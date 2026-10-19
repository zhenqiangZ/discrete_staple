import os
import sys
import csv
import time
import random
import argparse
from typing import List, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PAIRED_CSV, CKPT_DIR, VOCAB_SIZE, MAX_LEN,
    RL_EPOCHS, RL_BATCH_SIZE, RL_STEPS_PER_EPOCH,
    RL_LR, RL_ADV_COEFF, RL_K, RL_MIX_GT,
    RL_SAMPLE_STEPS, RL_SAMPLE_TEMP, RL_MASK_RATE_MAX,
)
from data.tokenizer import encode, decode, pad
from model.network import StapleDiffusionNetwork
from model.diffusion import random_mask, random_mask_rate
from reward_rules import compute_reward, detect_staple_pairs, staple_accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# ================================================================
# Data
# ================================================================

class PairedRLDataset(Dataset):
    def __init__(self, csv_path: str, max_len: int = MAX_LEN):
        self.max_len = max_len
        self.triples: List[Tuple[List[int], List[int], int]] = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                inp = row['input_seq'].strip()
                tgt = row['target_seq'].strip()
                cond = int(row['cond'])
                inp_tok = pad(encode(inp), max_len)
                tgt_tok = pad(encode(tgt), max_len)
                self.triples.append((inp_tok, tgt_tok, cond))
        print(f'PairedRLDataset: {len(self.triples)} triples')
        cond_counts = Counter(c for _, _, c in self.triples)
        for c in sorted(cond_counts):
            print(f'  cond={c}: {cond_counts[c]}')

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        inp, tgt, cond = self.triples[idx]
        return (torch.tensor(inp, dtype=torch.long),
                torch.tensor(tgt, dtype=torch.long),
                torch.tensor(cond, dtype=torch.long))


def create_rl_loaders(csv_path: str, batch_size: int = RL_BATCH_SIZE):
    dataset = PairedRLDataset(csv_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


# ================================================================
# Edit-based sampling
# ================================================================

@torch.no_grad()
def sample_edit(model: nn.Module,
                input_tokens: torch.Tensor,
                cond: torch.Tensor,
                steps: int = 1,
                temperature: float = RL_SAMPLE_TEMP,
                mask_rate_max: float = RL_MASK_RATE_MAX,
                ) -> torch.Tensor:
    model.eval()
    B, L = input_tokens.shape

    mask_rate = random.uniform(0.3, mask_rate_max)
    x_noisy, mask = random_mask(input_tokens, mask_rate)

    logits = model(x_noisy, cond)

    if temperature < 0.1:
        x = logits.argmax(dim=-1)
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        x = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, L)

    return x


# ================================================================
# Reward
# ================================================================

def compute_batch_rewards(tokens: torch.Tensor,
                          input_strs: List[str],
                          conds: List[int],
                          ) -> torch.Tensor:
    B = tokens.shape[0]
    rewards = torch.zeros(B)
    for i in range(B):
        seq = decode(tokens[i].tolist())
        r = compute_reward(seq, input_strs[i], conds[i])
        rewards[i] = r
    return rewards


# ================================================================
# RWR update
# ================================================================

def rwr_update(model: nn.Module,
               optimizer: torch.optim.Optimizer,
               input_tokens: torch.Tensor,
               cond: torch.Tensor,
               candidates: torch.Tensor,
               advantages: torch.Tensor,
               mix_gt: float = 0.0,
               gt_target: torch.Tensor = None,
               ):
    model.train()
    B, L = input_tokens.shape
    K = candidates.shape[0] // B
    total_loss = 0.0

    for k in range(K):
        start = k * B
        end = (k + 1) * B
        cand = candidates[start:end]
        adv = advantages[start:end]

        mask_rate = random_mask_rate(RL_MASK_RATE_MAX)
        x_noisy, mask = random_mask(input_tokens, mask_rate)

        logits = model(x_noisy, cond)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                             cand.reshape(-1), reduction='none')
        ce = ce.reshape(B, L)
        ce = ce[mask].mean()

        weight = (1.0 + RL_ADV_COEFF * adv).mean()
        total_loss += ce * weight

    total_loss /= K

    if mix_gt > 0 and gt_target is not None:
        mask_rate = random_mask_rate(RL_MASK_RATE_MAX)
        x_noisy, mask = random_mask(input_tokens, mask_rate)
        logits = model(x_noisy, cond)
        gt_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                  gt_target.reshape(-1))
        total_loss = (1 - mix_gt) * total_loss + mix_gt * gt_loss

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return total_loss.item()


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=RL_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=RL_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=RL_LR)
    parser.add_argument('--k', type=int, default=RL_K, help='candidates per AMP')
    parser.add_argument('--adv_coeff', type=float, default=RL_ADV_COEFF)
    parser.add_argument('--mix_gt', type=float, default=RL_MIX_GT)
    parser.add_argument('--ckpt', type=str, default=None,
                        help='pretrained checkpoint path')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    epochs = args.epochs
    batch_size = args.batch_size
    K = args.k
    save_dir = args.save_dir or os.path.join(CKPT_DIR, 'rl')
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print('\n[1] Loading paired data...')
    loader = create_rl_loaders(PAIRED_CSV, batch_size)

    # Load model
    print('\n[2] Loading model...')
    model = StapleDiffusionNetwork().to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f'  Loaded checkpoint: {args.ckpt}')
    else:
        print('  No checkpoint specified, using randomly initialized model')

    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {n_params:,}')

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # RL loop
    print(f'\n{"=" * 50}')
    print('RL Fine-tuning (Best-of-K RWR)')
    print(f'  epochs={epochs}, batch_size={batch_size}, K={K}')
    print(f'  lr={args.lr}, adv_coeff={args.adv_coeff}, mix_gt={args.mix_gt}')
    print(f'  sample_steps={RL_SAMPLE_STEPS}, temp={RL_SAMPLE_TEMP}')
    print(f'{"=" * 50}')

    best_mean_reward = -float('inf')
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_rewards = []
        epoch_accs = []
        epoch_steps = 0
        epoch_losses = []
        t0 = time.time()

        for batch in loader:
            input_tokens, target_tokens, cond = batch
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)
            cond = cond.to(device)
            B = input_tokens.shape[0]

            input_strs = [decode(t.tolist()) for t in input_tokens.cpu()]

            # Sample K candidates per AMP
            all_candidates = []
            for _ in range(K):
                cand = sample_edit(
                    model, input_tokens, cond,
                    steps=RL_SAMPLE_STEPS,
                    temperature=RL_SAMPLE_TEMP,
                    mask_rate_max=RL_MASK_RATE_MAX,
                )
                all_candidates.append(cand)
            all_candidates = torch.cat(all_candidates, dim=0)  # [K*B, L]

            # Compute rewards
            cond_list = cond.tolist()
            all_conds = cond_list * K
            all_input_strs = input_strs * K

            rewards = compute_batch_rewards(
                all_candidates, all_input_strs, all_conds,
            )
            rewards = rewards.to(device)

            # Normalize advantage per AMP group (K candidates each)
            advantages = torch.zeros_like(rewards)
            for i in range(B):
                group = rewards[i::B]
                if group.std() > 1e-8:
                    advantages[i::B] = (group - group.mean()) / group.std()
                advantages[i::B] = advantages[i::B].clamp(-2.0, 2.0)

            # RWR update
            loss = rwr_update(
                model, optimizer,
                input_tokens, cond,
                all_candidates, advantages,
                mix_gt=args.mix_gt,
                gt_target=target_tokens,
            )

            # Stats
            for i in range(B):
                best_idx = rewards[i::B].argmax().item() * B + i
                best_seq = decode(all_candidates[best_idx].tolist())
                acc = staple_accuracy(best_seq, cond_list[i])
                epoch_accs.append(acc)

            best_per_amp = [rewards[i::B].max().item() for i in range(B)]
            epoch_rewards.extend(best_per_amp)
            epoch_losses.append(loss)
            epoch_steps += 1
            global_step += 1

            if epoch_steps >= RL_STEPS_PER_EPOCH:
                break

        elapsed = time.time() - t0
        mean_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        mean_acc = np.mean(epoch_accs) if epoch_accs else 0.0
        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        # Demo
        print(f'\nEpoch {epoch}/{epochs} [{elapsed:.0f}s]')
        print(f'  Reward best/AMP: {mean_reward:.4f} | Acc: {mean_acc:.2%} | Loss: {mean_loss:.4f}')
        with torch.no_grad():
            demo_inp = input_tokens[:4].clone() if input_tokens.shape[0] >= 4 else input_tokens.clone()
            demo_cond = cond[:4].clone() if cond.shape[0] >= 4 else cond.clone()
            demo_cand = sample_edit(
                model, demo_inp, demo_cond,
                steps=RL_SAMPLE_STEPS, temperature=0.3,
            )
            for i in range(demo_inp.shape[0]):
                inp_seq = decode(demo_inp[i].tolist())
                out_seq = decode(demo_cand[i].tolist())
                c = demo_cond[i].item()
                pairs = detect_staple_pairs(out_seq)
                print(f'  [{i}] cond={c} | {inp_seq} -> {out_seq} (staples: {pairs})')

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            ckpt_path = os.path.join(save_dir, f'rl_best.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  [BEST] reward={mean_reward:.4f} -> saved {ckpt_path}')

    print(f'\nDone. Best mean reward: {best_mean_reward:.4f}')


if __name__ == '__main__':
    main()
