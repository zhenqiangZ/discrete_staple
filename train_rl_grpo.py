import os
import sys
import csv
import time
import random
import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PAIRED_CSV, CKPT_DIR, MAX_LEN,
    RL_EPOCHS, RL_BATCH_SIZE, RL_STEPS_PER_EPOCH, RL_LR, RL_K,
    GRPO_CLIP_EPS, GRPO_KL_BETA, GRPO_MASK_RATE,
    W_LLM, JUDGE_EVERY_N_STEPS, JUDGE_SUBSAMPLE_GROUPS, JUDGE_LITE_MODE,
    LLM_A_PROVIDER, LLM_A_MODEL, LLM_A_BASE_URL, LLM_A_API_KEY,
    LLM_B_PROVIDER, LLM_B_MODEL, LLM_B_BASE_URL, LLM_B_API_KEY,
    LLM_C_PROVIDER, LLM_C_MODEL, LLM_C_BASE_URL, LLM_C_API_KEY,
    SUBSAMPLE_PER_EPOCH, MIN_SEQ_LEN, MAX_SEQ_LEN_DESIGN, RECORD_DIR,
    PROXY_MODEL_PATH, PROXY_ENABLED, REVIEW_THRESHOLD, REVIEW_EVERY_N_EPOCHS,
)
from data.tokenizer import encode, decode, pad
from design import design_candidates
from model.network import StapleDiffusionNetwork
from model.diffusion import random_mask
from reward_rules import detect_staple_pairs, staple_accuracy
from predictors import load_predictor, batch_position_alignment
from llm_judge.evaluator import create_evaluator
from llm_judge.client import create_client
from llm_judge.record_store import RecordStore
from llm_judge.proxy_client import load_proxy_client

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# ================================================================
# Data
# ================================================================

class PairedRLDataset(Dataset):
    """Only parent AMPs (10-20 token length), cond assigned at runtime."""

    def __init__(self, csv_path: str, max_len: int = MAX_LEN):
        self.max_len = max_len
        seen = set()
        self.parents: List[List[int]] = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                inp = row['input_seq'].strip()
                inp_tok = encode(inp)
                l = len(inp_tok)
                if l < MIN_SEQ_LEN or l > MAX_SEQ_LEN_DESIGN:
                    continue
                key = tuple(inp_tok)
                if key in seen:
                    continue
                seen.add(key)
                self.parents.append(pad(inp_tok, max_len))
        print(f'PairedRLDataset: {len(self.parents)} parent AMPs (len {MIN_SEQ_LEN}-{MAX_SEQ_LEN_DESIGN})')

    def __len__(self):
        return len(self.parents)

    def __getitem__(self, idx):
        cond = random.choice([2, 3, 4])
        return torch.tensor(self.parents[idx], dtype=torch.long), torch.tensor(cond, dtype=torch.long)


# ================================================================
# GRPO update
# ================================================================

def grpo_update(
    model: nn.Module,
    log_probs_ref: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    x_noisy: torch.Tensor,
    amp_tokens: torch.Tensor,
    cond: torch.Tensor,
    candidates: torch.Tensor,
    group_indices: torch.Tensor,
    rewards: torch.Tensor,
    judged: torch.Tensor,
    clip_eps: float,
    kl_beta: float,
) -> dict:
    """
    Single GRPO update step.

    Args:
        model:         current policy (train mode)
        log_probs_ref: [B, L, V] reference log-softmax (no grad, same x_noisy)
        optimizer:     AdamW
        x_noisy:       [B, L] masked input (shared with ref pass)
        amp_tokens:    [B, L] original AMP sequences (for PAD mask)
        cond:          [B] condition labels
        candidates:    [B*K, L] candidate sequences
        group_indices: [B*K] batch-index mapping
        rewards:       [B*K] scalar rewards (0 for unjudged entries)
        judged:        [B*K] bool mask of entries with a valid reward
        clip_eps:      clipping epsilon
        kl_beta:       KL penalty coefficient

    Returns:
        dict of scalar loss components
    """
    # --- Current policy log-probs ---
    logits_θ = model(x_noisy, cond)
    log_probs_θ = F.log_softmax(logits_θ, dim=-1)

    # --- Importance ratio over edited positions only ---
    # Unchanged positions carry no action; summing over them would let the
    # ratio grow with sequence length and distort clipping.
    non_pad = (amp_tokens != 0).float()                            # [B, L]

    log_probs_θ_exp = log_probs_θ[group_indices]                   # [B*K, L, V]
    log_probs_ref_exp = log_probs_ref[group_indices]               # [B*K, L, V]

    log_p_θ = log_probs_θ_exp.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
    log_p_ref = log_probs_ref_exp.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)

    parent_exp = amp_tokens[group_indices]                         # [B*K, L]
    changed = (candidates != parent_exp) & (parent_exp != 0)       # [B*K, L]
    log_ratio = ((log_p_θ - log_p_ref) * changed.float()).sum(dim=-1).clamp(-10.0, 10.0)
    ratio = log_ratio.exp()

    # --- Group-normalised advantages (judged entries only) ---
    advantages = torch.zeros_like(rewards)
    for g in group_indices.unique():
        m = (group_indices == g) & judged
        if m.sum() < 2:
            continue
        grp = rewards[m]
        if grp.std() > 1e-8:
            advantages[m] = (grp - grp.mean()) / grp.std()
        advantages[m] = advantages[m].clamp(-2.0, 2.0)

    # --- Clipped surrogate (only judged candidates contribute) ---
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    pg_loss = -torch.min(surr1, surr2)[judged].mean()

    # --- KL divergence (all non-PAD positions) ---
    p_ref = log_probs_ref.exp()
    kl_per_pos = (p_ref * (log_probs_ref - log_probs_θ)).sum(dim=-1)
    kl = (kl_per_pos * non_pad).sum() / (non_pad.sum() + 1e-8)

    total_loss = pg_loss + kl_beta * kl

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        'total_loss': total_loss.item(),
        'pg_loss': pg_loss.item(),
        'kl': kl.item(),
        'mean_ratio': ratio[judged].mean().item(),
        'mean_adv': advantages[judged].mean().item(),
        'mean_reward': rewards[judged].mean().item(),
        'frac_clipped': ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps))[judged].float().mean().item(),
        'frac_pos_adv': (advantages[judged] > 0).float().mean().item(),
        'n_judged': int(judged.sum().item()),
        'mean_changed': changed.float().sum(dim=-1).mean().item(),
    }


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=RL_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=RL_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=RL_LR)
    parser.add_argument('--k', type=int, default=RL_K, help='candidates per AMP')
    parser.add_argument('--clip_eps', type=float, default=GRPO_CLIP_EPS)
    parser.add_argument('--kl_beta', type=float, default=GRPO_KL_BETA)
    parser.add_argument('--mask_rate', type=float, default=GRPO_MASK_RATE)
    parser.add_argument('--ckpt', type=str, default='./checkpoints/best.pt', help='pretrained checkpoint')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp_ckpt', type=str, default=None, help='AMP classifier checkpoint')
    parser.add_argument('--tox_ckpt', type=str, default=None, help='Toxicity classifier checkpoint')
    parser.add_argument('--w_llm', type=float, default=W_LLM,
                        help='LLM reward weight (0 = disabled)')
    parser.add_argument('--judge_every_n_steps', type=int, default=JUDGE_EVERY_N_STEPS)
    parser.add_argument('--judge_subsample_groups', type=int, default=JUDGE_SUBSAMPLE_GROUPS)
    parser.add_argument('--judge_lite', action='store_true', default=JUDGE_LITE_MODE,
                        help='Skip cross-review phase (3 calls vs 5 per group)')
    parser.add_argument('--record_dir', type=str, default=RECORD_DIR,
                        help='LLM judge record output directory')
    parser.add_argument('--subsample_per_epoch', type=int, default=SUBSAMPLE_PER_EPOCH,
                        help='Number of parent AMPs sampled per epoch')
    parser.add_argument('--steps_per_epoch', type=int, default=RL_STEPS_PER_EPOCH,
                        help='Max optimizer updates per epoch. Note this caps how much of '
                             'subsample_per_epoch is actually consumed: with K and batch_size, '
                             'one update sees batch_size*K candidates, so set this to about '
                             'subsample_per_epoch/batch_size to use the whole sample.')
    parser.add_argument('--proxy_model_path', type=str, default=PROXY_MODEL_PATH,
                        help='path to trained RewardProxyModel checkpoint')
    parser.add_argument('--proxy_enabled', action='store_true', default=PROXY_ENABLED,
                        help='use proxy model instead of LLM API for new evaluations')
    parser.add_argument('--review_threshold', type=int, default=REVIEW_THRESHOLD,
                        help='number of proxy entries triggering LLM review')
    parser.add_argument('--llm_a_provider', type=str, default=LLM_A_PROVIDER)
    parser.add_argument('--llm_a_model', type=str, default=LLM_A_MODEL)
    parser.add_argument('--llm_b_provider', type=str, default=LLM_B_PROVIDER)
    parser.add_argument('--llm_b_model', type=str, default=LLM_B_MODEL)
    parser.add_argument('--llm_c_provider', type=str, default=LLM_C_PROVIDER)
    parser.add_argument('--llm_c_model', type=str, default=LLM_C_MODEL)
    parser.add_argument('--llm_a_base_url', type=str, default=LLM_A_BASE_URL)
    parser.add_argument('--llm_b_base_url', type=str, default=LLM_B_BASE_URL)
    parser.add_argument('--llm_c_base_url', type=str, default=LLM_C_BASE_URL)
    parser.add_argument('--llm_a_api_key', type=str, default=LLM_A_API_KEY)
    parser.add_argument('--llm_b_api_key', type=str, default=LLM_B_API_KEY)
    parser.add_argument('--llm_c_api_key', type=str, default=LLM_C_API_KEY)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    epochs = args.epochs
    batch_size = args.batch_size
    K = args.k
    save_dir = args.save_dir or os.path.join(CKPT_DIR, 'rl_grpo')
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print('\n[1] Loading parent AMPs (10-20 token length)...')
    dataset = PairedRLDataset(PAIRED_CSV)

    # Load model
    print('\n[2] Loading model...')
    model = StapleDiffusionNetwork().to(device)
    ref_model = StapleDiffusionNetwork().to(device)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f'  Loaded checkpoint: {args.ckpt}')
    else:
        print('  No checkpoint specified, using randomly initialized model')

    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {n_params:,}')

    # Load predictor models
    print('\n[3] Loading AMP / toxicity predictors...')
    _project_root = os.path.dirname(os.path.abspath(__file__))
    amp_ckpt = args.amp_ckpt or os.path.join(_project_root, 'checkpoints', 'amp_classifier_best.pt')
    tox_ckpt = args.tox_ckpt or os.path.join(_project_root, 'checkpoints', 'toxicity_best.pt')
    amp_model = load_predictor(amp_ckpt, device)
    tox_model = load_predictor(tox_ckpt, device)

    # Load LLM Judge
    print('\n[4] Initializing LLM judge...')
    llm_evaluator = None
    if args.w_llm > 0:
        client_a = create_client(args.llm_a_provider, args.llm_a_model, args.llm_a_api_key, args.llm_a_base_url)
        client_b = create_client(args.llm_b_provider, args.llm_b_model, args.llm_b_api_key, args.llm_b_base_url)
        client_c = create_client(args.llm_c_provider, args.llm_c_model, args.llm_c_api_key, args.llm_c_base_url)
        llm_evaluator = create_evaluator(client_a, client_b, client_c,
                                         lite_mode=args.judge_lite,
                                         record_dir=args.record_dir)
        print(f'  LLM Judge: A={args.llm_a_model}, B={args.llm_b_model}, C={args.llm_c_model}')
        print(f'  Mode: {"Lite" if args.judge_lite else "Full"}')
        print(f'  Every {args.judge_every_n_steps} steps, {args.judge_subsample_groups} groups')
        if args.record_dir:
            print(f'  Records -> {args.record_dir}')
        if args.llm_a_base_url:
            print(f'  A base_url: {args.llm_a_base_url}')
        if args.llm_b_base_url:
            print(f'  B base_url: {args.llm_b_base_url}')
        if args.llm_c_base_url:
            print(f'  C base_url: {args.llm_c_base_url}')
    else:
        print(f'  Disabled (w_llm=0)')

    # Load RecordStore (persistent cache across epochs)
    print('\n[5] Initializing RecordStore...')
    record_store = None
    if args.record_dir:
        record_store = RecordStore(args.record_dir)
        print(f'  RecordStore loaded: {record_store.num_records()} entries, '
              f'{record_store.num_groups()} groups')
        _rs = record_store.summarize()
        if _rs['recovered_records']:
            print(f'  [note] recovered absolute scores from '
                  f'{_rs["recovered_records"]} legacy z-score record(s)')
        if _rs['unusable_records']:
            print(f'  [note] {_rs["unusable_records"]} record(s) had no usable scores')

    # Load Reward Proxy
    proxy_client = None
    if args.proxy_enabled:
        print('\n[6] Loading reward proxy...')
        proxy_client = load_proxy_client(args.proxy_model_path, device)
        if proxy_client is not None:
            print(f'  Proxy loaded: {args.proxy_model_path}')
        else:
            print(f'  Proxy load FAILED, falling back to LLM API')
            args.proxy_enabled = False

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # RL loop
    print(f'\n{"=" * 50}')
    print('RL Fine-tuning (GRPO + Multi-objective Reward)')
    print(f'  epochs={epochs}, batch_size={batch_size}, K={K}')
    print(f'  lr={args.lr}, clip_eps={args.clip_eps}, kl_beta={args.kl_beta}')
    print(f'  mask_rate={args.mask_rate}')
    _reward_src = 'LLM judge'
    if proxy_client is not None:
        _reward_src = 'cache -> proxy -> LLM judge'
    elif record_store is not None:
        _reward_src = 'cache -> LLM judge'
    print(f'  reward: {_reward_src} (w_llm={args.w_llm})')
    if args.proxy_enabled:
        print(f'  proxy={args.proxy_model_path} | review_threshold={args.review_threshold}')
    print(f'{"=" * 50}')

    if llm_evaluator is None and proxy_client is None:
        print('\n[warn] No reward source available '
              '(LLM judge disabled and no proxy model); exiting.')
        return

    best_mean_reward = -float('inf')
    global_step = 0

    for epoch in range(1, epochs + 1):
        # Subsample AMPs for this epoch
        n_sample = min(len(dataset), args.subsample_per_epoch)
        indices = random.sample(range(len(dataset)), n_sample)
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        epoch_rewards = []
        epoch_accs = []
        epoch_losses = []
        epoch_pg = []
        epoch_kl = []
        epoch_clipped = []
        epoch_adv = []
        epoch_r_llm = []
        epoch_judged = []
        t0 = time.time()
        update_step = 0
        last_tokens = None
        last_cond = None

        for batch in loader:
            # Only every N-th batch is judged.  The others carry no gradient
            # signal, so skip them before any forward pass or enumeration.
            if global_step % args.judge_every_n_steps != 0:
                global_step += 1
                continue

            input_tokens, cond = batch
            input_tokens = input_tokens.to(device)
            cond = cond.to(device)
            B = input_tokens.shape[0]

            input_strs = [decode(t.tolist()) for t in input_tokens.cpu()]
            cond_list = cond.tolist()

            # --- Step 1: noisy input (used by both ref and current) ---
            mask_rate_t = torch.full((B,), args.mask_rate, device=device)
            x_noisy, _ = random_mask(input_tokens, mask_rate_t)

            # --- Step 2: generate candidates via reference model ---
            ref_model.eval()
            with torch.no_grad():
                logits_ref = ref_model(x_noisy, cond)
                log_probs_ref = F.log_softmax(logits_ref, dim=-1)

                candidates, group_indices = design_candidates(
                    input_tokens, cond, log_probs_ref, K,
                )

            # --- Step 3: position alignment on parent (for LLM prompt data) ---
            amp_align = batch_position_alignment(amp_model, input_tokens)   # [B, L]
            tox_align = batch_position_alignment(tox_model, input_tokens)   # [B, L]

            # --- Step 4: reward (NaN marks groups that failed to score) ---
            # Priority: exact cache hit (raw scale) > proxy model > LLM API.
            if llm_evaluator is not None:
                llm_evaluator.set_step(epoch, global_step)
            n_groups = min(B, args.judge_subsample_groups)
            rewards_np = np.full(B * K, np.nan, dtype=np.float64)

            for g in range(n_groups):
                grp_s = g * K
                grp_e = (g + 1) * K
                parent_str = input_strs[g]
                cond_val = cond_list[g]
                cand_strs = [decode(candidates[i].tolist()) for i in range(grp_s, grp_e)]

                source = None
                result = None

                # (a) Reuse a previously scored identical group.
                if record_store is not None:
                    cached = [record_store.lookup_raw(parent_str, c, cond_val)
                              for c in cand_strs]
                    if all(s is not None for s in cached):
                        result = np.asarray(cached, dtype=np.float64)
                        source = 'cache'

                # (b) Cheap local proxy when enabled.
                if result is None and proxy_client is not None:
                    try:
                        raw = proxy_client.score_with_reasoning(
                            parent_str, cond_val, cand_strs)
                        result = np.asarray([float(r['score']) for r in raw],
                                            dtype=np.float64)
                        source = 'proxy'
                    except Exception as e:
                        print(f'  [warn] proxy scoring failed ({e}); using LLM API')

                # (c) Full multi-LLM pipeline.
                if result is None and llm_evaluator is not None:
                    seq_len = len(parent_str)
                    extra_data = {
                        'amp_alignments': amp_align[g, :seq_len].tolist(),
                        'tox_alignments': tox_align[g, :seq_len].tolist(),
                    }
                    result = np.asarray(
                        llm_evaluator.evaluate_group(
                            parent_str, cond_val, cand_strs, extra_data),
                        dtype=np.float64,
                    )
                    source = 'llm_api'

                if result is None:
                    # No cache hit, no proxy, no LLM judge: leave as NaN.
                    continue

                if result.shape[0] != K:
                    print(f'  [warn] judge returned {result.shape[0]} scores '
                          f'for {K} candidates; group skipped')
                    continue
                rewards_np[grp_s:grp_e] = result

                # Persist fresh scores so later epochs can reuse them.
                if record_store is not None and source in ('proxy', 'llm_api'):
                    for c, s in zip(cand_strs, result):
                        if np.isnan(s):
                            continue
                        record_store.put(parent_str, cond_val, c,
                                         float(s), source=source)

            judged = torch.tensor(~np.isnan(rewards_np), device=device)
            n_judged = int(judged.sum().item())
            global_step += 1

            if n_judged == 0:
                print('  [warn] no usable judge scores in this batch; update skipped')
                continue

            rewards = torch.tensor(
                np.nan_to_num(rewards_np, nan=0.0),
                device=device, dtype=torch.float32,
            )

            # --- Step 5: GRPO update (only judged candidates contribute) ---
            stats = grpo_update(
                model, log_probs_ref, optimizer,
                x_noisy, input_tokens, cond,
                candidates, group_indices, rewards, judged,
                args.clip_eps, args.kl_beta,
            )

            # --- Stats per group ---
            for i in range(B):
                grp_s = i * K
                grp_e = (i + 1) * K
                if not judged[grp_s:grp_e].any():
                    continue
                grp_rewards = rewards[grp_s:grp_e].clone()
                grp_rewards[~judged[grp_s:grp_e]] = -float('inf')
                best_idx = grp_s + grp_rewards.argmax().item()
                best_seq = decode(candidates[best_idx].tolist())
                epoch_accs.append(staple_accuracy(best_seq, cond_list[i]))
                epoch_rewards.append(grp_rewards.max().item())

            epoch_losses.append(stats['total_loss'])
            epoch_pg.append(stats['pg_loss'])
            epoch_kl.append(stats['kl'])
            epoch_clipped.append(stats['frac_clipped'])
            epoch_adv.append(stats['mean_adv'])
            epoch_r_llm.append(stats['mean_reward'])
            epoch_judged.append(n_judged)

            last_tokens, last_cond = input_tokens, cond
            update_step += 1

            if update_step >= args.steps_per_epoch:
                break

        elapsed = time.time() - t0
        mean_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        mean_acc = np.mean(epoch_accs) if epoch_accs else 0.0
        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        mean_pg = np.mean(epoch_pg) if epoch_pg else 0.0
        mean_kl = np.mean(epoch_kl) if epoch_kl else 0.0
        mean_clipped = np.mean(epoch_clipped) if epoch_clipped else 0.0
        mean_adv = np.mean(epoch_adv) if epoch_adv else 0.0

        mean_r_llm = np.mean(epoch_r_llm) if epoch_r_llm else 0.0
        mean_judged = np.mean(epoch_judged) if epoch_judged else 0.0

        print(f'\nEpoch {epoch}/{epochs} [{elapsed:.0f}s] | updates={update_step}')
        print(f'  Reward: {mean_reward:.4f} | Acc: {mean_acc:.2%} | Loss: {mean_loss:.4f}')
        print(f'  PG: {mean_pg:.4f} | KL: {mean_kl:.6f} | clip: {mean_clipped:.1%} | adv: {mean_adv:.3f}')
        if args.w_llm > 0 or proxy_client is not None:
            print(f'  r_score={mean_r_llm:.4f} | judged/update={mean_judged:.1f}')

        # Demo
        if last_tokens is not None:
            with torch.no_grad():
                n_demo = min(4, last_tokens.shape[0])
                demo_inp = last_tokens[:n_demo]
                demo_cond = last_cond[:n_demo]
                dm = torch.full((n_demo,), args.mask_rate, device=device)
                x_demo, _ = random_mask(demo_inp, dm)
                logits_demo = model(x_demo, demo_cond)
                demo_pred = logits_demo.argmax(dim=-1)
                for i in range(n_demo):
                    inp_seq = decode(demo_inp[i].tolist())
                    out_seq = decode(demo_pred[i].tolist())
                    c = demo_cond[i].item()
                    pairs = detect_staple_pairs(out_seq)
                    print(f'  [{i}] cond={c} | {inp_seq} -> {out_seq} (staples: {pairs})')

        # Phase 3: Review proxy-marked groups with LLM API
        if record_store is not None and llm_evaluator is not None \
                and epoch % REVIEW_EVERY_N_EPOCHS == 0:
            groups_to_review = record_store.get_proxy_marked_groups(
                threshold=args.review_threshold)
            if groups_to_review:
                print(f'  [Review] {len(groups_to_review)} groups with >= '
                      f'{args.review_threshold} proxy entries')
                n_reviewed = 0
                for parent_seq, cond_val in groups_to_review:
                    entries = record_store.get_group_entries(parent_seq, cond_val)
                    proxy_entries = [e for e in entries if e.get('source') == 'proxy']
                    if len(proxy_entries) < args.review_threshold:
                        continue
                    proxy_seqs = [e['seq'] for e in proxy_entries]
                    try:
                        new_scores = llm_evaluator.evaluate_group(
                            parent_seq, cond_val, proxy_seqs)
                    except Exception as ex:
                        print(f'    Review failed for {parent_seq} cond={cond_val}: {ex}')
                        continue
                    # Failed groups come back as NaN; keep the proxy values.
                    n_written = 0
                    for e, s in zip(proxy_entries, new_scores):
                        if np.isnan(s):
                            continue
                        record_store.put(parent_seq, cond_val, e['seq'],
                                         float(s), source='llm_api')
                        n_written += 1
                    if n_written:
                        n_reviewed += 1
                if n_reviewed:
                    print(f'    Reviewed {n_reviewed} groups with LLM API')
                    rs = record_store.summarize()
                    print(f'    Store: {rs["total_entries"]} entries, '
                          f'llm_api={rs["llm_api"]}, proxy={rs["proxy"]}, '
                          f'recovered={rs["recovered_entries"]}')

        # Note: ref_model stays frozen at the SFT policy (KL anchor).
        if epoch_rewards and mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            ckpt_path = os.path.join(save_dir, 'rl_grpo_best.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'mean_reward': mean_reward,
            }, ckpt_path)
            print(f'  [BEST] reward={mean_reward:.4f} -> saved {ckpt_path}')

    print(f'\nDone. Best mean reward: {best_mean_reward:.4f}')


if __name__ == '__main__':
    main()
