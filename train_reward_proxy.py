#!/usr/bin/env python3
"""
Phase 1: Train Reward Proxy Model from LLM Judge Records
=========================================================
Reads judge_records.jsonl, trains a small transformer + regression
head to predict LLM judge scores from (parent_seq, candidate_seq, cond).

Usage:
    python train_reward_proxy.py --data_dir logs/llm_judge_collection
"""

import os
import sys
import json
import time
import random
import math
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import VOCAB_SIZE, MAX_LEN
from data.tokenizer import encode, pad
from llm_judge.score_scale import SCALE_RECOVERED, resolve_scores


# ── Model ──────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class RewardProxyModel(nn.Module):
    """Small transformer encoder + regression head.
    Takes (parent, candidate, cond) and predicts a scalar score 0-10.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 4, max_len: int = MAX_LEN,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.cond_emb = nn.Embedding(5, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _encode(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.embedding(tokens)
        h = self.pos_enc(h)
        cond_emb = self.cond_emb(cond).unsqueeze(1)
        h = h + cond_emb
        h = self.dropout(h)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        mask = (tokens != 0).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    def forward(self, parent_tokens: torch.Tensor,
                candidate_tokens: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        parent_h = self._encode(parent_tokens, cond)
        cand_h = self._encode(candidate_tokens, cond)
        h = torch.cat([parent_h, cand_h], dim=-1)
        return self.regression_head(h).squeeze(-1)


# ── Dataset ────────────────────────────────────────────────────────

class RewardProxyDataset(Dataset):
    """Reads judge_records.jsonl files and builds (parent, candidate, cond, score) pairs."""

    def __init__(self, data_dir: str, max_len: int = MAX_LEN):
        self.max_len = max_len
        self.samples = []  # list of (parent_tokens, candidate_tokens, cond, score)
        self.n_recovered = 0
        self.n_skipped_unusable = 0

        pattern = os.path.join(data_dir, 'judge_records_*.jsonl')
        files = sorted(glob(pattern))
        if not files:
            raise FileNotFoundError(f'No judge_records_*.jsonl found in {data_dir}')

        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._extract_samples(record)

        if not self.samples:
            raise ValueError(f'No samples extracted from {len(files)} files in {data_dir}')

        print(f'RewardProxyDataset: {len(self.samples)} samples from {len(files)} files')
        if self.n_recovered:
            print(f'  recovered absolute scores from {self.n_recovered} '
                  f'legacy z-score record(s)')
        if self.n_skipped_unusable:
            print(f'  skipped {self.n_skipped_unusable} record(s) with no usable scores')

    def _extract_samples(self, record: dict):
        parent_seq = record.get('parent_seq', '')
        cond = record.get('cond', 2)
        candidates = record.get('candidates', [])

        # Normalise to absolute 0-10 scores.  Legacy z-score records are
        # recovered from their embedded phase-1/phase-3 payloads; records
        # with no recoverable payload are skipped rather than trained on a
        # relative target.
        scores, scale = resolve_scores(record)
        if scores is None:
            self.n_skipped_unusable += 1
            return
        if scale == SCALE_RECOVERED:
            self.n_recovered += 1

        if not parent_seq or not candidates:
            return

        parent_tokens = encode(parent_seq)

        for cand_info, score_f in zip(candidates, scores):
            cand_seq = cand_info.get('seq', '') if isinstance(cand_info, dict) else cand_info
            if not cand_seq:
                continue
            cand_tokens = encode(cand_seq)
            self.samples.append((
                parent_tokens,
                cand_tokens,
                cond,
                float(score_f),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        parent_tokens, cand_tokens, cond, score = self.samples[idx]
        parent_padded = torch.tensor(pad(parent_tokens, self.max_len), dtype=torch.long)
        cand_padded = torch.tensor(pad(cand_tokens, self.max_len), dtype=torch.long)
        return parent_padded, cand_padded, torch.tensor(cond, dtype=torch.long), torch.tensor(score, dtype=torch.float)


# ── Training ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for parent, candidate, cond, score in loader:
        parent = parent.to(device)
        candidate = candidate.to(device)
        cond = cond.to(device)
        score = score.to(device)

        pred = model(parent, candidate, cond)
        loss = F.mse_loss(pred, score)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(score)
        n += len(score)

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    n = 0
    for parent, candidate, cond, score in loader:
        parent = parent.to(device)
        candidate = candidate.to(device)
        cond = cond.to(device)
        score = score.to(device)

        pred = model(parent, candidate, cond)
        loss = F.mse_loss(pred, score)

        total_loss += loss.item() * len(score)
        all_preds.extend(pred.cpu().tolist())
        all_targets.extend(score.cpu().tolist())
        n += len(score)

    rmse = np.sqrt(total_loss / n)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    return total_loss / n, rmse, mae


def main():
    parser = argparse.ArgumentParser(description='Train reward proxy model from LLM judge records')
    parser.add_argument('--data_dir', type=str, default='logs/llm_judge_collection',
                        help='directory with judge_records_*.jsonl')
    parser.add_argument('--save_dir', type=str, default='checkpoints/reward_proxy',
                        help='output directory for model checkpoints')
    parser.add_argument('--d_model', type=int, default=256,
                        help='transformer hidden dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='optional pretrained diffusion model checkpoint for backbone init')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    print(f'\n[1] Loading judge records from {args.data_dir}...')
    dataset = RewardProxyDataset(args.data_dir)
    n_total = len(dataset)

    # Train/val split
    n_val = max(1, int(n_total * args.val_split))
    indices = list(range(n_total))
    random.shuffle(indices)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=args.batch_size, shuffle=False,
    )
    print(f'  Train: {len(train_idx)}, Val: {len(val_idx)}')

    # Build model
    print(f'\n[2] Building RewardProxyModel...')
    model = RewardProxyModel(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    if args.pretrained_ckpt:
        print(f'  Loading pretrained backbone from: {args.pretrained_ckpt}')
        ckpt = torch.load(args.pretrained_ckpt, map_location=device, weights_only=True)
        sd = ckpt.get('model_state_dict', ckpt)
        # Only load matching keys (embedding, pos_enc layers)
        model_dict = model.state_dict()
        for k in sd:
            if k in model_dict and sd[k].shape == model_dict[k].shape:
                model_dict[k] = sd[k]
        model.load_state_dict(model_dict)
        print(f'  Loaded matching weights from pretrained checkpoint')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {n_params:,}')
    print(f'  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_path = os.path.join(args.save_dir, 'best.pt')

    print(f'\n[3] Training...')
    print(f'  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_rmse, val_mae = evaluate(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
            }, best_path)

        print(f'  Epoch {epoch:3d}/{args.epochs} | '
              f'train: {train_loss:.4f} | val: {val_loss:.4f} | '
              f'RMSE: {val_rmse:.3f} | MAE: {val_mae:.3f} | '
              f'{time.time()-t0:.1f}s')

    print(f'\n{"=" * 50}')
    print(f'Done! Best val loss: {best_val_loss:.4f}')
    print(f'Best checkpoint: {best_path}')


if __name__ == '__main__':
    main()
