#!/usr/bin/env python3
"""
Phase 0: Data Collection for Reward Proxy Training
===================================================
For each parent AMP (len 10-20), generate candidates for cond 2/3/4
using the pretrained diffusion model, evaluate with LLM judge,
and save all records to JSON Lines.

Usage:
    python collect_judge_data.py [--ckpt checkpoints/best.pt] [--limit 100] [--judge_lite]
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PAIRED_CSV,
    GRPO_DESIGN_TOP_K,
    LLM_A_PROVIDER, LLM_A_MODEL, LLM_A_BASE_URL, LLM_A_API_KEY,
    LLM_B_PROVIDER, LLM_B_MODEL, LLM_B_BASE_URL, LLM_B_API_KEY,
    LLM_C_PROVIDER, LLM_C_MODEL, LLM_C_BASE_URL, LLM_C_API_KEY,
    MIN_SEQ_LEN, MAX_SEQ_LEN_DESIGN,
)
from data.tokenizer import decode
from model.network import StapleDiffusionNetwork
from model.diffusion import random_mask
from llm_judge.client import create_client
from llm_judge.evaluator import create_evaluator
from predictors import load_predictor, batch_position_alignment
from train_rl_grpo import PairedRLDataset, design_candidates


def main():
    parser = argparse.ArgumentParser(description='Collect LLM judge data for reward proxy training')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/best.pt',
                        help='pretrained model checkpoint')
    parser.add_argument('--k', type=int, default=GRPO_DESIGN_TOP_K,
                        help='candidates per group')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0,
                        help='limit number of parents to process (0 = all)')
    parser.add_argument('--judge_lite', action='store_true',
                        help='skip cross-review (3 calls vs 5 per group)')
    parser.add_argument('--record_dir', type=str, default='logs/llm_judge_collection',
                        help='output directory for judge records')
    parser.add_argument('--mask_rate', type=float, default=0.1,
                        help='mask rate for model forward pass')
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
    parser.add_argument('--amp_ckpt', type=str, default=None, help='AMP classifier checkpoint')
    parser.add_argument('--tox_ckpt', type=str, default=None, help='Toxicity classifier checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    K = args.k

    # [1] Load dataset
    print(f'\n[1] Loading parents (len {MIN_SEQ_LEN}-{MAX_SEQ_LEN_DESIGN})...')
    dataset = PairedRLDataset(PAIRED_CSV)
    total = len(dataset)
    limit = total if args.limit <= 0 else min(args.limit, total)
    print(f'  Total: {total}, processing: {limit}')

    # [2] Load model
    print('\n[2] Loading pretrained model...')
    model = StapleDiffusionNetwork().to(device)
    ref_model = StapleDiffusionNetwork().to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f'  Loaded: {args.ckpt}')
    else:
        print('  No checkpoint found, using randomly initialized model')
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {n_params:,}')

    # Load AMP / toxicity predictors for prompt data
    print('\n[3] Loading AMP / toxicity predictors...')
    _project_root = os.path.dirname(os.path.abspath(__file__))
    amp_ckpt = args.amp_ckpt or os.path.join(_project_root, 'checkpoints', 'amp_classifier_best.pt')
    tox_ckpt = args.tox_ckpt or os.path.join(_project_root, 'checkpoints', 'toxicity_best.pt')
    amp_model = load_predictor(amp_ckpt, device)
    tox_model = load_predictor(tox_ckpt, device)

    # [4] Init LLM judge
    print('\n[4] Initializing LLM judge...')
    client_a = create_client(args.llm_a_provider, args.llm_a_model, args.llm_a_api_key, args.llm_a_base_url)
    client_b = create_client(args.llm_b_provider, args.llm_b_model, args.llm_b_api_key, args.llm_b_base_url)
    client_c = create_client(args.llm_c_provider, args.llm_c_model, args.llm_c_api_key, args.llm_c_base_url)
    llm_evaluator = create_evaluator(
        client_a, client_b, client_c,
        lite_mode=args.judge_lite,
        record_dir=args.record_dir,
    )
    llm_evaluator.set_step(0, 0)
    print(f'  A: {args.llm_a_model}, B: {args.llm_b_model}, C: {args.llm_c_model}')
    print(f'  Mode: {"Lite" if args.judge_lite else "Full"}')
    print(f'  Records -> {args.record_dir}')

    # [5] Main loop
    total_groups = limit * 3  # 3 conds per parent
    n_calls = total_groups * 5  # Full pipeline always (5 calls per group)
    print(f'\n[5] Collecting judge data...')
    print(f'  {limit} parents x 3 conds x {K} candidates = {total_groups * K} evaluations')
    print(f'  LLM API calls: ~{n_calls}')
    print()

    t_start = time.time()
    n_groups = 0

    for idx in range(limit):
        # Access parent tokens directly (ignore random cond from __getitem__)
        parent_tokens = torch.tensor(dataset.parents[idx], dtype=torch.long)
        parent_seq = decode(parent_tokens.tolist())
        seq_len = len(parent_seq)
        parent_tokens = parent_tokens.unsqueeze(0).to(device)

        # Compute parent gradients once for all conds
        amp_align = batch_position_alignment(amp_model, parent_tokens)   # [1, L]
        tox_align = batch_position_alignment(tox_model, parent_tokens)   # [1, L]

        for cond_val in [2, 3, 4]:
            cond_tensor = torch.tensor([cond_val], device=device)

            mask_rate_t = torch.full((1,), args.mask_rate, device=device)
            x_noisy, _ = random_mask(parent_tokens, mask_rate_t)

            with torch.no_grad():
                logits = ref_model(x_noisy, cond_tensor)
                log_probs = F.log_softmax(logits, dim=-1)

            candidates, _ = design_candidates(parent_tokens, cond_tensor, log_probs, K)
            cand_strs = [decode(c.tolist()) for c in candidates.cpu()]

            extra_data = {
                'amp_alignments': amp_align[0, :seq_len].tolist(),
                'tox_alignments': tox_align[0, :seq_len].tolist(),
            }
            llm_evaluator.evaluate_group(parent_seq, cond_val, cand_strs, extra_data)
            n_groups += 1

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (limit - idx - 1) / rate if rate > 0 else 0
            print(f'  [{idx+1}/{limit}] parents, {n_groups} groups, '
                  f'{elapsed:.0f}s elapsed, ETA {eta:.0f}s')

    t_total = time.time() - t_start
    print(f'\n{"=" * 50}')
    print(f'Done! {n_groups} groups evaluated in {t_total:.1f}s')
    print(f'  Parents: {limit}')
    print(f'  Groups:  {n_groups}')
    print(f'  Records: {args.record_dir}')
    print(f'  Avg:     {t_total / max(n_groups, 1):.2f}s per group')


if __name__ == '__main__':
    main()
