#!/usr/bin/env python3
"""Evaluation harness for stapled peptide design.

Compares the SFT checkpoint, the GRPO checkpoint and a model-free random
baseline on the same parent set, reporting structural, physicochemical and
predicted-function metrics.

Usage
-----
    python evaluate.py                                  # SFT vs GRPO vs random
    python evaluate.py --systems sft,random --n_parents 100
    python evaluate.py --mode generate --n_parents 200   # raw generative policy
    python evaluate.py --holdout 0.2 --out reports/run1

Two generation modes
--------------------
``design`` (default)
    Mirrors the deployed inference path: enumerate every structurally valid
    staple placement for a parent, rank by model log-probability, keep top-K.
    This is exactly what ``design_candidates`` does during GRPO training, so
    the evaluation measures the same objective the model was optimised for.

``generate``
    Mask the parent and let the model reconstruct it, which probes the raw
    generative policy.  The mask rate defaults to 1.0 (full masking) because
    with partial masking the staple sites usually stay unmasked and therefore
    keep their original residue — compliance is then bounded by mask geometry
    rather than by model capability.  ``preservation`` is expected to be low
    in this mode (the parent information is masked away) and is **not** a
    quality signal; read ``Compl``, ``Orph%`` and ``Pairs`` instead.

Held-out caveat
---------------
``--holdout`` selects parents by a deterministic hash bucket, which makes the
parent set reproducible.  It is **not** a true held-out set: the shipped
checkpoints were trained on all of ``data/paired_data.csv``.  Numbers for
``sft``/``grpo`` are therefore in-distribution and should be read as an upper
bound, not as generalisation performance.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CKPT_DIR, MAX_LEN, MIN_SEQ_LEN, MAX_SEQ_LEN_DESIGN, PAIRED_CSV,
    TEMPERATURE,
)
from data.tokenizer import encode, decode, pad
from design import design_candidates
from eval_core import (
    flatten_parents, random_designs, summarize_designs, summarize_diversity,
    summarize_properties,
)
from model.diffusion import random_mask
from model.network import StapleDiffusionNetwork
from predictors import batch_score, load_predictor

SYSTEM_ORDER = ('sft', 'grpo', 'random')
COND_NAMES = {1: 'no staple', 2: 'i,i+4 (S5/S5)', 3: 'i,i+7 (S5/R8)', 4: 'double'}

# Console / report column layout: (metric key, header, format spec)
DESIGN_COLUMNS = [
    ('cond_compliance', 'Compl', '.3f'),
    ('staple_validity', 'Valid', '.3f'),
    ('valid_rate', 'Valid%', '.1%'),
    ('preservation', 'Pres', '.3f'),
    ('rule_reward', 'Rew', '.3f'),
    ('mean_pairs', 'Pairs', '.2f'),
    ('terminal_anchor_rate', 'Term%', '.1%'),
    ('mean_staple_offset', 'Offset', '.2f'),
    ('orphan_rate', 'Orph%', '.1%'),
    ('fallback_rate', 'Fall%', '.1%'),
]
FUNCTION_COLUMNS = [
    ('amp_mean', 'AMP', '.3f'),
    ('amp_pos_rate', 'AMP+', '.1%'),
    ('tox_mean', 'Tox', '.3f'),
    ('charge', 'Chg', '+.2f'),
    ('hydrophobicity', 'H', '+.3f'),
    ('hmoment', 'uH', '.3f'),
]
DIVERSITY_COLUMNS = [
    ('unique_ratio', 'Uniq', '.2f'),
    ('mean_pairwise_identity', 'Ident', '.2f'),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_parents(csv_path: str, n: int, seed: int, holdout: float) -> List[str]:
    """Load unique parent AMPs, apply an optional hash-based holdout, sample n."""
    seen = set()
    parents: List[str] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        if 'input_seq' not in header:
            raise ValueError(f'{csv_path} has no input_seq column')
        for line in f:
            seq = line.split(',', 1)[0].strip()
            if not seq or seq in seen:
                continue
            length = len(encode(seq))
            if length < MIN_SEQ_LEN or length > MAX_SEQ_LEN_DESIGN:
                continue
            seen.add(seq)
            parents.append(seq)

    if holdout > 0.0:
        buckets = max(1, int(round(1.0 / holdout)))
        parents = [p for p in parents if _hash_bucket(p, buckets) == 0]

    parents.sort()
    rng = random.Random(seed)
    if n < len(parents):
        parents = rng.sample(parents, n)
    else:
        rng.shuffle(parents)
    return parents


def _hash_bucket(seq: str, buckets: int) -> int:
    digest = hashlib.md5(seq.encode('utf-8')).hexdigest()
    return int(digest[:8], 16) % buckets


def _batches(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str, device: torch.device):
    """Load a checkpoint, tolerating numpy scalars in the metadata.

    ``best.pt`` loads under ``weights_only=True``; ``rl_grpo_best.pt`` stores
    ``mean_reward`` as a numpy scalar, which the weights-only unpickler
    rejects.  These are local project artifacts, so falling back to the
    unrestricted loader is acceptable — but it is announced, not silent.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        print(f'  [note] {os.path.basename(path)} needs the unrestricted '
              f'loader (numpy scalars in metadata)')
        return torch.load(path, map_location=device, weights_only=False)


def load_diffusion_model(ckpt_path: str, device: torch.device) -> StapleDiffusionNetwork:
    model = StapleDiffusionNetwork().to(device)
    ckpt = _load_checkpoint(ckpt_path, device)
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    meta = ''
    if isinstance(ckpt, dict) and 'epoch' in ckpt:
        meta = f" (epoch={ckpt['epoch']}"
        if 'mean_reward' in ckpt:
            meta += f", mean_reward={float(ckpt['mean_reward']):.4f}"
        elif 'val_loss' in ckpt:
            meta += f", val_loss={float(ckpt['val_loss']):.4f}"
        meta += ')'
    print(f'  loaded {os.path.basename(ckpt_path)}{meta}')
    return model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_designs(model, parents: Sequence[str], cond: int, k: int,
                     device: torch.device, batch_size: int,
                     max_len: int = MAX_LEN) -> List[str]:
    """Top-K staple placements per parent, parent-major order."""
    out: List[str] = []
    for chunk in _batches(parents, batch_size):
        tokens = torch.tensor(
            [pad(encode(p), max_len) for p in chunk],
            dtype=torch.long, device=device)
        cond_t = torch.full((len(chunk),), cond, dtype=torch.long, device=device)
        logits = model(tokens, cond_t)
        log_probs = F.log_softmax(logits, dim=-1)
        cands, groups = design_candidates(tokens, cond_t, log_probs, k)
        for i in range(len(chunk)):
            for row in cands[groups == i]:
                out.append(decode(row.tolist()))
    return out


@torch.no_grad()
def generate_masked(model, parents: Sequence[str], cond: int, k: int,
                    device: torch.device, batch_size: int,
                    mask_rate: float, temperature: float,
                    max_len: int = MAX_LEN) -> List[str]:
    """Mask the parent and let the model reconstruct it, ``k`` times each.

    Output stays parent-major (parent 0 repeated k times, then parent 1, ...)
    so it lines up with ``flatten_parents``.
    """
    out: List[str] = []
    for chunk in _batches(parents, batch_size):
        base = torch.tensor(
            [pad(encode(p), max_len) for p in chunk],
            dtype=torch.long, device=device)
        tokens = base.repeat_interleave(k, dim=0)
        cond_t = torch.full((tokens.shape[0],), cond, dtype=torch.long, device=device)
        rates = torch.full((tokens.shape[0],), mask_rate, device=device)
        x_noisy, mask = random_mask(tokens, rates)
        logits = model(x_noisy, cond_t)
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        sampled = torch.multinomial(
            probs.view(-1, probs.size(-1)), 1).view(x_noisy.shape)
        restored = torch.where(mask, sampled, x_noisy)
        for row in restored:
            out.append(decode(row.tolist()))
    return out


@torch.no_grad()
def score_sequences(model, seqs: Sequence[str], device: torch.device,
                    batch_size: int = 64, max_len: int = MAX_LEN) -> List[float]:
    scores: List[float] = []
    for chunk in _batches(seqs, batch_size):
        tokens = torch.tensor(
            [pad(encode(s), max_len) for s in chunk],
            dtype=torch.long, device=device)
        scores.extend(batch_score(model, tokens).cpu().tolist())
    return scores


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_system(designs: Sequence[str], parents_flat: Sequence[str], cond: int,
                    amp_scores: Sequence[float],
                    tox_scores: Sequence[float]) -> Dict[str, float]:
    metrics = summarize_designs(designs, parents_flat, cond)
    metrics.update(summarize_properties(designs))
    metrics.update(summarize_diversity(designs))
    metrics['amp_mean'] = _mean(amp_scores)
    metrics['amp_pos_rate'] = _mean([1.0 if s >= 0.5 else 0.0 for s in amp_scores])
    metrics['tox_mean'] = _mean(tox_scores)
    metrics['tox_pos_rate'] = _mean([1.0 if s >= 0.5 else 0.0 for s in tox_scores])
    metrics['n_designs'] = len(designs)
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _render_table(rows: List[List[str]], headers: List[str]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = '  '.join('{:<%d}' % w for w in widths)
    lines = [fmt.format(*headers), '  '.join('-' * w for w in widths)]
    lines.extend(fmt.format(*row) for row in rows)
    return '\n'.join(lines)


def _row(system: str, cond: int, m: Dict[str, float],
         columns: Sequence) -> List[str]:
    cells = [system, str(cond), str(m.get('n_designs', 0))]
    for key, _, spec in columns:
        value = m.get(key, 0.0)
        cells.append(format(value, spec) if isinstance(value, float) else str(value))
    return cells


def _headers(prefix: Sequence[str], columns: Sequence) -> List[str]:
    return list(prefix) + [h for _, h, _ in columns]


def print_report(results: Dict[str, Dict[str, Dict[str, float]]],
                 conds: Sequence[int], mode: str) -> None:
    print(f'\n{"=" * 78}')
    print(f'STRUCTURE  (mode={mode})')
    print('=' * 78)
    rows = []
    for system in SYSTEM_ORDER:
        if system not in results:
            continue
        for cond in conds:
            m = results[system].get(str(cond))
            if m:
                rows.append(_row(system, cond, m, DESIGN_COLUMNS))
    print(_render_table(rows, _headers(['System', 'Cond', 'N'], DESIGN_COLUMNS)))

    print(f'\n{"=" * 78}')
    print('PREDICTED FUNCTION + PROPERTIES')
    print('=' * 78)
    rows = []
    for system in SYSTEM_ORDER:
        if system not in results:
            continue
        for cond in conds:
            m = results[system].get(str(cond))
            if m:
                rows.append(_row(system, cond, m, FUNCTION_COLUMNS))
    print(_render_table(rows, _headers(['System', 'Cond', 'N'], FUNCTION_COLUMNS)))

    print(f'\n{"=" * 78}')
    print('DIVERSITY')
    print('=' * 78)
    rows = []
    for system in SYSTEM_ORDER:
        if system not in results:
            continue
        for cond in conds:
            m = results[system].get(str(cond))
            if m:
                rows.append(_row(system, cond, m, DIVERSITY_COLUMNS))
    print(_render_table(rows, _headers(['System', 'Cond', 'N'], DIVERSITY_COLUMNS)))


def write_markdown(path: str, results, conds, args, elapsed: float) -> None:
    lines = ['# Stapled peptide design — evaluation report', '']
    lines.append(f'- Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'- Mode: `{args.mode}`')
    lines.append(f'- Parents: {args.n_parents} (seed={args.seed}, '
                 f'holdout={args.holdout})')
    lines.append(f'- Candidates per parent: K={args.k}')
    lines.append(f'- Conditions: {", ".join(str(c) for c in conds)}')
    lines.append(f'- Runtime: {elapsed:.1f}s')
    lines.append('')

    lines.append('## Caveat')
    lines.append('')
    if args.mode == 'generate':
        lines.append(f'Masked reconstruction with `mask_rate={args.mask_rate}`. '
                     'With partial masking the staple sites usually stay unmasked, '
                     'so compliance is bounded by mask geometry rather than model '
                     'capability — full masking (1.0) is the default for that reason.')
        lines.append('')
        lines.append('`Pres` is **not** a quality signal here: masking removes the '
                     'parent information, so a low value is expected.')
        lines.append('')
    if args.holdout > 0:
        lines.append('`--holdout` uses a deterministic hash bucket, so the parent '
                     'set is reproducible. It is **not** a true held-out set: the '
                     'checkpoints were trained on all of `data/paired_data.csv`. '
                     'SFT/GRPO numbers are in-distribution and should be read as '
                     'an upper bound.')
    else:
        lines.append('All parents come from `data/paired_data.csv`, which the '
                     'checkpoints were trained on. SFT/GRPO numbers are '
                     'in-distribution and should be read as an upper bound.')
    lines.append('')

    for title, columns in (('Structure', DESIGN_COLUMNS),
                           ('Predicted function and properties', FUNCTION_COLUMNS),
                           ('Diversity', DIVERSITY_COLUMNS)):
        lines.append(f'## {title}')
        lines.append('')
        headers = _headers(['System', 'Cond', 'N'], columns)
        rows = []
        for system in SYSTEM_ORDER:
            if system not in results:
                continue
            for cond in conds:
                m = results[system].get(str(cond))
                if m:
                    rows.append(_row(system, cond, m, columns))
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('|' + '|'.join('---' for _ in headers) + '|')
        for row in rows:
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')

    lines.append('## Metric definitions')
    lines.append('')
    lines.append('| Metric | Definition |')
    lines.append('|---|---|')
    definitions = [
        ('Compl', '`staple_accuracy` — match between requested cond and design'),
        ('Valid', '`staple_site_validity` — gap in {4,7}, terminal margin, same helical face'),
        ('Valid%', 'Fraction of designs with every staple pair valid'),
        ('Pres', '`preservation_score` — scaffold residues kept unchanged'),
        ('Rew', '`compute_reward` — rule-based scalar the RWR baseline optimises'),
        ('Pairs', 'Mean number of detected staple pairs'),
        ('Term%', 'Fraction with a staple touching residue 0 or L-1'),
        ('Offset', 'Mean normalised staple start position (0=N-term, 1=C-term)'),
        ('Orph%', 'Fraction of S5/R8 residues in no detected pair (broken crosslink)'),
        ('Fall%', 'Fraction identical to the parent (no design produced)'),
        ('AMP', 'Mean AMP-classifier probability (higher is better)'),
        ('AMP+', 'Fraction predicted AMP-positive (p >= 0.5)'),
        ('Tox', 'Mean toxicity-classifier probability (lower is better)'),
        ('Chg', 'Mean net charge at pH 7'),
        ('H', 'Mean Eisenberg hydrophobicity <H>'),
        ('uH', 'Mean hydrophobic moment (helical wheel)'),
        ('Uniq', 'Distinct sequences / total designs'),
        ('Ident', 'Mean pairwise residue identity (1 - diversity)'),
    ]
    for key, text in definitions:
        lines.append(f'| {key} | {text} |')
    lines.append('')

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate stapled peptide design systems')
    parser.add_argument('--systems', type=str, default='sft,grpo,random',
                        help='comma-separated: sft,grpo,random')
    parser.add_argument('--conds', type=str, default='2,3,4',
                        help='comma-separated conditions')
    parser.add_argument('--n_parents', type=int, default=200,
                        help='number of parent AMPs to evaluate')
    parser.add_argument('--k', type=int, default=5,
                        help='candidates per parent')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--holdout', type=float, default=0.0,
                        help='fraction reserved by hash bucket (reproducible, not true held-out)')
    parser.add_argument('--mode', choices=['design', 'generate'], default='design')
    parser.add_argument('--mask_rate', type=float, default=1.0,
                        help='mask rate for --mode generate (1.0 = full masking)')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                        help='sampling temperature for --mode generate')
    parser.add_argument('--sft_ckpt', type=str,
                        default=os.path.join(CKPT_DIR, 'best.pt'))
    parser.add_argument('--grpo_ckpt', type=str,
                        default=os.path.join(CKPT_DIR, 'rl_grpo', 'rl_grpo_best.pt'))
    parser.add_argument('--amp_ckpt', type=str,
                        default=os.path.join(CKPT_DIR, 'amp_classifier_best.pt'))
    parser.add_argument('--tox_ckpt', type=str,
                        default=os.path.join(CKPT_DIR, 'toxicity_best.pt'))
    parser.add_argument('--parents_csv', type=str, default=PAIRED_CSV)
    parser.add_argument('--out', type=str, default='',
                        help='output prefix; writes <out>.json and <out>.md')
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(',') if s.strip()]
    unknown = [s for s in systems if s not in SYSTEM_ORDER]
    if unknown:
        parser.error(f'unknown system(s): {", ".join(unknown)}')
    conds = [int(c) for c in args.conds.split(',') if c.strip()]
    for c in conds:
        if c not in COND_NAMES:
            parser.error(f'unknown condition: {c}')

    if args.mode == 'generate' and 'random' in systems:
        print('[note] random baseline is meaningless in generate mode; skipping it')
        systems = [s for s in systems if s != 'random']
    if args.mode == 'generate' and args.mask_rate < 1.0:
        print(f'[note] mask_rate={args.mask_rate}: staple sites may stay unmasked, '
              f'so compliance is bounded by mask geometry rather than capability')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    t0 = time.time()

    print('\n[1] Loading parents...')
    parents = load_parents(args.parents_csv, args.n_parents, args.seed, args.holdout)
    if not parents:
        raise SystemExit('no parents matched the filters')
    lengths = [len(encode(p)) for p in parents]
    print(f'  {len(parents)} parents | length {min(lengths)}-{max(lengths)} '
          f'(mean {sum(lengths) / len(lengths):.1f})')

    print('\n[2] Loading predictors...')
    amp_model = load_predictor(args.amp_ckpt, device)
    tox_model = load_predictor(args.tox_ckpt, device)

    print('\n[3] Loading design models...')
    models = {}
    for system in systems:
        if system == 'random':
            continue
        ckpt = args.sft_ckpt if system == 'sft' else args.grpo_ckpt
        if not os.path.exists(ckpt):
            print(f'  [skip] {system}: checkpoint not found ({ckpt})')
            continue
        models[system] = load_diffusion_model(ckpt, device)

    active = [s for s in SYSTEM_ORDER if s in systems
              and (s == 'random' or s in models)]
    if not active:
        raise SystemExit('no runnable system')

    print(f'\n[4] Evaluating ({args.mode} mode)...')
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    rng = random.Random(args.seed)

    for system in active:
        results[system] = {}
        for cond in conds:
            t_cond = time.time()
            if system == 'random':
                designs = random_designs(parents, cond, args.k, rng)
            elif args.mode == 'design':
                designs = generate_designs(
                    models[system], parents, cond, args.k, device, args.batch_size)
            else:
                designs = generate_masked(
                    models[system], parents, cond, args.k, device, args.batch_size,
                    args.mask_rate, args.temperature)

            parents_flat = flatten_parents(parents, args.k)
            amp_scores = score_sequences(amp_model, designs, device)
            tox_scores = score_sequences(tox_model, designs, device)
            metrics = evaluate_system(
                designs, parents_flat, cond, amp_scores, tox_scores)
            results[system][str(cond)] = metrics
            print(f'  {system:<7} cond={cond}  n={len(designs):<5} '
                  f'compl={metrics["cond_compliance"]:.3f} '
                  f'valid={metrics["staple_validity"]:.3f} '
                  f'pres={metrics["preservation"]:.3f} '
                  f'amp={metrics["amp_mean"]:.3f} '
                  f'({time.time() - t_cond:.1f}s)')

    elapsed = time.time() - t0
    print_report(results, conds, args.mode)

    out_prefix = args.out or os.path.join(
        'reports', f'eval_{time.strftime("%Y%m%d_%H%M%S")}')
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'mode': args.mode,
        'config': {
            'n_parents': len(parents),
            'k': args.k,
            'seed': args.seed,
            'holdout': args.holdout,
            'conds': conds,
            'systems': active,
            'mask_rate': args.mask_rate if args.mode == 'generate' else None,
            'temperature': args.temperature if args.mode == 'generate' else None,
            'sft_ckpt': args.sft_ckpt,
            'grpo_ckpt': args.grpo_ckpt,
        },
        'elapsed_sec': round(elapsed, 1),
        'results': results,
    }
    with open(out_prefix + '.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_markdown(out_prefix + '.md', results, conds, args, elapsed)

    print(f'\nSaved: {out_prefix}.json')
    print(f'Saved: {out_prefix}.md')


if __name__ == '__main__':
    main()
