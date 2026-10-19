"""Split paired_data.csv into train / holdout by parent sequence.

Why this exists
---------------
``create_loaders`` already splits by parent sequence, but it splits *inside*
the file you hand it.  If you train on the full ``paired_data.csv`` and then
evaluate on parents drawn from that same file, the model has seen those
parents — the evaluation is in-distribution and reads as an upper bound.

This script carves out a parent-disjoint holdout once, so that:

    python train.py  --data data/paired_train.csv --out_dir checkpoints/sft_ho
    python evaluate.py --parents_csv data/paired_holdout.csv --holdout 1.0

is a genuine held-out measurement.

Usage
-----
    python data/make_holdout_split.py                  # 15% holdout, seed 42
    python data/make_holdout_split.py --frac 0.2
    python data/make_holdout_split.py --dry_run        # just report sizes
"""
import argparse
import csv
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(HERE, 'paired_data.csv')
DEFAULT_TRAIN = os.path.join(HERE, 'paired_train.csv')
DEFAULT_HOLDOUT = os.path.join(HERE, 'paired_holdout.csv')


def read_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    for need in ('input_seq', 'target_seq', 'cond'):
        if need not in (fields or []):
            raise SystemExit(f'{path} is missing column {need!r}; got {fields}')
    return fields, rows


def split(rows, frac, seed):
    by_parent = defaultdict(list)
    for r in rows:
        by_parent[r['input_seq']].append(r)

    keys = sorted(by_parent)
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_hold = int(round(len(keys) * frac))
    # Never produce an empty side on small inputs.
    n_hold = max(1, min(n_hold, len(keys) - 1)) if len(keys) > 1 else 0

    hold_keys = set(keys[:n_hold])
    train_rows = [r for k in keys if k not in hold_keys for r in by_parent[k]]
    hold_rows = [r for k in keys if k in hold_keys for r in by_parent[k]]
    return train_rows, hold_rows, by_parent, hold_keys


def write_rows(path, fields, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def describe(name, rows):
    per_cond = Counter(r['cond'] for r in rows)
    parents = {r['input_seq'] for r in rows}
    conds = ' '.join(f'cond{c}={per_cond.get(c, 0):,}'
                     for c in ('1', '2', '3', '4'))
    print(f'  {name:<8} rows={len(rows):>7,}  parents={len(parents):>6,}  {conds}')
    return parents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=DEFAULT_IN)
    ap.add_argument('--train_out', default=DEFAULT_TRAIN)
    ap.add_argument('--holdout_out', default=DEFAULT_HOLDOUT)
    ap.add_argument('--frac', type=float, default=0.15,
                    help='fraction of *parent sequences* held out')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f'input not found: {args.input}')

    fields, rows = read_rows(args.input)
    train_rows, hold_rows, by_parent, hold_keys = split(rows, args.frac, args.seed)

    print(f'Input: {args.input}  ({len(rows):,} rows, {len(by_parent):,} parents)')
    print(f'Split by parent sequence, frac={args.frac}, seed={args.seed}')
    tr_parents = describe('train', train_rows)
    ho_parents = describe('holdout', hold_rows)

    overlap = tr_parents & ho_parents
    print(f'  parent overlap: {len(overlap)}')
    if overlap:
        raise SystemExit('BUG: parent leak between train and holdout')

    if args.dry_run:
        print('\n[dry-run] nothing written')
        return

    write_rows(args.train_out, fields, train_rows)
    write_rows(args.holdout_out, fields, hold_rows)
    print(f'\nWrote {args.train_out} ({len(train_rows):,} rows)')
    print(f'Wrote {args.holdout_out} ({len(hold_rows):,} rows)')
    print('\nNext:')
    print(f'  python train.py --data {os.path.relpath(args.train_out)} --out_dir checkpoints/sft_ho')
    print(f'  python evaluate.py --parents_csv {os.path.relpath(args.holdout_out)} '
          f'--holdout 1.0 --sft_ckpt checkpoints/sft_ho/best.pt')


if __name__ == '__main__':
    main()
