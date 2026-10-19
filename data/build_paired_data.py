"""从 peptide_data.csv 生成 (input_seq, target_seq, cond) 配对数据.

每个父序列会产出多个订书变体（见 ``augment_stapled.SAMPLES_PER_COND``），
而不是旧版的「每种条件只取一个排序第一的候选」。后者让训练集退化成
唯一解，SFT 只能学到一条查表规则。

用法:
    python data/build_paired_data.py
"""

import csv
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.augment_stapled import (
    SAMPLES_DOUBLE, SAMPLES_PER_COND, augment_sequence,
)

INPUT_CSV = os.path.join(os.path.dirname(__file__), 'peptide_data.csv')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), 'paired_data.csv')


def build_pairs_for_amp(seq: str, rng: random.Random = None) -> list:
    """一条 AMP 的全部 (input, target, cond) 配对。"""
    return [(seq, target, cond)
            for target, _label, cond in augment_sequence(seq, 1, rng)]


def run(input_path=INPUT_CSV, output_path=OUTPUT_CSV, seed: int = 42):
    print(f"Reading: {input_path}")
    rng = random.Random(seed)

    all_rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            all_rows.append((row['sequence'].strip(), int(row['label'])))

    pos_seqs = [seq for seq, label in all_rows if label == 1]
    print(f"  Positive sequences: {len(pos_seqs)}")

    paired = []
    for idx, seq in enumerate(pos_seqs):
        paired.extend(build_pairs_for_amp(seq, rng))
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1}/{len(pos_seqs)}... pairs so far: {len(paired)}")

    cond_counts = Counter(cond for _, _, cond in paired)
    per_parent = Counter(cond for _, _, cond in paired)

    print(f"\nWriting: {output_path}")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['input_seq', 'target_seq', 'cond'])
        for inp, tgt, cond in paired:
            writer.writerow([inp, tgt, cond])

    cond_names = {1: 'self', 2: 'i,i+4', 3: 'i,i+7', 4: 'double'}
    print(f"\n{'=' * 50}")
    print(f"Paired Data Generation Complete")
    print(f"{'=' * 50}")
    print(f"  Total pairs: {len(paired)}")
    for cond in sorted(cond_counts):
        print(f"    cond={cond} ({cond_names.get(cond, '?')}): {cond_counts[cond]:>7}"
              f"  (~{cond_counts[cond] / max(1, len(pos_seqs)):.2f} per parent)")

    # 每个父序列每种条件的平均目标数 —— 1.0 意味着训练集退化成唯一解
    if len(pos_seqs):
        expected = {2: SAMPLES_PER_COND, 3: SAMPLES_PER_COND, 4: SAMPLES_DOUBLE}
        print(f"\n  Target diversity check (want > 1.0 for cond 2/3/4):")
        for cond in (2, 3, 4):
            actual = cond_counts.get(cond, 0) / len(pos_seqs)
            flag = 'OK' if actual > 1.0 else 'LOW'
            print(f"    cond={cond}: {actual:.2f} targets/parent "
                  f"(cap {expected[cond]}) [{flag}]")


if __name__ == '__main__':
    run()
