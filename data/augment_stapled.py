"""从 peptide_data.csv 生成订书肽模拟数据.

管线:
  原始 AMP (label=1) → 枚举 i,i+4 / i,i+7 订书位 → 螺旋面过滤 → S5/R8 替换 → CSV

关键设计（2026-10-19 修正）
--------------------------
旧版本对每个父序列只保留**一个** cond 2 / cond 3 目标，且候选排序里含有
``-abs(i - L/4)`` 这一人为位置先验。结果训练集里每个父序列只有唯一解，
模型学到的是「把 S5 放在 round(n/4)」这条查表规则，而不是订书位点分布。

现在改为从**全部合法候选**中均匀无放回采样 ``SAMPLES_PER_COND`` 个位点，
并去掉了位置先验项。双订书组合额外要求两组交叉键都能真实成键（不允许
出现孤儿 S5/R8 残基）。

输出格式: stapled_peptide_data.csv
  sequence  - 含 S5/R8 的订书肽序列 (原始 AMP 用标准 AA)
  label     - AMP 活性 (0/1, 继承自原始数据)
  cond      - 条件标签 (0=neg, 1=pos, 2=pos+i,i+4, 3=pos+i,i+7, 4=pos+双订书)
"""

import csv
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer import split_residues
from reward_rules import MIN_TERMINAL_MARGIN, staple_matching_size

random.seed(42)

# ================================================================
# 配置
# ================================================================
INPUT_CSV = os.path.join(os.path.dirname(__file__), 'peptide_data.csv')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), 'stapled_peptide_data.csv')

# 每个父序列、每种单订书类型采样多少个位点
SAMPLES_PER_COND = 4
# 每个父序列采样多少个双订书组合
SAMPLES_DOUBLE = 2
# 各类别的目标数量上限 (None=不限制)
CAP_PER_CLASS = {
    0: None,   # neg 全保留
    1: None,   # pos 全保留
    2: 20000,  # pos+i,i+4
    3: 20000,  # pos+i,i+7
    4: 12000,  # pos+双订书
}
# 最小序列长度 (太短无法订书)
MIN_LENGTH = 8
# 订书间距
STAPLE_GAPS = [4, 7]
# 螺旋倾向较强的残基 (仅用于候选排序, 不参与采样权重)
HELIX_FAVOURING = frozenset('AELMQKFI')

# ================================================================
# 螺旋面朝向过滤
# ================================================================

def helical_angle(pos: int) -> float:
    """α-螺旋中第 pos 位残基的螺旋轮角度 (度)."""
    return ((pos - 1) * 100) % 360


def angular_diff(angle1: float, angle2: float) -> float:
    """两个角度间的最小夹角."""
    d = abs(angle1 - angle2)
    return min(d, 360 - d)


def is_helical_face_compatible(i: int, j: int, max_angular_diff: float = 60.0) -> bool:
    """判断 i,j 是否在 α-螺旋的同一侧面 (订书可行)."""
    return angular_diff(helical_angle(i), helical_angle(j)) < max_angular_diff


def is_valid_staple_position(residues: list, i: int, j: int) -> bool:
    """订书位点基本约束: 非 Pro/Gly, 长度边界, 端部余量."""
    L = len(residues)
    if i < 0 or j >= L:
        return False
    if residues[i] in ('P', 'G') or residues[j] in ('P', 'G'):
        return False
    # 与 reward_rules.staple_site_validity 保持一致: 首尾各留一个残基,
    # 否则会生成结构上不可行、且会移除末端电荷的订书
    if i < MIN_TERMINAL_MARGIN or j > L - 1 - MIN_TERMINAL_MARGIN:
        return False
    return True

# ================================================================
# 订书位点枚举
# ================================================================

def _site_score(residues: list, candidate) -> float:
    """候选质量分: 区间内螺旋倾向残基越多越好.

    刻意**不含**任何位置项 —— 旧版的 ``-abs(i - L/4) * 0.5`` 会把所有
    父序列的订书都推到四分之一处, 正是训练集退化成唯一解的根因.
    """
    i, j, _ = candidate
    return float(sum(1 for p in range(i, j + 1) if residues[p] in HELIX_FAVOURING))


def enumerate_staple_candidates(seq: str) -> list:
    """枚举序列中所有可行的订书候选位.

    Returns:
        [(i, j, gap), ...]  其中 gap=j-i, 已按质量评分降序排列
    """
    residues = split_residues(seq)
    L = len(residues)
    candidates = []

    for gap in STAPLE_GAPS:
        for i in range(L - gap):
            j = i + gap
            if not is_valid_staple_position(residues, i, j):
                continue
            if not is_helical_face_compatible(i, j):
                continue
            candidates.append((i, j, gap))

    candidates.sort(key=lambda c: _site_score(residues, c), reverse=True)
    return candidates

# ================================================================
# 序列构造
# ================================================================

def generate_staple_pair_seq(seq: str, i: int, j: int, gap: int) -> str:
    """将序列中 i,j 替换为订书 AA, 返回新序列.

    gap=4 → S5,S5; gap=7 → S5,R8
    """
    residues = split_residues(seq)
    residues[i] = 'S5'
    residues[j] = 'R8' if gap == 7 else 'S5'
    return ''.join(residues)


def generate_double_staple_seq(seq: str, pair1, pair2) -> str:
    """生成双订书序列 (两组不重叠的订书对)."""
    residues = split_residues(seq)
    for i, j, gap in (pair1, pair2):
        residues[i] = 'S5'
        residues[j] = 'R8' if gap == 7 else 'S5'
    return ''.join(residues)


def is_overlapping(pair1, pair2):
    """检查两个订书对是否重叠."""
    (i1, j1, _) = pair1
    (i2, j2, _) = pair2
    return not (j1 < i2 or j2 < i1)


def _placement_of(pair1, pair2) -> dict:
    """位置 -> 订书残基名, 供成键检查使用."""
    placement = {}
    for i, j, gap in (pair1, pair2):
        placement[i] = 'S5'
        placement[j] = 'S5' if gap == 4 else 'R8'
    return placement

# ================================================================
# 采样
# ================================================================

def sample_sites(candidates: list, n: int, rng: random.Random) -> list:
    """从合法候选中均匀无放回采样 n 个位点.

    均匀是关键: 任何按质量分取 top-n 的做法都会重新引入位置/序列偏好,
    让模型重新退化成「总是选同一个区域」。
    """
    if not candidates:
        return []
    if len(candidates) <= n:
        return list(candidates)
    return rng.sample(candidates, n)


def sample_double_combos(candidates: list, n: int, rng: random.Random,
                         max_tries: int = 400) -> list:
    """采样 n 组不重叠、且两组交叉键都真实成立的订书组合."""
    out = []
    if len(candidates) < 2:
        return out
    seen = set()
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        p1, p2 = rng.sample(candidates, 2)
        if is_overlapping(p1, p2):
            continue
        placement = _placement_of(p1, p2)
        if len(placement) < 4:
            continue
        # 两组交叉键必须都能成键, 否则会留下孤儿 S5/R8
        if staple_matching_size(placement) < 2:
            continue
        key = tuple(sorted(placement))
        if key in seen:
            continue
        seen.add(key)
        out.append((p1, p2))
    return out

# ================================================================
# 主增强管线
# ================================================================

def augment_sequence(seq: str, label: int, rng: random.Random = None) -> list:
    """对单条序列生成多个订书变体.

    Returns:
        [(seq_str, label, cond), ...]
    """
    rng = rng or random

    if label == 0:
        # 阴性序列: 只保留原始 (cond=0), 不生成订书变体
        return [(seq, 0, 0)]

    results = [(seq, 1, 1)]  # 原始 AMP (cond=1)

    if len(split_residues(seq)) < MIN_LENGTH:
        return results

    candidates = enumerate_staple_candidates(seq)
    if not candidates:
        return results

    s4 = [c for c in candidates if c[2] == 4]
    s7 = [c for c in candidates if c[2] == 7]

    for group, cond in ((s4, 2), (s7, 3)):
        for i, j, gap in sample_sites(group, SAMPLES_PER_COND, rng):
            results.append((generate_staple_pair_seq(seq, i, j, gap), 1, cond))

    for p1, p2 in sample_double_combos(candidates, SAMPLES_DOUBLE, rng):
        results.append((generate_double_staple_seq(seq, p1, p2), 1, 4))

    return results

# ================================================================
# 入口
# ================================================================

def run(input_path=INPUT_CSV, output_path=OUTPUT_CSV, seed: int = 42):
    """执行数据增强并输出 CSV."""
    print(f"Reading: {input_path}")
    rng = random.Random(seed)

    all_rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append((row['sequence'].strip(), int(row['label'])))

    pos_count = sum(1 for _, l in all_rows if l == 1)
    neg_count = sum(1 for _, l in all_rows if l == 0)
    print(f"  Total: {len(all_rows)} sequences")
    print(f"  Pos: {pos_count}, Neg: {neg_count}")

    augmented = []
    class_counts = Counter()

    for idx, (seq, label) in enumerate(all_rows):
        for var_seq, var_label, var_cond in augment_sequence(seq, label, rng):
            cap = CAP_PER_CLASS.get(var_cond)
            if cap is not None and class_counts[var_cond] >= cap:
                continue
            augmented.append((var_seq, var_label, var_cond))
            class_counts[var_cond] += 1

        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(all_rows)}...")

    print(f"\nWriting: {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'label', 'cond'])
        for seq, label, cond in augmented:
            writer.writerow([seq, label, cond])

    print(f"\n{'=' * 50}")
    print(f"Augmentation Complete")
    print(f"{'=' * 50}")
    print(f"  Total: {len(augmented)} sequences")
    print(f"  Class distribution:")
    cond_names = {0: 'neg', 1: 'pos', 2: 'pos+i,i+4', 3: 'pos+i,i+7', 4: 'pos+双订书'}
    for cond in sorted(class_counts.keys()):
        print(f"    cond={cond} ({cond_names.get(cond, '?')}): {class_counts[cond]:>6}")

    return augmented


if __name__ == '__main__':
    run()
