"""Torch-free reward rules for stapled peptide design.

Kept separate from ``reward.py`` so that rule-based scoring and its unit
tests can run without installing torch.  ``reward.py`` re-exports these
names for backward compatibility.

Staple pairing uses a **maximum matching** rather than a greedy scan.  The
greedy version was order-dependent: given ``S5@0, S5@4, R8@7, S5@8`` it
paired ``(0,4)``, consumed position 4, and then reported a single pair while
the R8 and the last S5 were left dangling.  A maximum matching reports the
true number of crosslinks, which is what both the reward and the structural
metrics need.
"""

from typing import Dict, List, Tuple

from data.tokenizer import split_residues

STAPLE_RESIDUES = ('S5', 'R8')
MIN_TERMINAL_MARGIN = 1

# Reward weights.  ``w_valid`` used to be 0.15, which let a terminal-anchored
# staple — structurally implausible and synthetically awkward — still score
# about 0.85.  Raising it to 0.35 makes the structural constraint bite while
# keeping cond compliance the dominant term.
W_ACCURACY = 0.45
W_VALIDITY = 0.35
W_PRESERVATION = 0.20


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def _pair_gap(tok_i: str, tok_j: str) -> int:
    """Gap allowed between two staple residues, or 0 when they cannot pair."""
    if tok_i != 'S5':
        return 0
    if tok_j == 'S5':
        return 4
    if tok_j == 'R8':
        return 7
    return 0


def _max_matching(staple_pos: List[int], residues: List[str]) -> List[Tuple[int, int, int]]:
    """Largest set of disjoint staple pairs, tie-broken lexicographically.

    ``staple_pos`` is sorted; every pair must have its S5 on the left.
    Exponential in the number of staple residues, which is at most a handful
    in practice (cond 4 puts four).
    """
    if len(staple_pos) < 2:
        return []

    by_left: Dict[int, List[Tuple[int, Tuple[int, int, int]]]] = {}
    for a, i in enumerate(staple_pos):
        if residues[i] != 'S5':
            continue
        for b in range(a + 1, len(staple_pos)):
            j = staple_pos[b]
            gap = _pair_gap(residues[i], residues[j])
            if gap and j - i == gap:
                by_left.setdefault(a, []).append((b, (i, j, gap)))

    if not by_left:
        return []

    n = len(staple_pos)
    best: List[Tuple[int, int, int]] = []

    def rec(k: int, used: int, chosen: List[Tuple[int, int, int]]) -> None:
        nonlocal best
        if len(chosen) + (n - k) // 2 < len(best):
            return
        if k >= n:
            cand = sorted(chosen)
            if len(cand) > len(best) or (len(cand) == len(best) and cand < best):
                best = cand
            return
        if used >> k & 1:
            rec(k + 1, used, chosen)
            return
        rec(k + 1, used, chosen)                      # leave k unpaired
        for b, edge in by_left.get(k, ()):            # or pair k with b
            if used >> b & 1:
                continue
            rec(k + 1, used | (1 << b), chosen + [edge])

    rec(0, 0, [])
    return best


def detect_staple_pairs(seq: str) -> List[Tuple[int, int, int]]:
    """Detect staple pairs in a residue string.

    Returns a list of ``(i, j, gap)`` tuples.  ``gap`` is 4 for S5-S5
    (i,i+4) and 7 for S5-R8 (i,i+7).  Each position is used at most once,
    and the pairing maximises the number of crosslinks.
    """
    residues = split_residues(seq)
    staple_pos = [p for p, t in enumerate(residues) if t in STAPLE_RESIDUES]
    return _max_matching(staple_pos, residues)


def staple_matching_size(tokens_by_pos: Dict[int, str]) -> int:
    """Number of crosslinks a proposed placement can form.

    Used by candidate generators to reject placements that would leave a
    staple residue dangling, before the sequence is even built.
    """
    if len(tokens_by_pos) < 2:
        return 0
    staple_pos = sorted(tokens_by_pos)
    residues = ['A'] * (staple_pos[-1] + 1)
    for p, t in tokens_by_pos.items():
        residues[p] = t
    return len(_max_matching(staple_pos, residues))


def orphan_staple_rate(seq: str) -> float:
    """Fraction of S5/R8 residues that belong to no crosslink.

    An unpaired non-natural residue carries no bridge, so the design is
    chemically broken even when the pair count looks plausible.
    """
    residues = split_residues(seq)
    staple_pos = [p for p, t in enumerate(residues) if t in STAPLE_RESIDUES]
    if not staple_pos:
        return 0.0
    paired = 2 * len(_max_matching(staple_pos, residues))
    return (len(staple_pos) - paired) / len(staple_pos)


# ---------------------------------------------------------------------------
# Structural scores
# ---------------------------------------------------------------------------

def helical_angle(pos: int) -> float:
    """Idealised alpha-helix wheel angle (100 degrees per residue)."""
    return ((pos - 1) * 100) % 360


def angular_diff(a1: float, a2: float) -> float:
    d = abs(a1 - a2)
    return min(d, 360 - d)


def staple_accuracy(seq: str, cond: int) -> float:
    """Fractional match between the requested condition and the design.

    cond 1: no staple; cond 2: at least one i,i+4; cond 3: at least one
    i,i+7; cond 4: two staple pairs.
    """
    pairs = detect_staple_pairs(seq)
    if cond == 1:
        return 1.0 if len(pairs) == 0 else 0.0
    if cond == 2:
        if len(pairs) == 0:
            return 0.0
        n4 = sum(1 for p in pairs if p[2] == 4)
        return min(1.0, n4)
    if cond == 3:
        if len(pairs) == 0:
            return 0.0
        i7 = sum(1 for p in pairs if p[2] == 7)
        return min(1.0, i7)
    if cond == 4:
        return min(1.0, len(pairs) / 2.0)
    return 0.0


def staple_site_validity(seq: str) -> float:
    """Fraction of staple pairs that are structurally plausible.

    A pair is valid when it:
      * spans i,i+4 or i,i+7,
      * keeps at least ``MIN_TERMINAL_MARGIN`` residues from both termini,
      * sits on the same helical face (angular difference < 60 degrees).

    The terminal check matters because a staple anchored on the first or
    last residue is synthetically awkward and removes a terminal charge.
    """
    pairs = detect_staple_pairs(seq)
    if not pairs:
        return 1.0

    n_residues = len(split_residues(seq))
    valid = 0
    for i, j, gap in pairs:
        if gap not in (4, 7):
            continue
        if i < MIN_TERMINAL_MARGIN or j > n_residues - 1 - MIN_TERMINAL_MARGIN:
            continue
        if angular_diff(helical_angle(i), helical_angle(j)) < 60:
            valid += 1
    return valid / len(pairs)


def preservation_score(seq: str, input_seq: str) -> float:
    """Fraction of parent scaffold positions kept unchanged.

    Staple positions are not scaffold mutations: a residue replaced by
    S5/R8 is skipped instead of being counted as a mismatch.  Positions
    where the parent already carries a staple are skipped as well.
    """
    tokens_seq = split_residues(seq)
    tokens_inp = split_residues(input_seq)
    L = min(len(tokens_seq), len(tokens_inp))
    if L == 0:
        return 0.0

    kept = 0
    total = 0
    for cand, parent in zip(tokens_seq[:L], tokens_inp[:L]):
        if cand in STAPLE_RESIDUES or parent in STAPLE_RESIDUES:
            continue
        total += 1
        if cand == parent:
            kept += 1

    return kept / total if total else 1.0


def compute_reward(seq: str, input_seq: str, cond: int) -> float:
    """Rule-based scalar reward used by the RWR baseline."""
    acc = staple_accuracy(seq, cond)
    valid = staple_site_validity(seq)
    pres = preservation_score(seq, input_seq)
    return (W_ACCURACY * acc
            + W_VALIDITY * valid
            + W_PRESERVATION * pres)
