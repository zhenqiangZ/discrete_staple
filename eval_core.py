"""Torch-free evaluation core: metrics and baselines for stapled peptide design.

Kept free of torch/numpy so the metric definitions can be unit-tested and
reused by reporting code without the training environment.

Two responsibilities live here:

* **metrics** — cond compliance, staple validity, scaffold preservation,
  diversity, and property aggregation;
* **baselines** — ``random_staple_placement``, the model-free reference that
  the trained systems are compared against.
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

from compute_properties import compute_all
from data.tokenizer import split_residues
from reward_rules import (
    compute_reward,
    detect_staple_pairs,
    orphan_staple_rate,   # re-exported: defined with the pairing rules
    preservation_score,
    staple_accuracy,
    staple_site_validity,
)

S5 = 'S5'
R8 = 'R8'

_STAPLE_TOKENS = (S5, R8)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sequence_identity(a: str, b: str) -> float:
    """Residue-level identity over the overlapping prefix."""
    ra, rb = split_residues(a), split_residues(b)
    n = min(len(ra), len(rb))
    if n == 0:
        return 0.0
    return sum(1 for x, y in zip(ra[:n], rb[:n]) if x == y) / n


def unique_ratio(seqs: Sequence[str]) -> float:
    """Fraction of distinct sequences (1.0 = all different)."""
    return len(set(seqs)) / len(seqs) if seqs else 0.0


def mean_pairwise_identity(seqs: Sequence[str], max_pairs: int = 2000,
                           rng: Optional[random.Random] = None) -> float:
    """Mean pairwise identity, i.e. ``1 - diversity``.

    Exhaustive for small sets; random pair sampling above ``max_pairs`` so
    the cost stays bounded for thousands of designs.
    """
    n = len(seqs)
    if n < 2:
        return 1.0

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        vals = [sequence_identity(seqs[i], seqs[j])
                for i in range(n) for j in range(i + 1, n)]
        return _mean(vals)

    rng = rng or random
    vals = []
    for _ in range(max_pairs):
        i, j = rng.sample(range(n), 2)
        vals.append(sequence_identity(seqs[i], seqs[j]))
    return _mean(vals)


def has_terminal_anchor(seq: str) -> bool:
    """True when at least one staple pair touches the first or last residue."""
    pairs = detect_staple_pairs(seq)
    if not pairs:
        return False
    n = len(split_residues(seq))
    return any(i == 0 or j == n - 1 for i, j, _ in pairs)


def count_staples(seq: str) -> int:
    return len(detect_staple_pairs(seq))


def mean_staple_offset(seq: str) -> float:
    """Mean start position of staple pairs, normalised to ``[0, 1]``.

    Reveals positional bias: ~0 means staples cluster at the N-terminus,
    ~1 at the C-terminus, ~0.5 mid-sequence.  Returns ``nan`` when the
    sequence carries no staple or is too short to normalise.
    """
    pairs = detect_staple_pairs(seq)
    n = len(split_residues(seq))
    if not pairs or n < 2:
        return float('nan')
    return sum(i / (n - 1) for i, _, _ in pairs) / len(pairs)


# ---------------------------------------------------------------------------
# Design-level metrics
# ---------------------------------------------------------------------------

_EMPTY_DESIGN_SUMMARY = {
    'n': 0,
    'cond_compliance': 0.0,
    'staple_validity': 0.0,
    'valid_rate': 0.0,
    'preservation': 0.0,
    'rule_reward': 0.0,
    'mean_pairs': 0.0,
    'terminal_anchor_rate': 0.0,
    'mean_staple_offset': 0.0,
    'orphan_rate': 0.0,
    'fallback_rate': 0.0,
}


def summarize_designs(designs: Sequence[str], parents: Sequence[str],
                      cond: int) -> Dict[str, float]:
    """Structural metrics for a set of designs paired with their parents.

    ``fallback_rate`` counts designs identical to the parent, which is how
    the enumeration-based generator fills slots when it cannot find K
    distinct candidates.  A high value means the method silently produced
    no design for that slot.

    ``mean_staple_offset`` is the only positional metric here that separates
    two enumeration-based systems: every enumerated candidate is compliant
    and face-valid by construction, so what actually differs is *where* the
    model chooses to put the staple.

    ``rule_reward`` is ``compute_reward`` averaged over designs — the same
    scalar the RWR baseline optimises, so it ties this report back to the
    training objective.
    """
    n = len(designs)
    if n == 0 or n != len(parents):
        return dict(_EMPTY_DESIGN_SUMMARY)

    acc = [staple_accuracy(d, cond) for d in designs]
    val = [staple_site_validity(d) for d in designs]
    pres = [preservation_score(d, p) for d, p in zip(designs, parents)]
    reward = [compute_reward(d, p, cond) for d, p in zip(designs, parents)]
    pairs = [count_staples(d) for d in designs]
    term = [1.0 if has_terminal_anchor(d) else 0.0 for d in designs]
    fallback = [1.0 if d == p else 0.0 for d, p in zip(designs, parents)]
    orphan = [orphan_staple_rate(d) for d in designs]
    offsets = [mean_staple_offset(d) for d in designs]
    offsets = [o for o in offsets if o == o]  # drop NaN (no staple)

    return {
        'n': n,
        'cond_compliance': _mean(acc),
        'staple_validity': _mean(val),
        'valid_rate': _mean([1.0 if v >= 1.0 - 1e-9 else 0.0 for v in val]),
        'preservation': _mean(pres),
        'rule_reward': _mean(reward),
        'mean_pairs': _mean(pairs),
        'terminal_anchor_rate': _mean(term),
        'mean_staple_offset': _mean(offsets),
        'orphan_rate': _mean(orphan),
        'fallback_rate': _mean(fallback),
    }


def summarize_properties(seqs: Sequence[str]) -> Dict[str, float]:
    """Mean physicochemical properties over a set of sequences."""
    if not seqs:
        return {'charge': 0.0, 'hydrophobicity': 0.0,
                'hmoment': 0.0, 'mw': 0.0}
    props = [compute_all(s) for s in seqs]
    return {
        'charge': _mean([p['charge'] for p in props]),
        'hydrophobicity': _mean([p['hydrophobicity'] for p in props]),
        'hmoment': _mean([p['hmoment'] for p in props]),
        'mw': _mean([p['mw'] for p in props]),
    }


def summarize_diversity(seqs: Sequence[str],
                        rng: Optional[random.Random] = None) -> Dict[str, float]:
    """Uniqueness and mean pairwise identity of a design set."""
    return {
        'unique_ratio': unique_ratio(seqs),
        'mean_pairwise_identity': mean_pairwise_identity(seqs, rng=rng),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _apply_pair(residues: List[str], i: int, j: int) -> None:
    """Write an S5-S5 (gap 4) or S5-R8 (gap 7) staple into ``residues``."""
    gap = j - i
    residues[i] = S5
    residues[j] = S5 if gap == 4 else R8


def random_staple_placement(seq: str, cond: int,
                            rng: Optional[random.Random] = None) -> str:
    """Model-free baseline: place staples uniformly at random.

    Deliberately *unbiased* — terminal positions are allowed, because the
    point of the baseline is to show what a method that ignores structure
    would produce.  Returns the parent unchanged when the sequence is too
    short for the requested condition.
    """
    rng = rng or random
    residues = split_residues(seq)
    n = len(residues)
    if cond == 1:
        return ''.join(residues)

    if cond == 2:
        if n < 5:
            return ''.join(residues)
        i = rng.randrange(0, n - 4)
        _apply_pair(residues, i, i + 4)
        return ''.join(residues)

    if cond == 3:
        if n < 8:
            return ''.join(residues)
        i = rng.randrange(0, n - 7)
        _apply_pair(residues, i, i + 7)
        return ''.join(residues)

    if cond == 4:
        if n < 9:
            return ''.join(residues)
        # Rejection-sample two disjoint pairs; bounded so it always returns.
        for _ in range(200):
            gap1 = rng.choice((4, 7))
            gap2 = rng.choice((4, 7))
            if n <= gap1 or n <= gap2:
                continue
            a1 = rng.randrange(0, n - gap1)
            a2 = rng.randrange(0, n - gap2)
            j1, j2 = a1 + gap1, a2 + gap2
            if len({a1, j1, a2, j2}) < 4:
                continue
            _apply_pair(residues, a1, j1)
            _apply_pair(residues, a2, j2)
            return ''.join(residues)
        return ''.join(residues)

    return ''.join(residues)


def random_designs(parents: Sequence[str], cond: int, k: int,
                   rng: Optional[random.Random] = None) -> List[str]:
    """``k`` random designs per parent, flattened (parent-major order)."""
    rng = rng or random
    out: List[str] = []
    for parent in parents:
        for _ in range(k):
            out.append(random_staple_placement(parent, cond, rng))
    return out


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def flatten_parents(parents: Sequence[str], k: int) -> List[str]:
    """Repeat each parent ``k`` times so it lines up with flattened designs."""
    return [p for p in parents for _ in range(k)]
