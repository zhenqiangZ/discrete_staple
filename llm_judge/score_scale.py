"""Score-scale normalisation for judge records.

Two record generations exist:

``raw``
    Absolute 0-10 scores produced by the current pipeline.  ``final_scores``
    is directly usable as a regression target.

``zscore`` (legacy)
    Records written before the scale fix.  ``final_scores`` holds per-group
    z-scores, which are relative and therefore useless as an absolute target.
    Fortunately the same record still embeds the per-candidate scores from
    phases 1 and 3, so the absolute values can be recovered exactly.

This module is torch- and numpy-free so any consumer can use it.
"""
from typing import List, Optional, Tuple

SCALE_RAW = 'raw'
SCALE_RECOVERED = 'recovered'
SCALE_LEGACY = 'zscore'

_MIN_SCORE = 0.0
_MAX_SCORE = 10.0


def _indexed_scores(results, key: str) -> Optional[dict]:
    """Flatten a phase result payload into ``{candidate_idx: score}``."""
    if isinstance(results, dict):
        for value in results.values():
            if isinstance(value, list):
                results = value
                break
        else:
            return None
    if not isinstance(results, list) or not results:
        return None

    out = {}
    for pos, item in enumerate(results):
        if isinstance(item, dict):
            idx = item.get('idx', pos)
            raw = item.get(key)
            if raw is None:
                raw = item.get('final_score', item.get('score'))
        else:
            idx = pos
            raw = item
        try:
            out[int(idx)] = float(raw)
        except (TypeError, ValueError):
            return None
    return out


def _clamp(values: List[float]) -> List[float]:
    return [max(_MIN_SCORE, min(_MAX_SCORE, v)) for v in values]


def recover_raw_scores(record: dict) -> Optional[List[float]]:
    """Rebuild absolute 0-10 scores for a legacy z-score record.

    Preference order: phase-3 final judge, then the mean of the two phase-1
    scorers, then whichever single phase-1 scorer is complete.  Returns
    ``None`` when no complete set of candidate scores can be found.
    """
    candidates = record.get('candidates') or []
    n = len(candidates)
    if n == 0:
        return None

    phase3 = record.get('phase3') or {}
    llm_c = (phase3.get('llm_c') or {}).get('results')
    for key in ('final_score', 'score'):
        mapping = _indexed_scores(llm_c, key)
        if mapping is not None and len(mapping) == n and all(i in mapping for i in range(n)):
            return _clamp([mapping[i] for i in range(n)])

    phase1 = record.get('phase1') or {}
    a = _indexed_scores((phase1.get('llm_a') or {}).get('results'), 'score')
    b = _indexed_scores((phase1.get('llm_b') or {}).get('results'), 'score')
    a_ok = a is not None and len(a) == n and all(i in a for i in range(n))
    b_ok = b is not None and len(b) == n and all(i in b for i in range(n))

    if a_ok and b_ok:
        return _clamp([(a[i] + b[i]) / 2.0 for i in range(n)])
    if a_ok:
        return _clamp([a[i] for i in range(n)])
    if b_ok:
        return _clamp([b[i] for i in range(n)])
    return None


def resolve_scores(record: dict) -> Tuple[Optional[List[float]], Optional[str]]:
    """Return ``(scores, scale)`` with ``scale`` in {raw, recovered, None}."""
    stored = record.get('final_scores') or []
    scale = record.get('score_scale', SCALE_LEGACY)

    if scale == SCALE_RAW:
        try:
            values = [float(s) for s in stored]
        except (TypeError, ValueError):
            return None, None
        if any(v != v for v in values):  # NaN
            return None, None
        return values, SCALE_RAW

    recovered = recover_raw_scores(record)
    if recovered is None:
        return None, None
    return recovered, SCALE_RECOVERED
