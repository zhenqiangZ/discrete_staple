"""Tensor-side rewards: position-alignment reward on parent sequences.

Rule-based rewards live in ``reward_rules.py`` (torch-free) and are
re-exported here so existing ``from reward import ...`` calls keep working.
"""

from typing import List, Tuple

import torch

from data.tokenizer import decode
from reward_rules import (  # noqa: F401  (re-exported for compatibility)
    compute_reward,
    detect_staple_pairs,
    preservation_score,
    staple_accuracy,
    staple_site_validity,
)

__all__ = [
    'compute_reward',
    'detect_staple_pairs',
    'preservation_score',
    'staple_accuracy',
    'staple_site_validity',
    'compute_batch_rewards_alignment',
]


def compute_batch_rewards_alignment(
    candidates: torch.Tensor,
    input_strs: List[str],
    conds: List[int],
    amp_tokens: torch.Tensor,
    group_indices: torch.Tensor,
    amp_align: torch.Tensor,
    tox_align: torch.Tensor,
    w_amp: float = 0.30,
    w_tox: float = 0.30,
    w_staple: float = 0.20,
    w_pres: float = 0.20,
    impact_scale: float = 100.0,
) -> Tuple[torch.Tensor, dict]:
    """Reward based on parent-sequence position alignment.

    For each parent position i:
      align_amp[i] = dot(d(amp_logit)/d(emb_i), emb_i) / ||emb_i||
      align_tox[i] = dot(d(tox_logit)/d(emb_i), emb_i) / ||emb_i||

    A candidate that mutates position i gets the alignment value of the
    *parent* at that position interpreted as:

      align_amp[i] > 0  ->  current AA at i helps AMP activity
                           -> mutating it away is bad -> penalty
      align_amp[i] < 0  ->  current AA at i hurts AMP activity
                           -> mutating it away is good -> reward

      align_tox[i] > 0  ->  current AA at i increases toxicity
                           -> mutating it away is good -> reward
      align_tox[i] < 0  ->  current AA at i reduces toxicity
                           -> mutating it away is bad -> penalty

    Reward formula:
      mean_amp = mean over mutated positions of amp_align
      mean_tox = mean over mutated positions of tox_align

      r_amp  = -tanh(mean_amp * scale)
      r_tox  =  tanh(mean_tox * scale)
      reward = w_pres * pres + r_amp + r_tox

    Note: the staple-accuracy term is currently disabled; ``w_amp``,
    ``w_tox`` and ``w_staple`` are kept for API compatibility.

    Args:
        candidates:    [B*K, L] token ids (diffusion vocab)
        input_strs:    parent AMP decoded strings [B*K]
        conds:         condition per candidate [B*K]
        amp_tokens:    [B, L] parent tokens (diffusion vocab)
        group_indices: [B*K] which batch index each candidate belongs to
        amp_align:     [B, L] pre-computed amp alignment on parent seq
        tox_align:     [B, L] pre-computed tox alignment on parent seq
        w_*:           component weights (unused, kept for compatibility)
        impact_scale:  tanh scaling factor
    Returns:
        rewards:  [B*K] scalar rewards
        stats:    dict of components for logging
    """
    BK = candidates.shape[0]
    device = candidates.device

    # ---- Staple + preservation (string-based) ----
    r_staple = torch.zeros(BK, device=device)
    r_pres = torch.zeros(BK, device=device)
    for i in range(BK):
        seq = decode(candidates[i].tolist())
        r_staple[i] = staple_accuracy(seq, conds[i])
        r_pres[i] = preservation_score(seq, input_strs[i])

    # ---- Alignment-based reward (tensor-based) ----
    # For each candidate, find positions mutated from parent (excluding PAD)
    parent_exp = amp_tokens.to(candidates.device)[group_indices]
    mutated = (candidates != parent_exp) & (parent_exp != 0)   # [BK, L]

    amp_align_exp = amp_align.to(candidates.device)[group_indices]   # [BK, L]
    tox_align_exp = tox_align.to(candidates.device)[group_indices]

    # Mean alignment per candidate, only over mutated positions
    amp_mean = (amp_align_exp * mutated.float()).sum(dim=-1) / (mutated.float().sum(dim=-1) + 1e-8)
    tox_mean = (tox_align_exp * mutated.float()).sum(dim=-1) / (mutated.float().sum(dim=-1) + 1e-8)

    r_amp = -torch.tanh(amp_mean * impact_scale)
    r_tox = torch.tanh(tox_mean * impact_scale)

    # ---- Combined ----
    rewards = r_pres * w_pres + r_amp + r_tox

    stats = {
        'r_staple': r_staple.mean().item(),
        'r_pres': r_pres.mean().item(),
        'r_amp': r_amp.mean().item(),
        'r_tox': r_tox.mean().item(),
        'amp_align_mean': amp_mean.mean().item(),
        'tox_align_mean': tox_mean.mean().item(),
        'frac_mutated': mutated.float().mean().item(),
    }
    return rewards.detach(), stats
