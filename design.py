"""Staple candidate generation shared by GRPO training and evaluation.

``design_candidates`` enumerates every structurally valid staple placement
for a parent AMP, ranks them by the model's log-probability, and returns the
best ``K``.  Extracted from ``train_rl_grpo.py`` so that evaluation exercises
exactly the same inference path the RL loop optimises against.
"""

from typing import Tuple

import torch

from reward_rules import staple_matching_size

S5_TOKEN = 21
R8_TOKEN = 22

COND_NO_STAPLE = 1
COND_I_I4 = 2
COND_I_I7 = 3
COND_DOUBLE = 4


@torch.no_grad()
def design_candidates(
    amp_tokens: torch.Tensor,
    cond: torch.Tensor,
    log_probs: torch.Tensor,
    K: int,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Generate K candidates per sample via full enumeration.

    Enumerates ALL valid staple positions (not just top-k), ranks by
    log-probability, and returns the K best.  No extra forward pass.
    If fewer than K unique candidates exist, remaining slots are filled
    with the original AMP (no staples).

    Args:
        amp_tokens: [B, L] parent tokens (PAD = 0)
        cond:       [B] condition labels
        log_probs:  [B, L, V] reference log-softmax
        K:          candidates per parent

    Returns:
        candidates:    [B*K, L] candidate sequences
        group_indices: [B*K] batch-index each candidate belongs to
    """
    B, L = amp_tokens.shape
    is_pad = (amp_tokens == 0)

    all_candidates = []
    all_groups = []

    for i in range(B):
        c = cond[i].item()
        cands = []

        if c == COND_NO_STAPLE:
            for _ in range(K):
                all_candidates.append(amp_tokens[i:i + 1].clone())
                all_groups.append(i)
            continue

        if c == COND_I_I4:
            for a in range(L - 4):
                if is_pad[i, a] or is_pad[i, a + 4]:
                    continue
                seq = amp_tokens[i:i + 1].clone()
                seq[0, a] = S5_TOKEN
                seq[0, a + 4] = S5_TOKEN
                conf = (log_probs[i, a, S5_TOKEN] + log_probs[i, a + 4, S5_TOKEN]).item()
                cands.append((seq, conf))

        elif c == COND_I_I7:
            for a in range(L - 7):
                if is_pad[i, a] or is_pad[i, a + 7]:
                    continue
                seq = amp_tokens[i:i + 1].clone()
                seq[0, a] = S5_TOKEN
                seq[0, a + 7] = R8_TOKEN
                conf = (log_probs[i, a, S5_TOKEN] + log_probs[i, a + 7, R8_TOKEN]).item()
                cands.append((seq, conf))

        elif c == COND_DOUBLE:
            valid_pos = (~is_pad[i]).nonzero(as_tuple=True)[0].tolist()
            seen = set()
            for a1 in valid_pos:
                for gap1 in (4, 7):
                    j1 = a1 + gap1
                    if j1 >= L or is_pad[i, j1]:
                        continue
                    tok1, tok2 = (S5_TOKEN, S5_TOKEN) if gap1 == 4 else (S5_TOKEN, R8_TOKEN)
                    for a2 in valid_pos:
                        if a2 == a1:
                            continue
                        for gap2 in (4, 7):
                            j2 = a2 + gap2
                            if j2 >= L or is_pad[i, j2]:
                                continue
                            if len({a1, j1, a2, j2}) < 4:
                                continue
                            key = tuple(sorted([a1, j1, a2, j2]))
                            if key in seen:
                                continue
                            seen.add(key)
                            tok3, tok4 = (S5_TOKEN, S5_TOKEN) if gap2 == 4 else (S5_TOKEN, R8_TOKEN)
                            # Reject placements that would leave a staple residue
                            # without a partner: two crosslinks must be realisable.
                            placement = {
                                a1: 'S5',
                                j1: 'S5' if gap1 == 4 else 'R8',
                                a2: 'S5',
                                j2: 'S5' if gap2 == 4 else 'R8',
                            }
                            if len(placement) < 4 or staple_matching_size(placement) < 2:
                                continue
                            seq = amp_tokens[i:i + 1].clone()
                            seq[0, a1] = tok1
                            seq[0, j1] = tok2
                            seq[0, a2] = tok3
                            seq[0, j2] = tok4
                            conf = (
                                log_probs[i, a1, tok1]
                                + log_probs[i, j1, tok2]
                                + log_probs[i, a2, tok3]
                                + log_probs[i, j2, tok4]
                            ).item()
                            cands.append((seq, conf))

        cands.sort(key=lambda x: x[1], reverse=True)
        for seq, _ in cands[:K]:
            all_candidates.append(seq)
            all_groups.append(i)

        n_pad = K - min(len(cands), K)
        for _ in range(n_pad):
            all_candidates.append(amp_tokens[i:i + 1].clone())
            all_groups.append(i)

    candidates = torch.cat(all_candidates, dim=0)
    group_indices = torch.tensor(all_groups, dtype=torch.long, device=amp_tokens.device)
    return candidates, group_indices
