import torch
from typing import Tuple

from config import MASK_TOKEN


def random_mask(x: torch.Tensor, mask_rate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L = x.shape
    is_pad = (x == 0)
    mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)

    for i in range(B):
        valid_pos = (~is_pad[i]).nonzero(as_tuple=True)[0]
        if valid_pos.numel() == 0:
            continue
        n_valid = valid_pos.numel()
        rate = mask_rate[i].item() if mask_rate.dim() > 0 else mask_rate
        num_mask = max(1, int(n_valid * rate))
        perm = valid_pos[torch.randperm(n_valid, device=x.device)[:num_mask]]
        mask[i, perm] = True

    x_noisy = x.clone()
    x_noisy[mask] = MASK_TOKEN
    return x_noisy, mask


def random_mask_rate(batch_size: int, max_rate: float = 0.9) -> torch.Tensor:
    return torch.empty(batch_size).uniform_(0.0, max_rate)
