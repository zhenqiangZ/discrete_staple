#  python sample.py --cond 3 --temp 1 --steps 10 --count 5

import torch
import torch.nn.functional as F

from config import MASK_TOKEN, TEMPERATURE, NUM_SAMPLE_STEPS

DEFAULT_SAMPLE_LEN = 20


@torch.no_grad()
def sample(model, cond: torch.Tensor, steps: int = NUM_SAMPLE_STEPS,
           temperature: float = TEMPERATURE, max_len: int = DEFAULT_SAMPLE_LEN,
           device: str = 'cpu') -> torch.Tensor:
    model.eval()
    B = cond.shape[0]
    x = torch.full((B, max_len), MASK_TOKEN, dtype=torch.long, device=device)

    tokens_per_step = max(1, max_len // steps)

    for step in range(steps):
        logits = model(x, cond)
        probs = F.softmax(logits / temperature, dim=-1)

        x_new = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, max_len)

        num_update = tokens_per_step
        if step == steps - 1:
            num_update = (x == MASK_TOKEN).sum().item()

        remaining_mask = (x == MASK_TOKEN)
        for i in range(B):
            mask_pos = remaining_mask[i].nonzero(as_tuple=True)[0]
            if mask_pos.numel() == 0:
                continue
            n_update = min(num_update, mask_pos.numel())
            update_idx = mask_pos[torch.randperm(mask_pos.numel(), device=device)[:n_update]]
            x[i, update_idx] = x_new[i, update_idx]

    return x


@torch.no_grad()
def sample_with_confidence(model, cond: torch.Tensor, steps: int = NUM_SAMPLE_STEPS,
                           temperature: float = TEMPERATURE, max_len: int = DEFAULT_SAMPLE_LEN,
                           device: str = 'cpu') -> torch.Tensor:
    model.eval()
    B = cond.shape[0]
    x = torch.full((B, max_len), MASK_TOKEN, dtype=torch.long, device=device)

    for step in range(steps):
        logits = model(x, cond)
        probs = F.softmax(logits / temperature, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        confidence, x_new = log_probs.max(dim=-1)

        remaining_mask = (x == MASK_TOKEN)
        confidence[~remaining_mask] = float('inf')

        tokens_per_step = max(1, max_len // steps)
        for i in range(B):
            mask_pos = remaining_mask[i].nonzero(as_tuple=True)[0]
            if mask_pos.numel() == 0:
                continue
            n_update = min(tokens_per_step, mask_pos.numel())
            _, idx = confidence[i, mask_pos].sort(descending=True)
            update_idx = mask_pos[idx[:n_update]]
            x[i, update_idx] = x_new[i, update_idx]

    return x


@torch.no_grad()
def sample_autoregressive_final(model, cond: torch.Tensor,
                                 temperature: float = TEMPERATURE,
                                 max_len: int = DEFAULT_SAMPLE_LEN,
                                 device: str = 'cpu') -> torch.Tensor:
    model.eval()
    B = cond.shape[0]
    x = torch.full((B, max_len), MASK_TOKEN, dtype=torch.long, device=device)
    logits = model(x, cond)
    probs = F.softmax(logits / temperature, dim=-1)
    x = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, max_len)
    return x
