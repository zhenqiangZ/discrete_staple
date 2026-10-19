"""Wrapper for AMP and toxicity prediction models with signed gradient impact.

Predictor models are TransformerClassifier (train_models.py) pretrained for
AMP activity and toxicity classification.  This module provides batch scoring
and per-position gradient-impact analysis (first-order Taylor approximation).

Tokenizer alignment (predictor vocab=23, diffusion vocab=24):
  0  = PAD       (same)
  1-20 = AAs      (same)
  21 = S5, 22 = R8 (same)
  23 = MASK       (predictor has no MASK -> replaced with PAD)
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_models import TransformerClassifier

PREDICTOR_VOCAB = 23


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictor(ckpt_path: str, device: torch.device, target_vocab: int = 23) -> nn.Module:
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    ckpt_vocab = sd['embed.weight'].shape[0]
    if ckpt_vocab != target_vocab:
        print(f'  Expanding predictor embedding: {ckpt_vocab} -> {target_vocab}')
        model = TransformerClassifier(vocab_size=target_vocab)
        new_embed = model.embed.weight.data.clone()
        new_embed[:ckpt_vocab] = sd['embed.weight']
        sd['embed.weight'] = new_embed
    else:
        model = TransformerClassifier(vocab_size=ckpt_vocab)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f'  Loaded predictor ({type(model).__name__}): {ckpt_path}')
    return model


# ---------------------------------------------------------------------------
# Token preparation
# ---------------------------------------------------------------------------

def _prepare(tokens: torch.Tensor) -> torch.Tensor:
    """Map diffusion-vocab tokens to predictor-vocab.

    S5(21),R8(22) -> A(1); MASK(23) -> PAD(0); clamp remainder.
    """
    prepared = tokens.clone()
    prepared.masked_fill_(tokens == 21, 1)
    prepared.masked_fill_(tokens == 22, 1)
    prepared.masked_fill_(tokens == 23, 0)
    prepared.clamp_(0, PREDICTOR_VOCAB - 1)
    return prepared


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_score(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Sigmoid scores for sequences.

    Args:
        tokens: [B, L] diffusion-vocab tokens
    Returns:
        [B] sigmoid scores
    """
    x = _prepare(tokens)
    logits = model(x).squeeze(-1)
    return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Position alignment  (signed, on parent sequence only)
# ---------------------------------------------------------------------------

def batch_position_alignment(
    model: nn.Module,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Signed per-position alignment = dot(grad_logit, emb) / ||emb||.

    Computed on the original (parent) sequence tokens.  Positive at position i
    means the current amino acid at i *increases* the model's logit — mutating
    this position away from current AA would tend to *decrease* the logit.

    Formula:
      align_i = dot(∂logit/∂emb_i, emb_i) / ||emb_i||

    The denominator normalises out embedding magnitude so the result mainly
    reflects the angular alignment between gradient and embedding direction.

    Args:
        tokens: [B, L] parent-sequence tokens (diffusion vocab)
    Returns:
        [B, L] signed alignment values
    """
    prep = _prepare(tokens)
    emb = model.embed(prep)
    emb = emb.detach().requires_grad_(True)

    h = model.encoder(emb)
    h = h.mean(dim=1)
    logit = model.classifier(h).squeeze(-1)

    grad = torch.autograd.grad(logit.sum(), emb, only_inputs=True)[0]

    return ((grad * emb).sum(dim=-1) / (emb.norm(dim=-1) + 1e-8)).detach()
