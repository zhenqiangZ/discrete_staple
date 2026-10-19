import math
import torch
import torch.nn as nn

from config import VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, FFN_DIM, DROPOUT, NUM_CLASS, MAX_LEN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.cross_attn(h, cond_emb, cond_emb)[0]
        return x + h


class StapleDiffusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.pos_enc = PositionalEncoding(D_MODEL, MAX_LEN)
        self.cond_embedding = nn.Embedding(NUM_CLASS, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

        self.blocks = nn.ModuleList([
            TransformerBlock(D_MODEL, NUM_HEADS, FFN_DIM, DROPOUT)
            for _ in range(NUM_LAYERS)
        ])
        self.cross_attentions = nn.ModuleList([
            CrossAttentionBlock(D_MODEL, NUM_HEADS, DROPOUT)
            for _ in range(NUM_LAYERS)
        ])
        self.norm = nn.LayerNorm(D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        h = self.pos_enc(h)
        h = self.dropout(h)

        cond_emb = self.cond_embedding(cond).unsqueeze(1)

        for block, ca in zip(self.blocks, self.cross_attentions):
            h = block(h)
            h = ca(h, cond_emb)

        h = self.norm(h)
        logits = self.out_proj(h)
        return logits
