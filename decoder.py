import torch
import torch.nn as nn
import torch.nn.functional as F
from ffn import SWIGLUFFN
from mla import MultiHeadLatentAttention
from rmsnorm import RMSNorm

class DecoderBlock(nn.Module):
    def __init__(self, n_emb, n_hidden, n_heads, q_lora_rank, kv_lora_rank, nope_head_dim, rope_head_dim, dropout=0.0, max_len=1024):
        super().__init__()
        self.attn_norm = RMSNorm(n_emb)
        self.attn = MultiHeadLatentAttention(n_emb, n_heads, q_lora_rank, kv_lora_rank, nope_head_dim, rope_head_dim, dropout, max_len)
        self.ffn_norm = RMSNorm(n_emb)
        self.ffn = SWIGLUFFN(n_emb, n_hidden, dropout)

    def forward(self, x):
        attn_norm_x = self.attn_norm(x)
        h = x + self.attn(attn_norm_x)
        ffn_norm_h = self.ffn_norm(h)
        h = h + self.ffn(ffn_norm_h)
        return h
