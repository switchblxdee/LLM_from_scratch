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

class Decoder(nn.Module):
    def __init__(self, n_layers, n_vocab, n_emb, n_hidden, n_heads, q_lora_rank, kv_lora_rank, nope_head_dim, rope_head_dim, dropout=0.0, max_len=1024):
        super().__init__()
        
        self.embeddings = nn.Embedding(n_vocab, n_emb)
        self.final_norm = RMSNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, n_vocab, bias=False)
        
        self.blocks = nn.ModuleList([DecoderBlock(n_emb, n_hidden, n_heads, q_lora_rank, kv_lora_rank, nope_head_dim, rope_head_dim, dropout, max_len) for _ in range(n_layers)])

    def forward(self, x):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits