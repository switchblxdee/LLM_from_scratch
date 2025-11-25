import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import apply_rotary_emb, RotaryEmbedding

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, n_emb, n_heads, q_lora_rank, kv_lora_rank, nope_head_dim, rope_head_dim, dropout=0.0, max_len=1024):
        super().__init__()
        self.n_emb = n_emb
        self.n_heads = n_heads
        self.head_dim = nope_head_dim + rope_head_dim 
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.rotary_emb = RotaryEmbedding(rope_head_dim, max_seq_len=max_len)

        self.q_down_proj = nn.Linear(n_emb, q_lora_rank, bias=False)
        self.q_up_content = nn.Linear(q_lora_rank, n_heads * nope_head_dim, bias=False)
        self.q_up_rope = nn.Linear(q_lora_rank, n_heads * rope_head_dim, bias=False)

        self.kv_down_proj = nn.Linear(n_emb, kv_lora_rank, bias=False)
        self.v_up_proj = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)
        
        self.k_up_content = nn.Linear(kv_lora_rank, n_heads * nope_head_dim, bias=False)
        self.k_up_rope = nn.Linear(kv_lora_rank, n_heads * rope_head_dim, bias=False)

        self.o_proj = nn.Linear(n_heads * self.head_dim, n_emb, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size()
        
        q_latent = self.q_down_proj(x)
        kv_latent = self.kv_down_proj(x)
        
        q_content = self.q_up_content(q_latent)
        q_rope = self.q_up_rope(q_latent)
        
        v_content = self.v_up_proj(kv_latent)
        k_content = self.k_up_content(kv_latent)
        k_rope = self.k_up_rope(kv_latent)
        
        q_content = q_content.view(B, T, self.n_heads, self.nope_head_dim)
        q_rope = q_rope.view(B, T, self.n_heads, self.rope_head_dim)
        v_content = v_content.view(B, T, self.n_heads, self.head_dim)
        k_content = k_content.view(B, T, self.n_heads, self.nope_head_dim)
        k_rope = k_rope.view(B, T, self.n_heads, self.rope_head_dim)

        freqs_cis = self.rotary_emb(x, seq_len=T)
        q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, freqs_cis)

        q = torch.cat([q_content, q_rope], dim=-1)
        k = torch.cat([k_content, k_rope], dim=-1)
        v = v_content

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        
        output = probs @ v
        
        output = output.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        
        return self.o_proj(output)
        
