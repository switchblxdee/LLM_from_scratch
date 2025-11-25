import torch
import torch.nn as nn
import torch.nn.functional as F

class SWIGLUFFN(nn.Module):
    def __init__(self, n_emb, n_hidden, dropout=0.0):
        super().__init__()
        
        self.gate_proj = nn.Linear(n_emb, n_hidden, bias=False)
        self.up_proj = nn.Linear(n_emb, n_hidden, bias=False)
        self.down_proj = nn.Linear(n_hidden, n_emb, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        swi_gate = F.silu(gate)
        h = swi_gate * up 
        
        output = self.down_proj(h)
        output = self.dropout(output)
        
        return output