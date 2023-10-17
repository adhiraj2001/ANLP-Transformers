import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import TokenEmbedding
from attention import MHA
from layer_norm import LayerNorm

torch.cuda.empty_cache()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderLayer(nn.Module):

    def __init__(self, d_model=512, n_head=8, ffn_hidden=2048, dropout=0.1, eps=1e-6):

        super().__init__()
        
        self.mha = MHA(embed_dim=d_model, n_head=n_head)

        self.norm1 = LayerNorm(d_model=d_model, eps=eps)
        self.norm2 = LayerNorm(d_model=d_model, eps=eps)

        self.linear1 = nn.Linear(d_model, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, d_model)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        # skip_1 = x
        skip_1 = x.clone()

        x = self.mha(q=x, k=x, v=x, mask=mask)
        x = self.dropout1(x)

        x = self.norm1(x + skip_1)
        
        # skip_2 = x
        skip_2 = x.clone()

        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.linear2(x)
        # x = F.relu(x)
        x = self.dropout3(x)

        x = self.norm2(x + skip_2)

        return x


class EncoderBlock(nn.Module):

    def __init__(self, d_model=512, n_head=8, n_layers=2, ffn_hidden=2048, context_size=256, vocab_size=1000, dropout=0.1, eps=1e-6):

        super().__init__()
        
        self.encode = TokenEmbedding(vocab_size=vocab_size, d_model=d_model, context_size=context_size, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden, dropout=dropout, eps=eps) for i in range(n_layers)])

    def forward(self, x, mask=None):

        x = self.encode(x)

        for layer in self.layers:
            x = layer(x, mask)
        
        return x

