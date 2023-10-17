import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import TokenEmbedding
from attention import MHA
from layer_norm import LayerNorm

torch.cuda.empty_cache()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DecoderLayer(nn.Module):

    def __init__(self, d_model=512, n_head=8, ffn_hidden=2048, dropout=0.1, eps=1e-6):

        super().__init__()
        
        self.mha1 = MHA(embed_dim=d_model, n_head=n_head)
        self.mha2 = MHA(embed_dim=d_model, n_head=n_head)

        self.norm1 = LayerNorm(d_model=d_model, eps=eps)
        self.norm2 = LayerNorm(d_model=d_model, eps=eps)
        self.norm3 = LayerNorm(d_model=d_model, eps=eps)

        self.linear1 = nn.Linear(d_model, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, d_model)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.dropout4 = nn.Dropout(p=dropout)
    
    def forward(self, trg, enc, trg_mask=None, enc_mask=None):

        # skip_1 = trg
        skip_1 = trg.clone()

        x = self.mha1(q=trg, k=trg, v=trg, mask=trg_mask)
        x = self.dropout1(x)

        x = self.norm1(x + skip_1)

        if enc is not None:
            # skip_2 = x
            skip_2 = x.clone()

            x = self.mha2(q=x, k=enc, v=enc, mask=enc_mask)
            x = self.dropout2(x)

            x = self.norm2(x + skip_2)
        
        # skip_3 = x
        skip_3 = x.clone()

        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.linear2(x)
        # x = F.relu(x)
        x = self.dropout4(x)

        x = self.norm3(x + skip_3)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, d_model=512, n_head=8, n_layers=2, ffn_hidden=2048, context_size=256, vocab_size=1000, dropout=0.1, eps=1e-6):

        super().__init__()
        
        self.encode = TokenEmbedding(vocab_size=vocab_size, d_model=d_model, context_size=context_size, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden, dropout=dropout, eps=eps) for i in range(n_layers)])

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, trg, enc, trg_mask=None, enc_mask=None):

        trg = self.encode(trg)

        for layer in self.layers:
            trg = layer(trg, enc, trg_mask, enc_mask)
        
        output = self.linear(trg)
        # output = F.softmax(output)

        return output

