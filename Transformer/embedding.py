import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):

    def __init__(self, context_size=256, d_model=512):

        super().__init__()

        self.context_size = context_size
        self.d_model = d_model
    
    def forward(self, x):
        # NOTE : Recompute context_size here

        context_pos = torch.arange(start=0, end=self.context_size, step=1, dtype=torch.float, requires_grad=False, device=device)
        dim_2i = torch.arange(start=0, end=self.d_model, step=2, dtype=torch.float, requires_grad=False, device=device)

        pos_encoding = torch.zeros(self.context_size, self.d_model, requires_grad = False, device=device)
        
        # print(f'context_size: {self.context_size}')
        # print(f'd_model: {self.d_model}')
        # print(f'context_pos: {context_pos.size()}')
        # print(f'dim_2i: {dim_2i.size()}')
        # print(f'pos_encoding: {pos_encoding.size()}')
        
        pos_encoding[:, 0::2] = torch.sin(context_pos / (10000 ** (dim_2i / self.d_model)))
        pos_encoding[:, 1::2] = torch.cos(context_pos / (10000 ** (dim_2i / self.d_model)))

        # len = x.size(1)
        len = x.size(-2)

        return (x + pos_encoding[:len, ...])


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size=1000, d_model=512, context_size=256, dropout=0.1):

        super().__init__()

        # self.encode = nn.Embedding.from_pretrained(embedding_weights, freeze=True)
        self.encode = nn.Embedding(vocab_size, d_model)

        self.pos_encode = PositionalEncoding(context_size=context_size, d_model=d_model)
        self.drop_out = nn.Dropout(p=dropout)

    def forward(self, x):

        x = self.encode(x)
        x = self.pos_encode(x)
        x = self.drop_out(x)

        return x

