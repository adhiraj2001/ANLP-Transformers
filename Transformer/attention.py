import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.empty_cache()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, q, k, v, mask=None, n_inf=-10000):

        k_t = k.transpose(2, 3)
        # attn_score = (q @ k_t) / torch.sqrt(k.size(-1))
        attn_score = (q @ k_t) / math.sqrt(k.size(-1))

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, n_inf)

        attn_score = F.softmax(attn_score, dim=-1)

        v = (attn_score @ v)

        return v, attn_score

## From Paper:
# To avoid significant growth of computational cost and parametrization cost, we set p_v = p_k = p_v = embed_dim // n_head
# Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality
class MHA(nn.Module):

    def __init__(self, embed_dim=512, n_head=8):

        super().__init__()

        self.n_head = n_head
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_head

        assert self.head_dim * self.n_head == self.embed_dim, f'embed_dim({embed_dim}) must be divisible by n_head({n_head})'

        self.attention = Attention()

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        self.w_concat = nn.Linear(embed_dim, embed_dim)

    def split(self, x):

        # it is similar with group convolution (split by number of heads)
        x = x.view(x.size(0), x.size(1), self.n_head, self.head_dim).transpose(1, 2)

        return x

    def concat(self, x):

        # x = x.transpose(1, 2).view(x.size(0), x.size(1), self.embed_dim)
        x = x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), self.embed_dim)

        return x

    def forward(self, q, k, v, mask=None):

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        x, attn_score = self.attention(q, k, v, mask=mask)

        x = self.concat(x)
        x = self.w_concat(x)

        # TODO : implement visualization of attention map

        return x

