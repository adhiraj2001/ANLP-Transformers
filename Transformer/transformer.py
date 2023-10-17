import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import EncoderBlock
from decoder import DecoderBlock

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Transformer(nn.Module):

    def __init__(self, d_model=512, n_head=8, n_layers=2, ffn_hidden=2048, context_size=256, src_vocab_size=1000, trg_vocab_size=1000, src_pad=0, trg_pad=0, dropout=0.1, eps=1e-6):

        super().__init__()

        self.src_pad = src_pad
        self.trg_pad = trg_pad

        self.encoder = EncoderBlock(d_model=d_model, n_head=n_head, n_layers=n_layers, ffn_hidden=ffn_hidden, context_size=context_size, vocab_size=src_vocab_size, dropout=dropout, eps=eps)
        self.decoder = DecoderBlock(d_model=d_model, n_head=n_head, n_layers=n_layers, ffn_hidden=ffn_hidden, context_size=context_size, vocab_size=trg_vocab_size, dropout=dropout, eps=eps)

    def mask_src(self, src):

        src_mask = (src != self.src_pad).unsqueeze(1).unsqueeze(2)
        return src_mask

    def mask_trg(self, trg):

        trg_pad_mask = (trg != self.trg_pad).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        
        # trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, dtype=torch.ByteTensor)).to(device)
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, dtype=torch.uint8)).to(device)
        trg_mask = (trg_pad_mask & trg_sub_mask)

        return trg_mask
    
    def forward(self, src, trg):

        src_mask = self.mask_src(src)
        trg_mask = self.mask_trg(trg)

        enc_output = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_output, trg_mask, src_mask)

        return output

