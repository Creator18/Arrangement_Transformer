"""The grammar model: an encoder-decoder over piano note tokens.

The ENCODER reads notes that may be wrong (corrupted in pretraining; later, the
spectral ears' output); the DECODER writes the corrected piano part token by
token, attending to the encoder through cross-attention.  That shape is chosen
for what comes next: the song-to-piano arranger adds the AUDIO encoder's
features as further memory for the same decoder to attend to, so the decoder's
musical grammar learned here carries over unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import tokens as T


class GrammarModel(nn.Module):
    def __init__(self, d: int = 384, enc_layers: int = 4, dec_layers: int = 6, heads: int = 6,
                 max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.arch = dict(d=d, enc_layers=enc_layers, dec_layers=dec_layers, heads=heads,
                         max_len=max_len, dropout=dropout)
        self.emb = nn.Embedding(T.VOCAB, d, padding_idx=T.PAD)
        self.pos = nn.Embedding(max_len, d)
        mk = lambda: dict(d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=dropout,
                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**mk()), enc_layers,
                                             enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(**mk()), dec_layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, T.VOCAB)

    def embed(self, x):
        return self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]

    def encode(self, src):
        return self.encoder(self.embed(src), src_key_padding_mask=src == T.PAD)

    def decode(self, tgt_in, memory, src_pad):
        L = tgt_in.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tgt_in.device), 1)
        h = self.decoder(self.embed(tgt_in), memory, tgt_mask=causal,
                         tgt_key_padding_mask=tgt_in == T.PAD, memory_key_padding_mask=src_pad)
        return self.out(self.norm(h))

    def forward(self, src, tgt_in):
        return self.decode(tgt_in, self.encode(src), src == T.PAD)

    @torch.no_grad()
    def generate(self, src, max_len: int = 600):
        """Greedy decoding, batched; stops each row at EOS."""
        mem, pad = self.encode(src), src == T.PAD
        y = torch.full((src.shape[0], 1), T.BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            nxt = self.decode(y, mem, pad)[:, -1].argmax(-1)
            nxt[done] = T.PAD
            y = torch.cat([y, nxt[:, None]], 1)
            done |= nxt == T.EOS
            if done.all():
                break
        return y[:, 1:]
