import numpy as np
import torch
import torch.nn as nn
from allennlp.nn.util import masked_mean


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, kdim, vdim, hdim, odim, dropout=0.3):
        super(MultiHeadedAttention, self).__init__()
        self.d_k = hdim // n_head
        self.h = n_head
        self.q_proj = nn.Linear(kdim, hdim)
        self.k_proj = nn.Linear(kdim, hdim)
        self.v_proj = nn.Linear(vdim, hdim)
        self.out = nn.Linear(hdim, odim)
        self.attn = None
        self.drop = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask):
        B, qS, kD = q.shape
        B, kS, kD = k.shape
        # assert (mask.shape == B, kS)

        # 1) Do all the linear projections in batch from n_hid => h x d_k
        q = self.q_proj(q).view(B, qS, self.h, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(B, kS, self.h, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(B, kS, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        mask = mask.view(B, kS, 1, 1).transpose(1, 2)
        x, self.attn = self.attention(q, k, v, mask=mask, dropout=self.drop)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(B, qS, self.h * self.d_k)
        return self.out(x)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        # q:B,H,Q,D / k:B,H,D,K -> B,H,Q,K
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.transpose(-2, -1) == 0, -1e9)
        p_attn = nn.functional.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # attn:B,H,Q,K / v:B,H,K,D -> B,H,Q,D
        return torch.matmul(p_attn, value), p_attn


class Block(nn.Module):
    def __init__(self, n_head, hdim, ffdim, dropout):
        super(Block, self).__init__()
        self.inorm = nn.LayerNorm(hdim)
        self.attn = MultiHeadedAttention(n_head=n_head, kdim=hdim, vdim=hdim, hdim=hdim, odim=hdim)
        self.idrop = nn.Dropout(dropout)
        self.onorm = nn.LayerNorm(hdim)
        self.ff = nn.Sequential(nn.Linear(hdim, ffdim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffdim, hdim))
        self.odrop = nn.Dropout(dropout)

    def forward(self, x, mask):
        r = self.inorm(x)
        r = self.attn(r, r, r, mask)
        x = x + self.idrop(r)
        r = self.onorm(x)
        r = self.ff(r)
        x = x + self.odrop(r)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, n_head, hdim, ffdim, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(n_head=n_head, hdim=hdim, ffdim=ffdim, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_hid)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = 1 / (10000**(torch.arange(0., n_hid, 2.)) / n_hid)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) / np.sqrt(n_hid)
        self.pe = pe.requires_grad_(False).cuda()

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.shape[-2], :])


class Ctx(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.wtoi, self.wvec = args.wtoi, args.wvec

        hdim = 300
        self.posenc = PositionalEncoding(hdim)
        self.ctxenc = Encoder(num_layers=2, n_head=10, hdim=hdim, ffdim=hdim * 2, dropout=0.3)
        self.ctxagg = Encoder(num_layers=2, n_head=10, hdim=hdim, ffdim=hdim * 2, dropout=0.3)

    def pred(self, ws, ctxs):  # ws : no use
        # ctxs : B,C,S
        ctxs = torch.stack(ctxs)
        x = self.wvec[ctxs].cuda()  # B,C,S,D
        mask = (ctxs != -1).cuda()  # B,C,S
        B, C, S, D = x.shape

        x = x.reshape(B * C, S, D)
        mask = mask.reshape(B * C, S)
        x = self.posenc(x)  # B*C,S,D
        x = self.ctxenc(x, mask)  # B*C,S,D
        x = masked_mean(x, mask[:, :, None], dim=-2)  # B*C,D

        x = x.reshape(B, C, D)
        mask = mask.reshape(B, C, S).any(-1)  # B,C
        x = self.ctxagg(x, mask)  # B,C,D
        x = masked_mean(x, mask[:, :, None], dim=-2)  # B,D
        return x
