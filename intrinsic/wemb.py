import torch
import itertools
import math
import string
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from allennlp.nn.util import masked_softmax
from latsim import LatSim


def chain(x):
    return list(itertools.chain.from_iterable(x))


class CharCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hdim = args.cnn_dim
        self.ctoi = {c: i for i, c in enumerate(["<unk>"] + list(string.printable))}
        self.char_emb = nn.Embedding(len(self.ctoi), self.hdim)
        self.filter_num_width = args.cnn_filter
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.hdim,
                          out_channels=self.hdim,
                          kernel_size=filter_width,
                          padding=math.floor((filter_width - 1) / 2)), nn.ReLU())
            for filter_width in self.filter_num_width
        ])
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(args.cnn_dropout),
            nn.Linear(self.hdim * len(self.filter_num_width), self.hdim),
            nn.LayerNorm(self.hdim),
        )

    def forward(self, ws):
        cids = [torch.LongTensor([self.ctoi.get(c, self.ctoi["<unk>"]) for c in "<" + w + ">"]) for w in ws]  # B, C
        cids = pad_sequence(cids, True, -1)  # B,C
        _x = torch.cat([self.char_emb.weight, torch.zeros(1, self.hdim).cuda()])
        x = _x[cids].transpose(1, 2)  # B,C -> B, C, D, -> B, D, C
        x = [torch.max(conv(x), dim=-1)[0] for conv in self.convs]  # B, D, C -> 4, B, D, C -> 4, B, D
        x = torch.cat(x, dim=1)  # 4, B, D -> B, 4D
        x = self.linear(x)  # B, 4D -> B, D
        return x


def get_lr(dws, qws, L, R, measure, gpu):
    latsim = LatSim(words=dws, measure=measure, gpu=gpu)
    w2l, w2r = {}, {}
    if L > 0:
        for w in qws:
            xs = latsim.lat(w, L)
            w2l[w] = xs + ["<pad>"] * (L - len(xs))
    if R > 0:
        for w in qws:
            xs = latsim.ret(w, R, same=False)
            w2r[w] = xs + ["<pad>"] * (R - len(xs))
    latsim.simstring.db.close()
    # shutil.rmtree('db')
    return w2l, w2r


class Wemb(nn.Module):
    def __init__(self, args, qws):
        super().__init__()
        self.wtoi = args.wtoi
        self.wvec = args.wvec
        assert ("<pad>" not in self.wtoi)

        self.L, self.R, self.similarity = args.L, args.R, args.similarity
        qws = list(set(qws))
        self.w2l, self.w2r = get_lr(dws=self.wtoi.keys(),
                                    qws=qws,
                                    L=self.L,
                                    R=self.R,
                                    measure=self.similarity,
                                    gpu=args.cuda)

        self.word_dim = args.word_dim
        self.glv_wtoi = self.wtoi
        self.glv_wvec = torch.cat([self.wvec, torch.zeros(1, self.word_dim)])

        self.cdim = args.cnn_dim
        self.charcnn = CharCNN(args=args)
        self.lW = nn.Parameter(torch.zeros(self.cdim, self.cdim).uniform_(-0.1, 0.1))
        self.rW = nn.Parameter(torch.zeros(self.cdim, self.cdim).uniform_(-0.1, 0.1))
        self.gL = nn.Linear(args.L + args.R, 2)

    def pred(self, ws):
        ls_ = [self.w2l[w] for w in ws] if self.L > 0 else []
        rs_ = [self.w2r[w] for w in ws] if self.R > 0 else []

        _itow = sorted(set(ws + chain(ls_) + chain(rs_)) - {"<pad>"})
        _wtoi = {w: i for i, w in enumerate(_itow)}

        ce = nn.functional.normalize(self.charcnn(_itow), dim=-1)
        ce = torch.cat([ce, torch.zeros(1, self.cdim).cuda()], 0)

        qce = ce[[_wtoi[w] for w in ws]].unsqueeze(-1)  # q,1

        if self.L > 0:
            _lixs = torch.LongTensor([[_wtoi.get(w, -1) for w in x] for x in ls_])  # q, k
            la = masked_softmax(ce[_lixs] @ self.lW @ qce, (_lixs != -1).unsqueeze(-1).cuda(),
                                dim=-2,
                                memory_efficient=True)  # q,k,d x q,d,1 -> q,k,1
            lixs = torch.LongTensor([[self.glv_wtoi.get(w, -1) for w in x] for x in ls_])
            le = (la * self.glv_wvec[lixs].cuda()).sum(-2)
        if self.R > 0:
            _rixs = torch.LongTensor([[_wtoi.get(w, -1) for w in x] for x in rs_])  # q, k
            ra = masked_softmax(ce[_rixs] @ self.rW @ qce, (_rixs != -1).unsqueeze(-1).cuda(),
                                dim=-2,
                                memory_efficient=True)  # q,k,d x q,d,1 -> q,k,1
            rixs = torch.LongTensor([[self.glv_wtoi.get(w, -1) for w in x] for x in rs_])
            re = (ra * self.glv_wvec[rixs].cuda()).sum(-2)
        if self.R == 0:
            e = le
        elif self.L == 0:
            e = re
        else:
            a = self.gL(torch.cat([la, ra], -2).squeeze(-1)).softmax(-1)  # q,2
            e = (a.unsqueeze(-1) * torch.stack([le, re], -2)).sum(-2)
        return e

    def train_ws(self, ws):
        es = self.pred(ws)
        ts = self.glv_wvec[[self.glv_wtoi[w] for w in ws]].cuda()
        loss = -nn.functional.cosine_similarity(es, ts).mean()
        return loss
