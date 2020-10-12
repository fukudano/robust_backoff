import itertools
import torch
import torch.nn as nn
from allennlp.nn.util import masked_softmax
from charcnn import CharCNN
from latsim import LatSim


def chain(x):
    return list(itertools.chain.from_iterable(x))


def get_lr(dws, qws, L, R, lb):
    latsim = LatSim(dws)
    w2l, w2r = {}, {}
    for w in qws:
        xs = latsim.lat(w, L)
        w2l[w] = xs + ["<pad>"] * (L - len(xs))
    for w in qws:
        xs = latsim.ret(w, R, lb=lb, same=False)
        w2r[w] = xs + ["<pad>"] * (R - len(xs))
    return w2l, w2r


class Char(nn.Module):
    def __init__(self, args):
        super().__init__()
        L, R, lb = 7, 10, 0.0
        self.wtoi, self.wvec = args.wtoi, args.wvec

        qws = list(set(args.qws))
        self.w2l, self.w2r = get_lr(self.wtoi.keys(), qws, L, R, lb)

        self.cdim = 100
        self.charcnn = CharCNN(self.cdim)
        self.lW = nn.Parameter(torch.zeros(self.cdim, self.cdim).uniform_(-0.1, 0.1))
        self.rW = nn.Parameter(torch.zeros(self.cdim, self.cdim).uniform_(-0.1, 0.1))
        self.gL = nn.Linear(L + R, 2)

    def pred(self, ws):
        ls_ = [self.w2l[w] for w in ws]
        rs_ = [self.w2r[w] for w in ws]

        _itow = sorted(set(list(ws) + chain(ls_) + chain(rs_)) - {"<pad>"})
        _wtoi = {w: i for i, w in enumerate(_itow)}
        ce = nn.functional.normalize(self.charcnn(_itow), dim=-1)
        ce = torch.cat([ce, torch.zeros(1, self.cdim).cuda()], 0)

        _lixs = torch.LongTensor([[_wtoi.get(w, -1) for w in x] for x in ls_])  # q, k
        _rixs = torch.LongTensor([[_wtoi.get(w, -1) for w in x] for x in rs_])  # q, k

        qce = ce[[_wtoi[w] for w in ws]].unsqueeze(-1)  # q,1
        la = masked_softmax(ce[_lixs] @ self.lW @ qce, (_lixs != -1)[:, :, None].cuda(), dim=-2,
                            memory_efficient=True)  # q,k,d x q,d,1 -> q,k,1
        ra = masked_softmax(ce[_rixs] @ self.rW @ qce, (_rixs != -1)[:, :, None].cuda(), dim=-2,
                            memory_efficient=True)  # q,k,d x q,d,1 -> q,k,1

        lixs = torch.LongTensor([[self.wtoi.get(w, -1) for w in x] for x in ls_])
        rixs = torch.LongTensor([[self.wtoi.get(w, -1) for w in x] for x in rs_])
        le = (la * self.wvec[lixs].cuda()).sum(-2)
        re = (ra * self.wvec[rixs].cuda()).sum(-2)

        a = self.gL(torch.cat([la, ra], -2).squeeze(-1)).softmax(-1)  # q,2
        e = (a.unsqueeze(-1) * torch.stack([le, re], -2)).sum(-2)
        return e
