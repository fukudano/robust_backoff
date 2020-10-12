import itertools
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import simstring
from _simstring import Simstring


def chain(x):
    return list(itertools.chain.from_iterable(x))


class Wemb(nn.Module):
    def __init__(self, args):
        super().__init__()
        wtoi, wvec = args.wtoi, args.wvec

        sim = Simstring(words=args.wtoi.keys(), measure=simstring.jaccard, n=3)

        _itow, _wvec = [], []
        for w in args.tsk_words:
            if w in wtoi:
                _itow.append(w)
                _wvec.append(wvec[wtoi[w]])
            else:
                d = sim.search(w=w, K=1, same=True)
                if len(d) > 0:
                    d = d[0]
                    _itow.append(w)
                    _wvec.append(wvec[wtoi[d]])
        self.wtoi = {w: i for i, w in enumerate(_itow)}
        self.wvec = torch.stack(_wvec)

    def forward(self, batch):
        mask = pad_sequence([torch.LongTensor([w in self.wtoi for w in s]) for s in batch], True, -1)
        e = torch.zeros(*mask.shape, 300).cuda()
        wiv = [w for w in chain(batch) if w in self.wtoi]
        e[mask == 1] = self.wvec[[self.wtoi[w] for w in wiv]].cuda()
        return e
