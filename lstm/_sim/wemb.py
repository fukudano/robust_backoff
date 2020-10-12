import os
from more_itertools import chunked
import itertools
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from char import Char


def chain(x):
    return list(itertools.chain.from_iterable(x))


class Wemb(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.wtoi, self.wvec = args.wtoi, args.wvec
        self.sur = Char(args)

        if not args.train_emb:
            self.load_state_dict(torch.load(os.path.dirname(__file__)+"/model.pt"))
            self = self.eval().cuda()

            es, wtoi = self.make_wvec(args, args.qws)
            self.wtoi_ = {w: i for i, w in enumerate(args.tsk_words)}
            self.wvec_ = torch.zeros(len(self.wtoi_), 300)
            for w in args.tsk_words:
                if w in self.wtoi:
                    self.wvec_[self.wtoi_[w]] = self.wvec[self.wtoi[w]]
                elif w in wtoi:
                    self.wvec_[self.wtoi_[w]] = es[wtoi[w]]

    def pred(self, ws, ctxs):
        sur = self.sur.pred(ws)
        x = sur
        return x

    def train_batch(self, batch):
        ws, ctxs = zip(*batch)
        e = self.pred(ws, ctxs)
        t = self.wvec[[self.wtoi[w] for w in ws]].cuda()
        loss = -nn.functional.cosine_similarity(e, t).mean()
        return loss

    def make_wvec(self, args, qws):
        itow, es = [], []
        for batch in chunked(qws, 100):
            with torch.no_grad():
                itow.extend(batch)
                es.extend(self.pred(batch, None).cpu())
        es = torch.stack(es)
        wtoi = {w: i for i, w in enumerate(itow)}
        return es, wtoi

    def forward(self, sents):
        mask = pad_sequence([torch.LongTensor([w in self.wtoi_ for w in s]) for s in sents], True, -1)
        e = torch.zeros(*mask.shape, 300).cuda()
        e[mask == 1] = self.wvec_[[self.wtoi_[w] for w in chain(sents) if w in self.wtoi_]].cuda()
        return e
