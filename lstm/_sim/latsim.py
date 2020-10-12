import string
import itertools
from simstring_wrap import Simstring


def chain(x):
    return list(itertools.chain.from_iterable(x))


def lat(w, l, r, vocab):
    last = None
    ens = [None for _ in range(len(w) + 1)]
    ens[0] = (0, [], [])
    for b in range(0, len(w) - l + 1):
        if 0 < b < l:
            continue
        if b > 0 and ens[b] is None:
            continue
        for e in range(b + l, min(b + r, len(w)) + 1):
            if e < len(w) < e + l:
                continue
            s = w[b:e]
            q = s.strip(string.punctuation)
            if s == w:
                continue

            if s in vocab:
                cs, cq, cc = s, s, 1
            elif q in vocab:
                cs, cq, cc = s, q, 1
            elif len(s) == 1:
                cs, cq, cc = s, s, 1 * 10
            else:
                continue
            bc, bs, bq = ens[b]
            cn = (bc + cc, bs + [cs], bq + [cq])
            if ens[e] is None or cn[0] < ens[e][0]:
                ens[e] = cn
            if e == len(w):
                if last is None or cn[0] < last[0][0]:
                    last = [cn]
                elif cn[0] == last[0][0]:
                    last.append(cn)
        else:
            if (ens[-1] is not None and all(ens[-1][0] <= n[0] for n in ens[b + 1:] if n is not None)):
                break
    if last is None:
        return []
    return set(map(tuple, [x[-1] for x in last]))


class LatSim:
    def __init__(self, words):
        self.words = set(words)
        self.sursim = Simstring(self.words, be=True, unicode=True)

    def lat(self, w, k):
        ls = lat(w, 1, 30, self.words)
        ls = [w for w in chain(ls) if w in self.words]
        ls = sorted(set(ls), key=lambda x: -len(x))[:k]
        return ls

    def ret(self, w, k, lb=0.0, same=False):
        rs = self.sursim.search(w, k, lb=lb, same=same)
        return rs
