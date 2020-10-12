import numpy as np
from scipy.stats import spearmanr


def cosine(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2


class Dataset:
    def init(self):
        self.oovmask = None
        self.ws = sorted(set(sum(self.pairs, [])))

    def evaluate(self, es, vocab, ws=None, stoi=None):
        oovmask = np.array([l not in vocab or r not in vocab for l, r in self.pairs]).astype(bool)

        if stoi is None:
            assert (ws is not None)
            stoi = {s: i for i, s in enumerate(ws)}

        sims = []
        drop = 0
        for l, r in self.pairs:
            if (l in stoi and r in stoi and np.linalg.norm(es[stoi[l]]) > 0 and np.linalg.norm(es[stoi[r]]) > 0):
                le = es[stoi[l]]
                re = es[stoi[r]]
                s = cosine(le, re)
            else:
                s = 0
                drop += 1
            sims.append(s)
        sims = np.array(sims)

        r = spearmanr(sims, self.goldsims)[0] * 100
        oovr = spearmanr(sims[oovmask], self.goldsims[oovmask])[0] * 100
        wivr = spearmanr(sims[~oovmask], self.goldsims[~oovmask])[0] * 100
        oovrate = drop / len(self.pairs) * 100
        # stdout = f"all {r:2.1f}, wiv {wivr:2.1f}, oov {oovr:2.1f} (OOV: {oovrate:2.1f})"
        return r, wivr, oovr, oovrate, sims


class RW(Dataset):
    def __init__(self, args):
        pairs = []
        goldsims = []
        for line in open(args.rw_file, "r"):
            x = line.rstrip().split()
            pairs.append(x[:2])
            goldsims.append(float(x[2]))
        self.pairs = pairs
        self.goldsims = np.array(goldsims).astype(float)
        super().init()


class CARD(Dataset):
    def __init__(self, args):
        pairs = []
        goldsims = []
        for line in open(args.card_file, "r"):
            x = line.rstrip().split()
            if len(x) == 3:
                pairs.append(x[:2])
                goldsims.append(float(x[-1]))
            else:
                assert (False)
        self.pairs = pairs
        self.goldsims = np.array(goldsims).astype(float)
        super().init()
