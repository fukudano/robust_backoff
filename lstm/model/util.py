import torch
import itertools
import numpy as np


def build_stoi(sents):
    words, tags = map(lambda x: sorted(set(x)), zip(*chain(sents)))
    wtoi = {w: i for i, w in enumerate(["<unk>", "<pad>"] + words)}
    tags = ["<pad>"] + tags
    tag_itos = {i: w for i, w in enumerate(tags)}
    tag_stoi = {w: i for i, w in tag_itos.items()}
    return wtoi, words, tag_stoi, tag_itos


def cut_wvec(trgwtoi, srcwtoi, srcwvec):
    dim = 300
    trgwvec = torch.zeros(len(trgwtoi), dim).uniform_(-0.05, 0.05)
    cnt = 0
    for word in sorted(trgwtoi.keys()):
        if word in srcwtoi:
            trgwvec[trgwtoi[word]] = torch.FloatTensor(srcwvec[srcwtoi[word]])
            cnt += 1
    print(f"word match {cnt} / {len(trgwtoi)} ({cnt/len(trgwtoi)*100:.2f} %)")
    return trgwvec


def read_wvec(file):
    itow, wvec = [], []
    dim = 300
    with open(file, "r") as h:
        for line in h:
            x = line.rstrip(" \n").rsplit(" ")
            assert (len(x) == dim + 1)
            word, emb = x[0], list(map(float, x[1:]))
            itow.append(word)
            wvec.append(emb)
    wtoi = {w: i for i, w in enumerate(itow)}
    return wtoi, torch.FloatTensor(wvec)


def valid_IOB2(lstsents):
    import pdb
    for dataidx, sents in enumerate(lstsents):
        for sent in sents:
            words, tags = zip(*sent)
            for t in tags:
                if not (t == "O" or t.startswith("I-") or t.startswith("B-")):
                    pdb.set_trace()
            for l, r in zip(tags[:-1], tags[1:]):
                if l == "O" and r.startswith("I-"):
                    pdb.set_trace()


def seq2span(seq: list):  # IOB2 scheme
    spans = []
    typ = None
    b = None
    for ix, tag in enumerate(seq):
        if typ is not None and not tag.startswith("I-"):
            spans.append((b, ix - 1, typ))
            typ = None
        if tag.startswith("B-"):
            b = ix
            typ = tag[2:]
        elif tag.startswith("I-"):
            if tag[2:] != typ:
                import pdb
                pdb.set_trace()
    else:
        if typ is not None:
            spans.append((b, ix, typ))
    return spans


def eval_ner(sents: list, gold: list, pred: list, vocab: set):
    assert (len(sents) == len(gold) and len(gold) == len(pred))
    g, p, g_oov, p_oov, g_wiv, p_wiv = [], [], [], [], [], []
    for (six, sent), g_sent, p_sent in zip(enumerate(sents), gold, pred):
        assert (len(sent) == len(g_sent) and len(g_sent) == len(p_sent))
        g_spans = seq2span(g_sent)
        p_spans = seq2span(p_sent)
        for b, e, t in g_spans:
            g.append((six, b, e, t))
            if len(set(sent[b:e + 1]) - vocab) > 0:
                g_oov.append((six, b, e, t))
            else:
                g_wiv.append((six, b, e, t))
        for b, e, t in p_spans:
            p.append((six, b, e, t))
            if len(set(sent[b:e + 1]) - vocab) > 0:
                p_oov.append((six, b, e, t))
            else:
                p_wiv.append((six, b, e, t))

    def calc(g, p):
        if len(g) == 0 or len(p) == 0:
            f = 0
        else:
            c = set(g) & set(p)
            f = 2 * len(c) / (len(g) + len(p))
        return f

    f = calc(g, p)
    f_oov = calc(g_oov, p_oov)
    f_wiv = calc(g_wiv, p_wiv)
    return f, f_wiv, f_oov


def chain(x):
    return lambda x: list(itertools.chain.from_iterable(x))


def eval_pos(sents: list, gold: list, pred: list, vocab: set):
    assert (len(sents) == len(gold) and len(gold) == len(pred))
    for (six, sent), g_sent, p_sent in zip(enumerate(sents), gold, pred):
        assert (len(sent) == len(g_sent) and len(g_sent) == len(p_sent))
    g = np.array(chain(gold))
    p = np.array(chain(pred))
    m = np.array([w in vocab for w in chain(sents)]).astype(bool)
    acc = (g == p).mean()
    acc_oov = (g == p)[~m].mean()
    acc_wiv = (g == p)[m].mean()
    return acc, acc_wiv, acc_oov


class GoldPos:
    def __init__(self, tes, tra, wtoi):
        self.gold = chain([[t for w, t in s] for s in tes])
        travoc = set(chain([[w for w, t in s] for s in tra]))
        vocab = set(wtoi.keys()) | travoc

        maskvec = []
        for w, t in chain(tes):
            maskvec.append(w not in vocab)
        self.maskvec = np.array(maskvec).astype(bool)

    def eval(self, pred):
        pred = chain(pred)
        assert (len(pred) == len(self.gold))
        cor = np.array(pred) == np.array(self.gold)
        acc = cor.mean() * 100
        acc_wivvec = cor[~self.maskvec].mean() * 100
        acc_oovvec = cor[self.maskvec].mean() * 100
        return acc, acc_wivvec, acc_oovvec


class GoldNer:
    def __init__(self, tes: list, tra: list, wtoi: dict):
        gold = []
        for s in tes:
            ws, ts = zip(*s)
            gold.append([ws, ts])
        self.gold = gold
        travoc = set(chain([[w for w, t in s] for s in tra]))
        self.vocab = set(wtoi.keys()) | travoc

    def eval(self, pred):
        p, g, poov, pwiv, goov, gwiv = [], [], [], [], [], []
        assert (len(pred) == len(self.gold))
        for six, (ptags, (ws, gtags)) in enumerate(zip(pred, self.gold)):
            assert (len(ptags) == len(gtags))
            _p = [(six, b, e, t) for b, e, t in seq2span(ptags)]
            _g = [(six, b, e, t) for b, e, t in seq2span(gtags)]
            p += _p
            g += _g
            poov += [s for s in _p if len(set(ws[s[1]:s[2] + 1]) - self.vocab) > 0]
            pwiv += [s for s in _p if len(set(ws[s[1]:s[2] + 1]) - self.vocab) == 0]
            goov += [s for s in _g if len(set(ws[s[1]:s[2] + 1]) - self.vocab) > 0]
            gwiv += [s for s in _g if len(set(ws[s[1]:s[2] + 1]) - self.vocab) == 0]

        def calc(g, p):
            if len(g) == 0 or len(p) == 0:
                f = 0
            else:
                c = set(g) & set(p)
                f = 2 * len(c) / (len(g) + len(p))
            return f

        f = calc(g, p) * 100
        f_oov = calc(goov, poov) * 100
        f_wiv = calc(gwiv, pwiv) * 100
        return f, f_wiv, f_oov
