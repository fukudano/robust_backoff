import torch


# wordemb_file="glove.840B.300d.txt", wordfreq_file="freq_count.glove.840B.300d.txt.fix.txt"
def wordemb(args):
    w2f = {}
    for line in open(args.wordfreq_file, "r"):
        w, f = line.rstrip().replace(" ", "\t").split("\t")
        w2f[w] = int(f)

    itow, wvec = [], []
    for line in open(args.wordemb_file, "r"):
        x = line.rstrip().split(" ")
        itow.append(x[0])
        wvec.append(list(map(float, x[1:])))
    wtoi = {w: i for i, w in enumerate(itow)}
    wvec = torch.FloatTensor(wvec)
    w2f = {w: w2f.get(w, 0) for w in itow}
    return wtoi, wvec, w2f


def load_wordemb(file):
    itow, wvec = [], []
    with open(file, "r") as h:
        c = h.read()
        for r in c.split("\n"):
            x = r.rsplit(" ", 300)
            if len(x) == 301:
                w, e = x[0], [float(y) for y in x[1:]]
                itow.append(w)
                wvec.append(e)
    wtoi = {w: i for i, w in enumerate(itow)}
    wvec = torch.FloatTensor(wvec)
    return wtoi, wvec


def join_wordemb(allwords, wivwtoi, wivwvec, oovwtoi, oovwvec):
    itow, wvec = [], []
    for w in allwords:
        if w in oovwtoi:
            itow.append(w)
            wvec.append(oovwvec[oovwtoi[w]])
        elif w in wivwtoi:
            itow.append(w)
            wvec.append(wivwvec[wivwtoi[w]])
        else:
            itow.append(w)
            wvec.append(torch.zeros(300))
    wtoi = {w: i for i, w in enumerate(itow)}
    wvec = torch.stack(wvec)
    return wtoi, wvec
