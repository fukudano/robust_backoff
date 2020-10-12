import random
import torch
from collections import defaultdict


def get_ctxs(args, qws, maxctx=10, wnd=10):
    wtoi = args.wtoi
    # print(f"wtoi {len(wtoi)}")

    with open(args.corpus_vocab_file, "r") as h:
        x = [r.split(" ") for r in h.read().rstrip().split("\n")]
        w2f_cp = {w: int(f) for w, f in x}
    with open(args.corpus_file, "r") as h:
        lines = [l for l in h.readlines() if len(l) > 20]
    # print(f"w2f_cp {len(w2f_cp)}")
    # print(f"lines {len(lines)}")

    if args.train_emb:
        d = args.tra + args.dev + args.tes
        for s in d:
            for w, t in s:
                if w in w2f_cp:
                    w2f_cp[w] += 1
                else:
                    w2f_cp[w] = 1
        lines += [" ".join([w for w, t in s]) for s in d]

    # print("qws", len(qws))
    tws_cp = set(qws)
    # print(f"tws_cp {len(tws_cp)}")

    # print(f"maxctx {maxctx}")
    # print(f"wnd {wnd}")
    w2sents = defaultdict(list)
    for line in random.sample(lines, len(lines)):
        sent = line.split(" ")
        for ix, tw in enumerate(sent):
            if tw in tws_cp and len(w2sents[tw]) < maxctx:
                cl, cr = sent[:ix][-wnd:], sent[ix + 1:][:wnd]
                c = ["<pad>"] * (wnd - len(cl)) + cl + cr + ["<pad>"] * (wnd - len(cr))
                cwid = [wtoi.get(w, -1) for w in c]
                w2sents[tw].append(cwid)
    # print(f"w2sents {len(w2sents)}")

    # lens = np.array(list(map(len, w2sents.values())))
    # for th in [maxctx, maxctx // 10]:
    # print(f"th >= {th} :", (lens >= th).sum() / len(lens))

    wctx = []
    for w, sents in w2sents.items():
        sents += [[-1] * (wnd * 2)] * (maxctx - len(sents))
        wctx.append([w, torch.LongTensor(sents)])
    # print(f"wctx {len(wctx)}")
    return wctx
