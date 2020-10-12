import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--L', type=int, default=7)  #
parser.add_argument('--R', type=int, default=10)  #
parser.add_argument('--similarity', type=str, default="jaccard")
parser.add_argument('--wordemb_file', type=str, default="/data/fukuda/glove/glove.840B.300d.txt")
parser.add_argument(
    '--wordfreq_file',
    type=str,
    default=
    "/net/setagaya/disk6/fukuda/git/rareword/compact_reconstruction/resources/freq_count.glove.840B.300d.txt.fix.txt")

parser.add_argument('--cnn_dim', type=int, default=100)
parser.add_argument('--cnn_filter', type=str, default="1,3,5,7")
parser.add_argument('--cnn_dropout', type=float, default=0.3)
parser.add_argument('--word_dim', type=int, default=300)

parser.add_argument('--f_min', type=int, default=int(1e3))
parser.add_argument('--f_max', type=int, default=int(1e5))
parser.add_argument('--n_tradev', type=int, default=int(1e5))
parser.add_argument('--n_dev', type=int, default=int(1e3))
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch', type=int, default=1000)
parser.add_argument('--clip', type=int, default=1)

parser.add_argument('--rw_file', type=str, default="dataset/rw/rw.txt", help="Stanford Rare Word Similarity datasets")
parser.add_argument('--card_file',
                    type=str,
                    default="/net/setagaya/disk6/fukuda/git/rareword/card-660/dataset.tsv",
                    help="CARD-660 dataset")
parser.add_argument('--toefl_file',
                    type=str,
                    default="/net/setagaya/disk6/fukuda/git/rareword/TOEFL-Spell/Annotations.tsv",
                    help="TOEFL-Spell dataset")
args = parser.parse_args()

args.cnn_filter = [int(x) for x in args.cnn_filter.split(",")]

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
import numpy as np
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

import sys
import itertools
from more_itertools import chunked


def chain(x):
    return list(itertools.chain.from_iterable(x))


def cosine(v1, v2):
    n1 = np.linalg.norm(v1) + 1e-10
    n2 = np.linalg.norm(v2) + 1e-10
    return np.dot(v1, v2) / n1 / n2


if not os.path.exists("wotdemb.obj"):
    sys.path.append("../util")
    from wordemb import wordemb
    wordembobj = wordemb(args)
    torch.save(obj=wordembobj, f="wordemb.obj")
wordembobj = torch.load("wordemb.obj")
args.wtoi, args.wvec, args.w2f = wordembobj
# args.wtoi, args.wvec, args.w2f = wordemb(args)
# print(f"args.wtoi {len(args.wtoi)}")

from rwcard import CARD
card = CARD(args=args)

from toefl import toefl
_toefl = toefl(vocab=args.wtoi, args=args)
toefl_ws = [l for l, r in _toefl]

word_qws = [w for w in args.wtoi.keys() if args.f_min < args.w2f[w] < args.f_max]
if len(word_qws) > int(args.n_tradev):
    word_qws = random.sample(word_qws, int(args.n_tradev))

qws = list(set((word_qws + card.ws + toefl_ws)))
N = args.n_dev
x = random.sample(word_qws, len(word_qws))
tra, dev = x[:-N], x[-N:]
# print("word_qws", len(word_qws), "qws", len(qws))

out = ""
best = 0

from wemb import Wemb
net = Wemb(args=args, qws=qws).cuda()
optim = torch.optim.Adam(params=net.parameters(), lr=args.lr)

for epoch in range(args.epoch):
    net.train()
    lstloss = []
    for ws in chunked(random.sample(tra, len(tra)), args.batch):
        loss = net.train_ws(ws=ws)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optim.step()
        optim.zero_grad()
        lstloss.append(loss.item())

    net.eval()
    lstloss = []
    for ws in chunked(dev, args.batch):
        with torch.no_grad():
            loss = net.train_ws(ws=ws)
        lstloss.append(loss.item())
    cos_dev = -np.mean(lstloss) * 100

    with torch.no_grad():
        es = net.pred(ws=card.ws).cpu().detach()
        m = torch.ByteTensor([w in args.wtoi for w in card.ws])
        es[m] = args.wvec[[args.wtoi[w] for w in card.ws if w in args.wtoi]]
        r, wivr, oovr, oovrate, sims = card.evaluate(es.tolist(), args.wtoi, ws=card.ws)

    with torch.no_grad():
        cs = []
        for x in chunked(_toefl, 10000):
            bmis, btrg = zip(*x)
            bmis = list(bmis)
            with torch.no_grad():
                es = net.pred(ws=bmis).cpu().detach().numpy()
            for e, trg in zip(es, btrg):
                t = args.wvec[args.wtoi[trg]]
                c = cosine(t, e)
                cs.append(c)
        cos = np.mean(cs) * 100

    _out = f"L {args.L}, R {args.R}, sim {args.similarity}, epoch {epoch}\n"
    _out += f"cos_dev {cos_dev:.2f}\n"
    _out += f"CARD all {r:2.1f}, wiv {wivr:2.1f}, oov {oovr:2.1f} (OOV: {oovrate:2.1f})\n"
    _out += f"TOEFL {cos:.1f}\n"

    if cos_dev > best:
        best = cos_dev
        bestout = _out
        bestmodel = net.state_dict()
        bestargs = args
        # bestsims = sims
    # tqdm.write(_out)
    out += _out

# out += f"\nbest\n{bestout}\n"
open("log/log.txt", "a").write(bestout)

save_obj = {"state_dict": bestmodel}
for k in vars(args).keys():
    if k not in "wtoi,wvec,w2f".split(","):
        save_obj[k] = getattr(bestargs, k)
torch.save(obj=save_obj, f=f"log/model.pt")
