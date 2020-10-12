import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model', type=str, default="log/model.pt")

parser.add_argument('--wordemb_file', type=str, default="/data/fukuda/glove/glove.840B.300d.txt")
parser.add_argument(
    '--wordfreq_file',
    type=str,
    default=
    "/net/setagaya/disk6/fukuda/git/rareword/compact_reconstruction/resources/freq_count.glove.840B.300d.txt.fix.txt")

parser.add_argument('--rw_file', type=str, default="dataset/rw/rw.txt")
parser.add_argument('--card_file', type=str, default="/net/setagaya/disk6/fukuda/git/rareword/card-660/dataset.tsv")
parser.add_argument('--toefl_file',
                    type=str,
                    default="/net/setagaya/disk6/fukuda/git/rareword/TOEFL-Spell/Annotations.tsv")
args = parser.parse_args()

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import random
import numpy as np
import torch
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from more_itertools import chunked


def cosine(v1, v2):
    n1 = np.linalg.norm(v1) + 1e-10
    n2 = np.linalg.norm(v2) + 1e-10
    return np.dot(v1, v2) / n1 / n2


saved_obj = torch.load(args.model)
for k in saved_obj.keys():
    if k != "state_dict":
        setattr(args, k, saved_obj[k])

import sys
sys.path.append("../util")
from util import wordemb
args.wtoi, args.wvec, args.w2f = wordemb(args)

from rwcard import CARD
card = CARD(args=args)

from toefl import toefl
_toefl = toefl(vocab=args.wtoi, args=args)
toefl_ws = [l for l, r in _toefl]

qws = sorted(set((card.ws + toefl_ws)))

from wemb import Wemb
net = Wemb(args=args, qws=qws).cuda()
net.load_state_dict(saved_obj["state_dict"])

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

_out = f"L {args.L}, R {args.R}, sim {args.similarity}\n"
_out += f"CARD all {r:2.1f}, wiv {wivr:2.1f}, oov {oovr:2.1f} (OOV: {oovrate:2.1f})\n"
_out += f"TOEFL {cos:.1f}\n"
print(_out)
