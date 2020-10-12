import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch', type=int, default=50)
parser.add_argument('--data', default="ark")
parser.add_argument('--dataset_dir', default="dataset")
parser.add_argument('--wordemb_file', type=str, default="glove.840B.300d.txt")
parser.add_argument('--wordfreq_file', type=str, default="freq_count.glove.840B.300d.txt.fix.txt")
parser.add_argument('--path', default="../_app")
parser.add_argument('--log', default="log/log.txt")

parser.add_argument('--train_emb', action='store_true', default=False)
parser.add_argument('--fix', action="store_true")
parser.add_argument('--lstm', action="store_true")
parser.add_argument('--corpus_file', default="dataset/wikitext-103/train.txt")
parser.add_argument('--corpus_vocab_file', default="dataset/wikitext-103/train.txt.vocab")

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip', type=int, default=1)
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
from tqdm import tqdm
import itertools
chain = lambda x: list(itertools.chain.from_iterable(x))

import util

import sys
sys.path.append("../util")
from wordemb import wordemb
glv_wtoi, glv_wvec, glv_w2f = wordemb(args)
args.wtoi, args.wvec = glv_wtoi, torch.cat([glv_wvec, torch.zeros(1, 300)])

exec(f"from dataset import {args.data}")
tra, dev, tes = eval(args.data)(args)

sys.path.append("../word")
sys.path.append(args.path)

wtoi, words, args.tag_stoi, args.tag_itos = util.build_stoi(tra + dev + tes)

args.tsk_words = words
args.qws = list(set([w for w in words if w not in glv_wtoi]))

dev_gold = util.GoldPos(dev, tra, glv_wtoi)
tes_gold = util.GoldPos(tes, tra, glv_wtoi)

args.ner = False
from model import Net
net = Net(args).cuda()
optim = torch.optim.Adam(net.parameters(), args.lr)


def test(data, gold):
    net.eval()
    pred = []
    for batch in chunked(data, args.batch):
        sents, tags = list(zip(*map(lambda x: list(zip(*x)), batch)))
        with torch.no_grad():
            pred.extend(net.test_batch(sents))
    f, f_wiv, f_oov = gold.eval(pred)
    return (f, f_wiv, f_oov), pred


best = 0
for epoch in tqdm(range(args.epoch), ncols=0, disable=False):
    net.train()
    x = random.sample(tra, len(tra))
    for batch in tqdm(list(chunked(x, args.batch)), ncols=0, disable=True):
        sents, tags = list(zip(*map(lambda x: list(zip(*x)), batch)))
        loss = net.train_batch(sents, tags)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optim.step()
        optim.zero_grad()

    fs_dev, pred_dev = test(dev, dev_gold)
    fs_tes, _ = test(tes, tes_gold)

    stdout = "\n".join([
        "epoch {}".format(epoch),
        "dev all {:.2f} / vec wiv {:.2f} oov {:.2f}".format(*fs_dev),
        "tes all {:.2f} / vec wiv {:.2f} oov {:.2f}".format(*fs_tes),
    ])
    if fs_dev[0] > best:
        best = fs_dev[0]
        bestres = stdout
        # bestmodel = net.state_dict()
        # bestpred = pred_dev
    # print(stdout)
open(args.log, "a").write(args.data + "\n" + bestres + "\n")
