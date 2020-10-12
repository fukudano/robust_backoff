import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--path', type=str, default="_sim")
parser.add_argument('--log', type=str, default="log.txt")
parser.add_argument('--data', default="ark")
parser.add_argument('--dataset_dir', default="dataset")
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch', type=int, default=16)
args = parser.parse_args()

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import random
import numpy as np
import torch
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

import itertools
from torch.nn.utils.rnn import pad_sequence
from more_itertools import chunked

import warnings
warnings.simplefilter('ignore')


def chain(x):
    return lambda x: list(itertools.chain.from_iterable(x))


def proc(d):
    return [list(zip(*s)) for s in d]


import sys
sys.path.append("../../lstm/model/")
exec(f"from dataset import {args.data}")
tra, dev, tes = [proc(d) for d in eval(args.data)(args)]
tag_itos = ["<pad>"] + sorted(set(chain([t for s, t in tra + dev + tes])))
tag_stoi = {s: i for i, s in enumerate(tag_itos)}

sys.path.append("../../util")
from wordemb import wordemb
args.wtoi, args.wvec, args.w2f = wordemb(args)
args.tsk_words = sorted(set(chain([s for s, t in tra + dev + tes])))
args.qws = [w for w in args.tsk_words if w not in args.wtoi]
args.train_emb = False

sys.path.append(f"../../lstm/{args.path}")
from wemb import Wemb
wemb = Wemb(args)
wtoi, wvec = wemb.wtoi_, wemb.wvec_
wtoi["<unk>"] = len(wvec)
wvec = torch.cat([wvec, torch.zeros(1, 300)])
for w in args.qws:
    if w not in wtoi:
        wtoi[w] = wtoi["<unk>"]

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize(self, words):
    split_tokens = []
    mask = []
    orig_wids = []
    for word in words:
        sub_tokens = []
        for token in self.basic_tokenizer.tokenize(word, never_split=self.all_special_tokens):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                sub_tokens.append(sub_token)
        if len(sub_tokens) == 0:
            sub_tokens = [self.unk_token]
        split_tokens += sub_tokens
        mask += [True] + [False] * (len(sub_tokens) - 1)
        orig_wids += [wtoi[word]] * len(sub_tokens)
    return split_tokens, orig_wids, mask


def prep(data):
    o = []
    for org_sent, org_tag in data:
        tokens, orig_wids, mask = tokenize(tokenizer, org_sent)

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        mask = torch.ByteTensor([False] + mask + [False])

        ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokens))

        unkwid = wtoi["<unk>"]
        orig_wids = torch.LongTensor([unkwid] + orig_wids + [unkwid])

        tag_ids = torch.ones(len(ids)).fill_(-100).long()
        tag_ids[mask] = torch.LongTensor([tag_stoi[t] for t in org_tag])

        o.append([ids, orig_wids, tag_ids])
    return o


tra_, dev_, tes_ = [prep(d) for d in [tra, dev, tes]]


def cond(w):
    return w not in trawords and w not in args.wtoi


words = sorted(set(chain([s for s, t in tra + dev + tes])))
trawords = set(chain([s for s, t in tra]))


class GoldPos:
    def __init__(self, tes):
        self.gold = chain([t for s, t in tes])
        self.mask = np.array([cond(w) for w in chain([s for s, t in tes])]).astype(bool)

    def eval(self, pred):
        pred = chain(pred)
        assert (len(pred) == len(self.gold))
        cor = np.array(pred) == np.array(self.gold)
        acc = cor.mean() * 100
        acc_oov = cor[self.mask].mean() * 100
        return acc, acc_oov


dev_gold = GoldPos(dev)
tes_gold = GoldPos(tes)

from transformers import BertForTokenClassification
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tag_stoi))
        self.g2b = nn.Linear(300, 768)
        self.gate = nn.Linear(768, 1)

    def forward(self, sents, wids):
        b = self.net.bert.embeddings(input_ids=sents)
        a = self.gate(b).sigmoid()
        g = self.g2b(wvec[wids].cuda())
        x = (1 - a) * b + a * g
        return x

    def train_batch(self, batch):
        sents, wids, mask, tags = batchify(batch)
        x = self.forward(sents, wids)
        loss = self.net(inputs_embeds=x, attention_mask=mask, labels=tags)[0]
        return loss

    def test_batch(self, batch):
        sents, wids, mask, tags = batchify(batch)
        x = self.forward(sents, wids)
        scores = self.net(inputs_embeds=x, attention_mask=mask)[0]
        pred = scores.max(-1)[1]
        pred = [[tag_itos[i] for i in p[m]] for p, m in zip(pred, (tags != -100))]
        return pred


net = Net().cuda()
optim = torch.optim.Adam(net.parameters(), lr=args.lr)


def batchify(batch):
    sents, orig_wids, tags = zip(*batch)
    mask = pad_sequence([torch.ones(len(s)) for s in sents], batch_first=True, padding_value=0).byte().cuda()
    sents = pad_sequence(sents, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()
    wids = pad_sequence(orig_wids, True, -1).cuda()
    tags = pad_sequence(tags, batch_first=True, padding_value=-100).cuda()
    return sents, wids, mask, tags


def test(data, gold):
    preds = []
    with torch.no_grad():
        for batch in chunked(data, args.batch):
            preds += net.test_batch(batch)
    acc, acc_oov = gold.eval(preds)
    return acc, acc_oov, preds


best = 0
for epoch in range(args.epoch):
    net.train()
    x = random.sample(tra_, len(tra_))
    for batch in chunked(x, args.batch):
        loss = net.train_batch(batch)
        loss.backward()
        optim.step()
        optim.zero_grad()

    net.eval()
    dev_acc, _, dev_pred = test(dev_, dev_gold)

    out = f"epoch {epoch}\n"
    out += f"dev {dev_acc:.2f}\n"

    if dev_acc > best:
        best = dev_acc
        best_out = out
        best_devpred = dev_pred
ta, to, _ = test(tes_, tes_gold)
best_out += f"tes {ta:.2f} {to:.2f}"
# print(best_out)

open(f"log/{args.log}", "a").write(args.data + "\n" + best_out + "\n")
torch.save(obj=best_devpred, f=f"log/dev.{args.data}.{args.seed}.pred")
