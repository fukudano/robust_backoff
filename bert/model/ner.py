import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--path', type=str, default="_sim")
parser.add_argument('--log', type=str, default="log.txt")
parser.add_argument('--data', default="zhang")
parser.add_argument('--dataset_dir', default="dataset")
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--max_word', type=int, default=100)
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
    return [list(zip(*s)) for s in d if len(s) <= args.max_word]


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
    o, d = [], []
    for org_sent, org_tag in data:
        tokens, orig_wids, mask = tokenize(tokenizer, org_sent)

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        mask = torch.ByteTensor([False] + mask + [False])

        ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokens))

        unkwid = wtoi["<unk>"]
        orig_wids = torch.LongTensor([unkwid] + orig_wids + [unkwid])

        tag_ids = torch.ones(len(ids)).fill_(-100).long()
        tag_ids[mask] = torch.LongTensor([tag_stoi[t] for t in org_tag])

        if len(tokens) <= tokenizer.max_len:
            o.append([ids, orig_wids, tag_ids])
            d.append([org_sent, org_tag])
    return o, d


[tra_, tra], [dev_, dev], [tes_, tes] = [prep(d) for d in [tra, dev, tes]]

words = sorted(set(chain([s for s, t in tra + dev + tes])))
trawords = set(chain([s for s, t in tra]))


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


def cond(w):
    return w not in trawords and w not in args.wtoi


class GoldNer:
    def __init__(self, tes: list):
        self.gold = tes

    def eval(self, pred):
        p, g, poov, goov = [], [], [], []
        assert (len(pred) == len(self.gold))
        for six, (ptags, (ws, gtags)) in enumerate(zip(pred, self.gold)):
            assert (len(ptags) == len(gtags))
            _p = [(six, b, e, t) for b, e, t in seq2span(ptags)]
            _g = [(six, b, e, t) for b, e, t in seq2span(gtags)]
            p += _p
            g += _g
            poov += [s for s in _p if any([cond(w) for w in ws[s[1]:s[2] + 1]])]
            goov += [s for s in _g if any([cond(w) for w in ws[s[1]:s[2] + 1]])]

        def calc(g, p):
            if len(g) == 0 or len(p) == 0:
                f = 0
            else:
                c = set(g) & set(p)
                f = 2 * len(c) / (len(g) + len(p))
            return f

        f = calc(g, p) * 100
        f_oov = calc(goov, poov) * 100
        return f, f_oov, [g, p, goov, poov]


dev_gold = GoldNer(dev)
tes_gold = GoldNer(tes)

import torch.nn as nn
from transformers import BertForTokenClassification
from crf import CRF


def lens2mask(lens):
    return pad_sequence([torch.ones(x).byte() for x in lens], True, False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tag_stoi) + 2)
        self.g2b = nn.Linear(300, 768)
        self.gate = nn.Linear(768, 1)
        self.crf = CRF(tag_stoi)

    def forward(self, inputs, wids, attention_mask, labels):
        b = self.net.bert.embeddings(input_ids=inputs)
        a = self.gate(b).sigmoid()
        g = self.g2b(wvec[wids].cuda())
        x = (1 - a) * b + a * g

        logits = self.net(inputs_embeds=x, attention_mask=attention_mask)[0]
        first_mask = labels != -100
        mask = lens2mask(first_mask.sum(-1)).cuda()
        logits = torch.zeros(*mask.shape, logits.shape[-1]).cuda().masked_scatter(mask[:, :, None], logits[first_mask])
        labels = torch.zeros(*mask.shape).long().cuda().masked_scatter(mask, labels[first_mask])
        return logits, mask, labels

    def train_batch(self, inputs, wids, attention_mask, labels):
        logits, mask, labels = self.forward(inputs, wids, attention_mask, labels)
        loss = self.crf.neg_log_likelihood_loss(logits, mask, labels)
        return loss

    def test_batch(self, inputs, wids, attention_mask, labels):
        logits, mask, labels = self.forward(inputs, wids, attention_mask, labels)
        _, path = self.crf._viterbi_decode(logits, mask)
        pred = [[tag_itos[i] for i in p[m]] for p, m in zip(path, mask)]
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
            sents, wids, mask, tags = batchify(batch)
            preds += net.test_batch(sents, wids, mask, tags)
    f, f_oov, output = gold.eval(preds)
    return f, f_oov, preds, output


best = 0
for epoch in range(args.epoch):
    net.train()
    x = random.sample(tra_, len(tra_))
    for batch in chunked(x, args.batch):
        sents, wids, mask, tags = batchify(batch)
        loss = net.train_batch(sents, wids, mask, tags)
        loss.backward()
        optim.step()
        optim.zero_grad()

    net.eval()
    dev_f, _, dev_pred, _ = test(dev_, dev_gold)

    out = f"epoch {epoch}\n"
    out += f"dev {dev_f:.2f}\n"

    if dev_f > best:
        best = dev_f
        best_out = out
        best_devpred = dev_pred

ta, to, _, tes_output = test(tes_, tes_gold)
best_out += f"tes {ta:.2f} {to:.2f}"
# print(best_out)

open(f"log/{args.log}", "a").write(args.data + "\n" + best_out + "\n")
torch.save(obj=best_devpred, f=f"log/dev.{args.data}.{args.seed}.pred")
torch.save(obj=tes_output, f=f"log/tes.{args.data}.{args.seed}.output")
