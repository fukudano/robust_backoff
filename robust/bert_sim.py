import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dir', type=str, default=".")
parser.add_argument('--Ks', type=str, default="1,3,5")

parser.add_argument('--wordemb_file', type=str, default="glove.840B.300d.txt")
parser.add_argument('--wordfreq_file', type=str, default="freq_count.glove.840B.300d.txt.fix.txt")

parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--clip', type=float, default=1)
args = parser.parse_args()

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import random
import numpy as np
import torch
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

import warnings
warnings.simplefilter('ignore')

import itertools


def chain(x):
    return list(itertools.chain.from_iterable(x))


from dataset import sst
tra, dev, tes = sst()
tag_stoi = {"0": 0, "1": 1}

tess = []
for K in args.Ks.split(","):
    _, _, t = torch.load(f"{args.dir}/sst_{K}.{args.seed}.pt")
    tess.append(t)

import sys
sys.path.append("../util")
from wordemb import wordemb
args.wtoi, args.wvec, args.w2f = wordemb(args)
args.tsk_words = sorted(set(chain([s.split() for s, t in tra + dev + tes + sum(tess, [])])))
args.qws = [w for w in args.tsk_words if w not in args.wtoi]
args.tra = []
import sys
sys.path.append(f"../lstm/_sim")
from wemb import Wemb
wemb = Wemb(args)
wtoi, wvec = wemb.wtoi_, wemb.wvec_
wtoi["<unk>"] = len(wvec)
wvec = torch.cat([wvec, torch.zeros(1, 300)])

from more_itertools import chunked
from tqdm import tqdm

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from transformers import BertForSequenceClassification


def tokenize(words):
    split_tokens = []
    mask = []
    orig_wids = []
    for word in words:
        sub_tokens = []
        for token in tokenizer.basic_tokenizer.tokenize(word, never_split=tokenizer.all_special_tokens):
            for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                sub_tokens.append(sub_token)
        if len(sub_tokens) == 0:
            sub_tokens = [tokenizer.unk_token]
        split_tokens += sub_tokens
        mask += [True] + [False] * (len(sub_tokens) - 1)
        orig_wids += [wtoi.get(word, wtoi["<unk>"])] * len(sub_tokens)
    return split_tokens, orig_wids, mask


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.g2b = nn.Linear(300, 768)
        self.gate = nn.Linear(768, 1)

    def forward(self, lines):
        tknids, wids = [], []
        for line in lines:
            words = line.split()
            _tkns, _wids, _ = tokenize(words)
            tknids.append(torch.LongTensor(tokenizer.convert_tokens_to_ids(_tkns)))
            wids.append(torch.LongTensor(_wids))
        tknids = pad_sequence(tknids, True, tokenizer.pad_token_id).cuda()
        mask = tknids != tokenizer.pad_token_id
        wids = pad_sequence(wids, True, wtoi["<unk>"])

        # b = self.net.bert.embeddings(input_ids=tknids)
        # a = self.gate(b).softmax(-1)
        # g = self.g2b(wvec[wids].cuda())
        # o = (torch.stack([b, g], -1) @ a[:, :, :, None])[:, :, :, 0]
        # x = self.net(inputs_embeds=o, attention_mask=mask)[0]

        b = self.net.bert.embeddings(input_ids=tknids)
        a = self.gate(b).sigmoid()
        g = self.g2b(wvec[wids].cuda())
        o = (1 - a) * b + a * g
        x = self.net(inputs_embeds=o, attention_mask=mask)[0]

        # x = self.net(input_ids=tknids, attention_mask=mask)[0]
        return x


net = Net().cuda()
optim = torch.optim.Adam(net.parameters(), lr=args.lr)


def train_data(data):
    lines, tags = zip(*data)
    pred = net.forward(lines)
    gold = torch.LongTensor([tag_stoi[t] for t in tags]).cuda()
    loss = nn.functional.cross_entropy(pred, gold)
    return loss


def test(data):
    pred, gold = [], []
    for batch in chunked(data, args.batch):
        lines, tags = zip(*batch)
        with torch.no_grad():
            _pred = net.forward(lines).argmax(-1).tolist()
            pred.extend(_pred)
            gold.extend([tag_stoi[t] for t in tags])
    pred, gold = torch.LongTensor(pred), torch.LongTensor(gold)
    acc = (pred == gold).float().mean().item() * 100
    return acc


best = 0
for epoch in tqdm(range(args.epoch), ncols=0):
    net.train()
    x = random.sample(tra, len(tra))
    for batch in chunked(x, args.batch):
        loss = train_data(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optim.step()
        optim.zero_grad()

    net.eval()
    dev_acc = test(dev)
    tes_acc = test(tes)
    stdout = f"epoch {epoch}\ndev_acc {dev_acc:.2f}, tes_acc {tes_acc:.2f}"
    print(stdout)

    if dev_acc > best:
        best = dev_acc
        bestout = stdout
        bestmodel = net.state_dict()

stdout = [bestout]
net.load_state_dict(bestmodel)
net.eval()

for K, tes in zip(args.Ks.split(","), tess):
    dev_acc, tes_acc = test(dev), test(tes)
    stdout.append(f"K {K}: dev_acc {dev_acc:.2f}, tes_acc {tes_acc:.2f}")
stdout = "\n".join(stdout)
open(f"{args.dir}/log.txt", "a").write(f"{__file__} seed {args.seed}\n{stdout}\n")
# print(stdout)
