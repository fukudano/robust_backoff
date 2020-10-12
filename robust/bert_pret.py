import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dir', type=str, default=".")
parser.add_argument('--Ks', type=str, default="1,3,5")

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

from dataset import sst
tra, dev, tes = sst()
tag_stoi = {"0": 0, "1": 1}

from more_itertools import chunked
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from transformers import BertForSequenceClassification
net = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).cuda()


def forward(lines):
    tknids = [torch.LongTensor(tokenizer.encode(line)) for line in lines]
    tknids = pad_sequence(tknids, True, tokenizer.pad_token_id).cuda()
    mask = tknids != tokenizer.pad_token_id
    x = net(tknids, attention_mask=mask)[0]  # B,2
    return x


optim = torch.optim.Adam(net.parameters(), lr=args.lr)


def train_data(data):
    lines, tags = zip(*data)
    pred = forward(lines)
    gold = torch.LongTensor([tag_stoi[t] for t in tags]).cuda()
    loss = nn.functional.cross_entropy(pred, gold)
    return loss


def test(data):
    pred, gold = [], []
    for batch in chunked(data, args.batch):
        lines, tags = zip(*batch)
        with torch.no_grad():
            _pred = forward(lines).argmax(-1).tolist()
            pred.extend(_pred)
            gold.extend([tag_stoi[t] for t in tags])
    pred, gold = torch.LongTensor(pred), torch.LongTensor(gold)
    acc = (pred == gold).float().mean().item() * 100
    return acc


best = 0
for epoch in range(args.epoch):
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
    # print(stdout)

    if dev_acc > best:
        best = dev_acc
        bestout = stdout
        bestmodel = net.state_dict()

open(f"{args.dir}/log.txt", "a").write(f"{__file__} seed {args.seed}\n{bestout}\n")
torch.save(bestmodel, f"{args.dir}/base.{args.seed}.pt")
