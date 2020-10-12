import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dir', type=str, default=".")
parser.add_argument('--Ks', type=str, default="1,3,5")

parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-3)
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

from dataset import sst
tra, dev, tes = sst()
tag_stoi = {"0": 0, "1": 1}

from more_itertools import chunked

import sys
sys.path.append("../util")
from wordemb import wordemb
wtoi, wvec, w2f = wordemb(args)
wtoi["<unk>"] = len(wvec)
wvec = torch.cat([wvec, torch.zeros(1, 300)])

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def Lstm(lstm, emb, lens):
    sorted_lens, perm_idx = lens.sort(0, descending=True)
    _, reverse_mapping = perm_idx.sort(0, descending=False)
    rest_idx = torch.arange(0, len(sorted_lens))[reverse_mapping]

    in_pack = pack_padded_sequence(emb[perm_idx], sorted_lens, batch_first=True)
    out_pack, _ = lstm(in_pack)
    out, _ = pad_packed_sequence(out_pack, padding_value=0, batch_first=True)
    out = out[rest_idx]
    return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=200,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.5)
        self.lin = nn.Linear(400, 2)

    def forward(self, lines):
        wids = pad_sequence([torch.LongTensor([wtoi.get(w, wtoi["<unk>"]) for w in line.split()]) for line in lines],
                            True, -1)
        x = wvec[wids].cuda()
        x = Lstm(self.lstm, x, (wids != -1).sum(-1))
        x = x.max(-2)[0]
        x = self.lin(x)
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

stdout = [bestout]
net.load_state_dict(bestmodel)
net.eval()

for K in args.Ks.split(","):
    tra, dev, tes = torch.load(f"{args.dir}/sst_{K}.{args.seed}.pt")
    dev_acc, tes_acc = test(dev), test(tes)
    stdout.append(f"K {K}: dev_acc {dev_acc:.2f}, tes_acc {tes_acc:.2f}")
stdout = "\n".join(stdout)
open(f"{args.dir}/log.txt", "a").write(f"{__file__} seed {args.seed}\n{stdout}\n")
