import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dir', type=str, default=".")
parser.add_argument('--Ks', type=str, default="1,3,5")
parser.add_argument('--batch', type=int, default=16)
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

tag_stoi = {"0": 0, "1": 1}

from more_itertools import chunked
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from transformers import BertForSequenceClassification
net = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).cuda()
net.load_state_dict(torch.load(f"{args.dir}/base.{args.seed}.pt"))


def forward(lines):
    tknids = [torch.LongTensor(tokenizer.encode(line)) for line in lines]
    tknids = pad_sequence(tknids, True, tokenizer.pad_token_id).cuda()
    mask = tknids != tokenizer.pad_token_id
    x = net(tknids, attention_mask=mask)[0]  # B,2
    return x


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


net.eval()

stdout = []
for K in args.Ks.split(","):
    tra, dev, tes = torch.load(f"{args.dir}/sst_{K}.{args.seed}.pt")
    dev_acc, tes_acc = test(dev), test(tes)
    stdout.append(f"K {K}: dev_acc {dev_acc:.2f}, tes_acc {tes_acc:.2f}")
stdout = "\n".join(stdout)
open(f"{args.dir}/log.txt", "a").write(f"{__file__} seed {args.seed}\n{stdout}\n")
