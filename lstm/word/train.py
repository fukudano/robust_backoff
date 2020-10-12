import os
import argparse
import random
import numpy as np
import torch
import sys
from more_itertools import chunked
import string
import nltk
import itertools
sys.path.append("../util")
from wordemb import wordemb
from getctxs import get_ctxs
from wemb import Wemb


def chain(x):
    return lambda x: list(itertools.chain.from_iterable(x))


def main(args):
    stopwords = set(nltk.corpus.stopwords.words("english")) | set(string.punctuation)

    args.wtoi, args.wvec, args.w2f = wordemb(args)

    with open(args.corpus_vocab_file, "r") as h:
        x = [r.split(" ") for r in h.read().rstrip().split("\n")]
    w2f_cp = {w: int(f) for w, f in x}

    qws = [w for w, f in args.w2f.items() if w not in stopwords and w in w2f_cp and args.f_min < f < args.f_max]
    qws = random.sample(qws, args.n_tradev)
    # print(f"qws {len(qws)}")

    wctx = get_ctxs(args, qws, maxctx=args.maxctx, wnd=args.ctxwnd)

    N = args.n_dev
    x = random.sample(wctx, len(wctx))
    tra, dev = x[:-N], x[-N:]
    print(f"tra {len(tra)}, dev {len(dev)}")

    args.qws = list(set([w for w, _ in wctx]))
    net = Wemb(args).cuda()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    best = 0
    for epoch in range(args.epoch):
        net.train()
        lstloss = []
        x = random.sample(tra, len(tra))
        for batch in chunked(x, args.batch):
            loss = net.train_batch(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optim.step()
            optim.zero_grad()
            lstloss.append(loss.item())

        net.eval()
        lstloss = []
        for batch in chunked(dev, args.batch):
            with torch.no_grad():
                loss = net.train_batch(batch)
            lstloss.append(loss.item())
        dev_loss = np.mean(lstloss)
        # print(dev_loss)

        if dev_loss > best:
            best = dev_loss
            # best_loss = f"epoch {epoch}: {dev_loss}"
            bestmodel = net.state_dict()
    # print(best_loss)
    torch.save(bestmodel.state_dict(), "model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch', type=int, default=1000)
    parser.add_argument('--wordemb_file', type=str, default="glove.840B.300d.txt")
    parser.add_argument('--wordfreq_file', type=str, default="freq_count.glove.840B.300d.txt.fix.txt")

    parser.add_argument('--train_emb', action='store_true', default=True)
    parser.add_argument('--maxctx', type=int, default=10)
    parser.add_argument('--ctxwnd', type=int, default=10)
    parser.add_argument('--corpus_file', default="wikitext-103/train.txt")
    parser.add_argument('--corpus_vocab_file', default="wikitext-103/train.txt.vocab")

    parser.add_argument('--f_min', type=int, default=int(1e3))
    parser.add_argument('--f_max', type=int, default=int(1e5))
    parser.add_argument('--n_tradev', type=int, default=int(1e5))
    parser.add_argument('--n_dev', type=int, default=int(1e3))
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=1000)
    parser.add_argument('--clip', type=int, default=1)
    args = parser.parse_args()

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    main(args)
