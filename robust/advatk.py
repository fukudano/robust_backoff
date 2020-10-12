import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dir', type=str, default=".")
parser.add_argument('--Ks', type=str, default="1,3,5")
parser.add_argument('--N', type=int, default=10)
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

import random
import string
from nltk.corpus import stopwords as SW
from collections import defaultdict
punctuations = string.punctuation + ' '
stopwords = set(SW.words("english")) | set(punctuations)

MIN_LEN = 3


def init_keyboard_mappings():
    keyboard_mappings = defaultdict(lambda: [])
    keyboard = ["qwertyuiop", "asdfghjkl*", "zxcvbnm***"]
    row = len(keyboard)
    col = len(keyboard[0])

    dx = [0, -1, 1, 0, 0]
    dy = [0, 0, 0, -1, 1]

    for i in range(row):
        for j in range(col):
            for k in range(len(dx)):
                x_, y_ = i + dx[k], j + dy[k]
                if (x_ >= 0 and x_ < row) and (y_ >= 0 and y_ < col):
                    if keyboard[x_][y_] == '*' or keyboard[i][j] == '*':
                        continue
                    keyboard_mappings[keyboard[i][j]].append(keyboard[x_][y_])
    keyboard_mappings = dict(keyboard_mappings)
    return keyboard_mappings


keyboard_mappings = init_keyboard_mappings()


def drop(word):
    if len(word) < MIN_LEN or word in stopwords:
        return word
    rng = range(0, len(word))
    i = random.sample(rng, 1)[0]
    return word[:i] + word[i + 1:]


def swap(word):
    if len(word) < MIN_LEN or word in stopwords:
        return word
    rng = range(0, len(word) - 1)
    i = random.sample(rng, 1)[0]
    return word[:i] + word[i:i + 2][::-1] + word[i + 2:]


def add(word):
    if len(word) < MIN_LEN or word in stopwords:
        return word
    rng = range(0, len(word) + 1)
    i = random.sample(rng, 1)[0]
    key = get_keyboard_neighbors(word[i - 1]) if 0 <= i - 1 < len(word) else []\
     + get_keyboard_neighbors(word[i]) if 0 <= i < len(word) else []
    key = random.sample(key, 1)[0]
    return word[:i] + key + word[i:]


def subs(word):
    if len(word) < MIN_LEN or word in stopwords:
        return word
    rng = [i for i in range(0, len(word)) if word[i] in keyboard_mappings]
    if len(rng) == 0:
        return word
    i = random.sample(rng, 1)[0]
    key = get_keyboard_neighbors(word[i])[1:]
    key = random.sample(key, 1)[0]
    return word[:i] + key + word[i + 1:]


def get_keyboard_neighbors(ch):
    if ch not in keyboard_mappings:
        return [ch]
    return keyboard_mappings[ch]


def rand(word):
    func = random.sample("drop/swap/add/subs".split("/"), 1)[0]
    return eval(func)(word)


def attack_sent(words, prob=1, func="rand"):
    words_ = [eval(func)(w) if w not in specialtokens and random.random() <= prob else w for w in words]
    flag = [w != w_ for w, w_ in zip(words, words_) if w not in specialtokens]  # same length as tag
    return words_, flag


from dataset import sst
tra, dev, tes = sst()
tag_stoi = {"0": 0, "1": 1}

import torch as pt
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
specialtokens = set(tokenizer.special_tokens_map.values())

from transformers import BertForSequenceClassification
net = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).cuda()
net.load_state_dict(torch.load(f"{args.dir}/base.{args.seed}.pt"))


def tokenize(words):
    split_tokens = []
    mask = []
    for word in words:
        sub_tokens = []
        for token in tokenizer.basic_tokenizer.tokenize(word, never_split=tokenizer.all_special_tokens):
            for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                sub_tokens.append(sub_token)
        if len(sub_tokens) == 0:
            sub_tokens = [tokenizer.unk_token]
        split_tokens += sub_tokens
        mask += [True] + [False] * (len(sub_tokens) - 1)
    return split_tokens, mask


def maxgrads(words, gold):
    net.train()
    net.zero_grad()

    tkns, mask = tokenize(words)
    wixs = torch.LongTensor(mask).cumsum(0) - 1

    tknids = torch.LongTensor(tokenizer.convert_tokens_to_ids(tkns))
    tknids = tknids[None, :].cuda()
    mask = tknids != tokenizer.pad_token_id

    emb = net.bert.embeddings(input_ids=tknids).detach().requires_grad_(True)
    x = net(inputs_embeds=emb, attention_mask=mask)[0]  # B,2
    loss = nn.functional.cross_entropy(x, gold)
    loss.backward()

    grads = emb.grad.norm(dim=-1)[0]
    ixs = grads.argsort(descending=True)
    retwixs = wixs[ixs]
    retwixs = retwixs[np.sort(np.unique(retwixs, return_index=True)[1])]
    return retwixs


def evalscore(sents, gold):
    net.eval()

    sentids = []
    for words in sents:
        tkns, _ = tokenize(words)
        tknids = torch.LongTensor(tokenizer.convert_tokens_to_ids(tkns))
        sentids.append(tknids)
    sentids = pad_sequence(sentids, True, tokenizer.pad_token_id).cuda()
    mask = sentids != tokenizer.pad_token_id

    with torch.no_grad():
        emb = net.bert.embeddings(input_ids=sentids)
        x = net(inputs_embeds=emb, attention_mask=mask)[0]  # B,2
        score = x.softmax(-1)[:, gold]
    return score


def advatk(line, gold, K=1, N=10):
    words = line.split()
    gold = torch.LongTensor([tag_stoi[gold]]).cuda()

    words_atk, score = [x for x in words], 1
    wixs = maxgrads(words_atk, gold)[:K]
    for wix in wixs:
        w = words_atk[wix]

        typos = [rand(w) for _ in range(N)]
        _sents = [[(typo if i == wix else w) for i, w in enumerate(words_atk)] for typo in typos]

        _scores = evalscore(_sents, gold)
        min_scores, minix = _scores.min(0)
        if min_scores < score:
            words_atk = [x for x in _sents[minix]]
            score = min_scores
    line_ = " ".join(words_atk)
    return line_


def proc(d, K, N):
    return [[advatk(s, t, K=K, N=N), t] for s, t in d]


for K in map(int, args.Ks.split(",")):
    tes = proc(tes, K, args.N)
    torch.save([tra, dev, tes], f"{args.dir}/sst_{K}.{args.seed}.pt")
