def read(file, d="\t"):
    sents = []
    sent = []
    for line in open(file, "r"):
        line = line.rstrip()
        if line == "":
            if sent != []:
                sents.append(sent)
                sent = []
        else:
            word, tag = line.split(d)
            sent.append((word, tag))
    else:
        if sent != []:
            sents.append(sent)
    return sents


def read_(file):
    sents = []
    for line in open(file, "r"):
        sent = []
        for wt in line.rstrip().split():
            w, t = wt.rsplit("_", 1)
            sent.append([w, t])
        if len(sent) > 0:
            sents.append(sent)
    return sents


def IOBES_IOB2(insents):
    outsents = []
    for insent in insents:
        outsent = []
        for w, t in insent:
            if t.startswith("E-"):
                t = "I-" + t[2:]
            elif t.startswith("S-"):
                t = "B-" + t[2:]
            outsent.append([w, t])
        outsents.append(outsent)
    return outsents


def IOB_IOB2(insents):
    outsents = []
    for insent in insents:
        outsent = []
        for i, (w, t) in enumerate(insent):
            if t.startswith("I-") and (i == 0 or insent[i - 1][1] == "O" or insent[i - 1][1][2:] != t[2:]):
                t = "B-" + t[2:]
            outsent.append([w, t])
        outsents.append(outsent)
    return outsents


# Twitter POS
# GATE Twitter part-of-speech (twitie_tag) contains ARK, T-POS, DCU
# ARK Gimpel 2011
# Part-of-speech tagging for twitter: annotation, features, and experiments.
# T-POS Ritter 2011
# Named entity recognition in tweets: An experimental study.
# DCU foster 2011
# #hardtoparse: POS Tagging and Parsing the Twitterverse.

# Bio NER
# JNLPBA Collier 2004
# Introduction to the bio-entity recognition task at JNLPBA.
# BC2GM Smith 2008
# Overview of biocreative ii gene mention recognition
# BC4CHEMD Lu 2015
# CHEMDNER system with mixed conditional random fields and multi-scale word clustering
# BC5CDR Leaman 2016
# NCBI-disease Leaman 2016
# TaggerOne: joint named entity recognition and normalization with semi-Markov Models.

# Twitter NER
# WNUT17 Derczynski 2017
# Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition
# Twitter Zhang 2018
# Adaptive Co-Attention Network for Named Entity Recognition in Tweets


def ark(args):  # POS Twitter
    indir = args.dataset_dir + "/ark/"
    files = "oct27.train", "oct27.dev", "oct27.test"
    return [read(indir + file, d="\t") for file in files]
    # coverage tes/tra : words 70.60, vocab 36.66
    # coverage tra/glove : words 89.16, vocab 73.68
    # coverage tes/glove : words 89.39, vocab 77.85


def t_pos():  # POS Twitter
    indir = args.dataset_dir + "/t_pos/"
    files = "ritter_train.stanford", "ritter_dev.stanford", "ritter_eval.stanford"
    return [read_(indir + file) for file in files]
    # coverage tes/tra : words 73.99, vocab 48.48
    # coverage tra/glove : words 93.04, vocab 82.00
    # coverage tes/glove : words 92.23, vocab 84.97


def dcu(args):  # POS Twitter
    indir = args.dataset_dir + "/dcu/"
    files = "ritter_train.stanford", "foster_dev.stanford", "foster_eval.stanford"
    return [read_(indir + file) for file in files]
    # coverage tes/tra : words 63.71, vocab 36.15
    # coverage tra/glove : words 93.04, vocab 82.00
    # coverage tes/glove : words 95.64, vocab 90.64


def wnut(args):  # NER Twitter
    indir = args.dataset_dir + "/wnut/"
    files = "wnut17train.conll", "emerging.dev.conll", "emerging.test.annotated"
    return [read(indir + file, d="\t") for file in files]
    #coverage tes/tra : words 78.11, vocab 36.23
    #coverage tra/glove : words 91.65, vocab 71.04
    #coverage tes/glove : words 94.79, vocab 82.07


def zhang(args):  # NER Twitter
    def read(file, d="\t"):
        sents = []
        sent = []
        for line in open(file, "r"):
            line = line.rstrip()
            if line.startswith("IMGID:"):
                pass
            elif line == "":
                if sent != []:
                    sents.append(sent)
                    sent = []
            else:
                word, tag = line.split(d)
                sent.append((word, tag))
        else:
            if sent != []:
                sents.append(sent)
        return sents

    indir = args.dataset_dir + "/zhang/"
    files = "train", "dev", "test"
    return [IOB_IOB2(read(indir + file, d="\t")) for file in files]
    # coverage tes/tra : words 72.89, vocab 30.96
    # coverage tra/glove : words 83.65, vocab 57.16
    # coverage tes/glove : words 83.66, vocab 58.68


def jnlpba(args):
    indir = args.dataset_dir + "/jnlpba/"
    files = "train.tsv", "devel.tsv", "test.tsv"
    return [IOBES_IOB2(read(indir + file, d="\t")) for file in files]
    # coverage tes/tra : words 94.95, vocab 67.15
    # coverage tra/glove : words 96.42, vocab 67.44
    # coverage tes/glove : words 96.23, vocab 79.31


def bc2gm(args):
    indir = args.dataset_dir + "/bc2gm/"
    files = "train.tsv", "devel.tsv", "test.tsv"
    return [IOBES_IOB2(read(indir + file, d="\t")) for file in files]
    #coverage tes/tra : words 94.83, vocab 62.83
    #coverage tra/glove : words 98.11, vocab 81.18
    #coverage tes/glove : words 98.07, vocab 85.90


def bc4chemd(args):
    indir = args.dataset_dir + "/bc4chemd/"
    files = "train.tsv", "devel.tsv", "test.tsv"
    return [IOBES_IOB2(read(indir + file, d="\t")) for file in files]
    # coverage tes/tra : words 95.81, vocab 56.24
    # coverage tra/glove : words 98.12, vocab 80.38
    # coverage tes/glove : words 98.12, vocab 81.44


def bc5cdr(args):
    indir = args.dataset_dir + "/bc5cdr/"
    files = "train.tsv", "devel.tsv", "test.tsv"
    return [IOBES_IOB2(read(indir + file, d="\t")) for file in files]
    # coverage tes/tra : words 92.98, vocab 56.09
    # coverage tra/glove : words 99.17, vocab 94.69
    # coverage tes/glove : words 99.16, vocab 94.58


def ncbi_disease(args):
    indir = args.dataset_dir + "/ncbi_disease/"
    files = "train.tsv", "devel.tsv", "test.tsv"
    return [IOBES_IOB2(read(indir + file, d="\t")) for file in files]
    # coverage tes/tra : words 94.40, vocab 76.58
    # coverage tra/glove : words 98.32, vocab 88.56
    # coverage tes/glove : words 98.41, vocab 94.76


def conll(args):
    indir = args.dataset_dir + "/conll/"
    files = "eng.train", "eng.testa", "eng.testb"
    return [read(indir + file, d=" ") for file in files]
