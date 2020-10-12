import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--wordemb_file', type=str, default="glove.840B.300d.txt")
parser.add_argument('--wordfreq_file', type=str, default="freq_count.glove.840B.300d.txt.fix.txt")
parser.add_argument('--toefl_file', type=str, default="dataset/TOEFL-Spell/Annotations.tsv")
parser.add_argument('--output_file', type=str, default="toefl.txt")
args = parser.parse_args()


def toefl(vocab, args):
    with open(args.toefl_file, "r") as h:
        lines = h.readlines()
    data = []
    for line in lines[1:]:
        _, _, l, t, r = line.rstrip().split("\t")
        if t == "M" and l not in vocab and r in vocab:
            data.append([l, r])
    return data


import sys
sys.path.append("../util")
from wordemb import wordemb
args.wtoi, args.wvec, args.w2f = wordemb(args)

data = toefl(args.wtoi, args)
stdout = "\n".join([" ".join(r) for r in data])
with open(args.output_file, "w") as h:
    h.write(stdout)
