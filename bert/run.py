import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0)
parser.add_argument('--seeds', default="range(5)")
args = parser.parse_args()

import os
import subprocess
cwd = os.getcwd().split("/")[-1]

#ark,t_pos,dcu,wnut,zhang,jnlpba,bc2gm,bc4chemd,bc5cdr,ncbi_disease

subprocess.check_output("mkdir -p log", shell=True)

dataset_dir = "dataset"
seeds = eval(args.seeds)

otfile = "log.txt"
open(f"log/{otfile}", "w").write("")
# print(otfile)

cmnarg = f"--cuda {args.cuda} --dataset {dataset_dir} --path {cwd} --log {otfile}"

for data in "ark,t_pos,dcu".split(","):
    for seed in seeds:
        cmd = f"python ../model/pos.py --data {data} --seed {seed} {cmnarg}"
        # print(cmd)
        # open(f"log/{otfile}", "a").write(cmd + "\n")
        subprocess.Popen(cmd, shell=True).communicate()

for data in "wnut,zhang".split(","):
    for seed in seeds:
        cmd = f"python ../model/ner.py --data {data} --seed {seed} {cmnarg}"
        # print(cmd)
        # open(f"log/{otfile}", "a").write(cmd + "\n")
        subprocess.Popen(cmd, shell=True).communicate()

for data in "bc2gm,bc4chemd,bc5cdr,ncbi_disease".split(","):
    for seed in seeds:
        cmd = f"python ../model/ner.py --data {data} --seed {seed} {cmnarg}"
        # print(cmd)
        # open(f"log/{otfile}", "a").write(cmd + "\n")
        subprocess.Popen(cmd, shell=True).communicate()
