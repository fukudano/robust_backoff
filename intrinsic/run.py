import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cudas', type=str, default="0")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--Ls', type=str, default="range(11)")
parser.add_argument('--Rs', type=str, default="range(11)")
parser.add_argument('--similarities', type=str, default="jaccard")
args = parser.parse_args()

from itertools import product
import subprocess
from time import sleep

Ls, Rs = [[int(x) for x in eval(xs)] for xs in [args.Ls, args.Rs]]
similarities = args.similarities.split(",")

lrs = [[l, r, s] for l, r, s in product(Ls, Rs, similarities) if not (l == 0 and r == 0)]
cudas = args.cudas.split(",")[:len(lrs)]
lrs_ = [lrs[i::len(cudas)] for i, c in enumerate(cudas)]
assert len(lrs) == sum(map(len, lrs_))

procs = {}
for lrss, cuda in zip(lrs_, cudas):
    cmds = []
    for L, R, sim in lrss:
        cmd = f"python train.py --cuda {cuda} --seed {args.seed} --epoch {epoch} --L {L} --R {R} --similarity {sim}"
        cmds.append(cmd)
    cmds = ";\\\n".join(cmds)
    print(cmds, "\n")
    proc = subprocess.Popen(cmds, shell=True, stdout=subprocess.PIPE)
    procs[proc.pid] = proc
    # with open(f"run.{cuda}.sh", "w") as h:
    #     h.write(cmds)
with open("log/log.txt", "w") as h:
    h.write("")

try:
    while procs:
        for pid in list(procs):
            if procs[pid].poll() is not None:
                procs[pid].terminate()
                del procs[pid]
                print(pid, "terminated")
        sleep(1)
except KeyboardInterrupt:
    for pid in list(procs):
        del procs[pid]
print("exit")
