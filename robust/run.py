import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cudas', type=str, default="0")
parser.add_argument('--seeds', default="range(5)")
args = parser.parse_args()

import os
import subprocess
from time import sleep
cwd = os.getcwd().split("/")[-1]

subprocess.check_output("mkdir -p log", shell=True)

seeds = eval(args.seeds)

otfile = "log/log.txt"
open(otfile, "w").write("")
print(otfile)

cmds = []
for seed in seeds:
    cmnarg = f"--dir log --seed {seed} --cuda /cuda/"
    cmd = (f"python bert_pret.py {cmnarg};"
           f"python advatk.py {cmnarg};"
           f"python bert.py {cmnarg};"
           f"python bert_glv.py {cmnarg};"
           f"python bert_sim.py {cmnarg};"
           f"python lstm_unk.py {cmnarg};"
           f"python lstm_glv.py {cmnarg};"
           f"python lstm_sim.py {cmnarg};")
    cmds.append(cmd)

cudas = args.cudas.split(",")
cmds_chunked = [cmds[i::len(cudas)] for i, c in enumerate(cudas)]
assert len(cmds) == sum(map(len, cmds_chunked))

procs = {}
for cuda, cmd_batch in zip(cudas, cmds_chunked):
    cmd = "".join([cmd.replace("/cuda/", cuda) for cmd in cmd_batch])
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True)
    procs[proc.pid] = proc

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
print("finished")
