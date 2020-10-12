def toefl(vocab, args):
    with open(args.toefl_file, "r") as h:
        lines = h.readlines()
    data = []
    for line in lines[1:]:
        _, _, l, t, r = line.rstrip().split("\t")
        if t == "M" and l not in vocab and r in vocab:
            data.append([l, r])
    return data
