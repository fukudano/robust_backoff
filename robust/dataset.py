def read(infile):
    data = []
    for line in open(infile, 'r'):
        tag, words = line.strip().split(" ||| ")
        if tag == '0' or tag == '1': tag = '0'
        if tag == '3' or tag == '4': tag = '1'
        if tag == '2': continue
        data.append([words, tag])
    return data


def sst():
    indir = "dataset/"
    files = ["train.txt", "dev.txt", "test.txt"]
    tra, dev, tes = [read(indir + f) for f in files]
    return tra, dev, tes


if __name__ == "__main__":
    tra, dev, tes = sst()
    for n, d in zip("tra,dev,tes".split(","), [tra, dev, tes]):
        ls, ts = zip(*d)
        ws = set(sum([l.split(" ") for l in ls], []))
        print(n, len(d), len(ws))
