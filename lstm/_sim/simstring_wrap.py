import subprocess
import simstring


def w2s(w, l, r):
    o = []
    for i in range(l, r + 1):
        for j in range(len(w) - i + 1):
            o.append(w[j:j + i])
    return set(o)


def dic(x, y):
    return 2 * len(x & y) / (len(x) + len(y)) if (len(x) + len(y)) > 0 else 0


def jac(x, y):
    return len(x & y) / len(x | y) if len(x | y) > 0 else 0


def cos(x, y):
    return len(x & y) / (len(x) * len(y))**0.5 if (len(x) * len(y)) > 0 else 0


def ovr(x, y):
    return len(x & y) / min(len(x), len(y)) if min(len(x), len(y)) > 0 else 0


def mea(x, y, l, r, measure):
    x = w2s(' ' + x + ' ', l, r)
    y = w2s(' ' + y + ' ', l, r)
    return [dic, cos, jac, ovr][measure - 1](x, y)


class Simstring:
    def __init__(self, words, measure=3, n=3, be=True, unicode=True, file="sample.db"):
        self.n = n
        subprocess.check_output("mkdir -p db", shell=True)
        db = simstring.writer(f'./db/{file}', n, be, unicode)
        db.measure = measure
        for w in words:
            db.insert(w)
        db.close()
        db = simstring.reader(f"./db/{file}")
        db.measure = measure
        self.db = db

    def search(self, w, K, lb=0.0, same=False):
        for i in reversed(range(10)):
            self.db.threshold = i * 0.1
            try:
                ws = [x for x in self.db.retrieve(w) if same or x != w]
            except Exception as e:
                print(e)
                print(w)
                assert (False)
            if len(ws) >= K or ((i - 1) * 0.1) <= lb:
                break
        ws = sorted(ws, key=lambda x: -mea(w, x, self.n, self.n, self.db.measure))[:K]
        return ws
