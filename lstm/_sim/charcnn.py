import itertools
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import math
import string


def chain(x):
    return list(itertools.chain.from_iterable(x))


class CharCNN(nn.Module):
    def __init__(self, hdim, dropout=0.3):
        super(CharCNN, self).__init__()
        self.hdim = hdim
        self.ctoi = {c: i for i, c in enumerate(["<unk>"] + list(string.printable))}
        self.char_emb = nn.Embedding(len(self.ctoi), hdim)
        self.filter_num_width = [1, 3, 5, 7]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=hdim,
                          out_channels=hdim,
                          kernel_size=filter_width,
                          padding=math.floor((filter_width - 1) / 2)), nn.ReLU())
            for filter_width in self.filter_num_width
        ])
        self.linear = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hdim * len(self.filter_num_width), hdim),
                                    nn.LayerNorm(hdim))

    def forward(self, ws):
        cids = [torch.LongTensor([self.ctoi.get(c, self.ctoi["<unk>"]) for c in "<" + w + ">"]) for w in ws]  # B, C
        cids = pad_sequence(cids, True, -1)  # B,C
        _x = torch.cat([self.char_emb.weight, torch.zeros(1, self.hdim).cuda()])
        x = _x[cids].transpose(1, 2)  # B,C -> B, C, D, -> B, D, C
        x = [torch.max(conv(x), dim=-1)[0] for conv in self.convs]  # B, D, C -> 4, B, D, C -> 4, B, D
        x = torch.cat(x, dim=1)  # 4, B, D -> B, 4D
        x = self.linear(x)  # B, 4D -> B, D
        return x
