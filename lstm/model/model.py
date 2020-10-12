import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from crf import CRF
from wemb import Wemb


def Lstm(lstm, emb, lens):
    sorted_lens, perm_idx = lens.sort(0, descending=True)
    _, reverse_mapping = perm_idx.sort(0, descending=False)
    rest_idx = torch.arange(0, len(sorted_lens))[reverse_mapping]

    in_pack = pack_padded_sequence(emb[perm_idx], sorted_lens, batch_first=True)
    out_pack, _ = lstm(in_pack)
    out, _ = pad_packed_sequence(out_pack, padding_value=0, batch_first=True)
    out = out[rest_idx]
    return out


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.wemb = Wemb(args)
        self.drop = nn.Dropout(args.dropout)
        odim = len(args.tag_stoi)
        if args.ner:
            self.crf = CRF(args.tag_stoi)
            odim = len(args.tag_stoi) + 2
        if not args.lstm:
            self.ffn = nn.Sequential(nn.Linear(300, 400), nn.ReLU(), nn.Dropout(args.dropout))
        else:
            self.lstm = nn.LSTM(input_size=300,
                                hidden_size=200,
                                num_layers=2,
                                bias=True,
                                batch_first=True,
                                dropout=args.dropout,
                                bidirectional=True)
        self.hid2tag = nn.Linear(400, odim)

    def forward(self, batch):
        mask = pad_sequence([torch.ones(len(x)) for x in batch], True, 0).byte().cuda()
        if self.args.fix:
            with torch.no_grad():
                x = self.wemb.eval()(batch)
        else:
            x = self.wemb(batch)
        x = self.drop(x)
        if not self.args.lstm:
            x = self.ffn(x)
        else:
            x = Lstm(self.lstm, x, mask.sum(-1))
        x = self.hid2tag(x)
        return x, mask

    def train_batch(self, batch, tags):
        x, mask = self.forward(batch)
        tag_ids = pad_sequence([torch.LongTensor([self.args.tag_stoi[t] for t in s]) for s in tags], True,
                               self.args.tag_stoi["<pad>"]).cuda()
        if not self.args.ner:
            loss = nn.functional.cross_entropy(x[mask], tag_ids[mask])
        else:
            loss = self.crf.neg_log_likelihood_loss(x, mask, tag_ids)
        return loss

    def test_batch(self, batch):
        x, mask = self.forward(batch)
        if not self.args.ner:
            path = x.max(-1)[1]
        else:
            _, path = self.crf._viterbi_decode(x, mask)
        path = [p[m].tolist() for p, m in zip(path, mask)]
        tags = [[self.args.tag_itos[i] for i in s] for s in path]
        return tags
