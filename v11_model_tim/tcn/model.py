"""
TCN (Temporal Convolutional Network) untuk urutan IBI + ΔIBI (30 × 2, bermask).

Blok residual konvolusi dilatasi non-kausal (padding "same"). Input dan aktivasi setiap blok dikalikan mask, sehingga
nilai di posisi padding tidak memengaruhi output. Pooling = rata-rata bermask. Kontrak modul: lihat bersama/runner.py.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # v11_model_tim → paket `bersama`
from bersama import latih_dl as L  # noqa: E402

NAMA = "tcn"
INPUT = "urutan"
SETELAN = {"kanal": 32, "kernel": 3, "dilasi": (1, 2, 4, 8), "dropout": 0.1, **L.BAWAAN}
SETELAN_QUICK = {"epochs": 1}


class _Blok(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilasi, dropout):
        super().__init__()
        pad = dilasi * (kernel - 1) // 2
        self.c1 = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilasi)
        self.c2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilasi)
        self.n1, self.n2 = nn.BatchNorm1d(c_out), nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, m):                       # x (B, C, L) sudah bermask, m (B, 1, L)
        h = self.drop(torch.relu(self.n1(self.c1(x)))) * m
        h = self.drop(torch.relu(self.n2(self.c2(h)))) * m
        return torch.relu(h + self.skip(x)) * m


class TCN(nn.Module):
    def __init__(self, input_dim=2, kanal=32, kernel=3, dilasi=(1, 2, 4, 8), dropout=0.1, num_classes=3):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel TCN harus ganjil (padding 'same').")
        blok, c = [], input_dim
        for dl in dilasi:
            blok.append(_Blok(c, kanal, kernel, dl, dropout))
            c = kanal
        self.blok = nn.ModuleList(blok)
        self.fc = nn.Linear(kanal, num_classes)

    def forward(self, x, mask):                    # x (B, L, 2), mask (B, L)
        m = mask.unsqueeze(1)
        h = x.transpose(1, 2) * m
        for b in self.blok:
            h = b(h, m)
        return self.fc(h.sum(2) / m.sum(2).clamp(min=1))


def buat_net(setelan):
    return TCN(kanal=setelan["kanal"], kernel=setelan["kernel"], dilasi=tuple(setelan["dilasi"]),
               dropout=setelan["dropout"])


def latih(data, setelan, seed, device):
    return L.latih_paket_dl(lambda: buat_net(setelan), data, setelan, seed, device)


def prediksi(model, X, M, F, device):
    return L.prediksi_paket_dl(model, X, M, F, device)


def simpan(model, path):
    return L.simpan_paket_dl(model, path)


def info(model):
    return {"parameter": L.jumlah_parameter(model.net)}
