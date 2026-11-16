"""
InceptionTime (satu jaringan, bukan ensemble 5) untuk urutan IBI + ΔIBI (30 × 2, bermask).

Modul Inception: bottleneck 1×1 → konvolusi kernel (3, 5, 9) + cabang max-pool → 1×1, digabung (4 × filter kanal),
BatchNorm, ReLU. Shortcut residual setiap 3 modul. Semua cabang dikalikan mask; max-pool mengabaikan padding.
Kernel asli InceptionTime (10/20/40) lebih panjang dari window 30 detak, jadi diperkecil. Kontrak: bersama/runner.py.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # v11_model_tim → paket `bersama`
from bersama import latih_dl as L  # noqa: E402

NAMA = "inceptiontime"
INPUT = "urutan"
SETELAN = {"filter": 16, "kernel": (3, 5, 9), "kedalaman": 6, "dropout": 0.0, **L.BAWAAN}
SETELAN_QUICK = {"epochs": 1}
_SANGAT_KECIL = -1e4            # pengganti nilai padding saat max-pool (bukan -inf: menghindari NaN)


class _ModulInception(nn.Module):
    def __init__(self, c_in, filter, kernel):
        super().__init__()
        self.bottleneck = nn.Conv1d(c_in, filter, 1, bias=False)
        self.konv = nn.ModuleList(nn.Conv1d(filter, filter, k, padding=k // 2, bias=False) for k in kernel)
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.konv_pool = nn.Conv1d(c_in, filter, 1, bias=False)
        self.bn = nn.BatchNorm1d(filter * (len(kernel) + 1))

    def forward(self, x, m):                       # x (B, C, L) sudah bermask, m (B, 1, L)
        z = self.bottleneck(x)
        p = self.pool(x.masked_fill(m == 0, _SANGAT_KECIL)) * m
        h = torch.cat([k(z) for k in self.konv] + [self.konv_pool(p)], dim=1) * m
        return torch.relu(self.bn(h)) * m


class InceptionTime(nn.Module):
    def __init__(self, input_dim=2, filter=16, kernel=(3, 5, 9), kedalaman=6, dropout=0.0, num_classes=3):
        super().__init__()
        if any(k % 2 == 0 for k in kernel):
            raise ValueError("kernel InceptionTime harus ganjil (padding 'same').")
        c_out = filter * (len(kernel) + 1)
        self.modul, self.shortcut = nn.ModuleList(), nn.ModuleList()
        c, c_res = input_dim, input_dim
        for i in range(kedalaman):
            self.modul.append(_ModulInception(c, filter, kernel))
            c = c_out
            if i % 3 == 2:
                self.shortcut.append(nn.Sequential(nn.Conv1d(c_res, c_out, 1, bias=False), nn.BatchNorm1d(c_out)))
                c_res = c_out
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c_out, num_classes)

    def forward(self, x, mask):                    # x (B, L, 2), mask (B, L)
        m = mask.unsqueeze(1)
        h = x.transpose(1, 2) * m
        res, j = h, 0
        for i, modul in enumerate(self.modul):
            h = modul(h, m)
            if i % 3 == 2:
                h = torch.relu(h + self.shortcut[j](res)) * m
                res, j = h, j + 1
        return self.fc(self.drop(h.sum(2) / m.sum(2).clamp(min=1)))


def buat_net(setelan):
    return InceptionTime(filter=setelan["filter"], kernel=tuple(setelan["kernel"]), kedalaman=setelan["kedalaman"],
                         dropout=setelan["dropout"])


def latih(data, setelan, seed, device):
    return L.latih_paket_dl(lambda: buat_net(setelan), data, setelan, seed, device)


def prediksi(model, X, M, F, device):
    return L.prediksi_paket_dl(model, X, M, F, device)


def simpan(model, path):
    return L.simpan_paket_dl(model, path)


def info(model):
    return {"parameter": L.jumlah_parameter(model.net)}
