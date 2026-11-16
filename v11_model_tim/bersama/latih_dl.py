"""
latih_dl.py — loop training DL umum v11.

`latih_dl` sama persis dengan `sihedaf_cv.train_dl` (v3), dengan dua perbedaan:
- model dibuat lewat fungsi `buat_net()` (bukan hanya model di registry v2), dan
- bobot loss = class_weights(y) ** weight_power (bawaan 0,5 = bobot lunak, AGENTS.md).
Dengan weight_power = 1 dan model yang sama, hasilnya identik dengan v3 (tests/test_latih_dl.py, pola notebook v5).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from . import data as D

cv = D.cv
# Resep v3 + bobot lunak. Setiap model DL boleh menimpa lewat SETELAN-nya.
BAWAAN = dict(epochs=30, batch_size=256, lr=1e-3, weight_decay=1e-4, weight_power=0.5)


def latih_dl(buat_net: Callable[[], nn.Module], X, M, y, *, seed: int, device, epochs=30, batch_size=256,
             lr=1e-3, weight_decay=1e-4, weight_power=0.5):
    """Return (net dalam mode eval, mu, sd, detik). Urutan operasi = sihedaf_cv.train_dl (jangan diubah)."""
    cv.set_seed(seed)
    mu, sd = cv.zscore_fit(X, M)
    Xt = torch.from_numpy(cv.zscore_apply(X, M, mu, sd)).to(device)
    Mt = torch.from_numpy(M).to(device)
    yt = torch.from_numpy(y).to(device)
    n = len(y)
    n_batches = n // batch_size if n > batch_size else 1   # = drop_last, hindari batch 1 (BatchNorm)
    bs = batch_size if n > batch_size else n

    w = torch.tensor(cv.class_weights(y) ** weight_power, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    net = buat_net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator(device=device).manual_seed(seed)

    t0 = time.perf_counter()
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device, generator=gen)
        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs]
            opt.zero_grad(set_to_none=True)
            criterion(net(Xt[idx], Mt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return net.eval(), mu, sd, time.perf_counter() - t0


@dataclass
class PaketDL:
    """Model DL terlatih + normalisasi z-score dari data latihnya."""
    net: nn.Module
    mu: np.ndarray
    sd: np.ndarray
    setelan: dict


def latih_paket_dl(buat_net: Callable[[], nn.Module], data, setelan: dict, seed: int, device) -> PaketDL:
    """data = SimpleNamespace(X, M, F, y) dari runner. Kunci setelan training: lihat BAWAAN."""
    hp = {k: setelan.get(k, v) for k, v in BAWAAN.items()}
    net, mu, sd, _ = latih_dl(buat_net, data.X, data.M, data.y, seed=seed, device=device, **hp)
    return PaketDL(net, mu, sd, dict(setelan))


def prediksi_paket_dl(paket: PaketDL, X, M, F, device) -> np.ndarray:
    return cv.predict_dl(paket.net, X, M, paket.mu, paket.sd, device)


def simpan_paket_dl(paket: PaketDL, path: Path) -> Path:
    p = Path(path).with_suffix(".pt")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in paket.net.state_dict().items()},
                "mu": torch.from_numpy(paket.mu), "sd": torch.from_numpy(paket.sd), "setelan": paket.setelan}, p)
    return p


def jumlah_parameter(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
