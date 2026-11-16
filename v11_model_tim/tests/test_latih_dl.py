import numpy as np
import torch
from types import SimpleNamespace

from bersama import latih_dl
from sihedaf_models import build_model

KW = dict(input_dim=2, d_model=32, nhead=2, dim_ff=64, num_classes=3, dropout=0.1)


def test_paritas_dengan_v3(d, dev):
    """Pola notebook v5 sel 4: 2 epoch, latih = window idx % 5 != 0, uji 5.000 window. ±1 menit di GPU."""
    cv = latih_dl.cv
    tr = np.arange(len(d.y)) % 5 != 0
    te = np.flatnonzero(~tr)[:5000]
    a = latih_dl.latih_dl(lambda: build_model("MicroTransformer", **KW), d.X[tr], d.M[tr], d.y[tr],
                          seed=7, device=dev, epochs=2, weight_power=1.0)
    b = cv.train_dl("MicroTransformer", KW, d.X[tr], d.M[tr], d.y[tr], epochs=2, batch_size=256, lr=1e-3,
                    weight_decay=1e-4, seed=7, device=dev)
    pa = cv.predict_dl(a[0], d.X[te], d.M[te], a[1], a[2], dev)
    pb = cv.predict_dl(b[0], d.X[te], d.M[te], b[1], b[2], dev)
    assert np.abs(pa - pb).max() < 1e-4


def test_simpan_muat_pt(tmp_path, d, dev):
    idx = np.arange(0, len(d.y), 50)
    data = SimpleNamespace(X=d.X[idx], M=d.M[idx], F=d.F[idx], y=d.y[idx])
    paket = latih_dl.latih_paket_dl(lambda: build_model("MicroTransformer", **KW), data, {"epochs": 1}, 3, dev)
    p = latih_dl.simpan_paket_dl(paket, tmp_path / "r0_k0")
    assert p.name == "r0_k0.pt"
    isi = torch.load(p, weights_only=False)
    net = build_model("MicroTransformer", **KW).to(dev)
    net.load_state_dict(isi["state_dict"])
    baru = latih_dl.PaketDL(net.eval(), isi["mu"].numpy(), isi["sd"].numpy(), isi["setelan"])
    te = np.arange(1, 2000)
    assert np.allclose(latih_dl.prediksi_paket_dl(paket, d.X[te], d.M[te], None, dev),
                       latih_dl.prediksi_paket_dl(baru, d.X[te], d.M[te], None, dev), atol=1e-6)


def test_bawaan_resep_v3():
    assert latih_dl.BAWAAN == dict(epochs=30, batch_size=256, lr=1e-3, weight_decay=1e-4, weight_power=0.5)
