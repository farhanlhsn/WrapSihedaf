"""
fusi.py — logika v17 (pra-registrasi di RANCANGAN.md, dikunci 6 November 2026 sebelum model dilatih).

W-F: regresi logistik kecil pada [d1 (ECGFounder zero-shot), r (log-odds AF paket v3.1)], dilatih di CinC 2017 (EKG genggam, CC BY 3.0),
diuji di Icentia11k (wearable) dan dijaga di Holter/klinis (CPSC 2021, MIT-BIH, LTAFDB) serta sinus takikardia asli. Pembanding: D-1
(d1 > −1), paket v3.1, dan aturan naif D-1 ∨ v3.1. Semua masukan dibaca dari cache v13–v16.3; tidak ada inferensi ECGFounder baru.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "v16.3_replikasi"))
import replikasi as A163  # noqa: E402
from replikasi import A161, D, DK, Ev, H16, K15, K33, U162  # noqa: E402,F401

CACHE = REPO / "cache/v17"
SISTEM = ("W-F", "D-1", "v3.1 paket", "D-1 ∨ v3.1")


# ── model ──────────────────────────────────────────────────────────────
def log_odds_af(p3):
    """r = log((pAFIB + pAFL) / pN) dari probabilitas paket v3.1 (clip sama dengan v15.masukan)."""
    p3 = np.asarray(p3, np.float64)
    return np.log(np.clip(p3[..., 1] + p3[..., 2], 1e-7, 1)) - np.log(np.clip(p3[..., 0], 1e-7, 1))


def masukan(d1, p31):
    return np.column_stack([np.asarray(d1, np.float64), log_odds_af(p31)])


def latih(X, y):
    """StandardScaler + LogisticRegression(C=1, class_weight="balanced") — resep v15.latih_gabungan, tidak disalin."""
    return K15.latih_gabungan(X, np.asarray(y, np.int64))


def keputusan(model, d1, p31):
    return model.predict_proba(masukan(d1, p31))[:, 1] >= 0.5


def koefisien(model):
    """Batas keputusan dalam satuan asli: AF bila w_d1·d1 + w_r·r + b ≥ 0."""
    sk, lr = model.steps[0][1], model.steps[1][1]
    w = lr.coef_[0] / sk.scale_
    b = float(lr.intercept_[0] - np.sum(lr.coef_[0] * sk.mean_ / sk.scale_))
    return {"w_d1": float(w[0]), "w_r": float(w[1]), "b": b}


def prediksi(model, d1, p31):
    """Keputusan AF (bool) keempat sistem."""
    v31 = np.asarray(p31).argmax(1) != 0
    d = np.asarray(d1) > A163.AMBANG
    return {"W-F": keputusan(model, d1, p31), "D-1": d, "v3.1 paket": v31, "D-1 ∨ v3.1": d | v31}


def oof_cinc(Z, n_fold=5):
    """Validasi silang CinC 2017 per rekaman (GroupKFold, deterministik): keputusan W-F out-of-fold."""
    from sklearn.model_selection import GroupKFold
    X, y, g = masukan(Z["d1"], Z["p31"]), Z["y2"], Z["rekaman"]
    out = np.zeros(len(y), bool)
    for tr, te in GroupKFold(n_fold).split(X, y, g):
        out[te] = latih(X[tr], y[tr]).predict_proba(X[te])[:, 1] >= 0.5
    return out


# ── data (semua dari cache) ────────────────────────────────────────────
def data_cinc():
    Z, _ = A163.muat("CinC 2017")
    return Z


def data_icentia():
    I = H16.icentia()
    fast = np.load(REPO / "cache/v10/icentia_windows.npz")["fast"]
    return dict(d1=H16.tahap1(I["logit"])[1], p31=I["p31"], y2=(I["y"] > 0).astype(np.int64), unit=I["pasien"], fast=fast)


def data_cpsc():
    """CPSC 2021 label resmi (v16.2): window lolos cek, logit dan paket dari cache."""
    J = U162.bangun_jendela(resmi=True, log=lambda *_: None)
    ok = J["ok"]
    lo = np.load(U162.CACHE / "logit_ecgfounder_resmi.npy")
    return dict(d1=H16.tahap1(lo)[1], p31=np.load(U162.CACHE / "paket_resmi.npz")["p31"], y2=(J["y"][ok] > 0).astype(np.int64),
                unit=J["subjek"][ok], hr=J["hr"][ok])


def data_v163(ds):
    Z, _ = A163.muat(ds)
    return dict(d1=Z["d1"], p31=Z["p31"], y2=Z["y2"], unit=Z["subjek"], ritme=Z["ritme"], hr=Z["hr"])


def data_ptbxl():
    P = H16.ptbxl()
    return dict(d1=H16.tahap1(P["logit"])[1], p31=P["p31"], y2=(P["y"] > 0).astype(np.int64), unit=P["pasien"], st=P["st"], y=P["y"])


# ── evaluasi ───────────────────────────────────────────────────────────
def banding(Z, pred):
    """W-F vs D-1, W-F vs v3.1, D-1 ∨ v3.1 vs D-1 (bootstrap unit berpasangan) + titik keempat sistem."""
    y, u = Z["y2"], Z["unit"]
    b = {"W-F vs D-1": A161.banding2_pasien(y, pred["W-F"], pred["D-1"], u),
         "W-F vs v3.1": A161.banding2_pasien(y, pred["W-F"], pred["v3.1 paket"], u),
         "D-1 ∨ v3.1 vs D-1": A161.banding2_pasien(y, pred["D-1 ∨ v3.1"], pred["D-1"], u)}
    titik = {"W-F": b["W-F vs D-1"]["titik_a"], "D-1": b["W-F vs D-1"]["titik_b"], "v3.1 paket": b["W-F vs v3.1"]["titik_b"],
             "D-1 ∨ v3.1": b["D-1 ∨ v3.1 vs D-1"]["titik_a"]}
    return b, titik


def afdb(model):
    """AFDB (deskriptif): 5 ulangan OOF v3.1 + bias paket (v15._afdb_logp); D-1 tetap; metrik 2 kelas pola v11.1."""
    d = D.muat()
    d1 = A161.d1_afdb()
    y2 = (d.y > 0).astype(np.int64)
    pred = {k: [] for k in SISTEM}
    for r in range(len(d.folds)):
        p31 = np.exp(K15._afdb_logp(r))
        for k, v in prediksi(model, d1, p31).items():
            pred[k].append(v.astype(np.int64))
    W = A161.cv.bootstrap_weights(d.strata, Ev.N_BOOT, seed=D.SEED)
    hasil, boot = {}, {}
    for k, v in pred.items():
        hasil[k], boot[k] = DK.ringkas(DK.cm_pasien(y2, np.stack(v), d.records, d.patients), W)
    return hasil, {"W-F vs D-1": DK.banding(hasil["W-F"], hasil["D-1"], boot["W-F"], boot["D-1"]),
                   "W-F vs v3.1": DK.banding(hasil["W-F"], hasil["v3.1 paket"], boot["W-F"], boot["v3.1 paket"])}


def latihan(model):
    """Sinus takikardia asli (latihan fisik v3.3, mode perangkat): alarm ≥ 120 bpm dengan CI subjek, dan per pita detak."""
    L = H16.latihan()
    J = L["J"]
    _, d1 = H16.tahap1(L["logit"])
    hr_m, subj_m = (J[k][J["ok"]][L["mask"]] for k in ("hr", "subjek"))
    m120 = hr_m >= 120
    out = {}
    for k, v in prediksi(model, d1, L["p31"]).items():
        kel = H16.kelas_latihan(L, v.astype(np.int64))
        a = (kel[:, L["mask"]] != 0).mean(0)
        t, ci, _ = K33._boot_subjek(subj_m[m120], a[m120])
        out[k] = dict(alarm=t, ci=ci.tolist(), n=int(m120.sum()), pita=K33.alarm_per_pita(J, kel, L["mask"]))
    return out


def per_ritme(Z, pred):
    return {str(r): dict(n=int((Z["ritme"] == r).sum()), **{k: float(p[Z["ritme"] == r].mean()) for k, p in pred.items()})
            for r in np.unique(Z["ritme"])}


def simpan_model(model):
    CACHE.mkdir(parents=True, exist_ok=True)
    k = koefisien(model)
    (CACHE / "wf.json").write_text(json.dumps({**k, "masukan": ["d1 ECGFounder zero-shot", "log((pAFIB+pAFL)/pN) paket v3.1"],
                                               "ambang_p": 0.5, "data_latih": "CinC 2017 (CC BY 3.0)"}, indent=1))
    return k
