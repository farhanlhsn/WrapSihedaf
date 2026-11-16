"""
deteksi_af.py — logika v16.1 (pra-registrasi di RANCANGAN.md, dikunci 6 November 2026 sebelum metrik baru dihitung).

Detektor AF 2 kelas (Normal vs AF = AFIB atau AFL) dari tahap 1 v16: ECGFounder zero-shot tanpa penyetelan, AF bila
d1 = log((pAF + pAFL) / pN) > 0. Tidak ada training dan tidak ada parameter yang disetel.
  D-0 — keputusan itu (primer);
  D-L — D-0 ditambah window yang v3.1 sebut AFL ("perlu dicek" v16 dihitung sebagai alarm).
Pembanding 2 kelas memakai aturan v11.1: bias 3 kelas cross-fit, lalu AFIB + AFL digabung. Semua dari cache v3.3, v10, v11, v13, v15,
v16, kecuali paritas dan latensi fungsi prediksi (inferensi ECGFounder ulang pada ±600 window AFDB).
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "v16_hierarki_ecgfounder"))
import hierarki as H16  # noqa: E402
from hierarki import D, DK, E13, Ev, K33, T, cv  # noqa: E402,F401

CACHE = REPO / "cache/v16.1"
AMBANG_KURVA = np.arange(-4, 5)
REKAMAN_PARITAS = "08378"


def d1_afdb():
    """d1 tahap 1 untuk semua window AFDB (logit ECGFounder jalur 50 Hz dari cache v13)."""
    return H16.tahap1(E13.muat_afdb("50hz")[0])[1]


def ke2(pred3):
    """Keputusan 3 kelas → 2 kelas (AFIB dan AFL digabung)."""
    return (np.asarray(pred3) > 0).astype(np.int64)


def pembanding(d):
    """Keputusan 3 kelas (R, N) v1 (argmax), v3.1 dan BiGRU bawaan v11.1 (bias 3 kelas cross-fit), serta OOF keduanya."""
    from bersama import runner
    bg = runner.muat_hasil(D.lokasi_hasil() / "bigru/bawaan", d)
    if bg.ulangan != list(range(D.N_ULANGAN)):
        raise RuntimeError(f"BiGRU bawaan belum lengkap 5 ulangan: {bg.ulangan}")
    oof = {"v3.1": d.oof_v31, "BiGRU bawaan (v11.1)": bg.oof}
    p3 = {"v1 (argmax)": d.oof_v1.argmax(-1)}
    for k, o in oof.items():
        p3[k] = Ev.keputusan(o, d.folds, d.y)[0]
    return p3, oof


def sistem(d1, p3_v31):
    """D-0 dan D-L (R, N) dari d1 (N,) dan keputusan 3 kelas v3.1 (R, N)."""
    a = (d1 > 0)[None]
    return {"D-0": np.repeat(a, len(p3_v31), 0).astype(np.int64), "D-L": (a | (p3_v31 == 2)).astype(np.int64)}


def tanpa_04936(d, d1, oof):
    """Subset tanpa 04936: bias 3 kelas v3.1 dan BiGRU dipilih ulang di subset; D-L memakai v3.1 subset itu. Return (metrik, boot)."""
    m = d.records != T.SUSPECT
    pas = np.unique(d.records[m])
    W = cv.bootstrap_weights(cv.patient_strata(d.y[m], d.records[m], pas), Ev.N_BOOT, seed=D.SEED)
    y2 = (d.y[m] > 0).astype(np.int64)
    p3 = {k: Ev.keputusan(o[:, m], d.folds[:, m], d.y[m])[0] for k, o in oof.items()}
    pred = {"v1 (argmax)": ke2(d.oof_v1.argmax(-1)[:, m]), **{k: ke2(v) for k, v in p3.items()}, **sistem(d1[m], p3["v3.1"])}
    hasil, boot = {}, {}
    for k, p in pred.items():
        hasil[k], boot[k] = DK.ringkas(DK.cm_pasien(y2, p, d.records[m], pas), W)
    return hasil, boot


def per_pasien(d, pred2):
    """Sensitivitas (window AF) dan alarm (window Normal) per pasien, rata-rata ulangan; None bila pasien tidak punya kelas itu."""
    c = DK.cm_pasien((d.y > 0).astype(np.int64), pred2, d.records, d.patients).sum(0)
    R = len(pred2)
    out = {}
    for i, p in enumerate(d.patients):
        n0, n1 = c[i, 0].sum() / R, c[i, 1].sum() / R
        out[str(p)] = dict(n_normal=int(n0), n_af=int(n1), se=float(c[i, 1, 1] / c[i, 1].sum()) if n1 else None,
                           alarm=float(c[i, 0, 1] / c[i, 0].sum()) if n0 else None)
    return out


def alarm_normal(d, pred2):
    """Alarm AF pada window Normal AFDB per pita detak (pita v11.1) dan pada Normal teratur 80–95 bpm (CV IBI < 0,03)."""
    ibi, msk = d.X[..., 0], d.M.astype(bool)
    cv_ibi = np.array([ibi[i][msk[i]].std() / ibi[i][msk[i]].mean() for i in range(len(ibi))])
    teratur = (d.y == 0) & (d.hr >= 80) & (d.hr < 95) & (cv_ibi < 0.03)
    out = {k: dict(alarm=float((pred2[:, (d.y == 0) & (d.hr >= lo) & (d.hr < hi)] == 1).mean()),
                   n=int(((d.y == 0) & (d.hr >= lo) & (d.hr < hi)).sum())) for k, lo, hi in DK.PITA}
    out["teratur 80–95"] = dict(alarm=float((pred2[:, teratur] == 1).mean()), n=int(teratur.sum()))
    return out


def episode_af(d, pred2):
    """Metrik episode gaya EC57 (v3.3) untuk AF gabungan, rata-rata ulangan."""
    y2 = (d.y > 0).astype(np.int64)
    ms = [K33.metrik_episode(d.records, d.t, y2, p, 1) for p in pred2]
    return {m: float(np.mean([x[m] for x in ms])) for m in ms[0]}


def banding2_pasien(y2, a, b, pasien, n_boot=Ev.N_BOOT, seed=D.SEED):
    """2 kelas, bootstrap pasien berpasangan (pola ecgfounder.banding_pasien): {metrik: [Δ a − b, CI bawah, CI atas, P(Δ > 0)]},
    plus titik dan CI 95% masing-masing."""
    u, inv = np.unique(pasien, return_inverse=True)

    def cm(p):
        return np.bincount((inv * 2 + y2) * 2 + np.asarray(p, np.int64), minlength=len(u) * 4).reshape(len(u), 2, 2).astype(float)

    ca, cb = cm(a), cm(b)
    idx = np.random.default_rng(seed).integers(0, len(u), (n_boot, len(u)))
    ma, mb = DK.metrik(ca[idx].sum(1)), DK.metrik(cb[idx].sum(1))
    ta, tb = DK.metrik(ca.sum(0)), DK.metrik(cb.sum(0))
    out = {"titik_a": {}, "titik_b": {}}
    for m in ("Macro-F1", "Se", "Sp", "PPV"):
        dd = ma[m] - mb[m]
        out[m] = [float(ta[m] - tb[m]), *np.percentile(dd, [2.5, 97.5]).tolist(), float(np.mean(dd > 0))]
        out["titik_a"][m] = [float(ta[m]), *np.percentile(ma[m], [2.5, 97.5]).tolist()]
        out["titik_b"][m] = [float(tb[m]), *np.percentile(mb[m], [2.5, 97.5]).tolist()]
    return out


def kurva(d, d1, d1_lat120, d1_ptb, y_ptb, st_ptb):
    """Kurva operasi ambang d1 > t (deskriptif, tidak ada ambang yang dipilih)."""
    y2 = d.y > 0
    out = []
    for t in AMBANG_KURVA:
        a = d1 > t
        cm = np.bincount(y2.astype(np.int64) * 2 + a, minlength=4).reshape(2, 2).astype(float)
        ap = d1_ptb > t
        out.append(dict(t=int(t), se_afdb=float(a[y2].mean()), sp_afdb=float((~a[~y2]).mean()), mf1_afdb=float(DK.f1_makro_cm(cm)),
                        alarm_st=float((d1_lat120 > t).mean()), alarm_stach=float(ap[st_ptb].mean()),
                        se_ptb=float(ap[y_ptb > 0].mean()), sp_ptb=float((~ap[y_ptb == 0]).mean())))
    return out


# ── fungsi prediksi untuk uji (butir 9), paritas, latensi ──────────────
class DetektorAF:
    """Fungsi prediksi D-0: EKG 10 s pada 50 Hz, array (N, 500) → (d1 (N,), AF (N,) bool). Praproses dan model sama dengan v13
    (`ecgfounder.praproses` → Net1D 1-lead, checkpoint dicek SHA256, dimuat weights_only). CPU float32; GPU bf16 seperti cache v13."""

    def __init__(self, device="cpu"):
        self.device = device
        self.model = E13.muat_model(device)

    def __call__(self, ekg50, batch=64):
        x = E13.praproses(np.atleast_2d(np.asarray(ekg50, np.float64)), E13.hrv_core.FS_DEVICE)
        lo, _ = E13.inferensi(self.model, x, self.device, batch)
        d1 = H16.tahap1(lo)[1]
        return d1, d1 > 0


def jendela_paritas(rec=REKAMAN_PARITAS, n_lain=300):
    """Semua window AFL rekaman + n_lain window non-AFL berjarak tetap. Return (indeks AFDB, EKG 50 Hz (n, 500))."""
    recs, t0s, y = E13._afdb()
    idx = np.flatnonzero(recs == rec)
    lain = idx[y[idx] != 2]
    pilih = np.sort(np.r_[idx[y[idx] == 2], lain[np.linspace(0, len(lain) - 1, n_lain).astype(int)]])
    w, _ = E13.jendela_afdb(rec, t0s[pilih], "50hz")
    return pilih, w


def paritas(d1_cache, pilih, w, device="cpu"):
    """DetektorAF (dihitung ulang) vs d1 dari cache v13 (GPU bf16) pada window yang sama."""
    d1_baru, _ = DetektorAF(device)(w)
    sel = np.abs(d1_baru - d1_cache[pilih])
    return dict(n=int(len(pilih)), sama=float(np.mean((d1_baru > 0) == (d1_cache[pilih] > 0))), maks_selisih=float(sel.max()),
                median_selisih=float(np.median(sel)), device=device)


def latensi(w, device="cpu", batch=1, n=50):
    """Waktu per window (ms) untuk praproses + inferensi, median dari n panggilan dengan `batch` window (setelah 3 pemanasan)."""
    import torch
    det = DetektorAF(device)
    x = np.resize(w, (batch, w.shape[1]))
    for _ in range(3):
        det(x, batch)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        det(x, batch)
        t.append((time.perf_counter() - t0) * 1000 / batch)
    return dict(device=device, batch=batch, ms_per_window=float(np.median(t)), p90=float(np.percentile(t, 90)),
                threads=int(torch.get_num_threads()), gpu=torch.cuda.get_device_name(0) if device.startswith("cuda") else None)
