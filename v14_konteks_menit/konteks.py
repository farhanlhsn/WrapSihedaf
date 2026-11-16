"""
konteks.py — logika v14: tahap kedua yang membaca konteks beberapa menit (dinamika detak + keluaran v3.1 per window).

Rancangan dan hipotesis dikunci di RANCANGAN.md (6 November 2026). Fitur kausal: hanya window sekarang dan sebelumnya.
Tahap pertama di AFDB = OOF v3.1 (5 ulangan), di data eksternal = paket v3.1.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "v3_evaluasi_robust"), str(REPO / "v2_revisi"), str(REPO / "v3.2_uji_takikardia"),
                str(REPO / "v3.3_takikardia_asli"), str(REPO / "v11_model_tim")]
import sihedaf_cv as cv  # noqa: E402
import hrv_core  # noqa: E402

CACHE = REPO / "cache/v14"
OOF_DIR = REPO / "cache/oof_robust/detector_e30_r5_s2026"
N_REPEATS, SEED = 5, 2026
LANGKAH = 5.0                                     # langkah window AFDB (s)
CEPAT_TERATUR = (110.0, 0.05)
FITUR = ("logp_N", "logp_AF", "logp_AFL", "pAF_2m", "pAFL_2m", "pAF_10m", "pAFL_10m", "hr", "hr_10m", "hr_sd_2m",
         "hr_kemiringan_2m", "hr_lompat_2m", "lama_cepat_teratur", "frak_cepat_teratur_10m", "cv", "cv_2m")
XGB = dict(max_depth=4, n_estimators=300, learning_rate=0.05, tree_method="hist", objective="multi:softprob")


# ── fitur ──────────────────────────────────────────────────────────────
def fitur_rekaman(t, p, hr, cvv):
    """Satu rekaman, diurutkan waktu. t (n,) detik mulai window, p (n, 3), hr (n,), cvv (n,) → (n, len(FITUR))."""
    n = len(t)
    lp = np.log(np.clip(p, 1e-7, 1))
    a2 = np.searchsorted(t, t - 120.0, side="right")
    a10 = np.searchsorted(t, t - 600.0, side="right")
    cs = lambda v: np.r_[0.0, np.cumsum(v)]  # noqa: E731
    mean = lambda c, a: (c[np.arange(1, n + 1)] - c[a]) / (np.arange(1, n + 1) - a)  # noqa: E731
    c_af, c_afl, c_hr, c_cv = cs(p[:, 1]), cs(p[:, 2]), cs(hr), cs(cvv)
    ct = (hr >= CEPAT_TERATUR[0]) & (cvv <= CEPAT_TERATUR[1])
    c_ct = cs(ct.astype(float))
    sd2, slope2, jump2, lama = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    run_mulai = None
    for i in range(n):
        s = slice(a2[i], i + 1)
        h, tt = hr[s], t[s]
        if h.size > 1:
            sd2[i] = h.std()
            dt = tt - tt.mean()
            slope2[i] = float((dt * (h - h.mean())).sum() / max((dt * dt).sum(), 1e-9)) * 60.0
            jump2[i] = np.abs(np.diff(h)).max()
        if ct[i]:
            if run_mulai is None or (i > 0 and (not ct[i - 1] or t[i] - t[i - 1] > 2 * LANGKAH)):
                run_mulai = t[i]
            lama[i] = min((t[i] - run_mulai) / 60.0, 30.0)
        else:
            run_mulai = None
    return np.column_stack([lp, mean(c_af, a2), mean(c_afl, a2), mean(c_af, a10), mean(c_afl, a10), hr, mean(c_hr, a10),
                            sd2, slope2, jump2, lama, mean(c_ct, a10), cvv, mean(c_cv, a2)]).astype(np.float32)


def _afdb():
    z = np.load(REPO / "cache/dataset_v2_detector.npz")
    X, M = z["X"], z["M"]
    s = (X[:, :, 0] * M).sum(1) / M.sum(1)
    sd = np.sqrt(((X[:, :, 0] - s[:, None]) ** 2 * M).sum(1) / M.sum(1))
    return dict(y=z["y"], rec=z["records"].astype(str), t=z["t_start"].astype(np.float64), hr=60000.0 / s, cv=sd / s,
                folds=np.load(OOF_DIR / "folds.npz")["folds"],
                oof=np.stack([np.load(OOF_DIR / f"MicroTransformer_r{r}.npz")["oof"] for r in range(N_REPEATS)]))


def fitur_semua(rec, t, p, hr, cvv):
    """Fitur untuk banyak rekaman sekaligus (urutan baris dipertahankan)."""
    F = np.zeros((len(t), len(FITUR)), np.float32)
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        o = idx[np.argsort(t[idx], kind="stable")]
        F[o] = fitur_rekaman(t[o], p[o], hr[o], cvv[o])
    return F


# ── model ──────────────────────────────────────────────────────────────
def latih(F, y, seed, jenis="xgb"):
    w = (cv.class_weights(y) ** 0.5)[y]
    if jenis == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(**XGB, random_state=seed, n_jobs=8)
        m.fit(F, y, sample_weight=w)
        return m
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    m.fit(F, y, logisticregression__sample_weight=w)
    return m


def oof_v14(jenis="xgb", log=print):
    """OOF tahap kedua (R, N, 3) pada fold resmi; fitur ulangan r dari OOF v3.1 ulangan r. Disimpan ke cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p_out = CACHE / f"oof_{jenis}.npz"
    if p_out.exists():
        return np.load(p_out)["oof"]
    d = _afdb()
    out = np.zeros((N_REPEATS, len(d["y"]), 3), np.float32)
    for r in range(N_REPEATS):
        F = fitur_semua(d["rec"], d["t"], d["oof"][r], d["hr"], d["cv"])
        for k in range(5):
            tr, te = d["folds"][r] != k, d["folds"][r] == k
            m = latih(F[tr], d["y"][tr], SEED + 100 * r + k, jenis)
            out[r, te] = m.predict_proba(F[te])
        log(f"  {jenis} ulangan {r} selesai")
    np.savez_compressed(p_out, oof=out)
    return out


def model_penuh(jenis="xgb"):
    """Tahap kedua dilatih di seluruh AFDB (fitur dari OOF v3.1 kelima ulangan ditumpuk) + bias dari OOF v14 cross-fit
    (median bias fold). Untuk data eksternal dengan tahap pertama = paket v3.1."""
    d = _afdb()
    F = np.concatenate([fitur_semua(d["rec"], d["t"], d["oof"][r], d["hr"], d["cv"]) for r in range(N_REPEATS)])
    m = latih(F, np.tile(d["y"], N_REPEATS), SEED, jenis)
    oof = oof_v14(jenis)
    b = np.stack([cv.crossfit_bias(d["y"], oof[r], d["folds"][r])[1] for r in range(N_REPEATS)])
    return m, np.median(b.reshape(-1, 3), 0)


# ── uji stres konteks sintetis (H4) ────────────────────────────────────
def urutan_sintetis(jenis, target, sd_ms, seed, menit=15.0):
    """Waktu R-peak (detik) urutan 15 menit. jenis 'sinus': 80 bpm → target bertahap 5 menit lalu datar.
    'afl': sinus 80 bpm 5 menit, lalu AFL 2:1 mendadak (target bpm, CV 0,03)."""
    rng = np.random.default_rng(seed)
    t, out = 0.0, []
    while t < menit * 60:
        if jenis == "sinus":
            hr = 80 + (target - 80) * min(t / 300.0, 1.0)
            rr = 60000.0 / hr + rng.normal(0, sd_ms)
        else:
            rr = (60000.0 / 80 + rng.normal(0, sd_ms)) if t < 300 else 60000.0 / target * (1 + rng.normal(0, 0.03))
        t += max(rr, 250.0) / 1000.0
        out.append(t)
    return np.array(out)


def jendela_dari_rpeak(tp, langkah=LANGKAH):
    """Window 10 s langkah 5 s dari waktu R-peak: (t_mulai, ibi list, ok) — logika window sama dengan hrv_core."""
    ibi = np.diff(tp) * 1000.0
    t_end = tp[1:]
    hasil = []
    for s in np.arange(0.0, tp[-1] - hrv_core.WINDOW_SEC + 1e-9, langkah):
        i0, i1 = np.searchsorted(t_end, [s, s + hrv_core.WINDOW_SEC], side="left")
        w = ibi[i0:i1]
        ok, _ = hrv_core.window_quality(w)
        hasil.append((s, w, ok))
    return hasil


# ── tahap pertama paket v3.1 + tahap kedua pada urutan eksternal ───────
def paket_v31():
    from predictor import ECGPredictor
    return ECGPredictor(str(REPO / "v3.1_kalibrasi_produksi/model_export"), name="sihedaf_v3_1", device="cpu")


def nilai_urutan(t, ibi_list, P, m14, b14):
    """Window ok satu urutan (t mulai, list IBI) → (kelas v3.1, kelas v14, hr)."""
    seqs = [hrv_core.build_sequence(w) for w in ibi_list]
    p = P.predict_proba_sequences(np.stack([s[0] for s in seqs]), np.stack([s[1] for s in seqs]))
    hr = np.array([60000.0 / np.mean(w) for w in ibi_list])
    cvv = np.array([np.std(w) / np.mean(w) for w in ibi_list])
    F = fitur_rekaman(np.asarray(t, float), p, hr, cvv)
    k14 = cv.apply_bias(m14.predict_proba(F), b14).argmax(1)
    return p.argmax(1), k14, hr


def h3_ephnogram(m14, b14, log=print):
    """EPHNOGRAM, window langkah 5 s lewat detektor perangkat: kelas v3.1 dan v14 per window ok."""
    import takikardia_asli as K33
    P = paket_v31()
    out = {k: [] for k in ("subjek", "rekaman", "aktivitas", "t", "hr", "k31", "k14")}
    for rek in K33.rekaman_ephnogram():
        win, _ = hrv_core.signal_to_windows(np.nan_to_num(np.asarray(rek["ecg"], np.float64)), rek["fs"], LANGKAH)
        ok = [w for w in win if w["ok"]]
        if len(ok) < 2:
            continue
        k31, k14, hr = nilai_urutan([w["start_sec"] for w in ok], [w["ibi_ms"] for w in ok], P, m14, b14)
        n = len(ok)
        for k, v in (("subjek", [rek["subjek"]] * n), ("rekaman", [rek["rekaman"]] * n), ("aktivitas", [rek["aktivitas"]] * n),
                     ("t", [w["start_sec"] for w in ok]), ("hr", hr), ("k31", k31), ("k14", k14)):
            out[k].extend(list(v))
        log(f"  {rek['rekaman']}: {n} window ok, ≥120 bpm {(hr >= 120).sum()}")
    return {k: np.asarray(v) for k, v in out.items()}


def h4_sintetis(m14, b14, n_seed=10):
    """Uji stres konteks: fraksi alarm (A, bagian datar setelah 5 menit) dan deteksi AFL (B, window yang mulai ≥ 300 s)."""
    P = paket_v31()
    hasil = {}
    for jenis, targets in (("sinus", (120, 140, 160)), ("afl", (150,))):
        for target in targets:
            for sd in (10.0, 20.0):
                a31, a14 = [], []
                for s in range(n_seed):
                    tp = urutan_sintetis(jenis, target, sd, SEED + 1000 * s + target + int(sd))
                    w = [x for x in jendela_dari_rpeak(tp) if x[2]]
                    t = np.array([x[0] for x in w])
                    k31, k14, _ = nilai_urutan(t, [x[1] for x in w], P, m14, b14)
                    bag = t >= 300.0
                    if jenis == "sinus":
                        a31.append((k31[bag] != 0).mean()); a14.append((k14[bag] != 0).mean())
                    else:
                        a31.append((k31[bag] == 2).mean()); a14.append((k14[bag] == 2).mean())
                hasil[f"{jenis}_{target}_sd{int(sd)}"] = dict(v31=float(np.mean(a31)), v14=float(np.mean(a14)))
    return hasil
