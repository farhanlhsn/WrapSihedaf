"""
ekg.py — logika v6 (cabang sinyal EKG mentah 50 Hz), dipakai notebook dan laporan.

Varian (ditetapkan sebelum melihat hasil):
  ekg   : CNN 1-D pada potongan EKG 10 s @50 Hz (500 sampel) saja — apakah morfologi gelombang memuat info AFL?
  dual  : dua cabang — CNN EKG + cabang interval RR (badan MicroTransformer v2), digabung sebelum klasifikasi. Utama.
Keduanya: bobot kelas lunak (resep bawaan v5), fold/seed/window uji sama dengan referensi v3.1, bias cross-fit −7…+3.

Potongan EKG diambil dari sinyal yang sama dengan yang dipakai detektor R-peak v2 (lead 0, resample ke 50 Hz), jadi
sejajar dengan interval RR window yang sama (dibuktikan di build_ecg_windows).
"""
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "v5_training_afl"), str(REPO / "v4_smoothing_ensemble"),
                str(REPO / "v3_evaluasi_robust"), str(REPO / "v2_revisi")]
import sihedaf_cv as cv  # noqa: E402
import hrv_core  # noqa: E402

DATA_DIR = REPO / "data/afdb"
DATA_STD = REPO / "cache/dataset_v2_detector.npz"
OOF_DIR = REPO / "cache/oof_robust/detector_e30_r5_s2026"
CACHE_OUT = REPO / "cache/v6"
ECG_FILE = CACHE_OUT / "ecg_windows.npz"
FS = hrv_core.FS_DEVICE                     # 50 Hz
N_SAMP = int(hrv_core.WINDOW_SEC * FS)      # 500
BAND = (0.5, 20.0)                          # Hz; buang baseline wander & noise > 20 Hz, gelombang flutter 4–6 Hz tetap
N_REPEATS, SEED = 5, 2026
HP = dict(epochs=30, batch_size=256, lr=1e-3, weight_decay=1e-4, weight_power=0.5, flip_p=0.5)
RR_KWARGS = dict(input_dim=2, d_model=32, nhead=2, dim_ff=64, num_classes=3, dropout=0.1)
MODELS = {"ekg": "6a: CNN EKG saja", "dual": "6b: dua cabang EKG + RR"}
OOF_NAME = {"ekg": "EKG_CNN", "dual": "EKG_RR_DualBranch"}


# ── data ───────────────────────────────────────────────────────────────
def preprocess_record(sig, fs_in):
    """Sinyal mentah → 50 Hz (sama dengan hrv_core) → bandpass 0,5–20 Hz (filtfilt). Juga dipakai di produksi nanti."""
    from scipy import signal as sps
    x50 = hrv_core.resample_signal(sig, fs_in, FS)
    b, a = sps.butter(2, [BAND[0] / (FS / 2), BAND[1] / (FS / 2)], btype="band")
    return x50, sps.filtfilt(b, a, x50)


def normalize_window(w):
    """Standardisasi per window, dipotong ±8 (artefak besar tidak mendominasi)."""
    w = (w - w.mean()) / (w.std() + 1e-6)
    return np.clip(w, -8, 8)


def build_ecg_windows():
    """
    Potongan EKG 500 sampel untuk SETIAP window dataset v2 (urutan sama). Bukti sejajar: jumlah R-peak detektor v2
    di dalam [t, t+10) harus = jumlah IBI window (mask), kecuali dibatasi 30.
    """
    import wfdb
    std = np.load(DATA_STD)
    recs, t0s, M = std["records"].astype(str), std["t_start"].astype(np.float64), std["M"]
    out = np.zeros((len(recs), N_SAMP), dtype=np.float16)
    mismatch = 0
    for rec in np.unique(recs):
        idx = np.flatnonzero(recs == rec)
        path = str(DATA_DIR / rec)
        fs = wfdb.rdheader(path).fs
        sig = np.nan_to_num(wfdb.rdrecord(path, channels=[0]).p_signal[:, 0])
        x50, xf = preprocess_record(sig, fs)
        peaks_t = hrv_core.detect_rpeaks(x50, FS) / FS          # detektor v2 pada sinyal yang sama
        for i in idx:
            s = int(round(t0s[i] * FS))
            w = xf[s:s + N_SAMP]
            if w.size < N_SAMP:
                w = np.pad(w, (0, N_SAMP - w.size))
            out[i] = normalize_window(w).astype(np.float16)
            n_peaks = np.sum((peaks_t[1:] >= t0s[i]) & (peaks_t[1:] < t0s[i] + hrv_core.WINDOW_SEC))
            mismatch += int(min(n_peaks, hrv_core.MAX_BEATS) != int(M[i].sum()))
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ECG_FILE, ecg=out)
    return {"n": int(len(recs)), "mismatch": mismatch}


def class_spectra(max_per_class=4000, seed=0):
    """Rata-rata spektrum daya (Welch) per kelas dari window EKG ternormalisasi — apakah ada pita 4–6 Hz khas AFL?"""
    from scipy import signal as sps
    ecg, y = np.load(ECG_FILE)["ecg"], np.load(DATA_STD)["y"]
    rng = np.random.default_rng(seed)
    out = {}
    for c, name in enumerate(("Normal", "AFIB", "AFL")):
        sel = np.flatnonzero(y == c)
        sel = rng.choice(sel, min(max_per_class, sel.size), replace=False)
        f, p = sps.welch(ecg[sel].astype(np.float32), fs=FS, nperseg=250, axis=1)
        out[name] = np.median(p, 0)
    return f, out


# ── model ──────────────────────────────────────────────────────────────
def build(kind):
    import torch
    import torch.nn as nn
    from sihedaf_models import MicroTransformer, masked_mean

    class ResBlock(nn.Module):
        def __init__(self, cin, cout, stride, k=7):
            super().__init__()
            self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False); self.b1 = nn.BatchNorm1d(cout)
            self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False); self.b2 = nn.BatchNorm1d(cout)
            self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
            self.act = nn.SiLU()

        def forward(self, x):
            return self.act(self.b2(self.c2(self.act(self.b1(self.c1(x))))) + self.skip(x))

    class ECGEncoder(nn.Module):
        """1×500 → 256 fitur. Kernel 7 & 4 kali downsampling: medan pandang cukup untuk gelombang flutter (±10 sampel/siklus)."""
        def __init__(self, ch=(32, 64, 96, 128)):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(1, ch[0], 7, 1, 3, bias=False), nn.BatchNorm1d(ch[0]), nn.SiLU())
            self.blocks = nn.Sequential(*[ResBlock(a, b, 2) for a, b in zip((ch[0],) + ch[:-1], ch)])
            self.out_dim = 2 * ch[-1]

        def forward(self, e):
            h = self.blocks(self.stem(e.unsqueeze(1)))
            return torch.cat([h.mean(-1), h.amax(-1)], 1)

    class ECGOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = ECGEncoder(); self.fc = nn.Linear(self.enc.out_dim, 3)

        def forward(self, e, x, m):
            return self.fc(self.enc(e))

    class DualBranch(nn.Module):
        """CNN EKG + badan MicroTransformer (cabang RR, definisi v2) → gabung → klasifikasi."""
        def __init__(self):
            super().__init__()
            self.enc = ECGEncoder()
            self.rr = MicroTransformer(**RR_KWARGS)
            self.head = nn.Sequential(nn.Linear(self.enc.out_dim + RR_KWARGS["d_model"], 64), nn.SiLU(),
                                      nn.Dropout(0.2), nn.Linear(64, 3))

        def rr_features(self, x, m):
            h = self.rr.encoder(self.rr.pos_enc(self.rr.embedding(x)), src_key_padding_mask=(m == 0))
            return masked_mean(h, m)

        def forward(self, e, x, m):
            return self.head(torch.cat([self.enc(e), self.rr_features(x, m)], 1))

    return {"ekg": ECGOnly, "dual": DualBranch}[kind]()


# ── training ───────────────────────────────────────────────────────────
def train(kind, E, X, M, y, *, seed, device, epochs=None):
    """GPU-resident; bobot lunak; augmentasi: polaritas EKG dibalik acak (lead/elektroda perangkat bisa terbalik)."""
    import torch
    import torch.nn as nn
    cv.set_seed(seed)
    mu, sd = cv.zscore_fit(X, M)
    Et = torch.from_numpy(E).to(device)                                   # float16
    Xt = torch.from_numpy(cv.zscore_apply(X, M, mu, sd)).to(device)
    Mt, yt = torch.from_numpy(M).to(device), torch.from_numpy(y).to(device)
    n, bs = len(y), HP["batch_size"]
    nb = n // bs
    w = torch.tensor(cv.class_weights(y) ** HP["weight_power"], dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    net = build(kind).to(device)
    ep = epochs or HP["epochs"]
    opt = torch.optim.AdamW(net.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.perf_counter()
    net.train()
    for _ in range(ep):
        perm = torch.randperm(n, device=device, generator=gen)
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs]
            e = Et[idx].float()
            flip = torch.rand(len(idx), 1, device=device, generator=gen) < HP["flip_p"]
            e = torch.where(flip, -e, e)
            opt.zero_grad(set_to_none=True)
            crit(net(e, Xt[idx], Mt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return net.eval(), mu, sd, time.perf_counter() - t0


def predict(net, E, X, M, mu, sd, device, bs=4096):
    import torch
    out = []
    with torch.no_grad():
        Xn = cv.zscore_apply(X, M, mu, sd)
        for s in range(0, len(X), bs):
            e = torch.from_numpy(E[s:s + bs]).to(device).float()
            logit = net(e, torch.from_numpy(Xn[s:s + bs]).to(device), torch.from_numpy(M[s:s + bs]).to(device))
            out.append(torch.softmax(logit, 1).cpu().numpy())
    return np.concatenate(out)


def oof_path(kind, r):
    return OOF_DIR / f"{OOF_NAME[kind]}_r{r}.npz"


def run_job(job):
    import torch
    kind, r = job
    out = oof_path(kind, r)
    if out.exists():
        return kind, r, None
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    std, E = np.load(DATA_STD), np.load(ECG_FILE)["ecg"]
    X, M, y = std["X"], std["M"], std["y"]
    folds_r = np.load(OOF_DIR / "folds.npz")["folds"][r]
    oof = np.full((len(y), 3), np.nan, dtype=np.float32)
    times = []
    for k in range(5):
        tr, te = folds_r != k, folds_r == k
        net, mu, sd, t = train(kind, E[tr], X[tr], M[tr], y[tr], seed=SEED + 100 * r + k, device=device)
        oof[te] = predict(net, E[te], X[te], M[te], mu, sd, device)
        times.append(t)
        del net
        torch.cuda.empty_cache()
    assert not np.isnan(oof).any()
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, oof=oof, times=np.asarray(times))
    tmp.replace(out)
    return kind, r, float(np.mean(times))


def run_all_jobs(log=print):
    jobs = [(k, r) for r in range(N_REPEATS) for k in MODELS if not oof_path(k, r).exists()]
    log(f"{len(jobs)} job tersisa")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        for k, r, t in ex.map(run_job, jobs):
            log(f"  [{(time.perf_counter() - t0) / 60:5.1f} mnt] {k:<5} ulangan {r}: {t:.0f} s/fold")


# ── evaluasi (metrik v4 + format v5) ───────────────────────────────────
def evaluate():
    import pascaproses as P4
    d = P4.load()
    sources = {"v31": [d.oof["MicroTransformer"][r] for r in range(N_REPEATS)],
               **{k: [np.load(oof_path(k, r))["oof"] for r in range(N_REPEATS)] for k in MODELS}}
    # eksploratif: late fusion = rata-rata probabilitas MicroTransformer v3 dan CNN EKG
    sources["late"] = [(sources["v31"][r] + sources["ekg"][r]) / 2 for r in range(N_REPEATS)]
    pred, biases = {"v1": d.oof_v1.argmax(-1)}, {}
    for k, oofs in sources.items():
        adj = [cv.crossfit_bias(d.y, oofs[r], d.folds[r]) for r in range(N_REPEATS)]
        pred[k] = np.stack([a.argmax(1) for a, _ in adj])
        biases[k] = np.concatenate([b for _, b in adj])
    ev = P4.evaluate(d, pred)
    fp = {}
    for k, v in pred.items():
        cms = np.stack([cv.patient_cms(d.y, p, d.records, d.patients) for p in v])
        f = cms[:, :, :2, 2].sum(-1).mean(0)
        fp[k] = {"total": float(f.sum()), "pasien_lain": float(f[d.strata != 2].sum())}
    return {
        "labels": {"v1": "v1 (sebelum revisi), argmax", "v31": "v3.1 referensi (RR saja)", **MODELS,
                   "late": "Late fusion v3.1 + CNN EKG (eksploratif)"},
        "rows": ev.rows, "episodes": ev.epi, "pairs": ev.pairs,
        "afl_patients": ev.afl_patients, "afl_recall": ev.afl_recall, "fp": fp,
        "bias": {k: {"median": np.median(b, 0).tolist(), "min": b.min(0).tolist(), "max": b.max(0).tolist()}
                 for k, b in biases.items()},
        "train_time": {k: float(np.mean([np.load(oof_path(k, r))["times"].mean() for r in range(N_REPEATS)])) for k in MODELS},
    }
