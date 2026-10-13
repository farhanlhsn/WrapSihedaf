"""
latih.py — logika v5 (dua varian training MicroTransformer), dipakai notebook dan laporan.

  aflrapat   : data latih memakai window AFL yang digeser tiap 1 s (bukan 5 s). Data uji TETAP window standar 5 s.
  bobotlunak : bobot kelas di loss = akar dari bobot sekarang (N / (K·N_c))^0,5. Data sama dengan v3.

Fold, seed latih (2026 + 100·r + k), hiperparameter, dan data uji sama persis dengan referensi v3/v3.1, jadi selisihnya
hanya berasal dari resep training. Evaluasi memakai kode metrik v4 (window + episode) dan bias cross-fit grid −7…+3.
"""
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "v4_smoothing_ensemble"), str(REPO / "v3_evaluasi_robust"), str(REPO / "v2_revisi")]
import sihedaf_cv as cv  # noqa: E402
import hrv_core  # noqa: E402

DATA_DIR = REPO / "data/afdb"
DATA_STD = REPO / "cache/dataset_v2_detector.npz"
OOF_DIR = REPO / "cache/oof_robust/detector_e30_r5_s2026"
CACHE_OUT = REPO / "cache/v5"
DATA_DENSE = CACHE_OUT / "dataset_detector_afl1s.npz"
N_REPEATS, SEED = 5, 2026
HP = dict(epochs=30, batch_size=256, lr=1e-3, weight_decay=1e-4)
MODEL_KWARGS = dict(input_dim=2, d_model=32, nhead=2, dim_ff=64, num_classes=3, dropout=0.1)
VARIANTS = {"aflrapat": "5a: window AFL rapat (1 s) saat training",
            "bobotlunak": "5b: bobot kelas lunak (akar)"}

# Konfigurasi ekstraksi v2 (revisi_sihedaf_v2.ipynb, sel Config & AFDBExtractorV2, sumber "detector")
BLACKLIST = ("00735", "03665")
ECG_CHANNEL = 0
INVALID_STEP_SEC = 1.0
LABEL_MAP = {"(N": 0, "(AFIB": 1, "(AFL": 2}


# ── data ───────────────────────────────────────────────────────────────
def _process_record(rec, stride):
    """
    Salinan logika AFDBExtractorV2.process_record (sumber "detector") dari notebook v2 — kelas itu hanya ada di
    dalam sel notebook sehingga tidak bisa diimpor. build_dataset() membuktikan salinan ini identik dengan
    ../cache/dataset_v2_detector.npz bila stride standar.
    """
    import wfdb
    path = str(DATA_DIR / rec)
    fs = wfdb.rdheader(path).fs
    atr = wfdb.rdann(path, "atr")
    sig = np.nan_to_num(wfdb.rdrecord(path, channels=[ECG_CHANNEL]).p_signal[:, 0])
    x50 = hrv_core.resample_signal(sig, fs, hrv_core.FS_DEVICE)
    peaks, fs_pk = hrv_core.detect_rpeaks(x50, hrv_core.FS_DEVICE), float(hrv_core.FS_DEVICE)
    if peaks is None or len(peaks) < hrv_core.MIN_IBI + 1:
        return []
    rhythm = [(s, a.strip("\x00").strip()) for s, a in zip(atr.sample, atr.aux_note)
              if a.strip("\x00").strip().startswith("(")]
    r_samp = np.array([s for s, _ in rhythm])
    r_lab = np.array([LABEL_MAP.get(a, -1) for _, a in rhythm])
    t_end, ibi = hrv_core.peaks_to_ibi(peaks, fs_pk)
    close = peaks[1:] * (fs / fs_pk)
    idx = np.searchsorted(r_samp, close, side="right") - 1
    lab = np.where(idx >= 0, r_lab[np.clip(idx, 0, None)], -1)
    rows, t, W = [], 0.0, hrv_core.WINDOW_SEC
    while t + W <= t_end[-1]:
        i0, i1 = np.searchsorted(t_end, [t, t + W], side="left")
        wl = lab[i0:i1]
        ok, _ = hrv_core.window_quality(ibi[i0:i1])
        if ok and wl.size and wl[0] >= 0 and np.all(wl == wl[0]):
            c = int(wl[0])
            rows.append((rec, t, ibi[i0:i1].copy(), c))
            t += stride[c]
        else:
            t += INVALID_STEP_SEC
    return rows


def build_dataset(stride):
    recs = [l.strip() for l in open(DATA_DIR / "RECORDS") if l.strip() and l.strip() not in BLACKLIST]
    rows = [w for rec in recs for w in _process_record(rec, stride)]
    seqs = [hrv_core.build_sequence(w[2]) for w in rows]
    return {
        "X": np.stack([s[0] for s in seqs]).astype(np.float32),
        "M": np.stack([s[1] for s in seqs]).astype(np.float32),
        "y": np.array([w[3] for w in rows], dtype=np.int64),
        "records": np.array([w[0] for w in rows]),
        "t_start": np.array([w[1] for w in rows], dtype=np.float32),
    }


def verify_and_build_dense():
    """1) stride standar → harus identik dengan cache v2; 2) simpan versi AFL 1 s."""
    std = build_dataset({0: 5.0, 1: 5.0, 2: 5.0})
    ref = np.load(DATA_STD)
    for k in ("X", "M", "y", "t_start"):
        assert np.array_equal(std[k], ref[k]), f"salinan ekstraktor berbeda dari v2 pada '{k}'"
    assert np.array_equal(std["records"].astype(str), ref["records"].astype(str))
    dense = build_dataset({0: 5.0, 1: 5.0, 2: 1.0})
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_DENSE, **dense)
    return {"std": np.bincount(std["y"], minlength=3).tolist(), "dense": np.bincount(dense["y"], minlength=3).tolist()}


# ── training ───────────────────────────────────────────────────────────
def train_dl_weighted(X, M, y, *, weight_power, seed, device):
    """
    Sama dengan sihedaf_cv.train_dl (v3), kecuali bobot loss = class_weights(y) ** weight_power.
    weight_power = 1 → identik dengan resep v2/v3.
    """
    import torch
    import torch.nn as nn
    from sihedaf_models import build_model
    cv.set_seed(seed)
    mu, sd = cv.zscore_fit(X, M)
    Xt = torch.from_numpy(cv.zscore_apply(X, M, mu, sd)).to(device)
    Mt, yt = torch.from_numpy(M).to(device), torch.from_numpy(y).to(device)
    n, bs = len(y), HP["batch_size"]
    n_batches = n // bs if n > bs else 1
    bs = bs if n > bs else n
    w = torch.tensor(cv.class_weights(y) ** weight_power, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    net = build_model("MicroTransformer", **MODEL_KWARGS).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["epochs"])
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.perf_counter()
    net.train()
    for _ in range(HP["epochs"]):
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


def oof_path(variant, r):
    return OOF_DIR / f"MicroTransformer_{variant}_r{r}.npz"


def run_job(job):
    """Satu (varian, ulangan): latih 5 fold, simpan OOF pada window uji standar. Dilewati jika sudah ada."""
    import torch
    variant, r = job
    out = oof_path(variant, r)
    if out.exists():
        return variant, r, None
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = np.load(DATA_STD)
    train = np.load(DATA_DENSE) if variant == "aflrapat" else test
    rec_test, rec_train = test["records"].astype(str), train["records"].astype(str)
    folds_r = np.load(OOF_DIR / "folds.npz")["folds"][r]
    rec2fold = {p: int(folds_r[rec_test == p][0]) for p in np.unique(rec_test)}
    f_train = np.array([rec2fold[x] for x in rec_train])
    power = 0.5 if variant == "bobotlunak" else 1.0
    oof = np.full((len(test["y"]), 3), np.nan, dtype=np.float32)
    times = []
    for k in range(5):
        tr, te = f_train != k, folds_r == k
        net, mu, sd, t = train_dl_weighted(train["X"][tr], train["M"][tr], train["y"][tr],
                                           weight_power=power, seed=SEED + 100 * r + k, device=device)
        oof[te] = cv.predict_dl(net, test["X"][te], test["M"][te], mu, sd, device)
        times.append(t)
    assert not np.isnan(oof).any()
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, oof=oof, times=np.asarray(times))
    tmp.replace(out)
    return variant, r, float(np.mean(times))


def run_all_jobs(log=print):
    """10 job (2 varian × 5 ulangan) berurutan di satu proses GPU; urutan per ulangan supaya hasil awal cepat ada."""
    jobs = [(v, r) for r in range(N_REPEATS) for v in VARIANTS if not oof_path(v, r).exists()]
    log(f"{len(jobs)} job tersisa")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        for v, r, t in ex.map(run_job, jobs):
            log(f"  [{(time.perf_counter() - t0) / 60:5.1f} mnt] {v:<10} ulangan {r}: {t:.0f} s/fold")


# ── evaluasi (metrik v4: window + episode, bias cross-fit grid −7…+3) ──
def evaluate():
    import pascaproses as P4
    d = P4.load()
    pred, biases, fold_bias = {"v1": d.oof_v1.argmax(-1)}, {}, {}
    sources = {"v31": [d.oof["MicroTransformer"][r] for r in range(N_REPEATS)],
               **{v: [np.load(oof_path(v, r))["oof"] for r in range(N_REPEATS)] for v in VARIANTS}}
    for k, oofs in sources.items():
        adj = [cv.crossfit_bias(d.y, oofs[r], d.folds[r]) for r in range(N_REPEATS)]
        pred[k] = np.stack([a.argmax(1) for a, _ in adj])
        biases[k] = np.concatenate([b for _, b in adj])
        fold_bias[k] = [b for _, b in adj]
    ev = P4.evaluate(d, pred)

    # Tanpa bias sama sekali (argmax) — seberapa bergantung model pada kalibrasi bias
    W = cv.bootstrap_weights(d.strata, P4.N_BOOT, seed=SEED)
    cms_arg = {k: np.stack([cv.patient_cms(d.y, o.argmax(1), d.records, d.patients) for o in oofs])
               for k, oofs in sources.items()}
    boot_arg = {k: cv.boot_metrics(c, W) for k, c in cms_arg.items()}
    pt_arg = {k: cv.point_metrics(c) for k, c in cms_arg.items()}
    argmax_rows = {k: {m: [float(pt_arg[k][m].mean()), *map(float, np.percentile(boot_arg[k][m], [2.5, 97.5]))]
                       for m in ("Macro-F1", "AFL F1", "AFL P", "AFL R", "Normal R")} for k in sources}
    argmax_pairs = {}
    for k in VARIANTS:
        dd = boot_arg[k]["Macro-F1"] - boot_arg["v31"]["Macro-F1"]
        argmax_pairs[k] = [float(pt_arg[k]["Macro-F1"].mean() - pt_arg["v31"]["Macro-F1"].mean()),
                           *map(float, np.percentile(dd, [2.5, 97.5])), float(np.mean(dd > 0))]

    # Kepekaan: bias AFL hasil cross-fit digeser δ (simulasi bias meleset, mis. prevalensi perangkat berbeda)
    deltas = np.arange(-2.0, 2.01, 0.5)
    sens = {}
    for k, oofs in sources.items():
        mf1 = np.zeros((N_REPEATS, len(deltas)))
        for r in range(N_REPEATS):
            logp = np.log(np.clip(oofs[r], 1e-9, 1))
            for j, dl in enumerate(deltas):
                pr = np.empty(len(d.y), dtype=np.int64)
                for kk, b in zip(np.unique(d.folds[r]), fold_bias[k][r]):
                    te = d.folds[r] == kk
                    pr[te] = (logp[te] + b + np.array([0.0, 0.0, dl])).argmax(1)
                mf1[r, j] = cv._macro_f1_fast(d.y, pr)
        sens[k] = mf1.mean(0).tolist()
    train_time = {v: float(np.mean([np.load(oof_path(v, r))["times"].mean() for r in range(N_REPEATS)])) for v in VARIANTS}
    fp = {}
    for k, v in pred.items():
        cms = np.stack([cv.patient_cms(d.y, p, d.records, d.patients) for p in v])
        f = cms[:, :, :2, 2].sum(-1).mean(0)
        fp[k] = {"total": float(f.sum()), "pasien_lain": float(f[d.strata != 2].sum())}
    dense = np.load(DATA_DENSE)
    return {
        "labels": {"v1": "v1 (sebelum revisi), argmax", "v31": "v3.1 referensi (resep v2/v3)", **VARIANTS},
        "rows": ev.rows, "episodes": ev.epi, "pairs": ev.pairs,
        "afl_patients": ev.afl_patients, "afl_recall": ev.afl_recall, "fp": fp,
        "bias": {k: {"median": np.median(b, 0).tolist(), "min": b.min(0).tolist(), "max": b.max(0).tolist()}
                 for k, b in biases.items()},
        "class_weights_full": {"standar": cv.class_weights(d.y).tolist(),
                               "lunak": (cv.class_weights(d.y) ** 0.5).tolist(),
                               "aflrapat": cv.class_weights(dense["y"]).tolist()},
        "counts": {"standar": np.bincount(d.y, minlength=3).tolist(), "aflrapat": np.bincount(dense["y"], minlength=3).tolist()},
        "train_time": train_time,
        "argmax_rows": argmax_rows, "argmax_pairs": argmax_pairs,
        "bias_sensitivity": {"delta": deltas.tolist(), "macro_f1": sens},
    }
