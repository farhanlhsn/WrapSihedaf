"""Ukur ulang VERSI SEBELUM REVISI dengan protokol evaluasi robust.

Versi lama = ekstraksi_hrv.py (R-peak anotasi 250 Hz, stride N 5 s / AFIB 10 s /
AFL 1 s) + resep training model.ipynb (MicroTransformer, weighted CE, argmax)
+ front-end backend lama di dokumentasi.md (find_peaks polos, tanpa cek kualitas).

Resep lama dilatih ulang pada fold pasien yang SAMA dengan evaluasi_robust_v2.ipynb
(cache/oof_robust/detector_e30_r5_s2026/folds.npz), lalu tiap model fold diuji pada
tiga set uji pasien yang tidak ikut dilatih:
  lab  : window ekstraksi lama (yang dulu dipakai model.ipynb untuk melapor skor)
  det  : window detektor 50 Hz revisi (set uji yang sama dengan model produksi)
  fe   : window front-end lama pada sinyal 50 Hz (sistem lama end-to-end)

Jalankan di WSL dari folder v3_evaluasi_robust/:  python eval_versi_lama.py
Hasil di ../cache/oof_robust/detector_e30_r5_s2026/versi_lama/ (bisa dilanjutkan).
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wfdb
from scipy.signal import find_peaks, resample

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT / "v2_revisi", ROOT / "v1_lama"):   # sihedaf_cv, hrv_core, ekstraksi_hrv
    sys.path.insert(0, str(p))
import hrv_core  # noqa: E402
import sihedaf_cv as cv  # noqa: E402
from ekstraksi_hrv import AFDBExtractor  # noqa: E402

OOF_DIR = ROOT / "cache/oof_robust/detector_e30_r5_s2026"
OUT = OOF_DIR / "versi_lama"
SEED, N_REPEATS = 2026, 5
HP = dict(epochs=30, batch_size=256, lr=1e-3, weight_decay=1e-4)
# model.ipynb: MicroTransformer(input_dim=2, d_model=32, nhead=2, dim_ff=64, dropout=0.1)
MODEL_KWARGS = dict(input_dim=2, d_model=32, nhead=2, dim_ff=64, num_classes=3, dropout=0.1)
LABEL_MAP = {"(N": 0, "(AFIB": 1, "(AFL": 2}


def build_old_lab():
    """Dataset persis seperti ekstraksi_hrv.py (keluaran dialihkan ke folder ini)."""
    path = OUT / "old_lab.npz"
    if path.exists():
        return
    ex = AFDBExtractor(data_folder=str(ROOT / "data/afdb"), out_dir=str(OUT))
    ex.logger.setLevel(logging.WARNING)
    ex.out_sequence = "old_lab.npz"
    ex.out_tabular = "old_lab_tabular.csv"
    t0 = time.perf_counter()
    ex.run_pipeline()
    print(f"old_lab: {time.perf_counter() - t0:.0f}s")


def _old_backend_window(sig50):
    """Front-end backend lama (dokumentasi.md bagian 5) untuk satu window 10 s @ 50 Hz."""
    target_fs = 250
    signal = resample(sig50, int(len(sig50) * target_fs / hrv_core.FS_DEVICE))
    peaks, _ = find_peaks(signal, distance=target_fs * 0.4, height=np.mean(signal))
    if len(peaks) < 2:
        return None
    ibi_ms = np.diff(peaks) / float(target_fs) * 1000.0
    return hrv_core.build_sequence(ibi_ms)   # [IBI, delta_IBI], delta[0] = 0, pad ke 30


def build_old_frontend(records):
    """Window 10 s (stride 5 s) di dalam satu segmen ritme, diproses front-end lama."""
    path = OUT / "old_frontend.npz"
    if path.exists():
        return
    X, M, y, recs, n_total, n_fail = [], [], [], [], {}, {}
    W, STRIDE = int(hrv_core.WINDOW_SEC * hrv_core.FS_DEVICE), int(5 * hrv_core.FS_DEVICE)
    for rec in records:
        t0 = time.perf_counter()
        p = str(ROOT / "data/afdb" / rec)
        fs = wfdb.rdheader(p).fs
        sig = np.nan_to_num(wfdb.rdrecord(p, channels=[0]).p_signal[:, 0])
        x50 = hrv_core.resample_signal(sig, fs, hrv_core.FS_DEVICE)
        atr = wfdb.rdann(p, "atr")
        seg = [(s * hrv_core.FS_DEVICE / fs, LABEL_MAP.get(a.strip("\x00").strip(), -1))
               for s, a in zip(atr.sample, atr.aux_note) if a.strip("\x00").strip().startswith("(")]
        starts = np.array([s for s, _ in seg] + [len(x50)])
        labs = np.array([l for _, l in seg])
        tot = fail = 0
        for s in range(0, len(x50) - W + 1, STRIDE):
            i = np.searchsorted(starts, s, side="right") - 1
            if i < 0 or labs[i] < 0 or starts[i + 1] < s + W:
                continue            # window tidak berada di dalam satu segmen N/AFIB/AFL
            tot += 1
            out = _old_backend_window(x50[s:s + W])
            if out is None:
                fail += 1
                continue
            X.append(out[0]); M.append(out[1]); y.append(labs[i]); recs.append(rec)
        n_total[rec], n_fail[rec] = tot, fail
        print(f"  front-end lama {rec}: {tot} window, {fail} gagal (<2 puncak)  {time.perf_counter() - t0:.1f}s",
              flush=True)
    np.savez_compressed(path, X=np.stack(X), M=np.stack(M), y=np.array(y, dtype=np.int64),
                        records=np.array(recs), n_total=np.array([n_total[r] for r in records]),
                        n_fail=np.array([n_fail[r] for r in records]), rec_list=np.array(records))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    det = np.load(ROOT / "cache/dataset_v2_detector.npz")
    det_rec = det["records"].astype(str)
    patients = np.unique(det_rec)
    folds = np.load(OOF_DIR / "folds.npz")["folds"]

    build_old_lab()
    build_old_frontend(list(patients))
    lab = np.load(OUT / "old_lab.npz")
    fe = np.load(OUT / "old_frontend.npz")
    sets = {
        "lab": (lab["X"], lab["mask"], lab["records"].astype(str)),
        "det": (det["X"], det["M"], det_rec),
        "fe": (fe["X"], fe["M"], fe["records"].astype(str)),
    }
    print({k: v[0].shape[0] for k, v in sets.items()}, "| kelas lab:", np.bincount(lab["y"]).tolist())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for r in range(N_REPEATS):
        out = OUT / f"oof_r{r}.npz"
        if out.exists():
            continue
        rec2fold = {p: int(folds[r][det_rec == p][0]) for p in patients}
        f_sets = {k: np.array([rec2fold[x] for x in v[2]]) for k, v in sets.items()}
        oof = {k: np.full((len(v[2]), 3), np.nan, dtype=np.float32) for k, v in sets.items()}
        times = []
        for k in range(5):
            tr = f_sets["lab"] != k
            net, mu, sd, t = cv.train_dl("MicroTransformer", MODEL_KWARGS, lab["X"][tr], lab["mask"][tr],
                                         lab["y"][tr], seed=SEED + 100 * r + k, device=device, **HP)
            for name, (X, M, _) in sets.items():
                te = f_sets[name] == k
                oof[name][te] = cv.predict_dl(net, X[te], M[te], mu, sd, device)
            times.append(t)
            print(f"  ulangan {r} fold {k}: {t:.0f}s", flush=True)
        assert all(not np.isnan(v).any() for v in oof.values())
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, times=np.asarray(times), **oof)
        tmp.replace(out)
        print(f"✓ ulangan {r} selesai", flush=True)
    print("SELESAI")


if __name__ == "__main__":
    main()
