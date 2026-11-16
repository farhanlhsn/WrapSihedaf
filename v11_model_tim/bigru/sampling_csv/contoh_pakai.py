"""
contoh_pakai.py — membangun data latih dan data uji satu (ulangan, fold) dari CSV di folder ini. Butuh pandas + numpy.

    python contoh_pakai.py             # contoh: ulangan 0, fold 0, ketiga metode

Aturan: balancing hanya untuk data latih fold itu. Data uji selalu window asli pasien fold uji, tanpa balancing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FITUR = ["mean_ibi", "median_ibi", "min_ibi", "max_ibi", "sdnn", "rmssd", "pnn50", "cv", "sd1", "sd2", "sd1_sd2", "tpr",
         "sampen", "cosen", "hr_mean", "beat_count"]
MAKS_DETAK = 30


def muat(folder=Path(__file__).resolve().parent):
    """(window, sampling_under, sampling_over). Index tabel window = window_id."""
    folder = Path(folder)
    win = pd.read_csv(folder / "window.csv", dtype={"pasien": str, "ibi_ms": str})
    win = win.astype({f: np.float32 for f in FITUR + ["t_mulai_s"]}).set_index("window_id", drop=False)
    return win, pd.read_csv(folder / "sampling_under.csv"), pd.read_csv(folder / "sampling_over.csv")


def data_uji(win, r, k):
    """Window pasien fold k di ulangan r, distribusi asli."""
    return win[win[f"fold_u{r}"] == k]


def data_latih(win, under, over, r, k, metode=None):
    """metode None (tanpa balancing), "under", atau "over"."""
    latih = win[win[f"fold_u{r}"] != k]
    if metode is None:
        return latih
    if metode == "under":
        return win.loc[under.loc[(under.ulangan == r) & (under.fold == k), "window_id"].to_numpy()]
    if metode == "over":
        o = over[(over.ulangan == r) & (over.fold == k)]
        muncul = pd.Series(1, index=latih.index)
        muncul.loc[o.window_id.to_numpy()] = o.muncul.to_numpy()
        return latih.loc[latih.index.repeat(muncul.to_numpy())]
    raise ValueError(f"metode {metode!r} tidak dikenal: None, 'under', atau 'over'")


def urutan_ibi(df, maks=MAKS_DETAK):
    """Input model urutan: X (n, maks, 2) = [IBI ms, selisih IBI], mask M (n, maks). Sama dengan hrv_core.build_sequence."""
    X = np.zeros((len(df), maks, 2), dtype=np.float32)
    M = np.zeros((len(df), maks), dtype=np.float32)
    for i, s in enumerate(df["ibi_ms"]):
        ibi = np.array(s.split(";"), dtype=np.float32)[:maks]
        X[i, :len(ibi), 0] = ibi
        X[i, 1:len(ibi), 1] = np.diff(ibi)
        M[i, :len(ibi)] = 1.0
    return X, M


if __name__ == "__main__":
    win, under, over = muat()
    uji = data_uji(win, 0, 0)
    print(f"uji   ulangan 0 fold 0: {len(uji):7d} window, per kelas {uji.label.value_counts().sort_index().tolist()}")
    for metode in (None, "under", "over"):
        latih = data_latih(win, under, over, 0, 0, metode)
        print(f"latih {str(metode):6s}: {len(latih):7d} window, per kelas {latih.label.value_counts().sort_index().tolist()}")
