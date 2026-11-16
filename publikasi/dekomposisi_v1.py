"""Dekomposisi angka v1 (klaim 0,826 → front-end lama 0,418) per faktor, untuk paper evaluasi kritis.

Resep v1 (MicroTransformer, weighted CE, argmax; konstanta diimpor dari v3_evaluasi_robust/eval_versi_lama.py)
dilatih pada window ekstraksi v1 (`old_lab`: R-peak anotasi 250 Hz, stride N 5 s / AFIB 10 s / AFL 1 s), lalu tiap
model fold diuji pada pasien yang tidak dilatih di empat set uji. Training sama untuk semua langkah, jadi selisih
antar langkah hanya berasal dari cara mengukur:

  B  partisi v1 (StratifiedGroupKFold shuffle, seed 42), Macro-F1 dirata-rata per fold  ← cara model.ipynb melapor
  D  fold v3 (5 ulangan × 5 fold), rata-rata per fold
  E  fold v3, pooled OOF
  F  + window AFL uji dijarangkan ke stride ≥ 5 s (window 1 s tumpang tindih 90%)
  G  + window AFIB uji diberi bobot 2 (emulasi stride 5 s; ekstraksi v1 memakai 10 s)
  H  window ekstraksi v2 dengan R-peak anotasi (`dataset_v2_annotation`, stride seragam 5 s, cek kualitas hrv_core)
  I  window ekstraksi v2 dengan R-peak detektor 50 Hz (`dataset_v2_detector`, kondisi perangkat)
  J  front-end backend lama (`find_peaks` polos di sinyal 50 Hz, tanpa cek kualitas)

Ditambah demo bahwa StratifiedGroupKFold(shuffle=True) dengan 23 pasien hampir tidak mengubah partisi antar seed.

Jalankan dari root repo:
  python publikasi/dekomposisi_v1.py latih     # ±25 menit GPU; hasil per ulangan di cache, bisa dilanjutkan
  python publikasi/dekomposisi_v1.py hitung    # metrik + bootstrap pasien → cache/publikasi/dekomposisi_v1/ringkasan.json
"""
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
try:
    import wfdb  # noqa: F401
except ModuleNotFoundError:   # wfdb hanya dipakai pembangun dataset di eval_versi_lama; dataset sudah ada di cache
    sys.modules["wfdb"] = types.ModuleType("wfdb")
sys.path[:0] = [str(ROOT / "v3_evaluasi_robust"), str(ROOT / "v2_revisi"), str(ROOT / "v1_lama")]
import eval_versi_lama as evl  # noqa: E402
import sihedaf_cv as cv  # noqa: E402

OOF_DIR = ROOT / "cache/oof_robust/detector_e30_r5_s2026"
VL = OOF_DIR / "versi_lama"
OUT = ROOT / "cache/publikasi/dekomposisi_v1"
N_REPEATS, N_BOOT = evl.N_REPEATS, 2000
SEED_PARTISI_V1 = evl.SEED + 10_000          # seed latih partisi v1: SEED_PARTISI_V1 + k
# Skor yang dilaporkan model.ipynb (sama dengan CLAIM di v3_evaluasi_robust/laporan/build_report.py)
KLAIM = {"Macro-F1": 0.8257, "Macro-F1 sd": 0.0836}


def muat_set():
    lab = np.load(VL / "old_lab.npz")
    import pandas as pd
    tab = pd.read_csv(VL / "old_lab_tabular.csv", dtype={"record": str})
    assert len(tab) == len(lab["y"]) and (tab["label"].to_numpy() == lab["y"]).all() \
        and (tab["record"].to_numpy() == lab["records"].astype(str)).all(), "CSV tabular tidak sejajar dengan old_lab"
    ann = np.load(ROOT / "cache/dataset_v2_annotation.npz")
    det = np.load(ROOT / "cache/dataset_v2_detector.npz")
    fe = np.load(VL / "old_frontend.npz")
    return {
        "lab": dict(X=lab["X"], M=lab["mask"], y=lab["y"], rec=lab["records"].astype(str),
                    t=tab["window_start"].to_numpy(np.float64)),
        "ann": dict(X=ann["X"], M=ann["M"], y=ann["y"], rec=ann["records"].astype(str)),
        "det": dict(X=det["X"], M=det["M"], y=det["y"], rec=det["records"].astype(str)),
        "fe": dict(X=fe["X"], M=fe["M"], y=fe["y"], rec=fe["records"].astype(str)),
    }


def partisi_v1(lab):
    """Fold per window persis seperti model.ipynb: StratifiedGroupKFold(5, shuffle=True, random_state=42)."""
    from sklearn.model_selection import StratifiedGroupKFold
    f = np.empty(len(lab["y"]), dtype=np.int8)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42)
                                .split(lab["X"], lab["y"], lab["rec"])):
        f[va] = k
    return f


# ── Tahap 1: training (GPU, bisa dilanjutkan) ───────────────────────────
def latih():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    S = muat_set()
    lab = S["lab"]
    folds = np.load(OOF_DIR / "folds.npz")["folds"]
    patients = np.unique(S["det"]["rec"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}; window: " + ", ".join(f"{k} {len(v['y'])}" for k, v in S.items()), flush=True)

    def fit(tr, seed):
        net, mu, sd, t = cv.train_dl("MicroTransformer", evl.MODEL_KWARGS, lab["X"][tr], lab["M"][tr], lab["y"][tr],
                                     seed=seed, device=device, **evl.HP)
        return net, mu, sd, t

    for r in range(N_REPEATS):
        out = OUT / f"oof_r{r}.npz"
        if out.exists():
            continue
        rec2fold = {p: int(folds[r][S["det"]["rec"] == p][0]) for p in patients}
        fset = {k: np.array([rec2fold[x] for x in v["rec"]]) for k, v in S.items()}
        oof = {k: np.full((len(v["y"]), 3), np.nan, dtype=np.float32) for k, v in S.items()}
        for k in range(5):
            net, mu, sd, t = fit(fset["lab"] != k, evl.SEED + 100 * r + k)
            for name, d in S.items():
                te = fset[name] == k
                oof[name][te] = cv.predict_dl(net, d["X"][te], d["M"][te], mu, sd, device)
            print(f"  ulangan {r} fold {k}: {t:.0f}s", flush=True)
        assert all(not np.isnan(v).any() for v in oof.values())
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, **oof)
        tmp.replace(out)
        print(f"✓ ulangan {r}", flush=True)

    out = OUT / "oof_partisi_v1.npz"
    if not out.exists():
        f = partisi_v1(lab)
        oof = np.full((len(lab["y"]), 3), np.nan, dtype=np.float32)
        for k in range(5):
            net, mu, sd, t = fit(f != k, SEED_PARTISI_V1 + k)
            te = f == k
            oof[te] = cv.predict_dl(net, lab["X"][te], lab["M"][te], mu, sd, device)
            print(f"  partisi v1 fold {k}: {t:.0f}s", flush=True)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, lab=oof, folds=f)
        tmp.replace(out)
        print("✓ partisi v1", flush=True)
    print("SELESAI", flush=True)


# ── Tahap 2: metrik ──────────────────────────────────────────────────────
def jarangkan(t, rec, y, kelas=2, jarak=5.0):
    """Buang window kelas `kelas` yang mulai < `jarak` detik setelah window terakhir yang disimpan (per rekaman)."""
    keep = np.ones(len(y), bool)
    idx = np.flatnonzero(y == kelas)
    last = {}
    for i in idx[np.lexsort((t[idx], rec[idx]))]:
        if rec[i] in last and t[i] - last[rec[i]] < jarak - 1e-6:
            keep[i] = False
        else:
            last[rec[i]] = t[i]
    return keep


def cms_berbobot(y, pred, rec, patients, w):
    pidx = np.searchsorted(patients, rec)
    flat = (pidx * cv.K + y) * cv.K + pred
    return np.bincount(flat, weights=w, minlength=len(patients) * cv.K * cv.K).reshape(len(patients), cv.K, cv.K)


def rata_per_fold(y, pred, f):
    from sklearn.metrics import f1_score
    return [f1_score(y[f == k], pred[f == k], average="macro", zero_division=0) for k in np.unique(f)]


def demo_partisi(y, rec, seeds):
    """Berapa partisi pasien berbeda yang dihasilkan StratifiedGroupKFold(shuffle=True) untuk banyak seed."""
    from sklearn.model_selection import StratifiedGroupKFold
    patients = np.unique(rec)
    afl = set(np.unique(rec[y == 2]))
    parts, afl_parts = [], []
    for s in seeds:
        assign = {}
        for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(np.zeros(len(y)), y, rec)):
            for p in np.unique(rec[va]):
                assign[p] = k
        groups = [frozenset(p for p in patients if assign[p] == k) for k in range(5)]
        parts.append(frozenset(groups))
        afl_parts.append(frozenset(g & afl for g in groups if g & afl))
    top = max(set(afl_parts), key=afl_parts.count)
    return {"n_seed": len(seeds), "partisi_berbeda": len(set(parts)), "partisi_afl_berbeda": len(set(afl_parts)),
            "frac_partisi_afl_terbanyak": afl_parts.count(top) / len(seeds),
            "partisi_afl_terbanyak": sorted(sorted(g) for g in top)}


def hitung():
    S = muat_set()
    folds = np.load(OOF_DIR / "folds.npz")["folds"]
    det = S["det"]
    patients = np.unique(det["rec"])
    strata = cv.patient_strata(det["y"], det["rec"], patients)
    W = cv.bootstrap_weights(strata, N_BOOT, seed=evl.SEED)          # sama dengan laporan v3
    oofs = [np.load(OUT / f"oof_r{r}.npz") for r in range(N_REPEATS)]
    rec2fold = [{p: int(folds[r][det["rec"] == p][0]) for p in patients} for r in range(N_REPEATS)]
    fset = {k: [np.array([rec2fold[r][x] for x in v["rec"]]) for r in range(N_REPEATS)] for k, v in S.items()}
    lab = S["lab"]

    # set uji turunan dari window lab
    keep_afl = jarangkan(lab["t"], lab["rec"], lab["y"])
    w_afib = np.where(lab["y"] == 1, 2.0, 1.0)

    def cms_langkah(name, mask=None, w=None):
        d = S[name]
        m = np.ones(len(d["y"]), bool) if mask is None else mask
        ww = np.ones(m.sum()) if w is None else w[m]
        return np.stack([cms_berbobot(d["y"][m], o[name].argmax(1)[m], d["rec"][m], patients, ww) for o in oofs])

    langkah = {
        "E": ("Fold v3, pooled OOF (window v1)", cms_langkah("lab")),
        "F": ("+ AFL uji dijarangkan ke stride ≥ 5 s", cms_langkah("lab", keep_afl)),
        "G": ("+ AFIB uji bobot 2 (emulasi stride 5 s)", cms_langkah("lab", keep_afl, w_afib)),
        "H": ("Window v2, R-peak anotasi 250 Hz", cms_langkah("ann")),
        "I": ("Window v2, R-peak detektor 50 Hz", cms_langkah("det")),
        "J": ("Front-end lama find_peaks 50 Hz", cms_langkah("fe")),
    }
    mets = ("Macro-F1", "Normal R", "AFIB F1", "AFL P", "AFL R", "AFL F1")
    boot = {k: cv.boot_metrics(c, W) for k, (_, c) in langkah.items()}
    point = {k: cv.point_metrics(c) for k, (_, c) in langkah.items()}
    hasil = {}
    keys = list(langkah)
    for i, k in enumerate(keys):
        row = {"label": langkah[k][0]}
        for m in mets:
            row[m] = [float(point[k][m].mean()), *map(float, np.percentile(boot[k][m], [2.5, 97.5]))]
        if i:
            prev = keys[i - 1]
            for m in ("Macro-F1", "AFL P", "AFL F1"):
                dd = boot[k][m] - boot[prev][m]
                row[f"Δ {m}"] = [float(point[k][m].mean() - point[prev][m].mean()),
                                 *map(float, np.percentile(dd, [2.5, 97.5])), float(np.mean(dd > 0))]
        hasil[k] = row

    # langkah rata-rata per fold
    pv1 = np.load(OUT / "oof_partisi_v1.npz")
    fB = rata_per_fold(lab["y"], pv1["lab"].argmax(1), pv1["folds"])
    fD = [rata_per_fold(lab["y"], o["lab"].argmax(1), fset["lab"][r]) for r, o in enumerate(oofs)]
    hasil["A"] = {"label": "Klaim model.ipynb (partisi v1, rata-rata per fold)",
                  "Macro-F1": [KLAIM["Macro-F1"]], "sd_antar_fold": KLAIM["Macro-F1 sd"]}
    hasil["B"] = {"label": "Diulang: partisi v1, rata-rata per fold", "Macro-F1": [float(np.mean(fB))],
                  "sd_antar_fold": float(np.std(fB, ddof=1)), "per_fold": list(map(float, fB))}
    hasil["D"] = {"label": "Fold v3, rata-rata per fold", "Macro-F1": [float(np.mean(fD))],
                  "sd_antar_fold": float(np.std(np.ravel(fD), ddof=1)),
                  "sd_antar_ulangan": float(np.std(np.mean(fD, 1), ddof=1))}

    # paritas dengan OOF cache v3 (eval_versi_lama.py): run GPU berbeda, harus dekat
    lama = [np.load(VL / f"oof_r{r}.npz") for r in range(N_REPEATS)]
    paritas = {}
    for name in ("lab", "det", "fe"):
        d = S[name]
        c_baru = np.stack([cv.patient_cms(d["y"], o[name].argmax(1), d["rec"], patients) for o in oofs])
        c_lama = np.stack([cv.patient_cms(d["y"], o[name].argmax(1), d["rec"], patients) for o in lama])
        paritas[name] = {"baru": float(cv.point_metrics(c_baru)["Macro-F1"].mean()),
                         "cache_v3": float(cv.point_metrics(c_lama)["Macro-F1"].mean()),
                         "argmax_sama": float(np.mean([(a[name].argmax(1) == b[name].argmax(1)).mean()
                                                       for a, b in zip(oofs, lama)]))}

    # jumlah window per set uji
    def hitung_afl(y, rec, m=None):
        m = np.ones(len(y), bool) if m is None else m
        return {"kelas": np.bincount(y[m], minlength=3).tolist(),
                "afl_04936": int(((y == 2) & (rec == "04936") & m).sum())}
    jumlah = {"lab": hitung_afl(lab["y"], lab["rec"]), "lab_jarang": hitung_afl(lab["y"], lab["rec"], keep_afl),
              "ann": hitung_afl(S["ann"]["y"], S["ann"]["rec"]), "det": hitung_afl(det["y"], det["rec"]),
              "fe": hitung_afl(S["fe"]["y"], S["fe"]["rec"])}

    import sklearn
    seeds = list(range(100))
    demo = {"det": demo_partisi(det["y"], det["rec"], seeds), "lab": demo_partisi(lab["y"], lab["rec"], seeds),
            "fold_v3": {"partisi_berbeda": len({frozenset(frozenset(np.unique(det["rec"][folds[r] == k])) for k in range(5))
                                                 for r in range(N_REPEATS)}), "n_ulangan": N_REPEATS},
            "sklearn": sklearn.__version__}

    ring = {"langkah": {k: hasil[k] for k in "ABDEFGHIJ"}, "paritas": paritas, "jumlah_window": jumlah,
            "demo_partisi": demo, "n_boot": N_BOOT, "seed_partisi_v1": SEED_PARTISI_V1}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ringkasan.json").write_text(json.dumps(ring, indent=1, ensure_ascii=False))
    cetak(ring)


def cetak(ring):
    f = lambda x: f"{x:.3f}"
    print("| Langkah | Kondisi | Macro-F1 | AFL P | AFL R | AFL F1 | Δ Macro-F1 vs langkah sebelumnya [CI] (P) |")
    print("|---|---|---|---|---|---|---|")
    for k, r in ring["langkah"].items():
        mf = r["Macro-F1"]
        if len(mf) == 1:
            print(f"| {k} | {r['label']} | {f(mf[0])} ± {f(r['sd_antar_fold'])} (sd antar fold) | | | | |")
            continue
        cells = [f"{f(r[m][0])} [{r[m][1]:.2f}–{r[m][2]:.2f}]" for m in ("Macro-F1", "AFL P", "AFL R", "AFL F1")]
        d = r.get("Δ Macro-F1")
        dd = f"{d[0]:+.3f} [{d[1]:+.3f}; {d[2]:+.3f}] (P = {d[3]:.3f})" if d else ""
        print(f"| {k} | {r['label']} | " + " | ".join(cells) + f" | {dd} |")
    print("\nParitas dengan OOF cache v3:", json.dumps(ring["paritas"], ensure_ascii=False))
    print("Jumlah window:", json.dumps(ring["jumlah_window"], ensure_ascii=False))
    print("Demo partisi:", json.dumps(ring["demo_partisi"], ensure_ascii=False))


if __name__ == "__main__":
    {"latih": latih, "hitung": hitung}[sys.argv[1]]()
