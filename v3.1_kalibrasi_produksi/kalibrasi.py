"""
kalibrasi.py — logika v3.1, dipakai bersama oleh kalibrasi_produksi.ipynb dan laporan/build_report.py.

v3.1 tidak melatih model. Ia memakai OOF MicroTransformer dari v3 (fold & seed resmi) untuk:
1. memilih bias kelas produksi baru (grid −7…+3, dipilih pada OOF pooled 5 ulangan);
2. membandingkan bias produksi v2 `[0, 2, −3]`, prosedur baru, dan v1 dengan bootstrap pasien yang sama;
3. melihat efek bias pada model final (dilatih pada semua pasien) — hanya sebagai cek kewajaran.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "v3_evaluasi_robust"), str(REPO / "v2_revisi")]
import sihedaf_cv as cv  # noqa: E402

OOF_DIR = REPO / "cache/oof_robust/detector_e30_r5_s2026"
DATA = REPO / "cache/dataset_v2_detector.npz"
V2_EXPORT = REPO / "v2_revisi/model_export"
CACHE_OUT = REPO / "cache/v3_1"
N_REPEATS, SEED, N_BOOT = 5, 2026, 2000
OLD_GRID = np.linspace(-3, 3, 25)

# Varian yang dibandingkan (kunci → label laporan)
VARIANTS = {
    "v1": "v1 (sebelum revisi), argmax",
    "v2_argmax": "v2 tanpa bias (argmax)",
    "v2_prod": "v2 produksi: bias tetap [0; 2; −3]",
    "v2_cf": "Prosedur v2: cross-fit grid −3…+3",
    "v31_cf": "v3.1: cross-fit grid −7…+3 (estimasi jujur)",
    "v31_fixed": "v3.1: bias tetap baru (optimistis)",
}
PAIRS = [("v31_cf", "v2_prod"), ("v31_cf", "v1"), ("v2_prod", "v1"), ("v31_fixed", "v31_cf")]


def load():
    z = np.load(DATA)
    y, records = z["y"], z["records"].astype(str)
    patients = np.unique(records)
    return SimpleNamespace(
        y=y, records=records, patients=patients,
        strata=cv.patient_strata(y, records, patients),
        folds=np.load(OOF_DIR / "folds.npz")["folds"],
        oof=np.stack([np.load(OOF_DIR / f"MicroTransformer_r{r}.npz")["oof"] for r in range(N_REPEATS)]),
        oof_v1=np.stack([np.load(OOF_DIR / "versi_lama" / f"oof_r{r}.npz")["det"] for r in range(N_REPEATS)]),
        meta_v2=json.loads((V2_EXPORT / "sihedaf_v2.json").read_text()),
    )


def _crossfit(y, prob, folds_r, grid):
    """Bias fold k dipilih dari OOF fold lain (sama dengan sihedaf_cv.crossfit_bias, grid bisa diganti)."""
    adj, biases = np.empty_like(prob), []
    for k in np.unique(folds_r):
        tr, te = folds_r != k, folds_r == k
        b = cv.fit_class_bias(y[tr], prob[tr], grid=grid)
        adj[te] = cv.apply_bias(prob[te], b)
        biases.append(b)
    return adj, np.array(biases)


def fit_production_bias(d):
    """Bias tunggal untuk produksi + sebaran bias per fold (cross-fit) sebagai cek stabilitas."""
    b_new = cv.fit_class_bias(np.tile(d.y, N_REPEATS), d.oof.reshape(-1, 3), grid=cv.BIAS_GRID)
    new = [_crossfit(d.y, d.oof[r], d.folds[r], cv.BIAS_GRID) for r in range(N_REPEATS)]
    old = [_crossfit(d.y, d.oof[r], d.folds[r], OLD_GRID) for r in range(N_REPEATS)]
    return SimpleNamespace(b_new=b_new,
                           adj_new=np.stack([a for a, _ in new]), fold_new=np.concatenate([b for _, b in new]),
                           adj_old=np.stack([a for a, _ in old]), fold_old=np.concatenate([b for _, b in old]))


def evaluate(d, cal):
    b_v2 = np.asarray(d.meta_v2["class_bias"], dtype=np.float64)
    preds = {
        "v1": d.oof_v1.argmax(-1),
        "v2_argmax": d.oof.argmax(-1),
        "v2_prod": np.stack([cv.apply_bias(d.oof[r], b_v2).argmax(1) for r in range(N_REPEATS)]),
        "v2_cf": cal.adj_old.argmax(-1),
        "v31_cf": cal.adj_new.argmax(-1),
        "v31_fixed": np.stack([cv.apply_bias(d.oof[r], cal.b_new).argmax(1) for r in range(N_REPEATS)]),
    }
    cms = {k: np.stack([cv.patient_cms(d.y, p, d.records, d.patients) for p in v]) for k, v in preds.items()}
    W = cv.bootstrap_weights(d.strata, N_BOOT, seed=SEED)
    boot = {k: cv.boot_metrics(c, W) for k, c in cms.items()}
    point = {k: cv.point_metrics(c) for k, c in cms.items()}
    rows = {k: {m: [float(point[k][m].mean()), *map(float, np.percentile(boot[k][m], [2.5, 97.5]))]
                for m in cv.METRICS} for k in cms}
    pairs = {}
    for a, b in PAIRS:
        pairs[f"{a} vs {b}"] = {}
        for m in ("Macro-F1", "AFL F1", "AFL P", "AFL R", "Normal R"):
            dd = boot[a][m] - boot[b][m]
            pairs[f"{a} vs {b}"][m] = [float(point[a][m].mean() - point[b][m].mean()),
                                       *map(float, np.percentile(dd, [2.5, 97.5])), float(np.mean(dd > 0))]
    afl_pat = d.patients[d.strata == 2]
    pi = np.searchsorted(d.patients, afl_pat)
    afl_recall = {k: (cms[k][:, pi, 2, 2] / cms[k][:, pi, 2, :].sum(-1)).mean(0).tolist() for k in cms}
    fp = {k: cms[k][:, :, :2, 2].sum(-1).mean(0) for k in cms}       # window non-AFL → AFL, per pasien
    fp_summary = {k: {"total": float(v.sum()), "pasien_AFL": float(v[d.strata == 2].sum()),
                      "pasien_lain": float(v[d.strata != 2].sum())} for k, v in fp.items()}
    return SimpleNamespace(rows=rows, pairs=pairs, afl_patients=afl_pat.tolist(), afl_recall=afl_recall,
                           fp_summary=fp_summary, b_v2=b_v2.tolist())


def final_model_rates(d, biases: dict, device="cpu"):
    """
    Model FINAL (dilatih pada semua pasien) pada semua window detektor → proporsi prediksi per kelas.
    Ini in-sample: hanya untuk melihat arah efek bias (mis. berapa sering AFL ditebak), bukan akurasi.
    """
    from predictor import ECGPredictor
    z = np.load(DATA)
    P = ECGPredictor(str(V2_EXPORT), device=device)
    P.class_bias = np.zeros(3, dtype=np.float32)
    raw = np.concatenate([P.predict_proba_sequences(z["X"][s:s + 8192], z["M"][s:s + 8192])
                          for s in range(0, len(d.y), 8192)])
    out = {"aktual": (np.bincount(d.y, minlength=3) / len(d.y)).tolist()}
    for name, b in biases.items():
        pred = cv.apply_bias(raw, np.asarray(b, dtype=np.float64)).argmax(1)
        out[name] = (np.bincount(pred, minlength=3) / len(pred)).tolist()
    return out


def rate_biases(ev, cal):
    return {"v2 [0; 2; −3]": ev.b_v2, "v3.1 [0; −0,5; −4,5]": cal.b_new.tolist()}


def run_all(device="cpu"):
    """Semua angka v3.1 dalam satu dict (dipakai laporan; notebook memanggil langkahnya satu per satu)."""
    d = load()
    cal = fit_production_bias(d)
    ev = evaluate(d, cal)
    return assemble(d, cal, ev, final_model_rates(d, rate_biases(ev, cal), device=device))


def assemble(d, cal, ev, rates):
    summ = lambda a: {"median": np.median(a, 0).tolist(), "min": a.min(0).tolist(), "max": a.max(0).tolist()}
    return {
        "b_v2": ev.b_v2, "b_new": cal.b_new.tolist(),
        "grid": [float(cv.BIAS_GRID.min()), float(cv.BIAS_GRID.max()), float(cv.BIAS_GRID[1] - cv.BIAS_GRID[0])],
        "fold_bias_new": cal.fold_new.tolist(), "fold_bias_old": cal.fold_old.tolist(),
        "fold_bias_new_summary": summ(cal.fold_new), "fold_bias_old_summary": summ(cal.fold_old),
        "rows": ev.rows, "pairs": ev.pairs, "labels": VARIANTS,
        "afl_patients": ev.afl_patients, "afl_recall": ev.afl_recall, "fp_summary": ev.fp_summary,
        "final_model_rates": rates,
        "n_windows": int(len(d.y)), "n_patients": int(len(d.patients)),
    }
