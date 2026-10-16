"""
takikardia.py — logika v3.2 (uji stres takikardia & analisis sensitivitas), dipakai notebook dan laporan.

Evaluasi saja: tidak ada training, paket produksi tidak disentuh. Menindaklanjuti review eksternal:
1. Uji sintetis ritme teratur 60–170 bpm × 4 jenis variasi detak → model yang punya bobot (v1, v2, v3.1).
2. Window Normal cepat ASLI di AFDB → semua model yang OOF-nya ada (v1, v2, v3.1, v4, v5, v6).
3. Sensitivitas tanpa pasien 04936 (penyumbang 58% window AFL; label AFL-nya dipertanyakan reviewer).
4. Bootstrap bersarang: bias kelas dipilih ULANG di setiap sampel bootstrap (CI v3 belum memasukkan ini).
5. Simulasi aturan produk "takikardia teratur → perlu dicek".
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "v3_evaluasi_robust"), str(REPO / "v2_revisi")]
import sihedaf_cv as cv  # noqa: E402
import hrv_core  # noqa: E402

OOF_DIR = REPO / "cache/oof_robust/detector_e30_r5_s2026"
DATA = REPO / "cache/dataset_v2_detector.npz"
CACHE_OUT = REPO / "cache/v3_2"
N_REPEATS, SEED, N_BOOT = 5, 2026, 2000
CLASSES = ("Normal", "AFIB", "AFL")
BPMS = list(range(60, 175, 5))
JITTERS = {"tanpa variasi": ("none", 0.0), "seragam ±10 ms": ("uniform", 10.0),
           "Gauss sd 10 ms": ("normal", 10.0), "Gauss sd 20 ms": ("normal", 20.0)}
N_SYNTH = 50                                   # window per (detak, jenis variasi)
HR_BINS = [(0, 100), (100, 110), (110, 120), (120, 999)]
SUSPECT = "04936"
# Model dengan OOF (window AFDB) — kunci → (label, sumber probabilitas, cara keputusan)
OOF_MODELS = {
    "v1": "v1 (sebelum revisi), argmax",
    "v2_prod": "v2 produksi, bias [0; 2; −3]",
    "v31": "v3.1 produksi (referensi)",
    "v4_e1": "v4 ensemble MT + BiGRU",
    "v5_lunak": "v5 bobot lunak",
    "v6_late": "v6 late fusion RR + EKG",
}


# ── data & prediksi OOF ────────────────────────────────────────────────
def load():
    z = np.load(DATA)
    y, rec, X, M = z["y"], z["records"].astype(str), z["X"], z["M"]
    ibi = [X[i, :int(M[i].sum()), 0] for i in range(len(y))]
    hr = np.array([60000.0 / v.mean() for v in ibi])
    cvv = np.array([v.std() / v.mean() for v in ibi])          # koefisien variasi IBI (keteraturan)
    patients = np.unique(rec)
    load_oof = lambda name: np.stack([np.load(OOF_DIR / f"{name}_r{r}.npz")["oof"] for r in range(N_REPEATS)])
    mt = load_oof("MicroTransformer")
    probs = {"v2_prod": mt, "v31": mt,
             "v4_e1": (mt + load_oof("BiGRUAttention")) / 2,
             "v5_lunak": load_oof("MicroTransformer_bobotlunak"),
             "v6_late": (mt + load_oof("EKG_CNN")) / 2}
    v1 = np.stack([np.load(OOF_DIR / "versi_lama" / f"oof_r{r}.npz")["det"] for r in range(N_REPEATS)])
    return dict(y=y, rec=rec, hr=hr, cv=cvv, patients=patients, strata=cv.patient_strata(y, rec, patients),
                folds=np.load(OOF_DIR / "folds.npz")["folds"], probs=probs, v1=v1,
                b_v2=np.array([0.0, 2.0, -3.0]))


def oof_predictions(d, mask=None):
    """Prediksi semua model (R, N). mask: hanya window ini yang dipakai (bias cross-fit dihitung ulang di subset)."""
    m = np.ones(len(d["y"]), bool) if mask is None else mask
    y = d["y"][m]
    pred = {"v1": d["v1"][:, m].argmax(-1),
            "v2_prod": np.stack([cv.apply_bias(d["probs"]["v2_prod"][r][m], d["b_v2"]).argmax(1) for r in range(N_REPEATS)])}
    for k in ("v31", "v4_e1", "v5_lunak", "v6_late"):
        pred[k] = np.stack([cv.crossfit_bias(y, d["probs"][k][r][m], d["folds"][r][m])[0].argmax(1) for r in range(N_REPEATS)])
    return pred


# ── 1. uji sintetis ────────────────────────────────────────────────────
def synthetic_ibi(bpm, kind, scale, n_win, seed):
    rng = np.random.default_rng(seed)
    rr = 60000.0 / bpm
    n = int(hrv_core.WINDOW_SEC * 1000 // rr)
    noise = {"none": lambda: np.zeros(n), "uniform": lambda: rng.uniform(-scale, scale, n),
             "normal": lambda: rng.normal(0, scale, n)}[kind]
    return [rr + noise() for _ in range(n_win)]


def load_packages():
    """Model yang punya bobot: v1 (final.joblib), v2 dan v3.1 (bobot sama, bias beda)."""
    from predictor import ECGPredictor
    spec = importlib.util.spec_from_file_location("predictor_v1", REPO / "v1_lama/predictor.py")
    v1mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1mod)
    v1 = v1mod.ECGPredictor(str(REPO / "v1_lama/final.joblib"), device="cpu")
    v2 = ECGPredictor(str(REPO / "v2_revisi/model_export"), device="cpu")
    v31 = ECGPredictor(str(REPO / "v3.1_kalibrasi_produksi/model_export"), name="sihedaf_v3_1", device="cpu")

    def run_v1(seqs):
        X = np.stack([s[0] for s in seqs]); M = np.stack([s[1] for s in seqs])
        return np.array([r["class_id"] for r in v1.predict(X, M)])

    def run_new(P):
        def f(seqs):
            X = np.stack([s[0] for s in seqs]); M = np.stack([s[1] for s in seqs])
            return P.predict_proba_sequences(X, M).argmax(1)
        return f
    return {"v1": run_v1, "v2_prod": run_new(v2), "v31": run_new(v31)}


def stress_test(packages):
    """{model: {jenis variasi: (len(BPMS), 3) proporsi tebakan Normal/AFIB/AFL}}."""
    out = {m: {} for m in packages}
    for j, (jname, (kind, scale)) in enumerate(JITTERS.items()):
        for m in packages:
            out[m][jname] = np.zeros((len(BPMS), 3))
        for b, bpm in enumerate(BPMS):
            seqs = [hrv_core.build_sequence(v) for v in synthetic_ibi(bpm, kind, scale, N_SYNTH, SEED + 1000 * j + bpm)]
            for m, run in packages.items():
                out[m][jname][b] = np.bincount(run(seqs), minlength=3) / N_SYNTH
    return out


def normal_threshold(table):
    """Detak terendah (bpm) di mana < 50% window ditebak Normal."""
    frac_normal = table[:, 0]
    idx = np.flatnonzero(frac_normal < 0.5)
    return BPMS[idx[0]] if idx.size else None


# ── 2. window asli per pita detak ──────────────────────────────────────
def by_hr_bin(d, pred):
    """Untuk tiap kelas asli & pita detak: jumlah window dan % tebakan per kelas (rata-rata ulangan)."""
    res = {}
    for c, cname in enumerate(CLASSES):
        for lo, hi in HR_BINS:
            m = (d["y"] == c) & (d["hr"] >= lo) & (d["hr"] < hi)
            key = f"{cname} | {lo}–{hi if hi < 999 else '∞'} bpm"
            res[key] = {"n": int(m.sum()), "n_pasien": int(len(np.unique(d["rec"][m])))}
            for k, p in pred.items():
                res[key][k] = (np.stack([np.bincount(pr[m], minlength=3) for pr in p]).mean(0) / max(m.sum(), 1)).tolist()
    return res


# ── 3. sensitivitas tanpa 04936 ────────────────────────────────────────
def metrics_with_ci(d, pred, mask):
    rec, y = d["rec"][mask], d["y"][mask]
    patients = np.unique(rec)
    strata = cv.patient_strata(y, rec, patients)
    W = cv.bootstrap_weights(strata, N_BOOT, seed=SEED)
    cms = {k: np.stack([cv.patient_cms(y, p, rec, patients) for p in v]) for k, v in pred.items()}
    boot = {k: cv.boot_metrics(c, W) for k, c in cms.items()}
    point = {k: cv.point_metrics(c) for k, c in cms.items()}
    rows = {k: {m: [float(point[k][m].mean()), *map(float, np.percentile(boot[k][m], [2.5, 97.5]))]
                for m in ("Macro-F1", "AFL F1", "AFL P", "AFL R", "Normal R")} for k in cms}
    pairs = {}
    for k in cms:
        if k == "v31":
            continue
        dd = boot["v31"]["Macro-F1"] - boot[k]["Macro-F1"]
        pairs[k] = [float(point["v31"]["Macro-F1"].mean() - point[k]["Macro-F1"].mean()),
                    *map(float, np.percentile(dd, [2.5, 97.5])), float(np.mean(dd > 0))]
    return {"rows": rows, "pairs_v31_minus": pairs, "n_pasien": int(len(patients)),
            "n_pasien_afl": int((strata == 2).sum()), "n_afl": int((y == 2).sum())}


# ── 4. bootstrap bersarang (bias dipilih ulang di tiap sampel) ─────────
def _macro_f1_cm(cm):
    """cm (..., 3, 3) → Macro-F1 atas kelas yang ada (sama dengan sihedaf_cv._macro_f1_fast)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(-1), cm.sum(-2)
    denom = support + predicted
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    present = support > 0
    return (f1 * present).sum(-1) / np.maximum(present.sum(-1), 1)


def candidate_cms(d, probs):
    """(R, G, P, 3, 3): confusion matrix per pasien untuk SETIAP kandidat bias grid (urutan = fit_class_bias)."""
    grid = cv.BIAS_GRID
    biases = np.array([[0.0, b1, b2] for b1 in grid for b2 in grid])
    out = np.zeros((N_REPEATS, len(biases), len(d["patients"]), 3, 3))
    for r in range(N_REPEATS):
        logp = np.log(np.clip(probs[r], 1e-9, 1))
        for g, b in enumerate(biases):
            out[r, g] = cv.patient_cms(d["y"], (logp + b).argmax(1), d["rec"], d["patients"])
    return out, biases


def nested_bootstrap(d, cand, fixed_cms):
    """
    Untuk tiap sampel bootstrap (bobot pasien W_b): di tiap ulangan & fold, bias dipilih ulang pada pasien LATIH
    berbobot W_b, lalu dipasang ke pasien uji. Bobot semua-satu = cross-fit biasa (dicek di notebook).
    fixed_cms: {model: (R, P, 3, 3)} pembanding tanpa pemilihan bias (v1, v2 produksi).
    """
    W = cv.bootstrap_weights(d["strata"], N_BOOT, seed=SEED)
    W = np.vstack([np.ones(len(d["patients"])), W])                 # baris 0 = titik (tanpa resampling)
    pfold = np.array([[int(d["folds"][r][d["rec"] == p][0]) for p in d["patients"]] for r in range(N_REPEATS)])
    v31 = np.zeros((len(W), N_REPEATS))
    G, P = cand.shape[1], len(d["patients"])
    for r in range(N_REPEATS):
        flat = cand[r].reshape(G, P, 9)                                       # (G, P, 9)
        acc = np.zeros((len(W), 9))
        for k in range(5):
            tr, te = pfold[r] != k, pfold[r] == k
            # cm latih berbobot untuk semua kandidat sekaligus: (B, P_tr) @ (P_tr, G·9)
            train_cm = W[:, tr] @ flat[:, tr].transpose(1, 0, 2).reshape(tr.sum(), G * 9)
            gbest = _macro_f1_cm(train_cm.reshape(len(W), G, 3, 3)).argmax(1)  # kandidat pertama terbaik
            acc += np.einsum("bpc,bp->bc", flat[gbest][:, te], W[:, te])        # pasang ke pasien uji
        v31[:, r] = _macro_f1_cm(acc.reshape(len(W), 3, 3))
    v31 = v31.mean(1)
    fixed = {k: _macro_f1_cm(np.einsum("bp,rpij->brij", W, c)).mean(1) for k, c in fixed_cms.items()}
    out = {"v31_point": float(v31[0]), "v31_ci": np.percentile(v31[1:], [2.5, 97.5]).tolist()}
    for k, f in fixed.items():
        dd = v31[1:] - f[1:]
        out[k] = {"point": float(f[0]), "delta": float(v31[0] - f[0]),
                  "ci": np.percentile(dd, [2.5, 97.5]).tolist(), "p": float(np.mean(dd > 0))}
    return out


# ── 5. aturan produk ───────────────────────────────────────────────────
RULE_HR = (100, 110, 120)
RULE_CV = (0.03, 0.05, 0.08)


def product_rule(d, pred_v31, synth_v31=None):
    """
    Aturan: jika detak ≥ H bpm DAN koefisien variasi IBI ≤ C, keluaran = "takikardia teratur, perlu dicek"
    (bukan AFIB/AFL). Hitung % window tiap kelas asli yang berubah label, dan % alarm AFIB/AFL yang hilang.
    """
    out = {}
    alarm = pred_v31 != 0                                            # (R, N) tebakan AFIB/AFL
    for H in RULE_HR:
        for C in RULE_CV:
            hit = (d["hr"] >= H) & (d["cv"] <= C)
            row = {}
            for c, cname in enumerate(CLASSES):
                m = d["y"] == c
                row[f"{cname} kena aturan"] = float(hit[m].mean())
            # alarm AFIB/AFL yang dialihkan, dipisah: palsu (asli Normal) vs benar (asli AFIB/AFL)
            fa = alarm & (d["y"] == 0)[None]
            ta = alarm & (d["y"] != 0)[None]
            row["alarm palsu dialihkan"] = float((fa & hit[None]).sum() / max(fa.sum(), 1))
            row["alarm benar dialihkan"] = float((ta & hit[None]).sum() / max(ta.sum(), 1))
            row["AFL terdeteksi dialihkan"] = float((alarm & (pred_v31 == 2) & (d["y"] == 2)[None] & hit[None]).sum()
                                                    / max(((pred_v31 == 2) & (d["y"] == 2)[None]).sum(), 1))
            out[f"≥{H} bpm & CV ≤ {C:.2f}"] = row
    return out


def synthetic_cv():
    """Koefisien variasi IBI tipikal tiap jenis variasi sintetis pada 120 dan 170 bpm (acuan memilih ambang aturan)."""
    out = {}
    for jname, (kind, scale) in JITTERS.items():
        for bpm in (120, 170):
            v = synthetic_ibi(bpm, kind, scale, 200, SEED)
            out[f"{jname} @ {bpm} bpm"] = float(np.median([x.std() / x.mean() for x in v]))
    return out


def cached_predictions(d):
    """Bagian lambat (±12 menit: bias cross-fit & kandidat bias) disimpan di ../cache/v3_2/ dan dipakai ulang."""
    path = CACHE_OUT / "prediksi.npz"
    if path.exists():
        z = np.load(path)
        pred = {k[5:]: z[k] for k in z.files if k.startswith("semua")}
        pred_wo = {k[5:]: z[k] for k in z.files if k.startswith("tanpa")}
        return pred, pred_wo, z["cand"]
    keep = d["rec"] != SUSPECT
    pred = oof_predictions(d)
    pred_wo = {k: v for k, v in oof_predictions(d, keep).items() if k in ("v1", "v2_prod", "v31")}
    cand, _ = candidate_cms(d, d["probs"]["v31"])
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cand=cand, **{f"semua{k}": v.astype(np.int8) for k, v in pred.items()},
                        **{f"tanpa{k}": v.astype(np.int8) for k, v in pred_wo.items()})
    return pred, pred_wo, cand


def run_all(packages=None):
    d = load()
    packages = packages or load_packages()
    stress = stress_test(packages)
    pred, pred_wo, cand = cached_predictions(d)
    keep = d["rec"] != SUSPECT
    sens_all = metrics_with_ci(d, {k: pred[k] for k in ("v1", "v2_prod", "v31")}, np.ones(len(d["y"]), bool))
    sens_wo = metrics_with_ci(d, pred_wo, keep)
    fixed = {k: np.stack([cv.patient_cms(d["y"], p, d["rec"], d["patients"]) for p in pred[k]]) for k in ("v1", "v2_prod")}
    nested = nested_bootstrap(d, cand, fixed)
    cv_desc = {CLASSES[c]: {"median": float(np.median(d["cv"][d["y"] == c])),
                            "p10": float(np.percentile(d["cv"][d["y"] == c], 10))} for c in range(3)}
    cv_desc["Normal ≥120 bpm"] = {"median": float(np.median(d["cv"][(d["y"] == 0) & (d["hr"] >= 120)])),
                                  "p10": float(np.percentile(d["cv"][(d["y"] == 0) & (d["hr"] >= 120)], 10))}
    cv_desc["AFL ≥120 bpm"] = {"median": float(np.median(d["cv"][(d["y"] == 2) & (d["hr"] >= 120)])),
                               "p10": float(np.percentile(d["cv"][(d["y"] == 2) & (d["hr"] >= 120)], 10))}
    return {
        "bpms": BPMS, "jitters": list(JITTERS), "n_synth": N_SYNTH, "labels": OOF_MODELS,
        "stress": {m: {j: t.tolist() for j, t in v.items()} for m, v in stress.items()},
        "threshold": {m: {j: normal_threshold(t) for j, t in v.items()} for m, v in stress.items()},
        "by_hr": by_hr_bin(d, pred),
        "sens_all": sens_all, "sens_wo": sens_wo, "suspect": SUSPECT,
        "nested": nested,
        "rule": product_rule(d, pred["v31"]), "cv_desc": cv_desc, "synthetic_cv": synthetic_cv(),
        "afl_share_suspect": float(((d["rec"] == SUSPECT) & (d["y"] == 2)).sum() / (d["y"] == 2).sum()),
    }
