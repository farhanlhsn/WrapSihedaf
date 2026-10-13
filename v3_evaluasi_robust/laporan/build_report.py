"""Laporan PDF v3 (evaluasi robust), tanpa training ulang.

Dua tahap (sama seperti v2_revisi/laporan/build_report.py):
  1. --extract  dengan Python WSL SIHEDAF (numpy/matplotlib/sklearn): hitung ulang
     semua angka dari cache OOF (fungsi yang sama dengan evaluasi_robust.ipynb)
     lalu simpan tables.json + grafik PNG ke laporan/_build/.
  2. tanpa argumen dengan Python ber-ReportLab: rakit PDF ke
     laporan/laporan_evaluasi_robust.pdf.
Prasyarat: evaluasi_robust.ipynb dan eval_versi_lama.py sudah dijalankan (OOF di ../cache).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # = v3_evaluasi_robust/
REPO = HERE.parent
TMP = HERE / "laporan/_build"
OUT = HERE / "laporan/laporan_evaluasi_robust.pdf"
OOF_DIR = REPO / "cache/oof_robust/detector_e30_r5_s2026"
DATA = REPO / "cache/dataset_v2_detector.npz"
NB = HERE / "evaluasi_robust.ipynb"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402  (Arial → Liberation Sans → DejaVu)
SEED, N_REPEATS, N_BOOT = 2026, 5, 2000
MODELS = ["LightGBM", "MicroTransformer", "MaskedResNet1D", "BiGRUAttention"]
SHORT = {"LightGBM": "LightGBM", "MicroTransformer": "MicroTransformer",
         "MaskedResNet1D": "ResNet1D", "BiGRUAttention": "BiGRU"}
REF = "MicroTransformer +bias"
# layout (warna laporan sebelumnya) + palet data tervalidasi (dataviz validator: semua PASS)
TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
C_ARGMAX, C_BIAS, C_OLD = "#2a78d6", "#eb6834", "#1baf7a"
# Skor yang DILAPORKAN model.ipynb (versi sebelum revisi), skenario B1 MicroTransformer:
# rata-rata per fold, satu partisi StratifiedGroupKFold, data ekstraksi_hrv.py (anotasi 250 Hz).
CLAIM = {"Macro-F1": 0.8257, "Macro-F1 sd": 0.0836, "Normal R": 0.9275, "AFIB P": 0.8517,
         "AFIB R": 0.9445, "AFL P": 0.6144, "AFL R": 0.7633}


# ═════════════════════════════════════════════════════════════════════
# Tahap 1: angka + grafik (WSL)
# ═════════════════════════════════════════════════════════════════════
def extract():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter
    from sklearn.model_selection import StratifiedGroupKFold

    sys.path[:0] = [str(HERE), str(REPO / "v2_revisi")]   # sihedaf_cv, hrv_core
    import sihedaf_cv as cv

    TMP.mkdir(parents=True, exist_ok=True)
    z = np.load(DATA)
    y, records = z["y"], z["records"].astype(str)
    patients = np.unique(records)
    strata = cv.patient_strata(y, records, patients)
    folds = np.load(OOF_DIR / "folds.npz")["folds"]
    assert folds.shape == (N_REPEATS, len(y))

    oof = {m: np.stack([np.load(OOF_DIR / f"{m}_r{r}.npz")["oof"] for r in range(N_REPEATS)]) for m in MODELS}
    fold_time = {m: float(np.mean([np.load(OOF_DIR / f"{m}_r{r}.npz")["times"].mean() for r in range(N_REPEATS)]))
                 for m in MODELS}

    old_grid = np.linspace(-3, 3, 25)
    cms, biases, cms_old = {}, {}, {}
    for m in MODELS:
        for variant in ("argmax", "+bias", "+bias lama"):
            preds, bs = [], []
            for r in range(N_REPEATS):
                prob = oof[m][r]
                if variant != "argmax":
                    grid = old_grid if variant == "+bias lama" else cv.BIAS_GRID
                    adj = np.empty_like(prob)
                    for k in np.unique(folds[r]):
                        b = cv.fit_class_bias(y[folds[r] != k], prob[folds[r] != k], grid=grid)
                        adj[folds[r] == k] = cv.apply_bias(prob[folds[r] == k], b)
                        bs.append(b)
                    prob = adj
                preds.append(prob.argmax(1))
            stack = np.stack([cv.patient_cms(y, p, records, patients) for p in preds])
            if variant == "+bias lama":
                cms_old[m] = stack
            else:
                cms[f"{m} {variant}"] = stack
            if variant == "+bias":
                biases[m] = np.array(bs)

    W = cv.bootstrap_weights(strata, N_BOOT, seed=SEED)
    boot = {k: cv.boot_metrics(c, W) for k, c in cms.items()}
    point = {k: cv.point_metrics(c) for k, c in cms.items()}

    results = []
    for k in cms:
        row = {"key": k, "model": k.rsplit(" ", 1)[0], "variant": k.rsplit(" ", 1)[1],
               "sd_rep": float(point[k]["Macro-F1"].std())}
        for met in cv.METRICS:
            lo, hi = np.percentile(boot[k][met], [2.5, 97.5])
            row[met] = [float(point[k][met].mean()), float(lo), float(hi)]
        results.append(row)

    paired = []
    for k in cms:
        if k == REF:
            continue
        row = {"key": k}
        for met in ("Macro-F1", "AFL F1"):
            d = boot[k][met] - boot[REF][met]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = float(np.mean(d > 0))
            row[met] = [float(point[k][met].mean() - point[REF][met].mean()), float(lo), float(hi), p,
                        "lebih baik" if p >= 0.95 else "lebih buruk" if p <= 0.05 else "tidak terbedakan"]
        paired.append(row)

    grid_cmp = []
    for m in MODELS:
        old, new = cv.point_metrics(cms_old[m]), point[f"{m} +bias"]
        arg = point[f"{m} argmax"]
        grid_cmp.append({"model": m, **{f"{met}|{tag}": float(src[met].mean())
                                        for met in ("Macro-F1", "AFL F1", "AFL P", "AFL R")
                                        for tag, src in (("argmax", arg), ("lama", old), ("baru", new))}})

    afl_pat = patients[strata == 2]
    pi = np.searchsorted(patients, afl_pat)
    afl_recall = {}
    for m in MODELS:
        c = cms[f"{m} +bias"][:, pi]
        rec = c[:, :, 2, 2] / c[:, :, 2, :].sum(-1)
        afl_recall[m] = {"mean": rec.mean(0).tolist(), "min": rec.min(0).tolist(), "max": rec.max(0).tolist()}
    afl_windows = [int(((records == p) & (y == 2)).sum()) for p in afl_pat]
    fp = {m: cms[f"{m} +bias"][:, :, :2, 2].sum(-1).mean(0).tolist() for m in MODELS}

    # partisi lama (StratifiedGroupKFold shuffle) vs baru: pasien AFL per fold
    def afl_by_fold(f):
        return [", ".join(sorted(set(records[(f == k) & (y == 2)]))) or "-" for k in range(5)]
    old_parts = []
    for s in range(5):
        f = np.empty(len(y), dtype=int)
        for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(np.zeros(len(y)), y, records)):
            f[va] = k
        old_parts.append(afl_by_fold(f))
    new_parts = [afl_by_fold(folds[r]) for r in range(N_REPEATS)]

    # cocokkan dengan ringkasan notebook (string "0.832 [..]")
    import csv
    with open(OOF_DIR / "ringkasan.csv", encoding="utf-8") as fh:
        nb_rows = {r["Model"]: r for r in csv.DictReader(fh)}
    for r in results:
        assert nb_rows[r["key"]]["Macro-F1"].startswith(f"{r['Macro-F1'][0]:.3f}"), r["key"]
        assert nb_rows[r["key"]]["AFL F1"].startswith(f"{r['AFL F1'][0]:.3f}"), r["key"]

    payload = {
        "n_windows": int(len(y)), "class_counts": np.bincount(y, minlength=3).tolist(),
        "n_patients": int(len(patients)), "strata_counts": [int((strata == s).sum()) for s in (2, 1, 0)],
        "afl_patients": afl_pat.tolist(), "afl_windows": afl_windows,
        "results": results, "paired": paired, "grid_cmp": grid_cmp,
        "bias_afl": {m: [float(biases[m][:, 2].min()), float(np.median(biases[m][:, 2])), float(biases[m][:, 2].max())]
                     for m in MODELS},
        "bias_grid": [float(cv.BIAS_GRID.min()), float(cv.BIAS_GRID.max())],
        "afl_recall": afl_recall, "fp": fp, "patients": patients.tolist(), "strata": strata.tolist(),
        "fold_time": fold_time, "old_parts": old_parts, "new_parts": new_parts,
    }

    # ── versi sebelum revisi (hasil eval_versi_lama.py) ────────
    from sklearn.metrics import f1_score
    VL = OOF_DIR / "versi_lama"
    lab, fe = np.load(VL / "old_lab.npz"), np.load(VL / "old_frontend.npz")
    oofs = [np.load(VL / f"oof_r{r}.npz") for r in range(N_REPEATS)]
    sets = {"lab": (lab["y"], lab["records"].astype(str)), "det": (y, records),
            "fe": (fe["y"], fe["records"].astype(str))}
    rec2fold = [{p: int(folds[r][records == p][0]) for p in patients} for r in range(N_REPEATS)]
    old_cms, fold_style = {}, {}
    for name, (ys, rs) in sets.items():
        fl = [np.array([rec2fold[r][x] for x in rs]) for r in range(N_REPEATS)]
        old_cms[f"lama {name}"] = np.stack([cv.patient_cms(ys, o[name].argmax(1), rs, patients) for o in oofs])
        # gaya pelaporan model.ipynb: Macro-F1 dirata-rata per fold
        fold_style[name] = [float(np.mean([f1_score(ys[fl[r] == k], o[name].argmax(1)[fl[r] == k],
                                                    average="macro", zero_division=0) for k in range(5)]))
                            for r, o in enumerate(oofs)]
        if name == "det":
            old_cms["lama+bias det"] = np.stack([
                cv.patient_cms(ys, cv.crossfit_bias(ys, o[name], fl[r])[0].argmax(1), rs, patients)
                for r, o in enumerate(oofs)])
    old_cms["revisi argmax det"] = cms["MicroTransformer argmax"]
    old_cms["sekarang det"] = cms[REF]
    ob = {k: cv.boot_metrics(c, W) for k, c in old_cms.items()}
    op = {k: cv.point_metrics(c) for k, c in old_cms.items()}
    old_rows = {k: {met: [float(op[k][met].mean()), *map(float, np.percentile(ob[k][met], [2.5, 97.5]))]
                    for met in cv.METRICS} for k in old_cms}
    old_pairs = {}
    for a, b in (("sekarang det", "lama det"), ("lama+bias det", "lama det"),
                 ("revisi argmax det", "lama det"), ("sekarang det", "lama+bias det")):
        row = {}
        for met in ("Macro-F1", "AFL F1", "AFL P", "AFL R"):
            dd = ob[a][met] - ob[b][met]
            row[met] = [float(op[a][met].mean() - op[b][met].mean()), *map(float, np.percentile(dd, [2.5, 97.5])),
                        float(np.mean(dd > 0))]
        old_pairs[f"{a} vs {b}"] = row
    c_old = old_cms["lama det"][:, pi]
    old_afl_recall = (c_old[:, :, 2, 2] / c_old[:, :, 2, :].sum(-1)).mean(0).tolist()
    import hrv_core
    # window front-end lama yang akan DITOLAK cek kualitas revisi (IBI tidak fisiologis / terlalu sedikit)
    fe_bad = [not hrv_core.window_quality(xx[mm > 0, 0])[0] for xx, mm in zip(fe["X"], fe["M"])]
    payload["old"] = {
        "rows": old_rows, "pairs": old_pairs, "fold_style": fold_style,
        "lab_counts": np.bincount(lab["y"], minlength=3).tolist(),
        "lab_n": int(len(lab["y"])), "fe_n": int(len(fe["y"])),
        "fe_total": int(fe["n_total"].sum()), "fe_fail": int(fe["n_fail"].sum()),
        "fe_bad_quality": float(np.mean(fe_bad)),
        "afl_lab_per_patient": [int(((lab["records"].astype(str) == p) & (lab["y"] == 2)).sum()) for p in afl_pat],
        "afl_recall_det": old_afl_recall,
        "train_time": float(np.mean([o["times"].mean() for o in oofs])),
    }
    (TMP / "tables.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── grafik ─────────────────────────────────────────────────────────
    for style in ("regular", "bold"):
        font_manager.fontManager.addfont(font_path(style))
    family = font_manager.FontProperties(fname=font_path("regular")).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 9.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#C4CED8", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": GRAY, "ytick.color": INK,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

    def save(fig, name):
        fig.savefig(TMP / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=.08)
        plt.close(fig)

    def style(ax):
        ax.grid(axis="x", color="#E6EBF0", lw=.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(comma)

    R = {r["key"]: r for r in results}
    labels = [SHORT[m] for m in MODELS]

    # 1. forest: Macro-F1 & AFL F1, argmax vs +bias
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, met in zip(axes, ("Macro-F1", "AFL F1")):
        for i, m in enumerate(MODELS):
            for dy, v, col, mk in ((-.15, "argmax", C_ARGMAX, "o"), (.15, "+bias", C_BIAS, "s")):
                pt, lo, hi = R[f"{m} {v}"][met]
                ax.plot([lo, hi], [i + dy] * 2, color=col, lw=2, solid_capstyle="round")
                ax.plot(pt, i + dy, mk, color=col, ms=7.5, mec="white", mew=1.3,
                        label=("argmax (tanpa bias)" if v == "argmax" else "+ bias kelas") if i == 0 else None)
        ax.axvline(R[REF][met][0], color="#8A97A6", lw=1, ls="--")
        ax.set_title(met, loc="left", fontsize=10.5, fontweight="bold")
        style(ax)
    axes[0].set_yticks(range(len(MODELS)), labels)
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -.12), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "forest")

    # 2. selisih berpasangan vs referensi
    P = {p["key"]: p for p in paired}
    order = [k for k in (f"{m} {v}" for m in MODELS for v in ("+bias", "argmax")) if k != REF]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, met in zip(axes, ("Macro-F1", "AFL F1")):
        ax.axvline(0, color=INK, lw=1)
        for i, k in enumerate(order):
            d, lo, hi, _, verdict = P[k][met]
            col = C_BIAS if k.endswith("+bias") else C_ARGMAX
            ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
            ax.plot(d, i, "s" if k.endswith("+bias") else "o", ms=7.5, color=col,
                    mfc="white" if verdict == "tidak terbedakan" else col, mec=col, mew=1.6)
        ax.set_title(f"Δ {met} terhadap model produksi", loc="left", fontsize=10.5, fontweight="bold")
        style(ax)
    axes[0].set_yticks(range(len(order)), [f"{SHORT[k.rsplit(' ', 1)[0]]} {'+ bias' if k.endswith('+bias') else 'argmax'}"
                                           for k in order])
    axes[0].invert_yaxis()
    fig.text(.01, -.02, "Titik terisi = lebih buruk secara konsisten; titik kosong = tidak terbedakan. "
             "Garis = CI 95%. Kiri dari garis 0 = lebih buruk dari produksi.", fontsize=8.3, color=GRAY)
    fig.tight_layout()
    save(fig, "paired")

    # 3. efek bias: argmax → grid revisi → grid baru
    G = {g["model"]: g for g in grid_cmp}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, met in zip(axes, ("Macro-F1", "AFL F1")):
        for i, m in enumerate(MODELS):
            vals = [G[m][f"{met}|{t}"] for t in ("argmax", "lama", "baru")]
            ax.plot([min(vals), max(vals)], [i, i], color="#D5DCE3", lw=3, zorder=1)
            for v, col, mk, lab in zip(vals, (C_ARGMAX, C_OLD, C_BIAS), ("o", "D", "s"),
                                       ("argmax", "bias, grid revisi (−3…+3)", "bias, grid baru (−7…+3)")):
                ax.plot(v, i, mk, color=col, ms=7.5, mec="white", mew=1.3, zorder=3, label=lab if i == 0 else None)
        ax.set_title(met, loc="left", fontsize=10.5, fontweight="bold")
        style(ax)
    axes[0].set_yticks(range(len(MODELS)), labels)
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -.14), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "bias_effect")

    # 4. sebaran bias AFL optimal per fold
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    lo_g = cv.BIAS_GRID.min()
    ax.axvspan(lo_g - .3, -3, color="#FBE7DE", zorder=0)
    ax.axvline(-3, color=C_BIAS, lw=1.2, ls="--")
    ax.text(-3.08, -.62, "batas grid revisi (−3)", ha="right", va="center", fontsize=8.5, color=C_BIAS)
    rng = np.random.default_rng(0)
    for i, m in enumerate(MODELS):
        b = biases[m][:, 2]
        ax.scatter(b, i + rng.uniform(-.17, .17, b.size), s=20, color=C_ARGMAX, alpha=.75, edgecolor="white", lw=.5)
    ax.set_yticks(range(len(MODELS)), labels)
    ax.set_ylim(len(MODELS) - .5, -.9)
    ax.set_xlim(lo_g - .3, 1)
    ax.set_xlabel("Bias AFL optimal per fold (log-prob); 25 titik per model")
    style(ax)
    fig.tight_layout()
    save(fig, "bias_strip")

    # 5. heatmap recall AFL per pasien (+bias)
    mat = np.array([payload["old"]["afl_recall_det"]] + [afl_recall[m]["mean"] for m in MODELS]).T
    cmap = LinearSegmentedColormap.from_list("blue", ["#EEF4FC", "#86b6ef", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MODELS) + 1), ["Versi lama\n(argmax)"] + [f"{l}\n+ bias" for l in labels], fontsize=8.8)
    ax.axvline(.5, color="white", lw=4)
    ax.set_yticks(range(len(afl_pat)), [f"{p}   ({n} window AFL)" for p, n in zip(afl_pat, afl_windows)])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=9.5,
                    color="white" if mat[i, j] > .55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
    cb.set_label("Recall AFL (rata-rata 5 ulangan)", fontsize=8.5)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(comma)
    fig.tight_layout()
    save(fig, "afl_heatmap")

    # 6. AFL palsu per pasien, model produksi
    ref_fp = np.array(fp["MicroTransformer"])
    top = np.argsort(-ref_fp)[:8]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for j, i in enumerate(top):
        has_afl = strata[i] == 2
        ax.barh(j, ref_fp[i], color=C_BIAS if has_afl else C_ARGMAX, height=.62)
        ax.text(ref_fp[i] + 2, j, f"{ref_fp[i]:.0f}", va="center", fontsize=9)
    ax.set_yticks(range(len(top)), [patients[i] for i in top])
    ax.invert_yaxis()
    ax.set_xlabel("Window non-AFL yang diprediksi AFL (rata-rata per ulangan) — MicroTransformer + bias")
    ax.barh(0, 0, color=C_BIAS, label="pasien yang punya episode AFL")
    ax.barh(0, 0, color=C_ARGMAX, label="pasien tanpa AFL")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    save(fig, "fp_bar")
    # 7. versi sebelum revisi vs sekarang
    O = payload["old"]["rows"]
    rows_ov = [  # (key, label, warna, marker)
        ("lama lab", "Versi lama · uji data lab", C_ARGMAX, "o"),
        ("lama fe", "Versi lama · front-end lama (end-to-end)", C_ARGMAX, "o"),
        ("lama det", "Versi lama · uji detektor 50 Hz", C_ARGMAX, "o"),
        ("lama+bias det", "Versi lama + bias · uji detektor", C_OLD, "D"),
        ("sekarang det", "Sekarang (revisi + bias) · uji detektor", C_BIAS, "s"),
    ]
    claim = {"Macro-F1": CLAIM["Macro-F1"], "AFL F1": 2 * CLAIM["AFL P"] * CLAIM["AFL R"] / (CLAIM["AFL P"] + CLAIM["AFL R"])}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, met in zip(axes, ("Macro-F1", "AFL F1")):
        ax.plot(claim[met], 0, "D", ms=7.5, mfc="white", mec=INK, mew=1.5)
        for i, (k, _, col, mk) in enumerate(rows_ov, start=1):
            pt, lo, hi = O[k][met]
            ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
            ax.plot(pt, i, mk, color=col, ms=7.5, mec="white", mew=1.3)
        ax.axhline(.5, color="#E0E6EC", lw=.8)
        ax.set_title(met, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlim(0, 1)
        style(ax)
    axes[0].set_yticks(range(len(rows_ov) + 1), ["Klaim model.ipynb (dilaporkan)"] + [r[1] for r in rows_ov])
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "old_vs_new")
    print(f"OK tables.json + 7 grafik → {TMP}")


# ═════════════════════════════════════════════════════════════════════
# Tahap 2: PDF (ReportLab)
# ═════════════════════════════════════════════════════════════════════
def build():
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image

    d = json.loads((TMP / "tables.json").read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for name, style in (("Arial", "regular"), ("Arial-Bold", "bold"), ("Arial-Italic", "italic")):
        pdfmetrics.registerFont(TTFont(name, font_path(style)))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
    S = {
        "body": ParagraphStyle("body", fontName="Arial", fontSize=10.1, leading=14.6, textColor=HexColor(INK)),
        "small": ParagraphStyle("small", fontName="Arial", fontSize=8.7, leading=12, textColor=HexColor(GRAY)),
        "table": ParagraphStyle("table", fontName="Arial", fontSize=8.6, leading=11.5, textColor=HexColor(INK)),
        "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8.5, leading=11.5, textColor=white),
        "box": ParagraphStyle("box", fontName="Arial", fontSize=8.8, leading=11.8, textColor=HexColor(INK)),
    }
    c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
    c.setTitle("Laporan v3 - Evaluasi Robust SIHEDAF")
    c.setAuthor("Analisis v3_evaluasi_robust/evaluasi_robust.ipynb")
    c.setSubject("Langkah 1: fold pasien berulang, CI bootstrap pasien, perbandingan berpasangan, bias kelas")
    W_, H = A4
    M = 40
    CW = W_ - 2 * M
    state = {"page": 0}

    def fnum(v, nd=3):
        return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")

    def thou(v):
        return f"{int(v):,}".replace(",", ".")

    def para(txt, y, style="body", x=M, width=None, gap=9):
        p = Paragraph(txt, S[style])
        _, h = p.wrap(width or CW, H)
        assert y - h >= 51, f"hal {state['page']}: paragraf lewat footer: {txt[:50]}"
        p.drawOn(c, x, y - h)
        return y - h - gap

    def page(title, kicker):
        if state["page"]:
            c.showPage()
        state["page"] += 1
        n = state["page"]
        c.setFillColor(HexColor(TEAL)); c.rect(M, H - 41, 32, 4, fill=1, stroke=0)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial-Bold", 8)
        c.drawString(M + 43, H - 42, "SIHEDAF  /  V3 EVALUASI ROBUST")
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", 21); c.drawString(M, H - 79, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 99, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, "12 Oktober 2026 | Sumber: v3_evaluasi_robust/evaluasi_robust.ipynb, cache/oof_robust/detector_e30_r5_s2026")
        c.drawRightString(W_ - M, 29, f"{n:02d}")
        c.bookmarkPage(f"p{n}"); c.addOutlineEntry(title, f"p{n}", 0, False)
        return H - 121

    def section(title, y):
        c.setFont("Arial-Bold", 12); c.setFillColor(HexColor(INK)); c.drawString(M, y - 12, title)
        return y - 25

    def table(headers, rows, y, widths=None, zebra=True, bold_rows=()):
        widths = widths or [CW / len(headers)] * len(headers)
        cells = [[Paragraph(str(v), S["head"]) for v in headers]]
        cells += [[Paragraph(str(v), S["table"]) for v in row] for row in rows]
        t = Table(cells, colWidths=widths, hAlign="LEFT")
        st = [("BACKGROUND", (0, 0), (-1, 0), HexColor(INK)),
              ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
              ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
              ("LINEBELOW", (0, -1), (-1, -1), .5, HexColor("#D8E1E9"))]
        if zebra:
            st.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F1F5F8")]))
        for r in bold_rows:
            st.append(("BACKGROUND", (0, r), (-1, r), HexColor("#E3F1EF")))
        t.setStyle(TableStyle(st))
        _, h = t.wrap(CW, H)
        assert y - h >= 51, f"hal {state['page']}: tabel lewat footer ({y - h:.0f})"
        t.drawOn(c, M, y - h)
        return y - h - 12

    def chart(name, y, maxh):
        p = TMP / f"{name}.png"
        iw, ih = Image.open(p).size
        w = CW; h = w * ih / iw
        if h > maxh:
            h = maxh; w = h * iw / ih
        assert y - h >= 51, (name, y, h)
        c.drawImage(str(p), M + (CW - w) / 2, y - h, width=w, height=h, mask="auto")
        return y - h - 10

    def callout(title, txt, y, tint="#EDF6F5"):
        p = Paragraph(f"<b>{title}</b><br/>{txt}", S["body"])
        _, h = p.wrap(CW - 24, H)
        assert y - h - 20 >= 51, (title, y, h)
        c.setFillColor(HexColor(tint)); c.roundRect(M, y - h - 20, CW, h + 20, 6, fill=1, stroke=0)
        p.drawOn(c, M + 12, y - h - 10)
        return y - h - 31

    R = {r["key"]: r for r in d["results"]}
    P = {p["key"]: p for p in d["paired"]}
    G = {g["model"]: g for g in d["grid_cmp"]}
    ref = R[REF]
    mt_grid = G["MicroTransformer"]

    def ci(key, met, nd=3):
        pt, lo, hi = R[key][met]
        return f"{fnum(pt, nd)} <font color='{GRAY}' size='7.6'>[{fnum(lo, 2)}–{fnum(hi, 2)}]</font>"

    def pct(v):
        return f"{v * 100:.1f}%".replace(".", ",")

    fp_mt = d["fp"]["MicroTransformer"]
    pats, strata = d["patients"], d["strata"]
    fp_afl_share = sum(v for v, s in zip(fp_mt, strata) if s == 2) / sum(fp_mt)
    fp_non_afl_max = max(v for v, s in zip(fp_mt, strata) if s != 2)
    b_mt = d["bias_afl"]["MicroTransformer"]

    # ── Hal 1: ringkasan ──────────────────────────────────────────────
    y = page("Laporan v3: evaluasi robust SIHEDAF",
             "Langkah 1: memastikan perbedaan antar model itu nyata, bukan kebetulan | data detektor 50 Hz")
    cardw = (CW - 24) / 4
    # angka versi sebelum revisi
    OD = d["old"]
    OR, OP = OD["rows"], OD["pairs"]
    old_det, old_fe, now = OR["lama det"], OR["lama fe"], OR["sekarang det"]
    p_main = OP["sekarang det vs lama det"]
    fs_lab = OD["fold_style"]["lab"]

    def mean_sd(v):
        return f"{fnum(sum(v) / len(v))} ± {fnum((sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** .5)}"

    def verdict(p):
        return "lebih baik secara konsisten" if p >= .95 else "lebih buruk secara konsisten" if p <= .05 else "tidak terbedakan"

    cards = ((fnum(ref["Macro-F1"][0]), "Macro-F1 model produksi*"),
             (fnum(ref["AFL F1"][0]), "AFL F1 model produksi*"),
             (pct(ref["AFL P"][0]), "Precision AFL*"),
             (f"{fnum(old_det['Macro-F1'][0], 2)} → {fnum(now['Macro-F1'][0], 2)}", "Macro-F1 lama → sekarang**"))
    for i, (val, lab) in enumerate(cards):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 88, cardw, 88, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 24 if i < 3 else 18); c.drawString(x + 11, y - 37, val)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 8.7 if i < 3 else 8); c.drawString(x + 11, y - 61, lab)
    y -= 105
    y = para("*MicroTransformer + bias kelas (grid baru), rata-rata 5 ulangan prediksi out-of-fold. Rentang ketidakpastian "
             f"(CI 95%) Macro-F1 {fnum(ref['Macro-F1'][1], 2)}–{fnum(ref['Macro-F1'][2], 2)}, AFL F1 "
             f"{fnum(ref['AFL F1'][1], 2)}–{fnum(ref['AFL F1'][2], 2)}. **Versi sebelum revisi dilatih ulang dengan resep "
             "aslinya pada fold yang sama, diuji pada window yang sama. Ini <b>bukan</b> hasil uji pada perangkat asli.", y, "small")
    y = callout("Jawaban singkat",
                f"Dibanding <b>versi sebelum revisi</b>, sistem sekarang {verdict(p_main['Macro-F1'][3])}: pada data uji yang sama, "
                f"Macro-F1 {fnum(old_det['Macro-F1'][0])} → {fnum(now['Macro-F1'][0])} dan AFL F1 "
                f"{fnum(old_det['AFL F1'][0])} → {fnum(now['AFL F1'][0])}. Jika front-end lama ikut dihitung, versi lama hanya "
                f"Macro-F1 {fnum(old_fe['Macro-F1'][0])}. Di antara empat kandidat model, MicroTransformer + bias tetap yang terbaik; "
                "mengganti arsitektur berbasis interval RR tidak menambah apa-apa. Batas yang tersisa adalah <b>data AFL yang sangat "
                "sedikit</b> (7 pasien).", y)
    y = section("Temuan utama", y)
    d_bigru = P["BiGRUAttention +bias"]["Macro-F1"]
    items = [
        ("1. Revisi memperbaiki versi lama secara nyata.",
         f"Pada window uji yang sama, AFL precision naik {pct(old_det['AFL P'][0])} → {pct(now['AFL P'][0])} "
         f"(P(Δ Macro-F1 > 0) = {fnum(p_main['Macro-F1'][3], 2)}), dengan harga recall AFL turun "
         f"{pct(old_det['AFL R'][0])} → {pct(now['AFL R'][0])}. Sebagian besar kenaikan berasal dari bias kelas. "
         f"Front-end <i>find_peaks</i> lama membuat recall Normal jatuh ke {pct(old_fe['Normal R'][0])}."),
        ("2. Klaim lama hanya berlaku di kondisi lab.",
         f"Macro-F1 0,826 di model.ipynb bisa direproduksi ({mean_sd(fs_lab)}) dengan R-peak anotasi 250 Hz dan cara lapor lama, "
         "tapi tidak di kondisi perangkat 50 Hz."),
        ("3. Model produksi sudah terbaik di antara kandidat.",
         f"Semua varian lain lebih buruk secara konsisten, kecuali BiGRU + bias yang setara "
         f"(selisih Macro-F1 {fnum(d_bigru[0])}). ResNet1D yang unggul di notebook revisi tidak bertahan."),
        ("4. Bias kelas produksi terpotong.",
         f"Bias AFL optimal berada di {fnum(b_mt[0], 1)} … {fnum(b_mt[2], 1)}, selalu di bawah batas grid revisi (−3). "
         f"Dengan grid diperlebar, AFL F1 naik {fnum(mt_grid['AFL F1|lama'])} → {fnum(mt_grid['AFL F1|baru'])} tanpa melatih ulang."),
        ("5. Batasnya di data, bukan di model.",
         "Pasien 07910 dan 08215 jarang terdeteksi AFL oleh versi mana pun, termasuk versi lama (recall ≤ 0,21)."),
    ]
    for t, txt in items:
        y = para(f"<b>{t}</b> {txt}", y, gap=7)
    y -= 4
    para("Panduan baca: sebelum vs sesudah revisi (2–3), kenapa evaluasi dibenahi (4), protokol (5), hasil antar model (6), "
         "perbandingan berpasangan (7), bias kelas (8), AFL per pasien (9), kesimpulan (10), lampiran (11).", y, "small")

    # ── Hal 2: versi lama vs sekarang ─────────────────────────────────
    y = page("Sebelum vs sesudah revisi", "Versi lama dilatih ulang dengan resep aslinya pada fold pasien yang sama, lalu diuji di tiga kondisi")
    y = table(["Versi", "R-peak (front-end)", "Data latih", "Keputusan kelas"], [
        ["<b>Versi lama</b><br/>model.ipynb, final.joblib", "<i>find_peaks</i> polos di backend, tanpa filter dan tanpa cek kualitas",
         "ekstraksi_hrv.py: R-peak anotasi 250 Hz, stride AFL 1 s (overlap 90%)", "argmax"],
        ["<b>Revisi</b><br/>revisi_sihedaf_v2.ipynb", "detektor Pan-Tompkins 50 Hz + cek kualitas window",
         "R-peak detektor 50 Hz, stride 5 s untuk semua kelas", "bias kelas, grid −3…+3"],
        ["<b>Sekarang</b><br/>laporan ini", "sama dengan revisi", "sama dengan revisi", "bias kelas, grid −7…+3"],
    ], y, widths=[105, 140, 170, CW - 415])
    y = chart("old_vs_new", y, 180)
    y = para("Tiga kondisi uji untuk versi lama: <b>data lab</b> = window ekstraksi lama (dipakai model.ipynb untuk melapor skor); "
             "<b>uji detektor 50 Hz</b> = window yang sama persis dengan yang dipakai menguji model sekarang; <b>front-end lama</b> = "
             "sinyal 50 Hz diproses seperti backend lama (<i>find_peaks</i>), yaitu sistem lama secara utuh. Belah ketupat kosong = "
             "skor yang dilaporkan model.ipynb (tanpa CI; AFL F1 dihitung dari precision &amp; recall rata-rata fold).", y, "small")
    y = callout("Intinya",
            f"Pada window uji yang sama, sistem sekarang {verdict(p_main['Macro-F1'][3])} dari versi lama: Macro-F1 "
            f"{fnum(old_det['Macro-F1'][0])} → {fnum(now['Macro-F1'][0])}, AFL F1 {fnum(old_det['AFL F1'][0])} → "
            f"{fnum(now['AFL F1'][0])}, AFL precision {pct(old_det['AFL P'][0])} → {pct(now['AFL P'][0])}. Harganya: recall AFL "
            f"turun {pct(old_det['AFL R'][0])} → {pct(now['AFL R'][0])}, artinya alarm palsu jauh berkurang tetapi lebih banyak "
            "window AFL yang terlewat. Skor tinggi versi lama hanya muncul di kondisi lab. Dengan front-end aslinya, versi lama "
            f"hampir tidak bisa dipakai: Macro-F1 {fnum(old_fe['Macro-F1'][0])}, recall Normal {pct(old_fe['Normal R'][0])}.", y)

    callout("Kenapa front-end lama runtuh?",
            "<i>find_peaks</i> dipakai langsung pada sinyal mentah dengan ambang = rata-rata sinyal dan jarak minimum 0,4 detik "
            "(maksimal 150 bpm). Tanpa filter, gelombang T dan noise ikut terdeteksi atau detak terlewat. Nilai IBI-nya masih terlihat "
            f"wajar (hanya {pct(OD['fe_bad_quality'])} window yang akan ditolak cek kualitas revisi), tetapi urutannya kacau, "
            f"sehingga ritme normal terlihat tidak teratur dan sering ditebak AFIB (recall Normal {pct(old_fe['Normal R'][0])}).", y, "#FFF4EC")

    # ── Hal 3: rincian versi lama ─────────────────────────────────────
    y = page("Rincian perbandingan dengan versi lama", "Semua baris kecuali klaim: rata-rata 5 ulangan × 5 fold pasien, CI 95% bootstrap pasien")

    def orow(label, k):
        r = OR[k]
        return [label] + [f"{fnum(r[m][0])} <font color='{GRAY}' size='7.6'>[{fnum(r[m][1], 2)}–{fnum(r[m][2], 2)}]</font>"
                          for m in ("Macro-F1", "AFL F1", "AFL P", "AFL R")] + [fnum(r["Normal R"][0])]
    claim_afl_f1 = 2 * CLAIM["AFL P"] * CLAIM["AFL R"] / (CLAIM["AFL P"] + CLAIM["AFL R"])
    y = table(["Kondisi", "Macro-F1", "AFL F1", "AFL P", "AFL R", "Normal R"], [
        ["<b>Klaim model.ipynb</b> (rata-rata per fold, satu partisi)", f"{fnum(CLAIM['Macro-F1'])} ± {fnum(CLAIM['Macro-F1 sd'])}",
         f"≈ {fnum(claim_afl_f1)}", fnum(CLAIM["AFL P"]), fnum(CLAIM["AFL R"]), fnum(CLAIM["Normal R"])],
        ["Versi lama · data lab, <b>cara lapor lama</b> (rata-rata per fold)", mean_sd(fs_lab), "–", "–", "–", "–"],
        orow("Versi lama · data lab (pooled)", "lama lab"),
        orow("Versi lama · front-end lama", "lama fe"),
        orow("Versi lama · uji detektor 50 Hz", "lama det"),
        orow("Versi lama + bias · uji detektor", "lama+bias det"),
        orow("Revisi tanpa bias · uji detektor", "revisi argmax det"),
        orow("<b>Sekarang</b> (revisi + bias) · uji detektor", "sekarang det"),
    ], y, widths=[140, 80, 80, 80, 80, CW - 460], bold_rows=[8])
    y = section("Dari mana perbaikannya? (semua pada window uji detektor yang sama)", y)

    def prow(label, key):
        r = OP[key]
        cells = [label]
        for m in ("Macro-F1", "AFL F1", "AFL P"):
            dlt, lo, hi, _ = r[m]
            cells.append(f"{fnum(dlt)} <font color='{GRAY}' size='7.6'>[{fnum(lo, 2)}; {fnum(hi, 2)}]</font>")
        cells.append(fnum(r["Macro-F1"][3], 2))
        return cells
    y = table(["Perubahan dari versi lama", "Δ Macro-F1 [CI]", "Δ AFL F1 [CI]", "Δ AFL P [CI]", "P(Δ Macro-F1>0)"], [
        prow("Hanya menambah bias kelas", "lama+bias det vs lama det"),
        prow("Hanya melatih ulang di data detektor 50 Hz", "revisi argmax det vs lama det"),
        prow("<b>Keduanya</b> (= sistem sekarang)", "sekarang det vs lama det"),
    ], y, widths=[150, 95, 95, 95, CW - 435])
    pb, pr, pn = OP["lama+bias det vs lama det"], OP["revisi argmax det vs lama det"], OP["sekarang det vs lama+bias det"]
    y = callout("Cara membaca tabel di atas",
                f"Sekitar {pb['Macro-F1'][0] / p_main['Macro-F1'][0] * 100:.0f}% kenaikan Macro-F1 berasal dari <b>bias kelas</b> "
                f"(Δ {fnum(pb['Macro-F1'][0])} dari total {fnum(p_main['Macro-F1'][0])}). Melatih ulang di data "
                f"detektor <b>tanpa</b> bias justru sedikit lebih buruk ({fnum(pr['Macro-F1'][0])}, P = {fnum(pr['Macro-F1'][3], 2)}); "
                "kemungkinan karena versi lama memakai window AFL yang jauh lebih rapat (stride 1 detik), yang ternyata membantu "
                f"saat <i>training</i>. Digabung dengan bias, data detektor tetap memberi tambahan kecil tapi konsisten "
                f"({fnum(pn['Macro-F1'][0])} di atas versi lama + bias, P = {fnum(pn['Macro-F1'][3], 2)}).", y)
    lab_c = OD["lab_counts"]
    afl_lab = dict(zip(d["afl_patients"], OD["afl_lab_per_patient"]))
    y = callout("Kenapa klaim lama terlihat tinggi?",
                f"Ekstraksi lama menggeser window AFL tiap 1 detik, sehingga ada {thou(lab_c[2])} window AFL (data detektor: "
                f"{thou(d['class_counts'][2])}) yang 90% isinya tumpang tindih; {thou(afl_lab['04936'])} di antaranya (lebih dari separuh) "
                "berasal dari satu pasien, 04936. R-peak-nya dari anotasi 250 Hz yang bersih, bukan dari sinyal perangkat 50 Hz, "
                "dan skor dirata-rata per fold pada satu partisi. Resepnya sendiri jujur (pasien tidak bocor antar fold), tapi "
                "kondisi ujinya jauh lebih mudah daripada kondisi perangkat.", y)

    # ── Hal 4: kenapa dibenahi ────────────────────────────────────────
    y = page("Kenapa evaluasi perlu dibenahi", "Masalahnya ada di cara mengukur, bukan hanya di modelnya")
    y = para("Versi lama (<i>model.ipynb</i>) dan notebook revisi sama-sama mengevaluasi tiap model <b>satu kali</b> dengan 5 fold "
             "dan satu pembagian pasien. Di notebook revisi, selisih Macro-F1 antar model (0,68–0,76) lebih kecil daripada "
             "naik-turunnya skor antar fold (± 0,10–0,12); di versi lama, Macro-F1 dilaporkan 0,826 ± 0,084. Dari tabel seperti "
             "itu belum bisa disimpulkan model mana yang lebih baik.", y)
    y = section("Masalah yang ditemukan", y)
    y = table(["#", "Masalah", "Perbaikan"], [
        ["1", "Hanya satu pembagian pasien; hasil bisa kebetulan", "5 ulangan × 5 fold, pasien diacak ulang tiap ulangan"],
        ["2", "<b>StratifiedGroupKFold(shuffle=True)</b> menghasilkan pembagian pasien AFL yang <b>sama persis</b> untuk seed berapa pun (lihat tabel di bawah)",
         "Pasien diacak di dalam 3 strata (AFL / dominan AFIB / lainnya), lalu dibagikan bergiliran ke fold"],
        ["3", "Tidak ada rentang ketidakpastian", "Bootstrap tingkat pasien 2.000× → CI 95%"],
        ["4", "Model dibandingkan tanpa pasangan", "Selisih dihitung pada sampel bootstrap yang sama (berpasangan)"],
        ["5", "Tabel utama memakai R-peak anotasi, padahal produksi memakai detektor 50 Hz", "Semua evaluasi di data detektor"],
    ], y, widths=[22, 250, CW - 272])
    y = section("Bukti masalah #2: pasien AFL di fold uji", y)
    rows = [[f"SGKF, seed {s}"] + p for s, p in enumerate(d["old_parts"])]
    rows += [[f"<b>baru, ulangan {r}</b>"] + p for r, p in enumerate(d["new_parts"])]
    y = table(["Pembagian", "Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"], rows, y,
              widths=[95] + [(CW - 95) / 5] * 5)
    callout("Kenapa pembagian SGKF selalu sama?",
            "StratifiedGroupKFold mengurutkan pasien berdasarkan sebaran kelasnya, dan <i>shuffle</i> hanya mengacak pasien "
            "yang sebarannya kembar. Dengan 23 pasien yang sebarannya berbeda-beda, hampir tidak ada yang kembar, sehingga "
            "seed apa pun memberi pembagian yang sama. Mengulang CV dengan seed lain jadi tidak menambah informasi.", y, "#FFF4EC")

    # ── Hal 3: protokol baru ──────────────────────────────────────────
    y = page("Cara kerja protokol baru", "Dari 23 pasien sampai angka dengan rentang ketidakpastian")
    steps = [
        ("1  Strata pasien", f"{d['strata_counts'][0]} punya AFL · {d['strata_counts'][1]} dominan AFIB · {d['strata_counts'][2]} lainnya"),
        ("2  Bagi ke fold", "5 ulangan × 5 fold; pasien diacak per strata, dibagi bergiliran"),
        ("3  Latih & prediksi", "Tiap window diprediksi model yang tidak pernah melihat pasiennya (OOF)"),
        ("4  Bias kelas", "Penggeser ambang dipelajari dari fold lain (cross-fit)"),
        ("5  Bootstrap pasien", "Pasien diambil ulang 2.000× → CI 95% dan selisih berpasangan"),
    ]
    bw, gap = (CW - 4 * 10) / 5, 10
    bh = 104
    for i, (t, txt) in enumerate(steps):
        x = M + i * (bw + gap)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - bh, bw, bh, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.rect(x, y - 4, bw, 4, fill=1, stroke=0)
        p = Paragraph(f"<b>{t}</b><br/><br/>{txt}", S["box"])
        p.wrap(bw - 14, bh)
        p.drawOn(c, x + 7, y - 12 - p.height)
        if i < 4:
            c.setFillColor(HexColor(GRAY))
            ax_ = x + bw + 2
            path = c.beginPath()
            path.moveTo(ax_, y - bh / 2 + 4); path.lineTo(ax_ + 6, y - bh / 2); path.lineTo(ax_, y - bh / 2 - 4); path.close()
            c.drawPath(path, fill=1, stroke=0)
    y -= bh + 18
    cc = d["class_counts"]
    y = para(f"Data: {thou(d['n_windows'])} window 10 detik dari {d['n_patients']} pasien AFDB (Normal {thou(cc[0])}, "
             f"AFIB {thou(cc[1])}, AFL {thou(cc[2])}). R-peak berasal dari detektor 50 Hz, sama dengan kondisi perangkat. Empat model diuji: LightGBM (16 fitur HRV), "
             "MicroTransformer, ResNet1D, dan BiGRU (sekuens 30 interval RR). Hiperparameter sama dengan notebook revisi.", y)
    y = section("Istilah yang dipakai", y)
    y = table(["Istilah", "Arti sederhana"], [
        ["OOF (out-of-fold)", "Prediksi untuk pasien yang tidak ikut dilatih. Ini yang membuat skor jujur."],
        ["Precision AFL", "Dari semua tebakan AFL, berapa persen yang memang AFL. Rendah = banyak alarm palsu."],
        ["Recall AFL", "Dari semua window AFL sebenarnya, berapa persen yang tertangkap. Rendah = banyak AFL terlewat."],
        ["F1", "Rata-rata harmonik precision dan recall; tinggi hanya jika keduanya tinggi."],
        ["Macro-F1", "Rata-rata F1 tiga kelas (Normal, AFIB, AFL) dengan bobot sama, jadi AFL yang langka tetap dihitung penuh."],
        ["Bias kelas", "Angka yang menggeser ambang keputusan tiap kelas setelah model selesai dilatih. Negatif untuk AFL = model harus lebih yakin sebelum menyebut AFL."],
        ["CI 95%", "Rentang tempat skor sebenarnya kemungkinan besar berada. Lebar = masih banyak ketidakpastian."],
        ["Bootstrap pasien", "Mengambil ulang pasien secara acak (boleh terulang) ribuan kali untuk melihat seberapa besar skor berubah jika pasiennya berbeda."],
        ["P(Δ > 0)", "Peluang model lebih baik dari model produksi. ≥ 0,95 = lebih baik; ≤ 0,05 = lebih buruk; selebihnya = tidak terbedakan."],
    ], y, widths=[105, CW - 105])

    # ── Hal 4: hasil utama ────────────────────────────────────────────
    y = page("Hasil utama", "Rata-rata 5 ulangan, CI 95% bootstrap pasien | data detektor | baris hijau = model produksi")
    rows, bold = [], []
    for m in ("MicroTransformer", "BiGRUAttention", "MaskedResNet1D", "LightGBM"):
        for v in ("+bias", "argmax"):
            k = f"{m} {v}"
            if k == REF:
                bold.append(len(rows) + 1)
            rows.append([f"{SHORT[m]} {'+ bias' if v == '+bias' else 'argmax'}", ci(k, "Macro-F1"), ci(k, "AFL F1"),
                         ci(k, "AFL P"), ci(k, "AFL R"), fnum(R[k]["AFIB F1"][0]), fnum(R[k]["Normal R"][0])])
    y = table(["Model", "Macro-F1", "AFL F1", "AFL P", "AFL R", "AFIB<br/>F1", "Normal<br/>R"], rows, y,
              widths=[103, 80, 80, 80, 80, (CW - 423) / 2, (CW - 423) / 2], bold_rows=bold)
    y = chart("forest", y, 215)
    callout("Cara membaca",
            "Titik = skor rata-rata, garis = CI 95%. Garis putus-putus = model produksi. Hampir semua garis AFL F1 sangat "
            "panjang (sekitar 0,2–0,9) karena hanya ada 7 pasien AFL: skor sangat bergantung pada pasien mana yang ikut. "
            "Karena CI tunggal selebar ini, melihat apakah dua garis tumpang tindih <b>tidak cukup</b> untuk membandingkan "
            "model. Halaman berikutnya memakai perbandingan berpasangan yang lebih tajam.", y)

    # ── Hal 5: berpasangan ────────────────────────────────────────────
    y = page("Perbandingan berpasangan", f"Setiap model dibandingkan dengan model produksi ({SHORT['MicroTransformer']} + bias) pada sampel pasien yang sama")
    y = chart("paired", y, 235)
    rows = []
    for k in [f"{m} {v}" for m in ("BiGRUAttention", "MaskedResNet1D", "LightGBM", "MicroTransformer") for v in ("+bias", "argmax")]:
        if k == REF:
            continue
        a, b = P[k]["Macro-F1"], P[k]["AFL F1"]
        m, v = k.rsplit(" ", 1)
        def dci(x):
            return f"{fnum(x[0])} <font color='{GRAY}' size='7.6'>[{fnum(x[1], 2)}; {fnum(x[2], 2)}]</font>"
        rows.append([f"{SHORT[m]} {'+ bias' if v == '+bias' else 'argmax'}",
                     dci(a), fnum(a[3], 2), a[4], dci(b), fnum(b[3], 2), b[4]])
    y = table(["Model", "Δ Macro-F1 [CI 95%]", "P(Δ>0)", "Vonis", "Δ AFL F1 [CI 95%]", "P(Δ>0)", "Vonis"], rows, y,
              widths=[96, 104, 42, 66, 104, 42, CW - 454])
    callout("Kenapa CI tunggal lebar tapi selisihnya tetap jelas?",
            "Pasien yang sulit (misalnya 07910) sulit untuk <b>semua</b> model. Saat dua model dibandingkan pada pasien yang "
            "sama, kesulitan itu saling meniadakan, jadi yang tersisa adalah perbedaan modelnya sendiri. Contohnya, ResNet1D + bias "
            f"lebih rendah {fnum(-P['MaskedResNet1D +bias']['Macro-F1'][0])} Macro-F1 dengan P(Δ>0) = "
            f"{fnum(P['MaskedResNet1D +bias']['Macro-F1'][3], 2)}: kecil, tapi konsisten. BiGRU + bias benar-benar setara "
            f"dengan produksi (P = {fnum(P['BiGRUAttention +bias']['Macro-F1'][3], 2)}).", y)

    # ── Hal 6: bias kelas ─────────────────────────────────────────────
    y = page("Bias kelas: perbaikan termurah", "Pengaruh bias dan lebar grid pencariannya, pada OOF dan fold yang sama")
    y = para("Model DL dilatih dengan <i>weighted cross-entropy</i>: kesalahan pada AFL diberi bobot sekitar 48× karena AFL sangat langka. "
             "Akibatnya model terlalu sering menebak AFL. Bias kelas mengoreksi ini setelah training dengan menggeser ambang keputusan. "
             "Notebook revisi mencari bias di rentang −3…+3.", y)
    y = chart("bias_effect", y, 160)
    y = chart("bias_strip", y, 140)
    y = table(["Model", "Macro-F1 grid revisi", "Macro-F1 grid baru", "AFL F1 grid revisi", "AFL F1 grid baru",
               "AFL P grid revisi", "AFL P grid baru", "Bias AFL (median)"],
              [[SHORT[m]] + [fnum(G[m][f"{met}|{t}"]) for met in ("Macro-F1", "AFL F1", "AFL P") for t in ("lama", "baru")]
               + [fnum(d["bias_afl"][m][1], 2)] for m in ("MicroTransformer", "BiGRUAttention", "MaskedResNet1D", "LightGBM")],
              y, widths=[98] + [(CW - 98) / 7] * 7)
    callout("Artinya untuk model produksi",
            f"Bias AFL optimal MicroTransformer selalu di {fnum(b_mt[0], 1)} … {fnum(b_mt[2], 1)}, sedangkan bias produksi "
            "<b>[0, 2, −3]</b> berhenti tepat di batas grid revisi. Dengan grid diperlebar, precision AFL naik "
            f"{pct(mt_grid['AFL P|lama'])} → {pct(mt_grid['AFL P|baru'])} dan AFL F1 {fnum(mt_grid['AFL F1|lama'])} → "
            f"{fnum(mt_grid['AFL F1|baru'])}. Cukup mengubah <i>class_bias</i> di sihedaf_v2.json; bobot model tetap. "
            "Catatan: untuk ResNet1D dan LightGBM, grid lebar justru sedikit menurunkan skor (bias pilihan fold lain tidak "
            "selalu cocok), jadi manfaatnya berlaku untuk model produksi dan BiGRU, bukan semua model.", y)

    # ── Hal 7: AFL per pasien ─────────────────────────────────────────
    y = page("AFL per pasien", "Uji detektor 50 Hz | angka = recall AFL rata-rata 5 ulangan | kolom pertama = versi sebelum revisi")
    y = chart("afl_heatmap", y, 230)
    y = para("Versi lama (kolom pertama) punya recall AFL lebih tinggi di beberapa pasien (misalnya 06426) karena lebih sering "
             "menebak AFL, dengan precision yang jauh lebih rendah (hal. 3). Di antara model sekarang, hanya pasien <b>08378</b> dan "
             "<b>04908</b> yang terdeteksi baik. <b>07910</b> (96 window) dan <b>08215</b> (9 window) jarang terdeteksi oleh versi "
             "mana pun, dan <b>06426</b> rendah di semua model sekarang. Karena kegagalannya sama di semua arsitektur, penyebabnya kemungkinan besar pola RR pasien itu sendiri (misalnya AFL dengan konduksi teratur "
             "yang mirip ritme normal), bukan pilihan model.", y)
    y = chart("fp_bar", y, 170)
    callout("Dari mana AFL palsu berasal?",
            f"Sekitar <b>{fp_afl_share * 100:.0f}%</b> AFL palsu model produksi muncul pada pasien yang memang punya episode AFL "
            "(terutama 04936 dan 04043), kemungkinan window di sekitar awal/akhir episode. Pada pasien tanpa AFL, alarm palsu "
            f"paling banyak sekitar {fp_non_afl_max:.0f} window dari ±7.000 window per pasien. Pola seperti ini biasanya bisa "
            "dikurangi dengan menghaluskan prediksi antar-window (langkah 2).", y)

    # ── Hal 8: kesimpulan ─────────────────────────────────────────────
    y = page("Kesimpulan dan langkah berikutnya", "Yang sudah pasti, yang perlu diputuskan, dan yang belum bisa dijawab")
    y = section("Keputusan", y)
    y = table(["Hal", "Status", "Alasan"], [
        ["Ganti sistem lama (final.joblib + <i>find_peaks</i>) dengan pipeline revisi", "<b>Disarankan kuat</b>",
         f"Front-end lama membuat Macro-F1 turun ke {fnum(old_fe['Macro-F1'][0])}; di data uji yang sama pun revisi lebih baik"],
        ["Tetap pakai MicroTransformer sebagai model produksi", "Disarankan", "Terbaik bersama BiGRU; BiGRU tidak lebih baik dan latihnya lebih lambat"],
        ["Hentikan pencarian arsitektur RR-only (TCN, transformer lebih besar, dll.)", "Disarankan", "Empat arsitektur gagal di pasien AFL yang sama"],
        ["Kalibrasi ulang <i>class_bias</i> produksi dengan grid −7…+3", "<b>Menunggu persetujuan</b>", "Mengubah file yang dipakai backend (hanya metadata .json)"],
        ["Pakai protokol ini untuk semua eksperimen berikutnya", "Disarankan", "OOF dan fold sudah tersimpan; ide baru cukup dibandingkan berpasangan"],
    ], y, widths=[200, 95, CW - 295])
    y = section("Langkah 2 (berikutnya)", y)
    for t, txt in [
        ("Smoothing antar-window.", "Episode AFL berlangsung menit; menghaluskan prediksi window berurutan menargetkan AFL palsu di sekitar episode."),
        ("Ensemble MicroTransformer + BiGRU.", "Skor keduanya setara tapi pola recall per pasiennya berbeda (mis. 06426)."),
        ("Window AFL lebih rapat saat training saja.", "Versi lama (stride AFL 1 detik) mengalahkan revisi tanpa bias; "
         "kemungkinan kepadatan itu bekerja sebagai augmentasi. Evaluasi tetap pakai stride 5 detik."),
        ("Bobot kelas lebih lunak.", "Misalnya akar dari bobot sekarang, atau logit-adjusted loss, supaya tidak perlu koreksi bias sebesar −4,5."),
        ("Untuk 07910, 08215, 06426:", "cabang input sinyal EKG mentah 50 Hz atau data AFL tambahan, karena info interval RR tampaknya tidak cukup."),
    ]:
        y = para(f"• <b>{t}</b> {txt}", y, gap=5)
    y -= 6
    y = section("Batasan", y)
    for txt in [
        "Semua angka berasal dari 23 pasien AFDB yang di-downsample ke 50 Hz. Noise sensor, posisi elektroda, dan gerak pasien pada perangkat asli belum tersimulasi.",
        "Hanya 7 pasien AFL (1.152 window). CI AFL F1 selebar ±0,7: angka AFL adalah perkiraan kasar.",
        "Bias kelas dipilih dari data yang sama (cross-fit antar fold). Ini jujur di dalam CV, tapi tetap perlu dikonfirmasi pada data perangkat yang dilabeli dokter.",
        "Front-end lama disimulasikan pada lead 1 AFDB yang di-downsample ke 50 Hz; pada sinyal perangkat asli hasilnya bisa lebih baik atau lebih buruk.",
    ]:
        y = para(f"• {txt}", y, gap=5)

    # ── Hal 9: lampiran ───────────────────────────────────────────────
    y = page("Lampiran teknis", "Konfigurasi, waktu komputasi, dan cara mereproduksi")
    t = d["fold_time"]
    y = table(["Komponen", "Nilai"], [
        ["Data", f"cache/dataset_v2_detector.npz — {thou(d['n_windows'])} window, {d['n_patients']} pasien"],
        ["Pembagian", f"5 ulangan × 5 fold pasien, seed {SEED}; fold disimpan di folds.npz"],
        ["Training DL", "30 epoch, batch 256, AdamW lr 1e-3, weight decay 1e-4, cosine, weighted CE (sama dengan notebook revisi)"],
        ["Seed latih", "seed + 100·ulangan + fold"],
        ["Bias kelas", f"grid {fnum(d['bias_grid'][0], 0)}…+{fnum(d['bias_grid'][1], 0)} langkah 0,25; cross-fit antar fold"],
        ["Bootstrap", f"{thou(N_BOOT)} sampel, pasien diresampel per strata"],
        ["Waktu per fold", f"LightGBM {fnum(t['LightGBM'], 1)} s (CPU) · MicroTransformer {fnum(t['MicroTransformer'], 0)} s (GPU) · "
                           f"ResNet1D {fnum(t['MaskedResNet1D'], 0)} s (GPU) · BiGRU {fnum(t['BiGRUAttention'], 0)} s (CPU)"],
        ["Versi lama", f"ekstraksi_hrv.py apa adanya ({thou(OD['lab_n'])} window, sama dengan model.ipynb) + MicroTransformer "
                       f"resep model.ipynb, dilatih ulang pada fold yang sama ({fnum(OD['train_time'], 0)} s/fold, GPU); "
                       f"front-end lama: {thou(OD['fe_n'])} window, {thou(OD['fe_fail'])} gagal (&lt;2 puncak)"],
        ["Total waktu", "sekitar 1 jam 40 menit (kandidat) + 25 menit (versi lama) di WSL Debian, RTX 5060"],
    ], y, widths=[100, CW - 100])
    y = section("File", y)
    y = table(["File", "Isi"], [
        ["v3_evaluasi_robust/evaluasi_robust.ipynb", "Notebook evaluasi lengkap dengan output dan kesimpulan"],
        ["v3_evaluasi_robust/sihedaf_cv.py", "Fold berulang, training, bias kelas, metrik per pasien, bootstrap"],
        ["v3_evaluasi_robust/eval_versi_lama.py", "Ukur ulang versi sebelum revisi (v1_lama); hasil di .../versi_lama/"],
        ["v3_evaluasi_robust/laporan/build_report.py", "Pembuat laporan ini"],
        ["cache/oof_robust/detector_e30_r5_s2026/", "OOF tiap model × ulangan, folds.npz, ringkasan.csv, versi_lama/"],
        ["tools/run_notebook.py", "Runner notebook di WSL (kernel sihedaf-v2)"],
    ], y, widths=[215, CW - 215])
    y = section("Mereproduksi", y)
    para("Semua perintah di WSL dari root repo. Notebook: <font name='Courier' size='8.5'>python tools/run_notebook.py "
         "v3_evaluasi_robust/evaluasi_robust.ipynb</font> (job yang OOF-nya sudah ada dilewati). Versi lama (±25 menit, bisa "
         "dilanjutkan): <font name='Courier' size='8.5'>cd v3_evaluasi_robust &amp;&amp; python eval_versi_lama.py</font>. Laporan: "
         "<font name='Courier' size='8.5'>python v3_evaluasi_robust/laporan/build_report.py --extract</font>, lalu perintah yang sama "
         "tanpa argumen dengan Python yang memiliki ReportLab. Lihat README.md di root.", y)

    c.save()
    print(f"OK {OUT} ({state['page']} halaman)")


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else build()
