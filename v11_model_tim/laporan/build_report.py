"""Laporan PDF gabungan v11: enam model tim berbasis RR dibandingkan dengan protokol yang sama (penutupan v11).

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v11/gabungan.json (sel terakhir gabungan.ipynb) dan ringkasan.json tiap setelan, hitung ulang
     Macro-F1 dari OOF di cache dan cocokkan dengan ringkasan (assert), cocokkan selisih berpasangan di gabungan.json dengan
     ringkasan, lalu hitung yang tidak ada di notebook (bobot kelas penuh vs bawaan per model, margin skor AFL, AUROC AFL).
     Angka + grafik disimpan ke laporan/_build/.
  2. tanpa --extract: rakit laporan/laporan_v11_model_tim.pdf dengan ReportLab.
  (--grafik: gambar ulang grafik dari _build/v11.json tanpa menghitung ulang angka.)
Laporan per eksperimen tiap model ada di <model>/laporan/ (BiGRU, TCN, InceptionTime).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # v11_model_tim/laporan
V11 = HERE.parent
REPO = V11.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v11_model_tim.pdf"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402  (Arial → Liberation Sans → DejaVu)

# layout + palet data tervalidasi (validator dataviz: semua PASS; hijau < 3:1 → selalu didampingi tabel)
TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
BIRU, ORANYE, HIJAU = "#2a78d6", "#eb6834", "#1baf7a"
PASIEN_AFL = ["04043", "04908", "04936", "06426", "07910", "08215", "08378"]
TANGGAL = "3 November 2026"

MODEL = ["bigru", "tcn", "inceptiontime", "minirocket", "xgboost", "svm"]
NAMA = {"bigru": "BiGRU", "tcn": "TCN", "inceptiontime": "InceptionTime", "minirocket": "MiniRocket", "xgboost": "XGBoost",
        "svm": "SVM"}
ARSITEKTUR = {"bigru": "BiGRU dua arah + attention", "tcn": "Temporal Convolutional Network (konvolusi dilatasi)",
              "inceptiontime": "InceptionTime (modul Inception, 1 jaringan)",
              "minirocket": "MiniRocket (konvolusi acak) + regresi logistik", "xgboost": "XGBoost (gradient boosting pohon)",
              "svm": "SVM kernel RBF (aproksimasi Nystroem)"}
INPUT = {m: "urutan IBI + ΔIBI" for m in ("bigru", "tcn", "inceptiontime", "minirocket")} | {"xgboost": "16 fitur HRV",
                                                                                               "svm": "16 fitur HRV"}
# Setelan bobot kelas penuh per model dan referensinya (dilatih di mesin yang sama). TCN bawaan dilatih di T4, wp100 di L4,
# jadi pembandingnya bawaan_ulang (L4), sama dengan laporan TCN eksperimen 2.
BOBOT = {"bigru": ("wp100", "bawaan"), "tcn": ("wp100", "bawaan_ulang"), "inceptiontime": ("wp100", "bawaan"),
         "xgboost": ("bobot_penuh", "bawaan")}
# Tempat training (tidak tercatat di cache: ringkasan.json mencatat lingkungan evaluasi). Sumber: README tiap model.
TEMPAT = {"bigru/bawaan": "PC, GPU RTX 5060", "bigru/wp100": "PC, GPU RTX 5060", "tcn/bawaan": "Colab, GPU T4",
          "tcn/bawaan_ulang": "Colab, GPU L4", "tcn/wp100": "Colab, GPU L4", "inceptiontime/bawaan": "PC, GPU RTX 5060",
          "inceptiontime/wp100": "PC, GPU RTX 5060", "minirocket/bawaan": "PC, CPU", "xgboost/bawaan": "PC, CPU",
          "xgboost/bobot_penuh": "PC, CPU", "svm/bawaan": "Colab, CPU High-RAM"}


def ukuran(m, info):
    if "parameter" in info:
        return f"{info['parameter']:,} parameter".replace(",", ".")
    if m == "minirocket":
        return f"{info['fitur']:,} kernel".replace(",", ".")
    if m == "svm":
        return f"{info['komponen']:,} komponen".replace(",", ".")
    return f"{info['pohon']} pohon"


# ═════════════════════════════════════════════════════════════════════
# Tahap 1: angka + grafik
# ═════════════════════════════════════════════════════════════════════
def extract():
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    sys.path[:0] = [str(V11)]
    from bersama import data, evaluasi as E, runner

    akar = data.lokasi_hasil()
    G = json.loads((akar / "gabungan.json").read_text(encoding="utf-8"))
    d = data.muat()
    lp_mask = {p: (d.records == p) & (d.y == 2) for p in ("08378", "04936")}

    def margin(oof):
        """Margin skor AFL = log p(AFL) − log p kelas lain terbesar (sama dengan laporan TCN/InceptionTime)."""
        lp = np.log(np.clip(oof, 1e-9, 1))
        mg = lp[..., 2] - lp[..., :2].max(-1)
        return {**{p: float(np.median(mg[:, m])) for p, m in lp_mask.items()},
                "non_afl_p995": float(np.percentile(mg[:, d.y != 2], 99.5))}

    def rk(key):
        return json.loads((akar / key / "ringkasan.json").read_text(encoding="utf-8"))

    pakai = [f"{m}/bawaan" for m in MODEL] + [f"{m}/{s}" for m, (s, r) in BOBOT.items()] + ["tcn/bawaan_ulang"]
    assert set(G["kandidat"]) <= set(pakai), G["kandidat"]
    assert set(pakai) <= set(G["setelan"]), set(pakai) - set(G["setelan"])
    R, pred = {}, {}
    for key in pakai:
        r = rk(key)
        h = runner.muat_hasil(akar / key, d)
        assert h.ulangan == r["ulangan"] == list(range(data.N_ULANGAN)), key
        pb, bias = E.keputusan(h.oof, d.folds, d.y)
        for nama, p in (("bias", pb), ("argmax", h.oof.argmax(-1))):          # ringkasan = OOF di cache sekarang
            mf = float(np.mean([f1_score(d.y, q, average="macro", labels=[0, 1, 2]) for q in p]))
            assert abs(mf - r["metrik"][nama]["Macro-F1"][0]) < 1e-9, (key, nama)
        # pembanding v1/v3.1 di setiap ringkasan = pembanding di gabungan.json (sampel bootstrap sama)
        for k in ("v31", "v1"):
            assert np.allclose(r["metrik"][k]["Macro-F1"], G["acuan_metrik"][k]["Macro-F1"]), (key, k)
        pred[key] = pb
        bafl = bias[:, :, 2].ravel()
        R[key] = {"bias": r["metrik"]["bias"], "argmax": r["metrik"]["argmax"], "v31": r["banding_v31"],
                  "v1": r["banding_v1"], "afl": r["afl_per_pasien"]["bias"], "afl_argmax": r["afl_per_pasien"]["argmax"],
                  "palsu": r["afl_palsu"]["bias"], "biasstat": r["bias"],
                  "bias_afl_med": float(np.median(bafl)), "bias_afl_batas": int(np.isin(bafl, (-7.0, 3.0)).sum()),
                  "ambang": r["takikardia"]["model"]["ambang"],
                  "pita": {k.split(" | ")[1]: [v["n"], 1 - v["model"][0]] for k, v in r["pita_detak"].items()
                           if k.startswith("Normal")},
                  "waktu": r["waktu_per_fold"], "ukuran": ukuran(key.split("/")[0], r["info"]), "tempat": TEMPAT[key],
                  "info": r["info"], "auroc": float(np.mean([roc_auc_score(d.y == 2, o[:, 2]) for o in h.oof])),
                  "margin": margin(h.oof)}
    # selisih antar model di gabungan.json konsisten dengan ringkasan (prediksi yang sama)
    acuan = G["model_acuan"]
    for m, b in G["antar_model"].items():
        dl = R[f"{m}/bawaan"]["bias"]["Macro-F1"][0] - R[f"{acuan}/bawaan"]["bias"]["Macro-F1"][0]
        assert abs(b["Macro-F1"][0] - dl) < 1e-9, m
    for key, b in G["balancing"].items():
        m = key.split("/")[0]
        dl = rk(key)["metrik"]["bias"]["Macro-F1"][0] - R[f"{m}/bawaan"]["bias"]["Macro-F1"][0]
        assert abs(b["banding"]["Macro-F1"][0] - dl) < 1e-9, key
    bal = {key: {**b, "metrik": rk(key)["metrik"]["bias"], "ambang": rk(key)["takikardia"]["model"]["ambang"],
                 "alarm120": 1 - rk(key)["pita_detak"]["Normal | 120–∞ bpm"]["model"][0], "v31": rk(key)["banding_v31"],
                 "n_latih": rk(key)["info"]["n_latih"]} for key, b in G["balancing"].items()}
    # bobot kelas penuh vs referensi (tidak dihitung di notebook)
    bobot = {m: E.banding(d, pred[f"{m}/{s}"], pred[f"{m}/{r}"]) for m, (s, r) in BOBOT.items()}
    p_v31, _ = E.keputusan(d.oof_v31, d.folds, d.y)
    lp31 = d.oof_v31
    rr = rk("bigru/bawaan")
    payload = {
        "R": R, "acuan": acuan, "antar": G["antar_model"], "balancing": bal, "bobot": bobot, "tuning": G["tuning"],
        "kandidat": G["kandidat"], "tanpa": G["tanpa_04936"], "bersarang": G["bersarang"], "lingkungan": G["lingkungan"],
        "n_setelan": len(G["setelan"]), "setelan_per_model": {m: sum(s.startswith(m + "/") for s in G["setelan"]) for m in MODEL},
        "v31": {"metrik": G["acuan_metrik"]["v31"], "afl": rr["afl_per_pasien"]["v31"], "palsu": rr["afl_palsu"]["v31"],
                "pita": {k.split(" | ")[1]: [v["n"], 1 - v["v31"][0]] for k, v in rr["pita_detak"].items()
                         if k.startswith("Normal")},
                "auroc": float(np.mean([roc_auc_score(d.y == 2, o[:, 2]) for o in lp31])), "margin": margin(lp31)},
        "v1": {"metrik": G["acuan_metrik"]["v1"], "afl": rr["afl_per_pasien"]["v1"], "palsu": rr["afl_palsu"]["v1"]},
        "v31_vs_v1": E.banding(d, p_v31, d.oof_v1.argmax(-1)),
        "paket_v31": rr["takikardia"]["paket_v31"], "kontrol_v7": rr["takikardia"]["kontrol_v7"]["ambang"],
        "n_windows": int(len(d.y)), "class_counts": np.bincount(d.y).tolist(), "n_pasien": int(len(d.patients)),
        "afl_window": {p: int(((d.records == p) & (d.y == 2)).sum()) for p in PASIEN_AFL},
    }
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "v11.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    grafik()


def urut_bawaan(P):
    return sorted(MODEL, key=lambda m: -P["R"][f"{m}/bawaan"]["bias"]["Macro-F1"][0])


def grafik():
    """Grafik dari _build/v11.json saja (tanpa menghitung ulang angka), supaya tampilan bisa diperbaiki cepat."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter

    P = json.loads((TMP / "v11.json").read_text(encoding="utf-8"))
    R, K = P["R"], P["kandidat"]
    for style in ("regular", "bold"):
        font_manager.fontManager.addfont(font_path(style))
    family = font_manager.FontProperties(fname=font_path("regular")).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 9.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#C4CED8", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": GRAY, "ytick.color": INK,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
    pct = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")

    def save(fig, name):
        fig.savefig(TMP / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=.08)
        plt.close(fig)

    def style(ax, axis="x"):
        ax.grid(axis=axis, color="#E6EBF0", lw=.8)
        ax.set_axisbelow(True)
        if axis == "x":
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.xaxis.set_major_formatter(comma)

    def kn(k):
        m, s = k.split("/")
        return f"{NAMA[m]} {s}"

    ub = urut_bawaan(P)
    # 1. forest: v3.1, v1, enam model bawaan (urut), kandidat
    rows = [("v3.1 (produksi)", P["v31"]["metrik"], HIJAU, "D"), ("v1 (argmax)", P["v1"]["metrik"], GRAY, "D")]
    rows += [(f"{NAMA[m]} bawaan", R[f"{m}/bawaan"]["bias"], BIRU, "o") for m in ub]
    rows += [(f"{kn(k)} (kandidat)", R[k]["bias"], ORANYE, "s") for k in K]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, .34 * len(rows) + 1.0), sharey=True)
    for ax, met in zip(axes, ("Macro-F1", "AFL F1")):
        for i, (_, m, col, mk) in enumerate(rows):
            pt, lo, hi = m[met]
            ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round", alpha=.85)
            ax.plot(pt, i, mk, color=col, ms=7, mec="white", mew=1.2)
        ax.axvline(P["v31"]["metrik"][met][0], color=HIJAU, lw=1, ls="--")
        for y in (1.5, 1.5 + len(ub)):
            ax.axhline(y, color="#E0E6EC", lw=.8)
        ax.set_title(met + " (+ bias, CI 95%)", loc="left", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        style(ax)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "forest")

    def paired(baris, judul, nama, metrik=("Macro-F1", "AFL R"), tinggi=.36):
        fig, axes = plt.subplots(1, len(metrik), figsize=(7.2, tinggi * len(baris) + 1.2), sharey=True)
        for ax, met in zip(axes, metrik):
            ax.axvline(0, color=INK, lw=1)
            for i, (_, b) in enumerate(baris):
                dl, lo, hi, p = b[met]
                col, mk, isi = (HIJAU, "^", True) if p >= .95 else (ORANYE, "v", True) if p <= .05 else (BIRU, "o", False)
                ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
                ax.plot(dl, i, mk, ms=8, color=col, mfc=col if isi else "white", mec=col, mew=1.6)
            ax.set_title(f"Δ {met} {judul}", loc="left", fontsize=9.5, fontweight="bold")
            style(ax)
        axes[0].set_yticks(range(len(baris)), [x[0] for x in baris])
        axes[0].invert_yaxis()
        fig.text(.01, -.05 if len(baris) > 3 else -.08, "▲ hijau = lebih baik (P ≥ 0,95)   ▼ oranye = lebih buruk "
                 "(P ≤ 0,05)   ○ biru = tidak terbedakan. Garis = CI 95%.", fontsize=8.6, color=GRAY)
        fig.tight_layout()
        save(fig, nama)

    # 2. antar model (bawaan) vs model acuan
    paired([(NAMA[m], P["antar"][m]) for m in ub if m != P["acuan"]], f"vs {NAMA[P['acuan']]} bawaan", "antar",
           ("Macro-F1", "AFL F1"))
    # 3. balancing vs bawaan, dan bobot penuh vs referensinya
    paired([(f"{NAMA[k.split('/')[0]]} {k.split('/')[1]}", b["banding"]) for k, b in P["balancing"].items()]
           + [(f"{NAMA[m]} {s}", P["bobot"][m]) for m, (s, _) in BOBOT.items()], "vs referensi", "balancing", tinggi=.25)

    # 4. margin: keyakinan AFL di 08378 vs di window non-AFL, referensi → bobot penuh
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    gaya = dict(zip(BOBOT, (("o", BIRU), ("s", ORANYE), ("^", "#7a4cc2"), ("D", GRAY))))
    for m, (s, r) in BOBOT.items():
        a, b = R[f"{m}/{r}"]["margin"], R[f"{m}/{s}"]["margin"]
        mk, col = gaya[m]
        ax.annotate("", xy=(b["non_afl_p995"], b["08378"]), xytext=(a["non_afl_p995"], a["08378"]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
        ax.plot(a["non_afl_p995"], a["08378"], mk, color=col, mfc="white", mec=col, mew=1.6, ms=8)
        ax.plot(b["non_afl_p995"], b["08378"], mk, color=col, ms=8, mec="white", mew=1, label=f"{NAMA[m]}: {r} → {s}")
    v = P["v31"]["margin"]
    ax.plot(v["non_afl_p995"], v["08378"], "*", color=HIJAU, ms=13, mec="white", label="v3.1")
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, color="#C4CED8", lw=1, ls=":")
    ax.set_xlabel("Margin AFL di window non-AFL (persentil 99,5): calon AFL palsu")
    ax.set_ylabel("Margin AFL di 08378 (median)")
    ax.xaxis.set_major_formatter(comma); ax.yaxis.set_major_formatter(comma)
    ax.grid(color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), frameon=False, fontsize=8.6)
    fig.tight_layout()
    save(fig, "margin")

    # 5. heatmap recall AFL per pasien (+ bias)
    pendek = {"inceptiontime": "Inception-\nTime", "minirocket": "Mini-\nRocket"}
    kol = [("v3.1", P["v31"]["afl"]), ("v1", P["v1"]["afl"])] + [(pendek.get(m, NAMA[m]), R[f"{m}/bawaan"]["afl"]) for m in ub]
    kol += [(kn(k).replace(" ", "\n"), R[k]["afl"]) for k in K]
    mat = np.array([[k[1][p] for k in kol] for p in PASIEN_AFL])
    cmap = LinearSegmentedColormap.from_list("biru", ["#EEF4FC", "#86b6ef", BIRU, "#104281"])
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(kol)), [k[0] for k in kol], fontsize=7.8, rotation=0)
    for x in (1.5, 1.5 + len(ub)):
        ax.axvline(x, color="white", lw=4)
    ax.set_yticks(range(len(PASIEN_AFL)), [f"{p}  ({P['afl_window'][p]} window)" for p in PASIEN_AFL])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=7.8,
                    color="white" if mat[i, j] > .55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    cb.set_label("Recall AFL (+ bias, rata-rata 5 ulangan)", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(comma)
    fig.tight_layout()
    save(fig, "afl_heatmap")

    # 6. takikardia: alarm pada window Normal asli per pita detak (110–120 dan ≥ 120 bpm)
    baris = [("v3.1 (produksi)", P["v31"]["pita"])] + [(f"{NAMA[m]} bawaan", R[f"{m}/bawaan"]["pita"]) for m in ub]
    baris += [(f"{kn(k)} (kandidat)", R[k]["pita"]) for k in K]
    fig, ax = plt.subplots(figsize=(7.2, .25 * len(baris) + 1.0))
    for i, (_, p) in enumerate(baris):
        a, b = p["110–120 bpm"][1], p["120–∞ bpm"][1]
        ax.plot([a, b], [i, i], color="#D5DCE3", lw=3, zorder=1)
        ax.plot(a, i, "o", color=BIRU, ms=7, mec="white", mew=1.2, zorder=3, label="Normal 110–120 bpm" if i == 0 else None)
        ax.plot(b, i, "s", color=ORANYE, ms=7, mec="white", mew=1.2, zorder=3, label="Normal ≥ 120 bpm" if i == 0 else None)
    ax.set_yticks(range(len(baris)), [b[0] for b in baris])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel("% window Normal asli (OOF) yang ditebak AFIB/AFL = alarm palsu")
    ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=8.6, bbox_to_anchor=(0, 1.0))
    style(ax)
    ax.xaxis.set_major_formatter(pct)
    fig.tight_layout()
    save(fig, "takikardia")
    print(f"OK {TMP / 'v11.json'} + 6 grafik")


# ═════════════════════════════════════════════════════════════════════
# Tahap 2: PDF (ReportLab)
# ═════════════════════════════════════════════════════════════════════
def fnum(v, nd=3):
    return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")


def sg(v, nd=3):
    return ("+" if v >= 0 else "") + fnum(v, nd) if round(v, nd) != 0 else fnum(abs(v), nd)


def pct(v, nd=1):
    return f"{v * 100:.{nd}f}%".replace(".", ",")


def thou(v):
    return f"{int(v):,}".replace(",", ".")


def vonis(p):
    return "lebih baik" if p >= .95 else "lebih buruk" if p <= .05 else "tidak terbedakan"


def build():
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image

    d = json.loads((TMP / "v11.json").read_text(encoding="utf-8"))
    R, K = d["R"], d["kandidat"]
    ub = urut_bawaan(d)
    for name, style in (("Arial", "regular"), ("Arial-Bold", "bold"), ("Arial-Italic", "italic")):
        pdfmetrics.registerFont(TTFont(name, font_path(style)))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
    S = {
        "body": ParagraphStyle("body", fontName="Arial", fontSize=10, leading=14.3, textColor=HexColor(INK)),
        "small": ParagraphStyle("small", fontName="Arial", fontSize=8.6, leading=11.8, textColor=HexColor(GRAY)),
        "table": ParagraphStyle("table", fontName="Arial", fontSize=8.1, leading=10.6, textColor=HexColor(INK)),
        "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8.0, leading=10.6, textColor=white),
        "box": ParagraphStyle("box", fontName="Arial", fontSize=8.7, leading=11.7, textColor=HexColor(INK)),
    }
    c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
    c.setTitle("v11: enam model tim berbasis RR")
    c.setAuthor("v11_model_tim (pemilik repo dan tim)")
    W_, H = A4
    M = 40
    CW = W_ - 2 * M
    state = {"page": 0}

    def para(txt, y, style="body", x=M, width=None, gap=8):
        p = Paragraph(txt, S[style])
        _, h = p.wrap(width or CW, H)
        assert y - h >= 51, f"hal {state['page']}: paragraf lewat footer: {txt[:60]}"
        p.drawOn(c, x, y - h)
        return y - h - gap

    def page(title, kicker):
        if state["page"]:
            c.showPage()
        state["page"] += 1
        n = state["page"]
        c.setFillColor(HexColor(TEAL)); c.rect(M, H - 41, 32, 4, fill=1, stroke=0)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial-Bold", 8)
        c.drawString(M + 43, H - 42, "SIHEDAF  /  V11  /  ENAM MODEL TIM")
        fs = 20
        while pdfmetrics.stringWidth(title, "Arial-Bold", fs) > CW and fs > 13:
            fs -= .5
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", fs); c.drawString(M, H - 78, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 97, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, f"{TANGGAL} | Sumber: v11_model_tim/gabungan.ipynb, cache/v11/gabungan.json, cache/v11/<model>/<setelan>/")
        c.drawRightString(W_ - M, 29, f"{n:02d}")
        c.bookmarkPage(f"p{n}"); c.addOutlineEntry(title, f"p{n}", 0, False)
        return H - 118

    def section(title, y):
        c.setFont("Arial-Bold", 11.5); c.setFillColor(HexColor(INK)); c.drawString(M, y - 12, title)
        return y - 24

    def table(headers, rows, y, widths=None, bold_rows=()):
        widths = widths or [CW / len(headers)] * len(headers)
        cells = [[Paragraph(str(v), S["head"]) for v in headers]]
        cells += [[Paragraph(str(v), S["table"]) for v in row] for row in rows]
        tb = Table(cells, colWidths=widths, hAlign="LEFT")
        st = [("BACKGROUND", (0, 0), (-1, 0), HexColor(INK)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
              ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
              ("LINEBELOW", (0, -1), (-1, -1), .5, HexColor("#D8E1E9")),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F1F5F8")])]
        for r in bold_rows:
            st.append(("BACKGROUND", (0, r), (-1, r), HexColor("#E3F1EF")))
        tb.setStyle(TableStyle(st))
        _, h = tb.wrap(CW, H)
        assert y - h >= 51, f"hal {state['page']}: tabel lewat footer ({y - h:.0f})"
        tb.drawOn(c, M, y - h)
        return y - h - 11

    def chart(name, y, maxh):
        p = TMP / f"{name}.png"
        iw, ih = Image.open(p).size
        w = CW; h = w * ih / iw
        if h > maxh:
            h = maxh; w = h * iw / ih
        assert y - h >= 51, (name, y, h)
        c.drawImage(str(p), M + (CW - w) / 2, y - h, width=w, height=h, mask="auto")
        return y - h - 9

    def callout(title, txt, y, tint="#EDF6F5"):
        p = Paragraph(f"<b>{title}</b><br/>{txt}", S["body"])
        _, h = p.wrap(CW - 24, H)
        assert y - h - 20 >= 51, (state["page"], title, y, h)
        c.setFillColor(HexColor(tint)); c.roundRect(M, y - h - 20, CW, h + 20, 6, fill=1, stroke=0)
        p.drawOn(c, M + 12, y - h - 10)
        return y - h - 30

    def ci(m, nd=3):
        pt, lo, hi = m
        return f"{fnum(pt, nd)} <font color='{GRAY}' size='7'>[{fnum(lo, 2)}–{fnum(hi, 2)}]</font>"

    def dci(x):
        return f"{sg(x[0])} <font color='{GRAY}' size='7'>[{sg(x[1], 2)}…{sg(x[2], 2)}]</font>"

    def dp(x, nd=3):
        return f"{sg(x[0])} (P = {fnum(x[3], nd)})"

    def dpc(x):                                           # ringkas untuk kolom sempit: Δ (P)
        return f"{sg(x[0])} ({fnum(x[3], 2)})"

    def kn(k):
        m, s = k.split("/")
        return f"{NAMA[m]} {s}"

    def mf(k, met="Macro-F1"):
        return R[k]["bias"][met][0]

    def amb(k):
        return " / ".join(str(v) for v in R[k]["ambang"].values())

    v31, v1 = d["v31"]["metrik"], d["v1"]["metrik"]
    A = d["acuan"]
    best = ub[0]
    a120 = [R[f"{m}/bawaan"]["pita"]["120–∞ bpm"][1] for m in MODEL] + [R[k]["pita"]["120–∞ bpm"][1] for k in K]
    lebih_baik = [m for m in MODEL if R[f"{m}/bawaan"]["v31"]["Macro-F1"][3] >= .95]
    lebih_buruk = [m for m in MODEL if R[f"{m}/bawaan"]["v31"]["Macro-F1"][3] <= .05]
    tanpa, bs = d["tanpa"], d["bersarang"]
    bal = d["balancing"]
    amb_all = [v for k in [f"{m}/bawaan" for m in MODEL] + K for v in R[k]["ambang"].values()]
    amb_min, amb_max = min(amb_all), max(amb_all)
    maks_lambat = max([R[f"{m}/bawaan"]["afl"][p] for m in MODEL for p in ("07910", "08215")]
                      + [R[k]["afl"][p] for k in K for p in ("07910", "08215")] + [d["v31"]["afl"][p] for p in ("07910", "08215")])
    palsu_b = [R[f"{m}/bawaan"]["palsu"]["total"] for m in MODEL]
    min_palsu, max_palsu = min(palsu_b), max(palsu_b)

    # ── Hal 1: ringkasan ──────────────────────────────────────────────
    y = page("v11: enam model tim berbasis RR", f"6 arsitektur | {d['n_setelan']} setelan | 5 ulangan × 5 fold pasien | "
             f"data detektor 50 Hz | {TANGGAL}")
    k0 = K[0]
    cards = [(fnum(mf(f'{best}/bawaan')), f"Macro-F1 terbaik dengan resep sama ({NAMA[best]} bawaan)*"),
             (fnum(mf(k0)), f"{kn(k0)} (kandidat); vs v3.1 P = {fnum(R[k0]['v31']['Macro-F1'][3], 2)}"),
             (f"{len(lebih_baik)} / 6", "model lebih baik dari v3.1 (produksi)"),
             (f"{pct(min(a120), 0)}–{pct(max(a120), 0)}", "alarm palsu pada ritme Normal ≥ 120 bpm (v3.1 "
              f"{pct(d['v31']['pita']['120–∞ bpm'][1], 0)})")]
    cardw = (CW - 24) / 4
    for i, (val, lab) in enumerate(cards):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 84, cardw, 84, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 21); c.drawString(x + 11, y - 36, val)
        p = Paragraph(lab, ParagraphStyle("kartu", fontName="Arial", fontSize=8.1, leading=9.8, textColor=HexColor(GRAY)))
        p.wrap(cardw - 20, 40); p.drawOn(c, x + 11, y - 52 - p.height + 10)
    y -= 100
    y = para("*Semua angka: + bias kelas cross-fit, prediksi out-of-fold (OOF) rata-rata 5 ulangan × 5 fold pasien, data detektor "
             "50 Hz, CI 95% bootstrap pasien 2.000×. Pertanyaan pembimbing: model berbasis RR apa yang berbeda dari v1 dan "
             "apakah ada yang lebih baik? Juga: apakah data perlu di-balance?", y, "small")
    jaw = (f"<b>Tidak ada yang lebih baik dari model produksi v3.1.</b> Dengan resep yang sama, {NAMA[best]} terbaik "
           f"({fnum(mf(f'{best}/bawaan'))}, tidak terbedakan dari v3.1, P = {fnum(R[f'{best}/bawaan']['v31']['Macro-F1'][3], 2)}); "
           f"{len(lebih_buruk)} dari 6 model lebih buruk dari v3.1. Dengan bobot kelas penuh, BiGRU dan TCN naik ke "
           f"{fnum(mf(K[0]))} dan {fnum(mf(K[1]))}, setara v3.1, tetapi tidak melewatinya. Balancing data (arahan "
           "pembimbing) tidak lebih baik daripada bobot kelas. Semua model masih membunyikan alarm pada sebagian besar ritme "
           "Normal yang cepat. <b>Paket produksi tetap v3.1.</b>")
    y = callout("Jawaban singkat", jaw, y)
    y = section("Temuan utama", y)
    ba = bal.get("bigru/under"), bal.get("bigru/over"), bal.get("xgboost/smote"), bal.get("xgboost/under")
    tem = [
        ("1. Dengan resep yang sama, " + NAMA[best] + " terbaik.", "Selisih Macro-F1 terhadapnya: "
         + ", ".join(f"{NAMA[m]} {sg(d['antar'][m]['Macro-F1'][0])} (P = {fnum(d['antar'][m]['Macro-F1'][3], 3)})"
                     for m in ub[1:]) + ". Kemampuan mengurutkan AFL (AUROC) hampir sama; bedanya di keyakinan pada AFL 2:1."),
        ("2. Bobot kelas yang menentukan.", f"Bobot kelas penuh memulihkan AFL 2:1 (pasien 08378) pada BiGRU, TCN, dan "
         f"InceptionTime. BiGRU dan TCN naik ke tingkat v3.1; InceptionTime tidak, karena keyakinan AFL di window non-AFL ikut naik."),
        ("3. Balancing bukan jalan keluar.", f"Terhadap bawaan: BiGRU under {sg(ba[0]['banding']['Macro-F1'][0])} dan over "
         f"{sg(ba[1]['banding']['Macro-F1'][0])} (lebih buruk); XGBoost smote {sg(ba[2]['banding']['Macro-F1'][0])} dan under "
         f"{sg(ba[3]['banding']['Macro-F1'][0])} (lebih baik, tetapi tetap di bawah v3.1, P = "
         f"{fnum(ba[3]['v31']['Macro-F1'][3], 3)})."),
        ("4. Kandidat setara v3.1, bukan lebih baik.", "; ".join(
            f"{kn(k)}: vs v3.1 {sg(R[k]['v31']['Macro-F1'][0])} (P = {fnum(R[k]['v31']['Macro-F1'][3], 2)}), tanpa 04936 "
            f"{fnum(tanpa['rows'][k]['Macro-F1'][0])}, bersarang vs v1 P = {fnum(bs[k]['vs_v1']['p'], 2)}" for k in K) + "."),
        ("5. Takikardia belum terpecahkan.", f"Alarm pada ritme Normal ≥ 120 bpm {pct(min(a120), 0)}–{pct(max(a120), 0)} di "
         f"semua model; ambang uji stres sintetis {amb_min}–{amb_max} bpm. Interval RR saja tidak cukup (sama dengan v7)."),
    ]
    for judul, isi in tem:
        y = para(f"<b>{judul}</b> {isi}", y, gap=5)
    para("Halaman: perbandingan dengan versi sebelumnya (2), metode (3), antar model (4), balancing &amp; bobot kelas (5), "
         "keyakinan AFL (6), per pasien (7), takikardia (8), kandidat (9), kesimpulan &amp; batasan (10), lampiran (11).",
         y - 2, "small")

    # ── Hal 2: perbandingan dengan versi sebelumnya ───────────────────
    y = page("Perbandingan dengan versi sebelumnya", "Data uji, fold pasien, dan sampel bootstrap yang sama | + bias kelas cross-fit")
    rows = [["v1 (versi lama, argmax)", ci(v1["Macro-F1"]), ci(v1["AFL F1"]), fnum(v1["Normal R"][0]), "–",
             "–"],
            ["<b>v3.1 (produksi)</b>", ci(v31["Macro-F1"]), ci(v31["AFL F1"]), fnum(v31["Normal R"][0]), "–",
             dp(d["v31_vs_v1"]["Macro-F1"], 2)]]
    for k in [f"{m}/bawaan" for m in ub] + K:
        nm = f"{NAMA[k.split('/')[0]]} {k.split('/')[1]}" + (" (kandidat)" if k in K else "")
        rows.append([nm, ci(R[k]["bias"]["Macro-F1"]), ci(R[k]["bias"]["AFL F1"]), fnum(R[k]["bias"]["Normal R"][0]),
                     f"{dp(R[k]['v31']['Macro-F1'])} {vonis(R[k]['v31']['Macro-F1'][3])}",
                     dp(R[k]["v1"]["Macro-F1"], 2)])
    y = table(["Sistem", "Macro-F1 [CI]", "AFL F1 [CI]", "Normal R", "Δ Macro-F1 vs v3.1", "Δ vs v1"], rows, y,
              widths=[105, 78, 78, 38, 128, CW - 427], bold_rows=[2])
    y = chart("forest", y, 260)
    para("Belah ketupat hijau = v3.1 (garis putus-putus), abu-abu = v1, lingkaran biru = enam model dengan resep yang sama "
         "(bawaan, urut dari Macro-F1 tertinggi), kotak oranye = kandidat (bobot kelas penuh). Garis horizontal = CI 95%. Lebar CI "
         "AFL F1 berasal dari hanya 7 pasien AFL.", y, "small")

    # ── Hal 3: metode ─────────────────────────────────────────────────
    y = page("Metode", "Enam arsitektur, satu protokol: data, fold, seed, bobot kelas, bias, dan evaluasi sama")
    y = para("Arahan pembimbing: setiap anggota tim mencoba satu model berbasis interval RR yang berbeda dari model v1 (LightGBM, "
             "1D-ResNet, MicroTransformer, RandomForest/HistGB). Supaya adil, semua model memakai kode bersama "
             "(<i>v11_model_tim/bersama/</i>): data dan sidik jari yang sama, 5 ulangan × 5 fold pasien resmi, seed latih "
             "2026 + 100·r + k, bobot kelas lunak, bias kelas cross-fit, dan evaluasi protokol v3 + uji stres takikardia. "
             "Setelan <i>bawaan</i> = resep yang sama untuk semua; setelan lain adalah tuning per model.", y)
    rows = [[NAMA[m], ARSITEKTUR[m], INPUT[m], R[f"{m}/bawaan"]["ukuran"], R[f"{m}/bawaan"]["tempat"],
             f"{R[f'{m}/bawaan']['waktu']:.0f} s", str(d["setelan_per_model"][m])] for m in MODEL]
    y = table(["Model", "Arsitektur", "Input", "Ukuran", "Dilatih di (bawaan)", "s / fold", "Setelan"], rows, y,
              widths=[62, 150, 74, 72, 82, 38, CW - 478])
    y = para("Waktu per fold tidak sebanding antar baris: mesin berbeda, dan sebagian job berbagi GPU/CPU dengan proses lain "
             "(SVM: 5 proses di 8 core). Hasil antar mesin tetap bisa dibandingkan karena OOF, fold, dan sampel bootstrap sama.",
             y, "small")
    steps = [("1  Fold pasien", "5 ulangan × 5 fold resmi (folds.npz); seed latih 2026 + 100·r + k"),
             ("2  Latih & OOF", "Tiap window diprediksi model yang tidak pernah melihat pasiennya"),
             ("3  Bias kelas", "Penggeser ambang dipelajari dari fold lain (cross-fit, grid −7…+3)"),
             ("4  Bootstrap", "Pasien diambil ulang 2.000× → CI 95% dan selisih berpasangan"),
             ("5  Uji stres", "Ritme teratur sintetis 60–170 bpm dengan bobot model yang disimpan")]
    bw, bh = (CW - 40) / 5, 78
    for i, (tt, txt) in enumerate(steps):
        x = M + i * (bw + 10)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - bh, bw, bh, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.rect(x, y - 4, bw, 4, fill=1, stroke=0)
        p = Paragraph(f"<b>{tt}</b><br/>{txt}", S["box"])
        p.wrap(bw - 14, bh); p.drawOn(c, x + 7, y - 12 - p.height)
    y -= bh + 12
    y = section("Istilah", y)
    w_afl = d["n_windows"] / (3 * d["class_counts"][2])
    table(["Istilah", "Arti sederhana"], [
        ["OOF (out-of-fold)", "Prediksi untuk pasien yang tidak ikut dilatih. Ini yang membuat skor jujur."],
        ["Bobot kelas", "Kesalahan pada kelas langka (AFL) dihitung lebih berat saat training. Lunak (bawaan) = AFL "
                        f"±{w_afl ** .5:.0f}×; penuh (wp100) = AFL ±{w_afl:.0f}×."],
        ["Balancing", "Mengubah jumlah window latih per kelas: under = buang mayoritas, over = ulang minoritas, smote = buat "
                      "sintetis. Dijalankan di dalam setiap fold, hanya pada window latih."],
        ["Bias kelas", "Penggeser ambang keputusan setelah training. Negatif untuk AFL = model harus lebih yakin."],
        ["Margin skor AFL", "log p(AFL) − log p kelas lain terbesar, satuannya sama dengan bias."],
        ["P(Δ > 0)", "Peluang sistem lebih baik dari pembanding pada sampel pasien yang sama. ≥ 0,95 lebih baik; ≤ 0,05 lebih "
                     "buruk; selebihnya tidak terbedakan."],
    ], y, widths=[95, CW - 95])

    # ── Hal 4: antar model ────────────────────────────────────────────
    y = page("Antar model dengan resep yang sama", f"Setelan bawaan, berpasangan terhadap model terbaik ({NAMA[A]} bawaan)")
    y = chart("antar", y, 230)
    rows = []
    for m in ub:
        k = f"{m}/bawaan"
        b = d["antar"].get(m)
        rows.append([f"<b>{NAMA[m]}</b>" if m == A else NAMA[m], ci(R[k]["bias"]["Macro-F1"]),
                     "acuan" if b is None else f"{dci(b['Macro-F1'])} {vonis(b['Macro-F1'][3])}",
                     "–" if b is None else dpc(b["AFL F1"]), "–" if b is None else dpc(b["AFL R"]),
                     "–" if b is None else dpc(b["Normal R"]), fnum(R[k]["auroc"])])
    y = table(["Model (bawaan)", "Macro-F1 [CI]", f"Δ Macro-F1 vs {NAMA[A]} [CI]", "Δ AFL F1 (P)", "Δ AFL R (P)",
               "Δ Normal R (P)", "AUROC AFL"], rows, y, widths=[68, 80, 140, 62, 62, 62, CW - 474], bold_rows=[1])
    lemah = ", ".join(NAMA[m] for m in ub[1:] if d["antar"][m]["Macro-F1"][3] <= .05)
    callout("Intinya", f"Dengan resep yang sama, {NAMA[A]} paling baik; {lemah} lebih buruk darinya. AUROC AFL hampir sama di "
            f"semua model ({fnum(min(R[f'{m}/bawaan']['auroc'] for m in MODEL))}–"
            f"{fnum(max(R[f'{m}/bawaan']['auroc'] for m in MODEL))}; v3.1 {fnum(d['v31']['auroc'])}): semua model mengurutkan "
            "window AFL sama baiknya. Bedanya ada di keyakinan pada AFL 2:1 (halaman 6) dan AFL palsu (halaman 7), bukan di "
            "kemampuan dasar arsitekturnya.", y)

    # ── Hal 5: balancing & bobot kelas ────────────────────────────────
    y = page("Balancing data vs bobot kelas", "Arahan pembimbing diuji berpasangan: model sama, fold & bootstrap pasien sama")
    y = chart("balancing", y, 205)
    rows = []
    for k, b in bal.items():
        m, s = k.split("/")
        n = b["n_latih"]
        rows.append([f"{NAMA[m]} {s}", f"{NAMA[m]} bawaan", " / ".join(thou(x) for x in n), ci(b["metrik"]["Macro-F1"]),
                     f"{dci(b['banding']['Macro-F1'])} {vonis(b['banding']['Macro-F1'][3])}", dpc(b["banding"]["AFL R"]),
                     dpc(b["banding"]["Normal R"])])
    for m, (s, r) in BOBOT.items():
        b = d["bobot"][m]
        rows.append([f"{NAMA[m]} {s}", f"{NAMA[m]} {r}", "sama dengan referensi", ci(R[f"{m}/{s}"]["bias"]["Macro-F1"]),
                     f"{dci(b['Macro-F1'])} {vonis(b['Macro-F1'][3])}", dpc(b["AFL R"]), dpc(b["Normal R"])])
    y = table(["Setelan", "Referensi", "Window latih per kelas (r0 k0)", "Macro-F1 [CI]", "Δ Macro-F1 [CI]", "Δ AFL R (P)",
               "Δ Normal R (P)"], rows, y, widths=[80, 82, 72, 78, 115, 44, CW - 471])
    xu = bal["xgboost/under"]
    xs, xb = bal["xgboost/smote"], d["bobot"]["xgboost"]
    callout("Intinya", f"Untuk model urutan (BiGRU), balancing lebih buruk daripada bobot kelas: undersampling hanya "
            f"menyisakan ±1.000 window per kelas ({sg(bal['bigru/under']['banding']['Macro-F1'][0])}), oversampling "
            f"{sg(bal['bigru/over']['banding']['Macro-F1'][0])}, sedangkan bobot penuh {sg(d['bobot']['bigru']['Macro-F1'][0])}. "
            f"Untuk XGBoost, SMOTE ({sg(xs['banding']['Macro-F1'][0])}) sebesar efek bobot penuh ({sg(xb['Macro-F1'][0])}), dan "
            f"undersampling naik lebih banyak ({sg(xu['banding']['Macro-F1'][0])}), tetapi tetap di bawah v3.1 (Macro-F1 "
            f"{fnum(xu['metrik']['Macro-F1'][0])}) dan alarm pada Normal ≥ 120 bpm naik ke {pct(xu['alarm120'], 0)}. Resep "
            "terbaik tetap bobot kelas penuh pada model urutan.", y)

    # ── Hal 6: keyakinan AFL ──────────────────────────────────────────
    y = page("Kenapa bobot kelas penuh berhasil (dan kapan tidak)", "Margin skor AFL: keyakinan pada AFL 2:1 vs keyakinan AFL palsu")
    y = chart("margin", y, 230)
    rows = [["v3.1", "–", fnum(d["v31"]["margin"]["08378"], 2), fnum(d["v31"]["margin"]["non_afl_p995"], 2), "–",
             fnum(d["v31"]["afl"]["08378"], 2), fnum(v31["Macro-F1"][0])]]
    for m, (s, r) in BOBOT.items():
        for key in (f"{m}/{r}", f"{m}/{s}"):
            x = R[key]
            rows.append([kn(key), f"{fnum(x['bias_afl_med'], 2)} ({x['bias_afl_batas']}/25)", fnum(x["margin"]["08378"], 2),
                         fnum(x["margin"]["non_afl_p995"], 2), fnum(x["auroc"]), fnum(x["afl"]["08378"], 2),
                         fnum(x["bias"]["Macro-F1"][0])])
    y = table(["Sistem", "Bias AFL median (di batas −7)", "Margin AFL 08378", "Margin non-AFL p99,5", "AUROC AFL",
               "Recall 08378", "Macro-F1"], rows, y, widths=[110, 95, 66, 72, 54, 56, CW - 453])
    it = R["inceptiontime/wp100"]
    jarak = lambda x: x["margin"]["08378"] - x["margin"]["non_afl_p995"]  # noqa: E731
    callout("Intinya", "AFL hanya ditebak bila margin kira-kira melebihi besarnya bias AFL, dan bias cross-fit dipilih untuk "
            "menekan AFL palsu (window non-AFL dengan margin tinggi). Yang menentukan adalah <b>jarak</b> antara margin 08378 dan "
            f"margin non-AFL (di atas garis titik-titik). Bobot penuh menaikkan keduanya. Pada BiGRU dan TCN jaraknya melebar "
            f"(BiGRU {fnum(jarak(R['bigru/bawaan']), 2)} → {fnum(jarak(R['bigru/wp100']), 2)}, TCN "
            f"{fnum(jarak(R['tcn/bawaan_ulang']), 2)} → {fnum(jarak(R['tcn/wp100']), 2)}; v3.1 {fnum(d['v31']['margin']['08378'] - d['v31']['margin']['non_afl_p995'], 2)}), "
            "jadi 08378 bertahan setelah bias. Pada InceptionTime jaraknya hilang "
            f"({fnum(jarak(R['inceptiontime/bawaan']), 2)} → {fnum(jarak(it), 2)}): bias AFL menempel di −7 pada "
            f"{it['bias_afl_batas']} dari 25 fold, dan Macro-F1 tidak naik. Pada XGBoost (fitur ringkasan) bobot penuh hanya "
            f"sedikit menaikkan 08378 ({fnum(R['xgboost/bawaan']['afl']['08378'], 2)} → {fnum(R['xgboost/bobot_penuh']['afl']['08378'], 2)}).", y)

    # ── Hal 7: per pasien ─────────────────────────────────────────────
    y = page("AFL per pasien dan AFL palsu", "Recall AFL tiap pasien AFL (+ bias), rata-rata 5 ulangan | asal AFL palsu")
    y = chart("afl_heatmap", y, 230)
    rows = [["v3.1"] + [thou(round(d["v31"]["palsu"][k])) for k in ("total", "dari_normal", "dari_afib", "pasien_tanpa_afl")]]
    for k in [f"{m}/bawaan" for m in ub] + K:
        fp = R[k]["palsu"]
        rows.append([kn(k)] + [thou(round(fp[x])) for x in ("total", "dari_normal", "dari_afib", "pasien_tanpa_afl")])
    y = table(["Sistem (+ bias)", "AFL palsu / ulangan", "dari Normal", "dari AFIB", "di pasien tanpa AFL"], rows, y,
              widths=[150, 90, 90, 90, CW - 420])
    callout("Intinya", f"Pasien 08378 ({d['afl_window']['08378']} window, AFL 2:1 cepat dan teratur) adalah pembeda utama: "
            "model yang menangkapnya setelah bias mendekati v3.1. Pasien 07910 dan 08215 (AFL lambat atau tidak teratur) hampir "
            f"tidak terdeteksi oleh v3.1 maupun model v11 mana pun (recall ≤ {fnum(maks_lambat, 2)}), sama dengan temuan v12. "
            f"Semua model bawaan menghasilkan AFL palsu lebih banyak daripada v3.1 ({thou(round(min_palsu))}–"
            f"{thou(round(max_palsu))} vs {thou(round(d['v31']['palsu']['total']))} per ulangan).", y)

    # ── Hal 8: takikardia ─────────────────────────────────────────────
    y = page("Uji stres takikardia", "Protokol butir 9: ritme teratur sintetis dan window Normal asli yang cepat")
    y = chart("takikardia", y, 250)
    pk = list(d["v31"]["pita"])
    rows = [["v3.1 (paket)", str(d["paket_v31"])] + [pct(d["v31"]["pita"][k][1]) for k in pk]]
    rows.append(["kontrol v7 (per fold)", " / ".join(str(v) for v in d["kontrol_v7"].values())] + ["–"] * len(pk))
    for k in [f"{m}/bawaan" for m in ub] + K:
        rows.append([kn(k), amb(k)] + [pct(R[k]["pita"][x][1]) for x in pk])
    y = table(["Sistem", "Ambang (bpm): tanpa / ±10 / G10 / G20"] + [f"Normal {k}" for k in pk], rows, y,
              widths=[110, 120] + [(CW - 230) / len(pk)] * len(pk))
    y = para("Ambang = detak terendah di mana kurang dari separuh window sintetis ditebak Normal (makin tinggi makin baik). "
             "Kolom Normal = % window Normal asli (OOF) pada pita detak itu yang ditebak AFIB/AFL (makin rendah makin baik). "
             f"AFDB hanya punya {thou(d['v31']['pita']['120–∞ bpm'][0])} window Normal ≥ 120 bpm.", y, "small")
    callout("Intinya", f"Tidak ada model RR yang menyelesaikan alarm palsu pada sinus takikardia. Alarm pada Normal ≥ 120 bpm "
            f"{pct(min(a120), 0)}–{pct(max(a120), 0)} (v3.1 {pct(d['v31']['pita']['120–∞ bpm'][1], 0)}). Alarm terendah (TCN dan "
            "InceptionTime bawaan) datang bersama bias AFL yang kuat, yang juga menekan AFL asli (laporan per model). Ambang "
            "sintetis yang tinggi belum tentu berarti alarm rendah pada data asli (MiniRocket). Ini sejalan dengan v7–v10: yang "
            "dibutuhkan adalah rekaman sinus takikardia asli dari perangkat SIHEDAF, bukan arsitektur lain.", y, "#FFF4EC")

    # ── Hal 9: kandidat ───────────────────────────────────────────────
    y = page("Kandidat: bobot kelas penuh (BiGRU dan TCN)", "Protokol butir 10: tanpa pasien 04936 dan bootstrap bersarang")
    rows = [["v3.1 (produksi)", ci(tanpa["rows"]["v31"]["Macro-F1"]), fnum(tanpa["rows"]["v31"]["AFL F1"][0]), "–", "–"]]
    for k in K:
        x = tanpa["pairs_v31_minus"][k]
        rows.append([kn(k), ci(tanpa["rows"][k]["Macro-F1"]), fnum(tanpa["rows"][k]["AFL F1"][0]), dci(x), fnum(x[3], 2)])
    y = section(f"Tanpa pasien 04936 ({tanpa['n_pasien']} pasien, {tanpa['n_pasien_afl']} pasien AFL, "
                f"{thou(tanpa['n_afl'])} window AFL; bias dihitung ulang)", y)
    y = table(["Sistem", "Macro-F1 tanpa 04936", "AFL F1", "v3.1 − sistem [CI]", "P(v3.1 lebih baik)"], rows, y,
              widths=[120, 110, 60, 120, CW - 410])
    rows = []
    for k in K:
        b = bs[k]
        rows.append([kn(k), f"{fnum(b['kandidat']['titik'])} <font color='{GRAY}' size='7'>[{fnum(b['kandidat']['ci'][0], 2)}–"
                     f"{fnum(b['kandidat']['ci'][1], 2)}]</font>", f"{sg(b['vs_v1']['delta'])} <font color='{GRAY}' size='7'>"
                     f"[{sg(b['vs_v1']['ci'][0], 2)}…{sg(b['vs_v1']['ci'][1], 2)}]</font>", fnum(b["vs_v1"]["p"], 3),
                     vonis(b["vs_v1"]["p"])])
    y = section("Bootstrap bersarang vs v1 (bias dipilih ulang di setiap sampel; uji paling ketat)", y)
    y = table(["Kandidat", "Macro-F1 [CI]", "Δ vs v1 [CI]", "P", "Vonis"], rows, y, widths=[120, 110, 120, 60, CW - 410])
    rows = [[kn(k), f"{dp(R[k]['v31']['Macro-F1'], 2)} {vonis(R[k]['v31']['Macro-F1'][3])}", fnum(R[k]["afl"]["08378"], 2),
             amb(k), pct(R[k]["pita"]["120–∞ bpm"][1])] for k in K]
    y = section("Ringkasan kandidat (semua pasien)", y)
    y = table(["Kandidat", "Δ Macro-F1 vs v3.1", "Recall 08378", "Ambang (bpm)", "Alarm Normal ≥ 120"], rows, y,
              widths=[120, 160, 70, 100, CW - 450])
    callout("Intinya", "Kedua kandidat setara v3.1 pada semua pasien dan tanpa 04936, dan lebih baik dari v1 pada bootstrap "
            "bersarang (P = " + " dan ".join(fnum(bs[k]["vs_v1"]["p"], 3) for k in K) + "), seperti v3.1 pada uji yang sama "
            "(laporan v3.2). Tidak ada yang lebih baik dari v3.1, jadi menurut protokol butir 10 paket produksi tidak "
            "diganti. Alarm takikardianya juga tidak lebih rendah secara berarti.", y)

    # ── Hal 10: kesimpulan & batasan ──────────────────────────────────
    y = page("Kesimpulan, langkah berikutnya, dan batasan", "Yang sudah pasti, yang perlu diputuskan, dan yang belum bisa dijawab")
    tun = d["tuning"]
    rows = [["Mengganti paket produksi v3.1", "<b>Tidak</b>", "Tidak ada model atau kandidat yang lebih baik (protokol butir 4 dan 10)"],
            ["Model terbaik dengan resep sama", NAMA[best], f"Macro-F1 {fnum(mf(f'{best}/bawaan'))}; lebih baik dari "
             + ", ".join(NAMA[m] for m in ub[1:] if d['antar'][m]['Macro-F1'][3] <= .05)],
            ["Resep yang disarankan untuk model urutan", "Bobot kelas penuh", "BiGRU dan TCN wp100 setara v3.1; periksa juga "
             "margin non-AFL (InceptionTime)"],
            ["Balancing data", "Tidak disarankan", "Tidak lebih baik dari bobot kelas; under membuang hampir semua data"],
            ["Tuning lanjutan model RR", "Tidak diprioritaskan", "Seleksi cross-fit per model: "
             + ", ".join(f"{NAMA[m]} {fnum(t['metrik']['pilihan']['Macro-F1'][0])}" for m, t in tun.items())
             + f" (v3.1 {fnum(v31['Macro-F1'][0])})"]]
    y = section("Keputusan", y)
    y = table(["Hal", "Status", "Alasan"], rows, y, widths=[150, 100, CW - 250])
    y = section("Langkah berikutnya", y)
    for tt, txt in [("Data perangkat SIHEDAF berlabel.", "Masalah terbesar (alarm pada sinus takikardia, AFL lambat/tidak teratur) "
                     "tidak bisa diselesaikan dengan arsitektur RR lain. Yang dibutuhkan rekaman asli dari perangkat."),
                    ("v13 dengan hipotesis yang dikunci.", "Fitur p_konsisten/kuantisasi dari v12 terlihat memisahkan AFL dari "
                     "AFIB, tetapi dilihat setelah hasil. Uji ulang dengan aturan yang ditetapkan sebelumnya."),
                    ("Anggota tim.", "Hasil per model ada di README dan PDF tiap folder. Setelan baru dinilai dari recall 08378 "
                     "setelah bias, AFL palsu, dan bias di batas grid, bukan Macro-F1 saja.")]:
        y = para(f"• <b>{tt}</b> {txt}", y, gap=5)
    y -= 4
    y = section("Batasan", y)
    for txt in [f"Semua angka dari {d['n_pasien']} pasien AFDB ({len(PASIEN_AFL)} pasien AFL, {thou(d['class_counts'][2])} window "
                f"AFL; {pct(d['afl_window']['04936'] / d['class_counts'][2], 0)} dari 04936). CI AFL F1 sangat lebar; angka AFL adalah "
                "perkiraan kasar.",
                "Setelan selain bawaan dipilih setelah melihat hasil. Perkiraan jujur untuk model yang di-tuning adalah seleksi "
                "cross-fit (tabel di atas); kandidat wp100 ditetapkan dari pelajaran BiGRU sebelum TCN dan InceptionTime dilatih.",
                "Model dilatih di mesin berbeda (PC, Colab, Kaggle) dan sebagian dengan versi pustaka berbeda. Perbandingan tetap "
                "sah karena OOF, fold, dan sampel bootstrap sama, tetapi selisih kecil bisa berasal dari variasi training "
                "(TCN bawaan di dua GPU berbeda).",
                "MiniRocket: regresi logistik belum konvergen di 11 dari 25 fold. SVM, InceptionTime, dan MiniRocket hanya dilatih "
                "dengan sedikit setelan; tuning lebih banyak mungkin memperbaikinya, tetapi kecil kemungkinan melewati v3.1.",
                "AFDB hampir tidak punya sinus takikardia asli; uji stres memakai ritme sintetis."]:
        y = para(f"• {txt}", y, gap=4)

    # ── Hal 11: lampiran ──────────────────────────────────────────────
    y = page("Lampiran teknis", "Lingkungan, file, dan cara mereproduksi")
    lk = d["lingkungan"]
    pu = lk["pustaka"]
    y = table(["Komponen", "Nilai"], [
        ["Data", f"cache/dataset_v2_detector.npz: {thou(d['n_windows'])} window, {d['n_pasien']} pasien (dicek sidik_jari.json); "
                 "di Kaggle/Colab dari GitHub Release v11-data"],
        ["Pembagian", "cache/oof_robust/detector_e30_r5_s2026/folds.npz: 5 ulangan × 5 fold pasien, seed 2026"],
        ["Pembanding", "v3.1 = MicroTransformer + bias (OOF v3); v1 = OOF versi lama, argmax"],
        ["Bias & bootstrap", "grid −7…+3 langkah 0,25, cross-fit antar fold; 2.000 sampel pasien per strata (seed 2026)"],
        ["Training", "Per model, lihat tabel halaman 3 dan README tiap folder. Kode v11 commit b18d6bc + perubahan yang belum "
                     "di-commit."],
        ["Evaluasi gabungan", f"PC; Python {pu['python']}, numpy {pu['numpy']}, scikit-learn {pu['scikit-learn']}, "
                              f"torch {pu['torch']}. gabungan.ipynb lewat tools/run_notebook.py."],
        ["Pemeriksaan otomatis", "Macro-F1 (+ bias & argmax) setelan utama dihitung ulang dari OOF di cache dan harus sama dengan "
                                 "ringkasan.json; pembanding v1/v3.1, selisih antar model, dan selisih balancing di "
                                 "gabungan.json harus sama dengan ringkasan"],
        ["Dihitung di laporan", "Bobot penuh vs referensi per model (halaman 5), margin skor AFL dan AUROC AFL (halaman 4 dan 6): "
                                "dari OOF yang sama lewat bersama.evaluasi"],
    ], y, widths=[100, CW - 100])
    y = section("File", y)
    y = table(["File", "Isi"], [
        ["v11_model_tim/gabungan.ipynb", "Semua setelan, antar model, balancing, tuning cross-fit, kandidat; menulis gabungan.json"],
        ["v11_model_tim/&lt;model&gt;/", "model.py, jalankan.ipynb, README (hasil per model); BiGRU, TCN, InceptionTime juga "
                                         "eksperimen.py + laporan/"],
        ["v11_model_tim/bersama/", "Data + sidik jari, loop DL identik v3, runner 5×5, balancing per fold, evaluasi"],
        ["cache/v11/&lt;model&gt;/&lt;setelan&gt;/", "Bobot dan prediksi per fold, ringkasan.json, setelan.json"],
        ["cache/v11/gabungan.json", "Angka gabungan.ipynb (dibaca laporan ini)"],
        ["v11_model_tim/laporan/build_report.py", "Pembuat laporan ini"],
    ], y, widths=[200, CW - 200])
    y = section("Mereproduksi (dari root repo)", y)
    para("<font name='Courier' size='8.3'>python tools/run_notebook.py v11_model_tim/gabungan.ipynb</font> (±55 menit CPU), "
         "lalu <font name='Courier' size='8.3'>python v11_model_tim/laporan/build_report.py --extract</font> dan perintah yang "
         "sama tanpa --extract. Hasil teman dimasukkan dengan mengekstrak zip hasil (sel terakhir jalankan.ipynb) ke cache/v11/.",
         y)

    c.save()
    print(f"OK {OUT} ({state['page']} halaman)")
    return OUT


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    elif "--grafik" in sys.argv:
        grafik()
    else:
        build()
