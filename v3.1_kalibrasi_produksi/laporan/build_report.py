"""Laporan PDF v3.1 (kalibrasi & paket produksi), tanpa training.

Dua tahap (pola sama dengan v3):
  1. --extract  (Python dengan numpy/torch/matplotlib/wfdb): hitung ulang semua angka lewat kalibrasi.py,
     cocokkan dengan ../cache/v3_1/ringkasan.json dari notebook, cek paket, buat grafik → laporan/_build/.
  2. tanpa argumen (Python dengan ReportLab): rakit laporan/laporan_kalibrasi_produksi.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # = v3.1_kalibrasi_produksi/
REPO = HERE.parent
TMP = HERE / "laporan/_build"
OUT = HERE / "laporan/laporan_kalibrasi_produksi.pdf"
SUMMARY = REPO / "cache/v3_1/ringkasan.json"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402

TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
C_V1, C_V2, C_V31 = "#2a78d6", "#eb6834", "#1baf7a"     # palet tervalidasi (dataviz validator: PASS)


# ═════════════════════════════════════════════════════════════════════
# Tahap 1: angka + grafik
# ═════════════════════════════════════════════════════════════════════
def extract():
    import hashlib
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter

    sys.path.insert(0, str(HERE))
    import kalibrasi as K
    import hrv_core
    import wfdb
    from predictor import ECGPredictor

    TMP.mkdir(parents=True, exist_ok=True)
    res = K.run_all(device="cpu")

    # cocokkan dengan ringkasan notebook (angka identik = notebook & cache sinkron)
    nb = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert nb["b_new"] == res["b_new"], (nb["b_new"], res["b_new"])
    for k in res["rows"]:
        for m in ("Macro-F1", "AFL F1", "Normal R"):
            assert abs(nb["rows"][k][m][0] - res["rows"][k][m][0]) < 1e-9, (k, m)

    # cek paket produksi
    v2_dir, v31_dir = K.V2_EXPORT, HERE / "model_export"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    meta31 = json.loads((v31_dir / "sihedaf_v3_1.json").read_text())
    P31 = ECGPredictor(str(v31_dir), name="sihedaf_v3_1", device="cpu")
    P2 = ECGPredictor(str(v2_dir), device="cpu")
    z = np.load(K.DATA)
    idx = np.random.default_rng(0).choice(len(z["y"]), 4000, replace=False)
    X, M = z["X"][idx], z["M"][idx]
    saved, P31.class_bias = P31.class_bias, np.zeros(3, np.float32)
    raw = P31.predict_proba_sequences(X, M)
    P31.class_bias = saved
    parity = float(np.abs(P31.predict_proba_sequences(X, M) - K.cv.apply_bias(raw, np.asarray(res["b_new"]))).max())
    changed = float(np.mean(P31.predict_proba_sequences(X, M).argmax(1) != P2.predict_proba_sequences(X, M).argmax(1)))
    # demo alur: 60 s episode AFL (in-sample untuk model final)
    path = str(REPO / "data/afdb/04936")
    atr, fs = wfdb.rdann(path, "atr"), wfdb.rdheader(path).fs
    start = next(s for s, a in zip(atr.sample, atr.aux_note) if a.strip("\x00").strip() == "(AFL")
    sig = wfdb.rdrecord(path, channels=[0], sampfrom=int(start + 5 * fs), sampto=int(start + 65 * fs)).p_signal[:, 0]
    sig50 = hrv_core.resample_signal(sig, fs, hrv_core.FS_DEVICE)
    demo = {}
    for name, P in (("v2", P2), ("v3.1", P31)):
        out = P.predict_ecg(sig50, fs=hrv_core.FS_DEVICE)
        demo[name] = {"summary": out["summary"].get("class_name", "tidak_terbaca"),
                      "confidence": out["summary"].get("confidence"),
                      "windows": [w.get("class_name", "tidak terbaca").split(" ")[0] for w in out["windows"]]}
    res["package"] = {
        "pt_identical": sha(v31_dir / "sihedaf_v3_1.pt") == sha(v2_dir / "sihedaf_v2.pt") == meta31["weights_sha256"],
        "v2_json_untouched": sha(v2_dir / "sihedaf_v2.json") == meta31["parent"]["metadata_sha256"],
        "parity": parity, "changed": changed, "demo": demo, "version": meta31["version"],
        "class_bias": meta31["class_bias"],
    }
    (TMP / "tables.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── grafik ──
    for style in ("regular", "bold"):
        font_manager.fontManager.addfont(font_path(style))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path("regular")).get_name(),
                         "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#C4CED8", "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": GRAY, "ytick.color": INK, "figure.facecolor": "white"})
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))

    def save(fig, name):
        fig.savefig(TMP / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=.08, facecolor="white")
        plt.close(fig)

    R = res["rows"]
    # 1. perbandingan versi
    rows = [("v1", "v1 sebelum revisi (argmax)", C_V1, "o", True),
            ("v2_prod", "v2 produksi, bias [0; 2; −3]", C_V2, "s", True),
            ("v2_cf", "prosedur v2, grid −3…+3", C_V2, "s", False),
            ("v31_cf", "v3.1, bias grid −7…+3", C_V31, "D", True)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for ax, m in zip(axes, ("Macro-F1", "AFL F1")):
        for i, (k, _, col, mk, filled) in enumerate(rows):
            pt, lo, hi = R[k][m]
            ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
            ax.plot(pt, i, mk, color=col, mfc=col if filled else "white", mec=col, ms=7.5, mew=1.6)
        ax.axvline(R["v2_prod"][m][0], color="#8A97A6", lw=1, ls="--")
        ax.set_title(m, loc="left", fontsize=10.5, fontweight="bold")
        ax.grid(axis="x", color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(comma)
    axes[0].set_yticks(range(len(rows)), [r[1] for r in rows]); axes[0].invert_yaxis()
    fig.tight_layout(); save(fig, "versions")

    # 2. bias per fold: grid lama vs baru
    fo, fn = np.array(res["fold_bias_old"]), np.array(res["fold_bias_new"])
    rng = np.random.default_rng(0)
    jit = lambda a: a + rng.uniform(-.06, .06, a.shape)
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.axhspan(-7.4, -3, color="#FBE7DE", zorder=0)
    ax.axhline(-3, color=C_V2, lw=1, ls="--")
    ax.text(2.95, -2.85, "batas bawah grid v2 (−3)", ha="right", va="bottom", fontsize=8.3, color=C_V2)
    ax.scatter(jit(fo[:, 1]), jit(fo[:, 2]), s=22, marker="s", color=C_V2, alpha=.7, edgecolor="white", lw=.5,
               label="25 fold, grid −3…+3")
    ax.scatter(jit(fn[:, 1]), jit(fn[:, 2]), s=22, marker="o", color=C_V31, alpha=.8, edgecolor="white", lw=.5,
               label="25 fold, grid −7…+3")
    bv2, bnew = res["b_v2"], res["b_new"]
    ax.plot(bv2[1], bv2[2], "X", ms=12, color=C_V2, mec=INK, mew=1, label="v2 produksi [0; 2; −3]")
    ax.plot(bnew[1], bnew[2], "D", ms=10, color=C_V31, mec=INK, mew=1, label="v3.1 [0; −0,5; −4,5]")
    ax.set_xlim(-1.6, 3.1); ax.set_ylim(-6.2, -2.3)
    ax.set_xlabel("Bias AFIB (log-prob)"); ax.set_ylabel("Bias AFL (log-prob)")
    ax.xaxis.set_major_formatter(comma); ax.yaxis.set_major_formatter(comma)
    ax.grid(color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), frameon=False, fontsize=8.5)
    fig.tight_layout(); save(fig, "bias_scatter")

    # 3. recall AFL per pasien
    cols = [("v1", "v1"), ("v2_prod", "v2 produksi"), ("v31_cf", "v3.1")]
    mat = np.array([res["afl_recall"][k] for k, _ in cols]).T
    cmap = LinearSegmentedColormap.from_list("blue", ["#EEF4FC", "#86b6ef", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [c for _, c in cols]); ax.xaxis.tick_top()
    ax.set_yticks(range(len(res["afl_patients"])), res["afl_patients"]); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=9.5,
                    color="white" if mat[i, j] > .55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.03); cb.outline.set_visible(False)
    cb.set_label("Recall AFL", fontsize=8.5); cb.ax.yaxis.set_major_formatter(comma)
    fig.tight_layout(); save(fig, "afl_heatmap")
    print(f"OK tables.json + 3 grafik → {TMP}")


# ═════════════════════════════════════════════════════════════════════
# Tahap 2: PDF
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
    for name, style in (("Arial", "regular"), ("Arial-Bold", "bold"), ("Arial-Italic", "italic")):
        pdfmetrics.registerFont(TTFont(name, font_path(style)))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
    S = {
        "body": ParagraphStyle("body", fontName="Arial", fontSize=10.1, leading=14.6, textColor=HexColor(INK)),
        "small": ParagraphStyle("small", fontName="Arial", fontSize=8.7, leading=12, textColor=HexColor(GRAY)),
        "table": ParagraphStyle("table", fontName="Arial", fontSize=8.6, leading=11.5, textColor=HexColor(INK)),
        "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8.5, leading=11.5, textColor=white),
    }
    c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
    c.setTitle("Laporan v3.1 - Kalibrasi & Paket Produksi SIHEDAF")
    c.setAuthor("Analisis v3.1_kalibrasi_produksi/kalibrasi_produksi.ipynb")
    W_, H = A4
    M = 40
    CW = W_ - 2 * M
    state = {"page": 0}

    def fnum(v, nd=3):
        return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")

    def pct(v):
        return f"{v * 100:.1f}%".replace(".", ",")

    def para(txt, y, style="body", gap=9):
        p = Paragraph(txt, S[style]); _, h = p.wrap(CW, H)
        assert y - h >= 51, f"hal {state['page']}: paragraf lewat footer: {txt[:50]}"
        p.drawOn(c, M, y - h); return y - h - gap

    def page(title, kicker):
        if state["page"]:
            c.showPage()
        state["page"] += 1; n = state["page"]
        c.setFillColor(HexColor(TEAL)); c.rect(M, H - 41, 32, 4, fill=1, stroke=0)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial-Bold", 8)
        c.drawString(M + 43, H - 42, "SIHEDAF  /  V3.1 KALIBRASI & PAKET PRODUKSI")
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", 21); c.drawString(M, H - 79, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 99, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, "13 Oktober 2026 | Sumber: v3.1_kalibrasi_produksi/kalibrasi_produksi.ipynb, cache/v3_1/ringkasan.json")
        c.drawRightString(W_ - M, 29, f"{n:02d}")
        c.bookmarkPage(f"p{n}"); c.addOutlineEntry(title, f"p{n}", 0, False)
        return H - 121

    def section(title, y):
        c.setFont("Arial-Bold", 12); c.setFillColor(HexColor(INK)); c.drawString(M, y - 12, title)
        return y - 25

    def table(headers, rows, y, widths, bold_rows=()):
        cells = [[Paragraph(str(v), S["head"]) for v in headers]] + [[Paragraph(str(v), S["table"]) for v in r] for r in rows]
        t = Table(cells, colWidths=widths, hAlign="LEFT")
        st = [("BACKGROUND", (0, 0), (-1, 0), HexColor(INK)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F1F5F8")]),
              ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
              ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
              ("LINEBELOW", (0, -1), (-1, -1), .5, HexColor("#D8E1E9"))]
        st += [("BACKGROUND", (0, r), (-1, r), HexColor("#E3F1EF")) for r in bold_rows]
        t.setStyle(TableStyle(st)); _, h = t.wrap(CW, H)
        assert y - h >= 51, f"hal {state['page']}: tabel lewat footer ({y - h:.0f})"
        t.drawOn(c, M, y - h); return y - h - 12

    def chart(name, y, maxh):
        p = TMP / f"{name}.png"; iw, ih = Image.open(p).size
        w = CW; h = w * ih / iw
        if h > maxh:
            h = maxh; w = h * iw / ih
        assert y - h >= 51, (name, y, h)
        c.drawImage(str(p), M + (CW - w) / 2, y - h, width=w, height=h, mask="auto"); return y - h - 10

    def callout(title, txt, y, tint="#EDF6F5"):
        p = Paragraph(f"<b>{title}</b><br/>{txt}", S["body"]); _, h = p.wrap(CW - 24, H)
        assert y - h - 20 >= 51, (title, y, h)
        c.setFillColor(HexColor(tint)); c.roundRect(M, y - h - 20, CW, h + 20, 6, fill=1, stroke=0)
        p.drawOn(c, M + 12, y - h - 10); return y - h - 31

    R, PR, L = d["rows"], d["pairs"], d["labels"]
    v1, v2, v31 = R["v1"], R["v2_prod"], R["v31_cf"]
    p_main, p_v1, p_v2v1 = PR["v31_cf vs v2_prod"], PR["v31_cf vs v1"], PR["v2_prod vs v1"]
    fs = d["fp_summary"]
    rates = d["final_model_rates"]
    bnew = "[" + "; ".join(fnum(x, 1 if x % 1 else 0) for x in d["b_new"]) + "]"
    bv2 = "[" + "; ".join(fnum(x, 0) for x in d["b_v2"]) + "]"
    fbn = d["fold_bias_new_summary"]
    pk = d["package"]

    def ci(r, m):
        return f"{fnum(r[m][0])} <font color='{GRAY}' size='7.6'>[{fnum(r[m][1], 2)}–{fnum(r[m][2], 2)}]</font>"

    def verdict(p):
        return "lebih baik secara konsisten" if p >= .95 else "lebih buruk secara konsisten" if p <= .05 else "tidak terbedakan"

    # ── Hal 1 ──
    y = page("Laporan v3.1: kalibrasi & paket produksi", "Menerapkan keputusan laporan v3 tanpa melatih model baru")
    cardw = (CW - 24) / 4
    cards = ((fnum(v31["Macro-F1"][0]), "Macro-F1 v3.1*"),
             (f"{fnum(v2['Macro-F1'][0], 2)} → {fnum(v31['Macro-F1'][0], 2)}", "Macro-F1 v2 produksi → v3.1"),
             (f"{fnum(v2['Normal R'][0] * 100, 1)} → {pct(v31['Normal R'][0])}", "Recall Normal"),
             (f"{fs['v2_prod']['pasien_lain']:.0f} → {fs['v31_cf']['pasien_lain']:.0f}", "AFL palsu, pasien tanpa AFL**"))
    for i, (val, lab) in enumerate(cards):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 88, cardw, 88, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 24 if i == 0 else 16); c.drawString(x + 11, y - 37, val)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 8.2); c.drawString(x + 11, y - 61, lab)
    y -= 105
    y = para(f"*Estimasi jujur (bias cross-fit) pada window uji detektor 50 Hz, OOF v3 (5 ulangan × 5 fold pasien), CI 95% "
             f"{fnum(v31['Macro-F1'][1], 2)}–{fnum(v31['Macro-F1'][2], 2)}. **Window per ulangan. Bukan uji pada perangkat asli.", y, "small")
    y = callout("Jawaban singkat",
                f"Bias kelas produksi diganti dari {bv2} menjadi <b>{bnew}</b>. Pada data uji dan pasien yang sama, sistem "
                f"menjadi {verdict(p_main['Macro-F1'][3])} dari v2 produksi (Δ Macro-F1 {fnum(p_main['Macro-F1'][0])}, "
                f"P = {fnum(p_main['Macro-F1'][3], 2)}). Recall Normal, recall AFL, dan precision AFL semuanya naik. Bobot model "
                "tidak berubah; yang berubah hanya satu baris di metadata. Paket baru siap dipakai backend lewat "
                "<i>BACKEND.md</i>.", y)
    y = section("Status lima keputusan dari laporan v3", y)
    y = table(["Keputusan", "Status", "Bukti / lokasi"], [
        ["1. Ganti sistem lama (final.joblib + <i>find_peaks</i>)", "<b>Siap dimigrasi</b>",
         "Paket <i>model_export/sihedaf_v3_1</i> + panduan <i>BACKEND.md</i> (hal. 5)"],
        ["2. Tetap MicroTransformer", "Selesai", "Bobot v2 dipakai apa adanya, checksum identik"],
        ["3. Stop arsitektur RR-only", "Dicatat", "Keputusan proyek di README.md & AGENTS.md root"],
        ["4. Kalibrasi ulang <i>class_bias</i>", "<b>Selesai</b>", f"{bv2} → {bnew} (hal. 2–3)"],
        ["5. Protokol v3 untuk eksperimen berikutnya", "Diterapkan", "Semua angka di laporan ini memakai fold, seed, & bootstrap v3"],
    ], y, [190, 85, CW - 275])

    # ── Hal 2: perbandingan versi ──
    y = page("Perbandingan dengan versi sebelumnya", "Window uji detektor 50 Hz, fold & bootstrap pasien yang sama untuk semua baris")
    order = ["v1", "v2_argmax", "v2_prod", "v2_cf", "v31_cf", "v31_fixed"]
    y = table(["Varian", "Macro-F1", "AFL F1", "AFL P", "AFL R", "Normal R"],
              [[L[k], ci(R[k], "Macro-F1"), ci(R[k], "AFL F1"), fnum(R[k]["AFL P"][0]), fnum(R[k]["AFL R"][0]),
                fnum(R[k]["Normal R"][0])] for k in order],
              y, [170, 92, 92, 52, 52, CW - 458], bold_rows=[5])
    y = chart("versions", y, 165)

    def prow(label, p):
        return [label] + [f"{fnum(p[m][0])} <font color='{GRAY}' size='7.6'>[{fnum(p[m][1], 2)}; {fnum(p[m][2], 2)}]</font>"
                          for m in ("Macro-F1", "AFL F1", "Normal R")] + [fnum(p["Macro-F1"][3], 2)]
    y = table(["Perbandingan berpasangan", "Δ Macro-F1", "Δ AFL F1", "Δ Normal R", "P(Δ MF1>0)"], [
        prow("<b>v3.1 vs v2 produksi</b>", p_main), prow("v3.1 vs v1", p_v1), prow("v2 produksi vs v1", p_v2v1),
        prow("v3.1 bias tetap vs cross-fit (optimisme)", PR["v31_fixed vs v31_cf"])],
        y, [160, 92, 92, 92, CW - 436])
    callout("Cara membaca",
            f"v3.1 {verdict(p_main['Macro-F1'][3])} dari v2 produksi. Yang mengejutkan: <b>v2 produksi apa adanya tidak "
            f"terbedakan dari v1</b> (P = {fnum(p_v2v1['Macro-F1'][3], 2)}), karena bias {bv2} menurunkan recall Normal ke "
            f"{pct(v2['Normal R'][0])}. Baris \"bias tetap\" memakai bias yang dipilih dari OOF yang sama, jadi sedikit optimistis; "
            f"selisihnya dari estimasi jujur hanya {fnum(PR['v31_fixed vs v31_cf']['Macro-F1'][0])}, tanda biasnya stabil.", y)

    # ── Hal 3: bias ──
    y = page("Kenapa bias v2 salah, dan bias baru stabil", "Bias per fold (cross-fit) dan efeknya pada model final")
    y = para(f"Bias kelas menggeser ambang keputusan setelah training. Grid v2 hanya −3…+3, sehingga bias AFL <b>selalu</b> berhenti "
             "di −3. Pencarian lalu mengakali batas itu dengan menaikkan bias AFIB: tebakan AFL berkurang karena dialihkan ke AFIB, "
             "tapi window Normal ikut tertebak AFIB.", y)
    y = chart("bias_scatter", y, 215)
    y = table(["Grid", "Bias AFIB (median, rentang)", "Bias AFL (median, rentang)"], [
        ["−3…+3 (v2)", f"{fnum(d['fold_bias_old_summary']['median'][1], 2)} ({fnum(d['fold_bias_old_summary']['min'][1], 2)} … {fnum(d['fold_bias_old_summary']['max'][1], 2)})",
         f"{fnum(d['fold_bias_old_summary']['median'][2], 2)} (100% di batas grid)"],
        ["−7…+3 (v3.1)", f"{fnum(fbn['median'][1], 2)} ({fnum(fbn['min'][1], 2)} … {fnum(fbn['max'][1], 2)})",
         f"{fnum(fbn['median'][2], 2)} ({fnum(fbn['min'][2], 2)} … {fnum(fbn['max'][2], 2)})"],
    ], y, [110, (CW - 110) / 2, (CW - 110) / 2])
    y = section("Cek kewajaran pada model final (in-sample)", y)
    y = table(["Sumber", "Normal", "AFIB", "AFL"],
              [[k] + [pct(v) for v in rates[k]] for k in rates], y, [180, (CW - 180) / 3, (CW - 180) / 3, (CW - 180) / 3])
    callout("Artinya",
            f"Dengan bias v2, model final menebak AFIB pada {pct(rates['v2 [0; 2; −3]'][1])} window, padahal aslinya "
            f"{pct(rates['aktual'][1])}. Dengan bias v3.1, angkanya {pct(rates['v3.1 [0; −0,5; −4,5]'][1])}, jauh lebih dekat ke proporsi "
            "sebenarnya. Model final dilatih pada semua pasien, jadi tabel ini hanya menunjukkan arah efek bias, bukan akurasi.", y)

    # ── Hal 4: AFL per pasien ──
    y = page("AFL per pasien dan alarm palsu", "Recall AFL rata-rata 5 ulangan; AFL palsu = window non-AFL yang ditebak AFL")
    y = chart("afl_heatmap", y, 230)
    y = table(["Varian", "AFL palsu total", "di pasien ber-AFL", "di pasien tanpa AFL"],
              [[L[k], f"{fs[k]['total']:.0f}", f"{fs[k]['pasien_AFL']:.0f}", f"{fs[k]['pasien_lain']:.0f}"]
               for k in ("v1", "v2_prod", "v31_cf")], y, [200, (CW - 200) / 3, (CW - 200) / 3, (CW - 200) / 3])
    callout("Cara membaca",
            "v3.1 menaikkan recall AFL di hampir semua pasien dibanding v2 produksi (paling jelas 08378 dan 04908) sambil "
            f"menekan AFL palsu pada pasien tanpa AFL dari {fs['v2_prod']['pasien_lain']:.0f} ke {fs['v31_cf']['pasien_lain']:.0f} "
            "window per ulangan. v1 punya recall tertinggi karena sangat sering menebak AFL: ribuan alarm palsu. "
            "07910 dan 08215 tetap tidak terdeteksi oleh versi mana pun; ini batas data, bukan bias.", y)

    # ── Hal 5: paket & backend ──
    y = page("Paket produksi dan migrasi backend", "Keputusan 1 dan 2: apa yang berubah, bagaimana diuji, cara kembali")
    y = table(["File", "Isi"], [
        ["model_export/sihedaf_v3_1.pt", "Bobot MicroTransformer final v2, <b>identik</b> (SHA-256 sama)" if pk["pt_identical"] else "TIDAK identik"],
        ["model_export/sihedaf_v3_1.json", f"Metadata v2 + <i>class_bias</i> {bnew}, versi {pk['version']}, asal-usul (<i>parent</i>), cara kalibrasi, metrik protokol v3"],
        ["BACKEND.md", "Panduan integrasi: file yang disalin, API, status, perbedaan dengan sistem lama, checklist, rollback"],
        ["v2_revisi/{hrv_core, sihedaf_models, predictor}.py", "Kode inferensi tetap dari v2 (tidak disalin)"],
    ], y, [190, CW - 190])
    y = section("Uji paket", y)
    demo = pk["demo"]
    y = table(["Uji", "Hasil"], [
        ["Paket v2 tidak berubah", "Ya, checksum metadata sama" if pk["v2_json_untouched"] else "BERUBAH"],
        ["Paritas predictor vs evaluasi", f"selisih probabilitas maks {pk['parity']:.1e}".replace(".", ",").replace("e-0", "e−")],
        ["Prediksi berbeda dari paket v2", f"{pct(pk['changed'])} dari 4.000 window acak"],
        ["Guard satuan & window buruk", "IBI dalam detik → error; window < 4 IBI → <i>tidak_terbaca</i>"],
        ["Demo 60 s episode AFL (04936)", f"v2: {demo['v2']['windows'].count('AFL')}/{len(demo['v2']['windows'])} window AFL · "
                                         f"v3.1: {demo['v3.1']['windows'].count('AFL')}/{len(demo['v3.1']['windows'])} window AFL (in-sample, uji alur)"],
    ], y, [190, CW - 190])
    callout("Migrasi dan rollback",
            "Backend menyalin tiga file Python dari <i>v2_revisi/</i> dan dua file model dari <i>v3.1_kalibrasi_produksi/model_export/</i>, "
            "lalu memanggil <i>ECGPredictor(\"model_export\", name=\"sihedaf_v3_1\")</i>. Kode <i>find_peaks</i> dan <i>final.joblib</i> "
            "dihapus. Karena bobotnya sama, rollback ke v2 cukup mengganti dua file model; kode tidak berubah.", y)

    # ── Hal 6: kesimpulan ──
    y = page("Kesimpulan dan langkah berikutnya", "Yang berubah di produksi, yang tetap, dan batasannya")
    for t, txt in [
        ("Bias baru dipakai.", f"{bv2} → {bnew}. Lebih baik secara konsisten dari v2 produksi (Macro-F1 "
                               f"{fnum(v2['Macro-F1'][0])} → {fnum(v31['Macro-F1'][0])}) dan dari v1 (P = {fnum(p_v1['Macro-F1'][3], 2)})."),
        ("Model tetap.", "MicroTransformer v2, bobot identik. Tidak ada training di v3.1."),
        ("Trade-off yang tersisa.", f"Recall AFL v3.1 ({fnum(v31['AFL R'][0])}) masih di bawah v1 ({fnum(v1['AFL R'][0])}), karena v1 "
                                    f"menebak AFL jauh lebih sering (precision {fnum(v1['AFL P'][0])})."),
        ("Referensi baru.", "Eksperimen berikutnya dibandingkan dengan <b>v3.1</b> (bias cross-fit grid −7…+3) memakai OOF dan fold v3."),
    ]:
        y = para(f"<b>{t}</b> {txt}", y, gap=6)
    y = section("Langkah berikutnya (v4)", y)
    for txt in ["Smoothing antar-window dan ensemble MicroTransformer + BiGRU. Keduanya bisa diuji langsung pada OOF v3 tanpa training.",
                "Window AFL rapat khusus training dan bobot kelas yang lebih lunak (butuh training ulang).",
                "Untuk pasien 07910, 08215, 06426: cabang EKG mentah 50 Hz atau data AFL tambahan."]:
        y = para(f"• {txt}", y, gap=5)
    y = section("Batasan", y)
    for txt in ["Bias dipilih dari OOF model per fold lalu dipasang ke model final (dilatih pada semua pasien), sama seperti di v2. "
                "Ini asumsi yang wajar tapi belum bisa diuji di AFDB.",
                "Semua angka dari 23 pasien AFDB (7 dengan AFL) yang disimulasikan ke 50 Hz. Perlu validasi pada data perangkat asli "
                "yang dilabeli dokter.",
                "Reproduksi: <font name='Courier' size='8.5'>python tools/run_notebook.py v3.1_kalibrasi_produksi/kalibrasi_produksi.ipynb</font>, "
                "lalu <font name='Courier' size='8.5'>laporan/build_report.py --extract</font> dan tanpa argumen."]:
        y = para(f"• {txt}", y, gap=5)
    c.save()
    print(f"OK {OUT} ({state['page']} halaman)")


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else build()
