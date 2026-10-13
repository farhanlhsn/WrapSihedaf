"""Laporan PDF v5 (dua resep training untuk AFL).

Dua tahap (pola sama dengan v3–v4):
  1. --extract  (Python dengan numpy/torch/matplotlib): hitung ulang angka lewat latih.evaluate() dari OOF di cache,
     cocokkan dengan ../cache/v5/ringkasan.json dari notebook, buat grafik → laporan/_build/.
  2. tanpa argumen (Python dengan ReportLab): rakit laporan/laporan_training_afl.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # = v5_training_afl/
REPO = HERE.parent
TMP = HERE / "laporan/_build"
OUT = HERE / "laporan/laporan_training_afl.pdf"
SUMMARY = REPO / "cache/v5/ringkasan.json"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402

TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
COLOR = {"v1": "#8A97A6", "v31": "#2a78d6", "aflrapat": "#1baf7a", "bobotlunak": "#eb6834"}   # palet tervalidasi + abu netral
ORDER = ["v1", "v31", "aflrapat", "bobotlunak"]
SHORT = {"v1": "v1 (sebelum revisi)", "v31": "v3.1 (referensi)", "aflrapat": "5a: AFL rapat", "bobotlunak": "5b: bobot lunak"}


def extract():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter

    sys.path.insert(0, str(HERE))
    import latih as L

    TMP.mkdir(parents=True, exist_ok=True)
    res = L.evaluate()
    nb = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for k in res["rows"]:
        for m in ("Macro-F1", "AFL F1", "Normal R"):
            assert abs(nb["rows"][k][m][0] - res["rows"][k][m][0]) < 1e-9, (k, m)
    (TMP / "tables.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

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

    R, E = res["rows"], res["episodes"]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.0), sharey=True)
    for ax, (m, src, title) in zip(axes, (("Macro-F1", R, "Macro-F1"), ("AFL F1", R, "AFL F1"),
                                          ("deteksi episode", E, "Deteksi episode AFL"),
                                          ("alarm/jam", E, "Alarm palsu/jam ↓"))):
        for i, k in enumerate(ORDER):
            pt, lo, hi = src[k][m][:3]
            ax.plot([lo, hi], [i, i], color=COLOR[k], lw=2, solid_capstyle="round")
            ax.plot(pt, i, "o", color=COLOR[k], ms=6.5, mec="white", mew=1.2)
        ax.axvline(src["v31"][m][0], color="#8A97A6", lw=1, ls="--")
        ax.set_title(title, loc="left", fontsize=9.3, fontweight="bold")
        ax.grid(axis="x", color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); ax.xaxis.set_major_formatter(comma)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_yticks(range(len(ORDER)), [SHORT[k] for k in ORDER]); axes[0].invert_yaxis()
    fig.tight_layout(); save(fig, "forest")

    mat = np.array([res["afl_recall"][k] for k in ORDER]).T
    cmap = LinearSegmentedColormap.from_list("blue", ["#EEF4FC", "#86b6ef", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ORDER)), [SHORT[k] for k in ORDER], fontsize=8.6); ax.xaxis.tick_top()
    ax.set_yticks(range(len(res["afl_patients"])), res["afl_patients"]); ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=9.3,
                    color="white" if mat[i, j] > .55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03); cb.outline.set_visible(False)
    cb.set_label("Recall AFL", fontsize=8.5); cb.ax.yaxis.set_major_formatter(comma)
    fig.tight_layout(); save(fig, "afl_heatmap")
    print(f"OK tables.json + 2 grafik → {TMP}")


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
        "table": ParagraphStyle("table", fontName="Arial", fontSize=8.4, leading=11.2, textColor=HexColor(INK)),
        "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8.3, leading=11.2, textColor=white),
    }
    c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
    c.setTitle("Laporan v5 - Dua resep training untuk AFL")
    c.setAuthor("Analisis v5_training_afl/training_afl.ipynb")
    W_, H = A4
    M = 40
    CW = W_ - 2 * M
    state = {"page": 0}

    def fnum(v, nd=3):
        return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")

    def sgn(v, nd=3):
        return ("+" if v >= 0 else "") + fnum(v, nd)

    def pct(v):
        return f"{v * 100:.0f}%"

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
        c.drawString(M + 43, H - 42, "SIHEDAF  /  V5 RESEP TRAINING UNTUK AFL")
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", 21); c.drawString(M, H - 79, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 99, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, "13 Oktober 2026 | Sumber: v5_training_afl/training_afl.ipynb, cache/v5/, cache/oof_robust/")
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
              ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
              ("TOPPADDING", (0, 0), (-1, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
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

    R, E, P, L, B = d["rows"], d["episodes"], d["pairs"], d["labels"], d["bias"]
    A, AP, SN = d["argmax_rows"], d["argmax_pairs"], d["bias_sensitivity"]
    pa, pb = P["aflrapat"], P["bobotlunak"]

    def verdict(p):
        return "lebih baik" if p >= .95 else "lebih buruk" if p <= .05 else "tidak terbedakan"

    def ci(v, nd=3):
        return f"{fnum(v[0], nd)} <font color='{GRAY}' size='7.4'>[{fnum(v[1], 2)}–{fnum(v[2], 2)}]</font>"

    def dci(v, nd=3):
        return f"{sgn(v[0], nd)}<br/><font color='{GRAY}' size='7.2'>[{fnum(v[1], 2)}; {fnum(v[2], 2)}]</font>"

    # ── Hal 1 ──
    y = page("Laporan v5: dua resep training untuk AFL", "MicroTransformer dilatih ulang 2 × 25 kali pada fold yang sama dengan referensi v3.1")
    cardw = (CW - 24) / 4
    cards = ((sgn(pa["Macro-F1"][0]), "Δ Macro-F1, 5a AFL rapat"),
             (sgn(pb["Macro-F1"][0]), "Δ Macro-F1, 5b bobot lunak"),
             (f"{fnum(B['v31']['median'][2], 1)} → {fnum(B['bobotlunak']['median'][2], 2)}", "Bias AFL yang dibutuhkan, 5b"),
             ("v3.1", "Paket produksi tetap"))
    for i, (val, lab) in enumerate(cards):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 88, cardw, 88, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 20 if i != 2 else 16); c.drawString(x + 11, y - 37, val)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 7.9); c.drawString(x + 11, y - 61, lab)
    y -= 105
    y = para("Δ = varian − v3.1, berpasangan pada bootstrap pasien yang sama. Window uji detektor 50 Hz, bias cross-fit grid −7…+3. "
             "Fold, seed latih, dan hiperparameter identik dengan referensi; hanya resep training yang berbeda.", y, "small")
    y = callout("Jawaban singkat",
                f"<b>Tidak ada resep yang lebih baik dari v3.1.</b> 5a (window AFL rapat) setara dalam Macro-F1 (P = {fnum(pa['Macro-F1'][3], 2)}), "
                f"tetapi deteksi episode AFL justru turun secara konsisten. 5b (bobot lunak) juga setara (P = {fnum(pb['Macro-F1'][3], 2)}), "
                f"dengan bonus: model hanya butuh bias AFL {fnum(B['bobotlunak']['median'][2], 2)} (bukan {fnum(B['v31']['median'][2], 1)}), "
                f"dan tanpa bias pun jauh lebih baik ({fnum(A['bobotlunak']['Macro-F1'][0])} vs {fnum(A['v31']['Macro-F1'][0])}). "
                "Paket produksi tetap v3.1. Bobot lunak disarankan jadi resep bawaan untuk model baru.", y)
    y = section("Hipotesis (ditetapkan sebelum melihat hasil)", y)
    y = table(["Varian", "Hipotesis", "Hasil", "Status"], [
        ["5a: window AFL rapat (1 s)", "AFL recall/F1 naik, Macro-F1 tidak turun",
         f"Macro-F1 {sgn(pa['Macro-F1'][0])}, AFL F1 {sgn(pa['AFL F1'][0])} (tak terbedakan); deteksi episode {sgn(pa['deteksi episode'][0], 2)} (turun konsisten)",
         "<b>Ditolak</b>"],
        ["5b: bobot kelas lunak (akar)", "Tidak butuh bias ekstrem, Macro-F1 tidak turun",
         f"Bias AFL {fnum(B['bobotlunak']['median'][2], 2)}; Macro-F1 {sgn(pb['Macro-F1'][0])} (tak terbedakan)", "<b>Terbukti</b>"],
    ], y, [118, 130, 200, CW - 448])

    # ── Hal 2 ──
    y = page("Hasil utama", "Rata-rata 5 ulangan, CI 95% bootstrap pasien; alarm & episode memakai kode metrik v4")
    order = ["v1", "v31", "aflrapat", "bobotlunak"]
    y = table(["Varian", "Macro-F1", "AFL F1", "AFL P", "AFL R", "Normal R", "Alarm palsu/jam", "Episode terdeteksi"],
              [[L[k], ci(R[k]["Macro-F1"]), ci(R[k]["AFL F1"]), fnum(R[k]["AFL P"][0]), fnum(R[k]["AFL R"][0]),
                fnum(R[k]["Normal R"][0]), ci(E[k]["alarm/jam"], 2), ci(E[k]["deteksi episode"], 2)] for k in order],
              y, [112, 74, 74, 36, 36, 40, 70, CW - 442], bold_rows=[2])
    y = chart("forest", y, 160)
    y = section("Berpasangan dengan v3.1", y)
    y = table(["Varian", "Δ Macro-F1", "Δ AFL F1", "Δ AFL P", "Δ AFL R", "Δ Normal R", "Δ deteksi episode", "Vonis"],
              [[L[k], dci(P[k]["Macro-F1"]), dci(P[k]["AFL F1"]), dci(P[k]["AFL P"]), dci(P[k]["AFL R"]), dci(P[k]["Normal R"]),
                dci(P[k]["deteksi episode"], 2), verdict(P[k]["Macro-F1"][3]).replace("tidak terbedakan", "tidak<br/>terbedakan")]
               for k in ("v1", "aflrapat", "bobotlunak")],
              y, [100, 56, 56, 56, 56, 56, 60, CW - 440])

    # ── Hal 3 ──
    y = page("Kalibrasi: seberapa bergantung pada bias kelas", "Bias cross-fit per fold, hasil tanpa bias, dan bias yang sengaja digeser")
    y = table(["Varian", "Bias AFIB (median)", "Bias AFL (median)", "Bias AFL rentang", "Bobot AFL di loss"], [
        [L["v31"], fnum(B["v31"]["median"][1], 2), fnum(B["v31"]["median"][2], 2),
         f"{fnum(B['v31']['min'][2], 2)} … {fnum(B['v31']['max'][2], 2)}", fnum(d["class_weights_full"]["standar"][2], 1)],
        [L["aflrapat"], fnum(B["aflrapat"]["median"][1], 2), fnum(B["aflrapat"]["median"][2], 2),
         f"{fnum(B['aflrapat']['min'][2], 2)} … {fnum(B['aflrapat']['max'][2], 2)}", fnum(d["class_weights_full"]["aflrapat"][2], 1)],
        [L["bobotlunak"], fnum(B["bobotlunak"]["median"][1], 2), fnum(B["bobotlunak"]["median"][2], 2),
         f"{fnum(B['bobotlunak']['min'][2], 2)} … {fnum(B['bobotlunak']['max'][2], 2)}", fnum(d["class_weights_full"]["lunak"][2], 1)],
    ], y, [170, 75, 75, 95, CW - 415])
    y = section("Tanpa bias sama sekali (argmax)", y)
    y = table(["Varian", "Macro-F1", "AFL F1", "AFL P", "AFL R", "Δ Macro-F1 vs v3.1"],
              [[L[k], ci(A[k]["Macro-F1"]), ci(A[k]["AFL F1"]), fnum(A[k]["AFL P"][0]), fnum(A[k]["AFL R"][0]),
                "—" if k == "v31" else f"{sgn(AP[k][0])} (P = {fnum(AP[k][3], 2)}, {verdict(AP[k][3])})"]
               for k in ("v31", "aflrapat", "bobotlunak")],
              y, [150, 82, 82, 42, 42, CW - 398])
    y = section("Jika bias AFL meleset sebesar δ (Macro-F1)", y)
    dl = SN["delta"]
    y = table(["Varian"] + [("+" if x > 0 else "") + fnum(x, 1) for x in dl],
              [[SHORT[k]] + [fnum(v) for v in SN["macro_f1"][k]] for k in ("v31", "aflrapat", "bobotlunak")],
              y, [110] + [(CW - 110) / len(dl)] * len(dl))
    callout("Cara membaca",
            f"5b hanya butuh koreksi AFL setengahnya, dan tanpa bias pun Macro-F1-nya {fnum(A['bobotlunak']['Macro-F1'][0])} (v3.1: "
            f"{fnum(A['v31']['Macro-F1'][0])}), jadi lebih aman jika bias tidak terpasang. Tetapi jika bias terpasang lalu meleset "
            "(misalnya karena proporsi AFL di perangkat berbeda), ketiga model turun hampir sama cepat. Jadi 5b <b>tidak</b> lebih tahan "
            f"terhadap bias yang meleset. Catatan: bias AFL 5a menyentuh batas grid (−7) di sebagian fold.", y)

    # ── Hal 4 ──
    y = page("AFL per pasien dan alarm palsu", "Recall AFL rata-rata 5 ulangan")
    y = chart("afl_heatmap", y, 230)
    fp = d["fp"]
    y = table(["Varian", "AFL palsu (window/ulangan)", "di pasien tanpa AFL", "Alarm palsu/jam"],
              [[L[k], f"{fp[k]['total']:.0f}", f"{fp[k]['pasien_lain']:.0f}", fnum(E[k]["alarm/jam"][0], 2)]
               for k in order], y, [200, (CW - 200) / 3, (CW - 200) / 3, (CW - 200) / 3])
    callout("Yang terlihat",
            "5a tidak menolong pasien AFL yang sulit; ia malah menambah AFL palsu dan menurunkan recall pada 04043 dan 04908. "
            "5b menaikkan 04043 dan 06426 tapi menurunkan 08378, jadi hasil totalnya setara. "
            "07910 dan 08215 tetap 0 di semua resep berbasis interval RR, konsisten dengan v3 dan v4.", y)

    # ── Hal 5 ──
    y = page("Kesimpulan dan langkah berikutnya", "Keputusan v5")
    y = table(["Keputusan", "Alasan"], [
        ["<b>Paket produksi tetap v3.1</b>; v5 tidak dikemas", "Tidak ada resep yang lebih baik secara konsisten"],
        ["Window AFL rapat tidak dipakai", "Setara dalam Macro-F1, deteksi episode turun, AFL palsu bertambah"],
        ["<b>Bobot lunak jadi resep bawaan untuk model baru</b>",
         f"Setara, butuh koreksi bias setengahnya, tanpa bias jauh lebih baik ({sgn(AP['bobotlunak'][0])} Macro-F1)"],
        ["Tidak perlu mengganti model produksi ke 5b", "Tidak lebih tahan terhadap bias yang meleset; manfaatnya hanya jika bias tidak terpasang"],
    ], y, [230, CW - 230])
    y = section("Apa artinya untuk proyek", y)
    y = para("Setelah v3–v5, semua tuas yang hanya memakai interval RR sudah dicoba: arsitektur (4 model), kalibrasi bias, smoothing, "
             "ensemble, kepadatan data AFL, dan bobot kelas. Hasilnya berhenti di sekitar Macro-F1 0,83, dan pasien 07910 serta 08215 tidak "
             "pernah terdeteksi. Batasnya ada pada informasi input, bukan pada cara melatih.", y)
    y = section("Langkah berikutnya (v6)", y)
    for txt in ["<b>Cabang EKG mentah 50 Hz</b> (morfologi gelombang flutter 4–6 Hz masih terlihat di 50 Hz), digabung dengan cabang interval RR. "
                "Mulai dari baseline cepat (MiniRocket), lalu CNN dua cabang; pakai bobot lunak.",
                "Atau <b>data AFL tambahan</b> dari basis data lain (mis. Icentia11k, MIT-BIH Arrhythmia) untuk pretraining."]:
        y = para(f"• {txt}", y, gap=5)
    y = section("Batasan", y)
    for txt in ["Hanya 7 pasien AFL dan 13 episode; CI AFL sangat lebar.",
                "Reproduksi: <font name='Courier' size='8.3'>python tools/run_notebook.py v5_training_afl/training_afl.ipynb</font> "
                "(±60 menit GPU, bisa dilanjutkan), lalu <font name='Courier' size='8.3'>laporan/build_report.py --extract</font> dan tanpa argumen."]:
        y = para(f"• {txt}", y, gap=5)
    c.save()
    print(f"OK {OUT} ({state['page']} halaman)")


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else build()
