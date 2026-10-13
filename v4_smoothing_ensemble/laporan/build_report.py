"""Laporan PDF v4 (smoothing antar-window & ensemble), tanpa training.

Dua tahap (pola sama dengan v3/v3.1):
  1. --extract  (Python dengan numpy/matplotlib): hitung ulang semua angka lewat pascaproses.py dari
     ../cache/v4/pred.npz, cocokkan dengan ../cache/v4/ringkasan.json dari notebook, buat grafik → laporan/_build/.
  2. tanpa argumen (Python dengan ReportLab): rakit laporan/laporan_smoothing_ensemble.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # = v4_smoothing_ensemble/
REPO = HERE.parent
TMP = HERE / "laporan/_build"
OUT = HERE / "laporan/laporan_smoothing_ensemble.pdf"
SUMMARY = REPO / "cache/v4/ringkasan.json"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402

TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
C_REF, C_MT, C_E = "#2a78d6", "#1baf7a", "#eb6834"      # palet tervalidasi (dataviz validator: PASS)


# ═════════════════════════════════════════════════════════════════════
# Tahap 1
# ═════════════════════════════════════════════════════════════════════
def extract():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter

    sys.path.insert(0, str(HERE))
    import pascaproses as S

    TMP.mkdir(parents=True, exist_ok=True)
    res = S.run_all()                                         # memuat pred.npz dari cache (tanpa hitung ulang cross-fit)
    nb = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for k in res["rows"]:
        for m in ("Macro-F1", "AFL F1", "Normal R"):
            assert abs(nb["rows"][k][m][0] - res["rows"][k][m][0]) < 1e-9, (k, m)
        assert abs(nb["episodes"][k]["alarm/jam"][0] - res["episodes"][k]["alarm/jam"][0]) < 1e-9, k
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

    def style(ax):
        ax.grid(axis="x", color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(comma)

    R, E, L = res["rows"], res["episodes"], res["labels"]
    order = ["v31", "mt_off", "mt_rt", "e1", "e1_off", "e1_rt", "e2"]
    color = {"v31": C_REF, "mt_off": C_MT, "mt_rt": C_MT, "e1": C_E, "e1_off": C_E, "e1_rt": C_E, "e2": C_E}
    mark = {"v31": "o", "mt_off": "s", "mt_rt": "^", "e1": "o", "e1_off": "s", "e1_rt": "^", "e2": "D"}
    hollow = {"e2"}

    # 1. forest: Macro-F1, AFL F1, alarm/jam
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, (m, src, title) in zip(axes, (("Macro-F1", R, "Macro-F1"), ("AFL F1", R, "AFL F1"),
                                          ("alarm/jam", E, "Alarm AFL palsu/jam ↓"))):
        for i, k in enumerate(order):
            pt, lo, hi = src[k][m][:3]
            ax.plot([lo, hi], [i, i], color=color[k], lw=2, solid_capstyle="round")
            ax.plot(pt, i, mark[k], color=color[k], mfc="white" if k in hollow else color[k], mec=color[k], ms=7, mew=1.5)
        ax.axvline(src["v31"][m][0], color="#8A97A6", lw=1, ls="--")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        style(ax)
    short = {"v31": "v3.1 (referensi)", "mt_off": "MT + smoothing offline", "mt_rt": "MT + smoothing real-time",
             "e1": "Ensemble MT+BiGRU", "e1_off": "Ensemble + offline", "e1_rt": "Ensemble + real-time",
             "e2": "Ensemble 4 model (eksp.)"}
    axes[0].set_yticks(range(len(order)), [short[k] for k in order]); axes[0].invert_yaxis()
    fig.tight_layout(); save(fig, "forest")

    # 2. kurva per konfigurasi (dipaksa) — Macro-F1 dan alarm/jam
    pc = res["per_cfg"]
    off = [c for c in res["offline"] if c in pc]
    rt = [c for c in res["realtime"] if c in pc and c != "tanpa smoothing"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, (m, title) in zip(axes, (("Macro-F1", "Macro-F1"), ("alarm/jam", "Alarm AFL palsu/jam ↓"))):
        for j, (names, col, lab) in enumerate(((off, C_MT, "offline"), (rt, C_E, "real-time"))):
            ys = np.arange(len(names)) + (0 if lab == "offline" else len(off))
            ax.barh(ys, [pc[n][m] for n in names], color=col, height=.62, label=lab if ax is axes[0] else None)
            for yy, n in zip(ys, names):
                ax.text(pc[n][m], yy, f" {pc[n][m]:.3f}".replace(".", ","), va="center", fontsize=7.8, color=INK)
        style(ax); ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        if m == "Macro-F1":
            lo = min(pc[n][m] for n in off + rt)
            ax.set_xlim(lo - .02, max(pc[n][m] for n in off + rt) + .03)
    axes[0].set_yticks(range(len(off) + len(rt)), [n.replace(".", ",") for n in off + rt], fontsize=8.3)
    axes[0].invert_yaxis()
    fig.legend(loc="upper center", bbox_to_anchor=(.55, 1.06), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(); save(fig, "per_cfg")

    # 3. recall AFL per pasien
    cols = [("v1", "v1"), ("v31", "v3.1"), ("mt_off", "MT+offline"), ("e1", "Ensemble"), ("e1_off", "Ens.+offline")]
    mat = np.array([res["afl_recall"][k] for k, _ in cols]).T
    cmap = LinearSegmentedColormap.from_list("blue", ["#EEF4FC", "#86b6ef", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [c for _, c in cols]); ax.xaxis.tick_top()
    ax.set_yticks(range(len(res["afl_patients"])), res["afl_patients"]); ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=9.3,
                    color="white" if mat[i, j] > .55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.03); cb.outline.set_visible(False)
    cb.set_label("Recall AFL", fontsize=8.5); cb.ax.yaxis.set_major_formatter(comma)
    fig.tight_layout(); save(fig, "afl_heatmap")
    print(f"OK tables.json + 3 grafik → {TMP}")


# ═════════════════════════════════════════════════════════════════════
# Tahap 2 — diisi setelah hasil diketahui (lihat fungsi build di bawah)
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
        "table": ParagraphStyle("table", fontName="Arial", fontSize=8.4, leading=11.2, textColor=HexColor(INK)),
        "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8.3, leading=11.2, textColor=white),
    }
    c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
    c.setTitle("Laporan v4 - Smoothing antar-window & ensemble SIHEDAF")
    c.setAuthor("Analisis v4_smoothing_ensemble/smoothing_ensemble.ipynb")
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
        c.drawString(M + 43, H - 42, "SIHEDAF  /  V4 SMOOTHING ANTAR-WINDOW & ENSEMBLE")
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", 21); c.drawString(M, H - 79, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 99, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, "13 Oktober 2026 | Sumber: v4_smoothing_ensemble/smoothing_ensemble.ipynb, cache/v4/")
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

    R, E, P, L = d["rows"], d["episodes"], d["pairs"], d["labels"]
    ref = "v31"

    def verdict(p):
        return "lebih baik" if p >= .95 else "lebih buruk" if p <= .05 else "tidak terbedakan"

    def ci(v, nd=3):
        return f"{fnum(v[0], nd)} <font color='{GRAY}' size='7.4'>[{fnum(v[1], 2)}–{fnum(v[2], 2)}]</font>"

    def dci(v, nd=3, br=False):
        sep = "<br/>" if br else " "
        return f"{sgn(v[0], nd)}{sep}<font color='{GRAY}' size='7.2'>[{fnum(v[1], 2)}; {fnum(v[2], 2)}]</font>"

    e1, off, rt = "e1", "mt_off", "mt_rt"
    best_off = max(("mt_off", "e1_off"), key=lambda k: -E[k]["alarm/jam"][0])

    # ── Hal 1 ──
    y = page("Laporan v4: smoothing antar-window & ensemble", "Pasca-proses tanpa training, dibandingkan berpasangan dengan v3.1")
    cardw = (CW - 24) / 4
    cards = ((f"{sgn(P[e1]['Macro-F1'][0])}", "Δ Macro-F1, ensemble"),
             (f"{fnum(E[ref]['alarm/jam'][0], 2)} → {fnum(E[off]['alarm/jam'][0], 2)}", "Alarm palsu/jam, smoothing"),
             (f"{pct(E[ref]['deteksi episode'][0])} → {pct(E[off]['deteksi episode'][0])}", "Deteksi episode, smoothing"),
             ("v3.1", "Paket produksi tetap"))
    for i, (val, lab) in enumerate(cards):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 88, cardw, 88, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 20 if i in (0, 3) else 16); c.drawString(x + 11, y - 37, val)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 7.9); c.drawString(x + 11, y - 61, lab)
    y -= 105
    y = para("Window uji detektor 50 Hz, OOF v3 (5 ulangan × 5 fold pasien). Konfigurasi smoothing dan bias kelas dipilih per fold "
             "dari fold lain. \"Smoothing\" pada kartu = MicroTransformer + smoothing offline. Bukan uji pada perangkat asli.", y, "small")
    y = callout("Jawaban singkat",
                f"<b>Tidak ada varian yang lebih baik dari v3.1.</b> Ensemble MT + BiGRU setara (P = {fnum(P[e1]['Macro-F1'][3], 2)}), "
                "jadi tidak sebanding dengan biaya melatih dan mengirim model kedua. Smoothing bukan perbaikan, melainkan "
                f"<b>tukar-menukar</b>: alarm AFL palsu turun sekitar {E[ref]['alarm/jam'][0] / E[off]['alarm/jam'][0]:.0f}×, tetapi separuh episode AFL "
                "ikut hilang, terutama episode pendek. Paket produksi tetap v3.1; v4 tidak dikemas.", y)
    y = section("Hipotesis (ditetapkan sebelum melihat hasil)", y)
    rows = []
    for k in d["primary"]:
        pr = P[k]
        rows.append([L[k], f"{verdict(pr['Macro-F1'][3])} (P = {fnum(pr['Macro-F1'][3], 2)})",
                     f"{fnum(E[k]['alarm/jam'][0], 2)} vs {fnum(E[ref]['alarm/jam'][0], 2)}",
                     f"{pct(E[k]['deteksi episode'][0])} vs {pct(E[ref]['deteksi episode'][0])}"])
    y = table(["Varian", "Macro-F1 vs v3.1", "Alarm palsu/jam", "Episode AFL terdeteksi"], rows, y,
              [190, 120, 95, CW - 405])
    para("Ensemble 4 model diuji sebagai eksplorasi saja: tidak terbedakan dalam Macro-F1 dan alarm palsunya lebih banyak.", y, "small")

    # ── Hal 2 ──
    y = page("Hasil utama", "Rata-rata 5 ulangan, CI 95% bootstrap pasien; alarm & episode dihitung per pasien lalu di-bootstrap")
    order = ["v1", "v31", "mt_off", "mt_rt", "e1", "e1_off", "e1_rt", "e2"]
    y = table(["Varian", "Macro-F1", "AFL F1", "AFL R", "Normal R", "Alarm palsu/jam", "Episode terdeteksi"],
              [[L[k], ci(R[k]["Macro-F1"]), ci(R[k]["AFL F1"]), fnum(R[k]["AFL R"][0]), fnum(R[k]["Normal R"][0]),
                ci(E[k]["alarm/jam"], 2), ci(E[k]["deteksi episode"], 2)] for k in order],
              y, [140, 76, 76, 40, 44, 70, CW - 446], bold_rows=[2])
    y = chart("forest", y, 230)
    callout("Cara membaca",
            f"Garis putus-putus = v3.1. Ada {E[ref]['episode AFL']} episode AFL dalam {E[ref]['jam']:.0f} jam rekaman, jadi CI deteksi episode "
            "lebar. v1 terlihat paling banyak mendeteksi episode karena menebak AFL hampir di mana-mana "
            f"({fnum(E['v1']['alarm/jam'][0], 1)} alarm palsu per jam).", y)

    # ── Hal 3 ──
    y = page("Perbandingan berpasangan dengan v3.1", "Δ = varian − v3.1 pada sampel bootstrap pasien yang sama")
    rows = []
    for k in [k for k in order if k != ref]:
        pr = P[k]
        rows.append([L[k], dci(pr["Macro-F1"], br=True), dci(pr["AFL F1"], br=True), dci(pr["Normal R"], br=True),
                     dci(pr["alarm/jam"], 2, br=True), dci(pr["deteksi episode"], 2, br=True),
                     verdict(pr["Macro-F1"][3]).replace("tidak terbedakan", "tidak<br/>terbedakan").replace("lebih ", "lebih<br/>")])
    y = table(["Varian", "Δ Macro-F1", "Δ AFL F1", "Δ Normal R", "Δ alarm/jam", "Δ deteksi episode", "Vonis Macro-F1"],
              rows, y, [118, 66, 66, 66, 66, 66, CW - 448])
    po, pe = P[off], P[e1]
    y = callout("Yang konsisten (CI tidak menyentuh 0)",
                f"<b>Smoothing offline</b> menurunkan alarm palsu ({dci(po['alarm/jam'], 2)} per jam) dan menaikkan recall Normal "
                f"({dci(po['Normal R'])}), tapi juga menurunkan deteksi episode AFL ({dci(po['deteksi episode'], 2)}). Dalam Macro-F1 window "
                f"ia tidak terbedakan dari v3.1. <b>Ensemble</b> hanya konsisten menaikkan recall Normal ({dci(pe['Normal R'])}); Macro-F1, AFL F1, "
                "dan alarm palsunya tidak terbedakan. <b>Smoothing real-time</b> lebih buruk dalam Macro-F1.", y)
    callout("Kenapa ini penting",
            "Smoothing memindahkan sistem ke titik lain pada kurva yang sama: lebih sedikit alarm, lebih banyak AFL terlewat. "
            "Mana yang lebih baik bergantung pada pemakaian. Untuk skrining, AFL yang terlewat lebih mahal; untuk pemantauan "
            "rumah yang alarmnya mengganggu, sebaliknya. Ini keputusan produk, bukan hasil teknis.", y, "#FFF4EC")

    # ── Hal 4 ──
    y = page("Trade-off smoothing", "Hasil jika satu konfigurasi dipaksa untuk semua fold (diagnostik, bukan seleksi jujur)")
    y = chart("per_cfg", y, 250)
    ch = d["chosen"]
    rows = []
    for k in ("mt_off", "mt_rt", "e1_off", "e1_rt"):
        top = sorted(ch[k].items(), key=lambda kv: -kv[1])[:3]
        rows.append([L[k], ", ".join(f"{n.replace('.', ',')} ({v}×)" for n, v in top)])
    y = table(["Varian", "Konfigurasi terpilih cross-fit (dari 25 fold)"], rows, y, [210, CW - 210])
    pc = d["per_cfg"]
    callout("Yang terlihat",
            f"Semakin lebar smoothing, semakin sedikit alarm palsu (tanpa smoothing {fnum(pc['tanpa smoothing']['alarm/jam'], 2)} → "
            f"rata-rata ±120 s {fnum(pc['rata-rata ±120 s']['alarm/jam'], 2)} per jam). Tapi AFL F1 tidak pernah melampaui tanpa smoothing "
            f"({fnum(pc['tanpa smoothing']['AFL F1'])}). Pilihan cross-fit juga tidak stabil: konfigurasi berganti antar fold, dan "
            "kombinasi terbaik di tabel dipaksa ini tidak bisa dipakai sebagai klaim karena dipilih setelah melihat hasil.", y)

    # ── Hal 5 ──
    y = page("AFL per pasien", "Recall AFL rata-rata 5 ulangan; jumlah window AFL per pasien di laporan v3")
    y = chart("afl_heatmap", y, 240)
    callout("Episode pendek hilang",
            "Smoothing menghapus AFL pada pasien dengan episode pendek: 04043 (1 window) dan 06426 (21 window, ±2 menit) turun "
            "ke 0. Episode panjang (04908, 08378) tetap terdeteksi tapi recall-nya turun. Ensemble sedikit menaikkan 04043 dan 06426, "
            "tetapi tidak cukup untuk mengubah hasil keseluruhan. 07910 dan 08215 tetap tidak terdeteksi oleh varian mana pun.", y)

    # ── Hal 6 ──
    y = page("Kesimpulan dan langkah berikutnya", "Keputusan v4 dan rencana v5")
    y = table(["Keputusan", "Alasan"], [
        ["<b>Paket produksi tetap v3.1</b>; v4 tidak dikemas (tidak ada v4.1)", "Tidak ada varian yang lebih baik secara konsisten"],
        ["Ensemble MT + BiGRU tidak dipakai", f"Setara (P = {fnum(P[e1]['Macro-F1'][3], 2)}); butuh model final kedua dan inferensi 2×"],
        ["Smoothing tidak dijadikan default", "Mengurangi alarm palsu tapi menghilangkan separuh episode AFL"],
        ["Smoothing offline bisa jadi <i>opsi produk</i>", "Jika tim memutuskan alarm palsu lebih mahal dari AFL terlewat. Perlu keputusan pemilik produk"],
    ], y, [230, CW - 230])
    y = section("Langkah berikutnya (v5, perlu training)", y)
    for txt in ["<b>Window AFL rapat khusus training</b> (stride 1 s, evaluasi tetap 5 s). Petunjuk terkuat dari v3: resep v1 dengan "
                "stride itu lebih baik dari revisi tanpa bias.",
                "<b>Bobot kelas lebih lunak</b> (akar bobot sekarang / logit-adjusted loss), supaya model tidak bergantung pada bias −4,5.",
                "<b>Cabang EKG mentah 50 Hz</b> untuk 07910, 08215, 06426, yang gagal di semua versi berbasis interval RR."]:
        y = para(f"• {txt}", y, gap=5)
    y = section("Batasan", y)
    for txt in [f"Hanya {E[ref]['episode AFL']} episode AFL di 7 pasien: metrik episode sangat kasar.",
                "Smoothing offline mengasumsikan rekaman utuh tersedia sebelum diputuskan (cocok untuk <i>predict_ecg</i> per rekaman, "
                "tidak untuk alarm seketika).",
                "Reproduksi: <font name='Courier' size='8.3'>python tools/run_notebook.py v4_smoothing_ensemble/smoothing_ensemble.ipynb</font> "
                "(±14 menit), lalu <font name='Courier' size='8.3'>laporan/build_report.py --extract</font> dan tanpa argumen."]:
        y = para(f"• {txt}", y, gap=5)
    c.save()
    print(f"OK {OUT} ({state['page']} halaman)")


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else build()
