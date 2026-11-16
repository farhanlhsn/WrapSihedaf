"""Laporan PDF v11.1: evaluasi 2 kelas (Normal vs AF/AFL) untuk model yang sudah ada, dengan dua aturan keputusan.

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v11.1/dua_kelas.json (`python dua_kelas.py`), hitung ulang Macro-F1 2 kelas beberapa sistem dari OOF di
     cache untuk kedua aturan dan cocokkan (assert), ambil Macro-F1 3 kelas dari ringkasan.json v11, lalu simpan angka + grafik
     ke laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_dua_kelas.pdf dengan ReportLab.
  (--grafik: gambar ulang grafik dari _build/data.json saja.)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # v11.1_dua_kelas/laporan
VERSI = HERE.parent
REPO = VERSI.parent
TMP = HERE / "_build"
PDF = HERE / "laporan_dua_kelas.pdf"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402  (Arial → Liberation Sans → DejaVu)

TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
BIRU, ORANYE, HIJAU = "#2a78d6", "#eb6834", "#1baf7a"
URUT = ["v1 (argmax)", "v3.1", "BiGRU bawaan", "BiGRU wp100", "TCN bawaan", "TCN wp100", "XGBoost bawaan", "XGBoost bobot penuh"]
TIGA = {"BiGRU bawaan": "bigru/bawaan", "BiGRU wp100": "bigru/wp100", "TCN bawaan": "tcn/bawaan", "TCN wp100": "tcn/wp100",
        "XGBoost bawaan": "xgboost/bawaan", "XGBoost bobot penuh": "xgboost/bobot_penuh"}
NAMA_ATURAN = {"gabung": "3 kelas lalu digabung", "ulang": "bias 2 kelas dipilih ulang"}


# ═════════════════════════════════════════════════════════════════════
# Tahap 1: angka + grafik
# ═════════════════════════════════════════════════════════════════════
def extract():
    import numpy as np
    sys.path[:0] = [str(VERSI)]
    import dua_kelas as DK
    from bersama import data, evaluasi as EV, runner

    J = json.loads(DK.OUT.read_text(encoding="utf-8"))
    for a in ("ulang", "gabung"):
        assert list(J["aturan"][a]["sistem"]) == URUT, (a, list(J["aturan"][a]["sistem"]))
    d = data.muat()
    R = list(range(data.N_ULANGAN))
    y2 = (d.y > 0).astype(np.int64)
    # cocokkan dengan OOF di cache: hitung ulang Macro-F1 2 kelas dua sistem untuk kedua aturan
    for nama, oof in (("v3.1", d.oof_v31[R]), ("TCN bawaan", runner.muat_hasil(data.lokasi_hasil() / "tcn/bawaan", d).oof)):
        p_ulang, _ = DK.keputusan2(oof, d.folds[R], y2)
        p3, _ = EV.keputusan(oof, d.folds[R], d.y)
        for a, pred in (("ulang", p_ulang), ("gabung", (p3 > 0).astype(np.int64))):
            mf = float(DK.metrik(DK.cm_pasien(y2, pred, d.records, d.patients).sum(1))["Macro-F1"].mean())
            ref = J["aturan"][a]["sistem"][nama]["Macro-F1"][0]
            assert abs(mf - ref) < 1e-12, (a, nama, mf, ref)
    print("cocok dengan OOF di cache: v3.1 dan TCN bawaan, kedua aturan")
    tiga = {}
    for nama, p in TIGA.items():
        rk = json.loads((data.lokasi_hasil() / p / "ringkasan.json").read_text(encoding="utf-8"))
        tiga[nama] = rk["metrik"]["bias"]["Macro-F1"]
        tiga.setdefault("v3.1", rk["metrik"]["v31"]["Macro-F1"])
        tiga.setdefault("v1 (argmax)", rk["metrik"]["v1"]["Macro-F1"])
    # asal AFL teratur yang lambat (sumber alarm 85–90 bpm bila bias 2 kelas dipilih ulang)
    ibi, msk = d.X[..., 0], d.M.astype(bool)
    cv_ibi = np.array([ibi[i][msk[i]].std() / ibi[i][msk[i]].mean() for i in range(len(ibi))])
    lambat = (d.y == 2) & (d.hr >= 80) & (d.hr < 95) & (cv_ibi < 0.03)
    afl_lambat = {"total": int(lambat.sum()), "per_pasien": {p: int((lambat & (d.records == p)).sum()) for p in np.unique(d.records[lambat])},
                  "bpm_p5_p95": np.percentile(d.hr[lambat], [5, 95]).tolist(), "n_afl": int((d.y == 2).sum())}
    payload = {**J, "tiga": tiga, "urut": URUT, "afl_lambat": afl_lambat}
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    grafik()


def grafik():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter

    P = json.loads((TMP / "data.json").read_text(encoding="utf-8"))
    G, U = P["aturan"]["gabung"], P["aturan"]["ulang"]
    SG, SU = G["sistem"], U["sistem"]
    for style in ("regular", "bold"):
        font_manager.fontManager.addfont(font_path(style))
    family = font_manager.FontProperties(fname=font_path("regular")).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#C4CED8", "axes.labelcolor": INK, "text.color": INK, "xtick.color": GRAY,
                         "ytick.color": INK, "figure.facecolor": "white", "savefig.facecolor": "white"})
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

    # 1. forest: Macro-F1 2 kelas, aturan gabung (CI) + titik aturan ulang
    fig, axes = plt.subplots(1, 2, figsize=(7.2, .36 * len(P["urut"]) + 1.2), sharey=True)
    for ax, met in zip(axes, ("Macro-F1", "F1 AF1")):
        for i, n in enumerate(P["urut"]):
            pt, lo, hi = SG[n][met]
            ax.plot([lo, hi], [i, i], color=BIRU, lw=2, solid_capstyle="round", alpha=.85)
            ax.plot(pt, i, "o", color=BIRU, ms=7, mec="white", mew=1.2, label="3 kelas lalu digabung" if i == 0 else None)
            if n != "v1 (argmax)":
                ax.plot(SU[n][met][0], i, "D", color=ORANYE, ms=5.5, mfc="white", mew=1.4,
                        label="bias 2 kelas dipilih ulang" if i == 1 else None)
        ax.axvline(SG["v3.1"][met][0], color="#8A97A6", lw=1, ls="--")
        ax.axvline(.90, color=HIJAU, lw=.9, ls=":")
        ax.axhline(1.5, color="#E0E6EC", lw=.8)
        ax.set_title(("Macro-F1 2 kelas" if met == "Macro-F1" else "F1 AF/AFL") + " (CI 95%)", loc="left", fontsize=10, fontweight="bold")
        ax.set_xlim(.82, 1)
        style(ax)
    axes[0].set_yticks(range(len(P["urut"])), P["urut"])
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -.08), ncol=2, frameon=False, fontsize=8.4)
    fig.tight_layout()
    save(fig, "forest")

    # 2. selisih berpasangan vs v3.1 (aturan gabung)
    lain = [n for n in P["urut"] if n != "v3.1"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, .36 * len(lain) + 1.25), sharey=True)
    for ax, met in zip(axes, ("Macro-F1", "Sp")):
        ax.axvline(0, color=INK, lw=1)
        for i, n in enumerate(lain):
            dl, lo, hi, p = SG[n]["vs_v31"][met]
            col, mk, isi = (HIJAU, "^", True) if p >= .95 else (ORANYE, "v", True) if p <= .05 else (BIRU, "o", False)
            ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
            ax.plot(dl, i, mk, ms=8, color=col, mfc=col if isi else "white", mec=col, mew=1.6)
        ax.set_title(f"Δ {'Macro-F1' if met == 'Macro-F1' else 'spesifisitas'} vs v3.1", loc="left", fontsize=10, fontweight="bold")
        style(ax)
    axes[0].set_yticks(range(len(lain)), lain)
    axes[0].invert_yaxis()
    fig.text(.01, -.03, "Aturan 3 kelas lalu digabung. ▲ hijau = lebih baik (P ≥ 0,95)   ▼ oranye = lebih buruk (P ≤ 0,05)   "
             "○ biru = tidak terbedakan. Garis = CI 95%.", fontsize=8.4, color=GRAY)
    fig.tight_layout()
    save(fig, "paired")

    # 3. uji stres sintetis: ulang vs gabung untuk dua model
    jenis = ["tanpa variasi", "Gauss sd 10 ms"]
    bpm = SG["BiGRU bawaan"]["stres"]["bpm"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, j in zip(axes, jenis):
        for n, mk in (("BiGRU wp100", "s"), ("BiGRU bawaan", "o")):
            ax.plot(bpm, SU[n]["stres"]["proporsi_normal"][j], color=ORANYE, lw=1.6, marker=mk, ms=3.6, markevery=1,
                    ls="--" if n == "BiGRU bawaan" else "-", label=f"{n}, bias 2 kelas dipilih ulang")
            ax.plot(bpm, SG[n]["stres"]["proporsi_normal"][j], color=BIRU, lw=1.6, marker=mk, ms=3.6, markevery=1,
                    ls="--" if n == "BiGRU bawaan" else "-", label=f"{n}, 3 kelas lalu digabung")
        ax.axhline(.5, color="#C4CED8", lw=.8)
        ax.axvspan(82.5, 92.5, color="#FFF4EC", zorder=0)
        ax.set_title(j, fontsize=9, loc="left")
        ax.set_xticks([60, 80, 100, 120, 140, 160])
        ax.yaxis.set_major_formatter(pct)
        style(ax, "y")
    axes[0].set_ylabel("ditebak Normal")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -.2), ncol=2, frameon=False, fontsize=8)
    fig.text(.5, -.02, "Ritme teratur sintetis (bpm). Pita oranye muda = 85–90 bpm. Garis tipis = 50%.", ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "stres")

    # 4. alarm pada Normal teratur 80–95 bpm (asli), kedua aturan
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    y_pos = np.arange(len(P["urut"]))
    ax.barh(y_pos - .2, [SU[n]["teratur_80_95"] for n in P["urut"]], .38, color=ORANYE, label="bias 2 kelas dipilih ulang")
    ax.barh(y_pos + .2, [SG[n]["teratur_80_95"] for n in P["urut"]], .38, color=BIRU, label="3 kelas lalu digabung",
            hatch="///", edgecolor="white")
    for i, n in enumerate(P["urut"]):
        for dy, v in ((-.2, SU[n]["teratur_80_95"]), (.2, SG[n]["teratur_80_95"])):
            ax.text(v + .004, i + dy, f"{v * 100:.1f}%".replace(".", ","), va="center", fontsize=7.6, color=INK)
    ax.set_yticks(y_pos, P["urut"])
    ax.invert_yaxis()
    style(ax)
    ax.xaxis.set_major_formatter(pct)
    n_teratur = f"{P['n']['teratur_80_95']:,}".replace(",", ".")
    ax.set_xlabel(f"Window Normal teratur 80–95 bpm (CV IBI < 0,03; {n_teratur} window) yang menjadi alarm")
    ax.legend(frameon=False, fontsize=8.4, loc="lower right")
    fig.tight_layout()
    save(fig, "teratur")

    # 5. per pasien (aturan gabung)
    pas = P["pasien"]
    pilih = ["v3.1", "BiGRU bawaan", "TCN bawaan", "TCN wp100"]
    se = np.array([[G["per_pasien"][n]["se"][i] for n in pilih] for i in range(len(pas))], float)
    al = np.array([[G["per_pasien"][n]["alarm_normal"][i] for n in pilih] for i in range(len(pas))], float)
    cmap = LinearSegmentedColormap.from_list("biru", ["#EEF4FC", "#86b6ef", BIRU, "#104281"])
    cmap2 = LinearSegmentedColormap.from_list("oranye", ["#FFF4EC", "#f6b48f", ORANYE, "#8f3a12"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.4), sharey=True)
    for ax, mat, cm_, judul in ((axes[0], se, cmap, "Sensitivitas AF/AFL"), (axes[1], al, cmap2, "Bagian window Normal → alarm")):
        ax.imshow(np.ma.masked_invalid(mat), cmap=cm_, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pilih)), [p.replace(" ", "\n") for p in pilih], fontsize=8)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=7.4,
                        color="white" if (not np.isnan(v) and v > .55) else INK)
        ax.set_title(judul, fontsize=9.5, fontweight="bold", loc="left", pad=28)
    axes[0].set_yticks(range(len(pas)), pas, fontsize=8)
    fig.tight_layout()
    save(fig, "per_pasien")
    print(f"OK {TMP / 'data.json'} + 5 grafik")


# ═════════════════════════════════════════════════════════════════════
# Tahap 2: PDF
# ═════════════════════════════════════════════════════════════════════
def fnum(v, nd=3):
    return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")


def sg(v, nd=3):
    return ("+" if v >= 0 else "") + fnum(v, nd) if round(v, nd) != 0 else fnum(abs(v), nd)


def pct(v, nd=0):
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

    P = json.loads((TMP / "data.json").read_text(encoding="utf-8"))
    G, Ul = P["aturan"]["gabung"], P["aturan"]["ulang"]
    SG, SU, BG, BU, TG, TU = G["sistem"], Ul["sistem"], G["bersarang"], Ul["bersarang"], G["tanpa_04936"], Ul["tanpa_04936"]
    URT = P["urut"]
    for name, st in (("Arial", "regular"), ("Arial-Bold", "bold"), ("Arial-Italic", "italic")):
        pdfmetrics.registerFont(TTFont(name, font_path(st)))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
    ST = {"body": ParagraphStyle("body", fontName="Arial", fontSize=10, leading=14.3, textColor=HexColor(INK)),
          "small": ParagraphStyle("small", fontName="Arial", fontSize=8.6, leading=11.8, textColor=HexColor(GRAY)),
          "table": ParagraphStyle("table", fontName="Arial", fontSize=8.3, leading=11, textColor=HexColor(INK)),
          "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8.2, leading=11, textColor=white)}
    c = canvas.Canvas(str(PDF), pagesize=A4, pageCompression=1)
    c.setTitle("v11.1: evaluasi 2 kelas (Normal vs AF/AFL)")
    c.setAuthor("v11.1_dua_kelas (pemilik repo)")
    W_, H = A4
    M = 40
    CW = W_ - 2 * M
    state = {"page": 0}

    def para(txt, y, style="body", gap=8):
        p = Paragraph(txt, ST[style])
        _, h = p.wrap(CW, H)
        assert y - h >= 51, f"hal {state['page']}: paragraf lewat footer: {txt[:60]}"
        p.drawOn(c, M, y - h)
        return y - h - gap

    def page(title, kicker):
        if state["page"]:
            c.showPage()
        state["page"] += 1
        n = state["page"]
        c.setFillColor(HexColor(TEAL)); c.rect(M, H - 41, 32, 4, fill=1, stroke=0)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial-Bold", 8)
        c.drawString(M + 43, H - 42, "SIHEDAF  /  V11.1  /  EVALUASI 2 KELAS")
        fs = 20
        while pdfmetrics.stringWidth(title, "Arial-Bold", fs) > CW and fs > 13:
            fs -= .5
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", fs); c.drawString(M, H - 78, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 97, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, "3 November 2026 | Sumber: v11.1_dua_kelas/dua_kelas.py, cache/v11.1/dua_kelas.json")
        c.drawRightString(W_ - M, 29, f"{n:02d}")
        c.bookmarkPage(f"p{n}"); c.addOutlineEntry(title, f"p{n}", 0, False)
        return H - 118

    def section(title, y):
        c.setFont("Arial-Bold", 11.5); c.setFillColor(HexColor(INK)); c.drawString(M, y - 12, title)
        return y - 24

    def table(headers, rows, y, widths=None, bold_rows=()):
        widths = widths or [CW / len(headers)] * len(headers)
        cells = [[Paragraph(str(v), ST["head"]) for v in headers]] + [[Paragraph(str(v), ST["table"]) for v in r] for r in rows]
        tb = Table(cells, colWidths=widths, hAlign="LEFT")
        st = [("BACKGROUND", (0, 0), (-1, 0), HexColor(INK)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
              ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
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
        p = Paragraph(f"<b>{title}</b><br/>{txt}", ST["body"])
        _, h = p.wrap(CW - 24, H)
        assert y - h - 20 >= 51, (state["page"], title, y, h)
        c.setFillColor(HexColor(tint)); c.roundRect(M, y - h - 20, CW, h + 20, 6, fill=1, stroke=0)
        p.drawOn(c, M + 12, y - h - 10)
        return y - h - 30

    def ci(m, nd=3):
        return f"{fnum(m[0], nd)} <font color='{GRAY}' size='7.2'>[{fnum(m[1], 2)}–{fnum(m[2], 2)}]</font>"

    def dci(x):
        return f"{sg(x[0])} <font color='{GRAY}' size='7.2'>[{sg(x[1], 2)}…{sg(x[2], 2)}]</font>"

    model = [n for n in URT if n not in ("v1 (argmax)", "v3.1")]
    best = max(model, key=lambda n: SG[n]["Macro-F1"][0])
    lulus = [n for n in model if SG[n]["vs_v31"]["Macro-F1"][3] >= .95]
    lulus_v1 = [n for n in model if BG[n]["vs_v1"]["p"] >= .95]
    rng = lambda xs: f"{fnum(min(xs))}–{fnum(max(xs))}"  # noqa: E731
    amb = lambda st: " / ".join(str(v) for v in st["ambang"].values())  # noqa: E731
    al120 = lambda S, n: S[n]["alarm_normal"]["≥ 120"]  # noqa: E731
    sintetis_8590 = lambda S, n: 1 - min(min(S[n]["stres"]["proporsi_normal"][j][S[n]["stres"]["bpm"].index(b)]  # noqa: E731
                                             for b in (85, 90)) for j in S[n]["stres"]["proporsi_normal"])
    paling = max(model, key=lambda n: SU[n]["teratur_80_95"])

    # ── Hal 1: ringkasan ──────────────────────────────────────────────
    y = page("Evaluasi 2 kelas: Normal vs AF/AFL",
             f"Model yang sudah ada, tanpa training baru | 5 ulangan × 5 fold pasien | data detektor 50 Hz | {P['n']['pasien']} pasien AFDB")
    cards = [(fnum(SG[best]["Macro-F1"][0]), f"Macro-F1 2 kelas tertinggi ({best})*"),
             (sg(SG[best]["vs_v31"]["Macro-F1"][0]), f"vs v3.1 (P = {fnum(SG[best]['vs_v31']['Macro-F1'][3], 2)})"),
             (fnum(BG[best]["vs_v1"]["p"], 2), f"P vs v1, bootstrap bersarang ({best})"),
             (pct(SU[paling]["teratur_80_95"]), f"Normal teratur 80–95 bpm jadi alarm bila bias dipilih ulang ({paling})")]
    cardw = (CW - 24) / 4
    for i, (val, lab) in enumerate(cards):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 84, cardw, 84, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 23); c.drawString(x + 11, y - 36, val)
        p = Paragraph(lab, ParagraphStyle("kartu", fontName="Arial", fontSize=8.3, leading=10, textColor=HexColor(GRAY)))
        p.wrap(cardw - 20, 40); p.drawOn(c, x + 11, y - 52 - p.height + 10)
    y -= 100
    y = para("*Aturan keputusan \"3 kelas lalu digabung\": model dan bias 3 kelas seperti evaluasi biasa, lalu tebakan AFIB dan AFL "
             "disatukan menjadi AF/AFL (\"AF₁\" di literatur). Semua angka: prediksi out-of-fold rata-rata 5 ulangan × 5 fold pasien, "
             "CI 95% bootstrap pasien 2.000×. Pertanyaan: kalau AFL tidak dibedakan dari AFIB, seberapa baik model membedakan "
             "aritmia dari ritme normal?", y, "small")
    y = callout("Jawaban singkat",
                f"Target Macro-F1 ≥ 0,90 tercapai untuk tugas 2 kelas: {rng([SG[n]['Macro-F1'][0] for n in URT])} di semua sistem. "
                f"Terbaik <b>{best} {fnum(SG[best]['Macro-F1'][0])}</b>, lebih baik dari v3.1 (P = "
                f"{fnum(SG[best]['vs_v31']['Macro-F1'][3], 2)}) dan dari v1 pada uji paling ketat (P bersarang = "
                f"{fnum(BG[best]['vs_v1']['p'], 2)}). Cara mengambil keputusan penting: memilih ulang bias khusus 2 kelas memberi "
                f"Macro-F1 yang hampir sama, tetapi membuat alarm baru pada ritme normal teratur 80–95 bpm (sampai "
                f"{pct(SU[paling]['teratur_80_95'])}). Pakai keputusan 3 kelas lalu digabung. Alarm pada ritme normal ≥ 120 bpm tetap "
                f"tinggi ({pct(al120(SG, best))}).", y)
    y = section("Temuan utama", y)
    temuan = [
        ("1. Target ≥ 0,90 tercapai untuk 2 kelas.", f"Macro-F1 2 kelas {rng([SG[n]['Macro-F1'][0] for n in URT])}; 3 kelas untuk "
         f"model yang sama {rng([P['tiga'][n][0] for n in URT])}. Bedanya hampir seluruhnya kelas AFL."),
        ("2. Lebih baik dari v3.1:", ", ".join(f"{n} {sg(SG[n]['vs_v31']['Macro-F1'][0])}" for n in lulus) + " (P ≥ 0,95). "
         "Bersarang vs v1 lebih baik: " + (", ".join(lulus_v1) if lulus_v1 else "tidak ada") + "."),
        ("3. Aturan keputusan menentukan alarm palsu baru.", "Bias 2 kelas yang dipilih ulang melemahkan penekan AFL, sehingga ritme "
         "teratur 80–95 bpm disangka AFL lambat: " + ", ".join(f"{n} {pct(SU[n]['teratur_80_95'], 1)}" for n in ("v3.1", "BiGRU wp100", "BiGRU bawaan"))
         + f" vs {pct(max(SG[n]['teratur_80_95'] for n in URT), 1)} atau kurang dengan aturan gabung."),
        ("4. 04936 tidak lagi menentukan tingginya angka.", f"Tanpa 04936, Macro-F1 2 kelas {rng([TG[n]['Macro-F1'][0] for n in URT])}. "
         "Namun keunggulan atas v3.1 jadi tidak pasti: " + ", ".join(f"{n} P = {fnum(TG[n]['vs_v31']['Macro-F1'][3], 2)}" for n in lulus) + "."),
        ("5. Takikardia ≥ 120 bpm tetap masalah.", f"Normal ≥ 120 bpm → alarm {pct(al120(SG, 'v3.1'))} (v3.1), {pct(al120(SG, best))} ({best}). "
         "Penggabungan kelas tidak memisahkan ritme cepat-teratur dari aritmia."),
    ]
    for j, isi in temuan:
        y = para(f"<b>{j}</b> {isi}", y, gap=6)

    # ── Hal 2: 3 kelas vs 2 kelas ────────────────────────────────────
    y = page("Dari 3 kelas ke 2 kelas", "Data uji, fold pasien, dan sampel bootstrap yang sama")
    rows = [[n, ci(P["tiga"][n]), ci(SG[n]["Macro-F1"]), fnum(SU[n]["Macro-F1"][0]), sg(SG[n]["Macro-F1"][0] - P["tiga"][n][0])]
            for n in URT]
    y = table(["Sistem", "Macro-F1 3 kelas", "2 kelas: 3 kelas lalu digabung", "2 kelas: bias dipilih ulang", "Selisih 2 − 3 kelas"],
              rows, y, widths=[115, 105, 125, 100, CW - 445], bold_rows=[2])
    y = chart("forest", y, 270)
    para("Lingkaran biru + garis = aturan 3 kelas lalu digabung (CI 95%); belah ketupat oranye = bias 2 kelas dipilih ulang. Garis "
         "putus-putus = v3.1; titik-titik hijau = target 0,90. v1 tetap argmax (protokol v1).", y, "small")

    # ── Hal 3: metode ─────────────────────────────────────────────────
    y = page("Metode", "Tidak ada training baru; hanya cara mengambil keputusan dan menilai yang berubah")
    y = para("Model yang dinilai sama persis dengan evaluasi 3 kelas: OOF v1 dan v3.1 dari cache v3, serta BiGRU, TCN, dan XGBoost "
             "v11 (bobot kelas lunak = bawaan; bobot penuh = wp100 / bobot_penuh). Dua aturan keputusan dinilai dengan protokol yang "
             "sama (metrik pooled, CI bootstrap pasien, berpasangan, bersarang, tanpa 04936, uji stres):", y)
    y = table(["Aturan", "Cara", "Status"], [
        ["Bias 2 kelas dipilih ulang", "Probabilitas AF/AFL = p(AFIB) + p(AFL); satu bias AF/AFL dipilih dari fold lain (grid −7…+3) "
         "untuk Macro-F1 2 kelas.", "Rencana awal"],
        ["3 kelas lalu digabung", "Bias 3 kelas dipilih dari fold lain seperti evaluasi biasa; tebakan AFIB dan AFL disatukan.",
         "Ditambahkan setelah uji stres memperlihatkan alarm baru (tercatat sebagai analisis tambahan)"],
    ], y, widths=[110, 270, CW - 380])
    y = section("Istilah", y)
    y = table(["Istilah", "Arti sederhana"], [
        ["AF/AFL (AF₁)", "AFIB dan AFL digabung menjadi satu kelas aritmia; definisi yang dipakai literatur AF berbasis Holter."],
        ["Macro-F1 2 kelas", "Rata-rata F1 kelas Normal dan kelas AF/AFL."],
        ["Se / Sp / PPV", "Bagian AF/AFL yang terdeteksi / bagian Normal tanpa alarm / bagian alarm yang memang AF/AFL."],
        ["OOF, bias kelas", "Prediksi untuk pasien yang tidak ikut dilatih; penggeser ambang yang dipilih dari fold lain."],
        ["P(Δ > 0), bersarang", "Peluang lebih baik pada sampel pasien yang sama (≥ 0,95 lebih baik); bersarang = bias dipilih ulang di "
                                "setiap sampel bootstrap, uji paling ketat."],
    ], y, widths=[110, CW - 110])
    para(f"Data: {thou(P['n']['window'])} window 10 detik dari {P['n']['pasien']} pasien AFDB ({thou(P['n']['af1'])} AF/AFL, "
         f"{thou(P['n']['normal'])} Normal), R-peak dari detektor 50 Hz. Bootstrap pasien {thou(P['n_boot'])}×, distratifikasi, seed 2026.",
         y, "small")

    # ── Hal 4: hasil ──────────────────────────────────────────────────
    y = page("Hasil semua sistem", "Aturan 3 kelas lalu digabung | rata-rata 5 ulangan, CI 95% bootstrap pasien | baris hijau = v3.1")
    rows = [[n, ci(SG[n]["Macro-F1"]), ci(SG[n]["F1 AF1"]), ci(SG[n]["Se"]), ci(SG[n]["Sp"]), fnum(SG[n]["PPV"][0])] for n in URT]
    y = table(["Sistem", "Macro-F1", "F1 AF/AFL", "Se", "Sp", "PPV"], rows, y, widths=[110, 85, 85, 85, 85, CW - 450], bold_rows=[2])
    y = section("Pembanding: bias 2 kelas dipilih ulang", y)
    rows = [[n, ci(SU[n]["Macro-F1"]), fnum(SU[n]["Se"][0]), fnum(SU[n]["Sp"][0]),
             f"{fnum(SU[n]['bias']['median'], 2)} [{fnum(SU[n]['bias']['min'], 2)}…{fnum(SU[n]['bias']['maks'], 2)}]" if "bias" in SU[n] else "–"]
            for n in URT]
    y = table(["Sistem", "Macro-F1", "Se", "Sp", "Bias AF/AFL median [min…maks]"], rows, y, widths=[110, 95, 60, 60, CW - 325],
              bold_rows=[2])
    batas_g = [f"{n} ({pct(SG[n]['bias']['di_batas'])} fold)" for n in URT if SG[n].get("bias") and SG[n]["bias"]["di_batas"] > 0]
    batas_u = [n for n in URT if SU[n].get("bias") and SU[n]["bias"]["di_batas"] > 0]
    y = para("Bias aturan 3 kelas lalu digabung (cara sama dengan evaluasi 3 kelas v11), median AFIB / AFL: "
             + "; ".join(f"{n} {fnum(SG[n]['bias']['median'][0], 2)} / {fnum(SG[n]['bias']['median'][1], 2)}" for n in URT if SG[n].get("bias"))
             + ". Bias AFL yang jauh di bawah nol menekan tebakan AFL; aturan bias dipilih ulang tidak punya penekan ini (halaman 6).", y, "small")
    para("Batas grid (−7 atau +3) " + ("tidak tersentuh oleh aturan bias dipilih ulang" if not batas_u else f"tersentuh aturan ulang: {', '.join(batas_u)}")
         + ("; pada aturan gabung tersentuh oleh " + ", ".join(batas_g) + " (bias AFL −7, sama dengan evaluasi 3 kelas)." if batas_g else
            "; aturan gabung juga tidak menyentuhnya."), y, "small")

    # ── Hal 5: berpasangan ────────────────────────────────────────────
    y = page("Perbandingan berpasangan", "Sampel bootstrap pasien yang sama | aturan 3 kelas lalu digabung (dan aturan bias dipilih ulang)")
    y = chart("paired", y, 220)
    rows = []
    for n in [x for x in URT if x != "v3.1"]:
        a, u = SG[n]["vs_v31"], SU[n]["vs_v31"]
        bb = SG[n].get("vs_v1")
        rows.append([n, dci(a["Macro-F1"]), f"{fnum(a['Macro-F1'][3], 2)} {vonis(a['Macro-F1'][3])}",
                     f"{sg(a['Se'][0])} / {sg(a['Sp'][0])}", f"{sg(bb['Macro-F1'][0])} ({fnum(bb['Macro-F1'][3], 2)})" if bb else "–",
                     f"{sg(u['Macro-F1'][0])} ({fnum(u['Macro-F1'][3], 2)})"])
    y = table(["Sistem", "Δ vs v3.1 [CI]", "P, vonis", "Δ Se / Δ Sp", "Δ vs v1 (P)", "Aturan ulang: Δ vs v3.1 (P)"], rows, y,
              widths=[96, 104, 92, 80, 70, CW - 442])
    kelompok = {v: [n for n in URT if n != "v3.1" and vonis(SG[n]["vs_v31"]["Macro-F1"][3]) == v]
                for v in ("lebih baik", "tidak terbedakan", "lebih buruk")}
    callout("Intinya", " ".join(f"{v[0].upper() + v[1:]} dari v3.1: {', '.join(ns)}." for v, ns in kelompok.items() if ns)
            + " Pada model yang lebih baik, kenaikannya terutama dari spesifisitas (Δ Sp "
            + ", ".join(sg(SG[n]["vs_v31"]["Sp"][0]) for n in lulus) + " vs Δ Se "
            + ", ".join(sg(SG[n]["vs_v31"]["Se"][0]) for n in lulus) + "): lebih sedikit window Normal yang memicu alarm.", y)

    # ── Hal 6: alarm baru ─────────────────────────────────────────────
    y = page("Alarm baru pada ritme normal 80–95 bpm", "Uji stres sintetis dan window Normal asli yang teratur, dua aturan keputusan")
    y = chart("stres", y, 205)
    y = chart("teratur", y, 215)
    AL = P["afl_lambat"]
    asal = ", ".join(f"{n} dari {p}" for p, n in sorted(AL["per_pasien"].items(), key=lambda t: -t[1]))
    callout("Intinya",
            f"AFDB punya {AL['total']} window AFL yang teratur di {fnum(AL['bpm_p5_p95'][0], 0)}–{fnum(AL['bpm_p5_p95'][1], 0)} bpm "
            f"({asal}). Model belajar bahwa ritme teratur di sekitar itu bisa AFL. Dengan bias 3 kelas, AFL ditekan kuat (bias AFL median "
            f"{fnum(SG['v3.1']['bias']['median'][1], 2)} untuk v3.1), sehingga ritme Normal teratur 80–95 bpm hampir tidak pernah jadi "
            f"alarm. Saat bias dipilih ulang untuk 2 kelas, penekan itu hilang: {pct(SU[paling]['teratur_80_95'], 1)} window Normal teratur "
            f"80–95 bpm ({paling}) menjadi alarm, dan pada uji sintetis sampai {pct(sintetis_8590(SU, paling))} ritme 85–90 bpm. Aturan "
            "3 kelas lalu digabung menghindarinya dengan Macro-F1 yang hampir sama.", y, "#FFF4EC")

    # ── Hal 7: per pasien ─────────────────────────────────────────────
    y = page("Per pasien", "Aturan 3 kelas lalu digabung | sensitivitas AF/AFL dan bagian window Normal yang menjadi alarm, dijumlah 5 ulangan")
    y = chart("per_pasien", y, 420)
    pas, pp = P["pasien"], G["per_pasien"]
    al31 = [(v, p) for v, p in zip(pp["v3.1"]["alarm_normal"], pas) if v == v]
    se31 = [(v, p) for v, p in zip(pp["v3.1"]["se"], pas) if v == v]
    top_al = sorted(al31, reverse=True)[:3]
    para("Tanda \"–\" = pasien tidak punya window kelas itu. Alarm palsu terbanyak (v3.1): "
         + ", ".join(f"{p} {fnum(v, 2)} → {best} {fnum(pp[best]['alarm_normal'][pas.index(p)], 2)}" for v, p in top_al)
         + f". Sensitivitas terendah: {min(se31)[1]} ({fnum(min(se31)[0], 2)} untuk v3.1). Kesalahan terpusat di sedikit pasien, "
         "karena itu CI lebar.", y, "small")

    # ── Hal 8: butir 10 ───────────────────────────────────────────────
    y = page("Uji paling ketat (protokol butir 10)", "Bootstrap bersarang dan sensitivitas tanpa pasien 04936, kedua aturan")
    rows = []
    for n in URT:
        if n not in BG:
            continue
        g, u = BG[n], BU[n]
        rows.append([n, f"{fnum(g['titik'])} <font color='{GRAY}' size='7.2'>[{fnum(g['ci'][0], 2)}–{fnum(g['ci'][1], 2)}]</font>",
                     f"{sg(g['vs_v1']['delta'])} ({fnum(g['vs_v1']['p'], 2)})",
                     f"{sg(g['vs_v31']['delta'])} ({fnum(g['vs_v31']['p'], 2)})" if "vs_v31" in g else "–",
                     f"{fnum(u['vs_v1']['p'], 2)} / {fnum(u['vs_v31']['p'], 2) if 'vs_v31' in u else '–'}"])
    y = section("Bootstrap bersarang (bias dipilih ulang di setiap sampel)", y)
    y = table(["Sistem", "Gabung: Macro-F1 [CI]", "Gabung: vs v1 Δ (P)", "Gabung: vs v3.1 Δ (P)", "Ulang: P vs v1 / v3.1"], rows, y,
              widths=[105, 110, 100, 100, CW - 415], bold_rows=[1])
    y = section(f"Tanpa pasien 04936 ({P['n']['pasien_tanpa']} pasien, bias dihitung ulang)", y)
    rows = [[n, ci(TG[n]["Macro-F1"]), fnum(TG[n]["Sp"][0]),
             f"{sg(TG[n]['vs_v31']['Macro-F1'][0])} ({fnum(TG[n]['vs_v31']['Macro-F1'][3], 2)})" if "vs_v31" in TG[n] else "–",
             fnum(TU[n]["Macro-F1"][0])] for n in URT]
    y = table(["Sistem", "Gabung: Macro-F1", "Sp", "Gabung: vs v3.1 Δ (P)", "Ulang: Macro-F1"], rows, y,
              widths=[110, 110, 60, 120, CW - 400], bold_rows=[2])
    lulus_bs = [n for n in model if BG[n]["vs_v1"]["p"] >= .95 and BG[n]["vs_v31"]["p"] >= .95]
    lulus_tanpa = [n for n in model if TG[n]["vs_v31"]["Macro-F1"][3] >= .95]
    callout("Intinya", f"Pada uji bersarang, lebih baik dari v1 dan v3.1 sekaligus: {', '.join(lulus_bs) or 'tidak ada'} ({best}: P = "
            f"{fnum(BG[best]['vs_v1']['p'], 2)} dan {fnum(BG[best]['vs_v31']['p'], 2)}). Tanpa 04936 Macro-F1 semua sistem tetap "
            f"{rng([TG[n]['Macro-F1'][0] for n in URT])}, jadi untuk 2 kelas satu pasien AFL tidak lagi menentukan tingginya angka. "
            + ("Namun keunggulan atas v3.1 tidak lagi pasti tanpa 04936 (" + ", ".join(f"{n} P = {fnum(TG[n]['vs_v31']['Macro-F1'][3], 2)}"
                                                                          for n in lulus) + ")." if not lulus_tanpa else
               f"Keunggulan atas v3.1 bertahan tanpa 04936 untuk {', '.join(lulus_tanpa)}."), y)

    # ── Hal 9: takikardia & kesimpulan ────────────────────────────────
    y = page("Uji takikardia dan kesimpulan", "Window Normal asli per pita detak (aturan gabung) dan keputusan")
    pita = list(SG["v3.1"]["alarm_normal"])
    rows = [[n] + [pct(SG[n]["alarm_normal"][k], 1) for k in pita]
            + [amb(SG[n]["stres"]) if "stres" in SG[n] else (amb(SG[n]["stres_kontrol_v7"]) + " (kontrol v7)" if "stres_kontrol_v7" in SG[n] else "–")]
            for n in URT]
    y = table(["Sistem"] + [f"Normal {k} bpm" for k in pita] + ["Ambang sintetis (bpm)"], rows, y,
              widths=[96] + [62] * len(pita) + [CW - 96 - 62 * len(pita)], bold_rows=[2])
    y = para(f"Ambang sintetis = detak terendah di mana < 50% ritme teratur sintetis ditebak Normal (tanpa / ±10 / G10 / G20 ms). AFDB "
             f"hanya punya {P['n']['normal_per_pita']['≥ 120']} window Normal ≥ 120 bpm.", y, "small")
    y = section("Keputusan", y)
    y = table(["Hal", "Status", "Alasan"], [
        ["Laporkan Macro-F1 2 kelas berdampingan dengan 3 kelas", "<b>Disarankan</b>", "Sejalan dengan AF₁ di literatur; AFL dari "
         "interval RR saja tidak terpisah dari AF (v8–v12)"],
        ["Aturan keputusan 2 kelas", "<b>3 kelas lalu digabung</b>", "Macro-F1 setara, tanpa alarm baru pada ritme normal teratur "
         "80–95 bpm"],
        [f"{best} sebagai kandidat keluaran 2 kelas", "Kandidat", f"Lebih baik dari v3.1 (P = {fnum(SG[best]['vs_v31']['Macro-F1'][3], 2)}); "
         f"bersarang vs v1 P = {fnum(BG[best]['vs_v1']['p'], 2)}; tanpa 04936 vs v3.1 P = "
         f"{fnum(TG[best]['vs_v31']['Macro-F1'][3], 2)} (tidak pasti); belum dikemas"],
        ["Mengganti paket produksi", "Belum", "Butuh keputusan produk (2 vs 3 kelas) dan solusi alarm takikardia ≥ 120 bpm"],
    ], y, widths=[180, 100, CW - 280])
    y = section("Batasan", y)
    for txt in ["Validasi silang internal AFDB (23 pasien); angka literatur (F1 AF₁ 0,89–0,92) berasal dari data uji eksternal, jadi "
                "tidak sebanding langsung.",
                "Aturan \"3 kelas lalu digabung\" ditambahkan setelah melihat hasil uji stres; \"terbaik\" dipilih setelah melihat hasil. "
                "Karena itu uji bersarang yang dipakai untuk vonis.",
                "Model dilatih untuk 3 kelas; model yang dilatih langsung untuk 2 kelas tidak diuji. Metrik episode (butir 7) belum dihitung."]:
        y = para(f"• {txt}", y, gap=5)

    # ── Hal 10: lampiran ──────────────────────────────────────────────
    y = page("Lampiran teknis", "Lingkungan, file, dan cara mereproduksi")
    ln = P["lingkungan"]
    y = table(["Komponen", "Nilai"], [
        ["Data", f"cache/dataset_v2_detector.npz: {thou(P['n']['window'])} window, {P['n']['pasien']} pasien (dicek sidik_jari.json)"],
        ["Pembagian", "cache/oof_robust/detector_e30_r5_s2026/folds.npz: 5 ulangan × 5 fold pasien, seed 2026"],
        ["Model", "v1 & v3.1: OOF v3; BiGRU/TCN/XGBoost: cache/v11/&lt;model&gt;/&lt;setelan&gt;/ (fold diperiksa = fold resmi)"],
        ["Bias & bootstrap", "grid −7…+3 langkah 0,25, cross-fit antar fold; ulang = 1 bias AF/AFL, gabung = 2 bias (AFIB, AFL) "
                             "seperti 3 kelas; 2.000 sampel pasien per strata"],
        ["Pemeriksaan otomatis", "Macro-F1 2 kelas v3.1 dan TCN bawaan untuk kedua aturan dihitung ulang dari OOF dan harus sama "
                                 "dengan JSON; Macro-F1 bersarang dengan bobot semua-satu harus sama dengan cross-fit biasa (semua sistem); "
                                 "notebook mencocokkan angka yang dikutip di Kesimpulan dengan hasil run"],
        ["Lingkungan", ", ".join(f"{k} {v}" for k, v in ln["pustaka"].items() if v) + f"; commit {ln['commit']} + perubahan "
                       "v11/v11.1 yang belum di-commit"],
    ], y, widths=[110, CW - 110])
    y = section("Mereproduksi", y)
    para("Dari root repo: <font name='Courier' size='8.3'>python tools/run_notebook.py v11.1_dua_kelas/dua_kelas.ipynb</font> (±45 menit "
         "CPU; menulis cache/v11.1/dua_kelas.json). Lalu dari folder v11.1_dua_kelas: <font name='Courier' size='8.3'>python "
         "laporan/build_report.py --extract</font> dan <font name='Courier' size='8.3'>python laporan/build_report.py</font>.", y)
    c.save()
    print(f"OK {PDF} ({state['page']} halaman)")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    elif "--grafik" in sys.argv:
        grafik()
    else:
        build()
