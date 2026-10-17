"""Laporan PDF v8 (data eksternal Chapman–Shaoxing & Ningbo).

Dua tahap (pola sama dengan v3–v7):
  1. --extract  (Python dengan numpy/torch/matplotlib): hitung ulang lewat chapman.evaluate() dari cache/v8 (hasil run Kaggle
     yang dibawa pulang), cocokkan dengan ../cache/v8/ringkasan.json dari notebook, buat grafik → laporan/_build/.
     Di PC tim, evaluasi ulang juga memuat v1 (OOF cache v3) yang tidak ada di Kaggle.
  2. tanpa argumen (Python dengan ReportLab): rakit laporan/laporan_chapman_shaoxing.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # = v8_chapman_shaoxing/
REPO = HERE.parent
TMP = HERE / "laporan/_build"
OUT = HERE / "laporan/laporan_chapman_shaoxing.pdf"
SUMMARY = REPO / "cache/v8/ringkasan.json"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402

TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
B8 = "afdb_chapman_bersih"
MCOL = {"v1_paket": "#C4CED8", "v2_paket": "#A9B5C2", "v31_paket": "#8A97A6", "v31": "#8A97A6", "afdb": "#eb6834",
        "afdb_chapman": "#1baf7a", "chapman": "#A9B5C2", B8: "#2a78d6"}
MMARK = {"v1_paket": "x", "v2_paket": "P", "v31_paket": "s", "v31": "s", "afdb": "^", "afdb_chapman": "D", "chapman": "v", B8: "o"}
SHORT = {"v1": "v1", "v1_paket": "paket v1", "v2_paket": "paket v2", "v31_paket": "paket v3.1", "v31": "v31 (resep v3.1)",
         "afdb": "afdb (= v5b)", "afdb_chapman": "8a: + Chapman", "chapman": "8b: Chapman saja", B8: "8c: + Chapman tanpa AFL"}


def extract():
    sys.path.insert(0, str(HERE))
    import chapman as C

    TMP.mkdir(parents=True, exist_ok=True)
    res = C.evaluate()
    nb = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for k, r in nb["afdb"]["rows"].items():                  # semua yang dihitung di Kaggle harus sama persis
        for m in ("Macro-F1", "AFL F1", "AFL R", "Normal R"):
            assert abs(res["afdb"]["rows"][k][m][0] - r[m][0]) < 1e-12, (k, m)
    for k, p in nb["afdb"]["pairs"].items():
        assert abs(res["afdb"]["pairs"][k]["Macro-F1"][3] - p["Macro-F1"][3]) < 1e-12, k
    for k, r in nb["chapman"]["metrics"].items():
        assert abs(res["chapman"]["metrics"][k]["Macro-F1"][0] - r["Macro-F1"][0]) < 1e-12, k
    for k in nb["chapman"]["st_all"]:
        if k != "n":
            assert abs(res["chapman"]["st_all"][k][0] - nb["chapman"]["st_all"][k][0]) < 1e-12, k
    assert res["threshold"] == nb["threshold"]
    assert res["chapman"]["summary"] == nb["chapman"]["summary"]
    res["kaggle"] = {"evaluasi_lokal_menambah": [k for k in res["afdb"]["rows"] if k not in nb["afdb"]["rows"]]}
    (TMP / "tables.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    charts(res)


def charts(res=None):
    """Grafik dari hasil extract(); dengan --charts digambar ulang dari tables.json yang sudah diverifikasi."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter

    sys.path.insert(0, str(HERE))
    import chapman as C
    if res is None:
        res = json.loads((TMP / "tables.json").read_text(encoding="utf-8"))

    for style in ("regular", "bold"):
        font_manager.fontManager.addfont(font_path(style))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path("regular")).get_name(),
                         "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#C4CED8", "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": GRAY, "ytick.color": INK, "figure.facecolor": "white"})
    pctf = PercentFormatter(1.0, decimals=0)

    def save(fig, name):
        fig.savefig(TMP / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=.08, facecolor="white")
        plt.close(fig)

    # 1. label AFL: detak vs keteraturan per sumber
    ch = C.load_chapman()
    nbo = C.is_ningbo(ch["record"])
    z = np.load(C.DATA_STD)
    Mz, Iz = z["M"], z["X"][:, :, 0]
    mean = (Iz * Mz).sum(1) / Mz.sum(1)
    hra, cva = 60000 / mean, np.sqrt((((Iz - mean[:, None]) ** 2) * Mz).sum(1) / Mz.sum(1)) / mean
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), sharey=True)
    panels = (("AFL Ningbo (label tidak andal)", (ch["rhythm"] == "AFL") & nbo, "#eb6834", "^"),
              ("AFL Shaoxing", (ch["rhythm"] == "AFL") & ~nbo, "#2a78d6", "o"))
    for ax, (title, m, col, mk) in zip(axes, panels):
        idx = rng.permutation(np.flatnonzero(m))[:1500]
        ax.scatter(ch["hr"][idx], ch["cv"][idx], s=5, alpha=.4, color=col, marker=mk, lw=0)
        ax.set_title(f"{title} ({m.sum():,})".replace(",", "."), loc="left", fontsize=8.6, fontweight="bold")
    for c, col, mk, lab in ((1, "#8A97A6", "s", "AFIB"), (2, "#1baf7a", "D", "AFL")):
        m = z["y"] == c
        idx = rng.permutation(np.flatnonzero(m))[:1500]
        axes[2].scatter(hra[idx], cva[idx], s=5, alpha=.4, color=col, marker=mk, lw=0, label=f"{lab} AFDB")
    axes[2].set_title("AFDB (pembanding)", loc="left", fontsize=8.6, fontweight="bold")
    axes[2].legend(frameon=False, fontsize=7.8, markerscale=2.5, loc="upper right")
    for ax in axes:
        ax.set_xlim(40, 180); ax.set_ylim(0, .4); ax.grid(color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
        ax.set_xlabel("detak (bpm)", fontsize=8.5); ax.tick_params(labelsize=8)
    axes[0].set_ylabel("CV IBI (makin tinggi makin tidak teratur)", fontsize=8)
    fig.tight_layout(); save(fig, "label_afl")

    # 2. sinus takikardia asli: alarm per pita detak
    keys = ["v1_paket", "v2_paket", "v31_paket", "afdb_chapman", B8]
    bands = list(res["chapman"]["st_bands"])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(bands))
    for k in keys:
        if k not in res["chapman"]["st_all"]:
            continue
        v = [res["chapman"]["st_bands"][b][k][0] for b in bands]
        ax.plot(x, v, marker=MMARK[k], color=MCOL[k], ms=6, lw=2 if k in (B8, "v31_paket") else 1.4, label=SHORT[k])
    ax.set_xticks(x, [f"{b.replace('–∞', '+')} bpm\n({res['chapman']['st_bands'][b]['n']:,} pasien)".replace(",", ".")
                      for b in bands], fontsize=8)
    ax.set_ylabel("ST ditebak AFIB/AFL", fontsize=8.5); ax.yaxis.set_major_formatter(pctf); ax.set_ylim(0, 1.03)
    ax.grid(axis="y", color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.1), ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, "st_bands")

    # 3. uji stres sintetis
    skeys = ["v31_paket", "afdb", "afdb_chapman", B8]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.4), sharey=True)
    for ax, j in zip(axes, res["jitters"]):
        ax.axvspan(100, 170, color="#F6F7F9", zorder=0)
        for k in skeys:
            ax.plot(res["bpms"], [t[0] for t in res["stress"][k][j]], marker=MMARK[k], color=MCOL[k], ms=3, lw=1.4,
                    label=SHORT[k])
        ax.set_title(j, loc="left", fontsize=8.6, fontweight="bold")
        ax.set_xticks([60, 90, 120, 150]); ax.tick_params(labelsize=8)
        ax.grid(color="#E6EBF0", lw=.8); ax.set_axisbelow(True); ax.yaxis.set_major_formatter(pctf)
        ax.set_xlabel("bpm", fontsize=8.5)
    axes[0].set_ylabel("ditebak Normal", fontsize=8.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.12), ncol=4,
               frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, "stress")
    print(f"OK tables.json + 3 grafik → {TMP}")


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
        "table": ParagraphStyle("table", fontName="Arial", fontSize=8.3, leading=11, textColor=HexColor(INK)),
        "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8.2, leading=11, textColor=white),
    }
    c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
    c.setTitle("Laporan v8 - Data eksternal Chapman-Shaoxing")
    c.setAuthor("Analisis v8_chapman_shaoxing/chapman_kaggle.ipynb")
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

    def thou(v):
        return f"{round(v):,}".replace(",", ".")

    def ci(v, nd=3):
        return f"{fnum(v[0], nd)} <font color='{GRAY}' size='7.2'>[{fnum(v[1], 2)}–{fnum(v[2], 2)}]</font>"

    def cip(v):
        return f"{pct(v[0])} <font color='{GRAY}' size='7.2'>[{pct(v[1])}–{pct(v[2])}]</font>"

    def verdict(p):
        return "lebih baik" if p >= .95 else "lebih buruk" if p <= .05 else "tidak terbedakan"

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
        c.drawString(M + 43, H - 42, "SIHEDAF  /  V8 DATA EKSTERNAL CHAPMAN–SHAOXING")
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", 21); c.drawString(M, H - 79, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 99, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, "17 Oktober 2026 | Sumber: v8_chapman_shaoxing/chapman_kaggle.ipynb (run Colab), cache/v8/")
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
              ("LEFTPADDING", (0, 0), (-1, -1), 4.5), ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
              ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
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

    A, C_, TH = d["afdb"], d["chapman"], d["threshold"]
    R, P, PA = A["rows"], A["pairs"], A["pairs_ablasi"]
    st, stb, met = C_["st_all"], C_["st_bands"], C_["metrics"]
    rh = C_["rhythm"]
    pat = A["afl_patients"]
    i08 = pat.index("08378")
    rec = A["afl_recall"]
    th = lambda k: "/".join(str(TH[k][j]) for j in d["jitters"])
    alarm = lambda g, k: rh[g][k][1] + rh[g][k][2]
    hi = lambda b: b.replace("–∞", "+")

    # ── Hal 1: ringkasan ──
    y = page("Laporan v8: data eksternal Chapman–Shaoxing", "42 ribu pasien dari luar AFDB: sinus takikardia asli dan lebih banyak AFIB/AFL")
    cardw = (CW - 24) / 4
    cards = ((pct(st["v31_paket"][0]), "ST asli → alarm (paket v3.1)"),
             (pct(st[B8][0]), "ST asli → alarm (8c)"),
             (f"{fnum(rec[B8][i08], 2)}", f"recall AFL 08378, 8c (v31: {fnum(rec['v31'][i08], 2)})"),
             (f"{TH[B8]['tanpa variasi']} bpm", "ambang Normal 8c (v3.1: 115)"))
    for i, (val, lab) in enumerate(cards):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 88, cardw, 88, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 20); c.drawString(x + 11, y - 37, val)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 7.5); c.drawString(x + 11, y - 61, lab)
    y -= 105
    y = callout("Jawaban singkat",
                f"<b>Masalah takikardia terbukti pada data nyata.</b> Paket v3.1 membunyikan alarm AFIB/AFL pada "
                f"{pct(st['v31_paket'][0])} dari {thou(st['n'])} pasien sinus takikardia, dan {pct(stb['120–130']['v31_paket'][0])}–"
                f"{pct(stb['130–140']['v31_paket'][0])} pada detak ≥ 120 bpm. Menambah sinus takikardia asli ke data latih (8c) menurunkan "
                f"alarm itu ke {pct(st[B8][0])}, tanpa Macro-F1 AFDB turun secara terukur (P = {fnum(P[B8]['Macro-F1'][3], 2)}). "
                f"Harganya: AFL 2:1 dan SVT ikut disebut Normal (08378 {fnum(rec['v31'][i08], 2)} → {fnum(rec[B8][i08], 2)}; "
                f"{pct(rh['SVT'][B8][0])} SVT disebut Normal). Ini batas interval RR. <b>Paket produksi tetap v3.1.</b> "
                "Label AFL bagian Ningbo tidak andal dan tidak boleh dipakai.", y)
    y = section("Hipotesis dan hasilnya", y)
    y = table(["Varian", "Data latih", "Hasil", "Status"], [
        ["v31 (resep v3.1)", "AFDB, bobot penuh", f"Macro-F1 {fnum(R['v31']['Macro-F1'][0])}", "referensi"],
        ["afdb (= v5b)", "AFDB, bobot lunak", f"Δ {sgn(P['afdb']['Macro-F1'][0])} (P = {fnum(P['afdb']['Macro-F1'][3], 2)})", "setara"],
        ["8a (utama)", "AFDB + semua label Chapman", f"AFDB {fnum(R['afdb_chapman']['Macro-F1'][0])}; 08378 → {fnum(rec['afdb_chapman'][i08], 2)}",
         "<b>Ditolak</b> (label AFL Ningbo)"],
        ["8b", "Chapman saja", f"AFDB {fnum(R['chapman']['Macro-F1'][0])}; Normal R {fnum(R['chapman']['Normal R'][0])}", "<b>Ditolak</b>"],
        ["8c (setelah run pertama)", "AFDB + Chapman tanpa label AFL",
         f"alarm ST {pct(st['v31_paket'][0])} → {pct(st[B8][0])}; AFDB {fnum(R[B8]['Macro-F1'][0])}; AFL 2:1 hilang", "<b>Sebagian</b>"],
    ], y, [100, 125, CW - 315, 90], bold_rows=[5])
    y = section("Kriteria kandidat produksi untuk 8c (ditetapkan sebelum hasil)", y)
    drop = rec["v31"][i08] - rec[B8][i08]
    afl_r = met[B8]["AFL R"][0] >= met["v31_paket"]["AFL R"][0]
    y = table(["Kriteria", "Hasil", "Status"], [
        ["AFDB: Macro-F1 tidak lebih buruk dari v31", f"P = {fnum(P[B8]['Macro-F1'][3], 2)}", "lolos" if P[B8]["Macro-F1"][3] > .05 else "<b>tidak lolos</b>"],
        ["AFDB: tidak ada pasien AFL turun ≥ 0,20", f"08378 {fnum(rec['v31'][i08], 2)} → {fnum(rec[B8][i08], 2)}", "lolos" if drop < .2 else "<b>tidak lolos</b>"],
        ["Chapman: alarm ST ≤ separuh paket v3.1", f"{pct(st[B8][0])} vs {pct(st['v31_paket'][0])}", "lolos" if st[B8][0] <= st['v31_paket'][0] / 2 else "<b>tidak lolos</b>"],
        ["Chapman: recall AFL tidak lebih rendah dari paket", f"{fnum(met[B8]['AFL R'][0], 2)} vs {fnum(met['v31_paket']['AFL R'][0], 2)}", "lolos" if afl_r else "<b>tidak lolos</b>"],
        ["Uji stres: ambang ≥ 135 bpm di keempat variasi", th(B8) + " bpm", "lolos" if min(TH[B8].values()) >= 135 else "<b>tidak lolos</b>"],
    ], y, [215, CW - 305, 90])

    # ── Hal 2: data & label AFL ──
    y = page("Data Chapman dan masalah label AFL", "PhysioNet ecg-arrhythmia 1.0.0, lead II, front-end produksi, satu window 10 detik per pasien")
    sm = C_["summary"]
    per = sm["per ritme"]
    rows = [[g, thou(v["pasien"]), thou(v["window valid"]), "AFL (hanya Shaoxing andal)" if g == "AFL" else v["kelas"],
             fnum(v["detak median"], 0) + " bpm" if v["detak median"] else "—"]
            for g, v in per.items() if v["pasien"] >= 100]
    y = table(["Ritme", "Pasien", "Window valid", "Kelas SIHEDAF", "Detak median"], rows, y, [80, 90, 100, 120, CW - 390])
    y = para(f"Dari {thou(sm['pasien'])} pasien, {thou(sm['window valid'])} punya window 10 s yang lolos cek kualitas. Pasien dengan "
             "lebih dari satu kode ritme (mis. SA+SB) dan ritme di luar tiga kelas (SVT, AT, AVRT) tidak dipakai untuk training, tetapi "
             "prediksinya tetap dilaporkan.", y, "small")
    y = chart("label_afl", y, 165)
    lc = C_["afl_label_check"]
    callout("Label AFL bagian Ningbo tidak bisa dipakai",
            f"Bagian Ningbo (record di atas JS10646) tidak punya satu pun AFIB, tetapi punya {thou(lc['AFL (Ningbo)']['pasien'])} "
            f"\"AFL\" dengan detak median {fnum(lc['AFL (Ningbo)']['detak median'], 0)} bpm dan {pct(lc['AFL (Ningbo)']['CV > 0,10'])} tidak "
            f"teratur. Itu ciri AFIB (AFIB AFDB: {fnum(lc['AFIB (AFDB)']['detak median'], 0)} bpm, {pct(lc['AFIB (AFDB)']['CV > 0,10'])} tidak teratur), "
            f"bukan AFL (AFL AFDB: {fnum(lc['AFL (AFDB)']['detak median'], 0)} bpm, {pct(lc['AFL (AFDB)']['CV > 0,10'])}). Kemungkinan besar "
            "\"AF\" (atrial fibrillation) di Ningbo dipetakan ke kode SNOMED atrial flutter. Model 8a yang dilatih dengan label ini "
            "kehilangan deteksi AFL di AFDB. Metrik 3 kelas Chapman di laporan ini memakai AFL Shaoxing saja "
            f"({thou(C_['n_metrik']['AFL'])} pasien).", y, "#FFF4EC")

    # ── Hal 3: sinus takikardia asli ──
    y = page("Sinus takikardia asli", f"{thou(st['n'])} pasien ST Chapman; tidak satu pun dipakai untuk melatih paket")
    y = chart("st_bands", y, 185)
    keys = ["v1_paket", "v2_paket", "v31_paket", "afdb_chapman", B8]
    rows = [[f"ST {hi(b)} bpm", thou(stb[b]["n"])] + [cip(stb[b][k]) for k in keys] for b in stb]
    rows.append(["<b>ST semua</b>", thou(st["n"])] + [cip(st[k]) for k in keys])
    rows.append(["SR/SB/SA (sinus lain)", thou(C_["sinus_alarm"]["n"])] + [cip(C_["sinus_alarm"][k]) for k in keys])
    rows.append(["SVT → disebut Normal", thou(rh["SVT"]["n"])] + [pct(rh["SVT"][k][0]) for k in keys])
    y = table(["Kelompok", "Pasien"] + [SHORT[k] for k in keys], rows, y, [95, 45] + [(CW - 140) / 5] * 5, bold_rows=[len(stb) + 1])
    callout("Yang terlihat",
            f"Semua paket berbasis AFDB (v1, v2, v3.1) membunyikan alarm pada hampir semua sinus takikardia ≥ 120 bpm. Ini menegaskan "
            f"temuan sintetis v3.2 dengan data nyata. Model yang ikut melihat sinus takikardia asli (8a, 8c) menurunkan alarm ke "
            f"{pct(st['afdb_chapman'][0])} dan {pct(st[B8][0])}. Sisa alarm terkumpul di 140–150 bpm ({pct(stb['140–150'][B8][0])}), daerah "
            f"AFL 2:1. Tetapi ritme cepat teratur yang bukan sinus ikut disebut Normal: {pct(rh['SVT'][B8][0])} SVT oleh 8c, padahal paket "
            f"v3.1 menyebut {pct(rh['SVT']['v31_paket'][2])} SVT sebagai AFL (kelasnya salah, tetapi alarm tetap berbunyi).", y)

    # ── Hal 4: perbandingan di AFDB ──
    y = page("Perbandingan dengan versi sebelumnya (AFDB)", "Protokol v3: window detektor 50 Hz, 5 ulangan × 5 fold pasien, bias cross-fit, CI bootstrap pasien")
    order = [k for k in ("v1", "v31", "afdb", "afdb_chapman", "chapman", B8) if k in R]
    E = A["episodes"]
    y = table(["Model", "Macro-F1", "AFL F1", "AFL recall", "Normal recall", "AFL palsu / jam", "episode AFL"],
              [[SHORT[k], ci(R[k]["Macro-F1"]), ci(R[k]["AFL F1"]), ci(R[k]["AFL R"]), fnum(R[k]["Normal R"][0]),
                fnum(E[k]["alarm/jam"][0], 2), pct(E[k]["deteksi episode"][0])] for k in order],
              y, [92, 84, 84, 84, 55, 55, CW - 454], bold_rows=[order.index(B8) + 1])
    y = para("v1 = sistem sebelum revisi (OOF dari cache v3 di PC tim). v31 = resep v3.1 yang dilatih ulang di Colab "
             f"(Macro-F1 {fnum(R['v31']['Macro-F1'][0])}; di cache v3 0,832, selisih karena GPU berbeda). Semua Δ dihitung di lingkungan "
             "yang sama. <i>OOF</i> = prediksi untuk pasien yang tidak ikut dilatih; <i>CI</i> = rentang 95% dari 2.000 kali mengambil "
             "ulang pasien.", y, "small")
    rows = []
    for k in ("afdb_chapman", "chapman", B8):
        for ref, PP in (("v31", P[k]), ("afdb", PA[k])):
            rows.append([SHORT[k] if ref == "v31" else "", ref] +
                        [f"{sgn(PP[m][0])} <font color='{GRAY}' size='7.2'>(P = {fnum(PP[m][3], 2)})</font><br/>{verdict(PP[m][3])}"
                         for m in ("Macro-F1", "AFL R", "Normal R")])
    y = table(["Varian", "vs", "Δ Macro-F1", "Δ AFL recall", "Δ Normal recall"], rows, y,
              [120, 40, (CW - 160) / 3, (CW - 160) / 3, (CW - 160) / 3])
    callout("Cara membaca",
            f"8c setara dengan v31 di Macro-F1 (P = {fnum(P[B8]['Macro-F1'][3], 2)}), tetapi sedikit lebih buruk dari ablasi afdb, yaitu "
            f"resep yang sama tanpa Chapman (P = {fnum(PA[B8]['Macro-F1'][3], 2)}). Penurunannya berasal dari AFL: recall "
            f"{fnum(R['afdb']['AFL R'][0])} → {fnum(R[B8]['AFL R'][0])}. 8a dan 8b lebih buruk di hampir semua metrik.", y)

    # ── Hal 5: per pasien & Chapman 3 kelas ──
    y = page("Per pasien AFL dan metrik Chapman", "AFDB: recall AFL per pasien; Chapman: label andal saja (AFL Shaoxing)")
    order_p = [k for k in ("v1", "v31", "afdb", "afdb_chapman", B8) if k in rec]
    y = table(["Recall AFL per pasien"] + [SHORT[k] for k in order_p],
              [[p] + [fnum(rec[k][i], 2) for k in order_p] for i, p in enumerate(pat)],
              y, [110] + [(CW - 110) / len(order_p)] * len(order_p), bold_rows=[i08 + 1])
    fp = A["fp"]
    y = table(["Sumber AFL palsu & bias"] + [SHORT[k] for k in order_p], [
        ["AFL palsu / ulangan"] + [thou(fp[k]["total"]) for k in order_p],
        ["… dari pasien tanpa AFL"] + [thou(fp[k]["pasien_lain"]) for k in order_p],
        ["bias median (N; AFIB; AFL)"] + ["—" if k == "v1" else "; ".join(fnum(v, 2) for v in A["bias"][k]["median"]) for k in order_p]],
        y, [110] + [(CW - 110) / len(order_p)] * len(order_p))
    mk = ["v1_paket", "v31_paket", "v31", "afdb_chapman", B8]
    cp = C_["pairs"]["v31_paket"]
    y = table(["Chapman (label andal)", "Macro-F1", "AFL recall", "AFIB recall", "Normal recall", "Δ Macro-F1 vs paket v3.1"],
              [[SHORT[k], ci(met[k]["Macro-F1"]), ci(met[k]["AFL R"]), fnum(met[k]["AFIB R"][0]), fnum(met[k]["Normal R"][0]),
                "—" if k == "v31_paket" else f"{sgn(cp[k]['Macro-F1'][0])} ({verdict(cp[k]['Macro-F1'][3])})"] for k in mk],
              y, [100, 85, 85, 60, 60, CW - 390], bold_rows=[len(mk)])
    ab = C_["afl_bands"]
    y = table(["AFL Shaoxing per detak", "Pasien"] + [SHORT[k] for k in ("v31_paket", "afdb_chapman", B8)],
              [[f"{hi(b)} bpm", ab[b]["n"]] + [pct(ab[b][k]) if ab[b][k] is not None else "—" for k in ("v31_paket", "afdb_chapman", B8)]
               for b in ab], y, [130, 60] + [(CW - 190) / 3] * 3)
    para(f"Macro-F1 Chapman rendah untuk semua model karena hanya {thou(C_['n_metrik']['AFL'])} AFL dibanding "
         f"{thou(C_['n_metrik']['Normal'])} Normal: sedikit alarm palsu saja sudah menekan presisi AFL. 8c lebih baik di Normal dan "
         "Macro-F1, tetapi deteksi AFL cepat (≥ 130 bpm) turun tajam.", y, "small")

    # ── Hal 6: uji stres & sensitivitas ──
    y = page("Uji stres takikardia dan sensitivitas", "Protokol butir 9 (ritme teratur sintetis v3.2) dan butir 10 (tanpa pasien 04936)")
    y = chart("stress", y, 175)
    sk = ["v1_paket", "v31_paket", "v31", "afdb", "afdb_chapman", "chapman", B8]
    y = table(["Model"] + [f"Ambang: {j}" for j in d["jitters"]],
              [[SHORT[k]] + [f"{TH[k][j]} bpm" for j in d["jitters"]] for k in sk if k in TH],
              y, [120] + [(CW - 120) / 4] * 4, bold_rows=[len([k for k in sk if k in TH])])
    rows = []
    for lab_s, s in A["sens"].items():
        for k in ("v31", "afdb", B8):
            r = s["rows"][k]
            rows.append([lab_s if k == "v31" else "", SHORT[k], ci(r["Macro-F1"]), ci(r["AFL F1"]), ci(r["AFL R"]),
                         "—" if k == "v31" else fnum(s["pairs_v31_minus"][k][3], 2)])
    y = table(["Data", "Model", "Macro-F1", "AFL F1", "AFL recall", "P(v31 > model)"], rows, y, [70, 100, 90, 90, 90, CW - 440])
    callout("Cara membaca",
            f"Ambang = detak terendah di mana kurang dari separuh ritme teratur sintetis ditebak Normal. 8c menaikkannya dari 115 ke "
            f"{TH[B8]['tanpa variasi']} bpm, tetapi pada variasi sd 20 ms tetap {TH[B8]['Gauss sd 20 ms']} bpm. Lekukan di ±140–155 bpm "
            "adalah daerah AFL 2:1, tempat model 8a/8c tetap membunyikan alarm; di 165–170 bpm mereka kembali menebak Normal karena "
            "sinus takikardia Chapman mencapai detak itu. Tanpa 04936, 8c tetap tidak terbedakan dari v31 "
            f"(P = {fnum(A['sens']['tanpa 04936']['pairs_v31_minus'][B8][3], 2)}).", y)

    # ── Hal 7: kesimpulan ──
    y = page("Kesimpulan dan langkah berikutnya", "Yang terbukti, harganya, dan yang belum bisa dijawab")
    y = table(["Hal", "Keputusan / status"], [
        ["Paket produksi", "Tetap v3.1; v8 tidak dikemas"],
        ["Masalah takikardia", f"<b>Terbukti pada data nyata</b>: paket v3.1 alarm {pct(st['v31_paket'][0])} pada sinus takikardia, "
                               f"{pct(stb['120–130']['v31_paket'][0])}–{pct(stb['150–∞']['v31_paket'][0])} pada ≥ 120 bpm"],
        ["8c (+ sinus takikardia asli)", f"Alarm ST {pct(st[B8][0])}, AFDB setara v31; tetapi AFL 2:1 dan SVT disebut Normal. Tidak lolos kriteria"],
        ["Data Chapman", "Normal, AFIB, dan AFL Shaoxing dapat dipakai; <b>AFL Ningbo tidak</b>"],
        ["Batas interval RR", "Terkonfirmasi: ritme cepat yang teratur (ST, AFL 2:1, SVT) tidak bisa dipisahkan"],
    ], y, [150, CW - 150])
    y = section("Langkah berikutnya", y)
    for txt in ["<b>v9: cabang bentuk gelombang lead II 50 Hz</b> yang dilatih dengan AFDB + Chapman (label andal). Gelombang P "
                "(sinus) dan gelombang flutter (AFL) hanya terlihat di bentuk gelombang. v6 gagal karena hanya 23 pasien; sekarang ada "
                "±42 ribu. Model ini jauh lebih besar, jadi GPU yang lebih kuat (mis. Colab Pro) akan terasa manfaatnya.",
                "Pertimbangkan kelas keempat atau label produk \"takikardia teratur, perlu dicek\" untuk ritme yang memang tidak bisa "
                "dipisahkan dari interval RR."]:
        y = para(f"• {txt}", y, gap=5)
    y = section("Batasan", y)
    for txt in ["Chapman berisi pasien rumah sakit di Tiongkok dengan satu rekaman 10 detik per pasien; perangkat SIHEDAF merekam "
                "orang yang beraktivitas. Lead II belum tentu sama dengan lead perangkat.",
                "Label AFL yang andal hanya 416 pasien (Shaoxing), dan AFL-nya lebih tidak teratur daripada AFL di AFDB.",
                "8c ditambahkan setelah melihat run pertama (hipotesisnya ditulis sebelum hasil 8c ada).",
                "Reproduksi: notebook Kaggle/Colab-first di folder v8 (lihat README); laporan: "
                "<font name='Courier' size='8.3'>laporan/build_report.py --extract</font> lalu tanpa argumen."]:
        y = para(f"• {txt}", y, gap=5)
    c.save()
    print(f"OK {OUT} ({state['page']} halaman)")


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else charts() if "--charts" in sys.argv else build()
