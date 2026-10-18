"""Laporan PDF v10 (data wearable Icentia11k untuk cabang bentuk gelombang EKG).

Dua tahap (pola sama dengan v3–v9):
  1. --extract : evaluasi ulang dari cache/v10 + cache/v9 + cache/v8 (hasil run Kaggle yang dibawa pulang), cocokkan dengan
     ../cache/v10/ringkasan.json dari notebook, tambah eksplorasi AFDB & ringkasan data, lalu grafik → laporan/_build/.
     --charts menggambar ulang grafik saja.
  2. tanpa argumen (ReportLab): rakit laporan/laporan_icentia_wearable.pdf.
"""
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # = v10_icentia_wearable/
REPO = HERE.parent
TMP = HERE / "laporan/_build"
OUT = HERE / "laporan/laporan_icentia_wearable.pdf"
SUMMARY = REPO / "cache/v10/ringkasan.json"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402

TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
MCOL = {"v31_paket": "#8A97A6", "v31": "#8A97A6", "rr": "#eb6834", "v9_dual": "#1baf7a", "dual": "#2a78d6",
        "dual_tanpa_afdb": "#172B43", "v9_rr": "#C4CED8"}
MMARK = {"v31_paket": "s", "v31": "s", "rr": "^", "v9_dual": "D", "dual": "o", "dual_tanpa_afdb": "v", "v9_rr": "x"}
SHORT = {"v1": "v1", "v31_paket": "paket v3.1", "v31": "v31 (resep v3.1)", "v9_rr": "kontrol rr v9", "v9_dual": "9a (v9)",
         "rr": "kontrol rr", "dual": "10a: EKG + RR", "dual_tanpa_afdb": "10b: tanpa AFDB"}


def extract():
    sys.path.insert(0, str(HERE))
    import icentia as I
    TMP.mkdir(parents=True, exist_ok=True)
    res = I.evaluate()
    nb = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for k, r in nb["afdb"]["rows"].items():
        for m in ("Macro-F1", "AFL F1", "AFL R", "Normal R"):
            assert abs(res["afdb"]["rows"][k][m][0] - r[m][0]) < 1e-12, ("afdb", k, m)
    for grp in ("pairs", "pairs_9a", "pairs_9rr"):
        for k, p in nb["afdb"][grp].items():
            assert abs(res["afdb"][grp][k]["Macro-F1"][3] - p["Macro-F1"][3]) < 1e-12, (grp, k)
    for k, r in nb["icentia"]["rows"].items():
        for m in ("Macro-F1", "AFL R", "Normal R"):
            assert abs(res["icentia"]["rows"][k][m][0] - r[m][0]) < 1e-12, ("icentia", k, m)
    for ref, d in nb["icentia"]["pairs"].items():
        for k, p in d.items():
            assert abs(res["icentia"]["pairs"][ref][k]["Macro-F1"][3] - p["Macro-F1"][3]) < 1e-12, ("icentia pair", ref, k)
    for k, r in nb["chapman"]["metrics"].items():
        assert abs(res["chapman"]["metrics"][k]["Macro-F1"][0] - r["Macro-F1"][0]) < 1e-12, ("chapman", k)
    for k in nb["chapman"]["st_all"]:
        if k != "n":
            assert abs(res["chapman"]["st_all"][k][0] - nb["chapman"]["st_all"][k][0]) < 1e-12, k
    assert res["threshold"] == nb["threshold"]
    res["train_time"] = nb["train_time"]                      # waktu training dari run Kaggle
    res["eksplorasi"] = I.eksplorasi_afdb()
    sel = I.select_segments()
    res["data"] = {"scan": I.scan_summary(), "pilihan": sel["pasien"], "n_segmen": len(sel["segmen"]),
                   "window": I.icentia_summary(), "label": I.label_check()}
    (TMP / "tables.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    charts(res)


def charts(res=None):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter, FuncFormatter
    if res is None:
        res = json.loads((TMP / "tables.json").read_text(encoding="utf-8"))
    for style in ("regular", "bold"):
        font_manager.fontManager.addfont(font_path(style))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path("regular")).get_name(),
                         "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#C4CED8", "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": GRAY, "ytick.color": INK, "figure.facecolor": "white"})
    pctf = PercentFormatter(1.0, decimals=0)
    comma = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))

    def save(fig, name):
        fig.savefig(TMP / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=.08, facecolor="white")
        plt.close(fig)

    # 1. contoh bentuk gelombang: gambar dari notebook yang dijalankan di Kaggle
    nbk = json.loads((HERE / "icentia_wearable.ipynb").read_text(encoding="utf-8"))
    img = [o["data"]["image/png"] for c in nbk["cells"] for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
    assert len(img) == 2
    (TMP / "contoh_ekg.png").write_bytes(base64.b64decode(img[0]))

    # 2. ritme sinus cepat asli: Chapman (EKG klinis) dan Icentia (wearable)
    C, Ic = res["chapman"], res["icentia"]
    keys = ["v31_paket", "rr", "v9_dual", "dual"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
    panels = ((axes[0], "Chapman: sinus takikardia (EKG klinis)", C["st_bands"], list(C["st_bands"]), "pasien"),
              (axes[1], "Icentia: ritme Normal (wearable)", Ic["normal_hr"],
               [b for b in Ic["normal_hr"] if b.startswith(("100", "110", "120", "130", "140"))], "window"))
    for ax, title, tab, bands, unit in panels:
        x = np.arange(len(bands))
        for k in keys:
            ax.plot(x, [tab[b][k][0] for b in bands], marker=MMARK[k], color=MCOL[k], ms=5,
                    lw=2 if k in ("dual", "v31_paket") else 1.3, label=SHORT[k])
        ax.set_xticks(x, [f"{b.replace('–∞', '+')}\n({tab[b]['n']:,})".replace(",", ".") for b in bands], fontsize=7.2)
        ax.set_title(title, fontsize=8.6, loc="left", color=INK)
        ax.yaxis.set_major_formatter(pctf); ax.set_ylim(0, 1.03)
        ax.grid(axis="y", color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
        ax.set_xlabel(f"bpm ({unit})", fontsize=7.8)
    axes[0].set_ylabel("ditebak AFIB/AFL", fontsize=8.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.1), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, "takikardia_asli")

    # 3. tiga domain: Macro-F1 di AFDB, Chapman, Icentia
    A = res["afdb"]
    mk = ["v31", "rr", "v9_dual", "dual", "dual_tanpa_afdb"]
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    w = .27
    xs = np.arange(len(mk))
    doms = (("AFDB (Holter, 23 pasien)", [A["rows"][k]["Macro-F1"] for k in mk], "#8A97A6"),
            ("Chapman (EKG klinis, 34 ribu pasien)", [C["metrics"][k]["Macro-F1"] for k in mk], "#1baf7a"),
            ("Icentia (wearable, 2.420 pasien)", [Ic["rows"][k]["Macro-F1"] for k in mk], "#2a78d6"))
    for off, (lab, vals, col) in zip((-w, 0, w), doms):
        v = np.array(vals)
        ax.bar(xs + off, v[:, 0], w, color=col, label=lab, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
               error_kw=dict(ecolor=GRAY, lw=.8, capsize=2))
        for xi, val in zip(xs + off, v[:, 0]):
            ax.text(xi, .03, f"{val:.2f}".replace(".", ","), ha="center", fontsize=7, color="white", rotation=90, va="bottom")
    ax.set_xticks(xs, [SHORT[k] for k in mk], fontsize=8)
    ax.set_ylabel("Macro-F1", fontsize=8.5); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(comma)
    ax.grid(axis="y", color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.08), ncol=3, frameon=False, fontsize=7.8)
    fig.tight_layout(); save(fig, "domain3")

    # 4. recall AFL per pasien Icentia
    fig, ax = plt.subplots(figsize=(7.4, 2.3))
    for k in ("v31_paket", "v9_dual", "rr", "dual"):
        v = np.sort(Ic["afl_pasien_recall"][k])[::-1]
        ax.plot(np.arange(1, v.size + 1) / v.size, v, color=MCOL[k], lw=2 if k == "dual" else 1.3, label=SHORT[k],
                marker=MMARK[k], markevery=max(1, v.size // 12), ms=4.5)
    ax.set_xlabel(f"proporsi pasien AFL Icentia (n = {Ic['afl_pasien']['dual']['pasien']}), diurutkan dari recall tertinggi", fontsize=8)
    ax.set_ylabel("recall AFL pasien", fontsize=8.5); ax.xaxis.set_major_formatter(pctf); ax.yaxis.set_major_formatter(comma)
    ax.set_ylim(0, 1.02); ax.grid(color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
    ax.legend(fontsize=7.8, frameon=False, ncol=4, loc="upper right")
    fig.tight_layout(); save(fig, "afl_pasien")
    print(f"OK tables.json + 4 grafik → {TMP}")


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
    S = {"body": ParagraphStyle("body", fontName="Arial", fontSize=10.1, leading=14.6, textColor=HexColor(INK)),
         "small": ParagraphStyle("small", fontName="Arial", fontSize=8.7, leading=12, textColor=HexColor(GRAY)),
         "table": ParagraphStyle("table", fontName="Arial", fontSize=8.3, leading=11, textColor=HexColor(INK)),
         "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8.2, leading=11, textColor=white)}
    c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
    c.setTitle("Laporan v10 - Data wearable Icentia11k")
    c.setAuthor("Analisis v10_icentia_wearable/icentia_wearable.ipynb")
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

    def pv(p):
        return "P < 0,001" if p < .001 else f"P = {fnum(p, 3 if p < .01 else 2)}"

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
        c.drawString(M + 43, H - 42, "SIHEDAF  /  V10 DATA WEARABLE ICENTIA11K")
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", 21); c.drawString(M, H - 79, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 99, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, "18 Oktober 2026 | Sumber: v10_icentia_wearable/icentia_wearable.ipynb (run Kaggle 2× T4), cache/v10/, v9/, v8/")
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

    A, Cc, Ic, X, D = d["afdb"], d["chapman"], d["icentia"], d["eksplorasi"], d["data"]
    R, P, P9, P9r = A["rows"], A["pairs"], A["pairs_9a"], A["pairs_9rr"]
    met, st, stb, svt = Cc["metrics"], Cc["st_all"], Cc["st_bands"], Cc["svt_alarm"]
    IR, IP = Ic["rows"], Ic["pairs"]
    rec, pat = A["afl_recall"], A["afl_patients"]
    ip = {p_: pat.index(p_) for p_ in pat}
    nh = Ic["normal_hr"]
    hi_b = [b for b in nh if b.startswith(("120", "130", "140", "150"))]
    n_hi = sum(nh[b]["n"] for b in hi_b)
    a120 = {k: sum(nh[b][k][0] * nh[b]["n"] for b in hi_b) / n_hi for k in IR}     # pooled = rata-rata berbobot jumlah window
    cvt = Ic["afl_cv"]
    k_irr, k_reg = "tidak teratur (CV > 0,10)", "teratur (CV ≤ 0,10)"
    fpd = X["dual"]
    sc, wn, lb = D["scan"], D["window"], D["label"]
    epi = A["episodes"]

    # ── Hal 1: ringkasan ──
    y = page("Laporan v10: data wearable Icentia11k", "510 pasien AFL dari patch EKG satu lead untuk melatih cabang bentuk gelombang EKG")
    cardw = (CW - 24) / 4
    cards = ((fnum(IR["dual"]["Macro-F1"][0], 2), "Macro-F1 Icentia, 10a", f"(paket v3.1: {fnum(IR['v31_paket']['Macro-F1'][0], 2)})"),
             (pct(a120["dual"]), "Normal ≥ 120 bpm → alarm", f"Icentia, 10a (paket: {pct(a120['v31_paket'])})"),
             (fnum(met["dual"]["AFL R"][0], 2), "recall AFL Chapman, 10a", f"(paket v3.1: {fnum(met['v31_paket']['AFL R'][0], 2)})"),
             (fnum(R["dual"]["Macro-F1"][0], 2), "Macro-F1 AFDB, 10a", f"(v31: {fnum(R['v31']['Macro-F1'][0], 2)})"))
    for i, (val, lab1, lab2) in enumerate(cards):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 88, cardw, 88, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 20); c.drawString(x + 11, y - 37, val)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 7.4)
        c.drawString(x + 11, y - 59, lab1); c.drawString(x + 11, y - 70, lab2)
    y -= 105
    y = callout("Jawaban singkat",
                f"<b>Data wearable membuat model bentuk gelombang jauh lebih baik di dua domain eksternal, tetapi lebih buruk di AFDB.</b> "
                f"Pada 2.420 pasien Icentia yang tidak ikut dilatih, Macro-F1 naik dari {fnum(IR['v31_paket']['Macro-F1'][0])} (paket v3.1) "
                f"ke {fnum(IR['dual']['Macro-F1'][0])}, recall AFL dari {fnum(IR['v31_paket']['AFL R'][0], 2)} ke {fnum(IR['dual']['AFL R'][0], 2)}, "
                f"dan alarm pada ritme Normal ≥ 120 bpm turun {pct(a120['v31_paket'])} → {pct(a120['dual'])}. Di Chapman, 10a juga lebih baik "
                f"dari 9a (Macro-F1 {fnum(met['dual']['Macro-F1'][0])} vs {fnum(met['v9_dual']['Macro-F1'][0])}, {pv(Cc['pairs']['v9_dual']['dual']['Macro-F1'][3])}). "
                f"Tetapi di AFDB, Macro-F1 turun ke {fnum(R['dual']['Macro-F1'][0])} (lebih buruk dari v31, {pv(P['dual']['Macro-F1'][3])}): model "
                f"menyebut banyak window AFIB AFDB sebagai AFL. <b>Paket produksi tetap v3.1.</b>", y)
    y = section("Hipotesis dan hasilnya", y)
    y = table(["Varian", "Hipotesis (sebelum hasil)", "Hasil", "Status"], [
        ["kontrol rr", "Ambang uji stres sintetis > 115 bpm; AFL AFDB tidak membaik vs kontrol rr v9",
         f"ambang {min(d['threshold']['rr'].values())}–{max(d['threshold']['rr'].values())} bpm; AFDB Macro-F1 {sgn(P9r['rr']['Macro-F1'][0])} "
         f"({pv(P9r['rr']['Macro-F1'][3])})", "terbukti"],
        ["10a: EKG + RR", "AFDB: lebih baik dari 9a, AFL palsu turun, bias tidak di batas, 08378 ≥ 0,78. Icentia: lebih baik dari 9a "
         "& paket. Chapman: alarm ST ≤ 5%",
         f"AFDB {sgn(P9['dual']['Macro-F1'][0])} vs 9a ({pv(P9['dual']['Macro-F1'][3])}), AFL palsu {thou(A['fp']['v9_dual']['total'])} → "
         f"{thou(A['fp']['dual']['total'])}, 08378 {fnum(rec['dual'][ip['08378']], 2)}. Icentia {pv(IP['v9_dual']['dual']['Macro-F1'][3])}. "
         f"ST {pct(st['dual'][0])}", "<b>Sebagian</b>: Icentia & Chapman ya, AFDB tidak"],
        ["10b: tanpa AFDB", "AFDB tidak lebih buruk dari v31 (P > 0,05)",
         f"AFDB {fnum(R['dual_tanpa_afdb']['Macro-F1'][0])} ({pv(P['dual_tanpa_afdb']['Macro-F1'][3])})", "ditolak"],
        ["uji label AFL", "10a memisahkan AFL dari AFIB di Icentia jauh lebih baik dari kontrol rr",
         f"AFL tidak teratur ditebak AFL: rr {pct(cvt[k_irr]['AFL']['rr'][0])}, 10a {pct(cvt[k_irr]['AFL']['dual'][0])}; AFL F1 "
         f"{fnum(IR['rr']['AFL F1'][0])} → {fnum(IR['dual']['AFL F1'][0])}", "terbukti"],
    ], y, [72, 150, CW - 310, 88], bold_rows=[2])
    y = section("Temuan utama", y)
    for txt in [f"Icentia memberi {sc['kelompok pasien']['AFL']} pasien AFL (AFDB: 7), tetapi AFL-nya kebanyakan blok bervariasi: lambat dan "
                f"tidak teratur ({fnum(lb['AFL (Icentia)']['detak median'], 0)} bpm, CV {fnum(lb['AFL (Icentia)']['CV median'], 2)}), berbeda dari "
                f"AFL 2:1 di AFDB ({fnum(lb['AFL (AFDB)']['detak median'], 0)} bpm, CV {fnum(lb['AFL (AFDB)']['CV median'], 2)}).",
                f"Semua model lama hampir tidak pernah mendeteksi AFL Icentia (recall ≤ 0,05). 10a mendeteksi {fnum(IR['dual']['AFL R'][0], 2)} "
                f"dan hampir tidak membunyikan alarm pada sinus cepat di Icentia ({pct(a120['dual'])}) maupun Chapman ({pct(st['dual'][0])}).",
                f"Di AFDB, {pct(fpd['fp_afib'] / (fpd['fp_afib'] + fpd['fp_normal']))} AFL palsu 10a berasal dari window AFIB dan AUROC AFL turun "
                f"{fnum(X['v31']['auc_afl'])} → {fnum(X['dual']['auc_afl'])}; memperlebar bias tidak menolong. Batas AFL/AFIB tiap dataset berbeda.",
                f"Pasien AFDB 07910, yang gagal di semua model v3–v9 (recall ≤ 0,07), kini {fnum(rec['dual'][ip['07910']], 2)}."]:
        y = para(f"• {txt}", y, gap=5)

    # ── Hal 2: kriteria & perbandingan ──
    y = page("Kriteria kandidat dan perbandingan", "Kriteria ditetapkan sebelum hasil; pembanding pada data uji & fold yang sama")
    drops = [(p_, rec["v31"][i], rec["dual"][i]) for i, p_ in enumerate(pat) if rec["v31"][i] - rec["dual"][i] >= .2]
    y = table(["Kriteria 10a", "Hasil", "Status"], [
        ["AFDB: Macro-F1 tidak lebih buruk dari v31", f"{fnum(R['dual']['Macro-F1'][0])} vs {fnum(R['v31']['Macro-F1'][0])}, "
         f"{pv(P['dual']['Macro-F1'][3])}", "<b>tidak lolos</b>" if P["dual"]["Macro-F1"][3] <= .05 else "lolos"],
        ["AFDB: tidak ada pasien AFL turun ≥ 0,20", "; ".join(f"{p_} {fnum(a, 2)} → {fnum(b, 2)}" for p_, a, b in drops) or "—",
         "<b>tidak lolos</b>" if drops else "lolos"],
        ["Chapman: alarm ST ≤ separuh paket v3.1", f"{pct(st['dual'][0])} vs {pct(st['v31_paket'][0])}",
         "lolos" if st["dual"][0] <= st["v31_paket"][0] / 2 else "<b>tidak lolos</b>"],
        ["Chapman: recall AFL tidak lebih rendah dari paket", f"{fnum(met['dual']['AFL R'][0])} vs {fnum(met['v31_paket']['AFL R'][0])} "
         f"({pv(Cc['pairs']['v31_paket']['dual']['AFL R'][3])})", "lolos" if Cc["pairs"]["v31_paket"]["dual"]["AFL R"][3] > .05 else "<b>tidak lolos</b>"],
        ["Icentia: alarm Normal ≥ 120 bpm ≤ 25%", f"{pct(a120['dual'])} ({thou(n_hi)} window)", "lolos" if a120["dual"] <= .25 else "<b>tidak lolos</b>"],
        ["Icentia: recall AFL tidak lebih rendah dari paket", f"{fnum(IR['dual']['AFL R'][0])} vs {fnum(IR['v31_paket']['AFL R'][0])} "
         f"({pv(IP['v31_paket']['dual']['AFL R'][3])})", "lolos" if IP["v31_paket"]["dual"]["AFL R"][3] > .05 else "<b>tidak lolos</b>"],
    ], y, [200, CW - 290, 90])
    y = chart("domain3", y, 175)
    rows = []
    for k in ("v31", "v9_dual", "dual", "dual_tanpa_afdb"):
        rows.append([SHORT[k], ci(R[k]["Macro-F1"]), ci(met[k]["Macro-F1"]), ci(IR[k]["Macro-F1"]),
                     "referensi" if k == "v31" else f"{verdict(P[k]['Macro-F1'][3])} ({pv(P[k]['Macro-F1'][3])})"])
    rows.append(["v1 (sebelum revisi)", ci(R["v1"]["Macro-F1"]) if "v1" in R else "—", "—", "—",
                 f"{verdict(P['v1']['Macro-F1'][3])} ({pv(P['v1']['Macro-F1'][3])})" if "v1" in P else "—"])
    y = table(["Model", "Macro-F1 AFDB", "Macro-F1 Chapman", "Macro-F1 Icentia", "AFDB vs v31"], rows, y,
              [95, 95, 95, 95, CW - 380], bold_rows=[3])
    para("v31 = resep v3.1 yang dilatih ulang di lingkungan T4 (v8); v1 = sistem sebelum revisi (OOF cache v3, hanya AFDB). Paket v3.1 "
         f"sendiri: Macro-F1 Chapman {fnum(met['v31_paket']['Macro-F1'][0])}, Icentia {fnum(IR['v31_paket']['Macro-F1'][0])}. Bias kelas di "
         "Chapman & Icentia diambil dari AFDB, seperti di produksi.", y, "small")

    # ── Hal 3: data & metode ──
    y = page("Data Icentia11k dan cara kerja", "Semua anotasi dibaca; pasien AFL semua dipakai; praproses & arsitektur v6 tanpa diubah")
    per = sc["per ritme"]
    y = table(["Bagian", "Isi"], [
        ["Sumber", f"PhysioNet icentia11k-continuous-ecg 1.0: patch EKG satu lead (250 Hz) yang dipakai sehari-hari, {thou(sc['pasien'])} pasien, "
                   f"{thou(sc['segmen'])} segmen ±70 menit. Lisensi CC BY-NC-SA 4.0 (akademik/riset saja)"],
        ["Pemindaian", f"Semua file anotasi dibaca ({fnum(sc['GB anotasi'], 1)} GB, tidak disimpan): {thou(per['(AFL']['jam'])} jam AFL "
                       f"({per['(AFL']['pasien']} pasien), {thou(per['(AFIB']['jam'])} jam AFIB, {thou(per['(N']['jam'])} jam Normal. Pembaca "
                       "anotasi dibuktikan sama dengan wfdb.rdann"],
        ["Pemilihan", f"Kelompok pasien: {sc['kelompok pasien']['AFL']} AFL (episode ≥ 10 s), {sc['kelompok pasien']['AFIB']} AFIB, "
                      f"{thou(sc['kelompok pasien']['Normal'])} Normal. Dipakai: semua AFL & AFIB, 1.200 Normal acak; {thou(D['n_segmen'])} segmen"],
        ["Window", f"Front-end produksi (50 Hz, detektor R-peak v2), window 10 s, label = ritme anotasi. Batas per pasien 120 AFL, 40 AFIB, "
                   f"20 Normal + 12 Normal ≥ 100 bpm. Total {thou(wn['window'])} window (AFL {thou(wn['per kelas']['AFL'])}, AFIB "
                   f"{thou(wn['per kelas']['AFIB'])}, Normal {thou(wn['per kelas']['Normal'])}); EKG sejajar dengan RR (0 selisih)"],
        ["Training", "Resep v9 (CNN EKG + RR, bobot lunak, 30 epoch, seed 2026 + 100·r + k). Data latih: AFDB + Chapman berlabel andal + "
                     "Icentia, masing-masing fold latihnya. 10a & kontrol rr: 5×5 fold; 10b: 1×5 fold tanpa AFDB"],
        ["Evaluasi", "AFDB: protokol v3. Chapman: fold v8. Icentia: fold pasien baru (5 per ulangan); metrik 3 kelas hanya window acak; "
                     "CI bootstrap pasien bertingkat. Bias kelas di data eksternal dari AFDB"],
    ], y, [80, CW - 80])
    y = section("Cek label sebelum training (pelajaran v8)", y)
    keys_l = ["Normal (Icentia)", "AFIB (Icentia)", "AFL (Icentia)", "AFIB (AFDB)", "AFL (AFDB)"]
    y = table(["Kelas", "pasien", "window", "detak median", "CV IBI median", "window CV > 0,10"],
              [[k, thou(lb[k]["pasien"]), thou(lb[k]["window"]), f"{fnum(lb[k]['detak median'], 0)} bpm", fnum(lb[k]["CV median"]),
                pct(lb[k]["CV > 0,10"])] for k in keys_l], y, [120, 55, 65, 80, 80, CW - 400], bold_rows=[3])
    para(f"AFL Icentia kebanyakan AFL dengan blok AV yang berubah-ubah: lambat dan tidak teratur, mirip AFIB dari sisi interval RR. "
         f"Hanya {lb['AFL (Icentia), pasien ≥ 50% window teratur']['pasien']} dari {lb['AFL (Icentia)']['pasien']} pasien yang sebagian besar "
         "window AFL-nya teratur seperti AFL 2:1 di AFDB. Berbeda dengan Ningbo (v8), Icentia juga melabeli AFIB secara terpisah, jadi "
         "pelabel membedakan keduanya dari aktivitas atrium. Label tetap dipakai, dengan uji yang ditetapkan sebelum hasil (halaman 1).", y)

    # ── Hal 4: contoh EKG & Icentia ──
    y = page("Hasil di Icentia (wearable)",
             f"Pasien uji tidak ikut dilatih: {Ic['n_pasien']['AFL']} AFL, {Ic['n_pasien']['AFIB']} AFIB, {thou(Ic['n_pasien']['Normal'])} Normal; "
             "bias kelas dari AFDB")
    ik = ["v31_paket", "v31", "v9_dual", "rr", "dual", "dual_tanpa_afdb"]
    y = table(["Model", "Macro-F1", "AFL recall", "AFL F1", "AFIB R", "Normal R", "Normal ≥ 120 bpm → alarm"],
              [[SHORT[k], ci(IR[k]["Macro-F1"]), ci(IR[k]["AFL R"]), ci(IR[k]["AFL F1"]), fnum(IR[k]["AFIB R"][0]),
                fnum(IR[k]["Normal R"][0]), pct(a120[k])] for k in ik], y, [80, 82, 82, 82, 46, 50, CW - 422], bold_rows=[5])
    y = chart("afl_pasien", y, 140)
    y = table(["AFL Icentia (window acak)", "window"] + [SHORT[k] for k in ("v31_paket", "v9_dual", "rr", "dual")],
              [[f"{g}: ditebak {lab_}", thou(cvt[g]["n"])] + [cip(cvt[g][lab_][k]) for k in ("v31_paket", "v9_dual", "rr", "dual")]
               for g in (k_reg, k_irr) for lab_ in ("AFL", "AFIB")], y, [140, 45] + [(CW - 185) / 4] * 4)
    callout("Yang terlihat",
            f"Semua model lama hampir tidak pernah mendeteksi AFL Icentia (recall ≤ {fnum(max(IR[k]['AFL R'][0] for k in ('v31_paket', 'v31', 'v9_dual', 'rr')), 2)}): "
            f"AFL yang tidak teratur mereka sebut AFIB. 10a mendeteksi {pct(cvt[k_irr]['AFL']['dual'][0])} AFL tidak teratur dan "
            f"{pct(cvt[k_reg]['AFL']['dual'][0])} AFL teratur, dan hanya {pct(Ic['afl_pasien']['dual']['= 0'])} pasien AFL yang sama sekali tidak "
            f"terdeteksi (paket {pct(Ic['afl_pasien']['v31_paket']['= 0'])}). Harganya: {thou(Ic['afl_palsu_asal']['dual']['AFIB'])} window milik pasien AFIB "
            f"ditebak AFL per ulangan (paket {thou(Ic['afl_palsu_asal']['v31_paket']['AFIB'])}).", y)

    # ── Hal 5: takikardia asli & Chapman ──
    y = page("Ritme sinus cepat asli dan hasil Chapman", "Uji stres untuk model bentuk gelombang: sinus takikardia Chapman & Normal cepat Icentia")
    y = chart("takikardia_asli", y, 190)
    y = para(f"Angka dalam kurung = jumlah pasien (Chapman) atau window (Icentia). Pita Icentia 150+ bpm hanya {nh['150–∞']['n']} window, tidak "
         "digambar. Pada paket v3.1, hampir semua ritme sinus ≥ 120 bpm membunyikan alarm di kedua domain; model bentuk gelombang tidak.", y, "small")
    y -= 2
    ck = ["v31_paket", "v31", "v9_dual", "rr", "dual", "dual_tanpa_afdb"]
    y = table(["Chapman", "Macro-F1", "AFL recall (Shaoxing)", "AFIB R", "Normal R", "alarm ST", "alarm SVT (bagus)"],
              [[SHORT[k], ci(met[k]["Macro-F1"]), ci(met[k]["AFL R"]), fnum(met[k]["AFIB R"][0]), fnum(met[k]["Normal R"][0]),
                cip(st[k]), cip(svt[k])] for k in ck], y, [80, 85, 90, 48, 52, 78, CW - 433], bold_rows=[5])
    cp9 = Cc["pairs"]["v9_dual"]["dual"]
    callout("Yang terlihat",
            f"Menambah Icentia tidak merusak keunggulan 9a di Chapman, malah memperbaikinya: Macro-F1 {sgn(cp9['Macro-F1'][0])} "
            f"({pv(cp9['Macro-F1'][3])}) dan recall AFL Shaoxing {sgn(cp9['AFL R'][0])} ({pv(cp9['AFL R'][3])}), dengan alarm sinus takikardia tetap "
            f"{pct(st['dual'][0])}. Di Icentia, alarm pada Normal ≥ 120 bpm: paket {pct(a120['v31_paket'])}, kontrol rr {pct(a120['rr'])}, 9a "
            f"{pct(a120['v9_dual'])}, 10a {pct(a120['dual'])}.", y)

    # ── Hal 6: AFDB ──
    y = page("Hasil di AFDB (protokol v3)", "Window detektor 50 Hz, 5×5 fold pasien, bias cross-fit, CI bootstrap pasien")
    order = [k for k in ("v1", "v31", "v9_rr", "v9_dual", "rr", "dual", "dual_tanpa_afdb") if k in R]
    y = table(["Model", "Macro-F1", "AFL F1", "AFL recall", "Normal R", "AFL palsu / jam", "episode AFL"],
              [[SHORT[k], ci(R[k]["Macro-F1"]), ci(R[k]["AFL F1"]), ci(R[k]["AFL R"]), fnum(R[k]["Normal R"][0]),
                fnum(epi[k]["alarm/jam"][0], 2), pct(epi[k]["deteksi episode"][0])] for k in order],
              y, [80, 86, 86, 86, 52, 58, CW - 448], bold_rows=[order.index("dual") + 1])
    rows = []
    for k in ("rr", "dual", "dual_tanpa_afdb"):
        for ref, PP in (("v31", P[k]), ("9a", P9.get(k)), ("rr v9", P9r.get(k))):
            if PP is None:
                continue
            rows.append([SHORT[k] if ref == "v31" else "", ref] +
                        [f"{sgn(PP[m][0])} <font color='{GRAY}' size='7.2'>({pv(PP[m][3])})</font> {verdict(PP[m][3])}"
                         for m in ("Macro-F1", "AFL R", "Normal R")])
    y = table(["Model", "vs", "Δ Macro-F1", "Δ AFL recall", "Δ Normal recall"], rows, y,
              [85, 35, (CW - 120) / 3, (CW - 120) / 3, (CW - 120) / 3])
    ks = ["v31", "v9_dual", "rr", "dual", "dual_tanpa_afdb"]
    y = table(["recall AFL"] + pat, [[SHORT[k]] + [fnum(rec[k][i], 2) for i in range(len(pat))] for k in ks], y,
              [95] + [(CW - 95) / len(pat)] * len(pat), bold_rows=[4])

    # ── Hal 7: mengapa AFDB memburuk ──
    y = page("Mengapa AFDB memburuk", "Eksplorasi setelah hasil (bukan protokol): kalibrasi atau pemisahan kelas?")
    ks = [k for k in ks if k in X]
    y = table(["AFDB"] + [SHORT[k] for k in ks], [
        ["AUROC AFL lawan lain"] + [fnum(X[k]["auc_afl"]) for k in ks],
        ["AUROC AFIB lawan lain"] + [fnum(X[k]["auc_afib"]) for k in ks],
        ["Macro-F1, grid bias −7…+3 (protokol)"] + [fnum(R[k]["Macro-F1"][0]) for k in ks],
        ["Macro-F1, grid bias −15…+3"] + [fnum(X[k]["mf1_lebar"]) for k in ks],
        ["bias AFL (grid −15) min … maks"] + [f"{fnum(X[k]['bias_afl_lebar'][0], 2)} … {fnum(X[k]['bias_afl_lebar'][2], 2)}" for k in ks],
        ["AFL palsu: dari window Normal"] + [thou(X[k]["fp_normal"]) for k in ks],
        ["AFL palsu: dari window AFIB"] + [thou(X[k]["fp_afib"]) for k in ks],
    ], y, [150] + [(CW - 150) / len(ks)] * len(ks), bold_rows=[1])
    top = list(fpd["fp_per_pasien"].items())[:5]
    y = para("Pasien dengan AFL palsu terbanyak (10a, per ulangan): " + ", ".join(
        f"{q} {thou(v)}{' (pasien AFL)' if q in pat else ''}" for q, v in top) + ".", y, "small")
    callout("Tafsiran",
            f"Memperlebar grid bias tidak menolong ({fnum(X['dual']['mf1_lebar'])} vs {fnum(R['dual']['Macro-F1'][0])}), jadi masalahnya bukan "
            f"kalibrasi. AUROC AFL di AFDB turun dari {fnum(X['v31']['auc_afl'])} (v31) ke {fnum(X['v9_dual']['auc_afl'])} (9a) dan "
            f"{fnum(X['dual']['auc_afl'])} (10a), sedangkan AUROC AFIB tetap tinggi. {pct(fpd['fp_afib'] / (fpd['fp_afib'] + fpd['fp_normal']))} "
            "AFL palsu 10a berasal dari window AFIB. Dari Icentia, model belajar bahwa AFL bisa tidak teratur asalkan aktivitas atriumnya "
            "terorganisasi; pola itu juga ada pada banyak AFIB di AFDB (AF \"kasar\"). Penjelasan paling mungkin: <b>batas AFL/AFIB "
            "didefinisikan berbeda oleh pelabel AFDB dan Icentia</b>. Ini hipotesis; belum diuji dengan ahli. Catatan: 07910, pasien AFL yang "
            f"gagal di semua model v3–v9, kini {fnum(rec['dual'][ip['07910']], 2)} (10b {fnum(rec['dual_tanpa_afdb'][ip['07910']], 2)}).",
            y, "#FFF4EC")

    # ── Hal 8: uji stres sintetis & sensitivitas ──
    y = page("Uji stres sintetis dan sensitivitas", "Uji sintetis v3.2 hanya untuk model interval RR; sensitivitas tanpa pasien 04936")
    TH = d["threshold"]
    y = table(["Model (interval RR)"] + [f"Ambang: {j}" for j in d["jitters"]],
              [[SHORT[k]] + [f"{TH[k][j]} bpm" for j in d["jitters"]] for k in ("v31", "v9_rr", "rr")], y, [120] + [(CW - 120) / 4] * 4)
    y = para("Ambang = detak teratur sintetis tertinggi yang masih ≥ 50% ditebak Normal. Model bentuk gelombang tidak bisa diuji dengan ritme "
         "sintetis (tanpa gelombang); uji stresnya adalah ritme sinus cepat asli di halaman 5.", y, "small")
    y -= 4
    rows = []
    for lab_s, s in A["sens"].items():
        for k in ("v31", "v9_dual", "rr", "dual", "dual_tanpa_afdb"):
            r = s["rows"][k]
            rows.append([lab_s if k == "v31" else "", SHORT[k], ci(r["Macro-F1"]), ci(r["AFL F1"]), ci(r["AFL R"]),
                         "—" if k == "v31" else fnum(s["pairs_v31_minus"][k][3], 2)])
    y = table(["Data", "Model", "Macro-F1", "AFL F1", "AFL recall", "P(v31 > model)"], rows, y, [70, 90, 92, 92, 92, CW - 436])
    tt = d["train_time"]
    nt = d["n_train"]["dual"]
    para(f"Waktu training per fold di Kaggle T4: 10a {fnum(tt['dual'], 0)} s, 10b {fnum(tt['dual_tanpa_afdb'], 0)} s, kontrol rr "
         f"{fnum(tt['rr'], 0)} s (dua proses paralel di dua GPU; total training 3 jam 50 menit, evaluasi 50 menit). Data latih 10a per fold: "
         f"AFDB {thou(sum(nt[0]))}, Chapman {thou(sum(nt[1]))}, Icentia {thou(sum(nt[2]))} window.", y, "small")

    # ── Hal 9: kesimpulan ──
    y = page("Kesimpulan dan langkah berikutnya", "Yang terbukti, yang belum, dan arah berikutnya")
    y = table(["Hal", "Keputusan / status"], [
        ["Paket produksi", "Tetap v3.1; v10 tidak dikemas (gagal kriteria AFDB)"],
        ["Data wearable", f"<b>Bernilai</b> di dua domain eksternal: Icentia Macro-F1 {fnum(IR['v31_paket']['Macro-F1'][0], 2)} → "
                          f"{fnum(IR['dual']['Macro-F1'][0], 2)}, Chapman recall AFL {fnum(met['v9_dual']['AFL R'][0], 2)} → "
                          f"{fnum(met['dual']['AFL R'][0], 2)}; alarm sinus cepat rendah di kedua domain"],
        ["AFDB", f"<b>Lebih buruk</b>: Macro-F1 {fnum(R['dual']['Macro-F1'][0])}, AFL palsu dari window AFIB, 08378 hilang"],
        ["Label AFL Icentia", "Memuat informasi atrium yang nyata (uji yang ditetapkan sebelum hasil terbukti), tetapi fenotipenya berbeda dari AFDB"],
        ["Batas proyek sekarang", "Definisi AFL vs AFIB yang berbeda antar dataset, dan tidak ada data dari perangkat SIHEDAF sendiri"],
    ], y, [130, CW - 130])
    y = section("Langkah berikutnya", y)
    for txt in ["<b>Validasi dengan rekaman perangkat SIHEDAF yang dilabeli dokter.</b> Tiga domain publik kini memberi vonis berlawanan; "
                "hanya data perangkat yang bisa memutuskan model mana yang dipakai.",
                "Minta ahli meninjau label AFIB/AFL pasien AFDB 04936 dan 06995 (sumber AFL palsu terbesar) serta contoh AFL Icentia yang "
                "tidak teratur.",
                "Keputusan produk: apakah AFL perlu dibedakan dari AFIB di perangkat, atau cukup satu alarm \"AF/AFL\". Untuk alarm sinus "
                "takikardia, model bentuk gelombang jauh lebih baik dari paket v3.1 di kedua domain eksternal."]:
        y = para(f"• {txt}", y, gap=5)
    y = section("Batasan", y)
    for txt in ["Label Icentia semi-otomatis; batas AF kasar dan AFL dengan blok bervariasi memang sering tertukar.",
                "Window Icentia dibatasi per pasien dan segmen AFL dipilih yang paling banyak AFL-nya; prevalensi kelas di data uji Icentia "
                "tidak sama dengan pemakaian nyata.",
                "10b hanya 1 ulangan untuk Chapman & Icentia. Tafsiran di halaman 7 adalah hipotesis (eksplorasi setelah hasil).",
                "Model yang dilatih dengan Icentia terikat lisensi non-komersial (CC BY-NC-SA 4.0).",
                "Reproduksi: kernel Kaggle CPU (data Icentia, ±35 menit) lalu notebook v10 di Kaggle 2× T4 (±5 jam) dengan dataset privat "
                "sihedaf-v8-cache & sihedaf-v9-cache; laporan: <font name='Courier' size='8.3'>laporan/build_report.py --extract</font> "
                "lalu tanpa argumen."]:
        y = para(f"• {txt}", y, gap=5)
    c.save()
    print(f"OK {OUT} ({state['page']} halaman)")


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else charts() if "--charts" in sys.argv else build()
