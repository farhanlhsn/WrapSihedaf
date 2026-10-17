"""Laporan PDF v9 (cabang bentuk gelombang EKG, AFDB + Chapman).

Dua tahap (pola sama dengan v3–v8):
  1. --extract : evaluasi ulang dari cache/v9 + cache/v8 (hasil run Kaggle yang dibawa pulang), cocokkan dengan
     ../cache/v9/ringkasan.json dari notebook, lalu grafik → laporan/_build/. --charts menggambar ulang grafik saja.
  2. tanpa argumen (ReportLab): rakit laporan/laporan_gelombang_ekg.pdf.
"""
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # = v9_gelombang_ekg/
REPO = HERE.parent
TMP = HERE / "laporan/_build"
OUT = HERE / "laporan/laporan_gelombang_ekg.pdf"
SUMMARY = REPO / "cache/v9/ringkasan.json"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402

TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
B8 = "afdb_chapman_bersih"
MCOL = {"v31_paket": "#8A97A6", "v31": "#8A97A6", "rr": "#eb6834", B8: "#C4CED8", "dual": "#2a78d6", "ekg": "#1baf7a",
        "fusi": "#172B43"}
MMARK = {"v31_paket": "s", "v31": "s", "rr": "^", B8: "v", "dual": "o", "ekg": "D", "fusi": "P"}
SHORT = {"v1": "v1", "v1_paket": "paket v1", "v31_paket": "paket v3.1", "v31": "v31 (resep v3.1)", "afdb": "afdb (= v5b)",
         B8: "8c (v8)", "rr": "kontrol rr", "dual": "9a: EKG + RR", "ekg": "9b: EKG saja", "fusi": "9c: fusi"}


def extract():
    sys.path.insert(0, str(HERE))
    import gelombang as G
    TMP.mkdir(parents=True, exist_ok=True)
    res = G.evaluate()
    nb = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for k, r in nb["afdb"]["rows"].items():
        for m in ("Macro-F1", "AFL F1", "AFL R", "Normal R"):
            assert abs(res["afdb"]["rows"][k][m][0] - r[m][0]) < 1e-12, (k, m)
    for grp in ("pairs", "pairs_rr", "pairs_8c"):
        for k, p in nb["afdb"][grp].items():
            assert abs(res["afdb"][grp][k]["Macro-F1"][3] - p["Macro-F1"][3]) < 1e-12, (grp, k)
    for k, r in nb["chapman"]["metrics"].items():
        assert abs(res["chapman"]["metrics"][k]["Macro-F1"][0] - r["Macro-F1"][0]) < 1e-12, k
    for k in nb["chapman"]["st_all"]:
        if k != "n":
            assert abs(res["chapman"]["st_all"][k][0] - nb["chapman"]["st_all"][k][0]) < 1e-12, k
    assert res["threshold"] == nb["threshold"]
    res["train_time"] = nb["train_time"]                      # waktu training dari run Kaggle
    (TMP / "tables.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    charts(res)


def charts(res=None):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter
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

    # 1. contoh bentuk gelombang: gambar dari notebook yang dijalankan di Kaggle (EKG mentah Chapman tidak dibawa pulang)
    nbk = json.loads((HERE / "gelombang_ekg.ipynb").read_text(encoding="utf-8"))
    img = [o["data"]["image/png"] for c in nbk["cells"] for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
    assert len(img) == 1
    (TMP / "contoh_ekg.png").write_bytes(base64.b64decode(img[0]))

    # 2. sinus takikardia asli per pita detak
    C = res["chapman"]
    bands = list(C["st_bands"])
    keys = ["v31_paket", B8, "rr", "ekg", "dual"]                      # 9a digambar terakhir (di atas 9b yang nilainya mirip)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(bands))
    for k in keys:
        ax.plot(x, [C["st_bands"][b][k][0] for b in bands], marker=MMARK[k], color=MCOL[k], ms=6,
                lw=2 if k in ("dual", "ekg", "v31_paket") else 1.4, label=SHORT[k])
    ax.set_xticks(x, [f"{b.replace('–∞', '+')} bpm\n({C['st_bands'][b]['n']:,} pasien)".replace(",", ".") for b in bands], fontsize=8)
    ax.set_ylabel("ST ditebak AFIB/AFL", fontsize=8.5); ax.yaxis.set_major_formatter(pctf); ax.set_ylim(0, 1.03)
    ax.grid(axis="y", color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.1), ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, "st_bands")

    # 3. dua domain: Macro-F1 di AFDB vs di Chapman
    A = res["afdb"]
    mk = ["v31", B8, "rr", "dual", "ekg", "fusi"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = .38
    xs = np.arange(len(mk))
    for off, (lab, vals, col) in zip((-w / 2, w / 2), (("AFDB (Holter, 23 pasien)", [A["rows"][k]["Macro-F1"] for k in mk], "#8A97A6"),
                                                       ("Chapman (EKG klinis, 34 ribu pasien)", [C["metrics"][k]["Macro-F1"] for k in mk], "#2a78d6"))):
        v = np.array(vals)
        ax.bar(xs + off, v[:, 0], w, color=col, label=lab, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
               error_kw=dict(ecolor=GRAY, lw=.8, capsize=2))
        for xi, val in zip(xs + off, v[:, 0]):
            ax.text(xi, .03, f"{val:.2f}".replace(".", ","), ha="center", fontsize=7.4, color="white", rotation=90, va="bottom")
    ax.set_xticks(xs, [SHORT[k] for k in mk], fontsize=8)
    ax.set_ylabel("Macro-F1", fontsize=8.5); ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.grid(axis="y", color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.08), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, "domain")
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
    S = {"body": ParagraphStyle("body", fontName="Arial", fontSize=10.1, leading=14.6, textColor=HexColor(INK)),
         "small": ParagraphStyle("small", fontName="Arial", fontSize=8.7, leading=12, textColor=HexColor(GRAY)),
         "table": ParagraphStyle("table", fontName="Arial", fontSize=8.3, leading=11, textColor=HexColor(INK)),
         "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8.2, leading=11, textColor=white)}
    c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
    c.setTitle("Laporan v9 - Cabang bentuk gelombang EKG")
    c.setAuthor("Analisis v9_gelombang_ekg/gelombang_ekg.ipynb")
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
        c.drawString(M + 43, H - 42, "SIHEDAF  /  V9 CABANG BENTUK GELOMBANG EKG")
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", 21); c.drawString(M, H - 79, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 99, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, "18 Oktober 2026 | Sumber: v9_gelombang_ekg/gelombang_ekg.ipynb (run Kaggle 2× T4), cache/v9/, cache/v8/")
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

    A, Cc = d["afdb"], d["chapman"]
    R, P, PR, P8 = A["rows"], A["pairs"], A["pairs_rr"], A["pairs_8c"]
    met, st, stb = Cc["metrics"], Cc["st_all"], Cc["st_bands"]
    rec, pat = A["afl_recall"], A["afl_patients"]
    i08, i43 = pat.index("08378"), pat.index("04043")
    cp_rr, cp_pk = Cc["pairs"]["rr"], Cc["pairs"]["v31_paket"]
    svt = Cc["svt_alarm"]
    hi = lambda b: b.replace("–∞", "+")

    # ── Hal 1 ──
    y = page("Laporan v9: cabang bentuk gelombang EKG", "CNN pada EKG 50 Hz, dilatih dengan AFDB + 34 ribu pasien Chapman berlabel andal")
    cardw = (CW - 24) / 4
    cards = ((pct(st["dual"][0]), f"ST asli → alarm, 9a (paket v3.1: {pct(st['v31_paket'][0])})"),
             (fnum(met["dual"]["Macro-F1"][0], 2), f"Macro-F1 Chapman, 9a (paket: {fnum(met['v31_paket']['Macro-F1'][0], 2)})"),
             (fnum(R["dual"]["Macro-F1"][0], 2), f"Macro-F1 AFDB, 9a (v31: {fnum(R['v31']['Macro-F1'][0], 2)})"),
             (fnum(rec["dual"][i08], 2), f"recall AFL 08378, 9a (v31: {fnum(rec['v31'][i08], 2)})"))
    for i, (val, lab) in enumerate(cards):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 88, cardw, 88, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 20); c.drawString(x + 11, y - 37, val)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 7.2); c.drawString(x + 11, y - 61, lab)
    y -= 105
    y = callout("Jawaban singkat",
                f"<b>Bentuk gelombang memang memuat informasi yang hilang dari interval RR, tetapi model belum menyeberang antar-dataset.</b> "
                f"Pada pasien Chapman yang tidak ikut dilatih, model 9a hampir menghilangkan alarm palsu sinus takikardia "
                f"({pct(st['v31_paket'][0])} → {pct(st['dual'][0])}, juga di 140–150 bpm), menaikkan recall Normal ke "
                f"{fnum(met['dual']['Normal R'][0], 2)}, dan lebih baik dari kontrol interval RR di semua metrik (P = 1,00). Tetapi di AFDB "
                f"(Holter rawat jalan), Macro-F1 turun ke {fnum(R['dual']['Macro-F1'][0])} (P = {fnum(P['dual']['Macro-F1'][3], 2)} vs v31), "
                f"alarm AFL palsu melonjak, dan 08378 hilang. Bias kelasnya menyentuh kedua batas grid: tanda pergeseran domain. "
                "<b>Paket produksi tetap v3.1.</b>", y)
    y = section("Hipotesis dan hasilnya", y)
    y = table(["Varian", "Hipotesis (sebelum hasil)", "Hasil", "Status"], [
        ["kontrol rr", "Mirip 8c", f"AFDB {fnum(R['rr']['Macro-F1'][0])}; alarm ST {pct(st['rr'][0])}", "terbukti"],
        ["9a: EKG + RR", "vs rr di Chapman: AFL & alarm SVT naik, alarm ST tidak naik; 08378 ≥ 0,78",
         f"Chapman: AFL {sgn(cp_rr['dual']['AFL R'][0])} (P = {fnum(cp_rr['dual']['AFL R'][3], 2)}), SVT {pct(svt['rr'][0])} → "
         f"{pct(svt['dual'][0])}, ST {pct(st['rr'][0])} → {pct(st['dual'][0])}; 08378 {fnum(rec['dual'][i08], 2)}",
         "<b>Sebagian</b>: Chapman ya, AFDB tidak"],
        ["9b: EKG saja", "Memisahkan ST dari AFL lebih baik dari rr",
         f"Chapman Macro-F1 {fnum(met['ekg']['Macro-F1'][0])}, alarm ST {pct(st['ekg'][0])}; AFDB {fnum(R['ekg']['Macro-F1'][0])} "
         f"({verdict(P['ekg']['Macro-F1'][3])})", "<b>Sebagian</b>"],
        ["9c: fusi", "Eksploratif", f"AFDB {fnum(R['fusi']['Macro-F1'][0])}; Chapman {fnum(met['fusi']['Macro-F1'][0])}", "—"],
    ], y, [80, 150, CW - 320, 90], bold_rows=[2])
    y = section("Kriteria kandidat produksi untuk 9a (ditetapkan sebelum hasil)", y)
    st120 = max(stb[b]["dual"][0] for b in stb if not b.startswith(("100", "110")))
    drops = [(p, rec["v31"][i], rec["dual"][i]) for i, p in enumerate(pat) if rec["v31"][i] - rec["dual"][i] >= .2]
    table(["Kriteria", "Hasil", "Status"], [
        ["AFDB: Macro-F1 tidak lebih buruk dari v31", f"P = {fnum(P['dual']['Macro-F1'][3], 2)}", "lolos (tipis)" if P["dual"]["Macro-F1"][3] > .05 else "<b>tidak lolos</b>"],
        ["AFDB: tidak ada pasien AFL turun ≥ 0,20", "; ".join(f"{p} {fnum(a, 2)} → {fnum(b, 2)}" for p, a, b in drops) or "—",
         "<b>tidak lolos</b>" if drops else "lolos"],
        ["Chapman: alarm ST ≤ separuh paket v3.1", f"{pct(st['dual'][0])} vs {pct(st['v31_paket'][0])}", "lolos"],
        ["Chapman: recall AFL tidak lebih rendah dari paket", f"{fnum(met['dual']['AFL R'][0], 3)} vs {fnum(met['v31_paket']['AFL R'][0], 3)} "
         f"(P = {fnum(cp_pk['dual']['AFL R'][3], 2)})", "lolos (tidak terbedakan)" if cp_pk["dual"]["AFL R"][3] > .05 else "<b>tidak lolos</b>"],
        ["ST asli ≥ 120 bpm: alarm ≤ 25%", f"tertinggi {pct(st120)}", "lolos" if st120 <= .25 else "<b>tidak lolos</b>"],
    ], y, [210, CW - 310, 100])

    # ── Hal 2: data & metode ──
    y = page("Data dan cara kerja", "Praproses & arsitektur v6 tanpa diubah; bedanya hanya jumlah dan sumber pasien")
    y = table(["Bagian", "Isi"], [
        ["Data latih", "AFDB fold latih (5×5 fold resmi) + Chapman fold latih berlabel andal: 32.025 Normal (SR/SB/ST/SA), 1.696 AFIB, "
                       "416 AFL Shaoxing. AFL Ningbo tidak dipakai (label tidak andal, v8)"],
        ["Bentuk gelombang", "Kanal 0 AFDB / lead II Chapman → 50 Hz → bandpass 0,5–20 Hz → 500 sampel per window, dinormalisasi per window. "
                             "Dibuktikan sejajar dengan window interval RR: 0 selisih di 165.978 window AFDB dan 42.040 window Chapman"],
        ["Model", "9a: CNN residual (4 blok, ±390 ribu parameter) + badan MicroTransformer interval RR, digabung sebelum klasifikasi. "
                  "9b: CNN saja. Kontrol rr: MicroTransformer saja. Semua bobot kelas lunak, 30 epoch, seed sama"],
        ["Evaluasi", "AFDB: protokol v3 (pasien uji tidak ikut dilatih, bias cross-fit). Chapman: fold pasien v8, bias dari AFDB. "
                     "Pembanding v31/8c dari cache v8 (lingkungan T4 & fold yang sama)"],
        ["Uji stres", "Ritme sintetis v3.2 tidak punya bentuk gelombang, jadi untuk model EKG uji stresnya 6.937 sinus takikardia asli"],
    ], y, [100, CW - 100])
    y = chart("contoh_ekg", y, 260)
    para("Contoh tiga ritme cepat (130–160 bpm) yang sama teraturnya bila dilihat dari interval RR. Secara fisiologis pembedanya "
         "ada di aktivitas atrium: gelombang P sebelum tiap QRS pada sinus takikardia, gelombang flutter pada AFL. Pada 50 Hz dan "
         "satu lead, perbedaan ini halus dan tidak selalu tampak oleh mata; CNN mempelajarinya dari ribuan contoh. Gambar dari "
         "notebook yang dijalankan di Kaggle (desimal bertitik).", y, "small")

    # ── Hal 3: Chapman ──
    y = page("Hasil di Chapman (pasien yang tidak ikut dilatih)", f"Label andal: {thou(Cc['n_metrik']['Normal'])} Normal, "
             f"{thou(Cc['n_metrik']['AFIB'])} AFIB, {thou(Cc['n_metrik']['AFL'])} AFL Shaoxing; bias kelas dari AFDB")
    y = chart("st_bands", y, 165)
    keys = ["v31_paket", B8, "rr", "dual", "ekg", "fusi"]
    rows = [[SHORT[k], ci(met[k]["Macro-F1"]), ci(met[k]["AFL R"]), fnum(met[k]["AFIB R"][0]), fnum(met[k]["Normal R"][0]),
             cip(st[k]), cip(svt[k])] for k in keys]
    y = table(["Model", "Macro-F1", "AFL recall", "AFIB R", "Normal R", "alarm ST", "alarm SVT (bagus)"], rows, y,
              [80, 85, 85, 50, 55, 75, CW - 430], bold_rows=[4])
    callout("Yang terlihat",
            f"Model bentuk gelombang hampir tidak lagi membunyikan alarm pada sinus takikardia di semua pita detak (9a "
            f"{pct(stb['140–150']['dual'][0])} di 140–150 bpm; kontrol rr {pct(stb['140–150']['rr'][0])}). Ini persis informasi yang "
            f"tidak ada di interval RR. SVT lebih sering dibunyikan ({pct(svt['dual'][0])} vs {pct(svt['rr'][0])}) dan recall Normal naik ke "
            f"{fnum(met['dual']['Normal R'][0], 2)}. Recall AFL Shaoxing 9a {fnum(met['dual']['AFL R'][0])} (paket {fnum(met['v31_paket']['AFL R'][0])}); "
            f"9b lebih tinggi ({fnum(met['ekg']['AFL R'][0])}, P = {fnum(cp_pk['ekg']['AFL R'][3], 2)} vs paket). Semua Δ terhadap kontrol rr: P = 1,00.", y)

    # ── Hal 4: AFDB ──
    y = page("Hasil di AFDB (protokol v3)", "Window detektor 50 Hz, 5×5 fold pasien, bias cross-fit, CI bootstrap pasien")
    order = [k for k in ("v1", "v31", "afdb", B8, "rr", "dual", "ekg", "fusi") if k in R]
    E = A["episodes"]
    y = table(["Model", "Macro-F1", "AFL F1", "AFL recall", "Normal R", "AFL palsu / jam", "episode AFL"],
              [[SHORT[k], ci(R[k]["Macro-F1"]), ci(R[k]["AFL F1"]), ci(R[k]["AFL R"]), fnum(R[k]["Normal R"][0]),
                fnum(E[k]["alarm/jam"][0], 2), pct(E[k]["deteksi episode"][0])] for k in order],
              y, [80, 86, 86, 86, 52, 58, CW - 448], bold_rows=[order.index("dual") + 1])
    rows = []
    for k in ("dual", "ekg", "fusi"):
        for ref, PP in (("v31", P[k]), ("rr", PR[k]), ("8c", P8[k])):
            rows.append([SHORT[k] if ref == "v31" else "", ref] +
                        [f"{sgn(PP[m][0])} <font color='{GRAY}' size='7.2'>(P = {fnum(PP[m][3], 2)})</font> {verdict(PP[m][3])}"
                         for m in ("Macro-F1", "AFL R", "Normal R")])
    y = table(["Model", "vs", "Δ Macro-F1", "Δ AFL recall", "Δ Normal recall"], rows, y,
              [85, 35, (CW - 120) / 3, (CW - 120) / 3, (CW - 120) / 3])
    para("v1 = sistem sebelum revisi (OOF cache v3). v31, afdb, 8c = hasil v8 di lingkungan T4 yang sama. Kontrol rr = data latih "
         "sama dengan 9a tetapi tanpa bentuk gelombang.", y, "small")

    # ── Hal 5: mengapa AFDB berbeda ──
    y = page("Mengapa hasil AFDB dan Chapman berlawanan", "Pergeseran domain: Holter rawat jalan (AFDB) vs EKG klinis istirahat (Chapman)")
    y = chart("domain", y, 170)
    ks = ["v31", B8, "rr", "dual", "ekg", "fusi"]
    fp, bias = A["fp"], A["bias"]
    y = table(["AFDB"] + [SHORT[k] for k in ks], [
        ["AFL palsu / ulangan"] + [thou(fp[k]["total"]) for k in ks],
        ["… dari pasien tanpa AFL"] + [thou(fp[k]["pasien_lain"]) for k in ks],
        ["bias AFL min … maks"] + [f"{fnum(bias[k]['min'][2], 2)} … {fnum(bias[k]['max'][2], 2)}" for k in ks],
        ["recall AFL 08378"] + [fnum(rec[k][i08], 2) for k in ks],
        ["recall AFL 04043"] + [fnum(rec[k][i43], 2) for k in ks]],
        y, [100] + [(CW - 100) / 6] * 6)
    callout("Tafsiran",
            f"Hampir semua AFL palsu model bentuk gelombang di AFDB ({thou(fp['dual']['total'])} per ulangan untuk 9a) datang dari window "
            f"non-AFL milik pasien AFL sendiri (hanya {thou(fp['dual']['pasien_lain'])} dari pasien tanpa AFL). Bias AFL yang dipilih "
            f"cross-fit melompat dari {fnum(bias['dual']['min'][2], 0)} sampai {fnum(bias['dual']['max'][2], 0)}, menyentuh kedua batas grid. "
            "Itu tanda bahaya yang sama dengan v6: skala probabilitas model di AFDB tidak stabil antar fold. Penjelasan yang paling "
            "mungkin: model belajar AFL terutama dari 416 pasien Chapman (EKG istirahat, lead II), sedangkan AFL di AFDB direkam Holter "
            "(gerakan, noise, lead lain), dan AFL 2:1 yang sangat teratur seperti 08378 jarang di Chapman. Ini belum dibuktikan.", y, "#FFF4EC")

    # ── Hal 6: uji stres & sensitivitas ──
    y = page("Uji stres sintetis dan sensitivitas", "Uji sintetis v3.2 hanya untuk model interval RR; sensitivitas tanpa pasien 04936")
    TH = d["threshold"]
    y = table(["Model (interval RR)"] + [f"Ambang: {j}" for j in d["jitters"]],
              [[SHORT[k]] + [f"{TH[k][j]} bpm" for j in d["jitters"]] for k in ("v31", B8, "rr")], y, [120] + [(CW - 120) / 4] * 4)
    rows = []
    for lab_s, s in A["sens"].items():
        for k in ("v31", "rr", "dual", "ekg", "fusi"):
            r = s["rows"][k]
            rows.append([lab_s if k == "v31" else "", SHORT[k], ci(r["Macro-F1"]), ci(r["AFL F1"]), ci(r["AFL R"]),
                         "—" if k == "v31" else fnum(s["pairs_v31_minus"][k][3], 2)])
    y = table(["Data", "Model", "Macro-F1", "AFL F1", "AFL recall", "P(v31 > model)"], rows, y, [70, 90, 92, 92, 92, CW - 436])
    tt = d["train_time"]
    para(f"Waktu training per fold di Kaggle T4: 9a {fnum(tt['dual'], 0)} s, 9b {fnum(tt['ekg'], 0)} s, kontrol rr {fnum(tt['rr'], 0)} s "
         "(dua proses paralel di dua GPU). Model bentuk gelombang ±7× lebih lambat dilatih daripada model interval RR.", y, "small")

    # ── Hal 7: kesimpulan ──
    y = page("Kesimpulan dan langkah berikutnya", "Yang terbukti, yang belum, dan arah v10")
    y = table(["Hal", "Keputusan / status"], [
        ["Paket produksi", "Tetap v3.1; v9 tidak dikemas (gagal kriteria per pasien di AFDB)"],
        ["Informasi bentuk gelombang", f"<b>Terbukti bernilai</b> pada data eksternal: alarm ST {pct(st['v31_paket'][0])} → {pct(st['dual'][0])}, "
                                       f"Normal R {fnum(met['v31_paket']['Normal R'][0], 2)} → {fnum(met['dual']['Normal R'][0], 2)}"],
        ["Generalisasi ke AFDB", "<b>Belum</b>: AFL palsu melonjak, 08378 dan 04043 hilang, bias kelas tidak stabil"],
        ["Batas proyek sekarang", "Pergeseran domain antar-dataset (Holter vs EKG klinis), bukan kekurangan informasi atau pasien"],
    ], y, [150, CW - 150])
    y = section("Langkah berikutnya (v10)", y)
    for txt in ["<b>Validasi dengan rekaman perangkat SIHEDAF sendiri</b> yang dilabeli dokter. Ini satu-satunya cara tahu domain mana "
                "(Holter atau EKG klinis) yang lebih mirip perangkat, dan apakah hasil Chapman berlaku di sana.",
                "Training yang tahan domain: augmentasi noise/gerakan/baseline wander, amplitudo & polaritas, serta penyeimbangan "
                "kontribusi per dataset; kalibrasi bias terpisah per domain.",
                "Data AFL tambahan yang beragam (AFL 2:1 teratur seperti 08378), misalnya dari database Holter lain."]:
        y = para(f"• {txt}", y, gap=5)
    y = section("Batasan", y)
    for txt in ["AFL berlabel andal di Chapman hanya 416 pasien; AFDB hanya 7 pasien AFL.",
                "Lead AFDB (kanal 0) dan lead II Chapman tidak sama; perangkat SIHEDAF mungkin memakai lead lain lagi.",
                "Tafsiran pergeseran domain di halaman 5 adalah hipotesis, belum diuji.",
                "Reproduksi: notebook v9 di Kaggle (2× T4, ±6 jam) dengan dataset privat sihedaf-v8-cache; laporan: "
                "<font name='Courier' size='8.3'>laporan/build_report.py --extract</font> lalu tanpa argumen."]:
        y = para(f"• {txt}", y, gap=5)
    c.save()
    print(f"OK {OUT} ({state['page']} halaman)")


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else charts() if "--charts" in sys.argv else build()
