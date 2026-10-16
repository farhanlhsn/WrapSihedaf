"""Laporan PDF v3.2 (uji stres takikardia & analisis sensitivitas).

Dua tahap (pola sama dengan v3–v6):
  1. --extract  (Python dengan numpy/torch/matplotlib): hitung ulang lewat takikardia.run_all() (bagian lambat dari cache),
     cocokkan dengan ../cache/v3_2/ringkasan.json dari notebook, buat grafik → laporan/_build/.
  2. tanpa argumen (Python dengan ReportLab): rakit laporan/laporan_uji_takikardia.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # = v3.2_uji_takikardia/
REPO = HERE.parent
TMP = HERE / "laporan/_build"
OUT = HERE / "laporan/laporan_uji_takikardia.pdf"
SUMMARY = REPO / "cache/v3_2/ringkasan.json"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402

TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
MCOL = {"v1": "#8A97A6", "v2_prod": "#eb6834", "v31": "#2a78d6", "v6_late": "#1baf7a"}
MMARK = {"v1": "s", "v2_prod": "^", "v31": "o", "v6_late": "D"}
SHORT = {"v1": "v1", "v2_prod": "v2 produksi", "v31": "v3.1 produksi", "v4_e1": "v4 ensemble",
         "v5_lunak": "v5 bobot lunak", "v6_late": "v6 late fusion"}


def extract():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    sys.path.insert(0, str(HERE))
    import takikardia as T

    TMP.mkdir(parents=True, exist_ok=True)
    res = T.run_all()
    nb = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert nb["threshold"] == res["threshold"], "uji sintetis tidak sama dengan notebook"
    assert abs(nb["nested"]["v31_point"] - res["nested"]["v31_point"]) < 1e-12
    for k in ("v2_prod", "v1"):
        assert abs(nb["nested"][k]["p"] - res["nested"][k]["p"]) < 1e-12, k
    for s in ("sens_all", "sens_wo"):
        for k, r in res[s]["rows"].items():
            assert abs(nb[s]["rows"][k]["Macro-F1"][0] - r["Macro-F1"][0]) < 1e-12, (s, k)
    (TMP / "tables.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    for style in ("regular", "bold"):
        font_manager.fontManager.addfont(font_path(style))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path("regular")).get_name(),
                         "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#C4CED8", "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": GRAY, "ytick.color": INK, "figure.facecolor": "white"})
    pct = PercentFormatter(1.0, decimals=0)

    def save(fig, name):
        fig.savefig(TMP / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=.08, facecolor="white")
        plt.close(fig)

    # 1. uji sintetis: proporsi ditebak Normal vs bpm
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.4), sharey=True)
    for ax, j in zip(axes, res["jitters"]):
        ax.axvspan(100, 170, color="#F6F7F9", zorder=0)
        for m in ("v1", "v2_prod", "v31"):
            ax.plot(res["bpms"], [t[0] for t in res["stress"][m][j]], marker=MMARK[m], color=MCOL[m], ms=3.5, lw=1.6,
                    label=SHORT[m])
        ax.set_title(j, loc="left", fontsize=8.8, fontweight="bold")
        ax.set_xticks([60, 90, 120, 150]); ax.tick_params(labelsize=8)
        ax.grid(color="#E6EBF0", lw=.8); ax.set_axisbelow(True); ax.yaxis.set_major_formatter(pct)
        ax.set_xlabel("bpm", fontsize=8.5)
    axes[0].set_ylabel("ditebak Normal", fontsize=8.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.1), ncol=3,
               frameon=False, fontsize=8.5)
    fig.tight_layout(); save(fig, "stress")

    # 2. window Normal asli: proporsi alarm (AFIB+AFL) per pita detak
    bins = [k for k in res["by_hr"] if k.startswith("Normal")]
    labels = [k.split("| ")[1].replace("0–100", "<100").replace("120–∞", "≥120") for k in bins]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    x = np.arange(len(bins))
    for m in ("v1", "v2_prod", "v31", "v6_late"):
        ax.plot(x, [res["by_hr"][b][m][1] + res["by_hr"][b][m][2] for b in bins], marker=MMARK[m], color=MCOL[m],
                ms=6, lw=1.8, label=SHORT[m])
    ax.set_xticks(x, [f"{l}\n({res['by_hr'][b]['n']:,} window)".replace(",", ".") for l, b in zip(labels, bins)], fontsize=8.5)
    ax.set_ylabel("Normal ditebak AFIB/AFL", fontsize=8.5); ax.yaxis.set_major_formatter(pct); ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); save(fig, "real_bins")
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
    c.setTitle("Laporan v3.2 - Uji stres takikardia & analisis sensitivitas")
    c.setAuthor("Analisis v3.2_uji_takikardia/uji_takikardia.ipynb")
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
        c.drawString(M + 43, H - 42, "SIHEDAF  /  V3.2 UJI STRES TAKIKARDIA")
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", 21); c.drawString(M, H - 79, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 99, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, "16 Oktober 2026 | Sumber: v3.2_uji_takikardia/uji_takikardia.ipynb, cache/v3_2/, cache/oof_robust/")
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

    TH, BH, NE, SA, SW, RU = d["threshold"], d["by_hr"], d["nested"], d["sens_all"], d["sens_wo"], d["rule"]
    fast = BH["Normal | 120–∞ bpm"]; mid = BH["Normal | 110–120 bpm"]
    alarm = lambda b, m: b[m][1] + b[m][2]
    th31 = sorted(set(TH["v31"].values()))
    th_txt = "±" + "–".join(str(t) for t in th31) + " bpm"
    pv1_wo = SW["pairs_v31_minus"]["v1"][3]

    def verdict(p):
        return "lebih baik" if p >= .95 else "lebih buruk" if p <= .05 else "tidak terbedakan"

    # ── Hal 1 ──
    y = page("Laporan v3.2: uji stres takikardia", "Evaluasi saja, tanpa training; menindaklanjuti review eksternal atas paket v3.1")
    cardw = (CW - 24) / 4
    cards = ((th_txt, "Normal berhenti ditebak (v3.1)"), (pct(alarm(fast, "v31")), "Normal ≥120 bpm asli → alarm"),
             (fnum(NE["v2_prod"]["p"], 3), "P(v3.1 > v2), bersarang"), (fnum(pv1_wo, 2), "P(v3.1 > v1), tanpa 04936"))
    for i, (val, lab) in enumerate(cards):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 88, cardw, 88, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 20 if i else 17); c.drawString(x + 11, y - 37, val)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 7.9); c.drawString(x + 11, y - 61, lab)
    y -= 105
    y = callout("Jawaban singkat",
                f"<b>Temuan reviewer benar.</b> Semua paket (v1, v2, v3.1) berhenti menebak Normal pada ritme teratur di sekitar "
                f"110–115 bpm. Pada window Normal asli ≥120 bpm, v3.1 membunyikan alarm AFIB/AFL di {pct(alarm(fast, 'v31'))} window. "
                "Penyebabnya data latih: hanya 0,1% window Normal di AFDB yang di atas 120 bpm. Kesimpulan v3–v3.1 tetap berlaku, tetapi "
                f"sedikit kurang yakin: P(v3.1 > v2) {fnum(NE['v2_prod']['p'], 3)} setelah pemilihan bias ikut diacak, dan v3.1 vs v1 "
                f"menjadi tidak terbedakan tanpa pasien 04936 (P = {fnum(pv1_wo, 2)}).", y)
    y = section("Status klaim review", y)
    y = table(["Klaim", "Status", "Bukti di laporan ini"], [
        ["Ritme cepat teratur dianggap AFIB/AFL", "<b>Terbukti</b>", f"Ambang Normal v3.1 {th_txt}, semua jenis variasi (hal. 2)"],
        ["Di 120–130 bpm ditebak AFIB", "Sebagian", "AFIB jika variasi detak besar; AFL jika hampir tanpa variasi (hal. 2)"],
        ["Hanya 0,1% Normal > 120 bpm; AFL 78%", "<b>Terbukti</b>", "100 dari 99.000 window Normal; 903 dari 1.152 window AFL"],
        ["Bias −4,5 tidak menolong", "<b>Terbukti</b>", "v3.1 hanya sedikit lebih baik dari v2 di 110–120 bpm (hal. 3)"],
        ["Label AFL 04936 tidak bisa dipercaya", "Belum terverifikasi", "Tidak ada di halaman PhysioNet AFDB; sensitivitas tetap dihitung (hal. 4)"],
        ["CI belum memasukkan pemilihan bias", "<b>Benar</b>", "Bootstrap bersarang (hal. 4)"],
        ["Chapman–Shaoxing punya label sinus takikardia", "<b>Terbukti</b>", "Kode ST (SNOMED 427084000) di daftar kondisinya"],
    ], y, [175, 85, CW - 260])
    y = section("Cara uji", y)
    for txt in ["<b>Ritme teratur sintetis</b> (hal. 2): deretan interval RR buatan 60–170 bpm dengan sedikit variasi acak, lalu "
                "dimasukkan ke paket model asli lewat praproses produksi. Secara klinis ritme ini normal (sinus takikardia).",
                "<b>Window asli per pita detak</b> (hal. 3): prediksi OOF (out-of-fold: pasien uji tidak pernah ikut dilatih) dari "
                "evaluasi v3–v6, dikelompokkan menurut detak jantung window.",
                "<b>Sensitivitas</b> (hal. 4): evaluasi ulang tanpa pasien 04936, dan CI yang ikut memperhitungkan pemilihan bias kelas.",
                "<b>Aturan produk</b> (hal. 5): berapa banyak window AFDB yang akan diganti labelnya oleh aturan \"takikardia teratur\"."]:
        y = para(f"• {txt}", y, gap=4)

    # ── Hal 2 ──
    y = page("Uji sintetis: ritme teratur 60–170 bpm", f"{d['n_synth']} window 10 s per titik; model yang punya bobot (v4–v6 tidak menyimpan bobot)")
    y = chart("stress", y, 175)
    rows = []
    for m in ("v1", "v2_prod", "v31"):
        rows.append([SHORT[m]] + [f"{TH[m][j]} bpm" for j in d["jitters"]])
    y = table(["Model"] + [f"Ambang: {j}" for j in d["jitters"]], rows, y, [105] + [(CW - 105) / 4] * 4)
    st = d["stress"]["v31"]
    i120 = d["bpms"].index(120); i140 = d["bpms"].index(140)
    y = table(["v3.1 di 120 bpm", "Normal", "AFIB", "AFL"],
              [[j, pct(st[j][i120][0]), pct(st[j][i120][1]), pct(st[j][i120][2])] for j in d["jitters"]],
              y, [175, (CW - 175) / 3, (CW - 175) / 3, (CW - 175) / 3])
    callout("Cara membaca",
            "Ambang = detak terendah di mana kurang dari separuh window ditebak Normal. Area abu-abu di grafik = di atas 100 bpm (takikardia); garis yang tidak terlihat tertimpa garis v3.1 karena nilainya sama. Ketiga paket punya ambang di 110–115 bpm, "
            "jadi masalah ini sudah ada sejak v1 dan bukan dibuat oleh revisi. Di atas ambang, kelas yang ditebak bergantung pada "
            "variasi detak: ritme yang sangat teratur ditebak AFL, variasi sedikit lebih besar (sd 20 ms) ditebak AFIB. Keduanya salah "
            "untuk sinus takikardia. Untuk ritme di atas ±140 bpm yang benar-benar teratur, AFL 2:1 memang tidak bisa dibedakan dari "
            "interval RR saja.", y)

    # ── Hal 3 ──
    y = page("Window asli per pita detak jantung", "Prediksi OOF (rata-rata 5 ulangan) pada window AFDB; semua model yang punya OOF")
    y = chart("real_bins", y, 165)
    order = ["v1", "v2_prod", "v31", "v4_e1", "v5_lunak", "v6_late"]
    rows = []
    for key in ("Normal | 100–110 bpm", "Normal | 110–120 bpm", "Normal | 120–∞ bpm", "AFL | 120–∞ bpm"):
        b = BH[key]
        name = key.replace("120–∞", "≥120").replace("| ", "") + f" ({b['n']:,}".replace(",", ".") + f" window, {b['n_pasien']} pasien)"
        if key.startswith("AFL"):
            rows.append([name] + [pct(b[m][2]) for m in order])
        else:
            rows.append([name] + [pct(alarm(b, m)) for m in order])
    y = table(["Kelas & pita (Normal: % alarm; AFL: % terdeteksi)"] + [SHORT[m] for m in order], rows, y,
              [175] + [(CW - 175) / 6] * 6)
    cvd = d["cv_desc"]
    callout("Yang terlihat",
            f"Alarm palsu naik tajam mulai 110 bpm untuk semua model: {pct(alarm(mid, 'v31'))} pada 110–120 bpm dan "
            f"{pct(alarm(fast, 'v31'))} pada ≥120 bpm (v3.1). Late fusion v6 sedikit lebih baik ({pct(alarm(fast, 'v6_late'))}), tetapi "
            f"masalahnya sama. v1 mendeteksi AFL cepat paling banyak ({pct(BH['AFL | 120–∞ bpm']['v1'][2])}) karena cenderung menyebut "
            f"semua ritme cepat AFL: {pct(BH['AFIB | 120–∞ bpm']['v1'][2])} window AFIB ≥120 bpm juga ditebak AFL. Window Normal cepat di AFDB justru <b>tidak</b> teratur (koefisien variasi median "
            f"{fnum(cvd['Normal ≥120 bpm']['median'], 2)} vs {fnum(cvd['Normal']['median'], 3)} untuk semua Normal), kemungkinan "
            "karena denyut ektopik di pasien AF. Sinus takikardia teratur hampir tidak ada di AFDB, jadi skala masalah di dunia nyata "
            "belum bisa diukur dengan data ini.", y)

    # ── Hal 4 ──
    y = page("Sensitivitas: tanpa 04936 dan bootstrap bersarang", "Apakah kesimpulan v3/v3.1 bergantung pada satu pasien atau pada pemilihan bias?")
    y = para(f"Pasien 04936 menyumbang {pct(d['afl_share_suspect'])} window AFL. Reviewer menyebut labelnya dipertanyakan; halaman "
             "PhysioNet AFDB tidak menyebut hal ini, jadi sumbernya belum terverifikasi. Window 04936 dikeluarkan dari evaluasi (bias "
             "cross-fit dihitung ulang tanpa dia); model tidak dilatih ulang.", y)
    def ci(v, nd=3):
        return f"{fnum(v[0], nd)} <font color='{GRAY}' size='7.4'>[{fnum(v[1], 2)}–{fnum(v[2], 2)}]</font>"

    rows = []
    for lab, s in (("semua pasien", SA), ("tanpa 04936", SW)):
        for k in ("v1", "v2_prod", "v31"):
            r = s["rows"][k]
            p = "—" if k == "v31" else f"{fnum(s['pairs_v31_minus'][k][3], 2)} ({verdict(s['pairs_v31_minus'][k][3])})"
            rows.append([lab if k == "v1" else "", SHORT[k], ci(r["Macro-F1"]), ci(r["AFL F1"]), ci(r["AFL R"]), p])
    y = table(["Data", "Model", "Macro-F1", "AFL F1", "AFL recall", "P(v3.1 > model)"], rows, y,
              [62, 68, 96, 96, 96, CW - 418], bold_rows=[6])
    y = section("Bootstrap bersarang (bias dipilih ulang di setiap sampel)", y)
    y = para("CI di v3 dihitung dari prediksi yang biasnya sudah dipilih sekali per fold, lalu pasien diambil ulang 2.000 kali. "
             "Padahal bias itu sendiri dipilih dari data, jadi bisa saja berbeda bila pasiennya berbeda. Pada bootstrap "
             "<b>bersarang</b>, untuk setiap sampel pasien, bias v3.1 dipilih ulang dari pasien latih yang terambil, lalu dipasang ke "
             "pasien uji. Ketidakpastian pemilihan bias ikut masuk ke CI, sehingga CI-nya lebih lebar dan lebih jujur.", y)
    y = table(["Perbandingan", "Δ Macro-F1", "CI 95% bersarang", "P(Δ>0) bersarang", "P(Δ>0) v3 (bias tetap)"],
              [[f"v3.1 vs {SHORT[k]}", sgn(NE[k]["delta"]), f"{sgn(NE[k]['ci'][0])} … {sgn(NE[k]['ci'][1])}",
                fnum(NE[k]["p"], 3), fnum(SA["pairs_v31_minus"][k][3], 3)] for k in ("v2_prod", "v1")],
              y, [130, 70, 120, 90, CW - 410])
    callout("Artinya",
            f"Keunggulan v3.1 atas v2 bertahan di kedua uji (P ≥ {fnum(min(NE['v2_prod']['p'], SW['pairs_v31_minus']['v2_prod'][3]), 2)}). "
            f"Keunggulan atas v1 lebih rapuh: tetap konsisten setelah pemilihan bias diacak (P = {fnum(NE['v1']['p'], 3)}), tetapi tanpa "
            f"04936 turun ke P = {fnum(pv1_wo, 2)} (tidak terbedakan). Klaim \"P = 1,00\" di laporan v3.1 memang terlalu yakin; "
            "angka bersarang di sini yang lebih jujur.", y)

    # ── Hal 5 ──
    y = page("Simulasi aturan produk", "Detak ≥ H bpm DAN ritme teratur (koefisien variasi IBI ≤ C) → \"takikardia teratur, perlu dicek\"")
    y = para("Aturan ini tidak memperbaiki model; ia mengganti label keluaran untuk ritme cepat yang teratur. Ritme sinus takikardia "
             "teratur hampir tidak ada di AFDB, jadi <b>manfaatnya tidak bisa diukur di sini</b>. Yang bisa diukur adalah harganya: "
             "berapa banyak deteksi AFL yang ikut dialihkan, karena AFL 2:1 juga cepat dan teratur.", y)
    def pct1(v):
        return f"{v * 100:.1f}%".replace(".", ",")

    y = table(["Aturan", "Normal kena", "AFIB kena", "AFL kena", "Alarm palsu dialihkan", "AFL terdeteksi dialihkan"],
              [[k.replace(".", ","), pct1(RU[k]["Normal kena aturan"]), pct1(RU[k]["AFIB kena aturan"]), pct1(RU[k]["AFL kena aturan"]),
                pct1(RU[k]["alarm palsu dialihkan"]), pct1(RU[k]["AFL terdeteksi dialihkan"])] for k in RU],
              y, [120, 60, 60, 60, 95, CW - 395], bold_rows=[5])
    y = para("<i>Kena</i> = % window kelas itu (label asli) yang memenuhi aturan. <i>Alarm palsu dialihkan</i> = % window Normal yang "
             "ditebak AFIB/AFL dan diganti labelnya oleh aturan. <i>AFL terdeteksi dialihkan</i> = % window AFL yang benar ditebak AFL "
             "lalu diganti labelnya. CV = koefisien variasi IBI (simpangan baku / rata-rata); makin kecil, makin teratur.", y, "small")
    sc = d["synthetic_cv"]
    y = table(["Keteraturan (koefisien variasi IBI, median)", "Nilai"],
              [["Window AFL (AFDB)", fnum(cvd["AFL"]["median"])], ["Window AFIB (AFDB)", fnum(cvd["AFIB"]["median"])],
               ["Sintetis Gauss sd 20 ms @ 120 bpm", fnum(sc["Gauss sd 20 ms @ 120 bpm"])],
               ["Sintetis Gauss sd 20 ms @ 170 bpm", fnum(sc["Gauss sd 20 ms @ 170 bpm"])]],
              y, [260, CW - 260])
    k0 = "≥110 bpm & CV ≤ 0.05"
    callout("Trade-off untuk pemilik produk",
            f"Dengan aturan ≥110 bpm & CV ≤ 0,05 (baris disorot), hampir semua ritme teratur sintetis (CV 0–0,056) tidak lagi "
            f"dilaporkan sebagai AFIB/AFL dan AFIB nyaris tidak tersentuh ({pct1(RU[k0]['AFIB kena aturan'])}). Tetapi {pct(RU[k0]['AFL terdeteksi dialihkan'])} AFL yang "
            "berhasil dideteksi berubah label menjadi \"takikardia teratur, perlu dicek\". Pengguna tetap diberi peringatan, hanya "
            "tidak disebut AFL. Mau tidaknya memakai aturan ini adalah keputusan produk, bukan keputusan teknis.", y, "#FFF4EC")

    # ── Hal 6 ──
    y = page("Kesimpulan dan langkah berikutnya", "Yang berubah setelah review, dan yang belum bisa dijawab")
    y = table(["Hal", "Keputusan / status"], [
        ["Paket produksi v3.1", "Tetap; tidak ada perubahan bobot atau bias"],
        ["BACKEND.md", "Ditambah peringatan: ritme teratur ≥ ±115 bpm dilaporkan AFIB/AFL"],
        ["Protokol evaluasi (AGENTS.md)", "Uji stres takikardia wajib untuk setiap model baru; CI bersarang disarankan"],
        ["Aturan produk \"takikardia teratur\"", "<b>Menunggu keputusan pemilik produk</b> (trade-off di hal. 5)"],
        ["Klaim di laporan v3.1 (P = 1,00)", "Dikoreksi: P = " + fnum(NE["v2_prod"]["p"], 3) + " dengan pemilihan bias ikut diacak"],
    ], y, [190, CW - 190])
    y = section("Langkah berikutnya", y)
    for txt in ["<b>v7 dengan data eksternal yang memuat sinus takikardia</b> (mis. Chapman–Shaoxing: label ST, SVT, AF/flutter). "
                "Sekaligus menambah pasien AFL dan memungkinkan model belajar bahwa cepat tidak berarti tidak normal.",
                "Validasi pada rekaman perangkat asli saat aktivitas (naik tangga, olahraga ringan).",
                "Mencari sumber literatur soal anotasi 04936 sebelum memutuskan mengeluarkannya dari data latih."]:
        y = para(f"• {txt}", y, gap=5)
    y = section("Batasan", y)
    for txt in ["Uji sintetis memakai ritme dengan variasi acak sederhana; sinus takikardia sungguhan punya pola variabilitas sendiri.",
                "Analisis tanpa 04936 hanya mengeluarkannya dari evaluasi, tidak dari training.",
                "Model v4–v6 tidak menyimpan bobot, jadi hanya diuji di window AFDB (hal. 3), tidak di data sintetis.",
                "Bootstrap bersarang hanya dihitung untuk Macro-F1 v3.1 vs v2 dan v1.",
                "Reproduksi: <font name='Courier' size='8.3'>python tools/run_notebook.py v3.2_uji_takikardia/uji_takikardia.ipynb</font> "
                "(±10 menit pertama kali, CPU), lalu <font name='Courier' size='8.3'>laporan/build_report.py --extract</font> dan tanpa argumen."]:
        y = para(f"• {txt}", y, gap=5)
    c.save()
    print(f"OK {OUT} ({state['page']} halaman)")


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else build()
