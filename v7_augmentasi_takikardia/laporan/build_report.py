"""Laporan PDF v7 (augmentasi sinus takikardia sintetis).

Dua tahap (pola sama dengan v3–v6):
  1. --extract  (Python dengan numpy/torch/matplotlib): hitung ulang lewat augmentasi.evaluate() dari cache,
     cocokkan dengan ../cache/v7/ringkasan.json dari notebook, buat grafik → laporan/_build/.
  2. tanpa argumen (Python dengan ReportLab): rakit laporan/laporan_augmentasi_takikardia.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # = v7_augmentasi_takikardia/
REPO = HERE.parent
TMP = HERE / "laporan/_build"
OUT = HERE / "laporan/laporan_augmentasi_takikardia.pdf"
SUMMARY = REPO / "cache/v7/ringkasan.json"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402

TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
MCOL = {"v31_paket": "#8A97A6", "v31": "#8A97A6", "kontrol": "#eb6834", "bobotlunak": "#eb6834",
        "takikardia130": "#2a78d6", "takikardia160": "#1baf7a"}
MMARK = {"v31_paket": "s", "v31": "s", "kontrol": "^", "bobotlunak": "^", "takikardia130": "o", "takikardia160": "D"}
SHORT = {"v1": "v1", "v31": "v3.1", "v31_paket": "v3.1 paket produksi", "bobotlunak": "v5b bobot lunak",
         "kontrol": "kontrol (resep v5b)", "takikardia130": "7a: sintetis 100–130", "takikardia160": "7b: sintetis 100–160"}


def extract():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter

    sys.path.insert(0, str(HERE))
    import augmentasi as A

    TMP.mkdir(parents=True, exist_ok=True)
    res = A.evaluate()
    nb = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert nb["threshold"] == res["threshold"], "uji stres tidak sama dengan notebook"
    assert nb["parity"] == res["parity"]
    for k, r in res["rows"].items():
        for m in ("Macro-F1", "AFL F1", "AFL R", "Normal R"):
            assert abs(nb["rows"][k][m][0] - r[m][0]) < 1e-12, (k, m)
    for k, p in res["pairs"].items():
        assert abs(nb["pairs"][k]["Macro-F1"][3] - p["Macro-F1"][3]) < 1e-12, k
    for k, p in res["pairs_v5"].items():
        assert abs(nb["pairs_v5"][k]["Macro-F1"][3] - p["Macro-F1"][3]) < 1e-12, k
    assert nb["heldout"] == res["heldout"]

    # contoh sintetis vs AFL asli (sama dengan notebook bagian 1)
    z = np.load(A.DATA_STD)
    y, rec, Mz, Iz = z["y"], z["records"].astype(str), z["M"], z["X"][:, :, 0]
    mean = (Iz * Mz).sum(1) / Mz.sum(1)
    hr = 60000 / mean
    cvv = np.sqrt((((Iz - mean[:, None]) ** 2) * Mz).sum(1) / Mz.sum(1)) / mean
    ibis, shr, _ = A.synth_windows(A.load_beats(), np.unique(rec), 3000, A.VARIANTS["takikardia160"]["hr"], seed=1)
    scv = np.array([w.std() / w.mean() for w in ibis])
    res["synth_desc"] = {"cv_median": float(np.median(scv)), "cv_p90": float(np.percentile(scv, 90)),
                         "normal_cv_gt": float(np.mean(cvv[y == 0] > A.CV_MAX)),
                         "afib_cv_p10": float(np.percentile(cvv[y == 1], 10)),
                         "afl_130_155": float(np.mean((hr[y == 2] >= 130) & (hr[y == 2] < 155))),
                         "afl_100_130": int(((hr[y == 2] >= 100) & (hr[y == 2] < 130)).sum()),
                         "n_afl": int((y == 2).sum()), "cv_max": A.CV_MAX, "n_syn": A.N_SYN, "n_ho": A.N_HO}
    (TMP / "tables.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

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

    # 1. sintetis vs AFL asli
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    m = y == 2
    ax.scatter(shr[:1500], scv[:1500], s=5, color="#2a78d6", alpha=.35, label="sintetis 100–160 bpm (Normal)", marker="o", lw=0)
    ax.scatter(hr[m], cvv[m], s=9, color="#eb6834", alpha=.7, label="AFL asli (AFDB)", marker="^", lw=0)
    ax.axvspan(130, 155, color="#F6F7F9", zorder=0)
    ax.text(142.5, .285, "daerah AFL 2:1", ha="center", fontsize=8, color=GRAY)
    ax.set_xlim(60, 170); ax.set_ylim(0, .3)
    ax.set_xlabel("detak (bpm)", fontsize=8.5); ax.set_ylabel("CV IBI (makin kecil makin teratur)", fontsize=8.5)
    ax.grid(color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", markerscale=2)
    fig.tight_layout(); save(fig, "sintetis")

    # 2. uji stres
    keys = ["v31_paket", "kontrol", "takikardia130", "takikardia160"]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.4), sharey=True)
    for ax, j in zip(axes, res["jitters"]):
        ax.axvspan(100, 170, color="#F6F7F9", zorder=0)
        for k in keys:
            ax.plot(res["bpms"], [t[0] for t in res["stress"][k][j]], marker=MMARK[k], color=MCOL[k], ms=3.2, lw=1.5,
                    label=SHORT[k])
        ax.set_title(j, loc="left", fontsize=8.8, fontweight="bold")
        ax.set_xticks([60, 90, 120, 150]); ax.tick_params(labelsize=8)
        ax.grid(color="#E6EBF0", lw=.8); ax.set_axisbelow(True); ax.yaxis.set_major_formatter(pctf)
        ax.set_xlabel("bpm", fontsize=8.5)
    axes[0].set_ylabel("ditebak Normal", fontsize=8.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.12), ncol=4,
               frameon=False, fontsize=8.3)
    fig.tight_layout(); save(fig, "stress")

    # 3. sinus takikardia sintetis dari pasien uji
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    x = np.arange(len(res["ho_bins"]))
    for k in keys:
        ax.plot(x, [res["heldout"][k][b][0] for b in res["ho_bins"]], marker=MMARK[k], color=MCOL[k], ms=6, lw=1.8,
                label=SHORT[k])
    ax.set_xticks(x, [f"{b} bpm" for b in res["ho_bins"]], fontsize=8.5)
    ax.set_ylabel("ditebak Normal", fontsize=8.5); ax.yaxis.set_major_formatter(pctf); ax.set_ylim(0, 1.03)
    ax.grid(axis="y", color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.3, loc="upper right")
    fig.tight_layout(); save(fig, "heldout")
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
        "table": ParagraphStyle("table", fontName="Arial", fontSize=8.4, leading=11.2, textColor=HexColor(INK)),
        "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8.3, leading=11.2, textColor=white),
    }
    c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
    c.setTitle("Laporan v7 - Augmentasi sinus takikardia sintetis")
    c.setAuthor("Analisis v7_augmentasi_takikardia/augmentasi_takikardia.ipynb")
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
        return f"{fnum(v[0], nd)} <font color='{GRAY}' size='7.4'>[{fnum(v[1], 2)}–{fnum(v[2], 2)}]</font>"

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
        c.drawString(M + 43, H - 42, "SIHEDAF  /  V7 AUGMENTASI SINUS TAKIKARDIA")
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", 21); c.drawString(M, H - 79, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 99, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, "16 Oktober 2026 | Sumber: v7_augmentasi_takikardia/augmentasi_takikardia.ipynb, cache/v7/, cache/oof_robust/")
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

    R, P, P5, TH, HO, BH, SD = d["rows"], d["pairs"], d["pairs_v5"], d["threshold"], d["heldout"], d["by_hr"], d["synth_desc"]
    A7, B7 = "takikardia130", "takikardia160"
    th = lambda k: min(v for v in TH[k].values())
    th_txt = lambda k: (f"{th(k)} bpm" if len(set(TH[k].values())) == 1
                        else f"{min(TH[k].values())}–{max(TH[k].values())} bpm")
    alarm = lambda b, k: BH[b][k][1] + BH[b][k][2]
    pa = P[A7]["Macro-F1"]
    crit = {"Macro-F1 tidak lebih buruk dari v3.1": pa[3] > .05,
            "ambang ≥ 125 bpm di keempat jenis variasi": th(A7) >= 125,
            "recall AFL per pasien tidak runtuh": True}
    afl_pat = d["afl_patients"]
    rec31, rec7 = d["afl_recall"]["v31"], d["afl_recall"][A7]
    drops = [(p, a, b) for p, a, b in zip(afl_pat, rec31, rec7) if a - b >= .2]
    crit["recall AFL per pasien tidak runtuh"] = len(drops) == 0
    lolos = all(crit.values())
    crit_list = list(crit.values())
    E = d["episodes"]

    # ── Hal 1: ringkasan ──
    y = page("Laporan v7: augmentasi sinus takikardia", "Contoh Normal cepat sintetis dari detak asli pasien latih; tanpa data baru")
    cardw = (CW - 24) / 4
    cards = ((f"{th(A7)} bpm", "7a: ambang Normal (v3.1: 115)"),
             (pct(HO[A7]["120–130"][0]), "7a: takikardia 120–130 → Normal"),
             (sgn(pa[0]), f"Δ Macro-F1 7a vs v3.1, P = {fnum(pa[3], 2)}"),
             (fnum(R[A7]["AFL R"][0], 2), f"recall AFL 7a (v3.1: {fnum(R['v31']['AFL R'][0], 2)})"))
    for i, (val, lab) in enumerate(cards):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 88, cardw, 88, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 20); c.drawString(x + 11, y - 37, val)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 7.6); c.drawString(x + 11, y - 61, lab)
    y -= 105
    p08 = afl_pat.index("08378")
    y = callout("Jawaban singkat",
                f"<b>Sebagian bisa.</b> Contoh sinus takikardia sintetis 100–130 bpm (7a) menggeser batas alarm palsu dari 115 ke "
                f"{th(A7)} bpm, dan Macro-F1 tidak berubah secara terukur ({fnum(R[A7]['Macro-F1'][0])} vs "
                f"{fnum(R['v31']['Macro-F1'][0])}, P = {fnum(pa[3], 2)}). Harganya di AFL: recall AFL {fnum(R['v31']['AFL R'][0])} → "
                f"{fnum(R[A7]['AFL R'][0])}, terutama pasien 08378 yang AFL 2:1-nya sangat teratur ({fnum(rec31[p08], 2)} → "
                f"{fnum(rec7[p08], 2)}). Di atas ±135 bpm masalahnya tetap ada: varian 100–160 bpm (7b) rusak, karena di rentang itu "
                "sinus takikardia dan AFL 2:1 tidak bisa dibedakan dari interval RR. <b>Paket produksi tetap v3.1</b>; 7a kandidat bila "
                "pemilik produk menerima harganya.", y)
    y = section("Hipotesis dan hasilnya", y)
    y = table(["Varian", "Hipotesis (sebelum hasil)", "Hasil", "Status"], [
        ["7a: sintetis 100–130 bpm", "Ambang ≥ 125 bpm, Macro-F1 tidak lebih buruk dari v3.1",
         f"Ambang {th_txt(A7)}; Δ Macro-F1 {sgn(pa[0])} (P = {fnum(pa[3], 2)}, {verdict(pa[3])}); recall AFL turun",
         "<b>Terbukti</b>, dengan harga AFL"],
        ["7b: sintetis 100–160 bpm", "Ambang naik lebih jauh, recall AFL turun",
         f"Ambang {th_txt(B7)} (rusak); Macro-F1 {fnum(R[B7]['Macro-F1'][0])}, {verdict(P[B7]['Macro-F1'][3])}",
         "<b>Ditolak</b>"],
        ["kontrol (resep v5b)", "OOF sama dengan v5b", f"selisih probabilitas maks {fnum(d['parity']['selisih prob maks'], 1)}",
         "<b>Terbukti</b>"],
    ], y, [105, 140, CW - 340, 95])
    y = section("Kriteria kandidat produksi untuk 7a (ditetapkan sebelum hasil)", y)
    y = table(["Kriteria", "Hasil", "Status"], [
        ["Macro-F1 tidak lebih buruk dari v3.1 (P > 0,05)", f"P = {fnum(pa[3], 2)}", "lolos" if crit_list[0] else "<b>tidak lolos</b>"],
        ["Ambang ≥ 125 bpm di keempat jenis variasi", th_txt(A7), "lolos" if crit_list[1] else "<b>tidak lolos</b>"],
        ["Recall AFL per pasien tidak runtuh (turun < 0,20)",
         "; ".join(f"{p} {fnum(a, 2)} → {fnum(b, 2)}" for p, a, b in drops) or "semua pasien turun < 0,20",
         "lolos" if crit_list[2] else "<b>tidak lolos</b>"],
    ], y, [230, CW - 320, 90])

    # ── Hal 2: perbandingan ──
    y = page("Perbandingan dengan versi sebelumnya", "Window uji detektor 50 Hz, 5 ulangan × 5 fold pasien, bias cross-fit, CI 95% bootstrap pasien")
    order = ["v1", "v31", "bobotlunak", A7, B7]
    rows = [[SHORT[k], ci(R[k]["Macro-F1"]), ci(R[k]["AFL F1"]), ci(R[k]["AFL R"]), fnum(R[k]["Normal R"][0]),
             fnum(E[k]["alarm/jam"][0], 2), pct(E[k]["deteksi episode"][0])] for k in order]
    y = table(["Model", "Macro-F1", "AFL F1", "AFL recall", "Normal recall", "AFL palsu / jam", "episode AFL terdeteksi"], rows, y,
              [82, 86, 86, 86, 55, 55, CW - 450], bold_rows=[4])
    y = para("<i>OOF</i> (out-of-fold) = prediksi untuk pasien yang tidak ikut dilatih. <i>CI</i> = rentang 95% dari 2.000 kali "
             "mengambil ulang pasien. <i>Bias</i> = geseran skor kelas yang dipilih dari fold lain (cross-fit). v1 = sistem sebelum "
             "revisi, v3.1 = paket produksi, v5b = resep yang sama dengan v7 tanpa data sintetis.", y, "small")
    y = section("Selisih berpasangan (sampel bootstrap sama)", y)
    rows = []
    for k in (A7, B7):
        for ref, PP in (("v3.1", P[k]), ("v5b", P5[k])):
            rows.append([SHORT[k] if ref == "v3.1" else "", ref] +
                        [f"{sgn(PP[m][0])} <font color='{GRAY}' size='7.4'>(P = {fnum(PP[m][3], 2)})</font><br/>{verdict(PP[m][3])}"
                         for m in ("Macro-F1", "AFL R", "Normal R")])
    y = table(["Varian", "vs", "Δ Macro-F1", "Δ AFL recall", "Δ Normal recall"], rows, y, [120, 45, (CW - 165) / 3, (CW - 165) / 3,
                                                                                         (CW - 165) / 3])
    callout("Cara membaca",
            f"7a setara dengan v3.1 dan v5b di Macro-F1, sedikit lebih baik di recall Normal, dan lebih buruk dari v5b di recall AFL "
            f"(Δ {sgn(P5[A7]['AFL R'][0])}, P = {fnum(P5[A7]['AFL R'][3], 2)}). Jadi data sintetis menukar sebagian deteksi AFL dengan "
            f"lebih sedikit alarm palsu Normal. 7b lebih buruk di hampir semua metrik dan membunyikan AFL palsu "
            f"{fnum(E[B7]['alarm/jam'][0], 1)} kali per jam (v3.1: {fnum(E['v31']['alarm/jam'][0], 2)}).", y)

    # ── Hal 3: metode ──
    y = page("Cara membuat sinus takikardia sintetis", "Detak Normal asli dipadatkan waktunya; hanya dari pasien latih di setiap fold")
    for txt in [f"Urutan detak 23 rekaman diekstrak ulang dengan detektor produksi, lalu <b>dibuktikan identik</b> dengan dataset v2: "
                "window yang dipotong darinya sama bit demi bit (165.978 window).",
                "Untuk setiap contoh: pilih detak Normal acak dari pasien latih, pilih detak target acak dalam rentang varian, kalikan IBI "
                "dengan s = IBI target / rata-rata IBI sumber (s < 1, jadi hanya dipercepat), lalu potong 10 detik dengan fase acak.",
                f"Sinus takikardia itu teratur. Contoh dengan CV IBI > {fnum(SD['cv_max'], 2)} dibuang: {pct(SD['normal_cv_gt'])} window "
                f"Normal asli melebihi batas ini (denyut ektopik), sedangkan AFIB hampir selalu di atasnya (p10 = {fnum(SD['afib_cv_p10'], 2)}). "
                "Cepat dan tidak teratur adalah ciri AFIB, jadi tidak boleh diajarkan sebagai Normal.",
                f"{thou(SD['n_syn'])} contoh per fold (±7% window Normal latih), berlabel Normal. Resep training sama dengan v5b "
                "(bobot kelas lunak), fold, seed, dan hiperparameter sama dengan referensi. Fungsi training diimpor dari v5."]:
        y = para(f"• {txt}", y, gap=5)
    y -= 4
    y = chart("sintetis", y, 190)
    callout("Kenapa dua rentang",
            f"Di AFDB, {pct(SD['afl_130_155'])} window AFL berada di 130–155 bpm (AFL 2:1) dan hanya {SD['afl_100_130']} dari "
            f"{thou(SD['n_afl'])} di 100–130 bpm. 7a hanya mengisi daerah yang jarang AFL; 7b sengaja "
            "menutupi daerah AFL 2:1 untuk mengukur harga penuhnya. Grafik di atas memperlihatkan masalahnya: di 130–155 bpm, contoh "
            "Normal sintetis dan AFL asli sama-sama teratur.", y)

    # ── Hal 4: uji stres ──
    y = page("Uji stres takikardia", "Protokol butir 9: ritme teratur sintetis v3.2 dan sinus takikardia sintetis dari pasien uji")
    y = chart("stress", y, 170)
    y = table(["Model"] + [f"Ambang: {j}" for j in d["jitters"]],
              [[SHORT[k]] + [f"{TH[k][j]} bpm" for j in d["jitters"]] for k in ("v31_paket", "kontrol", A7, B7)],
              y, [120] + [(CW - 120) / 4] * 4, bold_rows=[3])
    y = section("Sinus takikardia sintetis dari pasien uji (lebih mirip detak asli)", y)
    y = chart("heldout", y, 150)
    callout("Yang terlihat",
            f"Pada model fold, ambang naik dari 115–120 bpm (kontrol) ke {th_txt(A7)} (7a). Pada sinus takikardia sintetis dari pasien "
            f"yang tidak ikut dilatih, 7a menebak Normal {pct(HO[A7]['110–120'][0])} di 110–120 bpm dan {pct(HO[A7]['120–130'][0])} di "
            f"120–130 bpm (paket v3.1: {pct(HO['v31_paket']['110–120'][0])} dan {pct(HO['v31_paket']['120–130'][0])}). Di atas 130 bpm "
            f"alarm tetap berbunyi ({pct(HO[A7]['130–140'][0])} Normal di 130–140, {pct(HO[A7]['140–150'][0])} di atas 140). "
            "7b tidak stabil: bias AFL +1,5 yang dipilih cross-fit membuat ritme sangat teratur di 85–90 bpm (mirip kelompok AFL lambat "
            "di hal. 3) ditebak abnormal, sehingga ambangnya tercatat 85 bpm, dan di 120–145 bpm alarm tetap berbunyi.", y)

    # ── Hal 5: per pita detak & per pasien ──
    y = page("Window asli per pita detak dan per pasien AFL", "Prediksi OOF pada window AFDB; rata-rata 5 ulangan")
    ks = ["v31", "bobotlunak", A7, B7]
    rows = []
    for key in ("Normal | 100–110 bpm", "Normal | 110–120 bpm", "Normal | 120–∞ bpm", "AFL | 110–120 bpm", "AFL | 120–∞ bpm"):
        b = BH[key]
        name = key.replace("120–∞", "≥120").replace("| ", "") + f" ({b['n']:,}".replace(",", ".") + " window)"
        rows.append([name] + [pct(b[k][2]) if key.startswith("AFL") else pct(alarm(key, k)) for k in ks])
    y = table(["Kelas & pita (Normal: % alarm; AFL: % terdeteksi)"] + [SHORT[k] for k in ks], rows, y,
              [175] + [(CW - 175) / 4] * 4)
    order_p = ["v1", "v31", "bobotlunak", A7, B7]
    rows = [[p] + [fnum(d["afl_recall"][k][i], 2) for k in order_p] for i, p in enumerate(afl_pat)]
    y = table(["Recall AFL per pasien"] + [SHORT[k] for k in order_p], rows, y, [115] + [(CW - 115) / 5] * 5,
              bold_rows=[p08 + 1])
    y = table(["Sumber AFL palsu & bias"] + [SHORT[k] for k in order_p], [
        ["AFL palsu / ulangan"] + [thou(d["fp"][k]["total"]) for k in order_p],
        ["… dari pasien tanpa AFL"] + [thou(d["fp"][k]["pasien_lain"]) for k in order_p],
        ["bias median (Normal, AFIB, AFL)"] + ["—" if k == "v1" else "; ".join(fnum(v, 2) for v in d["bias"][k]["median"])
                                              for k in order_p]],
        y, [115] + [(CW - 115) / 5] * 5)
    callout("Yang terlihat",
            f"Window Normal asli ≥120 bpm (umumnya tidak teratur) yang membunyikan alarm turun dari "
            f"{pct(alarm('Normal | 120–∞ bpm', 'v31'))} ke {pct(alarm('Normal | 120–∞ bpm', A7))}. Deteksi AFL ≥120 bpm turun dari "
            f"{pct(BH['AFL | 120–∞ bpm']['v31'][2])} ke {pct(BH['AFL | 120–∞ bpm'][A7][2])}; kerugian terbesar di 08378, yang AFL-nya "
            "paling teratur. 07910 dan 08215 tetap tidak terdeteksi oleh semua model.", y)

    # ── Hal 6: sensitivitas, kesimpulan ──
    y = page("Kesimpulan dan langkah berikutnya", "Termasuk sensitivitas tanpa 04936 dan batasan")
    rows = []
    for lab_s, s in d["sens"].items():
        for k in ("v31", A7, B7):
            r = s["rows"][k]
            pv = "—" if k == "v31" else f"{fnum(s['pairs_v31_minus'][k][3], 2)}"
            rows.append([lab_s if k == "v31" else "", SHORT[k], ci(r["Macro-F1"]), ci(r["AFL F1"]), ci(r["AFL R"]), pv])
    y = table(["Data", "Model", "Macro-F1", "AFL F1", "AFL recall", "P(v3.1 > model)"], rows, y,
              [68, 92, 92, 92, 92, CW - 436])
    y = table(["Hal", "Keputusan / status"], [
        ["Paket produksi", "Tetap v3.1; v7 tidak dikemas"],
        ["7a (100–130 bpm)", "<b>Kandidat, menunggu keputusan pemilik produk</b>: alarm palsu pada sinus takikardia sintetis 100–130 bpm "
                             "hampir hilang, dibayar dengan "
                             "sebagian deteksi AFL 2:1 (08378). Sebelum dikemas: latih ulang di semua pasien, bootstrap bersarang, "
                             "validasi pada sinus takikardia asli"],
        ["7b (100–160 bpm)", "Ditolak. Di 130–160 bpm interval RR tidak cukup untuk membedakan sinus takikardia dari AFL 2:1"],
        ["Resep", "Augmentasi sintetis tersedia di <font name='Courier' size='8'>augmentasi.synth_windows</font> untuk model berikutnya"],
    ], y, [110, CW - 110])
    y = section("Langkah berikutnya", y)
    for txt in ["<b>Data eksternal dengan sinus takikardia asli dan bentuk gelombang EKG</b> (mis. Chapman–Shaoxing): satu-satunya "
                "jalan untuk 135–160 bpm, sekaligus untuk memvalidasi hasil sintetis ini. Unduhan butuh persetujuan.",
                "Jika 7a dipilih: v7.1 = latih di semua pasien, bootstrap bersarang (protokol butir 10), uji pada rekaman perangkat "
                "saat aktivitas."]:
        y = para(f"• {txt}", y, gap=5)
    y = section("Batasan", y)
    for txt in ["Semua manfaat diukur pada sinus takikardia <b>sintetis</b>. Detak asli yang dipercepat mempertahankan variabilitas "
                "relatifnya, padahal pada takikardia sungguhan variabilitas biasanya mengecil.",
                "Sumber sintetis adalah pasien AF di AFDB; detak Normal mereka belum tentu mewakili orang sehat.",
                "Reproduksi: <font name='Courier' size='8.3'>python tools/run_notebook.py v7_augmentasi_takikardia/augmentasi_takikardia.ipynb</font> "
                "(±45 menit GPU, bisa dilanjutkan), lalu <font name='Courier' size='8.3'>laporan/build_report.py --extract</font> "
                "dan tanpa argumen."]:
        y = para(f"• {txt}", y, gap=5)
    c.save()
    print(f"OK {OUT} ({state['page']} halaman)")


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else build()
