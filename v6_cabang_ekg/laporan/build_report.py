"""Laporan PDF v6 (cabang sinyal EKG mentah 50 Hz).

Dua tahap (pola sama dengan v3–v5):
  1. --extract  (Python dengan numpy/torch/matplotlib): hitung ulang angka lewat ekg.evaluate() dari OOF di cache,
     cocokkan dengan ../cache/v6/ringkasan.json dari notebook, buat grafik → laporan/_build/.
  2. tanpa argumen (Python dengan ReportLab): rakit laporan/laporan_cabang_ekg.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # = v6_cabang_ekg/
REPO = HERE.parent
TMP = HERE / "laporan/_build"
OUT = HERE / "laporan/laporan_cabang_ekg.pdf"
SUMMARY = REPO / "cache/v6/ringkasan.json"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402

TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
COLOR = {"v1": "#8A97A6", "v31": "#2a78d6", "ekg": "#eb6834", "dual": "#1baf7a", "late": "#1baf7a"}
ORDER = ["v1", "v31", "ekg", "dual", "late"]
SHORT = {"v1": "v1 (sebelum revisi)", "v31": "v3.1 (RR saja)", "ekg": "6a: CNN EKG saja",
         "dual": "6b: dua cabang EKG + RR", "late": "Late fusion (eksp.)"}
CLASS_COLOR = {"Normal": "#2a78d6", "AFIB": "#eb6834", "AFL": "#1baf7a"}


def extract():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter

    sys.path.insert(0, str(HERE))
    import ekg as K

    TMP.mkdir(parents=True, exist_ok=True)
    res = K.evaluate()
    nb = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for k in res["rows"]:
        for m in ("Macro-F1", "AFL F1", "Normal R"):
            assert abs(nb["rows"][k][m][0] - res["rows"][k][m][0]) < 1e-9, (k, m)
    res["spectra"] = nb["spectra"]
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

    # 1. spektrum per kelas
    sp = res["spectra"]
    f = np.array(sp["f"])
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.axvspan(4, 6.5, color="#EEF1F4", zorder=0)
    ax.text(5.25, .97, "4–6,5 Hz", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.3, color=GRAY)
    for c in ("Normal", "AFIB", "AFL"):
        ax.semilogy(f, sp[c], color=CLASS_COLOR[c], lw=2, label=c)
    ax.set_xlim(0, 25); ax.set_xlabel("Frekuensi (Hz)"); ax.set_ylabel("Daya (median, relatif)")
    ax.xaxis.set_major_formatter(comma); ax.grid(color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout(); save(fig, "spectra")

    # 2. forest
    R, E = res["rows"], res["episodes"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3), sharey=True)
    for ax, (m, src, title) in zip(axes, (("Macro-F1", R, "Macro-F1"), ("AFL F1", R, "AFL F1"),
                                          ("deteksi episode", E, "Deteksi episode AFL"))):
        for i, k in enumerate(ORDER):
            pt, lo, hi = src[k][m][:3]
            ax.plot([lo, hi], [i, i], color=COLOR[k], lw=2, solid_capstyle="round")
            ax.plot(pt, i, "D" if k == "late" else "o", color=COLOR[k], mfc="white" if k == "late" else COLOR[k],
                    mec=COLOR[k], ms=6.5, mew=1.4)
        ax.axvline(src["v31"][m][0], color="#8A97A6", lw=1, ls="--")
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
        ax.grid(axis="x", color="#E6EBF0", lw=.8); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); ax.xaxis.set_major_formatter(comma)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_yticks(range(len(ORDER)), [SHORT[k] for k in ORDER]); axes[0].invert_yaxis()
    fig.tight_layout(); save(fig, "forest")

    # 3. recall AFL per pasien
    mat = np.array([res["afl_recall"][k] for k in ORDER]).T
    cmap = LinearSegmentedColormap.from_list("blue", ["#EEF4FC", "#86b6ef", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    heat_lab = {"v1": "v1", "v31": "v3.1\n(RR)", "ekg": "6a\nEKG saja", "dual": "6b\ndua cabang", "late": "Late\nfusion"}
    ax.set_xticks(range(len(ORDER)), [heat_lab[k] for k in ORDER], fontsize=8.6); ax.xaxis.tick_top()
    ax.set_yticks(range(len(res["afl_patients"])), res["afl_patients"]); ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=9.2,
                    color="white" if mat[i, j] > .55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.03); cb.outline.set_visible(False)
    cb.set_label("Recall AFL", fontsize=8.5); cb.ax.yaxis.set_major_formatter(comma)
    fig.tight_layout(); save(fig, "afl_heatmap")
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
    c.setTitle("Laporan v6 - Cabang sinyal EKG mentah 50 Hz")
    c.setAuthor("Analisis v6_cabang_ekg/cabang_ekg.ipynb")
    W_, H = A4
    M = 40
    CW = W_ - 2 * M
    state = {"page": 0}

    def fnum(v, nd=3):
        return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")

    def sgn(v, nd=3):
        return ("+" if v >= 0 else "") + fnum(v, nd)

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
        c.drawString(M + 43, H - 42, "SIHEDAF  /  V6 CABANG SINYAL EKG MENTAH")
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", 21); c.drawString(M, H - 79, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 99, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, "13 Oktober 2026 | Sumber: v6_cabang_ekg/cabang_ekg.ipynb, cache/v6/, cache/oof_robust/")
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

    R, E, P, L, B, fp = d["rows"], d["episodes"], d["pairs"], d["labels"], d["bias"], d["fp"]
    pe, pd_, pl = P["ekg"], P["dual"], P["late"]

    def verdict(p):
        return "lebih baik" if p >= .95 else "lebih buruk" if p <= .05 else "tidak terbedakan"

    def ci(v, nd=3):
        return f"{fnum(v[0], nd)} <font color='{GRAY}' size='7.4'>[{fnum(v[1], 2)}–{fnum(v[2], 2)}]</font>"

    def dci(v, nd=3):
        return f"{sgn(v[0], nd)}<br/><font color='{GRAY}' size='7.2'>[{fnum(v[1], 2)}; {fnum(v[2], 2)}]</font>"

    sp = d["spectra"]
    import numpy as np
    f = np.array(sp["f"])
    band, ref = (f >= 4) & (f <= 6.5), (f >= 1) & (f <= 20)
    share = {k: float(np.array(sp[k])[band].sum() / np.array(sp[k])[ref].sum()) for k in ("Normal", "AFIB", "AFL")}

    # ── Hal 1 ──
    y = page("Laporan v6: cabang sinyal EKG mentah", "Bentuk gelombang EKG 50 Hz ditambahkan ke interval RR, dibandingkan berpasangan dengan v3.1")
    cardw = (CW - 24) / 4
    cards = ((sgn(pe["Macro-F1"][0]), "Δ Macro-F1, 6a EKG saja"), (sgn(pd_["Macro-F1"][0]), "Δ Macro-F1, 6b dua cabang"),
             (sgn(pl["Macro-F1"][0]), "Δ Macro-F1, late fusion"), ("v3.1", "Paket produksi tetap"))
    for i, (val, lab) in enumerate(cards):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 88, cardw, 88, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 20); c.drawString(x + 11, y - 37, val)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 7.9); c.drawString(x + 11, y - 61, lab)
    y -= 105
    y = para("Window uji detektor 50 Hz, 5 ulangan × 5 fold pasien (sama dengan referensi), bias cross-fit grid −7…+3, bobot kelas "
             "lunak. Bukan uji pada perangkat asli.", y, "small")
    y = callout("Jawaban singkat",
                f"<b>Bentuk gelombang EKG belum membantu.</b> CNN EKG saja jauh lebih buruk dari v3.1 (Macro-F1 {fnum(R['ekg']['Macro-F1'][0])} "
                f"vs {fnum(R['v31']['Macro-F1'][0])}), dan model dua cabang cenderung lebih buruk (P = {fnum(pd_['Macro-F1'][3], 2)}). Masalahnya "
                "generalisasi antar pasien: dengan 23 pasien, CNN menghafal bentuk gelombang tiap pasien. Late fusion (eksploratif) setara "
                f"tapi menaikkan recall Normal secara konsisten. Pasien 07910 dan 08215 tetap tidak terdeteksi. Paket produksi tetap v3.1.", y)
    y = section("Hipotesis (ditetapkan sebelum melihat hasil)", y)
    y = table(["Varian", "Hipotesis", "Hasil", "Status"], [
        ["6a: CNN EKG saja", "Morfologi saja memuat info AFL",
         f"Macro-F1 {sgn(pe['Macro-F1'][0])} (lebih buruk); AFL F1 {fnum(R['ekg']['AFL F1'][0])}. Ada info AFL, tapi tidak cukup sendirian",
         "<b>Sebagian</b>"],
        ["6b: dua cabang EKG + RR", "Lebih baik dari v3.1, terutama pasien AFL yang sulit",
         f"Macro-F1 {sgn(pd_['Macro-F1'][0])} (P = {fnum(pd_['Macro-F1'][3], 2)}); 07910 & 08215 tetap 0", "<b>Ditolak</b>"],
        ["Late fusion (eksploratif)", "—", f"Macro-F1 {sgn(pl['Macro-F1'][0])} (tak terbedakan); recall Normal {sgn(pl['Normal R'][0])} (konsisten)",
         "Perlu konfirmasi"],
    ], y, [110, 120, 215, CW - 445])

    # ── Hal 2 ──
    y = page("Apakah informasi atrium terlihat di 50 Hz?", "Spektrum EKG per kelas dan kesejajaran potongan dengan interval RR")
    y = chart("spectra", y, 190)
    y = table(["Kelas", "Porsi daya 4–6,5 Hz (dari 1–20 Hz)"],
              [[k, fnum(share[k])] for k in ("Normal", "AFIB", "AFL")], y, [150, CW - 150])
    callout("Cara membaca",
            "Gelombang flutter (240–340 per menit = 4–5,7 Hz) dan fibrilasi (4–9 Hz) berada di bawah batas 25 Hz, jadi secara prinsip masih "
            f"terekam di 50 Hz. Kelas AFL dan AFIB memang punya porsi daya 4–6,5 Hz lebih besar ({fnum(share['AFL'])} dan {fnum(share['AFIB'])}) "
            f"daripada Normal ({fnum(share['Normal'])}). Tetapi puncak AFL di ±2,5, 5, dan 7,5 Hz adalah kelipatan laju jantung ±150 bpm "
            "(konduksi 2:1), jadi sebagian isi pita itu adalah irama ventrikel, bukan gelombang flutter murni. Informasi atrium ada, "
            "tapi lemah dan bercampur dengan QRS & gelombang T. "
            "Setiap potongan EKG dicek sejajar dengan interval RR window yang sama: dari 165.978 potongan, tidak ada yang jumlah "
            "R-peak-nya berbeda.", y)

    # ── Hal 3 ──
    y = page("Hasil utama", "Rata-rata 5 ulangan, CI 95% bootstrap pasien; metrik episode memakai kode v4")
    order = ["v1", "v31", "ekg", "dual", "late"]
    y = table(["Varian", "Macro-F1", "AFL F1", "AFL P", "AFL R", "Normal R", "AFIB F1", "Alarm palsu/jam"],
              [[L[k], ci(R[k]["Macro-F1"]), ci(R[k]["AFL F1"]), fnum(R[k]["AFL P"][0]), fnum(R[k]["AFL R"][0]),
                fnum(R[k]["Normal R"][0]), fnum(R[k]["AFIB F1"][0]), ci(E[k]["alarm/jam"], 2)] for k in order],
              y, [118, 74, 74, 36, 36, 40, 40, CW - 418], bold_rows=[2])
    y = chart("forest", y, 170)
    y = table(["Varian", "Δ Macro-F1", "Δ AFL F1", "Δ AFL P", "Δ AFL R", "Δ Normal R", "Δ deteksi episode", "Vonis"],
              [[L[k], dci(P[k]["Macro-F1"]), dci(P[k]["AFL F1"]), dci(P[k]["AFL P"]), dci(P[k]["AFL R"]), dci(P[k]["Normal R"]),
                dci(P[k]["deteksi episode"], 2), verdict(P[k]["Macro-F1"][3]).replace("tidak terbedakan", "tidak<br/>terbedakan")]
               for k in ("ekg", "dual", "late")],
              y, [100, 56, 56, 56, 56, 56, 60, CW - 440])

    # ── Hal 4 ──
    y = page("Per pasien: kenapa EKG tidak menolong", "Recall AFL rata-rata 5 ulangan; stabilitas bias kelas antar fold")
    y = chart("afl_heatmap", y, 215)
    y = table(["Varian", "AFL palsu (window/ulangan)", "di pasien tanpa AFL", "Bias AFIB (median)", "Bias AFL (rentang antar fold)"],
              [[L[k], f"{fp[k]['total']:.0f}", f"{fp[k]['pasien_lain']:.0f}", fnum(B[k]["median"][1], 2),
                f"{fnum(B[k]['min'][2], 2)} … {fnum(B[k]['max'][2], 2)}"] for k in ("v31", "ekg", "dual", "late")],
              y, [140, 95, 75, 70, CW - 380])
    callout("Yang terlihat",
            "CNN EKG menaikkan 06426 (0,17 → 0,44) tetapi menjatuhkan 08378 (0,96 → 0,46), sedangkan 07910 dan 08215 tetap 0. Bias yang "
            "dibutuhkan model berbasis EKG melompat dari fold ke fold (sampai −7…+3), tanda prediksinya tidak stabil terhadap pasien baru: "
            "bentuk gelombang sangat berbeda antar pasien dan antar lead, sementara hanya ada 23 pasien untuk belajar. Pada model dua cabang, "
            "cabang EKG yang besar ikut merusak pemisahan Normal–AFIB yang sebelumnya dikuasai cabang RR (recall Normal "
            f"{fnum(R['v31']['Normal R'][0])} → {fnum(R['dual']['Normal R'][0])}).", y)

    # ── Hal 5 ──
    y = page("Kesimpulan dan langkah berikutnya", "Keputusan v6 dan arah proyek")
    y = table(["Keputusan", "Alasan"], [
        ["<b>Paket produksi tetap v3.1</b>; v6 tidak dikemas", "Tidak ada varian yang lebih baik secara konsisten"],
        ["Model dua cabang tidak dipakai", f"Cenderung lebih buruk (Δ {sgn(pd_['Macro-F1'][0])}); cabang EKG overfit per pasien"],
        ["Late fusion dicatat sebagai kandidat", "Setara, recall Normal naik konsisten; perlu dikonfirmasi pada data lain sebelum dipakai"],
    ], y, [230, CW - 230])
    y = section("Apa artinya untuk proyek", y)
    y = para("v3–v6 sudah mencoba arsitektur, kalibrasi, pasca-proses, resep training, dan sekarang input bentuk gelombang. Semua berhenti "
             "di sekitar Macro-F1 0,83, dan dua dari tujuh pasien AFL tidak pernah terdeteksi oleh pendekatan apa pun. Batasnya adalah "
             "<b>jumlah pasien</b>, bukan model: hanya 7 pasien AFL dan 23 pasien total, terlalu sedikit untuk mempelajari pola yang "
             "berlaku untuk pasien baru, apalagi dari bentuk gelombang.", y)
    y = section("Langkah berikutnya (v7): data dengan lebih banyak pasien", y)
    for txt in ["<b>Basis data EKG publik dengan label AFL</b>, misalnya basis data 12-lead Chapman–Shaoxing di PhysioNet "
                "(rekaman 10 detik, puluhan ribu pasien; menurut deskripsinya termasuk ratusan rekaman AFL, perlu dicek ulang sebelum "
                "mengunduh). Lead tunggal di-downsample ke 50 Hz agar sesuai "
                "perangkat. Bisa dipakai untuk melatih, dan sebagai <b>uji eksternal</b> model sekarang.",
                "<b>Data perangkat asli yang dilabeli dokter</b>, sekecil apa pun, untuk validasi akhir.",
                "Mengunduh basis data eksternal butuh persetujuan pemilik repo; ukuran dan lisensi dicek dulu di PhysioNet."]:
        y = para(f"• {txt}", y, gap=5)
    y = section("Batasan", y)
    for txt in ["Satu arsitektur CNN dicoba tanpa penyetelan (30 epoch, 390 ribu parameter, augmentasi polaritas). CNN yang lebih kecil atau "
                "regularisasi lebih kuat mungkin sedikit lebih baik, tetapi tidak menjawab kekurangan pasien.",
                "Reproduksi: <font name='Courier' size='8.3'>python tools/run_notebook.py v6_cabang_ekg/cabang_ekg.ipynb</font> "
                "(±2 jam GPU, bisa dilanjutkan), lalu <font name='Courier' size='8.3'>laporan/build_report.py --extract</font> dan tanpa argumen."]:
        y = para(f"• {txt}", y, gap=5)
    c.save()
    print(f"OK {OUT} ({state['page']} halaman)")


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else build()
