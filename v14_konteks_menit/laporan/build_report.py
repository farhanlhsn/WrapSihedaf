"""Laporan PDF v14: konteks beberapa menit di atas keluaran per window.

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v14/ringkasan.json (sel akhir konteks_menit.ipynb), hitung ulang Macro-F1 v3.1 dan v14 dari OOF di
     cache dan cocokkan (assert), tambah pembanding v1, gambar grafik → laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_v14_konteks_menit.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v14_konteks_menit.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V), str(REPO / "v11_model_tim"), str(REPO / "v3.3_takikardia_asli")]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, LABEL_PALSU, Laporan, ci, daftar, fnum, pct, sg, sumbu_koma, thou, vonis  # noqa: E402

TANGGAL = "6 November 2026"
PASIEN_AFL = ["04043", "04908", "04936", "06426", "07910", "08215", "08378"]


def extract():
    import numpy as np
    import konteks as C
    from bersama import data as D, evaluasi as Ev
    R = json.loads((REPO / "cache/v14/ringkasan.json").read_text(encoding="utf-8"))
    d = D.muat()
    p31, _ = Ev.keputusan(d.oof_v31, d.folds, d.y)
    px, _ = Ev.keputusan(C.oof_v14("xgb"), d.folds, d.y)
    pl, _ = Ev.keputusan(C.oof_v14("lr"), d.folds, d.y)
    pv1 = d.oof_v1.argmax(-1)
    tab = Ev.tabel_metrik(d, {"v1": pv1, "v31": p31, "xgb": px, "lr": pl})
    for k in ("v31", "xgb", "lr"):
        for m in ("Macro-F1", "AFL F1", "AFL R"):
            assert abs(tab[k][m][0] - R["metrik"][k][m][0]) < 1e-9, (k, m)
    out = {"R": R, "v1": tab["v1"], "banding_v1": Ev.banding(d, px, pv1)}
    TMP.mkdir(exist_ok=True)
    (TMP / "v14.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    grafik(out)
    print("extract selesai; assert Macro-F1, AFL F1, AFL R (v3.1, xgb, lr) cocok dengan ringkasan.json")


def grafik(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from report_fonts import font_path
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font_path("regular"))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path("regular")).get_name(),
                         "font.size": 10, "axes.edgecolor": "#B8C4D0", "axes.labelcolor": INK, "xtick.color": GRAY,
                         "ytick.color": GRAY})
    R = out["R"]
    # H3: alarm per pita detak di EPHNOGRAM
    pita = R["h3_pita"]
    lab = list(pita)
    fig, ax = plt.subplots(figsize=(7.6, 3.0), dpi=200)
    x = range(len(lab))
    ax.plot(x, [pita[b]["v31"] * 100 for b in lab], "-o", color=BIRU, lw=2, ms=7, label="v3.1 paket")
    ax.plot(x, [pita[b]["v14"] * 100 for b in lab], "-s", color=ORANYE, lw=2, ms=7, label="v14 konteks (XGBoost)")
    ax.set_xticks(list(x), [f"{b}\n(n={thou(pita[b]['n'])})" for b in lab])
    ax.set_ylabel("% window ditebak AFIB/AFL"); ax.set_ylim(0, 105); ax.grid(axis="y", color="#E6ECF1")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, loc="upper left"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "h3.png"); plt.close(fig)
    # episode: Se episode AFL dan episode palsu per jam
    ep = R["episode"]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6), dpi=200)
    for ax, (judul, kelas, metrik) in zip(axs, (("Se episode AFL", "AFL", "se_episode"),
                                                ("AFL palsu per jam", "AFL", "palsu_per_jam"),
                                                ("AFIB palsu per jam", "AFIB", "palsu_per_jam"))):
        val = [ep[f"{n} {kelas}"][metrik] for n in ("v31", "xgb", "lr")]
        ax.bar(range(3), val, color=[BIRU, ORANYE, HIJAU], width=.6)
        for i, v in enumerate(val):
            ax.text(i, v, fnum(v, 2), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(3), ["v3.1", "v14 XGB", "v14 LR"], fontsize=8); ax.set_title(judul, fontsize=9, color=INK)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "episode.png"); plt.close(fig)


def build():
    o = json.loads((TMP / "v14.json").read_text(encoding="utf-8"))
    R = o["R"]
    M = R["metrik"]
    L = Laporan(OUT, "v14: konteks beberapa menit", "SIHEDAF  /  V14  /  KONTEKS MENIT",
                "v14_konteks_menit/konteks_menit.ipynb, cache/v14/", TANGGAL)
    bx = R["banding_v31"]["xgb"]["Macro-F1"]
    ep, h3 = R["episode"], R["h3"]
    # 1. ringkasan
    y = L.page("Ringkasan: konteks menit tidak menembus batas", "Hipotesis dikunci sebelum training (RANCANGAN.md); keempatnya tidak lolos")
    y = L.kartu([(fnum(M["xgb"]["Macro-F1"][0]), "Macro-F1 v14 (XGBoost)", f"v3.1 {fnum(M['v31']['Macro-F1'][0])}; P = {fnum(bx[3])}"),
                 (fnum(ep["xgb AFL"]["se_episode"], 2), "Se episode AFL", f"v3.1 {fnum(ep['v31 AFL']['se_episode'], 2)}"),
                 (pct(h3["k14"]["alarm"], 0), "alarm sinus takikardia asli", f"v3.1 {pct(h3['k31']['alarm'], 0)} (EPHNOGRAM)"),
                 (fnum(ep["xgb AFIB"]["palsu_per_jam"], 1), "episode AFIB palsu/jam", f"v3.1 {fnum(ep['v31 AFIB']['palsu_per_jam'], 1)}")], y)
    y = L.para("<b>Jawaban singkat.</b> Tahap kedua yang membaca dinamika detak 2–10 menit terakhir (cara mulai, kedataran, lama "
               "keadaan cepat-teratur) <b>tidak</b> mengurangi alarm palsu pada sinus takikardia dan <b>memperburuk</b> deteksi AFL di "
               f"AFDB: Macro-F1 {fnum(M['xgb']['Macro-F1'][0])} vs {fnum(M['v31']['Macro-F1'][0])} (Δ {sg(bx[0])}, P = {fnum(bx[3])}, "
               f"{vonis(bx[3])}). Satu-satunya perbaikan: episode AFIB palsu turun dari {fnum(ep['v31 AFIB']['palsu_per_jam'], 1)} ke "
               f"{fnum(ep['xgb AFIB']['palsu_per_jam'], 1)} per jam.", y)
    rows = [[k, "lolos" if v else "tidak lolos"] for k, v in R["vonis"].items()]
    y = L.table(["Hipotesis (kriteria dikunci)", "Hasil"], rows, y, widths=[L.CW * .78, L.CW * .22])
    y = L.para("<b>Temuan.</b> (1) Model tidak bisa belajar \"naik bertahap = sinus\" karena AFDB tidak punya episode sinus takikardia "
               "untuk dicontoh: di EPHNOGRAM (24 subjek sehat, latihan fisik) alarm ≥ 120 bpm tetap "
               f"{pct(h3['k14']['alarm'], 1)} [{pct(h3['k14']['ci'][0], 0)}–{pct(h3['k14']['ci'][1], 0)}] vs v3.1 "
               f"{pct(h3['k31']['alarm'], 1)}. (2) Dengan 7 pasien AFL, fitur konteks menghafal pola per pasien: recall AFL 04043 turun "
               f"{fnum(R['afl_per_pasien']['v31']['04043'], 2)} → {fnum(R['afl_per_pasien']['xgb']['04043'], 2)}, 08378 "
               f"{fnum(R['afl_per_pasien']['v31']['08378'], 2)} → {fnum(R['afl_per_pasien']['xgb']['08378'], 2)}. (3) Untuk AF yang "
               "terpecah-pecah, konteks memang membantu (lebih sedikit episode palsu), sejalan dengan literatur.", y)
    # 2. perbandingan
    y = L.page("Perbandingan dengan v3.1 dan v1", "AFDB, pooled OOF 5 ulangan, CI 95% bootstrap pasien, bias cross-fit")
    hdr = ["Sistem", "Macro-F1 [CI]", "AFL F1 [CI]", "AFL R", "AFL P", "Normal R"]
    nm = {"v1": "v1 (argmax)", "v31": "v3.1 (referensi)", "xgb": "v14 XGBoost (utama)", "lr": "v14 regresi logistik"}
    rows = []
    for k in ("v1", "v31", "xgb", "lr"):
        m = o["v1"] if k == "v1" else M[k]
        rows.append([nm[k], f"{fnum(m['Macro-F1'][0])} {ci(*m['Macro-F1'][1:])}", f"{fnum(m['AFL F1'][0])} {ci(*m['AFL F1'][1:])}",
                     fnum(m["AFL R"][0]), fnum(m["AFL P"][0]), fnum(m["Normal R"][0])])
    y = L.table(hdr, rows, y, widths=[L.CW * .24, L.CW * .2, L.CW * .2, L.CW * .12, L.CW * .12, L.CW * .12], sorot=(2,))
    rows = []
    for k in ("xgb", "lr"):
        for mt in ("Macro-F1", "AFL F1", "AFL R", "Normal R"):
            v = R["banding_v31"][k][mt]
            rows.append([nm[k], mt, f"{sg(v[0])} [{sg(v[1])}…{sg(v[2])}]", fnum(v[3]), vonis(v[3])])
    b1 = o["banding_v1"]["Macro-F1"]
    rows.append(["v14 XGBoost vs v1", "Macro-F1", f"{sg(b1[0])} [{sg(b1[1])}…{sg(b1[2])}]", fnum(b1[3]), vonis(b1[3])])
    y = L.table(["Kandidat vs v3.1", "Metrik", "Δ [CI 95%]", "P(Δ > 0)", "Vonis"], rows, y,
                widths=[L.CW * .26, L.CW * .14, L.CW * .3, L.CW * .12, L.CW * .18])
    y = L.section("Metrik episode (gaya EC57)", y)
    y = L.chart(TMP / "episode.png", y, 190)
    y = L.para("Se episode = bagian episode referensi yang tersentuh deteksi; episode palsu = episode deteksi tanpa tumpang tindih "
               "dengan referensi, per jam rekaman. AFDB hanya punya 14 episode AFL, jadi angka AFL sangat bergantung pada beberapa "
               "episode.", y, "small")
    # 3. metode
    y = L.page("Metode", "Fitur kausal (hanya masa lalu), fold resmi, hipotesis dikunci sebelum training")
    for t in ("<b>Tahap pertama.</b> OOF v3.1 (MicroTransformer v3) per window 10 s, langkah 5 s. Untuk data eksternal: paket v3.1.",
              "<b>Fitur (16).</b> log-probabilitas v3.1 window sekarang; rata-rata p(AFIB), p(AFL) 2 dan 10 menit; detak sekarang dan "
              "rata-rata 10 menit; simpangan baku dan kemiringan detak 2 menit; lompatan detak terbesar antar window dalam 2 menit; "
              "lama keadaan cepat-teratur (≥ 110 bpm, CV IBI ≤ 0,05) yang sedang berlangsung; bagian window cepat-teratur 10 menit; CV "
              "sekarang dan rata-rata 2 menit.",
              "<b>Model.</b> XGBoost (kedalaman 4, 300 pohon, laju 0,05, bobot kelas lunak) per (ulangan, fold) resmi, lalu bias "
              "cross-fit grid −7…+3. Sekunder: regresi logistik.",
              "<b>Uji sinus takikardia asli (H3).</b> EPHNOGRAM (PhysioNet, ODC-BY): 24 subjek sehat, rekaman kontinu 30 menit saat "
              "istirahat, berjalan, treadmill Bruce, dan bersepeda; diproses lewat detektor perangkat (50 Hz), window langkah 5 s.",
              "<b>Uji stres konteks sintetis (H4).</b> 15 menit: (A) sinus naik bertahap 80 → 120/140/160 bpm dalam 5 menit lalu datar; "
              "(B) sinus 80 bpm lalu AFL 2:1 mendadak 150 bpm (CV 0,03). Jitter Gauss 10 dan 20 ms, 10 urutan per kondisi.",
              "<b>Kenapa boleh dicoba.</b> Informasi lintas window (dinamika detak) adalah hal baru di luar window 10 s; v4 hanya "
              "merata-ratakan probabilitas. Dicatat sebagai pengecualian di RANCANGAN.md."):
        y = L.para(t, y)
    # 4. hasil sinus takikardia
    y = L.page("Hasil: sinus takikardia asli dan uji stres konteks", "H3 (EPHNOGRAM) dan H4 (sintetis)")
    y = L.chart(TMP / "h3.png", y, 230)
    y = L.para(f"EPHNOGRAM, window ≥ 120 bpm: n = {thou(h3['n'])} dari {h3['subjek']} subjek. Alarm v3.1 {pct(h3['k31']['alarm'], 1)} "
               f"[{pct(h3['k31']['ci'][0], 0)}–{pct(h3['k31']['ci'][1], 0)}], v14 {pct(h3['k14']['alarm'], 1)} "
               f"[{pct(h3['k14']['ci'][0], 0)}–{pct(h3['k14']['ci'][1], 0)}] (CI bootstrap subjek).", y, "small")
    def nama_h4(k):
        jenis, target, sd = k.split("_")
        return (f"sinus naik bertahap ke {target} bpm" if jenis == "sinus" else f"AFL 2:1 mendadak {target} bpm") + f", jitter {sd[2:]} ms"
    rows = [[nama_h4(k), pct(v["v31"], 0), pct(v["v14"], 0)] for k, v in R["h4"].items()]
    y = L.table(["Urutan sintetis 15 menit", "v3.1: alarm (sinus) / deteksi AFL", "v14"], rows, y,
                widths=[L.CW * .4, L.CW * .35, L.CW * .25])
    y = L.para("Sinus yang naik bertahap tetap dialarmkan hampir 100% oleh v14; hanya kondisi 120 bpm dengan jitter 20 ms yang "
               "turun. Deteksi AFL 2:1 mendadak tetap ±99%. Model belajar \"cepat-teratur = AFL\" lagi, sekarang dengan konteks.", y)
    # 5. per pasien
    y = L.page("Per pasien AFL dan asal AFL palsu", "Recall AFL per pasien (+ bias), rata-rata 5 ulangan")
    rows = [[p, fnum(R["afl_per_pasien"]["v31"][p], 2), fnum(R["afl_per_pasien"]["xgb"][p], 2), fnum(R["afl_per_pasien"]["lr"][p], 2)]
            for p in PASIEN_AFL]
    y = L.table(["Pasien", "v3.1", "v14 XGBoost", "v14 LR"], rows, y)
    rows = [[LABEL_PALSU[k], thou(R["afl_palsu"]["v31"][k]), thou(R["afl_palsu"]["xgb"][k]), thou(R["afl_palsu"]["lr"][k])]
            for k in R["afl_palsu"]["v31"]]
    y = L.table(["AFL palsu (window/ulangan)", "v3.1", "v14 XGBoost", "v14 LR"], rows, y)
    bm = R["bias_median"]
    y = L.para(f"Bias [Normal; AFIB; AFL] median: v3.1 {daftar(bm['v31'])}, v14 XGBoost {daftar(bm['xgb'])}; fraksi bias "
               f"v14 di batas grid {fnum(R['bias_di_batas']['xgb'], 2)}.", y, "small")
    # 6. kesimpulan, batasan, lampiran
    y = L.page("Kesimpulan, batasan, dan reproduksi", "")
    for t in ("<b>Kesimpulan.</b> Konteks menit yang dilatih hanya dengan AFDB tidak menyelesaikan alarm sinus takikardia dan "
              "merusak deteksi AFL. Informasi \"cara detak naik\" baru berguna bila data latih memuat episode sinus takikardia asli "
              "berlabel; EPHNOGRAM sekarang dipakai sebagai uji, jadi tidak boleh sekaligus dipakai melatih.",
              "<b>Yang masih berguna.</b> Konteks menurunkan episode AFIB palsu ±4×. Untuk alarm AF (bukan AFL), logika episode "
              "berbasis konteks layak dibawa ke uji perangkat.",
              "<b>Langkah berikutnya.</b> (1) Bila ada rekaman perangkat berlabel dengan aktivitas, latih ulang tahap kedua dengan pasien "
              "terpisah. (2) Untuk data publik: EPHNOGRAM + Wrist sebagai negatif latih dengan uji pada subjek lain (EPFL), dengan "
              "rancangan baru yang dikunci dulu.",
              "<b>Batasan.</b> AFDB: 23 pasien, 7 AFL, 14 episode AFL. EPHNOGRAM: subjek sehat muda, tanpa label ritme (diasumsikan sinus). "
              "Untuk data eksternal tahap pertama adalah paket v3.1 (dilatih penuh), sedangkan tahap kedua dilatih dengan OOF; distribusi "
              "probabilitasnya sedikit berbeda.",
              "<b>Reproduksi.</b> Dari root repo: <font face='Arial-Bold'>python tools/run_notebook.py v14_konteks_menit/konteks_menit.ipynb</font>, "
              "lalu <font face='Arial-Bold'>python v14_konteks_menit/laporan/build_report.py --extract</font> dan tanpa argumen. "
              "Butuh cache v3 (OOF dan fold), data EPHNOGRAM di data/ephnogram (tools/unduh_physionet.py ephnogram 1.0.0)."):
        y = L.para(t, y)
    lk = R["lingkungan"]
    y = L.para(f"Lingkungan: commit {lk['commit']} (+ perubahan belum di-commit), {lk['gpu']}, Python {lk['pustaka']['python']}, "
               f"numpy {lk['pustaka']['numpy']}, xgboost {lk['pustaka']['xgboost']}, scikit-learn {lk['pustaka']['scikit-learn']}.", y, "small")
    L.selesai()
    print("PDF:", OUT, L.page_no, "halaman")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
