"""Laporan PDF v15: kuantisasi RR dan konsistensi pra-QRS (pra-registrasi).

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v15/ringkasan.json (sel akhir kuantisasi_rr.ipynb), hitung ulang AUROC PTB-XL dan Macro-F1 AFDB dari
     cache lalu cocokkan (assert), tambah pembanding v1, gambar grafik → laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_v15_kuantisasi_rr.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v15_kuantisasi_rr.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V), str(REPO / "v11_model_tim")]
from laporan_sihedaf import BIRU, ORANYE, INK, GRAY, Laporan, ci, daftar, fnum, pct, sg, sumbu_koma, thou, vonis  # noqa: E402

TANGGAL = "6 November 2026"
PASIEN_AFL = ["04043", "04908", "04936", "06426", "07910", "08215", "08378"]
NAMA_F = {"kuantisasi": "kuantisasi RR", "p_konsisten": "konsistensi pra-QRS"}


def extract():
    import numpy as np
    import kuantisasi as Q
    from bersama import data as D, evaluasi as Ev
    R = json.loads((REPO / "cache/v15/ringkasan.json").read_text(encoding="utf-8"))
    P = dict(np.load(REPO / "cache/v15/ptbxl.npz"))
    for f in Q.FITUR_UJI:
        a = Q.auroc_boot(P["fitur"][:, Q.A12.FITUR.index(f)], P["y"], P["pasien"])
        assert abs(a["auroc"] - R["h1_ptbxl"][f]["auroc"]) < 1e-12, f
    d = D.muat()
    p31, _ = Ev.keputusan(d.oof_v31, d.folds, d.y)
    pg, _ = Ev.keputusan(Q.h2_afdb(), d.folds, d.y)
    pv1 = d.oof_v1.argmax(-1)
    tab = Ev.tabel_metrik(d, {"v1": pv1, "v31": p31, "gabungan": pg})
    for k in ("v31", "gabungan"):
        assert abs(tab[k]["Macro-F1"][0] - R["h2_afdb"]["metrik"][k]["Macro-F1"][0]) < 1e-9, k
    out = {"R": R, "v1": tab["v1"], "banding_v1": Ev.banding(d, pg, pv1)}
    TMP.mkdir(exist_ok=True)
    (TMP / "v15.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    grafik(out)
    print("extract selesai; assert AUROC PTB-XL (2 fitur) dan Macro-F1 AFDB (v3.1, gabungan) cocok dengan ringkasan.json")


def grafik(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from report_fonts import font_path
    font_manager.fontManager.addfont(font_path("regular"))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path("regular")).get_name(), "font.size": 10,
                         "axes.edgecolor": "#B8C4D0", "xtick.color": GRAY, "ytick.color": GRAY})
    R = out["R"]
    fig, ax = plt.subplots(figsize=(7.6, 2.4), dpi=200)
    baris = []
    for f in ("p_konsisten", "kuantisasi"):
        baris += [(f"{NAMA_F[f]}: semua AFL", R["h1_ptbxl"][f], BIRU, "o"), (f"{NAMA_F[f]}: AFL sulit", R["h1b_ptbxl"][f], ORANYE, "s")]
    for i, (lab, v, col, mk) in enumerate(baris[::-1]):
        ax.plot(v["ci"], [i, i], color=col, lw=2.2)
        ax.plot(v["auroc"], i, mk, color=col, ms=8, mec="white", mew=1.5)
        ax.text(v["ci"][1] + .01, i, fnum(v["auroc"]), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(baris)), [b[0] for b in baris[::-1]], fontsize=8.5)
    ax.axvline(.5, color=GRAY, lw=1, ls=":"); ax.axvline(.7, color="#1baf7a", lw=1.2, ls="--")
    ax.set_xlim(.4, 1.02); ax.set_xlabel("AUROC AFL vs AFIB (PTB-XL), CI 95% bootstrap pasien; garis putus = ambang 0,70")
    ax.spines[["top", "right"]].set_visible(False); sumbu_koma(ax, "x")
    fig.tight_layout(); fig.savefig(TMP / "auroc.png"); plt.close(fig)
    afl = R["h2_afdb"]["afl_per_pasien"]
    fig, ax = plt.subplots(figsize=(7.6, 2.5), dpi=200)
    x = range(len(PASIEN_AFL))
    ax.bar([i - .2 for i in x], [afl["v31"][p] for p in PASIEN_AFL], .38, color=BIRU, label="v3.1")
    ax.bar([i + .2 for i in x], [afl["gabungan"][p] for p in PASIEN_AFL], .38, color=ORANYE, label="gabungan (v3.1 + 2 fitur)", hatch="//", edgecolor="white")
    ax.set_xticks(list(x), PASIEN_AFL); ax.set_ylabel("recall AFL"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center"); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "afl_pasien.png"); plt.close(fig)


def build():
    o = json.loads((TMP / "v15.json").read_text(encoding="utf-8"))
    R = o["R"]
    S = R["syarat_cpsc"]
    H2 = R["h2_afdb"]
    L = Laporan(OUT, "v15: kuantisasi RR (pra-registrasi)", "SIHEDAF  /  V15  /  KUANTISASI RR",
                "v15_kuantisasi_rr/kuantisasi_rr.ipynb, cache/v15/", TANGGAL)
    bg = H2["banding_v31"]["Macro-F1"]
    hp = R["h2_ptbxl"]
    y = L.page("Ringkasan: uji utama tidak bisa dijalankan", "Hipotesis dikunci 6 November 2026 sebelum CPSC 2021 dibuka")
    y = L.kartu([("tidak cukup", "data CPSC 2021", f"AFL hanya {S['subjek']['(AFL']} subjek, {S['window']['(AFL']} window"),
                 (fnum(R["h1_ptbxl"]["p_konsisten"]["auroc"]), "AUROC pra-QRS (PTB-XL)", "AFL vs AFIB; analisis tambahan"),
                 (f"{fnum(hp['titik_b'][1], 2)} → {fnum(hp['titik_a'][1], 2)}", "recall AFL PTB-XL", f"P = {fnum(hp['AFL R'][3])}"),
                 (fnum(H2["metrik"]["gabungan"]["Macro-F1"][0]), "Macro-F1 AFDB gabungan", f"v3.1 {fnum(H2['metrik']['v31']['Macro-F1'][0])}; P = {fnum(bg[3])}")], y)
    y = L.para("<b>Jawaban singkat.</b> Hasil utama: <b>tidak cukup data</b>. CPSC 2021 ternyata hanya memuat AFL dari "
               f"{S['subjek']['(AFL']} subjek ({fnum(S['menit']['(AFL'], 1)} menit, {S['window']['(AFL']} window 10 s murni), jauh di "
               "bawah syarat yang dikunci (≥ 5 subjek, ≥ 200 window). Analisis tambahan di PTB-XL (dicatat sebagai penyimpangan) "
               "mendukung hipotesis bahwa kedua fitur memisahkan AFL dari AFIB, juga untuk AFL yang tidak cepat-teratur. Tetapi "
               f"menggabungkannya secara sederhana dengan v3.1 <b>memperburuk</b> AFDB (Macro-F1 Δ {sg(bg[0])}, P = {fnum(bg[3])}).", y)
    y = L.para("<b>Temuan.</b> (1) Di PTB-XL, konsistensi pra-QRS AUROC "
               f"{fnum(R['h1_ptbxl']['p_konsisten']['auroc'])} {ci(*R['h1_ptbxl']['p_konsisten']['ci'])}, kuantisasi RR "
               f"{fnum(R['h1_ptbxl']['kuantisasi']['auroc'])} {ci(*R['h1_ptbxl']['kuantisasi']['ci'])}; keduanya lolos ambang 0,70. "
               f"(2) Tahap gabungan menaikkan recall AFL PTB-XL {fnum(hp['titik_b'][1], 2)} → {fnum(hp['titik_a'][1], 2)} tanpa "
               f"menurunkan Macro-F1 secara bermakna (Δ {sg(hp['Macro-F1'][0])}, P = {fnum(hp['Macro-F1'][3])}). (3) Di AFDB, AFL "
               f"\"sulit\" membaik (06426 {fnum(H2['afl_per_pasien']['v31']['06426'], 2)} → {fnum(H2['afl_per_pasien']['gabungan']['06426'], 2)}), "
               f"tetapi 04936 jatuh {fnum(H2['afl_per_pasien']['v31']['04936'], 2)} → {fnum(H2['afl_per_pasien']['gabungan']['04936'], 2)}: "
               "informasinya ada, cara menggabungkannya belum.", y)
    y = L.page("Perbandingan dengan v3.1 dan v1", "AFDB: pooled OOF 5 ulangan, CI bootstrap pasien; PTB-XL: bootstrap pasien")
    rows = []
    nm = {"v1": "v1 (argmax)", "v31": "v3.1 (referensi)", "gabungan": "v15 gabungan (v3.1 + 2 fitur)"}
    for k in ("v1", "v31", "gabungan"):
        m = o["v1"] if k == "v1" else H2["metrik"][k]
        rows.append([nm[k], f"{fnum(m['Macro-F1'][0])} {ci(*m['Macro-F1'][1:])}", f"{fnum(m['AFL F1'][0])} {ci(*m['AFL F1'][1:])}",
                     fnum(m["AFL R"][0]), fnum(m["AFL P"][0]), fnum(m["Normal R"][0])])
    y = L.table(["AFDB", "Macro-F1 [CI]", "AFL F1 [CI]", "AFL R", "AFL P", "Normal R"], rows, y,
                widths=[L.CW * .28, L.CW * .2, L.CW * .2, L.CW * .1, L.CW * .1, L.CW * .12], sorot=(3,))
    rows = [[mt, f"{sg(v[0])} [{sg(v[1])}…{sg(v[2])}]", fnum(v[3]), vonis(v[3])] for mt, v in H2["banding_v31"].items()]
    b1 = o["banding_v1"]["Macro-F1"]
    rows.append(["Macro-F1 vs v1", f"{sg(b1[0])} [{sg(b1[1])}…{sg(b1[2])}]", fnum(b1[3]), vonis(b1[3])])
    y = L.table(["Gabungan vs v3.1 (AFDB)", "Δ [CI 95%]", "P(Δ > 0)", "Vonis"], rows, y,
                widths=[L.CW * .3, L.CW * .34, L.CW * .14, L.CW * .22])
    rows = [["Macro-F1", fnum(hp["titik_b"][0]), fnum(hp["titik_a"][0]), f"{sg(hp['Macro-F1'][0])} [{sg(hp['Macro-F1'][1])}…{sg(hp['Macro-F1'][2])}]", fnum(hp["Macro-F1"][3])],
            ["recall AFL", fnum(hp["titik_b"][1], 2), fnum(hp["titik_a"][1], 2), f"{sg(hp['AFL R'][0])} [{sg(hp['AFL R'][1])}…{sg(hp['AFL R'][2])}]", fnum(hp["AFL R"][3])]]
    y = L.table(["PTB-XL (tambahan)", "v3.1 paket", "gabungan", "Δ [CI 95%]", "P(Δ > 0)"], rows, y)
    st = R["stres"]
    rows = [[j, str(st["v31"][j]), str(st["gabungan"][j])] for j in st["v31"]]
    y = L.table(["Uji stres takikardia (protokol butir 9)", "v3.1: ambang bpm", "gabungan"], rows, y,
                widths=[L.CW * .5, L.CW * .25, L.CW * .25])
    y = L.para("Uji stres sintetis hanya punya IBI, jadi kedua fitur EKG diberi nilai median data latih (sesuai rancangan); ambang "
               "gabungan pada dasarnya mengikuti v3.1.", y, "small")
    y = L.page("Metode dan pra-registrasi", "RANCANGAN.md dikunci sebelum data uji dibuka; hash di logs/v15_praregistrasi.log")
    for t in ("<b>Fitur (tetap, dari v12).</b> <i>kuantisasi</i>: bagian interval RR yang rasionya terhadap siklus flutter (1/frekuensi "
              "atrium dominan dari sinyal sisa setelah pembatalan QRST) dalam ±0,15 dari bilangan bulat. <i>p_konsisten</i>: rata-rata "
              "korelasi antar-detak segmen −0,30…−0,10 s sebelum R (gelombang P/F yang terkunci ke QRS).",
              "<b>Hipotesis dikunci.</b> H1: AUROC AFL vs AFIB ≥ 0,70 dan batas bawah CI > 0,50 untuk kedua fitur di data baru. H2: "
              "regresi logistik (log-probabilitas v3.1 + 2 fitur) dilatih di AFDB + SHDB-AF, diuji tanpa penyetelan. Syarat data: ≥ 5 "
              "subjek dan ≥ 200 window AFL.",
              f"<b>Data uji utama.</b> CPSC 2021 ({thou(S['rekaman'])} rekaman yang dirilis; 11 rekaman di RECORDS tidak ada di SHA256SUMS "
              f"resmi). Ritme: AFIB {S['subjek']['(AFIB']} subjek ({thou(S['menit']['(AFIB'])} menit), Normal {S['subjek']['(N']} subjek, "
              f"AFL {S['subjek']['(AFL']} subjek. Syarat tidak terpenuhi.",
              "<b>Penyimpangan tercatat (RANCANGAN bagian 8).</b> H1/H2 dijalankan di PTB-XL 1.0.3 (lead I 100 Hz → 50 Hz, rekaman 10 s "
              f"saat istirahat) sebagai analisis tambahan: {thou(R['n_ptbxl']['rekaman'])} rekaman, kelas [sinus, AFIB, AFL] = "
              f"[{'; '.join(thou(x) for x in R['n_ptbxl']['kelas'])}]. AFL \"sulit\" = window AFL yang bukan cepat-teratur (pola RR v12)."):
        y = L.para(t, y)
    y = L.page("Hasil H1: pemisahan AFL vs AFIB (PTB-XL, tambahan)", "AUROC dengan CI bootstrap pasien berstrata")
    y = L.chart(TMP / "auroc.png", y, 200)
    rows = [[NAMA_F[f], f"{fnum(R['h1_ptbxl'][f]['auroc'])} {ci(*R['h1_ptbxl'][f]['ci'])}", f"{fnum(R['h1b_ptbxl'][f]['auroc'])} {ci(*R['h1b_ptbxl'][f]['ci'])}",
             "lolos" if R["h1_lulus_ptbxl"][f] else "tidak lolos"] for f in ("p_konsisten", "kuantisasi")]
    y = L.table(["Fitur", "Semua AFL", "AFL sulit", "Ambang H1"], rows, y)
    y = L.para(f"AFL: {R['h1_ptbxl']['kuantisasi']['n_afl']} rekaman dari {R['h1_ptbxl']['kuantisasi']['pasien_afl']} pasien; AFIB "
               f"{thou(R['h1_ptbxl']['kuantisasi']['n_afib'])}. Peringatan literatur (Climent dkk. 2008): RR saat AF juga bisa "
               "mengelompok di kelipatan siklus atrium, jadi AUROC kuantisasi lebih rendah dari konsistensi pra-QRS masuk akal.", y, "small")
    y = L.page("Per pasien AFL di AFDB", "Recall AFL (+ bias cross-fit), rata-rata 5 ulangan")
    y = L.chart(TMP / "afl_pasien.png", y, 200)
    rows = [[p, fnum(H2["afl_per_pasien"]["v31"][p], 2), fnum(H2["afl_per_pasien"]["gabungan"][p], 2)] for p in PASIEN_AFL]
    y = L.table(["Pasien", "v3.1", "gabungan"], rows, y)
    y = L.para(f"Bias median gabungan [Normal; AFIB; AFL] {daftar(H2['bias_median'])}, fraksi di batas grid {fnum(H2['bias_di_batas'], 2)}.", y, "small")
    y = L.page("Kesimpulan, batasan, dan reproduksi", "")
    for t in ("<b>Kesimpulan.</b> Uji yang direncanakan tidak bisa dijalankan karena data publik dengan AFL kontinu hampir tidak "
              "ada. Analisis tambahan menguatkan bahwa informasi atrium dari EKG 50 Hz (terutama konsistensi pra-QRS) membedakan AFL "
              "dari AFIB, termasuk AFL yang lolos dari model RR. Tetapi tahap gabungan yang sederhana tidak lebih baik di AFDB.",
              "<b>Langkah berikutnya.</b> Pakai fitur ini hanya untuk window yang sudah diputuskan AF (AFIB vs AFL), bukan untuk ketiga "
              "kelas sekaligus, dan uji dengan rekaman perangkat berlabel. Rancangan baru harus dikunci lagi sebelum melihat data.",
              "<b>Batasan.</b> PTB-XL: 10 s saat istirahat, sebagian label dari pembacaan otomatis yang dicek manusia; bukan Holter. "
              "AFDB: 7 pasien AFL dan 58% window AFL dari 04936, yang labelnya pernah dipertanyakan reviewer.",
              "<b>Reproduksi.</b> <font face='Arial-Bold'>python tools/run_notebook.py v15_kuantisasi_rr/kuantisasi_rr.ipynb</font>, lalu "
              "<font face='Arial-Bold'>build_report.py --extract</font> dan tanpa argumen. Data: CPSC 2021 dan PTB-XL lewat "
              "tools/unduh_physionet.py; SHDB-AF dan cache v12 dari v12."):
        y = L.para(t, y)
    lk = R["lingkungan"]
    y = L.para(f"Lingkungan: commit {lk['commit']} (+ perubahan belum di-commit), Python {lk['pustaka']['python']}, numpy "
               f"{lk['pustaka']['numpy']}, scikit-learn {lk['pustaka']['scikit-learn']}.", y, "small")
    L.selesai()
    print("PDF:", OUT, L.page_no, "halaman")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
