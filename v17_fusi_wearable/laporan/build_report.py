"""Laporan PDF v17: fusi ECGFounder + RR v3.1 yang dilatih dengan EKG genggam (CinC 2017), diuji di data wearable (Icentia11k).

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v17/ringkasan.json (sel akhir fusi.ipynb), latih ulang W-F dari cache CinC dan cocokkan koefisien serta
     Macro-F1 Icentia (assert), ambil angka v1 dari v16.2/v16.3, gambar grafik → laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_v17_fusi_wearable.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v17_fusi_wearable.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, Laporan, ci, fnum, pct, sg, sumbu_koma, thou, vonis  # noqa: E402,F401

TANGGAL = "6 November 2026"
SISTEM = ("W-F", "D-1", "v3.1 paket", "D-1 ∨ v3.1")
NAMA = {"W-F": "W-F (fusi, v17)", "D-1": "D-1 (v16.2)", "v3.1 paket": "v3.1 paket (produksi)", "D-1 ∨ v3.1": "D-1 ∨ v3.1 (naif)",
        "v1 paket": "v1 paket"}
GAYA = {"W-F": (ORANYE, "s"), "D-1": (HIJAU, "^"), "v3.1 paket": (BIRU, "o"), "D-1 ∨ v3.1": (GRAY, "D")}


def extract():
    import numpy as np
    import fusi as F
    R = json.loads((REPO / "cache/v17/ringkasan.json").read_text(encoding="utf-8"))
    cinc = F.data_cinc()
    m = F.latih(F.masukan(cinc["d1"], cinc["p31"]), cinc["y2"])
    k = F.koefisien(m)
    assert all(abs(k[x] - R["koefisien"][x]) < 1e-9 for x in ("w_d1", "w_r", "b"))
    ico = F.data_icentia()
    p = F.prediksi(m, ico["d1"], ico["p31"])
    b = F.A161.banding2_pasien(ico["y2"], p["W-F"], p["D-1"], ico["unit"])
    assert abs(b["titik_a"]["Macro-F1"][0] - R["icentia"]["titik"]["W-F"]["Macro-F1"][0]) < 1e-12
    # titik untuk grafik batas keputusan: sinus takikardia asli ≥ 120 bpm, dan sampel Icentia
    L = F.H16.latihan()
    _, d1l = F.H16.tahap1(L["logit"])
    m120 = L["hr"] >= 120
    rng = np.random.default_rng(2026)
    ia = rng.choice(np.flatnonzero(ico["y2"] == 1), 1500, replace=False)
    inn = rng.choice(np.flatnonzero(ico["y2"] == 0), 1500, replace=False)
    titik = {"sinus takikardia asli ≥ 120 bpm": [d1l[m120].tolist(), F.log_odds_af(L["p31"][m120]).tolist()],
             "Icentia AF (sampel)": [ico["d1"][ia].tolist(), F.log_odds_af(ico["p31"][ia]).tolist()],
             "Icentia non-AF (sampel)": [ico["d1"][inn].tolist(), F.log_odds_af(ico["p31"][inn]).tolist()]}
    r162 = json.loads((REPO / "cache/v16.2/ringkasan.json").read_text(encoding="utf-8"))
    r163 = json.loads((REPO / "cache/v16.3/ringkasan.json").read_text(encoding="utf-8"))
    v1 = {"CPSC 2021": r162["label_resmi"]["titik"]["v1 paket"], "MIT-BIH": r163["hasil"]["MIT-BIH"]["titik"]["v1 paket"],
          "LTAFDB": r163["hasil"]["LTAFDB"]["titik"]["v1 paket"], "CinC 2017": r163["hasil"]["CinC 2017"]["titik"]["v1 paket"]}
    out = {"R": R, "titik_batas": titik, "v1": v1}
    TMP.mkdir(exist_ok=True)
    (TMP / "v17.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    grafik(out)
    print("extract selesai; assert koefisien W-F dan Macro-F1 Icentia cocok dengan ringkasan.json")


def _gaya():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from report_fonts import font_path
    font_manager.fontManager.addfont(font_path("regular"))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path("regular")).get_name(), "font.size": 10,
                         "axes.edgecolor": "#B8C4D0", "xtick.color": GRAY, "ytick.color": GRAY})
    return plt


def _dataset(R):
    return [("Icentia11k", R["icentia"]["titik"]), ("CPSC 2021", R["holter"]["CPSC 2021"]["titik"]),
            ("MIT-BIH", R["holter"]["MIT-BIH"]["titik"]), ("LTAFDB", R["holter"]["LTAFDB"]["titik"]),
            ("CinC 2017 (OOF)", R["cinc"]["titik"]), ("PTB-XL", R["ptbxl"]["titik"]), ("AFDB", R["afdb"]["titik"])]


def grafik(o):
    import numpy as np
    plt = _gaya()
    R = o["R"]
    ds = _dataset(R)
    # 1. sensitivitas dan spesifisitas per dataset
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0), dpi=200, sharey=True)
    for ax, m, judul in zip(axs, ("Se", "Sp"), ("sensitivitas AF", "spesifisitas")):
        for i, (nm, tt) in enumerate(ds):
            for j, k in enumerate(("W-F", "D-1", "v3.1 paket")):
                v = tt[k][m]
                yy = i + (j - 1) * 0.22
                col, mk = GAYA[k]
                if len(v) == 3:
                    ax.plot(v[1:], [yy, yy], color=col, lw=1.4)
                ax.plot(v[0], yy, mk, color=col, ms=5.5, mec="white", mew=1, label=NAMA[k] if (i == 0 and m == "Se") else None)
        ax.set_title(judul, fontsize=9, color=INK); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#E6ECF1")
        sumbu_koma(ax, "x"); ax.tick_params(axis="x", labelsize=8)
    axs[0].set_yticks(range(len(ds)), [d[0] for d in ds], fontsize=8.5); axs[0].invert_yaxis()
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(TMP / "se_sp.png"); plt.close(fig)
    # 2. alarm sinus takikardia asli per pita detak
    lat = R["latihan"]
    pita = list(lat["W-F"]["pita"])
    fig, ax = plt.subplots(figsize=(7.6, 2.5), dpi=200)
    for k, ls in (("v3.1 paket", "-"), ("W-F", "-"), ("D-1", "--")):
        col, mk = GAYA[k]
        ax.plot(range(len(pita)), [lat[k]["pita"][p]["alarm"] * 100 for p in pita], ls, marker=mk, color=col, ms=6, mec="white", mew=1,
                lw=2, label=NAMA[k])
    ax.set_xticks(range(len(pita)), [p.replace("–∞", "+") for p in pita], fontsize=8)
    ax.set_ylabel("alarm AF (%)"); ax.set_ylim(0, 105); ax.legend(frameon=False, fontsize=8, ncol=1, loc="center right")
    ax.set_xlabel("detak window (EKG latihan fisik asli, mode perangkat)"); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "pita.png"); plt.close(fig)
    # 3. batas keputusan pada bidang (d1, r)
    k = R["koefisien"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=200)
    for (nm, (x, y)), col, mk in zip(o["titik_batas"].items(), (ORANYE, BIRU, GRAY), ("s", "o", ".")):
        ax.scatter(np.clip(x, -12, 12), np.clip(y, -12, 12), s=6 if mk != "." else 4, color=col, marker=mk, alpha=.45, lw=0, label=nm)
    xs = np.linspace(-12, 12, 50)
    ax.plot(xs, -(k["w_d1"] * xs + k["b"]) / k["w_r"], color=INK, lw=1.8, label="batas W-F (AF di kanan atas)")
    ax.axvline(-1, color=HIJAU, lw=1.8, ls="--", label="batas D-1 (d1 = −1)")
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12); ax.set_xlabel("d1 = log-odds AF ECGFounder"); ax.set_ylabel("r = log-odds AF paket v3.1")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", markerscale=2.5); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E6ECF1"); sumbu_koma(ax); sumbu_koma(ax, "x")
    fig.tight_layout(); fig.savefig(TMP / "batas.png"); plt.close(fig)


def _d(v, nd=3):
    return f"{sg(v[0], nd)} [{sg(v[1], nd)}…{sg(v[2], nd)}]"


def build():
    o = json.loads((TMP / "v17.json").read_text(encoding="utf-8"))
    R = o["R"]
    I, Ib, kr, lat, k = R["icentia"]["titik"], R["icentia"]["banding"], R["kriteria"], R["latihan"], R["koefisien"]
    L = Laporan(OUT, "v17: fusi ECGFounder + RR untuk data wearable", "SIHEDAF  /  V17  /  FUSI WEARABLE",
                "v17_fusi_wearable/fusi.ipynb, cache/v17/", TANGGAL)
    # ── 1. ringkasan ──
    y = L.page("Ringkasan: sensitivitas wearable naik, alarm saat aktivitas kembali", "Rancangan dikunci 6 November 2026 sebelum model dilatih")
    y = L.kartu([(fnum(I["W-F"]["Se"][0]), "sensitivitas Icentia (W-F)", f"D-1 {fnum(I['D-1']['Se'][0])}; v3.1 {fnum(I['v3.1 paket']['Se'][0])}"),
                 (fnum(I["W-F"]["Macro-F1"][0]), "Macro-F1 Icentia (W-F)", f"D-1 {fnum(I['D-1']['Macro-F1'][0])}; lebih baik"),
                 (pct(lat["W-F"]["alarm"]), "alarm sinus takikardia", f"≥ 120 bpm; D-1 {pct(lat['D-1']['alarm'])}"),
                 (f"{sum(v for kk, v in kr.items() if kk.startswith('K'))} dari 5", "kriteria lolos", "W-F tidak menggantikan D-1")], y)
    y = L.para("<b>Pertanyaan.</b> D-1 (ECGFounder zero-shot, AF bila d1 > −1) kurang sensitif pada EKG genggam/wearable satu lead, sedangkan "
               "model RR v3.1 lebih sensitif di sana tetapi mengalarmkan sinus takikardia. Apakah regresi logistik kecil yang menggabungkan "
               "keduanya, dilatih dengan EKG genggam CinC 2017 (lisensi CC BY 3.0), menaikkan sensitivitas wearable tanpa kehilangan "
               "spesifisitas D-1?", y)
    y = L.para("<b>Jawaban singkat: tidak tanpa harga.</b> Di Icentia11k (2.420 pasien wearable, tidak dipakai untuk melatih) W-F lebih "
               f"sensitif ({fnum(I['W-F']['Se'][0])} vs D-1 {fnum(I['D-1']['Se'][0])}) dan Macro-F1-nya lebih baik dari D-1 dan v3.1. Tetapi "
               f"W-F mengalarmkan {pct(lat['W-F']['alarm'])} sinus takikardia asli (batas yang dikunci 10%) dan lebih buruk dari D-1 di CPSC 2021 "
               "dan MIT-BIH. Kriteria K4 dan K5 gagal, jadi <b>D-1 tetap kandidat</b>; paket v3.1 tetap produksi.", y)
    ok = lambda v: "lolos" if v else "gagal"  # noqa: E731
    H = R["holter"]
    rows = [["K1 sensitivitas Icentia lebih baik dari D-1", ok(kr["K1_se_vs_D1"]), f"{_d(Ib['W-F vs D-1']['Se'])}, P = {fnum(Ib['W-F vs D-1']['Se'][3])}"],
            ["K2 Macro-F1 Icentia bukan lebih buruk dari D-1", ok(kr["K2_mf1_vs_D1"]), f"{_d(Ib['W-F vs D-1']['Macro-F1'])}, P = {fnum(Ib['W-F vs D-1']['Macro-F1'][3])}"],
            ["K3 Macro-F1 dan Se Icentia bukan lebih buruk dari v3.1", ok(kr["K3_vs_v31"]),
             f"{sg(Ib['W-F vs v3.1']['Macro-F1'][0])}; {sg(Ib['W-F vs v3.1']['Se'][0])}"],
            ["K4 alarm sinus takikardia asli ≤ 10%", ok(kr["K4_sinus_takikardia"]), f"{pct(lat['W-F']['alarm'])} {ci(lat['W-F']['ci'][0] * 100, lat['W-F']['ci'][1] * 100, 0)}"],
            ["K5 Holter bukan lebih buruk dari D-1", ok(kr["K5_holter"]),
             "; ".join(f"{nm} {sg(H[nm]['banding']['W-F vs D-1']['Macro-F1'][0])}" for nm in ("CPSC 2021", "MIT-BIH", "LTAFDB"))]]
    y = L.table(["Kriteria yang dikunci (W-F)", "Hasil", "Angka"], rows, y, widths=[L.CW * .43, L.CW * .1, L.CW * .47])
    y = L.para(f"<b>Temuan.</b> (1) W-F yang dipelajari: AF bila {fnum(k['w_d1'])}·d1 + {fnum(k['w_r'])}·r + {fnum(k['b'])} ≥ 0, kira-kira "
               f"d1 > {fnum(-k['b'] / k['w_d1'], 1)} − {fnum(k['w_r'] / k['w_d1'], 2)}·r: ambang ECGFounder diturunkan dan model RR ikut bersuara. "
               "(2) Suara RR itulah yang mengembalikan alarm pada detak cepat teratur. (3) Celah sensitivitas wearable bisa ditutup, tetapi titik "
               "kerjanya harus dipilih dari data perangkat yang memuat aktivitas fisik.", y)
    y = L.para("<b>Yang tidak bisa disimpulkan.</b> Tidak ada data perangkat. EKG Icentia dari v10 sudah difilter 0,5–20 Hz. CinC 2017 dan "
               "Icentia11k sudah pernah dilihat untuk D-1 (bukan untuk W-F).", y, "small")
    # ── 2. perbandingan ──
    y = L.page("Perbandingan dengan versi sebelumnya dan v1", "2 kelas; CI 95% bootstrap unit berpasangan 2.000×")
    y = L.chart(TMP / "se_sp.png", y, 200)
    rows = []
    for nm, tt in _dataset(R):
        for kk in ("W-F", "D-1", "v3.1 paket"):
            v = tt[kk]
            rows.append([nm if kk == "W-F" else "", NAMA[kk], fnum(v["Macro-F1"][0]), fnum(v["Se"][0]), fnum(v["Sp"][0])])
        kv1 = nm.replace(" (OOF)", "")
        if kv1 in o["v1"]:
            v = o["v1"][kv1]
            rows.append(["", NAMA["v1 paket"], fnum(v["Macro-F1"][0]), fnum(v["Se"][0]), fnum(v["Sp"][0])])
    y = L.table(["Dataset", "Sistem", "Macro-F1", "Sensitivitas", "Spesifisitas"], rows, y,
                widths=[L.CW * .2, L.CW * .3, L.CW * .16, L.CW * .17, L.CW * .17])
    # ── 3. metode ──
    y = L.page("Metode dan pra-registrasi", "RANCANGAN.md dikunci sebelum model dilatih; sha256 di logs/v17_praregistrasi.log")
    for tx in ("<b>Model W-F.</b> Regresi logistik (StandardScaler + LogisticRegression C = 1, kelas seimbang; resep v15) dengan dua masukan: "
               "d1 = log-odds AF ECGFounder zero-shot (sama dengan D-1) dan r = log((pAFIB + pAFL) / pN) dari paket v3.1. AF bila peluang ≥ 0,5. "
               "Tiga parameter, dilatih sekali di seluruh window CinC 2017 yang lolos cek (43.465 window, 3.898 AF, 8.172 rekaman).",
               "<b>Lisensi.</b> CinC 2017: CC BY 3.0 (boleh untuk produk dengan atribusi). Icentia11k: CC BY-NC-SA, hanya evaluasi.",
               "<b>Uji.</b> Utama: Icentia11k (115.722 window, 2.420 pasien, wearable). Penjaga: CPSC 2021 (label resmi), MIT-BIH, LTAFDB, dan "
               "sinus takikardia asli (latihan fisik v3.3). Deskriptif: AFDB (OOF v3.1 + bias paket, 5 ulangan), PTB-XL, validasi silang CinC "
               "per rekaman. Semua masukan dari cache; tidak ada inferensi ECGFounder baru dan tidak ada yang disetel di data uji.",
               "<b>Kriteria (dikunci).</b> K1 sensitivitas Icentia W-F lebih baik dari D-1 (P ≥ 0,95); K2 Macro-F1 Icentia bukan lebih buruk "
               "dari D-1; K3 Macro-F1 dan sensitivitas Icentia bukan lebih buruk dari v3.1; K4 alarm sinus takikardia asli ≥ 120 bpm ≤ 10%; K5 "
               "Macro-F1 CPSC 2021, MIT-BIH, LTAFDB bukan lebih buruk dari D-1. Semua lolos → W-F menggantikan D-1.",
               "<b>Penyimpangan.</b> Tidak ada."):
        y = L.para(tx, y)
    # ── 4. kenapa gagal ──
    y = L.page("Hasil: kenapa alarm saat aktivitas kembali", "Batas keputusan pada bidang (d1, r) dan alarm per pita detak")
    y = L.chart(TMP / "batas.png", y, 235)
    y = L.para("Sinus takikardia asli (kotak oranye) punya d1 rendah (ECGFounder: sinus) tetapi r tinggi (v3.1: AF). D-1 hanya melihat d1, "
               "jadi hampir semuanya di kiri garis hijau. Batas W-F miring: r yang tinggi menggeser ambang d1 ke kiri, sehingga sebagian besar "
               "detak cepat teratur ikut dialarmkan. Window AF Icentia yang terlewat D-1 (12.738) punya d1 median −2,7 dan r median 1,8; sinus "
               "takikardia asli punya d1 median −4,8 tetapi r median 6,0. Batas W-F menangkap keduanya sekaligus.", y)
    y = L.chart(TMP / "pita.png", y, 175)
    # ── 5. ritme ──
    y = L.page("Alarm per ritme non-AF (Holter)", "Proporsi window yang dialarmkan AF; MIT-BIH dan LTAFDB")
    rows = []
    for nm in ("MIT-BIH", "LTAFDB"):
        for r, v in H[nm]["ritme"].items():
            if r in ("(AFIB", "(AFL") or v["n"] < 100:
                continue
            rows.append([nm, r.strip("("), thou(v["n"]), pct(v["W-F"]), pct(v["D-1"]), pct(v["v3.1 paket"])])
    y = L.table(["Dataset", "Ritme", "n window", "W-F", "D-1", "v3.1 paket"], rows, y,
                widths=[L.CW * .16, L.CW * .14, L.CW * .16, L.CW * .18, L.CW * .18, L.CW * .18])
    y = L.para("Kode ritme MIT: N sinus normal, SBR sinus bradikardia, B bigemini ventrikel, T trigemini ventrikel, AB bigemini atrium, SVTA "
               "takiaritmia supraventrikel, BII blok AV derajat 2. W-F mengalarmkan lebih banyak ritme non-AF daripada D-1 (bigemini, sinus "
               "normal), kecuali sinus bradikardia LTAFDB yang justru turun (27,9% → 11,0%).", y, "small")
    ptk = R["ptbxl"]["kelas"]
    y = L.para(f"PTB-XL: AFL terdeteksi W-F {pct(ptk['W-F']['AFL'], 0)} (D-1 {pct(ptk['D-1']['AFL'], 0)}, v3.1 {pct(ptk['v3.1 paket']['AFL'], 0)}); "
               f"STACH dialarmkan {pct(ptk['W-F']['STACH'])} (D-1 {pct(ptk['D-1']['STACH'])}, v3.1 {pct(ptk['v3.1 paket']['STACH'])}).", y)
    # ── 6. kesimpulan ──
    y = L.page("Kesimpulan, langkah berikutnya, batasan, reproduksi", "")
    for tx in ("<b>Kesimpulan.</b> Celah sensitivitas D-1 di data wearable bisa ditutup dengan model kecil yang dilatih di EKG genggam, tetapi "
               "caranya (ambang ECGFounder lebih rendah dan suara RR) mengembalikan kelemahan lama model RR: alarm pada sinus takikardia saat "
               "aktivitas, serta spesifisitas Holter yang turun. Kriteria yang dikunci tidak terpenuhi; D-1 tetap kandidat.",
               "<b>Pelajaran.</b> Di CinC 2017 hanya 1,4% window non-AF yang ≥ 120 bpm, jadi model yang dilatih di sana hampir tidak melihat biaya "
               "alarm saat aktivitas. Ini anti-pattern yang sama dengan v1–v6 dan v16 H-X: data latih/penyetelan tanpa kondisi pemakaian penting.",
               "<b>Langkah berikutnya.</b> Data publik tidak bisa memilih titik kerja antara sensitivitas wearable dan alarm saat aktivitas. Itu "
               "harus dipilih dari rekaman perangkat SIHEDAF yang memuat aktivitas fisik (dan, bila ada, sensor gerak). Rancangan uji perangkat "
               "sebaiknya mengunci dua kandidat: D-1 (spesifik) dan W-F (sensitif), dengan sensitivitas dan alarm saat aktivitas sebagai kriteria.",
               "<b>Batasan.</b> EKG Icentia sudah difilter 0,5–20 Hz (v10); CinC tanpa ID pasien; sinus takikardia asli hanya 41 subjek sehat; "
               "AFDB dan PTB-XL sudah dipakai sebelumnya.",
               "<b>Reproduksi.</b> <font face='Arial-Bold'>python tools/run_notebook.py v17_fusi_wearable/fusi.ipynb</font>, lalu "
               "<font face='Arial-Bold'>build_report.py --extract</font> dan tanpa argumen. Butuh cache v10, v13, v16, v16.2, v16.3; tanpa GPU."):
        y = L.para(tx, y)
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
