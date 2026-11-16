"""Laporan PDF v16.3: replikasi detektor AF 2 kelas D-1 (ECGFounder, ambang −1) di CinC 2017, MIT-BIH, LTAFDB.

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v16.3/ringkasan.json (sel akhir replikasi.ipynb), hitung ulang Macro-F1 D-1 dan v3.1 tiap dataset dari cache
     per rekaman lalu cocokkan (assert), ambil hasil CPSC 2021 (v16.2) dan data yang sudah dipakai untuk rangkuman lintas dataset, gambar
     grafik → laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_v16_3_replikasi.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v16_3_replikasi.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, Laporan, ci, fnum, pct, sg, sumbu_koma, thou, vonis  # noqa: E402

TANGGAL = "6 November 2026"
DATASET = ("CinC 2017", "MIT-BIH", "LTAFDB")
SISTEM = ("v1 paket", "v3.1 paket", "D-0", "D-1")
NAMA = {"v1 paket": "v1 paket", "v3.1 paket": "v3.1 paket (referensi)", "D-0": "D-0 (v16.1, t = 0)", "D-1": "D-1 (primer, t = −1)"}
GAYA = {"v1 paket": (GRAY, "D"), "v3.1 paket": (BIRU, "o"), "D-0": (HIJAU, "^"), "D-1": (ORANYE, "s")}
METRIK = ("Macro-F1", "Se", "Sp", "PPV")
NAMA_RITME = {"(N": "sinus normal", "(SBR": "sinus bradikardia", "(B": "bigemini ventrikel", "(T": "trigemini ventrikel",
              "(AB": "bigemini atrium", "(SVTA": "takiaritmia supraventrikel", "(VT": "takikardia ventrikel", "(IVR": "idioventrikel",
              "(BII": "blok AV derajat 2", "(NOD": "ritme nodal", "(PREX": "pra-eksitasi (WPW)", "(VFL": "flutter ventrikel",
              "N": "normal (CinC)", "O": "ritme lain (CinC)", "(AFIB": "AFIB", "(AFL": "AFL", "A": "AF (CinC)"}


def extract():
    import replikasi as A
    R = json.loads((REPO / "cache/v16.3/ringkasan.json").read_text(encoding="utf-8"))
    for ds in DATASET:
        Z, _ = A.muat(ds)
        p = A.prediksi(Z)
        b = A.A161.banding2_pasien(Z["y2"], p["D-1"], p["v3.1 paket"], Z["subjek"])
        ref = R["hasil"][ds]["banding"]["D-1"]
        assert abs(b["Macro-F1"][0] - ref["Macro-F1"][0]) < 1e-12 and abs(b["titik_b"]["Macro-F1"][0] - ref["titik_b"]["Macro-F1"][0]) < 1e-12, ds
    r162 = json.loads((REPO / "cache/v16.2/ringkasan.json").read_text(encoding="utf-8"))
    lintas = {"CPSC 2021": r162["label_resmi"]["banding"]["D-1"]}
    lintas.update({ds: R["hasil"][ds]["banding"]["D-1"] for ds in DATASET})
    sudah = {"AFDB": r162["h4"]["afdb_banding"]["D-1"], "PTB-XL": r162["h4"]["eksternal"]["PTB-XL"], "Icentia11k": r162["h4"]["eksternal"]["Icentia11k"]}
    titik_afdb = {k: r162["h4"]["afdb"][k] for k in ("D-1", "v3.1")}
    out = {"R": R, "lintas": lintas, "sudah": sudah, "titik_afdb": titik_afdb,
           "n_cpsc": r162["label_resmi"]["n_uji"], "kriteria_cpsc": r162["label_resmi"]["kriteria"]}
    TMP.mkdir(exist_ok=True)
    (TMP / "v16_3.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    grafik(out)
    print("extract selesai; assert Macro-F1 D-1 dan v3.1 ketiga dataset cocok dengan ringkasan.json")


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


def grafik(o):
    plt = _gaya()
    R = o["R"]
    # 1. Macro-F1 / Se / Sp per dataset, empat sistem
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9), dpi=200, sharey=True)
    for ax, m, judul in zip(axs, ("Macro-F1", "Se", "Sp"), ("Macro-F1 2 kelas", "sensitivitas", "spesifisitas")):
        for i, ds in enumerate(DATASET):
            tt = R["hasil"][ds]["titik"]
            for j, k in enumerate(SISTEM):
                v = tt[k][m]
                yy = i + (1.5 - j) * 0.18
                col, mk = GAYA[k]
                ax.plot(v[1:], [yy, yy], color=col, lw=1.5)
                ax.plot(v[0], yy, mk, color=col, ms=5.5, mec="white", mew=1, label=NAMA[k].split(" (")[0] if (i == 0 and m == "Macro-F1") else None)
        ax.set_title(judul, fontsize=9, color=INK); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#E6ECF1")
        sumbu_koma(ax, "x"); ax.tick_params(axis="x", labelsize=8)
    axs[0].set_yticks(range(len(DATASET)), DATASET, fontsize=8.5); axs[0].invert_yaxis()
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(TMP / "metrik.png"); plt.close(fig)
    # 2. alarm per jenis ritme non-AF (MIT-BIH + LTAFDB + CinC), v3.1 vs D-1
    baris = []
    for ds in DATASET:
        for r, v in R["hasil"][ds]["ritme"].items():
            if v["n"] >= 200 and r not in ("(AFIB", "(AFL", "A"):
                baris.append((f"{ds}: {NAMA_RITME.get(r, r)}", v["v3.1 paket"], v["D-1"]))
    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(baris) + 0.8), dpi=200)
    y = range(len(baris))
    ax.barh([i + .2 for i in y], [b[1] * 100 for b in baris], .38, color=BIRU, label="v3.1 paket")
    ax.barh([i - .2 for i in y], [b[2] * 100 for b in baris], .38, color=ORANYE, hatch="//", edgecolor="white", label="D-1")
    ax.set_yticks(list(y), [b[0] for b in baris], fontsize=8); ax.invert_yaxis(); ax.set_xlabel("window non-AF yang dialarmkan AF (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#E6ECF1")
    sumbu_koma(ax, "x")
    fig.tight_layout(); fig.savefig(TMP / "ritme.png"); plt.close(fig)
    # 3. lintas dataset: Δ Macro-F1 dan Δ Se (D-1 − v3.1) dengan CI
    semua = [(n, v, "baru") for n, v in o["lintas"].items()] + [(n, v, "sudah dipakai") for n, v in o["sudah"].items()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.8), dpi=200, sharey=True)
    for ax, m, judul in ((a1, "Macro-F1", "Δ Macro-F1 (D-1 − v3.1)"), (a2, "Se", "Δ sensitivitas (D-1 − v3.1)")):
        for i, (n, v, jenis) in enumerate(semua):
            col, mk = (ORANYE, "s") if jenis == "baru" else (GRAY, "o")
            ax.plot(v[m][1:3], [i, i], color=col, lw=1.8)
            ax.plot(v[m][0], i, mk, color=col, ms=6, mec="white", mew=1)
        ax.axvline(0, color=INK, lw=.8); ax.set_title(judul, fontsize=9, color=INK)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#E6ECF1"); sumbu_koma(ax, "x"); ax.tick_params(axis="x", labelsize=8)
    a1.set_yticks(range(len(semua)), [f"{n}{'' if j == 'baru' else ' *'}" for n, _, j in semua], fontsize=8.5); a1.invert_yaxis()
    fig.text(.5, .01, "kotak oranye = data yang belum pernah dipakai; bulat abu-abu (*) = data yang dipakai untuk memilih ambang", ha="center",
             fontsize=7.5, color=GRAY)
    fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(TMP / "lintas.png"); plt.close(fig)


def _d(v, nd=3):
    return f"{sg(v[0], nd)} [{sg(v[1], nd)}…{sg(v[2], nd)}]"


def build():
    o = json.loads((TMP / "v16_3.json").read_text(encoding="utf-8"))
    R = o["R"]
    H = R["hasil"]
    kr = R["kriteria"]
    L = Laporan(OUT, "v16.3: replikasi detektor AF 2 kelas", "SIHEDAF  /  V16.3  /  REPLIKASI",
                "v16.3_replikasi/replikasi.ipynb, cache/v16.3/", TANGGAL)
    t = {ds: H[ds]["titik"] for ds in DATASET}
    b = {ds: H[ds]["banding"]["D-1"] for ds in DATASET}
    # ── 1. ringkasan ──
    y = L.page("Ringkasan: bertahan di Holter, kurang sensitif di EKG genggam", "Rancangan dikunci 6 November 2026 sebelum sinyal dibaca")
    y = L.kartu([(f"{R['lolos']} dari 3", "dataset lolos H1 + H2", f"vonis: replikasi {R['vonis_replikasi']}"),
                 (fnum(t["CinC 2017"]["D-1"]["Se"][0]), "sensitivitas CinC 2017", f"v3.1 {fnum(t['CinC 2017']['v3.1 paket']['Se'][0])}; lebih buruk"),
                 (fnum(t["LTAFDB"]["D-1"]["Se"][0]), "sensitivitas LTAFDB", f"v3.1 {fnum(t['LTAFDB']['v3.1 paket']['Se'][0])}; lebih baik"),
                 ("7 dari 7", "dataset Macro-F1 lebih baik", "4 baru + 3 sudah dipakai")], y)
    y = L.para("<b>Pertanyaan.</b> D-1 (ECGFounder zero-shot, AF bila d1 > −1) lolos uji pra-registrasi di CPSC 2021 (v16.2). Apakah itu "
               "bertahan di tiga dataset lain yang belum pernah dipakai: EKG genggam satu lead (CinC 2017), aritmia campuran (MIT-BIH), dan "
               "Holter 24 jam pasien AF (LTAFDB)? Tidak ada yang dilatih atau disetel.", y)
    y = L.para("<b>Jawaban singkat: sebagian.</b> Macro-F1 2 kelas D-1 lebih baik dari paket v3.1 di ketiga dataset, dan sensitivitasnya "
               f"lebih tinggi di MIT-BIH ({fnum(t['MIT-BIH']['D-1']['Se'][0])} vs {fnum(t['MIT-BIH']['v3.1 paket']['Se'][0])}) dan LTAFDB. "
               f"Tetapi di CinC 2017 sensitivitasnya lebih rendah ({fnum(t['CinC 2017']['D-1']['Se'][0])} vs "
               f"{fnum(t['CinC 2017']['v3.1 paket']['Se'][0])}), seperti di Icentia11k: pada EKG genggam/wearable satu lead D-1 melewatkan "
               "lebih banyak AF. Ambang tidak diubah; D-1 tetap kandidat uji perangkat dengan sensitivitas sebagai kriteria utama.", y)
    ok = lambda v: "lolos" if v else "gagal"  # noqa: E731
    rows = [[ds, ok(kr[ds]["H1_macro_f1"]), f"{_d(b[ds]['Macro-F1'])}, P = {fnum(b[ds]['Macro-F1'][3])}", ok(kr[ds]["H2_sensitivitas"]),
             f"{_d(b[ds]['Se'])}, P = {fnum(b[ds]['Se'][3])}"] for ds in DATASET]
    y = L.table(["Dataset", "H1 Macro-F1", "Δ Macro-F1 D-1 − v3.1", "H2 sensitivitas", "Δ sensitivitas D-1 − v3.1"], rows, y,
                widths=[L.CW * .15, L.CW * .13, L.CW * .3, L.CW * .14, L.CW * .28], sorot=(1,))
    rc, rl = H["CinC 2017"]["ritme"], H["LTAFDB"]["ritme"]
    y = L.para("<b>Temuan.</b> (1) Spesifisitas D-1 lebih baik di semua dataset; alarm pada ritme non-AF ≥ 120 bpm di CinC 2017 turun "
               f"{pct(H['CinC 2017']['pita']['v3.1 paket']['≥ 120']['alarm'], 0)} → {pct(H['CinC 2017']['pita']['D-1']['≥ 120']['alarm'], 0)}, "
               f"ritme lain (bukan AF) {pct(rc['O']['v3.1 paket'], 0)} → {pct(rc['O']['D-1'], 0)}. (2) Kelemahan baru: sinus bradikardia di "
               f"LTAFDB dialarmkan {pct(rl['(SBR']['D-1'], 0)} (v3.1 {pct(rl['(SBR']['v3.1 paket'], 0)}). (3) Episode AF palsu turun di MIT-BIH "
               f"dan LTAFDB, tetapi Se episode LTAFDB turun {fnum(H['LTAFDB']['episode']['v3.1 paket']['se_episode'], 2)} → "
               f"{fnum(H['LTAFDB']['episode']['D-1']['se_episode'], 2)} (episode pendek).", y)
    y = L.para("<b>Yang tidak bisa disimpulkan.</b> Tidak ada data perangkat SIHEDAF. CinC 2017 tidak punya ID pasien (bootstrap per rekaman) dan "
               "rekamannya pendek (9–61 s). MIT-BIH hanya punya 7 subjek ber-AF. Kanal MIT-BIH/LTAFDB bukan lead I.", y, "small")
    # ── 2. perbandingan ──
    y = L.page("Perbandingan dengan versi sebelumnya dan v1", "2 kelas; CI 95% bootstrap unit (subjek; CinC: rekaman)")
    y = L.chart(TMP / "metrik.png", y, 200)
    rows = []
    for ds in DATASET:
        for k in SISTEM:
            v = t[ds][k]
            rows.append([ds if k == SISTEM[0] else "", NAMA[k], f"{fnum(v['Macro-F1'][0])} {ci(*v['Macro-F1'][1:])}", fnum(v["Se"][0]),
                         fnum(v["Sp"][0]), fnum(v["PPV"][0])])
    y = L.table(["Dataset", "Sistem", "Macro-F1 [CI]", "Sensitivitas", "Spesifisitas", "PPV"], rows, y,
                widths=[L.CW * .14, L.CW * .26, L.CW * .2, L.CW * .14, L.CW * .14, L.CW * .12], sorot=(4, 8, 12))
    y = L.para("D-0 (ambang 0, v16.1) lebih spesifik tetapi kurang sensitif dari D-1 di ketiga dataset. v1 lebih buruk dari v3.1 di CinC 2017 "
               "dan MIT-BIH.", y, "small")
    # ── 3. metode ──
    y = L.page("Metode dan pra-registrasi", "RANCANGAN.md dikunci sebelum sinyal dibaca; sha256 di logs/v16.3_praregistrasi.log")
    n = {ds: H[ds]["n"] for ds in DATASET}
    for tx in ("<b>Sistem.</b> Sama dengan v16.2: D-1 (skor zero-shot ECGFounder 1-lead, AF bila d1 > −1), paket v3.1 (referensi), D-0 "
               "(ambang 0), paket v1. AFIB + AFL digabung. Tidak ada yang dilatih, ambang tidak disetel.",
               f"<b>Data.</b> CinC 2017 (PhysioNet challenge-2017, training, REFERENCE-v3.csv): {thou(n['CinC 2017']['window'])} window dari "
               f"{thou(n['CinC 2017']['unit'])} rekaman genggam AliveCor satu lead (AF = A; non-AF = N dan O; bising dibuang). MIT-BIH "
               f"Arrhythmia: {thou(n['MIT-BIH']['window'])} window, {n['MIT-BIH']['unit']} subjek ({n['MIT-BIH']['unit_af']} ber-AF), kanal 0, "
               f"ritme pacu dibuang. LTAFDB: {thou(n['LTAFDB']['window'])} window, {n['LTAFDB']['unit']} subjek, kanal 0. Label ritme MIT-BIH/"
               "LTAFDB konvensi MIT (berlaku sampai label berikutnya); window sebelum label pertama dibuang.",
               "<b>Window.</b> Window perangkat: resample 50 Hz, detektor R produksi, 10 s, langkah 5 s; hanya yang lolos cek kualitas. EKG "
               "untuk ECGFounder = 500 sampel 50 Hz yang sama.",
               "<b>Kriteria (dikunci).</b> Per dataset: H1 Macro-F1 2 kelas dan H2 sensitivitas D-1 tidak \"lebih buruk\" dari v3.1 (P(Δ > 0) "
               "> 0,05; bootstrap unit berpasangan 2.000×). Replikasi berhasil bila keduanya lolos di ketiga dataset; sebagian bila di satu "
               "atau dua. Bila sebagian: status ditinjau ulang, ambang tidak diubah.",
               "<b>Unduhan.</b> MIT-BIH dan LTAFDB dicek dengan SHA256SUMS resmi. challenge-2017 tidak punya SHA256SUMS; dicek dengan MD5 dari "
               "ETag mirror S3 (sama dengan training/MD5SUMS resmi). Kedua salinan REFERENCE-v3.csv berisi label yang sama.",
               "<b>Penyimpangan.</b> Tidak ada. Hasil per rekaman dihitung dua proses paralel (LTAFDB maju dan mundur); penulisan per rekaman "
               "atomik, rekaman yang sudah ada dilewati."):
        y = L.para(tx, y)
    # ── 4. alarm per ritme ──
    y = L.page("Hasil: alarm pada ritme non-AF", "Proporsi window non-AF yang dialarmkan AF; jenis ritme dengan ≥ 200 window")
    y = L.chart(TMP / "ritme.png", y, 260)
    rows = [[ds, NAMA[k]] + [pct(H[ds]["pita"][k][bnd]["alarm"]) if H[ds]["pita"][k][bnd]["alarm"] is not None else "–"
                             for bnd in H[ds]["pita"][k]] for ds in DATASET for k in ("v3.1 paket", "D-1")]
    pita = list(H["CinC 2017"]["pita"]["v3.1 paket"])
    y = L.table(["Dataset", "Sistem"] + [f"{p} bpm" for p in pita], rows, y,
                widths=[L.CW * .15, L.CW * .25] + [L.CW * .15] * 4, sorot=(2, 4, 6))
    rm = H["MIT-BIH"]["ritme"]
    y = L.para(f"Ritme jarang di MIT-BIH (< 200 window, tidak di grafik): SVTA {pct(rm['(SVTA']['v3.1 paket'], 0)} → {pct(rm['(SVTA']['D-1'], 0)} "
               f"({rm['(SVTA']['n']} window), takikardia ventrikel {pct(rm['(VT']['v3.1 paket'], 0)} → {pct(rm['(VT']['D-1'], 0)} ({rm['(VT']['n']}), "
               f"idioventrikel {pct(rm['(IVR']['v3.1 paket'], 0)} → {pct(rm['(IVR']['D-1'], 0)} ({rm['(IVR']['n']}). Sinus bradikardia LTAFDB "
               f"({thou(rl['(SBR']['n'])} window) dialarmkan {pct(rl['(SBR']['D-1'])} oleh D-1 vs {pct(rl['(SBR']['v3.1 paket'])} oleh v3.1: "
               "kelemahan baru yang perlu dicek di data perangkat.", y, "small")
    # ── 5. episode dan per unit ──
    y = L.page("Episode AF dan per unit", "EC57 pada window yang lolos cek (MIT-BIH, LTAFDB)")
    rows = []
    for ds in ("MIT-BIH", "LTAFDB"):
        for k in SISTEM:
            e = H[ds]["episode"][k]
            rows.append([ds if k == SISTEM[0] else "", NAMA[k], fnum(e["se_episode"], 2), fnum(e["ppv_episode"], 2), fnum(e["se_durasi"], 2),
                         fnum(e["ppv_durasi"], 2), fnum(e["palsu_per_jam"], 1)])
    y = L.table(["Dataset", "Sistem", "Se episode", "+P episode", "Se durasi", "+P durasi", "Palsu/jam"], rows, y,
                widths=[L.CW * .13, L.CW * .27, L.CW * .12, L.CW * .12, L.CW * .12, L.CW * .12, L.CW * .12], sorot=(4, 8))
    y = L.para(f"Di LTAFDB, D-1 melewatkan lebih banyak episode (Se episode {fnum(H['LTAFDB']['episode']['D-1']['se_episode'], 2)} vs "
               f"{fnum(H['LTAFDB']['episode']['v3.1 paket']['se_episode'], 2)}) tetapi menangkap lebih banyak durasi AF (Se durasi "
               f"{fnum(H['LTAFDB']['episode']['D-1']['se_durasi'], 2)} vs {fnum(H['LTAFDB']['episode']['v3.1 paket']['se_durasi'], 2)}): "
               "yang terlewat adalah episode pendek.", y)
    y = L.para(f"<b>Per unit.</b> Sensitivitas D-1 turun lebih dari 0,20 dari v3.1 di {len(H['LTAFDB']['se_turun_0_20'])} dari "
               f"{n['LTAFDB']['unit_af']} subjek LTAFDB ber-AF ({', '.join(H['LTAFDB']['se_turun_0_20'])}), {len(H['MIT-BIH']['se_turun_0_20'])} "
               f"dari {n['MIT-BIH']['unit_af']} subjek MIT-BIH, dan {len(H['CinC 2017']['se_turun_0_20'])} dari {n['CinC 2017']['unit_af']} "
               "rekaman AF CinC 2017 (rekaman pendek, 1–11 window, sehingga satu window sudah 0,1–1,0).", y)
    # ── 6. lintas dataset ──
    y = L.page("Rangkuman lintas dataset", "D-1 − paket v3.1, 2 kelas; tujuh dataset (empat baru, tiga sudah dipakai)")
    y = L.chart(TMP / "lintas.png", y, 210)
    rows = []
    for nm, v in list(o["lintas"].items()) + list(o["sudah"].items()):
        jenis = "baru" if nm in o["lintas"] else "sudah dipakai"
        rows.append([nm, jenis, f"{_d(v['Macro-F1'])}, P = {fnum(v['Macro-F1'][3])}", f"{_d(v['Se'])}, P = {fnum(v['Se'][3])}",
                     f"{sg(v['Sp'][0])}, P = {fnum(v['Sp'][3])}"])
    y = L.table(["Dataset", "Status", "Δ Macro-F1 [CI], P", "Δ sensitivitas [CI], P", "Δ spesifisitas, P"], rows, y,
                widths=[L.CW * .14, L.CW * .13, L.CW * .27, L.CW * .27, L.CW * .19], sorot=(1,))
    y = L.para("<b>Membaca.</b> Pada data Holter/klinis (CPSC 2021, MIT-BIH, LTAFDB, AFDB, PTB-XL) sensitivitas D-1 setara atau lebih baik. "
               "Pada EKG genggam/wearable satu lead (CinC 2017, Icentia11k) sensitivitasnya lebih rendah 3–5 poin. Spesifisitas dan Macro-F1 lebih "
               "baik di ketujuh dataset. Perangkat SIHEDAF wearable, jadi kelompok kedua lebih mirip kondisi pemakaian.", y)
    # ── 7. kesimpulan ──
    y = L.page("Kesimpulan, langkah berikutnya, batasan, reproduksi", "")
    for tx in ("<b>Kesimpulan.</b> Keunggulan D-1 sebagai detektor AF 2 kelas, yaitu alarm palsu jauh lebih sedikit dan Macro-F1 lebih baik, "
               "bertahan di tiga dataset baru. Kelemahannya juga konsisten: pada EKG genggam/wearable satu lead D-1 melewatkan lebih banyak AF "
               "daripada v3.1. Vonis yang dikunci: replikasi sebagian. Ambang tidak diubah.",
               "<b>Status.</b> D-1 tetap kandidat detektor AF 2 kelas untuk uji perangkat, tetapi uji itu harus mengunci sensitivitas sebagai "
               "kriteria utama (bukan hanya Macro-F1) dan memeriksa sinus bradikardia. Paket v3.1 tetap paket produksi.",
               "<b>Langkah berikutnya.</b> (1) Rekaman perangkat SIHEDAF berlabel: uji D-1 vs v3.1 dengan kriteria sensitivitas yang dikunci. "
               "(2) Bila sensitivitas di perangkat kurang: keputusan ambang (mis. lebih rendah dari −1) hanya boleh diambil dari data perangkat, "
               "dengan rancangan baru; ketujuh dataset publik ini sudah terpakai. (3) Jalur gabungan (D-1 untuk alarm, v3.1 sebagai cadangan "
               "sensitivitas) perlu dirancang dan diuji di data perangkat, bukan di data publik yang sudah dilihat.",
               "<b>Batasan.</b> CinC 2017 tanpa ID pasien dan rekaman pendek; MIT-BIH 7 subjek ber-AF; kanal MIT-BIH/LTAFDB bukan lead I; AFL "
               "hampir tidak ada di ketiga dataset. Bootstrap CinC per rekaman bisa mengecilkan CI bila satu pasien punya banyak rekaman.",
               "<b>Reproduksi.</b> <font face='Arial-Bold'>python tools/run_notebook.py v16.3_replikasi/replikasi.ipynb</font>, lalu "
               "<font face='Arial-Bold'>build_report.py --extract</font> dan tanpa argumen. Unduhan lewat tools/unduh_physionet.py (lihat "
               "README v16.3); butuh bobot ECGFounder dan cache v16.2 untuk rangkuman lintas dataset."):
        y = L.para(tx, y)
    lk = R["lingkungan"]
    y = L.para(f"Lingkungan: commit {lk['commit']} (+ perubahan belum di-commit), Python {lk['pustaka']['python']}, numpy "
               f"{lk['pustaka']['numpy']}, scikit-learn {lk['pustaka']['scikit-learn']}, torch {lk['pustaka']['torch']}, GPU {lk['gpu']}.",
               y, "small")
    L.selesai()
    print("PDF:", OUT, L.page_no, "halaman")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
