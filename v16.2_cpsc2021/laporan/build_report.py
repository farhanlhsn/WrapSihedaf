"""Laporan PDF v16.2: uji ambang t = −1 detektor AF 2 kelas (ECGFounder) di CPSC 2021.

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v16.2/ringkasan.json (sel akhir uji_cpsc.ipynb), hitung ulang Macro-F1 D-1 dan v3.1 di CPSC 2021 (kedua
     aturan label) dari cache lalu cocokkan (assert), gambar grafik → laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_v16_2_cpsc2021.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v16_2_cpsc2021.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, Laporan, ci, fnum, pct, sg, sumbu_koma, thou, vonis  # noqa: E402

TANGGAL = "6 November 2026"
SISTEM = ("v1 paket", "v3.1 paket", "D-0", "D-1")
NAMA = {"v1 paket": "v1 paket", "v3.1 paket": "v3.1 paket (referensi)", "D-0": "D-0 (v16.1, t = 0)", "D-1": "D-1 (primer, t = −1)"}
METRIK = ("Macro-F1", "Se", "Sp", "PPV")


def extract():
    import numpy as np
    import uji_cpsc as U
    R = json.loads((REPO / "cache/v16.2/ringkasan.json").read_text(encoding="utf-8"))
    for resmi, kunci, lo_nama, pk_nama in ((False, None, "logit_ecgfounder", "paket"), (True, "label_resmi", "logit_ecgfounder_resmi", "paket_resmi")):
        J = U.bangun_jendela(resmi=resmi, log=lambda *_: None)
        ok = J["ok"]
        _, d1 = U.H16.tahap1(np.load(U.CACHE / f"{lo_nama}.npy"))
        p31 = np.load(U.CACHE / f"{pk_nama}.npz")["p31"]
        y2 = (J["y"][ok] > 0).astype(np.int64)
        b = U.A161.banding2_pasien(y2, d1 > U.AMBANG, p31.argmax(1) != 0, J["subjek"][ok])
        ref = R["banding"]["D-1"] if kunci is None else R[kunci]["banding"]["D-1"]
        assert abs(b["Macro-F1"][0] - ref["Macro-F1"][0]) < 1e-12 and abs(b["titik_b"]["Macro-F1"][0] - ref["titik_b"]["Macro-F1"][0]) < 1e-12
    TMP.mkdir(exist_ok=True)
    (TMP / "v16_2.json").write_text(json.dumps({"R": R}, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    grafik(R)
    print("extract selesai; assert Macro-F1 D-1 dan v3.1 (dua aturan label) cocok dengan ringkasan.json")


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


def grafik(R):
    plt = _gaya()
    gaya = {"v1 paket": (GRAY, "D"), "v3.1 paket": (BIRU, "o"), "D-0": (HIJAU, "^"), "D-1": (ORANYE, "s")}
    # 1. Macro-F1, Se, Sp dengan CI untuk kedua analisis
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6), dpi=200, sharey=True)
    for ax, m, judul in zip(axs, ("Macro-F1", "Se", "Sp"), ("Macro-F1 2 kelas", "sensitivitas", "spesifisitas")):
        for j, (tt, lab) in enumerate(((R["titik"], "dikunci"), (R["label_resmi"]["titik"], "label resmi"))):
            for i, k in enumerate(SISTEM):
                v = tt[k][m]
                yy = i + (0.18 if j == 0 else -0.18)
                col, mk = gaya[k]
                ax.plot(v[1:], [yy, yy], color=col, lw=1.6, alpha=.55 if j == 0 else 1)
                ax.plot(v[0], yy, mk, color=col if j == 1 else "white", mec=col, ms=6, mew=1.4)
        ax.set_title(judul, fontsize=9, color=INK); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#E6ECF1")
        sumbu_koma(ax, "x"); ax.tick_params(axis="x", labelsize=8)
    axs[0].set_yticks(range(len(SISTEM)), [NAMA[k].split(" (")[0] for k in SISTEM], fontsize=8.5)
    fig.text(.5, .01, "CI 95% bootstrap subjek; penanda kosong = analisis dikunci, penanda penuh = label resmi", ha="center", fontsize=8, color=GRAY)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(TMP / "metrik.png"); plt.close(fig)
    # 2. alarm pada window non-AF per pita detak (label resmi)
    pita = R["label_resmi"]["alarm_pita"]
    pp = list(pita["v3.1 paket"])
    fig, ax = plt.subplots(figsize=(7.6, 2.6), dpi=200)
    lebar = .2
    for i, k in enumerate(SISTEM):
        col, _ = gaya[k]
        ys = [(pita[k][b]["alarm"] or 0) * 100 for b in pp]
        ax.bar([x + (i - 1.5) * lebar for x in range(len(pp))], ys, lebar * .92, color=col, label=NAMA[k].split(" (")[0],
               hatch=("//" if k == "D-1" else None), edgecolor="white" if k == "D-1" else None)
    ax.set_xticks(range(len(pp)), [f"{b} bpm\n(n = {thou(pita['v3.1 paket'][b]['n'])})" for b in pp], fontsize=8.5)
    ax.set_ylabel("alarm AF (%)"); ax.set_ylim(0, 105); ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "pita.png"); plt.close(fig)
    # 3. sensitivitas per subjek ber-AF (label resmi; window AF sama di kedua analisis)
    ps = R["label_resmi"]["per_subjek"]
    sub = sorted([s for s in ps["D-1"] if ps["D-1"][s]["se"] is not None], key=lambda s: ps["v3.1 paket"][s]["se"])
    fig, ax = plt.subplots(figsize=(7.6, 2.5), dpi=200)
    for i, s in enumerate(sub):
        ax.plot([i, i], [ps["v3.1 paket"][s]["se"], ps["D-1"][s]["se"]], color="#C9D3DD", lw=1.2, zorder=1)
    ax.scatter(range(len(sub)), [ps["v3.1 paket"][s]["se"] for s in sub], color=BIRU, marker="o", s=16, label="v3.1 paket", zorder=2)
    ax.scatter(range(len(sub)), [ps["D-1"][s]["se"] for s in sub], color=ORANYE, marker="s", s=16, label="D-1", zorder=3)
    ax.set_xticks(range(len(sub)), sub, rotation=90, fontsize=6.5); ax.set_ylabel("sensitivitas AF")
    ax.set_xlabel("subjek ber-AF, diurutkan menurut sensitivitas v3.1", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "se_subjek.png"); plt.close(fig)


def _d(v, nd=3):
    return f"{sg(v[0], nd)} [{sg(v[1], nd)}…{sg(v[2], nd)}]"


def _tabel_titik(L, tt, y, judul):
    rows = [[NAMA[k]] + [f"{fnum(tt[k][m][0])} {ci(*tt[k][m][1:])}" for m in METRIK] for k in SISTEM]
    return L.table([judul, "Macro-F1 [CI]", "Sensitivitas [CI]", "Spesifisitas [CI]", "PPV [CI]"], rows, y,
                   widths=[L.CW * .24, L.CW * .19, L.CW * .19, L.CW * .19, L.CW * .19], sorot=(4,))


def _tabel_banding(L, bd, y, judul):
    rows = []
    for k, lab in (("D-1", "D-1 − v3.1"), ("D-0", "D-0 − v3.1"), ("v1 paket", "v1 − v3.1"), ("D-1 vs D-0", "D-1 − D-0")):
        v = bd[k]
        rows.append([lab, f"{_d(v['Macro-F1'])}, P = {fnum(v['Macro-F1'][3])}", f"{sg(v['Se'][0])}, P = {fnum(v['Se'][3])}",
                     f"{sg(v['Sp'][0])}, P = {fnum(v['Sp'][3])}", vonis(v["Macro-F1"][3])])
    return L.table([judul, "Δ Macro-F1 [CI], P(Δ > 0)", "Δ Se, P", "Δ Sp, P", "Vonis Macro-F1"], rows, y,
                   widths=[L.CW * .18, L.CW * .36, L.CW * .16, L.CW * .16, L.CW * .14], sorot=(1,))


def build():
    R = json.loads((TMP / "v16_2.json").read_text(encoding="utf-8"))["R"]
    Lr = R["label_resmi"]
    t0, t1 = R["titik"], Lr["titik"]
    b0, b1 = R["banding"], Lr["banding"]
    kr0, kr1 = R["kriteria"], Lr["kriteria"]
    lat = R["h4"]["latihan"]
    L = Laporan(OUT, "v16.2: detektor AF 2 kelas di CPSC 2021", "SIHEDAF  /  V16.2  /  UJI CPSC 2021",
                "v16.2_cpsc2021/uji_cpsc.ipynb, cache/v16.2/", TANGGAL)
    # ── 1. ringkasan ──
    y = L.page("Ringkasan: ambang −1 bertahan di data baru", "Rancangan dikunci 6 November 2026 sebelum sinyal CPSC 2021 dibaca")
    y = L.kartu([(fnum(t1["D-1"]["Macro-F1"][0]), "Macro-F1 2 kelas (D-1)", f"v3.1 {fnum(t1['v3.1 paket']['Macro-F1'][0])}; P = {fnum(b1['D-1']['Macro-F1'][3], 2)}"),
                 (fnum(t1["D-1"]["Se"][0]), "sensitivitas AF", f"v3.1 {fnum(t1['v3.1 paket']['Se'][0])}; P = {fnum(b1['D-1']['Se'][3], 2)}"),
                 (fnum(t1["D-1"]["Sp"][0]), "spesifisitas", f"v3.1 {fnum(t1['v3.1 paket']['Sp'][0])}; lebih baik"),
                 (f"{fnum(Lr['episode']['D-1']['palsu_per_jam'], 1)}/jam", "episode AF palsu", f"v3.1 {fnum(Lr['episode']['v3.1 paket']['palsu_per_jam'], 1)}/jam")], y)
    y = L.para("<b>Pertanyaan.</b> v16.1 menemukan, <i>setelah</i> melihat hasil, bahwa detektor AF 2 kelas dari ECGFounder lebih seimbang "
               "pada ambang t = −1. Apakah itu bertahan di data yang belum pernah dinilai? Ambang dikunci, lalu diuji di CPSC 2021 "
               f"({thou(Lr['n_uji']['rekaman'])} rekaman Holter, {Lr['n_uji']['subjek']} subjek, {thou(Lr['n_uji']['window'])} window). Tidak "
               "ada yang dilatih atau disetel.", y)
    y = L.para("<b>Jawaban singkat: ya, untuk keluaran 2 kelas.</b> D-1 (AF bila d1 > −1) sama sensitifnya dengan paket v3.1 "
               f"({fnum(t1['D-1']['Se'][0])} vs {fnum(t1['v3.1 paket']['Se'][0])}) dan jauh lebih spesifik ({fnum(t1['D-1']['Sp'][0])} vs "
               f"{fnum(t1['v3.1 paket']['Sp'][0])}), sehingga Macro-F1 lebih baik ({fnum(t1['D-1']['Macro-F1'][0])} vs "
               f"{fnum(t1['v3.1 paket']['Macro-F1'][0])}). Kriteria yang dikunci lolos di kedua analisis (lihat penyimpangan label), jadi D-1 "
               "adalah <b>kandidat detektor AF 2 kelas untuk uji perangkat</b>. Bukan pengganti paket v3.1: untuk 3 kelas tidak ada yang berubah.", y)
    ok = lambda b: "lolos" if b else "gagal"  # noqa: E731
    rows = [["Macro-F1 2 kelas bukan lebih buruk dari v3.1", ok(kr0["H1a_macro_f1"]), f"P = {fnum(b0['D-1']['Macro-F1'][3])}",
             ok(kr1["H1a_macro_f1"]), f"P = {fnum(b1['D-1']['Macro-F1'][3])}"],
            ["Sensitivitas bukan lebih buruk dari v3.1", ok(kr0["H1b_sensitivitas"]), f"P = {fnum(b0['D-1']['Se'][3])}",
             ok(kr1["H1b_sensitivitas"]), f"P = {fnum(b1['D-1']['Se'][3])}"],
            ["Alarm sinus takikardia asli ≤ separuh v3.1", ok(kr0["sinus_takikardia"]), f"{pct(lat['D-1']['alarm'], 2)} vs {pct(lat['v3.1 paket']['alarm'])}",
             ok(kr1["sinus_takikardia"]), "sama"],
            ["Kandidat uji perangkat (keduanya lolos)", "ya" if kr0["kandidat_uji_perangkat"] else "tidak", "", "", ""]]
    y = L.table(["Kriteria yang dikunci (D-1)", "Dikunci", "Angka", "Label resmi", "Angka"], rows, y,
                widths=[L.CW * .4, L.CW * .11, L.CW * .2, L.CW * .13, L.CW * .16], sorot=(4,))
    y = L.para("<b>Temuan.</b> (1) Ambang −1 lebih baik dari ambang 0 di data baru: D-0 kehilangan sensitivitas "
               f"({fnum(t1['D-0']['Se'][0])}). (2) Alarm pada window non-AF turun {pct(Lr['per_kelas']['v3.1 paket']['Normal'])} → "
               f"{pct(Lr['per_kelas']['D-1']['Normal'])}; pada non-AF ≥ 120 bpm masih {pct(Lr['alarm_pita']['D-1']['≥ 120']['alarm'], 0)} "
               f"(v3.1 {pct(Lr['alarm_pita']['v3.1 paket']['≥ 120']['alarm'], 0)}). (3) Hanya {len(Lr['se_turun_0_20'])} dari "
               f"{sum(v['se'] is not None for v in Lr['per_subjek']['D-1'].values())} subjek ber-AF kehilangan sensitivitas > 0,20. (4) v1 paket "
               "lebih buruk dari v3.1 dengan label resmi.", y)
    y = L.para("<b>Yang tidak bisa disimpulkan.</b> CPSC 2021 adalah rekaman EKG dinamis (Holter) 2-lead 200 Hz, bukan perangkat SIHEDAF; AFL "
               "hanya 144 window dari 2 subjek; ambang −1 dipilih dari data yang lain (v16.1), bukan dari perangkat.", y, "small")
    # ── 2. perbandingan ──
    y = L.page("Perbandingan dengan versi sebelumnya dan v1", "CPSC 2021, 2 kelas; CI dan P bootstrap subjek berpasangan 2.000×")
    y = L.chart(TMP / "metrik.png", y, 185)
    y = _tabel_titik(L, t1, y, f"Label resmi ({Lr['n_uji']['subjek']} subjek)")
    y = _tabel_banding(L, b1, y, "Label resmi")
    y = _tabel_titik(L, t0, y, f"Dikunci ({R['n_uji']['subjek']} subjek)")
    y = _tabel_banding(L, b0, y, "Dikunci")
    # ── 3. metode ──
    y = L.page("Metode, pra-registrasi, dan penyimpangan", "RANCANGAN.md dikunci sebelum sinyal dibaca; sha256 di logs/v16.2_praregistrasi.log")
    d0, dr = R["data"], Lr["data"]
    for t in ("<b>Sistem.</b> D-1: skor zero-shot ECGFounder 1-lead (v13), AF bila d1 = log((pAF + pAFL) / pN) > −1. D-0: ambang 0 "
              "(v16.1). Pembanding: paket produksi v3.1 dan paket v1 pada interval RR window yang sama (AFIB + AFL digabung). Tidak ada "
              "yang dilatih; ambang −1 dikunci dari kurva v16.1.",
              "<b>Data.</b> CPSC 2021 (PhysioNet 1.0.0, 1.425 rekaman yang dirilis, Holter 2-lead 200 Hz). Kanal 0 (lead I, sama dengan "
              "ECGFounder 1-lead). Window perangkat: resample ke 50 Hz, detektor R produksi, 10 s, langkah 5 s; hanya window yang lolos cek "
              "kualitas (punya prediksi v3.1). EKG untuk ECGFounder = 500 sampel 50 Hz yang sama. Bootstrap per subjek.",
              "<b>Kriteria (dikunci).</b> Kandidat bila Macro-F1 2 kelas dan sensitivitas D-1 tidak \"lebih buruk\" dari v3.1 (P(Δ > 0) > "
              "0,05), dan alarm sinus takikardia asli ≤ separuh v3.1.",
              "<b>Penyimpangan: label Normal.</b> Aturan yang dikunci memakai pembaca segmen v15, yang hanya menganggap bagian setelah label "
              "<i>(N</i> sebagai Normal. Kode penilaian resmi CPSC 2021 (score_2021.py) memakai <i>(AFIB</i>/<i>(AFL</i> sebagai <b>awal</b> "
              "episode AF dan <i>(N</i> sebagai <b>akhir</b>; rekaman non-AF tidak punya label ritme. Analisis yang dikunci karena itu "
              f"membuang 721 rekaman non-AF (window Normal {thou(d0['Normal']['lolos'])} dari {d0['Normal']['subjek']} subjek). Analisis "
              f"label resmi: Normal {thou(dr['Normal']['lolos'])} window dari {dr['Normal']['subjek']} subjek. Window AF identik di keduanya "
              "(di-assert). Kandidat hanya bila kriteria lolos di kedua analisis; keduanya dilaporkan berdampingan.",
              "<b>Kendali mutu.</b> Prediksi paket pada window AF sama persis di kedua run (paket diproses per potongan 40.000 window setelah "
              f"prediksi v1 sekaligus kehabisan memori). Keputusan D-1 pada {thou(Lr['cek_af']['n'])} window AF sama "
              f"{pct(Lr['cek_af']['keputusan_d1_sama'])} (selisih d1 maks {fnum(Lr['cek_af']['maks_selisih_d1'], 3)}, batch GPU berbeda).",
              "<b>Protokol.</b> Fold dan bias tidak berlaku (data eksternal, tanpa parameter). Butir 5: per subjek. Butir 7: episode AF gaya "
              "EC57 pada window yang lolos cek. Butir 9: sinus takikardia asli (data v3.3) dan alarm non-AF per pita detak. H4 (AFDB, PTB-XL, "
              "Icentia11k) dilaporkan sebagai data yang sudah dipakai untuk memilih ambang."):
        y = L.para(t, y)
    # ── 4. alarm palsu ──
    y = L.page("Hasil: alarm palsu dan episode", "Label resmi CPSC 2021; window non-AF = di luar semua episode AF")
    y = L.chart(TMP / "pita.png", y, 190)
    pk = Lr["per_kelas"]
    rows = [[NAMA[k], pct(pk[k]["Normal"]), pct(pk[k]["AFIB"]), pct(pk[k]["AFL"]), pct(R["per_kelas"][k]["Normal"])] for k in SISTEM]
    y = L.table(["Proporsi window teralarm AF", "Non-AF (resmi)", "AFIB", "AFL (2 subjek)", "Normal (dikunci)"], rows, y,
                widths=[L.CW * .32, L.CW * .17, L.CW * .17, L.CW * .17, L.CW * .17], sorot=(4,))
    ep, ep0 = Lr["episode"], R["episode"]
    rows = [[NAMA[k], fnum(ep[k]["se_episode"], 2), fnum(ep[k]["ppv_episode"], 2), fnum(ep[k]["se_durasi"], 2), fnum(ep[k]["palsu_per_jam"], 1),
             fnum(ep0[k]["palsu_per_jam"], 2)] for k in SISTEM]
    y = L.table(["Episode AF (EC57)", "Se episode", "+P episode", "Se durasi", "Palsu/jam (resmi)", "Palsu/jam (dikunci)"], rows, y,
                widths=[L.CW * .28, L.CW * .13, L.CW * .13, L.CW * .13, L.CW * .16, L.CW * .17], sorot=(4,))
    y = L.para(f"Window non-AF ≥ 120 bpm di CPSC ({thou(Lr['alarm_pita']['v3.1 paket']['≥ 120']['n'])} window) masih dialarmkan "
               f"{pct(Lr['alarm_pita']['D-1']['≥ 120']['alarm'], 0)} oleh D-1, jauh lebih tinggi dari sinus takikardia saat latihan fisik "
               f"({pct(lat['D-1']['alarm'], 2)}). Rekaman non-AF CPSC bisa memuat aritmia lain (ektopik, "
               "takikardia supraventrikular); labelnya hanya \"bukan AF\". Ini perlu dicek di data perangkat.", y, "small")
    # ── 5. per subjek ──
    y = L.page("Per subjek", "Sensitivitas AF per subjek ber-AF (window AF sama di kedua analisis)")
    y = L.chart(TMP / "se_subjek.png", y, 185)
    ps = Lr["per_subjek"]
    se_ok = [s for s in ps["D-1"] if ps["D-1"][s]["se"] is not None]
    al_ok = [s for s in ps["D-1"] if ps["D-1"][s]["alarm"] is not None]
    lebih_rendah = sum(ps["D-1"][s]["se"] < ps["v3.1 paket"][s]["se"] for s in se_ok)
    al_turun = sum(ps["D-1"][s]["alarm"] < ps["v3.1 paket"][s]["alarm"] for s in al_ok)
    rows = [[s, thou(ps["D-1"][s]["n_af"]), fnum(ps["v3.1 paket"][s]["se"], 2), fnum(ps["D-1"][s]["se"], 2), fnum(ps["D-0"][s]["se"], 2)]
            for s in sorted(se_ok, key=lambda s: ps["D-1"][s]["se"] - ps["v3.1 paket"][s]["se"])[:8]]
    y = L.table(["8 subjek dengan selisih Se terburuk", "n window AF", "Se v3.1", "Se D-1", "Se D-0"], rows, y,
                widths=[L.CW * .32, L.CW * .17, L.CW * .17, L.CW * .17, L.CW * .17])
    y = L.para(f"Sensitivitas D-1 lebih rendah dari v3.1 di {lebih_rendah} dari {len(se_ok)} subjek ber-AF, tetapi hanya subjek "
               f"{', '.join(Lr['se_turun_0_20'])} yang turun lebih dari 0,20. Alarm pada window non-AF lebih rendah di {al_turun} dari "
               f"{len(al_ok)} subjek yang punya window non-AF.", y)
    # ── 6. H4 ──
    y = L.page("Data yang sudah dipakai (H4, bukan uji independen)", "Ambang −1 dipilih setelah melihat titik-titik ini di v16.1")
    h4 = R["h4"]
    rows = [["AFDB (OOF, aturan 2 kelas v11.1)", f"{fnum(h4['afdb']['D-1']['Macro-F1'][0])} / {fnum(h4['afdb']['v3.1']['Macro-F1'][0])}",
             f"{_d(h4['afdb_banding']['D-1']['Macro-F1'])}, P = {fnum(h4['afdb_banding']['D-1']['Macro-F1'][3])}",
             f"{fnum(h4['afdb']['D-1']['Se'][0])} / {fnum(h4['afdb']['v3.1']['Se'][0])}", f"{fnum(h4['afdb']['D-1']['Sp'][0])} / {fnum(h4['afdb']['v3.1']['Sp'][0])}"]]
    for n, v in h4["eksternal"].items():
        rows.append([n, f"{fnum(v['titik_a']['Macro-F1'][0])} / {fnum(v['titik_b']['Macro-F1'][0])}", f"{_d(v['Macro-F1'])}, P = {fnum(v['Macro-F1'][3])}",
                     f"{fnum(v['titik_a']['Se'][0])} / {fnum(v['titik_b']['Se'][0])}", f"{fnum(v['titik_a']['Sp'][0])} / {fnum(v['titik_b']['Sp'][0])}"])
    y = L.table(["Data", "Macro-F1 D-1 / v3.1", "Δ [CI], P(Δ > 0)", "Se D-1 / v3.1", "Sp D-1 / v3.1"], rows, y,
                widths=[L.CW * .24, L.CW * .16, L.CW * .3, L.CW * .15, L.CW * .15])
    sh = lat["D-1"]["banding"]
    y = L.para(f"<b>Sinus takikardia asli ≥ 120 bpm</b> (latihan fisik v3.3, mode perangkat): D-1 {pct(lat['D-1']['alarm'], 2)} "
               f"{ci(lat['D-1']['ci'][0] * 100, lat['D-1']['ci'][1] * 100, 1)} vs v3.1 {pct(lat['v3.1 paket']['alarm'])}; Δ {sg(sh['delta'] * 100, 1)} poin.", y)
    y = L.para(f"Di Icentia11k (wearable) Macro-F1 D-1 sedikit lebih baik, tetapi sensitivitasnya lebih rendah "
               f"({fnum(h4['eksternal']['Icentia11k']['titik_a']['Se'][0])} vs {fnum(h4['eksternal']['Icentia11k']['titik_b']['Se'][0])}, "
               f"P = {fnum(h4['eksternal']['Icentia11k']['Se'][3])}). Data perangkat wearable SIHEDAF mungkin lebih mirip Icentia daripada "
               "CPSC, jadi sensitivitas harus diperiksa lagi di sana.", y)
    # ── 7. kesimpulan ──
    y = L.page("Kesimpulan, langkah berikutnya, batasan, reproduksi", "")
    for t in ("<b>Kesimpulan.</b> Ambang −1 yang dipilih dari kurva v16.1 bertahan di data yang belum pernah dinilai: sebagai detektor AF "
              "2 kelas, ECGFounder sama sensitifnya dengan paket v3.1 dan jauh lebih spesifik, dengan episode AF palsu ±4× lebih sedikit. "
              "Ini sistem pertama di seri v13–v16.2 yang lolos kriteria kandidat yang dikunci di data baru. Hasilnya hanya untuk 2 kelas; "
              "untuk 3 kelas paket v3.1 tetap.",
              "<b>Langkah berikutnya.</b> (1) Uji D-1 pada rekaman perangkat SIHEDAF berlabel dengan "
              "<font face='Arial-Bold'>v16.1_dua_kelas/deteksi_af.DetektorAF</font> (ambang −1), rancangan dikunci dulu. (2) Keputusan produk: "
              "cukup \"AF ya/tidak\", atau AFL tetap perlu dibedakan (lewat v3.1 sebagai keluaran kedua). (3) Periksa alarm pada ritme non-AF "
              "cepat dan sensitivitas di data wearable sebelum dipakai. (4) Infrastruktur: ECGFounder 30,8 juta parameter di backend, ±30 ms "
              "per window di CPU.",
              "<b>Batasan.</b> CPSC 2021: EKG dinamis (Holter) 200 Hz, lead I, 105 subjek; label rekaman non-AF hanya \"bukan AF\". AFL hampir "
              "tidak ada (2 subjek). Ambang dipilih dari AFDB, PTB-XL, dan Icentia11k, lalu diuji sekali di CPSC 2021; pengujian ulang ambang "
              "lain di CPSC 2021 tidak lagi independen.",
              "<b>Reproduksi.</b> <font face='Arial-Bold'>python tools/run_notebook.py v16.2_cpsc2021/uji_cpsc.ipynb</font>, lalu "
              "<font face='Arial-Bold'>build_report.py --extract</font> dan tanpa argumen. Butuh data/cpsc2021 (tools/unduh_physionet.py), "
              "bobot ECGFounder, dan cache v3.3, v10, v13, v15, v16, v16.1."):
        y = L.para(t, y)
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
