"""Laporan PDF v16.8: gerbang kualitas sinyal (SQI templat QRS) untuk S1 (D-1 + episode AF ≥ 30 s).

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v16.8/ringkasan.json (sel akhir gerbang.ipynb), hitung ulang alarm palsu S1/S1G di afpdb + derau, Se episode
     Icentia, dan alarm palsu nstdb berat untuk q terpilih dari cache per rekaman (assert), gambar grafik.
  2. tanpa argumen: rakit laporan/laporan_v16_8_gerbang_kualitas.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v16_8_gerbang_kualitas.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, Laporan, fnum, pct, sg, sumbu_koma, thou  # noqa: E402

TANGGAL = "10 November 2026"
KONDISI = ("afpdb bersih", "afpdb + ma 12 dB", "afpdb + ma 6 dB", "afpdb + ma 0 dB", "afpdb + em 12 dB", "afpdb + em 6 dB",
           "afpdb + em 0 dB", "Icentia non-AF")
LABEL = {"afpdb bersih": "bersih", "afpdb + ma 12 dB": "otot\n12 dB", "afpdb + ma 6 dB": "otot\n6 dB", "afpdb + ma 0 dB": "otot\n0 dB",
         "afpdb + em 12 dB": "elektroda\n12 dB", "afpdb + em 6 dB": "elektroda\n6 dB", "afpdb + em 0 dB": "elektroda\n0 dB",
         "Icentia non-AF": "Icentia\ntanpa AF"}


def extract():
    import numpy as np
    import gerbang as G
    R = json.loads((G.CACHE / "ringkasan.json").read_text(encoding="utf-8"))
    q = R["tahapA"]["terpilih"]
    B = G.pasang_sqi("afpdb_bising", G.muat_bising())
    d = G.subset(B, (B["derau"] == "ma") & (B["snr"] == 6))
    for qq, s in ((None, "S1"), (q, "S1G")):
        assert abs(G.nilai(d, qq)["palsu_per_jam"] - R["validasi"]["afpdb + ma 6 dB"][s]["palsu_per_jam"]) < 1e-9, s
    ic, _ = G.K.muat("Icentia")
    ic = G.pasang_sqi("Icentia", ic)
    a = G.K.bagian(ic, ic["grup"] == "AF")
    for qq, s in ((None, "S1"), (q, "S1G")):
        assert abs(G.nilai(a, qq)["se_episode_30"] - R["validasi"]["Icentia AF"][s]["se_episode_30"]) < 1e-9, s
    ns, _ = G.K.muat("nstdb")
    ns = G.pasang_sqi("nstdb", ns)
    berat = G.subset(ns, np.isin(ns["grup"], ("12", "06", "00", "_6")))
    row = next(r for r in R["tahapA"]["grid"] if r["q"] == q)
    assert abs(G.nilai(berat, q)["palsu_per_jam"] - row["palsu_nstdb_berat"]) < 1e-9
    TMP.mkdir(exist_ok=True)
    (TMP / "v16_8.json").write_text(json.dumps({"R": R}, indent=1, ensure_ascii=False), encoding="utf-8")
    grafik(R)
    print("extract selesai; assert alarm palsu afpdb + ma 6 dB, Se Icentia AF (S1, S1G), dan nstdb berat (q terpilih) cocok dengan ringkasan.json")


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
    rows = R["tahapA"]["grid"]
    xs = list(range(len(rows)))
    lab = ["tanpa" if r["q"] is None else fnum(r["q"], 1) for r in rows]
    # 1. tahap A
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.7), dpi=200)
    axs[0].plot(xs, [r["palsu_nstdb_berat"] for r in rows], marker="s", color=ORANYE, label="nstdb 12 … −6 dB")
    axs[0].plot(xs, [r["palsu_nsrdb"] for r in rows], marker="o", color=BIRU, label="NSRDB (bersih)")
    axs[0].axhline(1, color=INK, lw=.8, ls="--"); axs[0].set_ylabel("alarm palsu per jam (S1)", fontsize=8.5)
    axs[0].legend(frameon=False, fontsize=7.5); axs[0].set_title("Alarm palsu", fontsize=9, color=INK)
    axs[1].plot(xs, [r["se_ltafdb"] for r in rows], marker="s", color=ORANYE, label="LTAFDB")
    axs[1].plot(xs, [r["se_mit"] for r in rows], marker="^", color=HIJAU, label="MIT-BIH")
    axs[1].set_ylabel("Se episode AF ≥ 30 s", fontsize=8.5); axs[1].legend(frameon=False, fontsize=7.5)
    axs[1].set_title("Sensitivitas", fontsize=9, color=INK)
    for ax in axs:
        ax.set_xticks(xs, lab, fontsize=8); ax.set_xlabel("ambang SQI q", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "tahapA.png"); plt.close(fig)
    # 2. validasi
    Vd = R["validasi"]
    fig, ax = plt.subplots(figsize=(7.6, 2.9), dpi=200)
    sis = (("S1", ORANYE, None, "S1"), ("S1G", ORANYE, "//", "S1 + gerbang"), ("R", BIRU, None, "v3.1 + 30 s"), ("RG", BIRU, "//", "v3.1 + 30 s + gerbang"))
    for i, (s, col, hat, nm) in enumerate(sis):
        ax.bar([x + (i - 1.5) * .2 for x in range(len(KONDISI))], [Vd[k][s]["palsu_per_jam"] for k in KONDISI], .19, color=col, hatch=hat,
               edgecolor="white", label=nm)
    ax.axhline(1, color=INK, lw=.8, ls="--")
    ax.axvspan(1.5, 2.5, color="#FFF1E8", zorder=0); ax.text(2, ax.get_ylim()[1] * .62, "V1", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(KONDISI)), [LABEL[k] for k in KONDISI], fontsize=7.5)
    ax.set_ylabel("alarm palsu per jam", fontsize=8.5); ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "validasi.png"); plt.close(fig)


def build():
    R = json.loads((TMP / "v16_8.json").read_text(encoding="utf-8"))["R"]
    A, Vd, k, b = R["tahapA"], R["validasi"], R["keputusan"], R["V1_boot"]
    ok = lambda x: "lolos" if x else "gagal"  # noqa: E731
    q = A["terpilih"]
    v1, ia, inn = Vd["afpdb + ma 6 dB"], Vd["Icentia AF"], Vd["Icentia non-AF"]
    L = Laporan(OUT, "v16.8: gerbang kualitas sinyal", "SIHEDAF  /  V16.8  /  GERBANG KUALITAS", "v16.8_gerbang_kualitas/gerbang.ipynb, cache/v16.8/",
                TANGGAL)
    y = L.page("Ringkasan: gerbang templat QRS tidak menolong derau otot", "Rancangan dikunci 10 November 2026 sebelum SQI/SNR dihitung (+ amandemen A)")
    y = L.kartu([(f"{fnum(v1['S1']['palsu_per_jam'], 2)} → {fnum(v1['S1G']['palsu_per_jam'], 2)}", "palsu/jam, derau otot 6 dB",
                  "target ≤ 1,0: gagal"),
                 (f"{fnum(Vd['afpdb + em 0 dB']['S1']['palsu_per_jam'], 1)} → {fnum(Vd['afpdb + em 0 dB']['S1G']['palsu_per_jam'], 1)}",
                  "palsu/jam, elektroda 0 dB", "jenis derau tahap A"),
                 (f"{fnum(ia['S1']['se_episode_30'], 3)} → {fnum(ia['S1G']['se_episode_30'], 3)}", "Se episode AF, Icentia", "biaya kecil"),
                 (fnum(q, 1), "ambang SQI terpilih", "batas atas grid")], y)
    y = L.para("<b>Pertanyaan.</b> Di v16.7, S1 (D-1 + aturan episode AF ≥ 30 s) memberi 5–8 alarm AF palsu per jam pada derau berat, karena "
               "window berderau lolos cek kualitas perangkat. Apakah gerbang kualitas sederhana, yaitu window dengan bentuk QRS yang tidak "
               "konsisten dianggap tidak terbaca, menekan alarm itu tanpa mengorbankan sensitivitas?", y)
    y = L.para(JAWABAN(R), y)
    rows = [["V1: alarm palsu S1G ≤ 1,0/jam, afpdb + derau otot 6 dB", ok(k["V1_palsu"]), fnum(v1["S1G"]["palsu_per_jam"], 2)],
            ["V1: S1G lebih rendah dari S1 (P(Δ&lt;0) ≥ 0,95)", ok(k["V1_lebih_rendah"]),
             f"Δ {sg(b['delta'], 2)} [{sg(b['ci'][0], 2)}; {sg(b['ci'][1], 2)}], P = {fnum(b['p_kurang'])}"],
            ["V2: Se episode Icentia ≥ 0,90 dan turun ≤ 0,02", ok(k["V2"]), f"{fnum(ia['S1']['se_episode_30'])} → {fnum(ia['S1G']['se_episode_30'])}"],
            ["V3: alarm palsu Icentia tanpa AF ≤ 1,0/jam", ok(k["V3"]), f"{fnum(inn['S1']['palsu_per_jam'], 2)} → {fnum(inn['S1G']['palsu_per_jam'], 2)}"],
            ["Keputusan: S1G menggantikan S1", "tidak", "S1 tetap sistem terbaik"]]
    y = L.table(["Kriteria yang dikunci", "Hasil", "Angka"], rows, y, widths=[L.CW * .52, L.CW * .1, L.CW * .38], sorot=(5,))
    y = L.para(TEMUAN(R), y)
    # 2. tahap A
    y = L.page("Tahap A: memilih ambang SQI di data pengembangan", "nstdb (derau elektroda), NSRDB, LTAFDB, MIT-BIH; semuanya sudah dipakai sebelumnya")
    y = L.chart(TMP / "tahapA.png", y, 190)
    rows = [["tanpa" if r["q"] is None else fnum(r["q"], 1), fnum(r["se_ltafdb"]), fnum(r["se_mit"]), fnum(r["palsu_nsrdb"], 2),
             fnum(r["palsu_nstdb_berat"], 2), pct(r["gerbang_nsrdb"], 1), pct(r["gerbang_nstdb_berat"], 1)] for r in A["grid"]]
    y = L.table(["q", "Se LTAFDB", "Se MIT-BIH", "Palsu/jam NSRDB", "Palsu/jam nstdb berat", "Digerbang NSRDB", "Digerbang nstdb"], rows, y,
                widths=[L.CW * .1, L.CW * .13, L.CW * .13, L.CW * .16, L.CW * .18, L.CW * .15, L.CW * .15], sorot=(len(rows),))
    y = L.para(TAHAPA_TEKS(R), y)
    # 3. validasi
    y = L.page("Tahap B: validasi sekali di data yang tidak dipakai memilih", "afpdb + derau nstdb (otot primer, elektroda sekunder) dan Icentia11k")
    y = L.chart(TMP / "validasi.png", y, 200)
    rows = [[k_.replace("afpdb + ma", "afpdb + otot").replace("afpdb + em", "afpdb + elektroda"), fnum(Vd[k_]["S1"]["palsu_per_jam"], 2),
             fnum(Vd[k_]["S1G"]["palsu_per_jam"], 2), fnum(Vd[k_]["R"]["palsu_per_jam"], 2), fnum(Vd[k_]["RG"]["palsu_per_jam"], 2),
             pct(Vd[k_]["S1G"]["frac_gerbang"], 1)] for k_ in KONDISI]
    rows.append(["Icentia AF: Se episode ≥ 30 s", fnum(ia["S1"]["se_episode_30"]), fnum(ia["S1G"]["se_episode_30"]), fnum(ia["R"]["se_episode_30"]),
                 fnum(ia["RG"]["se_episode_30"]), pct(ia["S1G"]["frac_gerbang"], 1)])
    y = L.table(["Kondisi (alarm palsu/jam)", "S1", "S1 + gerbang", "v3.1", "v3.1 + gerbang", "Digerbang"], rows, y,
                widths=[L.CW * .32, L.CW * .12, L.CW * .15, L.CW * .12, L.CW * .16, L.CW * .13], sorot=(3,))
    y = L.para(VALIDASI_TEKS(R), y)
    # 4. metode
    y = L.page("Metode, pra-registrasi, kendali mutu", "RANCANGAN.md dikunci; sha256 di logs/v16.8_praregistrasi.log")
    for tx in METODE(R):
        y = L.para(tx, y)
    # 5. kesimpulan
    y = L.page("Kesimpulan, langkah berikutnya, batasan, reproduksi", "")
    for tx in KESIMPULAN(R):
        y = L.para(tx, y)
    lk = R["lingkungan"]
    y = L.para(f"Lingkungan: PC desktop, GPU {lk['gpu']}, commit {lk['commit']} (+ perubahan belum di-commit), Python {lk['pustaka']['python']}, "
               f"numpy {lk['pustaka']['numpy']}, torch {lk['pustaka']['torch']}.", y, "small")
    L.selesai()
    print("PDF:", OUT, L.page_no, "halaman")


# ── teks naratif (ditulis setelah hasil dilihat; angka dari ringkasan) ──
def JAWABAN(R):
    Vd = R["validasi"]
    return ("<b>Jawaban singkat.</b> Tidak untuk derau otot. Tahap A memilih ambang q = "
            f"{fnum(R['tahapA']['terpilih'], 1)}, yang di data pengembangan menghapus semua alarm palsu nstdb. Pada derau otot 6 dB (kriteria "
            f"utama) alarm palsu hampir tidak berubah ({fnum(Vd['afpdb + ma 6 dB']['S1']['palsu_per_jam'], 2)} → "
            f"{fnum(Vd['afpdb + ma 6 dB']['S1G']['palsu_per_jam'], 2)} per jam). Gerbang hanya membantu untuk derau elektroda, jenis derau yang "
            f"dipakai memilih q ({fnum(Vd['afpdb + em 0 dB']['S1']['palsu_per_jam'], 2)} → {fnum(Vd['afpdb + em 0 dB']['S1G']['palsu_per_jam'], 2)} "
            "pada 0 dB). Biayanya di wearable nyata kecil, tetapi tanpa V1 gerbang ini tidak dipakai. <b>S1 tetap sistem terbaik.</b>")


def TEMUAN(R):
    Vd, A = R["validasi"], R["tahapA"]
    return ("<b>Temuan.</b> (1) Templat QRS mendeteksi derau elektroda yang mengubah bentuk QRS, tetapi derau otot (frekuensi tinggi) "
            "membuat QRS tetap mirip templatnya; alarm palsu saat derau otot tidak berasal dari window yang digerbang. (2) Di afpdb bersih "
            f"alarm palsu S1 ({fnum(Vd['afpdb bersih']['S1']['palsu_per_jam'], 2)}/jam) tidak terkait derau: gerbang hanya menurunkannya ke "
            f"{fnum(Vd['afpdb bersih']['S1G']['palsu_per_jam'], 2)}. (3) Gerbang yang sama banyak menolong v3.1 (mis. derau otot 0 dB "
            f"{fnum(Vd['afpdb + ma 0 dB']['R']['palsu_per_jam'], 2)} → {fnum(Vd['afpdb + ma 0 dB']['RG']['palsu_per_jam'], 2)}), jadi sebagian "
            "alarm v3.1 berasal dari R-peak yang salah. (4) Pemilihan q jatuh di batas atas grid (0,9); di nstdb gerbang q = 0,9 mengeluarkan "
            f"{pct(next(r for r in A['grid'] if r['q'] == A['terpilih'])['gerbang_nstdb_berat'], 0)} window, terlalu cocok dengan jenis derau itu.")


def TAHAPA_TEKS(R):
    g = {r["q"]: r for r in R["tahapA"]["grid"]}
    return ("Semua q memenuhi syarat sensitivitas (penurunan ≤ 0,01). Se bahkan naik sedikit pada q tinggi karena window yang digerbang juga "
            "keluar dari episode referensi (episode referensi ikut terpotong); ini artefak definisi, bukan perbaikan deteksi. Alarm palsu nstdb "
            f"berat turun dari {fnum(g[None]['palsu_nstdb_berat'], 2)} ke {fnum(g[0.9]['palsu_nstdb_berat'], 2)} hanya pada q = 0,9, di batas atas "
            "grid; batas itu tercatat sebagai tanda bahaya (AGENTS.md, anti-pattern grid).")


def VALIDASI_TEKS(R):
    Vd = R["validasi"]
    return ("Derau otot (V1, primer) belum pernah dipakai; derau elektroda sama jenisnya dengan tahap A. Alarm palsu dihitung per jam window "
            "yang dinilai (window yang digerbang tidak dihitung, sama seperti window yang gagal cek kualitas). Di Icentia gerbang mengeluarkan "
            f"{pct(Vd['Icentia AF']['S1G']['frac_gerbang'], 1)} window pasien AF dan {pct(Vd['Icentia non-AF']['S1G']['frac_gerbang'], 1)} window "
            "pasien tanpa AF; alarm palsu dan sensitivitas hampir tidak berubah.")


def METODE(R):
    return [
        "<b>SQI.</b> Sinyal kanal 0 → 50 Hz, R-peak detektor perangkat di seluruh rekaman, lalu pembersih.selaraskan v3.4 (sinyal bandpass "
        "detektor): korelasi tiap detak dengan templat QRS median detak dalam ±5 s. SQI window = median korelasi detak di [awal, awal + 10 s); "
        "tanpa korelasi valid = NaN. Gerbang q: window dengan SQI &lt; q atau NaN dikeluarkan (memutus run), lalu S1 seperti biasa.",
        "<b>Tahap A (dikunci).</b> Grid q = 0,5; 0,6; 0,7; 0,8; 0,9. Syarat: Se episode AF ≥ 30 s turun ≤ 0,01 di LTAFDB dan MIT-BIH. "
        "Di antaranya pilih alarm palsu gabungan nstdb SNR 12, 6, 0, −6 dB terendah (selisih ≤ 0,05/jam → q lebih kecil).",
        "<b>Tahap B.</b> afpdb n01–n50, p01–p50 (non-AF, 128 Hz) + derau nstdb kanal 0 (otot ma, elektroda em; 360 → 128 Hz) di seluruh "
        "rekaman pada SNR 12, 6, 0 dB. SNR = 10·log10(var sinyal / var derau terskala) setelah Butterworth orde 2 0,5–40 Hz (amandemen A); "
        "offset derau acak per rekaman (seed 2026 + 168); SNR terukur dicek = 6,000 dB pada contoh. Icentia: 600 segmen v16.7. Bootstrap rekaman "
        "berpasangan 2.000×, seed 2026, penyebut jam boleh berbeda.",
        "<b>Kendali mutu.</b> Laporan ini menghitung ulang alarm palsu afpdb + otot 6 dB (S1, S1G), Se Icentia AF (S1, S1G), dan alarm palsu "
        "nstdb berat pada q terpilih dari cache (assert).",
    ]


def KESIMPULAN(R):
    return [
        "<b>Kesimpulan.</b> Gerbang templat QRS tidak lolos (V1 gagal): derau otot, jenis derau yang tidak dilihat saat memilih ambang, tetap "
        "memicu alarm AF palsu. S1 = D-1 + aturan episode AF ≥ 30 s tetap sistem terbaik untuk deteksi AF 2 kelas; kelemahan derau berat tetap "
        "terbuka.",
        "<b>Langkah berikutnya.</b> Gerbang derau perlu fitur yang menangkap derau frekuensi tinggi (mis. rasio daya &gt; 40 Hz pada sinyal asli, "
        "sebelum diturunkan ke 50 Hz) atau ditanam di perangkat (deteksi lepas elektroda/akselerometer). Ambangnya dipilih dan diuji dengan "
        "rekaman derau perangkat SIHEDAF, bukan nstdb/afpdb yang kini terpakai.",
        "<b>Batasan.</b> Derau sintetis dijumlahkan ke seluruh rekaman (bukan blok); nstdb hanya punya 30 menit derau per jenis, jadi derau "
        "hampir sama di semua rekaman afpdb. q terpilih di batas grid. Window yang digerbang juga memotong episode referensi.",
        "<b>Reproduksi.</b> Dari root repo: python tools/run_notebook.py v16.8_gerbang_kualitas/gerbang.ipynb (butuh cache v16.3–v16.7, data "
        "afpdb, nstdb, ltafdb, mitdb, nsrdb, icentia_v167; pemrosesan afpdb berderau dan SQI ±6 menit). Lalu dari folder v16.8_gerbang_kualitas: "
        "python laporan/build_report.py --extract, lalu python laporan/build_report.py.",
    ]


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
