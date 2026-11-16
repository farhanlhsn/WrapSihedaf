"""Laporan PDF v16.7: uji konfirmasi S1 (D-1 + episode AF ≥ 30 s) di Icentia11k (wearable kontinu) dan nstdb (derau).

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v16.7/ringkasan.json (sel akhir konfirmasi.ipynb), hitung ulang Se episode dan alarm palsu per jam S1/R di
     Icentia dan alarm palsu S1 per SNR nstdb dari cache per rekaman lalu cocokkan (assert), gambar grafik.
  2. tanpa argumen: rakit laporan/laporan_v16_7_konfirmasi_wearable.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v16_7_konfirmasi_wearable.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, Laporan, fnum, pct, sg, sumbu_koma, thou  # noqa: E402

TANGGAL = "10 November 2026"
S1, RR, S3 = "S1 = D-1 + 30 s", "R = v3.1 + 30 s", "S3 = (D-1 atau v3.1) + 30 s"
SIS = (S1, RR, S3)
GAYA = {S1: (ORANYE, "s", "//"), RR: (BIRU, "o", None), S3: (HIJAU, "^", None)}
PENDEK = {S1: "S1 (D-1)", RR: "R (v3.1)", S3: "S3 (D-1 atau v3.1)"}
SNR = ("bersih", "24", "18", "12", "06", "00", "_6")
SNR_LAB = {"bersih": "bersih", "24": "24 dB", "18": "18 dB", "12": "12 dB", "06": "6 dB", "00": "0 dB", "_6": "−6 dB"}


def extract():
    import numpy as np
    import konfirmasi as K
    R = json.loads((K.CACHE / "ringkasan.json").read_text(encoding="utf-8"))
    ic, _ = K.muat("Icentia")
    for g in ("AF", "non-AF"):
        d = K.bagian(ic, ic["grup"] == g)
        for s, af in K.sistem(d).items():
            if s == S3:
                continue
            m = K.C.nilai(d, af, 30, 0)
            for kk in ("palsu_per_jam", "se_episode_30"):
                x, y = m[kk], R["icentia"][g][s][kk]
                assert (np.isnan(x) and np.isnan(y)) or abs(x - y) < 1e-9, (g, s, kk)
    ns, _ = K.muat("nstdb")
    for snr in K.SNR:
        d = K.bagian(ns, ns["grup"] == snr)
        assert abs(K.C.nilai(d, d["d1"], 30, 0)["palsu_per_jam"] - R["nstdb"][snr][S1]) < 1e-9, snr
    TMP.mkdir(exist_ok=True)
    (TMP / "v16_7.json").write_text(json.dumps({"R": R}, indent=1, ensure_ascii=False), encoding="utf-8")
    grafik(R)
    print("extract selesai; assert Se episode/alarm palsu Icentia (S1, R) dan alarm palsu S1 per SNR nstdb cocok dengan ringkasan.json")


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
    I = R["icentia"]
    # 1. Icentia: Se episode vs alarm palsu per jam (titik kerja)
    fig, ax = plt.subplots(figsize=(7.6, 2.9), dpi=200)
    for s in SIS:
        col, mk, _ = GAYA[s]
        ax.scatter(I["non-AF"][s]["palsu_per_jam"], I["AF"][s]["se_episode_30"], s=90, color=col, marker=mk, label=PENDEK[s], zorder=3)
    lo, hi = R["ci"]["palsu"], R["ci"]["se"]
    ax.errorbar(I["non-AF"][S1]["palsu_per_jam"], I["AF"][S1]["se_episode_30"],
                xerr=[[I["non-AF"][S1]["palsu_per_jam"] - lo[0]], [lo[1] - I["non-AF"][S1]["palsu_per_jam"]]],
                yerr=[[I["AF"][S1]["se_episode_30"] - hi[0]], [hi[1] - I["AF"][S1]["se_episode_30"]]], color=ORANYE, lw=1, capsize=3, zorder=2)
    ax.axvline(1, color=INK, lw=.8, ls="--"); ax.axhline(.9, color=INK, lw=.8, ls="--")
    ax.fill_between([0, 1], .9, 1.0, color="#E6F3F2", zorder=0)
    ax.text(.04, .993, "daerah target", fontsize=7.5, color=INK, va="top")
    ax.set_xlim(0, 3.2); ax.set_ylim(.85, 1.0)
    ax.set_xlabel("alarm palsu per jam (300 pasien tanpa AF)", fontsize=8.5); ax.set_ylabel("Se episode AF ≥ 30 s\n(300 pasien AF)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#E6ECF1")
    sumbu_koma(ax); sumbu_koma(ax, "x")
    fig.tight_layout(); fig.savefig(TMP / "icentia.png"); plt.close(fig)
    # 2. nstdb: alarm palsu per jam per SNR
    N = R["nstdb"]
    fig, ax = plt.subplots(figsize=(7.6, 2.5), dpi=200)
    for i, s in enumerate(SIS):
        col, _, hat = GAYA[s]
        ax.bar([x + (i - 1) * .26 for x in range(len(SNR))], [N[k][s] for k in SNR], .25, color=col, hatch=hat, edgecolor="white", label=PENDEK[s])
    ax.axhline(1, color=INK, lw=.8, ls="--")
    ax.set_xticks(range(len(SNR)), [SNR_LAB[k] for k in SNR], fontsize=8); ax.set_xlabel("SNR derau elektroda + gerak (118/119 MIT-BIH)", fontsize=8.5)
    ax.set_ylabel("alarm palsu per jam", fontsize=8.5); ax.legend(frameon=False, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "nstdb.png"); plt.close(fig)


def build():
    R = json.loads((TMP / "v16_7.json").read_text(encoding="utf-8"))["R"]
    I, k, N = R["icentia"], R["keputusan"], R["nstdb"]
    ok = lambda x: "lolos" if x else "gagal"  # noqa: E731
    a, n = I["AF"], I["non-AF"]
    L = Laporan(OUT, "v16.7: konfirmasi S1 di wearable", "SIHEDAF  /  V16.7  /  KONFIRMASI WEARABLE",
                "v16.7_konfirmasi_wearable/konfirmasi.ipynb, cache/v16.7/", TANGGAL)
    y = L.page("Ringkasan: S1 terkonfirmasi di rekaman wearable kontinu", "Rancangan dikunci 10 November 2026 sebelum sinyal dibaca")
    y = L.kartu([(fnum(a[S1]["se_episode_30"], 2), "Se episode AF ≥ 30 s (S1)", f"v3.1 + 30 s: {fnum(a[RR]['se_episode_30'], 2)}"),
                 (fnum(n[S1]["palsu_per_jam"], 2), "alarm palsu/jam (S1)", f"v3.1 + 30 s: {fnum(n[RR]['palsu_per_jam'], 2)}"),
                 (f"{thou(R['data']['Icentia']['jam'])} jam", "wearable satu lead", "600 pasien Icentia11k"),
                 (f"{fnum(N['06'][S1], 1)}/jam", "alarm palsu S1 pada SNR 6 dB", "derau berat: perlu gerbang")], y)
    y = L.para("<b>Pertanyaan.</b> S1 (D-1 + aturan episode AF ≥ 30 s) adalah kandidat terbaik untuk deteksi AF 2 kelas, tetapi buktinya "
               "dari data Holter dan di v16.6 hanya analisis sekunder. Kelemahan D-1 yang tercatat ada di EKG wearable. Apakah S1 memenuhi "
               "target produk di rekaman wearable kontinu sebagai hipotesis <b>primer</b>, dan seberapa tahan terhadap derau?", y)
    y = L.para(JAWABAN(R), y)
    rows = [["Target: Se episode AF ≥ 30 s ≥ 0,90 (300 pasien AF)", ok(k["target_se"]),
             f"{fnum(a[S1]['se_episode_30'])} [{fnum(R['ci']['se'][0])}–{fnum(R['ci']['se'][1])}]"],
            ["Target: alarm palsu ≤ 1,0/jam (300 pasien tanpa AF)", ok(k["target_palsu"]),
             f"{fnum(n[S1]['palsu_per_jam'], 2)} [{fnum(R['ci']['palsu'][0], 2)}–{fnum(R['ci']['palsu'][1], 2)}]"],
            ["H1: alarm palsu S1 &lt; v3.1 + 30 s (P(Δ&lt;0) ≥ 0,95)", ok(k["H1"]),
             f"Δ {sg(R['H1']['delta'], 2)} [{sg(R['H1']['ci'][0], 2)}; {sg(R['H1']['ci'][1], 2)}], P = {fnum(R['H1']['p_kurang'])}"],
            ["H2: Se episode S1 bukan lebih buruk dari v3.1 + 30 s", ok(k["H2"]),
             f"Δ {sg(R['H2']['delta'])} [{sg(R['H2']['ci'][0])}; {sg(R['H2']['ci'][1])}], P(Δ&gt;0) = {fnum(R['H2']['p_lebih'])}"],
            ["H3 nstdb: alarm palsu S1 ≤ 1,0/jam pada SNR ≥ 12 dB", ok(k["H3"]),
             f"24/18/12 dB: {fnum(N['24'][S1], 2)} / {fnum(N['18'][S1], 2)} / {fnum(N['12'][S1], 2)}"],
            ["Keputusan: S1 terkonfirmasi untuk wearable kontinu", "ya" if k["terkonfirmasi"] else "tidak", "H3 tidak ikut keputusan ini"]]
    y = L.table(["Kriteria yang dikunci", "Hasil", "Angka"], rows, y, widths=[L.CW * .5, L.CW * .1, L.CW * .4], sorot=(6,))
    y = L.para(TEMUAN(R), y)
    # 2. Icentia
    y = L.page("Icentia11k: titik kerja di wearable kontinu", "Bootstrap pasien berpasangan 2.000×, seed 2026; garis putus-putus = target")
    y = L.chart(TMP / "icentia.png", y, 200)
    rows = []
    for g, lab in (("AF", "300 pasien AF"), ("non-AF", "300 pasien tanpa AF")):
        for s in SIS:
            m = I[g][s]
            rows.append([lab if s == S1 else "", PENDEK[s], fnum(m["palsu_per_jam"], 2),
                         "–" if m["se_episode_30"] != m["se_episode_30"] else fnum(m["se_episode_30"]),
                         "–" if m["se_durasi"] != m["se_durasi"] else fnum(m["se_durasi"]),
                         "–" if not m["ppv_durasi"] else fnum(m["ppv_durasi"])])
    y = L.table(["Kelompok", "Sistem", "Palsu/jam", "Se episode ≥ 30 s", "Se durasi", "+P durasi"], rows, y,
                widths=[L.CW * .22, L.CW * .2, L.CW * .13, L.CW * .17, L.CW * .14, L.CW * .14], sorot=(1, 4))
    y = L.para(ICENTIA_TEKS(R), y)
    # 3. nstdb
    y = L.page("nstdb: uji stres derau elektroda dan gerak", "118 dan 119 MIT-BIH + derau (SNR 24 … −6 dB); 1 jam per SNR")
    y = L.chart(TMP / "nstdb.png", y, 175)
    rows = [[SNR_LAB[s]] + [fnum(N[s][x], 2) for x in SIS] + [fnum(N[s]["jam"], 2)] for s in SNR]
    y = L.table(["SNR", "S1", "R (v3.1)", "S3", "Jam"], rows, y, widths=[L.CW * .2, L.CW * .2, L.CW * .2, L.CW * .2, L.CW * .2])
    y = L.para(NSTDB_TEKS(R), y)
    # 4. metode
    y = L.page("Metode, pra-registrasi, kendali mutu", "RANCANGAN.md dikunci; sha256 di logs/v16.7_praregistrasi.log")
    for tx in METODE(R):
        y = L.para(tx, y)
    # 5. kesimpulan
    y = L.page("Kesimpulan, sistem terbaik, batasan, reproduksi", "")
    for tx in KESIMPULAN(R):
        y = L.para(tx, y)
    lk = R["lingkungan"]
    y = L.para(f"Lingkungan: PC desktop, GPU {lk['gpu']}, commit {lk['commit']} (+ perubahan belum di-commit), Python {lk['pustaka']['python']}, "
               f"numpy {lk['pustaka']['numpy']}, torch {lk['pustaka']['torch']}.", y, "small")
    L.selesai()
    print("PDF:", OUT, L.page_no, "halaman")


# ── teks naratif (ditulis setelah hasil dilihat; angka dari ringkasan) ──
def JAWABAN(R):
    I, N = R["icentia"], R["nstdb"]
    a, n = I["AF"], I["non-AF"]
    return ("<b>Jawaban singkat.</b> Ya untuk wearable kontinu: keempat kriteria primer lolos. S1 menangkap "
            f"{pct(a[S1]['se_episode_30'], 1)} episode AF ≥ 30 s (v3.1 + 30 s {pct(a[RR]['se_episode_30'], 1)}) dengan "
            f"{fnum(n[S1]['palsu_per_jam'], 2)} alarm palsu per jam pada pasien tanpa AF (v3.1 + 30 s {fnum(n[RR]['palsu_per_jam'], 2)}). "
            "Kekurangan sensitivitas D-1 per window di wearable tidak terbawa ke tingkat episode: satu episode cukup tertangkap di "
            f"sebagian window-nya. <b>Derau berat tetap masalah:</b> pada SNR 12 dB S1 memberi {fnum(N['12'][S1], 2)} alarm palsu per jam "
            f"(H3 gagal tipis), pada 6 dB ke bawah {fnum(min(N['06'][S1], N['00'][S1], N['_6'][S1]), 1)}–"
            f"{fnum(max(N['06'][S1], N['00'][S1], N['_6'][S1]), 1)}.")


def TEMUAN(R):
    I = R["icentia"]
    a, n = I["AF"], I["non-AF"]
    x = sorted(R["nonaf_per_pasien_S1"].values())
    n_alarm = sum(v > 0 for v in x)
    return ("<b>Temuan.</b> (1) Alarm palsu S1 terkumpul di sedikit pasien: "
            f"{n_alarm} dari {len(x)} pasien tanpa AF punya ≥ 1 alarm palsu, sisanya nol (tertinggi {fnum(x[-1], 1)} per jam). "
            f"(2) S3 (D-1 atau v3.1) menaikkan sensitivitas ke {fnum(a[S3]['se_episode_30'])}, tetapi alarm palsunya "
            f"{fnum(n[S3]['palsu_per_jam'], 2)} per jam, jadi tidak sepadan (deskriptif). (3) Pada waktu non-AF ≥ 120 bpm S1 mengalarmkan "
            f"{pct(R['pita'][S1]['≥ 120']['alarm'], 1)} window, v3.1 + 30 s {pct(R['pita'][RR]['≥ 120']['alarm'], 0)}. "
            f"(4) +P durasi S1 di pasien AF {fnum(a[S1]['ppv_durasi'])}: hampir semua waktu yang dialarmkan memang AF.")


def ICENTIA_TEKS(R):
    d = R["data"]["Icentia"]
    return ("Icentia11k adalah patch CardioSTAT satu lead (250 Hz) yang dipakai pasien rawat jalan hingga 2 minggu. Per pasien diambil satu "
            "segmen ±70 menit. Pasien AF: segmen dengan episode AF ≥ 30 s yang tidak pernah diproses v10 (pasiennya ada di seleksi v10, tetapi "
            "waktunya berbeda). Pasien tanpa AF: pasien yang tidak pernah dipakai. Window tanpa label ritme dibuang "
            f"({thou(d['dibuang_label'])} dari {thou(d['window_lolos'])} window lolos cek kualitas). D-1 dan v3.1 tidak pernah dilatih dengan Icentia.")


def NSTDB_TEKS(R):
    N = R["nstdb"]
    return ("Derau elektroda dan gerak ditambahkan ke rekaman 118 dan 119 MIT-BIH (tanpa AF) dalam blok 2 menit mulai menit ke-5. Tanpa derau "
            f"dan pada 24–18 dB S1 tidak memberi alarm palsu. Pada 12 dB S1 memberi {fnum(N['12'][S1], 2)} per jam (satu alarm dalam ±1 jam), "
            f"v3.1 {fnum(N['12'][RR], 2)}. Pada 6 dB ke bawah kedua sistem memberi 5–11 alarm palsu per jam: window berderau masih lolos cek "
            "kualitas perangkat. Data hanya 2 rekaman per SNR, jadi angka per SNR sangat kasar. Kesimpulannya, sebelum uji perangkat, S1 butuh "
            "gerbang kualitas sinyal yang lebih ketat. Gerbang itu harus dirancang dan diuji dengan data derau baru, bukan nstdb.")


def METODE(R):
    d = R["data"]
    return [
        "<b>Sistem.</b> S1 = d1 &gt; −1 (ECGFounder 1-lead zero-shot) lalu aturan v16.5: run window AF berjarak ≤ langkah + 0,5 s, alarm bila "
        "rentang ≥ 30 s. R = paket v3.1 dengan aturan yang sama. S3 = D-1 atau v3.1 (deskriptif). Tidak ada penyetelan.",
        "<b>Icentia11k</b> (PhysioNet, CC BY-NC-SA, hanya evaluasi). Seleksi dari pemindaian anotasi v10, seed 2026 + 167: 300 pasien AF (satu "
        "segmen dengan episode (AFIB/(AFL ≥ 30 s di luar seleksi v10) dan 300 pasien tanpa AF di segmen mana pun dan di luar seleksi v10 (satu "
        "segmen dengan (N ≥ 3.000 s). Unduh dari mirror S3 dengan SHA256. Label: (AFIB, (AFL = AF; (N = non-AF; berlaku sampai label ritme "
        f"berikutnya atau ')'. {thou(d['Icentia']['window'])} window, {thou(d['Icentia']['jam'])} jam. Unit bootstrap = pasien.",
        f"<b>nstdb</b> (PhysioNet 1.0.0, dari physionet.org karena tidak ada di mirror S3): 12 rekaman 30 menit, kanal 0, 360 Hz; "
        "pembanding bersih = 118 dan 119 MIT-BIH dengan mesin yang sama.",
        "<b>Rantai prediksi</b> = v16.4 (50 Hz, detektor R perangkat, cek kualitas window, ECGFounder, paket v3.1). Metrik episode = v16.5 "
        "(interval sebenarnya), bootstrap rasio = v16.6.",
        "<b>Kendali mutu.</b> Laporan ini menghitung ulang Se episode dan alarm palsu per jam S1 dan R di Icentia, serta alarm palsu S1 per "
        "SNR nstdb, dari cache per rekaman (assert).",
    ]


def KESIMPULAN(R):
    I = R["icentia"]
    return [
        "<b>Kesimpulan.</b> S1 terkonfirmasi sebagai hipotesis primer di rekaman wearable kontinu: Se episode AF ≥ 30 s "
        f"{fnum(I['AF'][S1]['se_episode_30'])}, alarm palsu {fnum(I['non-AF'][S1]['palsu_per_jam'], 2)} per jam, lebih baik dari v3.1 + 30 s di "
        "keduanya. Bersama v16.2–v16.6, S1 kini didukung oleh data Holter (CPSC 2021, MIT-BIH, LTAFDB, SHDB-AF, NSRDB, EDB, afpdb, incartdb), "
        "latihan fisik, dan wearable (Icentia11k).",
        "<b>Sistem terbaik saat ini.</b> Deteksi AF 2 kelas (episode): S1 = D-1 + aturan episode AF ≥ 30 s. Klasifikasi 3 kelas (AFIB vs AFL): "
        "tetap paket v3.1; belum ada sistem yang lebih baik. Kelemahan S1 yang tersisa: derau berat (SNR ≤ 6 dB), EKG pendek sesaat (10–60 s), "
        "dan tidak membedakan AFIB dari AFL.",
        "<b>Langkah berikutnya.</b> (1) Gerbang kualitas sinyal untuk derau berat, dirancang dan diuji dengan rekaman derau yang belum dipakai "
        "(atau rekaman perangkat). (2) Uji S1 pada rekaman perangkat SIHEDAF berlabel. (3) Mengemas S1 sebagai paket uji perangkat (versi minor) "
        "perlu persetujuan pemilik repo.",
        "<b>Batasan.</b> Pasien AF Icentia sama dengan pasien di seleksi v10, tetapi segmennya belum pernah diproses. Label Icentia dibuat "
        "tim teknisi Icentia, bukan ditinjau dokter untuk studi ini; 19% window tidak berlabel ritme dan dibuang. nstdb hanya 2 rekaman per SNR. Icentia berlisensi "
        "nonkomersial: hanya untuk evaluasi, tidak untuk melatih model produk.",
        "<b>Reproduksi.</b> Dari folder v16.7_konfirmasi_wearable: python -c \"import konfirmasi as K; K.unduh_icentia()\"; dari root repo "
        "python tools/unduh_physionet.py nstdb 1.0.0 --ke nstdb --sumber physionet --pola '*.dat' '*.hea' '*.atr' RECORDS; lalu python "
        "tools/run_notebook.py v16.7_konfirmasi_wearable/konfirmasi.ipynb (pemrosesan ±5 menit dengan 2 proses lewat K.jalankan). Lalu "
        "python laporan/build_report.py --extract dan python laporan/build_report.py dari folder versi.",
    ]


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
