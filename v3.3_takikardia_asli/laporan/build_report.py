"""Laporan PDF v3.3: sinus takikardia asli, aturan produk, abstain conformal, metrik episode.

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v3_3/ringkasan.json (sel akhir takikardia_asli.ipynb), hitung ulang alarm v3.1 ≥ 120 bpm dari
     prediksi di cache dan metrik episode AFL v3.1 dari OOF, cocokkan (assert), gambar grafik → laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_v3_3_takikardia_asli.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v3_3_takikardia_asli.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, Laporan, fnum, pct, sg, sumbu_koma, thou  # noqa: E402

TANGGAL = "6 November 2026"
PILIH = ["v31", "v7_130", "v7_160", "v8_8c", "v11_bigru_wp100"]
SINGKAT = {"v7_130": "v7a (ST sintetis 100–130)", "v7_160": "v7b (ST sintetis 100–160)", "v8_8a": "v8a (AFDB + Chapman)",
           "v8_8c": "v8c (AFDB + Chapman)", "v11_bigru": "v11 BiGRU", "v11_bigru_wp100": "v11 BiGRU bobot penuh",
           "v11_tcn_wp100": "v11 TCN bobot penuh", "v1": "v1", "v2_prod": "v2"}
GAYA = {"v31": (INK, "o", "-"), "v7_130": (BIRU, "s", "-"), "v7_160": (BIRU, "^", "--"), "v8_8c": (ORANYE, "D", "-"),
        "v11_bigru_wp100": (HIJAU, "v", ":")}


def extract():
    import numpy as np
    import takikardia_asli as K
    R = json.loads((REPO / "cache/v3_3/ringkasan.json").read_text(encoding="utf-8"))
    J = K.muat_jendela()
    ok = J["ok"]
    z = np.load(K.CACHE / "prediksi.npz")
    m = (J["mode"][ok] == "perangkat") & (J["hr"][ok] >= 120)
    t, _, _ = K._boot_subjek(J["subjek"][ok][m], (z["v31"][:, m] != 0).mean(0))
    assert abs(t - R["utama"]["v31"]["alarm"]) < 1e-12
    d = K.T.load()
    pred = np.stack([K.cv.crossfit_bias(d["y"], d["probs"]["v31"][r], d["folds"][r])[0].argmax(1) for r in range(5)])
    zz = np.load(REPO / "cache/dataset_v2_detector.npz")
    se = np.mean([K.metrik_episode(zz["records"].astype(str), zz["t_start"], d["y"], pred[r], 2)["se_episode"] for r in range(5)])
    assert abs(se - R["episode_v31"]["AFL"]["se_episode"]) < 1e-12
    TMP.mkdir(exist_ok=True)
    (TMP / "v33.json").write_text(json.dumps(R, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    grafik(R)
    print("extract selesai; assert alarm v3.1 ≥ 120 bpm dan Se episode AFL v3.1 cocok dengan ringkasan.json")


def grafik(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from report_fonts import font_path
    font_manager.fontManager.addfont(font_path("regular"))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path("regular")).get_name(), "font.size": 9.5,
                         "axes.edgecolor": "#B8C4D0", "xtick.color": GRAY, "ytick.color": GRAY})
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.9), dpi=200, sharey=True)
    for ax, s in zip(axs, ("EPFL", "Wrist", "EPHNOGRAM")):
        A = R["alarm_pita"][f"{s}/perangkat"]
        pita = [b for b in A["v31"] if not b.startswith("0–") and A["v31"][b]["n"] >= 5]
        for k in PILIH:
            col, mk, ls = GAYA[k]
            ax.plot(range(len(pita)), [A[k][b]["alarm"] * 100 for b in pita], ls=ls, marker=mk, color=col, lw=1.6, ms=5,
                    label=R["label"][k])
        ax.set_xticks(range(len(pita)), [b.replace(" bpm", "") for b in pita], rotation=45, fontsize=7.5)
        ax.set_title(s, fontsize=9.5, color=INK); ax.set_ylim(-3, 103); ax.grid(axis="y", color="#E6ECF1")
        ax.spines[["top", "right"]].set_visible(False); sumbu_koma(ax)
    axs[0].set_ylabel("% window ditebak AFIB/AFL")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7.2)
    fig.tight_layout(rect=(0, .17, 1, 1)); fig.savefig(TMP / "alarm_pita.png"); plt.close(fig)


def build():
    R = json.loads((TMP / "v33.json").read_text(encoding="utf-8"))
    U, lab = R["utama"], R["label"]
    L = Laporan(OUT, "v3.3: sinus takikardia asli", "SIHEDAF  /  V3.3  /  TAKIKARDIA ASLI",
                "v3.3_takikardia_asli/takikardia_asli.ipynb, cache/v3_3/", TANGGAL)
    terbaik = min((k for k in U if k != "v31"), key=lambda k: U[k]["alarm"])
    ab = R["abstain"]["kosong_ragu=False"]["asli_≥120"]
    ar = R["aturan_asli"]
    kena = [v["kena_aturan"] for k, v in ar.items() if k.endswith("perangkat")]
    y = L.page("Ringkasan: alarm palsu pada sinus takikardia asli", "Evaluasi saja; paket produksi tidak disentuh")
    y = L.kartu([(pct(U["v31"]["alarm"], 0), "alarm v3.1, sinus ≥ 120 bpm", f"{thou(U['v31']['n'])} window, {U['v31']['subjek']} subjek"),
                 (pct(U[terbaik]["alarm"], 0), "model terbaik", SINGKAT[terbaik]),
                 (f"{pct(min(kena), 0)}–{pct(max(kena), 0)}", "ditangkap aturan produk", "≥ 110 bpm & CV IBI ≤ 0,05"),
                 (pct(ab["tidak_yakin"], 0), "jadi \"tidak yakin\" (conformal)", "model yakin dalam kesalahannya")], y)
    y = L.para("<b>Jawaban singkat.</b> Pada EKG latihan fisik asli (EPFL, Wrist, EPHNOGRAM; 3 sumber, subjek sehat), paket "
               f"produksi v3.1 menyebut AFIB/AFL pada <b>{pct(U['v31']['alarm'], 1)}</b> window sinus ≥ 120 bpm "
               f"[{pct(U['v31']['ci'][0], 0)}–{pct(U['v31']['ci'][1], 0)}]. Temuan v3.2 dari data sintetis terkonfirmasi di data asli. "
               "Abstain conformal tidak menolong karena model yakin; aturan produk menangkap sebagian besar, dengan harga 86% AFL "
               "terdeteksi di AFDB ikut dialihkan (v3.2).", y)
    y = L.para("<b>Temuan.</b> (1) Model yang pernah dilatih dengan sinus takikardia (v7 sintetis, v8 Chapman asli) menurunkan "
               f"alarm, terbaik {lab[terbaik]}: {pct(U[terbaik]['alarm'], 1)} (Δ {fnum(U[terbaik]['banding']['delta'] * 100, 1)} poin). "
               "Tetapi v7/v8 punya trade-off AFL 2:1 di AFDB (laporan masing-masing). (2) Detektor R-peak 50 Hz sendiri bermasalah saat "
               "latihan: di EPFL hanya ±68% window lolos cek kualitas, CV IBI naik dibanding R-peak manual, dan akibatnya v3.1 juga "
               f"mengalarmkan {pct(R['alarm_pita']['EPHNOGRAM/perangkat']['v31']['0–100 bpm']['alarm'], 0)} window di bawah 100 bpm saat "
               f"beraktivitas (Wrist: {pct(R['alarm_pita']['Wrist/perangkat']['v31']['0–100 bpm']['alarm'], 0)} lewat detektor vs "
               f"{pct(R['alarm_pita']['Wrist/manual']['v31']['0–100 bpm']['alarm'], 0)} dengan R-peak manual). (3) Metrik episode v3.1 "
               f"di AFDB: Se episode AFL {fnum(R['episode_v31']['AFL']['se_episode'], 2)}, episode AFIB palsu "
               f"{fnum(R['episode_v31']['AFIB']['palsu_per_jam'], 1)}/jam (gaya EC57, dipakai v13–v15).", y)
    y = L.page("Perbandingan model", "Window ≥ 120 bpm, mode perangkat, ketiga sumber; CI dan P dari bootstrap subjek berpasangan")
    rows = []
    for k, v in U.items():
        b = v["banding"]
        rows.append([lab[k], f"{pct(v['alarm'], 1)} [{pct(v['ci'][0], 0)}–{pct(v['ci'][1], 0)}]",
                     "–" if b is None else f"{sg(b['delta'] * 100, 1)} poin", "–" if b is None else fnum(b["p_kurang"])])
    y = L.table(["Model", "Alarm AFIB/AFL [CI 95%]", "Δ vs v3.1", "P(lebih sedikit)"], rows, y,
                widths=[L.CW * .44, L.CW * .24, L.CW * .16, L.CW * .16], sorot=(3,))
    y = L.para("v3.2 (sintetis): ambang Normal → AFIB/AFL v3.1 115 bpm, v1 dan v2 lebih rendah. Model fold (v7, v8, v11) memakai bias "
               "cross-fit fold masing-masing; angka adalah rata-rata 25 model fold.", y, "small")
    rows = [[k, thou(v["subjek"]), thou(v["window"]), thou(v["lolos cek"]), fnum(v["% lolos"], 1), thou(v["detak median"]),
             thou(v["detak maks"]), fnum(v["CV median"], 3), thou(v["≥ 120 bpm"])] for k, v in R["data"].items()]
    y = L.table(["Sumber / mode", "Subjek", "Window", "Lolos cek", "% lolos", "Detak med.", "Detak maks", "CV IBI med.", "≥ 120 bpm"], rows, y,
                widths=[L.CW * .22] + [L.CW * .0975] * 8)
    y = L.page("Metode", "Pipeline perangkat dan R-peak manual; model dengan bobot tersimpan")
    for t in ("<b>Data latihan fisik (tanpa label ritme; subjek sehat, diasumsikan sinus).</b> EPFL high-intensity (Zenodo, CC BY 4.0): "
              "20 subjek, 99 segmen 20 s saat uji bersepeda maksimal, 250 Hz, R-peak manual direview dokter. Wrist (PhysioNet, ODC-BY): 8 "
              "subjek, EKG dada 256 Hz saat jalan, lari, bersepeda, R-peak referensi. EPHNOGRAM (PhysioNet, ODC-BY): 24 subjek, 69 "
              "rekaman 8 kHz (istirahat, berjalan, treadmill Bruce, bersepeda).",
              "<b>Pipeline.</b> Mode perangkat: sinyal → 50 Hz → detektor R-peak produksi → window 10 s tidak tumpang tindih → cek "
              "kualitas yang sama dengan AFDB. Mode manual: R-peak referensi → window yang sama (logika window dibuktikan identik dengan "
              "hrv_core.signal_to_windows).",
              "<b>Model.</b> Paket v1, v2, v3.1, dan model fold yang menyimpan bobot: v7a/v7b (sinus takikardia sintetis), v8a/v8c "
              "(Chapman–Shaoxing termasuk sinus takikardia asli), v11 BiGRU dan TCN.",
              "<b>Aturan produk</b> (v3.2): detak ≥ 110 bpm dan CV IBI ≤ 0,05 → keluaran \"takikardia teratur, perlu dicek\", bukan alarm.",
              "<b>Abstain conformal.</b> LAC split-conformal, cakupan 90%, ambang dipelajari cross-fit di AFDB (paket: dari OOF dengan bias "
              "paket). Himpunan prediksi yang memuat Normal dan kelas lain → \"tidak yakin\"; varian kedua juga menganggap himpunan kosong "
              "sebagai tidak yakin.",
              "<b>Metrik episode gaya EC57.</b> Episode = run window berurutan berkelas sama; tiap window mewakili 5 s di tengahnya "
              "(langkah AFDB 5 s). Se/+P episode dan durasi, episode palsu per jam."):
        y = L.para(t, y)
    y = L.page("Hasil per pita detak", "Mode perangkat; % window yang ditebak AFIB/AFL")
    y = L.chart(TMP / "alarm_pita.png", y, 300)
    y = L.para("Pita dengan < 5 window tidak digambar. Tabel lengkap per model, sumber, dan mode (perangkat/manual) ada di notebook.",
               y, "small")
    y = L.page("Aturan produk dan abstain", "Apa yang terjadi bila keluaran produk diubah tanpa melatih ulang")
    rows = [[k, thou(v["n"]), pct(v["alarm_paket"], 0), pct(v["kena_aturan"], 0), pct(v["alarm_sisa"], 0)] for k, v in ar.items()]
    y = L.table(["Sumber / mode (≥ 120 bpm)", "Window", "Alarm v3.1", "Kena aturan", "Alarm tersisa"], rows, y)
    af = R["aturan_afdb"]
    y = L.para(f"Biaya di AFDB (v3.1): {pct(af['AFL terdeteksi dialihkan'], 0)} AFL yang terdeteksi ikut dialihkan, "
               f"{pct(af['alarm benar dialihkan'], 1)} dari semua alarm benar, {pct(af['alarm palsu dialihkan'], 1)} dari alarm palsu.", y)
    rows = []
    for kk, nm in (("kosong_ragu=False", "himpunan Normal + kelas lain"), ("kosong_ragu=True", "+ himpunan kosong")):
        a = R["abstain"][kk]
        rows.append([nm, fnum(a["ambang_median"], 3), pct(a["afdb"]["Normal"][3], 1), pct(a["afdb"]["AFL"][3], 1),
                     pct(a["asli_≥120"]["alarm"], 1), pct(a["asli_≥120"]["tidak_yakin"], 1)])
    y = L.table(["Abstain: \"tidak yakin\" bila", "Ambang", "AFDB Normal ragu", "AFDB AFL ragu", "Asli ≥ 120: alarm", "Asli: ragu"], rows, y)
    y = L.para("Jaminan cakupan conformal hanya berlaku bila data uji mirip data kalibrasi. Sinus takikardia tidak ada di AFDB, dan model "
               "memberi probabilitas AFIB/AFL tinggi, jadi abstain tidak mengubah apa pun.", y, "small")
    y = L.section("Metrik episode v3.1 di AFDB", y)
    E = R["episode_v31"]
    rows = [[n, fnum(E[n]["se_episode"], 3), fnum(E[n]["ppv_episode"], 3), fnum(E[n]["se_durasi"], 3), fnum(E[n]["ppv_durasi"], 3),
             thou(E[n]["n_ref"]), fnum(E[n]["palsu_per_jam"], 2)] for n in ("AFIB", "AFL")]
    y = L.table(["Kelas", "Se episode", "+P episode", "Se durasi", "+P durasi", "Episode ref.", "Palsu/jam"], rows, y)
    y = L.page("Kesimpulan, batasan, dan reproduksi", "")
    for t in ("<b>Kesimpulan.</b> Sinus takikardia asli dialarmkan hampir selalu oleh model produksi. Perbaikan nyata hanya datang dari "
              "data latih yang memuat sinus takikardia (v7, v8), dengan trade-off AFL 2:1. Abstain conformal tidak berguna untuk masalah "
              "ini; aturan produk menolong tetapi mengorbankan AFL 2:1. Ini memperkuat kebutuhan rekaman perangkat berlabel saat aktivitas.",
              "<b>Batasan.</b> Data latihan fisik tidak punya label ritme (subjek sehat, diasumsikan sinus); EPFL hanya 20 s per segmen; "
              "subjek muda dan bugar. Detektor R-peak menjadi sumber kesalahan tambahan saat latihan.",
              "<b>Reproduksi.</b> Unduh: <font face='Arial-Bold'>tools/unduh_berkas.py</font> (EPFL, md5 Zenodo) dan "
              "<font face='Arial-Bold'>tools/unduh_physionet.py wrist 1.0.0 / ephnogram 1.0.0</font>. Lalu "
              "<font face='Arial-Bold'>python tools/run_notebook.py v3.3_takikardia_asli/takikardia_asli.ipynb</font> dan build_report "
              "dua tahap. Butuh cache v3, v7, v8, v11 (bobot model fold)."):
        y = L.para(t, y)
    lk = R["lingkungan"]
    y = L.para(f"Lingkungan: commit {lk['commit']} (+ perubahan belum di-commit), {lk['gpu']}, Python {lk['pustaka']['python']}, "
               f"torch {lk['pustaka']['torch']}, numpy {lk['pustaka']['numpy']}.", y, "small")
    L.selesai()
    print("PDF:", OUT, L.page_no, "halaman")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
