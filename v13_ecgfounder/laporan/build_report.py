"""Laporan PDF v13: model fondasi EKG ECGFounder 1-lead.

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v13/ringkasan.json (sel akhir ecgfounder.ipynb), hitung ulang Macro-F1 v3.1 dan probe/fusi 50 Hz dari
     OOF di cache dan cocokkan (assert), tambah pembanding v1, gambar grafik → laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_v13_ecgfounder.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v13_ecgfounder.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V), str(REPO / "v11_model_tim")]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, LABEL_PALSU, Laporan, ci, daftar, fnum, pct, sg, sumbu_koma, thou, vonis  # noqa: E402

TANGGAL = "6 November 2026"
PASIEN_AFL = ["04043", "04908", "04936", "06426", "07910", "08215", "08378"]
NAMA = {"v1": "v1 (argmax)", "v31": "v3.1 (referensi)", "zero_50hz": "zero-shot 50 Hz", "probe_50hz": "probe 50 Hz",
        "fusi_50hz": "fusi probe + v3.1, 50 Hz", "zero_250hz": "zero-shot 250 Hz (kontrol)", "probe_250hz": "probe 250 Hz (kontrol)",
        "fusi_250hz": "fusi 250 Hz (kontrol)"}


def extract():
    import numpy as np
    import ecgfounder as E
    from bersama import data as D, evaluasi as Ev
    R = json.loads((REPO / "cache/v13/ringkasan.json").read_text(encoding="utf-8"))
    d = D.muat()
    p31, _ = Ev.keputusan(d.oof_v31, d.folds, d.y)
    pr = np.load(E.CACHE / "oof_probe_50hz.npz")["oof"]
    pp, _ = Ev.keputusan(pr, d.folds, d.y)
    pf, _ = Ev.keputusan(E.fusi(pr, d.oof_v31), d.folds, d.y)
    pv1 = d.oof_v1.argmax(-1)
    tab = Ev.tabel_metrik(d, {"v1": pv1, "v31": p31, "probe_50hz": pp, "fusi_50hz": pf})
    for k in ("v31", "probe_50hz", "fusi_50hz"):
        for m in ("Macro-F1", "AFL R"):
            assert abs(tab[k][m][0] - R["metrik"][k][m][0]) < 1e-9, (k, m)
    out = {"R": R, "v1": tab["v1"], "banding_v1": {k: Ev.banding(d, p, pv1) for k, p in (("probe_50hz", pp), ("fusi_50hz", pf))}}
    TMP.mkdir(exist_ok=True)
    (TMP / "v13.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    grafik(out)
    print("extract selesai; assert Macro-F1 dan AFL R (v3.1, probe 50 Hz, fusi 50 Hz) cocok dengan ringkasan.json")


def grafik(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from report_fonts import font_path
    font_manager.fontManager.addfont(font_path("regular"))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path("regular")).get_name(), "font.size": 9.5,
                         "axes.edgecolor": "#B8C4D0", "xtick.color": GRAY, "ytick.color": GRAY})
    R = out["R"]
    M = R["metrik"] | {"v1": out["v1"]}
    urut = ["v31", "v1", "zero_50hz", "probe_50hz", "fusi_50hz", "zero_250hz", "probe_250hz", "fusi_250hz"]
    gaya = {"v31": (HIJAU, "D"), "v1": (GRAY, "D")} | {k: (BIRU if "50hz" in k and "250" not in k else ORANYE, "o" if "250" not in k else "s") for k in urut[2:]}
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.3), dpi=200, sharey=True)
    for ax, m in zip(axs, ("Macro-F1", "AFL F1")):
        for i, k in enumerate(urut[::-1]):
            col, mk = gaya[k]
            v = M[k][m]
            ax.plot(v[1:], [i, i], color=col, lw=2)
            ax.plot(v[0], i, mk, color=col, ms=7, mec="white", mew=1.2)
        ax.axvline(M["v31"][m][0], color=HIJAU, ls="--", lw=1)
        ax.set_title(f"{m} (+ bias, CI 95%)", fontsize=9.5, color=INK); ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#E6ECF1"); sumbu_koma(ax, "x")
    axs[0].set_yticks(range(len(urut)), [NAMA[k] for k in urut[::-1]], fontsize=8)
    fig.tight_layout(); fig.savefig(TMP / "forest.png"); plt.close(fig)
    H = R["h2_latihan"]
    sumber = sorted({s for v in H.values() for s in v["per_sumber"]})
    fig, ax = plt.subplots(figsize=(7.6, 2.6), dpi=200)
    nm = list(H)
    w = .8 / len(nm)
    warna = [INK, BIRU, ORANYE, HIJAU]
    pola = ["", "//", "..", "xx"]
    for i, k in enumerate(nm):
        ax.bar([j + (i - (len(nm) - 1) / 2) * w for j in range(len(sumber))], [H[k]["per_sumber"].get(s, 0) * 100 for s in sumber], w,
               color=warna[i % 4], hatch=pola[i % 4], edgecolor="white", label=k)
    ax.set_xticks(range(len(sumber)), sumber); ax.set_ylabel("% alarm, sinus ≥ 120 bpm"); ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.18))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "latihan.png"); plt.close(fig)


def build():
    o = json.loads((TMP / "v13.json").read_text(encoding="utf-8"))
    R = o["R"]
    M = R["metrik"]
    B = R["banding_v31"]
    H = R["h2_latihan"]
    L = Laporan(OUT, "v13: ECGFounder 1-lead", "SIHEDAF  /  V13  /  ECGFOUNDER", "v13_ecgfounder/ecgfounder.ipynb, cache/v13/", TANGGAL)
    b50 = R["banding_50_250"]
    y = L.page("Ringkasan: model fondasi EKG pada sinyal 50 Hz", "Rancangan dikunci sebelum metrik dihitung (RANCANGAN.md)")
    pt, ic = R["ptbxl"]["zero-shot"], R["icentia"]["zero-shot"]
    afl = R["afl_per_pasien"]
    y = L.kartu([(pct(H["zero-shot"]["alarm"], 0), "alarm sinus asli ≥ 120 bpm", f"zero-shot; v3.1 paket {pct(H['v3.1 paket']['alarm'], 0)}"),
                 (f"{fnum(pt['titik_a'][0])}", "Macro-F1 PTB-XL (zero-shot)", f"v3.1 {fnum(pt['titik_b'][0])}; P = {fnum(pt['Macro-F1'][3])}"),
                 (fnum(M["fusi_50hz"]["Macro-F1"][0]), "Macro-F1 AFDB (fusi)", f"v3.1 {fnum(M['v31']['Macro-F1'][0])}; P = {fnum(B['fusi_50hz']['Macro-F1'][3])}"),
                 (fnum(afl["zero_50hz"]["08378"], 2), "recall AFL 2:1 (08378)", f"zero-shot; v3.1 {fnum(afl['v31']['08378'], 2)}")], y)
    y = L.para("<b>Jawaban singkat.</b> Model fondasi EKG yang dipakai beku <b>menyelesaikan alarm palsu sinus takikardia</b> yang tidak "
               f"bisa diselesaikan model RR: pada EKG latihan fisik asli (50 Hz, jalur perangkat) alarm turun dari "
               f"{pct(H['v3.1 paket']['alarm'], 1)} (v3.1) ke {pct(H['zero-shot']['alarm'], 1)} [{pct(H['zero-shot']['ci'][0], 0)}–"
               f"{pct(H['zero-shot']['ci'][1], 0)}] (zero-shot), dan lebih baik dari v3.1 di dua dataset eksternal berlabel (PTB-XL, "
               "Icentia11k). Harganya ada di AFDB: AFL 2:1 (pasien 08378) ikut disebut sinus dan banyak AFIB disebut AFL, sehingga Macro-F1 "
               f"3 kelas di AFDB tidak mengalahkan v3.1 (fusi {fnum(M['fusi_50hz']['Macro-F1'][0])}, P = {fnum(B['fusi_50hz']['Macro-F1'][3])}).", y)
    rows = [[NAMA[k], f"{fnum(M[k]['Macro-F1'][0])} {ci(*M[k]['Macro-F1'][1:])}", f"{sg(B[k]['Macro-F1'][0])}", fnum(B[k]["Macro-F1"][3]),
             vonis(B[k]["Macro-F1"][3])] for k in ("zero_50hz", "probe_50hz", "fusi_50hz")]
    y = L.table(["AFDB (jalur perangkat 50 Hz)", "Macro-F1 [CI]", "Δ vs v3.1", "P(Δ > 0)", "Vonis"], rows, y,
                widths=[L.CW * .32, L.CW * .22, L.CW * .14, L.CW * .14, L.CW * .18])
    y = L.para("<b>Temuan.</b> (1) Head asli ECGFounder memisahkan sinus takikardia dari AF berdasarkan bentuk gelombang, juga di 50 Hz: "
               f"PTB-XL STACH dialarmkan {pct(R['ptbxl_st']['v3.1 paket'], 1)} oleh v3.1 dan {pct(R['ptbxl_st']['zero-shot'], 1)} oleh zero-shot. "
               f"(2) Eksternal: PTB-XL recall AFL {fnum(pt['titik_b'][1], 2)} → {fnum(pt['titik_a'][1], 2)}; Icentia Macro-F1 "
               f"{fnum(ic['titik_b'][0])} → {fnum(ic['titik_a'][0])} (P = {fnum(ic['Macro-F1'][3])}). (3) Probe yang dilatih di 23 pasien "
               "AFDB kehilangan keunggulan itu di data eksternal (pelajaran v6/v9: kepala baru menghafal AFDB). (4) Tidak ada biaya 50 Hz yang "
               f"terlihat: probe 50 Hz − 250 Hz {sg(b50['probe']['Macro-F1'][0])} (P = {fnum(b50['probe']['Macro-F1'][3])}). "
               f"(5) Tanda bahaya: bias zero-shot menyentuh batas grid di {pct(R['bias']['zero_50hz']['di_batas'], 0)} fold.", y)
    y = L.page("Perbandingan dengan v3.1 dan v1 (AFDB)", "Pooled OOF 5 ulangan, CI 95% bootstrap pasien, bias cross-fit")
    y = L.chart(TMP / "forest.png", y, 230)
    rows = []
    for k in ("v1", "v31", "zero_50hz", "probe_50hz", "fusi_50hz", "zero_250hz", "probe_250hz", "fusi_250hz"):
        m = o["v1"] if k == "v1" else M[k]
        rows.append([NAMA[k], f"{fnum(m['Macro-F1'][0])} {ci(*m['Macro-F1'][1:])}", f"{fnum(m['AFL F1'][0])} {ci(*m['AFL F1'][1:])}",
                     fnum(m["AFL R"][0]), fnum(m["AFL P"][0]), fnum(m["Normal R"][0])])
    y = L.table(["Sistem", "Macro-F1 [CI]", "AFL F1 [CI]", "AFL R", "AFL P", "Normal R"], rows, y,
                widths=[L.CW * .28, L.CW * .19, L.CW * .19, L.CW * .11, L.CW * .11, L.CW * .12], sorot=(2,))
    rows = [[f"{NAMA[k]} vs v1", f"{sg(v['Macro-F1'][0])} [{sg(v['Macro-F1'][1])}…{sg(v['Macro-F1'][2])}]", fnum(v["Macro-F1"][3]),
             vonis(v["Macro-F1"][3])] for k, v in o["banding_v1"].items()]
    rows += [[f"{m}: 50 Hz − 250 Hz", f"{sg(v['Macro-F1'][0])} [{sg(v['Macro-F1'][1])}…{sg(v['Macro-F1'][2])}]", fnum(v["Macro-F1"][3]),
              vonis(v["Macro-F1"][3])] for m, v in b50.items()]
    y = L.table(["Perbandingan Macro-F1", "Δ [CI 95%]", "P(Δ > 0)", "Vonis"], rows, y, widths=[L.CW * .36, L.CW * .3, L.CW * .14, L.CW * .2])
    y = L.page("Metode", "Model beku, tiga cara pakai, dua jalur sinyal")
    for t in ("<b>Model.</b> ECGFounder 1-lead (PKUDigitalHealth, MIT): Net1D 30,8 juta parameter, pralatih 10,8 juta EKG dari 1,8 juta "
              "pasien (HEEDB), head 150 label termasuk SINUS TACHYCARDIA dan ATRIAL FLUTTER. Checkpoint dicek SHA256 (API Hugging Face) dan "
              "dimuat weights_only (hanya skalar numpy yang diizinkan).",
              "<b>Sinyal.</b> AFDB kanal 0, window detektor yang sama dengan semua versi. Jalur 50hz: sinyal → 50 Hz (fungsi produksi) "
              "→ 500 Hz. Jalur kontrol 250hz: sinyal asli → 500 Hz. Lalu filter ECGFounder (notch 50 Hz, Butterworth 0,67–40 Hz, median "
              "baseline 0,4 s) dan z-score.",
              "<b>Cara pakai.</b> Zero-shot: logit [maks(NSR, SR, sinus takikardia), AF, AFL]. Probe: regresi logistik multinomial "
              "(C = 1, bobot kelas lunak) pada fitur 1.024 dimensi, dilatih per fold resmi. Fusi: rata-rata log-probabilitas probe dan "
              "v3.1. Semuanya dengan bias cross-fit grid −7…+3.",
              "<b>Uji eksternal.</b> EKG latihan fisik v3.3 (mode perangkat, window ≥ 120 bpm); PTB-XL (lead I 100 → 50 Hz; sinus "
              "termasuk STACH, AFIB, AFLT); Icentia11k (window 50 Hz v10, akademik saja). Model eksternal dilatih di seluruh AFDB; bias = "
              "median bias cross-fit. Uji stres sintetis butir 9 tidak bisa dipakai untuk model bentuk gelombang; diganti EKG asli."):
        y = L.para(t, y)
    y = L.page("Hasil eksternal", "Sinus takikardia asli, PTB-XL, Icentia11k — berdampingan dengan v3.1 (pelajaran v9)")
    y = L.chart(TMP / "latihan.png", y, 190)
    rows = [[k, f"{pct(v['alarm'], 1)} [{pct(v['ci'][0], 0)}–{pct(v['ci'][1], 0)}]", thou(v["n"])] for k, v in H.items()]
    y = L.table(["EKG latihan fisik, sinus ≥ 120 bpm", "Alarm AFIB/AFL [CI subjek]", "Window"], rows, y,
                widths=[L.CW * .4, L.CW * .4, L.CW * .2])
    for nama_d, kunci in (("PTB-XL", "ptbxl"), ("Icentia11k", "icentia")):
        rows = [[k, f"{fnum(v['titik_a'][0])} / {fnum(v['titik_b'][0])}", f"{sg(v['Macro-F1'][0])} (P = {fnum(v['Macro-F1'][3])})",
                 f"{fnum(v['titik_a'][1], 2)} / {fnum(v['titik_b'][1], 2)}", f"{sg(v['AFL R'][0], 2)} (P = {fnum(v['AFL R'][3])})"]
                for k, v in R[kunci].items()]
        y = L.table([f"{nama_d} ({thou(R['n'][kunci])} window)", "Macro-F1 model / v3.1", "Δ Macro-F1", "Recall AFL model / v3.1", "Δ recall AFL"],
                    rows, y, widths=[L.CW * .2, L.CW * .2, L.CW * .2, L.CW * .2, L.CW * .2])
    st = R["ptbxl_st"]
    y = L.para(f"PTB-XL sinus takikardia (STACH, {thou(R['n']['ptbxl_st'])} rekaman): alarm AFIB/AFL "
               + ", ".join(f"{k} {pct(v, 1)}" for k, v in st.items()) + ".", y, "small")
    y = L.page("Per pasien AFL, bias, dan episode", "AFDB, + bias cross-fit, rata-rata 5 ulangan")
    ks = ["v31", "zero_50hz", "probe_50hz", "fusi_50hz", "probe_250hz"]
    rows = [[p, *[fnum(R["afl_per_pasien"][k][p], 2) for k in ks]] for p in PASIEN_AFL]
    y = L.table(["Pasien", *[NAMA[k] for k in ks]], rows, y)
    rows = [[NAMA[k], daftar(R["bias"][k]["median"]), fnum(R["bias"][k]["di_batas"], 2)] for k in ks]
    y = L.table(["Sistem", "Bias median [Normal; AFIB; AFL]", "Fraksi di batas grid"], rows, y, widths=[L.CW * .4, L.CW * .4, L.CW * .2])
    E = R["episode"]
    rows = [[f"{NAMA[k.rsplit(' ', 1)[0]]}: {k.rsplit(' ', 1)[1]}", fnum(v["se_episode"], 3), fnum(v["ppv_episode"], 3),
             fnum(v["palsu_per_jam"], 2)] for k, v in E.items()]
    y = L.table(["Episode (gaya EC57)", "Se episode", "+P episode", "Palsu per jam"], rows, y)
    T = R["tanpa_04936"]["rows"]
    y = L.para("Tanpa pasien 04936 (protokol butir 10), Macro-F1: " + ", ".join(
        f"{NAMA.get(k, k)} {fnum(v['Macro-F1'][0])} {ci(*v['Macro-F1'][1:])}" for k, v in T.items()) + ".", y, "small")
    y = L.page("Kesimpulan, batasan, dan reproduksi", "")
    y = L.para("<b>Kesimpulan.</b> Pengetahuan pralatih dari 1,8 juta pasien adalah informasi baru pertama yang menurunkan alarm sinus "
               "takikardia secara besar tanpa melatih dengan data takikardia, dan lebih tahan antar-dataset daripada model RR. Tetapi AFL 2:1 "
               "di ±150 bpm, yang memang mirip sinus takikardia, ikut hilang. Bila keputusan produk cukup 2 kelas (Normal vs AF/AFL) atau AFL "
               "2:1 bisa ditangani terpisah, ECGFounder zero-shot di backend layak diuji pada rekaman perangkat. Bukan pengganti paket v3.1 "
               "untuk 3 kelas.", y)
    y = L.para("<b>Langkah berikutnya.</b> (1) Uji di rekaman perangkat SIHEDAF berlabel (terutama aktivitas dan AFL). (2) Rancangan baru "
               "yang dikunci: gabungkan zero-shot ECGFounder untuk memutuskan sinus vs AF, dan model RR atau fitur atrium (v15) hanya untuk "
               "AFIB vs AFL. (3) Evaluasi 2 kelas dengan protokol v11.1.", y)
    for t in ("<b>Batasan.</b> AFDB: 23 pasien, 7 AFL. Kanal AFDB bukan lead I. Sinyal 50 Hz kosong di atas 25 Hz, padahal model "
              "dilatih dengan pita sampai 40 Hz. Model 370 MB hanya cocok di backend, bukan di perangkat. FOUND-AF (benchmark luar) "
              "menggabungkan AF dan AFL, jadi angkanya tidak bisa dibandingkan langsung.",
              "<b>Reproduksi.</b> Unduh bobot dan kode: <font face='Arial-Bold'>tools/unduh_berkas.py</font> (SHA256 dari API Hugging "
              "Face; kode dari GitHub PKUDigitalHealth/ECGFounder cabang master, MIT). Lalu "
              "<font face='Arial-Bold'>python tools/run_notebook.py v13_ecgfounder/ecgfounder.ipynb</font> (GPU) dan build_report dua "
              "tahap. Data eksternal: PTB-XL (tools/unduh_physionet.py), cache v3.3 dan v10."):
        y = L.para(t, y)
    lk = R["lingkungan"]
    y = L.para(f"Lingkungan: commit {lk['commit']} (+ perubahan belum di-commit), {lk['gpu']}, Python {lk['pustaka']['python']}, torch "
               f"{lk['pustaka']['torch']}, scikit-learn {lk['pustaka']['scikit-learn']}.", y, "small")
    L.selesai()
    print("PDF:", OUT, L.page_no, "halaman")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
