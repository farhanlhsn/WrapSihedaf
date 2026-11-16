"""Laporan PDF v16: hierarki ECGFounder (sinus vs AF) lalu RR / fitur atrium (AFIB vs AFL).

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v16/ringkasan.json (sel akhir hierarki.ipynb), hitung ulang Macro-F1 AFDB H-0 dan alarm sinus
     takikardia H-0 dari cache lalu cocokkan (assert), tambah perbandingan dengan v1, gambar grafik → laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_v16_hierarki_ecgfounder.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v16_hierarki_ecgfounder.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, LABEL_PALSU, Laporan, ci, daftar, fnum, pct, sg, sumbu_koma, thou, vonis  # noqa: E402,F401

TANGGAL = "6 November 2026"
PASIEN_AFL = ["04043", "04908", "04936", "06426", "07910", "08215", "08378"]
SISTEM = ("v1 (argmax)", "v3.1", "v13 zero-shot", "v15 gabungan", "H-0", "H-A", "H-X")
NAMA = {"v1 (argmax)": "v1 (argmax)", "v3.1": "v3.1 (referensi)", "v13 zero-shot": "v13 ECGFounder zero-shot",
        "v15 gabungan": "v15 gabungan", "H-0": "H-0 (primer)", "H-A": "H-A (tahap 2 atrium)", "H-X": "H-X (tahap 1 disetel)"}


def extract():
    import numpy as np
    import hierarki as H
    from hierarki import D, E13, Ev, K33
    R = json.loads((REPO / "cache/v16/ringkasan.json").read_text(encoding="utf-8"))
    d = D.muat()
    lo, _ = E13.muat_afdb("50hz")
    pE, d1 = H.tahap1(lo)
    ph0, _ = H.keputusan_h(d.y, d1, H.d2_rr(d.oof_v31), d.folds)
    pv1 = d.oof_v1.argmax(-1)
    tab = Ev.tabel_metrik(d, {"H-0": ph0})
    assert abs(tab["H-0"]["Macro-F1"][0] - R["metrik"]["H-0"]["Macro-F1"][0]) < 1e-9
    L = H.latihan()
    kl, _ = H.keputusan_eksternal(L["logit"], L["p31"], R["b2_median"]["H-0"], np.array(R["bias_hx_median"]))
    m = L["hr"] >= 120
    t, _, _ = K33._boot_subjek(L["subjek"][m], (kl["H-0"][m] != 0).astype(float))
    assert abs(t - R["h1_latihan"]["H-0"]["alarm"]) < 1e-12
    p31, _ = Ev.keputusan(d.oof_v31, d.folds, d.y)
    assert abs(Ev.tabel_metrik(d, {"v3.1": p31})["v3.1"]["Macro-F1"][0] - R["metrik"]["v3.1"]["Macro-F1"][0]) < 1e-9
    alarm_afl_v31 = {p: float((p31[:, (d.records == p) & (d.y == 2)] != 0).mean()) for p in PASIEN_AFL}
    alarm_afl_v31["semua"] = float((p31[:, d.y == 2] != 0).mean())
    out = {"R": R, "banding_v1": {"H-0": Ev.banding(d, ph0, pv1)}, "alarm_afl_v31": alarm_afl_v31}
    TMP.mkdir(exist_ok=True)
    (TMP / "v16.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    grafik(out)
    print("extract selesai; assert Macro-F1 AFDB H-0 dan alarm sinus takikardia H-0 cocok dengan ringkasan.json")


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


def grafik(out):
    plt = _gaya()
    R = out["R"]
    # 1. Macro-F1 AFDB dengan CI
    fig, ax = plt.subplots(figsize=(7.6, 2.9), dpi=200)
    gaya = {"v3.1": (BIRU, "o"), "H-0": (ORANYE, "s"), "H-A": (ORANYE, "D"), "H-X": (ORANYE, "^")}
    for i, k in enumerate(SISTEM[::-1]):
        v = R["metrik"][k]["Macro-F1"]
        col, mk = gaya.get(k, (GRAY, "o"))
        ax.plot(v[1:], [i, i], color=col, lw=2.2)
        ax.plot(v[0], i, mk, color=col, ms=8, mec="white", mew=1.5)
        ax.text(v[0], i + .28, fnum(v[0]), ha="center", va="bottom", fontsize=8, color=INK)
    ax.axvline(R["metrik"]["v3.1"]["Macro-F1"][0], color=BIRU, lw=1, ls=":")
    ax.set_ylim(-.5, len(SISTEM) - .2)
    ax.set_yticks(range(len(SISTEM)), [NAMA[k] for k in SISTEM[::-1]], fontsize=8.5)
    ax.set_xlabel("Macro-F1 AFDB 3 kelas (pooled OOF, CI 95% bootstrap pasien); garis titik = v3.1")
    ax.spines[["top", "right"]].set_visible(False); sumbu_koma(ax, "x"); ax.grid(axis="x", color="#E6ECF1")
    fig.tight_layout(); fig.savefig(TMP / "macro_f1.png"); plt.close(fig)
    # 2. alarm per pita detak, latihan fisik
    h1 = R["h1_latihan"]
    pita = list(h1["v3.1 paket"]["pita"])
    fig, ax = plt.subplots(figsize=(7.6, 2.7), dpi=200)
    for k, col, mk, ls in (("v3.1 paket", BIRU, "o", "-"), ("H-0", ORANYE, "s", "-"), ("H-X", HIJAU, "^", "--")):
        ys = [h1[k]["pita"][p]["alarm"] * 100 for p in pita]
        lo = [h1[k]["pita"][p]["ci"][0] * 100 for p in pita]
        hi = [h1[k]["pita"][p]["ci"][1] * 100 for p in pita]
        ax.fill_between(range(len(pita)), lo, hi, color=col, alpha=.12, lw=0)
        ax.plot(range(len(pita)), ys, ls, marker=mk, color=col, ms=7, mec="white", mew=1.2, lw=2, label=k if k != "H-0" else "H-0 dan H-A")
    ax.set_xticks(range(len(pita)), [p.replace("–∞", "+") for p in pita], fontsize=8.5)
    ax.set_ylabel("alarm AFIB/AFL (%)"); ax.set_ylim(0, 105); ax.set_xlabel("detak window (latihan fisik asli, mode perangkat 50 Hz)")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "alarm_pita.png"); plt.close(fig)
    # 3. AFL per pasien: v3.1 (recall AFL, alarm AF apa pun) vs H-0 (alarm AF + 'perlu dicek' bertumpuk)
    afl, h5, a31 = R["afl_per_pasien"], R["h5"]["pasien_afl"], out["alarm_afl_v31"]
    fig, ax = plt.subplots(figsize=(7.6, 2.8), dpi=200)
    x = range(len(PASIEN_AFL))
    ax.bar([i - .2 for i in x], [a31[p] for p in PASIEN_AFL], .38, color="#BFD6F2", label="v3.1: alarm AFIB/AFL")
    ax.bar([i - .2 for i in x], [afl["v3.1"][p] for p in PASIEN_AFL], .38, color=BIRU, label="v3.1: ditebak AFL")
    ax.bar([i + .2 for i in x], [h5[p]["alarm"] for p in PASIEN_AFL], .38, color=ORANYE, label="H-0: alarm AFIB/AFL")
    ax.bar([i + .2 for i in x], [h5[p]["cek"] for p in PASIEN_AFL], .38, bottom=[h5[p]["alarm"] for p in PASIEN_AFL],
           color="white", edgecolor=ORANYE, hatch="///", lw=0, label="H-0: 'perlu dicek' (H5)")
    ax.set_xticks(list(x), [f"{p}\n(n={h5[p]['n']})" for p in PASIEN_AFL], fontsize=8.5)
    ax.set_ylabel("proporsi window AFL"); ax.set_ylim(0, 1.25); ax.set_yticks([0, .25, .5, .75, 1])
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center"); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "afl_pasien.png"); plt.close(fig)


def _d(v, nd=3):
    """Δ [CI] dari [Δ, bawah, atas, ...]."""
    return f"{sg(v[0], nd)} [{sg(v[1], nd)}…{sg(v[2], nd)}]"


def build():
    o = json.loads((TMP / "v16.json").read_text(encoding="utf-8"))
    R = o["R"]
    M, BD, T1 = R["metrik"], R["banding_v31"], R["tahap1_afdb"]
    H1, DK, EP, H5 = R["h1_latihan"], R["dua_kelas"], R["episode"], R["h5"]
    L = Laporan(OUT, "v16: hierarki ECGFounder lalu RR / fitur atrium", "SIHEDAF  /  V16  /  HIERARKI ECGFOUNDER",
                "v16_hierarki_ecgfounder/hierarki.ipynb, cache/v16/", TANGGAL)
    # ── 1. ringkasan ──
    y = L.page("Ringkasan: alarm palsu hilang, AFL ikut hilang", "Rancangan dikunci 6 November 2026 sebelum metrik dihitung")
    y = L.kartu([(pct(H1["H-0"]["alarm"]), "alarm sinus takikardia", f"≥ 120 bpm; v3.1 {pct(H1['v3.1 paket']['alarm'])}"),
                 (fnum(M["H-0"]["Macro-F1"][0]), "Macro-F1 AFDB 3 kelas", f"v3.1 {fnum(M['v3.1']['Macro-F1'][0])}; P = {fnum(BD['H-0']['Macro-F1'][3])}"),
                 (fnum(DK["H-0"]["Macro-F1"][0]), "Macro-F1 AFDB 2 kelas", f"v3.1 {fnum(DK['v3.1']['Macro-F1'][0])}; P = {fnum(DK['banding']['Macro-F1'][3], 2)}"),
                 (pct(T1["se_afl"], 0), "AFL lolos tahap 1", f"AFIB {pct(T1['se_afib'], 0)}; Normal benar {pct(T1['sp'])}")], y)
    y = L.para("<b>Pertanyaan.</b> ECGFounder (v13) mengenali sinus takikardia tetapi mencampur AFIB dan AFL; model RR (v3.1) memisahkan "
               "AFIB dari AFL tetapi menyebut sinus takikardia sebagai AF. Apakah pembagian tugas, yaitu ECGFounder untuk <i>sinus vs AF</i> "
               "lalu RR atau fitur atrium untuk <i>AFIB vs AFL</i>, mempertahankan keunggulan keduanya?", y)
    y = L.para("<b>Jawaban singkat: tidak untuk 3 kelas.</b> Tahap 1 hampir menghapus alarm palsu pada sinus takikardia asli "
               f"({pct(H1['v3.1 paket']['alarm'])} → {pct(H1['H-0']['alarm'])}) dan episode AFIB palsu "
               f"({fnum(EP['v3.1']['AFIB']['palsu_per_jam'], 1)} → {fnum(EP['H-0']['AFIB']['palsu_per_jam'], 1)} per jam). Tetapi ECGFounder "
               f"membaca {pct(1 - T1['se_afl'], 0)} window AFL AFDB sebagai sinus, sehingga AFL sudah hilang sebelum tahap 2 bekerja. "
               f"Macro-F1 3 kelas H-0 {fnum(M['H-0']['Macro-F1'][0])} vs v3.1 {fnum(M['v3.1']['Macro-F1'][0])} (lebih buruk). Kriteria yang "
               "dikunci: H1 lolos, H2 gagal, H4 lolos di PTB-XL tetapi gagal di Icentia, jadi v16 <b>bukan kandidat</b> sistem 3 kelas.", y)
    y = L.para("<b>Temuan.</b> (1) Untuk 2 kelas (AF ya/tidak) H-0 setara v3.1 "
               f"(Δ {sg(DK['banding']['Macro-F1'][0])}, P = {fnum(DK['banding']['Macro-F1'][3], 2)}) dengan spesifisitas "
               f"{fnum(DK['H-0']['Sp'][0])} vs {fnum(DK['v3.1']['Sp'][0])}, tetapi sensitivitas {fnum(DK['H-0']['Se'][0])} vs "
               f"{fnum(DK['v3.1']['Se'][0])}. (2) Tahap 2 fitur atrium (H-A) tidak berbeda dari RR, karena AFL sudah dibuang di tahap 1. "
               f"(3) Menyetel tahap 1 di AFDB (H-X) memulihkan sebagian AFL, tetapi {pct(R['parameter']['H-X']['di_batas'], 0)} nilai bias "
               f"menempel di batas grid, AFDB tetap lebih buruk, dan alarm sinus takikardia naik ke {pct(H1['H-X']['alarm'])}. (4) Keluaran "
               f"keempat \"takikardia teratur, perlu dicek\" menampung {pct(H5['pasien_afl']['08378']['cek'], 0)} AFL 2:1 (08378), tetapi juga "
               f"{pct(H5['latihan_≥120']['cek'], 0)} sinus takikardia asli: keduanya tetap tidak terpisah.", y)
    kr = R["kriteria"]
    rows = [["H1 sinus takikardia asli", "lolos" if kr["H1"] else "gagal"], ["H2 AFDB 3 kelas (bukan lebih buruk)", "lolos" if kr["H2"] else "gagal"],
            ["H4 PTB-XL (bukan lebih buruk)", "lolos" if kr["H4_ptbxl"] else "gagal"],
            ["H4 Icentia11k (bukan lebih buruk)", "lolos" if kr["H4_icentia"] else "gagal"],
            ["Kandidat uji perangkat (3 kelas)", "ya" if kr["kandidat_uji_perangkat"] else "tidak"],
            ["H2b AFDB 2 kelas (di luar kriteria)", f"{vonis(DK['banding']['Macro-F1'][3])} (P = {fnum(DK['banding']['Macro-F1'][3], 2)})"],
            ["H3 tahap 2 atrium vs RR (di luar kriteria)", f"{vonis(BD['H-A vs H-0']['Macro-F1'][3])} (AFDB), "
             f"{vonis(R['ptbxl']['H-A vs H-0']['Macro-F1'][3])} (PTB-XL)"]]
    y = L.table(["Kriteria yang dikunci (H-0)", "Hasil"], rows, y, widths=[L.CW * .6, L.CW * .4], sorot=(5,))
    y = L.para("<b>Yang tidak bisa disimpulkan.</b> Semua uji memakai data publik. Data latihan fisik tidak berlabel ritme (diasumsikan "
               "sinus), dan AFL AFDB berasal dari 7 pasien (58% dari 04936). Apakah AFL boleh dilaporkan lewat jalur \"perlu dicek\" adalah "
               "keputusan produk/klinis yang harus diuji pada rekaman perangkat berlabel.", y, "small")
    # ── 2. perbandingan ──
    y = L.page("Perbandingan dengan versi sebelumnya dan v1", "AFDB 3 kelas: pooled OOF 5 ulangan, CI 95% bootstrap pasien, berpasangan vs v3.1")
    y = L.chart(TMP / "macro_f1.png", y, 205)
    rows = [[NAMA[k], f"{fnum(M[k]['Macro-F1'][0])} {ci(*M[k]['Macro-F1'][1:])}", f"{fnum(M[k]['AFL F1'][0])} {ci(*M[k]['AFL F1'][1:])}",
             fnum(M[k]["AFL R"][0]), fnum(M[k]["AFL P"][0]), fnum(M[k]["Normal R"][0])] for k in SISTEM]
    y = L.table(["Sistem", "Macro-F1 [CI]", "AFL F1 [CI]", "AFL R", "AFL P", "Normal R"], rows, y,
                widths=[L.CW * .27, L.CW * .2, L.CW * .2, L.CW * .1, L.CW * .1, L.CW * .13], sorot=(5,))
    rows = []
    for k in ("H-0", "H-A", "H-X", "v13 zero-shot", "v15 gabungan", "v1 (argmax)"):
        v = BD[k]["Macro-F1"]
        rows.append([f"{NAMA[k]} − v3.1", _d(v), fnum(v[3]), vonis(v[3])])
    b1 = o["banding_v1"]["H-0"]["Macro-F1"]
    rows.append(["H-0 − v1", _d(b1), fnum(b1[3]), vonis(b1[3])])
    v = BD["H-A vs H-0"]["Macro-F1"]
    rows.append(["H-A − H-0 (H3)", _d(v), fnum(v[3]), vonis(v[3])])
    y = L.table(["Macro-F1 berpasangan", "Δ [CI 95%]", "P(Δ > 0)", "Vonis"], rows, y,
                widths=[L.CW * .34, L.CW * .32, L.CW * .13, L.CW * .21])
    y = L.para("v13 dan v15 dihitung ulang dari cache dan di-assert sama dengan ringkasan versinya. H-A dan H-0 memakai tahap 1 yang sama, "
               "jadi Normal R keduanya identik.", y, "small")
    # ── 3. metode ──
    y = L.page("Metode dan pra-registrasi", "RANCANGAN.md dikunci sebelum metrik dihitung; sha256 di logs/v16_praregistrasi.log")
    rows = [["H-0 (primer)", "ECGFounder zero-shot, tanpa penyetelan: AF bila log((pAF + pAFL) / pN) > 0", "RR v3.1: log(pAFL / pAFIB) + b2",
             "b2"],
            ["H-A", "sama dengan H-0", "regresi logistik [log p v3.1, kuantisasi, p_konsisten] (resep v15) + b2", "b2"],
            ["H-X", "skor log [pN, pAF·qAFIB, pAF·qAFL]", "RR v3.1 (q dinormalkan)", "bias 3 kelas"]]
    y = L.table(["Sistem", "Tahap 1: sinus vs AF", "Tahap 2: AFIB vs AFL", "Dipilih cross-fit"], rows, y,
                widths=[L.CW * .14, L.CW * .36, L.CW * .36, L.CW * .14])
    for t in ("<b>Kenapa tahap 1 tidak disetel.</b> AFDB hampir tidak punya sinus takikardia (0,1% window Normal di atas 120 bpm), "
              "sehingga ambang yang dipilih di AFDB tidak pernah melihat biaya alarm palsu (anti-pattern v1–v6). H-X dibuat untuk mengukur "
              "apa yang terjadi bila ambang itu tetap disetel.",
              "<b>Data (semua dari cache, tanpa inferensi baru).</b> AFDB: 165.978 window detektor 50 Hz, fold resmi 5×5; logit ECGFounder "
              "jalur 50 Hz dari v13; OOF v3.1; fitur atrium revisi 1 (v12). Tahap 2 atrium dilatih per (ulangan, fold) pada window AFIB/AFL "
              "fold latih + SHDB-AF. Sinus takikardia asli: EKG latihan fisik v3.3 (EPFL, Wrist, EPHNOGRAM; mode perangkat). Eksternal: "
              f"PTB-XL ({thou(R['n_ptbxl']['window'])} window, AFL {R['n_ptbxl']['afl']} dari {R['n_ptbxl']['pasien_afl']} pasien) dan "
              f"Icentia11k ({thou(R['n_icentia']['window'])} window, {thou(R['n_icentia']['pasien'])} pasien; akademik saja). Di data "
              "eksternal b2 dan bias = median cross-fit AFDB.",
              "<b>Hipotesis dan kriteria (dikunci).</b> H1: alarm sinus takikardia asli ≥ 120 bpm ≤ separuh v3.1 dan batas atas CI Δ < 0. "
              "H2: Macro-F1 AFDB 3 kelas H-0 bukan \"lebih buruk\" (P > 0,05). H2b: 2 kelas. H3: H-A vs H-0. H4: PTB-XL dan Icentia. "
              "H5: keluaran keempat \"perlu dicek\" (deskriptif). H6: AUROC AFL 2:1 vs sinus takikardia (eksploratif). "
              "<b>Kandidat uji perangkat</b> bila H1, H2 lolos dan H4 tidak \"lebih buruk\" di kedua data eksternal.",
              "<b>Protokol.</b> Butir 1–6 seperti biasa (bias/b2 dari fold lain, grid −7…+3). Butir 7: metrik episode gaya EC57. Butir 9: "
              "uji stres RR sintetis tidak berlaku karena tahap 1 membaca EKG; diganti EKG latihan fisik asli (sama dengan v13). Butir 10: "
              "bootstrap bersarang (parameter dipilih ulang di tiap sampel, berpasangan dengan v3.1 yang juga dipilih ulang; loop-nya di-assert "
              "sama dengan v3.2) dan tanpa 04936. Tidak ada penyimpangan dari rancangan.",
              "<b>Istilah.</b> OOF = prediksi tiap window oleh model yang tidak melihat pasiennya saat dilatih. CI 95% bootstrap pasien = "
              "rentang dari 2.000 kali mengambil ulang pasien. Bias/b2 = geseran skor kelas yang dipilih di fold lain."):
        y = L.para(t, y)
    # ── 4. H1 ──
    y = L.page("Hasil H1: sinus takikardia asli", "EKG latihan fisik, mode perangkat 50 Hz; CI dan Δ bootstrap subjek berpasangan")
    y = L.chart(TMP / "alarm_pita.png", y, 200)
    st = R["ptbxl_st"]
    rows = []
    for k, nama in (("v3.1 paket", "v3.1 paket (produksi)"), ("H-0", "H-0 dan H-A"), ("H-X", "H-X")):
        v = H1[k]
        rows.append([nama, f"{pct(v['alarm'])} {ci(v['ci'][0] * 100, v['ci'][1] * 100, 1)}", pct(v["per_sumber"]["EPFL"], 0),
                     pct(v["per_sumber"]["EPHNOGRAM"], 0), pct(v["per_sumber"]["Wrist"], 0), pct(v["pita"]["0–100 bpm"]["alarm"], 0),
                     pct(st[k])])
    y = L.table(["Sistem", "Alarm ≥ 120 bpm [CI]", "EPFL", "EPHNOGRAM", "Wrist", "< 100 bpm", "PTB-XL STACH"], rows, y,
                widths=[L.CW * .22, L.CW * .2, L.CW * .1, L.CW * .13, L.CW * .1, L.CW * .11, L.CW * .14], sorot=(2,))
    b = H1["H-0"]["banding"]
    y = L.para(f"<b>H1 lolos.</b> H-0 − v3.1 pada {thou(b['n'])} window ≥ 120 bpm: Δ {sg(b['delta'])} "
               f"[{sg(b['ci'][0])}…{sg(b['ci'][1])}]. Alarm di bawah 100 bpm, yang pada v3.1 berasal dari jitter detektor R 50 Hz saat bergerak "
               f"(v3.3), juga hilang ({pct(H1['v3.1 paket']['pita']['0–100 bpm']['alarm'], 0)} → {pct(H1['H-0']['pita']['0–100 bpm']['alarm'], 0)}). "
               f"Di PTB-XL, alarm pada sinus takikardia (STACH, {R['n_ptbxl']['stach']} rekaman) turun {pct(st['v3.1 paket'])} → {pct(st['H-0'])}. "
               "Satuan ukur: % window dengan keputusan AFIB atau AFL; CI per pita detak ada di grafik (pita berwarna).", y)
    y = L.para("Catatan: subjek latihan fisik sehat dan tidak berlabel ritme; window diasumsikan sinus. Bila ada ektopik saat latihan, "
               "angka alarm ini sedikit melebih-lebihkan alarm palsu.", y, "small")
    # ── 5. 2 kelas, episode, butir 10 ──
    y = L.page("AFDB 2 kelas, episode, dan uji ketahanan", "H2b; protokol butir 7 dan 10")
    db = DK["banding"]
    rows = [["v3.1 (bias 3 kelas, lalu digabung)", f"{fnum(DK['v3.1']['Macro-F1'][0])} {ci(*DK['v3.1']['Macro-F1'][1:])}",
             fnum(DK["v3.1"]["Se"][0]), fnum(DK["v3.1"]["Sp"][0]), fnum(DK["v3.1"]["PPV"][0])],
            ["H-0 (keputusan tahap 1)", f"{fnum(DK['H-0']['Macro-F1'][0])} {ci(*DK['H-0']['Macro-F1'][1:])}",
             fnum(DK["H-0"]["Se"][0]), fnum(DK["H-0"]["Sp"][0]), fnum(DK["H-0"]["PPV"][0])],
            ["H-0 − v3.1 [CI], P", f"{_d(db['Macro-F1'])}, P = {fnum(db['Macro-F1'][3])}", f"{sg(db['Se'][0])}, P = {fnum(db['Se'][3])}",
             f"{sg(db['Sp'][0])}, P = {fnum(db['Sp'][3], 2)}", ""]]
    y = L.table(["2 kelas (Normal vs AF/AFL)", "Macro-F1 [CI]", "Sensitivitas", "Spesifisitas", "PPV"], rows, y,
                widths=[L.CW * .29, L.CW * .29, L.CW * .15, L.CW * .14, L.CW * .13])
    y = L.para(f"Bersarang (bias v3.1 dipilih ulang di tiap sampel): Δ {sg(DK['bersarang']['delta'])} "
               f"[{sg(DK['bersarang']['ci_delta'][0])}…{sg(DK['bersarang']['ci_delta'][1])}], P = {fnum(DK['bersarang']['p'], 2)}: tidak "
               f"terbedakan. Sensitivitas turun karena AFL dan sebagian AFIB dibaca sinus (tahap 1: AFIB {pct(T1['se_afib'], 0)}, AFL "
               f"{pct(T1['se_afl'], 0)}); window Normal yang disebut AF turun {pct(1 - DK['v3.1']['Sp'][0])} → {pct(1 - DK['H-0']['Sp'][0])}.", y, "small")
    rows = []
    for k in ("v3.1", "H-0", "H-A", "H-X"):
        a, f = EP[k]["AFIB"], EP[k]["AFL"]
        rows.append([NAMA[k], fnum(a["se_episode"], 2), fnum(a["ppv_episode"], 2), fnum(a["palsu_per_jam"], 1),
                     fnum(f["se_episode"], 2), fnum(f["ppv_episode"], 2), fnum(f["palsu_per_jam"], 2)])
    y = L.table(["Episode (EC57)", "AFIB Se", "AFIB +P", "AFIB palsu/jam", "AFL Se", "AFL +P", "AFL palsu/jam"], rows, y,
                widths=[L.CW * .25, L.CW * .11, L.CW * .11, L.CW * .14, L.CW * .11, L.CW * .11, L.CW * .17])
    bs, tp = R["bersarang"], R["tanpa_04936"]
    rows = [[NAMA[k], f"{fnum(bs[k]['titik'])} {ci(*bs[k]['ci'])}", f"{sg(bs[k]['delta'])} [{sg(bs[k]['ci_delta'][0])}…{sg(bs[k]['ci_delta'][1])}]",
             fnum(bs[k]["p"], 4), f"{fnum(tp['rows'][k]['Macro-F1'][0])} vs {fnum(tp['rows']['v31']['Macro-F1'][0])}",
             fnum(1 - tp["pairs_v31_minus"][k][3], 4)] for k in ("H-0", "H-X")]
    y = L.table(["Butir 10", "Bersarang [CI]", "Δ vs v3.1 [CI]", "P(Δ > 0)", "Tanpa 04936 (vs v3.1)", "P(Δ > 0)"], rows, y,
                widths=[L.CW * .2, L.CW * .2, L.CW * .22, L.CW * .09, L.CW * .2, L.CW * .09])
    pm = R["parameter"]
    y = L.para(f"Parameter cross-fit: b2 median H-0 {fnum(pm['H-0']['median'], 2)} (rentang {fnum(pm['H-0']['min'], 2)}…{fnum(pm['H-0']['maks'], 2)}), "
               f"H-A {fnum(pm['H-A']['median'], 2)}; tidak ada di batas grid. Bias H-X {daftar(pm['H-X']['median'])} dengan "
               f"{pct(pm['H-X']['di_batas'], 0)} nilai di batas grid: tanda bahaya (AGENTS.md), penyetelan mendorong AF setinggi mungkin "
               "untuk mengejar AFL. Tanpa 04936 selisihnya tetap besar, tetapi CI lebar (tinggal 6 pasien AFL), jadi P tidak sampai 0,05.",
               y, "small")
    # ── 6. per pasien, H5, H6 ──
    y = L.page("Per pasien AFL, keluaran keempat (H5), dan AFL 2:1 (H6)", "AFDB, rata-rata 5 ulangan; n = jumlah window AFL pasien")
    y = L.chart(TMP / "afl_pasien.png", y, 195)
    afl, a31, hp = R["afl_per_pasien"], o["alarm_afl_v31"], H5["pasien_afl"]
    rows = [[p, str(hp[p]["n"]), fnum(afl["v3.1"][p], 2), fnum(a31[p], 2), fnum(afl["H-0"][p], 2), fnum(afl["H-A"][p], 2),
             fnum(afl["H-X"][p], 2), fnum(hp[p]["alarm"], 2), fnum(hp[p]["cek"], 2)] for p in PASIEN_AFL]
    y = L.table(["Pasien", "n", "v3.1 AFL", "v3.1 alarm", "H-0 AFL", "H-A AFL", "H-X AFL", "H-0 alarm", "perlu dicek"], rows, y,
                widths=[L.CW * .11] + [L.CW * .08] + [L.CW * .1] * 5 + [L.CW * .11, L.CW * .2], sorot=(7,))
    k5, l5, p5 = H5["kelas"], H5["latihan_≥120"], H5["ptbxl"]
    y = L.para(f"<b>H5.</b> \"Perlu dicek\" = tahap 1 sinus dan v3.1 AFL. Proporsinya: AFL AFDB {pct(k5['AFL']['cek'], 0)} (08378 "
               f"{pct(hp['08378']['cek'], 0)}), Normal AFDB {pct(k5['Normal']['cek'])}, AFIB AFDB {pct(k5['AFIB']['cek'])}, sinus takikardia "
               f"asli ≥ 120 bpm {pct(l5['cek'], 0)} {ci(l5['ci_cek'][0] * 100, l5['ci_cek'][1] * 100, 0)}, STACH PTB-XL {pct(p5['STACH']['cek'], 0)}, "
               f"AFL PTB-XL {pct(p5['AFL']['cek'], 0)}. Dengan v3.1, {pct(a31['semua'], 0)} window AFL AFDB memicu alarm AF; dengan H-0 "
               f"{pct(k5['AFL']['alarm'], 0)} alarm + {pct(k5['AFL']['cek'], 0)} \"perlu dicek\".", y)
    h6 = R["h6"]
    y = L.para(f"<b>H6 (eksploratif).</b> AUROC window AFL 08378 ({h6['n_pos']}) vs sinus takikardia asli ≥ 120 bpm ({thou(h6['n_neg'])}): "
               f"logit ECGFounder AFL − ST {fnum(h6['ecgfounder'], 2)}, skor RR v3.1 (log pAFL − log pN) {fnum(h6['v31'], 2)}. Satu pasien AFL dan "
               "dua dataset berbeda, jadi selisihnya bisa berasal dari pergeseran domain, bukan dari ritme.", y, "small")
    # ── 7. eksternal ──
    y = L.page("Data eksternal (H3, H4)", "Bootstrap pasien berpasangan vs paket v3.1; b2/bias = median cross-fit AFDB")
    rows = []
    for data, D_, n in (("PTB-XL", R["ptbxl"], ("H-0", "H-A", "H-X")), ("Icentia11k", R["icentia"], ("H-0", "H-X"))):
        for k in n:
            v = D_[k]
            rows.append([data, NAMA[k], f"{fnum(v['titik_a'][0])} / {fnum(v['titik_b'][0])}", _d(v["Macro-F1"]), fnum(v["Macro-F1"][3]),
                         vonis(v["Macro-F1"][3]), f"{fnum(v['titik_a'][1], 2)} / {fnum(v['titik_b'][1], 2)}"])
    y = L.table(["Data", "Sistem", "Macro-F1 sistem / v3.1", "Δ [CI 95%]", "P", "Vonis", "Recall AFL sistem / v3.1"], rows, y,
                widths=[L.CW * .12, L.CW * .2, L.CW * .14, L.CW * .19, L.CW * .08, L.CW * .13, L.CW * .14], sorot=(1, 4))
    v = R["ptbxl"]["H-A vs H-0"]
    y = L.para(f"<b>H3.</b> H-A − H-0 di PTB-XL: Macro-F1 {_d(v['Macro-F1'])}, P = {fnum(v['Macro-F1'][3])}; recall AFL "
               f"{fnum(v['titik_b'][1], 2)} → {fnum(v['titik_a'][1], 2)} (P = {fnum(v['AFL R'][3])}). Di AFDB "
               f"Δ {sg(BD['H-A vs H-0']['Macro-F1'][0])} (P = {fnum(BD['H-A vs H-0']['Macro-F1'][3])}). Tidak terbedakan di keduanya.", y)
    y = L.para(f"<b>H4.</b> PTB-XL (rekaman istirahat 10 s): H-0 lebih baik dari v3.1, sejalan dengan alarm pada sinus takikardia yang turun "
               f"({pct(R['ptbxl_st']['v3.1 paket'])} → {pct(R['ptbxl_st']['H-0'])}), tetapi recall AFL turun {fnum(R['ptbxl']['H-0']['titik_b'][1], 2)} → "
               f"{fnum(R['ptbxl']['H-0']['titik_a'][1], 2)}. Icentia11k (wearable, AFL kebanyakan blok bervariasi, v10): H-0 lebih buruk, "
               f"karena AFL adalah kelas terbesar di sana ({thou(R['n_icentia']['afl'])} window) dan recall AFL H-0 hanya "
               f"{fnum(R['icentia']['H-0']['titik_a'][1], 2)}. H-X kebalikannya: lebih buruk di PTB-XL, "
               "sedikit lebih baik di Icentia. Pelajaran v9/v10 berulang: ambang yang baik di satu dataset tidak berpindah.", y)
    # ── 8. kesimpulan ──
    y = L.page("Kesimpulan, langkah berikutnya, batasan, reproduksi", "")
    for t in ("<b>Kesimpulan.</b> ECGFounder 1-lead pada EKG 50 Hz adalah detektor AFIB yang sangat spesifik: alarm palsu pada sinus "
              "takikardia hampir nol, juga saat bergerak dengan detektor R 50 Hz. Tetapi AFL, terutama AFL 2:1 dan AFL AFDB pada umumnya, "
              "dibacanya sebagai sinus. Hierarki tidak bisa memulihkan AFL yang sudah dibuang di tahap 1, jadi tahap 2 (RR atau atrium) tidak "
              "berpengaruh. Menyetel tahap 1 di AFDB juga bukan jalan keluar: biasnya menempel di batas grid dan hasilnya tidak berpindah ke "
              "data lain.",
              "<b>Trade-off untuk produk.</b> Untuk keluaran 2 kelas (AF ya/tidak), H-0 setara v3.1 dengan spesifisitas jauh lebih tinggi dan "
              "episode AFIB palsu ±10× lebih sedikit, dibayar dengan AFL. Keluaran \"perlu dicek\" menampung sebagian besar AFL, tetapi juga "
              "sebagian besar sinus takikardia, jadi itu pemberitahuan lunak, bukan diagnosis. Menerima trade-off ini atau tidak adalah keputusan "
              "produk dan klinis.",
              "<b>Langkah berikutnya.</b> (1) Rekaman perangkat SIHEDAF berlabel tetap penentu: uji H-0 + \"perlu dicek\" di sana, dengan "
              "rancangan yang dikunci lagi. (2) Untuk AFL, informasi yang belum dipakai adalah lead lain/sampling lebih tinggi (gelombang F "
              "paling jelas di II, III, aVF, V1) atau model fondasi yang di-fine-tune dengan AFL dari banyak pasien; dengan 7 pasien AFL "
              "AFDB fine-tune akan overfit (v6). (3) ECGFounder (30,8 juta parameter) perlu inferensi di backend; ukur latensi dan biaya "
              "sebelum dipakai.",
              "<b>Batasan.</b> AFDB: 7 pasien AFL, 58% window dari 04936; label 08378 (AFL 2:1) dari satu pasien. Data latihan fisik tidak "
              "berlabel ritme. PTB-XL hanya 10 s saat istirahat; Icentia berlabel per segmen dan berisi AFL dengan fenotipe berbeda (v10). "
              "Kanal AFDB yang dipakai (ECG1) belum tentu sama dengan lead perangkat.",
              "<b>Reproduksi.</b> <font face='Arial-Bold'>python tools/run_notebook.py v16_hierarki_ecgfounder/hierarki.ipynb</font>, lalu "
              "<font face='Arial-Bold'>build_report.py --extract</font> dan tanpa argumen. Butuh cache v3.3, v10, v12, v13, v15 dan "
              "cache/oof_robust (lihat README v16); tidak butuh GPU."):
        y = L.para(t, y)
    lk = R["lingkungan"]
    y = L.para(f"Lingkungan: commit {lk['commit']} (+ perubahan belum di-commit), Python {lk['pustaka']['python']}, numpy "
               f"{lk['pustaka']['numpy']}, scikit-learn {lk['pustaka']['scikit-learn']}, torch {lk['pustaka']['torch']}. Notebook ±12 menit "
               "di PC (CPU; semua dari cache).", y, "small")
    L.selesai()
    print("PDF:", OUT, L.page_no, "halaman")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
