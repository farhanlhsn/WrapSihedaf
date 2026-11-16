"""Laporan PDF v3.4: pembersih detak (penyelarasan R-peak ke templat QRS + gerbang morfologi).

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v3_4/ringkasan.json (sel akhir pembersih_detak.ipynb), hitung ulang angka utama dari cache lalu cocokkan
     (assert), gambar grafik → laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_v3_4_pembersih_detak.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v3_4_pembersih_detak.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, Laporan, ci, fnum, pct, sg, sumbu_koma, thou  # noqa: E402

TANGGAL = "10 November 2026"
DS = ("CinC 2017", "MIT-BIH", "LTAFDB", "CPSC 2021")


def extract():
    import numpy as np
    import pembersih as P
    R = json.loads((P.CACHE / "ringkasan.json").read_text(encoding="utf-8"))
    # 1. AFDB: Macro-F1 dan recall AFL titik v3.1, F1, F2 dihitung ulang dari OOF
    lama, f1 = P.data_afdb()
    for k, d, pred in (("v31", lama, lama["pred"]), ("f1", f1, f1["pred"]), ("f2", f1, f1["pred_f2"])):
        t = P.cv.point_metrics(P.cms(d, pred))
        for m in ("Macro-F1", "AFL R", "AFIB R"):
            assert abs(float(t[m].mean()) - R["metrik_afdb"][k][m]["nilai"]) < 1e-12, (k, m)
    # 2. aktivitas: alarm H2 dihitung ulang dari window dan prediksi tersimpan
    L = P.jendela_latihan()
    K = P.prediksi_latihan(L)
    h2 = np.isin(L["sumber"], ["Wrist", "EPHNOGRAM"]) & (L["hr_lama"] < 100)
    assert abs(float((K["v31_lama"][h2] > 0).mean()) - R["H2"]["alarm_lama"]) < 1e-12
    assert abs(float((K["v31_f1"][h2] > 0).mean()) - R["H2"]["alarm_f1"]) < 1e-12
    assert abs(float((K["v31_f2"][h2] > 0).mean()) - R["H4_aktivitas"]["alarm_f2"]) < 1e-12
    # 3. eksternal: Se titik front-end lama dan F1 per dataset
    for ds in DS:
        Z = P.muat_eksternal(ds)
        af = Z["y2"] == 1
        e = R["eksternal"][ds]["f1_vs_lama"]
        assert abs(float((Z["v31_lama"][af] > 0).mean()) - e["titik_b"]["Se"][0]) < 1e-12, ds
        assert abs(float((Z["v31_f1"][af] > 0).mean()) - e["titik_a"]["Se"][0]) < 1e-12, ds
    TMP.mkdir(exist_ok=True)
    contoh = contoh_strip(P, L)
    X = json.loads((P.CACHE / "eksplorasi_f1b/ringkasan.json").read_text(encoding="utf-8"))
    for ds in DS:   # F1 di eksplorasi = F1 di notebook
        assert abs(X["eksternal"][ds]["f1_vs_lama"]["Se"][0] - R["eksternal"][ds]["f1_vs_lama"]["Se"][0]) < 1e-12, ds
    assert abs(X["latihan"]["H2"]["f1"] - R["H2"]["alarm_f1"]) < 1e-12
    strip_mitbih(P)
    (TMP / "v3_4.json").write_text(json.dumps({"R": R, "contoh": contoh, "X": X}, indent=1, default=float, ensure_ascii=False),
                                   encoding="utf-8")
    grafik(R)
    print("extract selesai; assert AFDB (Macro-F1, recall AFL/AFIB), alarm aktivitas H2/H4, Se eksternal cocok dengan ringkasan.json")


def contoh_strip(P, L):
    """Strip 6 s EPFL dengan R-peak manual, front-end lama, dan F1: rekaman dengan perbaikan jitter terbesar."""
    import numpy as np
    plt = _gaya()
    m = L["galat_sumber"] == "EPFL"
    gain = L["galat_lama"][m] - L["galat_f1"][m]
    subj = L["galat_subjek"][m][int(np.nanargmax(gain))]
    best = None
    for rek in P.T33.rekaman_epfl():
        if rek["subjek"] != subj:
            continue
        o = P.proses_sinyal(rek["ecg"], rek["fs"])
        ref = np.sort(rek["rpeak"]) / rek["fs"]
        e0, _ = P.cocokkan(o["p_lama"] / P.FS, ref, 0.1)
        e1, _ = P.cocokkan(o["p_f1"] / P.FS, ref, 0.1)
        g = np.std(e0 - np.median(e0)) - np.std(e1 - np.median(e1))
        if best is None or g > best[0]:
            best = (g, rek, o, ref)
    _, rek, o, ref = best
    y = P.bandpass(o["x"])
    t = np.arange(y.size) / P.FS
    # jendela 6 s dengan selisih lama vs F1 terbanyak
    beda = np.abs(o["p_lama"] - o["p_f1"]) > 1.0
    tb = o["p_lama"][beda] / P.FS
    t0 = max(0.0, float(np.median(tb)) - 3) if tb.size else 2.0
    w = (t >= t0) & (t < t0 + 6)
    fig, ax = plt.subplots(figsize=(7.6, 2.4), dpi=200)
    ax.plot(t[w], y[w], color="#9AA8B6", lw=1.0, label="EKG 50 Hz (bandpass)")
    for x in ref[(ref >= t0) & (ref < t0 + 6)]:
        ax.axvline(x, color=INK, lw=.7, ls=":", zorder=0)
    for p, col, mk, lab in ((o["p_lama"], BIRU, "v", "front-end lama"), (o["p_f1"], ORANYE, "o", "F1 (selaras templat)")):
        tt = p / P.FS
        k = (tt >= t0) & (tt < t0 + 6)
        ax.plot(tt[k], np.interp(tt[k], t, y), mk, color=col, ms=5, mfc="none" if mk == "o" else col, mew=1.3, label=lab, ls="none")
    ax.plot([], [], color=INK, lw=.7, ls=":", label="R-peak manual")
    ax.set_xlabel("detik"); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.18)); sumbu_koma(ax, "x")
    fig.tight_layout(); fig.savefig(TMP / "strip.png"); plt.close(fig)
    return {"subjek": str(subj), "rekaman": rek["rekaman"], "t0": t0}


def strip_mitbih(P):
    """MIT-BIH 210, window 825–835 s: titik lama, F1, dan F1b pada detak N dan V (anotasi)."""
    import numpy as np
    import eksplorasi_ektopik as E
    plt = _gaya()
    x, fs = P.A12.baca_dat(P.REPO / "data/mitdb/210", 0)
    s, k, _ = P.A12.baca_atr(P.REPO / "data/mitdb/210")
    b = np.isin(k, list(E.KODE))
    bs, bk = s[b] / fs, np.array([E.KODE[int(c)] for c in k[b]])
    o = P.proses_sinyal(x, fs)
    y = P.bandpass(o["x"])
    t = np.arange(y.size) / P.FS
    t0 = 825.0
    w = (t >= t0) & (t < t0 + 10)
    fig, ax = plt.subplots(figsize=(7.6, 2.4), dpi=200)
    ax.plot(t[w], y[w], color="#9AA8B6", lw=1.0)
    for p, col, mk, lab in ((o["p_lama"], BIRU, "v", "lama"), (o["p_f1"], ORANYE, "o", "F1"), (E.peaks_f1b(o), HIJAU, "x", "F1b")):
        tt = p / P.FS
        m = (tt >= t0) & (tt < t0 + 10)
        ax.plot(tt[m], np.interp(tt[m], t, y), mk, color=col, ms=5.5, mfc="none" if mk == "o" else col, mew=1.3, ls="none", label=lab)
    for bt, c in zip(bs, bk):
        if t0 <= bt < t0 + 10:
            ax.text(bt, y[w].max() * 1.12, c, ha="center", fontsize=8, color=INK if c == "N" else ORANYE, fontweight="bold")
    ax.set_xlabel("detik"); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False); sumbu_koma(ax, "x")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower right")
    fig.tight_layout(); fig.savefig(TMP / "mitbih210.png"); plt.close(fig)


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


GAYA = {"v1": (GRAY, "D", "v1"), "v31": (BIRU, "o", "v3.1 (referensi)"), "f1": (ORANYE, "s", "F1"), "f2": (HIJAU, "^", "F2")}


def grafik(R):
    import numpy as np
    plt = _gaya()
    # 1. AFDB: metrik dengan CI
    met = R["metrik_afdb"]
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.3), dpi=200, sharey=True)
    for ax, m, judul in zip(axs, ("Macro-F1", "Normal R", "AFIB R", "AFL R"), ("Macro-F1", "recall Normal", "recall AFIB", "recall AFL")):
        for i, k in enumerate(("v1", "v31", "f1", "f2")):
            col, mk, _ = GAYA[k]
            v = met[k][m]
            ax.plot(v["ci"], [i, i], color=col, lw=1.6, alpha=.6)
            ax.plot(v["nilai"], i, mk, color=col, ms=6)
        ax.set_title(judul, fontsize=9, color=INK); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#E6ECF1")
        sumbu_koma(ax, "x"); ax.tick_params(axis="x", labelsize=7.5)
    axs[0].set_yticks(range(4), [GAYA[k][2] for k in ("v1", "v31", "f1", "f2")], fontsize=8.5)
    fig.text(.5, .01, "titik = pooled OOF rata-rata 5 ulangan; garis = CI 95% bootstrap pasien", ha="center", fontsize=8, color=GRAY)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(TMP / "afdb.png"); plt.close(fig)
    # 2. aktivitas: alarm per pita detak (detak front-end sendiri), Wrist dan EPHNOGRAM
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.6), dpi=200, sharey=True)
    for ax, s in zip(axs, ("Wrist", "EPHNOGRAM")):
        pita = R["alarm_pita"][s]
        bands = [b for b in pita["v3.1 lama"] if b in pita["v3.1 + F1"] and pita["v3.1 lama"][b][0] >= 20]
        for i, (nama, kunci) in enumerate((("lama", "v3.1 lama"), ("F1", "v3.1 + F1"), ("F2", "v3.1 + F2"))):
            col = (BIRU, ORANYE, HIJAU)[i]
            ax.bar([x + (i - 1) * .27 for x in range(len(bands))], [pita[kunci][b][1] * 100 for b in bands], .25, color=col,
                   label=f"v3.1 + {nama}" if nama != "lama" else "v3.1, front-end lama", hatch=(None, "//", "..")[i], edgecolor="white")
        ax.set_xticks(range(len(bands)), [b.replace("∞", "") + ("+" if "∞" in b else "") for b in bands], fontsize=7.5)
        ax.set_title(s, fontsize=9, color=INK); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1")
        sumbu_koma(ax); ax.set_xlabel("detak (bpm)", fontsize=8)
    axs[0].set_ylabel("alarm AF/AFL (%)"); axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(TMP / "aktivitas.png"); plt.close(fig)
    # 3. eksternal: Δ Se dan Δ Sp (F1, F2 vs front-end lama) dengan CI
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.4), dpi=200, sharey=True)
    for ax, m, judul in zip(axs, ("Se", "Sp"), ("Δ sensitivitas AF", "Δ spesifisitas")):
        for i, ds in enumerate(DS):
            for j, (k, kunci) in enumerate((("f1", "f1_vs_lama"), ("f2", "f2_vs_lama"))):
                col, mk, _ = GAYA[k]
                v = R["eksternal"][ds][kunci][m]
                yy = i + (.15 if j == 0 else -.15)
                ax.plot([v[1] * 100, v[2] * 100], [yy, yy], color=col, lw=1.6, alpha=.6)
                ax.plot(v[0] * 100, yy, mk, color=col, ms=6, label=("F1" if j == 0 else "F2") if i == 0 else None)
        ax.axvline(0, color=INK, lw=.8); ax.set_title(judul + " (poin)", fontsize=9, color=INK)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#E6ECF1"); sumbu_koma(ax, "x")
    axs[0].set_yticks(range(len(DS)), DS, fontsize=8.5); axs[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(.5, .01, "vs paket v3.1 dengan front-end lama; CI 95% bootstrap subjek", ha="center", fontsize=8, color=GRAY)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(TMP / "eksternal.png"); plt.close(fig)
    # 4. uji stres sintetis: proporsi ditebak Normal (tanpa variasi) model fold F1 vs paket v3.1
    r32 = json.loads((REPO / "cache/v3_2/ringkasan.json").read_text(encoding="utf-8"))
    bpm = R["stres_bpm"]
    fig, ax = plt.subplots(figsize=(7.6, 2.2), dpi=200)
    ax.plot(bpm, np.array(r32["stress"]["v31"]["tanpa variasi"])[:, 0] * 100, "-o", color=BIRU, ms=3.5, label="paket v3.1")
    ax.plot(bpm, np.array(R["stres"]["tanpa variasi"]["normal"]) * 100, "-s", color=ORANYE, ms=3.5, label="model fold F1 (rata-rata 25)")
    ax.axhline(50, color=GRAY, lw=.8, ls="--")
    ax.set_xlabel("detak ritme teratur sintetis (bpm)", fontsize=8.5); ax.set_ylabel("ditebak Normal (%)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#E6ECF1"); ax.legend(frameon=False, fontsize=8); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "stres.png"); plt.close(fig)


def _b(v, nd=3):
    """dict banding pembersih → 'Δ [CI], P = …'."""
    return f"{sg(v['delta'], nd)} [{sg(v['ci'][0], nd)}; {sg(v['ci'][1], nd)}], P = {fnum(v['p'], 3)}"


def _b2(v, nd=3):
    """list banding2_pasien → 'Δ [CI], P = …'."""
    return f"{sg(v[0], nd)} [{sg(v[1], nd)}; {sg(v[2], nd)}], P = {fnum(v[3], 3)}"


def _vonis2(p):
    return "lebih baik" if p >= .95 else "lebih buruk" if p <= .05 else "tidak terbedakan"


def build():
    J = json.loads((TMP / "v3_4.json").read_text(encoding="utf-8"))
    R, contoh = J["R"], J["contoh"]
    met, bd, k = R["metrik_afdb"], R["banding_afdb"], R["keputusan"]
    eks = R["eksternal"]
    ok = lambda b: "lolos" if b else "gagal"  # noqa: E731
    L = Laporan(OUT, "v3.4: pembersih detak", "SIHEDAF  /  V3.4  /  PEMBERSIH DETAK",
                "v3.4_pembersih_detak/pembersih_detak.ipynb, cache/v3_4/", TANGGAL)
    # ── 1. ringkasan ──
    y = L.page("Ringkasan: detak lebih rapi, tetapi belum layak dipasang", "Rancangan dikunci sebelum hasil; ide dari pembaca EKG Tugas Akhir")
    jt = R["latihan_jitter"]["EPFL"]
    y = L.kartu([(f"{fnum(jt['jitter lama (ms, median rekaman)'], 1)} → {fnum(jt['jitter F1'], 1)} ms", "jitter R-peak (EPFL)", "vs R-peak manual"),
                 (f"{pct(R['H3a']['lolos_lama'], 0)} → {pct(R['H3a']['lolos_f1'], 0)}", "window EPFL lolos cek", "R-peak manual: 95%"),
                 (f"{pct(R['H2']['alarm_lama'], 0)} → {pct(R['H4_aktivitas']['alarm_f2'], 0)}", "alarm < 100 bpm, aktivitas", f"dengan F2; F1 saja {pct(R['H2']['alarm_f1'], 1)}"),
                 (f"{fnum(met['v31']['Macro-F1']['nilai'])} → {fnum(met['f1']['Macro-F1']['nilai'])}", "Macro-F1 AFDB (F1)", "bersarang: tidak terbedakan")], y)
    y = L.para("<b>Pertanyaan.</b> Detektor R-peak 50 Hz memilih puncak |sinyal|, sehingga pada sebagian detak titiknya melompat dari "
               "puncak R ke lembah S. Lompatan ini menambah \"getaran\" (jitter) pada jarak antar detak dan memicu alarm palsu saat aktivitas "
               "(v3.3). Pembaca EKG di Tugas Akhir pemilik repo membandingkan setiap QRS dengan templat median QRS di sekitarnya. Apakah "
               "cara itu memperbaiki SIHEDAF tanpa melatih model baru?", y)
    y = L.para("<b>Jawaban singkat: sebagian.</b> <b>F1</b> (setiap R-peak digeser ≤ 3 sampel ke posisi yang paling cocok dengan templat) "
               "membuat jarak antar detak saat latihan fisik jauh lebih rapi dan menurunkan alarm palsu. Di AFDB hasilnya sama atau sedikit lebih "
               f"baik. Tetapi di MIT-BIH sensitivitas AF turun tipis ({sg(eks['MIT-BIH']['f1_vs_lama']['Se'][0] * 100, 1)} poin, P = "
               f"{fnum(eks['MIT-BIH']['f1_vs_lama']['Se'][3], 3)}), jadi kriteria yang dikunci tidak lolos. <b>F2</b> (F1 + window ditolak bila "
               "bentuk QRS-nya tidak konsisten) menurunkan alarm palsu lebih banyak, tetapi menurunkan sensitivitas AF 1–3 poin di keempat data "
               "eksternal. <b>Keduanya tidak direkomendasikan untuk dipasang sekarang.</b> Paket produksi tetap v3.1.", y)
    rows = [["H1 AFDB: Macro-F1 dan recall AFL F1 bukan lebih buruk dari v3.1", ok(k["H1"]),
             f"Macro-F1 {sg(bd['f1_vs_v31']['Macro-F1']['delta'])}; AFL {sg(bd['f1_vs_v31']['AFL R']['delta'])}"],
            ["H2 alarm < 100 bpm saat aktivitas berkurang (F1)", ok(k["H2"]), f"{pct(R['H2']['alarm_lama'])} → {pct(R['H2']['alarm_f1'])}"],
            ["H3b alarm ≥ 120 bpm yang lolos aturan produk berkurang (F1)", ok(k["H3b"]), f"{pct(R['H3b']['sisa_lama'])} → {pct(R['H3b']['sisa_f1'])}"],
            ["H4 gerbang F2: alarm turun, biaya recall AFDB ≤ 0,02", ok(k["H4"]),
             f"{pct(R['H4_aktivitas']['alarm_f1'])} → {pct(R['H4_aktivitas']['alarm_f2'])}; AFIB −{fnum(k['biaya_f2_recall']['AFIB'])}"],
            ["H5 drop-in: bobot lama + front-end F1 bukan lebih buruk", ok(k["H5"]), f"Macro-F1 {sg(R['dropin']['banding']['Macro-F1']['delta'])}"],
            ["H6 Se di 4 data eksternal bukan lebih buruk (F1 / F2)", f"{ok(k['H6_f1'])} / {ok(k['H6_f2'])}", "F1: MIT-BIH; F2: keempatnya"],
            ["Keputusan: F1 / F2 direkomendasikan", f"{'ya' if k['F1_direkomendasikan'] else 'tidak'} / {'ya' if k['F2_direkomendasikan'] else 'tidak'}", ""]]
    y = L.table(["Kriteria yang dikunci", "Hasil", "Angka"], rows, y, widths=[L.CW * .55, L.CW * .13, L.CW * .32], sorot=(7,))
    y = L.para("<b>Temuan.</b> (1) Penyebab jitter terkonfirmasi: lompatan R/S, bukan resolusi 50 Hz. (2) Front-end tidak mengubah AFDB "
               "secara berarti (window, label, ketidakteraturan AF sama), sehingga bobot paket v3.1 bisa dipakai ulang (H5). (3) Sumber alarm "
               "terbesar saat aktivitas tetap model RR itu sendiri: alarm ≥ 120 bpm masih ±98%. (4) Gerbang morfologi berguna untuk spesifisitas, "
               "tetapi ia juga menolak AF dengan QRS yang bentuknya berubah-ubah.", y)
    # ── 2. perbandingan ──
    y = L.page("Perbandingan dengan v3.1 dan v1 (AFDB)", "Window detektor 50 Hz, fold pasien resmi; Δ berpasangan bootstrap pasien 2.000×")
    y = L.chart(TMP / "afdb.png", y, 170)
    rows = [[GAYA[x][2]] + [f"{fnum(met[x][m]['nilai'])} {ci(*met[x][m]['ci'])}" for m in ("Macro-F1", "Normal R", "AFIB R", "AFL R", "AFL F1")]
            for x in ("v1", "v31", "f1", "f2")]
    y = L.table(["Model", "Macro-F1 [CI]", "Recall Normal", "Recall AFIB", "Recall AFL", "F1 AFL"], rows, y,
                widths=[L.CW * .2] + [L.CW * .16] * 5, sorot=(3,))
    rows = []
    for kunci, lab in (("f1_vs_v31", "F1 − v3.1"), ("f2_vs_v31", "F2 − v3.1"), ("f2_vs_f1", "F2 − F1"), ("f1_vs_v1", "F1 − v1")):
        v = bd[kunci]
        rows.append([lab, _b(v["Macro-F1"]), v["Macro-F1"]["vonis"], f"{sg(v['AFL R']['delta'])}, {v['AFL R']['vonis']}",
                     f"{sg(v['AFIB R']['delta'])}, {v['AFIB R']['vonis']}"])
    y = L.table(["Banding", "Δ Macro-F1 [CI], P(Δ > 0)", "Vonis", "Δ recall AFL", "Δ recall AFIB"], rows, y,
                widths=[L.CW * .12, L.CW * .36, L.CW * .14, L.CW * .19, L.CW * .19], sorot=(1,))
    bs, t4 = R["bersarang"], R["tanpa_04936"]["banding"]
    y = L.para(f"<b>Protokol 10.</b> Dengan bootstrap bersarang (bias dipilih ulang di setiap sampel), Δ Macro-F1 F1 − v3.1 = "
               f"{sg(bs['delta'])} [{sg(bs['ci'][0])}; {sg(bs['ci'][1])}], P = {fnum(bs['p'])}: <b>tidak terbedakan</b> (biasa: "
               f"P = {fnum(bd['f1_vs_v31']['Macro-F1']['p'])}). Tanpa pasien 04936: Δ {sg(t4['Macro-F1']['delta'])}, P = "
               f"{fnum(t4['Macro-F1']['p'], 2)}. Pada {thou(R['window_sama']['n'])} window yang sama di kedua dataset: Δ "
               f"{sg(R['window_sama']['banding']['Macro-F1']['delta'])}, P = {fnum(R['window_sama']['banding']['Macro-F1']['p'], 2)}.", y)
    y = L.para("Model F1 = MicroTransformer resep v3 (sama dengan v3.1, bobot kelas penuh) yang dilatih ulang di window F1. v1 dan v3.1 "
               "dihitung di window front-end lama; perbandingan tetap berpasangan karena pasien dan bobot bootstrap sama.", y, "small")
    # ── 3. metode ──
    y = L.page("Metode: apa yang diubah", "Detektor produksi tidak diubah; lapisan baru di antara R-peak dan window")
    y = L.chart(TMP / "strip.png", y, 150)
    y = L.para(f"<i>Contoh EPFL subjek {contoh['subjek']} ({contoh['rekaman']}): segitiga = titik front-end lama, lingkaran = F1, garis titik = "
               "R-peak manual dokter. Front-end lama kadang menunjuk lembah S, bukan puncak R.</i>", y, "small")
    for t in ("<b>Front-end lama.</b> Sinyal di-resample ke 50 Hz, difilter 5–15 Hz, lalu kandidat QRS dicari (gaya Pan-Tompkins). Titik "
              "akhirnya puncak <i>nilai mutlak</i> sinyal ±80 ms di sekitar kandidat. Bila lembah S lebih dalam dari puncak R, titiknya pindah "
              "±30–60 ms. Pada detak berikutnya bisa kembali ke R. Jarak antar detak jadi bergetar walaupun ritmenya teratur.",
              "<b>F1: penyelarasan ke templat.</b> Untuk setiap detak, templat = median bentuk QRS (±0,1 s) semua detak dalam ±5 s. "
              "Detak digeser −3…+3 sampel ke posisi dengan korelasi tertinggi terhadap templat, lalu diperhalus dengan interpolasi parabola "
              "(di bawah 1 sampel). Ini cara pembaca EKG Tugas Akhir (fungsi <i>_dominant_beats</i>): bentuk QRS dibandingkan dengan templatnya.",
              "<b>F2: gerbang morfologi.</b> Detak \"dominan\" bila korelasinya dengan templat ≥ 0,8 dan ukurannya 0,6–1,67 kali templat. "
              "Window ditolak (\"sinyal buruk\", tidak ada alarm) bila kurang dari 60% detaknya dominan. Semua ambang diambil dari Tugas "
              "Akhir, tidak disetel di data mana pun.",
              "<b>Tidak dipakai:</b> aturan detak terlewat dan pembuangan detak prematur dari Tugas Akhir. Keduanya bisa menghapus "
              "ketidakteraturan yang justru menjadi tanda AF.",
              "<b>Evaluasi.</b> (a) AFDB dengan protokol penuh: dataset dibangun ulang dengan F1, MicroTransformer resep v3.1 dilatih ulang 5 "
              "ulangan × 5 fold pasien resmi, bias cross-fit. (b) Paket v3.1 apa adanya (drop-in) di data latihan fisik EPFL, Wrist, EPHNOGRAM "
              "(v3.3) dan di empat data eksternal (CinC 2017, MIT-BIH, LTAFDB, CPSC 2021; amandemen A, dikunci sebelum hasil dilihat). Window "
              "yang gagal cek kualitas dihitung \"tidak ada alarm\"."):
        y = L.para(t, y)
    # ── 4. aktivitas ──
    y = L.page("Hasil: latihan fisik (sinus, alarm = palsu)", "EPFL, Wrist, EPHNOGRAM mode perangkat; paket v3.1 dengan tiga front-end")
    y = L.chart(TMP / "aktivitas.png", y, 180)
    dt = R["latihan_data"]
    rows = [[s, thou(dt[s]["window"]), f"{pct(dt[s]['lolos QC lama'], 0)} → {pct(dt[s]['lolos QC f1'], 0)}",
             f"{fnum(dt[s]['CV median lama'])} → {fnum(dt[s]['CV median f1'])}", pct(dt[s]["lolos gerbang F2 (dari lolos QC F1)"], 0),
             (f"{fnum(R['latihan_jitter'][s]['jitter lama (ms, median rekaman)'], 1)} → {fnum(R['latihan_jitter'][s]['jitter F1'], 1)}"
              if s in R["latihan_jitter"] else "–")] for s in ("EPFL", "Wrist", "EPHNOGRAM")]
    y = L.table(["Sumber", "Window", "Lolos cek kualitas", "CV IBI median", "Lolos gerbang F2", "Jitter (ms)"], rows, y,
                widths=[L.CW * .17, L.CW * .12, L.CW * .19, L.CW * .18, L.CW * .17, L.CW * .17])
    h2, h3, h4 = R["H2"], R["H3b"], R["H4_aktivitas"]
    rows = [["H2: alarm < 100 bpm (Wrist + EPHNOGRAM), lama → F1", f"{pct(h2['alarm_lama'])} → {pct(h2['alarm_f1'])}",
             f"{sg(h2['delta'] * 100, 1)} poin [{sg(h2['ci'][0] * 100, 1)}; {sg(h2['ci'][1] * 100, 1)}]", fnum(h2["p_kurang"], 3)],
            ["H4: alarm < 100 bpm, F1 → F2", f"{pct(h4['alarm_f1'])} → {pct(h4['alarm_f2'])}",
             f"{sg(h4['delta'] * 100, 1)} poin [{sg(h4['ci'][0] * 100, 1)}; {sg(h4['ci'][1] * 100, 1)}]", fnum(h4["p_kurang"], 3)],
            ["H3b: alarm ≥ 120 bpm lolos aturan produk, lama → F1", f"{pct(h3['sisa_lama'])} → {pct(h3['sisa_f1'])}",
             f"{sg(h3['delta'] * 100, 1)} poin [{sg(h3['ci'][0] * 100, 1)}; {sg(h3['ci'][1] * 100, 1)}]", fnum(h3["p_kurang"], 3)]]
    y = L.table(["Uji (bootstrap subjek berpasangan)", "Alarm", "Δ [CI 95%]", "P(Δ < 0)"], rows, y,
                widths=[L.CW * .44, L.CW * .17, L.CW * .25, L.CW * .14])
    a1 = R["alarm_120"]
    y = L.para(f"<b>Sinus takikardia ≥ 120 bpm tetap dialarmkan:</b> v3.1 lama {pct(a1['v3.1 lama']['alarm'])}, + F1 "
               f"{pct(a1['v3.1 + F1']['alarm'])}, + F2 {pct(a1['v3.1 + F2']['alarm'])}, model fold F1 {pct(a1['fold F1']['alarm'])}. "
               "Front-end tidak menyelesaikan masalah utama v3.2/v3.3; ia hanya membuat aturan produk (≥ 110 bpm & CV ≤ 0,05) lebih "
               "sering berlaku. Di EPFL, F1 membuat lebih banyak window bisa dianalisis, sehingga jumlah alarm mentah justru naik "
               f"({pct(R['H3b_per_sumber']['EPFL']['alarm lama'], 0)} → {pct(R['H3b_per_sumber']['EPFL']['alarm F1'], 0)}); sesudah aturan "
               f"produk sisanya turun ({pct(R['H3b_per_sumber']['EPFL']['sisa lama'], 0)} → {pct(R['H3b_per_sumber']['EPFL']['sisa F1'], 0)}).", y)
    # ── 5. eksternal ──
    y = L.page("Hasil: empat data eksternal (amandemen A)", "Paket v3.1 2 kelas (AF vs bukan AF); bootstrap subjek berpasangan 2.000×")
    y = L.chart(TMP / "eksternal.png", y, 165)
    rows = []
    for ds in DS:
        e = eks[ds]
        for nama, kunci in (("F1", "f1_vs_lama"), ("F2", "f2_vs_lama")):
            b = e[kunci]
            rows.append([f"{ds} / {nama}" if nama == "F1" else f"   {nama}",
                         f"{fnum(b['titik_b']['Se'][0])} → {fnum(b['titik_a']['Se'][0])}", f"P = {fnum(b['Se'][3])}, {_vonis2(b['Se'][3])}",
                         f"{fnum(b['titik_b']['Sp'][0])} → {fnum(b['titik_a']['Sp'][0])}", f"{fnum(b['titik_b']['Macro-F1'][0])} → {fnum(b['titik_a']['Macro-F1'][0])}"])
    y = L.table(["Data / front-end", "Se lama → baru", "Vonis Se", "Sp lama → baru", "Macro-F1"], rows, y,
                widths=[L.CW * .2, L.CW * .18, L.CW * .26, L.CW * .18, L.CW * .18])
    n_txt = "; ".join(f"{ds}: {thou(eks[ds]['window'])} window, {eks[ds]['subjek_AF']} subjek AF" for ds in DS)
    y = L.para(f"<i>{n_txt}.</i>", y, "small")
    bands = list(eks["LTAFDB"]["alarm_pita_nonaf"])
    rows = [[ds] + [" / ".join(pct(eks[ds]["alarm_pita_nonaf"][b][fe], 0) if eks[ds]["alarm_pita_nonaf"][b][fe] is not None else "–"
                                for fe in ("lama", "f1", "f2")) for b in bands] for ds in DS]
    y = L.table(["Alarm non-AF (lama / F1 / F2)"] + [f"{b.replace('∞', '')}{'+' if '∞' in b else ''} bpm" for b in bands], rows, y,
                widths=[L.CW * .24] + [L.CW * .19] * 4)
    ps = eks["MIT-BIH"]["per_subjek_af"]
    y = L.para("<b>Mengapa F1 gagal H6.</b> Di MIT-BIH penurunannya datang dari dua subjek: 210 (Se "
               f"{fnum(ps['210']['lama'])} → {fnum(ps['210']['f1'])}) dan 217 ({fnum(ps['217']['lama'])} → {fnum(ps['217']['f1'])}, hanya "
               f"{ps['217']['n']} window AF). Seri bootstrap hanya 4%, jadi bukan artefak. Di tiga data lain Se F1 setara. F2 menolak window AF "
               f"dengan QRS berubah-ubah (MIT-BIH 222: {pct(ps['222']['ditolak_gerbang'], 0)} window AF ditolak), sehingga Se turun di "
               "keempat data, sementara spesifisitas naik di keempatnya.", y)
    # ── 6. per pasien & uji stres ──
    y = L.page("Per pasien AFDB, episode, dan uji stres takikardia", "Protokol butir 5, 7, dan 9")
    ap = R["afl_per_pasien"]
    rows = [[p, thou(ap["v31"][p]["n"]), fnum(ap["v31"][p]["recall"], 2), fnum(ap["f1"][p]["recall"], 2), fnum(ap["f2"][p]["recall"], 2)]
            for p in ap["v31"]]
    y = L.table(["Pasien AFL", "Window AFL", "Recall v3.1", "Recall F1", "Recall F2"], rows, y, widths=[L.CW * .2] * 5)
    ep = R["episode"]
    rows = [[f"{kl}", *[f"{fnum(ep[x][kl]['se_episode'], 2)} / {fnum(ep[x][kl]['ppv_episode'], 2)} / {fnum(ep[x][kl]['palsu_per_jam'], 2)}"
                        for x in ("v31", "f1", "f2")]] for kl in ("AFIB", "AFL")]
    y = L.table(["Episode (Se / +P / palsu per jam)", "v3.1", "F1", "F2"], rows, y, widths=[L.CW * .31, L.CW * .23, L.CW * .23, L.CW * .23])
    fp = R["afl_palsu"]
    y = L.para(f"<b>AFL palsu</b> (per ulangan): dari Normal pasien tanpa AFL {fnum(fp['v31']['Normal, pasien tanpa AFL'], 1)} → "
               f"{fnum(fp['f1']['Normal, pasien tanpa AFL'], 1)} (F1), dari AFIB pasien AFL {fnum(fp['v31']['AFIB, pasien AFL'], 1)} → "
               f"{fnum(fp['f1']['AFIB, pasien AFL'], 1)}. Jumlah absolutnya kecil, tetapi AFL palsu dari Normal naik ±4×. Bias cross-fit F1 "
               f"median {fnum(R['bias_f1']['median'][1], 2)} (AFIB) dan {fnum(R['bias_f1']['median'][2], 2)} (AFL), tidak ada yang di batas grid.", y)
    y = L.chart(TMP / "stres.png", y, 135)
    st = R["stres"]
    y = L.para(f"<b>Uji stres sintetis.</b> Model fold F1 berhenti menebak Normal pada {st['tanpa variasi']['ambang']} bpm (tanpa variasi) "
               f"dan {st['Gauss sd 20 ms']['ambang']} bpm (variasi Gauss 20 ms); paket v3.1: 115 bpm. Front-end tidak berperan pada IBI sintetis, "
               "jadi selisih ini dari latih ulang, bukan dari F1.", y)
    # ── 7. kesimpulan ──
    y = L.page("Kesimpulan, langkah berikutnya, batasan", "")
    for t in ("<b>Kesimpulan.</b> Ide dari Tugas Akhir bekerja sesuai mekanismenya: F1 menghapus lompatan R/S sehingga jitter di EPFL turun "
              "±4× (10,7 → 2,8 ms; Wrist 14,7 → 10,4 ms) dan lebih banyak window bisa dianalisis. AFDB tidak berubah secara berarti, dan bobot paket v3.1 bisa dipakai "
              "tanpa latih ulang. Namun aturan yang dikunci menuntut sensitivitas AF tidak memburuk di <i>setiap</i> data eksternal. F1 gagal "
              "tipis di MIT-BIH (dua subjek), sedangkan F2 menukar sensitivitas dengan spesifisitas di keempat data. Paket produksi tetap v3.1.",
              "<b>Yang tidak bisa disimpulkan.</b> Kegagalan F1 sangat tipis (−0,7 poin; dengan seri dihitung separuh, P ≈ 0,06). Itu bukan "
              "bukti bahwa F1 merugikan, tetapi juga bukan bukti aman. Menyetel ulang ambang atau konteks templat di data yang sama tidak lagi "
              "independen.",
              "<b>Langkah berikutnya.</b> (1) Lihat window AF subjek 210 dan 217 MIT-BIH yang berubah keputusan: apakah penyelarasan memindahkan "
              "titik ke gelombang lain (QRS aberan, artefak)? (2) Bila perbaikannya jelas (mis. tidak menggeser detak yang korelasinya rendah), "
              "kunci rancangan baru dan uji di data yang belum dipakai: rekaman perangkat SIHEDAF. (3) Gerbang F2 lebih cocok sebagai keluaran "
              "\"sinyal buruk\" yang terpisah dari alarm, dengan ambang dari data perangkat. (4) Masalah sinus takikardia tetap butuh detektor AF "
              "berbasis EKG (v16.2, D-1), bukan front-end RR.",
              "<b>Batasan.</b> Data latihan fisik adalah orang sehat tanpa label ritme (diasumsikan sinus); hipotesis H2/H3 arahnya sudah terlihat "
              "di probe sebelum rancangan dikunci, jadi bukan uji independen. Data eksternal dinilai dengan paket v3.1 yang dilatih di front-end "
              "lama (drop-in). Semua data publik direkam dengan alat lain dari perangkat SIHEDAF."):
        y = L.para(t, y)
    # ── 8. analisis pasca-hasil ──
    X = J["X"]
    y = L.page("Analisis pasca-hasil: mengapa F1 gagal di MIT-BIH", "Eksploratif, dilakukan sesudah hasil; data sudah dipakai, jadi bukan uji independen")
    y = L.chart(TMP / "mitbih210.png", y, 150)
    y = L.para("<i>MIT-BIH 210, 825–835 s. Huruf = anotasi detak (N normal, V ektopik ventrikel). Segitiga = front-end lama, lingkaran = "
               "F1, silang = F1b. Pada detak V, F1 melompat ±50–60 ms ke tonjolan kecil sebelum kompleks.</i>", y, "small")
    y = L.para("<b>Penyebab.</b> Di 9 dari 10 window AF yang hilang (210 dan 217) ada detak V yang digeser ke batas lag (−3 sampel). "
               "Templatnya median detak <i>normal</i>, sehingga bentuk PVC tidak punya puncak korelasi yang benar. IBI di sekitar PVC "
               "berubah dan model menggeser keputusan ke Normal; beberapa window memang sudah di tepi (probabilitas AFIB 0,50–0,58 dengan front-end lama). "
               "Penyebabnya bukan AF yang \"dihaluskan\": CV IBI window itu justru naik.", y)
    g = X["galat_mitbih"]
    rows = [[c, *[f"{fnum(g[fe][c]['median'], 1)} ms; {pct(g[fe][c]['lebih_30ms'])}" for fe in ("lama", "f1", "f1b")]] for c in ("N", "V", "/")]
    y = L.table(["Jenis detak MIT-BIH (median |galat|; > 30 ms)", "lama", "F1", "F1b"], rows, y,
                widths=[L.CW * .4, L.CW * .2, L.CW * .2, L.CW * .2])
    rows = []
    for ds in DS:
        a, b = X["eksternal"][ds]["f1_vs_lama"], X["eksternal"][ds]["f1b_vs_lama"]
        rows.append([ds, f"{sg(a['Se'][0] * 100, 2)} (P = {fnum(a['Se'][3])})", f"{sg(b['Se'][0] * 100, 2)} (P = {fnum(b['Se'][3])})",
                     f"{sg(a['Sp'][0] * 100, 1)} / {sg(b['Sp'][0] * 100, 1)}"])
    y = L.table(["Data eksternal", "Δ Se F1 (poin)", "Δ Se F1b (poin)", "Δ Sp F1 / F1b"], rows, y,
                widths=[L.CW * .22, L.CW * .28, L.CW * .28, L.CW * .22])
    lt, ad = X["latihan"], X["afdb_dropin"]["banding"]
    y = L.para(f"<b>F1b: geser hanya detak dominan</b> (korelasi ≥ 0,8, ukuran 0,6–1,67; satu varian, ditetapkan sebelum dihitung). Masalah "
               f"PVC hampir hilang dan Se MIT-BIH pulih, tetapi di CPSC 2021 Se turun {sg(X['eksternal']['CPSC 2021']['f1b_vs_lama']['Se'][0] * 100, 2)} "
               f"poin dengan P = {fnum(X['eksternal']['CPSC 2021']['f1b_vs_lama']['Se'][3])} (341 ribu window, jadi selisih sekecil ini sudah "
               f"\"signifikan\"). Jitter EPFL {fnum(lt['jitter']['EPFL'], 1)} ms (F1 2,8; lama 10,7), lolos QC EPFL {pct(lt['epfl_lolos_qc'], 0)}, "
               f"alarm &lt; 100 bpm {pct(lt['H2']['f1b'])} (lama {pct(lt['H2']['lama'])}). AFDB drop-in Macro-F1 {sg(ad['Macro-F1']['delta'])} "
               f"(P = {fnum(ad['Macro-F1']['p'])}).", y)
    y = L.para("<b>Pelajaran untuk rancangan berikutnya.</b> (1) Templat dari detak normal tidak boleh dipakai untuk menggeser detak yang "
               "bentuknya lain; F1b memperbaikinya. (2) Kriteria \"bukan lebih buruk\" tanpa margin terlalu ketat untuk data sebesar LTAFDB/CPSC: "
               "selisih −0,2 poin pun gagal. Rancangan berikutnya sebaiknya memakai margin non-inferioritas yang ditetapkan sebelumnya "
               "(mis. −1 poin) dan diuji di data perangkat, karena ketujuh data publik sudah terpakai.", y)
    # ── 9. lampiran ──
    y = L.page("Lampiran teknis: bukti, penyimpangan, reproduksi", "")
    pr = R["paritas"]
    for t in (f"<b>Bukti kendali mutu.</b> (1) Tes pytest 7/7 lolos, termasuk salinan detektor = <i>hrv_core.detect_rpeaks</i> bit demi bit. "
              f"(2) Ekstraktor v3.4 dengan front-end lama menghasilkan dataset AFDB identik bit demi bit dengan <i>dataset_v2_detector.npz</i> "
              f"({thou(sum(R['window_afdb']['lama']))} window). (3) Uji paritas training: fungsi job v3.4 pada dataset lama mereproduksi OOF "
              f"MicroTransformer v3 ulangan 0 dengan selisih maksimum {fnum(pr['selisih_maks'], 1)}. (4) Window latihan fisik front-end lama "
              "identik dengan v3.3. (5) Data eksternal: front-end lama di-assert sama dengan <i>signal_to_windows</i> di setiap rekaman. "
              "(6) <i>build_report.py --extract</i> menghitung ulang angka utama dari cache dan meng-assert sama dengan ringkasan notebook.",
              f"<b>Pra-registrasi.</b> RANCANGAN.md dikunci 10 November 2026 03:43 sebelum training. Amandemen A (data eksternal) 04:01, sebelum "
              "hasil AFDB dilihat. Penyimpangan B (pelaporan): vonis \"identik\" bila Δ = 0 di semua sampel bootstrap (F2 vs F1 pada AFL) dan "
              "tabel Se per subjek; keputusan tidak berubah. Hash di <i>logs/v3.4_praregistrasi.log</i>.",
              f"<b>Window AFDB.</b> Lama (Normal, AFIB, AFL): {', '.join(thou(v) for v in R['window_afdb']['lama'])}; F1: "
              f"{', '.join(thou(v) for v in R['window_afdb']['f1'])}. Window yang lolos gerbang F2: Normal {pct(R['gerbang_afdb']['Normal'])}, "
              f"AFIB {pct(R['gerbang_afdb']['AFIB'])}, AFL {pct(R['gerbang_afdb']['AFL'])}. Drop-in (H5): model v7 kontrol (resep v5b, front-end "
              "lama, ulangan 0) di window F1 vs OOF-nya sendiri.",
              "<b>Reproduksi.</b> <font face='Arial-Bold'>python tools/run_notebook.py v3.4_pembersih_detak/pembersih_detak.ipynb</font> "
              "(jalankan dulu <i>pembersih.bangun_afdb()</i>, <i>pembersih.jalankan_semua()</i> untuk training, dan "
              "<i>pembersih.jalankan_eksternal()</i>; semuanya bisa dilanjutkan dan dilewati bila cache ada), lalu <font face='Arial-Bold'>build_report.py --extract</font> "
              "dan tanpa argumen. Halaman 8: <font face='Arial-Bold'>python eksplorasi_ektopik.py</font> (±10 menit, menulis "
              "cache/v3_4/eksplorasi_f1b/). Butuh data/afdb, data latihan fisik v3.3, data/cinc2017, mitdb, ltafdb, cpsc2021, cache v3, "
              "v3.2, v3.3, v7."):
        y = L.para(t, y)
    lk = R["lingkungan"]
    y = L.para(f"Lingkungan: PC desktop, GPU {lk['gpu']}, commit {lk['commit']} (+ perubahan belum di-commit), Python {lk['pustaka']['python']}, "
               f"numpy {lk['pustaka']['numpy']}, torch {lk['pustaka']['torch']}. Dataset AFDB 3 menit (6 proses), training 21 menit "
               f"(36–50 s/fold), data eksternal 5 menit (4 proses), notebook {fnum(R['durasi_notebook_menit'], 0)} menit.", y, "small")
    L.selesai()
    print("PDF:", OUT, L.page_no, "halaman")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
