"""Laporan PDF v16.1: detektor AF 2 kelas dari tahap 1 v16 (ECGFounder tanpa penyetelan).

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v16.1/ringkasan.json (sel akhir deteksi_af.ipynb), hitung ulang Macro-F1 2 kelas D-0 di AFDB dan PTB-XL
     dari cache lalu cocokkan (assert), gambar grafik → laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_v16_1_dua_kelas.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v16_1_dua_kelas.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, Laporan, ci, fnum, pct, sg, sumbu_koma, thou, vonis  # noqa: E402

TANGGAL = "6 November 2026"
SISTEM = ("v1 (argmax)", "v3.1", "BiGRU bawaan (v11.1)", "D-0", "D-L")
NAMA = {"v1 (argmax)": "v1 (argmax)", "v3.1": "v3.1 (referensi)", "BiGRU bawaan (v11.1)": "BiGRU bawaan (v11.1)",
        "D-0": "D-0 (primer)", "D-L": "D-L (D-0 + AFL v3.1)", "v3.1 paket": "v3.1 paket"}


def extract():
    import numpy as np
    import deteksi_af as A
    from deteksi_af import D, DK, H16
    R = json.loads((REPO / "cache/v16.1/ringkasan.json").read_text(encoding="utf-8"))
    d = D.muat()
    d1 = A.d1_afdb()
    y2 = (d.y > 0).astype(np.int64)
    W = A.cv.bootstrap_weights(d.strata, A.Ev.N_BOOT, seed=D.SEED)
    m, _ = DK.ringkas(DK.cm_pasien(y2, np.repeat((d1 > 0).astype(np.int64)[None], len(d.folds), 0), d.records, d.patients), W)
    assert abs(m["Macro-F1"][0] - R["afdb"]["D-0"]["Macro-F1"][0]) < 1e-12
    P = H16.ptbxl()
    _, d1p = H16.tahap1(P["logit"])
    b = A.banding2_pasien((P["y"] > 0).astype(np.int64), d1p > 0, P["p31"].argmax(1) != 0, P["pasien"])
    assert abs(b["titik_a"]["Macro-F1"][0] - R["ptbxl"]["D-0"]["titik_a"]["Macro-F1"][0]) < 1e-12
    TMP.mkdir(exist_ok=True)
    (TMP / "v16_1.json").write_text(json.dumps({"R": R}, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    grafik(R)
    print("extract selesai; assert Macro-F1 2 kelas D-0 (AFDB, PTB-XL) cocok dengan ringkasan.json")


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
    # 1. alarm per pita detak, latihan fisik
    lat = R["latihan"]
    pita = list(lat["v3.1 paket"]["pita"])
    fig, ax = plt.subplots(figsize=(7.6, 2.7), dpi=200)
    for k, col, mk, ls in (("v3.1 paket", BIRU, "o", "-"), ("BiGRU bawaan (v11.1)", GRAY, "D", ":"), ("D-L", HIJAU, "^", "--"),
                           ("D-0", ORANYE, "s", "-")):
        ys = [lat[k]["pita"][p]["alarm"] * 100 for p in pita]
        ax.fill_between(range(len(pita)), [lat[k]["pita"][p]["ci"][0] * 100 for p in pita],
                        [lat[k]["pita"][p]["ci"][1] * 100 for p in pita], color=col, alpha=.10, lw=0)
        ax.plot(range(len(pita)), ys, ls, marker=mk, color=col, ms=6.5, mec="white", mew=1.1, lw=2, label=NAMA[k])
    ax.set_xticks(range(len(pita)), [p.replace("–∞", "+") for p in pita], fontsize=8.5)
    ax.set_ylabel("alarm AF (%)"); ax.set_ylim(0, 105); ax.set_xlabel("detak window (EKG latihan fisik asli, mode perangkat 50 Hz)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.16)); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "alarm_pita.png"); plt.close(fig)
    # 2. kurva operasi: sensitivitas (kiri) dan alarm palsu (kanan) per ambang
    k, ki = R["kurva"], R["kurva_icentia_tambahan"]
    t = [x["t"] for x in k]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.8), dpi=200)
    for ys, col, mk, ls, lab in (([x["se_afdb"] for x in k], BIRU, "o", "-", "AFDB"), ([x["se_ptb"] for x in k], ORANYE, "s", "--", "PTB-XL"),
                                 ([x["se"] for x in ki], HIJAU, "^", ":", "Icentia11k*")):
        a1.plot(t, ys, ls, marker=mk, color=col, ms=5.5, mec="white", mew=1, lw=1.8, label=lab)
    a1.set_ylabel("sensitivitas AF"); a1.set_ylim(0, 1.02)
    for ys, col, mk, ls, lab in (([1 - x["sp_afdb"] for x in k], BIRU, "o", "-", "Normal AFDB"),
                                 ([x["alarm_st"] for x in k], ORANYE, "s", "--", "sinus takikardia asli ≥ 120"),
                                 ([x["alarm_stach"] for x in k], HIJAU, "^", ":", "STACH PTB-XL")):
        a2.plot(t, ys, ls, marker=mk, color=col, ms=5.5, mec="white", mew=1, lw=1.8, label=lab)
    a2.set_ylabel("alarm palsu"); a2.set_ylim(0, .3)
    for a in (a1, a2):
        a.axvline(0, color=GRAY, lw=1, ls=":"); a.set_xlabel("ambang t (AF bila d1 > t); garis titik = D-0")
        a.legend(frameon=False, fontsize=7.5, loc="lower left" if a is a1 else "upper right")
        a.spines[["top", "right"]].set_visible(False); a.grid(color="#E6ECF1"); sumbu_koma(a); a.set_xticks(t)
    fig.tight_layout(); fig.savefig(TMP / "kurva.png"); plt.close(fig)
    # 3. sensitivitas per pasien ber-AF: v3.1 vs D-0
    pp = R["per_pasien"]
    pas = [p for p in pp["D-0"] if pp["D-0"][p]["se"] is not None]
    fig, ax = plt.subplots(figsize=(7.6, 2.6), dpi=200)
    x = range(len(pas))
    for i, p in enumerate(pas):
        ax.plot([i, i], [pp["v3.1"][p]["se"], pp["D-0"][p]["se"]], color="#C9D3DD", lw=1.5, zorder=1)
    ax.scatter(x, [pp["v3.1"][p]["se"] for p in pas], color=BIRU, marker="o", s=26, label="v3.1", zorder=2)
    ax.scatter(x, [pp["D-0"][p]["se"] for p in pas], color=ORANYE, marker="s", s=26, label="D-0", zorder=3)
    ax.set_xticks(list(x), pas, rotation=60, fontsize=7.5); ax.set_ylabel("sensitivitas AF"); ax.set_ylim(.55, 1.03)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "se_pasien.png"); plt.close(fig)


def _d(v, nd=3):
    return f"{sg(v[0], nd)} [{sg(v[1], nd)}…{sg(v[2], nd)}]"


def build():
    R = json.loads((TMP / "v16_1.json").read_text(encoding="utf-8"))["R"]
    A, B, T4, BT = R["afdb"], R["banding"], R["tanpa_04936"], R["tanpa_04936_banding"]
    lat, ptb, ico, kr, ep = R["latihan"], R["ptbxl"], R["icentia"], R["kriteria"], R["episode"]
    L = Laporan(OUT, "v16.1: detektor AF 2 kelas (ECGFounder)", "SIHEDAF  /  V16.1  /  DETEKTOR AF 2 KELAS",
                "v16.1_dua_kelas/deteksi_af.ipynb, cache/v16.1/", TANGGAL)
    # ── 1. ringkasan ──
    y = L.page("Ringkasan: sangat spesifik, gagal di data wearable", "Rancangan dikunci 6 November 2026 sebelum metrik baru dihitung")
    y = L.kartu([(fnum(A["D-0"]["Macro-F1"][0]), "Macro-F1 AFDB 2 kelas", f"v3.1 {fnum(A['v3.1']['Macro-F1'][0])}; P = {fnum(B['D-0']['Macro-F1'][3], 2)}"),
                 (pct(lat["D-0"]["alarm"]), "alarm sinus takikardia", f"≥ 120 bpm; v3.1 {pct(lat['v3.1 paket']['alarm'])}"),
                 (fnum(ptb["D-0"]["titik_a"]["Macro-F1"][0]), "Macro-F1 PTB-XL", f"v3.1 {fnum(ptb['D-0']['titik_b']['Macro-F1'][0])}; lebih baik"),
                 (fnum(ico["D-0"]["titik_a"]["Macro-F1"][0]), "Macro-F1 Icentia11k", f"v3.1 {fnum(ico['D-0']['titik_b']['Macro-F1'][0])}; lebih buruk")], y)
    y = L.para("<b>Pertanyaan.</b> Bila SIHEDAF cukup melaporkan \"AF ya/tidak\" (AFIB dan AFL digabung), apakah keputusan sinus vs AF "
               "dari ECGFounder (tahap 1 v16, tanpa penyetelan dan tanpa training) lebih aman daripada v3.1 tanpa kehilangan terlalu banyak AF?", y)
    y = L.para("<b>Jawaban singkat: belum.</b> D-0 jauh lebih spesifik: alarm palsu pada sinus takikardia asli hampir nol, pada ritme "
               f"Normal AFDB {pct(1 - A['D-0']['Sp'][0])} (v3.1 {pct(1 - A['v3.1']['Sp'][0])}), episode AF palsu "
               f"{fnum(ep['v3.1']['palsu_per_jam'], 1)} → {fnum(ep['D-0']['palsu_per_jam'], 1)} per jam. Macro-F1 2 kelas setara v3.1 di AFDB "
               f"(juga tanpa 04936) dan lebih baik di PTB-XL. Tetapi sensitivitasnya lebih rendah di ketiga data, dan di Icentia11k (wearable) "
               f"turun {fnum(ico['D-0']['titik_b']['Se'][0])} → {fnum(ico['D-0']['titik_a']['Se'][0])}, sehingga Macro-F1 lebih buruk. "
               "Menurut kriteria yang dikunci, D-0 <b>belum menjadi kandidat</b> untuk uji perangkat.", y)
    rows = [["AFDB 2 kelas (bukan lebih buruk)", "lolos" if kr["afdb"] else "gagal", f"P = {fnum(B['D-0']['Macro-F1'][3])}"],
            ["AFDB tanpa 04936", "lolos" if kr["tanpa_04936"] else "gagal", f"P = {fnum(BT['D-0']['Macro-F1'][3])}"],
            ["PTB-XL 2 kelas", "lolos" if kr["ptbxl"] else "gagal", f"P = {fnum(ptb['D-0']['Macro-F1'][3])}"],
            ["Icentia11k 2 kelas", "lolos" if kr["icentia"] else "gagal", f"P = {fnum(ico['D-0']['Macro-F1'][3])}"],
            ["Alarm sinus takikardia asli ≤ separuh v3.1", "lolos" if kr["sinus_takikardia"] else "gagal",
             f"{pct(lat['D-0']['alarm'])} vs {pct(lat['v3.1 paket']['alarm'])}"],
            ["Kandidat uji perangkat", "ya" if kr["kandidat_uji_perangkat"] else "tidak", "Icentia11k gagal"]]
    y = L.table(["Kriteria yang dikunci (D-0)", "Hasil", "Angka"], rows, y, widths=[L.CW * .5, L.CW * .15, L.CW * .35], sorot=(6,))
    y = L.para("<b>Temuan.</b> (1) Sensitivitas AF D-0 lebih rendah dari v3.1 di AFDB "
               f"({fnum(A['D-0']['Se'][0])} vs {fnum(A['v3.1']['Se'][0])}), PTB-XL ({fnum(ptb['D-0']['titik_a']['Se'][0])} vs "
               f"{fnum(ptb['D-0']['titik_b']['Se'][0])}) dan Icentia11k; yang hilang terutama AFL (PTB-XL: {pct(R['ptbxl_kelas']['D-0']['AFL'], 0)} "
               f"AFL terdeteksi vs {pct(R['ptbxl_kelas']['v3.1 paket']['AFL'], 0)}). (2) D-L (menambah AFL v3.1) memulihkan sebagian "
               f"sensitivitas tetapi alarm sinus takikardia naik lagi ke {pct(lat['D-L']['alarm'])}. (3) Kurva ambang menunjukkan "
               "ambang yang lebih rendah menukar sedikit spesifisitas dengan sensitivitas; tidak dipilih di sini (lihat halaman 7).", y)
    # ── 2. perbandingan ──
    y = L.page("Perbandingan dengan versi sebelumnya dan v1", "AFDB 2 kelas: pooled OOF 5 ulangan, CI 95% bootstrap pasien, berpasangan")
    rows = [[NAMA[k], f"{fnum(A[k]['Macro-F1'][0])} {ci(*A[k]['Macro-F1'][1:])}", f"{fnum(A[k]['Se'][0])} {ci(*A[k]['Se'][1:])}",
             f"{fnum(A[k]['Sp'][0])} {ci(*A[k]['Sp'][1:])}", fnum(A[k]["PPV"][0])] for k in SISTEM]
    y = L.table(["AFDB 2 kelas", "Macro-F1 [CI]", "Sensitivitas [CI]", "Spesifisitas [CI]", "PPV"], rows, y,
                widths=[L.CW * .27, L.CW * .2, L.CW * .2, L.CW * .2, L.CW * .13], sorot=(4,))
    rows = []
    for k, lab in (("D-0", "D-0 − v3.1"), ("D-L", "D-L − v3.1"), ("BiGRU bawaan (v11.1)", "BiGRU (v11.1) − v3.1"),
                   ("v1 (argmax)", "v1 − v3.1"), ("D-0 vs BiGRU", "D-0 − BiGRU (v11.1)"), ("D-L vs D-0", "D-L − D-0")):
        v = B[k]
        rows.append([lab, f"{_d(v['Macro-F1'])}, P = {fnum(v['Macro-F1'][3])}", f"{sg(v['Se'][0])}, P = {fnum(v['Se'][3])}",
                     f"{sg(v['Sp'][0])}, P = {fnum(v['Sp'][3])}", vonis(v["Macro-F1"][3])])
    y = L.table(["Berpasangan", "Δ Macro-F1 [CI], P(Δ > 0)", "Δ Se, P", "Δ Sp, P", "Vonis Macro-F1"], rows, y,
                widths=[L.CW * .2, L.CW * .34, L.CW * .16, L.CW * .16, L.CW * .14])
    bs = R["bersarang_v16"]
    y = L.para(f"<b>Bootstrap bersarang</b> (v16; bias v3.1 dipilih ulang di tiap sampel, D-0 tanpa parameter): Δ {sg(bs['delta'])} "
               f"[{sg(bs['ci_delta'][0])}…{sg(bs['ci_delta'][1])}], P = {fnum(bs['p'], 2)}.", y, "small")
    rows = [[NAMA[k], f"{fnum(T4[k]['Macro-F1'][0])} {ci(*T4[k]['Macro-F1'][1:])}", fnum(T4[k]["Se"][0]), fnum(T4[k]["Sp"][0]),
             "–" if k == "v3.1" else f"{sg(BT[k]['Macro-F1'][0])}, P = {fnum(BT[k]['Macro-F1'][3])}"] for k in SISTEM]
    y = L.table(["Tanpa 04936 (H2)", "Macro-F1 [CI]", "Se", "Sp", "Δ vs v3.1, P"], rows, y,
                widths=[L.CW * .27, L.CW * .22, L.CW * .13, L.CW * .13, L.CW * .25], sorot=(4,))
    y = L.para("v3.1 dan BiGRU di-assert sama dengan ringkasan v11.1, D-0 sama dengan H2b v16. Aturan 2 kelas pembanding: bias 3 kelas "
               "cross-fit lalu AFIB + AFL digabung (v11.1).", y, "small")
    # ── 3. metode ──
    y = L.page("Metode dan pra-registrasi", "RANCANGAN.md dikunci sebelum metrik baru; sha256 di logs/v16.1_praregistrasi.log")
    rows = [["D-0 (primer)", "AF bila d1 = log((pAF + pAFL) / pN) > 0 dari skor zero-shot ECGFounder 1-lead (v13/v16)", "tidak ada"],
            ["D-L (sekunder)", "D-0, atau v3.1 menyebut AFL (\"perlu dicek\" v16 dihitung sebagai alarm)", "bias v3.1"],
            ["v3.1, BiGRU (v11.1), v1", "bias 3 kelas cross-fit (v1: argmax), lalu AFIB + AFL digabung", "bias 3 kelas"]]
    y = L.table(["Sistem", "Keputusan AF", "Parameter"], rows, y, widths=[L.CW * .2, L.CW * .62, L.CW * .18])
    for t in ("<b>Versi minor.</b> Tidak ada training dan tidak ada ambang yang disetel. ECGFounder dipakai beku (bobot dan kode resmi, "
              "lisensi MIT, dicek SHA256). Data dan cache sama dengan v16: AFDB (165.978 window detektor 50 Hz, fold resmi 5×5), EKG "
              f"latihan fisik asli v3.3, PTB-XL ({thou(R['n']['ptbxl'])} window, AF {thou(R['n']['ptbxl_af'])}) dan Icentia11k "
              f"({thou(R['n']['icentia'])} window, AF {thou(R['n']['icentia_af'])}, {thou(R['n']['icentia_pasien'])} pasien; akademik saja).",
              "<b>Kriteria (dikunci).</b> D-0 kandidat uji perangkat bila Macro-F1 2 kelas tidak \"lebih buruk\" dari v3.1 di AFDB, AFDB tanpa "
              "04936, PTB-XL, dan Icentia11k, dan alarm sinus takikardia asli ≤ separuh v3.1. Penurunan sensitivitas dilaporkan apa adanya, "
              "tanpa margin tambahan. Dua bagian sudah diketahui dari v16 sebelum dikunci (AFDB dan sinus takikardia).",
              "<b>Protokol.</b> Butir 1–4 untuk 2 kelas (pola v11.1); butir 5: sensitivitas dan alarm per pasien; butir 7: episode AF "
              "gabungan gaya EC57; butir 9: sinus takikardia asli per pita detak dan alarm Normal AFDB per pita (uji stres RR sintetis tidak "
              "berlaku karena D-0 membaca EKG); butir 10: bootstrap bersarang (v16) dan tanpa 04936. Data eksternal: bootstrap pasien "
              "berpasangan 2.000× vs paket v3.1 (argmax, digabung).",
              "<b>Penyimpangan tercatat.</b> Setelah run pertama, kurva ambang juga dihitung untuk Icentia11k (RANCANGAN bagian 8) karena "
              "hanya kriteria Icentia yang gagal. Deskriptif; D-0 dan kriteria tidak berubah.",
              "<b>Istilah.</b> Sensitivitas (Se) = bagian window AF yang dialarmkan. Spesifisitas (Sp) = bagian window Normal yang tidak "
              "dialarmkan. PPV = bagian alarm yang benar AF. OOF, CI, dan bias: lihat laporan v16."):
        y = L.para(t, y)
    # ── 4. alarm palsu ──
    y = L.page("Hasil: alarm palsu (butir 9) dan episode", "Latihan fisik: CI bootstrap subjek; AFDB: window Normal per pita detak")
    y = L.chart(TMP / "alarm_pita.png", y, 200)
    rows = []
    for k in ("v3.1 paket", "BiGRU bawaan (v11.1)", "D-0", "D-L"):
        v = lat[k]
        bd = v.get("banding")
        rows.append([NAMA[k], f"{pct(v['alarm'])} {ci(v['ci'][0] * 100, v['ci'][1] * 100, 1)}", pct(v["per_sumber"]["EPFL"], 0),
                     pct(v["per_sumber"]["EPHNOGRAM"], 0), pct(v["per_sumber"]["Wrist"], 0),
                     "–" if bd is None else f"{sg(bd['delta'] * 100, 1)} poin"])
    y = L.table(["Sinus takikardia asli ≥ 120 bpm", "Alarm [CI]", "EPFL", "EPHNOGRAM", "Wrist", "Δ vs v3.1"], rows, y,
                widths=[L.CW * .27, L.CW * .2, L.CW * .1, L.CW * .14, L.CW * .1, L.CW * .19], sorot=(3,))
    an = R["alarm_normal"]
    kol = list(an["v3.1"])
    rows = [[NAMA[k]] + [pct(an[k][b]["alarm"]) for b in kol] for k in SISTEM]
    y = L.table(["Normal AFDB (n window)"] + [f"{b} ({thou(an['v3.1'][b]['n'])})" for b in kol], rows, y,
                widths=[L.CW * .25] + [L.CW * .15] * 5, sorot=(4,))
    rows = [[NAMA[k], fnum(ep[k]["se_episode"], 2), fnum(ep[k]["ppv_episode"], 2), fnum(ep[k]["se_durasi"], 2), fnum(ep[k]["palsu_per_jam"], 1)]
            for k in ("v3.1", "BiGRU bawaan (v11.1)", "D-0", "D-L")]
    y = L.table(["Episode AF (EC57, AFDB)", "Se episode", "+P episode", "Se durasi", "Episode palsu/jam"], rows, y,
                widths=[L.CW * .3, L.CW * .16, L.CW * .16, L.CW * .16, L.CW * .22], sorot=(3,))
    # ── 5. eksternal ──
    y = L.page("Hasil: data eksternal (H1)", "Bootstrap pasien berpasangan 2.000× vs paket v3.1 (argmax, AFIB + AFL digabung)")
    rows = []
    for data, D_ in (("PTB-XL", ptb), ("Icentia11k", ico)):
        for k in ("D-0", "D-L"):
            v = D_[k]
            rows.append([data, NAMA[k], f"{fnum(v['titik_a']['Macro-F1'][0])} / {fnum(v['titik_b']['Macro-F1'][0])}",
                         f"{_d(v['Macro-F1'])}, P = {fnum(v['Macro-F1'][3])}", vonis(v["Macro-F1"][3]),
                         f"{fnum(v['titik_a']['Se'][0])} / {fnum(v['titik_b']['Se'][0])}", f"{fnum(v['titik_a']['Sp'][0])} / {fnum(v['titik_b']['Sp'][0])}"])
    y = L.table(["Data", "Sistem", "Macro-F1 sistem / v3.1", "Δ [CI], P(Δ > 0)", "Vonis", "Se sistem / v3.1", "Sp sistem / v3.1"], rows, y,
                widths=[L.CW * .1, L.CW * .15, L.CW * .13, L.CW * .28, L.CW * .1, L.CW * .12, L.CW * .12], sorot=(1, 3))
    pk, ik = R["ptbxl_kelas"], R["icentia_kelas"]
    rows = [[NAMA[k], pct(pk[k]["AFIB"], 0), pct(pk[k]["AFL"], 0), pct(pk[k]["STACH"]), pct(pk[k]["sinus lain"]),
             pct(ik[k]["AFIB"], 0), pct(ik[k]["AFL"], 0), pct(ik[k]["Normal"])] for k in ("v3.1 paket", "D-0", "D-L")]
    y = L.table(["Alarm AF per kelas asli", "PTB AFIB", "PTB AFL", "PTB STACH", "PTB sinus lain", "Icentia AFIB", "Icentia AFL",
                 "Icentia Normal"], rows, y, widths=[L.CW * .2] + [L.CW * .1142] * 7, sorot=(2,))
    y = L.para("<b>Membaca tabel.</b> Di PTB-XL (rekaman 10 s saat istirahat) D-0 lebih baik karena hampir tidak pernah mengalarmkan "
               f"ritme sinus (STACH {pct(pk['v3.1 paket']['STACH'])} → {pct(pk['D-0']['STACH'])}); harganya AFL ({pct(pk['v3.1 paket']['AFL'], 0)} → "
               f"{pct(pk['D-0']['AFL'], 0)}) dan sedikit AFIB. Di Icentia11k (wearable), AF dan AFL adalah kelas besar ({thou(R['n']['icentia_af'])} "
               "window), sehingga sensitivitas yang hilang (AFIB maupun AFL) lebih mahal daripada alarm Normal yang dihemat. Pelajaran v9/v10 "
               "berulang: dataset yang berbeda memberi vonis berbeda.", y)
    # ── 6. per pasien ──
    y = L.page("Per pasien AFDB (butir 5)", "Sensitivitas AF dan alarm pada window Normal, rata-rata 5 ulangan")
    y = L.chart(TMP / "se_pasien.png", y, 185)
    pp = R["per_pasien"]
    rows = []
    for p in pp["D-0"]:
        a, b = pp["v3.1"][p], pp["D-0"][p]
        f = lambda v: "–" if v is None else fnum(v, 2)  # noqa: E731
        rows.append([p, thou(b["n_af"]), f(a["se"]), f(b["se"]), thou(b["n_normal"]), f(a["alarm"]), f(b["alarm"])])
    turun = set(R["se_turun_0_20"])
    y = L.table(["Pasien", "n AF", "Se v3.1", "Se D-0", "n Normal", "Alarm v3.1", "Alarm D-0"], rows, y,
                widths=[L.CW * .13, L.CW * .13, L.CW * .14, L.CW * .14, L.CW * .14, L.CW * .16, L.CW * .16],
                sorot=tuple(i + 1 for i, p in enumerate(pp["D-0"]) if p in turun))
    y = L.para(f"Sensitivitas D-0 turun lebih dari 0,20 di {', '.join(sorted(turun))} (disorot): 08378 kehilangan AFL 2:1 dan sebagian "
               "AFIB-nya, 05121 sebagian AFIB, 05091 hanya punya 9 window AF. Alarm pada window Normal turun di semua pasien yang punya ritme "
               "Normal, paling besar di 08219 dan 06995.", y, "small")
    # ── 7. kurva + fungsi prediksi ──
    y = L.page("Kurva ambang (deskriptif) dan fungsi prediksi", "Tidak ada ambang yang dipilih; * Icentia ditambahkan setelah hasil")
    y = L.chart(TMP / "kurva.png", y, 200)
    ki = {x["t"]: x for x in R["kurva_icentia_tambahan"]}
    rows = [[sg(x["t"], 0) if x["t"] else "0 (D-0)", fnum(x["se_afdb"]), fnum(x["sp_afdb"]), pct(x["alarm_st"]), pct(x["alarm_stach"]),
             fnum(x["se_ptb"]), fnum(ki[x["t"]]["se"]), fnum(ki[x["t"]]["sp"]), fnum(ki[x["t"]]["mf1"])] for x in R["kurva"]]
    y = L.table(["t", "Se AFDB", "Sp AFDB", "Sinus ≥120", "STACH", "Se PTB", "Se Icentia*", "Sp Icentia*", "MF1 Icentia*"], rows, y,
                widths=[L.CW * .11] + [L.CW * .11125] * 8, sorot=(5,))
    par, lc, lg = R["paritas"], R["latensi"]["cpu"], R["latensi"]["gpu"]
    k1, i1 = next(x for x in R["kurva"] if x["t"] == -1), ki[-1]
    y = L.para(f"<b>Membaca kurva.</b> Pada t = −1, sensitivitas AFDB {fnum(k1['se_afdb'])} (v3.1 {fnum(A['v3.1']['Se'][0])}) dengan "
               f"spesifisitas {fnum(k1['sp_afdb'])}, alarm sinus takikardia asli {pct(k1['alarm_st'])}, dan di Icentia11k Macro-F1 "
               f"{fnum(i1['mf1'])} (paket v3.1 {fnum(ico['D-0']['titik_b']['Macro-F1'][0])}). Ini terlihat <i>setelah</i> hasil, di data yang "
               "sama, jadi bukan temuan: ambang lain harus dikunci lebih dulu dan diuji di data yang belum dipakai (mis. CPSC 2021, yang "
               "sinyalnya belum pernah dinilai, atau rekaman perangkat).", y)
    y = L.para(f"<b>Fungsi prediksi (H6).</b> <font face='Arial-Bold'>deteksi_af.DetektorAF</font> (EKG 10 s 50 Hz → keputusan) dijalankan "
               f"ulang di CPU float32 pada {par['n']} window 08378: keputusan sama {pct(par['sama'])} dengan cache v13 (GPU bf16), selisih d1 "
               f"maksimum {fnum(par['maks_selisih'], 2)}, median {fnum(par['median_selisih'], 3)}. Latensi praproses + inferensi: "
               f"{fnum(lc['ms_per_window'], 1)} ms/window di CPU ({lc['threads']} thread, batch 1)"
               + (f", {fnum(lg['ms_per_window'], 2)} ms/window di GPU ({lg['gpu']}, batch {lg['batch']})." if lg else "."), y)
    # ── 8. kesimpulan ──
    y = L.page("Kesimpulan, langkah berikutnya, batasan, reproduksi", "")
    for t in ("<b>Kesimpulan.</b> Sebagai detektor AF 2 kelas, ECGFounder tanpa penyetelan sangat spesifik: alarm palsu pada sinus "
              "takikardia asli, ritme Normal AFDB, dan sinus PTB-XL hampir hilang, dan episode AF palsu turun ±10×. Macro-F1-nya setara v3.1 di "
              "AFDB dan lebih baik di PTB-XL. Tetapi sensitivitasnya lebih rendah di semua data, terutama untuk AFL, dan di Icentia11k "
              "(wearable) Macro-F1 lebih buruk. Kriteria yang dikunci tidak terpenuhi, jadi D-0 belum menjadi kandidat uji perangkat.",
              "<b>Trade-off untuk produk.</b> D-0 menukar sensitivitas dengan spesifisitas. Untuk skrining yang alarmnya dibaca dokter, "
              "alarm palsu yang ±10× lebih sedikit bisa lebih berharga daripada beberapa poin sensitivitas; untuk deteksi yang tidak boleh "
              "melewatkan AF, sebaliknya. Ini keputusan produk/klinis.",
              "<b>Langkah berikutnya.</b> (1) Kunci ambang D-0 lain (mis. t = −1) dan uji pada data yang belum dipakai: CPSC 2021 (sudah "
              "diunduh di v15, sinyalnya belum pernah dinilai; AF paroksismal dan persisten) dan rekaman perangkat SIHEDAF berlabel. (2) Jalankan DetektorAF di backend pada rekaman perangkat (latensi CPU ±30 ms/window cukup "
              "untuk window 10 s). (3) Bila AFL harus terdeteksi, D-0 tidak cukup: gabungkan dengan jalur \"perlu dicek\" dan terima alarm "
              "sinus takikardia, atau cari data AFL dari banyak pasien.",
              "<b>Batasan.</b> AFDB: 23 pasien, 2 di antaranya tanpa ritme Normal. Data latihan fisik tidak berlabel ritme (diasumsikan sinus). "
              "PTB-XL hanya 10 s saat istirahat; Icentia11k berlabel per segmen dan fenotipe AFL-nya berbeda (v10). Kanal AFDB (ECG1) dan "
              "lead I PTB-XL belum tentu sama dengan lead perangkat.",
              "<b>Reproduksi.</b> <font face='Arial-Bold'>python tools/run_notebook.py v16.1_dua_kelas/deteksi_af.ipynb</font>, lalu "
              "<font face='Arial-Bold'>build_report.py --extract</font> dan tanpa argumen. Butuh cache v3.3, v10, v11 (BiGRU bawaan), v11.1, "
              "v13, v15, v16 dan bobot ECGFounder di data/ecgfounder/ (untuk paritas dan latensi)."):
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
