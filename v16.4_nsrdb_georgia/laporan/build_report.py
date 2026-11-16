"""Laporan PDF v16.4: D-1 dan v3.1 di NSRDB (Holter sehat 24 jam) dan Georgia 12-lead (data yang belum pernah dipakai).

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v16.4/ringkasan.json (sel akhir uji_baru.ipynb), hitung ulang angka utama dari cache per rekaman lalu
     cocokkan (assert), gambar grafik → laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_v16_4_nsrdb_georgia.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v16_4_nsrdb_georgia.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, Laporan, ci, fnum, pct, sg, sumbu_koma, thou, vonis  # noqa: E402

TANGGAL = "10 November 2026"
SISTEM = ("v1 paket", "v3.1 paket", "D-0", "D-1")
NAMA = {"v1 paket": "v1 paket", "v3.1 paket": "v3.1 paket (referensi)", "D-0": "D-0 (t = 0)", "D-1": "D-1 (primer, t = −1)"}
GAYA = {"v1 paket": (GRAY, "D"), "v3.1 paket": (BIRU, "o"), "D-0": (HIJAU, "^"), "D-1": (ORANYE, "s")}
KEL = ("sinus normal", "sinus takikardia", "sinus bradikardia", "lainnya")


def extract():
    import numpy as np
    import uji_baru as U
    R = json.loads((U.CACHE / "ringkasan.json").read_text(encoding="utf-8"))
    Zg, _ = U.muat("Georgia")
    Zn, _ = U.muat("NSRDB")
    pg, pn = U.R163.prediksi(Zg), U.R163.prediksi(Zn)
    af = Zg["y2"] == 1
    for k in ("D-1", "v3.1 paket"):
        assert abs(float(pg[k][af].mean()) - R["georgia"]["titik"][k]["Se"][0]) < 1e-12, k
        assert abs(float((~pg[k][~af]).mean()) - R["georgia"]["titik"][k]["Sp"][0]) < 1e-12, k
        st = Zg["kelompok"] == "sinus takikardia"
        assert abs(float(pg[k][st].mean()) - R["georgia"]["kelompok"]["sinus takikardia"][k]["alarm"]) < 1e-12, k
        assert abs(float(pn[k].mean()) - R["nsrdb"]["alarm"][k]["alarm"]) < 1e-12, k
    # eksplorasi pasca-hasil (bukan bagian keputusan): AF Georgia yang terlewat menurut keteraturan RR
    cv, afl = Zg["cv"], Zg["afl"].astype(bool)
    eks = {}
    for nama, m in (("AF tidak teratur (CV IBI ≥ 0,08)", af & (cv >= .08)), ("AF teratur (CV IBI < 0,08)", af & (cv < .08)),
                    ("AFL tanpa AF", af & afl)):
        eks[nama] = dict(n=int(m.sum()), v31=float(pg["v3.1 paket"][m].mean()), d1=float(pg["D-1"][m].mean()))
    lewat = af & ~pg["D-1"]
    eks["lewat"] = dict(n=int(lewat.sum()), cv_med=float(np.median(cv[lewat])), hr_med=float(np.median(Zg["hr"][lewat])),
                        cv_tangkap=float(np.median(cv[af & pg["D-1"]])), dua_lewat=int((lewat & ~pg["v3.1 paket"]).sum()),
                        cv_dua_lewat=float(np.median(cv[lewat & ~pg["v3.1 paket"]])))
    TMP.mkdir(exist_ok=True)
    (TMP / "v16_4.json").write_text(json.dumps({"R": R, "eks": eks}, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    grafik(R)
    print("extract selesai; assert Se/Sp Georgia, alarm sinus takikardia, alarm NSRDB cocok dengan ringkasan.json")


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
    G, N = R["georgia"], R["nsrdb"]
    # 1. Georgia: alarm per kelompok non-AF + sensitivitas AF/AFL
    grup = [("sinus normal", G["kelompok"]), ("sinus takikardia", G["kelompok"]), ("sinus bradikardia", G["kelompok"]),
            ("lainnya", G["kelompok"]), ("AF", G["subkelompok"]), ("AFL", G["subkelompok"])]
    fig, ax = plt.subplots(figsize=(7.6, 2.8), dpi=200)
    lebar = .2
    for i, k in enumerate(SISTEM):
        col, _ = GAYA[k]
        ys = [d[g][k]["alarm"] * 100 for g, d in grup]
        ax.bar([x + (i - 1.5) * lebar for x in range(len(grup))], ys, lebar * .92, color=col, label=NAMA[k].split(" (")[0],
               hatch=("//" if k == "D-1" else None), edgecolor="white" if k == "D-1" else None)
    ax.set_xticks(range(len(grup)), [f"{g}\n(n = {thou(d[g]['D-1']['n'])})" for g, d in grup], fontsize=7.8)
    ax.axvline(3.5, color=GRAY, lw=.8, ls="--")
    ax.text(1.5, 104, "non-AF: alarm = palsu", ha="center", fontsize=8, color=GRAY)
    ax.text(4.5, 104, "AF: alarm = benar", ha="center", fontsize=8, color=GRAY)
    ax.set_ylabel("dialarmkan AF (%)"); ax.set_ylim(0, 112); ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper left",
                                                                       bbox_to_anchor=(0, .93))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "georgia.png"); plt.close(fig)
    # 2. NSRDB: alarm per subjek, v3.1 vs D-1
    ps = N["per_subjek"]
    sub = sorted(ps, key=lambda s: ps[s]["v3.1 paket"])
    fig, ax = plt.subplots(figsize=(7.6, 2.4), dpi=200)
    for i, s in enumerate(sub):
        ax.plot([i, i], [ps[s]["v3.1 paket"] * 100, ps[s]["D-1"] * 100], color="#C9D3DD", lw=1.2, zorder=1)
    ax.scatter(range(len(sub)), [ps[s]["v3.1 paket"] * 100 for s in sub], color=BIRU, marker="o", s=18, label="v3.1 paket", zorder=2)
    ax.scatter(range(len(sub)), [ps[s]["D-1"] * 100 for s in sub], color=ORANYE, marker="s", s=18, label="D-1", zorder=3)
    ax.set_xticks(range(len(sub)), sub, rotation=90, fontsize=7); ax.set_ylabel("window dialarmkan (%)", fontsize=8.5)
    ax.set_xlabel("subjek NSRDB (orang sehat), diurutkan menurut alarm v3.1", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "nsrdb.png"); plt.close(fig)
    # 3. alarm non-AF per pita detak (NSRDB dan Georgia)
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.5), dpi=200, sharey=True)
    for ax, (judul, pita) in zip(axs, (("NSRDB", N["pita"]), ("Georgia (non-AF)", G["pita"]))):
        bands = list(pita["D-1"])
        for i, k in enumerate(("v3.1 paket", "D-1")):
            col, mk = GAYA[k]
            ys = [(pita[k][b]["alarm"] or 0) * 100 for b in bands]
            ax.bar([x + (i - .5) * .38 for x in range(len(bands))], ys, .36, color=col, label=k,
                   hatch=("//" if k == "D-1" else None), edgecolor="white" if k == "D-1" else None)
        ax.set_xticks(range(len(bands)), [f"{b}\n(n = {thou(pita['D-1'][b]['n'])})" for b in bands], fontsize=7)
        ax.set_title(judul, fontsize=9, color=INK); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1")
        sumbu_koma(ax)
    axs[0].set_ylabel("alarm AF (%)"); axs[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(TMP / "pita.png"); plt.close(fig)


def _d(v, nd=3):
    return f"{sg(v[0], nd)} [{sg(v[1], nd)}; {sg(v[2], nd)}]"


def build():
    J = json.loads((TMP / "v16_4.json").read_text(encoding="utf-8"))
    R = J["R"]
    R["eks"] = J["eks"]
    G, N, k = R["georgia"], R["nsrdb"], R["keputusan"]
    t, b = G["titik"], G["banding"]
    kel, sub = G["kelompok"], G["subkelompok"]
    na = N["alarm"]
    ok = lambda x: "lolos" if x else "gagal"  # noqa: E731
    jam = R["data"]["NSRDB"]["window_lolos"] * 5 / 3600
    L = Laporan(OUT, "v16.4: D-1 di NSRDB dan Georgia", "SIHEDAF  /  V16.4  /  NSRDB & GEORGIA",
                "v16.4_nsrdb_georgia/uji_baru.ipynb, cache/v16.4/", TANGGAL)
    # ── 1. ringkasan ──
    y = L.page(RINGKASAN_JUDUL(k), "Rancangan dikunci 10 November 2026 sebelum sinyal kedua dataset dibaca")
    y = L.kartu([(f"{pct(na['v3.1 paket']['alarm'], 1)} → {pct(na['D-1']['alarm'], 1)}", "alarm palsu orang sehat", f"NSRDB, 18 subjek, ±{thou(jam)} jam"),
                 (f"{pct(kel['sinus takikardia']['v3.1 paket']['alarm'], 0)} → {pct(kel['sinus takikardia']['D-1']['alarm'], 0)}",
                  "alarm sinus takikardia", "Georgia, label dokter"),
                 (f"{fnum(t['v3.1 paket']['Se'][0], 2)} → {fnum(t['D-1']['Se'][0], 2)}", "sensitivitas AF (Georgia)", f"P = {fnum(b['D-1']['Se'][3], 3)}"),
                 (f"{fnum(t['v3.1 paket']['Macro-F1'][0], 2)} → {fnum(t['D-1']['Macro-F1'][0], 2)}", "Macro-F1 2 kelas (Georgia)",
                  f"P = {fnum(b['D-1']['Macro-F1'][3], 3)}")], y)
    y = L.para("<b>Pertanyaan.</b> D-1 (ECGFounder tanpa penyetelan, AF bila d1 &gt; −1) adalah kandidat detektor AF 2 kelas. Ketujuh data "
               "publik sebelumnya sudah dipakai. Dua data yang belum pernah disentuh repo ini menjawab dua pertanyaan produk: berapa alarm "
               "AF palsu pada orang sehat sepanjang hari (NSRDB, Holter 24 jam), dan apakah keunggulan D-1 bertahan di EKG klinis lain dengan "
               "label dokter untuk sinus takikardia, sinus bradikardia, dan AFL (Georgia, AS).", y)
    y = L.para(JAWABAN(R), y)
    rows = [["H1 Georgia: Macro-F1 D-1 bukan lebih buruk dari v3.1", ok(k["H1_macro_f1"]), _d(b["D-1"]["Macro-F1"]) + f", P = {fnum(b['D-1']['Macro-F1'][3])}"],
            ["H2 Georgia: sensitivitas D-1 bukan lebih buruk", ok(k["H2_sensitivitas"]), _d(b["D-1"]["Se"]) + f", P = {fnum(b['D-1']['Se'][3])}"],
            ["H3 Georgia: alarm sinus takikardia D-1 ≤ separuh v3.1", ok(k["H3_sinus_takikardia"]),
             f"{pct(kel['sinus takikardia']['D-1']['alarm'])} vs {pct(kel['sinus takikardia']['v3.1 paket']['alarm'])}"],
            ["H4 NSRDB: alarm palsu D-1 lebih rendah (P(Δ&lt;0) ≥ 0,95)", ok(k["H4_nsrdb"]),
             f"Δ {sg(N['banding_d1_v31']['delta'] * 100, 2)} poin, P = {fnum(N['banding_d1_v31']['p_kurang'])}"],
            ["Sekunder: Se non-inferior (batas bawah CI &gt; −0,02)", ok(k["sekunder_noninferior_se"]), f"batas bawah {sg(b['D-1']['Se'][1])}"],
            ["Keputusan: D-1 tetap kandidat", "ya" if k["semua_lolos"] else "ditinjau", ""]]
    y = L.table(["Kriteria yang dikunci", "Hasil", "Angka"], rows, y, widths=[L.CW * .5, L.CW * .12, L.CW * .38], sorot=(6,))
    y = L.para(TEMUAN(R), y)
    # ── 2. Georgia ──
    y = L.page("Georgia: perbandingan dengan v3.1 dan v1", "EKG klinis 12-lead 10 s (lead I), Emory, AS; bootstrap rekaman berpasangan 2.000×")
    y = L.chart(TMP / "georgia.png", y, 190)
    rows = [[NAMA[s]] + [f"{fnum(t[s][m][0])} {ci(*t[s][m][1:])}" for m in ("Macro-F1", "Se", "Sp", "PPV")] for s in SISTEM]
    y = L.table(["Sistem", "Macro-F1 [CI]", "Sensitivitas [CI]", "Spesifisitas [CI]", "PPV [CI]"], rows, y,
                widths=[L.CW * .24, L.CW * .19, L.CW * .19, L.CW * .19, L.CW * .19], sorot=(4,))
    rows = [[lab, f"{_d(b[s]['Macro-F1'])}, P = {fnum(b[s]['Macro-F1'][3])}", f"{sg(b[s]['Se'][0])}, P = {fnum(b[s]['Se'][3])}",
             f"{sg(b[s]['Sp'][0])}, P = {fnum(b[s]['Sp'][3])}", vonis(b[s]["Macro-F1"][3])]
            for s, lab in (("D-1", "D-1 − v3.1"), ("D-0", "D-0 − v3.1"), ("v1 paket", "v1 − v3.1"))]
    y = L.table(["Banding", "Δ Macro-F1 [CI], P(Δ &gt; 0)", "Δ Se, P", "Δ Sp, P", "Vonis Macro-F1"], rows, y,
                widths=[L.CW * .15, L.CW * .37, L.CW * .17, L.CW * .17, L.CW * .14], sorot=(1,))
    rows = [[g, thou(d["D-1"]["n"]), *[f"{pct(d[s]['alarm'])} {ci(d[s]['ci'][0] * 100, d[s]['ci'][1] * 100, 0)}" for s in ("v3.1 paket", "D-1")]]
            for g, d in [(g, kel[g]) for g in KEL] + [(g, sub[g]) for g in sub]]
    y = L.table(["Kelompok (alarm AF; untuk AF/AFL = sensitivitas)", "EKG", "v3.1 [CI %]", "D-1 [CI %]"], rows, y,
                widths=[L.CW * .4, L.CW * .14, L.CW * .23, L.CW * .23])
    # ── 3. NSRDB ──
    y = L.page("NSRDB: alarm palsu pada orang sehat sepanjang hari", f"18 subjek, ±{thou(jam)} jam Holter 128 Hz; semua window non-AF")
    y = L.chart(TMP / "nsrdb.png", y, 165)
    ep = N["episode"]
    rows = [[NAMA[s], f"{pct(na[s]['alarm'], 2)} {ci(na[s]['ci'][0] * 100, na[s]['ci'][1] * 100, 2)}", pct(N["alarm_tanpa_artefak"][s]["alarm"], 2),
             fnum(ep[s]["palsu_per_jam"], 2)] for s in SISTEM]
    y = L.table(["Sistem", "Window dialarmkan [CI %]", "Tanpa window berartefak", "Episode AF palsu per jam"], rows, y,
                widths=[L.CW * .28, L.CW * .28, L.CW * .22, L.CW * .22], sorot=(4,))
    y = L.para(NSRDB_TEKS(R), y)
    # ── 4. pita detak ──
    y = L.page("Alarm non-AF per pita detak", "Detak dari R-peak detektor perangkat; protokol butir 9")
    y = L.chart(TMP / "pita.png", y, 180)
    y = L.para(PITA_TEKS(R), y)
    # ── 5. metode ──
    y = L.page("Metode, pra-registrasi, kendali mutu", "RANCANGAN.md dikunci sebelum sinyal dibaca; sha256 di logs/v16.4_praregistrasi.log")
    for tx in METODE(R):
        y = L.para(tx, y)
    # ── 6. kesimpulan ──
    y = L.page("Kesimpulan, langkah berikutnya, batasan, reproduksi", "")
    for tx in KESIMPULAN(R):
        y = L.para(tx, y)
    lk = R["lingkungan"]
    y = L.para(f"Lingkungan: PC desktop, GPU {lk['gpu']}, commit {lk['commit']} (+ perubahan belum di-commit), Python {lk['pustaka']['python']}, "
               f"numpy {lk['pustaka']['numpy']}, torch {lk['pustaka']['torch']}.", y, "small")
    L.selesai()
    print("PDF:", OUT, L.page_no, "halaman")


# Teks naratif: ditulis setelah hasil dilihat (angka tetap diambil dari ringkasan).
def RINGKASAN_JUDUL(k):
    return "Ringkasan: D-1 jauh lebih spesifik, tetapi kurang sensitif di EKG klinis 10 s"


def JAWABAN(R):
    G, N = R["georgia"], R["nsrdb"]
    t, kel = G["titik"], G["kelompok"]
    return ("<b>Jawaban singkat.</b> Tiga dari empat kriteria lolos. Pada orang sehat sepanjang hari (NSRDB) D-1 mengalarmkan "
            f"{pct(N['alarm']['D-1']['alarm'], 1)} window, v3.1 {pct(N['alarm']['v3.1 paket']['alarm'], 1)}. Di Georgia Macro-F1 D-1 jauh "
            f"lebih baik ({fnum(t['D-1']['Macro-F1'][0])} vs {fnum(t['v3.1 paket']['Macro-F1'][0])}) dan alarm sinus takikardia turun dari "
            f"{pct(kel['sinus takikardia']['v3.1 paket']['alarm'], 0)} ke {pct(kel['sinus takikardia']['D-1']['alarm'], 0)}. "
            f"<b>Tetapi H2 gagal:</b> sensitivitas AF D-1 lebih rendah ({fnum(t['D-1']['Se'][0])} vs {fnum(t['v3.1 paket']['Se'][0])}, "
            "P(Δ &gt; 0) = 0,000). Sesuai rancangan, ambang tidak diubah; status kandidat ditinjau (halaman terakhir).")


def TEMUAN(R):
    G, N, e = R["georgia"], R["nsrdb"], R["eks"]
    t, sub = G["titik"], G["subkelompok"]
    ti, ta = e["AF tidak teratur (CV IBI ≥ 0,08)"], e["AF teratur (CV IBI < 0,08)"]
    return ("<b>Temuan.</b> (1) Kekurangan sensitivitas D-1 kini terlihat di tiga data non-Holter: CinC 2017, Icentia11k (v16.3), dan "
            "Georgia. Di keempat data Holter (AFDB, LTAFDB, MIT-BIH, CPSC 2021) D-1 setara atau lebih baik. "
            f"(2) Bahkan pada AF Georgia yang RR-nya tidak teratur, D-1 menangkap {fnum(ti['d1'])} vs v3.1 {fnum(ti['v31'])} "
            f"(n = {thou(ti['n'])}); jadi bukan hanya soal label. (3) {thou(ta['n'])} EKG berlabel AF/AFL justru teratur (CV IBI &lt; 0,08); "
            f"keduanya hampir tidak menangkapnya (v3.1 {fnum(ta['v31'])}, D-1 {fnum(ta['d1'])}): AFL dengan konduksi tetap atau label tingkat "
            f"rekaman yang tidak berlaku sepanjang 10 s. (4) D-1 nyaris tidak mendeteksi AT/SVT ({pct(sub['AT/SVT']['D-1']['alarm'], 0)} vs "
            f"v3.1 {pct(sub['AT/SVT']['v3.1 paket']['alarm'], 0)}); untuk detektor AF itu benar, bukan kesalahan. (5) PPV D-1 "
            f"{fnum(t['D-1']['PPV'][0])} vs v3.1 {fnum(t['v3.1 paket']['PPV'][0])}: di populasi klinis dengan AF 7%, dua dari tiga alarm "
            "v3.1 adalah palsu. Butir (2)–(3) adalah eksplorasi setelah hasil, bukan uji yang dikunci.")


def NSRDB_TEKS(R):
    N = R["nsrdb"]
    na, ep = N["alarm"], N["episode"]
    ps = N["per_subjek"]
    mx = max(ps, key=lambda s: ps[s]["D-1"])
    return (f"D-1 menurunkan window teralarm dari {pct(na['v3.1 paket']['alarm'], 1)} ke {pct(na['D-1']['alarm'], 1)} "
            f"(Δ {sg(N['banding_d1_v31']['delta'] * 100, 2)} poin, CI [{sg(N['banding_d1_v31']['ci'][0] * 100, 2)}; "
            f"{sg(N['banding_d1_v31']['ci'][1] * 100, 2)}], bootstrap subjek); D-1 lebih rendah pada {sum(ps[x]['D-1'] < ps[x]['v3.1 paket'] for x in ps)} dari {len(ps)} subjek. "
            f"Namun tanpa aturan durasi, ini tetap {fnum(ep['D-1']['palsu_per_jam'], 1)} episode AF palsu per jam (v3.1 "
            f"{fnum(ep['v3.1 paket']['palsu_per_jam'], 1)}), terlalu banyak untuk perangkat. Subjek tertinggi D-1: {mx} "
            f"({pct(ps[mx]['D-1'], 1)}). Membuang window berartefak hampir tidak mengubah angka, jadi alarm palsu bukan terutama dari derau. "
            "Karena itu v16.5 menguji aturan episode klinis (AF ≥ 30 s) di atas keputusan per window.")


def PITA_TEKS(R):
    N, G = R["nsrdb"]["pita"], R["georgia"]["pita"]
    return ("v3.1 mengalarmkan hampir semua ritme non-AF ≥ 120 bpm, di NSRDB "
            f"({pct(N['v3.1 paket']['≥ 120']['alarm'], 0)}) dan Georgia ({pct(G['v3.1 paket']['≥ 120']['alarm'], 0)}); ini mengulang temuan "
            f"v3.2–v3.3 di dua data baru. D-1 tetap rendah di Georgia ({pct(G['D-1']['≥ 120']['alarm'], 1)}), tetapi di NSRDB naik ke "
            f"{pct(N['D-1']['110–120']['alarm'], 0)} (110–120 bpm) dan {pct(N['D-1']['≥ 120']['alarm'], 0)} (≥ 120 bpm). Window NSRDB adalah Holter "
            "ambulatori 128 Hz sepanjang hari (lebih banyak derau dan gerak) sedangkan Georgia adalah EKG istirahat di klinik. Di bawah 100 bpm, "
            f"alarm D-1 di NSRDB {pct(N['D-1']['< 100']['alarm'], 1)} vs v3.1 {pct(N['v3.1 paket']['< 100']['alarm'], 1)}.")


def METODE(R):
    d = R["data"]
    pk = d["Georgia"]["per_kelompok"]
    return [
        "<b>Data.</b> NSRDB (PhysioNet, 18 subjek tanpa aritmia berarti, Holter ±24 jam 128 Hz, kanal 0): semua window non-AF; kode artefak "
        "14/16 ditandai dan dianalisis dengan/tanpa window itu. Georgia (PhysioNet/CinC Challenge 2021, Emory, AS, "
        f"{thou(d['Georgia']['rekaman'])} EKG 12-lead 10 s 500 Hz): lead I, label SNOMED dari header. Kelompok berurutan prioritas: AF/AFL "
        f"({thou(pk['AF/AFL'][0])}), sinus takikardia ({thou(pk['sinus takikardia'][0])}), sinus bradikardia ({thou(pk['sinus bradikardia'][0])}), "
        f"sinus normal ({thou(pk['sinus normal'][0])}), lainnya ({thou(pk['lainnya'][0])}). Unit bootstrap = subjek (NSRDB) atau rekaman (Georgia).",
        "<b>Rantai prediksi.</b> Sama dengan v16.3: resampling ke 50 Hz, detektor R-peak perangkat (hrv_core), cek kualitas window, lalu "
        "ECGFounder 1-lead zero-shot (d1 = log((pAF + pAFL)/pN)), paket v3.1, dan paket v1. Georgia: satu window 10 s per rekaman. "
        "D-1 = d1 &gt; −1 (dikunci di v16.2), D-0 = d1 &gt; 0.",
        "<b>Kriteria (dikunci).</b> H1 Macro-F1 dan H2 sensitivitas D-1 bukan lebih buruk dari v3.1 (P(Δ &gt; 0) &gt; 0,05), H3 alarm sinus "
        "takikardia D-1 ≤ separuh v3.1, H4 alarm NSRDB D-1 lebih rendah (P(Δ &lt; 0) ≥ 0,95). Sekunder: non-inferioritas sensitivitas dengan "
        "margin −0,02 (batas bawah CI). Bootstrap 2.000×, seed 2026, berpasangan.",
        f"<b>Kendali mutu.</b> Jumlah rekaman per kelompok sama dengan hitungan dari header (assert). Window lolos cek: Georgia "
        f"{thou(d['Georgia']['window_lolos'])} / {thou(d['Georgia']['window'])}, NSRDB {thou(d['NSRDB']['window_lolos'])} / "
        f"{thou(d['NSRDB']['window'])}. Laporan ini menghitung ulang Se, Sp, alarm sinus takikardia, dan alarm NSRDB dari cache per "
        "rekaman dan mencocokkannya dengan ringkasan notebook (assert).",
    ]


def KESIMPULAN(R):
    k = R["keputusan"]
    t = R["georgia"]["titik"]
    return [
        "<b>Kesimpulan.</b> D-1 tetap jauh lebih spesifik daripada v3.1 di dua data baru: alarm palsu orang sehat turun "
        f"±{fnum(R['nsrdb']['alarm']['v3.1 paket']['alarm'] / R['nsrdb']['alarm']['D-1']['alarm'], 1)}× dan alarm "
        "sinus takikardia klinis turun dari separuh ke ±2%. Tetapi sensitivitasnya tidak setara di EKG klinis 10 s "
        f"(Δ Se {sg(t['D-1']['Se'][0] - t['v3.1 paket']['Se'][0])}). Dengan v16.3, kekurangan sensitivitas 3–6 poin kini konsisten di tiga data "
        "non-Holter. <b>Status:</b> D-1 tetap kandidat untuk perangkat Holter/patch yang merekam terus-menerus, di mana AF yang terlewat di satu "
        "window bisa tertangkap di window berikutnya (v16.5 mengujinya lewat sensitivitas episode). Untuk pemeriksaan sesaat 10–60 s "
        "(EKG genggam), D-1 belum cukup sensitif.",
        "<b>Langkah berikutnya.</b> (1) v16.5: aturan episode AF ≥ 30 s untuk D-1 dan v3.1 (alarm palsu per jam di NSRDB, EDB, latihan "
        "fisik; sensitivitas episode di empat data Holter). (2) Ambang atau fusi untuk rekaman pendek hanya dari data perangkat SIHEDAF; "
        "ketujuh data publik dan Georgia kini terpakai untuk D-1.",
        "<b>Batasan.</b> Label Georgia berlaku per rekaman, bukan per detik; sebagian EKG berlabel AF tampak teratur. Hanya lead I yang dipakai "
        "(perangkat satu lead), sehingga AF dengan gelombang f kecil di lead I sulit bagi model bentuk gelombang. NSRDB hanya 18 subjek; "
        "CI-nya lebar. Eksplorasi menurut keteraturan RR dilakukan setelah hasil dilihat.",
        "<b>Reproduksi.</b> Dari root repo: python tools/unduh_physionet.py nsrdb 1.0.0 --ke nsrdb --pola '*.hea' '*.dat' '*.atr' RECORDS; "
        "python tools/unduh_physionet.py challenge-2021 1.0.3 --ke challenge2021 --sumber physionet --thread 6 --pola 'training/georgia/*'; "
        "python tools/run_notebook.py v16.4_nsrdb_georgia/uji_baru.ipynb. Lalu dari folder v16.4_nsrdb_georgia: python laporan/build_report.py "
        "--extract, lalu python laporan/build_report.py.",
    ]


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
