"""Laporan PDF v16.9: sinus bradikardia dan detak lambat pada S1 (D-1 + episode AF ≥ 30 s).

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v16.9/ringkasan.json (sel akhir bradikardia.ipynb), hitung ulang laju alarm palsu S1/R pada sinus bradikardia
     LTAFDB dan fraksi alarm detak lambat/acuan Icentia11k dari cache per rekaman (assert), gambar grafik.
  2. tanpa argumen: rakit laporan/laporan_v16_9_sinus_bradikardia.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v16_9_sinus_bradikardia.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, Laporan, ci, fnum, pct, thou  # noqa: E402

TANGGAL = "10 November 2026"
PITA = ("< 50", "50–60", "60–80", "80–100", "≥ 100")
DATASET = ("LTAFDB", "MIT-BIH", "CPSC 2021", "NSRDB", "EDB", "SHDB-AF", "afpdb", "incartdb", "svdb", "Icentia11k")


def extract():
    import numpy as np
    import bradikardia as B
    R = json.loads((B.CACHE / "ringkasan.json").read_text(encoding="utf-8"))
    d = B.muat("LTAFDB")
    al = B.sistem(d)
    for s, k in (("S1", "nilai"), ("R", "nilai_b")):
        pl, _, _ = B.episode(d, al[s])
        u = B.laju_ritme(d, pl, B.ritme_episode(d, pl), "(SBR")
        lj = sum(v[0] for v in u.values()) / sum(v[1] for v in u.values())
        assert abs(lj - R["P1"]["(SBR"]["laju"][k]) < 1e-9, s
    ic = B.muat("Icentia11k")
    a = B.sistem(ic)["S1"]
    for band, k in ((B.LAMBAT, "lambat"), (B.ACUAN, "acuan")):
        u = B.fraksi_pita(ic, a, *band)
        f = sum(v[0] for v in u.values()) / sum(v[1] for v in u.values())
        assert abs(f - R["P3"]["Icentia11k"]["S1"][k]) < 1e-9, k
    assert np.isclose(R["P1"]["(SBR"]["jam"], B.jam(int(((d["ritme"] == "(SBR") & (d["y2"] == 0)).sum())))
    TMP.mkdir(exist_ok=True)
    (TMP / "v16_9.json").write_text(json.dumps({"R": R}, indent=1, ensure_ascii=False), encoding="utf-8")
    grafik(R)
    print("extract selesai; assert laju palsu sinus bradikardia LTAFDB (S1, R) dan fraksi alarm lambat/acuan Icentia (S1) cocok")


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
    import numpy as np
    from laporan_sihedaf import sumbu_koma
    plt = _gaya()
    # 1. per subjek (sinus bradikardia LTAFDB)
    ps = {u: v for u, v in R["P2"]["per_subjek"].items() if v["palsu_s1"] > 0 or v["palsu_r"] > 0}
    us = list(ps)
    fig, ax = plt.subplots(figsize=(7.6, 2.6), dpi=200)
    x = np.arange(len(us))
    ax.bar(x - .2, [ps[u]["palsu_s1"] for u in us], .38, color=ORANYE, label="S1 (D-1 + 30 s)")
    ax.bar(x + .2, [ps[u]["palsu_r"] for u in us], .38, color=BIRU, hatch="//", edgecolor="white", label="v3.1 + 30 s")
    ax.set_xticks(x, [f"subjek {u}\n{fnum(ps[u]['jam_sbr'], 1)} jam" for u in us], fontsize=7.5)
    ax.set_ylabel("episode AF palsu\nselama sinus bradikardia", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "subjek.png"); plt.close(fig)
    # 2. laju palsu per pita detak, semua dataset
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=200, sharey=True)
    xs = np.arange(len(PITA))
    for ax, s, col, mk, nama in ((axs[0], "S1", ORANYE, "o", "S1 (D-1 + 30 s)"), (axs[1], "R", BIRU, "s", "v3.1 + 30 s (sumbu dipotong di 12)")):
        M = np.array([[R["per_pita"][ds][p][f"laju_{s}"] for p in PITA] for ds in DATASET])
        for i, ds in enumerate(DATASET):
            if ds == "Icentia11k":
                continue
            ax.plot(xs, M[i], color="#C9D3DD", lw=.9, zorder=1)
        ax.plot(xs, M[DATASET.index("Icentia11k")], color=HIJAU, marker="^", lw=1.4, label="Icentia11k (wearable)", zorder=3)
        ax.plot(xs, np.nanmedian(M, 0), color=col, marker=mk, lw=2, label="median 10 dataset", zorder=4)
        ax.plot([], [], color="#C9D3DD", lw=.9, label="dataset lain")
        ax.axhline(1, color=INK, lw=.8, ls="--")
        ax.axvspan(-.4, 1.5, color="#F3F6F9", zorder=0)
        ax.set_title(nama, fontsize=9, color=INK)
        ax.set_xticks(xs, PITA, fontsize=8); ax.set_xlabel("detak window (bpm)", fontsize=8)
        ax.set_ylim(0, 12)
        ax.legend(frameon=False, fontsize=7, loc="upper right" if s == "S1" else "upper left")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    axs[0].set_ylabel("episode AF palsu per jam", fontsize=8.5)
    fig.tight_layout(); fig.savefig(TMP / "pita.png"); plt.close(fig)


def build():
    R = json.loads((TMP / "v16_9.json").read_text(encoding="utf-8"))["R"]
    sb, nn, P3, P4, Ek = R["P1"]["(SBR"], R["P1"]["(N"], R["P3"], R["P4"], R["eksplorasi"]
    ya = lambda x: "ya" if x else "tidak"  # noqa: E731
    L = Laporan(OUT, "v16.9: sinus bradikardia pada S1", "SIHEDAF  /  V16.9  /  SINUS BRADIKARDIA",
                "v16.9_sinus_bradikardia/bradikardia.ipynb, cache/v16.9/", TANGGAL)
    # 1. ringkasan
    y = L.page("Ringkasan: detak lambat adalah kelemahan nyata S1, tetapi terkonsentrasi",
               "Evaluasi dari cache; rancangan dikunci 10 November 2026 sebelum alarm per ritme/pita dihitung")
    af_lambat = [P4[k]["frac_window_lambat"] for k in ("LTAFDB", "CPSC 2021", "SHDB-AF", "Icentia11k")]
    y = L.kartu([(f"{fnum(sb['laju']['nilai'], 1)} vs {fnum(sb['laju']['nilai_b'], 1)}", "palsu/jam, bradikardia",
                  "S1 vs v3.1, LTAFDB"),
                 (pct(R["P2"]["porsi_top3_s1"], 0), "alarm itu dari 3 subjek", f"{R['P2']['subjek_dengan_palsu_s1']} dari {sb['subjek']} subjek"),
                 (f"{1 + len(R['P3_dataset_lemah_selain_ltafdb'])} dari 10", "dataset: S1 lebih sering", "alarm saat < 60 bpm"),
                 (f"{pct(min(af_lambat), 0)}–{pct(max(af_lambat), 0)}", "window AF berdetak < 60", "aturan detak tidak bisa")], y)
    y = L.para("<b>Pertanyaan.</b> Di v16.3, D-1 menyebut 28% window sinus bradikardia LTAFDB sebagai AF (v3.1: 4%). Apakah ini masih "
               "menjadi alarm setelah aturan episode ≥ 30 s (S1), apakah hanya terjadi di LTAFDB, dan bisakah ditekan dengan aturan detak?", y)
    y = L.para(JAWABAN(R), y)
    rows = [["P1: S1 &gt; 1,0 palsu/jam saat sinus bradikardia dan lebih tinggi dari v3.1 (P ≥ 0,95)", ya(R["P1_kelemahan_nyata"]),
             f"{fnum(sb['laju']['nilai'], 2)} {ci(*sb['laju']['ci'])} vs {fnum(sb['laju']['nilai_b'], 2)}; P = {fnum(sb['laju']['p_lebih'])}"],
            ["P3: S1 lebih sering alarm saat &lt; 60 bpm (vs 60–100) di ≥ 2 dataset selain LTAFDB", ya(R["P3_kelemahan_umum"]),
             ", ".join(R["P3_dataset_lemah_selain_ltafdb"])],
            ["Aturan berbasis detak dipilih?", "tidak", "di luar rancangan; lihat P4"]]
    y = L.table(["Kriteria yang dikunci", "Positif?", "Angka"], rows, y, widths=[L.CW * .55, L.CW * .1, L.CW * .35])
    y = L.para(TEMUAN(R), y)
    # 2. P1/P2
    y = L.page("P1–P2: sinus bradikardia berlabel di LTAFDB", "84 subjek Holter ±24 jam; 25 subjek punya label (SBR")
    y = L.chart(TMP / "subjek.png", y, 175)
    rows = []
    for nama, v in (("sinus bradikardia (SBR)", sb), ("sinus normal (N)", nn)):
        rows.append([nama, fnum(v["jam"], 1), str(v["subjek"]), thou(v["palsu_s1"]), thou(v["palsu_r"]),
                     f"{fnum(v['laju']['nilai'], 2)} {ci(*v['laju']['ci'])}", f"{fnum(v['laju']['nilai_b'], 2)} {ci(*v['laju']['ci_b'])}",
                     f"{pct(v['fraksi_waktu']['nilai'], 1)} / {pct(v['fraksi_waktu']['nilai_b'], 1)}"])
    y = L.table(["Ritme", "Jam", "Subjek", "Palsu S1", "Palsu v3.1", "S1 per jam", "v3.1 per jam", "Waktu alarm S1 / v3.1"], rows, y,
                widths=[L.CW * .19, L.CW * .07, L.CW * .08, L.CW * .09, L.CW * .1, L.CW * .17, L.CW * .16, L.CW * .14], sorot=(1,))
    J = R["P2"]
    rows = [[k, pct(J["jarak_s1"][k], 0), pct(J["jarak_r"][k], 0), pct(J["jarak_dasar"][k], 0)] for k in J["jarak_dasar"] if k != "rekaman tanpa AF"]
    y = L.table(["Jarak ke episode AF terdekat", "Episode palsu S1", "Episode palsu v3.1", "Semua window SBR"], rows, y,
                widths=[L.CW * .34, L.CW * .22, L.CW * .22, L.CW * .22])
    y = L.para(P12_TEKS(R), y)
    # 3. P3
    y = L.page("P3: detak lambat di sepuluh dataset (window non-AF)", "Detak dari detektor perangkat 50 Hz; daerah abu-abu = detak < 60 bpm")
    y = L.chart(TMP / "pita.png", y, 205)
    rows = []
    for ds in DATASET:
        o = P3[ds]
        rows.append([ds, fnum(R["data"][ds]["jam_lambat"], 0), pct(o["S1"]["lambat"], 2), pct(o["S1"]["acuan"], 2), fnum(o["S1"]["p_lebih"], 2),
                     pct(o["R"]["lambat"], 2), pct(o["R"]["acuan"], 2), ya(o["S1_lemah"])])
    sor = tuple(i + 1 for i, ds in enumerate(DATASET) if P3[ds]["S1_lemah"])
    y = L.table(["Dataset", "Jam &lt; 60", "S1 &lt; 60", "S1 60–100", "S1 P(Δ&gt;0)", "v3.1 &lt; 60", "v3.1 60–100", "S1 lemah"], rows, y,
                widths=[L.CW * .17, L.CW * .1, L.CW * .11, L.CW * .12, L.CW * .13, L.CW * .12, L.CW * .13, L.CW * .12], sorot=sor)
    y = L.para(P3_TEKS(R), y)
    # 4. P4 + eksplorasi
    y = L.page("P4 dan eksplorasi: AF lambat, dan apa yang membedakan window yang dialarmkan", "P4 deskriptif; eksplorasi dilihat setelah hasil")
    rows = [[ds, thou(v["window_af"]), pct(v["frac_window_lambat"], 1), f"{thou(v['episode_lambat'])} / {thou(v['episode_30'])}",
             "–" if v["se_s1_lambat"] != v["se_s1_lambat"] else fnum(v["se_s1_lambat"])] for ds, v in P4.items()]
    y = L.table(["Dataset", "Window AF", "AF &lt; 60 bpm", "Episode AF ≥ 30 s lambat", "Se S1 episode lambat"], rows, y,
                widths=[L.CW * .2, L.CW * .17, L.CW * .17, L.CW * .25, L.CW * .21])
    y = L.para(P4_TEKS(R), y)
    a, t = Ek["alarm_vs_tidak"]["dialarmkan S1"], Ek["alarm_vs_tidak"]["tidak dialarmkan"]
    rows = [["Window", thou(a["window"]), thou(t["window"])],
            ["Detak anotasi (median, bpm)", fnum(a["hr_ref_median"], 1), fnum(t["hr_ref_median"], 1)],
            ["Detak detektor − anotasi (median, bpm)", fnum(a["selisih_hr_median"], 2), fnum(t["selisih_hr_median"], 2)],
            ["CV RR anotasi (median)", fnum(a["cv_ref_median"], 3), fnum(t["cv_ref_median"], 3)],
            ["Window dengan CV RR &gt; 0,10", pct(a["frac_cv_ref_gt_010"], 1), pct(t["frac_cv_ref_gt_010"], 1)],
            ["Window dengan detak ektopik", pct(a["frac_ada_ektopik"], 1), pct(t["frac_ada_ektopik"], 1)],
            ["Window dengan detak anotasi &lt; 45 bpm", pct(a["frac_hr_ref_lt_45"], 1), pct(t["frac_hr_ref_lt_45"], 1)]]
    y = L.table(["Window sinus bradikardia LTAFDB", "Di dalam alarm S1", "Tidak dialarmkan"], rows, y,
                widths=[L.CW * .5, L.CW * .25, L.CW * .25])
    y = L.para(EKS_TEKS(R), y)
    # 5. metode + kesimpulan
    y = L.page("Metode, kesimpulan, batasan, reproduksi", "RANCANGAN.md dikunci; sha256 di logs/v16.9_praregistrasi.log")
    for tx in METODE(R) + KESIMPULAN(R):
        y = L.para(tx, y)
    lk = R["lingkungan"]
    y = L.para(f"Lingkungan: PC desktop, commit {lk['commit']} (+ perubahan belum di-commit), Python {lk['pustaka']['python']}, "
               f"numpy {lk['pustaka']['numpy']}; notebook ±{fnum(R['durasi_menit'], 0)} menit tanpa GPU (semua dari cache).", y, "small")
    L.selesai()
    print("PDF:", OUT, L.page_no, "halaman")


# ── teks naratif (ditulis setelah hasil dilihat; angka dari ringkasan) ──
def JAWABAN(R):
    sb = R["P1"]["(SBR"]
    ic = R["per_pita"]["Icentia11k"]
    return ("<b>Jawaban singkat.</b> Ya, ini kelemahan nyata, tetapi sempit. Selama sinus bradikardia di LTAFDB, S1 memberi "
            f"{fnum(sb['laju']['nilai'], 1)} alarm AF palsu per jam (v3.1: {fnum(sb['laju']['nilai_b'], 1)}), padahal pada sinus normal subjek "
            f"yang sama S1 jauh lebih baik dari v3.1 ({fnum(R['P1']['(N']['laju']['nilai'], 2)} vs {fnum(R['P1']['(N']['laju']['nilai_b'], 2)}). "
            f"{pct(R['P2']['porsi_top3_s1'], 0)} alarm itu berasal dari 3 subjek. Pola yang sama, lebih lemah, muncul di dataset lain: alarm S1 "
            "paling sedikit pada 60–100 bpm dan naik pada detak lambat (dan cepat). Di wearable Icentia11k misalnya "
            f"{fnum(ic['< 50']['laju_S1'], 2)} dan {fnum(ic['50–60']['laju_S1'], 2)} palsu/jam pada &lt; 50 dan 50–60 bpm, vs "
            f"{fnum(ic['60–80']['laju_S1'], 2)} pada 60–80. Aturan \"abaikan AF bila detak lambat\" tidak bisa dipakai: {_af_lambat(R)} window AF "
            "sungguhan di Holter dan wearable berdetak &lt; 60 bpm.")


def _af_lambat(R):
    v = [R["P4"][k]["frac_window_lambat"] for k in ("LTAFDB", "CPSC 2021", "SHDB-AF", "Icentia11k")]
    return f"{pct(min(v), 0)}–{pct(max(v), 0)}"


def _se_lambat(R):
    v = [x["se_s1_lambat"] for x in R["P4"].values() if x["se_s1_lambat"] == x["se_s1_lambat"]]
    return f"{fnum(min(v), 2)}–{fnum(max(v), 2)}"


def TEMUAN(R):
    Ek = R["eksplorasi"]["alarm_vs_tidak"]
    return ("<b>Temuan.</b> (1) Window sinus bradikardia yang dialarmkan S1 tidak berbeda dari yang tidak dialarmkan: detaknya sama "
            f"(median {fnum(Ek['dialarmkan S1']['hr_ref_median'], 1)} vs {fnum(Ek['tidak dialarmkan']['hr_ref_median'], 1)} bpm), sama teraturnya "
            f"(CV RR {fnum(Ek['dialarmkan S1']['cv_ref_median'], 3)}), tidak lebih banyak detak ektopik, dan detektor perangkat menghitung detak "
            "dengan benar. Jadi alarm tidak datang dari ritme yang tidak teratur atau R-peak yang salah, tetapi dari bentuk gelombang yang dilihat "
            "ECGFounder pada subjek tertentu. (2) Karena itu v3.1 (yang hanya melihat RR) hampir tidak pernah salah di sini. (3) Sebagian besar "
            f"alarm jauh dari episode AF sungguhan ({pct(R['P2']['jarak_s1']['> 60 menit'], 0)} lebih dari 1 jam), jadi bukan sekadar sisa "
            "episode AF. (4) S1 tetap mendeteksi AF lambat dengan baik "
            f"(Se episode {_se_lambat(R)}).")


def P12_TEKS(R):
    J = R["P2"]
    top = list(J["per_subjek"])[:3]
    return (f"Episode palsu dihitung setelah aturan 30 s pada seluruh rekaman, lalu diberi label ritme terbanyak di dalamnya. Subjek {', '.join(top)} "
            f"menyumbang {pct(J['porsi_top3_s1'], 0)} episode palsu S1 saat sinus bradikardia; {J['subjek_dengan_palsu_s1']} dari "
            f"{R['P1']['(SBR']['subjek']} subjek punya setidaknya satu. CI laju lebar karena itu. Sebaran jarak episode palsu S1 ke episode AF "
            f"terdekat agak lebih dekat dari sebaran semua window sinus bradikardia ({pct(J['jarak_s1']['< 10 menit'], 0)} vs "
            f"{pct(J['jarak_dasar']['< 10 menit'], 0)} dalam 10 menit), tetapi {pct(J['jarak_s1']['> 60 menit'], 0)} lebih dari 1 jam dari AF, "
            "jadi sebagian besar alarm bukan sisa episode AF. "
            "Semua 25 subjek LTAFDB ber-AF, sehingga pertanyaan \"rekaman tanpa AF\" tidak bisa dijawab di sini.")


def P3_TEKS(R):
    lemah = R["P3_dataset_lemah_selain_ltafdb"]
    pp = R["per_pita"]
    n50 = sum(pp[ds]["< 50"]["laju_S1"] > 1.0 for ds in DATASET)
    kalah = [f"{ds} {p}" for ds in DATASET for p in ("< 50", "50–60") if pp[ds][p]["laju_S1"] > pp[ds][p]["laju_R"]]
    return ("Fraksi waktu non-AF yang berada di dalam alarm, pada detak &lt; 60 vs 60–100 bpm. Kriteria yang dikunci (P(Δ&gt;0) ≥ 0,95 dan "
            f"minimal 2× lipat) terpenuhi di LTAFDB, {' dan '.join(lemah)}, sehingga detak lambat dinyatakan kelemahan umum S1. Besarnya kecil di "
            f"sebagian besar dataset (sering &lt; 2% waktu), tetapi pada &lt; 50 bpm laju alarm palsu S1 di atas target 1,0/jam di {n50} dari 10 "
            "dataset. Grafik memperlihatkan bentuk U: S1 paling sedikit alarm di 60–100 bpm. Di pita 60–100 dan ≥ 100 bpm v3.1 lebih sering "
            f"alarm di semua dataset; di detak lambat S1 lebih buruk dari v3.1 pada: {', '.join(kalah)} bpm. Pita svdb dan MIT-BIH &lt; 50 bpm "
            "hanya ±1 jam.")


def P4_TEKS(R):
    return ("AF dengan respons ventrikel lambat sering ada di Holter dan wearable (mis. pada pasien dengan obat penurun detak). Episode AF ≥ 30 s "
            f"dengan median detak &lt; 60 bpm dideteksi S1 dengan Se {_se_lambat(R)}, jadi menekan alarm berdasarkan detak saja akan membuang AF "
            "sungguhan dalam jumlah besar. Detak di tabel dari detektor perangkat.")


def EKS_TEKS(R):
    pr = R["eksplorasi"]["pita_hr_ref"]
    return ("Detak dibaca dari anotasi referensi LTAFDB (.atr), bukan detektor. Semua window sinus bradikardia memang &lt; 60 bpm menurut "
            f"anotasi. Alarm S1 menurut pita detak anotasi: &lt; 40 bpm {pct(pr['< 40']['alarm_s1'], 1)}, 40–45 {pct(pr['40–45']['alarm_s1'], 1)}, "
            f"45–50 {pct(pr['45–50']['alarm_s1'], 1)}, 50–55 {pct(pr['50–55']['alarm_s1'], 1)}; polanya tidak monoton dan mengikuti subjek, bukan "
            "detak. Eksplorasi ini dilihat setelah hasil dan tidak dipakai memilih apa pun.")


def METODE(R):
    return [
        "<b>Sistem.</b> S1 = D-1 (ECGFounder 1-lead zero-shot, AF bila d1 &gt; −1) + aturan episode AF ≥ 30 s (v16.5, tanpa celah), referensi "
        "v3.1 + aturan yang sama. Keputusan per window dari cache v16.2–v16.7; tidak ada sinyal yang dibaca ulang dan tidak ada training. "
        "Jumlah episode palsu diperiksa sama dengan episode.metrik v16.5 (assert) di semua dataset.",
        "<b>Metrik.</b> Episode palsu = episode deteksi tanpa tumpang tindih dengan AF referensi. Laju per ritme/pita = episode palsu ritme/pita "
        "itu ÷ jam window non-AF ritme/pita itu (label ritme terbanyak / median detak di dalam episode). Fraksi waktu = proporsi window non-AF "
        "di dalam alarm. Bootstrap subjek (atau rekaman) berpasangan 2.000×, seed 2026.",
        "<b>Kendali mutu.</b> Laporan menghitung ulang laju palsu sinus bradikardia LTAFDB (S1, v3.1) dan fraksi alarm lambat/acuan Icentia11k "
        "(S1) dari cache (assert). Protokol butir 9 (uji stres takikardia) tidak berlaku: tidak ada model baru; S1 sudah diuji di v16.",
    ]


def KESIMPULAN(R):
    return [
        "<b>Kesimpulan.</b> S1 tetap sistem terbaik untuk deteksi AF 2 kelas, tetapi punya kelemahan kedua yang sekarang terukur: alarm AF "
        "palsu pada detak lambat, terutama &lt; 50 bpm dan pada sebagian kecil orang. v3.1 tidak punya kelemahan ini karena hanya melihat RR "
        "yang teratur. Aturan detak sederhana bukan solusi karena AF lambat sering terjadi.",
        "<b>Langkah berikutnya.</b> Uji perangkat harus memuat rekaman tidur/malam dan orang dengan detak istirahat rendah, dengan kriteria "
        "alarm palsu per jam pada &lt; 50 bpm. Hipotesis (belum diuji, butuh data perangkat): saat detak lambat dan RR sangat teratur, alarm "
        "D-1 diturunkan menjadi \"perlu dicek\". Versi global (D-1 dan v3.1, S2 di v16.5) kehilangan sensitivitas, jadi hanya varian terbatas "
        "pada detak lambat yang masuk akal diuji.",
        "<b>Batasan.</b> Label sinus bradikardia hanya ada di LTAFDB (25 subjek, semuanya ber-AF); 3 subjek menentukan hasil P1. Detak lambat "
        "di dataset lain didefinisikan dari detektor, bukan label ritme, sehingga bisa memuat ritme lain. Georgia (EKG 10 s) tidak menunjukkan "
        "masalah (v16.4: 5,0% vs 5,6%), mungkin karena pasien klinisnya berbeda; aturan 30 s tidak bisa diterapkan di sana. Paket v1 tidak "
        "dinilai: pertanyaannya tentang S1 vs paket produksi v3.1.",
        "<b>Reproduksi.</b> Dari root repo: python tools/run_notebook.py v16.9_sinus_bradikardia/bradikardia.ipynb (butuh cache v16.2–v16.7 "
        f"dan anotasi data/ltafdb; ±{fnum(R['durasi_menit'], 0)} menit, tanpa GPU). Lalu dari folder v16.9_sinus_bradikardia: python laporan/build_report.py --extract, "
        "lalu python laporan/build_report.py.",
    ]


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
