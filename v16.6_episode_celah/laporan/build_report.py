"""Laporan PDF v16.6: aturan episode AF dengan toleransi celah (dipilih di data lama) dan validasi di SHDB-AF, afpdb, incartdb, svdb.

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v16.6/ringkasan.json (sel akhir celah.ipynb), hitung ulang alarm palsu per jam dan Se episode di data
     validasi dari cache per rekaman lalu cocokkan (assert), bootstrap S1 (D-1 + 30 s) vs v3.1 + 30 s di SHDB-AF, gambar grafik.
  2. tanpa argumen: rakit laporan/laporan_v16_6_episode_celah.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v16_6_episode_celah.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, Laporan, fnum, pct, sg, sumbu_koma, thou  # noqa: E402

TANGGAL = "10 November 2026"
SIS = ("D-1 terpilih", "v3.1 terpilih", "D-1 30 s", "v3.1 30 s")
GAYA = {"D-1 terpilih": (ORANYE, "s", "//"), "v3.1 terpilih": (BIRU, "o", "//"), "D-1 30 s": (ORANYE, "s", None), "v3.1 30 s": (BIRU, "o", None)}
VAL = ("SHDB-AF", "afpdb", "incartdb", "svdb")


def extract():
    import numpy as np
    import celah as C
    R = json.loads((C.CACHE / "ringkasan.json").read_text(encoding="utf-8"))
    sp, g = R["tahapA"]["terpilih"]["span"], R["tahapA"]["terpilih"]["celah"]
    tamb = {}
    for ds in VAL:
        d, _ = C.muat(ds)
        hs = {}
        for s, (k, a, b) in {"D-1 terpilih": ("d1", sp, g), "v3.1 terpilih": ("v31", sp, g), "D-1 30 s": ("d1", 30, 0),
                             "v3.1 30 s": ("v31", 30, 0)}.items():
            m = C.nilai(d, d[k], a, b)
            for kk in ("palsu_per_jam", "se_episode_30"):
                x, y = m[kk], R["hasil"][ds][s][kk]
                assert (np.isnan(x) and np.isnan(y)) or abs(x - y) < 1e-9, (ds, s, kk)
            hs[s] = m
        tamb[ds] = {"ci_palsu_S1": C.boot_rasio(hs["D-1 30 s"]["palsu_unit"], hs["D-1 30 s"]["palsu_unit"])["ci_a"],
                    "S1_vs_R_palsu": C.boot_rasio(hs["D-1 30 s"]["palsu_unit"], hs["v3.1 30 s"]["palsu_unit"])}
        if ds == "SHDB-AF":
            tamb[ds]["S1_vs_R_se"] = C.boot_rasio(hs["D-1 30 s"]["se_unit"], hs["v3.1 30 s"]["se_unit"])
            tamb[ds]["ci_se_S1"] = C.boot_rasio(hs["D-1 30 s"]["se_unit"], hs["D-1 30 s"]["se_unit"])["ci_a"]
            pu = hs["D-1 30 s"]["palsu_unit"]
            tamb[ds]["per_subjek_S1"] = {u: v[0] / v[1] for u, v in pu.items() if v[1] > 0}
    TMP.mkdir(exist_ok=True)
    (TMP / "v16_6.json").write_text(json.dumps({"R": R, "T": tamb}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    grafik(R, tamb)
    print("extract selesai; assert alarm palsu per jam dan Se episode (4 sistem × 4 data validasi) cocok dengan ringkasan.json")


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


def grafik(R, T):
    plt = _gaya()
    # 1. grid tahap A: alarm palsu NSRDB vs Se episode minimum
    rows = R["tahapA"]["grid"]
    fig, ax = plt.subplots(figsize=(7.6, 3.0), dpi=200)
    mk = {0: ("o", BIRU), 1: ("^", HIJAU), 2: ("s", ORANYE)}
    for gg in (0, 1, 2):
        rr = [r for r in rows if r["celah"] == gg]
        ax.plot([max(r["palsu NSRDB"], 1e-3) for r in rr], [r["Se30 min"] for r in rr], marker=mk[gg][0], color=mk[gg][1], lw=1,
                label=f"celah {gg} window", ms=6)
        for r in rr:
            ax.annotate(f"{r['span']} s", (max(r["palsu NSRDB"], 1e-3), r["Se30 min"]), textcoords="offset points", xytext=(5, -3),
                        fontsize=7, color=GRAY)
    t = R["tahapA"]["terpilih"]
    sel = next(r for r in rows if r["span"] == t["span"] and r["celah"] == t["celah"])
    ax.scatter([sel["palsu NSRDB"]], [sel["Se30 min"]], s=160, facecolors="none", edgecolors=INK, lw=1.2, zorder=5)
    ax.axvline(1, color=INK, lw=.8, ls="--"); ax.axhline(.9, color=INK, lw=.8, ls="--")
    ax.set_xscale("log"); ax.set_xlabel("alarm palsu per jam di NSRDB (log; 0 digambar di 0,001)", fontsize=8.5)
    ax.set_ylabel("Se episode ≥ 30 s,\nminimum 4 data AF", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right"); ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#E6ECF1")
    sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "grid.png"); plt.close(fig)
    # 2. validasi: alarm palsu per jam per data, 4 sistem
    H = R["hasil"]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8), dpi=200, gridspec_kw=dict(width_ratios=[5, 2]))
    for i, s in enumerate(SIS):
        col, _, hat = GAYA[s]
        axs[0].bar([x + (i - 1.5) * .2 for x in range(4)], [H[ds][s]["palsu_per_jam"] for ds in VAL], .19, color=col, hatch=hat,
                   edgecolor="white", label=s.replace("terpilih", f"{t['span']} s"))
        axs[1].bar(i, H["SHDB-AF"][s]["se_episode_30"], .7, color=col, hatch=hat, edgecolor="white")
    axs[0].axhline(1, color=INK, lw=.8, ls="--"); axs[0].set_xticks(range(4), ["SHDB-AF\n(non-AF)", "afpdb", "incartdb", "svdb\n(deskriptif)"], fontsize=8)
    axs[0].set_ylabel("alarm palsu per jam", fontsize=8.5); axs[0].legend(frameon=False, fontsize=7, ncol=2)
    axs[1].axhline(.9, color=INK, lw=.8, ls="--"); axs[1].set_ylim(.5, 1); axs[1].set_xticks(range(4), ["D-1", "v3.1", "D-1", "v3.1"], fontsize=7.5)
    axs[1].set_title(f"SHDB-AF: Se episode ≥ 30 s\n({t['span']} s arsir | 30 s polos)", fontsize=8, color=INK)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "validasi.png"); plt.close(fig)
    # 3. SHDB-AF: alarm non-AF per pita detak
    P = R["pita"]["SHDB-AF"]
    bands = list(P["D-1 30 s"])
    fig, ax = plt.subplots(figsize=(7.6, 2.4), dpi=200)
    for i, s in enumerate(SIS):
        col, _, hat = GAYA[s]
        ax.bar([x + (i - 1.5) * .2 for x in range(len(bands))], [(P[s][b]["alarm"] or 0) * 100 for b in bands], .19, color=col, hatch=hat,
               edgecolor="white", label=s.replace("terpilih", f"{t['span']} s"))
    ax.set_xticks(range(len(bands)), [f"{b} bpm\n(n = {thou(P['D-1 30 s'][b]['n'])})" for b in bands], fontsize=7.5)
    ax.set_ylabel("window non-AF\nberalarm (%)", fontsize=8.5); ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "pita.png"); plt.close(fig)


def build():
    J = json.loads((TMP / "v16_6.json").read_text(encoding="utf-8"))
    R, T = J["R"], J["T"]
    H, k = R["hasil"], R["keputusan"]
    t = R["tahapA"]["terpilih"]
    ok = lambda x: "lolos" if x else "gagal"  # noqa: E731
    s1 = H["SHDB-AF"]["D-1 30 s"]
    L = Laporan(OUT, "v16.6: aturan episode dengan celah", "SIHEDAF  /  V16.6  /  CELAH & VALIDASI",
                "v16.6_episode_celah/celah.ipynb, cache/v16.6/", TANGGAL)
    y = L.page("Ringkasan: konfigurasi terpilih gagal, aturan klinis 30 s lolos di data baru",
               "Rancangan dikunci 10 November 2026 sebelum sinyal data validasi dibaca (plus amandemen A, juga sebelum sinyal)")
    y = L.kartu([(fnum(H["SHDB-AF"]["D-1 terpilih"]["se_episode_30"], 2), f"Se episode, {t['span']} s (terpilih)", "SHDB-AF, target 0,90"),
                 (fnum(s1["se_episode_30"], 2), "Se episode, D-1 + 30 s", f"SHDB-AF, {thou(R['data']['SHDB-AF']['unit'])} subjek"),
                 (fnum(s1["palsu_per_jam"], 2), "alarm palsu/jam, D-1 + 30 s", f"SHDB-AF, {thou(R['data']['SHDB-AF']['jam'])} jam"),
                 (fnum(H["SHDB-AF"]["v3.1 30 s"]["palsu_per_jam"], 2), "alarm palsu/jam, v3.1 + 30 s", "SHDB-AF")], y)
    y = L.para("<b>Pertanyaan.</b> v16.5: D-1 + aturan episode ≥ 30 s hampir memenuhi target produk (NSRDB 1,23 &gt; 1,0 alarm palsu per "
               "jam). Apakah durasi minimum lain dan toleransi celah (run tidak putus karena 1–2 window yang terlewat) memberi titik kerja "
               "yang lebih baik, dan apakah titik kerja itu bertahan di data yang belum pernah dipakai?", y)
    y = L.para(JAWABAN(R, T), y)
    rows = [["Target: Se episode SHDB-AF ≥ 0,90 (konfigurasi terpilih)", ok(k["target_se"]),
             f"{fnum(R['target']['se_shdb'])} [{fnum(R['ci']['se_shdb'][0])}–{fnum(R['ci']['se_shdb'][1])}]"],
            ["Target: alarm palsu ≤ 1,0/jam di SHDB-AF, afpdb, incartdb", ok(k["target_palsu"]),
             ", ".join(fnum(v, 2) for v in R["target"]["palsu"].values())],
            ["H1: palsu/jam SHDB-AF D-1 &lt; v3.1 (konfigurasi sama)", ok(k["H1"]),
             f"Δ {sg(R['H1']['delta'], 2)} [{sg(R['H1']['ci'][0], 2)}; {sg(R['H1']['ci'][1], 2)}], P = {fnum(R['H1']['p_kurang'])}"],
            ["H2: Se episode SHDB-AF D-1 bukan lebih buruk dari v3.1", ok(k["H2"]),
             f"Δ {sg(R['H2']['delta'])}, P(Δ &gt; 0) = {fnum(R['H2']['p_lebih'])}"],
            ["Keputusan: konfigurasi terpilih = sistem kandidat terbaik", "tidak", "target Se gagal"]]
    y = L.table(["Kriteria yang dikunci", "Hasil", "Angka"], rows, y, widths=[L.CW * .5, L.CW * .1, L.CW * .4], sorot=(5,))
    y = L.para(TEMUAN(R, T), y)
    # 2. tahap A
    y = L.page("Tahap A: grid durasi × celah di data pengembangan", "Data v16.5 (sudah dipakai); sistem D-1; pemilihan dikunci")
    y = L.chart(TMP / "grid.png", y, 205)
    rows = [[f"{r['span']} s", str(r["celah"]), fnum(r["palsu NSRDB"], 3), fnum(r["palsu EDB"], 3), fnum(r["palsu Latihan fisik"], 3),
             fnum(r["Se30 min"])] for r in R["tahapA"]["grid"]]
    sel = [i + 1 for i, r in enumerate(R["tahapA"]["grid"]) if r["span"] == t["span"] and r["celah"] == t["celah"]]
    y = L.table(["Durasi min.", "Celah", "Palsu/jam NSRDB", "Palsu/jam EDB", "Palsu/jam latihan", "Se ep. ≥ 30 s (min 4 data)"], rows, y,
                widths=[L.CW * .14, L.CW * .1, L.CW * .19, L.CW * .17, L.CW * .19, L.CW * .21], sorot=tuple(sel))
    y = L.para(GRID_TEKS(R), y)
    # 3. tahap B
    y = L.page("Tahap B: validasi di data yang belum pernah dipakai", "Bootstrap subjek/rekaman berpasangan 2.000×, seed 2026")
    y = L.chart(TMP / "validasi.png", y, 195)
    rows = []
    for ds in VAL:
        for s in SIS:
            m = H[ds][s]
            rows.append([ds if s == SIS[0] else "", s.replace("terpilih", f"{t['span']} s"), fnum(m["palsu_per_jam"], 2),
                         "–" if m["se_episode_30"] != m["se_episode_30"] else fnum(m["se_episode_30"]),
                         "–" if m["se_durasi"] != m["se_durasi"] else fnum(m["se_durasi"]),
                         "–" if not m["ppv_durasi"] else fnum(m["ppv_durasi"])])
    y = L.table(["Data", "Sistem", "Palsu/jam", "Se episode ≥ 30 s", "Se durasi", "+P durasi"], rows, y,
                widths=[L.CW * .15, L.CW * .2, L.CW * .14, L.CW * .19, L.CW * .16, L.CW * .16])
    # 4. sekunder S1
    y = L.page("Analisis sekunder: S1 = D-1 + 30 s (aturan v16.5) di data baru", "S1 ditetapkan di v16.5 sebelum data ini dilihat")
    y = L.para(S1_TEKS(R, T), y)
    sh = T["SHDB-AF"]
    rows = [[ds, f"{fnum(H[ds]['D-1 30 s']['palsu_per_jam'], 2)} [{fnum(T[ds]['ci_palsu_S1'][0], 2)}–{fnum(T[ds]['ci_palsu_S1'][1], 2)}]",
             fnum(H[ds]["v3.1 30 s"]["palsu_per_jam"], 2),
             f"{sg(T[ds]['S1_vs_R_palsu']['delta'], 2)}, P = {fnum(T[ds]['S1_vs_R_palsu']['p_kurang'])}"] for ds in VAL]
    y = L.table(["Data", "S1 palsu/jam [CI]", "v3.1 + 30 s", "Δ S1 − v3.1, P(Δ &lt; 0)"], rows, y,
                widths=[L.CW * .2, L.CW * .3, L.CW * .2, L.CW * .3], sorot=(1,))
    y = L.para(f"Se episode ≥ 30 s SHDB-AF: S1 {fnum(s1['se_episode_30'])} [{fnum(sh['ci_se_S1'][0])}–{fnum(sh['ci_se_S1'][1])}] vs v3.1 + 30 s "
               f"{fnum(H['SHDB-AF']['v3.1 30 s']['se_episode_30'])}; Δ {sg(sh['S1_vs_R_se']['delta'])} [{sg(sh['S1_vs_R_se']['ci'][0])}; "
               f"{sg(sh['S1_vs_R_se']['ci'][1])}], P(Δ &gt; 0) = {fnum(sh['S1_vs_R_se']['p_lebih'])}.", y)
    y = L.chart(TMP / "pita.png", y, 165)
    y = L.para(PITA_TEKS(R, T), y)
    # 5. metode
    y = L.page("Metode, pra-registrasi, kendali mutu", "RANCANGAN.md dikunci; sha256 di logs/v16.6_praregistrasi.log")
    for tx in METODE(R):
        y = L.para(tx, y)
    # 6. kesimpulan
    y = L.page("Kesimpulan, sistem terbaik saat ini, batasan, reproduksi", "")
    for tx in KESIMPULAN(R, T):
        y = L.para(tx, y)
    lk = R["lingkungan"]
    y = L.para(f"Lingkungan: PC desktop, GPU {lk['gpu']}, commit {lk['commit']} (+ perubahan belum di-commit), Python {lk['pustaka']['python']}, "
               f"numpy {lk['pustaka']['numpy']}, torch {lk['pustaka']['torch']}.", y, "small")
    L.selesai()
    print("PDF:", OUT, L.page_no, "halaman")


# ── teks naratif (ditulis setelah hasil dilihat; angka dari ringkasan) ──
def JAWABAN(R, T):
    H, t = R["hasil"], R["tahapA"]["terpilih"]
    s1, sel = H["SHDB-AF"]["D-1 30 s"], H["SHDB-AF"]["D-1 terpilih"]
    return ("<b>Jawaban singkat.</b> Toleransi celah tidak membantu: alarm palsu naik jauh lebih cepat daripada sensitivitas. Tahap A memilih "
            f"durasi {t['span']} s tanpa celah. Di SHDB-AF konfigurasi itu memenuhi batas alarm palsu ({fnum(sel['palsu_per_jam'], 2)}/jam) tetapi "
            f"sensitivitas episode hanya {fnum(sel['se_episode_30'], 2)}, jadi <b>gagal</b>. Aturan klinis v16.5 (S1 = D-1 + 30 s) yang diuji "
            f"sebagai analisis sekunder justru <b>memenuhi kedua target di semua data baru</b>: Se SHDB-AF {fnum(s1['se_episode_30'], 3)}, alarm "
            f"palsu {fnum(s1['palsu_per_jam'], 2)} (SHDB-AF), {fnum(H['afpdb']['D-1 30 s']['palsu_per_jam'], 2)} (afpdb), "
            f"{fnum(H['incartdb']['D-1 30 s']['palsu_per_jam'], 2)} (incartdb) per jam (nilai titik; CI alarm palsu SHDB-AF sampai "
            f"{fnum(T['SHDB-AF']['ci_palsu_S1'][1], 2)}).")


def TEMUAN(R, T):
    H = R["hasil"]
    return ("<b>Temuan.</b> (1) Kriteria pemilihan yang dikunci salah sasaran: sensitivitas diukur untuk episode ≥ 30 s, tetapi aturan "
            "≥ 45 s tidak bisa mengalarmkan episode 30–45 s, dan batas alarm palsu yang ketat di NSRDB mendorong pemilihan ke sana. "
            "(2) Sejak v16.5, S1 ditetapkan dari definisi klinis, tanpa penyetelan, dan kini lolos di 2.305 jam Holter pasien AF yang belum "
            f"pernah dilihat. (3) Dengan aturan yang sama, v3.1 memberi alarm palsu {fnum(H['SHDB-AF']['v3.1 30 s']['palsu_per_jam'], 2)} "
            f"(SHDB-AF) sampai {fnum(H['incartdb']['v3.1 30 s']['palsu_per_jam'], 2)} (incartdb) per jam. (4) svdb (aritmia supraventrikular, "
            f"ritme tidak dianotasi) tetap sulit: S1 {fnum(H['svdb']['D-1 30 s']['palsu_per_jam'], 2)} per jam (deskriptif).")


def GRID_TEKS(R):
    rows = {(r["span"], r["celah"]): r for r in R["tahapA"]["grid"]}
    return ("Celah 1 window menaikkan Se minimum (30 s: "
            f"{fnum(rows[(30, 0)]['Se30 min'])} → {fnum(rows[(30, 1)]['Se30 min'])}) tetapi alarm palsu NSRDB hampir tiga kali "
            f"({fnum(rows[(30, 0)]['palsu NSRDB'], 2)} → {fnum(rows[(30, 1)]['palsu NSRDB'], 2)}). Pada orang sehat, window AF palsu muncul "
            "berkelompok dengan sela pendek, sehingga celah menyambung kelompok itu menjadi episode. Hanya lima konfigurasi yang memenuhi batas "
            "alarm palsu di ketiga data negatif; di antaranya 45 s tanpa celah punya Se minimum tertinggi, jadi terpilih sesuai aturan. Lingkaran "
            "hitam = terpilih.")


def S1_TEKS(R, T):
    H = R["hasil"]
    return ("Rancangan v16.6 mencantumkan D-1 + 30 s tanpa celah sebagai pembanding sekunder. Aturan ini identik dengan S1 v16.5, yang "
            "ditetapkan sebelum data validasi dilihat, sehingga angkanya di bawah ini adalah uji independen dari aturan itu. Tetapi keputusan "
            "yang dikunci untuk v16.6 merujuk ke konfigurasi terpilih, jadi status S1 ditulis sebagai temuan sekunder, bukan vonis utama. "
            f"Di SHDB-AF S1 menangkap {pct(H['SHDB-AF']['D-1 30 s']['se_episode_30'], 1)} dari {thou(H['SHDB-AF']['D-1 30 s']['n_ref_30'])} "
            "episode AF ≥ 30 s. Kedua target terpenuhi pada nilai titik, tetapi CI-nya menyentuh batas: Se [" +
            f"{fnum(T['SHDB-AF']['ci_se_S1'][0])}–{fnum(T['SHDB-AF']['ci_se_S1'][1])}], alarm palsu SHDB-AF [{fnum(T['SHDB-AF']['ci_palsu_S1'][0], 2)}–"
            f"{fnum(T['SHDB-AF']['ci_palsu_S1'][1], 2)}] per jam.")


def PITA_TEKS(R, T):
    P = R["pita"]["SHDB-AF"]
    ps = T["SHDB-AF"]["per_subjek_S1"]
    n_atas = sum(v > 1 for v in ps.values())
    return (f"Pada waktu non-AF pasien SHDB-AF, S1 mengalarmkan {pct(P['D-1 30 s']['≥ 120']['alarm'], 0)} window ≥ 120 bpm, v3.1 + 30 s "
            f"{pct(P['v3.1 30 s']['≥ 120']['alarm'], 0)}. Di bawah 100 bpm: {pct(P['D-1 30 s']['< 100']['alarm'], 1)} vs "
            f"{pct(P['v3.1 30 s']['< 100']['alarm'], 1)}. Per subjek, {n_atas} dari {len(ps)} subjek SHDB-AF di atas 1 alarm palsu per jam "
            f"dengan S1 (tertinggi {fnum(max(ps.values()), 2)}).")


def METODE(R):
    d = R["data"]
    return [
        "<b>Aturan dengan celah.</b> Run = window AF di mana jarak awal antar window AF berurutan ≤ langkah × (g + 1) + 0,5 s (hingga g window "
        "tidak-AF atau gagal cek kualitas dijembatani). Alarm untuk run dengan rentang ≥ durasi minimum; semua window di rentang itu ditandai "
        "alarm. g = 0 dan 30 s identik dengan aturan v16.5 (assert di semua data pengembangan).",
        "<b>Tahap A (dikunci).</b> Grid durasi {30, 45, 60, 90} s × celah {0, 1, 2} untuk D-1 di data v16.5 (AFDB, LTAFDB, MIT-BIH, CPSC 2021; "
        "NSRDB, EDB, latihan fisik). Di antara konfigurasi dengan alarm palsu ≤ 1,0/jam di ketiga data negatif, pilih Se episode ≥ 30 s minimum "
        "di empat data AF terbesar (seri ≤ 0,005 → alarm palsu NSRDB + EDB terendah).",
        "<b>Tahap B (data baru).</b> " + "; ".join(f"{ds}: {thou(v['window'])} window, {fnum(v['jam'], 0)} jam, {thou(v['unit'])} unit" for ds, v in d.items()) +
        ". SHDB-AF (Jepang, ODC-BY): 98 Holter 24 jam 200 Hz, ECG1, unit = subjek. afpdb: n01–n50 dan p01–p50 (rekaman c dan t tidak "
        "dipakai). incartdb: lead I, unit = pasien. svdb: ritme tidak dianotasi, deskriptif. Rantai prediksi = v16.4 (50 Hz, detektor "
        "perangkat, cek kualitas, ECGFounder d1 &gt; −1, paket v3.1).",
        "<b>Label.</b> Label ritme = aux diawali '(' dan berlaku sampai label berikutnya; AF = (AFIB, (AFL, (WPWAF; window yang melintasi "
        "batas dibuang. Amandemen A (sebelum sinyal dibaca): incartdb dan svdb hanya menandai perubahan dari ritme normal, jadi waktu sebelum "
        "label pertama dihitung non-AF.",
        "<b>Kendali mutu.</b> Laporan ini menghitung ulang alarm palsu per jam dan Se episode untuk 4 sistem × 4 data validasi dari cache per "
        "rekaman dan mencocokkannya dengan ringkasan notebook (assert).",
    ]


def KESIMPULAN(R, T):
    H = R["hasil"]
    return [
        "<b>Kesimpulan.</b> Vonis utama v16.6 gagal: konfigurasi yang dipilih di data lama (45 s) kehilangan terlalu banyak episode AF di "
        "SHDB-AF. Toleransi celah tidak membantu. Aturan klinis 30 s tanpa celah (S1) bertahan di data baru dan memenuhi kedua target di "
        "SHDB-AF, afpdb, dan incartdb.",
        "<b>Sistem terbaik saat ini (detektor AF 2 kelas, episode).</b> D-1 (ECGFounder 1-lead zero-shot, d1 &gt; −1) + aturan episode AF "
        "≥ 30 s. Bukti: CPSC 2021 (v16.2), MIT-BIH/LTAFDB (v16.3), NSRDB/Georgia (v16.4), episode di 7 data (v16.5), dan SHDB-AF/afpdb/incartdb "
        "(v16.6). Kelemahan yang tercatat: sensitivitas lebih rendah di EKG pendek non-Holter (CinC 2017, Icentia, Georgia), alarm palsu sedikit "
        "di atas 1/jam di NSRDB dan svdb, dan tidak membedakan AFIB dari AFL. Paket 3 kelas tetap v3.1; tidak ada model_export yang diubah.",
        "<b>Langkah berikutnya.</b> Semua data publik Holter yang relevan kini terpakai. Titik kerja S1 harus diuji di rekaman perangkat SIHEDAF "
        "berlabel (alarm palsu per jam saat aktivitas harian dan Se episode), dengan rancangan yang dikunci sebelum data dilihat.",
        "<b>Batasan.</b> S1 di v16.6 adalah analisis sekunder. SHDB-AF dianotasi satu orang; label episode referensi = run window berlabel AF. "
        "incartdb hanya punya 4 episode AF ≥ 30 s. svdb mungkin berisi AF/SVT tanpa label. Waktu non-AF SHDB-AF milik pasien AF, jadi alarm "
        "palsunya bisa lebih tinggi daripada pada orang sehat.",
        "<b>Reproduksi.</b> Dari root repo: python tools/unduh_physionet.py svdb 1.0.0 --ke svdb (juga afpdb, incartdb; --pola '*.dat' '*.hea' "
        "'*.atr' '*.qrs' RECORDS), SHDB-AF lewat python v12_shdb_atrium/atrium.py unduh; lalu python tools/run_notebook.py "
        "v16.6_episode_celah/celah.ipynb (pemrosesan per rekaman ±22 menit dengan 2 proses, lewat celah.jalankan). Lalu dari folder "
        "v16.6_episode_celah: python laporan/build_report.py --extract, lalu python laporan/build_report.py.",
    ]


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
