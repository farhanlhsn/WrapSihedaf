"""Laporan PDF v16.5: aturan episode AF ≥ 30 s untuk D-1 dan v3.1 (alarm palsu per jam vs episode AF yang terdeteksi).

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract: baca cache/v16.5/ringkasan.json (sel akhir episode.ipynb), hitung ulang alarm palsu per jam NSRDB dan EDB dari cache per
     rekaman lalu cocokkan (assert), gambar grafik → laporan/_build/.
  2. tanpa argumen: rakit laporan/laporan_v16_5_episode_30s.pdf.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
REPO = V.parent
TMP = HERE / "_build"
OUT = HERE / "laporan_v16_5_episode_30s.pdf"
sys.path[:0] = [str(REPO / "tools"), str(V)]
from laporan_sihedaf import BIRU, ORANYE, HIJAU, INK, GRAY, Laporan, fnum, pct, sg, sumbu_koma, thou  # noqa: E402

TANGGAL = "10 November 2026"
SISTEM = ("v3.1", "D-1", "S2 (D-1 & v3.1)")
NAMA = {"v3.1": "R = v3.1 + 30 s (referensi)", "D-1": "S1 = D-1 + 30 s (primer)", "S2 (D-1 & v3.1)": "S2 = (D-1 dan v3.1) + 30 s"}
PENDEK = {"v3.1": "R (v3.1)", "D-1": "S1 (D-1)", "S2 (D-1 & v3.1)": "S2 (D-1 dan v3.1)"}
GAYA = {"v3.1": (BIRU, "o"), "D-1": (ORANYE, "s"), "S2 (D-1 & v3.1)": (HIJAU, "^")}
AF_DS = ("AFDB", "LTAFDB", "MIT-BIH", "CPSC 2021")
NEG_DS = ("NSRDB", "EDB", "Latihan fisik")


def extract():
    import numpy as np
    import episode as E
    R = json.loads((E.CACHE / "ringkasan.json").read_text(encoding="utf-8"))
    for ds, fn in (("NSRDB", E._nsrdb), ("EDB", E._edb)):
        d = fn()
        for s, af in (("D-1", d["d1"]), ("v3.1", d["v31"][0])):
            al = E.aturan(d["rec"], d["start"], af, d["langkah"], 30)
            m = E.metrik(d["rec"], d["start"], d["y2"], al, d["langkah"], d["unit"])
            assert abs(m["palsu_per_jam"] - R["hasil"][ds][s]["30"]["palsu_per_jam"]) < 1e-9, (ds, s)
    TMP.mkdir(exist_ok=True)
    (TMP / "v16_5.json").write_text(json.dumps({"R": R}, indent=1, ensure_ascii=False), encoding="utf-8")
    grafik(R)
    print("extract selesai; assert alarm palsu per jam NSRDB dan EDB (S1, R) cocok dengan ringkasan.json")


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
    H = R["hasil"]
    # 1. alarm palsu per jam (30 s) di data negatif + Se episode ≥ 30 s di data AF
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.7), dpi=200, gridspec_kw=dict(width_ratios=[3, 4]))
    for i, s in enumerate(SISTEM):
        col, mk = GAYA[s]
        axs[0].bar([x + (i - 1) * .26 for x in range(3)], [H[ds][s]["30"]["palsu_per_jam"] for ds in NEG_DS], .24, color=col,
                   label=PENDEK[s], hatch=("//" if s == "D-1" else None), edgecolor="white" if s == "D-1" else None)
        axs[1].plot(range(4), [H[ds][s]["30"]["se_episode_30"] for ds in AF_DS], marker=mk, color=col, lw=1.4, ms=6, label=PENDEK[s])
    axs[0].axhline(R_TARGET_PALSU, color=INK, lw=.9, ls="--"); axs[0].text(2.45, R_TARGET_PALSU + .2, "target 1,0", ha="right", fontsize=7.5)
    axs[0].set_xticks(range(3), ["NSRDB", "EDB", "latihan\nfisik"], fontsize=8); axs[0].set_ylabel("alarm palsu per jam", fontsize=8.5)
    axs[0].legend(frameon=False, fontsize=7.5); axs[0].set_title("Data tanpa AF (aturan 30 s)", fontsize=9, color=INK)
    axs[1].axhline(.9, color=INK, lw=.9, ls="--"); axs[1].text(-.1, .903, "target 0,90", ha="left", va="bottom", fontsize=7.5)
    axs[1].legend(frameon=False, fontsize=7.5, loc="lower left")
    axs[1].set_xticks(range(4), AF_DS, fontsize=8); axs[1].set_ylim(.8, 1.0); axs[1].set_ylabel("Se episode AF ≥ 30 s", fontsize=8.5)
    axs[1].set_title("Data dengan AF (aturan 30 s)", fontsize=9, color=INK)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "utama.png"); plt.close(fig)
    # 2. kurva durasi: alarm palsu NSRDB vs durasi (log) dan Se episode ≥ 30 s LTAFDB
    K = R["kurva"]
    dur = ["0", "30", "60", "120", "360"]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.6), dpi=200)
    for s in SISTEM:
        col, mk = GAYA[s]
        axs[0].plot(range(5), [max(K[s]["NSRDB"][d][0], 1e-3) for d in dur], marker=mk, color=col, label=PENDEK[s], lw=1.3)
        axs[1].plot(range(5), [K[s]["LTAFDB"][d][1] for d in dur], marker=mk, color=col, lw=1.3)
    axs[0].set_yscale("log"); axs[0].axhline(1, color=INK, lw=.8, ls="--")
    axs[0].set_title("NSRDB: alarm palsu per jam", fontsize=9, color=INK); axs[0].legend(frameon=False, fontsize=7.5)
    axs[1].set_title("LTAFDB: Se episode AF ≥ 30 s", fontsize=9, color=INK)
    for ax in axs:
        ax.set_xticks(range(5), ["tanpa", "30 s", "60 s", "120 s", "360 s"], fontsize=8); ax.set_xlabel("durasi minimum aturan", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1")
    sumbu_koma(axs[1])
    fig.tight_layout(); fig.savefig(TMP / "kurva.png"); plt.close(fig)
    # 3. NSRDB per subjek (S1)
    ps = R["nsrdb_per_subjek_S1"]
    sub = sorted(ps, key=lambda k: ps[k])
    fig, ax = plt.subplots(figsize=(7.6, 2.2), dpi=200)
    ax.bar(range(len(sub)), [ps[k] for k in sub], color=ORANYE, hatch="//", edgecolor="white")
    ax.axhline(1, color=INK, lw=.9, ls="--")
    ax.set_xticks(range(len(sub)), sub, rotation=90, fontsize=7); ax.set_ylabel("alarm palsu per jam", fontsize=8.5)
    ax.set_xlabel("subjek NSRDB, S1 = D-1 + 30 s", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6ECF1"); sumbu_koma(ax)
    fig.tight_layout(); fig.savefig(TMP / "subjek.png"); plt.close(fig)


R_TARGET_PALSU = 1.0


def build():
    R = json.loads((TMP / "v16_5.json").read_text(encoding="utf-8"))["R"]
    H, T = R["hasil"], R["target"]
    ok = lambda x: "lolos" if x else "gagal"  # noqa: E731
    L = Laporan(OUT, "v16.5: aturan episode AF ≥ 30 s", "SIHEDAF  /  V16.5  /  EPISODE ≥ 30 s",
                "v16.5_episode_30s/episode.ipynb, cache/v16.5/", TANGGAL)
    s1 = H
    rn = H["NSRDB"]["D-1"]["0"]["palsu_per_jam"] / H["NSRDB"]["D-1"]["30"]["palsu_per_jam"]
    y = L.page(f"Ringkasan: aturan 30 s memangkas alarm palsu D-1 ±{fnum(rn, 0)}×, S1 hampir memenuhi target",
               "Rancangan dikunci 10 November 2026 sebelum aturan dihitung dan sebelum sinyal EDB dibaca")
    y = L.kartu([(f"{fnum(s1['NSRDB']['D-1']['0']['palsu_per_jam'], 1)} → {fnum(s1['NSRDB']['D-1']['30']['palsu_per_jam'], 2)}",
                  "alarm palsu/jam, orang sehat", "NSRDB, D-1, tanpa → 30 s"),
                 (f"{fnum(min(T['D-1']['se'].values()), 2)}–{fnum(max(T['D-1']['se'].values()), 2)}", "Se episode AF ≥ 30 s (S1)",
                  "4 data AF Holter"),
                 (f"{fnum(s1['Latihan fisik']['v3.1']['30']['palsu_per_jam'], 1)} → {fnum(s1['Latihan fisik']['D-1']['30']['palsu_per_jam'], 2)}",
                  "alarm palsu/jam saat olahraga", "v3.1 → D-1, aturan 30 s"),
                 ("tidak ada", "sistem yang memenuhi target", "S1 gagal tipis di NSRDB")], y)
    y = L.para("<b>Pertanyaan.</b> Tanpa aturan durasi, D-1 memberi ±21 episode AF palsu per jam pada orang sehat (v16.4), v3.1 ±52. "
               "Pedoman klinis menyebut episode AF yang tercatat sebagai AF yang berlangsung ≥ 30 detik. Dengan aturan itu, berapa alarm "
               "palsu per jam, dan berapa episode AF yang masih terdeteksi?", y)
    y = L.para(JAWABAN(R), y)
    rows = []
    for s in SISTEM:
        t = T[s]
        rows.append([NAMA[s], ", ".join(f"{fnum(v, 2)}" for v in t["palsu"].values()), ", ".join(f"{fnum(v, 2)}" for v in t["se"].values()),
                     "ya" if t["memenuhi"] else "tidak"])
    y = L.table(["Sistem (aturan 30 s)", "Palsu/jam NSRDB, EDB, latihan", "Se ep. AFDB, LTAFDB, MIT-BIH, CPSC", "Target"], rows, y,
                widths=[L.CW * .3, L.CW * .25, L.CW * .3, L.CW * .15], sorot=(2,))
    h1 = R["H1"]["nsrdb"]
    rows = [["H1: S1 vs R, palsu/jam NSRDB lebih rendah (P(Δ&lt;0) ≥ 0,95)", ok(R["H1"]["lolos"]),
             f"Δ {sg(h1['delta'], 2)} [{sg(h1['ci'][0], 2)}; {sg(h1['ci'][1], 2)}], P = {fnum(h1['p_kurang'])}"],
            ["H2: Se durasi LTAFDB S1 turun ≤ 0,05 vs tanpa aturan", ok(R["H2"]["lolos"]), f"turun {fnum(R['H2']['turun_se_durasi_ltafdb'])}"],
            ["Target produk (alarm ≤ 1,0/jam dan Se ≥ 0,90)", "gagal", "tidak ada sistem yang memenuhi"]]
    y = L.table(["Kriteria yang dikunci", "Hasil", "Angka"], rows, y, widths=[L.CW * .5, L.CW * .12, L.CW * .38])
    y = L.para(TEMUAN(R), y)
    # 2. hasil lengkap
    y = L.page("Hasil per dataset dan sistem (aturan 30 s)", "Episode gaya EC57; R di AFDB = OOF v3.1, rata-rata 5 ulangan")
    y = L.chart(TMP / "utama.png", y, 185)
    rows = []
    for ds in AF_DS + NEG_DS:
        for s in SISTEM:
            m, m0 = H[ds][s]["30"], H[ds][s]["0"]
            se = "–" if ds in ("NSRDB", "Latihan fisik") else fnum(m["se_episode_30"])
            sd = "–" if ds in ("NSRDB", "Latihan fisik") else fnum(m["se_durasi"])
            rows.append([ds if s == "v3.1" else "", s.split(" (")[0] if s != "S2 (D-1 & v3.1)" else "S2", fnum(m0["palsu_per_jam"], 1),
                         fnum(m["palsu_per_jam"], 2), se, sd])
    y = L.table(["Data", "Sistem", "Palsu/jam tanpa aturan", "Palsu/jam 30 s", "Se episode ≥ 30 s", "Se durasi"], rows, y,
                widths=[L.CW * .18, L.CW * .14, L.CW * .18, L.CW * .16, L.CW * .18, L.CW * .16])
    # 3. kurva + subjek
    y = L.page("Kurva durasi dan alarm palsu per subjek", "Durasi lain hanya deskriptif (RANCANGAN bagian 5)")
    y = L.chart(TMP / "kurva.png", y, 175)
    y = L.para(KURVA_TEKS(R), y)
    y = L.chart(TMP / "subjek.png", y, 150)
    y = L.para(SUBJEK_TEKS(R), y)
    # 4. metode
    y = L.page("Metode, pra-registrasi, kendali mutu", "RANCANGAN.md dikunci; sha256 di logs/v16.5_praregistrasi.log")
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
    H, T = R["hasil"], R["target"]
    return ("<b>Jawaban singkat.</b> Aturan 30 s memangkas alarm palsu D-1 di NSRDB dari "
            f"{fnum(H['NSRDB']['D-1']['0']['palsu_per_jam'], 1)} ke {fnum(H['NSRDB']['D-1']['30']['palsu_per_jam'], 2)} per jam, sementara "
            f"{fnum(min(T['D-1']['se'].values()), 2)}–{fnum(max(T['D-1']['se'].values()), 2)} episode AF ≥ 30 s tetap terdeteksi. S1 memenuhi "
            f"target sensitivitas dan alarm palsu di EDB ({fnum(T['D-1']['palsu']['EDB'], 2)}) dan latihan fisik "
            f"({fnum(T['D-1']['palsu']['Latihan fisik'], 2)}), tetapi <b>gagal tipis di NSRDB</b> ({fnum(T['D-1']['palsu']['NSRDB'], 2)} &gt; 1,0). "
            "S2 memenuhi semua batas alarm palsu tetapi sensitivitasnya di bawah 0,90; R (v3.1) gagal di keduanya. Jadi tidak ada sistem yang "
            "memenuhi target, dan sesuai rancangan tidak ada yang dipilih.")


def TEMUAN(R):
    H = R["hasil"]
    rasio = [H[ds]["D-1"]["0"]["palsu_per_jam"] / H[ds]["D-1"]["30"]["palsu_per_jam"] for ds in AF_DS + NEG_DS]
    se_all = [H[ds][s]["30"]["se_episode"] for ds in AF_DS for s in SISTEM]
    return ("<b>Temuan.</b> (1) Hampir semua alarm palsu D-1 adalah run AF pendek: aturan 30 s menurunkan alarm palsunya "
            f"{fnum(min(rasio), 0)}–{fnum(max(rasio), 0)}× di ketujuh data. "
            f"(2) Saat latihan fisik v3.1 tetap memberi {fnum(H['Latihan fisik']['v3.1']['30']['palsu_per_jam'], 1)} alarm palsu per jam dengan "
            f"aturan 30 s, D-1 {fnum(H['Latihan fisik']['D-1']['30']['palsu_per_jam'], 2)}: alarm sinus takikardia v3.1 bukan kejadian "
            "singkat, sehingga aturan durasi tidak menolongnya. (3) Di MIT-BIH, D-1 + 30 s lebih sensitif "
            f"({fnum(H['MIT-BIH']['D-1']['30']['se_episode_30'])} vs {fnum(H['MIT-BIH']['v3.1']['30']['se_episode_30'])}) dan alarm palsunya "
            f"{fnum(H['MIT-BIH']['D-1']['30']['palsu_per_jam'], 2)} vs {fnum(H['MIT-BIH']['v3.1']['30']['palsu_per_jam'], 2)} per jam. "
            f"(4) Se episode semua episode (termasuk &lt; 30 s) rendah untuk semua sistem ({fnum(min(se_all), 2)}–{fnum(max(se_all), 2)}), sesuai definisi aturan; episode "
            "pendek memang tidak dialarmkan.")


def KURVA_TEKS(R):
    K = R["kurva"]
    return ("Aturan yang lebih panjang terus menurunkan alarm palsu (NSRDB D-1: "
            f"{fnum(K['D-1']['NSRDB']['30'][0], 2)} pada 30 s, {fnum(K['D-1']['NSRDB']['60'][0], 3)} pada 60 s), tetapi Se episode ≥ 30 s turun "
            f"tajam (LTAFDB {fnum(K['D-1']['LTAFDB']['30'][1])} → {fnum(K['D-1']['LTAFDB']['60'][1])}). Sebagian penurunan itu karena "
            "definisi: aturan 60 s tidak bisa mengalarmkan episode 30–60 s. Eksplorasi setelah hasil (v16.6, deskriptif) menunjukkan run "
            "juga sering terputus oleh satu window yang terlewat; v16.6 menguji toleransi celah dengan rancangan baru. Sumbu kiri "
            "logaritmik; nilai 0 digambar di 0,001.")


def SUBJEK_TEKS(R):
    ps = R["nsrdb_per_subjek_S1"]
    n_atas = sum(v > 1 for v in ps.values())
    return (f"Pada S1, {n_atas} dari {len(ps)} subjek NSRDB di atas 1 alarm palsu per jam; tertinggi {max(ps, key=ps.get)} "
            f"({fnum(max(ps.values()), 2)}). Alarm palsu tersebar di banyak orang, bukan satu subjek bermasalah.")


def METODE(R):
    d = R["data"]
    return [
        "<b>Aturan (dikunci, dari definisi klinis).</b> Keputusan AF per window diurutkan per rekaman. Run = window AF berturut-turut dengan "
        "jarak awal ≤ langkah + 0,5 s (window yang tidak lolos cek kualitas memutus run). Alarm untuk run dengan rentang ≥ 30 s (langkah 5 s "
        "→ ≥ 5 window; latihan fisik langkah 10 s → ≥ 3 window). Durasi 60/120/360 s dan tanpa aturan hanya deskriptif.",
        "<b>Sistem.</b> S1 = D-1 (ECGFounder zero-shot, d1 &gt; −1) + aturan (primer); S2 = AF hanya bila D-1 dan paket v3.1 sama-sama AF; "
        "R = paket v3.1 (AFDB: OOF MicroTransformer + bias cross-fit, 5 ulangan).",
        "<b>Data.</b> " + "; ".join(f"{ds} {thou(v['window'])} window ({fnum(v['jam'], 0)} jam, {thou(v['unit'])} unit)" for ds, v in d.items()) +
        ". EDB (European ST-T, 90 rekaman × 2 jam) belum pernah dipakai; AF hanya di satu rekaman (e0409).",
        "<b>Metrik.</b> Episode gaya EC57 dengan interval sebenarnya: Se episode untuk episode referensi ≥ 30 s (primer) dan semua; "
        "+P episode; Se/+P durasi; alarm palsu per jam = episode deteksi tanpa tumpang tindih referensi / jam pengamatan. H1: bootstrap "
        "subjek berpasangan 2.000×, seed 2026.",
        "<b>Kendali mutu.</b> Alarm per window NSRDB tanpa aturan sama dengan v16.4 (assert). Laporan ini menghitung ulang alarm palsu per "
        "jam S1 dan R di NSRDB dan EDB dari cache per rekaman (assert).",
    ]


def KESIMPULAN(R):
    T = R["target"]
    return [
        "<b>Kesimpulan.</b> Aturan episode klinis ≥ 30 s adalah langkah paling berpengaruh sejauh ini untuk alarm palsu: S1 (D-1 + 30 s) "
        f"turun ke {fnum(T['D-1']['palsu']['NSRDB'], 2)} per jam pada orang sehat dan {fnum(T['D-1']['palsu']['Latihan fisik'], 2)} saat "
        "olahraga, dengan Se episode AF ≥ 30 s ≥ 0,91 di keempat data Holter. Target produk yang dikunci belum terpenuhi (NSRDB sedikit di "
        f"atas 1,0). v3.1 + 30 s tetap memberi {fnum(R['hasil']['Latihan fisik']['v3.1']['30']['palsu_per_jam'], 1)} alarm palsu per jam saat olahraga.",
        "<b>Langkah berikutnya.</b> v16.6: pilih durasi minimum dan toleransi celah dari grid kecil di data ini (kriteria dikunci), lalu uji "
        "sekali di data yang belum pernah dipakai (SHDB-AF, afpdb, incartdb). Ambang d1 tetap −1.",
        "<b>Batasan.</b> NSRDB hanya 18 subjek; EDB hanya punya satu episode AF. Label episode referensi = run window berlabel AF (bukan detik "
        "persis). Aturan dihitung pada window 10 s dengan langkah 5 s; perangkat dengan langkah lain perlu dihitung ulang. Kurva durasi "
        "dilihat setelah hasil dan tidak dipakai untuk memilih.",
        "<b>Reproduksi.</b> Dari root repo: python tools/unduh_physionet.py edb 1.0.0 --ke edb --pola '*.hea' '*.dat' '*.atr' RECORDS; "
        "lalu python tools/run_notebook.py v16.5_episode_30s/episode.ipynb (butuh cache v13, v16.1–v16.4, v3.3). Lalu dari folder "
        "v16.5_episode_30s: python laporan/build_report.py --extract, lalu python laporan/build_report.py.",
    ]


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        build()
