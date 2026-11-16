"""Laporan PDF eksperimen InceptionTime v11: satu PDF per eksperimen (definisi eksperimen di ../eksperimen.py).

Pola sama dengan ../../tcn/laporan/build_report.py, dengan pembanding model = kandidat v11 (BiGRU dan TCN `wp100`, OOF di
cache/v11/{bigru,tcn}/wp100/). Halaman diagnosa memuat bias AFL per fold, AUROC AFL, dan margin skor AFL, karena pertanyaan
InceptionTime sama dengan TCN: apakah AFL 2:1 (08378) bertahan setelah bias kelas.

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract [e1 e2]: baca ringkasan.json tiap setelan (= sel evaluasi jalankan.ipynb) dan file perbandingan
     (`python eksperimen.py evaluasi eN`) dari cache/v11/inceptiontime, cocokkan dengan OOF di cache (assert), lalu simpan
     angka + grafik ke laporan/_build/.
  2. tanpa --extract [e1 e2]: rakit laporan/laporan_inceptiontime_eN_*.pdf dengan ReportLab.
  (--grafik [eN]: gambar ulang grafik dari _build/eN.json tanpa menghitung ulang angka.)
Tanpa nama eksperimen = semua eksperimen yang file perbandingannya sudah ada.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # v11_model_tim/inceptiontime/laporan
MODEL = HERE.parent
REPO = MODEL.parents[1]
TMP = HERE / "_build"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402  (Arial → Liberation Sans → DejaVu)

# layout + palet data tervalidasi (validator dataviz: semua PASS; hijau < 3:1 → selalu didampingi tabel)
TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
BIRU, ORANYE, HIJAU = "#2a78d6", "#eb6834", "#1baf7a"
PASIEN_AFL = ["04043", "04908", "04936", "06426", "07910", "08215", "08378"]
NAMA_LAIN = {"bigru": "BiGRU wp100", "tcn": "TCN wp100"}      # kandidat v11 (urutan = eksperimen.LAIN)

# Per laporan: nama file, setelan yang kurvanya digambar di uji takikardia (≤ 3), baris kunci perbandingan versi, dan
# tempat training (tidak tercatat di cache: ringkasan.json mencatat lingkungan evaluasi, bukan training).
LAPORAN = {
    "e1": dict(pdf="laporan_inceptiontime_e1_bawaan.pdf", kurva=["bawaan"], kunci=["bawaan"],
               tempat={"bawaan": "PC, RTX 5060"}, jalur={"bawaan": "jalankan.ipynb"}),
    "e2": dict(pdf="laporan_inceptiontime_e2_wp100.pdf", kurva=["bawaan", "wp100"], kunci=["bawaan", "wp100"],
               tempat={"bawaan": "PC, RTX 5060", "wp100": "PC, RTX 5060"},
               jalur={"bawaan": "jalankan.ipynb", "wp100": "eksperimen.py"}),
}


# ═════════════════════════════════════════════════════════════════════
# Tahap 1: angka + grafik
# ═════════════════════════════════════════════════════════════════════
def extract(e):
    import numpy as np
    from sklearn.metrics import f1_score

    sys.path[:0] = [str(MODEL.parent), str(MODEL)]
    from eksperimen import EKSPERIMEN, LAIN, _margin
    from bersama import data, evaluasi as E, runner

    assert list(LAIN) == list(NAMA_LAIN), LAIN
    eks, cfg = EKSPERIMEN[e], LAPORAN[e]
    akar = data.lokasi_hasil() / "inceptiontime"
    B = json.loads((akar / eks["banding"]).read_text(encoding="utf-8"))
    ref = eks["referensi"]
    semua = list(dict.fromkeys([ref, *eks["setelan"]]))
    assert semua == B["setelan"], (semua, B["setelan"])
    teks_setelan = {}
    for x in EKSPERIMEN.values():
        teks_setelan.update(x["setelan"])

    d = data.muat()
    R = {}
    for s in semua:
        rk = json.loads((akar / s / "ringkasan.json").read_text(encoding="utf-8"))
        # ringkasan.json (= sel evaluasi notebook) harus cocok dengan OOF yang ada di cache sekarang
        h = runner.muat_hasil(akar / s, d)
        assert h.ulangan == rk["ulangan"] == list(range(data.N_ULANGAN)), s
        pb, bias = E.keputusan(h.oof, d.folds, d.y)
        for nama, pred in (("bias", pb), ("argmax", h.oof.argmax(-1))):
            mf = float(np.mean([f1_score(d.y, p, average="macro", labels=[0, 1, 2]) for p in pred]))
            assert abs(mf - rk["metrik"][nama]["Macro-F1"][0]) < 1e-9, (s, nama, mf, rk["metrik"][nama]["Macro-F1"][0])
        dg = B["diagnosa"][s]
        assert np.allclose(bias[:, :, 2], dg["bias_afl"]), s                  # diagnosa dari OOF yang sama
        mg = _margin(d, h.oof)
        assert all(abs(mg[k] - dg["margin"][k]) < 1e-9 for k in mg), s
        pita = {k.split(" | ")[1]: [v["n"], 1 - v["model"][0]] for k, v in rk["pita_detak"].items() if k.startswith("Normal")}
        bafl = np.ravel(dg["bias_afl"])
        R[s] = {"bias": rk["metrik"]["bias"], "argmax": rk["metrik"]["argmax"], "v31": rk["banding_v31"],
                "v1": rk["banding_v1"], "afl": rk["afl_per_pasien"]["bias"], "afl_argmax": rk["afl_per_pasien"]["argmax"],
                "palsu": rk["afl_palsu"]["bias"], "palsu_argmax": rk["afl_palsu"]["argmax"], "biasstat": rk["bias"],
                "afl_di_batas": int(np.isin(bafl, (-7.0, 3.0)).sum()), "n_bias_afl": int(bafl.size),
                "ambang": rk["takikardia"]["model"]["ambang"], "proporsi": rk["takikardia"]["model"]["proporsi"],
                "pita": pita, "waktu": rk["waktu_per_fold"], "param": rk["info"]["parameter"],
                "n_latih": rk["info"]["n_latih"], "setelan_efektif": rk["setelan"], "setelan": teks_setelan[s],
                "tempat": cfg["tempat"][s], "jalur": cfg["jalur"][s], "lingkungan": rk["lingkungan"],
                "auroc": float(np.mean(dg["auroc_afl"])), "bias_afl": dg["bias_afl"], "vs_lain": dg["banding_lain"],
                "margin": dg["margin"]}
    rr = json.loads((akar / ref / "ringkasan.json").read_text(encoding="utf-8"))
    for s in semua:                                   # pembanding v3.1/v1 sama di semua ringkasan
        rk = json.loads((akar / s / "ringkasan.json").read_text(encoding="utf-8"))
        assert rk["metrik"]["v31"] == rr["metrik"]["v31"] and rk["metrik"]["v1"] == rr["metrik"]["v1"], s
    # selisih berpasangan konsisten dengan ringkasan; tanpa-04936 memuat setelan yang sama
    for s in semua:
        if s != ref:
            b = B["banding"][f"{s}_vs_{ref}"]["banding"]["Macro-F1"]
            assert abs(b[0] - (R[s]["bias"]["Macro-F1"][0] - R[ref]["bias"]["Macro-F1"][0])) < 1e-9, s
        assert s in B["tanpa_04936"]["rows"], s
    G = B["pembanding"]
    lain = {}
    for m, sl in LAIN.items():                        # metrik pembanding di file perbandingan = ringkasan.json model itu
        f = data.lokasi_hasil() / m / sl
        rg = json.loads((f / "ringkasan.json").read_text(encoding="utf-8"))
        assert np.allclose(rg["metrik"]["bias"]["Macro-F1"], G[m]["metrik"]["Macro-F1"]), m
        hg = runner.muat_hasil(f, d)
        _, bg = E.keputusan(hg.oof, d.folds, d.y)
        lain[m] = {**G[m], "nama": NAMA_LAIN[m], "auroc": float(np.mean(G[m]["auroc_afl"])), "bias_afl": bg[:, :, 2].tolist(),
                   "ambang": rg["takikardia"]["model"]["ambang"], "tanpa_kunci": f"{m}_{sl}",
                   "pita": {k.split(" | ")[1]: [v["n"], 1 - v["model"][0]] for k, v in rg["pita_detak"].items()
                            if k.startswith("Normal")}}
    p_v31, _ = E.keputusan(d.oof_v31, d.folds, d.y)
    p_v1 = d.oof_v1.argmax(-1)
    pita31 = {k.split(" | ")[1]: [v["n"], 1 - v["v31"][0]] for k, v in rr["pita_detak"].items() if k.startswith("Normal")}
    afl_win = {p: int(((d.records == p) & (d.y == 2)).sum()) for p in PASIEN_AFL}
    payload = {
        "e": e, "judul": eks["judul"], "tanggal": eks["tanggal"], "referensi": ref, "baru": list(eks["setelan"]),
        "semua": semua, "file_banding": eks["banding"], **cfg, "R": R, "lain": lain,
        "v31": {"metrik": rr["metrik"]["v31"], "afl": rr["afl_per_pasien"]["v31"], "palsu": rr["afl_palsu"]["v31"],
                "pita": pita31, "auroc": float(np.mean(G["v31"]["auroc_afl"])), "margin": G["v31"]["margin"]},
        "v1": {"metrik": rr["metrik"]["v1"], "afl": rr["afl_per_pasien"]["v1"], "palsu": rr["afl_palsu"]["v1"]},
        "v31_vs_v1": E.banding(d, p_v31, p_v1),
        "banding": B["banding"], "tanpa": B["tanpa_04936"], "bersarang": B.get("bersarang", {}),
        "kontrol_v7": rr["takikardia"]["kontrol_v7"], "paket_v31": rr["takikardia"]["paket_v31"],
        "bpm": rr["takikardia"]["model"]["bpm"], "n_windows": int(len(d.y)), "class_counts": np.bincount(d.y).tolist(),
        "n_pasien": int(len(d.patients)), "afl_window": afl_win,
    }
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / f"{e}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    grafik(e)


def grafik(e):
    """Grafik dari _build/<e>.json saja (tanpa menghitung ulang angka), supaya tampilan bisa diperbaiki cepat."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter

    P = json.loads((TMP / f"{e}.json").read_text(encoding="utf-8"))
    R, ref, semua, afl_win, L = P["R"], P["referensi"], P["semua"], P["afl_window"], P["lain"]
    for style in ("regular", "bold"):
        font_manager.fontManager.addfont(font_path(style))
    family = font_manager.FontProperties(fname=font_path("regular")).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 9.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#C4CED8", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": GRAY, "ytick.color": INK,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
    pct = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")

    def save(fig, name):
        fig.savefig(TMP / f"{e}_{name}.png", dpi=220, bbox_inches="tight", pad_inches=.08)
        plt.close(fig)

    def style(ax, axis="x"):
        ax.grid(axis=axis, color="#E6EBF0", lw=.8)
        ax.set_axisbelow(True)
        if axis == "x":
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.xaxis.set_major_formatter(comma)

    nama = lambda s: f"InceptionTime {s}"  # noqa: E731

    # 1. forest: Macro-F1 & AFL F1 (+ bias) dengan CI; pembanding di atas
    rows = [("v3.1 (produksi)", P["v31"]["metrik"], HIJAU, "D"), ("v1 (argmax)", P["v1"]["metrik"], GRAY, "D")]
    rows += [(L[m]["nama"], L[m]["metrik"], "#8A97A6", "P") for m in L]
    rows += [(nama(s), R[s]["bias"], ORANYE if s == ref else BIRU, "s" if s == ref else "o") for s in semua]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, .36 * len(rows) + 1.0), sharey=True)
    for ax, met in zip(axes, ("Macro-F1", "AFL F1")):
        for i, (_, m, col, mk) in enumerate(rows):
            pt, lo, hi = m[met]
            ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round", alpha=.85)
            ax.plot(pt, i, mk, color=col, ms=7, mec="white", mew=1.2)
        ax.axvline(R[ref]["bias"][met][0], color="#8A97A6", lw=1, ls="--")
        ax.axhline(1.5 + len(L), color="#E0E6EC", lw=.8)
        ax.set_title(met + " (+ bias, CI 95%)", loc="left", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        style(ax)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "forest")

    # 2. selisih berpasangan vs referensi (hanya bila ada setelan lain)
    lain = [s for s in semua if s != ref]
    if lain:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, .36 * len(lain) + 1.25), sharey=True)
        for ax, met in zip(axes, ("Macro-F1", "AFL R")):
            ax.axvline(0, color=INK, lw=1)
            for i, s in enumerate(lain):
                dl, lo, hi, p = P["banding"][f"{s}_vs_{ref}"]["banding"][met]
                col, mk, isi = (HIJAU, "^", True) if p >= .95 else (ORANYE, "v", True) if p <= .05 else (BIRU, "o", False)
                ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
                ax.plot(dl, i, mk, ms=8, color=col, mfc=col if isi else "white", mec=col, mew=1.6)
            ax.set_title(f"Δ {met} terhadap {ref}", loc="left", fontsize=10, fontweight="bold")
            style(ax)
        axes[0].set_yticks(range(len(lain)), lain)
        axes[0].invert_yaxis()
        fig.text(.01, -.06, "▲ hijau = lebih baik (P ≥ 0,95)   ▼ oranye = lebih buruk (P ≤ 0,05)   ○ biru = tidak terbedakan. "
                 "Garis = CI 95%.", fontsize=8.6, color=GRAY)
        fig.tight_layout()
        save(fig, "paired")

    # 3. heatmap recall AFL per pasien (+ bias); satu setelan: tambah kolom argmax untuk menunjukkan efek bias
    kol = [("v3.1", P["v31"]["afl"])] + [(L[m]["nama"].replace(" ", "\n"), L[m]["afl_per_pasien"]) for m in L]
    if len(semua) == 1:
        kol += [(f"{semua[0]}\nargmax", R[semua[0]]["afl_argmax"]), (f"{semua[0]}\n+ bias", R[semua[0]]["afl"])]
    else:
        kol += [(f"{s}\nargmax", R[s]["afl_argmax"]) for s in semua if s != ref]
        kol += [(f"{s}\n+ bias", R[s]["afl"]) for s in semua]
    mat = np.array([[k[1][p] for k in kol] for p in PASIEN_AFL])
    cmap = LinearSegmentedColormap.from_list("biru", ["#EEF4FC", "#86b6ef", BIRU, "#104281"])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(kol)), [k[0] for k in kol], fontsize=8.6)
    ax.axvline(len(L) + .5, color="white", lw=4)
    ax.set_yticks(range(len(PASIEN_AFL)), [f"{p}  ({afl_win[p]} window)" for p in PASIEN_AFL])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=8.4,
                    color="white" if mat[i, j] > .55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    cb.set_label("Recall AFL (rata-rata 5 ulangan)", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(comma)
    fig.tight_layout()
    save(fig, "afl_heatmap")

    # 4. diagnosa: bias AFL tiap fold (25 titik per sistem), garis = median; batas grid −7 dan +3
    baris = [(L[m]["nama"], L[m]["bias_afl"], "#8A97A6", "P") for m in L]
    baris += [(nama(s), R[s]["bias_afl"], ORANYE if s == ref else BIRU, "s" if s == ref else "o") for s in semua]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, .38 * len(baris) + 1.0))
    for i, (_, b, col, mk) in enumerate(baris):
        v = np.ravel(b)
        ax.scatter(v, i + rng.uniform(-.17, .17, len(v)), s=16, marker=mk, color=col, alpha=.75, edgecolors="white",
                   linewidths=.4, zorder=3)
        ax.plot([np.median(v)] * 2, [i - .3, i + .3], color=INK, lw=1.6, zorder=4)
    for x in (-7, 3):
        ax.axvline(x, color="#C4CED8", lw=1, ls=":")
    ax.axhline(len(L) - .5, color="#E0E6EC", lw=.8)
    ax.set_xlim(-7.4, 3.4)
    ax.set_yticks(range(len(baris)), [b[0] for b in baris])
    ax.invert_yaxis()
    ax.set_xlabel("Bias AFL terpilih per fold (5 ulangan × 5 fold); garis gelap = median, garis titik-titik = batas grid")
    style(ax)
    fig.tight_layout()
    save(fig, "bias_afl")

    # 5. tanpa 04936: semua pasien vs tanpa 04936
    T = P["tanpa"]["rows"]
    baris = [("v3.1", P["v31"]["metrik"]["Macro-F1"][0], T["v31"]["Macro-F1"][0])]
    baris += [(L[m]["nama"], L[m]["metrik"]["Macro-F1"][0], T[L[m]["tanpa_kunci"]]["Macro-F1"][0]) for m in L]
    baris += [(nama(s), R[s]["bias"]["Macro-F1"][0], T[s]["Macro-F1"][0]) for s in semua]
    tinggi = .3 * len(baris) + .9
    fig, ax = plt.subplots(figsize=(7.2, tinggi))
    for i, (_, a, b) in enumerate(baris):
        ax.plot([a, b], [i, i], color="#D5DCE3", lw=3, zorder=1)
        ax.plot(a, i, "o", color=BIRU, ms=7, mec="white", mew=1.2, zorder=3, label="semua 23 pasien" if i == 0 else None)
        ax.plot(b, i, "s", color=ORANYE, ms=7, mec="white", mew=1.2, zorder=3, label="tanpa 04936" if i == 0 else None)
    ax.set_yticks(range(len(baris)), [b[0] for b in baris])
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 (+ bias; tanpa 04936: bias dihitung ulang di 22 pasien)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.62 / (tinggi - .8)), ncol=2, frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    save(fig, "tanpa")

    # 6. uji takikardia: % ditebak Normal per bpm, 4 jenis variasi (small multiples)
    bpm = P["bpm"]
    jenis = list(R[ref]["proporsi"])
    gaya = list(zip(P["kurva"], (ORANYE, BIRU, HIJAU), ("s", "o", "^")))
    fig, axes = plt.subplots(1, len(jenis), figsize=(7.2, 2.5), sharey=True)
    for ax, j in zip(axes, jenis):
        kv = [p[0] for p in P["kontrol_v7"]["proporsi"][j]]
        ax.plot(bpm, kv, color="#8A97A6", lw=1.3, ls="--", label="kontrol v7 (MicroTransformer)")
        for s, col, mk in gaya:
            ax.plot(bpm, [p[0] for p in R[s]["proporsi"][j]], color=col, lw=1.8, marker=mk, ms=3.6, markevery=2,
                    label=nama(s))
        ax.axhline(.5, color="#C4CED8", lw=.8)
        ax.axvline(P["paket_v31"], color=HIJAU, lw=1, ls=":")
        ax.set_title(j, fontsize=9, loc="left")
        ax.set_xticks([60, 100, 140, 170])
        ax.yaxis.set_major_formatter(pct)
        style(ax, "y")
    axes[0].set_ylabel("ditebak Normal")
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.3, -.2), ncol=len(gaya) + 1, frameon=False, fontsize=8.6)
    fig.text(.5, -.02, "Ritme teratur sintetis (bpm). Garis hijau titik-titik = ambang paket v3.1 (115 bpm).", ha="center",
             fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "takikardia")
    print(f"OK {TMP / (e + '.json')} + {6 if lain else 5} grafik")


# ═════════════════════════════════════════════════════════════════════
# Tahap 2: PDF (ReportLab)
# ═════════════════════════════════════════════════════════════════════
def fnum(v, nd=3):
    return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")


def sg(v, nd=3):
    return ("+" if v >= 0 else "") + fnum(v, nd) if round(v, nd) != 0 else fnum(abs(v), nd)


def pct(v, nd=1):
    return f"{v * 100:.{nd}f}%".replace(".", ",")


def thou(v):
    return f"{int(v):,}".replace(",", ".")


def vonis(p):
    return "lebih baik" if p >= .95 else "lebih buruk" if p <= .05 else "tidak terbedakan"


class A:
    """Akses angka satu laporan (semua angka di teks lewat sini, tidak diketik manual)."""

    def __init__(self, d):
        self.d, self.R, self.ref, self.L = d, d["R"], d["referensi"], d["lain"]

    def m(self, s, met="Macro-F1", i=0):
        return self.R[s]["bias"][met][i]

    def f(self, s, met="Macro-F1"):
        return fnum(self.m(s, met))

    def b(self, s, met="Macro-F1"):                       # vs referensi: [Δ, lo, hi, P]
        return self.d["banding"][f"{s}_vs_{self.ref}"]["banding"][met]

    def bd(self, s, met="Macro-F1"):
        x = self.b(s, met)
        return f"{sg(x[0])} (P = {fnum(x[3], 2)})"

    def v31(self, s, met="Macro-F1"):
        return self.R[s]["v31"][met]

    def g(self, s, m="bigru", met="Macro-F1"):            # vs kandidat v11 m (+ bias): [Δ, lo, hi, P]
        return self.R[s]["vs_lain"][m][met]

    def gd(self, s, m="bigru", met="Macro-F1", nd=3):
        x = self.g(s, m, met)
        return f"{sg(x[0])} (P = {fnum(x[3], nd)})"

    def gp(self, s, m="bigru", met="Macro-F1", nd=3):     # sama dengan gd, untuk ditulis di dalam kurung
        x = self.g(s, m, met)
        return f"{sg(x[0])}, P = {fnum(x[3], nd)}"

    def bp(self, s, met="Macro-F1"):                      # sama dengan bd, untuk ditulis di dalam kurung
        x = self.b(s, met)
        return f"{sg(x[0])}, P = {fnum(x[3], 2)}"

    def afl(self, s, p, jenis="afl"):
        return fnum(self.R[s][jenis][p], 2)

    def tanpa(self, s, met="Macro-F1"):
        return self.d["tanpa"]["rows"][s][met][0]

    def tanpa_l(self, m):
        return self.tanpa(self.L[m]["tanpa_kunci"])

    def tp(self, s):                                      # v3.1 − s tanpa 04936: [Δ, lo, hi, P]
        return self.d["tanpa"]["pairs_v31_minus"][s]

    def amb(self, s):
        return " / ".join(str(v) for v in self.R[s]["ambang"].values())

    def pita(self, s, k="120–∞ bpm"):
        return self.R[s]["pita"][k][1]

    def bs(self, s):
        return self.d["bersarang"][s]

    def mg(self, s, k="08378"):
        return fnum(self.R[s]["margin"][k], 2)

    def mgl(self, m, k="08378"):
        return fnum(self.L[m]["margin"][k], 2)

    def rentang_bias(self, s):                             # bias AFL 25 fold: (min, maks, median)
        return self._rentang(self.R[s]["bias_afl"])

    def rentang_bias_l(self, m):
        return self._rentang(self.L[m]["bias_afl"])

    @staticmethod
    def _rentang(b):
        import numpy as np
        v = np.ravel(b)
        return float(v.min()), float(v.max()), float(np.median(v))


# ── teks per eksperimen (angka lewat A, narasi ditulis setelah melihat hasil) ───────────────────────
def teks_e1(a):
    d, L = a.d, a.L
    s = "bawaan"
    lo, hi, med = a.rentang_bias(s)
    pa, pb = a.R[s]["palsu_argmax"]["total"], a.R[s]["palsu"]["total"]
    p31 = d["v31"]["palsu"]["total"]
    return dict(
        pertanyaan="Apakah InceptionTime (modul Inception, satu jaringan) dengan setelan bawaan setara model produksi dan "
                   "kandidat v11 (BiGRU dan TCN dengan bobot kelas penuh)?",
        cards=[(a.f(s), "Macro-F1 InceptionTime bawaan*"), (sg(a.v31(s)[0]), f"Selisih vs v3.1 (P = {fnum(a.v31(s)[3], 3)})"),
               (a.afl(s, "08378"), f"Recall AFL 08378 (tanpa bias {a.afl(s, '08378', 'afl_argmax')})"),
               (fnum(a.R[s]["auroc"]), f"AUROC AFL (v3.1 {fnum(d['v31']['auroc'])})")],
        jawaban=f"<b>Tidak.</b> InceptionTime bawaan <b>lebih buruk</b> daripada v3.1 (Macro-F1 {a.f(s)} vs "
                f"{fnum(d['v31']['metrik']['Macro-F1'][0])}, P = {fnum(a.v31(s)[3], 3)}) dan kedua kandidat v11 (BiGRU wp100 "
                f"{a.gp(s, 'bigru')}; TCN wp100 {a.gp(s, 'tcn')}), dan setara v1 (P = {fnum(a.R[s]['v1']['Macro-F1'][3], 2)}). "
                f"Kemampuan memisahkan AFL sama (AUROC {fnum(a.R[s]['auroc'])}, v3.1 {fnum(d['v31']['auroc'])}). Polanya sama "
                f"dengan TCN bawaan, tetapi lebih ringan: tanpa bias model terlalu sering menyebut AFL, dan bias cross-fit yang "
                f"menekannya ikut menurunkan AFL 2:1 pasien 08378 ({a.afl(s, '08378', 'afl_argmax')} → {a.afl(s, '08378')}).",
        temuan=[
            ("1. Lebih buruk dari v3.1 dan kandidat v11.", f"Δ Macro-F1 vs v3.1 {sg(a.v31(s)[0])} (P = {fnum(a.v31(s)[3], 3)}), "
             f"vs BiGRU wp100 {a.gd(s, 'bigru')}, vs TCN wp100 {a.gd(s, 'tcn')}."),
            ("2. Tanpa bias: terlalu banyak AFL.", f"Recall AFL {fnum(a.R[s]['argmax']['AFL R'][0])}, tetapi "
             f"{thou(round(pa))} AFL palsu per ulangan ({fnum(pa / p31, 1)}× v3.1)."),
            ("3. Keyakinan pada AFL 2:1 sedang.", f"Margin skor AFL di 08378 {a.mg(s)} (kandidat v11 {a.mgl('bigru')} dan "
             f"{a.mgl('tcn')}, v3.1 {fnum(d['v31']['margin']['08378'], 2)}). Bias AFL median {fnum(med, 2)} menurunkan AFL palsu "
             f"ke {thou(round(pb))} per ulangan, tetapi ikut menurunkan 08378 ke {a.afl(s, '08378')}."),
            ("4. Alarm detak cepat lebih rendah.", f"Ambang {a.amb(s)} bpm (v3.1 {d['paket_v31']}); alarm pada Normal ≥ 120 bpm "
             f"{pct(a.pita(s))} (v3.1 {pct(d['v31']['pita']['120–∞ bpm'][1])}). Sebagian karena bias AFL yang kuat."),
            ("5. Tanpa 04936.", f"Macro-F1 {fnum(a.tanpa(s))} (v3.1 {fnum(a.tanpa('v31'))}, BiGRU wp100 "
             f"{fnum(a.tanpa_l('bigru'))}, TCN wp100 {fnum(a.tanpa_l('tcn'))})."),
        ],
        hasil="InceptionTime: enam modul Inception (bottleneck 1×1, konvolusi kernel 3, 5, 9 dan cabang max-pool, 16 filter per "
              "cabang = 64 kanal), shortcut residual setiap 3 modul, pooling rata-rata bermask. Kernel asli (10/20/40) diperkecil "
              "karena window hanya 30 detak, dan hanya satu jaringan (aslinya ensemble 5). Input dan resep training sama dengan "
              "BiGRU dan TCN v11 bawaan (urutan IBI + ΔIBI, 30 epoch, bobot kelas lunak). Dijalankan di PC (GPU RTX 5060) "
              "lewat jalankan.ipynb.",
        paired=f"Terhadap BiGRU wp100, InceptionTime bawaan lebih buruk di AFL F1 ({a.gp(s, 'bigru', 'AFL F1')}) dan recall AFL "
               f"({a.gp(s, 'bigru', 'AFL R')}), sedangkan recall Normal tidak berbeda ({a.gp(s, 'bigru', 'Normal R', 2)}). "
               f"Terhadap TCN wp100 polanya sama: AFL F1 {a.gd(s, 'tcn', 'AFL F1')}, recall AFL {a.gd(s, 'tcn', 'AFL R')}.",
        pasien=f"Tanpa bias, InceptionTime menangkap 08378 sepenuhnya ({a.afl(s, '08378', 'afl_argmax')}), tetapi juga menebak AFL "
               f"di {thou(round(a.R[s]['palsu_argmax']['dari_normal']))} window Normal dan "
               f"{thou(round(a.R[s]['palsu_argmax']['dari_afib']))} window AFIB per ulangan. Bias cross-fit menekan AFL palsu ke "
               f"{thou(round(pb))}, dengan harga 08378 ({a.afl(s, '08378', 'afl_argmax')} → {a.afl(s, '08378')}), 04908 "
               f"({a.afl(s, '04908', 'afl_argmax')} → {a.afl(s, '04908')}), dan 04936 ({a.afl(s, '04936', 'afl_argmax')} → "
               f"{a.afl(s, '04936')}). Tidak ada bias di batas grid.",
        diagnosa=f"AUROC AFL {fnum(a.R[s]['auroc'])} hampir sama dengan v3.1 ({fnum(d['v31']['auroc'])}) dan kandidat v11 "
                 f"({fnum(L['bigru']['auroc'])}, {fnum(L['tcn']['auroc'])}): urutan skor AFL sama baiknya. Bedanya ada di margin "
                 f"skor AFL. Di 08378 (AFL 2:1) median margin InceptionTime {a.mg(s)}, jauh di bawah kandidat v11 "
                 f"({a.mgl('bigru')} dan {a.mgl('tcn')}) dan v3.1 ({fnum(d['v31']['margin']['08378'], 2)}). Bias AFL-nya median "
                 f"{fnum(med, 2)} (rentang {fnum(lo, 2)} … {fnum(hi, 2)}). AFL hanya ditebak bila margin kira-kira melebihi besarnya "
                 "bias, jadi sebagian window 08378 hilang. Kerusakannya lebih kecil daripada TCN bawaan, yang margin 08378-nya "
                 "lebih tipis dan kehilangan hampir seluruh 08378 (laporan TCN eksperimen 1).",
        tanpa=f"Tanpa 04936, Macro-F1 InceptionTime {fnum(a.tanpa(s))}, jauh di bawah v3.1 ({fnum(a.tanpa('v31'))}) dan kandidat "
              f"v11 ({fnum(a.tanpa_l('bigru'))} dan {fnum(a.tanpa_l('tcn'))}). Tanpa pasien itu, 08378 menjadi bagian terbesar AFL "
              f"yang tersisa, dan InceptionTime kehilangan sebagian pasien ini setelah bias. v3.1 − InceptionTime "
              f"{sg(a.tp(s)[0])} [{sg(a.tp(s)[1], 2)}…{sg(a.tp(s)[2], 2)}], P = {fnum(a.tp(s)[3], 2)}.",
        taki=f"InceptionTime berhenti menebak Normal pada {a.amb(s)} bpm (paket v3.1: {d['paket_v31']}). Alarm pada Normal ≥ 120 bpm "
             f"{pct(a.pita(s))} (v3.1 {pct(d['v31']['pita']['120–∞ bpm'][1])}, BiGRU wp100 {pct(L['bigru']['pita']['120–∞ bpm'][1])}, "
             f"TCN wp100 {pct(L['tcn']['pita']['120–∞ bpm'][1])}). Ini sebagian karena bias AFL yang kuat, bukan karena "
             "InceptionTime lebih memahami takikardia.",
        keputusan=[["InceptionTime bawaan sebagai kandidat produksi", "<b>Tidak</b>",
                    f"Lebih buruk dari v3.1 (P = {fnum(a.v31(s)[3], 3)}) dan BiGRU wp100 (P = {fnum(a.g(s, 'bigru')[3], 3)})"],
                   ["Menaikkan keyakinan pada AFL 2:1 (08378 turun setelah bias)", "Diteruskan",
                    "Eksperimen 2: bobot kelas penuh (wp100)"]],
        langkah=[("Bobot kelas penuh.", "Resep yang memulihkan 08378 di BiGRU dan TCN (margin 08378 naik ke ±5). Satu-satunya "
                  "hipotesis eksperimen 2, ditetapkan sebelum training."),
                 ("Ukuran berhasil.", "08378 tetap terdeteksi setelah bias tanpa menambah AFL palsu, bukan Macro-F1 saja.")],
        batasan=["Hanya setelan bawaan, satu kali training per (ulangan, fold), satu jaringan (bukan ensemble 5).",
                 "Bootstrap bersarang tidak dijalankan karena InceptionTime bawaan bukan kandidat."],
    )


def teks_e2(a):
    d, L = a.d, a.L
    wp, bw = "wp100", "bawaan"
    lo, hi, med = a.rentang_bias(wp)
    lo0, hi0, med0 = a.rentang_bias(bw)
    nb, nn = a.R[wp]["afl_di_batas"], a.R[wp]["n_bias_afl"]
    pw, pb = a.R[wp]["palsu"], a.R[bw]["palsu"]
    return dict(
        pertanyaan="Apakah bobot kelas penuh (weight_power 1,0), resep yang memulihkan AFL 2:1 pada BiGRU dan TCN, juga "
                   "memperbaiki InceptionTime?",
        cards=[(a.f(wp), f"Macro-F1 wp100* (bawaan {a.f(bw)})"),
               (a.afl(wp, "08378"), f"Recall AFL 08378 (bawaan {a.afl(bw, '08378')})"),
               (sg(a.b(wp)[0]), f"Selisih vs bawaan (P = {fnum(a.b(wp)[3], 2)})"),
               (f"{nb} / {nn}", "Fold dengan bias AFL di batas grid −7")],
        jawaban=f"<b>Tidak.</b> Bobot penuh memulihkan AFL 2:1 pasien 08378 ({a.afl(bw, '08378')} → {a.afl(wp, '08378')}), "
                f"tetapi Macro-F1 tidak berubah ({a.f(bw)} → {a.f(wp)}, P = {fnum(a.b(wp)[3], 2)}). Berbeda dengan BiGRU dan "
                f"TCN, InceptionTime juga makin yakin AFL di window yang bukan AFL. Bias AFL harus lebih kuat sampai menempel di "
                f"batas grid −7 pada {nb} dari {nn} fold, dan AFL palsu naik. InceptionTime wp100 lebih buruk daripada BiGRU wp100 "
                f"({a.gp(wp, 'bigru')}) dan tidak terbedakan dari v3.1 ({sg(a.v31(wp)[0])}, P = {fnum(a.v31(wp)[3], 3)}).",
        temuan=[
            ("1. Bobot penuh memulihkan AFL 2:1.", f"Margin skor AFL di 08378 naik {a.mg(bw)} → {a.mg(wp)}, setara kandidat v11 "
             f"({a.mgl('bigru')}, {a.mgl('tcn')}). Recall 08378 {a.afl(bw, '08378')} → {a.afl(wp, '08378')}; recall AFL "
             f"{a.bd(wp, 'AFL R')}."),
            ("2. Keyakinan AFL yang salah ikut naik.", f"Margin di window non-AFL (persentil 99,5) {a.mg(bw, 'non_afl_p995')} → "
             f"{a.mg(wp, 'non_afl_p995')}, dua kali kandidat v11 ({a.mgl('bigru', 'non_afl_p995')}, {a.mgl('tcn', 'non_afl_p995')}). "
             f"AFL palsu {thou(round(pb['total']))} → {thou(round(pw['total']))} per ulangan."),
            ("3. Bias menempel di batas grid.", f"Bias AFL median {fnum(med0, 2)} → {fnum(med, 2)}; {nb} dari {nn} fold di −7 "
             "(tanda bahaya AGENTS.md: grid mungkin memotong optimum)."),
            ("4. Hasil akhirnya sama.", f"vs bawaan {a.bd(wp)}, vs v3.1 {sg(a.v31(wp)[0])} (P = {fnum(a.v31(wp)[3], 3)}), vs BiGRU "
             f"wp100 {a.gd(wp, 'bigru')}. Bootstrap bersarang vs v1: P = {fnum(a.bs(wp)['vs_v1']['p'], 2)}."),
            ("5. Takikardia hampir sama.", f"Ambang {a.amb(wp)} bpm (bawaan {a.amb(bw)}); alarm Normal ≥ 120 bpm "
             f"{pct(a.pita(bw))} → {pct(a.pita(wp))} (v3.1 {pct(d['v31']['pita']['120–∞ bpm'][1])})."),
        ],
        hasil="Dua setelan, masing-masing 5 ulangan × 5 fold, di PC yang sama (GPU RTX 5060). Setelan bawaan dari eksperimen 1 "
              "menjadi referensi berpasangan; wp100 hanya mengubah weight_power 0,5 → 1,0. Setelan wp100 ditetapkan sebelum training sebagai "
              "satu-satunya hipotesis, karena resep ini memulihkan 08378 di BiGRU dan TCN. Seed dan fold sama, jadi selisihnya "
              "hanya berasal dari bobot kelas (dan noise training GPU).",
        paired=f"wp100 vs bawaan: Macro-F1 {a.bd(wp)}, tidak terbedakan. Recall AFL naik ({a.bp(wp, 'AFL R')}, lebih baik), "
               f"tetapi presisi AFL turun sedikit ({a.bp(wp, 'AFL P')}), dan recall Normal tidak berbeda "
               f"({a.bp(wp, 'Normal R')}). Terhadap kandidat v11, wp100 lebih buruk daripada BiGRU wp100 ({a.gp(wp, 'bigru')}) dan "
               f"tidak terbedakan dari TCN wp100 ({a.gp(wp, 'tcn')}).",
        pasien=f"wp100 memulihkan 08378 ({a.afl(bw, '08378')} → {a.afl(wp, '08378')}, {d['afl_window']['08378']} window) dan "
               f"sedikit 04908 ({a.afl(bw, '04908')} → {a.afl(wp, '04908')}), tetapi 04043 turun ({a.afl(bw, '04043')} → "
               f"{a.afl(wp, '04043')}; hanya {d['afl_window']['04043']} window), dan 04936 tetap ({a.afl(wp, '04936')}). AFL palsu "
               f"naik {thou(round(pb['total']))} → {thou(round(pw['total']))}. Kenaikannya dari ritme Normal "
               f"({thou(round(pb['dari_normal']))} → {thou(round(pw['dari_normal']))}), sedangkan di pasien tanpa AFL justru turun "
               f"({thou(round(pb['pasien_tanpa_afl']))} → {thou(round(pw['pasien_tanpa_afl']))}). Kolom terakhir dihitung dari "
               f"bias AFIB dan AFL bersama; bias AFL saja: {nb} dari {nn} fold.",
        diagnosa=f"AUROC AFL hampir tidak berubah ({fnum(a.R[bw]['auroc'])} → {fnum(a.R[wp]['auroc'])}): bobot penuh tidak "
                 f"memperbaiki urutan skor, hanya menggeser skor AFL naik. Pada BiGRU dan TCN wp100, margin 08378 mencapai ±5 sementara "
                 f"margin non-AFL tetap di bawah 3 ({a.mgl('bigru', 'non_afl_p995')}, {a.mgl('tcn', 'non_afl_p995')}). Pada "
                 f"InceptionTime keduanya naik bersama: 08378 {a.mg(bw)} → {a.mg(wp)}, non-AFL "
                 f"{a.mg(bw, 'non_afl_p995')} → {a.mg(wp, 'non_afl_p995')}. Bias cross-fit lalu harus menekan seluruh skor AFL "
                 f"(median {fnum(med, 2)}, rentang {fnum(lo, 2)} … {fnum(hi, 2)}). {nb} fold menempel di −7, jadi optimumnya "
                 "mungkin lebih rendah lagi.",
        tanpa=f"Tanpa 04936, wp100 {fnum(a.tanpa(wp))} (bawaan {fnum(a.tanpa(bw))}), masih di bawah v3.1 ({fnum(a.tanpa('v31'))}) "
              f"dan kandidat v11 ({fnum(a.tanpa_l('bigru'))}, {fnum(a.tanpa_l('tcn'))}). v3.1 − wp100 {sg(a.tp(wp)[0])} "
              f"(P = {fnum(a.tp(wp)[3], 2)}); v3.1 − bawaan {sg(a.tp(bw)[0])} (P = {fnum(a.tp(bw)[3], 2)}). Ketergantungan pada "
              "04936 berkurang, tetapi tidak hilang seperti pada BiGRU dan TCN wp100.",
        taki=f"wp100 berhenti menebak Normal pada {a.amb(wp)} bpm (bawaan {a.amb(bw)}, paket v3.1 {d['paket_v31']}). Alarm pada "
             f"Normal ≥ 120 bpm {pct(a.pita(wp))} (bawaan {pct(a.pita(bw))}, BiGRU wp100 {pct(L['bigru']['pita']['120–∞ bpm'][1])}, "
             f"TCN wp100 {pct(L['tcn']['pita']['120–∞ bpm'][1])}, v3.1 {pct(d['v31']['pita']['120–∞ bpm'][1])}). Bobot kelas "
             "tidak mengubah masalah sinus takikardia.",
        keputusan=[["InceptionTime wp100 sebagai kandidat", "<b>Tidak</b>",
                    f"Tidak lebih baik dari bawaan (P = {fnum(a.b(wp)[3], 2)}); lebih buruk dari BiGRU wp100 "
                    f"(P = {fnum(a.g(wp, 'bigru')[3], 3)}); bias di batas grid"],
                   ["Kandidat v11", "Tetap", "BiGRU wp100 dan TCN wp100 (setara v3.1)"],
                   ["Tuning InceptionTime lebih lanjut", "Tidak diprioritaskan",
                    "Masalahnya AFL palsu dari ritme Normal, bukan keyakinan pada 08378"]],
        langkah=[("Jika anggota tim meneruskan InceptionTime.", "Cari setelan yang menurunkan margin AFL di window non-AFL "
                  "(mis. dropout 0,2 atau kedalaman 3) dengan bobot penuh. Nilai dari recall 08378 setelah bias, AFL palsu, dan "
                  "jumlah bias di batas grid, bukan Macro-F1 saja."),
                 ("gabungan.ipynb.", "Masukkan InceptionTime bawaan dan wp100 bersama lima model lain.")],
        batasan=[f"Bias AFL di batas grid pada {nb} dari {nn} fold. Grid −7…+3 bagian dari protokol, jadi tidak diperlebar untuk "
                 "satu model; skor wp100 mungkin sedikit diremehkan.",
                 "Dua setelan, masing-masing dilatih sekali. Selisih kecil antar setelan bisa berasal dari noise training (di TCN, "
                 "setelan bawaan yang sama memberi Macro-F1 berbeda di dua GPU; laporan TCN eksperimen 2).",
                 "Penjelasan mekanisme (modul Inception mengenali urutan teratur-cepat tanpa membedakan AFL 2:1 dari sinus cepat) "
                 "adalah dugaan; belum diuji."],
    )


TEKS = {"e1": teks_e1, "e2": teks_e2}


def build(e):
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image

    d = json.loads((TMP / f"{e}.json").read_text(encoding="utf-8"))
    a = A(d)
    t = TEKS[e](a)
    R, ref, L = d["R"], d["referensi"], d["lain"]
    out = HERE / d["pdf"]
    for name, style in (("Arial", "regular"), ("Arial-Bold", "bold"), ("Arial-Italic", "italic")):
        pdfmetrics.registerFont(TTFont(name, font_path(style)))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
    S = {
        "body": ParagraphStyle("body", fontName="Arial", fontSize=10, leading=14.3, textColor=HexColor(INK)),
        "small": ParagraphStyle("small", fontName="Arial", fontSize=8.6, leading=11.8, textColor=HexColor(GRAY)),
        "table": ParagraphStyle("table", fontName="Arial", fontSize=8.3, leading=11, textColor=HexColor(INK)),
        "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8.2, leading=11, textColor=white),
        "box": ParagraphStyle("box", fontName="Arial", fontSize=8.7, leading=11.7, textColor=HexColor(INK)),
    }
    c = canvas.Canvas(str(out), pagesize=A4, pageCompression=1)
    no = e[1:]
    c.setTitle(f"InceptionTime v11 eksperimen {no}: {d['judul']}")
    c.setAuthor("v11_model_tim/inceptiontime (dijalankan pemilik repo)")
    c.setSubject(t["pertanyaan"])
    W_, H = A4
    M = 40
    CW = W_ - 2 * M
    state = {"page": 0}
    semua = d["semua"]
    lain = [s for s in semua if s != ref]
    nL = len(L)

    def para(txt, y, style="body", x=M, width=None, gap=8):
        p = Paragraph(txt, S[style])
        _, h = p.wrap(width or CW, H)
        assert y - h >= 51, f"hal {state['page']}: paragraf lewat footer: {txt[:60]}"
        p.drawOn(c, x, y - h)
        return y - h - gap

    def page(title, kicker):
        if state["page"]:
            c.showPage()
        state["page"] += 1
        n = state["page"]
        c.setFillColor(HexColor(TEAL)); c.rect(M, H - 41, 32, 4, fill=1, stroke=0)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial-Bold", 8)
        c.drawString(M + 43, H - 42, f"SIHEDAF  /  V11 INCEPTIONTIME  /  EKSPERIMEN {no}")
        fs = 20
        while pdfmetrics.stringWidth(title, "Arial-Bold", fs) > CW and fs > 13:
            fs -= .5
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", fs); c.drawString(M, H - 78, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 97, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, f"{d['tanggal']} | Sumber: v11_model_tim/inceptiontime/eksperimen.py, "
                            f"cache/v11/inceptiontime/{d['file_banding']}")
        c.drawRightString(W_ - M, 29, f"{n:02d}")
        c.bookmarkPage(f"p{n}"); c.addOutlineEntry(title, f"p{n}", 0, False)
        return H - 118

    def section(title, y):
        c.setFont("Arial-Bold", 11.5); c.setFillColor(HexColor(INK)); c.drawString(M, y - 12, title)
        return y - 24

    def table(headers, rows, y, widths=None, bold_rows=()):
        widths = widths or [CW / len(headers)] * len(headers)
        cells = [[Paragraph(str(v), S["head"]) for v in headers]]
        cells += [[Paragraph(str(v), S["table"]) for v in row] for row in rows]
        tb = Table(cells, colWidths=widths, hAlign="LEFT")
        st = [("BACKGROUND", (0, 0), (-1, 0), HexColor(INK)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
              ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
              ("LINEBELOW", (0, -1), (-1, -1), .5, HexColor("#D8E1E9")),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F1F5F8")])]
        for r in bold_rows:
            st.append(("BACKGROUND", (0, r), (-1, r), HexColor("#E3F1EF")))
        tb.setStyle(TableStyle(st))
        _, h = tb.wrap(CW, H)
        assert y - h >= 51, f"hal {state['page']}: tabel lewat footer ({y - h:.0f})"
        tb.drawOn(c, M, y - h)
        return y - h - 11

    def chart(name, y, maxh):
        p = TMP / f"{e}_{name}.png"
        iw, ih = Image.open(p).size
        w = CW; h = w * ih / iw
        if h > maxh:
            h = maxh; w = h * iw / ih
        assert y - h >= 51, (name, y, h)
        c.drawImage(str(p), M + (CW - w) / 2, y - h, width=w, height=h, mask="auto")
        return y - h - 9

    def callout(title, txt, y, tint="#EDF6F5"):
        p = Paragraph(f"<b>{title}</b><br/>{txt}", S["body"])
        _, h = p.wrap(CW - 24, H)
        assert y - h - 20 >= 51, (state["page"], title, y, h)
        c.setFillColor(HexColor(tint)); c.roundRect(M, y - h - 20, CW, h + 20, 6, fill=1, stroke=0)
        p.drawOn(c, M + 12, y - h - 10)
        return y - h - 30

    def ci(m, nd=3):
        pt, lo, hi = m
        return f"{fnum(pt, nd)} <font color='{GRAY}' size='7.2'>[{fnum(lo, 2)}–{fnum(hi, 2)}]</font>"

    def dci(x):
        return f"{sg(x[0])} <font color='{GRAY}' size='7.2'>[{sg(x[1], 2)}…{sg(x[2], 2)}]</font>"

    def kode(s):
        isi = ", ".join(f"{k}={v}" for k, v in R[s]["setelan"].items()) or "(bawaan)"
        return f"<font name='Courier' size='7'>{isi}</font>"

    def nm(s):
        return f"<b>{s}</b>" if s == ref else s

    v31, v1 = d["v31"]["metrik"], d["v1"]["metrik"]
    w_afl = d["n_windows"] / (3 * d["class_counts"][2])         # class_weights seimbang untuk AFL
    i_ref = 2 + semua.index(ref)                                # baris referensi di tabel yang diawali v3.1 (baris 0 = judul)
    i_ref_l = 2 + nL + semua.index(ref)                         # ... yang diawali v3.1 dan kandidat v11

    # ── Hal 1: ringkasan ──────────────────────────────────────────────
    y = page(f"Eksperimen {no}: {d['judul']}",
             f"InceptionTime (v11, dijalankan pemilik repo) | 5 ulangan × 5 fold pasien | data detektor 50 Hz | {d['tanggal']}")
    cardw = (CW - 24) / 4
    for i, (val, lab) in enumerate(t["cards"]):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 84, cardw, 84, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 23); c.drawString(x + 11, y - 36, val)
        p = Paragraph(lab, ParagraphStyle("kartu", fontName="Arial", fontSize=8.3, leading=10, textColor=HexColor(GRAY)))
        p.wrap(cardw - 20, 40); p.drawOn(c, x + 11, y - 52 - p.height + 10)
    y -= 100
    y = para(f"*Semua angka: InceptionTime + bias kelas, prediksi out-of-fold rata-rata 5 ulangan × 5 fold pasien, data detektor "
             f"50 Hz, CI 95% bootstrap pasien 2.000×. Pertanyaan: {t['pertanyaan']}", y, "small")
    y = callout("Jawaban singkat", t["jawaban"], y)
    y = section("Temuan utama", y)
    for judul, isi in t["temuan"]:
        y = para(f"<b>{judul}</b> {isi}", y, gap=6)
    daftar = ["perbandingan dengan versi sebelumnya (2)", "metode (3)", "hasil (4)", "berpasangan (5)", "per pasien (6)",
              "diagnosa bias (7)", "tanpa 04936 (8)", "uji takikardia (9)", "kesimpulan &amp; batasan (10)", "lampiran teknis (11)"]
    para("Halaman: " + ", ".join(daftar) + ".", y - 2, "small")

    # ── Hal 2: perbandingan dengan versi sebelumnya ───────────────────
    y = page("Perbandingan dengan versi sebelumnya", "Data uji, fold pasien, dan sampel bootstrap yang sama | + bias kelas cross-fit")
    rows = [["v1 (versi lama, argmax)", ci(v1["Macro-F1"]), ci(v1["AFL F1"]), fnum(v1["Normal R"][0]), "–", "–"],
            ["<b>v3.1 (produksi)</b>", ci(v31["Macro-F1"]), ci(v31["AFL F1"]), fnum(v31["Normal R"][0]), "–",
             f"{sg(d['v31_vs_v1']['Macro-F1'][0])} (P = {fnum(d['v31_vs_v1']['Macro-F1'][3], 2)})"]]
    for m in L:
        rows.append([f"{L[m]['nama']} (kandidat v11)", ci(L[m]["metrik"]["Macro-F1"]), ci(L[m]["metrik"]["AFL F1"]),
                     fnum(L[m]["metrik"]["Normal R"][0]), "–", "–"])
    for s in d["kunci"]:
        rows.append([f"InceptionTime <b>{s}</b>", ci(R[s]["bias"]["Macro-F1"]), ci(R[s]["bias"]["AFL F1"]),
                     fnum(R[s]["bias"]["Normal R"][0]), f"{sg(R[s]['v31']['Macro-F1'][0])} (P = {fnum(R[s]['v31']['Macro-F1'][3], 3)})",
                     f"{sg(R[s]['v1']['Macro-F1'][0])} (P = {fnum(R[s]['v1']['Macro-F1'][3], 2)})"])
    y = table(["Sistem", "Macro-F1 [CI]", "AFL F1 [CI]", "Normal R", "Δ Macro-F1 vs v3.1", "Δ Macro-F1 vs v1"], rows, y,
              widths=[140, 82, 82, 50, 83, CW - 437], bold_rows=[2])
    y = chart("forest", y, 270)
    y = para("Belah ketupat hijau = v3.1, abu-abu = v1, plus abu-abu = kandidat v11 (BiGRU dan TCN wp100). Kotak oranye = "
             + (f"InceptionTime {ref} (referensi), lingkaran biru = setelan lain. " if lain else f"InceptionTime {ref}. ")
             + "Garis putus-putus = InceptionTime " + ref + ". Garis horizontal = CI 95%. Lebar CI AFL F1 berasal dari hanya "
             "7 pasien AFL.", y, "small")
    bb = list(d["bersarang"])
    if bb:
        rows = [[s, f"{fnum(d['bersarang'][s]['kandidat']['titik'])} <font color='{GRAY}' size='7.2'>"
                    f"[{fnum(d['bersarang'][s]['kandidat']['ci'][0], 2)}–{fnum(d['bersarang'][s]['kandidat']['ci'][1], 2)}]</font>",
                 f"{sg(d['bersarang'][s]['vs_v1']['delta'])} <font color='{GRAY}' size='7.2'>[{sg(d['bersarang'][s]['vs_v1']['ci'][0], 2)}…"
                 f"{sg(d['bersarang'][s]['vs_v1']['ci'][1], 2)}]</font>", fnum(d["bersarang"][s]["vs_v1"]["p"], 2),
                 vonis(d["bersarang"][s]["vs_v1"]["p"])] for s in bb]
        y = section("Bootstrap bersarang vs v1 (bias dipilih ulang di setiap sampel; uji paling ketat)", y)
        table(["Setelan", "Macro-F1 [CI]", "Δ vs v1 [CI]", "P", "Vonis"], rows, y, widths=[110, 110, 120, 60, CW - 400])

    # ── Hal 3: metode ─────────────────────────────────────────────────
    y = page("Metode", "Satu model (InceptionTime), satu protokol; hanya setelan training yang berubah")
    y = para(t["hasil"], y)
    rows = []
    for s in semua:
        n = R[s]["n_latih"]
        rows.append([nm(s), kode(s), f"{R[s]['tempat']} ({R[s]['jalur']})", thou(R[s]["param"]),
                     " / ".join(thou(x) for x in n), f"{R[s]['waktu']:.0f} s"])
    y = table(["Setelan", "SETELAN (selain bawaan)", "Dilatih di", "Param.", "Window latih per kelas (r0 k0)", "s / fold"],
              rows, y, widths=[62, 120, 125, 46, 112, CW - 465], bold_rows=[1 + semua.index(ref)])
    y = para(f"Bawaan: 6 modul Inception (kernel 3, 5, 9 + max-pool; 16 filter per cabang), shortcut residual tiap 3 modul, "
             f"dropout 0, 30 epoch, batch 256, AdamW lr 1e-3, weight decay 1e-4, cosine, cross-entropy dengan bobot kelas "
             f"<i>class_weights(y) ** weight_power</i> (bawaan 0,5 = lunak). Data: {thou(d['n_windows'])} window 10 detik dari "
             f"{d['n_pasien']} pasien AFDB (Normal {thou(d['class_counts'][0])}, AFIB {thou(d['class_counts'][1])}, AFL "
             f"{thou(d['class_counts'][2])}), R-peak dari detektor 50 Hz.", y, "small")
    steps = [("1  Fold pasien", "5 ulangan × 5 fold resmi (folds.npz); seed latih 2026 + 100·r + k"),
             ("2  Latih & OOF", "Tiap window diprediksi model yang tidak pernah melihat pasiennya"),
             ("3  Bias kelas", "Penggeser ambang dipelajari dari fold lain (cross-fit, grid −7…+3)"),
             ("4  Bootstrap", "Pasien diambil ulang 2.000× → CI 95% dan selisih berpasangan"),
             ("5  Uji stres", "Ritme teratur sintetis 60–170 bpm dengan bobot model yang disimpan")]
    bw, bh = (CW - 40) / 5, 80
    for i, (tt, txt) in enumerate(steps):
        x = M + i * (bw + 10)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - bh, bw, bh, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.rect(x, y - 4, bw, 4, fill=1, stroke=0)
        p = Paragraph(f"<b>{tt}</b><br/>{txt}", S["box"])
        p.wrap(bw - 14, bh); p.drawOn(c, x + 7, y - 12 - p.height)
    y -= bh + 14
    y = section("Istilah", y)
    table(["Istilah", "Arti sederhana"], [
        ["OOF (out-of-fold)", "Prediksi untuk pasien yang tidak ikut dilatih. Ini yang membuat skor jujur."],
        ["Bobot kelas", "Kesalahan pada kelas langka (AFL) dihitung lebih berat saat training. weight_power 1,0 = penuh "
                        f"(AFL ±{w_afl:.0f}×), 0,5 = lunak (akar, AFL ±{w_afl ** .5:.0f}×)."],
        ["Bias kelas", "Penggeser ambang keputusan setelah training. Negatif untuk AFL = model harus lebih yakin sebelum menyebut AFL."],
        ["Margin skor AFL", "log p(AFL) − log p kelas lain terbesar, satuannya sama dengan bias. AFL ditebak kira-kira bila "
                            "margin lebih besar dari besarnya bias AFL."],
        ["AUROC AFL", "Seberapa baik skor AFL mengurutkan window AFL di atas window lain, tanpa peduli ambang (1 = sempurna)."],
        ["CI 95%", "Rentang tempat skor sebenarnya kemungkinan besar berada."],
        ["P(Δ > 0)", "Peluang setelan lebih baik dari pembanding pada sampel pasien yang sama. ≥ 0,95 lebih baik; ≤ 0,05 lebih "
                     "buruk; selebihnya tidak terbedakan."],
    ], y, widths=[100, CW - 100])

    # ── Hal 4: hasil ──────────────────────────────────────────────────
    y = page("Hasil semua setelan", "+ bias kelas cross-fit | rata-rata 5 ulangan, CI 95% bootstrap pasien | baris hijau = referensi")
    rows = [["v3.1 (produksi)", ci(v31["Macro-F1"]), ci(v31["AFL F1"]), fnum(v31["AFL P"][0]), ci(v31["AFL R"]),
             fnum(v31["AFIB F1"][0]), fnum(v31["Normal R"][0]), "–"]]
    for s in semua:
        m = R[s]["bias"]
        rows.append([nm(s), ci(m["Macro-F1"]), ci(m["AFL F1"]), fnum(m["AFL P"][0]), ci(m["AFL R"]),
                     fnum(m["AFIB F1"][0]), fnum(m["Normal R"][0]), fnum(R[s]["argmax"]["Macro-F1"][0])])
    y = table(["Setelan", "Macro-F1", "AFL F1", "AFL P", "AFL R", "AFIB F1", "Normal R", "Macro-F1 argmax"], rows, y,
              widths=[70, 80, 80, 44, 80, 46, 46, CW - 446], bold_rows=[i_ref])
    rows = [[f"{nm(s)}, {k}", ci(R[s][j]["Macro-F1"]), ci(R[s][j]["AFL F1"]), fnum(R[s][j]["AFL P"][0]), ci(R[s][j]["AFL R"]),
             fnum(R[s][j]["Normal R"][0]), thou(round(R[s]["palsu" if j == "bias" else "palsu_argmax"]["total"]))]
            for s in semua for k, j in (("tanpa bias", "argmax"), ("+ bias", "bias"))]
    y = section("Efek bias kelas (argmax vs + bias cross-fit)", y)
    y = table(["InceptionTime", "Macro-F1", "AFL F1", "AFL P", "AFL R", "Normal R", "AFL palsu / ulangan"], rows, y,
              widths=[100, 82, 82, 44, 82, 50, CW - 440])
    callout("Cara membaca", "Angka tertinggi di tabel tidak otomatis lebih baik: selisih harus diuji berpasangan (halaman "
            "berikutnya). Tanpa bias, InceptionTime menebak AFL terlalu sering. Bias kelas menukar recall AFL dengan AFL palsu "
            "yang jauh lebih sedikit; selisih Macro-F1 argmax dan + bias menunjukkan seberapa besar koreksi ambang yang "
            "dibutuhkan.", y)

    # ── Hal 5: berpasangan ────────────────────────────────────────────
    y = page("Perbandingan berpasangan", "Sampel bootstrap pasien yang sama | + bias kelas cross-fit")
    if lain:
        y = chart("paired", y, 150)
        rows = []
        for s in lain:
            b = a.b(s)
            rows.append([s, dci(b), fnum(b[3], 2), vonis(b[3]), f"{sg(a.b(s, 'AFL F1')[0])} ({fnum(a.b(s, 'AFL F1')[3], 2)})",
                         f"{sg(a.b(s, 'AFL R')[0])} ({fnum(a.b(s, 'AFL R')[3], 2)})",
                         f"{sg(a.b(s, 'Normal R')[0])} ({fnum(a.b(s, 'Normal R')[3], 2)})"])
        y = table([f"Setelan vs {ref}", "Δ Macro-F1 [CI]", "P", "Vonis", "Δ AFL F1 (P)", "Δ AFL R (P)", "Δ Normal R (P)"], rows,
                  y, widths=[70, 105, 34, 76, 75, 75, CW - 435])

    def dp(x):                                            # Δ Macro-F1, P, dan vonis dalam satu sel
        return f"{sg(x[0])} (P = {fnum(x[3], 3)}) {vonis(x[3])}"

    rows = [[nm(s), dp(R[s]["v31"]["Macro-F1"])] + [dp(R[s]["vs_lain"][m]["Macro-F1"]) for m in L] for s in semua]
    y = section("Δ Macro-F1 terhadap v3.1 dan kandidat v11 (semua + bias)", y)
    y = table(["InceptionTime", "vs v3.1"] + [f"vs {L[m]['nama']}" for m in L], rows, y,
              widths=[85] + [(CW - 85) / (1 + nL)] * (1 + nL))
    rows = [[nm(s)] + [f"{sg(R[s]['vs_lain'][m][met][0])} (P = {fnum(R[s]['vs_lain'][m][met][3], 2)})"
                       for m in L for met in ("AFL F1", "AFL R")] for s in semua]
    y = table(["InceptionTime"] + [f"{L[m]['nama']}: Δ {met}" for m in L for met in ("AFL F1", "AFL R")], rows, y,
              widths=[85] + [(CW - 85) / (2 * nL)] * (2 * nL))
    callout("Intinya", t["paired"], y)

    # ── Hal 6: per pasien ─────────────────────────────────────────────
    y = page("AFL per pasien dan AFL palsu", "Recall AFL tiap pasien AFL, rata-rata 5 ulangan | asal AFL palsu")
    y = chart("afl_heatmap", y, 225)
    rows = [["v3.1"] + [thou(round(d["v31"]["palsu"][k])) for k in ("total", "dari_normal", "dari_afib", "pasien_tanpa_afl")]
            + ["–", "–"]]
    rows += [[L[m]["nama"]] + [thou(round(L[m]["afl_palsu"][k])) for k in ("total", "dari_normal", "dari_afib",
                                                                            "pasien_tanpa_afl")] + ["–", "–"] for m in L]
    for s in semua:
        fp, bi = R[s]["palsu"], R[s]["biasstat"]
        rows.append([nm(s)] + [thou(round(fp[k])) for k in ("total", "dari_normal", "dari_afib")]
                    + [thou(round(fp["pasien_tanpa_afl"])), f"{fnum(bi['median'][2], 2)} [{fnum(bi['min'][2], 2)}…{fnum(bi['maks'][2], 2)}]",
                       pct(bi["di_batas"], 0)])
    y = table(["Sistem", "AFL palsu / ulangan", "dari Normal", "dari AFIB", "di pasien tanpa AFL", "Bias AFL median [min…maks]",
               "Bias di batas grid"], rows, y, widths=[70, 70, 58, 58, 72, 105, CW - 433], bold_rows=[i_ref_l])
    callout("Intinya", t["pasien"], y)

    # ── Hal 7: diagnosa bias ──────────────────────────────────────────
    y = page("Diagnosa: bias AFL per fold, AUROC, dan margin", "Apakah AFL 2:1 (08378) bertahan setelah bias kelas, dan kenapa")
    y = chart("bias_afl", y, 220)
    mgn = lambda m: [fnum(m["08378"], 2), fnum(m["04936"], 2), fnum(m["non_afl_p995"], 2)]  # noqa: E731
    rows = [["v3.1", fnum(d["v31"]["auroc"]), "–"] + mgn(d["v31"]["margin"])]
    for m in L:
        lo, hi, med = a.rentang_bias_l(m)
        rows.append([L[m]["nama"], fnum(L[m]["auroc"]), f"{fnum(med, 2)} [{fnum(lo, 2)}…{fnum(hi, 2)}]"] + mgn(L[m]["margin"]))
    for s in semua:
        lo, hi, med = a.rentang_bias(s)
        rows.append([nm(s), fnum(R[s]["auroc"]), f"{fnum(med, 2)} [{fnum(lo, 2)}…{fnum(hi, 2)}]"] + mgn(R[s]["margin"]))
    y = table(["Sistem", "AUROC AFL", "Bias AFL median [min…maks]", "Margin AFL 08378 (AFL 2:1)", "Margin AFL 04936",
               "Margin non-AFL (persentil 99,5)"], rows, y, widths=[80, 58, 118, 88, 70, CW - 414], bold_rows=[i_ref_l])
    y = para("Margin = log p(AFL) − log p kelas lain terbesar per window (median di window AFL pasien itu, 5 ulangan digabung), "
             "satuan yang sama dengan bias kelas. Makin besar margin di window AFL dan makin kecil di window non-AFL, makin aman "
             "AFL dari bias negatif.", y, "small")
    callout("Intinya", t["diagnosa"], y)

    # ── Hal 8: tanpa 04936 ────────────────────────────────────────────
    tn = d["tanpa"]
    y = page("Ketergantungan pada pasien 04936",
             f"{pct(d['afl_window']['04936'] / d['class_counts'][2], 0)} window AFL berasal dari 04936 | tanpa pasien ini: "
             f"{tn['n_pasien']} pasien, {tn['n_pasien_afl']} pasien AFL, {thou(tn['n_afl'])} window AFL")
    y = chart("tanpa", y, 250)
    rows = [["v3.1", ci(tn["rows"]["v31"]["Macro-F1"]), fnum(tn["rows"]["v31"]["AFL F1"][0]), "–", "–"]]
    for k, label in [(L[m]["tanpa_kunci"], L[m]["nama"]) for m in L] + [(s, nm(s)) for s in semua]:
        x = tn["pairs_v31_minus"][k]
        rows.append([label, ci(tn["rows"][k]["Macro-F1"]), fnum(tn["rows"][k]["AFL F1"][0]), dci(x), fnum(x[3], 2)])
    y = table(["Sistem", "Macro-F1 tanpa 04936", "AFL F1", "v3.1 − sistem [CI]", "P(v3.1 lebih baik)"], rows, y,
              widths=[120, 105, 60, 120, CW - 405], bold_rows=[i_ref_l])
    callout("Intinya", t["tanpa"], y)

    # ── Hal 9: uji takikardia ─────────────────────────────────────────
    y = page("Uji stres takikardia", "Protokol butir 9: ritme teratur sintetis dan window Normal asli yang cepat")
    y = chart("takikardia", y, 190)
    pk = list(d["v31"]["pita"])
    rows = [["v3.1 (paket)", f"{d['paket_v31']}"] + [pct(d["v31"]["pita"][k][1]) for k in pk]]
    rows.append(["kontrol v7 (per fold)", " / ".join(str(v) for v in d["kontrol_v7"]["ambang"].values())] + ["–"] * len(pk))
    rows += [[L[m]["nama"], " / ".join(str(v) for v in L[m]["ambang"].values())] + [pct(L[m]["pita"][k][1]) for k in pk]
             for m in L]
    for s in semua:
        rows.append([nm(s), a.amb(s)] + [pct(R[s]["pita"][k][1]) for k in pk])
    y = table(["Sistem", "Ambang (bpm): tanpa / ±10 / G10 / G20"] + [f"Normal {k}" for k in pk], rows, y,
              widths=[90, 130] + [(CW - 220) / len(pk)] * len(pk), bold_rows=[i_ref_l + 1])
    y = para("Ambang = detak terendah di mana kurang dari separuh window sintetis ditebak Normal (makin tinggi makin baik). "
             "Kolom Normal = % window Normal asli (OOF) pada pita detak itu yang ditebak AFIB/AFL (makin rendah makin baik). "
             f"AFDB hanya punya {thou(d['v31']['pita']['120–∞ bpm'][0])} window Normal ≥ 120 bpm.", y, "small")
    callout("Intinya", t["taki"], y, "#FFF4EC")

    # ── Hal 10: kesimpulan & batasan ──────────────────────────────────
    y = page("Kesimpulan, langkah berikutnya, dan batasan", "Yang sudah pasti, yang perlu diputuskan, dan yang belum bisa dijawab")
    y = section("Keputusan", y)
    y = table(["Hal", "Status", "Alasan"], t["keputusan"], y, widths=[190, 90, CW - 280])
    y = section("Langkah berikutnya", y)
    for tt, txt in t["langkah"]:
        y = para(f"• <b>{tt}</b> {txt}", y, gap=5)
    y -= 4
    y = section("Batasan", y)
    for txt in t["batasan"] + [
            f"Semua angka dari {d['n_pasien']} pasien AFDB ({len(PASIEN_AFL)} pasien AFL, {thou(d['class_counts'][2])} window "
            "AFL) yang di-downsample ke 50 Hz. CI AFL F1 sangat lebar; angka AFL adalah perkiraan kasar.",
            "AFDB hampir tidak punya sinus takikardia asli; uji stres memakai ritme sintetis. Masalah domain antar-dataset "
            "(v8–v10) tidak tersentuh eksperimen ini.",
            "Bias kelas dipilih cross-fit di data yang sama. " + ("Bootstrap bersarang (halaman 2) memperhitungkan "
            "ketidakpastiannya." if d["bersarang"] else "Bootstrap bersarang, yang memperhitungkannya, tidak dijalankan di "
            "eksperimen ini.")]:
        y = para(f"• {txt}", y, gap=5)

    # ── Hal 11: lampiran ──────────────────────────────────────────────
    y = page("Lampiran teknis", "Lingkungan, file, dan cara mereproduksi")
    lk = R[ref]["lingkungan"]
    pk_ = lk["pustaka"]
    y = table(["Komponen", "Nilai"], [
        ["Data", f"cache/dataset_v2_detector.npz: {thou(d['n_windows'])} window, {d['n_pasien']} pasien (dicek sidik_jari.json)"],
        ["Pembagian", "cache/oof_robust/detector_e30_r5_s2026/folds.npz: 5 ulangan × 5 fold pasien, seed 2026"],
        ["Pembanding", "v3.1 = MicroTransformer + bias (OOF v3); v1 = OOF versi lama, argmax; kandidat v11 = "
                       "cache/v11/{bigru,tcn}/wp100/"],
        ["Bias & bootstrap", "grid −7…+3 langkah 0,25, cross-fit antar fold; 2.000 sampel pasien per strata"],
        ["Training & evaluasi", f"PC desktop, GPU {lk['gpu']}; Python {pk_['python']}, torch {pk_['torch']}, numpy {pk_['numpy']}, "
                                f"scikit-learn {pk_['scikit-learn']}; kode v11 commit {lk['commit']} + perubahan yang belum "
                                "di-commit. Batas memori 5 GB per job (systemd-run), tanpa OOM."],
        ["Jalur", "; ".join(f"{s}: {R[s]['jalur']}" for s in semua) + ". Training lewat bersama.runner.jalankan dan evaluasi "
                  "lewat bersama.evaluasi, sama untuk notebook dan eksperimen.py."],
        ["Pemeriksaan otomatis", "Macro-F1 (+ bias & argmax) tiap setelan dihitung ulang dari OOF di cache dan harus sama dengan "
                                 "ringkasan.json; bias per fold, margin, dan metrik kandidat v11 harus sama dengan file perbandingan"],
    ], y, widths=[100, CW - 100])
    y = section("File", y)
    y = table(["File", "Isi"], [
        ["v11_model_tim/inceptiontime/eksperimen.py", "Definisi eksperimen, training (runner), evaluasi (bersama.evaluasi)"],
        ["v11_model_tim/inceptiontime/model.py", "InceptionTime: modul Inception bermask, satu jaringan"],
        ["v11_model_tim/bersama/", "Data + sidik jari, loop DL identik v3, runner 5×5, evaluasi"],
        ["cache/v11/inceptiontime/&lt;setelan&gt;/", "Bobot (.pt) dan prediksi (.npz) per fold, ringkasan.json, setelan.json"],
        [f"cache/v11/inceptiontime/{d['file_banding']}", "Perbandingan berpasangan, vs kandidat v11, diagnosa bias & margin, "
         "tanpa 04936" + (", bootstrap bersarang" if d["bersarang"] else "")],
        ["v11_model_tim/inceptiontime/laporan/build_report.py", "Pembuat laporan ini"],
    ], y, widths=[200, CW - 200])
    y = section("Mereproduksi (dari folder v11_model_tim/inceptiontime)", y)
    para(f"Training: <font name='Courier' size='8.3'>python eksperimen.py latih {e}</font> (bisa dilanjutkan; fold yang ada "
         f"dilewati). Evaluasi: <font name='Courier' size='8.3'>python eksperimen.py evaluasi {e}</font>. Laporan: "
         f"<font name='Courier' size='8.3'>python laporan/build_report.py --extract {e}</font>, lalu perintah yang sama tanpa "
         "--extract. Satu setelan juga bisa dilatih lewat jalankan.ipynb (ubah SETELAN dan NAMA_SETELAN).", y)

    c.save()
    print(f"OK {out} ({state['page']} halaman)")
    return out


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if "--extract" in sys.argv:
        sys.path[:0] = [str(MODEL.parent), str(MODEL)]
        from eksperimen import EKSPERIMEN
        from bersama import data
        akar = data.lokasi_hasil() / "inceptiontime"
        for e in args or [k for k in EKSPERIMEN if (akar / EKSPERIMEN[k]["banding"]).exists()]:
            extract(e)
    elif "--grafik" in sys.argv:                          # gambar ulang dari _build/<e>.json saja
        for e in args or sorted(p.stem for p in TMP.glob("e?.json")):
            grafik(e)
    else:
        for e in args or sorted(p.stem for p in TMP.glob("e?.json")):
            build(e)
