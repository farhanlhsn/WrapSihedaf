"""Laporan PDF eksperimen BiGRU v11: satu PDF per eksperimen (definisi eksperimen di ../eksperimen.py).

Dua tahap (pola v3_evaluasi_robust/laporan/build_report.py):
  1. --extract [e1 e2 e3]: baca ringkasan.json tiap setelan (= sel evaluasi jalankan.ipynb) dan file perbandingan
     (`python eksperimen.py evaluasi eN`) dari cache/v11/bigru, cocokkan dengan OOF di cache (assert), lalu simpan
     angka + grafik ke laporan/_build/.
  2. tanpa --extract [e1 e2 e3]: rakit laporan/laporan_bigru_eN_*.pdf dengan ReportLab.
  (--grafik [eN]: gambar ulang grafik dari _build/eN.json tanpa menghitung ulang angka.)
Tanpa nama eksperimen = semua eksperimen yang file perbandingannya sudah ada.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # v11_model_tim/bigru/laporan
BIGRU = HERE.parent
REPO = BIGRU.parents[1]
TMP = HERE / "_build"
sys.path.insert(0, str(REPO / "tools"))
from report_fonts import font_path  # noqa: E402  (Arial → Liberation Sans → DejaVu)

# layout + palet data tervalidasi (validator dataviz: semua PASS; hijau < 3:1 → selalu didampingi tabel)
TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
BIRU, ORANYE, HIJAU = "#2a78d6", "#eb6834", "#1baf7a"
PASIEN_AFL = ["04043", "04908", "04936", "06426", "07910", "08215", "08378"]

# Per laporan: nama file, setelan yang kurvanya digambar di uji takikardia (≤ 3), baris kunci perbandingan versi,
# dan tempat training (tidak tercatat di cache; dari log antrean & kernel Kaggle).
LAPORAN = {
    "e1": dict(pdf="laporan_bigru_e1_balancing.pdf", kurva=["bawaan", "under"], kunci=["bawaan", "under"],
               kaggle=[]),
    "e2": dict(pdf="laporan_bigru_e2_tuning.pdf", kurva=["bawaan", "wp100", "over"], kunci=["bawaan", "wp100", "lr5e4"],
               kaggle=["h16", "h64", "lr5e4"]),
    "e3": dict(pdf="laporan_bigru_e3_regularisasi.pdf", kurva=["wp100", "w1_wd1e2", "w1_h16"],
               kunci=["bawaan", "wp100", "w1_wd1e3"],
               kaggle=["w1_wd1e3", "w1_wd1e2", "w1_lr5e4"]),
}
PC, KAGGLE = "PC, RTX 5060", "Kaggle, T4"


# ═════════════════════════════════════════════════════════════════════
# Tahap 1: angka + grafik
# ═════════════════════════════════════════════════════════════════════
def extract(e):
    import numpy as np
    from sklearn.metrics import f1_score

    sys.path[:0] = [str(BIGRU.parent), str(BIGRU)]
    from eksperimen import EKSPERIMEN
    from bersama import data, evaluasi as E, runner

    eks, cfg = EKSPERIMEN[e], LAPORAN[e]
    akar = data.lokasi_hasil() / "bigru"
    B = json.loads((akar / eks["banding"]).read_text(encoding="utf-8"))
    ref = eks["referensi"]
    semua = list(dict.fromkeys([ref, *eks["setelan"], *eks["pembanding"]]))
    assert set(semua) <= set(B["setelan"]), (semua, B["setelan"])
    teks_setelan = {"bawaan": {}}
    for x in EKSPERIMEN.values():
        teks_setelan.update(x["setelan"])

    d = data.muat()
    R = {}
    for s in semua:
        rk = json.loads((akar / s / "ringkasan.json").read_text(encoding="utf-8"))
        # ringkasan.json (= sel evaluasi notebook) harus cocok dengan OOF yang ada di cache sekarang
        h = runner.muat_hasil(akar / s, d)
        assert h.ulangan == rk["ulangan"] == list(range(data.N_ULANGAN)), s
        pb, _ = E.keputusan(h.oof, d.folds, d.y)
        for nama, pred in (("bias", pb), ("argmax", h.oof.argmax(-1))):
            mf = float(np.mean([f1_score(d.y, p, average="macro", labels=[0, 1, 2]) for p in pred]))
            assert abs(mf - rk["metrik"][nama]["Macro-F1"][0]) < 1e-9, (s, nama, mf, rk["metrik"][nama]["Macro-F1"][0])
        pita = {k.split(" | ")[1]: [v["n"], 1 - v["model"][0]] for k, v in rk["pita_detak"].items() if k.startswith("Normal")}
        R[s] = {"bias": rk["metrik"]["bias"], "argmax": rk["metrik"]["argmax"], "v31": rk["banding_v31"],
                "v1": rk["banding_v1"], "afl": rk["afl_per_pasien"]["bias"], "palsu": rk["afl_palsu"]["bias"],
                "biasstat": rk["bias"], "ambang": rk["takikardia"]["model"]["ambang"],
                "proporsi": rk["takikardia"]["model"]["proporsi"], "pita": pita, "waktu": rk["waktu_per_fold"],
                "param": rk["info"]["parameter"], "n_latih": rk["info"]["n_latih"], "setelan_efektif": rk["setelan"],
                "setelan": teks_setelan[s], "tempat": KAGGLE if s in cfg["kaggle"] else PC, "lingkungan": rk["lingkungan"]}
    rr = json.loads((akar / ref / "ringkasan.json").read_text(encoding="utf-8"))
    for s in semua:                                   # pembanding v3.1/v1 sama di semua ringkasan
        rk = json.loads((akar / s / "ringkasan.json").read_text(encoding="utf-8"))
        assert rk["metrik"]["v31"] == rr["metrik"]["v31"] and rk["metrik"]["v1"] == rr["metrik"]["v1"], s
    # selisih berpasangan konsisten dengan ringkasan; tanpa-04936 dan bersarang memuat setelan yang sama
    for s in semua:
        if s != ref:
            b = B["banding"][f"{s}_vs_{ref}"]["banding"]["Macro-F1"]
            assert abs(b[0] - (R[s]["bias"]["Macro-F1"][0] - R[ref]["bias"]["Macro-F1"][0])) < 1e-9, s
        assert s in B["tanpa_04936"]["rows"], s
    # cocokkan juga dengan output jalankan.ipynb yang dijalankan (bila salinannya ada di logs/)
    nb = REPO / "logs" / "v11_bigru" / "jalankan_bigru_bawaan_dijalankan.ipynb"
    if nb.exists() and "bawaan" in R:
        teks = json.dumps(json.loads(nb.read_text(encoding="utf-8")))
        m = R["bawaan"]["bias"]["Macro-F1"]
        assert f"bigru + bias      {m[0]:.3f} [{m[1]:.2f}\\u2013{m[2]:.2f}]" in teks, "angka notebook tidak cocok"
        print("cocok dengan output jalankan.ipynb (bawaan)")

    # pembanding: v3.1, v1, dan BiGRU v3 bobot penuh (OOF v3, mesin lain) terhadap v3.1 & v1
    oofd = data.lokasi_data() / data.OOF
    v3 = np.stack([np.load(oofd / f"BiGRUAttention_r{r}.npz")["oof"] for r in range(data.N_ULANGAN)])
    p_v3, _ = E.keputusan(v3, d.folds, d.y)
    p_v31, _ = E.keputusan(d.oof_v31, d.folds, d.y)
    p_v1 = d.oof_v1.argmax(-1)
    assert abs(float(np.mean([f1_score(d.y, p, average="macro") for p in p_v3])) - B["v3_bobot_penuh"]["metrik"]["Macro-F1"][0]) < 1e-9
    p1 = akar / EKSPERIMEN["e1"]["banding"]
    bersarang_bawaan = (json.loads(p1.read_text(encoding="utf-8"))["bersarang"]["bawaan"] if p1.exists()
                        else json.loads((akar / "banding_gpu.json").read_text(encoding="utf-8"))["bersarang_bawaan_vs_v1"])
    pita31 = {k.split(" | ")[1]: [v["n"], 1 - v["v31"][0]] for k, v in rr["pita_detak"].items() if k.startswith("Normal")}
    afl_win = {p: int(((d.records == p) & (d.y == 2)).sum()) for p in PASIEN_AFL}
    payload = {
        "e": e, "judul": eks["judul"], "tanggal": eks["tanggal"], "referensi": ref, "baru": list(eks["setelan"]),
        "pembanding": eks["pembanding"], "semua": semua, "file_banding": eks["banding"], **cfg, "R": R,
        "v31": {"metrik": rr["metrik"]["v31"], "afl": rr["afl_per_pasien"]["v31"], "palsu": rr["afl_palsu"]["v31"],
                "pita": pita31},
        "v1": {"metrik": rr["metrik"]["v1"], "afl": rr["afl_per_pasien"]["v1"], "palsu": rr["afl_palsu"]["v1"]},
        "v3": {**B["v3_bobot_penuh"], "vs_v31": E.banding(d, p_v3, p_v31), "vs_v1": E.banding(d, p_v3, p_v1)},
        "v31_vs_v1": E.banding(d, p_v31, p_v1), "bersarang_bawaan": bersarang_bawaan,
        "banding": B["banding"], "tanpa": B["tanpa_04936"], "crossfit": B.get("pilih_crossfit"),
        "bersarang": B.get("bersarang", {}),
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
    R, ref, semua, afl_win = P["R"], P["referensi"], P["semua"], P["afl_window"]
    # ── grafik ─────────────────────────────────────────────────────────
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

    # 1. forest: Macro-F1 & AFL F1 (+ bias) dengan CI; v3.1 dan v1 di atas
    rows = [("v3.1 (produksi)", P["v31"]["metrik"], HIJAU, "D"), ("v1 (argmax)", P["v1"]["metrik"], GRAY, "D")]
    rows += [(s, R[s]["bias"], ORANYE if s == ref else BIRU, "s" if s == ref else "o") for s in semua]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, .36 * len(rows) + 1.0), sharey=True)
    for ax, met in zip(axes, ("Macro-F1", "AFL F1")):
        for i, (_, m, col, mk) in enumerate(rows):
            pt, lo, hi = m[met]
            ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round", alpha=.85)
            ax.plot(pt, i, mk, color=col, ms=7, mec="white", mew=1.2)
        ax.axvline(R[ref]["bias"][met][0], color="#8A97A6", lw=1, ls="--")
        ax.axhline(1.5, color="#E0E6EC", lw=.8)
        ax.set_title(met + " (+ bias, CI 95%)", loc="left", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        style(ax)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "forest")

    # 2. selisih berpasangan vs referensi
    lain = [s for s in semua if s != ref]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, .36 * len(lain) + 1.25), sharey=True)
    for ax, met in zip(axes, ("Macro-F1", "AFL F1")):
        ax.axvline(0, color=INK, lw=1)
        for i, s in enumerate(lain):
            dl, lo, hi, p = P["banding"][f"{s}_vs_{ref}"]["banding"][met]
            if max(abs(dl), abs(lo), abs(hi)) < 1e-12:                     # identik (kontrol noise)
                col, mk, isi = GRAY, "D", True
            else:
                col, mk, isi = (HIJAU, "^", True) if p >= .95 else (ORANYE, "v", True) if p <= .05 else (BIRU, "o", False)
            ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
            ax.plot(dl, i, mk, ms=8, color=col, mfc=col if isi else "white", mec=col, mew=1.6)
        ax.set_title(f"Δ {met} terhadap {ref}", loc="left", fontsize=10, fontweight="bold")
        style(ax)
    axes[0].set_yticks(range(len(lain)), lain)
    axes[0].invert_yaxis()
    ada_identik = any(max(map(abs, P["banding"][f"{s}_vs_{ref}"]["banding"]["Macro-F1"][:3])) < 1e-12 for s in lain)
    fig.text(.01, -.03, "▲ hijau = lebih baik (P ≥ 0,95)   ▼ oranye = lebih buruk (P ≤ 0,05)   ○ biru = tidak terbedakan. "
             + ("Belah ketupat abu-abu = identik. " if ada_identik else "") + "Garis = CI 95%.", fontsize=8.6, color=GRAY)
    fig.tight_layout()
    save(fig, "paired")

    # 3. heatmap recall AFL per pasien
    kol = [("v3.1", P["v31"]["afl"]), ("BiGRU\nv3", P["v3"]["afl_per_pasien"])]
    kol += [(s.replace("_", "_\n", 1) if len(s) > 6 else s, R[s]["afl"]) for s in semua]
    mat = np.array([[k[1][p] for k in kol] for p in PASIEN_AFL])
    cmap = LinearSegmentedColormap.from_list("biru", ["#EEF4FC", "#86b6ef", BIRU, "#104281"])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(kol)), [k[0] for k in kol], fontsize=7.8 if len(kol) > 9 else 8.6)
    ax.axvline(1.5, color="white", lw=4)
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
    cb.set_label("Recall AFL (+ bias, rata-rata 5 ulangan)", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(comma)
    fig.tight_layout()
    save(fig, "afl_heatmap")

    # 4. tanpa 04936: semua pasien vs tanpa 04936
    T = P["tanpa"]["rows"]
    baris = [("v3.1", P["v31"]["metrik"]["Macro-F1"][0], T["v31"]["Macro-F1"][0]),
             ("BiGRU v3 bobot penuh", P["v3"]["metrik"]["Macro-F1"][0], T["v3_bobot_penuh"]["Macro-F1"][0])]
    baris += [(s, R[s]["bias"]["Macro-F1"][0], T[s]["Macro-F1"][0]) for s in semua]
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

    # 5. uji takikardia: % ditebak Normal per bpm, 4 jenis variasi (small multiples)
    bpm = P["bpm"]
    jenis = list(R[ref]["proporsi"])
    gaya = list(zip(P["kurva"], (BIRU, ORANYE, HIJAU), ("o", "s", "^")))
    fig, axes = plt.subplots(1, len(jenis), figsize=(7.2, 2.5), sharey=True)
    for ax, j in zip(axes, jenis):
        kv = [p[0] for p in P["kontrol_v7"]["proporsi"][j]]
        ax.plot(bpm, kv, color="#8A97A6", lw=1.3, ls="--", label="kontrol v7 (MicroTransformer)")
        for s, col, mk in gaya:
            ax.plot(bpm, [p[0] for p in R[s]["proporsi"][j]], color=col, lw=1.8, marker=mk, ms=3.6, markevery=2, label=s)
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
    print(f"OK {TMP / (e + '.json')} + 5 grafik")


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


def identik(b):
    """Selisih berpasangan nol di semua sampel bootstrap (mis. bawaan_ulang): P(Δ > 0) = 0 bukan berarti lebih buruk."""
    return abs(b[0]) < 1e-12 and abs(b[1]) < 1e-12 and abs(b[2]) < 1e-12


def vonis(p, b=None):
    if b is not None and identik(b):
        return "identik"
    return "lebih baik" if p >= .95 else "lebih buruk" if p <= .05 else "tidak terbedakan"


class A:
    """Akses angka satu laporan (semua angka di teks lewat sini, tidak diketik manual)."""

    def __init__(self, d):
        self.d, self.R, self.ref = d, d["R"], d["referensi"]

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

    def afl(self, s, p):
        return fnum(self.R[s]["afl"][p], 2)

    def tanpa(self, s, met="Macro-F1"):
        return self.d["tanpa"]["rows"][s][met][0]

    def tp(self, s):                                      # v3.1 − s tanpa 04936: [Δ, lo, hi, P]
        return self.d["tanpa"]["pairs_v31_minus"][s]

    def amb(self, s):
        return " / ".join(str(v) for v in self.R[s]["ambang"].values())

    def pita(self, s, k="120–∞ bpm"):
        return self.R[s]["pita"][k][1]

    def bs(self, s):
        return self.d["bersarang"][s]


# ── teks per eksperimen (angka lewat A, narasi ditulis setelah melihat hasil) ───────────────────────
def teks_e1(a):
    d = a.d
    pu = a.b("under")
    v3 = d["v3"]["metrik"]
    return dict(
        pertanyaan="Apakah BiGRU dengan bobot kelas lunak setara model produksi, dan apakah undersampling data latih (arahan "
                   "pembimbing) membantu?",
        cards=[(a.f("bawaan"), "Macro-F1 BiGRU bawaan*"), (sg(a.v31("bawaan")[0]), f"Selisih vs v3.1 (P = {fnum(a.v31('bawaan')[3], 2)})"),
               (a.f("under"), "Macro-F1 undersampling"), (fnum(a.m("bawaan", "Normal R")), "Recall Normal bawaan")],
        jawaban=f"BiGRU dengan bobot kelas lunak <b>setara</b> model produksi v3.1 (Macro-F1 {a.f('bawaan')} vs "
                f"{fnum(d['v31']['metrik']['Macro-F1'][0])}, P = {fnum(a.v31('bawaan')[3], 2)}) dan lebih jarang membuat alarm "
                f"palsu (recall Normal {fnum(a.m('bawaan', 'Normal R'))}). <b>Undersampling jauh lebih buruk</b>: Macro-F1 "
                f"{a.f('under')}, selisih {sg(pu[0])} (P = {fnum(pu[3], 2)}). Kelemahan bawaan: AFL 2:1 di pasien 08378 "
                f"({a.afl('bawaan', '08378')}) dan skor yang bergantung pada pasien 04936.",
        temuan=[
            ("1. Setara v3.1 dan BiGRU v3.", f"Selisih Macro-F1 {sg(a.v31('bawaan')[0])} terhadap v3.1 (P = "
             f"{fnum(a.v31('bawaan')[3], 2)}). BiGRU v3 (bobot penuh, mesin lain) {fnum(v3['Macro-F1'][0])}."),
            ("2. Lebih sedikit alarm palsu.", f"Recall Normal {fnum(a.m('bawaan', 'Normal R'))} vs v3.1 "
             f"{fnum(d['v31']['metrik']['Normal R'][0])} (P = {fnum(a.v31('bawaan', 'Normal R')[3], 2)}). Ambang takikardia "
             f"{a.amb('bawaan')} bpm, sedikit di atas paket v3.1 ({d['paket_v31']})."),
            ("3. Harga bobot lunak: AFL 2:1.", f"Recall 08378 {a.afl('bawaan', '08378')} (v3.1 "
             f"{fnum(d['v31']['afl']['08378'], 2)}). Tanpa 04936, Macro-F1 turun ke {fnum(a.tanpa('bawaan'))} (v3.1 "
             f"{fnum(a.tanpa('v31'))})."),
            ("4. Undersampling merusak.", f"Tiap kelas dipotong ke jumlah AFL (±{thou(a.R['under']['n_latih'][0])} window per "
             f"kelas). AFL palsu naik {thou(round(a.R['bawaan']['palsu']['total']))} → "
             f"{thou(round(a.R['under']['palsu']['total']))} per ulangan, "
             f"dan {pct(a.pita('under', '100–110 bpm'), 0)} window Normal 100–110 bpm ditebak AFIB/AFL."),
            ("5. Uji paling ketat.", f"Bootstrap bersarang (bias dipilih ulang tiap sampel): bawaan vs v1 P = "
             f"{fnum(a.bs('bawaan')['vs_v1']['p'], 2)}, artinya belum pasti lebih baik dari v1."),
        ],
        hasil="Bobot lunak (akar bobot seimbang) adalah bawaan semua model DL baru (AGENTS.md). Undersampling menyamakan jumlah "
              "window tiap kelas dengan membuang sebagian besar Normal dan AFIB, sehingga bobot kelas tidak dibutuhkan lagi.",
        paired=f"Undersampling lebih buruk di semua metrik utama: Macro-F1 {sg(pu[0])}, AFL F1 {sg(a.b('under', 'AFL F1')[0])}, "
               f"recall Normal {sg(a.b('under', 'Normal R')[0])} (semua P = 0,00). Dengan ±3 ribu window latih, model kehilangan "
               "hampir semua contoh ritme Normal yang cepat, padahal itu sumber alarm palsu terbesar.",
        pasien=f"Bawaan menangkap 04908 ({a.afl('bawaan', '04908')}) lebih baik dari v3.1 (04043 juga, tetapi pasien itu hanya "
               f"punya {d['afl_window']['04043']} window AFL), "
               f"tetapi 08378 (AFL 2:1, detak teratur cepat) turun ke {a.afl('bawaan', '08378')}. Undersampling kehilangan 08378 "
               f"hampir seluruhnya ({a.afl('under', '08378')}).",
        tanpa=f"Tanpa 04936, v3.1 tetap {fnum(a.tanpa('v31'))} dan BiGRU v3 bobot penuh {fnum(a.tanpa('v3_bobot_penuh'))}, "
              f"sedangkan bawaan turun ke {fnum(a.tanpa('bawaan'))}. Selisih v3.1 − bawaan {sg(a.tp('bawaan')[0])} "
              f"[{sg(a.tp('bawaan')[1], 2)}…{sg(a.tp('bawaan')[2], 2)}]: CI lebar, tetapi turunnya besar. Penyebab utamanya 08378, "
              "yang tanpa 04936 menjadi bagian terbesar AFL yang tersisa.",
        taki=f"Bawaan berhenti menebak Normal pada {a.amb('bawaan')} bpm (paket v3.1: {d['paket_v31']}), dan alarm pada Normal "
             f"≥ 120 bpm {pct(a.pita('bawaan'))} (v3.1 {pct(d['v31']['pita']['120–∞ bpm'][1])}). Undersampling sudah berhenti di "
             f"{a.amb('under')} bpm.",
        keputusan=[["Bobot kelas lunak sebagai bawaan BiGRU v11", "Dipakai", "Setara v3.1, recall Normal lebih tinggi"],
                   ["Undersampling data latih (arahan pembimbing)", "<b>Tidak disarankan</b>",
                    f"Lebih buruk di semua metrik (Δ Macro-F1 {sg(pu[0])}, P = {fnum(pu[3], 2)})"],
                   ["Memulihkan AFL 2:1 (08378) dan ketergantungan pada 04936", "Diteruskan", "Eksperimen 2: bobot kelas, "
                    "kapasitas, resep, oversampling"]],
        langkah=[("Bobot kelas di antara lunak dan penuh.", "weight_power 0,75 dan 1,0 untuk menguji trade-off 08378 vs recall Normal."),
                 ("Oversampling untuk model urutan.", "SMOTE tidak bisa dipakai pada urutan IBI; padanannya salinan acak window."),
                 ("Kapasitas dan resep.", "hidden 16/64, lr, epoch; dipilih lewat seleksi cross-fit, bukan dari tabel.")],
        batasan=["Satu kali training per (ulangan, fold). Eksperimen 2 kemudian menunjukkan training ini deterministik di mesin "
                 "yang sama (ulangan setelan bawaan identik bit demi bit).",
                 "BiGRU v3 dilatih di CPU mesin lain; perbandingannya tetap sah karena fold dan data uji sama."],
    )


def teks_e2(a):
    d = a.d
    cf = d["crossfit"]
    wp, lr = "wp100", "lr5e4"
    from collections import Counter
    lain2 = [s for s in d["baru"] + d["pembanding"] if s not in (wp, "under", "bawaan_ulang")]
    pilih = Counter(x for t in cf["terpilih"] for x in t).most_common()
    return dict(
        pertanyaan="Bisakah setelan lain memperbaiki kelemahan bawaan (AFL 2:1 di 08378, ketergantungan pada 04936), dan apakah "
                   "oversampling membantu model urutan?",
        cards=[(a.f(wp), "Macro-F1 bobot penuh (wp100)*"), (fnum(a.tanpa(wp)), "Tanpa 04936 (bawaan " + fnum(a.tanpa("bawaan"), 2) + ")"),
               (a.afl(wp, "08378"), "Recall AFL 08378 (bawaan " + a.afl("bawaan", "08378") + ")"),
               (sg(a.b("over")[0]), f"Oversampling vs bawaan (P = {fnum(a.b('over')[3], 2)})")],
        jawaban=f"<b>Tidak ada setelan yang lebih baik dari bawaan pada Macro-F1</b> (semua P &lt; 0,95). Yang paling berguna "
                f"adalah <b>bobot kelas penuh</b> (wp100): Macro-F1 {a.f(wp)} setara v3.1 (P = {fnum(a.v31(wp)[3], 2)}), 08378 "
                f"kembali terdeteksi ({a.afl('bawaan', '08378')} → {a.afl(wp, '08378')}), dan tanpa 04936 skornya {fnum(a.tanpa(wp))} "
                f"(bawaan {fnum(a.tanpa('bawaan'))}). <b>Oversampling dan undersampling lebih buruk</b> daripada pembobotan kelas.",
        temuan=[
            ("1. Training deterministik.", "bawaan_ulang identik bit demi bit dengan bawaan, jadi selisih antar setelan lokal "
             "bukan noise run. Resep yang sama di CPU (BiGRU v3) dan GPU (wp100) memberi Macro-F1 yang sama "
             f"(Δ {sg(d['banding']['wp100_vs_v3_bobot_penuh']['banding']['Macro-F1'][0])})."),
            ("2. Bobot penuh paling tahan uji.", f"Bootstrap bersarang vs v1 P = {fnum(a.bs(wp)['vs_v1']['p'], 2)} (bawaan "
             f"{fnum(d['bersarang_bawaan']['vs_v1']['p'], 2)}). "
             f"Harganya: recall Normal {fnum(a.m('bawaan', 'Normal R'))} → {fnum(a.m(wp, 'Normal R'))} dan alarm Normal "
             f"≥ 120 bpm {pct(a.pita('bawaan'), 0)} → {pct(a.pita(wp), 0)}."),
            ("3. lr 5e-4 tertinggi tapi rapuh.", f"Macro-F1 {a.f(lr)}, vs bawaan {a.bd(lr)}; tanpa 04936 jatuh ke "
             f"{fnum(a.tanpa(lr))}, sama dengan bawaan."),
            ("4. Lebih banyak = overfit.", f"50 epoch {a.bd('ep50')}, hidden 64 {a.bd('h64')}. hidden 16 ({thou(a.R['h16']['param'])} parameter) "
             f"setara bawaan ({a.bd('h16')})."),
            ("5. Balancing kalah dari pembobotan.", f"over {a.bd('over')}, under {a.bd('under')}. Seleksi cross-fit dari 8 "
             f"setelan juga tidak menolong ({fnum(cf['metrik']['pilihan']['Macro-F1'][0])})."),
        ],
        hasil="Delapan setelan baru, masing-masing 5 ulangan × 5 fold, dibandingkan dengan bawaan pada sampel bootstrap pasien "
              "yang sama. Tiga setelan (h16, h64, lr5e4) dilatih di Kaggle T4 karena GPU lokal penuh.",
        paired=f"Hanya tiga setelan yang berbeda secara konsisten dari bawaan, dan ketiganya lebih buruk: ep50 "
               f"{sg(a.b('ep50')[0])}, over {sg(a.b('over')[0])}, under {sg(a.b('under')[0])}. Bobot penuh dan lr 5e-4 menaikkan "
               f"AFL F1 ({sg(a.b(wp, 'AFL F1')[0])} dan {sg(a.b(lr, 'AFL F1')[0])}), tetapi belum pasti (P "
               f"{fnum(a.b(wp, 'AFL F1')[3], 2)} dan {fnum(a.b(lr, 'AFL F1')[3], 2)}). Semua setelan dengan bobot lebih besar "
               "menurunkan recall Normal sedikit (P = 0,00). Seleksi cross-fit memilih setelan yang berbeda-beda per fold: "
               + ", ".join(f"{k} {v}×" for k, v in pilih) + ".",
        pasien=f"Bobot penuh memulihkan 08378 ({a.afl(wp, '08378')}, {d['afl_window']['08378']} window) dan 06426 "
               f"({a.afl(wp, '06426')}). 04043 turun ke {a.afl(wp, '04043')}, tetapi pasien ini hanya punya "
               f"{d['afl_window']['04043']} window AFL, jadi angkanya hanya berarti window itu tertebak di "
               f"{round(a.R[wp]['afl']['04043'] * 5)} dari 5 ulangan. Oversampling menghafal ±{thou(a.R['bawaan']['n_latih'][2])} window AFL yang disalin "
               f"±{a.R['over']['n_latih'][2] / a.R['bawaan']['n_latih'][2]:.0f}×: 04908 turun ke "
               f"{a.afl('over', '04908')} dan AFL palsu naik ke {thou(round(a.R['over']['palsu']['total']))} per ulangan. "
               f"{pct(a.R['over']['biasstat']['di_batas'], 0)} nilai bias over menyentuh batas grid, tanda bahaya.",
        tanpa=f"Hanya bobot penuh yang tetap dekat v3.1 tanpa 04936: {fnum(a.tanpa(wp))} vs {fnum(a.tanpa('v31'))} "
              f"(v3.1 − wp100 {sg(a.tp(wp)[0])}, P = {fnum(a.tp(wp)[3], 2)}). Setelan lain, termasuk lr5e4 yang tertinggi pada "
              f"data penuh, berada di {fnum(min(a.tanpa(s) for s in lain2))}–{fnum(max(a.tanpa(s) for s in lain2))} "
              f"(under {fnum(a.tanpa('under'))}). Makin besar bobot AFL, makin kecil ketergantungan pada 04936: bobot 0,5 → 0,75 "
              f"→ 1,0 memberi {fnum(a.tanpa('bawaan'))} → {fnum(a.tanpa('wp075'))} → {fnum(a.tanpa(wp))}.",
        taki=f"Ambang takikardia hanya bergeser 115–125 bpm di semua setelan. Bobot penuh: {a.amb(wp)} bpm, alarm Normal "
             f"≥ 120 bpm {pct(a.pita(wp))} (v3.1 {pct(d['v31']['pita']['120–∞ bpm'][1])}). hidden 64 punya ambang tertinggi "
             f"({a.amb('h64')}). Masalah sinus takikardia bukan soal tuning.",
        keputusan=[["Bobot penuh (weight_power 1,0) bila BiGRU jadi kandidat", "<b>Disarankan</b>",
                    "Satu-satunya yang tahan uji tanpa 04936; alasan tercatat untuk menyimpang dari bobot lunak"],
                   ["Oversampling / undersampling untuk BiGRU", "<b>Tidak disarankan</b>", "Keduanya lebih buruk (P = 0,00)"],
                   ["Model lebih kecil (hidden 16)", "Boleh", f"Setara bawaan dengan "
                    f"{fnum(a.R['bawaan']['param'] / a.R['h16']['param'], 1)}× lebih sedikit parameter"],
                   ["Epoch lebih banyak / hidden lebih besar", "Tidak disarankan", "Lebih buruk atau tidak lebih baik"]],
        langkah=[("Regularisasi di atas bobot penuh.", "Epoch 20, weight decay 1e-3/1e-2, batch 512, hidden 16, lr 5e-4 "
                  "(eksperimen 3), lalu seleksi cross-fit."),
                 ("Dropout.", "Butuh perubahan model.py dengan nilai bawaan yang menjaga hasil lama identik.")],
        batasan=["wp100 dipilih setelah melihat hasil. Resepnya sudah ada sejak v3, tetapi pada data penuh ia tetap tidak "
                 f"terbedakan dari bawaan (P = {fnum(a.b(wp)[3], 2)}).",
                 "h16, h64, dan lr5e4 dilatih di GPU lain (T4). Determinisme di sana tidak diperiksa."],
    )


def teks_e3(a):
    d = a.d
    cf = d["crossfit"]
    wp, h16, wd3, wd2 = "wp100", "w1_h16", "w1_wd1e3", "w1_wd1e2"
    from collections import Counter
    baru = d["baru"]
    tanpa_h16 = [s for s in baru if s != h16]
    pilih = Counter(x for t in cf["terpilih"] for x in t).most_common()
    rng = lambda xs, nd=3: f"{fnum(min(xs), nd)}–{fnum(max(xs), nd)}"  # noqa: E731
    dmf = [a.b(s)[0] for s in tanpa_h16]
    terbaik = max(baru, key=lambda s: a.m(s))
    return dict(
        pertanyaan="Apakah regularisasi di atas bobot kelas penuh (epoch lebih sedikit, weight decay, batch lebih besar, model "
                   "lebih kecil, lr lebih kecil) memperbaiki BiGRU?",
        cards=[(a.f(terbaik), f"Macro-F1 tertinggi ({terbaik})*"), (sg(a.b(terbaik)[0]), f"{terbaik} vs wp100 (P = "
               f"{fnum(a.b(terbaik)[3], 2)})"), (fnum(cf["metrik"]["pilihan"]["Macro-F1"][0]), "Seleksi cross-fit 7 setelan"),
               (a.f(h16), f"hidden 16 + bobot penuh (P = {fnum(a.b(h16)[3], 2)})")],
        jawaban=f"<b>Tidak.</b> Tidak ada setelan yang lebih baik dari bobot penuh (wp100). Lima setelan berada dalam "
                f"{sg(min(dmf))} … {sg(max(dmf))} Macro-F1 dari wp100 (semua tidak terbedakan), dan hidden 16 <b>lebih buruk</b> "
                f"({sg(a.b(h16)[0])}, P = {fnum(a.b(h16)[3], 2)}). Seleksi cross-fit dari ketujuh setelan memberi "
                f"{fnum(cf['metrik']['pilihan']['Macro-F1'][0])}, sama dengan wp100 ({a.f(wp)}) dan v3.1 (P = "
                f"{fnum(cf['banding_v31']['Macro-F1'][3], 2)}). Resep bobot penuh sudah di titik jenuh: tuning BiGRU di ruang ini selesai.",
        temuan=[
            ("1. Titik jenuh.", f"weight decay 1e-3 {a.bd(wd3)}, 1e-2 {a.bd(wd2)}, batch 512 {a.bd('w1_bs512')}, 20 epoch "
             f"{a.bd('w1_ep20')}, lr 5e-4 {a.bd('w1_lr5e4')}."),
            ("2. Keunggulan bobot penuh tetap.", f"Tanpa 04936 semua setelan di {rng([a.tanpa(s) for s in tanpa_h16])} (v3.1 "
             f"{fnum(a.tanpa('v31'))}, bawaan {fnum(a.tanpa('bawaan'))}). Bootstrap bersarang vs v1: "
             + ", ".join(f"{s} P = {fnum(a.bs(s)['vs_v1']['p'], 2)}" for s in d["bersarang"]) + "."),
            ("3. hidden 16 + bobot penuh rusak.", f"AFL F1 {sg(a.b(h16, 'AFL F1')[0])} (P = {fnum(a.b(h16, 'AFL F1')[3], 2)}), "
             f"08378 {a.afl(h16, '08378')}. Di eksperimen 2, hidden 16 dengan bobot lunak setara bawaan: model kecil tidak "
             "cocok dengan bobot AFL yang besar (penyebab pastinya belum diuji)."),
            ("4. Alarm pada detak cepat tidak membaik.", f"Recall Normal semua setelan sedikit di bawah wp100; alarm Normal "
             f"≥ 120 bpm {pct(min(a.pita(s) for s in baru), 0)}–{pct(max(a.pita(s) for s in baru), 0)} (wp100 {pct(a.pita(wp), 0)}). "
             f"weight decay 1e-2 sedikit terbaik (ambang {a.amb(wd2)})."),
            ("5. Tidak ada pemenang stabil.", "Seleksi cross-fit memilih " + ", ".join(f"{k} {v}×" for k, v in pilih)
             + " dari 25 fold."),
        ],
        hasil="Enam setelan di atas bobot kelas penuh (wp100 dari eksperimen 2), masing-masing 5 ulangan × 5 fold, dibandingkan "
              "berpasangan dengan wp100. Arahnya regularisasi, karena di eksperimen 2 setelan berkapasitas atau berlatih lebih "
              "banyak justru lebih buruk. Tiga setelan dilatih di Kaggle T4; bawaan ikut sebagai pembanding.",
        paired=f"Hanya hidden 16 yang berbeda secara konsisten dari wp100, dan lebih buruk ({sg(a.b(h16)[0])}). Weight decay "
               f"1e-3 dan 1e-2 sedikit di atas wp100 ({sg(a.b(wd3)[0])} dan {sg(a.b(wd2)[0])}, P {fnum(a.b(wd3)[3], 2)} dan "
               f"{fnum(a.b(wd2)[3], 2)}): kecil, dan keduanya dilatih di mesin lain (Kaggle T4). Semua setelan menurunkan recall "
               f"Normal sedikit ({sg(min(a.b(s, 'Normal R')[0] for s in baru))} … {sg(max(a.b(s, 'Normal R')[0] for s in baru))}). "
               f"Bawaan vs wp100: {a.bd('bawaan')}, recall Normal {sg(a.b('bawaan', 'Normal R')[0])}.",
        pasien=f"Semua setelan kecuali hidden 16 tetap menangkap 08378 ({rng([a.R[s]['afl']['08378'] for s in tanpa_h16], 2)}). Yang "
               f"berubah antar setelan terutama 06426 ({rng([a.R[s]['afl']['06426'] for s in baru], 2)}) dan 04908 "
               f"({rng([a.R[s]['afl']['04908'] for s in baru], 2)}). AFL palsu "
               f"{thou(round(min(a.R[s]['palsu']['total'] for s in baru)))}–{thou(round(max(a.R[s]['palsu']['total'] for s in baru)))} "
               f"per ulangan (wp100 {thou(round(a.R[wp]['palsu']['total']))}). Tidak ada bias di batas grid.",
        tanpa=f"Ketergantungan pada 04936 tetap kecil untuk semua setelan berbobot penuh: {rng([a.tanpa(s) for s in tanpa_h16])} "
              f"tanpa 04936, dekat v3.1 ({fnum(a.tanpa('v31'))}) dan wp100 ({fnum(a.tanpa(wp))}). Hidden 16 turun ke "
              f"{fnum(a.tanpa(h16))}, bawaan {fnum(a.tanpa('bawaan'))}.",
        taki=f"Ambang hampir sama di semua setelan ({a.amb(wp)} untuk wp100). weight decay 1e-2 sedikit lebih baik ({a.amb(wd2)}, "
             f"alarm Normal ≥ 120 bpm {pct(a.pita(wd2))}); lr 5e-4 dan hidden 16 lebih buruk ({pct(a.pita('w1_lr5e4'))} dan "
             f"{pct(a.pita(h16))}). Bawaan dengan bobot lunak tetap paling jarang alarm ({pct(a.pita('bawaan'))}). Tidak ada yang "
             "mendekati penyelesaian masalah sinus takikardia.",
        keputusan=[["Resep BiGRU kandidat: weight_power 1,0 (wp100)", "<b>Dipertahankan</b>",
                    "Tidak ada regularisasi yang terbedakan; resep paling sederhana dipakai"],
                   ["Hentikan tuning hiperparameter BiGRU", "<b>Disarankan</b>",
                    f"Seleksi cross-fit = wp100 = v3.1; sisa selisih {sg(min(dmf))} … {sg(max(dmf))}"],
                   ["hidden 16 dengan bobot penuh", "Tidak disarankan", f"{sg(a.b(h16)[0])} (P = {fnum(a.b(h16)[3], 2)})"],
                   ["Dropout / GRU 2 lapis", "Tidak diprioritaskan", "Regularisasi lain tidak memberi apa-apa"]],
        langkah=[("Pakai wp100 sebagai hasil BiGRU di gabungan.ipynb.", "Untuk perbandingan antar model tim v11."),
                 ("Perbaikan berikutnya butuh data, bukan tuning.", "Sesuai AGENTS.md: pasien baru atau rekaman perangkat "
                  "SIHEDAF berlabel. Alarm pada sinus takikardia tidak bisa diselesaikan dari interval RR AFDB.")],
        batasan=["Tiga setelan dilatih di Kaggle T4 dan dibandingkan dengan wp100 dari GPU lokal. Selisih sekecil ±0,003 bisa "
                 "berasal dari beda mesin.",
                 "Seleksi cross-fit hanya di antara tujuh setelan berbobot penuh (bawaan tidak ikut)."],
    )


TEKS = {"e1": teks_e1, "e2": teks_e2, "e3": teks_e3}


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
    R, ref = d["R"], d["referensi"]
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
    c.setTitle(f"BiGRU v11 eksperimen {no}: {d['judul']}")
    c.setAuthor("v11_model_tim/bigru (pemilik repo)")
    c.setSubject(t["pertanyaan"])
    W_, H = A4
    M = 40
    CW = W_ - 2 * M
    state = {"page": 0}

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
        c.drawString(M + 43, H - 42, f"SIHEDAF  /  V11 BIGRU  /  EKSPERIMEN {no}")
        fs = 20
        while pdfmetrics.stringWidth(title, "Arial-Bold", fs) > CW and fs > 13:
            fs -= .5
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", fs); c.drawString(M, H - 78, title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(M, H - 97, kicker)
        c.setStrokeColor(HexColor("#D8E1E9")); c.line(M, 43, W_ - M, 43)
        c.setFont("Arial", 8)
        c.drawString(M, 29, f"{d['tanggal']} | Sumber: v11_model_tim/bigru/eksperimen.py, cache/v11/bigru/{d['file_banding']}")
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

    v31, v1 = d["v31"]["metrik"], d["v1"]["metrik"]
    w_afl = d["n_windows"] / (3 * d["class_counts"][2])         # class_weights seimbang untuk AFL
    semua = d["semua"]
    lain = [s for s in semua if s != ref]

    # ── Hal 1: ringkasan ──────────────────────────────────────────────
    y = page(f"Eksperimen {no}: {d['judul']}",
             f"BiGRU + attention (v11, pemilik repo) | 5 ulangan × 5 fold pasien | data detektor 50 Hz | {d['tanggal']}")
    cardw = (CW - 24) / 4
    for i, (val, lab) in enumerate(t["cards"]):
        x = M + i * (cardw + 8)
        c.setFillColor(HexColor("#EDF3F8")); c.roundRect(x, y - 84, cardw, 84, 6, fill=1, stroke=0)
        c.setFillColor(HexColor(TEAL)); c.setFont("Arial-Bold", 23); c.drawString(x + 11, y - 36, val)
        p = Paragraph(lab, ParagraphStyle("kartu", fontName="Arial", fontSize=8.3, leading=10, textColor=HexColor(GRAY)))
        p.wrap(cardw - 20, 40); p.drawOn(c, x + 11, y - 52 - p.height + 10)
    y -= 100
    y = para(f"*Semua angka: BiGRU + bias kelas, prediksi out-of-fold rata-rata 5 ulangan × 5 fold pasien, data detektor 50 Hz, "
             f"CI 95% bootstrap pasien 2.000×. Pertanyaan: {t['pertanyaan']}", y, "small")
    y = callout("Jawaban singkat", t["jawaban"], y)
    y = section("Temuan utama", y)
    for judul, isi in t["temuan"]:
        y = para(f"<b>{judul}</b> {isi}", y, gap=6)
    para("Halaman: perbandingan dengan versi sebelumnya (2), metode (3), hasil (4), berpasangan (5), per pasien (6), tanpa "
         "04936 (7), uji takikardia (8), kesimpulan &amp; batasan (9), lampiran teknis (10).", y - 2, "small")

    # ── Hal 2: perbandingan dengan versi sebelumnya ───────────────────
    y = page("Perbandingan dengan versi sebelumnya", "Data uji, fold pasien, dan sampel bootstrap yang sama | + bias kelas cross-fit")
    v3 = d["v3"]
    rows = [["v1 (versi lama, argmax)", ci(v1["Macro-F1"]), ci(v1["AFL F1"]), fnum(v1["Normal R"][0]), "–", "–"],
            ["<b>v3.1 (produksi)</b>", ci(v31["Macro-F1"]), ci(v31["AFL F1"]), fnum(v31["Normal R"][0]), "–",
             f"{sg(d['v31_vs_v1']['Macro-F1'][0])} (P = {fnum(d['v31_vs_v1']['Macro-F1'][3], 2)})"],
            ["BiGRU v3, bobot penuh (CPU, mesin lain)", ci(v3["metrik"]["Macro-F1"]), ci(v3["metrik"]["AFL F1"]),
             fnum(v3["metrik"]["Normal R"][0]), f"{sg(v3['vs_v31']['Macro-F1'][0])} (P = {fnum(v3['vs_v31']['Macro-F1'][3], 2)})",
             f"{sg(v3['vs_v1']['Macro-F1'][0])} (P = {fnum(v3['vs_v1']['Macro-F1'][3], 2)})"]]
    for s in d["kunci"]:
        rows.append([f"BiGRU v11 <b>{s}</b>", ci(R[s]["bias"]["Macro-F1"]), ci(R[s]["bias"]["AFL F1"]),
                     fnum(R[s]["bias"]["Normal R"][0]), f"{sg(R[s]['v31']['Macro-F1'][0])} (P = {fnum(R[s]['v31']['Macro-F1'][3], 2)})",
                     f"{sg(R[s]['v1']['Macro-F1'][0])} (P = {fnum(R[s]['v1']['Macro-F1'][3], 2)})"])
    y = table(["Sistem", "Macro-F1 [CI]", "AFL F1 [CI]", "Normal R", "Δ Macro-F1 vs v3.1", "Δ Macro-F1 vs v1"], rows, y,
              widths=[140, 82, 82, 50, 83, CW - 437], bold_rows=[2])
    y = chart("forest", y, 270)
    y = para("Belah ketupat hijau = v3.1, abu-abu = v1. Kotak oranye = referensi eksperimen ini, lingkaran biru = setelan lain. "
             "Garis putus-putus = referensi. Garis horizontal = CI 95%. Lebar CI AFL F1 berasal dari hanya 7 pasien AFL.", y, "small")
    bb = [s for s in d["bersarang"]]
    if bb:
        rows = [[s, f"{fnum(d['bersarang'][s]['kandidat']['titik'])} <font color='{GRAY}' size='7.2'>"
                    f"[{fnum(d['bersarang'][s]['kandidat']['ci'][0], 2)}–{fnum(d['bersarang'][s]['kandidat']['ci'][1], 2)}]</font>",
                 f"{sg(d['bersarang'][s]['vs_v1']['delta'])} <font color='{GRAY}' size='7.2'>[{sg(d['bersarang'][s]['vs_v1']['ci'][0], 2)}…"
                 f"{sg(d['bersarang'][s]['vs_v1']['ci'][1], 2)}]</font>", fnum(d["bersarang"][s]["vs_v1"]["p"], 2),
                 vonis(d["bersarang"][s]["vs_v1"]["p"])] for s in bb]
        y = section("Bootstrap bersarang vs v1 (bias dipilih ulang di setiap sampel; uji paling ketat)", y)
        table(["Setelan", "Macro-F1 [CI]", "Δ vs v1 [CI]", "P", "Vonis"], rows, y, widths=[110, 110, 120, 60, CW - 400])

    # ── Hal 3: metode ─────────────────────────────────────────────────
    y = page("Metode", "Satu model (BiGRU + attention), satu protokol; hanya setelan training yang berubah")
    y = para(t["hasil"], y)
    rows = []
    for s in semua:
        n = R[s]["n_latih"]
        rows.append([f"<b>{s}</b>" if s == ref else s, kode(s), R[s]["tempat"], thou(R[s]["param"]),
                     " / ".join(thou(x) for x in n), f"{R[s]['waktu']:.0f} s"])
    y = table(["Setelan", "SETELAN (selain bawaan)", "Dilatih di", "Param.", "Window latih per kelas (r0 k0)", "s / fold"],
              rows, y, widths=[72, 167, 68, 46, 112, CW - 465], bold_rows=[1])
    y = para(f"Bawaan: hidden 32, 30 epoch, batch 256, AdamW lr 1e-3, weight decay 1e-4, cosine, cross-entropy dengan bobot kelas "
             f"<i>class_weights(y) ** weight_power</i> (bawaan 0,5 = lunak). Data: {thou(d['n_windows'])} window 10 detik dari "
             f"{d['n_pasien']} pasien AFDB (Normal {thou(d['class_counts'][0])}, AFIB {thou(d['class_counts'][1])}, AFL "
             f"{thou(d['class_counts'][2])}), R-peak dari detektor 50 Hz."
             + (" Waktu per fold tidak sebanding antar baris: proses berbagi GPU dan sebagian dilatih di Kaggle."
                if d["kaggle"] else ""), y, "small")
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
        ["Balancing", "Menyamakan jumlah window tiap kelas di data latih: under = membuang, over = menyalin acak. Hanya di "
                      "data latih tiap fold; data uji tetap asli."],
        ["Bias kelas", "Penggeser ambang keputusan setelah training. Negatif untuk AFL = model harus lebih yakin sebelum menyebut AFL."],
        ["CI 95%", "Rentang tempat skor sebenarnya kemungkinan besar berada."],
        ["P(Δ > 0)", "Peluang setelan lebih baik dari pembanding pada sampel pasien yang sama. ≥ 0,95 lebih baik; ≤ 0,05 lebih "
                     "buruk; selebihnya tidak terbedakan."],
        ["Seleksi cross-fit", "Setelan untuk tiap fold uji dipilih dari hasil fold lain, jadi tuning tidak mengintip data uji."],
    ], y, widths=[100, CW - 100])

    # ── Hal 4: hasil ──────────────────────────────────────────────────
    y = page("Hasil semua setelan", "+ bias kelas cross-fit | rata-rata 5 ulangan, CI 95% bootstrap pasien | baris hijau = referensi")
    rows = [["v3.1 (produksi)", ci(v31["Macro-F1"]), ci(v31["AFL F1"]), fnum(v31["AFL P"][0]), ci(v31["AFL R"]),
             fnum(v31["AFIB F1"][0]), fnum(v31["Normal R"][0]), "–"]]
    for s in semua:
        m = R[s]["bias"]
        rows.append([f"<b>{s}</b>" if s == ref else s, ci(m["Macro-F1"]), ci(m["AFL F1"]), fnum(m["AFL P"][0]), ci(m["AFL R"]),
                     fnum(m["AFIB F1"][0]), fnum(m["Normal R"][0]), fnum(R[s]["argmax"]["Macro-F1"][0])])
    y = table(["Setelan", "Macro-F1", "AFL F1", "AFL P", "AFL R", "AFIB F1", "Normal R", "Macro-F1 argmax"], rows, y,
              widths=[70, 80, 80, 44, 80, 46, 46, CW - 446], bold_rows=[2])
    cf = d.get("crossfit")
    if cf:
        m = cf["metrik"]["pilihan"]
        b = cf["banding_v31"]["Macro-F1"]
        y = callout("Seleksi cross-fit (tuning tanpa mengintip data uji)",
                    f"Dari {len(cf['kandidat'])} setelan ({', '.join(cf['kandidat'])}), setelan untuk tiap fold uji dipilih dari "
                    f"fold lain. Hasilnya Macro-F1 {fnum(m['Macro-F1'][0])} [{fnum(m['Macro-F1'][1], 2)}–{fnum(m['Macro-F1'][2], 2)}], "
                    f"AFL F1 {fnum(m['AFL F1'][0])}; vs v3.1 {sg(b[0])} (P = {fnum(b[3], 2)}). Ini perkiraan jujur untuk "
                    "\"BiGRU yang di-tuning\": lebih rendah dari setelan terbaik di tabel, karena pilihan terbaik di fold lain "
                    "tidak selalu terbaik di fold uji.", y)
    y = callout("Cara membaca", "Angka tertinggi di tabel tidak otomatis lebih baik: selisih antar setelan harus diuji "
                "berpasangan (halaman berikutnya). Macro-F1 argmax = tanpa bias kelas; selisihnya dengan + bias menunjukkan "
                "seberapa besar koreksi ambang yang dibutuhkan.", y)

    # ── Hal 5: berpasangan ────────────────────────────────────────────
    y = page("Perbandingan berpasangan", f"Setiap setelan dibandingkan dengan {ref} pada sampel bootstrap pasien yang sama")
    y = chart("paired", y, 240)
    rows = []
    for s in lain:
        b = a.b(s)
        rows.append([s, dci(b), fnum(b[3], 2), vonis(b[3], b), f"{sg(a.b(s, 'AFL F1')[0])} ({fnum(a.b(s, 'AFL F1')[3], 2)})",
                     f"{sg(a.b(s, 'AFL R')[0])} ({fnum(a.b(s, 'AFL R')[3], 2)})",
                     f"{sg(a.b(s, 'Normal R')[0])} ({fnum(a.b(s, 'Normal R')[3], 2)})"])
    y = table([f"Setelan vs {ref}", "Δ Macro-F1 [CI]", "P", "Vonis", "Δ AFL F1 (P)", "Δ AFL R (P)", "Δ Normal R (P)"], rows, y,
              widths=[70, 105, 34, 76, 75, 75, CW - 435])
    callout("Intinya", t["paired"], y)

    # ── Hal 6: per pasien ─────────────────────────────────────────────
    y = page("AFL per pasien dan AFL palsu", "Recall AFL (+ bias) tiap pasien AFL, rata-rata 5 ulangan | asal AFL palsu")
    y = chart("afl_heatmap", y, 225)
    rows = [["v3.1"] + [thou(round(d["v31"]["palsu"][k])) for k in ("total", "dari_normal", "dari_afib", "pasien_tanpa_afl")]
            + ["–", "–"]]
    for s in semua:
        fp, bi = R[s]["palsu"], R[s]["biasstat"]
        rows.append([f"<b>{s}</b>" if s == ref else s] + [thou(round(fp[k])) for k in ("total", "dari_normal", "dari_afib")]
                    + [thou(round(fp["pasien_tanpa_afl"])), f"{fnum(bi['median'][2], 2)} [{fnum(bi['min'][2], 2)}…{fnum(bi['maks'][2], 2)}]",
                     pct(bi["di_batas"], 0)])
    y = table(["Sistem", "AFL palsu / ulangan", "dari Normal", "dari AFIB", "di pasien tanpa AFL", "Bias AFL median [min…maks]",
               "Bias di batas grid"], rows, y, widths=[70, 70, 58, 58, 72, 105, CW - 433], bold_rows=[2])
    callout("Intinya", t["pasien"], y)

    # ── Hal 7: tanpa 04936 ────────────────────────────────────────────
    tn = d["tanpa"]
    y = page("Ketergantungan pada pasien 04936",
             f"{pct(d['afl_window']['04936'] / d['class_counts'][2], 0)} window AFL berasal dari 04936 | tanpa pasien ini: {tn['n_pasien']} pasien, {tn['n_pasien_afl']} pasien AFL, "
             f"{thou(tn['n_afl'])} window AFL")
    y = chart("tanpa", y, 250)
    rows = [["v3.1", ci(tn["rows"]["v31"]["Macro-F1"]), fnum(tn["rows"]["v31"]["AFL F1"][0]), "–", "–"]]
    for s in ["v3_bobot_penuh"] + semua:
        nm = "BiGRU v3 bobot penuh" if s == "v3_bobot_penuh" else (f"<b>{s}</b>" if s == ref else s)
        x = tn["pairs_v31_minus"][s]
        rows.append([nm, ci(tn["rows"][s]["Macro-F1"]), fnum(tn["rows"][s]["AFL F1"][0]), dci(x), fnum(x[3], 2)])
    y = table(["Sistem", "Macro-F1 tanpa 04936", "AFL F1", "v3.1 − sistem [CI]", "P(v3.1 lebih baik)"], rows, y,
              widths=[120, 105, 60, 120, CW - 405], bold_rows=[3])
    callout("Intinya", t["tanpa"], y)

    # ── Hal 8: uji takikardia ─────────────────────────────────────────
    y = page("Uji stres takikardia", "Protokol butir 9: ritme teratur sintetis dan window Normal asli yang cepat")
    y = chart("takikardia", y, 190)
    pk = list(d["v31"]["pita"])
    rows = [["v3.1 (paket)", f"{d['paket_v31']}"] + [pct(d["v31"]["pita"][k][1]) for k in pk]]
    rows.append(["kontrol v7 (per fold)", " / ".join(str(v) for v in d["kontrol_v7"]["ambang"].values())] + ["–"] * len(pk))
    for s in semua:
        rows.append([f"<b>{s}</b>" if s == ref else s, a.amb(s)] + [pct(R[s]["pita"][k][1]) for k in pk])
    y = table(["Sistem", "Ambang (bpm): tanpa / ±10 / G10 / G20"] + [f"Normal {k}" for k in pk], rows, y,
              widths=[90, 130] + [(CW - 220) / len(pk)] * len(pk), bold_rows=[3])
    y = para("Ambang = detak terendah di mana kurang dari separuh window sintetis ditebak Normal (makin tinggi makin baik). "
             "Kolom Normal = % window Normal asli (OOF) pada pita detak itu yang ditebak AFIB/AFL (makin rendah makin baik). "
             f"AFDB hanya punya {thou(d['v31']['pita']['120–∞ bpm'][0])} window Normal ≥ 120 bpm.", y, "small")
    callout("Intinya", t["taki"], y, "#FFF4EC")

    # ── Hal 9: kesimpulan & batasan ───────────────────────────────────
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
            "Bias kelas dipilih cross-fit di data yang sama. Bootstrap bersarang memperhitungkan ketidakpastiannya."]:
        y = para(f"• {txt}", y, gap=5)

    # ── Hal 10: lampiran ──────────────────────────────────────────────
    y = page("Lampiran teknis", "Lingkungan, file, dan cara mereproduksi")
    ln = R[ref]["lingkungan"]
    pk_ = ", ".join(f"{k} {v}" for k, v in ln["pustaka"].items() if v)
    kg = [s for s in semua if R[s]["tempat"] == KAGGLE]
    y = table(["Komponen", "Nilai"], [
        ["Data", f"cache/dataset_v2_detector.npz: {thou(d['n_windows'])} window, {d['n_pasien']} pasien (dicek sidik_jari.json)"],
        ["Pembagian", "cache/oof_robust/detector_e30_r5_s2026/folds.npz: 5 ulangan × 5 fold pasien, seed 2026"],
        ["Pembanding", "v3.1 = MicroTransformer + bias (OOF v3); v1 = OOF versi lama, argmax; BiGRU v3 = BiGRUAttention_r*.npz"],
        ["Bias & bootstrap", "grid −7…+3 langkah 0,25, cross-fit antar fold; 2.000 sampel pasien per strata"],
        ["Training lokal", f"{PC}; {pk_}"],
        ["Training Kaggle", ("GPU T4 ×2, torch 2.11.0+cu128, tiga proses; kode v11 dibawa di dalam script kernel, data dari "
                             "GitHub Release v11-data (dicek sidik jari). Setelan: " + ", ".join(kg)) if kg else "–"],
        ["Kode", f"commit {ln['commit']} + perubahan v11 yang belum di-commit; tes v11 lolos"],
        ["Pemeriksaan otomatis", "Macro-F1 (+ bias & argmax) tiap setelan dihitung ulang dari OOF di cache dan harus sama dengan "
                                 "ringkasan.json; selisih berpasangan harus konsisten dengan ringkasan"],
    ], y, widths=[100, CW - 100])
    y = section("File", y)
    y = table(["File", "Isi"], [
        ["v11_model_tim/bigru/eksperimen.py", "Definisi eksperimen, training (runner), evaluasi (bersama.evaluasi)"],
        ["v11_model_tim/bigru/model.py", "BiGRU + attention (diimpor dari v2_revisi/sihedaf_models.py)"],
        ["v11_model_tim/bersama/", "Data + sidik jari, loop DL identik v3, runner 5×5, balancing per fold, evaluasi"],
        ["cache/v11/bigru/&lt;setelan&gt;/", "Bobot (.pt) dan prediksi (.npz) per fold, ringkasan.json, setelan.json"],
        [f"cache/v11/bigru/{d['file_banding']}", "Perbandingan berpasangan, tanpa 04936, seleksi cross-fit, bootstrap bersarang"],
        ["v11_model_tim/bigru/laporan/build_report.py", "Pembuat laporan ini"],
    ], y, widths=[200, CW - 200])
    y = section("Mereproduksi (dari folder v11_model_tim/bigru)", y)
    para(f"Training: <font name='Courier' size='8.3'>python eksperimen.py latih {e}</font> (bisa dilanjutkan; satu setelan: tambahkan "
         f"namanya). Evaluasi: <font name='Courier' size='8.3'>python eksperimen.py evaluasi {e}</font>. Laporan: "
         f"<font name='Courier' size='8.3'>python laporan/build_report.py --extract {e}</font>, lalu perintah yang sama tanpa "
         "--extract. Satu setelan juga bisa dilatih lewat jalankan.ipynb (ubah SETELAN dan NAMA_SETELAN).", y)

    c.save()
    print(f"OK {out} ({state['page']} halaman)")
    return out


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if "--extract" in sys.argv:
        sys.path[:0] = [str(BIGRU.parent), str(BIGRU)]
        from eksperimen import EKSPERIMEN
        from bersama import data
        akar = data.lokasi_hasil() / "bigru"
        for e in args or [k for k in EKSPERIMEN if (akar / EKSPERIMEN[k]["banding"]).exists()]:
            extract(e)
    elif "--grafik" in sys.argv:                          # gambar ulang dari _build/<e>.json saja
        for e in args or sorted(p.stem for p in TMP.glob("e?.json")):
            grafik(e)
    else:
        for e in args or sorted(p.stem for p in TMP.glob("e?.json")):
            build(e)
