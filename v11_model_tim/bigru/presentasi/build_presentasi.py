"""Halaman presentasi BiGRU v11 untuk pembimbing: apa saja yang sudah dicoba, dalam bahasa sederhana.

Semua angka dibaca dari `../laporan/_build/e1.json`–`e3.json` (keluaran `laporan/build_report.py --extract`, yang sudah
di-assert terhadap OOF dan `ringkasan.json`) dan dari `cache/v11/xgboost/banding_balancing.json`. Bagian "Dari v1 ke v3.1"
membaca `laporan/_build/tables.json` milik v3, v3.1, dan v3.2 (jalankan `build_report.py --extract` versi itu bila belum
ada) serta output sel `v1_lama/model.ipynb`. Bagian "Cara sampling" membaca dataset dan fold v11 (`bersama.data`) serta
jumlah window latih per fold yang dicatat runner (`info["n_latih"]` di `cache/v11/bigru/<setelan>/r*_k*.npz`). Tidak ada
angka hasil yang diketik tangan. Grafik digambar sebagai SVG di dalam halaman, jadi halaman bisa dibuka tanpa internet
(font Google jatuh ke Arial/Liberation Sans).

Pakai (cwd bebas):
    python build_presentasi.py                  # -> presentasi_bigru.html (halaman utuh, bisa dibuka offline)
    python build_presentasi.py --artifact OUT   # -> potongan halaman tanpa <html>/<head>, untuk dipublikasikan
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "laporan" / "_build"
REPO = HERE.parents[2]
CACHE = REPO / "cache" / "v11"
sys.path.insert(0, str(HERE.parents[1]))                     # v11_model_tim
import numpy as np  # noqa: E402
from bersama import data as D, evaluasi as EV  # noqa: E402

EKS = ("e1", "e2", "e3")
E = {e: json.loads((BUILD / f"{e}.json").read_text()) for e in EKS}
E1 = E["e1"]
PASIEN_AFL = list(E1["afl_window"])
PITA = list(E1["v31"]["pita"])

# Penjelasan setelan untuk pembaca (teks saja; nilai parameternya diambil dari setelan.json lewat _build)
KET = {
    "bawaan": "Bobot kelas lunak, resep v3",
    "under": "Undersampling ke jumlah AFL",
    "bawaan_ulang": "Bawaan diulang persis (cek noise)",
    "wp075": "Bobot kelas ¾",
    "wp100": "Bobot kelas penuh",
    "over": "Oversampling sampai sama dengan Normal",
    "h16": "Model lebih kecil",
    "h64": "Model lebih besar",
    "lr5e4": "Learning rate setengah",
    "ep50": "Latih lebih lama",
    "w1_h16": "Bobot penuh + model kecil",
    "w1_ep20": "Bobot penuh + latih lebih singkat",
    "w1_bs512": "Bobot penuh + batch lebih besar",
    "w1_wd1e3": "Bobot penuh + weight decay 10×",
    "w1_wd1e2": "Bobot penuh + weight decay 100×",
    "w1_lr5e4": "Bobot penuh + learning rate setengah",
}
KANDIDAT = "wp100"
HL = ' class="hl"'
REF_EKS = {"e1": "bawaan", "e2": "bawaan", "e3": "wp100"}


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def gabung():
    """Satukan hasil per setelan dari tiga eksperimen; setelan yang muncul di beberapa file harus identik."""
    R, asal, tanpa = {}, {}, {}
    for e in EKS:
        for s, r in E[e]["R"].items():
            if s in R:
                assert R[s]["bias"] == r["bias"] and R[s]["afl"] == r["afl"], f"{s} beda antara eksperimen"
            else:
                R[s] = r
        for s in E[e]["baru"]:
            asal[s] = e
        for s, row in E[e]["tanpa"]["rows"].items():
            if s in tanpa:
                assert tanpa[s] == row, f"tanpa 04936 {s} beda antara eksperimen"
            tanpa[s] = row
    assert set(R) == set(asal) == set(KET), (sorted(R), sorted(KET))
    return R, asal, tanpa


R, ASAL, TANPA = gabung()
SETELAN = [s for e in EKS for s in E[e]["baru"]]
N_FOLD = {s: len([p for p in (CACHE / "bigru" / s).glob("r[0-9]_k[0-9].npz")]) for s in SETELAN}
assert set(N_FOLD.values()) == {25}, N_FOLD
assert sum(E1["class_counts"]) == E1["n_windows"] and sum(E1["afl_window"].values()) == E1["class_counts"][2]


def banding_ref(s, k="Macro-F1"):
    """Selisih [Δ, lo, hi, P] terhadap referensi eksperimennya (e1/e2: bawaan, e3: wp100); None untuk referensinya sendiri."""
    e = ASAL[s]
    key = f"{s}_vs_{REF_EKS[e]}"
    return E[e]["banding"][key]["banding"][k] if key in E[e]["banding"] else None


def v31_vs(s, k="Macro-F1"):
    return R[s]["v31"][k]


XGB = json.loads((CACHE / "xgboost" / "banding_balancing.json").read_text())["banding"]

# versi sebelumnya, untuk bagian "Dari v1 ke v3.1"
V3T = json.loads((REPO / "v3_evaluasi_robust" / "laporan" / "_build" / "tables.json").read_text())
V31T = json.loads((REPO / "v3.1_kalibrasi_produksi" / "laporan" / "_build" / "tables.json").read_text())
V32T = json.loads((REPO / "v3.2_uji_takikardia" / "laporan" / "_build" / "tables.json").read_text())
LAMA = V3T["old"]["rows"]


def klaim_v1():
    """Macro-F1 MicroTransformer yang dilaporkan notebook v1 (rata-rata per fold ± sd), dibaca dari output selnya."""
    nb = json.loads((REPO / "v1_lama" / "model.ipynb").read_text())
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            m = re.search(r"Micro-Transformer\s+([0-9.]+)\s*±\s*([0-9.]+)", "".join(o.get("text", [])))
            if m:
                return float(m[1]), float(m[2])
    raise ValueError("Skor MicroTransformer tidak ditemukan di output v1_lama/model.ipynb")


KLAIM_V1 = klaim_v1()


def muat_sampling():
    """Angka bagian "Cara sampling": window, fold, data latih per fold (dicatat runner saat training), strata bootstrap."""
    d = D.muat()
    rk = [(r, k) for r in range(D.N_ULANGAN) for k in range(D.N_FOLD)]
    info = lambda s, r, k: json.loads(str(np.load(CACHE / "bigru" / s / f"r{r}_k{k}.npz")["info"]))["n_latih"]  # noqa: E731
    n = {s: np.array([info(s, r, k) for r, k in rk]) for s in ("bawaan", "under", "over")}
    asli = np.array([np.bincount(d.y[d.folds[r] != k], minlength=3) for r, k in rk])
    assert np.array_equal(n["bawaan"], asli)
    assert np.array_equal(n["under"], np.repeat(asli.min(1, keepdims=True), 3, 1))
    assert np.array_equal(n["over"], np.repeat(asli.max(1, keepdims=True), 3, 1))
    assert all(n[s][0].tolist() == R[s]["n_latih"] for s in n) and R["wp100"]["n_latih"] == R["bawaan"]["n_latih"]
    assert len(d.y) == E1["n_windows"] and np.bincount(d.y, minlength=3).tolist() == E1["class_counts"]
    geser = np.diff(d.t)[d.records[1:] == d.records[:-1]]
    assert np.median(geser) == geser.min()
    assert len(np.load(CACHE / "bigru" / "bawaan" / "r0_k0.npz")["idx"]) == int((d.folds[0] == 0).sum())
    wp ={s: R[s]["setelan_efektif"]["weight_power"] for s in ("bawaan", "wp100")}
    w = D.cv.class_weights(np.repeat(np.arange(3), asli[0]))
    return SimpleNamespace(
        detik=D.hrv_core.WINDOW_SEC, geser=float(geser.min()), maks=D.hrv_core.MAX_BEATS,
        median_detak=int(np.median(d.M.sum(1))), n=n, rasio=asli[:, 0] / asli[:, 2], afl=(asli[:, 2].min(), asli[:, 2].max()),
        n_uji0=int((d.folds[0] == 0).sum()), w={s: w ** p for s, p in wp.items()}, wp=wp,
        strata=np.bincount(d.strata, minlength=3), n_rk=len(rk))


S = muat_sampling()


# ── format angka (koma desimal, titik ribuan, minus "−") ──────────────────────────────────────────────────────────────
def fnum(x, d=3):
    s = f"{abs(x):.{d}f}"
    nol = float(s) == 0
    return ("−" if x < 0 and not nol else "") + s.replace(".", ",")


def sg(x, d=3):
    s = fnum(x, d)
    return s if s.startswith("−") or float(s.replace(",", ".")) == 0 else "+" + s


def thou(n):
    return f"{int(round(n)):,}".replace(",", ".")


def pct(x, d=1):
    return fnum(100 * x, d) + "%"


def fp(p):
    return fnum(p, 2)


def vonis(b):
    d, lo, hi, p = b
    if d == lo == hi == 0:
        return "identik"
    return "lebih baik" if p >= 0.95 else "lebih buruk" if p <= 0.05 else "tidak terbedakan"


def ci(t, d=2):
    return f"[{fnum(t[1], d)}–{fnum(t[2], d)}]"


def dci(t, d=2):
    return f"[{sg(t[1], d)}…{sg(t[2], d)}]"


def esc(s):
    return html.escape(str(s), quote=True)


def kode(s):
    return f"<code>{esc(s)}</code>"


def perubahan(s):
    """Kunci SETELAN yang diubah dari bawaan, satu per baris."""
    return "<br>".join(kode(f"{k} = {v}") for k, v in R[s]["setelan"].items()) or '<span>tidak ada</span>'


# ── angka yang dipakai di teks ────────────────────────────────────────────────────────────────────────────────────────
M = lambda s, k="Macro-F1": R[s]["bias"][k][0]  # noqa: E731
V31 = E1["v31"]["metrik"]
V1 = E1["v1"]["metrik"]
N_SET = len(SETELAN)
N_MODEL = sum(N_FOLD.values())
N_KAGGLE = sum(R[s]["tempat"].startswith("Kaggle") for s in SETELAN)
N_PASIEN = E1["n_pasien"]
N_PASIEN_AFL = len(PASIEN_AFL)
N_WIN = E1["n_windows"]
N_AFL = E1["class_counts"][2]
P04936 = E1["afl_window"]["04936"] / N_AFL
B_UNDER = banding_ref("under")
B_OVER = banding_ref("over")
B_WP100 = banding_ref("wp100")
B_EP50 = banding_ref("ep50")
B_W1H16 = banding_ref("w1_h16")
E3_LAIN = [s for s in E["e3"]["baru"] if s != "w1_h16"]
E3_RENTANG = (min(banding_ref(s)[0] for s in E3_LAIN), max(banding_ref(s)[0] for s in E3_LAIN))
assert all(vonis(banding_ref(s)) == "tidak terbedakan" for s in E3_LAIN)
assert not any(vonis(banding_ref(s)) == "lebih baik" for s in SETELAN if banding_ref(s))
assert vonis(B_UNDER) == vonis(B_OVER) == vonis(B_EP50) == vonis(B_W1H16) == "lebih buruk"
N_LATIH_BAWAAN = sum(R["bawaan"]["n_latih"])
N_LATIH_UNDER = sum(R["under"]["n_latih"])
SALIN_AFL = R["over"]["n_latih"][2] / R["bawaan"]["n_latih"][2]
CF2 = E["e2"]["crossfit"]["metrik"]["pilihan"]["Macro-F1"][0]
CF3 = E["e3"]["crossfit"]["metrik"]["pilihan"]["Macro-F1"][0]
XGB_UNDER = XGB["under_vs_bawaan"]["banding"]["Macro-F1"]
BERSARANG_WP100 = E["e2"]["bersarang"]["wp100"]["vs_v1"]
BERSARANG_BAWAAN = E1["bersarang"]["bawaan"]["vs_v1"]
PITA_TINGGI = PITA[-1]
N_NORMAL_CEPAT = E1["v31"]["pita"][PITA_TINGGI][0]
assert vonis(XGB_UNDER) == "lebih baik"
assert not any(vonis(v31_vs(s)) == "lebih baik" for s in SETELAN)
BURUK_V31 = [s for s in SETELAN if vonis(v31_vs(s)) == "lebih buruk"]
BURUK_REF = [s for s in SETELAN if banding_ref(s) and vonis(banding_ref(s)) == "lebih buruk"]
AMBANG = {s: list(R[s]["ambang"].values()) for s in SETELAN}
AMB_MIN, AMB_MAKS = min(min(v) for v in AMBANG.values()), max(max(v) for v in AMBANG.values())
AMB_MIN_S = [s for s in SETELAN if min(AMBANG[s]) == AMB_MIN]
AMB_MAKS_S = [s for s in SETELAN if max(AMBANG[s]) == AMB_MAKS]
assert TANPA["bawaan"]["Macro-F1"][0] < TANPA[KANDIDAT]["Macro-F1"][0] - 0.1
assert TANPA["lr5e4"]["Macro-F1"][0] < TANPA[KANDIDAT]["Macro-F1"][0] - 0.1
assert BERSARANG_WP100["p"] >= 0.95 and BERSARANG_BAWAAN["p"] < 0.95
assert all(r["afl"][p] <= 0.05 for r in (E1["v31"], R["bawaan"], R["wp100"], R["over"], R["under"]) for p in ("07910", "08215"))

# v1 → v3.1: laporan v3/v3.1/v3.2 harus memakai angka v1 dan v3.1 yang sama dengan laporan BiGRU
SAMA = lambda a, b: all(abs(x - y) < 1e-9 for x, y in zip(a, b))  # noqa: E731
assert SAMA(LAMA["lama det"]["Macro-F1"], V1["Macro-F1"]) and SAMA(LAMA["sekarang det"]["Macro-F1"], V31["Macro-F1"])
assert SAMA(V31T["rows"]["v1"]["Macro-F1"], V1["Macro-F1"]) and SAMA(V31T["rows"]["v31_cf"]["Macro-F1"], V31["Macro-F1"])
assert V31T["package"]["pt_identical"], "bobot v3.1 harus identik dengan v2"
P_V31_V1 = V31T["pairs"]["v31_cf vs v1"]["Macro-F1"]
assert abs(P_V31_V1[0] - E1["v31_vs_v1"]["Macro-F1"][0]) < 1e-9
NEST_V1 = V32T["nested"]["v1"]
TANPA_V1 = V32T["sens_wo"]["pairs_v31_minus"]["v1"]
V2PROD = V31T["rows"]["v2_prod"]
BIAS_V2, BIAS_V31, GRID_V31 = V31T["b_v2"], V31T["b_new"], V31T["grid"][:2]
AMB_V1, AMB_V31 = V32T["threshold"]["v1"], V32T["threshold"]["v31"]
assert set(AMB_V31.values()) == {E1["paket_v31"]}
assert vonis(P_V31_V1) == "lebih baik" and NEST_V1["p"] >= 0.95 and vonis(TANPA_V1) == "tidak terbedakan"
assert all(vonis(v31_vs(KANDIDAT, k)) == "tidak terbedakan" for k in ("Macro-F1", "AFL F1", "AFL P", "AFL R"))
assert vonis(v31_vs(KANDIDAT, "Normal R")) == "lebih baik"


# ── grafik SVG ────────────────────────────────────────────────────────────────────────────────────────────────────────
def marker(kind, x, y, cls):
    """circle = tidak terbedakan / data penuh, square = lebih buruk / tanpa 04936, diamond = pembanding."""
    if kind == "square":
        return f'<rect class="{cls} ring" x="{x - 5.5:.1f}" y="{y - 5.5:.1f}" width="11" height="11" rx="1.5"/>'
    if kind == "diamond":
        return f'<path class="{cls} ring" d="M{x:.1f} {y - 7:.1f}L{x + 7:.1f} {y:.1f}L{x:.1f} {y + 7:.1f}L{x - 7:.1f} {y:.1f}Z"/>'
    return f'<circle class="{cls} ring" cx="{x:.1f}" cy="{y:.1f}" r="5.5"/>'


def svg_forest():
    """Selisih Macro-F1 setiap setelan terhadap v3.1, dikelompokkan per eksperimen."""
    W, xl, x0, x1, xr = 880, 16, 330, 690, 712
    dom = (-0.16, 0.06)
    ticks = [-0.16, -0.12, -0.08, -0.04, 0.0, 0.04]
    X = lambda v: x0 + (v - dom[0]) / (dom[1] - dom[0]) * (x1 - x0)  # noqa: E731
    rows = [("ref", "v1", "Sistem awal tim", [-E1["v31_vs_v1"]["Macro-F1"][0], -E1["v31_vs_v1"]["Macro-F1"][2],
                                               -E1["v31_vs_v1"]["Macro-F1"][1], 1 - E1["v31_vs_v1"]["Macro-F1"][3]])]
    judul = {e: f"Eksperimen {e[1]} · {E[e]['tanggal']}" for e in EKS}
    for e in EKS:
        rows.append(("grup", judul[e], None, None))
        for s in E[e]["baru"]:
            rows.append(("row", s, KET[s], v31_vs(s)))
    out, y = [], 64
    body = []
    for kind, nama, ket, b in rows:
        if kind == "grup":
            y += 10
            body.append(f'<text class="t-grup" x="{xl}" y="{y + 14}">{esc(nama)}</text>'
                        f'<line class="sep" x1="{xl}" x2="{W - 16}" y1="{y + 22}" y2="{y + 22}"/>')
            y += 30
            continue
        h = 38
        cy = y + h / 2
        d, lo, hi, p = b
        v = "pembanding" if kind == "ref" else vonis(b)
        assert v in ("tidak terbedakan", "lebih buruk", "pembanding"), (nama, v)
        hl = nama == KANDIDAT
        tip = (f"{nama}: Macro-F1 {fnum(M(nama))} {ci(R[nama]['bias']['Macro-F1'])}; "
               if kind == "row" else f"v1: Macro-F1 {fnum(V1['Macro-F1'][0])}; ")
        tip += f"selisih dengan v3.1 {sg(d)} {dci(b)}, P = {fp(p)}"
        g = [f'<g class="baris" tabindex="0" data-tip="{esc(tip)}">',
             f'<rect class="{"hl" if hl else "hit"}" x="{xl - 8}" y="{y + 1}" width="{W - 2 * xl + 16}" height="{h - 2}" rx="6"/>',
             f'<text class="t-nama" x="{xl}" y="{cy - 3}">{esc(nama)}</text>',
             f'<text class="t-ket" x="{xl}" y="{cy + 12}">{esc(ket)}</text>']
        if hl:
            g.append(f'<text class="t-badge" x="{xl + 8 + 8.6 * len(nama)}" y="{cy - 3}">KANDIDAT</text>')
        a, bb = max(lo, dom[0]), min(hi, dom[1])
        cls = "s0" if kind == "ref" else ("s2" if v == "lebih buruk" else "s1")
        g.append(f'<line class="ci {cls}-stroke" x1="{X(a):.1f}" x2="{X(bb):.1f}" y1="{cy}" y2="{cy}"/>')
        if lo < dom[0]:
            g.append(f'<path class="{cls}" d="M{x0 - 2} {cy}l8 -5v10z"/>')
        if d < dom[0]:
            g.append(f'<text class="t-luar" x="{x0 + 12}" y="{cy - 8}">titik {fnum(d)}, di luar skala</text>')
        else:
            g.append(marker({"s0": "diamond", "s1": "circle", "s2": "square"}[cls], X(d), cy, cls))
        g.append(f'<text class="t-nilai" x="{xr}" y="{cy - 3}">{sg(d)}</text>'
                 f'<text class="t-vonis" x="{xr}" y="{cy + 12}">{esc(v)}{"" if kind == "ref" else f", P = {fp(p)}"}</text>')
        g.append("</g>")
        body.append("".join(g))
        y += h
    H = y + 46
    grid = []
    for t in ticks:
        grid.append(f'<line class="{"nol" if t == 0 else "grid"}" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="56" y2="{y + 4}"/>'
                    f'<text class="t-tick" x="{X(t):.1f}" y="{y + 22}" text-anchor="middle">{sg(t, 2) if t else "0"}</text>')
    head = (f'<text class="t-ax" x="{X(0):.1f}" y="48" text-anchor="middle">v3.1 (paket produksi)</text>'
            f'<text class="t-ax" x="{(x0 + x1) / 2}" y="{y + 40}" text-anchor="middle">'
            f'Selisih Macro-F1 terhadap v3.1 · kanan = lebih baik · garis = CI 95%</text>'
            f'<g class="legend">{marker("circle", 24, 18, "s1")}<text x="36" y="22">tidak terbedakan dari v3.1</text>'
            f'{marker("square", 236, 18, "s2")}<text x="248" y="22">lebih buruk dari v3.1</text>'
            f'{marker("diamond", 424, 18, "s0")}<text x="436" y="22">sistem awal (pembanding)</text></g>')
    out.append(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Selisih Macro-F1 setiap setelan BiGRU terhadap v3.1">'
               + "".join(grid) + head + "".join(body) + "</svg>")
    return "".join(out)


def svg_dumbbell():
    """Macro-F1 data penuh vs tanpa pasien 04936."""
    W, xl, x0, x1, xr = 880, 16, 290, 690, 712
    dom = (0.5, 0.9)
    X = lambda v: x0 + (v - dom[0]) / (dom[1] - dom[0]) * (x1 - x0)  # noqa: E731
    sistem = [("v31", "v3.1", "Paket produksi"), ("bawaan", "bawaan", KET["bawaan"]), ("wp100", "wp100", KET["wp100"]),
              ("lr5e4", "lr5e4", KET["lr5e4"]), ("h16", "h16", KET["h16"]), ("over", "over", KET["over"]),
              ("under", "under", KET["under"])]
    y, body = 52, []
    for key, nama, ket in sistem:
        full = V31["Macro-F1"][0] if key == "v31" else M(key)
        tnp = TANPA[key]["Macro-F1"][0]
        assert dom[0] <= min(full, tnp) and max(full, tnp) <= dom[1], (key, full, tnp)
        h, cy = 40, y + 20
        hl = key == KANDIDAT
        tip = f"{nama}: data penuh {fnum(full)} → tanpa 04936 {fnum(tnp)} {ci(TANPA[key]['Macro-F1'])}"
        body.append(
            f'<g class="baris" tabindex="0" data-tip="{esc(tip)}">'
            f'<rect class="{"hl" if hl else "hit"}" x="{xl - 8}" y="{y + 1}" width="{W - 2 * xl + 16}" height="{h - 2}" rx="6"/>'
            f'<text class="t-nama" x="{xl}" y="{cy - 3}">{esc(nama)}</text><text class="t-ket" x="{xl}" y="{cy + 12}">{esc(ket)}</text>'
            f'<line class="conn" x1="{X(full):.1f}" x2="{X(tnp):.1f}" y1="{cy}" y2="{cy}"/>'
            f'{marker("circle", X(full), cy, "s1")}{marker("square", X(tnp), cy, "s2")}'
            f'<text class="t-nilai" x="{xr}" y="{cy + 5}">{fnum(full)} → {fnum(tnp)}</text></g>')
        y += h
    grid = "".join(
        f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="46" y2="{y + 4}"/>'
        f'<text class="t-tick" x="{X(t):.1f}" y="{y + 22}" text-anchor="middle">{fnum(t, 1)}</text>'
        for t in (0.5, 0.6, 0.7, 0.8, 0.9))
    head = (f'<g class="legend">{marker("circle", 24, 18, "s1")}<text x="36" y="22">data penuh ({N_PASIEN} pasien)</text>'
            f'{marker("square", 236, 18, "s2")}<text x="248" y="22">tanpa pasien 04936 ({E1["tanpa"]["n_pasien"]} pasien)</text></g>'
            f'<text class="t-ax" x="{(x0 + x1) / 2}" y="{y + 40}" text-anchor="middle">Macro-F1 (+ bias)</text>')
    return f'<svg viewBox="0 0 {W} {y + 50}" role="img" aria-label="Macro-F1 dengan dan tanpa pasien 04936">{grid}{head}{"".join(body)}</svg>'


def svg_palsu():
    """Jumlah window AFL palsu per ulangan (dari 5 ulangan OOF)."""
    W, xl, x0, x1 = 880, 16, 250, 770
    sistem = [("v31", "v3.1", "Paket produksi", E1["v31"]["palsu"]["total"]),
              ("bawaan", "bawaan", "Bobot kelas lunak", R["bawaan"]["palsu"]["total"]),
              ("wp100", "wp100", "Bobot kelas penuh", R["wp100"]["palsu"]["total"]),
              ("over", "over", "Oversampling", R["over"]["palsu"]["total"]),
              ("under", "under", "Undersampling", R["under"]["palsu"]["total"])]
    top = 2000
    assert max(v for *_, v in sistem) <= top
    X = lambda v: x0 + v / top * (x1 - x0)  # noqa: E731
    y, body = 16, []
    for key, nama, ket, v in sistem:
        h, cy = 40, y + 20
        x = X(v)
        bar = f'M{x0} {cy - 10}H{x - 4:.1f}a4 4 0 0 1 4 4v12a4 4 0 0 1 -4 4H{x0}Z'
        body.append(
            f'<g class="baris" tabindex="0" data-tip="{esc(f"{nama}: {thou(v)} window AFL palsu per ulangan")}">'
            f'<rect class="hit" x="{xl - 8}" y="{y + 1}" width="{W - 2 * xl + 16}" height="{h - 2}" rx="6"/>'
            f'<text class="t-nama" x="{xl}" y="{cy - 3}">{esc(nama)}</text><text class="t-ket" x="{xl}" y="{cy + 12}">{esc(ket)}</text>'
            f'<path class="{"s0" if key == "v31" else "s1"}" d="{bar}"/>'
            f'<text class="t-nilai" x="{x + 8:.1f}" y="{cy + 5}">{thou(v)}</text></g>')
        y += h
    grid = "".join(
        f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="10" y2="{y + 4}"/>'
        f'<text class="t-tick" x="{X(t):.1f}" y="{y + 22}" text-anchor="middle">{thou(t)}</text>'
        for t in (0, 500, 1000, 1500, 2000))
    ax = (f'<text class="t-ax" x="{(x0 + x1) / 2}" y="{y + 40}" text-anchor="middle">'
          f'Window yang salah disebut AFL, per ulangan (lebih kecil = lebih baik)</text>')
    return f'<svg viewBox="0 0 {W} {y + 50}" role="img" aria-label="Jumlah AFL palsu per ulangan">{grid}{ax}{"".join(body)}</svg>'


def svg_rr():
    """Ilustrasi: model hanya melihat jarak antar detak (IBI). Bukan data pasien."""
    contoh = [
        ("Normal", "teratur", [812, 788, 804, 796, 820, 790, 806, 798, 815, 792, 800, 808, 794]),
        ("AFIB", "tidak teratur", [520, 880, 610, 735, 455, 960, 580, 700, 495, 845, 640, 770, 530, 900, 610]),
        ("AFL 2:1", "teratur, cepat", [430, 431, 429, 432, 430, 431, 429, 430, 432, 431, 429, 430, 431, 432,
                                       430, 429, 431, 430, 432, 431, 429, 430, 431]),
        ("Normal cepat", "sinus takikardia", [440, 452, 446, 438, 450, 444, 455, 441, 448, 443, 451, 439,
                                              447, 445, 453, 442, 449, 444, 450, 446, 441, 452]),
    ]
    W, x0, x1, span = 880, 210, 864, 10000
    X = lambda ms: x0 + ms / span * (x1 - x0)  # noqa: E731
    out, y = [], 26
    out.append(f'<text class="t-ax" x="{x0}" y="14">satu window = 10 detik (maks. 30 detak) · ilustrasi, bukan data pasien</text>')
    for i, (nama, sifat, ibi) in enumerate(contoh):
        bpm = 60000 / (sum(ibi) / len(ibi))
        base = y + 44
        t, peaks = 160.0, []
        for d in [0] + ibi:
            t += d
            if t > span - 60:
                break
            peaks.append(t)
        spikes = "".join(
            f'M{X(p) - 5:.1f} {base}L{X(p) - 2:.1f} {base + 4}L{X(p):.1f} {base - 28}L{X(p) + 2.5:.1f} {base + 7}L{X(p) + 5:.1f} {base}'
            for p in peaks)
        a, b = X(peaks[1]), X(peaks[2])
        bracket = (f'<path class="brk" d="M{a:.1f} {base - 34}v-5H{b:.1f}v5"/>'
                   f'<text class="t-ibi" x="{(a + b) / 2:.1f}" y="{base - 43}" text-anchor="middle">{thou(peaks[2] - peaks[1])} ms</text>')
        warn = ' class="rr-mirip"' if i >= 2 else ""
        out.append(
            f'<g{warn}><text class="t-nama" x="16" y="{base - 10}">{esc(nama)}</text>'
            f'<text class="t-ket" x="16" y="{base + 6}">{esc(sifat)} · ±{thou(round(bpm / 5) * 5)} bpm</text>'
            f'<line class="base" x1="{x0}" x2="{x1}" y1="{base}" y2="{base}"/><path class="ekg" d="{spikes}"/>{bracket}</g>')
        y += 74
    return f'<svg viewBox="0 0 {W} {y + 8}" role="img" aria-label="Ilustrasi jarak antar detak untuk empat ritme">{"".join(out)}</svg>'


def fbias(b):
    return "[" + "; ".join(fnum(x, 1) for x in b) + "]"


def svg_versi():
    """Macro-F1 di data uji perangkat (detektor 50 Hz), dari sistem v1 sampai v3.1 dan kandidat BiGRU."""
    W, xl, x0, x1, xr = 880, 16, 300, 690, 712
    dom = (0.3, 1.0)
    X = lambda v: x0 + (v - dom[0]) / (dom[1] - dom[0]) * (x1 - x0)  # noqa: E731
    sistem = [("v1 sistem utuh", "front-end find_peaks lama di backend", LAMA["lama fe"]["Macro-F1"], False),
              ("v1 model", "detak dari detektor baru, tanpa bias", V1["Macro-F1"], False),
              ("v2 produksi", f"bobot v2, bias {fbias(BIAS_V2)}", V2PROD["Macro-F1"], False),
              ("v3.1", f"bobot v2, bias {fbias(BIAS_V31)} · paket aktif", V31["Macro-F1"], True),
              ("BiGRU wp100", "kandidat v11 (halaman ini)", R[KANDIDAT]["bias"]["Macro-F1"], False)]
    y, body = 16, []
    for nama, ket, (t, lo, hi), hl in sistem:
        assert dom[0] <= lo and hi <= dom[1], (nama, lo, hi)
        h, cy = 40, y + 20
        body.append(
            f'<g class="baris" tabindex="0" data-tip="{esc(f"{nama}: Macro-F1 {fnum(t)} {ci((t, lo, hi))}")}">'
            f'<rect class="{"hl" if hl else "hit"}" x="{xl - 8}" y="{y + 1}" width="{W - 2 * xl + 16}" height="{h - 2}" rx="6"/>'
            f'<text class="t-nama" x="{xl}" y="{cy - 3}">{esc(nama)}</text><text class="t-ket" x="{xl}" y="{cy + 12}">{esc(ket)}</text>'
            f'<line class="ci s1-stroke" x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{cy}" y2="{cy}"/>{marker("circle", X(t), cy, "s1")}'
            f'<text class="t-nilai" x="{xr}" y="{cy - 3}">{fnum(t)}</text>'
            f'<text class="t-vonis" x="{xr}" y="{cy + 12}">{ci((t, lo, hi))}</text></g>')
        y += h
    ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    grid = "".join(
        f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="10" y2="{y + 4}"/>'
        f'<text class="t-tick" x="{X(v):.1f}" y="{y + 22}" text-anchor="middle">{fnum(v, 1)}</text>' for v in ticks)
    ax = (f'<text class="t-ax" x="{(x0 + x1) / 2}" y="{y + 40}" text-anchor="middle">'
          f'Macro-F1 di data uji perangkat (detektor 50 Hz) · garis = CI 95%</text>')
    return f'<svg viewBox="0 0 {W} {y + 50}" role="img" aria-label="Macro-F1 dari v1 sampai v3.1">{grid}{ax}{"".join(body)}</svg>'


def kotak(x, y, w, h, judul, *isi, sorot=False):
    teks = "".join(f'<text class="d-isi" x="{x + 12}" y="{y + 40 + 16 * i}">{esc(t)}</text>' for i, t in enumerate(isi))
    return (f'<rect class="{"d-sorot" if sorot else "d-kotak"}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>'
            f'<text class="d-judul" x="{x + 12}" y="{y + 22}">{esc(judul)}</text>{teks}')


def panah(*titik):
    """Garis siku (segmen tegak/datar) dengan kepala panah di titik terakhir."""
    (xa, ya), (xb, yb) = titik[-2], titik[-1]
    dx, dy = (xb > xa) - (xb < xa), (yb > ya) - (yb < ya)
    ux, uy = xb - 8 * dx, yb - 8 * dy
    d = "M" + " L".join(f"{x} {y}" for x, y in (*titik[:-1], (ux, uy)))
    kepala = f"{xb},{yb} {ux - 4.5 * dy},{uy + 4.5 * dx} {ux + 4.5 * dy},{uy - 4.5 * dx}"
    return f'<path class="d-panah" d="{d}"/><polygon class="d-kepala" points="{kepala}"/>'


def label(x, y, *baris, anchor="middle"):
    return "".join(f'<text class="d-label" x="{x}" y="{y + 14 * i}" text-anchor="{anchor}">{esc(t)}</text>'
                   for i, t in enumerate(baris))


def svg_alur():
    """Alur satu fold: pasien dibagi dulu, balancing hanya di cabang latih, data uji tetap asli, lalu bootstrap pasien."""
    n0 = {s: int(S.n[s][0].sum()) for s in S.n}
    st = S.strata
    opsi = [("bobot lunak (bawaan)", n0["bawaan"]), ("bobot penuh (wp100)", n0["bawaan"]),
            ("undersampling", n0["under"]), ("oversampling", n0["over"])]
    grup = (f'<rect class="d-sorot" x="446" y="8" width="204" height="114" rx="8"/>'
            f'<text class="d-judul" x="458" y="28">Bantu AFL · pilih satu</text>'
            + "".join(f'<text class="d-isi" x="458" y="{50 + 20 * i}">{esc(a)}</text>'
                      f'<text class="d-isi angka" x="638" y="{50 + 20 * i}" text-anchor="end">{thou(b)}</text>'
                      for i, (a, b) in enumerate(opsi)))
    isi = [
        kotak(16, 104, 136, 84, "Data AFDB", f"{N_PASIEN} pasien", f"{thou(N_WIN)} window",
              f"{fnum(S.detik, 0)} dtk, geser {fnum(S.geser, 0)}"),
        panah((152, 146), (166, 146), (166, 64), (240, 64)), panah((152, 146), (166, 146), (166, 228), (240, 228)),
        label(172, 140, "dibagi", "per pasien", anchor="start"),
        kotak(240, 30, 160, 68, "Pasien latih", "4 dari 5 kelompok", f"{thou(n0['bawaan'])} window"),
        kotak(240, 194, 160, 68, "Pasien uji", "1 kelompok", f"{thou(S.n_uji0)} window"),
        panah((400, 64), (446, 64)), grup,
        panah((650, 64), (690, 64)), label(670, 56, "latih"),
        kotak(690, 30, 174, 68, "BiGRU", "satu model per fold", f"{thou(N_MODEL // N_SET)} model per setelan"),
        panah((777, 98), (777, 194)), label(785, 150, "memprediksi", anchor="start"),
        panah((400, 228), (690, 228)), label(545, 220, "tidak diubah, distribusi asli"),
        kotak(690, 194, 174, 68, "Prediksi", "window uji", "pasien tak pernah dilihat"),
        panah((777, 262), (777, 316)), label(769, 284, f"diulang {S.n_rk}×", f"{D.N_FOLD} fold × {D.N_ULANGAN} ulangan", anchor="end"),
        kotak(690, 316, 174, 68, "OOF gabungan", "tiap window diuji", "sekali per ulangan"),
        panah((690, 350), (650, 350)),
        kotak(446, 316, 204, 68, "Bootstrap pasien", f"{thou(EV.N_BOOT)}× ambil ulang pasien",
              f"{st[2]} AFL · {st[1]} AFIB · {st[0]} lain"),
        panah((446, 350), (400, 350)),
        kotak(240, 316, 160, 68, "CI 95% dan P", "sampel sama untuk", "semua setelan"),
    ]
    return ('<svg viewBox="0 0 880 396" role="img" aria-label="Alur satu fold: pasien dibagi dulu, balancing hanya pada data '
            'latih, data uji tetap asli, lalu pasien diambil ulang untuk CI dan P">' + "".join(isi) + "</svg>")


def svg_komposisi():
    """Komposisi kelas dan jumlah window latih per cara, contoh ulangan 0 fold 0."""
    W, x0, lebar, t0, tmaks = 880, 230, 330, 600, 190
    nmaks = max(int(S.n[s][0].sum()) for s in S.n)
    nb = int(S.n["bawaan"][0].sum())
    baris = [("bawaan", "Tanpa balancing", ["bawaan dan wp100: data sama", "AFL dibantu lewat bobot loss"], "data latih asli"),
             ("under", "Undersampling", ["acak tanpa pengembalian", "sampai = jumlah AFL"],
              f"{pct(S.n['under'][0].sum() / nb)} dari data asli"),
             ("over", "Oversampling", ["acak dengan pengembalian", "AFIB dan AFL sampai = Normal"],
              f"tiap window AFL muncul ±{fnum(SALIN_AFL, 0)}×")]
    kelas = ["Normal", "AFIB", "AFL"]
    out = ["".join(f'<rect class="s{i + 1}" x="{x0 + 92 * i}" y="9" width="11" height="11" rx="2"/>'
                   f'<text class="t-ax" x="{x0 + 92 * i + 17}" y="19">{k}</text>' for i, k in enumerate(kelas)),
           f'<text class="t-ax" x="{t0}" y="19">jumlah window latih</text>']
    y = 36
    for s, nama, ket, catatan in baris:
        n = S.n[s][0]
        tot = int(n.sum())
        seg, x = [], x0
        for i, v in enumerate(n):
            w = lebar * v / tot
            seg.append(f'<rect class="s{i + 1}" x="{x:.1f}" y="{y + 8}" width="{max(w - 2, 2):.1f}" height="16" rx="2"/>')
            x += w
        tip = f"{nama}: " + " · ".join(f"{k} {thou(v)}" for k, v in zip(kelas, n)) + f" = {thou(tot)} window"
        out.append(
            f'<g class="baris" tabindex="0" data-tip="{esc(tip)}">'
            f'<rect class="hit" x="8" y="{y}" width="{W - 16}" height="64" rx="6"/>'
            f'<text class="d-judul" x="16" y="{y + 21}">{esc(nama)}</text>'
            + "".join(f'<text class="t-ket" x="16" y="{y + 38 + 15 * i}">{esc(t)}</text>' for i, t in enumerate(ket))
            + "".join(seg)
            + f'<text class="t-ket" x="{x0}" y="{y + 42}">'
            + " · ".join(f"{k} {pct(v / tot)}" for k, v in zip(kelas, n)) + "</text>"
            f'<rect class="s0" x="{t0}" y="{y + 8}" width="{max(tmaks * tot / nmaks, 2):.1f}" height="16" rx="2"/>'
            f'<text class="t-nilai" x="{t0 + tmaks * tot / nmaks + 8:.1f}" y="{y + 21}">{thou(tot)}</text>'
            f'<text class="t-ket" x="{t0}" y="{y + 42}">{esc(catatan)}</text></g>')
        y += 74
    return (f'<svg viewBox="0 0 {W} {y - 4}" role="img" aria-label="Komposisi kelas dan jumlah window latih: tanpa balancing, '
            f'undersampling, oversampling">{"".join(out)}</svg>')


# ── tabel ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def tabel_ubah():
    rows = [
        ("Detak dicari dengan cara berbeda saat latihan dan saat dipakai. Latihan memakai anotasi dokter (250 Hz, rapi), "
         "backend memakai <code>find_peaks</code> tanpa filter pada sinyal 50 Hz, maksimal 150 bpm, hanya 10 detik pertama.",
         "Satu modul <code>hrv_core</code> untuk latihan dan perangkat: detektor R-peak Pan-Tompkins 50 Hz (sampai 240 bpm), "
         "seluruh sinyal dibagi window 10 detik. Model dilatih dengan detak dari detektor ini.", "v2"),
        ("Tidak ada cek kualitas sinyal.", "Window ditolak bila jarak antar detak tidak fisiologis atau detaknya terlalu sedikit.", "v2"),
        ("Window AFL digeser 1 detik (tumpang tindih 90%), Normal 5 detik, AFIB 10 detik. AFL di data uji jadi terhitung "
         "berlipat, sehingga skor terlihat tinggi.",
         "Semua kelas digeser sama, 5 detik. Window rapat hanya boleh untuk menambah data latih.", "v2"),
        ("Bila file model hilang, predictor diam-diam memakai bobot acak. Model disimpan sebagai pickle tanpa asal-usul.",
         "Model + metadata JSON dengan checksum SHA-256. File hilang atau tidak cocok = error, server tidak jalan.", "v2"),
        ("Dinilai dari satu pembagian pasien dan rata-rata per fold. Selisih antar model ternyata noise.",
         "5 ulangan × 5 fold pasien, OOF gabungan, CI bootstrap pasien, perbandingan berpasangan.", "v3"),
        ("Tanpa bias kelas (v1), lalu bias v2 dari grid sempit −3…+3 yang memotong optimum.",
         f"Bias {fbias(BIAS_V31)} dipilih cross-fit dengan grid {sg(GRID_V31[0], 0)}…{sg(GRID_V31[1], 0)}.", "v3.1"),
        ("Belum pernah diuji pada ritme normal yang cepat.",
         "Uji stres takikardia wajib untuk setiap kandidat (ritme teratur buatan 60–170 bpm).", "v3.2"),
    ]
    body = "".join(f'<tr><td>{a}</td><td>{b}</td><td class="nw"><strong>{v}</strong></td></tr>' for a, b, v in rows)
    return ('<table class="tabel ubah-tabel"><thead><tr><th>Masalah di v1</th><th>Perbaikan</th><th>Versi</th></tr></thead>'
            f'<tbody>{body}</tbody></table>')


def tabel_v1_v31():
    w = KANDIDAT
    kol = [("v1", V1, E1["v1"]["palsu"], AMB_V1), ("v3.1", V31, E1["v31"]["palsu"], AMB_V31),
           (f"BiGRU {w}", R[w]["bias"], R[w]["palsu"], R[w]["ambang"])]
    baris = [("Macro-F1", "Macro-F1", True), ("AFL F1", "AFL F1", True),
             ("AFL presisi", "AFL P", False), ("AFL recall", "AFL R", True), ("Normal recall", "Normal R", False)]
    rows = []
    for lbl, k, pakai_ci in baris:
        sel = "".join(f'<td class="num">{fnum(m[k][0])}{f"<span>{ci(m[k])}</span>" if pakai_ci else ""}</td>' for _, m, _, _ in kol)
        rows.append(f'<tr><th scope="row">{lbl}</th>{sel}</tr>')
    rows.append('<tr><th scope="row">AFL palsu di pasien tanpa AFL<span>window per ulangan</span></th>'
                + "".join(f'<td class="num">{thou(p["pasien_tanpa_afl"])}</td>' for _, _, p, _ in kol) + "</tr>")
    rows.append('<tr><th scope="row">Ambang takikardia (bpm)<span>4 jenis variasi detak</span></th>'
                + "".join(f'<td class="num">{" / ".join(str(v) for v in a.values())}</td>' for *_, a in kol) + "</tr>")
    head = "".join(f'<th class="num">{esc(n)}</th>' for n, *_ in kol)
    return f'<table class="tabel"><thead><tr><th>Data uji perangkat, OOF</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def sel_recall(r):
    p = round(100 * r)
    cls = "terang" if p >= 55 else ""
    return f'<td class="heat {cls}" style="--p:{p}%">{fnum(r, 2)}</td>'


def tabel_pasien():
    sistem = [("v3.1", "Paket produksi", E1["v31"]["afl"]), ("bawaan", KET["bawaan"], R["bawaan"]["afl"]),
              ("wp100", KET["wp100"], R["wp100"]["afl"]), ("over", KET["over"], R["over"]["afl"]),
              ("under", KET["under"], R["under"]["afl"])]
    head = "".join(
        f'<th class="num{" sorot" if p in ("04936", "08378") else ""}">{p}<span>{thou(E1["afl_window"][p])} window</span></th>'
        for p in PASIEN_AFL)
    rows = "".join(
        f'<tr{HL if nama == KANDIDAT else ""}><th scope="row">{kode(nama) if nama != "v3.1" else nama}'
        f'<span>{esc(ket)}</span></th>{"".join(sel_recall(afl[p]) for p in PASIEN_AFL)}</tr>'
        for nama, ket, afl in sistem)
    return f'<table class="tabel heat-tabel"><thead><tr><th>Sistem</th>{head}</tr></thead><tbody>{rows}</tbody></table>'


def tabel_balancing():
    sistem = [("bawaan", "Bobot kelas lunak (bawaan)"), ("wp100", "Bobot kelas penuh"),
              ("over", "Oversampling"), ("under", "Undersampling")]
    rows = []
    for s, nama in sistem:
        n = R[s]["n_latih"]
        b = banding_ref(s)
        cmp_ = "referensi" if b is None else f"{sg(b[0])} · {vonis(b)}"
        rows.append(
            f'<tr{HL if s == KANDIDAT else ""}><th scope="row">{kode(s)}<span>{esc(nama)}</span></th>'
            f'<td class="num">{" / ".join(thou(v) for v in n)}</td><td class="num">{fnum(M(s))}</td>'
            f'<td>{cmp_}</td><td class="num">{fnum(M(s, "AFL F1"))}</td>'
            f'<td class="num">{pct(R[s]["pita"][PITA[1]][1])}</td></tr>')
    return ('<table class="tabel"><thead><tr><th>Setelan</th><th class="num">Window latih<span>Normal / AFIB / AFL</span></th>'
            '<th class="num">Macro-F1</th><th>vs bawaan</th><th class="num">AFL F1</th>'
            f'<th class="num">Alarm palsu<span>Normal {esc(PITA[1])}</span></th></tr></thead><tbody>{"".join(rows)}</tbody></table>')


def tabel_takikardia():
    sistem = [("v31", "v3.1", "Paket produksi"), ("bawaan", "bawaan", KET["bawaan"]), ("wp100", "wp100", KET["wp100"]),
              ("w1_wd1e2", "w1_wd1e2", KET["w1_wd1e2"]), ("over", "over", KET["over"]), ("under", "under", KET["under"])]
    rows = []
    for key, nama, ket in sistem:
        if key == "v31":
            amb, pita = f"{E1['paket_v31']} (paket)", E1["v31"]["pita"]
        else:
            amb, pita = " / ".join(str(v) for v in R[key]["ambang"].values()), R[key]["pita"]
        rows.append(f'<tr{HL if key == KANDIDAT else ""}><th scope="row">{kode(nama) if key != "v31" else nama}'
                    f'<span>{esc(ket)}</span></th><td class="num">{amb}</td>'
                    + "".join(f'<td class="num">{pct(pita[b][1])}</td>' for b in PITA) + "</tr>")
    head = "".join(f'<th class="num">{esc(b.replace("–∞ bpm", " bpm ke atas"))}<span>{thou(E1["v31"]["pita"][b][0])} window</span></th>'
                   for b in PITA)
    return (f'<table class="tabel"><thead><tr><th>Sistem</th><th class="num">Ambang (bpm)<span>4 jenis variasi detak</span></th>'
            f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def tabel_lampiran():
    rows = []
    for e in EKS:
        rows.append(f'<tr class="grup"><th colspan="7">Eksperimen {e[1]} · {esc(E[e]["judul"])} · referensi '
                    f'{kode(REF_EKS[e])}</th></tr>')
        for s in E[e]["baru"]:
            b = banding_ref(s)
            ref = "—" if b is None else f"{sg(b[0])}<span>{dci(b)} · P = {fp(b[3])}</span><span>{vonis(b)}</span>"
            v = v31_vs(s)
            rows.append(
                f'<tr{HL if s == KANDIDAT else ""}><th scope="row">{kode(s)}<span>{esc(KET[s])}</span></th>'
                f'<td class="ubah">{perubahan(s)}<span>dilatih di {esc(R[s]["tempat"])}</span></td>'
                f'<td class="num">{thou(R[s]["param"])}</td>'
                f'<td class="num">{fnum(M(s))}<span>{ci(R[s]["bias"]["Macro-F1"])}</span></td>'
                f'<td class="num">{fnum(TANPA[s]["Macro-F1"][0])}</td><td class="num">{ref}</td>'
                f'<td class="num">{sg(v[0])}<span>P = {fp(v[3])}</span></td></tr>')
    return ('<table class="tabel lampiran"><thead><tr><th>Setelan</th><th>Perubahan dari bawaan</th>'
            '<th class="num">Parameter</th><th class="num">Macro-F1<span>CI 95%</span></th><th class="num">Tanpa 04936</th>'
            '<th class="num">vs referensi<span>eksperimennya</span></th><th class="num">vs v3.1</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


# ── halaman ───────────────────────────────────────────────────────────────────────────────────────────────────────────
CSS = """
/* Layout: satu kolom baca ±70 karakter; grafik & tabel boleh lebih lebar dan bergulir di wadahnya sendiri.
   Hero memakai kisi kertas EKG (1 mm / 5 mm) sebagai satu-satunya ornamen. */
:root {
  --bg: #f3f7f7; --surface: #fbfcfc; --ink: #172b43; --ink-2: #3f5165; --muted: #5a6977;
  --line: #d6e0e2; --grid: #e4ebec; --accent: #008b83; --accent-ink: #00706a; --wash: rgba(0, 139, 131, .07);
  --ekg-1: rgba(0, 139, 131, .07); --ekg-5: rgba(0, 139, 131, .14);
  --s0: #7d8a96; --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --seq: #1c5cab; --on-seq: #ffffff; --code: #e6eeef;
  --f-display: "Archivo", "Arial Narrow", "Liberation Sans Narrow", Arial, sans-serif;
  --f-body: "Atkinson Hyperlegible", "Liberation Sans", Arial, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "Liberation Mono", monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0e1823; --surface: #13202f; --ink: #e5edf2; --ink-2: #b6c4cf; --muted: #8e9eac;
  --line: #273a4e; --grid: #1e2e3f; --accent: #2fb3a9; --accent-ink: #5fcbc2; --wash: rgba(47, 179, 169, .10);
  --ekg-1: rgba(47, 179, 169, .06); --ekg-5: rgba(47, 179, 169, .12);
  --s0: #8e9eac; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --seq: #2f72c8; --on-seq: #ffffff; --code: #1b2b3c; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0e1823; --surface: #13202f; --ink: #e5edf2; --ink-2: #b6c4cf; --muted: #8e9eac;
  --line: #273a4e; --grid: #1e2e3f; --accent: #2fb3a9; --accent-ink: #5fcbc2; --wash: rgba(47, 179, 169, .10);
  --ekg-1: rgba(47, 179, 169, .06); --ekg-5: rgba(47, 179, 169, .12);
  --s0: #8e9eac; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --seq: #2f72c8; --on-seq: #ffffff; --code: #1b2b3c; color-scheme: dark }

* { box-sizing: border-box }
html { scroll-behavior: smooth; scroll-padding-top: 64px }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto } }
body { margin: 0; background: var(--bg); color: var(--ink); font: 17px/1.6 var(--f-body);
  -webkit-text-size-adjust: 100%; text-rendering: optimizeLegibility }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--bg) }
.wrap { max-width: 1000px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)) }
h1, h2, h3 { font-family: var(--f-display); font-stretch: 87.5%; line-height: 1.12; text-wrap: balance; margin: 0 }
h1 { font-size: clamp(2.1rem, 5.4vw, 3.4rem); font-weight: 800; letter-spacing: -.01em }
h2 { font-size: clamp(1.55rem, 3.4vw, 2.1rem); font-weight: 750 }
h3 { font-size: 1.15rem; font-weight: 700 }
p { margin: 0; max-width: 68ch }
a { color: var(--accent-ink) }
code { font-family: var(--f-mono); font-size: .86em; background: var(--code); padding: .08em .35em; border-radius: 4px; white-space: nowrap }
strong { font-weight: 700 }
.eyebrow { font: 600 .78rem/1.2 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent-ink) }
.muted { color: var(--muted) }
:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; border-radius: 4px }

/* navigasi */
nav.atas { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg);
  border-bottom: 1px solid var(--line) }
nav.atas .wrap { display: flex; gap: 4px; overflow-x: auto; scrollbar-width: none; padding-block: 8px }
nav.atas .wrap::-webkit-scrollbar { display: none }
nav.atas a { flex: none; text-decoration: none; color: var(--ink-2); font-size: .86rem; padding: 6px 10px; border-radius: 999px }
nav.atas a:hover { background: var(--wash); color: var(--ink) }

/* hero: kisi kertas EKG */
header.hero { background-color: var(--bg);
  background-image: linear-gradient(var(--ekg-5) 1px, transparent 1px), linear-gradient(90deg, var(--ekg-5) 1px, transparent 1px),
    linear-gradient(var(--ekg-1) 1px, transparent 1px), linear-gradient(90deg, var(--ekg-1) 1px, transparent 1px);
  background-size: 50px 50px, 50px 50px, 10px 10px, 10px 10px; border-bottom: 1px solid var(--line) }
header.hero .wrap { display: grid; gap: 22px; padding-block: 56px 40px }
.lead { font-size: 1.16rem; color: var(--ink-2); max-width: 64ch }
.tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
  display: grid; gap: 4px; align-content: start }
.tile .lbl { font-size: .84rem; color: var(--muted) }
.tile .val { font: 750 1.9rem/1.1 var(--f-display); font-stretch: 87.5% }
.tile .sub { font-size: .88rem; color: var(--ink-2) }
.tile.utama { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent) }
@media (max-width: 820px) { .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)) } }
@media (max-width: 440px) { .tiles { grid-template-columns: minmax(0, 1fr) } }

/* section */
main section { padding-block: 52px 8px; display: grid; gap: 18px }
main section > * { min-width: 0 }
.jawab { display: grid; gap: 10px; margin: 0; padding: 0; list-style: none; max-width: 76ch }
.jawab li { display: grid; grid-template-columns: 28px minmax(0, 1fr); gap: 10px; align-items: start }
.jawab li::before { content: ""; width: 12px; height: 12px; margin-top: .45em; border-radius: 3px; background: var(--accent) }
.istilah { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px 28px; margin: 0 }
.istilah div { border-top: 1px solid var(--line); padding-top: 10px }
.istilah dt { font-weight: 700 }
.istilah dd { margin: 4px 0 0; color: var(--ink-2) }
@media (max-width: 720px) { .istilah { grid-template-columns: minmax(0, 1fr) } }

/* perjalanan: urutan nyata, jadi bernomor */
ol.jalan { list-style: none; margin: 0; padding: 0; display: grid; gap: 0; counter-reset: j }
ol.jalan > li { display: grid; grid-template-columns: 112px minmax(0, 1fr); gap: 18px; padding-block: 18px;
  border-top: 1px solid var(--line) }
ol.jalan .tgl { font: 600 .82rem/1.3 var(--f-mono); color: var(--muted) }
ol.jalan .tgl b { display: block; font: 750 1.5rem/1.1 var(--f-display); font-stretch: 87.5%; color: var(--accent-ink) }
ol.jalan .isi { display: grid; gap: 8px }
ol.jalan .qa { display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 4px 12px }
ol.jalan .qa dt { font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); padding-top: .2em }
ol.jalan .qa dd { margin: 0 }
ol.jalan .isi ul { margin: 0; padding-left: 1.1em; display: grid; gap: 6px; max-width: 68ch; list-style: disc }
@media (max-width: 600px) { ol.jalan > li, ol.jalan .qa { grid-template-columns: minmax(0, 1fr); gap: 6px } }

/* grafik */
figure { margin: 0; display: grid; gap: 10px }
figcaption { font-size: .92rem; color: var(--muted); max-width: 76ch }
.kanvas { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; overflow-x: auto; padding: 8px 4px }
.kanvas svg { display: block; width: 100%; min-width: 720px; height: auto }
svg text { font-family: var(--f-body); fill: var(--ink) }
svg .t-nama { font: 600 14px var(--f-mono); fill: var(--ink) }
svg .t-ket { font-size: 12px; fill: var(--muted) }
svg .t-grup { font: 700 13px var(--f-display); letter-spacing: .06em; text-transform: uppercase; fill: var(--accent-ink) }
svg .t-badge { font: 700 10px var(--f-mono); letter-spacing: .08em; fill: var(--accent-ink) }
svg .t-nilai { font: 600 14px var(--f-mono); fill: var(--ink); font-variant-numeric: tabular-nums }
svg .t-vonis, svg .t-luar { font-size: 12px; fill: var(--muted) }
svg .t-tick { font-size: 12px; fill: var(--muted); font-variant-numeric: tabular-nums }
svg .t-ax { font-size: 12.5px; fill: var(--muted) }
svg .t-ibi { font: 500 11.5px var(--f-mono); fill: var(--ink-2) }
svg .legend text { font-size: 13px; fill: var(--ink-2) }
svg .grid { stroke: var(--grid); stroke-width: 1 }
svg .nol { stroke: var(--ink-2); stroke-width: 1.5 }
svg .sep { stroke: var(--line); stroke-width: 1 }
svg .hit { fill: transparent }
svg .hl { fill: var(--wash) }
svg .baris:hover .hit, svg .baris:focus .hit { fill: var(--wash) }
svg .baris:focus { outline: none }
svg .baris:focus-visible .hit, svg .baris:focus-visible .hl { stroke: var(--accent); stroke-width: 2 }
svg .ci { stroke-width: 2.5; stroke-linecap: round }
svg .s0-stroke { stroke: var(--s0) } svg .s1-stroke { stroke: var(--s1) } svg .s2-stroke { stroke: var(--s2) }
svg .s0 { fill: var(--s0) } svg .s1 { fill: var(--s1) } svg .s2 { fill: var(--s2) } svg .s3 { fill: var(--s3) }
svg .ring { stroke: var(--surface); stroke-width: 2 }
svg .conn { stroke: var(--line); stroke-width: 3; stroke-linecap: round }
svg .base { stroke: var(--grid); stroke-width: 1 }
svg .ekg { fill: none; stroke: var(--ink); stroke-width: 1.6; stroke-linejoin: round; stroke-linecap: round }
svg .rr-mirip .ekg { stroke: var(--s2) }
svg .brk { fill: none; stroke: var(--muted); stroke-width: 1 }
/* diagram alur: kotak netral, satu kotak sorot (tempat data latih diubah), panah berlabel */
svg .d-kotak { fill: var(--surface); stroke: var(--line); stroke-width: 1.5 }
svg .d-sorot { fill: var(--wash); stroke: var(--accent); stroke-width: 1.5 }
svg .d-judul { font: 700 13.5px var(--f-body); fill: var(--ink) }
svg .d-isi { font-size: 12.5px; fill: var(--ink-2) }
svg .d-isi.angka { font-family: var(--f-mono); font-variant-numeric: tabular-nums }
svg .d-label { font-size: 12px; fill: var(--ink-2) }
svg .d-panah { fill: none; stroke: var(--ink-2); stroke-width: 1.5; stroke-linejoin: round }
svg .d-kepala { fill: var(--ink-2) }

/* tabel */
.gulir { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--surface) }
.tabel { border-collapse: collapse; width: 100%; font-size: .9rem; font-variant-numeric: tabular-nums }
.tabel th, .tabel td { padding: 9px 10px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--grid) }
.tabel thead th { font-size: .78rem; font-weight: 700; color: var(--ink-2); background: var(--bg); position: sticky; top: 0 }
.tabel th span, .tabel td span { display: block; font-size: .78rem; font-weight: 400; color: var(--muted); white-space: nowrap }
.tabel .num { text-align: right; white-space: nowrap }
.tabel tbody th { font-weight: 400 }
.tabel tbody th span { white-space: normal; min-width: 150px }
.tabel tr.hl { background: var(--wash) }
.tabel tr.grup th { background: var(--bg); font-family: var(--f-display); font-weight: 700; color: var(--accent-ink); font-size: .85rem }
.tabel th.sorot { color: var(--ink) }
.heat-tabel td.heat { text-align: center; font-weight: 600; min-width: 64px;
  background: color-mix(in oklab, var(--seq) var(--p), var(--surface)) }
.heat-tabel td.heat.terang { color: var(--on-seq) }
.tabel .nw { white-space: nowrap }
.lampiran td.ubah code { display: inline-block; margin-block: 1px }

.catatan { border-left: 3px solid var(--accent); padding: 4px 0 4px 14px; color: var(--ink-2); max-width: 74ch }
.dua { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px 28px }
@media (max-width: 760px) { .dua { grid-template-columns: minmax(0, 1fr) } }
.kartu { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; display: grid;
  gap: 8px; align-content: start }
.kartu ul { margin: 0; padding-left: 1.1em; display: grid; gap: 6px; align-content: start }
footer { padding-block: 48px 64px; color: var(--muted); font-size: .88rem }

.tip { position: fixed; z-index: 20; pointer-events: none; max-width: 340px; background: var(--ink); color: var(--bg);
  font-size: .84rem; line-height: 1.4; padding: 7px 10px; border-radius: 6px }
"""

JS = """
(() => {
  const t = document.createElement('div'); t.className = 'tip'; t.hidden = true; document.body.appendChild(t);
  const show = (el, x, y) => {
    t.textContent = el.getAttribute('data-tip'); t.hidden = false;
    const r = t.getBoundingClientRect();
    let L = x + 14, T = y + 14;
    if (L + r.width > innerWidth - 8) L = x - r.width - 14;
    if (T + r.height > innerHeight - 8) T = y - r.height - 14;
    t.style.left = Math.max(8, L) + 'px'; t.style.top = Math.max(8, T) + 'px';
  };
  document.addEventListener('pointermove', e => {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (el) show(el, e.clientX, e.clientY); else t.hidden = true;
  });
  document.addEventListener('focusin', e => {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (el) { const r = el.getBoundingClientRect(); show(el, r.left + 24, r.top + r.height / 2); }
  });
  document.addEventListener('focusout', () => { t.hidden = true; });
  document.addEventListener('scroll', () => { t.hidden = true; }, { passive: true });
})();
"""

FONTS = ("https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,600..800"
         "&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap")


def halaman():
    b, w = "bawaan", KANDIDAT
    afl08378 = (R[b]["afl"]["08378"], R[w]["afl"]["08378"])
    tiles = f"""
<div class="tiles">
  <div class="tile"><span class="lbl">Setelan yang dicoba</span><span class="val">{N_SET}</span>
    <span class="sub">{thou(N_MODEL)} model dilatih ({N_SET} × 25 fold pasien)</span></div>
  <div class="tile utama"><span class="lbl">Kandidat akhir</span><span class="val">bobot penuh</span>
    <span class="sub"><code>weight_power</code> 1,0 · Macro-F1 {fnum(M(w))}, setara v3.1 ({fnum(V31["Macro-F1"][0])})</span></div>
  <div class="tile"><span class="lbl">Tanpa pasien 04936</span><span class="val">{fnum(TANPA[w]["Macro-F1"][0])}</span>
    <span class="sub">bawaan turun ke {fnum(TANPA[b]["Macro-F1"][0])}</span></div>
  <div class="tile"><span class="lbl">Balancing data untuk BiGRU</span><span class="val">lebih buruk</span>
    <span class="sub">under {sg(B_UNDER[0])}, over {sg(B_OVER[0])} vs bawaan</span></div>
</div>"""

    jawab = f"""
<ul class="jawab">
  <li><span><strong>BiGRU setara dengan paket produksi v3.1, tetapi tidak lebih baik.</strong> Dari {N_SET} setelan, tidak ada
    yang lebih baik dari v3.1. Dengan {N_PASIEN} pasien ({N_PASIEN_AFL} di antaranya punya AFL), selisih kecil tidak bisa
    dibedakan dari kebetulan.</span></li>
  <li><span><strong>Balancing data (arahan pembimbing) membuat BiGRU lebih buruk daripada pembobotan kelas.</strong>
    Undersampling {sg(B_UNDER[0])} dan oversampling {sg(B_OVER[0])} Macro-F1 (P = {fp(B_UNDER[3])} dan {fp(B_OVER[3])}). Di XGBoost
    justru undersampling membantu ({sg(XGB_UNDER[0])}, P = {fp(XGB_UNDER[3])}), jadi efek balancing bergantung pada model.</span></li>
  <li><span><strong>Bobot kelas penuh paling tahan uji.</strong> AFL 2:1 di pasien 08378 kembali terdeteksi
    ({fnum(afl08378[0], 2)} → {fnum(afl08378[1], 2)}), dan hasilnya tidak lagi bergantung pada pasien 04936.</span></li>
  <li><span><strong>Tuning sudah jenuh.</strong> Regularisasi dan pemilihan setelan otomatis tidak menambah apa-apa.
    Perbaikan berikutnya butuh data baru (pasien lain, rekaman perangkat SIHEDAF berlabel), bukan hiperparameter.</span></li>
</ul>"""

    istilah = [
        ("Fold pasien (5 ulangan × 5 fold)",
         f"{N_PASIEN} pasien dibagi lima kelompok. Model dilatih dengan empat kelompok dan diuji pada kelompok sisanya, bergiliran. "
         "Pembagian diacak ulang lima kali, jadi satu setelan = 25 model. Pasien uji tidak pernah dilihat saat latihan."),
        ("Window", f"Potongan rekaman 10 detik (maks. 30 detak). Data uji berisi {thou(N_WIN)} window dari detektor 50 Hz, "
                   "sama dengan kondisi di perangkat."),
        ("Macro-F1", "Rata-rata skor F1 tiga kelas (Normal, AFIB, AFL). AFL yang langka dihitung sama penting dengan Normal."),
        ("CI 95%", f"Rentang ketidakpastian dari mengambil ulang pasien 2.000 kali. Lebar karena pasiennya hanya {N_PASIEN}."),
        ("Perbandingan berpasangan dan P",
         "Dua setelan dinilai pada sampel ulang pasien yang sama. P = peluang setelan baru lebih baik. "
         "P ≥ 0,95 lebih baik, P ≤ 0,05 lebih buruk, selain itu tidak terbedakan."),
        ("Bias kelas", "Geseran skor tiap kelas sebelum memilih tebakan, untuk menyeimbangkan alarm. Nilainya dipelajari dari fold "
                       "lain (cross-fit), jadi pasien uji tidak ikut menentukan."),
        ("Uji stres takikardia", "Ritme teratur buatan 60–170 bpm. Dicari detak di mana model berhenti menyebut Normal. "
                                 "Penting karena data latih hampir tidak punya ritme normal yang cepat."),
        ("Tanpa 04936", f"{pct(P04936, 0)} window AFL berasal dari satu pasien (04936). Hasil dihitung ulang tanpa pasien ini "
                        "untuk melihat apakah model hanya pandai di pasien itu."),
    ]
    istilah_html = "".join(f"<div><dt>{esc(a)}</dt><dd>{esc(b_)}</dd></div>" for a, b_ in istilah)

    langkah = [
        ("22 Okt", "Langkah 0", "Memeriksa dataset balancing tim di Drive",
         "Bisakah CSV over/undersampling dari Drive dipakai langsung?",
         "Tidak langsung. Isinya turunan dataset v1: 6 fitur, satu pembagian pasien saja, dan kolom pasien sudah hilang, sehingga "
         "tidak bisa disejajarkan dengan fold v11. Caranya diulang sebagai opsi <code>balancing</code> yang dijalankan di dalam "
         "setiap fold, hanya pada data latih."),
        ("23 Okt", "Eksperimen 1", E1["judul"],
         "Seberapa baik BiGRU dengan bobot kelas lunak (aturan proyek), dan apakah undersampling membantu?",
         f"Bawaan: Macro-F1 {fnum(M(b))}, setara v3.1 (P = {fp(v31_vs(b)[3])}). Undersampling {sg(B_UNDER[0])}, lebih buruk "
         f"(P = {fp(B_UNDER[3])}), karena hanya {thou(N_LATIH_UNDER)} dari {thou(N_LATIH_BAWAAN)} window latih yang tersisa. "
         f"Kelemahan bawaan: AFL 2:1 di pasien 08378 hanya {fnum(afl08378[0], 2)} (v3.1 {fnum(E1['v31']['afl']['08378'], 2)})."),
        ("2 Nov", "Eksperimen 2", E["e2"]["judul"],
         "Apakah setelan lain memperbaiki kelemahan itu, dan apakah oversampling membantu?",
         f"Tidak ada yang lebih baik dari bawaan pada Macro-F1. Bobot penuh ({kode('wp100')}) memulihkan 08378 "
         f"({fnum(afl08378[0], 2)} → {fnum(afl08378[1], 2)}) dan tetap {fnum(TANPA[w]['Macro-F1'][0])} tanpa 04936. Oversampling "
         f"{sg(B_OVER[0])} dan latih lebih lama {sg(B_EP50[0])}, keduanya lebih buruk. Ulangan bawaan identik bit demi bit, jadi "
         f"training-nya deterministik. Dijalankan paralel: {N_SET_E2_PC()} setelan di PC, {N_SET_E2_KAGGLE()} di Kaggle."),
        ("2 Nov", "Eksperimen 3", E["e3"]["judul"],
         "Apakah regularisasi di atas bobot penuh menambah sesuatu?",
         f"Tidak. Lima setelan berada dalam {sg(E3_RENTANG[0])} … {sg(E3_RENTANG[1])} dari {kode('wp100')}, semuanya tidak "
         f"terbedakan. Bobot penuh dengan model kecil lebih buruk ({sg(B_W1H16[0])}). Pemilihan setelan otomatis per fold "
         f"(cross-fit) memberi {fnum(CF3)}, sama dengan {kode('wp100')}."),
    ]
    jalan = "".join(
        f'<li><div class="tgl"><b>{esc(tgl)}</b>{esc(lbl)}</div><div class="isi"><h3>{esc(jdl)}</h3>'
        f'<dl class="qa"><dt>Pertanyaan</dt><dd>{esc(q)}</dd><dt>Jawaban</dt><dd>{a}</dd></dl></div></li>'
        for tgl, lbl, jdl, q, a in langkah)

    cc, st = E1["class_counts"], S.strata
    bobot = lambda s: f"AFL {fnum(S.w[s][2], 1)}, AFIB {fnum(S.w[s][1], 2)}, Normal {fnum(S.w[s][0], 2)}"  # noqa: E731
    tahap = [
        ("1", "Window", "Memotong rekaman menjadi window",
         f"<p>Sinyal EKG {N_PASIEN} pasien diturunkan ke 50 Hz, lalu detaknya dicari dengan detektor yang sama dengan perangkat. "
         f"Rekaman dipotong menjadi window {fnum(S.detik, 0)} detik yang digeser {fnum(S.geser, 0)} detik, jadi dua window "
         f"berurutan tumpang tindih {pct(1 - S.geser / S.detik, 0)}. Satu window berisi maksimal {S.maks} detak (median "
         f"{S.median_detak}). Hasilnya {thou(N_WIN)} window: Normal {thou(cc[0])}, AFIB {thou(cc[1])}, AFL {thou(cc[2])}.</p>"
         "<p>Window tidak dibuat ulang. Datanya sama persis dengan v3 (dicek SHA-256), supaya hasil BiGRU bisa dibandingkan "
         "dengan v1 dan v3.1 pada window yang sama.</p>"),
        ("2", "Fold", "Membagi pasien, bukan window",
         f"<p>Pasien dibagi lima kelompok: empat untuk latihan, satu untuk uji, bergiliran. Pembagiannya diulang "
         f"{D.N_ULANGAN} kali dengan susunan berbeda, jadi ada {S.n_rk} pasangan latih–uji. Satu pasien tidak pernah ada di data "
         "latih dan data uji sekaligus, sehingga model selalu diuji pada pasien yang belum pernah dilihatnya.</p>"
         f"<p>Karena AFL menumpuk di {N_PASIEN_AFL} pasien, isi data latih sangat berubah antar fold: AFL {thou(S.afl[0])}–"
         f"{thou(S.afl[1])} window, dengan perbandingan Normal : AFL dari {thou(S.rasio.min())} : 1 sampai "
         f"{thou(S.rasio.max())} : 1.</p>"),
        ("3", "Balancing", "Membantu AFL, hanya di data latih",
         "<p>Empat cara dibandingkan. Semuanya dijalankan setelah pembagian, pada window latih fold itu saja, dengan seed yang "
         f"sama dengan seed latihan ({D.SEED} + 100 × ulangan + fold). Jadi setiap cara bisa dibandingkan berpasangan dengan "
         "bawaan. Data uji tidak pernah diubah.</p><ul>"
         f"<li><strong>Bobot kelas lunak</strong> ({kode('bawaan')}): data tidak diubah. Kesalahan tiap kelas diberi bobot "
         f"saat latihan: {bobot('bawaan')}.</li>"
         f"<li><strong>Bobot kelas penuh</strong> ({kode('wp100')}): sama, tetapi bobotnya {bobot('wp100')}.</li>"
         "<li><strong>Undersampling</strong>: setiap kelas dipilih acak, tanpa pengembalian, sampai sama dengan jumlah AFL.</li>"
         "<li><strong>Oversampling</strong>: window AFIB dan AFL disalin acak, dengan pengembalian, sampai sama dengan jumlah "
         "Normal.</li></ul>"
         "<p>Setelah undersampling atau oversampling, bobot kelas otomatis bernilai 1, jadi AFL tidak dibantu dua kali. SMOTE "
         "tidak dipakai untuk BiGRU (alasannya di bagian Balancing).</p>"),
        ("4", "Bootstrap", "Mengukur ketidakpastian dengan mengambil ulang pasien",
         f"<p>Untuk CI dan P, yang diambil ulang adalah pasien, bukan window, sebanyak {thou(EV.N_BOOT)} kali. Window dari "
         "pasien yang sama saling mirip, jadi mengambil ulang window membuat CI terlalu sempit.</p>"
         f"<p>Pengambilannya per kelompok: {st[2]} pasien punya AFL, {st[1]} pasien dominan AFIB, dan {st[0]} lainnya. Setiap "
         f"sampel tetap berisi {st[2]} pasien AFL. Semua setelan dinilai pada {thou(EV.N_BOOT)} sampel yang sama (seed "
         f"{D.SEED}), lalu P = bagian sampel di mana setelan baru menang.</p>"),
    ]
    tahap_html = "".join(
        f'<li><div class="tgl"><b>{no}</b>{esc(lbl)}</div><div class="isi"><h3>{esc(jdl)}</h3>{isi}</div></li>'
        for no, lbl, jdl, isi in tahap)

    pita_rendah, pita_tinggi = R[b]["pita"][PITA[1]][1], R["under"]["pita"][PITA[1]][1]
    nested_w, nested_b = BERSARANG_WP100, BERSARANG_BAWAAN

    return f"""
<nav class="atas" aria-label="Bagian halaman"><div class="wrap">
  <a href="#ringkasan">Ringkasan</a><a href="#latar">Latar</a><a href="#cara-menilai">Metode</a><a href="#sampling">Sampling</a>
  <a href="#v1-ke-v31">v1 → v3.1</a><a href="#perjalanan">Langkah</a><a href="#hasil">Hasil</a><a href="#balancing">Balancing</a>
  <a href="#pasien">Pasien</a><a href="#tanpa-04936">Tanpa 04936</a><a href="#detak-cepat">Detak cepat</a>
  <a href="#kesimpulan">Kesimpulan</a><a href="#batasan">Batasan</a><a href="#lampiran">Lampiran</a>
</div></nav>

<header class="hero" id="ringkasan"><div class="wrap">
  <span class="eyebrow">SIHEDAF v11 · model tim · BiGRU + attention</span>
  <h1>Apa saja yang sudah dicoba pada BiGRU</h1>
  <p class="lead">Ringkasan tiga eksperimen ({E1["tanggal"]} – {E["e3"]["tanggal"]}) untuk diskusi dengan pembimbing.
    Model membaca jarak antar detak jantung dan memilih satu dari tiga kelas: Normal, AFIB, atau AFL.</p>
  {tiles}
  {jawab}
</div></header>

<main class="wrap">
<section id="latar">
  <span class="eyebrow">Latar</span>
  <h2>Yang dibaca model hanya jarak antar detak</h2>
  <p>BiGRU membaca urutan jarak antar detak (IBI, dalam milidetik) dari dua arah, lalu lapisan <em>attention</em> memilih detak
    yang paling penting. Modelnya kecil: {thou(R[b]["param"])} parameter. Karena hanya melihat jarak antar detak, beberapa ritme
    tampak sangat mirip.</p>
  <figure><div class="kanvas">{svg_rr()}</div>
    <figcaption>AFL 2:1 dan ritme normal yang cepat (sinus takikardia) sama-sama teratur dan cepat. Dari jarak antar detak saja,
      keduanya hampir tidak bisa dibedakan. Inilah sumber alarm palsu pada detak cepat (lihat bagian "Detak cepat").</figcaption></figure>
  <div class="dua">
    <div class="kartu"><h3>Datanya</h3><ul>
      <li>MIT-BIH AFDB: {N_PASIEN} pasien, {thou(N_WIN)} window uji.</li>
      <li>AFL hanya {pct(N_AFL / N_WIN)} window ({thou(N_AFL)}), dari {N_PASIEN_AFL} pasien.</li>
      <li>{pct(P04936, 0)} window AFL berasal dari satu pasien (04936).</li></ul></div>
    <div class="kartu"><h3>Pembandingnya</h3><ul>
      <li><strong>v3.1</strong>: paket produksi aktif (MicroTransformer), Macro-F1 {fnum(V31["Macro-F1"][0])}.</li>
      <li><strong>v1</strong>: sistem awal tim, Macro-F1 {fnum(V1["Macro-F1"][0])} bila diukur dengan cara yang sama.</li>
      <li>Penjelasan keduanya ada di bagian <a href="#v1-ke-v31">Dari v1 ke v3.1</a>.</li></ul></div>
  </div>
</section>

<section id="cara-menilai">
  <span class="eyebrow">Cara menilai</span>
  <h2>Satu protokol untuk semua setelan</h2>
  <p>Protokolnya sama dengan v3 dan dipakai semua anggota tim. Istilah yang muncul di halaman ini:</p>
  <dl class="istilah">{istilah_html}</dl>
</section>

<section id="sampling">
  <span class="eyebrow">Cara sampling</span>
  <h2>Pasien dibagi dulu, baru data latih boleh diubah</h2>
  <p>Sampling di sini terdiri dari empat tahap. Urutannya yang paling penting: data dibagi per pasien lebih dulu, lalu
    balancing (kalau dipakai) hanya mengubah data latih. Data uji selalu dibiarkan seperti aslinya, karena itulah kondisi
    yang akan dihadapi perangkat.</p>
  <figure><div class="kanvas">{svg_alur()}</div>
    <figcaption>Alur satu fold. Kotak berwarna adalah satu-satunya tempat data diubah. Angka window adalah contoh ulangan 0,
      fold 0; susunan pasien berubah di setiap fold. Kelompok bootstrap: pasien yang punya AFL, pasien dominan AFIB, dan
      lainnya.</figcaption></figure>
  <ol class="jalan">{tahap_html}</ol>
  <figure><div class="kanvas">{svg_komposisi()}</div>
    <figcaption>Data latih ulangan 0, fold 0. Undersampling hanya menyisakan {pct(N_LATIH_UNDER / N_LATIH_BAWAAN)} window
      latih. Oversampling membuat setiap window AFL muncul rata-rata ±{fnum(SALIN_AFL, 0)} kali. Bobot kelas tidak mengubah
      data, jadi {kode("bawaan")} dan {kode("wp100")} memakai data latih yang sama. Arahkan kursor ke baris untuk jumlah per
      kelas. Hasil ketiganya ada di bagian <a href="#balancing">Balancing</a>.</figcaption></figure>
  <p class="catatan">Yang sengaja tidak dilakukan: balancing sebelum data dibagi, termasuk memakai CSV balancing turunan v1
    di Drive. Kalau di-balance lebih dulu, data uji ikut seimbang, sehingga skornya tidak lagi menggambarkan kondisi nyata.
    Data latih setiap fold tersedia sebagai CSV di <code>cache/v11/bigru/sampling_csv.zip</code> (dibuat
    <code>ekspor_sampling.py</code>, isinya dicek sama dengan data latih saat training).</p>
</section>

<section id="v1-ke-v31">
  <span class="eyebrow">Pembanding</span>
  <h2>Dari v1 ke v3.1: apa yang diubah</h2>
  <p><strong>v3.1 adalah paket model yang sekarang dipakai backend</strong>
    (<code>v3.1_kalibrasi_produksi/model_export/</code>). Modelnya MicroTransformer, arsitektur yang sama dengan model akhir v1.
    Yang berubah adalah semua yang ada di sekeliling model: cara mencari detak, data latih, cara menilai, dan bias kelas.
    v3.1 sendiri tidak melatih apa pun. Bobotnya identik dengan v2 (checksum sama). Yang dihitung ulang hanya bias kelasnya:
    {fbias(BIAS_V2)} menjadi {fbias(BIAS_V31)} (Normal; AFIB; AFL).</p>
  <div class="gulir">{tabel_ubah()}</div>
  <figure><div class="kanvas">{svg_versi()}</div>
    <figcaption>Notebook v1 melaporkan Macro-F1 {fnum(KLAIM_V1[0])} ± {fnum(KLAIM_V1[1])}. Angka itu rata-rata per fold dari satu
      pembagian pasien, dengan detak dari anotasi dokter dan window AFL yang tumpang tindih. Diukur dengan data uji perangkat dan
      protokol v3, model v1 mendapat {fnum(V1["Macro-F1"][0])}. Bersama front-end <code>find_peaks</code> yang dulu dipakai
      backend, sistem utuhnya hanya {fnum(LAMA["lama fe"]["Macro-F1"][0])}.</figcaption></figure>
  <div class="gulir">{tabel_v1_v31()}</div>
  <ul class="jawab">
    <li><span><strong>v3.1 lebih baik dari v1.</strong> Selisih Macro-F1 {sg(P_V31_V1[0])} {dci(P_V31_V1)}, P = {fp(P_V31_V1[3])}.
      Bila pemilihan bias ikut diacak (bootstrap bersarang), P = {fnum(NEST_V1["p"], 3)}, vonisnya tetap.</span></li>
    <li><span><strong>Tetapi tanpa pasien 04936, keduanya tidak terbedakan</strong> ({sg(TANPA_V1[0])} {dci(TANPA_V1)},
      P = {fp(TANPA_V1[3])}). Satu pasien masih sangat menentukan.</span></li>
    <li><span><strong>Trade-off:</strong> recall AFL turun ({fnum(V1["AFL R"][0])} → {fnum(V31["AFL R"][0])}). v1 jauh lebih
      sering menebak AFL, sehingga AFL palsunya di pasien tanpa AFL {thou(E1["v1"]["palsu"]["pasien_tanpa_afl"])} window per
      ulangan, sedangkan v3.1 hanya {thou(E1["v31"]["palsu"]["pasien_tanpa_afl"])}.</span></li>
    <li><span><strong>Posisi BiGRU:</strong> kandidat {kode(KANDIDAT)} tidak terbedakan dari v3.1 pada Macro-F1 dan ketiga metrik
      AFL (P = {", ".join(fp(v31_vs(KANDIDAT, k)[3]) for k in ("Macro-F1", "AFL F1", "AFL P", "AFL R"))}). Recall Normal-nya
      sedikit lebih tinggi ({sg(v31_vs(KANDIDAT, "Normal R")[0])}, P = {fp(v31_vs(KANDIDAT, "Normal R")[3])}). Karena Macro-F1
      tidak lebih baik, BiGRU tidak diusulkan menggantikan v3.1.</span></li>
  </ul>
</section>

<section id="perjalanan">
  <span class="eyebrow">Perjalanan</span>
  <h2>Empat langkah, berurutan</h2>
  <ol class="jalan">{jalan}</ol>
</section>

<section id="hasil">
  <span class="eyebrow">Semua hasil</span>
  <h2>Tidak ada setelan yang mengalahkan v3.1</h2>
  <p>Setiap baris adalah satu setelan (25 model). Titik = selisih Macro-F1 terhadap v3.1, garis = rentang ketidakpastiannya.
    Kalau garis memotong garis tegak v3.1, selisihnya belum bisa dibedakan dari kebetulan.</p>
  <figure><div class="kanvas">{svg_forest()}</div>
    <figcaption>Arahkan kursor ke baris untuk melihat Macro-F1 dan CI-nya. Angka lengkap ada di Lampiran.
      Lebih buruk dari v3.1: {", ".join(kode(s) for s in BURUK_V31)}. Dibanding referensi eksperimennya sendiri (perbandingan
      yang lebih peka, karena modelnya mirip), yang lebih buruk: {", ".join(kode(s) for s in BURUK_REF)}.</figcaption></figure>
</section>

<section id="balancing">
  <span class="eyebrow">Arahan pembimbing</span>
  <h2>Balancing data vs pembobotan kelas</h2>
  <p>AFL sangat langka, jadi model perlu dibantu agar tidak mengabaikannya. Ada dua cara. <strong>Balancing</strong> mengubah data
    latih: membuang data kelas besar (undersampling) atau menyalin data kelas kecil (oversampling). <strong>Pembobotan kelas</strong>
    membiarkan data apa adanya, tetapi kesalahan pada AFL diberi bobot lebih besar saat latihan. SMOTE tidak bisa dipakai untuk
    BiGRU, karena mencampur dua urutan detak tidak menghasilkan ritme yang masuk akal. Penggantinya adalah oversampling dengan
    menyalin window asli.</p>
  <div class="gulir">{tabel_balancing()}</div>
  <figure><div class="kanvas">{svg_palsu()}</div>
    <figcaption>AFL palsu = window non-AFL yang ditebak AFL, rata-rata per ulangan.</figcaption></figure>
  <div class="dua">
    <div class="kartu"><h3>Kenapa undersampling merusak</h3>
      <p>Hanya {pct(N_LATIH_UNDER / N_LATIH_BAWAAN)} window latih yang tersisa. Model jarang melihat ritme normal, terutama yang
        cepat, sehingga {pct(pita_tinggi, 0)} ritme normal {esc(PITA[1])} disebut AFIB/AFL (bawaan {pct(pita_rendah, 0)}).</p></div>
    <div class="kartu"><h3>Kenapa oversampling merusak</h3>
      <p>Setiap window AFL disalin ±{fnum(SALIN_AFL, 0)}×. Model menghafal salinan yang sama, dan AFL palsu naik dari
        {thou(R[b]["palsu"]["total"])} ke {thou(R["over"]["palsu"]["total"])} per ulangan.</p></div>
  </div>
  <p class="catatan">Jawaban untuk BiGRU: pakai pembobotan kelas, bukan balancing. Hasil ini tidak berlaku untuk semua model. Di XGBoost
    (16 fitur HRV), undersampling justru lebih baik dari bawaannya ({sg(XGB_UNDER[0])}, P = {fp(XGB_UNDER[3])}). Jadi setiap
    anggota tim perlu menguji balancing pada modelnya sendiri, dengan opsi <code>balancing</code> yang sama.</p>
</section>

<section id="pasien">
  <span class="eyebrow">Per pasien</span>
  <h2>Recall AFL di tujuh pasien AFL</h2>
  <p>Recall = bagian window AFL pasien itu yang tertebak AFL. Pasien 08378 adalah satu-satunya contoh AFL 2:1 yang besar.
    Dengan bobot lunak, recall pasien ini turun ke {fnum(afl08378[0], 2)}. Bobot penuh memulihkannya ke {fnum(afl08378[1], 2)}.</p>
  <div class="gulir">{tabel_pasien()}</div>
  <p class="muted">Pasien 04043 hanya punya {E1["afl_window"]["04043"]} window AFL, jadi angkanya tidak bermakna. 07910 dan 08215
    hampir tidak terdeteksi oleh semua sistem di tabel, termasuk v3.1.</p>
</section>

<section id="tanpa-04936">
  <span class="eyebrow">Uji ketahanan</span>
  <h2>Apa yang terjadi tanpa pasien 04936</h2>
  <p>Kalau hasil sebuah setelan runtuh saat satu pasien dibuang, model itu belum bisa dipercaya untuk pasien baru. Bobot lunak
    (bawaan) dan learning rate kecil runtuh. Bobot penuh tetap dekat v3.1.</p>
  <figure><div class="kanvas">{svg_dumbbell()}</div>
    <figcaption>Bias kelas dihitung ulang tanpa pasien 04936 ({E1["tanpa"]["n_pasien"]} pasien, {E1["tanpa"]["n_pasien_afl"]}
      pasien AFL, {thou(E1["tanpa"]["n_afl"])} window AFL).</figcaption></figure>
</section>

<section id="detak-cepat">
  <span class="eyebrow">Uji stres takikardia</span>
  <h2>Ritme normal yang cepat masih jadi alarm</h2>
  <p>AFDB hampir tidak punya ritme normal yang cepat: hanya {thou(N_NORMAL_CEPAT)} window Normal di atas 120 bpm. Karena itu
    setiap model diuji juga dengan ritme teratur buatan. Ambang = detak di mana model berhenti menyebut Normal. Kolom pita
    detak = persentase window Normal asli yang disebut AFIB/AFL.</p>
  <div class="gulir">{tabel_takikardia()}</div>
  <p class="catatan">Ambang dari {N_SET} setelan BiGRU berada di {AMB_MIN}–{AMB_MAKS} bpm (v3.1: {E1["paket_v31"]}). Yang terendah
    {", ".join(kode(s) for s in AMB_MIN_S)}, yang tertinggi {", ".join(kode(s) for s in AMB_MAKS_S)}. Pada ritme normal di atas
    120 bpm, kandidat {kode(KANDIDAT)} masih menyebut {pct(R[KANDIDAT]["pita"][PITA_TINGGI][1], 0)} sebagai AFIB/AFL (v3.1
    {pct(E1["v31"]["pita"][PITA_TINGGI][1], 0)}). Tuning hanya menggeser ambang beberapa bpm. Data eksternal (v8–v10) juga belum
    menyelesaikannya. Yang dibutuhkan adalah rekaman perangkat SIHEDAF yang berlabel.</p>
</section>

<section id="kesimpulan">
  <span class="eyebrow">Kesimpulan</span>
  <h2>Kandidat BiGRU dan langkah berikutnya</h2>
  <div class="dua">
    <div class="kartu"><h3>Kandidat: <code>{{"weight_power": 1.0}}</code></h3><ul>
      <li>Macro-F1 {fnum(M(w))} {ci(R[w]["bias"]["Macro-F1"])}, setara v3.1 (P = {fp(v31_vs(w)[3])}).</li>
      <li>Tanpa 04936: {fnum(TANPA[w]["Macro-F1"][0])} (v3.1 {fnum(TANPA["v31"]["Macro-F1"][0])}).</li>
      <li>Lebih baik dari v1 walau bias dipilih ulang di setiap sampel (bootstrap bersarang, P = {fp(nested_w["p"])}).
        Untuk bawaan, P = {fp(nested_b["p"])}.</li>
      <li>Resepnya sama dengan BiGRU v3, tetapi kini bobot modelnya tersimpan, sehingga bisa diuji takikardia.</li></ul></div>
    <div class="kartu"><h3>Yang diusulkan</h3><ul>
      <li>BiGRU tidak menggantikan paket produksi v3.1, karena hasilnya setara, bukan lebih baik.</li>
      <li>Tuning BiGRU dihentikan. Setelan tambahan hanya bergeser di dalam rentang ketidakpastian.</li>
      <li>Bandingkan BiGRU dengan model tim lain di <code>gabungan.ipynb</code> setelah semua model selesai dilatih.</li>
      <li>Langkah yang paling mungkin membantu: menambah pasien, terutama rekaman perangkat SIHEDAF berlabel.</li></ul></div>
  </div>
</section>

<section id="batasan">
  <span class="eyebrow">Batasan</span>
  <h2>Yang belum bisa disimpulkan</h2>
  <ul class="jawab">
    <li><span>Hanya {N_PASIEN} pasien dan {N_PASIEN_AFL} pasien AFL, sehingga CI lebar dan banyak hasil "tidak terbedakan".</span></li>
    <li><span>{kode("wp100")} dipilih setelah melihat hasil. Resepnya memang sudah ada sejak v3, tetapi pada data penuh ia tetap
      tidak terbedakan dari bawaan (P = {fp(B_WP100[3])}).</span></li>
    <li><span>{N_KAGGLE} setelan dilatih di Kaggle (GPU T4), sisanya di PC. Selisih sekecil ±0,003 bisa berasal dari beda mesin.</span></li>
    <li><span>Semua angka dari AFDB. Ini bukan uji pada perangkat asli, dan masalah ritme normal yang cepat belum selesai.</span></li>
    <li><span>Dropout dan GRU dua lapis belum dicoba, karena butuh mengubah arsitektur. Karena regularisasi lain tidak membantu,
      peluangnya kecil.</span></li>
  </ul>
</section>

<section id="lampiran">
  <span class="eyebrow">Lampiran</span>
  <h2>Semua setelan</h2>
  <p>Macro-F1 memakai bias kelas cross-fit. Referensi eksperimen 1 dan 2 adalah {kode("bawaan")}, referensi eksperimen 3 adalah
    {kode("wp100")}.</p>
  <div class="gulir">{tabel_lampiran()}</div>
  <p class="muted">Sumber: <code>v11_model_tim/bigru/eksperimen.py</code> (definisi dan training), <code>cache/v11/bigru/</code>
    (hasil per fold). Laporan lengkap: <code>laporan_bigru_e1_balancing.pdf</code>, <code>laporan_bigru_e2_tuning.pdf</code>,
    <code>laporan_bigru_e3_regularisasi.pdf</code> di <code>v11_model_tim/bigru/laporan/</code>.</p>
</section>
</main>

<footer><div class="wrap">Dibuat oleh <code>v11_model_tim/bigru/presentasi/build_presentasi.py</code> dari cache eksperimen.
  Pemilihan setelan otomatis (cross-fit) di eksperimen 2 memberi {fnum(CF2)} dan di eksperimen 3 {fnum(CF3)}.</div></footer>
<script>{JS}</script>
"""


def N_SET_E2_PC():
    return sum(not R[s]["tempat"].startswith("Kaggle") for s in E["e2"]["baru"])


def N_SET_E2_KAGGLE():
    return sum(R[s]["tempat"].startswith("Kaggle") for s in E["e2"]["baru"])


def kepala():
    return (f'<title>BiGRU SIHEDAF v11</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'<link rel="preconnect" href="https://fonts.googleapis.com">'
            f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            f'<link rel="stylesheet" href="{FONTS}">\n<style>{CSS}</style>\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, help="tulis potongan halaman (tanpa <html>/<head>) ke path ini")
    a = ap.parse_args()
    isi = halaman()
    if a.artifact:
        a.artifact.write_text(kepala() + isi, encoding="utf-8")
        print(a.artifact)
        return
    out = HERE / "presentasi_bigru.html"
    out.write_text('<!doctype html>\n<html lang="id">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                   + kepala() + "</head>\n<body>\n" + isi + "\n</body>\n</html>\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
