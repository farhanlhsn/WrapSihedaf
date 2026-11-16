"""Angka dan gambar paper 1 (ICIMPACT 2027), semuanya dihitung dari cache, tidak disalin manual.

Sumber:
  cache/publikasi/dekomposisi_v1/ringkasan.json   dekomposisi langkah A–J (publikasi/dekomposisi_v1.py; di naskah A–I)
  v3_evaluasi_robust/laporan/_build/tables.json   hasil v3 (referensi, kandidat, grid bias, recall AFL per pasien)
  cache/v11/<model>/<setelan>/ringkasan.json      model tim v11, banding terhadap v3.1
  cache/v3_2/ringkasan.json                       bootstrap bersarang dan sensitivitas tanpa 04936
  v1_lama/model.ipynb                             skor per fold yang dilaporkan versi lama

Jalankan dari root repo:  python publikasi/paper1/angka.py
Keluaran: publikasi/paper1/angka.json, angka.tex (makro LaTeX, termasuk baris Tabel I), gambar/*.pdf (+ .png pratinjau).
"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "gambar"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#172B43", "#526477", "#d9dde3"

V11 = [("bigru", "bawaan", "BiGRU"), ("tcn", "bawaan", "TCN"), ("inceptiontime", "bawaan", "InceptionTime"),
       ("minirocket", "bawaan", "MiniRocket"), ("xgboost", "bawaan", "XGBoost"), ("svm", "bawaan", "SVM"),
       ("bigru", "wp100", "BiGRU"), ("tcn", "wp100", "TCN"), ("inceptiontime", "wp100", "InceptionTime")]
V3 = [("LightGBM +bias", "LightGBM"), ("MaskedResNet1D +bias", "ResNet1D"), ("BiGRUAttention +bias", "BiGRU")]
AFL_PATIENTS = ["04043", "04908", "04936", "06426", "07910", "08215", "08378"]


def load(p):
    return json.loads((ROOT / p).read_text())


def vonis(p):
    return "better" if p >= 0.95 else "worse" if p <= 0.05 else "n.s."


def log_v1_per_fold():
    """Macro-F1 per fold MicroTransformer B1 dari output model.ipynb (blok antara ✓ ResNet dan ✓ Transformer)."""
    nb = load("v1_lama/model.ipynb")
    txt = "".join("".join(o.get("text", "")) for c in nb["cells"] for o in c.get("outputs", []))
    seg = txt.split("✓ B1 / 1D-ResNet")[1].split("✓ B1 / Micro-Transformer")[0]
    mf = [float(x) for x in re.findall(r"Macro-F1=([0-9.]+)", seg)]
    afl = [float(x) for x in re.findall(r"AFL-Rec=([0-9.]+)", seg)]
    assert len(mf) == 5 and len(afl) == 5, (mf, afl)
    return mf, afl


def kumpulkan():
    dek = load("cache/publikasi/dekomposisi_v1/ringkasan.json")
    v3 = load("v3_evaluasi_robust/laporan/_build/tables.json")
    v32 = load("cache/v3_2/ringkasan.json")
    L = dek["langkah"]

    # dekomposisi harus sama dengan laporan v3 (model fold identik)
    old = v3["old"]["rows"]
    for k, key in (("E", "lama lab"), ("I", "lama det"), ("J", "lama fe")):
        assert abs(L[k]["Macro-F1"][0] - old[key]["Macro-F1"][0]) < 1e-9, k
    assert abs(L["A"]["Macro-F1"][0] - 0.8257) < 1e-9
    assert all(v["argmax_sama"] == 1.0 for v in dek["paritas"].values())

    # partisi B = partisi asli model.ipynb: skor per fold harus berdekatan
    log_mf, log_afl = log_v1_per_fold()
    b_mf = L["B"]["per_fold"]
    assert max(abs(a - b) for a, b in zip(log_mf, b_mf)) < 0.03, (log_mf, b_mf)

    res = {r["key"]: r for r in v3["results"]}
    paired = {r["key"]: r for r in v3["paired"]}
    ref = res["MicroTransformer +bias"]
    assert abs(ref["Macro-F1"][0] - v32["nested"]["v31_point"]) < 1e-9

    model = []
    for key, nama in V3:
        d = paired[key]["Macro-F1"]
        model.append(dict(grup="v3", nama=nama, delta=d[0], lo=d[1], hi=d[2], p=d[3], mf=res[key]["Macro-F1"][0]))
    for m, s, nama in V11:
        r = load(f"cache/v11/{m}/{s}/ringkasan.json")
        d = r["banding_v31"]["Macro-F1"]
        model.append(dict(grup="soft" if s == "bawaan" else "full", nama=nama, delta=d[0], lo=d[1], hi=d[2], p=d[3],
                          mf=r["metrik"]["bias"]["Macro-F1"][0]))
    for x in model:
        x["vonis"] = vonis(x["p"])
    assert not any(x["vonis"] == "better" for x in model), "ada model yang lebih baik dari referensi"

    gc = {g["model"]: g for g in v3["grid_cmp"]}["MicroTransformer"]
    i04936 = AFL_PATIENTS.index("04936")
    # recall AFL tertinggi di antara semua konfigurasi (+bias) untuk dua rekaman yang selalu gagal
    rec_max = {}
    for p in ("07910", "08215"):
        vals = [v3["afl_recall"][k]["mean"][AFL_PATIENTS.index(p)] for k in ("LightGBM", "MaskedResNet1D", "BiGRUAttention")]
        vals += [load(f"cache/v11/{m}/{s}/ringkasan.json")["afl_per_pasien"]["bias"][p] for m, s, _ in V11]
        rec_max[p] = max(vals)
    pairs = v3["old"]["pairs"]
    bias_share = pairs["lama+bias det vs lama det"]["Macro-F1"][0] / pairs["sekarang det vs lama det"]["Macro-F1"][0]
    return dict(
        rec_max=rec_max, bias_share=bias_share,
        langkah=L, demo=dek["demo_partisi"], jumlah=dek["jumlah_window"],
        log_v1=dict(mf=log_mf, afl=log_afl), b_per_fold=b_mf,
        ref=dict(mf=ref["Macro-F1"], afl_f1=ref["AFL F1"]), model=model,
        grid=dict(lama=gc["Macro-F1|lama"], baru=gc["Macro-F1|baru"], bias_afl=v3["bias_afl"]["MicroTransformer"]),
        afl_pasien=dict(zip(AFL_PATIENTS, v3["afl_recall"]["MicroTransformer"]["mean"])),
        afl_window=dict(zip(AFL_PATIENTS, v3["afl_windows"])),
        afl_share_04936=v3["afl_windows"][i04936] / sum(v3["afl_windows"]),
        normal_r=dict(p1=v32["sens_all"]["rows"]["v2_prod"]["Normal R"][0], p2=v32["sens_all"]["rows"]["v31"]["Normal R"][0]),
        robust=dict(semua=v32["sens_all"]["pairs_v31_minus"], tanpa=v32["sens_wo"]["pairs_v31_minus"],
                    bersarang=dict(v1=v32["nested"]["v1"], v2=v32["nested"]["v2_prod"])),
        mf_p1=v32["sens_all"]["rows"]["v2_prod"]["Macro-F1"], n_pasien=v32["sens_all"]["n_pasien"],
    )


# ── LaTeX ───────────────────────────────────────────────────────────────
# Huruf langkah di naskah dibuat berurutan (dekomposisi_v1.py tidak punya langkah C). Nama makro tetap memakai huruf cache.
TAMPIL = dict(zip("ABDEFGHIJ", "ABCDEFGHI"))


def n(x, d=3):
    s = f"{x:.{d}f}"
    return f"\\ensuremath{{{s.replace('-', '-')}}}"


def ci(lo, hi, d=2):
    return f"[{lo:.{d}f}, {hi:.{d}f}]".replace("-", "$-$")


def sd(x, d=3):
    if abs(x) < 0.5 * 10 ** -d:
        return f"{0:.{d}f}"
    return f"{x:+.{d}f}".replace("-", "$-$").replace("+", "$+$")


def tulis_tex(A):
    L, m = A["langkah"], {}
    for k in "ABD":
        m[f"mf{k}"] = n(L[k]["Macro-F1"][0])
        m[f"sd{k}"] = n(L[k]["sd_antar_fold"])
    for k in "EFGHIJ":
        m[f"mf{k}"] = n(L[k]["Macro-F1"][0])
        m[f"ci{k}"] = ci(*L[k]["Macro-F1"][1:])
        m[f"aflP{k}"] = n(L[k]["AFL P"][0])
        m[f"aflR{k}"] = n(L[k]["AFL R"][0])
        if "Δ Macro-F1" in L[k]:
            d = L[k]["Δ Macro-F1"]
            m[f"d{k}"] = sd(d[0])
            m[f"dci{k}"] = ci(d[1], d[2], 3)
            m[f"dp{k}"] = f"{d[3]:.3f}"
    m["dWindows"] = sd(L["G"]["Macro-F1"][0] - L["E"]["Macro-F1"][0])
    m["dPartition"] = sd(L["D"]["Macro-F1"][0] - L["B"]["Macro-F1"][0])
    m["dPooled"] = sd(L["E"]["Macro-F1"][0] - L["D"]["Macro-F1"][0])
    # besaran mutlak untuk kalimat seperti "cost 0.016" / "lowered by 0.087"
    mutlak = {"F": L["F"]["Macro-F1"][0] - L["E"]["Macro-F1"][0], "G": L["G"]["Macro-F1"][0] - L["F"]["Macro-F1"][0],
              "I": L["I"]["Macro-F1"][0] - L["H"]["Macro-F1"][0], "J": L["J"]["Macro-F1"][0] - L["I"]["Macro-F1"][0],
              "Windows": L["G"]["Macro-F1"][0] - L["E"]["Macro-F1"][0],
              "Partition": L["D"]["Macro-F1"][0] - L["B"]["Macro-F1"][0],
              "Pooled": L["E"]["Macro-F1"][0] - L["D"]["Macro-F1"][0]}
    for k, v in mutlak.items():
        m[f"ad{k}"] = n(abs(v))
    m["maxRecE"] = n(A["rec_max"]["07910"], 2)
    m["maxRecF"] = n(A["rec_max"]["08215"], 2)
    m["biasShare"] = f"{100 * A['bias_share']:.0f}"
    m["nSeeds"] = str(A["demo"]["det"]["n_seed"])
    m["nPartitions"] = str(A["demo"]["det"]["partisi_berbeda"])
    m["sklearnVersion"] = A["demo"]["sklearn"]
    m["nAflLab"] = f"{A['jumlah']['lab']['kelas'][2]:,}"
    m["nAflThin"] = f"{A['jumlah']['lab_jarang']['kelas'][2]:,}"
    m["nAflDet"] = f"{A['jumlah']['det']['kelas'][2]:,}"
    for kunci, nama in (("lab", "Lab"), ("det", "Det")):
        for i, kls in enumerate(("Norm", "Af")):
            m[f"n{kls}{nama}"] = f"{A['jumlah'][kunci]['kelas'][i]:,}"
    m["refMF"] = n(A["ref"]["mf"][0])
    m["refCI"] = ci(*A["ref"]["mf"][1:])
    m["refAflF"] = n(A["ref"]["afl_f1"][0])
    m["refAflCI"] = ci(*A["ref"]["afl_f1"][1:])
    m["mfPone"] = n(A["mf_p1"][0])
    m["ciPone"] = ci(*A["mf_p1"][1:])
    m["gridOld"] = n(A["grid"]["lama"])
    m["gridNew"] = n(A["grid"]["baru"])
    b = A["grid"]["bias_afl"]
    m["biasAflMed"] = f"$-${abs(b[1]):.1f}".replace(".0", ".0")
    m["biasAflRange"] = f"$-${abs(b[0]):.1f} to $-${abs(b[2]):.1f}"
    m["normalRone"] = n(A["normal_r"]["p1"], 2)
    m["normalRtwo"] = n(A["normal_r"]["p2"], 2)
    m["share"] = f"{100 * A['afl_share_04936']:.0f}"
    m["nPatients"] = str(A["n_pasien"])
    for p, r in A["afl_pasien"].items():
        m[f"rec{chr(65 + AFL_PATIENTS.index(p))}"] = n(r, 2)
    mods = A["model"]
    m["nWorse"] = str(sum(x["vonis"] == "worse" for x in mods))
    m["nNs"] = str(sum(x["vonis"] == "n.s." for x in mods))
    m["nConfig"] = str(len(mods))
    for key, x in (("BigruSoft", mods[3]), ("BigruFull", mods[9]), ("TcnFull", mods[10]), ("TcnSoft", mods[4])):
        m[f"mf{key}"] = n(x["mf"])
        m[f"d{key}"] = sd(x["delta"])
        m[f"p{key}"] = f"{x['p']:.2f}"
    R = A["robust"]
    for nama, (dd, k) in {"FixOne": (R["semua"]["v1"], 0), "FixTwo": (R["semua"]["v2_prod"], 0),
                          "WoOne": (R["tanpa"]["v1"], 0), "WoTwo": (R["tanpa"]["v2_prod"], 0)}.items():
        m[f"d{nama}"] = sd(dd[0])
        m[f"ci{nama}"] = ci(dd[1], dd[2], 3)
        m[f"p{nama}"] = f"{dd[3]:.2f}"
    for nama, dd in (("NestOne", R["bersarang"]["v1"]), ("NestTwo", R["bersarang"]["v2"])):
        m[f"d{nama}"] = sd(dd["delta"])
        m[f"ci{nama}"] = ci(*dd["ci"], 3)
        m[f"p{nama}"] = f"{dd['p']:.3f}"
    rows = [
        ("A", "Reported in the original notebook: one partition, per-fold mean, P0 windows", f"\\mfA{{}} $\\pm$ \\sdA", "--"),
        ("B", "Re-run, same partition and recipe", f"\\mfB{{}} $\\pm$ \\sdB", sd(L["B"]["Macro-F1"][0] - L["A"]["Macro-F1"][0])),
        ("D", "5$\\times$5 patient-level folds, per-fold mean", f"\\mfD{{}} $\\pm$ \\sdD", "\\dPartition"),
        ("E", "Pooled out-of-fold predictions, P0 windows", "\\mfE{} \\ciE", "\\dPooled"),
        ("F", "AFL test windows thinned from 1-s to $\\geq$5-s stride", "\\mfF{} \\ciF", "\\dF{} \\dciF"),
        ("G", "AF test windows weighted $\\times$2 (emulates 5-s stride)", "\\mfG{} \\ciG", "\\dG{} \\dciG"),
        ("H", "P1 window extraction, annotation R-peaks (250\\,Hz)", "\\mfH{} \\ciH", "\\dH{} \\dciH"),
        ("I", "P1 window extraction, 50\\,Hz detector R-peaks", "\\mfI{} \\ciI", "\\dI{} \\dciI"),
        ("J", "Originally deployed peak-finding front-end, no quality gate", "\\mfJ{} \\ciJ", "\\dJ{} \\dciJ"),
    ]
    body = []
    for k, cond, mf, d in rows:
        afl = f"\\aflP{k} & \\aflR{k}" if k in "EFGHIJ" else "-- & --"
        body.append(f"{TAMPIL[k]} & {cond} & {mf} & {d} & {afl} \\\\")
    # baris tabel sebagai makro: \input di dalam tabular memicu "Misplaced \noalign" sebelum \bottomrule
    m["tabelDekomposisi"] = "\n" + "\n".join(body) + "\n"
    lines = ["% Dibuat otomatis oleh angka.py dari cache. Jangan diedit manual."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (HERE / "angka.tex").write_text("\n".join(lines) + "\n")


# ── Gambar ──────────────────────────────────────────────────────────────
def gaya():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    nama = {f.name for f in font_manager.fontManager.ttflist}
    fam = next((f for f in ("Arial", "Liberation Sans") if f in nama), "DejaVu Sans")
    plt.rcParams.update({"font.family": fam, "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                         "xtick.major.width": 0.6, "ytick.major.width": 0, "axes.titlesize": 7.5})
    return plt


def simpan(fig, nama):
    # ukuran tetap selebar kolom IEEE (3,5 in), tanpa bbox "tight", supaya huruf 7 pt tidak ikut mengecil di naskah
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{nama}.pdf")
    fig.savefig(FIG / f"{nama}.png", dpi=220)


def gambar_dekomposisi(A):
    plt = gaya()
    L = A["langkah"]
    keys = list("ABDEFGHIJ")
    teks = {"A": "Reported", "B": "Re-run", "D": "5×5 patient folds", "E": "Pooled OOF", "F": "AFL stride ≥5 s",
            "G": "AF weight ×2", "H": "P1, annot. R-peaks", "I": "P1, 50 Hz detector", "J": "Deployed front-end"}
    label = {k: f"{TAMPIL[k]}  {v}" for k, v in teks.items()}
    fig = plt.figure(figsize=(3.5, 2.45))
    ax = fig.add_axes([0.37, 0.15, 0.51, 0.70])
    y = np.arange(len(keys))[::-1]
    prev = None
    for yi, k in zip(y, keys):
        v = L[k]["Macro-F1"]
        if k in "ABD":
            ax.errorbar(v[0], yi, xerr=L[k]["sd_antar_fold"], fmt="s", ms=4.5, color=ORANGE, ecolor=ORANGE,
                        elinewidth=1, capsize=0, zorder=3)
        else:
            ax.errorbar(v[0], yi, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=4.5, color=BLUE, ecolor=BLUE,
                        elinewidth=1, capsize=0, zorder=3)
        if prev is not None:
            d = v[0] - prev
            teks = "0.000" if abs(d) < 0.0005 else f"{d:+.3f}".replace("-", "−")
            ax.text(1.005, yi, teks, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.5,
                    color=INK)
        prev = v[0]
    ax.axvline(A["ref"]["mf"][0], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(A["ref"]["mf"][0] + 0.012, 0, "P2\nreference", ha="left", va="center", fontsize=6, color=MUTED)
    ax.set_yticks(y, [label[k] for k in keys])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlabel("Macro-F1")
    ax.text(1.005, len(keys) - 0.45, "Δ", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=6.5,
            color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    h1 = Line2D([0], [0], marker="s", ms=4.5, color=ORANGE, lw=1, label="per-fold mean ± SD")
    h2 = Line2D([0], [0], marker="o", ms=4.5, color=BLUE, lw=1, label="pooled, 95% patient-bootstrap CI")
    ax.legend(handles=[h1, h2], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False, fontsize=6,
              handletextpad=0.3, borderaxespad=0.1)
    simpan(fig, "dekomposisi")
    plt.close(fig)


def gambar_forest(A):
    plt = gaya()
    grup = {"v3": "Screening (full weights)", "soft": "Soft class weights", "full": "Full class weights"}
    rows, ylab = [], []
    for g in ("v3", "soft", "full"):
        rows.append(("hdr", grup[g]))
        rows += [("m", x) for x in A["model"] if x["grup"] == g]
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(3.5, 2.7))
    ax = fig.add_axes([0.40, 0.13, 0.47, 0.79])
    y = np.arange(len(rows))[::-1]
    for yi, (t, x) in zip(y, rows):
        if t == "hdr":
            continue
        worse = x["vonis"] == "worse"
        c, mk = (ORANGE, "s") if worse else (BLUE, "o")
        ax.errorbar(x["delta"], yi, xerr=[[x["delta"] - x["lo"]], [x["hi"] - x["delta"]]], fmt=mk, ms=4.2, color=c,
                    ecolor=c, elinewidth=1, capsize=0, zorder=3)
        ax.text(1.005, yi, "P < 0.01" if x["p"] < 0.005 else f"P = {x['p']:.2f}",transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=6, color=MUTED)
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    ax.set_xlim(-0.28, 0.06)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(y, [x if t == "hdr" else "    " + x["nama"] for t, x in rows])
    ax.tick_params(axis="y", length=0)
    for lab, (t, _) in zip(ax.get_yticklabels(), rows):
        if t == "hdr":
            lab.set_fontweight("bold")
    ax.set_xlabel("Δ Macro-F1 vs recalibrated reference (95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    h1 = Line2D([0], [0], marker="s", ms=4.2, color=ORANGE, lw=1, label="worse (P ≤ 0.05)")
    h2 = Line2D([0], [0], marker="o", ms=4.2, color=BLUE, lw=1, label="not distinguishable")
    ax.legend(handles=[h1, h2], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=6,
              handletextpad=0.3, borderaxespad=0.1)
    simpan(fig, "forest")
    plt.close(fig)


if __name__ == "__main__":
    A = kumpulkan()
    (HERE / "angka.json").write_text(json.dumps(A, indent=1, ensure_ascii=False, default=float))
    tulis_tex(A)
    gambar_dekomposisi(A)
    gambar_forest(A)
    print("model:", [(x["grup"], x["nama"], round(x["delta"], 3), x["vonis"]) for x in A["model"]])
    print("log v1 per fold:", A["log_v1"]["mf"], "| B:", [round(x, 4) for x in A["b_per_fold"]])
    print("OK:", sorted(p.name for p in HERE.iterdir()))
