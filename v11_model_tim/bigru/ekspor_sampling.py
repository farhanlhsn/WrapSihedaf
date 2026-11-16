"""
ekspor_sampling.py — CSV hasil sampling BiGRU (undersampling / oversampling per fold) untuk dibagikan ke luar repo.

    python ekspor_sampling.py          # → cache/v11/bigru/sampling_csv/ + sampling_csv.zip

Indeks window diambil dengan memanggil bersama.balancing.terapkan (fungsi yang sama dengan runner) memakai seed latih
protokol, jadi isinya sama dengan data latih yang dipakai training. Setelah ditulis, CSV dibaca ulang dan data latih setiap
(ulangan, fold, metode) dibangun lagi dengan contoh_pakai.py, lalu di-assert sama dengan keluaran balancing.terapkan pada data
asli, dan jumlah per kelas sama dengan info["n_latih"] di cache hasil training.
"""
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent                       # v11_model_tim/bigru
sys.path[:0] = [str(HERE.parent), str(HERE)]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bersama import balancing as B, data as D  # noqa: E402
import hrv_core  # noqa: E402

OUT = D.lokasi_hasil() / "bigru" / "sampling_csv"
HASIL = D.lokasi_hasil() / "bigru"
KELAS = np.array(["Normal", "AFIB", "AFL"])
METODE = ("under", "over")
SETELAN = {"bawaan": None, "under": "under", "over": "over", "wp100": None}   # setelan BiGRU yang dirangkum
VONIS = lambda p: "lebih baik" if p >= 0.95 else "lebih buruk" if p <= 0.05 else "tidak terbedakan"


def indeks(d, r, k, metode):
    """Indeks global window latih fold (r, k) setelah balancing, urutan sama dengan keluaran balancing.terapkan."""
    tr = np.flatnonzero(d.folds[r] != k)
    ns = SimpleNamespace(X=tr, M=tr, F=tr, y=d.y[tr])         # array indeks ikut dipotong/disalin persis seperti data
    return B.terapkan(ns, metode, D.seed_latih(r, k), "urutan").X


def tulis_window(d):
    n = d.M.sum(1).astype(int)
    ibi = [";".join(str(v) for v in d.X[i, :n[i], 0]) for i in range(len(d.y))]
    kolom = {"window_id": np.arange(len(d.y)), "pasien": d.records, "t_mulai_s": d.t.astype(np.float32),
             "label": d.y, "kelas": KELAS[d.y], "n_detak": n}
    kolom |= {f: d.F[:, j] for j, f in enumerate(hrv_core.FEATURE_NAMES)}
    kolom |= {f"fold_u{r}": d.folds[r] for r in range(D.N_ULANGAN)}
    kolom["ibi_ms"] = ibi
    pd.DataFrame(kolom).to_csv(OUT / "window.csv", index=False)


def tulis_sampling(d):
    under, over, jumlah = [], [], []
    for r in range(D.N_ULANGAN):
        for k in range(D.N_FOLD):
            tr = np.flatnonzero(d.folds[r] != k)
            iu, io = indeks(d, r, k, "under"), indeks(d, r, k, "over")
            under.append(pd.DataFrame({"ulangan": r, "fold": k, "window_id": iu}))
            muncul = np.bincount(io, minlength=len(d.y))
            ada = np.flatnonzero(muncul > 1)
            over.append(pd.DataFrame({"ulangan": r, "fold": k, "window_id": ada, "muncul": muncul[ada]}))
            for metode, idx in (("bawaan", tr), ("under", iu), ("over", io)):
                n = np.bincount(d.y[idx], minlength=3)
                jumlah.append(dict(ulangan=r, fold=k, metode=metode, n_normal=n[0], n_afib=n[1], n_afl=n[2],
                                   total=n.sum(), pasien_uji=" ".join(np.unique(d.records[d.folds[r] == k]))))
    pd.concat(under).to_csv(OUT / "sampling_under.csv", index=False)
    pd.concat(over).to_csv(OUT / "sampling_over.csv", index=False)
    pd.DataFrame(jumlah).to_csv(OUT / "jumlah_per_fold.csv", index=False)


def tulis_hasil():
    """Ringkasan hasil BiGRU per setelan sampling, dari JSON laporan (bukan disalin manual)."""
    E = [json.loads((HERE / "laporan" / "_build" / f"e{i}.json").read_text(encoding="utf-8")) for i in (1, 2, 3)]
    R, banding = {}, {}
    for e in E:
        for s, r in e["R"].items():
            assert s not in R or R[s]["bias"] == r["bias"], f"{s} beda antara eksperimen"
            R.setdefault(s, r)
        banding |= {k: v["banding"] for k, v in e["banding"].items()}
    v31 = E[1]["v31"]["metrik"]
    baris = [dict(setelan="v3.1 (produksi, referensi)", balancing="", weight_power="",
                  **{f"{m}{x}": v31[m][i] for m in ("Macro-F1", "AFL R", "AFL F1") for i, x in enumerate(("", " bawah", " atas"))})]
    for s, bal in SETELAN.items():
        m = R[s]["bias"]
        wp = 1.0 if s == "wp100" else 0.5
        row = dict(setelan=s, balancing=bal or "tanpa", weight_power=wp,
                   **{f"{k}{x}": m[k][i] for k in ("Macro-F1", "AFL R", "AFL F1") for i, x in enumerate(("", " bawah", " atas"))})
        for ref, b in (("bawaan", banding.get(f"{s}_vs_bawaan")), ("v3.1", R[s]["v31"])):
            if b is None:
                continue
            for k in ("Macro-F1", "AFL R"):
                dlt, lo, hi, p = b[k]
                row |= {f"Δ {k} vs {ref}": dlt, f"Δ {k} vs {ref} bawah": lo, f"Δ {k} vs {ref} atas": hi,
                        f"P {k} vs {ref}": p, f"vonis {k} vs {ref}": VONIS(p)}
        baris.append(row)
    df = pd.DataFrame(baris)
    df.columns = [c.replace("Macro-F1", "macro_f1").replace("AFL R", "afl_recall").replace("AFL F1", "afl_f1")
                  .replace("Δ ", "delta_").replace("P ", "p_").replace(" vs ", "_vs_").replace(" bawah", "_ci_bawah")
                  .replace(" atas", "_ci_atas").replace("vonis ", "vonis_").replace("v3.1", "v31") for c in df.columns]
    df.round(4).to_csv(OUT / "hasil_bigru_sampling.csv", index=False)


def verifikasi(d):
    spec = importlib.util.spec_from_file_location("contoh_pakai", OUT / "contoh_pakai.py")
    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    win, under, over = cp.muat(OUT)
    assert np.array_equal(win["window_id"].to_numpy(), np.arange(len(d.y)))
    assert np.array_equal(win[hrv_core.FEATURE_NAMES].to_numpy(np.float32), d.F), "fitur tidak sama bit demi bit"
    assert np.array_equal(win["t_mulai_s"].to_numpy(np.float32), d.t.astype(np.float32))
    X, M = cp.urutan_ibi(win)
    assert np.array_equal(X, d.X) and np.array_equal(M, d.M), "urutan IBI tidak sama bit demi bit"
    for r in range(D.N_ULANGAN):
        for k in range(D.N_FOLD):
            tr = np.flatnonzero(d.folds[r] != k)
            asli = SimpleNamespace(X=d.X[tr], M=d.M[tr], F=d.F[tr], y=d.y[tr])
            uji = cp.data_uji(win, r, k)
            assert np.array_equal(uji["window_id"].to_numpy(), np.flatnonzero(d.folds[r] == k))
            for metode in METODE:
                latih = cp.data_latih(win, under, over, r, k, metode)
                ref = B.terapkan(asli, metode, D.seed_latih(r, k), "urutan")
                F = latih[hrv_core.FEATURE_NAMES].to_numpy(np.float32)
                urut = lambda a: a[np.lexsort(a.T[::-1])]
                assert np.array_equal(urut(F), urut(ref.F)) and np.array_equal(np.sort(latih["label"]), np.sort(ref.y)), \
                    f"data latih {metode} u{r} k{k} beda"
                info = json.loads(str(np.load(HASIL / metode / f"r{r}_k{k}.npz")["info"]))
                assert np.bincount(latih["label"], minlength=3).tolist() == info["n_latih"], (metode, r, k)
            info = json.loads(str(np.load(HASIL / "bawaan" / f"r{r}_k{k}.npz")["info"]))
            assert np.bincount(d.y[tr], minlength=3).tolist() == info["n_latih"], ("bawaan", r, k)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for nama in ("BACA_SAYA.txt", "contoh_pakai.py"):
        shutil.copy(HERE / "sampling_csv" / nama, OUT / nama)
    d = D.muat()
    tulis_window(d)
    tulis_sampling(d)
    tulis_hasil()
    verifikasi(d)
    zp = OUT.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(OUT.iterdir()):
            zf.write(f, f"sampling_csv/{f.name}")
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:28s} {f.stat().st_size / 1e6:7.2f} MB")
    print(f"{zp} {zp.stat().st_size / 1e6:.1f} MB — verifikasi lolos")


if __name__ == "__main__":
    main()
