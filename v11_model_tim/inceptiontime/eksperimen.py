"""
eksperimen.py — definisi, training, dan evaluasi eksperimen InceptionTime v11 (dijalankan pemilik repo). Satu sumber untuk
README ("Hasil", "Hasil 2"). Pola sama dengan ../tcn/eksperimen.py dan ../bigru/eksperimen.py.

    python eksperimen.py latih e2              # semua setelan eksperimen (bisa dilanjutkan; fold yang ada dilewati)
    python eksperimen.py evaluasi e2           # ringkasan.json per setelan + perbandingan → cache/v11/inceptiontime/<banding>

Jalankan dengan cwd = folder ini, sama dengan jalankan.ipynb. Training memakai bersama.runner.jalankan (sama dengan sel
training notebook), evaluasi memakai bersama.evaluasi (sama dengan gabungan.ipynb). Tidak ada metrik yang ditulis ulang.
Selain perbandingan dengan referensi, setiap setelan dibandingkan berpasangan dengan BiGRU dan TCN `wp100` (kandidat v11),
dan dicatat AUROC AFL, bias per fold, dan margin skor AFL (definisi sama dengan ../tcn/laporan/build_report.py).
"""
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # v11_model_tim/inceptiontime
sys.path[:0] = [str(HERE.parent), str(HERE)]
import numpy as np  # noqa: E402
from bersama import data, evaluasi as E, runner  # noqa: E402

# Setiap eksperimen: setelan baru (dilatih), referensi (pembanding berpasangan utama), dan analisis tambahan.
# Nama setelan = NAMA_SETELAN = folder cache/v11/inceptiontime/<nama>/.
EKSPERIMEN = {
    "e1": dict(
        judul="InceptionTime setelan bawaan", tanggal="3 November 2026", banding="banding_e1_bawaan.json",
        referensi="bawaan", setelan={"bawaan": {}}, bersarang=[]),
    "e2": dict(
        judul="Bobot kelas penuh", tanggal="3 November 2026", banding="banding_e2_wp100.json",
        referensi="bawaan",                           # dilatih di GPU yang sama (RTX 5060), jadi tidak perlu bawaan_ulang
        setelan={"bawaan": {}, "wp100": {"weight_power": 1.0}}, bersarang=["wp100"]),
}
LAIN = {"bigru": "wp100", "tcn": "wp100"}          # kandidat v11 model lain (cache/v11/<model>/<setelan>/)


def _log(m, t0=time.time()):
    print(f"[{(time.time() - t0) / 60:5.1f} mnt] {m}", flush=True)


def latih(nama_eks: str, hanya=None) -> None:
    import model as MDL
    for nama, setelan in EKSPERIMEN[nama_eks]["setelan"].items():
        if hanya and nama not in hanya:
            continue
        _log(f"latih {nama} {setelan}")
        runner.jalankan(MDL, nama, setelan, log=_log)


def _margin(d, oof):
    """Margin skor AFL = log p(AFL) − log p kelas lain terbesar (skala yang sama dengan bias kelas): median di window AFL
    08378 (AFL 2:1) dan 04936, dan persentil 99,5 di window non-AFL (calon AFL palsu). Sama dengan laporan TCN."""
    lp = np.log(np.clip(oof, 1e-9, 1))
    mg = lp[..., 2] - lp[..., :2].max(-1)
    pilih = {p: (d.records == p) & (d.y == 2) for p in ("08378", "04936")}
    return {**{p: float(np.median(mg[:, m])) for p, m in pilih.items()},
            "non_afl_p995": float(np.percentile(mg[:, d.y != 2], 99.5))}


def evaluasi(nama_eks: str, n_proses: int = 4) -> Path:
    eks = EKSPERIMEN[nama_eks]
    d = data.muat()
    akar = data.lokasi_hasil() / "inceptiontime"
    lain = {m: data.lokasi_hasil() / m / s for m, s in LAIN.items()}
    ref = eks["referensi"]
    semua = list(dict.fromkeys([ref, *eks["setelan"]]))
    for f in [akar / s for s in semua] + list(lain.values()):
        if len(list(f.glob("r?_k?.npz"))) != 25:
            raise RuntimeError(f"{f} belum lengkap 25 fold; jalankan `python eksperimen.py latih {nama_eks}` dulu.")
    for s in semua:
        if not (akar / s / "ringkasan.json").exists():
            _log(f"laporan_model {s}")
            E.laporan_model(akar / s, d)

    def tugas(t):                                              # dijalankan di proses anak (fork: d diwarisi)
        from sklearn.metrics import roc_auc_score
        jenis, arg = t
        h = lambda s: runner.muat_hasil(akar / s, d)  # noqa: E731
        if jenis == "banding":
            return t, E.banding_setelan(d, akar / arg, akar / ref)
        if jenis == "diagnosa":                               # vs kandidat model lain (+ bias), AUROC, bias, margin
            hs = h(arg)
            assert hs.ulangan == list(range(data.N_ULANGAN))
            p_s, b_s = E.keputusan(hs.oof, d.folds, d.y)
            r = {"auroc_afl": [float(roc_auc_score(d.y == 2, o[:, 2])) for o in hs.oof], "margin": _margin(d, hs.oof),
                 "bias_afl": b_s[:, :, 2].tolist(), "bias_afib": b_s[:, :, 1].tolist(), "banding_lain": {}}
            for m, f in lain.items():
                p_m, _ = E.keputusan(runner.muat_hasil(f, d).oof, d.folds, d.y)
                r["banding_lain"][m] = E.banding(d, p_s, p_m)
            return t, r
        if jenis == "pembanding":                             # metrik, AUROC, dan margin model lain + v3.1, sekali saja
            r = {}
            for m, f in lain.items():
                hm = runner.muat_hasil(f, d)
                p_m, _ = E.keputusan(hm.oof, d.folds, d.y)
                r[m] = {"setelan": f.name, "metrik": E.tabel_metrik(d, {"bias": p_m})["bias"],
                        "afl_per_pasien": E.afl_per_pasien(d, p_m), "afl_palsu": E.afl_palsu(d, p_m),
                        "auroc_afl": [float(roc_auc_score(d.y == 2, o[:, 2])) for o in hm.oof],
                        "margin": _margin(d, hm.oof)}
            r["v31"] = {"auroc_afl": [float(roc_auc_score(d.y == 2, o[:, 2])) for o in d.oof_v31],
                        "margin": _margin(d, d.oof_v31)}
            return t, r
        if jenis == "tanpa":
            oofs = {"v31": d.oof_v31, **{f"{m}_{f.name}": runner.muat_hasil(f, d).oof for m, f in lain.items()},
                    **{s: h(s).oof for s in arg}}
            return t, E.tanpa_04936(d, oofs, list(range(data.N_ULANGAN)))
        if jenis == "bersarang":
            return t, E.bootstrap_bersarang(d, h(arg).oof)
        raise ValueError(jenis)

    daftar = ([("banding", s) for s in semua if s != ref] + [("diagnosa", s) for s in semua]
              + [("pembanding", None), ("tanpa", tuple(semua))] + [("bersarang", s) for s in eks["bersarang"]])
    out = {"eksperimen": nama_eks, "referensi": ref, "setelan": semua, "banding": {}, "diagnosa": {}, "bersarang": {}}
    _log(f"{len(daftar)} tugas, {n_proses} proses")
    with ProcessPoolExecutor(n_proses, mp_context=mp.get_context("fork")) as ex:
        global _TUGAS
        _TUGAS = tugas
        for fut in as_completed([ex.submit(_jalankan_tugas, t) for t in daftar]):
            (jenis, arg), r = fut.result()
            _log(f"selesai {jenis} {arg if jenis in ('banding', 'diagnosa', 'bersarang') else ''}")
            if jenis == "banding":
                out["banding"][f"{arg}_vs_{ref}"] = r
            elif jenis == "diagnosa":
                out["diagnosa"][arg] = r
            elif jenis == "pembanding":
                out["pembanding"] = r
            elif jenis == "tanpa":
                out["tanpa_04936"] = r
            else:
                out["bersarang"][arg] = r
    p = akar / eks["banding"]
    p.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    _log(f"selesai → {p}")
    return p


_TUGAS = None


def _jalankan_tugas(t):
    return _TUGAS(t)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("latih", "evaluasi") or sys.argv[2] not in EKSPERIMEN:
        sys.exit(__doc__)
    os.chdir(HERE)
    if sys.argv[1] == "latih":
        latih(sys.argv[2], set(sys.argv[3:]) or None)
    else:
        evaluasi(sys.argv[2], int(os.environ.get("N_PROSES", "4")))
