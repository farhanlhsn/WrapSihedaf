"""
evaluasi.py — evaluasi standar v11 (protokol AGENTS.md butir 1–6, 9; butir 10 opsional).

Semua perhitungan memakai kode v3/v3.2/v4 lewat impor:
- bias kelas cross-fit grid −7…+3 (`sihedaf_cv.crossfit_bias`);
- metrik pooled OOF rata-rata ulangan + CI 95% bootstrap pasien 2.000× stratified (seed 2026);
- perbandingan berpasangan dengan sampel bootstrap yang sama; vonis P ≥ 0,95 lebih baik, ≤ 0,05 lebih buruk;
- uji stres takikardia v3.2 dengan bias fold masing-masing model fold.
Pembanding v1 dan v3.1 selalu dipotong ke ulangan yang sama dengan kandidat.
"""
from __future__ import annotations

import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np

from . import data as D
from . import runner

cv, T = D.cv, D.T
import pascaproses as P4  # noqa: E402  (v4, sys.path diatur oleh data.py)

AMBANG_V31_PAKET = 115          # v3.2: paket produksi v3.1 berhenti menebak Normal mulai 115 bpm
BATAS_GRID = (float(cv.BIAS_GRID.min()), float(cv.BIAS_GRID.max()))      # (−7, +3)
N_BOOT = 2000
METRIK_BANDING = ("Macro-F1", "AFL F1", "AFL P", "AFL R", "Normal R")
_W: dict = {}


def vonis(p: float) -> str:
    return "lebih baik" if p >= 0.95 else "lebih buruk" if p <= 0.05 else "tidak terbedakan"


def _bobot_boot(d) -> np.ndarray:
    key = id(d)
    if key not in _W:
        _W[key] = cv.bootstrap_weights(d.strata, N_BOOT, seed=D.SEED)
    return _W[key]


def _cms(d, pred) -> np.ndarray:
    """(R, P, 3, 3) confusion matrix per ulangan & pasien."""
    return np.stack([cv.patient_cms(d.y, np.asarray(p, dtype=np.int64), d.records, d.patients) for p in pred])


def keputusan(oof: np.ndarray, folds: np.ndarray, y: np.ndarray):
    """Bias cross-fit per ulangan. Return (prediksi (R, N), bias (R, 5, 3))."""
    pred, bias = [], []
    for o, f in zip(oof, folds):
        adj, b = cv.crossfit_bias(y, o, f)
        pred.append(adj.argmax(1))
        bias.append(b)
    return np.stack(pred), np.stack(bias)


def tabel_metrik(d, pred: dict) -> dict:
    """{nama: {metrik: [titik, CI bawah, CI atas]}}; titik = rata-rata ulangan dari metrik pooled."""
    W = _bobot_boot(d)
    out = {}
    for nama, p in pred.items():
        c = _cms(d, p)
        boot, titik = cv.boot_metrics(c, W), cv.point_metrics(c)
        out[nama] = {m: [float(titik[m].mean()), *map(float, np.percentile(boot[m], [2.5, 97.5]))]
                     for m in cv.METRICS}
    return out


def banding(d, pred_kandidat: np.ndarray, pred_ref: np.ndarray) -> dict:
    """{metrik: [Δ titik, CI bawah, CI atas, P(Δ > 0)]}, sampel bootstrap pasien yang sama."""
    if len(pred_kandidat) != len(pred_ref):
        raise ValueError("Kandidat dan referensi harus memakai ulangan yang sama.")
    W = _bobot_boot(d)
    ck, cr = _cms(d, pred_kandidat), _cms(d, pred_ref)
    bk, br = cv.boot_metrics(ck, W), cv.boot_metrics(cr, W)
    tk, tr = cv.point_metrics(ck), cv.point_metrics(cr)
    out = {}
    for m in METRIK_BANDING:
        dd = bk[m] - br[m]
        out[m] = [float(tk[m].mean() - tr[m].mean()), *map(float, np.percentile(dd, [2.5, 97.5])),
                  float(np.mean(dd > 0))]
    return out


def afl_per_pasien(d, pred) -> dict:
    """Recall AFL tiap pasien AFL (7 pasien), rata-rata ulangan."""
    c = _cms(d, pred)
    afl = d.patients[d.strata == 2]
    pi = np.searchsorted(d.patients, afl)
    rec = (c[:, pi, 2, 2] / c[:, pi, 2, :].sum(-1)).mean(0)
    return {str(p): float(v) for p, v in zip(afl, rec)}


def afl_palsu(d, pred) -> dict:
    """Jumlah window non-AFL yang ditebak AFL (rata-rata per ulangan), menurut kelas asli dan jenis pasien."""
    dari = _cms(d, pred)[:, :, :2, 2]                        # (R, P, 2): aktual Normal/AFIB → tebakan AFL
    return {"total": float(dari.sum((1, 2)).mean()),
            "dari_normal": float(dari[..., 0].sum(1).mean()),
            "dari_afib": float(dari[..., 1].sum(1).mean()),
            "pasien_tanpa_afl": float(dari[:, d.strata != 2].sum((1, 2)).mean())}


def bias_di_batas(bias: np.ndarray) -> float:
    """Proporsi bias AFIB/AFL yang menyentuh batas grid (−7 atau +3). > 0 = tanda bahaya (AGENTS.md)."""
    b = np.asarray(bias)[..., 1:]
    return float(np.mean(np.isclose(b, BATAS_GRID[0]) | np.isclose(b, BATAS_GRID[1])))


def ambang_takikardia(stres: np.ndarray, bias: np.ndarray, uji) -> dict:
    """
    stres (R, 5, S, 3): prediksi tiap model fold pada set uji stres; bias (R, 5, 3): bias fold-nya.
    Return proporsi tebakan Normal/AFIB/AFL per (jenis variasi, bpm) dan ambang bpm berhenti menebak Normal.
    """
    kelas = np.stack([cv.apply_bias(stres[i, k], bias[i, k]).argmax(1)
                      for i in range(stres.shape[0]) for k in range(stres.shape[1])])     # (model, S)
    proporsi, ambang = {}, {}
    for j in T.JITTERS:
        tab = []
        for b in T.BPMS:
            m = (uji.jenis == j) & (uji.bpm == b)
            tab.append(np.bincount(kelas[:, m].ravel(), minlength=3) / (kelas.shape[0] * m.sum()))
        tab = np.array(tab)
        proporsi[j] = tab.tolist()
        ambang[j] = T.normal_threshold(tab)
    return {"bpm": list(T.BPMS), "proporsi": proporsi, "ambang": ambang}


def _lingkungan() -> dict:
    def git(*args):
        try:
            r = subprocess.run(["git", *args], cwd=D.REPO, capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    commit = git("rev-parse", "--short", "HEAD")
    status = git("status", "--porcelain", "--", "v11_model_tim")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "tidak ada"
    except ImportError:
        gpu = "tidak diketahui"
    pustaka = {"python": platform.python_version()}
    for nama in ("numpy", "scipy", "torch", "scikit-learn", "xgboost", "aeon", "imbalanced-learn"):
        try:
            pustaka[nama] = metadata.version(nama)
        except metadata.PackageNotFoundError:
            pustaka[nama] = None
    return {"commit": commit or "tidak diketahui",
            "ada_perubahan": "tidak diketahui" if status is None else bool(status),
            "gpu": gpu, "pustaka": pustaka}


def laporan_model(folder: Path, d) -> dict:
    """Evaluasi lengkap satu setelan model; ditulis juga ke <folder>/ringkasan.json."""
    folder = Path(folder)
    h = runner.muat_hasil(folder, d)
    R = h.ulangan
    folds = d.folds[R]
    pred_bias, bias = keputusan(h.oof, folds, d.y)
    pred_v31, _ = keputusan(d.oof_v31[R], folds, d.y)
    pred = {"v1": d.oof_v1[R].argmax(-1), "v31": pred_v31, "argmax": h.oof.argmax(-1), "bias": pred_bias}
    uji = D.set_uji_stres()
    _, bias_kontrol = keputusan(d.kontrol_v7.oof[None], d.folds[:1], d.y)
    b2 = bias.reshape(-1, 3)
    r = {
        "model": h.model, "nama_setelan": folder.name, "setelan": h.setelan, "quick": h.quick, "ulangan": R,
        "metrik": tabel_metrik(d, pred),
        "banding_v31": banding(d, pred_bias, pred_v31),
        "banding_v1": banding(d, pred_bias, pred["v1"]),
        "afl_per_pasien": {k: afl_per_pasien(d, p) for k, p in pred.items()},
        "afl_palsu": {k: afl_palsu(d, p) for k, p in pred.items()},
        "bias": {"median": np.median(b2, 0).tolist(), "min": b2.min(0).tolist(), "maks": b2.max(0).tolist(),
                 "di_batas": bias_di_batas(bias)},
        "takikardia": {"model": ambang_takikardia(h.stres, bias, uji),
                       "kontrol_v7": ambang_takikardia(d.kontrol_v7.stres[None], bias_kontrol, uji),
                       "paket_v31": AMBANG_V31_PAKET},
        "pita_detak": T.by_hr_bin(dict(y=d.y, rec=d.records, hr=d.hr), {"v31": pred_v31, "model": pred_bias}),
        "waktu_per_fold": float(h.waktu.mean()),
        "info": h.info[0] if h.info else {},
        "lingkungan": _lingkungan(),
    }
    (folder / "ringkasan.json").write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return r


def kumpulkan_ringkasan(akar: Path | None = None) -> list:
    """Semua ringkasan.json non-QUICK di <lokasi_hasil>/<model>/<setelan>/, untuk gabungan.ipynb."""
    akar = Path(akar) if akar else D.lokasi_hasil()
    out = []
    for p in sorted(akar.glob("*/*/ringkasan.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if not r.get("quick", False):
            out.append(r)
    return out


# ── opsional: tuning dan protokol butir 10 ──────────────────────────────
def pilih_crossfit(d, folders: list) -> dict:
    """
    Tuning tanpa mengintip fold uji: beberapa setelan dari model yang sama, dan untuk tiap fold uji dipakai setelan
    terbaik di fold lain (v4 `crossfit_select`, bias juga dipilih cross-fit). Hanya ulangan yang lengkap di semua setelan.
    """
    hasil = {Path(f).name: runner.muat_hasil(f, d) for f in folders}
    if len({h.model for h in hasil.values()}) != 1:
        raise ValueError("pilih_crossfit hanya untuk setelan dari satu model yang sama.")
    R = sorted(set.intersection(*(set(h.ulangan) for h in hasil.values())))
    if not R:
        raise RuntimeError("Tidak ada ulangan yang lengkap di semua setelan.")
    pred, terpilih = [], []
    for r in R:
        cands = {nama: h.oof[h.ulangan.index(r)] for nama, h in hasil.items()}
        p, _, chosen = P4.crossfit_select(d.y, cands, d.folds[r])
        pred.append(p.astype(np.int64))
        terpilih.append(chosen)
    pred = np.stack(pred)
    pred_v31, _ = keputusan(d.oof_v31[R], d.folds[R], d.y)
    return {"ulangan": R, "terpilih": terpilih, "metrik": tabel_metrik(d, {"pilihan": pred, "v31": pred_v31}),
            "banding_v31": banding(d, pred, pred_v31)}


def banding_setelan(d, folder_kandidat: Path, folder_ref: Path) -> dict:
    """
    Dua setelan dari model yang SAMA, berpasangan (mis. balancing "smote" vs "bawaan" tanpa balancing). Keduanya + bias
    cross-fit, ulangan yang lengkap di keduanya, sampel bootstrap pasien yang sama. Juga recall AFL per pasien.
    """
    hk, hr = runner.muat_hasil(folder_kandidat, d), runner.muat_hasil(folder_ref, d)
    if hk.model != hr.model:
        raise ValueError(f"banding_setelan hanya untuk satu model; didapat {hk.model} vs {hr.model}.")
    R = sorted(set(hk.ulangan) & set(hr.ulangan))
    if not R:
        raise RuntimeError("Tidak ada ulangan yang lengkap di kedua setelan.")
    pk, _ = keputusan(hk.oof[[hk.ulangan.index(r) for r in R]], d.folds[R], d.y)
    pr, _ = keputusan(hr.oof[[hr.ulangan.index(r) for r in R]], d.folds[R], d.y)
    return {"model": hk.model, "kandidat": Path(folder_kandidat).name, "referensi": Path(folder_ref).name,
            "ulangan": R, "banding": banding(d, pk, pr),
            "afl_per_pasien": {"kandidat": afl_per_pasien(d, pk), "referensi": afl_per_pasien(d, pr)}}


def tanpa_04936(d, oofs: dict, ulangan: list) -> dict:
    """Sensitivitas tanpa pasien 04936 (58% window AFL); bias cross-fit dihitung ulang di subset (pola v7)."""
    if "v31" not in oofs:
        raise ValueError("oofs harus memuat kunci 'v31' (pembanding).")
    mask = d.records != T.SUSPECT
    pr = {k: np.stack([cv.crossfit_bias(d.y[mask], o[i][mask], d.folds[r][mask])[0].argmax(1)
                       for i, r in enumerate(ulangan)]) for k, o in oofs.items()}
    return T.metrics_with_ci(dict(y=d.y, rec=d.records), pr, mask)


def bootstrap_bersarang(d, oof: np.ndarray) -> dict:
    """Kandidat (bias dipilih ulang di setiap sampel bootstrap) vs v1 argmax. Butuh 5 ulangan lengkap."""
    oof = np.asarray(oof)
    if oof.shape[0] != T.N_REPEATS:
        raise ValueError(f"bootstrap_bersarang butuh {T.N_REPEATS} ulangan lengkap, ada {oof.shape[0]}.")
    dT = dict(y=d.y, rec=d.records, patients=d.patients, strata=d.strata, folds=d.folds)
    cand, _ = T.candidate_cms(dT, oof)
    out = T.nested_bootstrap(dT, cand, {"v1": _cms(d, d.oof_v1.argmax(-1))})
    # T.nested_bootstrap menamai kandidatnya "v31" (asalnya v3.2); di sini itu = model kandidat.
    return {"kandidat": {"titik": out["v31_point"], "ci": out["v31_ci"]}, "vs_v1": out["v1"]}
