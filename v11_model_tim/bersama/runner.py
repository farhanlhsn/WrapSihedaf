"""
runner.py — menjalankan satu setelan model di 5 ulangan × 5 fold pasien (protokol v3). Bisa dilanjutkan.

Kontrak `model.py` (setiap folder model):
    NAMA: str                    nama folder model, mis. "tcn"
    INPUT: str                   "fitur" (16 fitur HRV, F) atau "urutan" (X, M); menentukan balancing yang boleh
    SETELAN: dict                setelan bawaan (ditimpa dari sel pertama notebook)
    SETELAN_QUICK: dict          ditimpa saat SIHEDAF_QUICK=1 (uji alur, bukan hasil)
    latih(data, setelan, seed, device) -> model      data = SimpleNamespace(X, M, F, y) window latih
                                                     (setelah balancing "smote": X dan M = None)
    prediksi(model, X, M, F, device) -> (n, 3)       probabilitas, baris berjumlah 1
    simpan(model, path) -> Path                      path tanpa ekstensi; kembalikan path berkas bobot
    info(model) -> dict                              mis. {"parameter": 15000}

Setelan bersama untuk semua model (SETELAN_BERSAMA): `balancing` (None, "under", "over", "smote"; lihat balancing.py).
Balancing dijalankan di sini, pada window latih setiap fold, sebelum `latih()`.

Keluaran per (ulangan r, fold k) di lokasi_hasil()/NAMA/NAMA_SETELAN/:
    r{r}_k{k}.<ext>   bobot model, ditulis DULU
    r{r}_k{k}.npz     oof, idx, stres, waktu, info (+ n_latih per kelas); ditulis atomik TERAKHIR (= tanda fold selesai)
Sesi yang terputus: jalankan sel yang sama lagi; fold yang sudah punya .npz dilewati.
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import balancing as B
from . import data as D

SETELAN_BERSAMA = {"balancing": None}


def _normal(obj) -> object:
    """Bentuk JSON (tuple → list) supaya setelan bisa dibandingkan dengan yang tersimpan."""
    return json.loads(json.dumps(obj, sort_keys=True))


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cek_proba(p: np.ndarray, n: int, asal: str) -> None:
    if p.shape != (n, 3) or not np.all(np.isfinite(p)) or not np.allclose(p.sum(1), 1.0, atol=1e-3):
        raise ValueError(f"prediksi() untuk {asal} harus probabilitas (n, 3) yang barisnya berjumlah 1; "
                         f"didapat bentuk {p.shape}")


def subset_quick(idx: np.ndarray, y: np.ndarray, seed: int, n: int = 3000, min_afl: int = 300) -> np.ndarray:
    """Sebagian kecil window latih untuk uji alur: maksimal `min_afl` AFL (semua jika lebih sedikit), sisanya acak."""
    rng = np.random.default_rng(seed)
    afl, lain = idx[y[idx] == 2], idx[y[idx] != 2]
    a = rng.choice(afl, size=min(min_afl, len(afl)), replace=False)
    b = rng.choice(lain, size=min(n - len(a), len(lain)), replace=False)
    return np.sort(np.concatenate([a, b]))


def jalankan(mdl, nama_setelan: str, setelan: dict, *, ulangan=None, log=print) -> Path:
    """Latih + prediksi semua (ulangan, fold) yang belum ada. Return folder hasil."""
    if not nama_setelan or any(c in nama_setelan for c in "/\\:"):
        raise ValueError(f"NAMA_SETELAN tidak valid: {nama_setelan!r}")
    tak_dikenal = sorted(set(setelan) - set(mdl.SETELAN) - set(SETELAN_BERSAMA))
    if tak_dikenal:
        raise ValueError(f"Kunci SETELAN tidak dikenal untuk {mdl.NAMA}: {tak_dikenal}. "
                         f"Kunci yang ada: {sorted({**SETELAN_BERSAMA, **mdl.SETELAN})}")
    efektif = {**SETELAN_BERSAMA, **mdl.SETELAN, **setelan, **(mdl.SETELAN_QUICK if D.QUICK else {})}
    input_model = getattr(mdl, "INPUT", None)
    B.periksa(efektif["balancing"], input_model)
    folder = D.lokasi_hasil() / mdl.NAMA / nama_setelan
    folder.mkdir(parents=True, exist_ok=True)
    kunci = {"model": mdl.NAMA, "setelan": _normal(efektif), "quick": D.QUICK}
    p_setelan = folder / "setelan.json"
    if p_setelan.exists():
        lama = json.loads(p_setelan.read_text(encoding="utf-8"))
        if lama != kunci:
            raise ValueError(
                f"Setelan untuk NAMA_SETELAN={nama_setelan!r} berbeda dengan yang tersimpan di {p_setelan}.\n"
                f"  tersimpan: {lama}\n  sekarang : {kunci}\n"
                "Ganti NAMA_SETELAN setiap kali SETELAN diubah, supaya hasil lama tidak tercampur.")
    else:
        p_setelan.write_text(json.dumps(kunci, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    d = D.muat()
    uji = D.set_uji_stres()
    dev = _device()
    ulangan = ([0] if D.QUICK else list(range(D.N_ULANGAN))) if ulangan is None else list(ulangan)
    t_mulai = time.perf_counter()
    for r in ulangan:
        for k in range(D.N_FOLD):
            out = folder / f"r{r}_k{k}.npz"
            if out.exists():
                continue
            tr = np.flatnonzero(d.folds[r] != k)
            te = np.flatnonzero(d.folds[r] == k)
            seed = D.seed_latih(r, k)
            if D.QUICK:
                tr = subset_quick(tr, d.y, seed)
            data = SimpleNamespace(X=d.X[tr], M=d.M[tr], F=d.F[tr], y=d.y[tr])
            data = B.terapkan(data, efektif["balancing"], seed, input_model)       # hanya window latih fold ini
            n_latih = np.bincount(data.y, minlength=3).tolist()
            t0 = time.perf_counter()
            model = mdl.latih(data, efektif, seed, dev)
            waktu = time.perf_counter() - t0
            p_te = np.asarray(mdl.prediksi(model, d.X[te], d.M[te], d.F[te], dev))
            p_st = np.asarray(mdl.prediksi(model, uji.X, uji.M, uji.F, dev))
            _cek_proba(p_te, len(te), f"fold uji r{r} k{k}")
            _cek_proba(p_st, len(uji.X), "set uji stres")
            mdl.simpan(model, folder / f"r{r}_k{k}")
            tmp = folder / f"r{r}_k{k}.tmp.npz"
            np.savez(tmp, oof=p_te.astype(np.float32), idx=te.astype(np.int64), stres=p_st.astype(np.float32),
                     waktu=np.float64(waktu), info=json.dumps({**mdl.info(model), "n_latih": n_latih}))
            tmp.replace(out)
            log(f"[{(time.perf_counter() - t_mulai) / 60:6.1f} mnt] {mdl.NAMA}/{nama_setelan} "
                f"ulangan {r} fold {k}: latih {waktu:.0f} s, window latih per kelas {n_latih}")
    return folder


def muat_hasil(folder: Path, d=None) -> SimpleNamespace:
    """
    OOF (R, N, 3), prediksi uji stres (R, 5, S, 3), dan waktu (R, 5) dari ulangan yang LENGKAP 5 fold.
    Ulangan yang belum lengkap dilewati; tidak ada satu pun yang lengkap → RuntimeError.
    """
    folder = Path(folder)
    kunci = json.loads((folder / "setelan.json").read_text(encoding="utf-8"))
    folds = d.folds if d is not None else np.load(D.lokasi_data() / D.BERKAS[1])["folds"]
    lengkap = [r for r in range(D.N_ULANGAN)
               if all((folder / f"r{r}_k{k}.npz").exists() for k in range(D.N_FOLD))]
    if not lengkap:
        raise RuntimeError(f"Belum ada ulangan yang lengkap 5 fold di {folder}. Jalankan sel training dulu.")
    oof = np.full((len(lengkap), folds.shape[1], 3), np.nan, dtype=np.float32)
    stres, waktu, info = [], [], []
    for i, r in enumerate(lengkap):
        st, wk = [], []
        for k in range(D.N_FOLD):
            z = np.load(folder / f"r{r}_k{k}.npz")
            idx = z["idx"]
            if not np.array_equal(idx, np.flatnonzero(folds[r] == k)):
                raise RuntimeError(f"{folder}/r{r}_k{k}.npz tidak sesuai fold resmi (data atau fold berbeda).")
            oof[i, idx] = z["oof"]
            st.append(z["stres"])
            wk.append(float(z["waktu"]))
            info.append(json.loads(str(z["info"])))
        stres.append(np.stack(st))
        waktu.append(wk)
    if np.isnan(oof).any():
        raise RuntimeError(f"OOF di {folder} tidak menutup semua window.")
    return SimpleNamespace(oof=oof, stres=np.stack(stres), waktu=np.array(waktu), ulangan=lengkap,
                           model=kunci["model"], setelan=kunci["setelan"], quick=kunci["quick"], info=info,
                           folder=folder)


def zip_hasil(folder: Path) -> Path:
    """Zip satu folder setelan untuk dibagikan ke tim. Diekstrak di <repo>/cache/v11/ oleh penerima."""
    folder = Path(folder)
    tujuan = folder.parent / f"{folder.parent.name}_{folder.name}.zip"
    with zipfile.ZipFile(tujuan, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(folder.iterdir()):
            if p.is_file() and ".tmp" not in p.name:
                zf.write(p, f"{folder.parent.name}/{folder.name}/{p.name}")
    return tujuan
