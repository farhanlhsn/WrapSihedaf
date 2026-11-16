"""
balancing.py — opsi balancing data latih, dijalankan runner DI DALAM setiap fold (arahan pembimbing, 22 Oktober 2026).

Bawaan v11 tetap tanpa balancing (bobot kelas lunak + bias cross-fit, alasan di ../README.md). Opsi ini ada supaya cara
balancing yang diminta pembimbing bisa diuji dengan protokol yang sama dan dibandingkan berpasangan dengan setelan bawaan:
- hanya window latih fold itu yang di-balance; window uji dan set uji stres tetap dengan distribusi asli;
- tidak ada kolom pasien yang hilang dan tidak ada pembagian lain, jadi pasien uji tidak pernah masuk data latih.

Metode (kunci setelan `balancing`, berlaku untuk semua model):
    None     tanpa balancing (bawaan)
    "under"  undersampling acak: setiap kelas dipotong menjadi jumlah kelas terkecil (AFL), tanpa pengembalian
    "over"   oversampling acak: AFIB dan AFL disalin acak (dengan pengembalian) sampai sama dengan jumlah Normal. Salinan
             utuh window asli, jadi berlaku untuk semua model, termasuk model urutan (padanan SMOTE untuk urutan IBI)
    "smote"  SMOTE (imbalanced-learn, k = 5) pada 16 fitur HRV: AFIB dan AFL dinaikkan ke jumlah Normal. Hanya untuk
             model fitur (`INPUT = "fitur"`). Interpolasi urutan IBI dengan panjang berbeda tidak bermakna, jadi model
             urutan → error. Fitur distandardisasi dulu (dari data latih fold, seperti pada kNN), lalu dikembalikan
             ke skala asli karena model melakukan normalisasinya sendiri.

Setelah balancing, bobot kelas lunak (`data.bobot_sampel`, loss DL) otomatis bernilai 1 untuk semua kelas
(class_weights data seimbang = 1), jadi tidak ada koreksi ganda.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

METODE = (None, "under", "over", "smote")
K_SMOTE = 5


def periksa(metode, input_model) -> None:
    """Error sebelum training dimulai bila metode tidak dikenal atau tidak cocok dengan input model."""
    if metode not in METODE:
        raise ValueError(f"balancing={metode!r} tidak dikenal. Pilihan: {list(METODE)}")
    if metode == "smote" and input_model != "fitur":
        raise ValueError(f"balancing='smote' hanya untuk model fitur HRV (INPUT = 'fitur'), model ini INPUT = "
                         f"{input_model!r}. Untuk model urutan pakai balancing='under' atau 'over'.")


def terapkan(data: SimpleNamespace, metode, seed: int, input_model) -> SimpleNamespace:
    """data = SimpleNamespace(X, M, F, y) window latih satu fold → data latih yang sudah di-balance."""
    periksa(metode, input_model)
    if metode is None:
        return data
    if metode == "under":
        return _under(data, seed)
    if metode == "over":
        return _over(data, seed)
    return _smote(data, seed)


def _under(data, seed):
    rng = np.random.default_rng(seed)
    n = np.bincount(data.y, minlength=3)
    if (n == 0).any():
        raise ValueError(f"Undersampling butuh ketiga kelas di data latih; jumlah per kelas {n.tolist()}.")
    m = int(n.min())
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(data.y == c), size=m, replace=False) for c in range(3)]))
    return SimpleNamespace(X=data.X[idx], M=data.M[idx], F=data.F[idx], y=data.y[idx])


def _over(data, seed):
    rng = np.random.default_rng(seed)
    n = np.bincount(data.y, minlength=3)
    if (n == 0).any():
        raise ValueError(f"Oversampling butuh ketiga kelas di data latih; jumlah per kelas {n.tolist()}.")
    m = int(n.max())
    # window asli di depan (seperti SMOTE), salinan acak di belakang
    tambah = [rng.choice(np.flatnonzero(data.y == c), size=m - int(n[c]), replace=True) for c in range(3)]
    idx = np.concatenate([np.arange(len(data.y)), *tambah])
    return SimpleNamespace(X=data.X[idx], M=data.M[idx], F=data.F[idx], y=data.y[idx])


def _smote(data, seed):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as e:
        raise ImportError("balancing='smote' butuh imbalanced-learn: pip install 'imbalanced-learn>=0.12'") from e
    from sklearn.preprocessing import StandardScaler
    n = np.bincount(data.y, minlength=3)
    if n.min() <= K_SMOTE:
        raise ValueError(f"SMOTE (k = {K_SMOTE}) butuh > {K_SMOTE} window per kelas; jumlah per kelas {n.tolist()}.")
    F = np.asarray(data.F, dtype=np.float64)
    scaler = StandardScaler().fit(F)
    Zb, yb = SMOTE(k_neighbors=K_SMOTE, random_state=seed).fit_resample(scaler.transform(F), data.y)
    # imbalanced-learn menaruh window asli di depan, sintetis di belakang. Window asli dipakai persis (tanpa
    # pembulatan bolak-balik skala).
    if not np.array_equal(yb[:len(F)], data.y):
        raise RuntimeError("Urutan keluaran SMOTE tidak seperti yang diharapkan (window asli di depan).")
    Fb = np.concatenate([F, scaler.inverse_transform(Zb[len(F):])]).astype(np.float32)
    return SimpleNamespace(X=None, M=None, F=Fb, y=np.asarray(yb, dtype=data.y.dtype))
