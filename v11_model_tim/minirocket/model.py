"""
MiniRocket (aeon) untuk urutan IBI + ΔIBI (2 kanal × 30 detak), lalu regresi logistik dengan bobot sampel lunak.

Input dinormalisasi z-score bermask (sama dengan model DL), jadi posisi padding selalu bernilai 0. MiniRocket tidak
mendukung panjang berbeda, jadi urutan padding 30 dipakai apa adanya. Jumlah kernel dikurangi dari 10.000 menjadi
2.016 (kelipatan 84) agar matriks fitur muat di RAM. Fitur dihitung per batch.
aeon dipasang terpisah: `pip install -r requirements-minirocket.txt`. Kontrak modul: lihat bersama/runner.py.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # v11_model_tim → paket `bersama`
from bersama import data as D  # noqa: E402

NAMA = "minirocket"
INPUT = "urutan"
SETELAN = {"kernel": 2016, "C": 1.0, "max_iter": 1000}
SETELAN_QUICK = {"kernel": 84}
BATCH = 8192


def _MiniRocket():
    try:
        from aeon.transformations.collection.convolution_based import MiniRocket
    except ImportError as e:
        raise ImportError("Pustaka aeon belum terpasang: pip install -r requirements-minirocket.txt") from e
    return MiniRocket


def _bentuk(X, M, mu, sd) -> np.ndarray:
    """(n, 30, 2) → (n, 2, 30) float64, format koleksi aeon (n_cases, n_channels, n_timepoints)."""
    return np.ascontiguousarray(D.cv.zscore_apply(X, M, mu, sd).transpose(0, 2, 1), dtype=np.float64)


def _fitur(transformer, X, M, mu, sd) -> np.ndarray:
    return np.concatenate([np.asarray(transformer.transform(_bentuk(X[s:s + BATCH], M[s:s + BATCH], mu, sd)),
                                      dtype=np.float32) for s in range(0, len(X), BATCH)])


def latih(data, setelan, seed, device):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    mu, sd = D.cv.zscore_fit(data.X, data.M)
    transformer = _MiniRocket()(n_kernels=setelan["kernel"], random_state=seed, n_jobs=-1)
    transformer.fit(_bentuk(data.X, data.M, mu, sd))
    scaler = StandardScaler()
    Z = scaler.fit_transform(_fitur(transformer, data.X, data.M, mu, sd))
    clf = LogisticRegression(C=setelan["C"], max_iter=setelan["max_iter"], random_state=seed)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(Z, data.y, sample_weight=D.bobot_sampel(data.y))
    konvergen = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    if not konvergen:
        print(f"  ⚠ LogisticRegression belum konvergen (max_iter={setelan['max_iter']}); dicatat di info.")
    return {"mu": mu, "sd": sd, "transformer": transformer, "scaler": scaler, "clf": clf,
            "n_fitur": int(Z.shape[1]), "konvergen": konvergen}


def prediksi(model, X, M, F, device):
    return np.concatenate([
        model["clf"].predict_proba(model["scaler"].transform(
            _fitur(model["transformer"], X[s:s + BATCH], M[s:s + BATCH], model["mu"], model["sd"])))
        for s in range(0, len(X), BATCH)])


def simpan(model, path):
    import joblib
    p = Path(path).with_suffix(".joblib")
    joblib.dump(model, p)
    return p


def info(model):
    return {"fitur": model["n_fitur"], "konvergen": bool(model["konvergen"])}
