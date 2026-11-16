"""
SVM kernel RBF (aproksimasi Nystroem) pada 16 fitur HRV, dengan bobot sampel lunak.

SVM RBF eksak (sklearn.svm.SVC) tidak sanggup ±130 ribu window latih × 25 fold, jadi kernel RBF diaproksimasi dengan
`Nystroem` lalu dipisahkan oleh SVM linear (LinearSVC, squared hinge, primal). SVM tidak menghasilkan probabilitas: skor keputusannya
diubah ke pseudo-probabilitas dengan softmax, dan bias cross-fit menyesuaikan skalanya. Kontrak: bersama/runner.py.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # v11_model_tim → paket `bersama`
from bersama import data as D  # noqa: E402

NAMA = "svm"
INPUT = "fitur"
SETELAN = {"komponen": 2000, "gamma": None, "C": 1.0, "max_iter": 5000}
SETELAN_QUICK = {"komponen": 200}


def latih(data, setelan, seed, device):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.kernel_approximation import Nystroem
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    model = make_pipeline(
        StandardScaler(),
        Nystroem(kernel="rbf", gamma=setelan["gamma"], n_components=setelan["komponen"], random_state=seed),
        LinearSVC(C=setelan["C"], max_iter=setelan["max_iter"], dual="auto", random_state=seed),
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(data.F, data.y, linearsvc__sample_weight=D.bobot_sampel(data.y))
    model.konvergen_ = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    if not model.konvergen_:
        print(f"  ⚠ LinearSVC belum konvergen (max_iter={setelan['max_iter']}); dicatat di info.")
    return model


def prediksi(model, X, M, F, device):
    s = model.decision_function(F)
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def simpan(model, path):
    import joblib
    p = Path(path).with_suffix(".joblib")
    joblib.dump(model, p)
    return p


def info(model):
    return {"komponen": int(model.named_steps["nystroem"].n_components), "konvergen": bool(model.konvergen_)}
