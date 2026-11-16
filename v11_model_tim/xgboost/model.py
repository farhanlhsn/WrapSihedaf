"""
XGBoost pada 16 fitur HRV (`F`, sama dengan LightGBM v1/v3) dengan bobot sampel lunak.

Catatan nama: folder ini bernama `xgboost`, tetapi tidak punya __init__.py, jadi `import xgboost` tetap memuat pustaka
aslinya (tests/test_model_xgboost.py). Kontrak modul: lihat bersama/runner.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # v11_model_tim → paket `bersama`
from bersama import data as D  # noqa: E402

NAMA = "xgboost"
INPUT = "fitur"
SETELAN = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "tree_method": "hist", "perangkat": "cpu",
           "weight_power": 0.5}                                      # bobot kelas ** weight_power; 1,0 = penuh
SETELAN_QUICK = {"n_estimators": 20}


def latih(data, setelan, seed, device):
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=setelan["n_estimators"], learning_rate=setelan["learning_rate"],
                          max_depth=setelan["max_depth"], tree_method=setelan["tree_method"],
                          device=setelan["perangkat"], objective="multi:softprob", eval_metric="mlogloss",
                          random_state=seed, n_jobs=-1)
    model.fit(data.F, data.y, sample_weight=D.bobot_sampel(data.y, setelan["weight_power"]))
    return model


def prediksi(model, X, M, F, device):
    return model.predict_proba(F)


def simpan(model, path):
    p = Path(path).with_suffix(".json")
    model.save_model(p)
    return p


def info(model):
    return {"pohon": int(model.get_booster().num_boosted_rounds())}
