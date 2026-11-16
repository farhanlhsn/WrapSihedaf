"""Tes tiga model klasik (xgboost, svm, minirocket): probabilitas, determinisme, bobot sampel, simpan."""
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bersama import data


def _data_kecil(d, langkah=60):
    idx = np.arange(0, len(d.y), langkah)
    return SimpleNamespace(X=d.X[idx], M=d.M[idx], F=d.F[idx], y=d.y[idx])


def _cek_kontrak(M, d, setelan, tmp_path, akhiran):
    dl = _data_kecil(d)
    m1 = M.latih(dl, setelan, 11, None)
    m2 = M.latih(dl, setelan, 11, None)
    te = np.arange(5, 4000, 3)
    p1 = M.prediksi(m1, d.X[te], d.M[te], d.F[te], None)
    p2 = M.prediksi(m2, d.X[te], d.M[te], d.F[te], None)
    assert p1.shape == (len(te), 3) and np.allclose(p1.sum(1), 1, atol=1e-6)
    assert np.allclose(p1, p2)                                       # seed sama → hasil sama
    assert M.simpan(m1, tmp_path / "r0_k0").suffix == akhiran
    return m1


def test_import_xgboost_asli_dari_folder_model(monkeypatch):
    pytest.importorskip("xgboost")
    monkeypatch.chdir(data.V11 / "xgboost")
    monkeypatch.syspath_prepend(str(data.V11))
    monkeypatch.syspath_prepend(".")
    monkeypatch.delitem(sys.modules, "xgboost", raising=False)
    assert hasattr(importlib.import_module("xgboost"), "XGBClassifier")


def test_xgboost(muat_model, d, tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    M = muat_model("xgboost")
    dipakai = {}
    asli = data.bobot_sampel

    def mata_mata(y, pangkat=0.5):
        dipakai["pangkat"] = pangkat
        return asli(y, pangkat)
    monkeypatch.setattr(data, "bobot_sampel", mata_mata)
    m = _cek_kontrak(M, d, {**M.SETELAN, "n_estimators": 20}, tmp_path, ".json")
    assert dipakai.get("pangkat") == 0.5 and M.info(m) == {"pohon": 20}
    _cek_kontrak(M, d, {**M.SETELAN, "n_estimators": 20, "weight_power": 1.0}, tmp_path, ".json")
    assert dipakai["pangkat"] == 1.0


def test_bobot_sampel_pangkat():
    y = np.array([0, 0, 0, 1, 2, 2])
    assert np.allclose(data.bobot_sampel(y, 1.0), data.cv.class_weights(y)[y])
    assert np.allclose(data.bobot_sampel(y), data.cv.class_weights(y)[y] ** 0.5)


def test_svm(muat_model, d, tmp_path):
    M = muat_model("svm")
    m = _cek_kontrak(M, d, {**M.SETELAN, "komponen": 100}, tmp_path, ".joblib")
    assert set(M.info(m)) == {"komponen", "konvergen"}


def test_minirocket(muat_model, d, tmp_path):
    pytest.importorskip("aeon", reason="aeon belum terpasang: pip install -r minirocket/requirements-minirocket.txt")
    M = muat_model("minirocket")
    m = _cek_kontrak(M, d, {**M.SETELAN, "kernel": 84}, tmp_path, ".joblib")
    assert M.info(m)["fitur"] > 0


def test_minirocket_prediksi_per_batch(muat_model, d, monkeypatch):
    pytest.importorskip("aeon")
    M = muat_model("minirocket")
    m = M.latih(_data_kecil(d), {**M.SETELAN, "kernel": 84}, 1, None)
    monkeypatch.setattr(M, "BATCH", 1000)
    p_kecil = M.prediksi(m, d.X[:2500], d.M[:2500], None, None)
    monkeypatch.setattr(M, "BATCH", 8192)
    assert np.allclose(p_kecil, M.prediksi(m, d.X[:2500], d.M[:2500], None, None), atol=1e-6)
