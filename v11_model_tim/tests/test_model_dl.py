"""Tes tiga model DL (bigru, tcn, inceptiontime): bentuk, invarian padding, ukuran, dan kontrak runner."""
from types import SimpleNamespace

import numpy as np
import pytest

from bersama import latih_dl
from conftest import cek_invarian_padding

MODEL_DL = ("bigru", "tcn", "inceptiontime")


@pytest.mark.parametrize("nama", MODEL_DL)
def test_bentuk_dan_invarian_padding(nama, muat_model):
    M = muat_model(nama)
    cek_invarian_padding(M.buat_net(M.SETELAN))


@pytest.mark.parametrize("nama", MODEL_DL)
def test_maks_100_ribu_parameter(nama, muat_model):
    M = muat_model(nama)
    assert latih_dl.jumlah_parameter(M.buat_net(M.SETELAN)) <= 100_000


@pytest.mark.parametrize("nama", MODEL_DL)
def test_setelan_bawaan_resep_v3(nama, muat_model):
    M = muat_model(nama)
    assert M.NAMA == nama
    assert {k: M.SETELAN[k] for k in latih_dl.BAWAAN} == latih_dl.BAWAAN
    assert M.SETELAN_QUICK == {"epochs": 1}


@pytest.mark.parametrize("nama", MODEL_DL)
def test_kontrak_runner(nama, muat_model, d, dev, tmp_path):
    """latih 1 epoch pada data kecil → prediksi probabilitas, simpan .pt, info berisi jumlah parameter."""
    M = muat_model(nama)
    idx = np.arange(0, len(d.y), 40)
    data_latih = SimpleNamespace(X=d.X[idx], M=d.M[idx], F=d.F[idx], y=d.y[idx])
    model = M.latih(data_latih, {**M.SETELAN, "epochs": 1}, 5, dev)
    p = M.prediksi(model, d.X[:3000], d.M[:3000], d.F[:3000], dev)
    assert p.shape == (3000, 3) and np.allclose(p.sum(1), 1, atol=1e-4)
    assert M.simpan(model, tmp_path / "r0_k0").suffix == ".pt"
    assert M.info(model)["parameter"] == latih_dl.jumlah_parameter(model.net)


def test_kernel_genap_ditolak(muat_model):
    with pytest.raises(ValueError):
        muat_model("tcn").TCN(kernel=4)
    with pytest.raises(ValueError):
        muat_model("inceptiontime").InceptionTime(kernel=(3, 4, 9))
