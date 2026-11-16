import os
import shutil
import zipfile

import numpy as np
import pytest

from bersama import data


def test_repo_dan_bentuk(d):
    assert (data.REPO / "AGENTS.md").exists()
    assert d.X.shape == (165978, 30, 2) and d.F.shape == (165978, 16) and d.folds.shape == (5, 165978)
    assert d.oof_v31.shape == d.oof_v1.shape == (5, 165978, 3)
    assert d.kontrol_v7.oof.shape == (165978, 3) and d.kontrol_v7.stres.shape == (5, 4600, 3)
    assert len(d.patients) == 23 and int((d.strata == 2).sum()) == 7
    assert np.all(np.isfinite(d.hr)) and 20 < np.median(d.hr) < 200


def test_seed_latih():
    assert data.seed_latih(0, 0) == 2026 and data.seed_latih(4, 3) == 2429


def _salin_berkas(tujuan):
    for rel in data.BERKAS:
        (tujuan / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(data.lokasi_data() / rel, tujuan / rel)


def test_sidik_jari_rusak_error(tmp_path):
    _salin_berkas(tmp_path)
    p = tmp_path / "dataset_v2_detector.npz"
    isi = bytearray(p.read_bytes())
    isi[len(isi) // 2] ^= 0xFF
    p.write_bytes(bytes(isi))
    with pytest.raises(RuntimeError, match="dataset_v2_detector.npz"):
        data.periksa_sidik_jari(tmp_path)


def test_sidik_jari_berkas_hilang(tmp_path):
    _salin_berkas(tmp_path)
    (tmp_path / "v7/uji_kontrol_r0.npz").unlink()
    with pytest.raises(RuntimeError, match="hilang"):
        data.periksa_sidik_jari(tmp_path)


def test_zip_bolak_balik(tmp_path):
    z = data.buat_zip(tmp_path / "v11_data.zip")
    zipfile.ZipFile(z).extractall(tmp_path / "x")
    data.periksa_sidik_jari(tmp_path / "x")          # tidak error


def test_siapkan_data_dari_url(tmp_path, monkeypatch):
    """Jalur Kaggle/Colab: cache repo tidak ada → unduh zip (URL file:// menggantikan GitHub Release)."""
    z = data.buat_zip(tmp_path / "v11_data.zip")
    # setenv "" (bukan delenv): "" dianggap tidak diset, dan monkeypatch memulihkan nilai yang ditulis siapkan_data
    monkeypatch.setenv("SIHEDAF_DATA", "")
    monkeypatch.setattr(data, "REPO", tmp_path / "repo_palsu")
    monkeypatch.setattr(data, "REPO_UNDUHAN", tmp_path / "unduh")
    akar = data.siapkan_data(z.as_uri())
    assert akar == tmp_path / "unduh"
    assert os.environ["SIHEDAF_DATA"] == str(akar)
    data.periksa_sidik_jari(akar)


def test_siapkan_data_url_gagal(tmp_path, monkeypatch):
    monkeypatch.setenv("SIHEDAF_DATA", "")
    monkeypatch.setattr(data, "REPO", tmp_path / "repo_palsu")
    monkeypatch.setattr(data, "REPO_UNDUHAN", tmp_path / "u")
    with pytest.raises(RuntimeError, match="SIHEDAF_DATA"):
        data.siapkan_data((tmp_path / "tidak_ada.zip").as_uri())


def test_siapkan_data_sihedaf_data_salah_tidak_mengunduh(tmp_path, monkeypatch):
    monkeypatch.setenv("SIHEDAF_DATA", str(tmp_path / "kosong"))
    with pytest.raises(RuntimeError, match="hilang"):
        data.siapkan_data("file:///tidak/dipakai.zip")


def test_uji_stres_fitur_dari_ibi_float64():
    u = data.set_uji_stres()
    assert len(u.ibi) == 4600 and u.X.shape == (4600, 30, 2) and u.F.shape == (4600, 16)
    assert np.array_equal(u.F[7], data.hrv_core.hrv_features(u.ibi[7]))
    assert u.bpm[0] == 60 and u.bpm[-1] == 170 and u.jenis[0] == "tanpa variasi"


def test_uji_stres_sama_dengan_v7():
    sys_path = str(data.REPO / "v7_augmentasi_takikardia")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import augmentasi
    Xv7, Mv7, _ = augmentasi.stress_set()
    u = data.set_uji_stres()
    assert np.array_equal(u.X, Xv7) and np.array_equal(u.M, Mv7)


def test_bobot_sampel_lunak():
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    w = data.bobot_sampel(y)
    penuh = len(y) / (3 * np.array([90, 9, 1]))
    assert np.allclose(w[[0, 90, 99]], penuh ** 0.5)
