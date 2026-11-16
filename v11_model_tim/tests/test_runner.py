import json
from pathlib import Path

import numpy as np
import pytest

from bersama import data, runner


class Palsu:
    """Model palsu yang cepat: memprediksi proporsi kelas data latih. Mencatat (r, k) setiap kali dilatih."""
    NAMA = "palsu"
    SETELAN = {"p": 1}
    SETELAN_QUICK = {}
    panggilan = []

    @staticmethod
    def latih(data_latih, setelan, seed, device):
        Palsu.panggilan.append(divmod(seed - data.SEED, 100))
        return np.bincount(data_latih.y, minlength=3) / len(data_latih.y)

    @staticmethod
    def prediksi(model, X, M, F, device):
        return np.tile(model, (len(X), 1))

    @staticmethod
    def simpan(model, path):
        p = Path(path).with_suffix(".npy")
        np.save(p, model)
        return p

    @staticmethod
    def info(model):
        return {"palsu": True}


@pytest.fixture
def hasil(tmp_path, monkeypatch):
    monkeypatch.setenv("SIHEDAF_HASIL", str(tmp_path))
    Palsu.panggilan.clear()
    return tmp_path


def test_lanjut_setelah_putus(hasil, d):
    f = runner.jalankan(Palsu, "a", {"p": 1}, ulangan=[0])
    assert sorted(Palsu.panggilan) == [(0, k) for k in range(5)]
    (f / "r0_k3.npz").unlink()
    (f / "r0_k1.tmp.npz").write_bytes(b"rusak")                     # sisa sesi yang mati di tengah simpan
    Palsu.panggilan.clear()
    runner.jalankan(Palsu, "a", {"p": 1}, ulangan=[0])
    assert Palsu.panggilan == [(0, 3)]                                # hanya fold yang hilang dilatih ulang
    assert (f / "r0_k3.npy").exists()


def test_setelan_berubah_error(hasil):
    runner.jalankan(Palsu, "a", {"p": 1}, ulangan=[0])
    with pytest.raises(ValueError, match="NAMA_SETELAN"):
        runner.jalankan(Palsu, "a", {"p": 2}, ulangan=[0])


def test_kunci_setelan_salah_ketik_error(hasil):
    with pytest.raises(ValueError, match="tidak dikenal"):
        runner.jalankan(Palsu, "b", {"pp": 1}, ulangan=[0])


def test_hasil_tidak_lengkap(hasil, d):
    f = runner.jalankan(Palsu, "a", {"p": 1}, ulangan=[0, 1])
    for k in range(1, 5):
        (f / f"r1_k{k}.npz").unlink()
    h = runner.muat_hasil(f, d)
    assert h.ulangan == [0] and h.oof.shape == (1, len(d.y), 3) and h.stres.shape == (1, 5, 4600, 3)
    kosong = hasil / "palsu" / "kosong"
    kosong.mkdir(parents=True)
    (kosong / "setelan.json").write_text(json.dumps({"model": "palsu", "setelan": {}, "quick": False}))
    with pytest.raises(RuntimeError, match="lengkap"):
        runner.muat_hasil(kosong, d)


def test_oof_menutup_semua_window(hasil, d):
    f = runner.jalankan(Palsu, "a", {"p": 1}, ulangan=[2])
    h = runner.muat_hasil(f, d)
    assert h.ulangan == [2] and not np.isnan(h.oof).any()
    for k in range(5):
        z = np.load(f / f"r2_k{k}.npz")
        assert np.array_equal(z["idx"], np.flatnonzero(d.folds[2] == k))


def test_prediksi_bukan_probabilitas_error(hasil):
    class Logit(Palsu):
        NAMA = "logit"

        @staticmethod
        def prediksi(model, X, M, F, device):
            return np.tile([2.0, -1.0, 0.5], (len(X), 1))
    with pytest.raises(ValueError, match="probabilitas"):
        runner.jalankan(Logit, "a", {"p": 1}, ulangan=[0])


def test_quick_terpisah(hasil, monkeypatch, d):
    monkeypatch.setattr(data, "QUICK", True)
    assert data.lokasi_hasil() == hasil / "_quick"          # SIHEDAF_HASIL diset (mis. Google Drive)
    monkeypatch.delenv("SIHEDAF_HASIL")
    assert data.lokasi_hasil().name == "v11_quick"


def test_subset_quick(d):
    s = runner.subset_quick(np.arange(len(d.y)), d.y, 1)
    assert len(s) == 3000 and int((d.y[s] == 2).sum()) >= 300 and len(np.unique(s)) == 3000


def test_balancing_under_hanya_di_data_latih(hasil, d):
    f = runner.jalankan(Palsu, "under", {"balancing": "under"}, ulangan=[0])
    for k in range(5):
        z = np.load(f / f"r0_k{k}.npz")
        n = json.loads(str(z["info"]))["n_latih"]
        afl_latih = int(((d.folds[0] != k) & (d.y == 2)).sum())
        assert n == [afl_latih] * 3                                       # dipotong ke jumlah AFL fold latih
        assert np.allclose(np.load(f / f"r0_k{k}.npy"), 1 / 3)            # model palsu melihat data seimbang
        assert np.array_equal(z["idx"], np.flatnonzero(d.folds[0] == k))  # fold uji utuh, distribusi asli
    assert json.loads((f / "setelan.json").read_text())["setelan"]["balancing"] == "under"
    assert data.bobot_sampel(np.repeat(np.arange(3), 5)).tolist() == [1.0] * 15  # seimbang → bobot lunak = 1


def test_tanpa_balancing_n_latih_asli(hasil, d):
    f = runner.jalankan(Palsu, "a", {"p": 1}, ulangan=[0])
    n = json.loads(str(np.load(f / "r0_k0.npz")["info"]))["n_latih"]
    assert n == np.bincount(d.y[d.folds[0] != 0], minlength=3).tolist()
    assert json.loads((f / "setelan.json").read_text())["setelan"]["balancing"] is None


def test_balancing_salah_error_sebelum_training(hasil):
    with pytest.raises(ValueError, match="INPUT"):
        runner.jalankan(Palsu, "s", {"balancing": "smote"}, ulangan=[0])  # Palsu tidak punya INPUT = "fitur"
    with pytest.raises(ValueError, match="tidak dikenal"):
        runner.jalankan(Palsu, "x", {"balancing": "oversampling"}, ulangan=[0])
    assert Palsu.panggilan == [] and not (hasil / "palsu" / "s").exists()


def test_semua_model_punya_input(muat_model):
    from bersama import MODEL_TIM
    harapan = {"svm": "fitur", "xgboost": "fitur"}                        # aeon/xgboost diimpor di dalam latih()
    for nama in MODEL_TIM:
        assert muat_model(nama).INPUT == harapan.get(nama, "urutan")


def test_zip_hasil(hasil):
    f = runner.jalankan(Palsu, "a", {"p": 1}, ulangan=[0])
    import zipfile
    nama = zipfile.ZipFile(runner.zip_hasil(f)).namelist()
    assert "palsu/a/r0_k0.npz" in nama and "palsu/a/setelan.json" in nama
