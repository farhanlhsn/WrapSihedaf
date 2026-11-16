"""Tes v3.4: `python -m pytest v3.4_pembersih_detak/test_pembersih.py -q` dari root repo."""
import numpy as np
import pytest

import pembersih as P

FS = P.FS


def _ekg(rr_ms, polaritas=None, noise=0.0, seed=0, dur_s=None):
    """EKG sintetis 250 Hz: QRS bifasik (R lalu S) per detak; polaritas[i] = rasio S/R (S dalam bila > 1)."""
    fs = 250.0
    t_r = np.cumsum(rr_ms) / 1000.0
    n = int((t_r[-1] + 1.0) * fs) if dur_s is None else int(dur_s * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    sr = np.ones(len(t_r)) * 0.4 if polaritas is None else np.asarray(polaritas)
    for i, tr in enumerate(t_r):
        x += 1.0 * np.exp(-0.5 * ((t - tr) / 0.010) ** 2)
        x -= sr[i] * np.exp(-0.5 * ((t - tr - 0.030) / 0.010) ** 2)
        x += 0.15 * np.exp(-0.5 * ((t - tr + 0.16) / 0.025) ** 2)        # P
        x += 0.25 * np.exp(-0.5 * ((t - tr - 0.25) / 0.045) ** 2)        # T
    x += np.random.default_rng(seed).normal(0, noise, n)
    return x, fs, t_r


def test_detektor_salinan_identik_sintetis():
    rng = np.random.default_rng(1)
    for seed in range(5):
        x, fs, _ = _ekg(rng.normal(700, 80, 60), noise=0.05, seed=seed)
        x50 = P.hrv_core.resample_signal(x, fs, FS)
        assert np.array_equal(P.detektor_salinan(x50), P.hrv_core.detect_rpeaks(x50, FS))


@pytest.mark.skipif(not (P.AFDB / "04015.dat").exists(), reason="AFDB tidak ada")
def test_detektor_salinan_identik_afdb():
    x, fs = P.A12.baca_dat(P.AFDB / "04015", 0)
    x50 = P.hrv_core.resample_signal(np.nan_to_num(x[:int(1800 * fs)]), fs, FS)
    assert np.array_equal(P.detektor_salinan(x50), P.hrv_core.detect_rpeaks(x50, FS))


def test_penyelarasan_mengurangi_lompatan_r_ke_s():
    # 25% detak punya S lebih dalam dari R → |y| kadang memilih S; penyelarasan mengembalikan ke R
    rng = np.random.default_rng(3)
    rr = np.full(80, 500.0)
    sr = np.where(rng.random(80) < 0.25, 1.15, 0.85)
    x, fs, t_r = _ekg(rr, polaritas=sr, noise=0.02, seed=3)
    o = P.proses_sinyal(x, fs)
    def jitter(p):
        e, _ = P.cocokkan(p / FS, t_r, tol=0.1)
        return np.std(e - np.median(e)) * 1000
    assert jitter(o["p_f1"]) < 0.6 * jitter(o["p_lama"])
    assert np.all(np.diff(o["p_f1"]) > 0)


def test_ritme_teratur_tanpa_lompatan_tidak_digeser_jauh():
    x, fs, _ = _ekg(np.full(60, 800.0), noise=0.0)
    o = P.proses_sinyal(x, fs)
    assert np.max(np.abs(o["p_f1"] - o["p_lama"])) <= P.MAX_LAG + 0.5
    assert o["dominan"].all()


def test_ketidakteraturan_af_dipertahankan():
    rng = np.random.default_rng(7)
    rr = rng.uniform(350, 1000, 120)
    x, fs, t_r = _ekg(rr, noise=0.02, seed=7)
    o = P.proses_sinyal(x, fs)
    ibi0, ibi1 = np.diff(o["p_lama"]) / FS * 1000, np.diff(o["p_f1"]) / FS * 1000
    assert abs(ibi1.std() / ibi0.std() - 1) < 0.05


def test_gerbang_menolak_artefak():
    # detak normal + lonjakan artefak berbentuk lain → detak artefak tidak dominan
    x, fs, t_r = _ekg(np.full(40, 600.0), noise=0.01)
    t = np.arange(x.size) / fs
    for ta in t_r[::2] + 0.3:
        x += 1.2 * np.exp(-0.5 * ((t - ta) / 0.040) ** 2)              # gelombang lebar, bukan QRS
    o = P.proses_sinyal(x, fs)
    assert (~o["dominan"]).sum() >= 1
    t_end = o["p_f1"][1:] / FS
    assert P.fraksi_dominan(t_end, o["dominan"][1:], 5.0) < 1.0


def test_cocokkan_satu_satu():
    e, j = P.cocokkan(np.array([1.0, 1.01, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert len(j) == len(np.unique(j)) == 2
    assert np.allclose(sorted(np.abs(e)), [0.0, 0.0])
