"""Tes opsi balancing per fold (bersama/balancing.py). Data sintetis kecil: tidak butuh cache, torch, atau GPU."""
from types import SimpleNamespace

import numpy as np
import pytest

from bersama import balancing as B


def _data(n=(600, 200, 40), seed=0):
    """Tiga kelas tidak seimbang dengan fitur yang skalanya sangat berbeda (seperti F: ms vs rasio)."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), n).astype(np.int64)
    F = (rng.normal(size=(len(y), 16)) + y[:, None]) * np.geomspace(1e-2, 1e4, 16)
    X = rng.normal(size=(len(y), 30, 2)).astype(np.float32)
    M = np.ones((len(y), 30), dtype=np.float32)
    return SimpleNamespace(X=X, M=M, F=F.astype(np.float32), y=y)


def _baris(a):
    return {r.tobytes() for r in np.ascontiguousarray(a)}


def test_tanpa_balancing_tidak_mengubah_apa_pun():
    d = _data()
    assert B.terapkan(d, None, 1, "urutan") is d


def test_under_seimbang_dan_subset_asli():
    d = _data()
    u = B.terapkan(d, "under", 7, "urutan")
    assert np.bincount(u.y, minlength=3).tolist() == [40, 40, 40]
    # setiap window adalah window asli yang utuh (X, M, F, y sejajar), tanpa duplikat
    asli = {(x.tobytes(), f.tobytes(), int(c)) for x, f, c in zip(d.X, d.F, d.y)}
    hasil = [(x.tobytes(), f.tobytes(), int(c)) for x, f, c in zip(u.X, u.F, u.y)]
    assert set(hasil) <= asli and len(set(hasil)) == len(hasil)
    assert (B.terapkan(d, "under", 7, "urutan").F == u.F).all()          # seed sama → hasil sama
    assert not np.array_equal(B.terapkan(d, "under", 8, "urutan").F, u.F)


def test_under_juga_untuk_model_fitur():
    assert np.bincount(B.terapkan(_data(), "under", 1, "fitur").y).tolist() == [40, 40, 40]


def test_smote_seimbang_ke_kelas_terbanyak():
    pytest.importorskip("imblearn")
    d = _data()
    s = B.terapkan(d, "smote", 3, "fitur")
    assert np.bincount(s.y, minlength=3).tolist() == [600, 600, 600]
    assert s.X is None and s.M is None and s.F.dtype == np.float32
    # window asli dipakai persis dan ada di depan; Normal tidak ditambah
    assert np.array_equal(s.F[:len(d.y)], d.F) and np.array_equal(s.y[:len(d.y)], d.y)
    assert (s.y[len(d.y):] != 0).all()
    np.testing.assert_array_equal(B.terapkan(d, "smote", 3, "fitur").F, s.F)   # deterministik


def test_smote_sintetis_di_antara_tetangga_kelas_sendiri():
    pytest.importorskip("imblearn")
    d = _data()
    s = B.terapkan(d, "smote", 3, "fitur")
    for c in (1, 2):
        asli = d.F[d.y == c].astype(np.float64)
        sint = s.F[len(d.y):][s.y[len(d.y):] == c].astype(np.float64)
        lo, hi = asli.min(0), asli.max(0)
        tol = 1e-4 * (hi - lo)
        assert ((sint >= lo - tol) & (sint <= hi + tol)).all()               # interpolasi, bukan ekstrapolasi
        assert len(_baris(s.F[s.y == c])) > 0.9 * (s.y == c).sum()          # sebagian besar baru, bukan salinan


def test_smote_untuk_model_urutan_error():
    with pytest.raises(ValueError, match="under"):
        B.terapkan(_data(), "smote", 1, "urutan")
    with pytest.raises(ValueError, match="INPUT"):
        B.periksa("smote", None)                                             # model tanpa INPUT


def test_metode_tidak_dikenal_error():
    with pytest.raises(ValueError, match="tidak dikenal"):
        B.periksa("SMOTE", "fitur")
    with pytest.raises(ValueError, match="tidak dikenal"):
        B.periksa("oversampling", "urutan")


@pytest.mark.parametrize("input_model", ["urutan", "fitur"])
def test_over_seimbang_ke_kelas_terbanyak_dengan_salinan_asli(input_model):
    d = _data()
    o = B.terapkan(d, "over", 5, input_model)
    assert np.bincount(o.y, minlength=3).tolist() == [600, 600, 600]
    # window asli dipakai persis dan ada di depan; Normal tidak ditambah
    n = len(d.y)
    for a in ("X", "M", "F", "y"):
        assert np.array_equal(getattr(o, a)[:n], getattr(d, a))
    assert (o.y[n:] != 0).all()
    # tambahan = salinan utuh window asli dari kelas yang sama (X, M, F, y sejajar), bukan interpolasi
    asli = {(x.tobytes(), m.tobytes(), f.tobytes(), int(c)) for x, m, f, c in zip(d.X, d.M, d.F, d.y)}
    tambahan = {(x.tobytes(), m.tobytes(), f.tobytes(), int(c)) for x, m, f, c in zip(o.X[n:], o.M[n:], o.F[n:], o.y[n:])}
    assert tambahan <= asli
    # semua window AFL ikut tersalin (bukan hanya sebagian kecil)
    assert len({k for k in tambahan if k[3] == 2}) == 40


def test_over_deterministik_per_seed():
    d = _data()
    o = B.terapkan(d, "over", 5, "urutan")
    assert np.array_equal(B.terapkan(d, "over", 5, "urutan").X, o.X)
    assert not np.array_equal(B.terapkan(d, "over", 6, "urutan").X, o.X)


def test_over_butuh_ketiga_kelas():
    with pytest.raises(ValueError, match="ketiga kelas"):
        B.terapkan(_data(n=(50, 20, 0)), "over", 1, "urutan")


def test_kelas_terlalu_sedikit_error():
    pytest.importorskip("imblearn")
    with pytest.raises(ValueError, match="window per kelas"):
        B.terapkan(_data(n=(50, 20, 4)), "smote", 1, "fitur")
    with pytest.raises(ValueError, match="ketiga kelas"):
        B.terapkan(_data(n=(50, 20, 0)), "under", 1, "urutan")
