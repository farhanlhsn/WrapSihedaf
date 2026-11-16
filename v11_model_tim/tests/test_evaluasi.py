import json

import numpy as np
import pytest

from bersama import data, evaluasi


def test_v31_dari_oof_cocok_dengan_v3(d):
    """MicroTransformer OOF + bias cross-fit = 0,832 (cache/oof_robust/.../ringkasan.csv, v3)."""
    pred, bias = evaluasi.keputusan(d.oof_v31, d.folds, d.y)
    assert pred.shape == (5, len(d.y)) and bias.shape == (5, 5, 3)
    assert round(evaluasi.tabel_metrik(d, {"k": pred})["k"]["Macro-F1"][0], 3) == 0.832


def test_banding_diri_sendiri(d):
    pred, _ = evaluasi.keputusan(d.oof_v31[:1], d.folds[:1], d.y)
    b = evaluasi.banding(d, pred, pred)
    assert b["Macro-F1"][0] == 0 and b["Macro-F1"][3] == 0


def test_banding_ulangan_beda_error(d):
    with pytest.raises(ValueError):
        evaluasi.banding(d, d.oof_v1[:2].argmax(-1), d.oof_v1[:1].argmax(-1))


def test_ambang_kontrol_v7(d):
    """Angka v7: kontrol MicroTransformer per fold berhenti menebak Normal di 115–120 bpm."""
    _, b = evaluasi.keputusan(d.kontrol_v7.oof[None], d.folds[:1], d.y)
    a = evaluasi.ambang_takikardia(d.kontrol_v7.stres[None], b, data.set_uji_stres())
    assert a["ambang"]["tanpa variasi"] in (115, 120)
    assert len(a["proporsi"]["tanpa variasi"]) == 23


def test_bias_di_batas():
    b = np.zeros((1, 5, 3))
    b[0, 0, 2] = -7.0
    assert evaluasi.bias_di_batas(b) == 0.1          # 1 dari 10 nilai AFIB/AFL


def test_afl_per_pasien_dan_palsu(d):
    pred = d.oof_v1[:1].argmax(-1)
    rec = evaluasi.afl_per_pasien(d, pred)
    assert len(rec) == 7 and "04936" in rec and all(0 <= v <= 1 for v in rec.values())
    fp = evaluasi.afl_palsu(d, pred)
    assert fp["total"] == pytest.approx(fp["dari_normal"] + fp["dari_afib"])


def _folder_palsu(tmp_path, d, ulangan=(0,)):
    """Hasil palsu: OOF BiGRUAttention v3 (hanya ada di cache repo, tidak di zip) + prediksi uji stres kontrol v7."""
    f = tmp_path / "bigru_v3" / "bawaan"
    f.mkdir(parents=True)
    (f / "setelan.json").write_text(json.dumps({"model": "bigru_v3", "setelan": {}, "quick": False}))
    for r in ulangan:
        oof = np.load(data.REPO / "cache" / data.OOF / f"BiGRUAttention_r{r}.npz")["oof"]
        for k in range(5):
            te = np.flatnonzero(d.folds[r] == k)
            np.savez(f / f"r{r}_k{k}.npz", oof=oof[te], idx=te, stres=d.kontrol_v7.stres[k], waktu=1.0,
                     info=json.dumps({"parameter": 1}))
    return f


def test_laporan_model_ulangan_terpotong(tmp_path, d):
    f = _folder_palsu(tmp_path, d)
    r = evaluasi.laporan_model(f, d)
    assert r["ulangan"] == [0] and (f / "ringkasan.json").exists()
    pred_v31, _ = evaluasi.keputusan(d.oof_v31[[0]], d.folds[[0]], d.y)
    titik = evaluasi.tabel_metrik(d, {"v31": pred_v31})["v31"]["Macro-F1"][0]
    assert r["metrik"]["v31"]["Macro-F1"][0] == pytest.approx(titik)              # v3.1 dipotong ke ulangan 0
    assert set(r["lingkungan"]) == {"commit", "ada_perubahan", "gpu", "pustaka"}
    assert set(r["afl_per_pasien"]) == {"v1", "v31", "argmax", "bias"}
    json.dumps(r)                                                                  # bisa diserialisasi


def test_kumpulkan_ringkasan_abaikan_quick(tmp_path):
    for nama, quick in (("a", False), ("b", True)):
        p = tmp_path / "m" / nama
        p.mkdir(parents=True)
        (p / "ringkasan.json").write_text(json.dumps({"nama_setelan": nama, "quick": quick}))
    assert [r["nama_setelan"] for r in evaluasi.kumpulkan_ringkasan(tmp_path)] == ["a"]


def test_pilih_crossfit(tmp_path, d):
    f1 = _folder_palsu(tmp_path, d)
    f2 = tmp_path / "bigru_v3" / "kedua"
    f2.mkdir()
    for p in f1.iterdir():
        (f2 / p.name).write_bytes(p.read_bytes())
    t = evaluasi.pilih_crossfit(d, [f1, f2])
    assert t["ulangan"] == [0] and len(t["terpilih"][0]) == 5
    assert t["banding_v31"]["Macro-F1"][3] >= 0


def test_banding_setelan(tmp_path, d):
    f1 = _folder_palsu(tmp_path, d, ulangan=(0, 1))
    f2 = tmp_path / "bigru_v3" / "kedua"
    f2.mkdir()
    for p in f1.iterdir():
        if not p.name.startswith("r1_"):                                          # setelan kedua: hanya ulangan 0
            (f2 / p.name).write_bytes(p.read_bytes())
    b = evaluasi.banding_setelan(d, f2, f1)
    assert b["ulangan"] == [0] and b["kandidat"] == "kedua" and b["referensi"] == "bawaan"
    assert b["banding"]["Macro-F1"][0] == 0 and b["banding"]["Macro-F1"][3] == 0     # isi sama → Δ = 0
    assert b["afl_per_pasien"]["kandidat"] == b["afl_per_pasien"]["referensi"]
    lain = tmp_path / "model_lain" / "x"
    lain.mkdir(parents=True)
    for p in f2.iterdir():
        (lain / p.name).write_bytes(p.read_bytes())
    (lain / "setelan.json").write_text(json.dumps({"model": "model_lain", "setelan": {}, "quick": False}))
    with pytest.raises(ValueError, match="satu model"):
        evaluasi.banding_setelan(d, lain, f1)


def test_bersarang_butuh_5_ulangan(d):
    with pytest.raises(ValueError):
        evaluasi.bootstrap_bersarang(d, d.oof_v31[:1])


def test_vonis():
    assert evaluasi.vonis(0.95) == "lebih baik" and evaluasi.vonis(0.05) == "lebih buruk"
    assert evaluasi.vonis(0.5) == "tidak terbedakan"
