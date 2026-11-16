"""
eksperimen.py — definisi, training, dan evaluasi eksperimen TCN v11 (dijalankan pemilik repo). Satu sumber untuk README
("Hasil", "Hasil 2") dan laporan PDF di laporan/. Pola sama dengan ../bigru/eksperimen.py.

    python eksperimen.py latih e2              # semua setelan eksperimen (bisa dilanjutkan; fold yang ada dilewati)
    python eksperimen.py latih e2 k16 d02      # sebagian setelan saja (mis. satu proses per setelan di satu GPU)
    python eksperimen.py evaluasi e2           # ringkasan.json per setelan + perbandingan → cache/v11/tcn/<banding>

Jalankan dengan cwd = folder ini, sama dengan jalankan.ipynb. Training memakai bersama.runner.jalankan (sama dengan sel
training notebook), evaluasi memakai bersama.evaluasi (sama dengan gabungan.ipynb). Tidak ada metrik yang ditulis ulang.
Selain perbandingan dengan referensi, setiap setelan dibandingkan berpasangan dengan BiGRU v11 (bawaan dan wp100), dan
dicatat AUROC AFL serta bias AFL per fold (kelemahan TCN bawaan: bias yang melompat antar fold menghapus AFL 08378).
"""
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # v11_model_tim/tcn
sys.path[:0] = [str(HERE.parent), str(HERE)]
import numpy as np  # noqa: E402
from bersama import data, evaluasi as E, runner  # noqa: E402

# Setiap eksperimen: setelan baru (dilatih), referensi (pembanding berpasangan utama), pembanding lama (ikut dibandingkan,
# tidak dilatih ulang), dan analisis tambahan. Nama setelan = NAMA_SETELAN = folder cache/v11/tcn/<nama>/.
EKSPERIMEN = {
    "e1": dict(
        judul="TCN setelan bawaan", tanggal="2 November 2026", banding="banding_tcn.json",
        referensi="bawaan", setelan={"bawaan": {}}, pembanding=[], crossfit=False, bersarang=[]),
    "e2": dict(
        judul="Bobot kelas, kapasitas, dan regularisasi", tanggal="2 November 2026", banding="banding_e2_tuning.json",
        referensi="bawaan_ulang",                     # bawaan dilatih ulang di GPU yang sama dengan setelan lain
        setelan={"bawaan_ulang": {}, "wp100": {"weight_power": 1.0}, "wp025": {"weight_power": 0.25},
                 "k16": {"kanal": 16}, "d02": {"dropout": 0.2}, "dil124": {"dilasi": [1, 2, 4]}},
        pembanding=["bawaan"], crossfit=True, bersarang="top3"),
}
BIGRU = ("bawaan", "wp100")          # setelan BiGRU v11 pembanding (cache/v11/bigru/<nama>/)


def _log(m, t0=time.time()):
    print(f"[{(time.time() - t0) / 60:5.1f} mnt] {m}", flush=True)


def latih(nama_eks: str, hanya=None) -> None:
    import model as MDL
    eks = EKSPERIMEN[nama_eks]
    for nama, setelan in eks["setelan"].items():
        if hanya and nama not in hanya:
            continue
        _log(f"latih {nama} {setelan}")
        runner.jalankan(MDL, nama, setelan, log=_log)


def evaluasi(nama_eks: str, n_proses: int = 4) -> Path:
    eks = EKSPERIMEN[nama_eks]
    d = data.muat()
    akar = data.lokasi_hasil() / "tcn"
    akar_bigru = data.lokasi_hasil() / "bigru"
    ref = eks["referensi"]
    semua = list(dict.fromkeys([ref, *eks["setelan"], *eks["pembanding"]]))
    for f in [akar / s for s in semua] + [akar_bigru / s for s in BIGRU]:
        if len(list(f.glob("r?_k?.npz"))) != 25:
            raise RuntimeError(f"{f} belum lengkap 25 fold; jalankan `python eksperimen.py latih {nama_eks}` dulu.")
    ringkasan = {}
    for s in semua:
        p = akar / s / "ringkasan.json"
        if not p.exists():
            _log(f"laporan_model {s}")
            E.laporan_model(akar / s, d)
        ringkasan[s] = json.loads(p.read_text(encoding="utf-8"))

    def tugas(t):                                              # dijalankan di proses anak (fork: d diwarisi)
        from sklearn.metrics import roc_auc_score
        jenis, arg = t
        h = lambda s: runner.muat_hasil(akar / s, d)  # noqa: E731
        if jenis == "banding":
            return t, E.banding_setelan(d, akar / arg, akar / ref)
        if jenis == "diagnosa":                               # vs BiGRU (+ bias), AUROC AFL, bias AFL per fold
            hs = h(arg)
            assert hs.ulangan == list(range(data.N_ULANGAN))
            p_s, b_s = E.keputusan(hs.oof, d.folds, d.y)
            r = {"auroc_afl": [float(roc_auc_score(d.y == 2, o[:, 2])) for o in hs.oof],
                 "bias_afl": b_s[:, :, 2].tolist(), "bias_afib": b_s[:, :, 1].tolist(), "banding_bigru": {}}
            for g in BIGRU:
                p_g, _ = E.keputusan(runner.muat_hasil(akar_bigru / g, d).oof, d.folds, d.y)
                r["banding_bigru"][g] = {"bias": E.banding(d, p_s, p_g), "argmax": E.banding(d, hs.oof.argmax(-1), p_g)}
            return t, r
        if jenis == "bigru":                                  # metrik & AUROC pembanding BiGRU, sekali saja
            r = {}
            for g in BIGRU:
                hg = runner.muat_hasil(akar_bigru / g, d)
                p_g, _ = E.keputusan(hg.oof, d.folds, d.y)
                r[g] = {"metrik": E.tabel_metrik(d, {"bias": p_g})["bias"], "afl_per_pasien": E.afl_per_pasien(d, p_g),
                        "afl_palsu": E.afl_palsu(d, p_g),
                        "auroc_afl": [float(roc_auc_score(d.y == 2, o[:, 2])) for o in hg.oof]}
            r["v31"] = {"auroc_afl": [float(roc_auc_score(d.y == 2, o[:, 2])) for o in d.oof_v31]}
            return t, r
        if jenis == "crossfit":
            pc = E.pilih_crossfit(d, [akar / s for s in arg])
            pc["terpilih"] = [list(map(str, x)) for x in pc["terpilih"]]
            return t, {"kandidat": list(arg), **pc}
        if jenis == "tanpa":
            oofs = {"v31": d.oof_v31, **{f"bigru_{g}": runner.muat_hasil(akar_bigru / g, d).oof for g in BIGRU},
                    **{s: h(s).oof for s in arg}}
            return t, E.tanpa_04936(d, oofs, list(range(data.N_ULANGAN)))
        if jenis == "bersarang":
            return t, E.bootstrap_bersarang(d, h(arg).oof)
        raise ValueError(jenis)

    kand = [s for s in semua if s not in eks["pembanding"]]
    if eks["bersarang"] == "top3":
        mf = {s: ringkasan[s]["metrik"]["bias"]["Macro-F1"][0] for s in eks["setelan"]}
        bersarang = sorted(mf, key=mf.get, reverse=True)[:3]
    else:
        bersarang = list(eks["bersarang"])
    daftar = ([("banding", s) for s in semua if s != ref] + [("diagnosa", s) for s in semua] + [("bigru", None),
              ("tanpa", tuple(semua))] + ([("crossfit", tuple(kand))] if eks["crossfit"] else [])
              + [("bersarang", s) for s in bersarang])
    out = {"eksperimen": nama_eks, "referensi": ref, "setelan": semua, "banding": {}, "diagnosa": {}, "bersarang": {}}
    _log(f"{len(daftar)} tugas, {n_proses} proses")
    with ProcessPoolExecutor(n_proses, mp_context=mp.get_context("fork")) as ex:
        global _TUGAS
        _TUGAS = tugas
        for fut in as_completed([ex.submit(_jalankan_tugas, t) for t in daftar]):
            (jenis, arg), r = fut.result()
            _log(f"selesai {jenis} {arg if jenis in ('banding', 'diagnosa', 'bersarang') else ''}")
            if jenis == "banding":
                out["banding"][f"{arg}_vs_{ref}"] = r
            elif jenis == "diagnosa":
                out["diagnosa"][arg] = r
            elif jenis == "bigru":
                out["pembanding"] = r
            elif jenis == "crossfit":
                out["pilih_crossfit"] = r
            elif jenis == "tanpa":
                out["tanpa_04936"] = r
            else:
                out["bersarang"][arg] = r
    p = akar / eks["banding"]
    p.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    _log(f"selesai → {p}")
    return p


_TUGAS = None


def _jalankan_tugas(t):
    return _TUGAS(t)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("latih", "evaluasi") or sys.argv[2] not in EKSPERIMEN:
        sys.exit(__doc__)
    os.chdir(HERE)
    if sys.argv[1] == "latih":
        latih(sys.argv[2], set(sys.argv[3:]) or None)
    else:
        evaluasi(sys.argv[2], int(os.environ.get("N_PROSES", "4")))
