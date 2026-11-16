"""
eksperimen.py — definisi, training, dan evaluasi eksperimen BiGRU v11 (pemilik repo). Satu sumber untuk README ("Hasil",
"Hasil 2", "Hasil 3") dan laporan PDF di laporan/.

    python eksperimen.py latih e3              # semua setelan eksperimen (bisa dilanjutkan; fold yang ada dilewati)
    python eksperimen.py latih e3 w1_h16       # satu setelan saja (mis. dibagi ke beberapa GPU / Kaggle)
    python eksperimen.py evaluasi e3           # ringkasan.json per setelan + perbandingan → cache/v11/bigru/<banding>

Jalankan dengan cwd = folder ini, sama dengan jalankan.ipynb. Training memakai bersama.runner.jalankan (sama dengan sel
training notebook), evaluasi memakai bersama.evaluasi (sama dengan gabungan.ipynb). Tidak ada metrik yang ditulis ulang.
"""
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # v11_model_tim/bigru
sys.path[:0] = [str(HERE.parent), str(HERE)]
import numpy as np  # noqa: E402
from bersama import data, evaluasi as E, runner  # noqa: E402

# Setiap eksperimen: setelan baru (dilatih), referensi (pembanding berpasangan utama), pembanding lama (ikut dibandingkan,
# tidak dilatih ulang), dan analisis tambahan. Nama setelan = NAMA_SETELAN = folder cache/v11/bigru/<nama>/.
EKSPERIMEN = {
    "e1": dict(
        judul="Bobot kelas lunak vs undersampling", tanggal="23 Oktober 2026", banding="banding_e1_balancing.json",
        referensi="bawaan", setelan={"bawaan": {}, "under": {"balancing": "under"}}, pembanding=[],
        crossfit=False, bersarang=["bawaan", "under"], vs_v3=["bawaan"]),
    "e2": dict(
        judul="Bobot kelas, kapasitas, resep, dan oversampling", tanggal="2 November 2026", banding="banding_tuning.json",
        referensi="bawaan",
        setelan={"bawaan_ulang": {}, "wp075": {"weight_power": 0.75}, "wp100": {"weight_power": 1.0},
                 "over": {"balancing": "over"}, "h16": {"hidden": 16}, "h64": {"hidden": 64}, "lr5e4": {"lr": 0.0005},
                 "ep50": {"epochs": 50}},
        pembanding=["under"], crossfit=True, bersarang="top3", vs_v3=["wp100"]),
    "e3": dict(
        judul="Regularisasi di atas bobot kelas penuh", tanggal="2 November 2026", banding="banding_putaran3.json",
        referensi="wp100",
        setelan={"w1_h16": {"weight_power": 1.0, "hidden": 16}, "w1_ep20": {"weight_power": 1.0, "epochs": 20},
                 "w1_bs512": {"weight_power": 1.0, "batch_size": 512},
                 "w1_wd1e3": {"weight_power": 1.0, "weight_decay": 0.001},
                 "w1_wd1e2": {"weight_power": 1.0, "weight_decay": 0.01},
                 "w1_lr5e4": {"weight_power": 1.0, "lr": 0.0005}},
        pembanding=["bawaan"], crossfit=True, bersarang="top3", vs_v3=[]),
}
DUPLIKAT = {"bawaan_ulang"}          # setelan kontrol noise: tidak ikut seleksi cross-fit / bersarang


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
    akar = data.lokasi_hasil() / "bigru"
    ref = eks["referensi"]
    semua = [ref, *eks["setelan"], *eks["pembanding"]]
    semua = list(dict.fromkeys(semua))
    for s in semua:
        if len(list((akar / s).glob("r?_k?.npz"))) != 25:
            raise RuntimeError(f"{akar / s} belum lengkap 25 fold; jalankan `python eksperimen.py latih {nama_eks}` dulu.")
    ringkasan = {}
    for s in semua:
        p = akar / s / "ringkasan.json"
        if not p.exists():
            _log(f"laporan_model {s}")
            E.laporan_model(akar / s, d)
        ringkasan[s] = json.loads(p.read_text(encoding="utf-8"))
    oofd = data.lokasi_data() / data.OOF

    def v3():
        return np.stack([np.load(oofd / f"BiGRUAttention_r{r}.npz")["oof"] for r in range(data.N_ULANGAN)])

    def tugas(t):                                              # dijalankan di proses anak (fork: d diwarisi)
        jenis, arg = t
        h = lambda s: runner.muat_hasil(akar / s, d)  # noqa: E731
        if jenis == "banding":
            return t, E.banding_setelan(d, akar / arg, akar / ref)
        if jenis == "v3":
            p_v3, _ = E.keputusan(v3(), d.folds, d.y)
            r = {"metrik": E.tabel_metrik(d, {"v3": p_v3})["v3"], "afl_per_pasien": E.afl_per_pasien(d, p_v3),
                 "afl_palsu": E.afl_palsu(d, p_v3), "banding": {}}
            for s in arg:                                     # paritas lintas mesin (resep sama, mesin berbeda)
                p_s, _ = E.keputusan(h(s).oof, d.folds, d.y)
                r["banding"][f"{s}_vs_v3_bobot_penuh"] = {"banding": E.banding(d, p_s, p_v3)}
            return t, r
        if jenis == "crossfit":
            pc = E.pilih_crossfit(d, [akar / s for s in arg])
            pc["terpilih"] = [list(map(str, x)) for x in pc["terpilih"]]
            return t, {"kandidat": list(arg), **pc}
        if jenis == "tanpa":
            oofs = {"v31": d.oof_v31, "v3_bobot_penuh": v3(), **{s: h(s).oof for s in arg}}
            return t, E.tanpa_04936(d, oofs, list(range(data.N_ULANGAN)))
        if jenis == "bersarang":
            return t, E.bootstrap_bersarang(d, h(arg).oof)
        raise ValueError(jenis)

    kand = [s for s in semua if s not in DUPLIKAT and s not in eks["pembanding"]]
    if eks["bersarang"] == "top3":
        mf = {s: ringkasan[s]["metrik"]["bias"]["Macro-F1"][0] for s in eks["setelan"] if s not in DUPLIKAT}
        bersarang = sorted(mf, key=mf.get, reverse=True)[:3]
    else:
        bersarang = list(eks["bersarang"])
    daftar = ([("banding", s) for s in semua if s != ref] + [("v3", tuple(eks["vs_v3"])), ("tanpa", tuple(semua))]
              + ([("crossfit", tuple(kand))] if eks["crossfit"] else []) + [("bersarang", s) for s in bersarang])
    out = {"eksperimen": nama_eks, "referensi": ref, "setelan": semua, "banding": {}, "bersarang": {}}
    _log(f"{len(daftar)} tugas, {n_proses} proses")
    with ProcessPoolExecutor(n_proses, mp_context=mp.get_context("fork")) as ex:
        global _TUGAS
        _TUGAS = tugas
        for fut in as_completed([ex.submit(_jalankan_tugas, t) for t in daftar]):
            (jenis, arg), r = fut.result()
            _log(f"selesai {jenis} {arg if jenis in ('banding', 'bersarang') else ''}")
            if jenis == "banding":
                out["banding"][f"{arg}_vs_{ref}"] = r
            elif jenis == "v3":
                out["banding"].update(r.pop("banding"))
                out["v3_bobot_penuh"] = r
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
