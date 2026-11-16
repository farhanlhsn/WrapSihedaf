"""
eksplorasi_ektopik.py — analisis PASCA-HASIL v3.4 (eksploratif, bukan uji independen; data sudah dipakai di v3.4).

Temuan: F1 gagal H6 karena MIT-BIH 210/217. Di sana F1 menggeser detak ektopik ventrikel (V) −50…−62 ms ke batas lag: templat =
median detak normal, sehingga bentuk PVC tidak punya puncak korelasi yang benar.

Varian F1b (satu varian, ditetapkan sebelum dihitung): geser HANYA detak dominan (korelasi ≥ 0,8 dan ukuran 0,6–1,67, ambang yang
sama); detak lain tetap di posisi detektor lama. Dinilai dengan paket v3.1 (drop-in) dan model v7 kontrol (drop-in AFDB, pola H5).
Hasil hanya menjadi hipotesis untuk rancangan berikutnya yang diuji di data perangkat.
"""
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import pembersih as P

OUT = P.CACHE / "eksplorasi_f1b"
KODE = {1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S", 10: "E", 11: "j", 12: "/", 13: "Q"}


def peaks_f1b(o):
    return np.where(o["dominan"] & np.isfinite(o["korelasi"]), o["p_f1"], o["p_lama"])


def galat_mitbih():
    """|galat| posisi per jenis detak (N, V, /) vs anotasi MIT-BIH, lama / F1 / F1b (offset median detak N dibuang)."""
    tot = {}
    for rec in (P.REPO / "data/mitdb/RECORDS").read_text().split():
        x, fs = P.A12.baca_dat(P.REPO / "data/mitdb" / rec, 0)
        s, k, _ = P.A12.baca_atr(P.REPO / "data/mitdb" / rec)
        b = np.isin(k, list(KODE))
        bs, bk = s[b] / fs, np.array([KODE[int(c)] for c in k[b]])
        o = P.proses_sinyal(x, fs)
        for fe, p in (("lama", o["p_lama"]), ("f1", o["p_f1"]), ("f1b", peaks_f1b(o))):
            e, j = P.cocokkan(p / P.FS, bs, 0.15)
            off = np.median(e[bk[j] == "N"]) if (bk[j] == "N").any() else np.median(e)
            for c in ("N", "V", "/"):
                tot.setdefault(fe, {}).setdefault(c, []).append(np.abs(e[bk[j] == c] - off) * 1000)
    return {fe: {c: {"n": int(sum(len(a) for a in v)), "median": float(np.median(np.concatenate(v))),
                     "lebih_30ms": float(np.mean(np.concatenate(v) > 30))} for c, v in d.items()} for fe, d in tot.items()}


def _eks(arg):
    """Satu rekaman eksternal: kelas paket v3.1 untuk F1b pada grid window yang sama dengan cache v3.4."""
    ds, rec, lab = arg
    p = OUT / ds.replace(" ", "_") / (rec.replace("/", "__") + ".npz")
    if p.exists():
        return
    if "v31" not in P._ALAT:
        P._ALAT["v31"] = P.paket_v31()
    x, fs, label, _ = P._sinyal_label(ds, rec, lab)
    o = P.proses_sinyal(np.nan_to_num(np.asarray(x, np.float64)), fs)
    t_end, ibi = P.hrv_core.peaks_to_ibi(peaks_f1b(o), P.FS)
    kls, ok_l = [], []
    for s, w, ok, _ in P._grid(t_end, ibi, o["x"].size / P.FS):
        if label(s) is None:
            continue
        ok_l.append(ok)
        kls.append(P.hrv_core.build_sequence(w) if ok else None)
    k = np.full(len(kls), -1, np.int8)
    ok = np.array(ok_l, bool)
    if ok.any():
        X = np.stack([a[0] for a in kls if a is not None]); M = np.stack([a[1] for a in kls if a is not None])
        k[ok] = P._ALAT["v31"].predict_proba_sequences(X, M).argmax(1)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p.with_suffix(".tmp.npz"), v31_f1b=k, ok_f1b=ok)
    p.with_suffix(".tmp.npz").replace(p)


def jalankan_eksternal(workers=4, log=print):
    jobs = [(ds, rec, lab) for ds in ("LTAFDB", "CPSC 2021", "MIT-BIH", "CinC 2017") for rec, lab in P.daftar_eksternal(ds)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(_eks, jobs, chunksize=4))
    log(f"eksternal F1b {(time.perf_counter() - t0) / 60:.1f} menit")


def evaluasi_eksternal():
    R163, U162 = P._modul_eksternal()
    out = {}
    for ds in P.EKSTERNAL:
        Z = P.muat_eksternal(ds)
        kb = np.concatenate([np.load(OUT / ds.replace(" ", "_") / (rec.replace("/", "__") + ".npz"))["v31_f1b"]
                             for rec, _ in P.daftar_eksternal(ds)])
        assert len(kb) == len(Z["y2"])
        y2 = Z["y2"].astype(np.int64)
        out[ds] = {"f1b_vs_lama": U162.A161.banding2_pasien(y2, kb > 0, Z["v31_lama"] > 0, Z["subjek"]),
                   "f1_vs_lama": U162.A161.banding2_pasien(y2, Z["v31_f1"] > 0, Z["v31_lama"] > 0, Z["subjek"])}
    return out


def latihan():
    """Jitter EPFL/Wrist dan alarm < 100 bpm (himpunan H2) untuk F1b dengan paket v3.1."""
    L = P.jendela_latihan()
    K = P.prediksi_latihan(L)
    v31 = P.paket_v31()
    gal = {"EPFL": [], "Wrist": []}
    kb, i = np.full(len(L["start"]), -1), 0
    for sumber, gen in (("EPFL", P.T33.rekaman_epfl), ("Wrist", P.T33.rekaman_wrist), ("EPHNOGRAM", P.T33.rekaman_ephnogram)):
        for rek in gen():
            o = P.proses_sinyal(rek["ecg"], rek["fs"])
            pb = peaks_f1b(o)
            if rek.get("rpeak") is not None:
                e, _ = P.cocokkan(pb / P.FS, np.sort(rek["rpeak"]) / rek["fs"], 0.10)
                gal[sumber].append(np.std(e - np.median(e)) * 1000 if e.size > 2 else np.nan)
            t_end, ibi = P.hrv_core.peaks_to_ibi(pb, P.FS)
            for s, w, ok, _ in P._grid(t_end, ibi, o["x"].size / P.FS, stride=P.hrv_core.WINDOW_SEC):
                assert L["rekaman"][i] == rek["rekaman"] and L["start"][i] == s
                if ok:
                    X, M = P.hrv_core.build_sequence(w)
                    kb[i] = v31.predict_proba_sequences(X[None], M[None]).argmax(1)[0]
                i += 1
    assert i == len(L["start"])
    h2 = np.isin(L["sumber"], ["Wrist", "EPHNOGRAM"]) & (L["hr_lama"] < 100)
    a_l, a_b = (K["v31_lama"] > 0).astype(float), (kb > 0).astype(float)
    return {"jitter": {s: float(np.nanmedian(v)) for s, v in gal.items()},
            "H2": {"lama": float(a_l[h2].mean()), "f1": float((K["v31_f1"][h2] > 0).mean()), "f1b": float(a_b[h2].mean()),
                   **P.boot_subjek_selisih(L["subjek"][h2], a_b[h2] - a_l[h2])},
            "epfl_lolos_qc": float((kb[L["sumber"] == "EPFL"] >= 0).mean())}


def afdb_dropin(workers=4):
    """AFDB window F1b → model v7 kontrol r0 (front-end lama) dengan bias fold lamanya vs OOF-nya di window lama (pola H5)."""
    path = OUT / "dataset_f1b.npz"
    if not path.exists():
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            hasil = list(ex.map(_afdb_rek, P.daftar_rekaman()))
        np.savez_compressed(path, **P._ke_array(hasil))
    z = np.load(path)
    import torch
    from sihedaf_models import build_model
    lama, _ = P.data_afdb()
    zl = np.load(P.DATA_LAMA)
    oof = np.load(P.REPO / "cache/v7/MicroTransformer_kontrol_r0.npz")["oof"]
    adj, b = P.cv.crossfit_bias(zl["y"], oof, np.load(P.OOF_DIR / "folds.npz")["folds"][0])
    folds = P.fold_per_window(z["records"])[0]
    pr = np.zeros(len(z["y"]), np.int64)
    for k in range(5):
        w = torch.load(P.REPO / f"cache/v7/model/kontrol_r0_k{k}.pt", map_location="cpu", weights_only=True)
        net = build_model("MicroTransformer", **w["model_kwargs"]); net.load_state_dict(w["state_dict"]); net.eval()
        te = folds == k
        pr[te] = P.cv.apply_bias(P.cv.predict_dl(net, z["X"][te], z["M"][te], w["mu"].numpy(), w["sd"].numpy(), torch.device("cpu")),
                                 b[k]).argmax(1)
    W = P.bobot_bootstrap(lama["strata"])
    dl = dict(y=zl["y"], rec=zl["records"].astype(str), patients=lama["patients"])
    db = dict(y=z["y"], rec=z["records"].astype(str), patients=lama["patients"])
    ml, bl = P.ringkas_metrik(P.cms(dl, adj.argmax(1)[None]), W)
    mb, bb = P.ringkas_metrik(P.cms(db, pr[None]), W)
    return {"lama": ml, "f1b": mb, "banding": P.banding(bb, bl, {m: v["nilai"] for m, v in mb.items()}, {m: v["nilai"] for m, v in ml.items()})}


def _afdb_rek(rec):
    x, fs = P.A12.baca_dat(P.AFDB / rec, P.L5.ECG_CHANNEL)
    s, _, aux = P.A12.baca_atr(P.AFDB / rec)
    rhythm = [(int(a), b.strip("\x00").strip()) for a, b in zip(s, aux) if b.strip("\x00").strip().startswith("(")]
    r_samp = np.array([a for a, _ in rhythm]); r_lab = np.array([P.L5.LABEL_MAP.get(b, -1) for _, b in rhythm])
    o = P.proses_sinyal(x, fs)
    p = peaks_f1b(o)
    t_end, ibi = P.hrv_core.peaks_to_ibi(p, P.FS)
    return rec, P._window_afdb(t_end, ibi, P._label_detak(p, fs, r_samp, r_lab), o["dominan"][1:])


if __name__ == "__main__":
    import json
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)  # noqa: E731
    R = {}
    R["galat_mitbih"] = galat_mitbih(); log("galat MIT-BIH selesai")
    R["latihan"] = latihan(); log("latihan selesai")
    jalankan_eksternal(log=log)
    R["eksternal"] = evaluasi_eksternal(); log("eksternal selesai")
    R["afdb_dropin"] = afdb_dropin(); log("AFDB drop-in selesai")
    P.simpan_json(R, OUT / "ringkasan.json"); log("SELESAI")
