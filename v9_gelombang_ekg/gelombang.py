"""
gelombang.py — logika v9 (cabang bentuk gelombang EKG, dilatih dengan AFDB + Chapman berlabel andal).

Masalah (v7–v8): dari interval RR saja, ritme cepat yang teratur (sinus takikardia, AFL 2:1, SVT) tidak bisa dipisahkan.
Pembedanya ada di bentuk gelombang (gelombang P vs gelombang flutter). v6 mencoba CNN EKG dengan 23 pasien AFDB dan overfit
per pasien; v9 memakai arsitektur & resep v6 yang sama (diimpor dari v6_cabang_ekg/ekg.py), ditambah ±42 ribu pasien Chapman.

Data latih (semua varian sama): AFDB fold latih + Chapman fold latih dengan label andal: Normal (SR/SB/ST/SA), AFIB, dan AFL
bagian Shaoxing saja (AFL Ningbo tidak andal, v8). Fold AFDB resmi dan fold Chapman v8 dipakai ulang.

Varian (ditetapkan sebelum melihat hasil):
  rr    : kontrol — MicroTransformer interval RR saja (resep v5b), data latih sama
  dual  : 9a (utama) — dua cabang CNN EKG + RR (DualBranch v6)
  ekg   : 9b — CNN EKG saja (ECGOnly v6)
  fusi  : 9c — rata-rata probabilitas rr dan ekg (dihitung saat evaluasi, tanpa training)
Pembanding dari cache v8 (lingkungan T4 yang sama): v31, afdb, 8c; paket v1/v2/v3.1.

SIHEDAF_QUICK=1: uji alur cepat (1 ulangan, 1 epoch, 4.000 window latih per fold), hanya untuk memeriksa kode.
"""
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "v8_chapman_shaoxing"), str(REPO / "v6_cabang_ekg")]
import chapman as C8  # noqa: E402  (juga memasang path v2–v7)
import ekg as E6  # noqa: E402

cv, hrv_core, L5, T = C8.cv, C8.hrv_core, C8.L5, C8.T

QUICK = os.environ.get("SIHEDAF_QUICK") == "1"
CACHE_OUT = REPO / ("cache/v9_quick" if QUICK else "cache/v9")       # uji cepat tidak tercampur dengan hasil sungguhan
OOF_V9 = CACHE_OUT / "oof"
MODELS = CACHE_OUT / "model"
CH_ECG = REPO / "cache/v9/chapman_ecg.npz"                             # data asli, dipakai bersama
if QUICK:
    C8.N_BOOT = T.N_BOOT = 200
V8_OOF = REPO / "cache/v8/oof"
N_REPEATS = 1 if QUICK else C8.N_REPEATS
SEED = C8.SEED
VARIANTS = {"rr": "kontrol: RR saja, AFDB + Chapman andal",
            "dual": "9a: CNN EKG + RR, AFDB + Chapman andal",
            "ekg": "9b: CNN EKG saja, AFDB + Chapman andal"}
REFS = {"v31": "v31 (resep v3.1, AFDB saja; v8)", "afdb": "afdb (= v5b; v8)", "afdb_chapman_bersih": "8c: RR, tanpa AFL Chapman (v8)"}
PRIMARY = "dual"


# ── 1. data: EKG AFDB (v6) & EKG Chapman sejajar dengan window RR v8 ──
def ensure_afdb_ecg(log=print):
    """Potongan EKG AFDB v6 (dibuat ulang bila belum ada); sejajar dengan window RR (selisih jumlah R-peak harus 0)."""
    if E6.ECG_FILE.exists():
        return {"n": int(len(np.load(E6.ECG_FILE)["ecg"])), "mismatch": "cache"}
    if not (L5.DATA_DIR / "RECORDS").exists():
        import subprocess
        subprocess.run([sys.executable, str(REPO / "tools/download_afdb.py")], check=True)
    t0 = time.perf_counter()
    info = E6.build_ecg_windows()
    assert info["mismatch"] == 0, f"EKG AFDB tidak sejajar dengan window RR: {info}"
    log(f"EKG AFDB: {info['n']} window, sejajar ({(time.perf_counter() - t0) / 60:.1f} mnt)")
    return info


_PATHS = None


def _chapman_paths():
    global _PATHS
    if _PATHS is None:
        _PATHS = {p.stem: p for p in (C8.CH_DIR / "WFDBRecords").rglob("*.hea")}
    return _PATHS


def _ecg_record(args):
    """Satu pasien Chapman → (EKG 500 sampel float16, jumlah IBI yang cocok dengan window RR v8)."""
    from scipy.io import loadmat
    rec, ibi_ref = args
    hea = _chapman_paths()[rec]
    h = C8.parse_header(hea.read_text(encoding="utf-8", errors="replace"))
    val = loadmat(str(hea.with_suffix(".mat")))["val"]
    i = h["leads"].index(C8.LEAD)
    sig = (val[i].astype(np.float64) - h["baseline"][i]) / h["gain"][i]
    _, xf = E6.preprocess_record(sig, h["fs"])
    w = xf[:E6.N_SAMP]
    if w.size < E6.N_SAMP:
        w = np.pad(w, (0, E6.N_SAMP - w.size))
    wins, _ = hrv_core.signal_to_windows(sig, h["fs"], stride_sec=5.0)       # langkah RR yang sama dengan v8
    same = bool(wins) and np.array_equal(np.asarray(wins[0]["ibi_ms"], np.float32)[:hrv_core.MAX_BEATS], ibi_ref)
    return E6.normalize_window(w).astype(np.float16), same


def build_chapman_ecg(log=print, workers=None):
    """EKG lead II (50 Hz, bandpass v6) untuk setiap window Chapman v8 yang valid; dibuktikan sejajar (IBI identik)."""
    if CH_ECG.exists():
        z = np.load(CH_ECG)
        return {"n": int(len(z["ecg"])), "tidak_sejajar": int(z["tidak_sejajar"])}
    ch = C8.load_chapman()
    n_ibi = ch["M"].sum(1).astype(int)
    jobs = [(r, ch["X"][j, :n_ibi[j], 0]) for j, r in enumerate(ch["record"])]
    workers = workers or max(1, os.cpu_count() or 2)
    log(f"EKG Chapman: {len(jobs)} window, {workers} proses ...")
    t0, out, same = time.perf_counter(), [], []
    with mp.get_context("spawn").Pool(workers) as pool:
        for b0 in range(0, len(jobs), 2500):
            try:
                res = pool.map_async(_ecg_record, jobs[b0:b0 + 2500], chunksize=32).get(timeout=900)
            except mp.TimeoutError:
                raise RuntimeError(f"pool EKG macet di batch {b0}")
            out += [r[0] for r in res]; same += [r[1] for r in res]
            log(f"  {len(out)}/{len(jobs)} ({(time.perf_counter() - t0) / 60:.1f} mnt)")
    bad = int(len(same) - sum(same))
    assert bad == 0, f"{bad} window EKG Chapman tidak sejajar dengan window RR v8"
    CH_ECG.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CH_ECG, ecg=np.stack(out), record=ch["record"], tidak_sejajar=bad)
    return {"n": len(out), "tidak_sejajar": bad}


def train_mask(ch):
    """Label Chapman yang dipakai untuk training: Normal, AFIB, dan AFL Shaoxing (bukan AFL Ningbo)."""
    return (ch["cls"] >= 0) & ~((ch["rhythm"] == "AFL") & C8.is_ningbo(ch["record"]))


# ── 2. training ────────────────────────────────────────────────────────
def job_path(variant, r):
    return OOF_V9 / f"{variant}_r{r}.npz"


def run_job(job):
    """Satu (varian, ulangan): 5 fold; OOF AFDB + Chapman, uji stres sintetis (hanya rr), bobot model."""
    import torch
    variant, r = job
    out = job_path(variant, r)
    if out.exists():
        return variant, r, None
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{C8._GPU}" if torch.cuda.is_available() else "cpu")
    z = np.load(C8.DATA_STD)
    X, M, y = z["X"], z["M"], z["y"]
    Ea = np.load(E6.ECG_FILE)["ecg"]
    fa = np.load(C8.FOLDS)["folds"][r]
    ch = C8.load_chapman()
    Ec = np.load(CH_ECG)["ecg"]
    assert np.array_equal(np.load(CH_ECG)["record"], ch["record"])
    fc = C8.chapman_folds(ch["rhythm"])[r]
    use = train_mask(ch)
    Xs, Ms, _ = C8.A7.stress_set()
    oof_a = np.full((len(y), 3), np.nan, np.float32)
    oof_c = np.full((len(ch["cls"]), 3), np.nan, np.float32)
    stress = np.full((5, len(Xs), 3), np.nan, np.float32)
    times = []
    MODELS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 100 * r)
    epochs = 1 if QUICK else None
    for k in range(5):
        ta, tc = np.flatnonzero(fa != k), np.flatnonzero(use & (fc != k))
        if QUICK:
            ta, tc = rng.choice(ta, 3000, replace=False), rng.choice(tc, 1000, replace=False)
        Xtr, Mtr = np.concatenate([X[ta], ch["X"][tc]]), np.concatenate([M[ta], ch["M"][tc]])
        ytr = np.concatenate([y[ta], ch["cls"][tc]])
        seed = SEED + 100 * r + k
        if variant == "rr":
            if QUICK:
                L5.HP["epochs"] = 1
            net, mu, sd, t = L5.train_dl_weighted(Xtr, Mtr, ytr, weight_power=0.5, seed=seed, device=device)
            pa = lambda Xq, Mq, Eq: cv.predict_dl(net, Xq, Mq, mu, sd, device)
            stress[k] = cv.predict_dl(net, Xs, Ms, mu, sd, device)
        else:
            Etr = np.concatenate([Ea[ta], Ec[tc]])
            net, mu, sd, t = E6.train(variant, Etr, Xtr, Mtr, ytr, seed=seed, device=device, epochs=epochs)
            pa = lambda Xq, Mq, Eq: E6.predict(net, Eq, Xq, Mq, mu, sd, device)
        oof_a[fa == k] = pa(X[fa == k], M[fa == k], Ea[fa == k])
        oof_c[fc == k] = pa(ch["X"][fc == k], ch["M"][fc == k], Ec[fc == k])
        torch.save({"state_dict": {a: b.cpu() for a, b in net.state_dict().items()}, "mu": torch.from_numpy(mu),
                    "sd": torch.from_numpy(sd), "variant": variant, "repeat": r, "fold": k}, MODELS / f"{variant}_r{r}_k{k}.pt")
        times.append(t)
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert not np.isnan(oof_a).any() and not np.isnan(oof_c).any()
    OOF_V9.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, oof_afdb=oof_a, oof_chapman=oof_c, stress=stress, times=np.asarray(times))
    tmp.replace(out)
    return variant, r, float(np.mean(times))


def all_jobs():
    # CNN dulu per ulangan (paling lama), rr terakhir
    return [(v, r) for r in range(N_REPEATS) for v in ("dual", "ekg", "rr")]


def run_all_jobs(log=print, n_gpu=None):
    """Satu proses per GPU (pola v8); job yang sudah ada dilewati."""
    import torch
    n_gpu = n_gpu or max(1, torch.cuda.device_count())
    jobs = [j for j in all_jobs() if not job_path(*j).exists()]
    log(f"{len(jobs)} job tersisa dari {len(all_jobs())}, {n_gpu} proses (satu per GPU)")
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for i in range(n_gpu):
        queue.put(i)
    with ProcessPoolExecutor(n_gpu, mp_context=ctx, initializer=C8._init_gpu, initargs=(queue,)) as ex:
        for v, r, t in ex.map(run_job, jobs):
            log(f"  [{(time.perf_counter() - t0) / 60:5.1f} mnt] {v:<5} ulangan {r}: {t:.0f} s/fold")


# ── 3. evaluasi ────────────────────────────────────────────────────────
def load_sources():
    """{nama: [dict(oof_afdb, oof_chapman, stress)] per ulangan} untuk varian v9, fusi, dan pembanding v8."""
    src = {}
    for v in VARIANTS:
        src[v] = [dict(np.load(job_path(v, r))) for r in range(N_REPEATS)]
    src["fusi"] = [{"oof_afdb": (a["oof_afdb"] + b["oof_afdb"]) / 2, "oof_chapman": (a["oof_chapman"] + b["oof_chapman"]) / 2}
                   for a, b in zip(src["rr"], src["ekg"])]
    for v in REFS:
        p = [V8_OOF / f"{v}_r{r}.npz" for r in range(N_REPEATS)]
        assert all(q.exists() for q in p), f"hasil v8 untuk {v} tidak ada di {V8_OOF} (jalankan v8 atau salin cache/v8)"
        src[v] = [dict(np.load(q)) for q in p]
    return src


def evaluate():
    import pascaproses as P4
    if QUICK:
        P4.N_BOOT = 200
    z = np.load(C8.DATA_STD)
    y, rec = z["y"], z["records"].astype(str)
    starts = np.flatnonzero(np.r_[True, rec[1:] != rec[:-1]])
    patients = np.unique(rec)
    folds = np.load(C8.FOLDS)["folds"][:N_REPEATS]
    d = SimpleNamespace(y=y, records=rec, t=z["t_start"].astype(np.float64), patients=patients,
                        bounds=list(zip(starts, np.r_[starts[1:], len(y)])), strata=cv.patient_strata(y, rec, patients),
                        folds=folds)
    Mz, Iz = z["M"], z["X"][:, :, 0]
    hr_a = 60000.0 / ((Iz * Mz).sum(1) / Mz.sum(1))
    S = load_sources()
    keys = ["v31", "afdb", "afdb_chapman_bersih", "rr", "dual", "ekg", "fusi"]

    # ── AFDB (protokol v3) ──
    pred, fold_bias = {}, {}
    for k in keys:
        adj = [cv.crossfit_bias(y, S[k][r]["oof_afdb"], folds[r]) for r in range(N_REPEATS)]
        pred[k] = np.stack([a.argmax(1) for a, _ in adj]); fold_bias[k] = [b for _, b in adj]
    v1o = C8.OOF_DIR / "versi_lama"
    if (v1o / "oof_r0.npz").exists():
        pred["v1"] = np.stack([np.load(v1o / f"oof_r{r}.npz")["det"].argmax(-1) for r in range(N_REPEATS)])
    ev = P4.evaluate(d, pred)
    W = cv.bootstrap_weights(d.strata, C8.N_BOOT, seed=SEED)
    cms = {k: np.stack([cv.patient_cms(y, p, rec, patients) for p in pred[k]]) for k in keys}
    boot = {k: cv.boot_metrics(c, W) for k, c in cms.items()}
    point = {k: cv.point_metrics(c) for k, c in cms.items()}

    def pair(a, b):
        return {m: [float(point[a][m].mean() - point[b][m].mean()),
                    *map(float, np.percentile(boot[a][m] - boot[b][m], [2.5, 97.5])),
                    float(np.mean(boot[a][m] - boot[b][m] > 0))] for m in ("Macro-F1", "AFL F1", "AFL R", "Normal R")}
    pairs_rr = {k: pair(k, "rr") for k in ("dual", "ekg", "fusi")}
    pairs_8c = {k: pair(k, "afdb_chapman_bersih") for k in ("rr", "dual", "ekg", "fusi")}
    fp = {}
    for k, v in pred.items():
        c = np.stack([cv.patient_cms(y, p, rec, patients) for p in v])
        f = c[:, :, :2, 2].sum(-1).mean(0)
        fp[k] = {"total": float(f.sum()), "pasien_lain": float(f[d.strata != 2].sum())}
    dd = dict(y=y, rec=rec, hr=hr_a)
    by_hr = T.by_hr_bin(dd, {k: pred[k] for k in keys})
    sens = {}
    for lab, mask in (("semua pasien", np.ones(len(y), bool)), ("tanpa 04936", rec != T.SUSPECT)):
        pr = {k: np.stack([cv.crossfit_bias(y[mask], S[k][r]["oof_afdb"][mask], folds[r][mask])[0].argmax(1)
                           for r in range(N_REPEATS)]) for k in keys}
        sens[lab] = T.metrics_with_ci(dd, pr, mask)

    # ── Chapman (label andal; bias dari AFDB per model fold) ──
    ch = C8.load_chapman()
    fc = C8.chapman_folds(ch["rhythm"])[:N_REPEATS]
    contrib = {}
    for k in keys:
        pb = []
        for r in range(N_REPEATS):
            o, p = S[k][r]["oof_chapman"], np.empty(len(ch["cls"]), np.int64)
            for f in range(5):
                m = fc[r] == f
                p[m] = cv.apply_bias(o[m], fold_bias[k][r][f]).argmax(1)
            pb.append(C8._onehot(p))
        contrib[k] = np.mean(pb, 0)
    runners, pk_note = C8.load_packages()
    for k, f in runners.items():
        contrib[k] = C8._onehot(np.concatenate([f(ch["X"][i:i + 4096], ch["M"][i:i + 4096]) for i in range(0, len(ch["X"]), 4096)]))
    ningbo = C8.is_ningbo(ch["record"])
    m3 = train_mask(ch)
    cls3 = ch["cls"][m3]
    cm_pt = {k: np.stack([v[m3][cls3 == c].sum(0) for c in range(3)]) for k, v in contrib.items()}
    bt = C8.chapman_boot({k: v[m3] for k, v in contrib.items()}, cls3)
    mets = {k: {m: [float(C8._cm_metrics(cm_pt[k])[m]), *map(float, np.percentile(C8._cm_metrics(bt[k])[m], [2.5, 97.5]))]
                for m in ("Macro-F1", "AFL F1", "AFL R", "AFL P", "AFIB F1", "AFIB R", "Normal R")} for k in contrib}
    ch_pairs = {}
    for ref in ("v31_paket", "rr", "afdb_chapman_bersih"):
        for k in contrib:
            if k == ref:
                continue
            row = {}
            for m in ("Macro-F1", "AFL R", "Normal R"):
                dm = C8._cm_metrics(bt[k])[m] - C8._cm_metrics(bt[ref])[m]
                row[m] = [float(C8._cm_metrics(cm_pt[k])[m] - C8._cm_metrics(cm_pt[ref])[m]),
                          *map(float, np.percentile(dm, [2.5, 97.5])), float(np.mean(dm > 0))]
            ch_pairs.setdefault(ref, {})[k] = row
    groups = {}
    for g in sorted(set(ch["rhythm"]), key=lambda g: (g not in C8.GROUP_CLASS, g)):
        if g == "AFL":
            groups["AFL (Shaoxing)"] = (ch["rhythm"] == g) & ~ningbo
            groups["AFL (Ningbo, label tidak andal)"] = (ch["rhythm"] == g) & ningbo
        else:
            groups[g] = ch["rhythm"] == g
    rhythm_tab = {g: {"n": int(m.sum()), **{k: v[m].mean(0).tolist() for k, v in contrib.items()}} for g, m in groups.items()}
    alarm = {k: v[:, 1] + v[:, 2] for k, v in contrib.items()}
    st = ch["rhythm"] == "ST"
    band = lambda m, lo, hi: m & (ch["hr"] >= lo) & (ch["hr"] < hi)
    st_bands = {f"{lo}–{hi if hi < 999 else '∞'}": {"n": int(band(st, lo, hi).sum()),
                                                    **C8.boot_mean({k: a[band(st, lo, hi)] for k, a in alarm.items()})}
                for lo, hi in C8.ST_BANDS}
    st_all = {"n": int(st.sum()), **C8.boot_mean({k: a[st] for k, a in alarm.items()})}
    svt = ch["rhythm"] == "SVT"
    svt_alarm = {"n": int(svt.sum()), **C8.boot_mean({k: a[svt] for k, a in alarm.items()})}
    sinus = np.isin(ch["rhythm"], ["SR", "SB", "SA"])
    sinus_alarm = {"n": int(sinus.sum()), **C8.boot_mean({k: a[sinus] for k, a in alarm.items()})}
    afl = (ch["rhythm"] == "AFL") & ~ningbo
    afl_bands = {f"{lo}–{hi if hi < 999 else '∞'}": {"n": int(band(afl, lo, hi).sum()),
                                                     **{k: float(v[band(afl, lo, hi), 2].mean()) if band(afl, lo, hi).any() else None
                                                        for k, v in contrib.items()}}
                 for lo, hi in [(0, 100), (100, 130), (130, 160), (160, 999)]}

    # ── uji stres sintetis: hanya model interval RR (ritme sintetis tidak punya bentuk gelombang) ──
    _, _, meta = C8.A7.stress_set()
    mj = np.array([m[0] for m in meta]); mb = np.array([m[1] for m in meta])
    stress = {}
    for k in ("rr", "v31", "afdb_chapman_bersih"):
        cls_s = np.stack([cv.apply_bias(S[k][r]["stress"][f], fold_bias[k][r][f]).argmax(1)
                          for r in range(N_REPEATS) for f in range(5)])
        stress[k] = {j: [(np.bincount(cls_s[:, (mj == j) & (mb == b)].ravel(), minlength=3)
                          / cls_s[:, (mj == j) & (mb == b)].size).tolist() for b in T.BPMS] for j in T.JITTERS}
    threshold = {k: {j: T.normal_threshold(np.array(t)) for j, t in s.items()} for k, s in stress.items()}

    return {
        "labels": {**VARIANTS, "fusi": "9c: fusi rr + ekg", **REFS, "v1": "v1 (OOF cache PC tim)",
                   "v1_paket": "paket v1", "v2_paket": "paket v2", "v31_paket": "paket produksi v3.1"},
        "primary": PRIMARY, "quick": QUICK,
        "afdb": {"rows": ev.rows, "episodes": ev.epi, "pairs": ev.pairs, "pairs_rr": pairs_rr, "pairs_8c": pairs_8c,
                 "afl_patients": ev.afl_patients, "afl_recall": ev.afl_recall, "fp": fp, "by_hr": by_hr, "sens": sens,
                 "bias": {k: {"median": np.median(np.concatenate(b), 0).tolist(), "min": np.concatenate(b).min(0).tolist(),
                              "max": np.concatenate(b).max(0).tolist()} for k, b in fold_bias.items()}},
        "chapman": {"metrics": mets, "pairs": ch_pairs, "rhythm": rhythm_tab, "st_bands": st_bands, "st_all": st_all,
                    "svt_alarm": svt_alarm, "sinus_alarm": sinus_alarm, "afl_bands": afl_bands, "paket_catatan": pk_note,
                    "n_metrik": {"Normal": int((cls3 == 0).sum()), "AFIB": int((cls3 == 1).sum()), "AFL": int((cls3 == 2).sum())}},
        "stress": stress, "threshold": threshold, "bpms": T.BPMS, "jitters": list(T.JITTERS),
        "train_time": {v: float(np.mean([e["times"].mean() for e in S[v]])) for v in VARIANTS},
    }


def pack_results(dest):
    import zipfile
    dest = Path(dest)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(CACHE_OUT.rglob("*")):
            if p.is_file() and not p.name.endswith((".tmp.npz", ".zip")) and p != CH_ECG:
                zf.write(p, Path("cache/v9") / p.relative_to(CACHE_OUT))
    return dest, dest.stat().st_size / 1e6
