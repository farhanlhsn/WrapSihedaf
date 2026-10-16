"""
augmentasi.py — logika v7 (augmentasi sinus takikardia sintetis), dipakai notebook dan laporan.

Masalah (v3.2): semua model menebak ritme teratur ≥ ±115 bpm sebagai AFIB/AFL karena AFDB hampir tidak punya ritme Normal
yang cepat. Class weighting / bias tidak menolong (v3.2 hal. 3), SMOTE tidak punya bahan (100 window Normal > 120 bpm,
semuanya tidak teratur).

Ide: buat contoh "Normal cepat" dari urutan detak Normal ASLI pasien latih yang dipadatkan waktunya (IBI × s, s < 1) ke
detak target. Pola variabilitasnya asli, hanya lajunya dinaikkan. Window sintetis dipotong 10 s seperti window asli.

  kontrol      : tanpa sintetis = resep v5b (bobot lunak). Hanya ulangan 0: uji paritas + pembanding uji stres per fold.
  takikardia130: + N_SYN window sintetis per fold, detak target 100–130 bpm (di bawah daerah AFL 2:1 di AFDB).
  takikardia160: + N_SYN window sintetis per fold, detak target 100–160 bpm (menutupi daerah AFL 2:1; eksploratif).

Resep: bobot kelas lunak (akar) seperti aturan AGENTS.md, fold/seed/hiperparameter/window uji sama dengan referensi.
Fungsi training = v5_training_afl/latih.train_dl_weighted (diimpor, bukan disalin). Bobot setiap model fold disimpan
(syarat uji stres, AGENTS.md protokol butir 9).
"""
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "v3.2_uji_takikardia"), str(REPO / "v5_training_afl"),
                str(REPO / "v4_smoothing_ensemble"), str(REPO / "v3_evaluasi_robust"), str(REPO / "v2_revisi")]
import sihedaf_cv as cv  # noqa: E402
import hrv_core  # noqa: E402
import latih as L5  # noqa: E402
import takikardia as T  # noqa: E402

DATA_DIR = REPO / "data/afdb"
DATA_STD = REPO / "cache/dataset_v2_detector.npz"
OOF_DIR = REPO / "cache/oof_robust/detector_e30_r5_s2026"
CACHE_OUT = REPO / "cache/v7"
BEATS = CACHE_OUT / "detak_afdb.npz"
MODELS = CACHE_OUT / "model"
N_REPEATS, SEED = 5, 2026
WEIGHT_POWER = 0.5                       # bobot lunak (resep bawaan model baru, v5)
N_SYN = 6000                             # window sintetis per fold (±7% window Normal latih)
N_HO = 1000                              # window uji sintetis dari pasien UJI per fold (pasien tidak ikut dilatih)
HO_RANGE = (100.0, 160.0)
CV_MAX = 0.10                            # batas keteraturan window sintetis (AFIB p10 = 0,13; lihat notebook bagian 1)
HO_BINS = [(100, 110), (110, 120), (120, 130), (130, 140), (140, 150), (150, 160)]
VARIANTS = {
    "kontrol": dict(label="kontrol: resep v5b tanpa sintetis", hr=None, repeats=(0,)),
    "takikardia130": dict(label="7a: + sintetis 100–130 bpm", hr=(100.0, 130.0), repeats=tuple(range(N_REPEATS))),
    "takikardia160": dict(label="7b: + sintetis 100–160 bpm", hr=(100.0, 160.0), repeats=tuple(range(N_REPEATS))),
}
PRIMARY = "takikardia130"                # ditetapkan sebelum melihat hasil


# ── 1. urutan detak per rekaman ────────────────────────────────────────
def record_beats(rec):
    """
    Urutan detak satu rekaman: (t_end detik, IBI ms, label ritme per detak). Langkah deteksinya sama dengan
    v5 latih._process_record (= ekstraktor v2 sumber "detector"); verify_beats() membuktikan window yang dipotong dari
    urutan ini identik bit demi bit dengan ../cache/dataset_v2_detector.npz.
    """
    import wfdb
    path = str(DATA_DIR / rec)
    fs = wfdb.rdheader(path).fs
    atr = wfdb.rdann(path, "atr")
    sig = np.nan_to_num(wfdb.rdrecord(path, channels=[L5.ECG_CHANNEL]).p_signal[:, 0])
    x50 = hrv_core.resample_signal(sig, fs, hrv_core.FS_DEVICE)
    peaks, fs_pk = hrv_core.detect_rpeaks(x50, hrv_core.FS_DEVICE), float(hrv_core.FS_DEVICE)
    if peaks is None or len(peaks) < hrv_core.MIN_IBI + 1:
        return np.zeros(0), np.zeros(0), np.zeros(0, np.int64)
    rhythm = [(s, a.strip("\x00").strip()) for s, a in zip(atr.sample, atr.aux_note)
              if a.strip("\x00").strip().startswith("(")]
    r_samp = np.array([s for s, _ in rhythm])
    r_lab = np.array([L5.LABEL_MAP.get(a, -1) for _, a in rhythm])
    t_end, ibi = hrv_core.peaks_to_ibi(peaks, fs_pk)
    close = peaks[1:] * (fs / fs_pk)
    idx = np.searchsorted(r_samp, close, side="right") - 1
    lab = np.where(idx >= 0, r_lab[np.clip(idx, 0, None)], -1)
    return t_end, ibi, lab


def windows_from_beats(t_end, ibi, lab, stride=5.0):
    """Potong window 10 s seperti ekstraktor v2 (window valid digeser `stride`, tidak valid 1 s)."""
    rows, t, W = [], 0.0, hrv_core.WINDOW_SEC
    if len(t_end) == 0:
        return rows
    while t + W <= t_end[-1]:
        i0, i1 = np.searchsorted(t_end, [t, t + W], side="left")
        wl = lab[i0:i1]
        ok, _ = hrv_core.window_quality(ibi[i0:i1])
        if ok and wl.size and wl[0] >= 0 and np.all(wl == wl[0]):
            rows.append((t, ibi[i0:i1].copy(), int(wl[0])))
            t += stride
        else:
            t += L5.INVALID_STEP_SEC
    return rows


def build_beats():
    """Ekstrak urutan detak 23 rekaman → cache, lalu buktikan identik dengan dataset v2. Dilewati jika sudah ada."""
    if not BEATS.exists():
        recs = [l.strip() for l in open(DATA_DIR / "RECORDS") if l.strip() and l.strip() not in L5.BLACKLIST]
        out = {}
        for rec in recs:
            t_end, ibi, lab = record_beats(rec)
            out.update({f"{rec}__t": t_end, f"{rec}__ibi": ibi, f"{rec}__lab": lab})
        CACHE_OUT.mkdir(parents=True, exist_ok=True)
        tmp = BEATS.with_name(BEATS.stem + ".tmp.npz")
        np.savez_compressed(tmp, **out)
        tmp.replace(BEATS)
    return verify_beats()


def load_beats():
    z = np.load(BEATS)
    recs = sorted({k.split("__")[0] for k in z.files})
    return {r: (z[f"{r}__t"], z[f"{r}__ibi"], z[f"{r}__lab"]) for r in recs}


def verify_beats():
    beats = load_beats()
    rows = [(rec, *w) for rec, (t, i, l) in beats.items() for w in windows_from_beats(t, i, l)]
    seqs = [hrv_core.build_sequence(w[2]) for w in rows]
    ref = np.load(DATA_STD)
    X = np.stack([s[0] for s in seqs]).astype(np.float32)
    M = np.stack([s[1] for s in seqs]).astype(np.float32)
    y = np.array([w[3] for w in rows], dtype=np.int64)
    t = np.array([w[1] for w in rows], dtype=np.float32)
    rec = np.array([w[0] for w in rows])
    for k, v in (("X", X), ("M", M), ("y", y), ("t_start", t)):
        assert np.array_equal(v, ref[k]), f"window dari urutan detak berbeda dengan dataset v2 pada '{k}'"
    assert np.array_equal(rec, ref["records"].astype(str))
    n_beats = sum(len(b[1]) for b in beats.values())
    n_norm = sum(int((b[2] == 0).sum()) for b in beats.values())
    return {"rekaman": len(beats), "detak": n_beats, "detak Normal": n_norm, "window": int(len(y))}


# ── 2. window sintetis ─────────────────────────────────────────────────
def synth_windows(beats, patients, n, hr_range, seed, max_tries=50):
    """
    n window "Normal cepat" dari detak Normal pasien `patients`: mulai di detak acak, detak target ~ U(hr_range),
    IBI × s (s = IBI target / rata-rata IBI 30 detak sumber, harus ≤ 1), fase awal window acak, panjang 10 s.
    Semua detak sumber harus berlabel Normal dan lolos window_quality setelah dipadatkan; detak window harus di dalam
    hr_range dan CV IBI ≤ CV_MAX.
    Return: daftar IBI (ms), detak aktual (bpm), pasien sumber.
    """
    rng = np.random.default_rng(seed)
    pool = [(p, np.flatnonzero(beats[p][2] == 0)) for p in patients]
    pool = [(p, ix) for p, ix in pool if ix.size]
    sizes = np.array([ix.size for _, ix in pool], dtype=np.float64)
    W = hrv_core.WINDOW_SEC * 1000.0
    out_ibi, out_hr, out_src = [], [], []
    tries = 0
    while len(out_ibi) < n:
        tries += 1
        assert tries < n * max_tries, "terlalu banyak sampel ditolak; cek data detak"
        j = rng.choice(len(pool), p=sizes / sizes.sum())
        p, ix = pool[j]
        i = int(ix[rng.integers(ix.size)])
        _, ibi, lab = beats[p]
        target = rng.uniform(*hr_range)
        src = ibi[i:i + 60]
        if src.size < 30:
            continue
        s = (60000.0 / target) / src[:30].mean()
        if s > 1.0:                                       # sumber sudah lebih cepat dari target → bukan pemadatan
            continue
        scaled = src * s
        end = np.cumsum(scaled)
        phase = rng.uniform(0.0, scaled[0])
        k = np.flatnonzero((end >= phase) & (end < phase + W))
        if k.size == 0 or k[-1] >= src.size - 1:          # sumber terlalu pendek untuk 10 s
            continue
        w = scaled[k[0]:k[-1] + 1]
        if np.any(lab[i + k[0]:i + k[-1] + 1] != 0) or not hrv_core.window_quality(w)[0]:
            continue
        # sinus takikardia = teratur: buang segmen berdenyut ektopik (cepat + tidak teratur adalah ciri AFIB)
        if not (hr_range[0] <= 60000.0 / w.mean() < hr_range[1]) or w.std() / w.mean() > CV_MAX:
            continue
        out_ibi.append(w.astype(np.float32))
        out_hr.append(60000.0 / w.mean())
        out_src.append(p)
    return out_ibi, np.array(out_hr), np.array(out_src)


def to_arrays(ibis):
    seqs = [hrv_core.build_sequence(v) for v in ibis]
    return np.stack([s[0] for s in seqs]).astype(np.float32), np.stack([s[1] for s in seqs]).astype(np.float32)


def stress_set():
    """Window uji stres v3.2 (ritme teratur 60–170 bpm × 4 jenis variasi × 50), urutan: variasi, bpm, window."""
    ibis, meta = [], []
    for j, (jname, (kind, scale)) in enumerate(T.JITTERS.items()):
        for bpm in T.BPMS:
            ibis += T.synthetic_ibi(bpm, kind, scale, T.N_SYNTH, T.SEED + 1000 * j + bpm)
            meta += [(jname, bpm)] * T.N_SYNTH
    X, M = to_arrays(ibis)
    return X, M, meta


# ── 3. training ────────────────────────────────────────────────────────
def job_paths(variant, r):
    oof = (CACHE_OUT if variant == "kontrol" else OOF_DIR) / f"MicroTransformer_{variant}_r{r}.npz"
    return oof, CACHE_OUT / f"uji_{variant}_r{r}.npz"


def run_job(job):
    """Satu (varian, ulangan): latih 5 fold, simpan OOF, prediksi uji stres & sintetis pasien uji, bobot model."""
    import torch
    variant, r = job
    p_oof, p_extra = job_paths(variant, r)
    if p_oof.exists() and p_extra.exists():
        return variant, r, None
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = VARIANTS[variant]
    z = np.load(DATA_STD)
    X, M, y, rec = z["X"], z["M"], z["y"], z["records"].astype(str)
    folds_r = np.load(OOF_DIR / "folds.npz")["folds"][r]
    beats = load_beats()
    Xs, Ms, _ = stress_set()
    oof = np.full((len(y), 3), np.nan, dtype=np.float32)
    stress = np.zeros((5, len(Xs), 3), np.float32)
    ho_p, ho_hr, ho_src, times, n_syn = [], [], [], [], []
    MODELS.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        tr, te = folds_r != k, folds_r == k
        Xtr, Mtr, ytr = X[tr], M[tr], y[tr]
        if cfg["hr"] is not None:
            ibis, _, _ = synth_windows(beats, np.unique(rec[tr]), N_SYN, cfg["hr"], seed=SEED + 100 * r + k + 7000)
            Xa, Ma = to_arrays(ibis)
            Xtr, Mtr = np.concatenate([Xtr, Xa]), np.concatenate([Mtr, Ma])
            ytr = np.concatenate([ytr, np.zeros(len(Xa), np.int64)])
        n_syn.append(len(ytr) - int(tr.sum()))
        net, mu, sd, t = L5.train_dl_weighted(Xtr, Mtr, ytr, weight_power=WEIGHT_POWER,
                                              seed=SEED + 100 * r + k, device=device)
        oof[te] = cv.predict_dl(net, X[te], M[te], mu, sd, device)
        stress[k] = cv.predict_dl(net, Xs, Ms, mu, sd, device)
        # uji sintetis dari pasien UJI fold ini (seed tidak bergantung varian → window sama untuk semua varian)
        ibis, hr, src = synth_windows(beats, np.unique(rec[te]), N_HO, HO_RANGE, seed=SEED + 100 * r + k + 9000)
        Xh, Mh = to_arrays(ibis)
        ho_p.append(cv.predict_dl(net, Xh, Mh, mu, sd, device)); ho_hr.append(hr); ho_src.append(src)
        torch.save({"state_dict": {a: b.cpu() for a, b in net.state_dict().items()}, "mu": torch.from_numpy(mu), "sd": torch.from_numpy(sd),
                    "model_kwargs": L5.MODEL_KWARGS, "variant": variant, "repeat": r, "fold": k},
                   MODELS / f"{variant}_r{r}_k{k}.pt")
        times.append(t)
    assert not np.isnan(oof).any()
    for path, arrays in ((p_extra, dict(stress=stress, ho_p=np.stack(ho_p), ho_hr=np.stack(ho_hr),
                                        ho_src=np.stack(ho_src), n_syn=np.array(n_syn))),
                         (p_oof, dict(oof=oof, times=np.asarray(times)))):
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(path)
    return variant, r, float(np.mean(times))


def all_jobs():
    return [(v, r) for r in range(N_REPEATS) for v, c in VARIANTS.items() if r in c["repeats"]]


def run_all_jobs(log=print):
    """Semua job berurutan di satu proses GPU; urutan per ulangan supaya hasil awal cepat ada. Bisa dilanjutkan."""
    jobs = [j for j in all_jobs() if not all(p.exists() for p in job_paths(*j))]
    log(f"{len(jobs)} job tersisa dari {len(all_jobs())}")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        for v, r, t in ex.map(run_job, jobs):
            log(f"  [{(time.perf_counter() - t0) / 60:5.1f} mnt] {v:<14} ulangan {r}: {t:.0f} s/fold")


# ── 4. evaluasi ────────────────────────────────────────────────────────
def _decide(oofs, folds, y):
    """Bias cross-fit per ulangan → prediksi (R, N) dan bias per (ulangan, fold)."""
    adj = [cv.crossfit_bias(y, o, folds[r]) for r, o in enumerate(oofs)]
    return np.stack([a.argmax(1) for a, _ in adj]), [b for _, b in adj]


def evaluate():
    import pascaproses as P4
    d = P4.load()
    z = np.load(DATA_STD)
    Mz, Iz = z["M"], z["X"][:, :, 0]                       # baca sekali (akses NpzFile membaca ulang seluruh array)
    hr = 60000.0 / ((Iz * Mz).sum(1) / Mz.sum(1))
    main =[v for v in VARIANTS if v != "kontrol"]
    sources = {"v31": [d.oof["MicroTransformer"][r] for r in range(N_REPEATS)],
               "bobotlunak": [np.load(OOF_DIR / f"MicroTransformer_bobotlunak_r{r}.npz")["oof"] for r in range(N_REPEATS)],
               **{v: [np.load(job_paths(v, r)[0])["oof"] for r in range(N_REPEATS)] for v in main}}
    pred, fold_bias = {"v1": d.oof_v1.argmax(-1)}, {}
    for k, oofs in sources.items():
        pred[k], fold_bias[k] = _decide(oofs, d.folds, d.y)
    ev = P4.evaluate(d, pred)

    # Δ berpasangan terhadap v5b (ablasi: resep sama tanpa sintetis), sampel bootstrap sama
    W = cv.bootstrap_weights(d.strata, P4.N_BOOT, seed=SEED)
    cms = {k: np.stack([cv.patient_cms(d.y, p, d.records, d.patients) for p in pred[k]]) for k in ["bobotlunak", *main]}
    boot = {k: cv.boot_metrics(c, W) for k, c in cms.items()}
    point = {k: cv.point_metrics(c) for k, c in cms.items()}
    pairs_v5 = {v: {m: [float(point[v][m].mean() - point["bobotlunak"][m].mean()),
                        *map(float, np.percentile(boot[v][m] - boot["bobotlunak"][m], [2.5, 97.5])),
                        float(np.mean(boot[v][m] - boot["bobotlunak"][m] > 0))]
                    for m in ("Macro-F1", "AFL F1", "AFL R", "Normal R")} for v in main}

    # paritas: kontrol (ulangan 0) vs OOF v5b ulangan 0
    ctrl = np.load(job_paths("kontrol", 0)[0])["oof"]
    parity = {"selisih prob maks": float(np.abs(ctrl - sources["bobotlunak"][0]).max()),
              "prediksi sama": float(np.mean(_decide([ctrl], d.folds[:1], d.y)[0][0] == pred["bobotlunak"][0]))}

    # sumber AFL palsu
    fp = {}
    for k, v in pred.items():
        c = np.stack([cv.patient_cms(d.y, p, d.records, d.patients) for p in v])
        f = c[:, :, :2, 2].sum(-1).mean(0)
        fp[k] = {"total": float(f.sum()), "pasien_lain": float(f[d.strata != 2].sum())}

    # window asli per pita detak (v3.2) dan sensitivitas tanpa 04936
    dd = dict(y=d.y, rec=d.records, hr=hr)
    by_hr = T.by_hr_bin(dd, {k: pred[k] for k in ["v31", "bobotlunak", *main]})
    sens = {}
    for lab, mask in (("semua pasien", np.ones(len(d.y), bool)), ("tanpa 04936", d.records != T.SUSPECT)):
        pr = {k: np.stack([cv.crossfit_bias(d.y[mask], o[mask], d.folds[r][mask])[0].argmax(1)
                           for r, o in enumerate(sources[k])]) for k in ["v31", *main]}
        sens[lab] = T.metrics_with_ci(dd, pr, mask)

    # uji stres sintetis (v3.2) & sintetis pasien uji — tiap model fold memakai bias cross-fit fold-nya
    _, _, meta = stress_set()
    meta_j = np.array([m[0] for m in meta]); meta_b = np.array([m[1] for m in meta])
    ctrl_bias = _decide([ctrl], d.folds[:1], d.y)[1]
    stress, ho = {}, {}
    for v in VARIANTS:
        reps = VARIANTS[v]["repeats"]
        fb = ctrl_bias if v == "kontrol" else fold_bias[v]
        cls_s, cls_h, hr_h = [], [], []
        for r in reps:
            e = np.load(job_paths(v, r)[1])
            b = fb[reps.index(r)]
            for k in range(5):
                cls_s.append(cv.apply_bias(e["stress"][k], b[k]).argmax(1))
                cls_h.append(cv.apply_bias(e["ho_p"][k], b[k]).argmax(1)); hr_h.append(e["ho_hr"][k])
        cls_s = np.stack(cls_s)                                        # (model, window)
        stress[v] = {j: [np.bincount(cls_s[:, (meta_j == j) & (meta_b == bpm)].ravel(), minlength=3)
                         / (cls_s.shape[0] * T.N_SYNTH) for bpm in T.BPMS] for j in T.JITTERS}
        stress[v] = {j: np.array(t).tolist() for j, t in stress[v].items()}
        ch, hh = np.concatenate(cls_h), np.concatenate(hr_h)
        ho[v] = {f"{lo}–{hi}": np.bincount(ch[(hh >= lo) & (hh < hi)], minlength=3).astype(float).tolist()
                 for lo, hi in HO_BINS}
        ho[v] = {kk: (np.array(c) / max(sum(c), 1)).tolist() for kk, c in ho[v].items()}
    # pembanding: paket produksi v3.1 (dilatih dengan SEMUA pasien, jadi pada uji pasien-uji ia sedikit diuntungkan)
    v32 = __import__("json").loads((T.CACHE_OUT / "ringkasan.json").read_text(encoding="utf-8"))
    stress["v31_paket"] = v32["stress"]["v31"]
    run31 = T.load_packages()["v31"]
    beats = load_beats()
    rec = np.load(DATA_STD)["records"].astype(str)
    ch, hh = [], []
    for k in range(5):
        te = d.folds[0] == k
        ibis, hr_k, _ = synth_windows(beats, np.unique(rec[te]), N_HO, HO_RANGE, seed=SEED + k + 9000)
        ch.append(run31([hrv_core.build_sequence(v) for v in ibis])); hh.append(hr_k)
    ch, hh = np.concatenate(ch), np.concatenate(hh)
    ho["v31_paket"] = {f"{lo}–{hi}": (np.bincount(ch[(hh >= lo) & (hh < hi)], minlength=3)
                                      / max(((hh >= lo) & (hh < hi)).sum(), 1)).tolist() for lo, hi in HO_BINS}
    threshold = {v: {j: T.normal_threshold(np.array(t)) for j, t in s.items()} for v, s in stress.items()}

    extra0 = np.load(job_paths(PRIMARY, 0)[1])
    train_time = {v: float(np.mean([np.load(job_paths(v, r)[0])["times"].mean() for r in VARIANTS[v]["repeats"]]))
                  for v in VARIANTS}
    return {
        "labels": {"v1": "v1 (sebelum revisi), argmax", "v31": "v3.1 referensi", "bobotlunak": "v5b bobot lunak",
                   **{v: c["label"] for v, c in VARIANTS.items()}},
        "primary": PRIMARY,
        "rows": ev.rows, "episodes": ev.epi, "pairs": ev.pairs, "pairs_v5": pairs_v5,
        "afl_patients": ev.afl_patients, "afl_recall": ev.afl_recall, "fp": fp,
        "bias": {k: {"median": np.median(np.concatenate(b), 0).tolist(), "min": np.concatenate(b).min(0).tolist(),
                     "max": np.concatenate(b).max(0).tolist()} for k, b in fold_bias.items()},
        "parity": parity, "by_hr": by_hr, "sens": sens,
        "stress": stress, "threshold": threshold, "bpms": T.BPMS, "jitters": list(T.JITTERS),
        "heldout": ho, "ho_bins": [f"{lo}–{hi}" for lo, hi in HO_BINS],
        "n_syn": extra0["n_syn"].tolist(), "train_time": train_time,
    }
