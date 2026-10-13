"""
pascaproses.py — logika v4 (smoothing antar-window & ensemble), dipakai notebook dan laporan.

Tidak ada training. Semua bekerja di atas OOF v3 (fold & seed resmi):
- smoothing: probabilitas window dihaluskan dengan window lain dari rekaman yang sama
  (rata-rata bergerak atau HMM), versi offline (pakai window sesudahnya) dan real-time (hanya sebelumnya);
- ensemble: rata-rata probabilitas beberapa model.
Konfigurasi smoothing DAN bias kelas dipilih per fold dari fold lain (cross-fit), jadi tidak ada pilihan yang
mengintip pasien uji. Referensi = v3.1 (MicroTransformer, bias cross-fit grid −7…+3, tanpa smoothing).
"""
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "v3_evaluasi_robust"), str(REPO / "v2_revisi")]
import sihedaf_cv as cv  # noqa: E402

OOF_DIR = REPO / "cache/oof_robust/detector_e30_r5_s2026"
DATA = REPO / "cache/dataset_v2_detector.npz"
CACHE_OUT = REPO / "cache/v4"
N_REPEATS, SEED, N_BOOT = 5, 2026, 2000
GAP_SEC = 10.0          # window berjarak > ini dianggap tidak bersambung (episode terputus)
STEP_SEC = 5.0          # jarak normal antar window

# Kandidat smoothing; ("tanpa", 0) ikut supaya cross-fit boleh memilih "tidak dihaluskan".
OFFLINE = [("tanpa", 0)] + [("rata-rata", h) for h in (15, 30, 60, 120)] + [("HMM", p) for p in (0.9, 0.97, 0.99)]
REALTIME = [("tanpa", 0)] + [("rata-rata kausal", h) for h in (15, 30, 60, 120)] + [("HMM kausal", p) for p in (0.9, 0.97, 0.99)]
ENSEMBLES = {"E1": ["MicroTransformer", "BiGRUAttention"],
             "E2": ["MicroTransformer", "BiGRUAttention", "MaskedResNet1D", "LightGBM"]}

# Varian yang dilaporkan: kunci → (label, basis probabilitas, kandidat smoothing)
VARIANTS = {
    "v1": ("v1 (sebelum revisi), argmax", None, None),
    "v31": ("v3.1 referensi: MicroTransformer", "MicroTransformer", [("tanpa", 0)]),
    "mt_off": ("MicroTransformer + smoothing offline", "MicroTransformer", OFFLINE),
    "mt_rt": ("MicroTransformer + smoothing real-time", "MicroTransformer", REALTIME),
    "e1": ("Ensemble MT + BiGRU", "E1", [("tanpa", 0)]),
    "e1_off": ("Ensemble MT + BiGRU + smoothing offline", "E1", OFFLINE),
    "e1_rt": ("Ensemble MT + BiGRU + smoothing real-time", "E1", REALTIME),
    "e2": ("Ensemble 4 model (eksploratif)", "E2", [("tanpa", 0)]),
}
PRIMARY = ["mt_off", "mt_rt", "e1", "e1_off", "e1_rt"]     # ditetapkan sebelum melihat hasil
REF = "v31"


def cfg_name(c):
    kind, val = c
    if kind == "tanpa":
        return "tanpa smoothing"
    return f"{kind} ±{val} s" if kind == "rata-rata" else (f"{kind} {val} s" if "rata" in kind else f"{kind} p={val}")


# ── data ───────────────────────────────────────────────────────────────
def load():
    z = np.load(DATA)
    y, records, t = z["y"], z["records"].astype(str), z["t_start"].astype(np.float64)
    patients = np.unique(records)
    # rekaman kontigu & urut waktu (diperiksa di notebook)
    starts = np.flatnonzero(np.r_[True, records[1:] != records[:-1]])
    bounds = list(zip(starts, np.r_[starts[1:], len(y)]))
    oof = {m: np.stack([np.load(OOF_DIR / f"{m}_r{r}.npz")["oof"] for r in range(N_REPEATS)])
           for m in ("MicroTransformer", "BiGRUAttention", "MaskedResNet1D", "LightGBM")}
    for e, members in ENSEMBLES.items():
        oof[e] = np.mean([oof[m] for m in members], axis=0)
    return SimpleNamespace(
        y=y, records=records, t=t, patients=patients, bounds=bounds,
        strata=cv.patient_strata(y, records, patients),
        folds=np.load(OOF_DIR / "folds.npz")["folds"], oof=oof,
        oof_v1=np.stack([np.load(OOF_DIR / "versi_lama" / f"oof_r{r}.npz")["det"] for r in range(N_REPEATS)]),
    )


# ── smoothing ──────────────────────────────────────────────────────────
def smooth_mean(P, t, bounds, half_width, causal):
    """Rata-rata probabilitas window dalam ±half_width detik (kausal: hanya ke belakang)."""
    out = np.empty_like(P)
    for s, e in bounds:
        tt, pp = t[s:e], P[s:e]
        cs = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(pp, 0)])
        lo = np.searchsorted(tt, tt - half_width, "left")
        hi = np.arange(1, e - s + 1) if causal else np.searchsorted(tt, tt + half_width, "right")
        out[s:e] = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    return out


def smooth_hmm(P, t, bounds, p_stay, causal):
    """
    HMM 3 keadaan. Emisi ∝ probabilitas model (model dilatih dengan bobot kelas seimbang, jadi keluarannya
    mendekati likelihood dengan prior seragam). Transisi per 5 s: tetap p_stay, pindah (1−p_stay)/2 ke tiap
    kelas lain; jeda lebih panjang → p_stay^(Δt/5). Kausal = forward saja; offline = forward-backward.
    """
    out = np.empty_like(P)
    for s, e in bounds:
        pp = np.clip(P[s:e], 1e-9, 1.0)
        n = e - s
        stay = p_stay ** (np.r_[STEP_SEC, np.diff(t[s:e])] / STEP_SEC)
        alpha = np.empty((n, 3))
        a = pp[0] / pp[0].sum()
        alpha[0] = a
        for i in range(1, n):
            st = stay[i]
            a = (st * a + (1 - st) * (1 - a) / 2) * pp[i]
            a /= a.sum()
            alpha[i] = a
        if causal:
            out[s:e] = alpha
            continue
        beta = np.ones(3)
        out[e - 1] = alpha[-1]
        for i in range(n - 2, -1, -1):
            st = stay[i + 1]
            m = beta * pp[i + 1]
            beta = st * m + (1 - st) * (m.sum() - m) / 2
            beta /= beta.sum()
            post = alpha[i] * beta
            out[s + i] = post / post.sum()
    return out


def apply_smoothing(P, d, cfg):
    kind, val = cfg
    if kind == "tanpa":
        return P
    if kind.startswith("rata"):
        return smooth_mean(P, d.t, d.bounds, val, causal="kausal" in kind)
    return smooth_hmm(P, d.t, d.bounds, val, causal="kausal" in kind)


# ── cross-fit: pilih konfigurasi + bias dari fold lain ─────────────────
def crossfit_select(y, cands, folds_r):
    """
    cands: {cfg: probabilitas (N,3)}. Untuk tiap fold uji: tiap kandidat diberi bias terbaik di fold latih,
    kandidat dengan Macro-F1 fold-latih tertinggi dipakai untuk fold uji.
    Return (pred terpilih (N,), {cfg: pred jika cfg itu dipaksa}, [cfg terpilih per fold]).
    """
    per_cfg = {c: np.empty(len(y), dtype=np.int8) for c in cands}
    logps = {c: np.log(np.clip(P, 1e-9, 1)) for c, P in cands.items()}
    chosen_pred = np.empty(len(y), dtype=np.int8)
    chosen = []
    for k in np.unique(folds_r):
        tr, te = folds_r != k, folds_r == k
        best = None
        for c, P in cands.items():
            b = cv.fit_class_bias(y[tr], P[tr])
            logp = logps[c]
            score = cv._macro_f1_fast(y[tr], (logp[tr] + b).argmax(1))
            per_cfg[c][te] = (logp[te] + b).argmax(1)
            if best is None or score > best[0]:
                best = (score, c)
        chosen_pred[te] = per_cfg[best[1]][te]
        chosen.append(best[1])
    return chosen_pred, per_cfg, chosen


def _repeat_job(r):
    """Semua varian untuk satu ulangan (dijalankan paralel per ulangan)."""
    d = load()
    out = {"pred": {"v1": d.oof_v1[r].argmax(1).astype(np.int8)}, "chosen": {}, "per_cfg": {}}
    cache = {}
    for key, (_, base, pool) in VARIANTS.items():
        if base is None:
            continue
        cands = {}
        for c in pool:
            if (base, c) not in cache:
                cache[(base, c)] = apply_smoothing(d.oof[base][r], d, c)
            cands[c] = cache[(base, c)]
        pred, per_cfg, chosen = crossfit_select(d.y, cands, d.folds[r])
        out["pred"][key] = pred
        out["chosen"][key] = [cfg_name(c) for c in chosen]
        if key in ("mt_off", "mt_rt"):
            out["per_cfg"].update({cfg_name(c): p for c, p in per_cfg.items()})
    return r, out


def run_predictions(workers=N_REPEATS):
    """Prediksi semua varian × ulangan (paralel). Disimpan ke cache/v4/pred.npz agar laporan tidak menghitung ulang."""
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    path = CACHE_OUT / "pred.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["pred"].item(), z["chosen"].item(), z["per_cfg"].item()
    res = {}
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        for r, out in ex.map(_repeat_job, range(N_REPEATS)):
            res[r] = out
    pred = {k: np.stack([res[r]["pred"][k] for r in range(N_REPEATS)]) for k in res[0]["pred"]}
    chosen = {k: [c for r in range(N_REPEATS) for c in res[r]["chosen"][k]] for k in res[0]["chosen"]}
    per_cfg = {k: np.stack([res[r]["per_cfg"][k] for r in range(N_REPEATS)]) for k in res[0]["per_cfg"]}
    np.savez(path, pred=np.array(pred, dtype=object), chosen=np.array(chosen, dtype=object),
             per_cfg=np.array(per_cfg, dtype=object))
    return pred, chosen, per_cfg


# ── metrik episode ─────────────────────────────────────────────────────
def _runs(mask, t):
    """Indeks awal-akhir (inklusif) deret True yang bersambung (jeda ≤ GAP_SEC)."""
    brk = np.r_[True, (~mask[:-1]) | (np.diff(t) > GAP_SEC)]
    starts = np.flatnonzero(mask & brk)
    out = []
    for s in starts:
        e = s
        while e + 1 < len(mask) and mask[e + 1] and t[e + 1] - t[e] <= GAP_SEC:
            e += 1
        out.append((s, e))
    return out


def episode_counts(d, pred):
    """Per pasien: (alarm AFL palsu, episode AFL sebenarnya, episode AFL terdeteksi, jam rekaman)."""
    P = len(d.patients)
    fa, ep, det, hours = np.zeros(P), np.zeros(P), np.zeros(P), np.zeros(P)
    for s, e in d.bounds:
        p = np.searchsorted(d.patients, d.records[s])
        tt, yy, pr = d.t[s:e], d.y[s:e], pred[s:e]
        hours[p] = (tt[-1] - tt[0] + 10) / 3600
        fa[p] = sum(1 for a, b in _runs(pr == 2, tt) if not (yy[a:b + 1] == 2).any())
        true_eps = _runs(yy == 2, tt)
        ep[p] = len(true_eps)
        det[p] = sum(1 for a, b in true_eps if (pr[a:b + 1] == 2).any())
    return fa, ep, det, hours


# ── evaluasi ───────────────────────────────────────────────────────────
def evaluate(d, pred):
    W = cv.bootstrap_weights(d.strata, N_BOOT, seed=SEED)
    cms = {k: np.stack([cv.patient_cms(d.y, p, d.records, d.patients) for p in v]) for k, v in pred.items()}
    boot = {k: cv.boot_metrics(c, W) for k, c in cms.items()}
    point = {k: cv.point_metrics(c) for k, c in cms.items()}
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    rows = {k: {m: [float(point[k][m].mean()), *ci(boot[k][m])] for m in cv.METRICS} for k in cms}

    # episode: per pasien per ulangan → bootstrap pasien yang sama
    epi, epi_boot = {}, {}
    for k, v in pred.items():
        counts = np.array([episode_counts(d, p) for p in v])            # (R, 4, P)
        fa, ep, det, hours = counts[:, 0].mean(0), counts[0, 1], counts[:, 2].mean(0), counts[0, 3]
        epi_boot[k] = {"alarm/jam": (W @ fa) / (W @ hours), "deteksi episode": (W @ det) / np.maximum(W @ ep, 1e-9)}
        epi[k] = {"alarm/jam": [float(fa.sum() / hours.sum()), *ci(epi_boot[k]["alarm/jam"])],
                  "deteksi episode": [float(det.sum() / ep.sum()), *ci(epi_boot[k]["deteksi episode"])],
                  "alarm palsu (total/ulangan)": float(fa.sum()), "episode AFL": int(ep.sum()),
                  "alarm palsu pasien tanpa AFL": float(fa[d.strata != 2].sum()), "jam": float(hours.sum())}

    pairs = {}
    for k in pred:
        if k == REF:
            continue
        row = {}
        for m in ("Macro-F1", "AFL F1", "AFL P", "AFL R", "Normal R"):
            dd = boot[k][m] - boot[REF][m]
            row[m] = [float(point[k][m].mean() - point[REF][m].mean()), *ci(dd), float(np.mean(dd > 0))]
        for m in ("alarm/jam", "deteksi episode"):
            dd = epi_boot[k][m] - epi_boot[REF][m]
            row[m] = [epi[k][m][0] - epi[REF][m][0], *ci(dd), float(np.mean(dd > 0))]
        pairs[k] = row

    afl_pat = d.patients[d.strata == 2]
    pi = np.searchsorted(d.patients, afl_pat)
    afl_recall = {k: (cms[k][:, pi, 2, 2] / cms[k][:, pi, 2, :].sum(-1)).mean(0).tolist() for k in cms}
    return SimpleNamespace(rows=rows, epi=epi, pairs=pairs, afl_patients=afl_pat.tolist(), afl_recall=afl_recall)


def per_cfg_table(d, per_cfg):
    """Metrik titik tiap konfigurasi smoothing MT jika DIPAKSA (bias tetap cross-fit). Diagnostik, bukan seleksi jujur."""
    out = {}
    for name, v in per_cfg.items():
        pm = cv.point_metrics(np.stack([cv.patient_cms(d.y, p, d.records, d.patients) for p in v]))
        counts = np.array([episode_counts(d, p) for p in v])
        out[name] = {"Macro-F1": float(pm["Macro-F1"].mean()), "AFL F1": float(pm["AFL F1"].mean()),
                     "Normal R": float(pm["Normal R"].mean()),
                     "alarm/jam": float(counts[:, 0].sum(1).mean() / counts[0, 3].sum())}
    return out


def assemble(d, pred, chosen, per_cfg, ev):
    from collections import Counter
    return {
        "labels": {k: v[0] for k, v in VARIANTS.items()}, "primary": PRIMARY, "ref": REF,
        "rows": ev.rows, "episodes": ev.epi, "pairs": ev.pairs,
        "chosen": {k: dict(Counter(v).most_common()) for k, v in chosen.items()},
        "per_cfg": per_cfg_table(d, per_cfg),
        "afl_patients": ev.afl_patients, "afl_recall": ev.afl_recall,
        "n_windows": int(len(d.y)), "n_patients": int(len(d.patients)),
        "offline": [cfg_name(c) for c in OFFLINE], "realtime": [cfg_name(c) for c in REALTIME],
    }


def run_all():
    d = load()
    pred, chosen, per_cfg = run_predictions()
    return assemble(d, pred, chosen, per_cfg, evaluate(d, pred))
