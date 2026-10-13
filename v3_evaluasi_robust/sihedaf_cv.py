"""
sihedaf_cv.py — utilitas CV & evaluasi robust SIHEDAF.

Training, prediksi, dan kalibrasi bias disalin dari sel 16
`revisi_sihedaf_v2.ipynb` (hiperparameter & loss sama). Perbedaan:
- Batch diambil langsung dari tensor yang sudah di GPU (tanpa DataLoader),
  supaya CV berulang (banyak seed × fold) muat dalam waktu wajar.
- Fold dibentuk berulang (repeated StratifiedGroupKFold per pasien).
- Metrik dihitung dari confusion matrix per pasien, sehingga bootstrap
  level pasien (dan perbandingan berpasangan antar model) murah.
"""
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

# Arsitektur model dipakai dari v2 (satu definisi, tidak disalin).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2_revisi"))
from sihedaf_models import build_model  # noqa: E402

K = 3
CLASSES = ["Normal", "AFIB", "AFL"]


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Fold pasien berulang ───────────────────────────────────────────────
def patient_strata(y, records, patients):
    """Strata per pasien: 2 = punya AFL, 1 = dominan AFIB (>50% window), 0 = lainnya."""
    strata = np.zeros(len(patients), dtype=np.int64)
    for i, p in enumerate(patients):
        yp = y[records == p]
        strata[i] = 2 if (yp == 2).any() else int(np.mean(yp == 1) > 0.5)
    return strata


def repeated_patient_folds(y, records, n_splits=5, n_repeats=5, seed=42):
    """
    Return (R, N) int8: fold uji tiap window untuk tiap ulangan.

    Tidak memakai StratifiedGroupKFold(shuffle=True): dengan 23 pasien,
    partisinya hampir deterministik (pasien AFL selalu jatuh di fold yang
    sama untuk seed berapa pun). Sebagai gantinya pasien diacak di dalam
    strata (AFL / dominan AFIB / lainnya) lalu dibagikan round-robin, jadi
    tiap fold kebagian pasien AFL dan tiap ulangan benar-benar berbeda.
    """
    patients = np.unique(records)
    strata = patient_strata(y, records, patients)
    pidx = np.searchsorted(patients, records)
    rng = np.random.default_rng(seed)
    folds = np.empty((n_repeats, len(y)), dtype=np.int8)
    for r in range(n_repeats):
        order = np.concatenate([rng.permutation(np.flatnonzero(strata == s)) for s in (2, 1, 0)])
        assign = np.empty(len(patients), dtype=np.int64)
        assign[order] = (np.arange(len(order)) + rng.integers(n_splits)) % n_splits
        folds[r] = assign[pidx]
    return folds


# ── Deep learning ──────────────────────────────────────────────────────
def class_weights(y):
    counts = np.bincount(y, minlength=K).astype(np.float64)
    return len(y) / (K * np.maximum(counts, 1))


def zscore_fit(X, M):
    v = X[M.astype(bool)]
    return v.mean(0).astype(np.float32), (v.std(0) + 1e-8).astype(np.float32)


def zscore_apply(X, M, mu, sd):
    return ((X - mu) / sd * M[..., None]).astype(np.float32)


def train_dl(model_name, model_kwargs, X, M, y, *, epochs, batch_size, lr, weight_decay, seed, device):
    """Weighted CE + AdamW + cosine, sama dengan train_dl di notebook revisi."""
    set_seed(seed)
    mu, sd = zscore_fit(X, M)
    Xt = torch.from_numpy(zscore_apply(X, M, mu, sd)).to(device)
    Mt = torch.from_numpy(M).to(device)
    yt = torch.from_numpy(y).to(device)
    n = len(y)
    n_batches = n // batch_size if n > batch_size else 1   # = drop_last, hindari batch 1 (BatchNorm)
    bs = batch_size if n > batch_size else n

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(y), dtype=torch.float32, device=device))
    net = build_model(model_name, **model_kwargs).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator(device=device).manual_seed(seed)

    t0 = time.perf_counter()
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device, generator=gen)
        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs]
            opt.zero_grad(set_to_none=True)
            criterion(net(Xt[idx], Mt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return net.eval(), mu, sd, time.perf_counter() - t0


@torch.no_grad()
def predict_dl(net, X, M, mu, sd, device, bs=8192):
    Xn = zscore_apply(X, M, mu, sd)
    out = []
    for s in range(0, len(Xn), bs):
        logits = net(torch.from_numpy(Xn[s:s + bs]).to(device), torch.from_numpy(M[s:s + bs]).to(device))
        out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, K), np.float32)


# ── LightGBM ───────────────────────────────────────────────────────────
def make_lgbm(seed):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              class_weight="balanced", random_state=seed, verbose=-1, n_jobs=-1)


# ── CV satu ulangan ────────────────────────────────────────────────────
def cv_oof(fit_predict, data, folds_r, seed):
    """fit_predict(train_idx, test_idx, seed) → prob test. Return (oof, waktu per fold)."""
    oof = np.full((len(data["y"]), K), np.nan, dtype=np.float32)
    times = []
    for k in np.unique(folds_r):
        tr, te = np.flatnonzero(folds_r != k), np.flatnonzero(folds_r == k)
        t0 = time.perf_counter()
        oof[te] = fit_predict(tr, te, seed + int(k))
        times.append(time.perf_counter() - t0)
    assert not np.isnan(oof).any()
    return oof, times


def run_dl_job(job: dict):
    """
    Satu (model, ulangan): CV semua fold, simpan OOF ke job["out"] (.npz).
    Fungsi level-modul supaya bisa dijalankan paralel lewat ProcessPoolExecutor
    (model kecil → GPU kurang terpakai; bottleneck-nya overhead Python per batch).
    Sudah ada hasilnya → dilewati, jadi run yang terputus bisa dilanjutkan.
    """
    out = Path(job["out"])
    if out.exists():
        return str(out), None
    torch.set_num_threads(job.get("threads", 2))
    z = np.load(job["data_path"])
    X, M, y = z["X"], z["M"], z["y"]
    folds_r = np.load(job["folds_path"])["folds"][job["repeat"]]
    device = torch.device(job["device"])
    hp = job["hparams"]

    def fit_predict(tr, te, seed):
        net, mu, sd, _ = train_dl(job["model"], job["model_kwargs"], X[tr], M[tr], y[tr],
                                  seed=seed, device=device, **hp)
        return predict_dl(net, X[te], M[te], mu, sd, device)

    oof, times = cv_oof(fit_predict, {"y": y}, folds_r, seed=job["seed"])
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, oof=oof, times=np.asarray(times))
    tmp.replace(out)
    return str(out), float(np.mean(times))


# ── Kalibrasi bias kelas (salinan notebook revisi) ─────────────────────
def _macro_f1_fast(y, pred):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1[cm.sum(1) > 0].mean()


# Notebook revisi memakai linspace(-3, 3, 25); bias AFL model DL (weighted CE →
# AFL sangat over-prediksi) mentok di -3, jadi grid diperlebar ke bawah.
BIAS_GRID = np.linspace(-7, 3, 41)


def fit_class_bias(y, proba, grid=BIAS_GRID):
    logp = np.log(np.clip(proba, 1e-9, 1))
    best, best_b = -1, np.zeros(K)
    for b1 in grid:
        for b2 in grid:
            b = np.array([0.0, b1, b2])
            s = _macro_f1_fast(y, (logp + b).argmax(1))
            if s > best:
                best, best_b = s, b
    return best_b


def apply_bias(proba, b):
    z = np.log(np.clip(proba, 1e-9, 1)) + b
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def crossfit_bias(y, proba, folds_r):
    """Bias fold k dipelajari dari OOF fold lain → evaluasi tetap jujur."""
    adj = np.empty_like(proba)
    biases = []
    for k in np.unique(folds_r):
        b = fit_class_bias(y[folds_r != k], proba[folds_r != k])
        adj[folds_r == k] = apply_bias(proba[folds_r == k], b)
        biases.append(b)
    return adj, np.array(biases)


# ── Metrik dari confusion matrix ───────────────────────────────────────
def patient_cms(y, pred, records, patients):
    """Confusion matrix per pasien: (P, K, K), baris = aktual."""
    pidx = np.searchsorted(patients, records)
    flat = (pidx * K + y) * K + pred
    return np.bincount(flat, minlength=len(patients) * K * K).reshape(len(patients), K, K).astype(np.float64)


def metrics_from_cm(cm):
    """cm: (..., K, K) → dict metrik, masing-masing berbentuk (...)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(-1)
    predicted = cm.sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(predicted > 0, tp / predicted, 0.0)
        rec = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return {
        "Macro-F1": f1.mean(-1),
        "Normal R": rec[..., 0],
        "AFIB P": prec[..., 1], "AFIB R": rec[..., 1], "AFIB F1": f1[..., 1],
        "AFL P": prec[..., 2], "AFL R": rec[..., 2], "AFL F1": f1[..., 2],
    }


METRICS = ["Macro-F1", "Normal R", "AFIB P", "AFIB R", "AFIB F1", "AFL P", "AFL R", "AFL F1"]


# ── Bootstrap level pasien ─────────────────────────────────────────────
def bootstrap_weights(strata, n_boot=2000, seed=0):
    """
    Bobot resampling pasien (n_boot, P), stratified per strata (mis. punya
    AFL / tidak) supaya tiap sampel bootstrap tetap berisi pasien AFL.
    """
    rng = np.random.default_rng(seed)
    P = len(strata)
    W = np.zeros((n_boot, P))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        draws = rng.choice(idx, size=(n_boot, len(idx)), replace=True)
        for b in range(n_boot):
            W[b] += np.bincount(draws[b], minlength=P)
    return W


def boot_metrics(cms, W):
    """
    cms: (R, P, K, K) per ulangan & pasien; W: (B, P).
    Return dict metrik → (B,) = rata-rata antar ulangan dari metrik pooled
    pada pasien hasil resampling.
    """
    pooled = np.einsum("bp,rpij->brij", W, cms)          # (B, R, K, K)
    return {k: v.mean(1) for k, v in metrics_from_cm(pooled).items()}


def point_metrics(cms):
    """Metrik pooled per ulangan: dict → (R,)."""
    return metrics_from_cm(cms.sum(1))
