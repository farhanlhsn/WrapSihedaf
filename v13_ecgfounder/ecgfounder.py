"""
ecgfounder.py — logika v13: model fondasi EKG ECGFounder 1-lead (PKUDigitalHealth, lisensi MIT) pada window 10 s SIHEDAF.

Pertanyaan: apakah pengetahuan pralatih dari 10,8 juta EKG (1,8 juta pasien), yang head-nya sudah memuat sinus takikardia dan
AFL, menembus batas model RR, dan berapa yang hilang karena perangkat hanya 50 Hz?

Alur per window (sama untuk semua data): sinyal → (jalur "50hz": resample ke 50 Hz dengan hrv_core seperti perangkat) →
resample ke 500 Hz → filter ECGFounder (notch 50 Hz, Butterworth 0,67–40 Hz, median baseline 0,4 s) → z-score → Net1D.
Jalur kontrol "250hz": sinyal asli AFDB 250 Hz langsung ke 500 Hz.

Tiga cara pakai, semuanya dengan fold resmi 5×5 dan bias cross-fit (protokol v3):
  zero-shot  — logit head [maks(NSR, SR, ST), AF, AFL] → 3 kelas;
  probe      — fitur 1.024 dimensi beku → regresi logistik per fold (data latih fold itu saja);
  fusi       — rata-rata log-probabilitas probe dan v3.1 (OOF MicroTransformer).
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt, resample_poly

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "v3_evaluasi_robust"), str(REPO / "v2_revisi"), str(REPO / "v12_shdb_atrium"),
                str(REPO / "v11_model_tim")]
import sihedaf_cv as cv  # noqa: E402
import hrv_core  # noqa: E402
import atrium as A12  # noqa: E402

KODE = REPO / "data/ecgfounder/kode"
BOBOT = REPO / "data/ecgfounder/1_lead_ECGFounder.pth"
SHA_BOBOT = "f863a38897fb49a27fec7e44008ea3c7bdbd29c77fa4a02ecbb8c56df4f37603"   # API Hugging Face (LFS)
CACHE = REPO / "cache/v13"
FS_MODEL, N_MODEL = 500, 5000
N_REPEATS, SEED = 5, 2026
LABEL = {"NSR": "NORMAL SINUS RHYTHM", "SR": "SINUS RHYTHM", "AF": "ATRIAL FIBRILLATION", "ST": "SINUS TACHYCARDIA",
         "AFL": "ATRIAL FLUTTER", "SVT": "SUPRAVENTRICULAR TACHYCARDIA"}


def indeks_label():
    tugas = [b.strip() for b in (KODE / "tasks.txt").read_text().splitlines() if b.strip()]
    assert len(tugas) == 150, len(tugas)
    return {k: tugas.index(v) for k, v in LABEL.items()}


# ── praproses (sama dengan util.filter_bandpass + z-score di repo ECGFounder) ──
def filter_ecgfounder(x, fs=FS_MODEL):
    """util.filter_bandpass ECGFounder untuk array (N, L): notch 50 Hz Q 30, Butterworth orde 4 0,67–40 Hz (filtfilt),
    lalu dikurangi median bergerak 0,4 s (baseline)."""
    b, a = iirnotch(50, 30, fs)
    y = filtfilt(b, a, x, axis=-1)
    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    y = filtfilt(b, a, y, axis=-1)
    k = int(0.4 * fs) + 1
    k += (k % 2 == 0)
    return y - np.stack([medfilt(r, kernel_size=k) for r in y])


def praproses(win, fs_in):
    """win (N, L) pada fs_in → (N, 1, 5000) float32 siap Net1D."""
    w = np.asarray(win, np.float64)
    up = resample_poly(w, FS_MODEL, int(fs_in), axis=-1)
    up = up[:, :N_MODEL] if up.shape[1] >= N_MODEL else np.pad(up, ((0, 0), (0, N_MODEL - up.shape[1])))
    y = filter_ecgfounder(up)
    y = (y - y.mean(1, keepdims=True)) / (y.std(1, keepdims=True) + 1e-8)
    return y[:, None, :].astype(np.float32)


# ── model ──────────────────────────────────────────────────────────────
def muat_model(device="cuda"):
    """Net1D 1-lead dengan head 150 label asli. Checkpoint dicek SHA256; dimuat weights_only (hanya skalar numpy diizinkan)."""
    import torch
    h = hashlib.sha256()
    with BOBOT.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    if h.hexdigest() != SHA_BOBOT:
        raise ValueError("checksum checkpoint ECGFounder tidak cocok")
    spec = importlib.util.spec_from_file_location("net1d_ecgfounder", KODE / "net1d.py")
    net1d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(net1d)
    model = net1d.Net1D(in_channels=1, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                        m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16, verbose=False,
                        use_bn=False, use_do=False, n_classes=150, return_features=True)
    izin = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(izin):
        z = torch.load(BOBOT, map_location="cpu", weights_only=True)
    model.load_state_dict(z["state_dict"], strict=True)
    return model.to(device).eval()


def inferensi(model, x, device="cuda", batch=128):
    """x (N, 1, 5000) → (logit (N, 150) float32, fitur (N, 1024) float16)."""
    import torch
    lo, fi = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        for i in range(0, len(x), batch):
            o, f = model(torch.from_numpy(x[i:i + batch]).to(device))
            lo.append(o.float().cpu().numpy())
            fi.append(f.float().cpu().numpy().astype(np.float16))
    return np.concatenate(lo), np.concatenate(fi)


# ── AFDB ───────────────────────────────────────────────────────────────
def _afdb():
    z = np.load(REPO / "cache/dataset_v2_detector.npz")
    return z["records"].astype(str), z["t_start"].astype(np.float64), z["y"]


def jendela_afdb(rec, t0s, jalur):
    """Window 10 s kanal 0 AFDB (sama dengan v6) untuk satu rekaman. jalur '50hz' (seperti perangkat) atau '250hz'."""
    x, fs = A12.baca_dat(REPO / "data/afdb" / rec, 0)
    x = np.nan_to_num(x)
    if jalur == "50hz":
        x, fs = hrv_core.resample_signal(x, fs, hrv_core.FS_DEVICE), hrv_core.FS_DEVICE
    n = int(round(hrv_core.WINDOW_SEC * fs))
    out = np.zeros((len(t0s), n))
    for j, t in enumerate(t0s):
        s = int(round(t * fs))
        w = x[s:s + n]
        out[j, :w.size] = w
    return out, fs


def jalankan_afdb(jalur, device="cuda", log=print):
    """Logit + fitur ECGFounder untuk semua window AFDB, per rekaman (bisa dilanjutkan) → cache/v13/afdb_<jalur>/."""
    recs, t0s, _ = _afdb()
    out_dir = CACHE / f"afdb_{jalur}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    for rec in np.unique(recs):
        p = out_dir / f"{rec}.npz"
        if p.exists():
            continue
        model = model or muat_model(device)
        idx = np.flatnonzero(recs == rec)
        w, fs = jendela_afdb(rec, t0s[idx], jalur)
        lo, fi = inferensi(model, praproses(w, fs), device)
        np.savez(p.with_suffix(".tmp.npz"), idx=idx, logit=lo, fitur=fi)
        p.with_suffix(".tmp.npz").replace(p)
        log(f"  {jalur} {rec}: {len(idx)} window")
    return muat_afdb(jalur)


def muat_afdb(jalur):
    recs, _, _ = _afdb()
    logit = np.zeros((len(recs), 150), np.float32)
    fitur = np.zeros((len(recs), 1024), np.float16)
    isi = np.zeros(len(recs), bool)
    for p in sorted((CACHE / f"afdb_{jalur}").glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        z = np.load(p)
        logit[z["idx"]], fitur[z["idx"]], isi[z["idx"]] = z["logit"], z["fitur"], True
    if not isi.all():
        raise RuntimeError(f"cache AFDB {jalur} belum lengkap: {int((~isi).sum())} window kosong")
    return logit, fitur


# ── tiga cara pakai → OOF (R, N, 3) ────────────────────────────────────
def skor_zero_shot(logit):
    """[maks(NSR, SR, ST), AF, AFL] → log-softmax → probabilitas 3 kelas."""
    i = indeks_label()
    z = np.stack([logit[:, [i["NSR"], i["SR"], i["ST"]]].max(1), logit[:, i["AF"]], logit[:, i["AFL"]]], 1)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def oof_probe(fitur, y, folds, C=1.0, pangkat=0.5, log=print):
    """Regresi logistik multinomial pada fitur beku, per (ulangan, fold): standardisasi & bobot kelas lunak dari data latih
    fold itu (AGENTS.md: class_weights ** 0,5). Return OOF (R, N, 3)."""
    from sklearn.linear_model import LogisticRegression
    X = fitur.astype(np.float32)
    out = np.zeros((len(folds), len(y), 3), np.float32)
    for r, f in enumerate(folds):
        for k in range(5):
            tr, te = f != k, f == k
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            w = (cv.class_weights(y[tr]) ** pangkat)[y[tr]]
            m = LogisticRegression(C=C, max_iter=1000, tol=1e-4)
            m.fit((X[tr] - mu) / sd, y[tr], sample_weight=w)
            out[r, te] = m.predict_proba((X[te] - mu) / sd)
        log(f"  probe ulangan {r} selesai")
    return out


def fusi(p_a, p_b):
    """Rata-rata log-probabilitas dua sumber → probabilitas."""
    z = 0.5 * (np.log(np.clip(p_a, 1e-7, 1)) + np.log(np.clip(p_b, 1e-7, 1)))
    z -= z.max(-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(-1, keepdims=True)).astype(np.float32)


def ptbxl():
    """
    PTB-XL 1.0.3 (100 Hz, lead I) → satu window 10 s per rekaman. Kelas ritme: Normal = SR/SBRAD/SARRH/STACH (sinus,
    termasuk sinus takikardia), AFIB, AFL (AFLT). Rekaman dengan AFIB dan AFLT sekaligus, pacu, atau SVT dibuang.
    Return dict: ecg (N, 1000) mV, y, st (sinus takikardia), pasien, ecg_id.
    """
    import ast
    import pandas as pd
    akar = REPO / "data/ptbxl"
    d = pd.read_csv(akar / "ptbxl_database.csv")
    scp = d.scp_codes.apply(ast.literal_eval)
    sinus = scp.apply(lambda s: any(k in s for k in ("SR", "SBRAD", "SARRH", "STACH")))
    af, afl = scp.apply(lambda s: "AFIB" in s), scp.apply(lambda s: "AFLT" in s)
    buang = scp.apply(lambda s: any(k in s for k in ("PACE", "SVTAC", "PSVT"))) | (af & afl)
    y = np.where(afl, 2, np.where(af, 1, np.where(sinus, 0, -1)))
    pilih = (~buang) & (y >= 0)
    d, y, scp = d[pilih], y[pilih], scp[pilih]
    ecg = np.stack([A12.baca_dat(akar / f, 0)[0] for f in d.filename_lr])
    return dict(ecg=np.nan_to_num(ecg), fs=100.0, y=y.astype(np.int64), st=scp.apply(lambda s: "STACH" in s).to_numpy(),
                pasien=d.patient_id.to_numpy().astype(np.int64), ecg_id=d.ecg_id.to_numpy(),
                manusia=d.validated_by_human.to_numpy())


def ke_50hz(win, fs):
    """Window (N, L) pada fs → 50 Hz dengan hrv_core.resample_signal (seperti perangkat), per window."""
    return np.stack([hrv_core.resample_signal(w, fs, hrv_core.FS_DEVICE)[:500] for w in win]), float(hrv_core.FS_DEVICE)


def rr_paket_v31(win, fs):
    """Probabilitas paket v3.1 dari RR window (detektor produksi pada sinyal yang di-resample ke 50 Hz). NaN bila tak terbaca."""
    from predictor import ECGPredictor
    P = ECGPredictor(str(REPO / "v3.1_kalibrasi_produksi/model_export"), name="sihedaf_v3_1", device="cpu")
    out = np.full((len(win), 3), np.nan, np.float32)
    seqs, ok = [], []
    for i, w in enumerate(win):
        ws, _ = hrv_core.signal_to_windows(np.asarray(w, np.float64), fs, hrv_core.WINDOW_SEC)
        if ws and ws[0]["ok"]:
            seqs.append(hrv_core.build_sequence(ws[0]["ibi_ms"]))
            ok.append(i)
    if ok:
        out[ok] = P.predict_proba_sequences(np.stack([s[0] for s in seqs]), np.stack([s[1] for s in seqs]))
    return out


def jendela_latihan():
    """Window EKG 10 s data latihan fisik v3.3 (mode perangkat, window yang lolos cek kualitas), sejajar dengan
    cache/v3_3/jendela.npz. Return (ecg (N, L) pada fs asli tiap sumber dikonversi ke 50 Hz, indeks baris jendela)."""
    sys.path.insert(0, str(REPO / "v3.3_takikardia_asli"))
    import takikardia_asli as K
    J = K.muat_jendela()
    loader = {"EPFL": K.rekaman_epfl, "Wrist": K.rekaman_wrist, "EPHNOGRAM": K.rekaman_ephnogram}
    out, baris = [], []
    for s in np.unique(J["sumber"]):
        for rek in loader[s]():
            m = np.flatnonzero((J["sumber"] == s) & (J["rekaman"] == rek["rekaman"]) & (J["mode"] == "perangkat") & J["ok"])
            x = np.nan_to_num(np.asarray(rek["ecg"], np.float64))
            x50 = hrv_core.resample_signal(x, rek["fs"], hrv_core.FS_DEVICE)
            for i in m:
                a = int(round(J["start"][i] * hrv_core.FS_DEVICE))
                w = x50[a:a + 500]
                out.append(np.pad(w, (0, 500 - w.size)))
                baris.append(i)
    return np.stack(out), np.array(baris)


def model_probe_penuh(fitur, y, C=1.0, pangkat=0.5):
    """Probe dilatih di SELURUH AFDB (untuk data eksternal: latihan fisik, PTB-XL, Icentia). Return (model, mu, sd)."""
    from sklearn.linear_model import LogisticRegression
    X = fitur.astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=C, max_iter=1000, tol=1e-4)
    m.fit((X - mu) / sd, y, sample_weight=(cv.class_weights(y) ** pangkat)[y])
    return m, mu, sd


# ── data eksternal ─────────────────────────────────────────────────────
def bias_median(oof, y, folds):
    """Median bias cross-fit (25 fold) sebuah OOF — bias untuk model yang dilatih penuh dan dipakai di data eksternal."""
    return np.median(np.stack([cv.crossfit_bias(y, oof[r], folds[r])[1] for r in range(len(folds))]).reshape(-1, 3), 0)


def tanpa_bias_paket(p_paket):
    """Probabilitas paket v3.1 (sudah termasuk bias paket) → probabilitas mentah (bias dikurangkan lagi di ruang log)."""
    import json
    b = np.array(json.loads((REPO / "v3.1_kalibrasi_produksi/model_export/sihedaf_v3_1.json").read_text())["class_bias"])
    z = np.log(np.clip(p_paket, 1e-7, 1)) - b
    z -= z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def banding_pasien(y, pred_a, pred_b, pasien, n_boot=2000, seed=SEED):
    """Macro-F1 dan recall AFL (a − b) dengan bootstrap pasien berpasangan; P = P(Δ > 0). Juga titik masing-masing."""
    u, inv = np.unique(pasien, return_inverse=True)

    def cm(pred):
        return np.bincount((inv * 3 + y) * 3 + pred, minlength=len(u) * 9).reshape(len(u), 3, 3).astype(float)

    def met(c):
        tp = np.diagonal(c, axis1=-2, axis2=-1)
        rec = tp / np.maximum(c.sum(-1), 1)
        pre = tp / np.maximum(c.sum(-2), 1)
        f1 = 2 * pre * rec / np.maximum(pre + rec, 1e-12)
        return f1.mean(-1), rec[..., 2]

    ca, cb = cm(pred_a), cm(pred_b)
    idx = np.random.default_rng(seed).integers(0, len(u), (n_boot, len(u)))
    fa, ra = met(ca[idx].sum(1))
    fb, rb = met(cb[idx].sum(1))
    ta, tra = met(ca.sum(0))
    tb, trb = met(cb.sum(0))
    return {"Macro-F1": [float(ta - tb), *np.percentile(fa - fb, [2.5, 97.5]).tolist(), float(np.mean(fa - fb > 0))],
            "AFL R": [float(tra - trb), *np.percentile(ra - rb, [2.5, 97.5]).tolist(), float(np.mean(ra - rb > 0))],
            "titik_a": [float(ta), float(tra)], "titik_b": [float(tb), float(trb)]}


def jalankan_eksternal(nama, win50, device="cuda"):
    """Logit + fitur ECGFounder untuk window 50 Hz eksternal → cache/v13/<nama>.npz (dilewati bila sudah ada)."""
    p = CACHE / f"{nama}.npz"
    if p.exists():
        z = np.load(p)
        return z["logit"], z["fitur"]
    model = muat_model(device)
    lo, fi = [], []
    for i in range(0, len(win50), 4096):
        a, b = inferensi(model, praproses(win50[i:i + 4096], hrv_core.FS_DEVICE), device)
        lo.append(a); fi.append(b)
    lo, fi = np.concatenate(lo), np.concatenate(fi)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, logit=lo, fitur=fi)
    return lo, fi
