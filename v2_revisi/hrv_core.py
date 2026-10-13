"""
hrv_core.py — SATU sumber kebenaran untuk praproses SIHEDAF.

Dipakai oleh ekstraksi dataset, training, DAN predictor backend, sehingga
data latih dan data produksi diproses dengan kode yang persis sama
(menutup masalah training-serving skew pada versi sebelumnya).

Konvensi:
- IBI selalu dalam MILIDETIK.
- Sekuens model berbentuk (MAX_BEATS, 2) = [IBI, delta_IBI], delta_IBI[0] = 0.
- Mask: 1 = detak valid, 0 = padding.
"""
from fractions import Fraction

import numpy as np
from scipy import signal as sps

# ── Konstanta praproses (ikut diekspor ke metadata model) ──────────────
FS_DEVICE = 50          # sampling rate perangkat produksi (Hz)
WINDOW_SEC = 10.0       # panjang window analisis (detik)
MAX_BEATS = 30          # panjang sekuens model
IBI_MIN_MS = 300.0      # batas fisiologis bawah (200 bpm)
IBI_MAX_MS = 2000.0     # batas fisiologis atas (30 bpm)
MAX_OUTLIERS = 1        # maksimum IBI di luar rentang per window
MIN_IBI = 4             # minimum IBI per window agar window dianggap valid

LABELS_MAP = {0: "Normal", 1: "AFIB (Atrial Fibrillation)", 2: "AFL (Atrial Flutter)"}

FEATURE_NAMES = [
    "mean_ibi", "median_ibi", "min_ibi", "max_ibi", "sdnn", "rmssd",
    "pnn50", "cv", "sd1", "sd2", "sd1_sd2", "tpr", "sampen", "cosen",
    "hr_mean", "beat_count",
]


def preprocessing_config() -> dict:
    """Konstanta praproses, untuk disimpan di metadata model."""
    return {
        "fs_device": FS_DEVICE, "window_sec": WINDOW_SEC, "max_beats": MAX_BEATS,
        "ibi_min_ms": IBI_MIN_MS, "ibi_max_ms": IBI_MAX_MS,
        "max_outliers": MAX_OUTLIERS, "min_ibi": MIN_IBI,
        "ibi_unit": "ms", "sequence_features": ["ibi_ms", "delta_ibi_ms"],
    }


# ── Sinyal → R-peak ────────────────────────────────────────────────────
def resample_signal(x: np.ndarray, fs_in: float, fs_out: float = FS_DEVICE) -> np.ndarray:
    """Resample polifase (anti-aliasing bawaan)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    frac = Fraction(int(round(fs_out)), int(round(fs_in))).limit_denominator(1000)
    return sps.resample_poly(np.asarray(x, dtype=np.float64), frac.numerator, frac.denominator)


def detect_rpeaks(ecg: np.ndarray, fs: float, chunk_sec: float = 10.0) -> np.ndarray:
    """
    Detektor R-peak ala Pan-Tompkins yang tahan untuk fs rendah (50 Hz).

    1. Clip artefak ekstrem, bandpass 5–15 Hz.
    2. Turunan → kuadrat → moving-window integration (150 ms).
    3. Threshold adaptif per potongan `chunk_sec` detik.
    4. Jarak minimum 250 ms (sampai 240 bpm; versi lama dibatasi 150 bpm,
       padahal AFL 2:1 sering tepat di ~150 bpm).
    5. Refinement ke puncak |sinyal| + interpolasi parabolik sub-sampel,
       untuk mengurangi kuantisasi 20 ms pada 50 Hz.

    Return: posisi R-peak dalam satuan SAMPEL (float, boleh pecahan).
    """
    x = np.nan_to_num(np.asarray(ecg, dtype=np.float64).ravel())
    if x.size < int(2 * fs):
        return np.array([], dtype=np.float64)

    lo, hi = np.percentile(x, [0.5, 99.5])
    x = np.clip(x, lo, hi)

    nyq = fs / 2.0
    high = min(15.0, 0.8 * nyq)
    b, a = sps.butter(2, [5.0 / nyq, high / nyq], btype="band")
    y = sps.filtfilt(b, a, x)

    d = np.gradient(y)
    w = max(1, int(round(0.15 * fs)))
    mwi = np.convolve(d * d, np.ones(w) / w, mode="same")

    # Threshold adaptif per chunk
    thr = np.empty_like(mwi)
    n_chunk = max(1, int(round(chunk_sec * fs)))
    for s in range(0, mwi.size, n_chunk):
        seg = mwi[s:s + n_chunk]
        thr[s:s + n_chunk] = 0.3 * np.percentile(seg, 98)

    cand, _ = sps.find_peaks(mwi, distance=max(1, int(round(0.25 * fs))))
    cand = cand[mwi[cand] >= thr[cand]]
    if cand.size == 0:
        return np.array([], dtype=np.float64)

    # Refinement ke puncak |y| di sekitar kandidat (±80 ms)
    half = max(1, int(round(0.08 * fs)))
    ay = np.abs(y)
    peaks = []
    for c in cand:
        s, e = max(0, c - half), min(ay.size, c + half + 1)
        p = s + int(np.argmax(ay[s:e]))
        # interpolasi parabolik sub-sampel
        if 0 < p < ay.size - 1:
            y0, y1, y2 = ay[p - 1], ay[p], ay[p + 1]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            off = float(np.clip(off, -0.5, 0.5))
        else:
            off = 0.0
        peaks.append(p + off)

    peaks = np.unique(np.round(np.asarray(peaks), 3))
    # buang duplikat yang terlalu dekat setelah refinement
    keep = np.concatenate(([True], np.diff(peaks) >= 0.25 * fs))
    return peaks[keep]


# ── R-peak → IBI ───────────────────────────────────────────────────────
def peaks_to_ibi(peaks: np.ndarray, fs: float):
    """Return (t_end_detik, ibi_ms). t_end = waktu R-peak penutup tiap IBI."""
    peaks = np.asarray(peaks, dtype=np.float64)
    if peaks.size < 2:
        return np.array([]), np.array([])
    ibi_ms = np.diff(peaks) / fs * 1000.0
    t_end = peaks[1:] / fs
    return t_end, ibi_ms


def window_quality(ibi_ms: np.ndarray):
    """Return (ok, alasan). Aturan yang SAMA untuk training dan inferensi."""
    ibi_ms = np.asarray(ibi_ms, dtype=np.float64)
    if ibi_ms.size < MIN_IBI:
        return False, f"terlalu sedikit detak ({ibi_ms.size} IBI < {MIN_IBI})"
    n_out = int(np.sum((ibi_ms < IBI_MIN_MS) | (ibi_ms > IBI_MAX_MS)))
    if n_out > MAX_OUTLIERS:
        return False, f"{n_out} IBI di luar {IBI_MIN_MS:.0f}–{IBI_MAX_MS:.0f} ms"
    return True, "ok"


def build_sequence(ibi_ms: np.ndarray):
    """IBI satu window → (X[MAX_BEATS,2], mask[MAX_BEATS]). delta_IBI[0] = 0."""
    ibi = np.asarray(ibi_ms, dtype=np.float32)[:MAX_BEATS]
    n = ibi.size
    X = np.zeros((MAX_BEATS, 2), dtype=np.float32)
    mask = np.zeros(MAX_BEATS, dtype=np.float32)
    X[:n, 0] = ibi
    if n > 1:
        X[1:n, 1] = np.diff(ibi)
    mask[:n] = 1.0
    return X, mask


# ── Fitur HRV untuk cabang tabular ─────────────────────────────────────
def _sample_entropy(x: np.ndarray, m: int, r: float) -> float:
    """SampEn sederhana (O(n²), aman untuk n ≤ ~40)."""
    n = x.size
    if n <= m + 1 or r <= 0:
        return np.nan

    def _count(mm):
        tpl = np.array([x[i:i + mm] for i in range(n - m)])
        dist = np.max(np.abs(tpl[:, None, :] - tpl[None, :, :]), axis=2)
        iu = np.triu_indices(len(tpl), k=1)
        return np.sum(dist[iu] <= r)

    B, A = _count(m), _count(m + 1)
    if A == 0 or B == 0:
        # batas atas yang terdefinisi (hindari inf)
        return float(np.log((n - m) * (n - m - 1) / 2.0))
    return float(-np.log(A / B))


def hrv_features(ibi_ms: np.ndarray) -> np.ndarray:
    """Vektor fitur HRV sesuai urutan FEATURE_NAMES."""
    ibi = np.asarray(ibi_ms, dtype=np.float64)
    d = np.diff(ibi)
    mean = ibi.mean()
    sdnn = ibi.std(ddof=1) if ibi.size > 1 else 0.0
    rmssd = np.sqrt(np.mean(d ** 2)) if d.size else 0.0
    pnn50 = np.mean(np.abs(d) > 50.0) if d.size else 0.0
    sd1 = np.sqrt(0.5) * d.std(ddof=1) if d.size > 1 else 0.0
    sd2_sq = 2 * sdnn ** 2 - sd1 ** 2
    sd2 = np.sqrt(sd2_sq) if sd2_sq > 0 else 0.0
    sd1_sd2 = sd1 / sd2 if sd2 > 0 else 0.0
    if ibi.size >= 3:
        mid = ibi[1:-1]
        tp = np.sum(((mid > ibi[:-2]) & (mid > ibi[2:])) | ((mid < ibi[:-2]) & (mid < ibi[2:])))
        tpr = tp / (ibi.size - 2)
    else:
        tpr = 0.0
    sampen = _sample_entropy(ibi, m=1, r=0.2 * sdnn) if sdnn > 0 else 0.0
    # COSEn (Lake & Moorman): SampEn dengan r tetap + ln(2r) − ln(mean RR), satuan detik
    r_s = 0.03
    se_abs = _sample_entropy(ibi / 1000.0, m=1, r=r_s)
    cosen = se_abs + np.log(2 * r_s) - np.log(mean / 1000.0) if np.isfinite(se_abs) else 0.0
    feats = [
        mean, np.median(ibi), ibi.min(), ibi.max(), sdnn, rmssd, pnn50,
        sdnn / mean if mean > 0 else 0.0, sd1, sd2, sd1_sd2, tpr,
        np.nan_to_num(sampen), np.nan_to_num(cosen), 60000.0 / mean, float(ibi.size),
    ]
    return np.asarray(feats, dtype=np.float32)


# ── Sinyal penuh → window siap model (dipakai predictor) ───────────────
def signal_to_windows(ecg: np.ndarray, fs: float, stride_sec: float = 5.0):
    """
    EKG mentah (fs berapa pun) → list window.
    Sinyal diproses di FS_DEVICE agar identik dengan kondisi produksi.
    """
    x = resample_signal(ecg, fs, FS_DEVICE)
    peaks = detect_rpeaks(x, FS_DEVICE)
    t_end, ibi = peaks_to_ibi(peaks, FS_DEVICE)
    dur = x.size / FS_DEVICE
    out = []
    starts = np.arange(0.0, max(dur - WINDOW_SEC, 0.0) + 1e-9, stride_sec)
    for s in starts:
        i0, i1 = np.searchsorted(t_end, [s, s + WINDOW_SEC], side="left")
        w_ibi = ibi[i0:i1]
        ok, reason = window_quality(w_ibi)
        out.append({"start_sec": float(s), "ibi_ms": w_ibi, "ok": ok, "reason": reason})
    return out, peaks
