"""
pembersih.py — logika v3.4 (pembersih detak), dipakai notebook, tes, dan laporan. Rancangan: RANCANGAN.md (dikunci).

Front-end di atas detektor produksi `hrv_core.detect_rpeaks` (tidak diubah):
  F1  penyelarasan setiap R-peak ke templat QRS median ±5 s (sinyal bertanda, bukan |y|), lag −3…+3 sampel + parabola.
  F2  F1 + gerbang morfologi: window ditolak bila < 60% detak penutup IBI-nya dominan (korelasi ≥ 0,8, ukuran 0,6–1,67).
Ide dari pembaca EKG Tugas Akhir pemilik repo (asd_ecg/measurement.py, `_dominant_beats`).

Bagian:
  1. front-end (bandpass yang dibuktikan identik dengan detektor, penyelarasan, gerbang)
  2. AFDB: ekstraktor v2 sumber "detector" dibangun ulang tanpa wfdb (dibuktikan identik dengan cache v2), dataset F1
  3. training MicroTransformer resep v3 per fold (bobot disimpan), uji paritas
  4. evaluasi AFDB (protokol 1–8, 10)
  5. data latihan fisik (v3.3) dan alarm
"""
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.signal as sps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "v3.3_takikardia_asli"), str(REPO / "v3.2_uji_takikardia"),
                str(REPO / "v5_training_afl"), str(REPO / "v12_shdb_atrium"), str(REPO / "v3_evaluasi_robust"),
                str(REPO / "v2_revisi")]
import hrv_core  # noqa: E402
import sihedaf_cv as cv  # noqa: E402
import atrium as A12  # noqa: E402  (pembaca WFDB tanpa wfdb, dibuktikan sama di v12)
import latih as L5  # noqa: E402  (konstanta ekstraktor v2 dan resep: BLACKLIST, LABEL_MAP, HP, MODEL_KWARGS)
import takikardia as T  # noqa: E402  (v3.2)
import takikardia_asli as T33  # noqa: E402  (v3.3)

FS = hrv_core.FS_DEVICE
AFDB = REPO / "data/afdb"
DATA_LAMA = REPO / "cache/dataset_v2_detector.npz"
OOF_DIR = REPO / "cache/oof_robust/detector_e30_r5_s2026"
CACHE = REPO / "cache/v3_4"
MODELS = CACHE / "model"
N_REPEATS, SEED, N_BOOT = 5, 2026, 2000
CLASSES = ("Normal", "AFIB", "AFL")

# parameter front-end (RANCANGAN bagian 2; tidak disetel)
SEGMEN_S = 0.10          # setengah lebar segmen QRS
MAX_LAG = 3              # sampel (60 ms di 50 Hz)
KONTEKS_S = 5.0          # templat dari detak dengan |Δt| ≤ 5 s
MIN_TEMPLAT = 3
MIN_KORELASI = 0.8
RENTANG_UKURAN = (0.6, 1.67)
MIN_DOMINAN = 0.6


# ══ 1. front-end ═══════════════════════════════════════════════════════
def bandpass(x, fs=FS):
    """Sinyal acuan detektor: salinan langkah 1 `hrv_core.detect_rpeaks` (dibuktikan oleh `detektor_salinan`)."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64).ravel())
    lo, hi = np.percentile(x, [0.5, 99.5])
    x = np.clip(x, lo, hi)
    nyq = fs / 2.0
    high = min(15.0, 0.8 * nyq)
    b, a = sps.butter(2, [5.0 / nyq, high / nyq], btype="band")
    return sps.filtfilt(b, a, x)


def detektor_salinan(ecg, fs=FS, chunk_sec=10.0):
    """`hrv_core.detect_rpeaks` dibangun dari `bandpass` di atas. Hanya untuk bukti: hasilnya harus sama bit demi bit."""
    x = np.nan_to_num(np.asarray(ecg, dtype=np.float64).ravel())
    if x.size < int(2 * fs):
        return np.array([], dtype=np.float64)
    y = bandpass(x, fs)
    d = np.gradient(y)
    w = max(1, int(round(0.15 * fs)))
    mwi = np.convolve(d * d, np.ones(w) / w, mode="same")
    thr = np.empty_like(mwi)
    n_chunk = max(1, int(round(chunk_sec * fs)))
    for s in range(0, mwi.size, n_chunk):
        thr[s:s + n_chunk] = 0.3 * np.percentile(mwi[s:s + n_chunk], 98)
    cand, _ = sps.find_peaks(mwi, distance=max(1, int(round(0.25 * fs))))
    cand = cand[mwi[cand] >= thr[cand]]
    if cand.size == 0:
        return np.array([], dtype=np.float64)
    half = max(1, int(round(0.08 * fs)))
    ay = np.abs(y)
    peaks = []
    for c in cand:
        s, e = max(0, c - half), min(ay.size, c + half + 1)
        p = s + int(np.argmax(ay[s:e]))
        if 0 < p < ay.size - 1:
            y0, y1, y2 = ay[p - 1], ay[p], ay[p + 1]
            den = y0 - 2 * y1 + y2
            off = float(np.clip(0.5 * (y0 - y2) / den if den != 0 else 0.0, -0.5, 0.5))
        else:
            off = 0.0
        peaks.append(p + off)
    peaks = np.unique(np.round(np.asarray(peaks), 3))
    return peaks[np.concatenate(([True], np.diff(peaks) >= 0.25 * fs))]


def selaraskan(y, peaks, fs=FS):
    """
    F1: R-peak (sampel, pecahan) → (posisi baru, korelasi terbaik, rasio ukuran, dominan).
    Templat tiap detak = median segmen QRS detak dengan |Δt| ≤ KONTEKS_S. Detak tanpa templat (< MIN_TEMPLAT segmen lengkap)
    atau di ujung sinyal tidak digeser, korelasi NaN, dan dianggap dominan.
    """
    peaks = np.asarray(peaks, np.float64)
    n = peaks.size
    out, kor, rasio, dom = peaks.copy(), np.full(n, np.nan), np.full(n, np.nan), np.ones(n, bool)
    if n == 0:
        return out, kor, rasio, dom
    half = int(round(SEGMEN_S * fs))
    off = np.arange(-half, half + 1)
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    c = np.round(peaks).astype(np.int64)
    lengkap = (c - half - MAX_LAG >= 0) & (c + half + MAX_LAG < y.size)
    seg0 = y[np.clip(c[:, None] + off[None], 0, y.size - 1)]                      # (n, 2h+1)
    t = peaks / fs
    lo = np.searchsorted(t, t - KONTEKS_S, side="left")
    hi = np.searchsorted(t, t + KONTEKS_S, side="right")
    cum = np.concatenate(([0], np.cumsum(lengkap)))
    tpl = np.full(seg0.shape, np.nan)
    for i in np.flatnonzero(lengkap & (cum[hi] - cum[lo] >= MIN_TEMPLAT)):
        j = np.arange(lo[i], hi[i])
        tpl[i] = np.median(seg0[j[lengkap[j]]], axis=0)
    ok = np.isfinite(tpl).all(1)
    if not ok.any():
        return out, kor, rasio, dom
    ci = c[ok]
    E = y[ci[:, None, None] + lags[None, :, None] + off[None, None, :]]             # (m, lag, 2h+1)
    En = E - E.mean(-1, keepdims=True)
    sE = En.std(-1, keepdims=True)
    En = np.where(sE > 0, En / np.where(sE > 0, sE, 1), 0.0)
    Tm = tpl[ok]
    Tn = Tm - Tm.mean(-1, keepdims=True)
    sT = Tn.std(-1, keepdims=True)
    Tn = np.where(sT > 0, Tn / np.where(sT > 0, sT, 1), 0.0)
    cs = (En * Tn[:, None, :]).mean(-1)                                               # (m, lag)
    k = cs.argmax(1)
    m = np.arange(len(k))
    a = cs[m, np.clip(k - 1, 0, None)]
    b = cs[m, k]
    d = cs[m, np.clip(k + 1, None, len(lags) - 1)]
    tengah = (k > 0) & (k < len(lags) - 1)
    den = a - 2 * b + d
    geser = np.where(tengah & (den != 0), 0.5 * (a - d) / np.where(den != 0, den, 1), 0.0)
    geser = np.clip(geser, -0.5, 0.5)
    out[ok] = ci + lags[k] + geser
    kor[ok] = b
    ukuran_t = np.ptp(Tm, axis=1)
    rasio[ok] = np.where(ukuran_t > 0, np.ptp(E[m, k], axis=1) / np.where(ukuran_t > 0, ukuran_t, 1), np.inf)
    dom[ok] = (b >= MIN_KORELASI) & (rasio[ok] >= RENTANG_UKURAN[0]) & (rasio[ok] <= RENTANG_UKURAN[1])
    o = np.argsort(out, kind="stable")
    assert np.array_equal(o, np.arange(n)), "urutan detak berubah setelah penyelarasan"
    return out, kor, rasio, dom


def proses_sinyal(ecg, fs_in):
    """EKG mentah → sinyal 50 Hz, R-peak front-end lama, R-peak F1, dan status dominan per detak."""
    x = hrv_core.resample_signal(np.nan_to_num(np.asarray(ecg, np.float64)), fs_in, FS)
    p0 = hrv_core.detect_rpeaks(x, FS)
    p1, kor, rasio, dom = selaraskan(bandpass(x), p0)
    return dict(x=x, p_lama=p0, p_f1=p1, korelasi=kor, rasio=rasio, dominan=dom)


def fraksi_dominan(t_end, dom_tutup, t0):
    """Fraksi detak penutup IBI yang dominan dalam window [t0, t0+10) (IBI yang sama dengan yang dipilih window)."""
    i0, i1 = np.searchsorted(t_end, [t0, t0 + hrv_core.WINDOW_SEC], side="left")
    return float(dom_tutup[i0:i1].mean()) if i1 > i0 else np.nan


# ══ 2. AFDB ════════════════════════════════════════════════════════════
def daftar_rekaman():
    return [l.strip() for l in open(AFDB / "RECORDS") if l.strip() and l.strip() not in L5.BLACKLIST]


def _qrs_referensi(rec):
    """Anotasi QRS referensi AFDB (.qrs, format MIT sama dengan .atr) lewat pembaca v12."""
    import shutil
    import tempfile
    d = Path(tempfile.mkdtemp())
    try:
        (d / f"{rec}.atr").symlink_to(AFDB / f"{rec}.qrs")
        s, _, _ = A12.baca_atr(d / rec)
    finally:
        shutil.rmtree(d)
    return s


def _label_detak(peaks, fs_rek, r_samp, r_lab):
    close = peaks[1:] * (fs_rek / FS)
    idx = np.searchsorted(r_samp, close, side="right") - 1
    return np.where(idx >= 0, r_lab[np.clip(idx, 0, None)], -1)


def _window_afdb(t_end, ibi, lab, dom_tutup=None):
    """Ekstraktor v2 (window valid digeser 5 s, tidak valid 1 s) = salinan v5 `latih._process_record`, + fraksi dominan."""
    rows, t, W = [], 0.0, hrv_core.WINDOW_SEC
    if len(t_end) == 0:
        return rows
    while t + W <= t_end[-1]:
        i0, i1 = np.searchsorted(t_end, [t, t + W], side="left")
        wl = lab[i0:i1]
        ok, _ = hrv_core.window_quality(ibi[i0:i1])
        if ok and wl.size and wl[0] >= 0 and np.all(wl == wl[0]):
            fd = float(dom_tutup[i0:i1].mean()) if dom_tutup is not None else np.nan
            rows.append((t, ibi[i0:i1].copy(), int(wl[0]), fd))
            t += 5.0
        else:
            t += L5.INVALID_STEP_SEC
    return rows


def proses_afdb(rec):
    """Satu rekaman AFDB → window front-end lama dan F1, plus galat posisi vs .qrs per ritme (deskriptif)."""
    x, fs = A12.baca_dat(AFDB / rec, L5.ECG_CHANNEL)
    s, _, aux = A12.baca_atr(AFDB / rec)
    rhythm = [(int(a), b.strip("\x00").strip()) for a, b in zip(s, aux) if b.strip("\x00").strip().startswith("(")]
    r_samp = np.array([a for a, _ in rhythm])
    r_lab = np.array([L5.LABEL_MAP.get(b, -1) for _, b in rhythm])
    P = proses_sinyal(x, fs)
    out = {"rec": rec}
    for fe, p in (("lama", P["p_lama"]), ("f1", P["p_f1"])):
        if p is None or len(p) < hrv_core.MIN_IBI + 1:
            out[fe] = []
            continue
        t_end, ibi = hrv_core.peaks_to_ibi(p, FS)
        lab = _label_detak(p, fs, r_samp, r_lab)
        out[fe] = _window_afdb(t_end, ibi, lab, P["dominan"][1:] if fe == "f1" else None)
    # galat posisi vs .qrs (offset median per rekaman dibuang; hanya deviasi yang memengaruhi IBI)
    ref = np.sort(_qrs_referensi(rec)) / fs
    lab_ref = np.searchsorted(r_samp / fs, ref, side="right") - 1
    lab_ref = np.where(lab_ref >= 0, r_lab[np.clip(lab_ref, 0, None)], -1)
    for fe, p in (("lama", P["p_lama"]), ("f1", P["p_f1"])):
        e, j = cocokkan(np.asarray(p) / FS, ref)
        out[f"galat_{fe}"] = ((e - np.median(e)) * 1000).astype(np.float32) if e.size else np.zeros(0, np.float32)
        out[f"ritme_{fe}"] = lab_ref[j].astype(np.int8)
    out["dominan_detak"] = np.array([np.mean(P["dominan"]), np.nanmean(P["korelasi"] < MIN_KORELASI)])
    return out


def cocokkan(det_s, ref_s, tol=0.15):
    """Pasangan satu-satu detak deteksi ↔ referensi terdekat (≤ tol s). Return (galat det − ref, indeks referensi)."""
    det_s = np.asarray(det_s)
    j = np.clip(np.searchsorted(ref_s, det_s), 1, len(ref_s) - 1)
    pilih = np.where(np.abs(ref_s[j - 1] - det_s) <= np.abs(ref_s[j] - det_s), j - 1, j)
    e = det_s - ref_s[pilih]
    m = np.abs(e) <= tol
    # satu referensi paling banyak satu pasangan: ambil yang terdekat
    urut = np.lexsort((np.abs(e), pilih))
    unik = np.zeros(len(e), bool)
    _, first = np.unique(pilih[urut], return_index=True)
    unik[urut[first]] = True
    m &= unik
    return e[m], pilih[m]


def _ke_array(rows_per_rec):
    rows = [(rec, *w) for rec, ws in rows_per_rec for w in ws]
    seqs = [hrv_core.build_sequence(w[2]) for w in rows]
    return {"X": np.stack([q[0] for q in seqs]).astype(np.float32), "M": np.stack([q[1] for q in seqs]).astype(np.float32),
            "y": np.array([w[3] for w in rows], dtype=np.int64), "records": np.array([w[0] for w in rows]),
            "t_start": np.array([w[1] for w in rows], dtype=np.float32),
            "dominan": np.array([w[4] for w in rows], dtype=np.float32)}


def bangun_afdb(workers=6, log=print):
    """Dataset F1 + galat posisi → cache. Front-end lama harus identik bit demi bit dengan dataset v2 (assert)."""
    p_ds, p_gl = CACHE / "dataset_f1.npz", CACHE / "galat_afdb.npz"
    if p_ds.exists() and p_gl.exists():
        return muat_afdb_f1()
    t0 = time.perf_counter()
    recs = daftar_rekaman()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        hasil = list(ex.map(proses_afdb, recs))
    log(f"  {len(recs)} rekaman diproses ({(time.perf_counter() - t0) / 60:.1f} menit)")
    lama = _ke_array([(h["rec"], h["lama"]) for h in hasil])
    ref = np.load(DATA_LAMA)
    for k in ("X", "M", "y", "t_start"):
        assert np.array_equal(lama[k], ref[k]), f"ekstraktor v3.4 (front-end lama) berbeda dari dataset v2 pada '{k}'"
    assert np.array_equal(lama["records"].astype(str), ref["records"].astype(str))
    log(f"  front-end lama identik dengan dataset_v2_detector ({len(lama['y'])} window)")
    f1 = _ke_array([(h["rec"], h["f1"]) for h in hasil])
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = p_ds.with_name(p_ds.stem + ".tmp.npz")
    np.savez_compressed(tmp, **f1)
    tmp.replace(p_ds)
    gl = {}
    for h in hasil:
        for k in ("galat_lama", "galat_f1", "ritme_lama", "ritme_f1", "dominan_detak"):
            gl[f"{h['rec']}__{k}"] = h[k]
    tmp = p_gl.with_name(p_gl.stem + ".tmp.npz")
    np.savez_compressed(tmp, **gl)
    tmp.replace(p_gl)
    return muat_afdb_f1()


def muat_afdb_f1():
    z = np.load(CACHE / "dataset_f1.npz")
    return {k: z[k] for k in z.files}


def fold_per_window(records_baru):
    """(R, N_baru): fold pasien resmi (folds.npz, per window dataset v2) dipetakan ke window baru lewat rekaman."""
    ref = np.load(DATA_LAMA)
    rec_lama = ref["records"].astype(str)
    folds = np.load(OOF_DIR / "folds.npz")["folds"]
    out = np.zeros((folds.shape[0], len(records_baru)), folds.dtype)
    for r in range(folds.shape[0]):
        peta = {}
        for p in np.unique(rec_lama):
            f = np.unique(folds[r][rec_lama == p])
            assert f.size == 1, "fold resmi tidak per pasien"
            peta[p] = f[0]
        out[r] = np.array([peta[p] for p in records_baru.astype(str)])
    return out


# ══ 3. training ════════════════════════════════════════════════════════
def _path_job(varian, r):
    if varian == "paritas":
        return CACHE / "paritas_r0.npz"
    return CACHE / "oof" / f"MicroTransformer_{varian}_r{r}.npz"


def jalankan_job(job):
    """
    Satu (varian, ulangan): latih 5 fold dengan fungsi referensi `sihedaf_cv.train_dl` (resep v3 = v3.1), simpan OOF dan bobot.
    varian "paritas": dataset v2 lama, ulangan 0 (bukti fungsi job = referensi). varian "f1": dataset F1.
    """
    import torch
    varian, r = job
    out = _path_job(varian, r)
    if out.exists():
        return varian, r, None
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if varian == "paritas":
        z = np.load(DATA_LAMA)
        folds_r = np.load(OOF_DIR / "folds.npz")["folds"][r]
    else:
        z = np.load(CACHE / f"dataset_{varian}.npz")
        folds_r = fold_per_window(z["records"])[r]
    X, M, y = z["X"], z["M"], z["y"]
    oof = np.full((len(y), 3), np.nan, np.float32)
    times = []
    for k in range(5):
        tr, te = folds_r != k, folds_r == k
        net, mu, sd, t = cv.train_dl("MicroTransformer", L5.MODEL_KWARGS, X[tr], M[tr], y[tr], seed=SEED + 100 * r + k,
                                     device=device, **L5.HP)
        oof[te] = cv.predict_dl(net, X[te], M[te], mu, sd, device)
        times.append(t)
        if varian != "paritas":
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {a: b.cpu() for a, b in net.state_dict().items()}, "mu": torch.from_numpy(mu),
                        "sd": torch.from_numpy(sd), "model_kwargs": L5.MODEL_KWARGS, "varian": varian, "repeat": r, "fold": k},
                       MODELS / f"{varian}_r{r}_k{k}.pt")
    assert not np.isnan(oof).any()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, oof=oof, times=np.asarray(times))
    tmp.replace(out)
    return varian, r, float(np.mean(times))


def semua_job():
    return [("paritas", 0)] + [("f1", r) for r in range(N_REPEATS)]


def jalankan_semua(log=print):
    """Berurutan dalam satu proses GPU (model kecil; paralel tidak lebih cepat). Job yang sudah ada dilewati."""
    jobs = [j for j in semua_job() if not _path_job(*j).exists()]
    log(f"{len(jobs)} job tersisa dari {len(semua_job())}")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        for v, r, t in ex.map(jalankan_job, jobs):
            log(f"  [{(time.perf_counter() - t0) / 60:5.1f} mnt] {v:<8} ulangan {r}: {t:.0f} s/fold")


def cek_paritas():
    a = np.load(_path_job("paritas", 0))["oof"]
    b = np.load(OOF_DIR / "MicroTransformer_r0.npz")["oof"]
    return {"selisih_maks": float(np.abs(a - b).max()), "argmax_sama": float((a.argmax(1) == b.argmax(1)).mean()),
            "n": int(len(a))}


# ══ 4. evaluasi AFDB ═══════════════════════════════════════════════════
def _ibi(X, M):
    return [X[i, :int(M[i].sum()), 0] for i in range(len(X))]


def data_afdb():
    """Dua dataset (lama = referensi v3.1, f1) dengan pasien, strata (dari dataset lama), fold, OOF, dan prediksi +bias."""
    lama = T.load()
    z = muat_afdb_f1()
    ib = _ibi(z["X"], z["M"])
    f1 = dict(y=z["y"], rec=z["records"].astype(str), t=z["t_start"], dominan=z["dominan"],
              hr=np.array([60000.0 / v.mean() for v in ib]), cv=np.array([v.std() / v.mean() for v in ib]),
              patients=lama["patients"], strata=lama["strata"], folds=fold_per_window(z["records"]))
    assert np.array_equal(np.unique(f1["rec"]), lama["patients"])
    f1["probs"] = np.stack([np.load(_path_job("f1", r))["oof"] for r in range(N_REPEATS)])
    lama["t"] = np.load(DATA_LAMA)["t_start"]
    pred_lama, bias_lama, pred_f1, bias_f1 = [], [], [], []
    for r in range(N_REPEATS):
        a, b = cv.crossfit_bias(lama["y"], lama["probs"]["v31"][r], lama["folds"][r])
        pred_lama.append(a.argmax(1)); bias_lama.append(b)
        a, b = cv.crossfit_bias(f1["y"], f1["probs"][r], f1["folds"][r])
        pred_f1.append(a.argmax(1)); bias_f1.append(b)
    lama["pred"], lama["bias"] = np.stack(pred_lama), np.stack(bias_lama)
    f1["pred"], f1["bias"] = np.stack(pred_f1), np.stack(bias_f1)
    f1["lolos_gerbang"] = f1["dominan"] >= MIN_DOMINAN
    f1["pred_f2"] = np.where(f1["lolos_gerbang"][None], f1["pred"], 0)          # ditolak = sinyal buruk = tidak ada alarm
    return lama, f1


def cms(d, pred, mask=None):
    m = np.ones(len(d["y"]), bool) if mask is None else mask
    return np.stack([cv.patient_cms(d["y"][m], p[m], d["rec"][m], d["patients"]) for p in pred])


def bobot_bootstrap(strata):
    return cv.bootstrap_weights(strata, N_BOOT, seed=SEED)


def ringkas_metrik(c, W):
    """c: (R, P, 3, 3). Titik (rata-rata ulangan) + CI 95% dan sampel bootstrap per metrik."""
    titik = cv.point_metrics(c)
    boot = cv.boot_metrics(c, W)
    return {m: dict(nilai=float(titik[m].mean()), ci=np.percentile(boot[m], [2.5, 97.5]).tolist()) for m in cv.METRICS}, boot


def banding(boot_a, boot_b, titik_a, titik_b, metrik=("Macro-F1", "AFL R", "AFL F1", "AFIB R", "Normal R")):
    """Δ = a − b, CI dan P(Δ>0) dari sampel bootstrap yang sama; vonis protokol."""
    out = {}
    for m in metrik:
        dd = boot_a[m] - boot_b[m]
        p = float(np.mean(dd > 0))
        # Δ = 0 di semua sampel (mis. F2 vs F1 pada AFL: gerbang tidak menolak window AFL) → "identik", bukan "lebih buruk"
        identik = bool(np.all(dd == 0)) and titik_a[m] == titik_b[m]
        out[m] = dict(delta=float(titik_a[m] - titik_b[m]), ci=np.percentile(dd, [2.5, 97.5]).tolist(), p=p, p_sama=float(np.mean(dd == 0)),
                      vonis="identik" if identik else
                      ("lebih baik" if p >= 0.95 else ("lebih buruk" if p <= 0.05 else "tidak terbedakan")))
    return out


def afl_per_pasien(d, pred):
    out = {}
    for p in d["patients"]:
        m = (d["rec"] == p) & (d["y"] == 2)
        if m.sum():
            out[p] = dict(n=int(m.sum()), recall=float((pred[:, m] == 2).mean()))
    return out


def asal_afl_palsu(d, pred):
    """AFL palsu per ulangan, dipisah menurut kelas asli dan apakah pasiennya punya AFL."""
    punya_afl = np.isin(d["rec"], np.unique(d["rec"][d["y"] == 2]))
    palsu = (pred == 2) & (d["y"] != 2)[None]
    return {"Normal, pasien tanpa AFL": float(palsu[:, (d["y"] == 0) & ~punya_afl].sum() / N_REPEATS),
            "Normal, pasien AFL": float(palsu[:, (d["y"] == 0) & punya_afl].sum() / N_REPEATS),
            "AFIB, pasien tanpa AFL": float(palsu[:, (d["y"] == 1) & ~punya_afl].sum() / N_REPEATS),
            "AFIB, pasien AFL": float(palsu[:, (d["y"] == 1) & punya_afl].sum() / N_REPEATS)}


def episode(d, pred):
    out = {}
    for kelas, n in ((1, "AFIB"), (2, "AFL")):
        ms = [T33.metrik_episode(d["rec"], d["t"], d["y"], pred[r], kelas) for r in range(N_REPEATS)]
        out[n] = {m: float(np.mean([x[m] for x in ms])) for m in ms[0]}
    return out


def kandidat_cms(d):
    """(R, G, P, 3, 3) per kandidat bias grid (pola v3.2 `candidate_cms`) untuk probabilitas d['probs']."""
    probs = d["probs"] if not isinstance(d["probs"], dict) else d["probs"]["v31"]
    return T.candidate_cms(d, probs)[0]


def seri_bersarang(d, cand, W):
    """
    Macro-F1 per sampel bootstrap dengan bias dipilih ULANG di setiap sampel (logika `takikardia.nested_bootstrap`, dipakai
    untuk dua dataset dengan bobot pasien yang sama). Baris 0 = titik (bobot semua-satu = cross-fit biasa).
    """
    W = np.vstack([np.ones(W.shape[1]), W])
    pfold = np.array([[int(d["folds"][r][d["rec"] == p][0]) for p in d["patients"]] for r in range(N_REPEATS)])
    G, P = cand.shape[1], len(d["patients"])
    hasil = np.zeros((len(W), N_REPEATS))
    for r in range(N_REPEATS):
        flat = cand[r].reshape(G, P, 9)
        acc = np.zeros((len(W), 9))
        for k in range(5):
            tr, te = pfold[r] != k, pfold[r] == k
            train_cm = W[:, tr] @ flat[:, tr].transpose(1, 0, 2).reshape(tr.sum(), G * 9)
            gbest = T._macro_f1_cm(train_cm.reshape(len(W), G, 3, 3)).argmax(1)
            acc += np.einsum("bpc,bp->bc", flat[gbest][:, te], W[:, te])
        hasil[:, r] = T._macro_f1_cm(acc.reshape(len(W), 3, 3))
    return hasil.mean(1)


def window_sama(lama, f1):
    """Indeks window yang ada di kedua dataset dengan (rekaman, t_start, label) sama."""
    kunci_l = {(r, round(float(t), 3), int(y)): i for i, (r, t, y) in enumerate(zip(lama["rec"], lama["t"], lama["y"]))}
    il, i1 = [], []
    for j, (r, t, y) in enumerate(zip(f1["rec"], f1["t"], f1["y"])):
        i = kunci_l.get((r, round(float(t), 3), int(y)))
        if i is not None:
            il.append(i); i1.append(j)
    return np.array(il), np.array(i1)


def prediksi_subset(d, idx, probs):
    """Prediksi +bias untuk window idx saja; bias cross-fit dihitung ulang di subset (pola v3.2 `oof_predictions`)."""
    return np.stack([cv.crossfit_bias(d["y"][idx], probs[r][idx], d["folds"][r][idx])[0].argmax(1) for r in range(N_REPEATS)])


def subset(d, idx, pred):
    """Dataset ringkas (y, rec, patients, pred) untuk window idx; pred sudah dihitung untuk subset."""
    return dict(y=d["y"][idx], rec=d["rec"][idx], t=d["t"][idx], patients=d["patients"], pred=pred)


# ── drop-in (H5): model resep v5b front-end lama (v7 kontrol, ulangan 0) di window F1 ──
def prediksi_dropin(f1, device="cpu"):
    import torch
    from sihedaf_models import build_model
    z_l = np.load(DATA_LAMA)
    oof_lama = np.load(REPO / "cache/v7/MicroTransformer_kontrol_r0.npz")["oof"]
    folds_l = np.load(OOF_DIR / "folds.npz")["folds"][0]
    adj_lama, b = cv.crossfit_bias(z_l["y"], oof_lama, folds_l)
    zf = muat_afdb_f1()
    p = np.zeros((len(f1["y"]), 3), np.float32)
    for k in range(5):
        w = torch.load(REPO / f"cache/v7/model/kontrol_r0_k{k}.pt", map_location="cpu", weights_only=True)
        net = build_model("MicroTransformer", **w["model_kwargs"])
        net.load_state_dict(w["state_dict"])
        net = net.to(device).eval()
        te = f1["folds"][0] == k
        p[te] = cv.apply_bias(cv.predict_dl(net, zf["X"][te], zf["M"][te], w["mu"].numpy(), w["sd"].numpy(), torch.device(device)), b[k])
    return adj_lama.argmax(1)[None], p.argmax(1)[None]


# ══ 5. latihan fisik ═══════════════════════════════════════════════════
def jendela_latihan(log=print):
    """
    Window mode perangkat EPFL, Wrist, EPHNOGRAM untuk front-end lama dan F1 (grid sama dengan v3.3, langkah 10 s), fraksi
    dominan F1, dan galat posisi vs R-peak manual (EPFL, Wrist). Front-end lama harus sama dengan cache/v3_3/jendela.npz.
    """
    path = CACHE / "latihan.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    kol = {k: [] for k in ("sumber", "subjek", "rekaman", "start", "ok_lama", "ok_f1", "hr_lama", "hr_f1", "cv_lama", "cv_f1",
                           "dominan")}
    X = {"lama": [], "f1": []}
    galat = {"sumber": [], "subjek": [], "lama": [], "f1": []}
    for sumber, gen in (("EPFL", T33.rekaman_epfl), ("Wrist", T33.rekaman_wrist), ("EPHNOGRAM", T33.rekaman_ephnogram)):
        n0 = len(kol["start"])
        for rek in gen():
            P = proses_sinyal(rek["ecg"], rek["fs"])
            dur = P["x"].size / FS
            per_fe = {}
            for fe in ("lama", "f1"):
                t_end, ibi = hrv_core.peaks_to_ibi(P[f"p_{fe}"], FS)
                per_fe[fe] = (t_end, T33._potong(t_end, ibi, dur))
            t_end1 = per_fe["f1"][0]
            dom_tutup = P["dominan"][1:]
            for (s, w0, ok0, _), (s1, w1, ok1, _) in zip(per_fe["lama"][1], per_fe["f1"][1]):
                assert s == s1
                for k, v in (("sumber", sumber), ("subjek", rek["subjek"] or rek["rekaman"]), ("rekaman", rek["rekaman"]),
                             ("start", s), ("ok_lama", ok0), ("ok_f1", ok1),
                             ("hr_lama", 60000.0 / w0.mean() if w0.size else np.nan), ("hr_f1", 60000.0 / w1.mean() if w1.size else np.nan),
                             ("cv_lama", w0.std() / w0.mean() if w0.size else np.nan), ("cv_f1", w1.std() / w1.mean() if w1.size else np.nan),
                             ("dominan", fraksi_dominan(t_end1, dom_tutup, s))):
                    kol[k].append(v)
                for fe, w, ok in (("lama", w0, ok0), ("f1", w1, ok1)):
                    xs, ms = hrv_core.build_sequence(w) if ok else (np.zeros((hrv_core.MAX_BEATS, 2), np.float32),
                                                                      np.zeros(hrv_core.MAX_BEATS, np.float32))
                    X[fe].append((xs, ms))
            if rek.get("rpeak") is not None:
                ref = np.sort(rek["rpeak"]) / rek["fs"]
                galat["sumber"].append(sumber); galat["subjek"].append(rek["subjek"])
                for fe in ("lama", "f1"):
                    e, _ = cocokkan(P[f"p_{fe}"] / FS, ref, tol=0.10)
                    galat[fe].append(float(np.std(e - np.median(e)) * 1000) if e.size > 2 else np.nan)
        log(f"  {sumber}: {len(kol['start']) - n0} window")
    out = {k: np.array(v) for k, v in kol.items()}
    for fe in ("lama", "f1"):
        out[f"X_{fe}"] = np.stack([a for a, _ in X[fe]])
        out[f"M_{fe}"] = np.stack([b for _, b in X[fe]])
    for k, v in galat.items():
        out[f"galat_{k}"] = np.array(v)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def cek_latihan_v33(L):
    """Front-end lama v3.4 = window mode perangkat v3.3 (start, ok, IBI, detak)."""
    J = T33.muat_jendela()
    m = J["mode"] == "perangkat"
    assert np.array_equal(J["rekaman"][m], L["rekaman"]) and np.array_equal(J["start"][m], L["start"])
    assert np.array_equal(J["ok"][m], L["ok_lama"])
    assert np.array_equal(J["X"][m], L["X_lama"]) and np.array_equal(J["M"][m], L["M_lama"])
    assert np.allclose(J["hr"][m], L["hr_lama"], equal_nan=True)
    return int(m.sum())


def paket_v31():
    from predictor import ECGPredictor
    return ECGPredictor(str(REPO / "v3.1_kalibrasi_produksi/model_export"), name="sihedaf_v3_1", device="cpu")


def prediksi_latihan(L, device="cpu"):
    """Kelas per window (N), −1 = tidak lolos QC. Paket v3.1 untuk front-end lama, F1, F2; keluarga fold F1 (R·5, N)."""
    import torch
    from sihedaf_models import build_model
    path = CACHE / "prediksi_latihan.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    v31 = paket_v31()
    out = {}
    for fe in ("lama", "f1"):
        ok = L[f"ok_{fe}"]
        k = np.full(len(ok), -1, np.int64)
        k[ok] = v31.predict_proba_sequences(L[f"X_{fe}"][ok], L[f"M_{fe}"][ok]).argmax(1)
        out[f"v31_{fe}"] = k
    out["v31_f2"] = np.where(L["ok_f1"] & (L["dominan"] < MIN_DOMINAN), 0, out["v31_f1"])
    # keluarga fold F1 dengan bias fold (pola v3.3 prediksi_keluarga)
    lama, f1 = data_afdb()
    ok = L["ok_f1"]
    fam = np.full((N_REPEATS * 5, len(ok)), -1, np.int64)
    dev = torch.device(device)
    for r in range(N_REPEATS):
        for k in range(5):
            w = torch.load(MODELS / f"f1_r{r}_k{k}.pt", map_location="cpu", weights_only=True)
            net = build_model("MicroTransformer", **w["model_kwargs"])
            net.load_state_dict(w["state_dict"])
            p = cv.predict_dl(net.to(dev).eval(), L["X_f1"][ok], L["M_f1"][ok], w["mu"].numpy(), w["sd"].numpy(), dev)
            fam[5 * r + k, ok] = cv.apply_bias(p, f1["bias"][r][k]).argmax(1)
    out["fold_f1"] = fam
    np.savez_compressed(path, **out)
    return out


def boot_subjek_selisih(subj, delta):
    """Δ rata-rata per window, bootstrap subjek berpasangan (pola v3.3). P = P(Δ < 0)."""
    t, ci, b = T33._boot_subjek(subj, delta)
    return dict(delta=t, ci=ci.tolist(), p_kurang=float((b < 0).mean()), n=int(len(delta)), subjek=int(np.unique(subj).size))


def alarm(kelas):
    """kelas (N) atau (G, N) → fraksi alarm per window (lolos QC dan bukan Normal)."""
    k = np.atleast_2d(kelas)
    return ((k > 0)).mean(0)


def stres_fold_f1(f1):
    """Uji stres sintetis v3.2 untuk 25 model fold F1 (bias fold); tabel = rata-rata proporsi antar model."""
    import torch
    from sihedaf_models import build_model
    tabel = {j: np.zeros((len(T.BPMS), 3)) for j in T.JITTERS}
    for r in range(N_REPEATS):
        for k in range(5):
            w = torch.load(MODELS / f"f1_r{r}_k{k}.pt", map_location="cpu", weights_only=True)
            net = build_model("MicroTransformer", **w["model_kwargs"])
            net.load_state_dict(w["state_dict"])
            net.eval()
            b = f1["bias"][r][k]

            def run(seqs, net=net, w=w, b=b):
                X = np.stack([s[0] for s in seqs]); M = np.stack([s[1] for s in seqs])
                return cv.apply_bias(cv.predict_dl(net, X, M, w["mu"].numpy(), w["sd"].numpy(), torch.device("cpu")), b).argmax(1)
            res = T.stress_test({"m": run})["m"]
            for j in tabel:
                tabel[j] += res[j] / (N_REPEATS * 5)
    return tabel


def simpan_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=float, ensure_ascii=False))


# ══ 6. data eksternal (amandemen A) ════════════════════════════════════
EKSTERNAL = ("CinC 2017", "MIT-BIH", "LTAFDB", "CPSC 2021")
_ALAT = {}


def _modul_eksternal():
    """Pembaca dan label v16.3 (CinC, MIT-BIH, LTAFDB) dan v16.2 (CPSC 2021, label resmi); diimpor saat dipakai."""
    sys.path[:0] = [str(REPO / "v16.3_replikasi"), str(REPO / "v16.2_cpsc2021")]
    import replikasi as R163
    import uji_cpsc as U162
    return R163, U162


def daftar_eksternal(ds):
    R163, U162 = _modul_eksternal()
    if ds == "CPSC 2021":
        return [(rec, None) for rec in U162.K15.rekaman_cpsc()]
    return R163.rekaman(ds)


def _sinyal_label(ds, rec, label_rek):
    """(x, fs, fungsi label(start) → 0/1 (bukan AF / AF) atau None, subjek)."""
    R163, U162 = _modul_eksternal()
    if ds == "CPSC 2021":
        x, fs = A12.baca_dat(U162.K15.CPSC / rec, 0)
        f = U162._label_resmi(rec)
        return x, fs, (lambda s: None if f(s) is None else int(f(s) > 0)), U162.subjek(rec)
    base = REPO / "data" / R163.FOLDER[ds] / rec
    if ds == "CinC 2017":
        x, fs = R163.baca_cinc(base)
        return x, fs, (lambda s: int(label_rek in R163.AF)), R163.subjek(ds, rec)
    x, fs = A12.baca_dat(base, 0)
    seg = R163.K15.segmen_ritme(str(base))

    def f(s):
        cocok = [lb for a, b, lb in seg if a <= s and s + hrv_core.WINDOW_SEC <= b]
        if not cocok or cocok[0] in R163.BUANG:
            return None
        return int(cocok[0] in R163.AF)
    return x, fs, f, R163.subjek(ds, rec)


def proses_eksternal(arg):
    """Satu rekaman → cache/v3_4/eksternal/<ds>/<rekaman>.npz: window berlabel (grid signal_to_windows), QC, detak, CV, fraksi
    dominan, dan kelas paket v3.1 untuk front-end lama dan F1 (−1 = gagal QC). Front-end lama di-assert = signal_to_windows."""
    ds, rec, label_rek = arg
    p = CACHE / "eksternal" / ds.replace(" ", "_") / (rec.replace("/", "__") + ".npz")
    if p.exists():
        return ds, rec
    if "v31" not in _ALAT:
        _ALAT["v31"] = paket_v31()
    x, fs, label, subj = _sinyal_label(ds, rec, label_rek)
    x = np.nan_to_num(np.asarray(x, np.float64))
    ref, _ = hrv_core.signal_to_windows(x, fs)
    P_ = proses_sinyal(x, fs)
    dur = P_["x"].size / FS
    kol = {k: [] for k in ("y2", "start", "ok_lama", "ok_f1", "hr_lama", "hr_f1", "cv_lama", "cv_f1", "dominan")}
    X = {"lama": [], "f1": []}
    per = {}
    for fe in ("lama", "f1"):
        t_end, ibi = hrv_core.peaks_to_ibi(P_[f"p_{fe}"], FS)
        per[fe] = (t_end, _grid(t_end, ibi, dur))
    assert len(per["lama"][1]) == len(ref)
    for (s, w, ok, a), r in zip(per["lama"][1], ref):
        assert s == r["start_sec"] and ok == r["ok"] and np.array_equal(w, r["ibi_ms"]), f"{ds} {rec}: front-end lama ≠ signal_to_windows"
    for (s, w0, ok0, _), (_, w1, ok1, _) in zip(per["lama"][1], per["f1"][1]):
        lab = label(s)
        if lab is None:
            continue
        for k, v in (("y2", lab), ("start", s), ("ok_lama", ok0), ("ok_f1", ok1),
                     ("hr_lama", 60000.0 / w0.mean() if w0.size else np.nan), ("hr_f1", 60000.0 / w1.mean() if w1.size else np.nan),
                     ("cv_lama", w0.std() / w0.mean() if w0.size else np.nan), ("cv_f1", w1.std() / w1.mean() if w1.size else np.nan),
                     ("dominan", fraksi_dominan(per["f1"][0], P_["dominan"][1:], s))):
            kol[k].append(v)
        for fe, w, ok in (("lama", w0, ok0), ("f1", w1, ok1)):
            if ok:
                X[fe].append(hrv_core.build_sequence(w))
    out = {k: np.array(v) for k, v in kol.items()}
    for fe in ("lama", "f1"):
        k = np.full(len(kol["y2"]), -1, np.int8)
        ok = out[f"ok_{fe}"].astype(bool) if len(kol["y2"]) else np.zeros(0, bool)
        if ok.any():
            Xa = np.stack([a for a, _ in X[fe]]); Ma = np.stack([b for _, b in X[fe]])
            k[ok] = np.concatenate([_ALAT["v31"].predict_proba_sequences(Xa[i:i + 40000], Ma[i:i + 40000]).argmax(1)
                                    for i in range(0, len(Xa), 40000)])
        out[f"v31_{fe}"] = k
    out["subjek"] = np.array(subj)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p.with_suffix(".tmp.npz"), **out)
    p.with_suffix(".tmp.npz").replace(p)
    return ds, rec


def _grid(t_end, ibi, dur, stride=5.0):
    """Grid window `hrv_core.signal_to_windows` (langkah 5 s, tidak bergantung QC) dari urutan detak."""
    out = []
    for s in np.arange(0.0, max(dur - hrv_core.WINDOW_SEC, 0.0) + 1e-9, stride):
        i0, i1 = np.searchsorted(t_end, [s, s + hrv_core.WINDOW_SEC], side="left")
        w = ibi[i0:i1]
        ok, a = hrv_core.window_quality(w)
        out.append((float(s), w, ok, a))
    return out


def jalankan_eksternal(workers=6, log=print):
    """Semua rekaman keempat dataset (bisa dilanjutkan). Urutan: rekaman panjang (LTAFDB) dulu agar beban seimbang."""
    jobs = [(ds, rec, lab) for ds in ("LTAFDB", "CPSC 2021", "MIT-BIH", "CinC 2017") for rec, lab in daftar_eksternal(ds)]
    jobs = [j for j in jobs if not (CACHE / "eksternal" / j[0].replace(" ", "_") / (j[1].replace("/", "__") + ".npz")).exists()]
    log(f"{len(jobs)} rekaman tersisa")
    t0, n = time.perf_counter(), 0
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        for ds, rec in ex.map(proses_eksternal, jobs, chunksize=1):
            n += 1
            if n % 200 == 0 or n == len(jobs):
                log(f"  [{(time.perf_counter() - t0) / 60:5.1f} mnt] {n}/{len(jobs)} ({ds})")


def muat_eksternal(ds):
    bag = {}
    for rec, _ in daftar_eksternal(ds):
        z = np.load(CACHE / "eksternal" / ds.replace(" ", "_") / (rec.replace("/", "__") + ".npz"))
        n = z["y2"].size
        if n == 0:
            continue
        for k in z.files:
            if k != "subjek":
                bag.setdefault(k, []).append(z[k])
        bag.setdefault("subjek", []).append(np.full(n, str(z["subjek"])))
    return {k: np.concatenate(v) for k, v in bag.items()}
