"""
gerbang.py — logika v16.8 (pra-registrasi di RANCANGAN.md + amandemen A, dikunci 10 November 2026 sebelum SQI/SNR dihitung).

SQI window = median korelasi detak ke templat QRS lokal (`pembersih.selaraskan` v3.4, diimpor). Gerbang q mengeluarkan window dengan
SQI < q (atau NaN) seperti window yang gagal cek kualitas, lalu S1 (D-1 + episode AF ≥ 30 s). Tahap A memilih q di nstdb/LTAFDB/MIT-BIH;
tahap B memvalidasi di afpdb + derau nstdb (`ma` primer, `em` sekunder) dan Icentia11k. Rantai prediksi = v16.4, aturan/metrik = v16.5/16.6,
data Icentia/nstdb = v16.7 (diimpor, tidak disalin).
"""
import json
import sys
import zlib
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import scipy.signal as sps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "v16.7_konfirmasi_wearable"), str(REPO / "v3.4_pembersih_detak")]
import konfirmasi as K  # noqa: E402
from konfirmasi import C, E, R163, U164, hrv_core  # noqa: E402

CACHE = REPO / "cache/v16.8"
GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
SNR_VAL = (12, 6, 0)
DERAU = ("ma", "em")
SEED = 2026 + 168
FS50 = hrv_core.FS_DEVICE
SIG = {"LTAFDB": REPO / "data/ltafdb", "MIT-BIH": REPO / "data/mitdb", "NSRDB": REPO / "data/nsrdb", "nstdb": REPO / "data/nstdb",
       "mitdb": REPO / "data/mitdb", "Icentia": REPO / "data/icentia_v167", "afpdb": REPO / "data/afpdb", "afpdb_bising": REPO / "data/afpdb"}
A12 = R163.A12


# ── sinyal (termasuk afpdb + derau) ────────────────────────────────────
def _bp(x, fs):
    """Bandpass SNR (amandemen A): Butterworth orde 2, 0,5–40 Hz, filtfilt."""
    b, a = sps.butter(2, [0.5 / (fs / 2), min(40.0, 0.45 * fs) / (fs / 2)], btype="band")
    return sps.filtfilt(b, a, x)


def bising(rec):
    """'n01__ma__06' → (sinyal afpdb kanal 0 + derau nstdb kanal 0 pada SNR itu, fs)."""
    nama, derau, snr = rec.split("__")
    x, fs = A12.baca_dat(SIG["afpdb"] / nama, 0)
    x = np.nan_to_num(np.asarray(x, np.float64))
    nz, fsn = A12.baca_dat(SIG["nstdb"] / derau, 0)
    nz = hrv_core.resample_signal(np.nan_to_num(np.asarray(nz, np.float64)), fsn, fs)
    rng = np.random.default_rng(SEED + zlib.crc32(nama.encode()))
    if nz.size >= x.size:
        o = int(rng.integers(0, nz.size - x.size + 1))
        seg = nz[o:o + x.size]
    else:                                                    # tidak terjadi untuk afpdb (30 menit), dijaga saja
        seg = np.resize(np.roll(nz, -int(rng.integers(0, nz.size))), x.size)
    ps, pn = np.var(_bp(x, fs)), np.var(_bp(seg, fs))
    return x + seg * np.sqrt(ps / (pn * 10 ** (int(snr) / 10))), fs


def _baca_dat(base, kanal=0):
    base = Path(base)
    if "__" in base.name:
        return bising(base.name)
    return A12.baca_dat(base, kanal)


U164.A12.baca_dat = _baca_dat                              # proxy namespace dari celah.py; hanya U164 yang memakainya
U164.AKAR["afpdb_bising"] = SIG["afpdb"]


def sinyal(ds, rec):
    if ds == "afpdb_bising":
        return bising(rec)
    x, fs = A12.baca_dat(SIG[ds] / rec, 0)
    return np.nan_to_num(np.asarray(x, np.float64)), fs


# ── SQI ────────────────────────────────────────────────────────────────
def sqi_rekaman(ds, rec, start):
    import pembersih as PB
    x, fs = sinyal(ds, rec)
    x50 = hrv_core.resample_signal(x, fs, FS50)
    p = hrv_core.detect_rpeaks(x50, FS50)
    out = np.full(len(start), np.nan, np.float32)
    if p is None or len(p) < 3:
        return out
    _, kor, _, _ = PB.selaraskan(PB.bandpass(x50), p)
    t = np.asarray(p, float) / FS50
    lo = np.searchsorted(t, start, side="left")
    hi = np.searchsorted(t, start + hrv_core.WINDOW_SEC, side="left")
    for i, (a, b) in enumerate(zip(lo, hi)):
        k = kor[a:b]
        k = k[np.isfinite(k)]
        if k.size:
            out[i] = np.median(k)
    return out


def _sqi_job(job):
    ds, rec, start = job
    p = CACHE / "sqi" / ds / (rec.replace("/", "__") + ".npz")
    if p.exists():
        return rec
    s = sqi_rekaman(ds, rec, start)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p.with_suffix(".tmp.npz"), start=start, sqi=s)
    p.with_suffix(".tmp.npz").replace(p)
    return rec


def pasang_sqi(ds, d, workers=4, log=print):
    """Hitung (bila belum) dan pasang kolom 'sqi' sejajar dengan window d."""
    recs = list(dict.fromkeys(d["rec"].tolist()))
    jobs = []
    for r in recs:
        m = d["rec"] == r
        jobs.append((ds, r, np.unique(d["start"][m])))
    todo = [j for j in jobs if not (CACHE / "sqi" / ds / (j[1].replace("/", "__") + ".npz")).exists()]
    if todo:
        log(f"  SQI {ds}: {len(todo)} rekaman")
        with get_context("spawn").Pool(workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_sqi_job, todo, chunksize=1)):
                if (i + 1) % max(1, len(todo) // 5) == 0:
                    log(f"    {i + 1}/{len(todo)}")
    q = np.full(d["rec"].size, np.nan, np.float32)
    for r in recs:
        z = np.load(CACHE / "sqi" / ds / (r.replace("/", "__") + ".npz"))
        m = np.flatnonzero(d["rec"] == r)
        q[m] = z["sqi"][np.searchsorted(z["start"], d["start"][m])]
    d["sqi"] = q
    return d


# ── gerbang & metrik ───────────────────────────────────────────────────
def subset(d, m):
    n = d["rec"].size
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[-1] == n and k != "langkah":
            out[k] = v[..., m]
        else:
            out[k] = v
    return out


def gerbang(d, q):
    return d if q is None else subset(d, np.nan_to_num(d["sqi"], nan=-1.0) >= q)


def nilai(d, q, sistem="d1"):
    g = gerbang(d, q)
    m = C.nilai(g, g[sistem], 30, 0)
    m["frac_gerbang"] = 1.0 - g["rec"].size / max(d["rec"].size, 1)
    return m


def boot2(ua, ub, n_boot=2000, seed=2026):
    """Δ rasio (a − b), penyebut boleh beda: bootstrap unit berpasangan."""
    u = sorted(set(ua) | set(ub))
    A = np.array([ua.get(k, [0, 0]) for k in u], float)
    B = np.array([ub.get(k, [0, 0]) for k in u], float)
    idx = np.random.default_rng(seed).integers(0, len(u), (n_boot, len(u)))
    with np.errstate(invalid="ignore", divide="ignore"):
        ra = A[idx, 0].sum(1) / A[idx, 1].sum(1)
        rb = B[idx, 0].sum(1) / B[idx, 1].sum(1)
    dd = ra - rb
    return dict(delta=float(A[:, 0].sum() / A[:, 1].sum() - B[:, 0].sum() / B[:, 1].sum()), ci=np.nanpercentile(dd, [2.5, 97.5]).tolist(),
                p_kurang=float(np.nanmean(dd < 0)), p_lebih=float(np.nanmean(dd > 0)), ci_a=np.nanpercentile(ra, [2.5, 97.5]).tolist())


# ── data ───────────────────────────────────────────────────────────────
def data_dev(log=print):
    D = {"LTAFDB": E._ext("LTAFDB"), "MIT-BIH": E._ext("MIT-BIH"), "NSRDB": E._nsrdb()}
    for k in ("nstdb", "mitdb"):
        D[k], _ = K.muat(k)
    for k in list(D):
        D[k] = pasang_sqi(k, D[k], log=log)
    return D


def rekaman_bising():
    return [f"{n}__{dz}__{s:02d}" for dz in DERAU for s in SNR_VAL for n in K.C.rekaman("afpdb")]


def jalankan_bising(bagian=0, n_bagian=1, log=print):
    d = CACHE / "afpdb_bising"
    d.mkdir(parents=True, exist_ok=True)
    alat = None
    daftar = rekaman_bising()
    for j, rec in enumerate(daftar):
        if j % n_bagian != bagian:
            continue
        p = d / f"{rec}.npz"
        if p.exists():
            continue
        alat = alat or R163.Alat()
        z = U164.proses_rekaman("afpdb_bising", rec, alat)
        np.savez(p.with_suffix(".tmp.npz"), **z)
        p.with_suffix(".tmp.npz").replace(p)
        if (j + 1) % 50 == 0:
            log(f"  afpdb_bising: {j + 1}/{len(daftar)}")


def muat_bising():
    bag = {k: [] for k in ("rec", "start", "d1", "p31", "unit", "derau", "snr")}
    for rec in rekaman_bising():
        z = np.load(CACHE / "afpdb_bising" / f"{rec}.npz")
        if bool(z["kosong"]):
            continue
        n = z["start"].size
        nama, dz, s = rec.split("__")
        for k, v in (("rec", np.full(n, rec)), ("start", z["start"].astype(float)), ("d1", z["d1"]), ("p31", z["p31"]),
                     ("unit", np.full(n, nama)), ("derau", np.full(n, dz)), ("snr", np.full(n, int(s)))):
            bag[k].append(v)
    Z = {k: np.concatenate(v) for k, v in bag.items()}
    return dict(rec=Z["rec"], start=Z["start"], y2=np.zeros(Z["rec"].size, int), unit=Z["unit"], langkah=5.0, derau=Z["derau"],
                snr=Z["snr"], d1=Z["d1"] > R163.AMBANG, v31=(Z["p31"].argmax(1) != 0))


def simpan_json(obj, path):
    C.simpan_json(obj, path)
