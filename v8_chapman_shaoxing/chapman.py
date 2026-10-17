"""
chapman.py — logika v8 (data eksternal Chapman–Shaoxing & Ningbo, PhysioNet "ecg-arrhythmia" 1.0.0), dipakai notebook
dan laporan. Dirancang untuk dijalankan di Kaggle (GPU T4, Internet aktif), tetapi juga jalan di WSL/Colab.

Data: 45.152 pasien, satu EKG 12-lead 10 detik per pasien, 500 Hz, label ritme SNOMED CT di header (#Dx), CC BY 4.0.
Yang dipakai hanya lead II, diproses dengan front-end produksi (hrv_core.signal_to_windows: resample 50 Hz, deteksi R-peak,
cek kualitas) → satu window 10 s per pasien, sama seperti perangkat merekam 10 detik.

Pemetaan ke kelas SIHEDAF: Normal = SR, SB, ST, SA (irama sinus); AFIB = AFIB; AFL = AF (atrial flutter).
Ritme lain (SVT, AT, AVNRT, AVRT, WAVN) dan pasien dengan >1 kode ritme tidak dipakai untuk training/metrik 3 kelas,
tetapi prediksinya tetap dilaporkan.

Varian (semua MicroTransformer, fold AFDB resmi 5×5, seed latih 2026 + 100·r + k):
  v31          : referensi = resep v3/v3.1 (bobot kelas penuh), AFDB saja, dilatih ulang di lingkungan yang sama.
  afdb         : ablasi = bobot lunak (resep bawaan model baru, = v5b), AFDB saja.
  afdb_chapman : 8a (utama) = bobot lunak, AFDB + Chapman. Pasien Chapman juga dibagi 5 fold per ulangan.
  chapman      : 8b (eksploratif) = bobot lunak, Chapman saja; AFDB menjadi data uji eksternal murni.
"""
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "v7_augmentasi_takikardia"), str(REPO / "v3.2_uji_takikardia"),
                str(REPO / "v5_training_afl"), str(REPO / "v4_smoothing_ensemble"), str(REPO / "v3_evaluasi_robust"),
                str(REPO / "v2_revisi")]
import sihedaf_cv as cv  # noqa: E402
import hrv_core  # noqa: E402
import latih as L5  # noqa: E402
import takikardia as T  # noqa: E402
import augmentasi as A7  # noqa: E402

DATA_STD = REPO / "cache/dataset_v2_detector.npz"
OOF_DIR = REPO / "cache/oof_robust/detector_e30_r5_s2026"
FOLDS = OOF_DIR / "folds.npz"
CACHE_OUT = REPO / "cache/v8"
OOF_V8 = CACHE_OUT / "oof"
MODELS = CACHE_OUT / "model"
CH_DIR = Path(os.environ.get("SIHEDAF_CHAPMAN_DIR", REPO / "data/chapman/ecg-arrhythmia-1.0.0"))   # bisa diarahkan ke unduhan yang sudah ada
CH_WIN = CACHE_OUT / "chapman_windows.npz"
S3_BASE = "https://physionet-open.s3.amazonaws.com/ecg-arrhythmia/1.0.0/"     # mirror resmi PhysioNet (AWS Open Data)
N_REPEATS, SEED, N_BOOT = 5, 2026, 2000
LEAD = "II"

RHYTHM = {"426783006": "SR", "426177001": "SB", "427084000": "ST", "427393009": "SA", "164889003": "AFIB",
          "164890007": "AFL", "426761007": "SVT", "713422000": "AT", "233896004": "AVNRT", "233897008": "AVRT",
          "195101003": "WAVN"}
GROUP_CLASS = {"SR": 0, "SB": 0, "ST": 0, "SA": 0, "AFIB": 1, "AFL": 2}
CLASSES = ("Normal", "AFIB", "AFL")
ST_BANDS = [(100, 110), (110, 120), (120, 130), (130, 140), (140, 150), (150, 999)]

VARIANTS = {
    "v31": dict(label="referensi v3.1 (bobot penuh, AFDB saja)", power=1.0, afdb=True, chapman=False),
    "afdb": dict(label="ablasi: bobot lunak, AFDB saja (= v5b)", power=0.5, afdb=True, chapman=False),
    "afdb_chapman": dict(label="8a: bobot lunak, AFDB + Chapman", power=0.5, afdb=True, chapman=True),
    "chapman": dict(label="8b: bobot lunak, Chapman saja", power=0.5, afdb=False, chapman=True),
    # Ditambahkan setelah run pertama: label AFL bagian Ningbo terbukti tidak andal (afl_label_check). Hipotesis ditulis
    # sebelum hasilnya ada: alarm palsu ST turun seperti 8a, AFL di AFDB tidak lebih buruk dari v31.
    "afdb_chapman_bersih": dict(label="8c: bobot lunak, AFDB + Chapman tanpa label AFL", power=0.5, afdb=True,
                                chapman=True, chapman_classes=(0, 1)),
}
PRIMARY = "afdb_chapman"                 # ditetapkan sebelum melihat hasil
SHAOXING_MAX = 10646                     # JS00001–JS10646 = Chapman–Shaoxing; sisanya Ningbo (tanpa satu pun AFIB, "AFL"-nya
                                         # berperilaku seperti AFIB: detak ±88 bpm, 86% tidak teratur → label AFL tidak andal)

# Sidik jari dataset AFDB detektor & fold resmi dari PC tim (16 Oktober 2026). Dataset yang diekstrak ulang di mesin lain
# harus sama strukturnya (y, records, t_start); X boleh berbeda sedikit bila versi pustaka berbeda (dilaporkan).
AFDB_FINGERPRINT = {"n": 165978, "counts": [99000, 65826, 1152], "y": "3555740bcefa5aee", "records": "023e3250f2b6d06e",
                    "t_start": "c9788052eae7d50f", "X": "fcea1c422a58e6ec", "M": "4905d145d0bb51ab"}
AFDB_FOLDS = {"04015": [4, 3, 1, 3, 1], "04043": [3, 1, 4, 4, 4], "04048": [2, 4, 3, 0, 2], "04126": [0, 0, 1, 4, 0],
              "04746": [1, 3, 2, 2, 0], "04908": [1, 4, 3, 2, 0], "04936": [4, 3, 2, 4, 1], "05091": [2, 4, 2, 2, 0],
              "05121": [1, 4, 3, 0, 1], "05261": [3, 1, 4, 0, 2], "06426": [4, 0, 1, 1, 4], "06453": [4, 1, 2, 2, 4],
              "06995": [3, 2, 0, 3, 3], "07162": [2, 2, 0, 0, 4], "07859": [0, 1, 4, 1, 0], "07879": [0, 3, 0, 3, 2],
              "07910": [3, 2, 3, 3, 2], "08215": [0, 0, 0, 3, 3], "08219": [0, 2, 4, 1, 4], "08378": [2, 1, 2, 0, 3],
              "08405": [4, 0, 4, 1, 3], "08434": [1, 0, 3, 4, 3], "08455": [3, 2, 1, 4, 1]}


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ── 1. AFDB: dataset detektor & fold resmi ─────────────────────────────
def ensure_afdb(log=print):
    """Pakai cache/dataset_v2_detector.npz bila ada; jika tidak, unduh AFDB dan ekstrak ulang (ekstraktor v5 = v2)."""
    if not DATA_STD.exists():
        if not (L5.DATA_DIR / "RECORDS").exists():
            t0 = time.perf_counter()
            log("unduh AFDB (±600 MB) ...")
            subprocess.run([sys.executable, str(REPO / "tools/download_afdb.py")], check=True)
            log(f"AFDB terunduh ({(time.perf_counter() - t0) / 60:.1f} mnt)")
        log("ekstrak window detektor 50 Hz dari 23 rekaman AFDB ...")
        orig = L5._process_record                            # dibungkus hanya untuk log progres; logika v5 tidak diubah

        def logged(rec, stride):
            t = time.perf_counter()
            rows = orig(rec, stride)
            log(f"  {rec}: {len(rows)} window ({time.perf_counter() - t:.0f}s)")
            return rows
        L5._process_record = logged
        try:
            ds = L5.build_dataset({0: 5.0, 1: 5.0, 2: 5.0})
        finally:
            L5._process_record = orig
        DATA_STD.parent.mkdir(parents=True, exist_ok=True)
        tmp = DATA_STD.with_name(DATA_STD.stem + ".tmp.npz")
        np.savez_compressed(tmp, **ds)
        tmp.replace(DATA_STD)
    z = np.load(DATA_STD)
    rec = z["records"].astype(str)
    got = {"n": int(len(z["y"])), "counts": np.bincount(z["y"], minlength=3).tolist(), "y": _h(z["y"].astype(np.int64)),
           "records": hashlib.sha256("|".join(rec).encode()).hexdigest()[:16],
           "t_start": _h(z["t_start"].astype(np.float32)), "X": _h(z["X"].astype(np.float32)),
           "M": _h(z["M"].astype(np.float32))}
    for k in ("n", "counts", "y", "records", "t_start", "M"):
        assert got[k] == AFDB_FINGERPRINT[k], f"dataset AFDB berbeda dari PC tim pada '{k}': {got[k]} vs {AFDB_FINGERPRINT[k]}"
    return {"identik_bit_demi_bit": got["X"] == AFDB_FINGERPRINT["X"], **got}


def ensure_folds():
    """Fold resmi v3 (dibuat ulang bila belum ada), dibuktikan sama dengan fold PC tim."""
    z = np.load(DATA_STD)
    y, rec = z["y"], z["records"].astype(str)
    if not FOLDS.exists():
        FOLDS.parent.mkdir(parents=True, exist_ok=True)
        np.savez(FOLDS, folds=cv.repeated_patient_folds(y, rec, 5, N_REPEATS, SEED))
    folds = np.load(FOLDS)["folds"]
    for p, f in AFDB_FOLDS.items():
        assert [int(folds[r][rec == p][0]) for r in range(N_REPEATS)] == f, f"fold pasien {p} berbeda dari fold resmi"
    return folds


# ── 2. Chapman: unduh & ekstrak window ─────────────────────────────────
def _sha_list(session):
    """Daftar (path, sha256) resmi dari SHA256SUMS.txt PhysioNet."""
    r = session.get(S3_BASE + "SHA256SUMS.txt", timeout=60)
    r.raise_for_status()
    return [(q[1].strip(), q[0]) for q in (l.split(maxsplit=1) for l in r.text.splitlines()) if len(q) == 2]


def download_chapman(log=print, threads=48, max_tries=8):
    """
    Unduh dari mirror resmi PhysioNet di AWS S3 (±5 GB, ±90 ribu file .hea/.mat) secara paralel ke data/chapman/.
    Setiap file dicek dengan SHA256SUMS.txt resmi; file yang sudah ada dan cocok dilewati, jadi bisa dilanjutkan.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import requests
    flag = CH_DIR / ".lengkap"
    if flag.exists():
        return CH_DIR
    CH_DIR.mkdir(parents=True, exist_ok=True)
    local = threading.local()

    def sess():
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s

    files = [(q, h) for q, h in _sha_list(requests.Session())
             if q.endswith((".hea", ".mat")) or q == "ConditionNames_SNOMED-CT.csv"]

    def get(item):
        rel, sha = item
        dest = CH_DIR / rel
        if dest.exists():
            data = dest.read_bytes()
            if hashlib.sha256(data).hexdigest() == sha:
                return len(data), 0
        for attempt in range(max_tries):
            try:
                r = sess().get(S3_BASE + rel, timeout=60)
                r.raise_for_status()
                data = r.content
                if hashlib.sha256(data).hexdigest() != sha:
                    raise OSError("checksum tidak cocok")
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
                return len(data), 1
            except (requests.RequestException, OSError) as e:
                if attempt == max_tries - 1:
                    raise RuntimeError(f"gagal mengunduh {rel}: {e}") from e
                time.sleep(2 * (attempt + 1))

    t0, nbytes, nnew = time.perf_counter(), 0, 0
    log(f"{len(files)} file dari {S3_BASE} ({threads} thread) ...")
    with ThreadPoolExecutor(threads) as ex:
        for i, (n, new) in enumerate(ex.map(get, files)):
            nbytes += n; nnew += new
            if (i + 1) % 10000 == 0 or i + 1 == len(files):
                dt = time.perf_counter() - t0
                log(f"  {i + 1}/{len(files)} file, {nbytes / 1e9:.2f} GB, {nbytes / 1e6 / max(dt, 1e-9):.0f} MB/s, "
                    f"{dt / 60:.1f} mnt")
    flag.write_text(json.dumps({"files": len(files), "bytes": nbytes, "baru": nnew}))
    log(f"Chapman siap: {nbytes / 1e9:.2f} GB, {nnew} file baru, {(time.perf_counter() - t0) / 60:.1f} mnt")
    return CH_DIR


def parse_header(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    head = lines[0].split()
    n_sig, fs = int(head[1]), float(head[2])
    leads, gain, base = [], [], []
    for l in lines[1:1 + n_sig]:
        f = l.split()
        g = f[2].split("/")[0]
        if "(" in g:                                         # "gain(baseline)"
            g, b = g.rstrip(")").split("(")
            base.append(float(b))
        else:
            base.append(float(f[4]))                         # baseline = adc_zero bila tidak ditulis
        gain.append(float(g) or 200.0)
        leads.append(f[-1])
    meta = {}
    for l in lines[1 + n_sig:]:
        if l.startswith("#") and ":" in l:
            k, v = l[1:].split(":", 1)
            meta[k.strip()] = v.strip()
    dx = [c.strip() for c in meta.get("Dx", "").split(",") if c.strip()]
    return {"name": head[0], "fs": fs, "leads": leads, "gain": gain, "baseline": base, "dx": dx,
            "age": meta.get("Age", ""), "sex": meta.get("Sex", "")}


def _process_record(hea_name):
    """Satu pasien → dict (ritme, alasan, IBI window 10 s dari lead II lewat front-end produksi)."""
    from scipy.io import loadmat
    rec = Path(hea_name).stem
    out = {"record": rec, "rhythm": "", "codes": "", "age": "", "sex": "", "ok": False, "reason": "", "ibi": None}
    try:
        h = parse_header((CH_DIR / hea_name).read_text(encoding="utf-8", errors="replace"))
        rh = sorted({RHYTHM[c] for c in h["dx"] if c in RHYTHM})
        out.update(rhythm="+".join(rh), codes=",".join(h["dx"]), age=h["age"], sex=h["sex"])
        if len(rh) != 1:
            out["reason"] = "tanpa kode ritme" if not rh else "lebih dari satu kode ritme"
            return out
        val = loadmat(str(CH_DIR / (hea_name[:-4] + ".mat")))["val"]
        i = h["leads"].index(LEAD)
        sig = (val[i].astype(np.float64) - h["baseline"][i]) / h["gain"][i]
        wins, _ = hrv_core.signal_to_windows(sig, h["fs"], stride_sec=5.0)
        if not wins:
            out["reason"] = "sinyal < 10 s"
            return out
        w = wins[0]
        out.update(ok=bool(w["ok"]), reason="ok" if w["ok"] else w["reason"], ibi=np.asarray(w["ibi_ms"], np.float32))
    except Exception as e:  # dicatat per pasien dan dilaporkan, bukan diabaikan
        out["reason"] = f"error {type(e).__name__}: {str(e)[:80]}"
    return out


def build_chapman_windows(log=print, workers=None):
    """Ekstrak satu window per pasien (lead II) → cache/v8/chapman_windows.npz. Dilewati bila sudah ada."""
    if CH_WIN.exists():
        return chapman_summary()
    names = sorted(p.relative_to(CH_DIR).as_posix() for p in (CH_DIR / "WFDBRecords").rglob("*.hea"))
    assert len(names) > 40000, f"hanya {len(names)} header di {CH_DIR}; jalankan download_chapman() dulu"
    workers = workers or max(1, (os.cpu_count() or 2))
    log(f"{len(names)} header, {workers} proses ...")
    t0 = time.perf_counter()
    rows = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for b0 in range(0, len(names), 2500):                # per batch: progres teratur + gagal cepat bila pool macet
            batch = names[b0:b0 + 2500]
            try:
                rows += pool.map_async(_process_record, batch, chunksize=32).get(timeout=900)
            except mp.TimeoutError:
                raise RuntimeError(f"pool ekstraksi macet: batch {b0}–{b0 + len(batch)} tidak selesai dalam 15 menit")
            dt = time.perf_counter() - t0
            log(f"  {len(rows)}/{len(names)} ({dt / 60:.1f} mnt, sisa ±{dt / len(rows) * (len(names) - len(rows)) / 60:.1f} mnt)")
    X = np.zeros((len(rows), hrv_core.MAX_BEATS, 2), np.float32)
    M = np.zeros((len(rows), hrv_core.MAX_BEATS), np.float32)
    hr, cvv, nb = np.full(len(rows), np.nan), np.full(len(rows), np.nan), np.zeros(len(rows), np.int16)
    for j, r in enumerate(rows):
        if r["ibi"] is not None and r["ibi"].size:
            X[j], M[j] = hrv_core.build_sequence(r["ibi"])
            hr[j], cvv[j], nb[j] = 60000.0 / r["ibi"].mean(), r["ibi"].std() / r["ibi"].mean(), r["ibi"].size
    rhythm = np.array([r["rhythm"] for r in rows])
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    tmp = CH_WIN.with_name(CH_WIN.stem + ".tmp.npz")
    np.savez_compressed(tmp, X=X, M=M, hr=hr, cv=cvv, n_ibi=nb, rhythm=rhythm,
                        cls=np.array([GROUP_CLASS.get(r, -1) for r in rhythm], np.int64),
                        ok=np.array([r["ok"] for r in rows]), reason=np.array([r["reason"] for r in rows]),
                        record=np.array([r["record"] for r in rows]), codes=np.array([r["codes"] for r in rows]),
                        age=np.array([r["age"] for r in rows]), sex=np.array([r["sex"] for r in rows]))
    tmp.replace(CH_WIN)
    log(f"selesai {(time.perf_counter() - t0) / 60:.1f} mnt")
    return chapman_summary()


def is_ningbo(record):
    return np.array([int(r[2:]) > SHAOXING_MAX for r in np.atleast_1d(record)])


def afl_label_check():
    """Detak & keteraturan per ritme dan sumber, dibandingkan dengan AFDB: apakah label AFL masuk akal?"""
    ch = load_chapman()
    nb = is_ningbo(ch["record"])
    rows = {}
    for src, m_src in (("Shaoxing", ~nb), ("Ningbo", nb)):
        for g in ("SR", "ST", "AFIB", "AFL"):
            m = m_src & (ch["rhythm"] == g)
            rows[f"{g} ({src})"] = {"pasien": int(m.sum()),
                                    "detak median": float(np.median(ch["hr"][m])) if m.any() else None,
                                    "CV median": float(np.median(ch["cv"][m])) if m.any() else None,
                                    "CV > 0,10": float(np.mean(ch["cv"][m] > 0.10)) if m.any() else None}
    z = np.load(DATA_STD)
    Mz, Iz = z["M"], z["X"][:, :, 0]
    mean = (Iz * Mz).sum(1) / Mz.sum(1)
    hr = 60000.0 / mean
    cvv = np.sqrt((((Iz - mean[:, None]) ** 2) * Mz).sum(1) / Mz.sum(1)) / mean
    for c, g in ((1, "AFIB (AFDB)"), (2, "AFL (AFDB)")):
        m = z["y"] == c
        rows[g] = {"pasien": int(len(np.unique(z["records"][m]))), "detak median": float(np.median(hr[m])),
                   "CV median": float(np.median(cvv[m])), "CV > 0,10": float(np.mean(cvv[m] > 0.10))}
    return rows


def chapman_summary():
    z = np.load(CH_WIN)
    reason = np.array([r if not r.startswith("error") else "error" for r in z["reason"]])
    rh, ok = z["rhythm"], z["ok"]
    groups = sorted(set(rh[rh != ""]), key=lambda g: (g not in GROUP_CLASS, g))
    per = {g: {"pasien": int((rh == g).sum()), "window valid": int(((rh == g) & ok).sum()),
               "kelas": CLASSES[GROUP_CLASS[g]] if g in GROUP_CLASS else "tidak dipakai",
               "detak median": float(np.nanmedian(z["hr"][(rh == g) & ok])) if ((rh == g) & ok).any() else None}
           for g in groups}
    return {"pasien": int(len(rh)), "pasien unik": int(len(np.unique(z["record"]))), "window valid": int(ok.sum()),
            "alasan": {k: int(v) for k, v in zip(*np.unique(reason, return_counts=True))}, "per ritme": per}


def load_chapman():
    """Window Chapman yang lolos cek kualitas (semua ritme, termasuk yang tidak dipakai untuk training)."""
    z = np.load(CH_WIN)
    m = z["ok"]
    return {k: z[k][m] for k in ("X", "M", "hr", "cv", "rhythm", "cls", "record")}


def chapman_folds(rhythm):
    """(R, N) fold per pasien Chapman, diacak dalam tiap ritme lalu dibagikan round-robin (seperti fold AFDB)."""
    rng = np.random.default_rng(SEED + 1000)
    folds = np.empty((N_REPEATS, len(rhythm)), np.int8)
    for r in range(N_REPEATS):
        for g in np.unique(rhythm):
            idx = rng.permutation(np.flatnonzero(rhythm == g))
            folds[r, idx] = (np.arange(idx.size) + rng.integers(5)) % 5
    return folds


# ── 3. training ────────────────────────────────────────────────────────
def job_path(variant, r):
    return OOF_V8 / f"{variant}_r{r}.npz"


def parity_check(device, n_epochs=2):
    """latih.train_dl_weighted(bobot pangkat 1) harus identik dengan sihedaf_cv.train_dl (resep v3) pada seed sama."""
    z = np.load(DATA_STD)
    tr = np.arange(len(z["y"])) % 5 != 0
    te = np.flatnonzero(~tr)[:5000]
    saved = L5.HP["epochs"]
    L5.HP["epochs"] = n_epochs
    try:
        a = L5.train_dl_weighted(z["X"][tr], z["M"][tr], z["y"][tr], weight_power=1.0, seed=7, device=device)
    finally:
        L5.HP["epochs"] = saved
    b = cv.train_dl("MicroTransformer", L5.MODEL_KWARGS, z["X"][tr], z["M"][tr], z["y"][tr], epochs=n_epochs,
                    batch_size=L5.HP["batch_size"], lr=L5.HP["lr"], weight_decay=L5.HP["weight_decay"], seed=7, device=device)
    pa = cv.predict_dl(a[0], z["X"][te], z["M"][te], a[1], a[2], device)
    pb = cv.predict_dl(b[0], z["X"][te], z["M"][te], b[1], b[2], device)
    return float(np.abs(pa - pb).max())


_GPU = 0


def _init_gpu(queue):
    """Initializer proses pekerja: ambil satu nomor GPU (mode multi-GPU, satu proses per GPU)."""
    global _GPU
    _GPU = queue.get()


def run_job(job):
    """Satu (varian, ulangan): latih 5 fold, prediksi fold uji AFDB + Chapman + uji stres, simpan bobot model."""
    import torch
    variant, r = job
    out = job_path(variant, r)
    if out.exists():
        return variant, r, None
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{_GPU}" if torch.cuda.is_available() else "cpu")
    cfg = VARIANTS[variant]
    z = np.load(DATA_STD)
    X, M, y = z["X"], z["M"], z["y"]
    fa = np.load(FOLDS)["folds"][r]
    ch = load_chapman()
    fc = chapman_folds(ch["rhythm"])[r]
    use_ch = np.isin(ch["cls"], cfg.get("chapman_classes", (0, 1, 2)))
    Xs, Ms, _ = A7.stress_set()
    oof_a = np.full((len(y), 3), np.nan, np.float32)
    oof_c = np.full((len(ch["cls"]), 3), np.nan, np.float32)
    stress = np.zeros((5, len(Xs), 3), np.float32)
    times, n_train = [], []
    MODELS.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        parts = []
        if cfg["afdb"]:
            parts.append((X[fa != k], M[fa != k], y[fa != k]))
        if cfg["chapman"]:
            m = use_ch & (fc != k)
            parts.append((ch["X"][m], ch["M"][m], ch["cls"][m]))
        Xtr, Mtr, ytr = (np.concatenate(p) for p in zip(*parts))
        net, mu, sd, t = L5.train_dl_weighted(Xtr, Mtr, ytr, weight_power=cfg["power"], seed=SEED + 100 * r + k,
                                              device=device)
        oof_a[fa == k] = cv.predict_dl(net, X[fa == k], M[fa == k], mu, sd, device)
        oof_c[fc == k] = cv.predict_dl(net, ch["X"][fc == k], ch["M"][fc == k], mu, sd, device)
        stress[k] = cv.predict_dl(net, Xs, Ms, mu, sd, device)
        torch.save({"state_dict": {a: b.cpu() for a, b in net.state_dict().items()}, "mu": torch.from_numpy(mu),
                    "sd": torch.from_numpy(sd), "model_kwargs": L5.MODEL_KWARGS, "variant": variant, "repeat": r,
                    "fold": k}, MODELS / f"{variant}_r{r}_k{k}.pt")
        times.append(t); n_train.append(np.bincount(ytr, minlength=3))
    assert not np.isnan(oof_a).any() and not np.isnan(oof_c).any()
    OOF_V8.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, oof_afdb=oof_a, oof_chapman=oof_c, stress=stress, times=np.asarray(times), n_train=np.stack(n_train))
    tmp.replace(out)
    return variant, r, float(np.mean(times))


def all_jobs():
    return [(v, r) for r in range(N_REPEATS) for v in VARIANTS]


def run_all_jobs(log=print, n_gpu=None):
    """
    Semua job, urut per ulangan supaya hasil awal cepat ada. Satu proses per GPU (Kaggle T4 x2 → 2 job bersamaan);
    job saling lepas dan seed-nya tetap, jadi hasilnya sama dengan menjalankan berurutan. Bisa dilanjutkan.
    """
    import torch
    n_gpu = n_gpu or max(1, torch.cuda.device_count())
    jobs = [j for j in all_jobs() if not job_path(*j).exists()]
    log(f"{len(jobs)} job tersisa dari {len(all_jobs())}, {n_gpu} proses (satu per GPU)")
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for i in range(n_gpu):
        queue.put(i)
    with ProcessPoolExecutor(n_gpu, mp_context=ctx, initializer=_init_gpu, initargs=(queue,)) as ex:
        for v, r, t in ex.map(run_job, jobs):
            log(f"  [{(time.perf_counter() - t0) / 60:5.1f} mnt] {v:<13} ulangan {r}: {t:.0f} s/fold")


# ── 4. evaluasi ────────────────────────────────────────────────────────
def _cm_metrics(cm):
    """cm (..., 3, 3), baris = kelas asli → metrik per kelas & Macro-F1."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    rec = tp / np.maximum(cm.sum(-1), 1e-12)
    prec = tp / np.maximum(cm.sum(-2), 1e-12)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return {"Macro-F1": f1.mean(-1), "AFL F1": f1[..., 2], "AFL R": rec[..., 2], "AFL P": prec[..., 2],
            "AFIB F1": f1[..., 1], "AFIB R": rec[..., 1], "Normal R": rec[..., 0]}


def _boot_counts(groups, n_boot, seed, chunk=100):
    """Bootstrap pasien bertingkat: untuk tiap kelompok, berapa kali tiap pasien terambil (dibangkitkan per potongan)."""
    rng = np.random.default_rng(seed)
    for b0 in range(0, n_boot, chunk):
        b1 = min(n_boot, b0 + chunk)
        yield b0, b1, [(idx, rng.multinomial(idx.size, np.full(idx.size, 1.0 / idx.size), size=b1 - b0).astype(np.float32))
                       for idx in groups]


def chapman_boot(contrib, cls, n_boot=N_BOOT, seed=SEED):
    """contrib {model: (N, 3) proporsi tebakan per pasien}; cls kelas asli → {model: (B, 3, 3)} CM bootstrap."""
    groups = [np.flatnonzero(cls == c) for c in range(3)]
    out = {k: np.zeros((n_boot, 3, 3)) for k in contrib}
    for b0, b1, cnts in _boot_counts(groups, n_boot, seed):
        for c, (idx, cnt) in enumerate(cnts):
            for k, v in contrib.items():
                out[k][b0:b1, c, :] = cnt @ v[idx]
    return out


def boot_mean(values, n_boot=N_BOOT, seed=SEED):
    """values {model: (n,)} untuk satu kelompok pasien → (titik, CI 2,5–97,5) rata-rata, sampel bootstrap sama."""
    n = len(next(iter(values.values())))
    if n == 0:
        return {k: [float("nan")] * 3 for k in values}
    acc = {k: np.zeros(n_boot) for k in values}
    for b0, b1, cnts in _boot_counts([np.arange(n)], n_boot, seed):
        cnt = cnts[0][1]
        for k, v in values.items():
            acc[k][b0:b1] = cnt @ np.asarray(v, np.float32) / n
    return {k: [float(np.mean(values[k])), *map(float, np.percentile(a, [2.5, 97.5]))] for k, a in acc.items()}


def _onehot(pred):
    return np.eye(3, dtype=np.float32)[pred]


def load_packages():
    """Paket yang ada di repo: v3.1 dan v2 (wajib), v1 final.joblib (bila gagal dimuat, alasannya dilaporkan)."""
    import importlib.util
    from predictor import ECGPredictor
    pk = {"v31_paket": ECGPredictor(str(REPO / "v3.1_kalibrasi_produksi/model_export"), name="sihedaf_v3_1", device="cpu"),
          "v2_paket": ECGPredictor(str(REPO / "v2_revisi/model_export"), device="cpu")}
    run = {k: (lambda P: lambda X, M: P.predict_proba_sequences(X, M).argmax(1))(P) for k, P in pk.items()}
    note = {}
    try:
        spec = importlib.util.spec_from_file_location("predictor_v1", REPO / "v1_lama/predictor.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        v1 = mod.ECGPredictor(str(REPO / "v1_lama/final.joblib"), device="cpu")
        run["v1_paket"] = lambda X, M: np.array([r["class_id"] for r in v1.predict(X, M)])
    except Exception as e:
        note["v1_paket"] = f"tidak bisa dimuat: {type(e).__name__}: {e}"
    return run, note


def evaluate():
    import pascaproses as P4
    z = np.load(DATA_STD)
    y, rec = z["y"], z["records"].astype(str)
    t = z["t_start"].astype(np.float64)
    starts = np.flatnonzero(np.r_[True, rec[1:] != rec[:-1]])
    patients = np.unique(rec)
    d = SimpleNamespace(y=y, records=rec, t=t, patients=patients, bounds=list(zip(starts, np.r_[starts[1:], len(y)])),
                        strata=cv.patient_strata(y, rec, patients), folds=np.load(FOLDS)["folds"])
    Mz, Iz = z["M"], z["X"][:, :, 0]
    hr_a = 60000.0 / ((Iz * Mz).sum(1) / Mz.sum(1))
    E = {v: [dict(np.load(job_path(v, r))) for r in range(N_REPEATS)] for v in VARIANTS}   # baca sekali ke memori

    # ── AFDB (protokol v3): bias cross-fit, Δ berpasangan vs v31 (dilatih ulang) dan vs ablasi ──
    pred, fold_bias = {}, {}
    for v in VARIANTS:
        adj = [cv.crossfit_bias(y, E[v][r]["oof_afdb"], d.folds[r]) for r in range(N_REPEATS)]
        pred[v] = np.stack([a.argmax(1) for a, _ in adj])
        fold_bias[v] = [b for _, b in adj]
    v1_oof = OOF_DIR / "versi_lama"
    if (v1_oof / "oof_r0.npz").exists():                    # hanya ada di cache PC tim (bukan di Kaggle)
        pred["v1"] = np.stack([np.load(v1_oof / f"oof_r{r}.npz")["det"].argmax(-1) for r in range(N_REPEATS)])
    ev = P4.evaluate(d, pred)
    W = cv.bootstrap_weights(d.strata, N_BOOT, seed=SEED)
    cms = {k: np.stack([cv.patient_cms(y, p, rec, patients) for p in pred[k]]) for k in VARIANTS}
    boot = {k: cv.boot_metrics(c, W) for k, c in cms.items()}
    point = {k: cv.point_metrics(c) for k, c in cms.items()}
    pairs_ablasi = {v: {m: [float(point[v][m].mean() - point["afdb"][m].mean()),
                            *map(float, np.percentile(boot[v][m] - boot["afdb"][m], [2.5, 97.5])),
                            float(np.mean(boot[v][m] - boot["afdb"][m] > 0))]
                        for m in ("Macro-F1", "AFL F1", "AFL R", "Normal R")} for v in VARIANTS if v not in ("v31", "afdb")}
    fp = {}
    for k, v in pred.items():
        c = np.stack([cv.patient_cms(y, p, rec, patients) for p in v])
        f = c[:, :, :2, 2].sum(-1).mean(0)
        fp[k] = {"total": float(f.sum()), "pasien_lain": float(f[d.strata != 2].sum())}
    dd = dict(y=y, rec=rec, hr=hr_a)
    by_hr = T.by_hr_bin(dd, {k: pred[k] for k in VARIANTS})
    sens = {}
    for lab, mask in (("semua pasien", np.ones(len(y), bool)), ("tanpa 04936", rec != T.SUSPECT)):
        pr = {v: np.stack([cv.crossfit_bias(y[mask], E[v][r]["oof_afdb"][mask], d.folds[r][mask])[0].argmax(1)
                           for r in range(N_REPEATS)]) for v in VARIANTS}
        sens[lab] = T.metrics_with_ci(dd, pr, mask)

    # ── Chapman (data eksternal): bias dari AFDB per model fold (seperti produksi) dan tanpa bias ──
    ch = load_chapman()
    fc = chapman_folds(ch["rhythm"])
    contrib, contrib_raw = {}, {}
    for v in VARIANTS:
        pb, pr0 = [], []
        for r in range(N_REPEATS):
            o = E[v][r]["oof_chapman"]
            p = np.empty(len(o), np.int64)
            for k in range(5):
                m = fc[r] == k
                p[m] = cv.apply_bias(o[m], fold_bias[v][r][k]).argmax(1)
            pb.append(_onehot(p)); pr0.append(_onehot(o.argmax(1)))
        contrib[v], contrib_raw[v] = np.mean(pb, 0), np.mean(pr0, 0)
    runners, pk_note = load_packages()
    for k, f in runners.items():
        contrib[k] = _onehot(np.concatenate([f(ch["X"][i:i + 4096], ch["M"][i:i + 4096])
                                             for i in range(0, len(ch["X"]), 4096)]))
    ningbo = is_ningbo(ch["record"])
    m_all = ch["cls"] >= 0
    m3 = m_all & ~((ch["rhythm"] == "AFL") & ningbo)       # label andal: AFL Ningbo dikeluarkan dari metrik 3 kelas
    cls3 = ch["cls"][m3]
    cm_pt = {k: np.stack([v[m3][cls3 == c].sum(0) for c in range(3)]) for k, v in contrib.items()}
    cm_pt_raw = {k: np.stack([v[m3][cls3 == c].sum(0) for c in range(3)]) for k, v in contrib_raw.items()}
    bt = chapman_boot({k: v[m3] for k, v in contrib.items()}, cls3)
    mets = {k: {m: [float(_cm_metrics(cm_pt[k])[m]), *map(float, np.percentile(_cm_metrics(bt[k])[m], [2.5, 97.5]))]
                for m in ("Macro-F1", "AFL F1", "AFL R", "AFL P", "AFIB F1", "AFIB R", "Normal R")} for k in contrib}
    mets_raw = {k: {m: float(_cm_metrics(cm_pt_raw[k])[m]) for m in ("Macro-F1", "AFL F1", "AFL R", "Normal R")}
                for k in contrib_raw}
    cls_all = ch["cls"][m_all]
    mets_all_labels = {k: {m: float(_cm_metrics(np.stack([v[m_all][cls_all == c].sum(0) for c in range(3)]))[m])
                           for m in ("Macro-F1", "AFL F1", "AFL R", "AFIB R")} for k, v in contrib.items()}
    ch_pairs = {}
    for ref in ("v31", "v31_paket"):
        for k in contrib:
            if k == ref:
                continue
            dm = _cm_metrics(bt[k])["Macro-F1"] - _cm_metrics(bt[ref])["Macro-F1"]
            da = _cm_metrics(bt[k])["AFL R"] - _cm_metrics(bt[ref])["AFL R"]
            ch_pairs.setdefault(ref, {})[k] = {
                "Macro-F1": [float(_cm_metrics(cm_pt[k])["Macro-F1"] - _cm_metrics(cm_pt[ref])["Macro-F1"]),
                             *map(float, np.percentile(dm, [2.5, 97.5])), float(np.mean(dm > 0))],
                "AFL R": [float(_cm_metrics(cm_pt[k])["AFL R"] - _cm_metrics(cm_pt[ref])["AFL R"]),
                          *map(float, np.percentile(da, [2.5, 97.5])), float(np.mean(da > 0))]}
    groups = {}
    for g in sorted(set(ch["rhythm"]), key=lambda g: (g not in GROUP_CLASS, g)):
        if g == "AFL":
            groups["AFL (Shaoxing)"] = (ch["rhythm"] == g) & ~ningbo
            groups["AFL (Ningbo, label tidak andal)"] = (ch["rhythm"] == g) & ningbo
        else:
            groups[g] = ch["rhythm"] == g
    rhythm_tab = {g: {"n": int(m.sum()), **{k: v[m].mean(0).tolist() for k, v in contrib.items()}} for g, m in groups.items()}
    st = ch["rhythm"] == "ST"
    alarm = {k: v[:, 1] + v[:, 2] for k, v in contrib.items()}
    st_bands = {f"{lo}–{hi if hi < 999 else '∞'}": {"n": int((st & (ch["hr"] >= lo) & (ch["hr"] < hi)).sum()),
                                                    **boot_mean({k: a[st & (ch["hr"] >= lo) & (ch["hr"] < hi)]
                                                                 for k, a in alarm.items()})}
                for lo, hi in ST_BANDS}
    st_all = {"n": int(st.sum()), **boot_mean({k: a[st] for k, a in alarm.items()})}
    sinus_slow = np.isin(ch["rhythm"], ["SR", "SB", "SA"])
    sinus_alarm = {"n": int(sinus_slow.sum()), **boot_mean({k: a[sinus_slow] for k, a in alarm.items()})}
    afl = (ch["rhythm"] == "AFL") & ~ningbo                 # hanya AFL Shaoxing (label andal)
    afl_bands = {f"{lo}–{hi if hi < 999 else '∞'}": {"n": int((afl & (ch["hr"] >= lo) & (ch["hr"] < hi)).sum()),
                                                     **{k: float(v[afl & (ch["hr"] >= lo) & (ch["hr"] < hi), 2].mean())
                                                        if (afl & (ch["hr"] >= lo) & (ch["hr"] < hi)).any() else None
                                                        for k, v in contrib.items()}}
                 for lo, hi in [(0, 100), (100, 130), (130, 160), (160, 999)]}

    # ── uji stres sintetis (protokol butir 9) ──
    _, _, meta = A7.stress_set()
    mj = np.array([m[0] for m in meta]); mb = np.array([m[1] for m in meta])
    stress = {}
    for v in VARIANTS:
        cls_s = np.stack([cv.apply_bias(E[v][r]["stress"][k], fold_bias[v][r][k]).argmax(1)
                          for r in range(N_REPEATS) for k in range(5)])
        stress[v] = {j: [(np.bincount(cls_s[:, (mj == j) & (mb == b)].ravel(), minlength=3) / cls_s[:, (mj == j) & (mb == b)].size).tolist()
                         for b in T.BPMS] for j in T.JITTERS}
    seq_run = {k: (lambda f: lambda seqs: f(np.stack([s[0] for s in seqs]), np.stack([s[1] for s in seqs])))(f)
               for k, f in runners.items()}
    stress.update({k: {j: np.asarray(tab).tolist() for j, tab in tabs.items()} for k, tabs in T.stress_test(seq_run).items()})
    threshold = {k: {j: T.normal_threshold(np.array(tab)) for j, tab in s.items()} for k, s in stress.items()}

    return {
        "labels": {**{v: c["label"] for v, c in VARIANTS.items()}, "v31_paket": "paket produksi v3.1",
                   "v2_paket": "paket v2", "v1_paket": "paket v1 (final.joblib)", "v1": "v1 (OOF cache PC tim)"},
        "primary": PRIMARY,
        "afdb": {"rows": ev.rows, "episodes": ev.epi, "pairs": ev.pairs, "pairs_ablasi": pairs_ablasi,
                 "afl_patients": ev.afl_patients, "afl_recall": ev.afl_recall, "fp": fp, "by_hr": by_hr, "sens": sens,
                 "bias": {k: {"median": np.median(np.concatenate(b), 0).tolist(), "min": np.concatenate(b).min(0).tolist(),
                              "max": np.concatenate(b).max(0).tolist()} for k, b in fold_bias.items()}},
        "chapman": {"summary": chapman_summary(), "afl_label_check": afl_label_check(), "metrics": mets,
                    "metrics_tanpa_bias": mets_raw, "metrics_semua_label": mets_all_labels, "pairs": ch_pairs,
                    "n_metrik": {"Normal": int((cls3 == 0).sum()), "AFIB": int((cls3 == 1).sum()), "AFL": int((cls3 == 2).sum())},
                    "rhythm": rhythm_tab, "st_bands": st_bands, "st_all": st_all, "sinus_alarm": sinus_alarm,
                    "afl_bands": afl_bands, "paket_catatan": pk_note},
        "stress": stress, "threshold": threshold, "bpms": T.BPMS, "jitters": list(T.JITTERS),
        "train_time": {v: float(np.mean([e["times"].mean() for e in E[v]])) for v in VARIANTS},
        "n_train": {v: E[v][0]["n_train"].mean(0).tolist() for v in VARIANTS},
    }


def pack_results(dest):
    """Kumpulkan keluaran v8 (ringkasan, OOF, window Chapman, bobot model) ke satu zip untuk dibawa pulang."""
    dest = Path(dest)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(CACHE_OUT.rglob("*")):
            if p.is_file() and not p.name.endswith((".tmp.npz", ".zip")):
                zf.write(p, Path("cache/v8") / p.relative_to(CACHE_OUT))     # diekstrak di root repo → cache/v8/
        zf.write(FOLDS, "cache/oof_robust/detector_e30_r5_s2026/folds.npz")
    return dest, dest.stat().st_size / 1e6
