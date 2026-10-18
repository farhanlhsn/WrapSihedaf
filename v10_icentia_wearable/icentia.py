"""
icentia.py — logika v10 (data wearable Icentia11k + AFDB + Chapman, cabang bentuk gelombang EKG v6/v9).

Masalah (v9): CNN bentuk gelombang yang dilatih dengan AFDB + Chapman memisahkan sinus takikardia dari AFL di Chapman (EKG klinis
istirahat), tetapi gagal di AFDB (Holter): AFL palsu melonjak, 08378 hilang, bias kelas melompat. Batasnya pergeseran domain.
Icentia11k (PhysioNet "icentia11k-continuous-ecg" 1.0) berasal dari patch EKG satu lead yang dipakai sehari-hari (CardioSTAT,
250 Hz, ±70 menit per segmen), 11.000 pasien. Domainnya dekat dengan perangkat SIHEDAF (wearable satu lead), berisi aktivitas
harian (sinus takikardia asli saat bergerak) dan ratusan pasien AFL. Lisensi CC BY-NC-SA 4.0: hanya untuk akademik/riset;
model yang dilatih dengan data ini tidak boleh dipakai komersial tanpa izin.

Alur data (dijalankan sekali, di Kaggle CPU; hasil disimpan ke cache/v10/):
  1. scan_icentia      : baca SEMUA file anotasi (.atr, ±23 GB, tidak disimpan) → durasi tiap ritme per segmen.
  2. select_segments   : pilih pasien & segmen (aturan tetap, seed tetap; ditulis sebelum melihat hasil model).
  3. build_icentia_windows : unduh segmen terpilih (dicek SHA256), front-end produksi (50 Hz, detektor R-peak v2) → window
                         10 s interval RR + potongan EKG v6 yang sejajar. Label window = ritme anotasi (semua detak sama).

Varian (ditetapkan sebelum melihat hasil; resep training v9 tanpa perubahan):
  rr               : kontrol — MicroTransformer interval RR (resep v5b), AFDB + Chapman andal + Icentia
  dual             : 10a (utama) — CNN EKG + RR (DualBranch v6), AFDB + Chapman andal + Icentia
  dual_tanpa_afdb  : 10b (eksploratif) — DualBranch, Chapman andal + Icentia saja (1 ulangan × 5 fold); AFDB menjadi domain
                     yang tidak pernah dilihat, seperti perangkat SIHEDAF nanti
Pembanding: v31 (v8), 9a dual & kontrol rr v9 (cache v9, bobot modelnya juga dipakai untuk memprediksi Icentia), paket v3.1.

SIHEDAF_QUICK=1: uji alur cepat (1 ulangan, 1 epoch, ±5.000 window latih per fold), hanya untuk memeriksa kode.
"""
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "v9_gelombang_ekg")]
import gelombang as G9  # noqa: E402  (juga memasang path v2–v8)

C8, E6, cv, hrv_core, L5, T = G9.C8, G9.E6, G9.cv, G9.hrv_core, G9.L5, G9.T

QUICK = os.environ.get("SIHEDAF_QUICK") == "1"
CACHE_OUT = REPO / ("cache/v10_quick" if QUICK else "cache/v10")     # uji cepat tidak tercampur dengan hasil sungguhan
OOF_V10 = CACHE_OUT / "oof"
MODELS = CACHE_OUT / "model"
IC_DIR = REPO / "cache/v10"                                            # data Icentia, dipakai bersama (juga oleh uji cepat)
IC_SCAN = IC_DIR / "icentia_scan.npz"
IC_SEL = IC_DIR / "icentia_pilihan.json"
IC_WIN = IC_DIR / "icentia_windows.npz"
V8_OOF, V8_MODEL = REPO / "cache/v8/oof", REPO / "cache/v8/model"
V9_OOF, V9_MODEL = REPO / "cache/v9/oof", REPO / "cache/v9/model"
S3_IC = "https://physionet-open.s3.amazonaws.com/icentia11k-continuous-ecg/1.0/"   # mirror resmi PhysioNet (AWS Open Data)
if QUICK:
    C8.N_BOOT = T.N_BOOT = 200
N_REPEATS = 1 if QUICK else C8.N_REPEATS
SEED = C8.SEED
N_BOOT = C8.N_BOOT
SEG_LEN = 2 ** 20 + 1                       # sampel per segmen Icentia (250 Hz, ±70 menit)
FS_IC = 250
IC_LABEL = {"(N": 0, "(AFIB": 1, "(AFL": 2}  # label ritme Icentia; ")" menutup episode (sesudahnya tidak berlabel)
CLASSES = ("Normal", "AFIB", "AFL")
GROUPS = ("Normal", "AFIB", "AFL")          # kelompok pasien: punya AFL (≥ 10 s) / punya AFIB / tanpa keduanya

# Aturan pemilihan (ditetapkan sebelum melihat hasil model). Semua pasien AFL dipakai; pasien AFIB & Normal diambil acak.
SEL = dict(n_afib_pat=800, n_norm_pat=1200, afl_segs=6, afib_segs=2, norm_segs=1, min_ep_s=10.0, min_norm_s=300.0,
           cap={2: 120, 1: 40, 0: 20}, cap_fast=12, fast_hr=100.0, stride=5.0)
HR_BANDS = [(0, 60), (60, 100), (100, 110), (110, 120), (120, 130), (130, 140), (140, 150), (150, 999)]

VARIANTS = {"rr": "kontrol: RR saja, AFDB + Chapman + Icentia",
            "dual": "10a: CNN EKG + RR, AFDB + Chapman + Icentia",
            "dual_tanpa_afdb": "10b: CNN EKG + RR, Chapman + Icentia (AFDB tidak dilihat)"}
REFS = {"v31": ("v31 (resep v3.1, AFDB saja; v8)", V8_OOF / "v31_r{r}.npz", V8_MODEL / "v31_r{r}_k{k}.pt", "rr"),
        "v9_rr": ("kontrol rr v9 (AFDB + Chapman)", V9_OOF / "rr_r{r}.npz", V9_MODEL / "rr_r{r}_k{k}.pt", "rr"),
        "v9_dual": ("9a: EKG + RR v9 (AFDB + Chapman)", V9_OOF / "dual_r{r}.npz", V9_MODEL / "dual_r{r}_k{k}.pt", "dual")}
PRIMARY = "dual"


# ── 1. Icentia: baca anotasi semua segmen ─────────────────────────────
_SHA = None


def ic_sha(session=None):
    """{path relatif: sha256} dari SHA256SUMS.txt resmi (±1,6 juta baris, dibaca sekali per proses)."""
    global _SHA
    if _SHA is None:
        import requests
        r = (session or requests).get(S3_IC + "SHA256SUMS.txt", timeout=300)
        r.raise_for_status()
        _SHA = {q[1].strip(): q[0] for q in (l.split(maxsplit=1) for l in r.text.splitlines()) if len(q) == 2}
    return _SHA


def _is_rhythm(text):
    return text.startswith("(") or text == ")"


def parse_atr_rhythm(data):
    """
    Bytes file .atr (format anotasi MIT) → [(sampel, teks)] untuk anotasi ritme Icentia ("(N", "(AFIB", "(AFL", ")").
    Divektorkan (Icentia menulis teks aux "None" di setiap detak, ±7.000 per file): hanya SKIP (kode 59, jarang) yang
    diproses berurutan; isi teks aux berupa ASCII sehingga tidak pernah terbaca sebagai kode ≥ 59. Seperti wfdb, setiap word
    bukan-khusus (kode ≤ 58, termasuk kode 0 dengan interval > 0) menambah waktu. Dibuktikan sama dengan wfdb.rdann
    (verify_parser).
    """
    b = np.frombuffer(data[:len(data) // 2 * 2], dtype=np.uint8)
    w = b.view("<u2")
    A, I = (w >> 10).astype(np.int64), (w & 1023).astype(np.int64)
    skip_j, nxt = [], 0
    for j in np.flatnonzero(A == 59):                        # SKIP: 2 word berikutnya = interval 32-bit (word tinggi dulu)
        if j >= nxt and j + 2 < w.size:
            skip_j.append(j)
            nxt = j + 3
    skip_j = np.asarray(skip_j, np.int64)
    payload = np.zeros(w.size, bool)
    payload[skip_j + 1] = payload[skip_j + 2] = True
    aux_j = np.flatnonzero((A == 63) & ~payload)
    n = (I[aux_j] + 1) // 2                                  # AUX: I byte teks, dipad ke genap
    idx = np.repeat(aux_j + 1, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    payload[idx[idx < w.size]] = True
    end = np.flatnonzero((w == 0) & ~payload)
    stop = int(end[0]) if end.size else w.size
    inc = np.where(~payload & (A <= 58), I, 0)
    v = (w[skip_j + 1].astype(np.int64) << 16) | w[skip_j + 2].astype(np.int64)
    inc[skip_j] = np.where(v >= 1 << 31, v - (1 << 32), v)
    t = np.cumsum(inc[:stop])
    aux_j = aux_j[aux_j < stop]
    first = b[np.minimum(2 * aux_j + 2, b.size - 1)]
    out = []
    for j in aux_j[(first == ord("(")) | (first == ord(")"))]:
        s = data[2 * j + 2:2 * j + 2 + int(I[j])].split(b"\x00")[0].decode("ascii", "replace")
        if _is_rhythm(s):
            out.append((int(t[j]), s))
    return out


def rhythm_episodes(events, seg_len=SEG_LEN):
    """[(sampel, teks)] → {label: [(awal, akhir)]}; episode berakhir di ")" atau label berikutnya."""
    ev = [(s, a.strip()) for s, a in events if a.strip().startswith("(") or a.strip() == ")"]
    out = {}
    for i, (s, a) in enumerate(ev):
        if a.startswith("("):
            e = ev[i + 1][0] if i + 1 < len(ev) else seg_len
            out.setdefault(a, []).append((s, e))
    return out


def verify_parser(names, session=None, log=print):
    """parse_atr_rhythm harus sama persis dengan wfdb.rdann (sampel & teks) pada contoh file."""
    import requests
    import wfdb
    s = session or requests.Session()
    tmp = Path(tempfile.mkdtemp(prefix="icv_"))
    try:
        n_ev = 0
        for n in names:
            data = s.get(S3_IC + n, timeout=60).content
            (tmp / Path(n).name).write_bytes(data)
            a = wfdb.rdann(str(tmp / Path(n).stem), "atr")
            ref = [(int(x), str(t)) for x, t in zip(a.sample, a.aux_note) if t and _is_rhythm(str(t))]
            got = parse_atr_rhythm(data)
            assert got == ref, f"parser .atr berbeda dari wfdb pada {n}: {got[:5]} vs {ref[:5]}"
            n_ev += len(ref)
        log(f"parser .atr = wfdb.rdann pada {len(names)} file ({n_ev} anotasi ritme)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _scan_chunk(names):
    """Satu proses: unduh .atr (thread), hitung durasi & episode tiap ritme per segmen. File tidak disimpan."""
    import requests
    loc = threading.local()

    def one(n):
        for attempt in range(8):
            try:
                if not hasattr(loc, "s"):
                    loc.s = requests.Session()
                r = loc.s.get(S3_IC + n, timeout=60)
                r.raise_for_status()
                ep = rhythm_episodes(parse_atr_rhythm(r.content))
                return n[:-4], len(r.content), {a: (sum(e - s for s, e in v), len(v), max(e - s for s, e in v))
                                                for a, v in ep.items()}
            except Exception as e:
                if attempt == 7:
                    raise RuntimeError(f"gagal membaca {n}: {e}") from e
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(32) as ex:
        return list(ex.map(one, names))


def scan_icentia(log=print, workers=None, chunk=1000):
    """Durasi (s), jumlah episode, dan episode terpanjang tiap ritme untuk setiap segmen → cache/v10/icentia_scan.npz."""
    if IC_SCAN.exists():
        return scan_summary()
    import requests
    t0 = time.perf_counter()
    sha = ic_sha(requests.Session())
    names = sorted(n for n in sha if n.endswith(".atr"))
    log(f"{len(names)} file .atr ({time.perf_counter() - t0:.0f}s untuk SHA256SUMS)")
    assert len(names) > 500000, f"hanya {len(names)} file .atr di SHA256SUMS"
    verify_parser([n for n in names if n.startswith("p00/p00010/")][:3] + names[:20:4])
    workers = workers or max(2, 2 * (os.cpu_count() or 2))
    chunks = [names[i:i + chunk] for i in range(0, len(names), chunk)]
    rows, nbytes = [], 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for b0 in range(0, len(chunks), workers * 4):
            try:
                res = pool.map_async(_scan_chunk, chunks[b0:b0 + workers * 4], chunksize=1).get(timeout=1800)
            except mp.TimeoutError:
                raise RuntimeError(f"pemindaian macet di potongan {b0}")
            for part in res:
                rows += part
                nbytes += sum(r[1] for r in part)
            dt = time.perf_counter() - t0
            log(f"  {len(rows)}/{len(names)} segmen, {nbytes / 1e9:.1f} GB, {len(rows) / dt:.0f} file/s, "
                f"sisa ±{dt / len(rows) * (len(names) - len(rows)) / 60:.0f} mnt")
    labels = sorted({a for r in rows for a in r[2]})
    dur = np.zeros((len(rows), len(labels)), np.float32)
    n_ep = np.zeros((len(rows), len(labels)), np.int32)
    longest = np.zeros((len(rows), len(labels)), np.float32)
    for i, (_, _, d) in enumerate(rows):
        for a, (s, n, mx) in d.items():
            j = labels.index(a)
            dur[i, j], n_ep[i, j], longest[i, j] = s / FS_IC, n, mx / FS_IC
    IC_DIR.mkdir(parents=True, exist_ok=True)
    tmp = IC_SCAN.with_name(IC_SCAN.stem + ".tmp.npz")
    np.savez_compressed(tmp, segment=np.array([r[0] for r in rows]), nbytes=np.array([r[1] for r in rows], np.int64),
                        labels=np.array(labels), dur_s=dur, n_episode=n_ep, longest_s=longest)
    tmp.replace(IC_SCAN)
    log(f"pemindaian selesai: {(time.perf_counter() - t0) / 60:.1f} mnt, {nbytes / 1e9:.1f} GB anotasi")
    return scan_summary()


def _load_scan():
    z = np.load(IC_SCAN)
    seg = z["segment"].astype(str)
    return seg, np.array([s.split("/")[1] for s in seg]), list(z["labels"].astype(str)), z


def scan_summary():
    seg, pat, labels, z = _load_scan()
    up = np.unique(pat)
    per = {}
    for j, a in enumerate(labels):
        m = z["n_episode"][:, j] > 0
        per[a] = {"jam": float(z["dur_s"][:, j].sum() / 3600), "episode": int(z["n_episode"][:, j].sum()),
                  "segmen": int(m.sum()), "pasien": int(len(np.unique(pat[m])))}
    g = patient_groups()
    return {"pasien": int(len(up)), "segmen": int(len(seg)), "GB anotasi": float(z["nbytes"].sum() / 1e9),
            "per ritme": per, "kelompok pasien": {GROUPS[c]: int((g["group"] == c).sum()) for c in range(3)},
            "tanpa label ritme / tidak dipakai": int((g["group"] < 0).sum())}


def patient_groups():
    """Kelompok tiap pasien dari hasil pindai: 2 = punya episode AFL ≥ 10 s, 1 = AFIB ≥ 10 s (tanpa AFL),
    0 = tanpa AFIB/AFL sama sekali dan punya ritme Normal ≥ 300 s dalam satu segmen, −1 = lainnya."""
    seg, pat, labels, z = _load_scan()
    col = {a: labels.index(a) for a in IC_LABEL if a in labels}
    up, inv = np.unique(pat, return_inverse=True)
    def any_p(mask):
        return np.bincount(inv, weights=mask.astype(float), minlength=len(up)) > 0
    lon = lambda a: z["longest_s"][:, col[a]] if a in col else np.zeros(len(seg))
    dur = lambda a: z["dur_s"][:, col[a]] if a in col else np.zeros(len(seg))
    nep = lambda a: z["n_episode"][:, col[a]] if a in col else np.zeros(len(seg))
    afl = any_p(lon("(AFL") >= SEL["min_ep_s"])
    afib = any_p(lon("(AFIB") >= SEL["min_ep_s"]) & ~afl
    clean = ~any_p((nep("(AFL") > 0) | (nep("(AFIB") > 0))
    norm = clean & any_p(dur("(N") >= SEL["min_norm_s"])
    group = np.full(len(up), -1)
    group[norm], group[afib], group[afl] = 0, 1, 2
    return {"patient": up, "group": group}


# ── 2. pemilihan pasien & segmen ───────────────────────────────────────
def select_segments(log=print):
    """Aturan SEL, seed tetap → cache/v10/icentia_pilihan.json. Batas window per pasien dibagi rata ke segmennya."""
    if IC_SEL.exists():
        return json.loads(IC_SEL.read_text(encoding="utf-8"))
    seg, pat, labels, z = _load_scan()
    col = {a: labels.index(a) for a in IC_LABEL}
    g = patient_groups()
    grp = dict(zip(g["patient"], g["group"]))
    rng = np.random.default_rng(SEED + 3000)
    chosen = {2: sorted(p for p, c in grp.items() if c == 2)}
    for c, n in ((1, SEL["n_afib_pat"]), (0, SEL["n_norm_pat"])):
        pool = sorted(p for p, v in grp.items() if v == c)
        chosen[c] = sorted(rng.choice(pool, min(n, len(pool)), replace=False).tolist())
    by_pat = {}
    for i, p in enumerate(pat):
        by_pat.setdefault(p, []).append(i)
    jobs = []
    for c in (2, 1, 0):
        for p in chosen[c]:
            idx = np.array(by_pat[p])
            if c == 0:
                ok = idx[z["dur_s"][idx, col["(N"]] >= SEL["min_norm_s"]]
                pick = sorted(rng.choice(ok, min(SEL["norm_segs"], ok.size), replace=False).tolist())
            else:
                a = "(AFL" if c == 2 else "(AFIB"
                ok = idx[z["longest_s"][idx, col[a]] >= SEL["min_ep_s"]]
                ok = ok[np.argsort(-z["dur_s"][ok, col[a]], kind="stable")]
                pick = sorted(ok[:SEL["afl_segs" if c == 2 else "afib_segs"]].tolist())
            n = len(pick)
            caps = {k: math.ceil(v / n) for k, v in SEL["cap"].items()}
            for i in pick:
                jobs.append({"segment": str(seg[i]), "patient": str(p), "group": int(c), "caps": caps,
                             "cap_fast": math.ceil(SEL["cap_fast"] / n)})
    out = {"aturan": SEL, "pasien": {GROUPS[c]: len(v) for c, v in chosen.items()}, "segmen": jobs}
    IC_DIR.mkdir(parents=True, exist_ok=True)
    IC_SEL.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                      encoding="utf-8")
    log(f"terpilih: {out['pasien']} pasien, {len(jobs)} segmen")
    return out


# ── 3. window interval RR + EKG dari segmen terpilih ───────────────────
def label_windows(peaks, fs_sig, events, stride=None):
    """
    Logika ekstraktor AFDB v2 (latih._process_record) untuk Icentia: label tiap IBI = ritme anotasi pada R-peak penutupnya;
    ")" dan label lain = tidak berlabel. Window 10 s valid bila lolos cek kualitas & semua IBI berlabel sama.
    Return [(t_start, ibi_ms, kelas)].
    """
    stride = stride or SEL["stride"]
    ev = [(s, a.strip()) for s, a in events if a.strip().startswith("(") or a.strip() == ")"]
    if not ev:
        return []
    fs_pk = float(hrv_core.FS_DEVICE)
    r_samp = np.array([s for s, _ in ev])
    r_lab = np.array([IC_LABEL.get(a, -1) for _, a in ev])
    t_end, ibi = hrv_core.peaks_to_ibi(peaks, fs_pk)
    close = peaks[1:] * (fs_sig / fs_pk)
    idx = np.searchsorted(r_samp, close, side="right") - 1
    lab = np.where(idx >= 0, r_lab[np.clip(idx, 0, None)], -1)
    rows, t, W = [], 0.0, hrv_core.WINDOW_SEC
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


def _fetch(session, rel, sha, dest, max_tries=8):
    for attempt in range(max_tries):
        try:
            r = session.get(S3_IC + rel, timeout=120)
            r.raise_for_status()
            if hashlib.sha256(r.content).hexdigest() != sha:
                raise OSError("checksum tidak cocok")
            dest.write_bytes(r.content)
            return len(r.content)
        except Exception as e:
            if attempt == max_tries - 1:
                raise RuntimeError(f"gagal mengunduh {rel}: {e}") from e
            time.sleep(2 * (attempt + 1))


_SESSION = None


def _process_segment(job):
    """Satu segmen: unduh (SHA256), window RR + EKG sejajar, pilih acak sesuai batas. File dihapus sesudahnya."""
    import requests
    import wfdb
    global _SESSION
    _SESSION = _SESSION or requests.Session()
    seg = job["segment"]
    tmp = Path(tempfile.mkdtemp(prefix="ics_"))
    out = {"segment": seg, "reason": "ok", "n_kandidat": [0, 0, 0], "n_kandidat_cepat": 0}
    try:
        nbytes = sum(_fetch(_SESSION, f"{seg}.{ext}", job["sha"][ext], tmp / f"{Path(seg).name}.{ext}")
                     for ext in ("hea", "dat", "atr"))
        base = str(tmp / Path(seg).name)
        rec = wfdb.rdrecord(base, channels=[0])
        sig = np.nan_to_num(rec.p_signal[:, 0])
        ann = wfdb.rdann(base, "atr")
        events = [(int(s), str(a)) for s, a in zip(ann.sample, ann.aux_note) if a and str(a) != "None"]
        x50, xf = E6.preprocess_record(sig, rec.fs)
        peaks = hrv_core.detect_rpeaks(x50, hrv_core.FS_DEVICE)
        if peaks is None or len(peaks) < hrv_core.MIN_IBI + 1:
            out["reason"] = "R-peak terlalu sedikit"
            return out
        rows = label_windows(peaks, rec.fs, events)
        y = np.array([r[2] for r in rows], np.int64)
        hr = np.array([60000.0 / r[1].mean() for r in rows]) if rows else np.zeros(0)
        out["n_kandidat"] = np.bincount(y, minlength=3).tolist()
        rng = np.random.default_rng(int(hashlib.sha256(seg.encode()).hexdigest()[:8], 16) + SEED)
        keep, fast = [], []
        for c in range(3):
            idx = np.flatnonzero(y == c)
            k = min(job["caps"][c], idx.size)
            pick = rng.choice(idx, k, replace=False) if k else np.zeros(0, int)
            keep += pick.tolist(); fast += [False] * len(pick)
            if c == 0:
                rest = np.setdiff1d(idx[hr[idx] >= SEL["fast_hr"]], pick)
                out["n_kandidat_cepat"] = int((hr[idx] >= SEL["fast_hr"]).sum())
                kf = min(job["cap_fast"], rest.size)
                extra = rng.choice(rest, kf, replace=False) if kf else np.zeros(0, int)
                keep += extra.tolist(); fast += [True] * len(extra)
        order = np.argsort(keep, kind="stable")
        keep, fast = np.asarray(keep, int)[order], np.asarray(fast, bool)[order]
        peaks_t = peaks / hrv_core.FS_DEVICE
        X = np.zeros((len(keep), hrv_core.MAX_BEATS, 2), np.float32)
        M = np.zeros((len(keep), hrv_core.MAX_BEATS), np.float32)
        ecg = np.zeros((len(keep), E6.N_SAMP), np.float16)
        mismatch = 0
        for j, i in enumerate(keep):
            t0, ibi, _ = rows[i]
            X[j], M[j] = hrv_core.build_sequence(ibi)
            s = int(round(t0 * E6.FS))
            w = xf[s:s + E6.N_SAMP]
            if w.size < E6.N_SAMP:
                w = np.pad(w, (0, E6.N_SAMP - w.size))
            ecg[j] = E6.normalize_window(w).astype(np.float16)
            n_pk = np.sum((peaks_t[1:] >= t0) & (peaks_t[1:] < t0 + hrv_core.WINDOW_SEC))   # bukti sejajar (cara v6)
            mismatch += int(min(n_pk, hrv_core.MAX_BEATS) != int(M[j].sum()))
        out.update(X=X, M=M, ecg=ecg, y=y[keep], hr=hr[keep].astype(np.float32),
                   t=np.array([rows[i][0] for i in keep], np.float32), fast=fast, mismatch=mismatch, nbytes=nbytes,
                   n_peaks=int(len(peaks)))
    except RuntimeError:
        raise                                                # unduhan gagal setelah 8 kali: hentikan, jangan diam-diam
    except Exception as e:  # dicatat per segmen dan dilaporkan, bukan diabaikan
        out["reason"] = f"error {type(e).__name__}: {str(e)[:80]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def build_icentia_windows(log=print, workers=None, batch=240):
    """Semua segmen terpilih → cache/v10/icentia_windows.npz (window RR + EKG + label + pasien). Dilewati bila ada."""
    if IC_WIN.exists():
        return icentia_summary()
    import requests
    sel = select_segments(log)
    sha = ic_sha(requests.Session())
    jobs = [dict(j, caps={int(a): b for a, b in j["caps"].items()}, sha={e: sha[f"{j['segment']}.{e}"] for e in ("hea", "dat", "atr")})
            for j in sel["segmen"]]
    workers = workers or max(2, 2 * (os.cpu_count() or 2))
    log(f"{len(jobs)} segmen, {workers} proses ...")
    t0 = time.perf_counter()
    res = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for b0 in range(0, len(jobs), batch):
            try:
                res += pool.map_async(_process_segment, jobs[b0:b0 + batch], chunksize=2).get(timeout=3600)
            except mp.TimeoutError:
                raise RuntimeError(f"pool ekstraksi Icentia macet di batch {b0}")
            dt = time.perf_counter() - t0
            log(f"  {len(res)}/{len(jobs)} segmen ({dt / 60:.1f} mnt, sisa ±{dt / len(res) * (len(jobs) - len(res)) / 60:.0f} mnt)")
    good = [(j, r) for j, r in zip(jobs, res) if "X" in r and len(r["y"])]
    mismatch = sum(r["mismatch"] for _, r in good)
    assert mismatch == 0, f"{mismatch} window EKG Icentia tidak sejajar dengan window RR"
    cat = lambda k: np.concatenate([r[k] for _, r in good])
    rep = lambda k: np.concatenate([np.repeat(j[k], len(r["y"])) for j, r in good])
    IC_DIR.mkdir(parents=True, exist_ok=True)
    tmp = IC_WIN.with_name(IC_WIN.stem + ".tmp.npz")
    np.savez_compressed(tmp, X=cat("X"), M=cat("M"), ecg=cat("ecg"), y=cat("y"), hr=cat("hr"), t=cat("t"), fast=cat("fast"),
                        patient=rep("patient"), segment=rep("segment"), group=rep("group").astype(np.int8),
                        seg_name=np.array([r["segment"] for r in res]), seg_reason=np.array([r["reason"] for r in res]),
                        seg_kandidat=np.array([r["n_kandidat"] for r in res], np.int64),
                        seg_kandidat_cepat=np.array([r["n_kandidat_cepat"] for r in res], np.int64))
    tmp.replace(IC_WIN)
    log(f"selesai {(time.perf_counter() - t0) / 60:.1f} mnt, {sum(r.get('nbytes', 0) for r in res) / 1e9:.1f} GB diunduh")
    return icentia_summary()


def load_icentia():
    z = np.load(IC_WIN)
    return {k: z[k] for k in ("X", "M", "ecg", "y", "hr", "t", "fast", "group")} | \
        {"patient": z["patient"].astype(str), "segment": z["segment"].astype(str)}


def icentia_summary():
    z = np.load(IC_WIN)
    y, g, fast, pat = z["y"], z["group"], z["fast"], z["patient"].astype(str)
    reason = np.array([r if not r.startswith("error") else "error" for r in z["seg_reason"].astype(str)])
    tab = {f"pasien {GROUPS[c]}": {CLASSES[k]: int(((g == c) & (y == k)).sum()) for k in range(3)} | {
        "pasien": int(len(np.unique(pat[g == c])))} for c in (2, 1, 0)}
    return {"window": int(len(y)), "per kelas": dict(zip(CLASSES, np.bincount(y, minlength=3).tolist())),
            "Normal tambahan ≥ 100 bpm": int(fast.sum()), "per kelompok pasien": tab,
            "segmen": dict(zip(*map(lambda a: a.tolist(), np.unique(reason, return_counts=True)))),
            "kandidat window (sebelum dibatasi)": dict(zip(CLASSES, z["seg_kandidat"].sum(0).tolist())),
            "kandidat Normal ≥ 100 bpm": int(z["seg_kandidat_cepat"].sum())}


def label_check():
    """Detak & keteraturan per kelas Icentia vs AFDB: apakah label masuk akal secara fisiologis (pelajaran v8)?
    Ditambah: berapa pasien AFL Icentia yang sebagian besar window AFL-nya teratur (seperti AFL 2:1 di AFDB)."""
    ic = load_icentia()
    rows = {}
    Ii = ic["X"][:, :, 0]
    mean_i = (Ii * ic["M"]).sum(1) / ic["M"].sum(1)
    cv_i = np.sqrt((((Ii - mean_i[:, None]) ** 2) * ic["M"]).sum(1) / ic["M"].sum(1)) / mean_i
    a = (ic["y"] == 2) & ~ic["fast"]
    up, inv = np.unique(ic["patient"][a], return_inverse=True)
    frac = np.bincount(inv, weights=(cv_i[a] <= 0.10).astype(float)) / np.bincount(inv)
    def stat(M, I, m):
        mean = (I * M).sum(1) / M.sum(1)
        cvv = np.sqrt((((I - mean[:, None]) ** 2) * M).sum(1) / M.sum(1)) / mean
        return {"window": int(m.sum()), "detak median": float(np.median(60000 / mean[m])),
                "CV median": float(np.median(cvv[m])), "CV > 0,10": float(np.mean(cvv[m] > 0.10))}
    acak = ~ic["fast"]
    for c in range(3):
        m = (ic["y"] == c) & acak
        rows[f"{CLASSES[c]} (Icentia)"] = {"pasien": int(len(np.unique(ic["patient"][m]))), **stat(ic["M"], ic["X"][:, :, 0], m)}
    z = np.load(C8.DATA_STD)
    for c in range(3):
        m = z["y"] == c
        rows[f"{CLASSES[c]} (AFDB)"] = {"pasien": int(len(np.unique(z["records"][m]))), **stat(z["M"], z["X"][:, :, 0], m)}
    rows["AFL (Icentia), pasien ≥ 50% window teratur"] = {"pasien": int((frac >= 0.5).sum()), "window": int(a.sum())}
    return rows


def icentia_folds(ic):
    """(R, N) fold per window Icentia: pasien diacak dalam tiap kelompok lalu dibagikan round-robin (seperti fold AFDB)."""
    up, inv = np.unique(ic["patient"], return_inverse=True)
    gp = np.zeros(len(up), np.int64)
    gp[inv] = ic["group"]
    rng = np.random.default_rng(SEED + 2000)
    folds = np.empty((C8.N_REPEATS, len(up)), np.int8)
    for r in range(C8.N_REPEATS):
        for c in range(3):
            idx = rng.permutation(np.flatnonzero(gp == c))
            folds[r, idx] = (np.arange(idx.size) + rng.integers(5)) % 5
    return folds[:, inv]


# ── 4. training ────────────────────────────────────────────────────────
def job_path(variant, r):
    return OOF_V10 / f"{variant}_r{r}.npz"


def run_job(job):
    """Satu (varian, ulangan): 5 fold; OOF AFDB + Chapman + Icentia, uji stres sintetis (hanya rr), bobot model."""
    import torch
    variant, r = job
    out = job_path(variant, r)
    if out.exists():
        return variant, r, None
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{C8._GPU}" if torch.cuda.is_available() else "cpu")
    lodo = variant == "dual_tanpa_afdb"
    kind = "rr" if variant == "rr" else "dual"
    z = np.load(C8.DATA_STD)
    X, M, y = z["X"], z["M"], z["y"]
    Ea = np.load(E6.ECG_FILE)["ecg"]
    fa = np.load(C8.FOLDS)["folds"][r]
    ch = C8.load_chapman()
    Ec = np.load(G9.CH_ECG)["ecg"]
    assert np.array_equal(np.load(G9.CH_ECG)["record"], ch["record"])
    fc = C8.chapman_folds(ch["rhythm"])[r]
    use = G9.train_mask(ch)
    ic = load_icentia()
    fi = icentia_folds(ic)[r]
    Xs, Ms, _ = C8.A7.stress_set()
    oof_a = np.full((5 if lodo else 1, len(y), 3), np.nan, np.float32)    # 10b: tiap model memprediksi SEMUA window AFDB
    oof_c = np.full((len(ch["cls"]), 3), np.nan, np.float32)
    oof_i = np.full((len(ic["y"]), 3), np.nan, np.float32)
    stress = np.full((5, len(Xs), 3), np.nan, np.float32)
    times, n_train = [], []
    MODELS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 100 * r)
    epochs = 1 if QUICK else None
    for k in range(5):
        ta = np.zeros(0, int) if lodo else np.flatnonzero(fa != k)
        tc, ti = np.flatnonzero(use & (fc != k)), np.flatnonzero(fi != k)
        if QUICK:
            ta = ta if lodo else rng.choice(ta, 3000, replace=False)
            tc, ti = rng.choice(tc, 1000, replace=False), rng.choice(ti, 1500, replace=False)
        Xtr = np.concatenate([X[ta], ch["X"][tc], ic["X"][ti]])
        Mtr = np.concatenate([M[ta], ch["M"][tc], ic["M"][ti]])
        ytr = np.concatenate([y[ta], ch["cls"][tc], ic["y"][ti]])
        seed = SEED + 100 * r + k
        if kind == "rr":
            if QUICK:
                L5.HP["epochs"] = 1
            net, mu, sd, t = L5.train_dl_weighted(Xtr, Mtr, ytr, weight_power=0.5, seed=seed, device=device)
            pa = lambda Xq, Mq, Eq: cv.predict_dl(net, Xq, Mq, mu, sd, device)
            stress[k] = cv.predict_dl(net, Xs, Ms, mu, sd, device)
        else:
            Etr = np.concatenate([Ea[ta], Ec[tc], ic["ecg"][ti]])
            net, mu, sd, t = E6.train("dual", Etr, Xtr, Mtr, ytr, seed=seed, device=device, epochs=epochs)
            pa = lambda Xq, Mq, Eq: E6.predict(net, Eq, Xq, Mq, mu, sd, device)
        if lodo:
            oof_a[k] = pa(X, M, Ea)
        else:
            oof_a[0, fa == k] = pa(X[fa == k], M[fa == k], Ea[fa == k])
        oof_c[fc == k] = pa(ch["X"][fc == k], ch["M"][fc == k], Ec[fc == k])
        oof_i[fi == k] = pa(ic["X"][fi == k], ic["M"][fi == k], ic["ecg"][fi == k])
        torch.save({"state_dict": {a: b.cpu() for a, b in net.state_dict().items()}, "mu": torch.from_numpy(mu),
                    "sd": torch.from_numpy(sd), "variant": variant, "kind": kind, "repeat": r, "fold": k},
                   MODELS / f"{variant}_r{r}_k{k}.pt")
        times.append(t); n_train.append([np.bincount(p, minlength=3) for p in (y[ta], ch["cls"][tc], ic["y"][ti])])
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert not np.isnan(oof_a).any() and not np.isnan(oof_c).any() and not np.isnan(oof_i).any()
    OOF_V10.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, oof_afdb=oof_a[0] if not lodo else np.zeros(0), afdb_per_model=oof_a if lodo else np.zeros(0),
             oof_chapman=oof_c, oof_icentia=oof_i, stress=stress, times=np.asarray(times), n_train=np.asarray(n_train))
    tmp.replace(out)
    return variant, r, float(np.mean(times))


def all_jobs():
    # CNN paling lama dulu; 10b (1 ulangan) disisipkan setelah ulangan pertama; rr terakhir
    jobs = []
    for r in range(N_REPEATS):
        jobs.append(("dual", r))
        if r == 0:
            jobs.append(("dual_tanpa_afdb", 0))
    return jobs + [("rr", r) for r in range(N_REPEATS)]


def run_all_jobs(log=print, n_gpu=None):
    """Satu proses per GPU (pola v8/v9); job yang sudah ada dilewati."""
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
            log(f"  [{(time.perf_counter() - t0) / 60:5.1f} mnt] {v:<15} ulangan {r}: {t:.0f} s/fold")


def ref_icentia_path(name, r):
    return OOF_V10 / f"ref_icentia_{name}_r{r}.npz"


def predict_refs_icentia(log=print):
    """Bobot model pembanding (v8 v31, v9 rr & dual) → prediksi Icentia. Pasien Icentia fold k diprediksi model fold k
    (semua pasien Icentia tidak pernah dilihat model ini), supaya bias fold yang sama bisa dipakai."""
    import torch
    from sihedaf_models import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ic = load_icentia()
    fi = icentia_folds(ic)
    for name, (_, _, mpath, kind) in REFS.items():
        for r in range(N_REPEATS):
            out = ref_icentia_path(name, r)
            if out.exists():
                continue
            o = np.full((len(ic["y"]), 3), np.nan, np.float32)
            for k in range(5):
                ck = torch.load(str(mpath).format(r=r, k=k), map_location="cpu", weights_only=False)
                mu, sd = ck["mu"].numpy(), ck["sd"].numpy()
                net = (build_model("MicroTransformer", **L5.MODEL_KWARGS) if kind == "rr" else E6.build("dual"))
                net.load_state_dict(ck["state_dict"])
                net = net.to(device).eval()
                m = fi[r] == k
                o[m] = (cv.predict_dl(net, ic["X"][m], ic["M"][m], mu, sd, device) if kind == "rr"
                        else E6.predict(net, ic["ecg"][m], ic["X"][m], ic["M"][m], mu, sd, device))
            assert not np.isnan(o).any()
            OOF_V10.mkdir(parents=True, exist_ok=True)
            np.savez(out, oof_icentia=o)
        log(f"  prediksi Icentia {name}: {N_REPEATS} ulangan")


# ── 5. evaluasi ────────────────────────────────────────────────────────
def load_sources():
    """{nama: [dict(oof_afdb, oof_chapman, oof_icentia, stress)] per ulangan}. 10b: OOF AFDB disusun per ulangan dari
    prediksi kelima modelnya (pasien AFDB di fold k memakai model k; semua pasien AFDB tidak pernah dilihat)."""
    folds = np.load(C8.FOLDS)["folds"]
    src = {}
    for v in VARIANTS:
        if v == "dual_tanpa_afdb":
            e = dict(np.load(job_path(v, 0)))
            per = []
            for r in range(N_REPEATS):
                o = np.empty(e["afdb_per_model"].shape[1:], np.float32)
                for k in range(5):
                    o[folds[r] == k] = e["afdb_per_model"][k][folds[r] == k]
                per.append({"oof_afdb": o, **({k: e[k] for k in ("oof_chapman", "oof_icentia")} if r == 0 else {})})
            src[v] = per
        else:
            src[v] = [dict(np.load(job_path(v, r))) for r in range(N_REPEATS)]
    for name, (_, oof_tpl, _, _) in REFS.items():
        p = [Path(str(oof_tpl).format(r=r)) for r in range(N_REPEATS)]
        assert all(q.exists() for q in p), f"hasil pembanding {name} tidak ada ({p[0]})"
        src[name] = [dict(np.load(q)) | dict(np.load(ref_icentia_path(name, r))) for r, q in enumerate(p)]
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
    keys = ["v31", "v9_rr", "v9_dual", "rr", "dual", "dual_tanpa_afdb"]
    nr = {k: sum("oof_icentia" in e for e in S[k]) for k in keys}       # 10b: Chapman/Icentia hanya ulangan 0

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    # ── AFDB (protokol v3) ──
    pred, fold_bias = {}, {}
    for k in keys:
        adj = [cv.crossfit_bias(y, S[k][r]["oof_afdb"], folds[r]) for r in range(N_REPEATS)]
        pred[k] = np.stack([a.argmax(1) for a, _ in adj]); fold_bias[k] = [b for _, b in adj]
    v1o = C8.OOF_DIR / "versi_lama"
    if (v1o / "oof_r0.npz").exists():
        pred["v1"] = np.stack([np.load(v1o / f"oof_r{r}.npz")["det"].argmax(-1) for r in range(N_REPEATS)])
    ev = P4.evaluate(d, pred)
    W = cv.bootstrap_weights(d.strata, N_BOOT, seed=SEED)
    cms = {k: np.stack([cv.patient_cms(y, p, rec, patients) for p in pred[k]]) for k in keys}
    boot = {k: cv.boot_metrics(c, W) for k, c in cms.items()}
    point = {k: cv.point_metrics(c) for k, c in cms.items()}

    def pair(a, b, bt=boot, pt=point, ms=("Macro-F1", "AFL F1", "AFL R", "Normal R")):
        return {m: [float(pt[a][m].mean() - pt[b][m].mean()), *ci(bt[a][m] - bt[b][m]),
                    float(np.mean(bt[a][m] - bt[b][m] > 0))] for m in ms}
    pairs_9a = {k: pair(k, "v9_dual") for k in ("dual", "dual_tanpa_afdb")}
    pairs_9rr = {"rr": pair("rr", "v9_rr")}
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

    def biased(k, oof_key, fold_r):
        """Prediksi kelas di data eksternal: bias fold AFDB yang sama dengan model fold itu (seperti produksi)."""
        out = []
        for r in range(nr[k]):
            o, p = S[k][r][oof_key], np.empty(len(fold_r[r]), np.int64)
            for f in range(5):
                m = fold_r[r] == f
                p[m] = cv.apply_bias(o[m], fold_bias[k][r][f]).argmax(1)
            out.append(p)
        return np.stack(out)
    runners, pk_note = C8.load_packages()
    runners = {k: f for k, f in runners.items() if k == "v31_paket"}

    # ── Chapman (label andal; bias dari AFDB per model fold) ──
    ch = C8.load_chapman()
    fc = C8.chapman_folds(ch["rhythm"])
    contrib = {k: np.mean([C8._onehot(p) for p in biased(k, "oof_chapman", fc)], 0) for k in keys}
    for k, f in runners.items():
        contrib[k] = C8._onehot(np.concatenate([f(ch["X"][i:i + 4096], ch["M"][i:i + 4096]) for i in range(0, len(ch["X"]), 4096)]))
    m3 = G9.train_mask(ch)
    cls3 = ch["cls"][m3]
    cm_pt = {k: np.stack([v[m3][cls3 == c].sum(0) for c in range(3)]) for k, v in contrib.items()}
    bt = C8.chapman_boot({k: v[m3] for k, v in contrib.items()}, cls3)
    ch_mets = {k: {m: [float(C8._cm_metrics(cm_pt[k])[m]), *ci(C8._cm_metrics(bt[k])[m])]
                   for m in ("Macro-F1", "AFL F1", "AFL R", "AFL P", "AFIB F1", "AFIB R", "Normal R")} for k in contrib}
    ch_pairs = {}
    for ref in ("v31_paket", "v9_dual"):
        for k in contrib:
            if k != ref:
                ch_pairs.setdefault(ref, {})[k] = {
                    m: [float(C8._cm_metrics(cm_pt[k])[m] - C8._cm_metrics(cm_pt[ref])[m]),
                        *ci(C8._cm_metrics(bt[k])[m] - C8._cm_metrics(bt[ref])[m]),
                        float(np.mean(C8._cm_metrics(bt[k])[m] - C8._cm_metrics(bt[ref])[m] > 0))]
                    for m in ("Macro-F1", "AFL R", "Normal R")}
    alarm = {k: v[:, 1] + v[:, 2] for k, v in contrib.items()}
    band = lambda m, lo, hi: m & (ch["hr"] >= lo) & (ch["hr"] < hi)
    st = ch["rhythm"] == "ST"
    st_bands = {f"{lo}–{hi if hi < 999 else '∞'}": {"n": int(band(st, lo, hi).sum()),
                                                    **C8.boot_mean({k: a[band(st, lo, hi)] for k, a in alarm.items()})}
                for lo, hi in C8.ST_BANDS}
    st_all = {"n": int(st.sum()), **C8.boot_mean({k: a[st] for k, a in alarm.items()})}
    svt = ch["rhythm"] == "SVT"
    svt_alarm = {"n": int(svt.sum()), **C8.boot_mean({k: a[svt] for k, a in alarm.items()})}

    # ── Icentia (pasien uji tidak pernah dilatih; bias dari AFDB) ──
    ic = load_icentia()
    fi = icentia_folds(ic)
    yi, acak = ic["y"], ~ic["fast"]
    ipat = np.unique(ic["patient"])
    pidx = np.searchsorted(ipat, ic["patient"])
    gp = np.zeros(len(ipat), np.int64)
    gp[pidx] = ic["group"]
    pred_i = {k: biased(k, "oof_icentia", fi) for k in keys}
    for k, f in runners.items():
        pred_i[k] = np.concatenate([f(ic["X"][i:i + 4096], ic["M"][i:i + 4096]) for i in range(0, len(yi), 4096)])[None]
    Wi = cv.bootstrap_weights(gp, N_BOOT, seed=SEED)
    cms_i = {k: np.stack([cv.patient_cms(yi[acak], p[acak], ic["patient"][acak], ipat) for p in v]) for k, v in pred_i.items()}
    boot_i = {k: cv.boot_metrics(c, Wi) for k, c in cms_i.items()}
    point_i = {k: cv.point_metrics(c) for k, c in cms_i.items()}
    ic_rows = {k: {m: [float(point_i[k][m].mean()), *ci(boot_i[k][m])] for m in cv.METRICS} for k in cms_i}
    ic_pairs = {ref: {k: pair(k, ref, boot_i, point_i) for k in cms_i if k != ref} for ref in ("v31", "v9_dual", "v31_paket")}

    def rate(num, den):
        """num, den (R, P) per pasien → rasio pooled (titik) + CI bootstrap pasien yang sama (Wi)."""
        n, dn = num.mean(0), den.mean(0)
        b = (Wi @ n) / np.maximum(Wi @ dn, 1e-12)
        return [float(n.sum() / max(dn.sum(), 1e-12)), *ci(b)]

    def per_pat(mask, hit):
        """(R, P): jumlah window `mask` per pasien dan yang `hit` (hit: (R, N) bool)."""
        den = np.bincount(pidx[mask], minlength=len(ipat)).astype(float)
        num = np.stack([np.bincount(pidx[mask], weights=h[mask].astype(float), minlength=len(ipat)) for h in hit])
        return num, np.broadcast_to(den, num.shape)
    nm = yi == 0
    ic_hr = {}
    for lo, hi in HR_BANDS:
        m = nm & (ic["hr"] >= lo) & (ic["hr"] < hi)
        ic_hr[f"{lo}–{hi if hi < 999 else '∞'}"] = {"n": int(m.sum()), "pasien": int(len(np.unique(pidx[m]))),
                                                    **{k: rate(*per_pat(m, p > 0)) for k, p in pred_i.items()}}
    ic_norm_origin = {GROUPS[c]: {"n": int((nm & acak & (ic["group"] == c)).sum()),
                                  **{k: rate(*per_pat(nm & acak & (ic["group"] == c), p > 0)) for k, p in pred_i.items()}}
                      for c in range(3)}
    afl = yi == 2
    ic_afl_hr = {f"{lo}–{hi if hi < 999 else '∞'}": {"n": int((afl & acak & (ic["hr"] >= lo) & (ic["hr"] < hi)).sum()),
                                                     **{k: rate(*per_pat(afl & acak & (ic["hr"] >= lo) & (ic["hr"] < hi), p == 2))
                                                        for k, p in pred_i.items()}}
                 for lo, hi in [(0, 100), (100, 130), (130, 160), (160, 999)]}
    Ii = ic["X"][:, :, 0]
    mean_i = (Ii * ic["M"]).sum(1) / ic["M"].sum(1)
    cv_i = np.sqrt((((Ii - mean_i[:, None]) ** 2) * ic["M"]).sum(1) / ic["M"].sum(1)) / mean_i
    ic_afl_cv = {}                                           # AFL Icentia kebanyakan tidak teratur (blok bervariasi), AFDB teratur
    for name, m in (("teratur (CV ≤ 0,10)", afl & acak & (cv_i <= 0.10)), ("tidak teratur (CV > 0,10)", afl & acak & (cv_i > 0.10))):
        ic_afl_cv[name] = {"n": int(m.sum()), "AFL": {k: rate(*per_pat(m, p == 2)) for k, p in pred_i.items()},
                           "AFIB": {k: rate(*per_pat(m, p == 1)) for k, p in pred_i.items()}}
    afl_pat = np.flatnonzero((gp == 2) & (np.bincount(pidx[afl & acak], minlength=len(ipat)) >= 5))
    ic_afl_pat = {k: (cms_i[k][:, afl_pat, 2, 2] / cms_i[k][:, afl_pat, 2, :].sum(-1)).mean(0) for k in cms_i}
    ic_afl_sum = {k: {"pasien": int(len(afl_pat)), "median": float(np.median(v)), "≥ 0,5": float(np.mean(v >= 0.5)),
                      "= 0": float(np.mean(v == 0))} for k, v in ic_afl_pat.items()}
    ic_fp = {k: {GROUPS[c]: float(np.mean([((p == 2) & (yi != 2) & acak & (ic["group"] == c)).sum() for p in v]))
                 for c in range(3)} for k, v in pred_i.items()}

    # ── uji stres sintetis: hanya model interval RR ──
    _, _, meta = C8.A7.stress_set()
    mj = np.array([m[0] for m in meta]); mb = np.array([m[1] for m in meta])
    stress = {}
    for k in ("rr", "v31", "v9_rr"):
        cls_s = np.stack([cv.apply_bias(S[k][r]["stress"][f], fold_bias[k][r][f]).argmax(1)
                          for r in range(N_REPEATS) for f in range(5)])
        stress[k] = {j: [(np.bincount(cls_s[:, (mj == j) & (mb == b)].ravel(), minlength=3)
                          / cls_s[:, (mj == j) & (mb == b)].size).tolist() for b in T.BPMS] for j in T.JITTERS}
    threshold = {k: {j: T.normal_threshold(np.array(t)) for j, t in s.items()} for k, s in stress.items()}

    return {
        "labels": {**VARIANTS, **{k: v[0] for k, v in REFS.items()}, "v1": "v1 (OOF cache PC tim)",
                   "v31_paket": "paket produksi v3.1"},
        "primary": PRIMARY, "quick": QUICK, "n_ulangan_eksternal": nr,
        "afdb": {"rows": ev.rows, "episodes": ev.epi, "pairs": ev.pairs, "pairs_9a": pairs_9a, "pairs_9rr": pairs_9rr,
                 "afl_patients": ev.afl_patients, "afl_recall": ev.afl_recall, "fp": fp, "by_hr": by_hr, "sens": sens,
                 "bias": {k: {"median": np.median(np.concatenate(b), 0).tolist(), "min": np.concatenate(b).min(0).tolist(),
                              "max": np.concatenate(b).max(0).tolist()} for k, b in fold_bias.items()}},
        "chapman": {"metrics": ch_mets, "pairs": ch_pairs, "st_bands": st_bands, "st_all": st_all, "svt_alarm": svt_alarm,
                    "paket_catatan": pk_note,
                    "n_metrik": {"Normal": int((cls3 == 0).sum()), "AFIB": int((cls3 == 1).sum()), "AFL": int((cls3 == 2).sum())}},
        "icentia": {"rows": ic_rows, "pairs": ic_pairs, "normal_hr": ic_hr, "normal_asal": ic_norm_origin,
                    "afl_hr": ic_afl_hr, "afl_cv": ic_afl_cv, "afl_pasien": ic_afl_sum, "afl_pasien_recall": {k: v.tolist() for k, v in ic_afl_pat.items()},
                    "afl_palsu_asal": ic_fp, "n_pasien": {GROUPS[c]: int((gp == c).sum()) for c in range(3)},
                    "n_window_metrik": dict(zip(CLASSES, np.bincount(yi[acak], minlength=3).tolist()))},
        "stress": stress, "threshold": threshold, "bpms": T.BPMS, "jitters": list(T.JITTERS),
        "train_time": {v: float(np.mean([np.load(job_path(v, r))["times"].mean() for r in range(nr[v])])) for v in VARIANTS},
        "n_train": {v: np.load(job_path(v, 0))["n_train"].mean(0).tolist() for v in VARIANTS},
    }


def eksplorasi_afdb(keys=("v31", "v9_dual", "dual", "dual_tanpa_afdb"), wide=np.linspace(-15, 3, 73)):
    """
    Eksploratif, dibuat SETELAH hasil (bukan bagian protokol): apakah kegagalan AFDB model bentuk gelombang soal kalibrasi
    (bias AFL terpotong di −7) atau soal pemisahan kelas? (1) AUROC satu-lawan-lain dari OOF AFDB (tidak bergantung bias),
    (2) Macro-F1 bila grid bias cross-fit diperlebar ke −15, (3) asal AFL palsu per kelas asli dan per pasien (bias protokol).
    """
    from sklearn.metrics import roc_auc_score
    z = np.load(C8.DATA_STD)
    y, rec = z["y"], z["records"].astype(str)
    pats = np.unique(rec)
    afl_p = np.unique(rec[y == 2])
    folds = np.load(C8.FOLDS)["folds"][:N_REPEATS]
    S = load_sources()
    out = {}
    for k in keys:
        row = {"auc_afl": [], "auc_afib": [], "mf1_lebar": [], "aflr_lebar": [], "bias_afl_lebar": [], "fp_normal": [],
               "fp_afib": [], "fp_pasien_afl": []}
        per = {}
        for r in range(N_REPEATS):
            o = S[k][r]["oof_afdb"]
            row["auc_afl"].append(roc_auc_score(y == 2, o[:, 2])); row["auc_afib"].append(roc_auc_score(y == 1, o[:, 1]))
            adj = np.empty_like(o)
            for f in range(5):
                m = folds[r] == f
                b = cv.fit_class_bias(y[~m], o[~m], grid=wide)
                adj[m] = cv.apply_bias(o[m], b)
                row["bias_afl_lebar"].append(float(b[2]))
            met = cv.metrics_from_cm(cv.patient_cms(y, adj.argmax(1), rec, pats).sum(0))
            row["mf1_lebar"].append(float(met["Macro-F1"])); row["aflr_lebar"].append(float(met["AFL R"]))
            p = cv.crossfit_bias(y, o, folds[r])[0].argmax(1)
            fp = (p == 2) & (y != 2)
            row["fp_normal"].append(int((fp & (y == 0)).sum())); row["fp_afib"].append(int((fp & (y == 1)).sum()))
            row["fp_pasien_afl"].append(int((fp & np.isin(rec, afl_p)).sum()))
            for q, n in zip(*np.unique(rec[fp], return_counts=True)):
                per[q] = per.get(q, 0.0) + n / N_REPEATS
        b = np.array(row.pop("bias_afl_lebar"))
        out[k] = {a: float(np.mean(v)) for a, v in row.items()} | {
            "bias_afl_lebar": [float(b.min()), float(np.median(b)), float(b.max())],
            "fp_per_pasien": dict(sorted(((q, float(v)) for q, v in per.items()), key=lambda t: -t[1])[:8])}
    return out


def pack_results(dest):
    """Keluaran v10 (ringkasan, OOF, bobot model) → satu zip. Data Icentia (cache/v10/icentia_*) dibawa terpisah."""
    import zipfile
    dest = Path(dest)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(CACHE_OUT.rglob("*")):
            if p.is_file() and not p.name.endswith((".tmp.npz", ".zip")) and not p.name.startswith("icentia_"):
                zf.write(p, Path("cache") / CACHE_OUT.name / p.relative_to(CACHE_OUT))
    return dest, dest.stat().st_size / 1e6
