"""
atrium.py — logika v12 (cabang aktivitas atrium dari EKG + data Holter SHDB-AF). Rancangan: RANCANGAN.md.

Alur:
  1. unduh_shdb        : sinyal 98 rekaman beranotasi SHDB-AF dari mirror S3 PhysioNet, dicek SHA256, bisa dilanjutkan.
  2. baca_dat/baca_atr : pembaca WFDB format 16 (SHDB-AF) dan 212 (AFDB) serta anotasi MIT, tanpa pustaka wfdb (tidak ada di venv
                         proyek). Dibuktikan sama dengan wfdb.rdrecord/rdann (bukti_pembaca, dijalankan lewat `uv run --with wfdb`).
  3. fitur_atrium      : pembatalan QRST (templat median detak) → fitur aktivitas atrium per window 10 s @50 Hz.
  4. bangun_shdb       : front-end produksi (50 Hz, detektor R-peak v2) + label window cara ekstraktor AFDB v2 (label_windows v10),
                         batas per subjek (SEL), potongan EKG v6, fitur atrium → cache/v12/shdb/<rekaman>.npz (bisa dilanjutkan).
  5. bangun_afdb       : fitur atrium untuk setiap window dataset v2 (urutan sama); potongan EKG dibuktikan sama dengan cache v6.
  6. gerbang           : AUROC fitur per dataset (RANCANGAN 7.2) + contoh sinyal sisa & spektrum per kelas (7.3).

Kode versi lama dipakai lewat sys.path, tidak disalin: hrv_core (v2), ekg (v6: preprocess_record, normalize_window), icentia (v10:
label_windows, logika ekstraktor AFDB v2).

Data SHDB-AF: PhysioNet `shdb-af` 1.0.1 (Tsutsui dkk. 2024), lisensi ODC-BY 1.0. Sitasi: Tsutsui K, Biton Brimer S,
Ben-Moshe N, dkk. SHDB-AF: a Japanese Holter ECG database of atrial fibrillation. arXiv:2406.16974 (2024); PhysioNet,
doi:10.13026/10mk-y852.
"""
import csv
import hashlib
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SHDB_DIR = REPO / "data/shdb_af/1.0.1"
AFDB_DIR = REPO / "data/afdb"
S3_SHDB = "https://physionet-open.s3.amazonaws.com/shdb-af/1.0.1/"    # mirror resmi PhysioNet (AWS Open Data)
CACHE = REPO / "cache/v12"
SEED = 2026
FS = 50                                                  # sampling rate perangkat (hrv_core.FS_DEVICE)
N_SAMP = 500                                             # 10 s @50 Hz (v6)
KELAS = ("Normal", "AFIB", "AFL")

# Aturan pemilihan window SHDB-AF (RANCANGAN 6.2, ditetapkan sebelum melihat hasil model). Batas per subjek; subjek dengan dua
# rekaman membagi batasnya rata ke kedua rekaman.
SEL = dict(cap={2: 1500, 1: 600, 0: 400}, cap_cepat=400, hr_cepat=100.0, stride=5.0)


# ── 1. unduh ───────────────────────────────────────────────────────────
def _sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def sha_resmi():
    """{nama file: sha256} dari SHA256SUMS.txt resmi (sudah diunduh bersama anotasi)."""
    out = {}
    for line in (SHDB_DIR / "SHA256SUMS.txt").read_text().splitlines():
        q = line.split(maxsplit=1)
        if len(q) == 2:
            out[q[1].strip()] = q[0]
    return out


def rekaman_beranotasi():
    """Rekaman yang punya file .atr (98 di versi 1.0.1)."""
    return sorted(p.stem for p in SHDB_DIR.glob("*.atr"))


def _unduh_satu(nama, sha, max_coba=6, log=print):
    tujuan = SHDB_DIR / nama
    if tujuan.exists() and _sha256(tujuan) == sha:
        return nama, "sudah ada"
    sementara = tujuan.with_suffix(tujuan.suffix + ".part")
    for coba in range(1, max_coba + 1):
        try:
            with urllib.request.urlopen(S3_SHDB + nama, timeout=120) as r, open(sementara, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            if _sha256(sementara) != sha:
                raise IOError("SHA256 tidak cocok")
            sementara.replace(tujuan)
            return nama, f"diunduh (percobaan {coba})"
        except Exception as e:                                       # noqa: BLE001  (dicoba ulang, lalu error)
            log(f"  {nama}: {e} (percobaan {coba}/{max_coba})")
            time.sleep(min(60, 5 * coba))
    raise RuntimeError(f"gagal mengunduh {nama} setelah {max_coba} percobaan")


def unduh_shdb(workers=4, log=print):
    """Sinyal (.dat) semua rekaman beranotasi. File yang sudah ada dan SHA256-nya cocok dilewati."""
    sha = sha_resmi()
    nama = [f"{r}.dat" for r in rekaman_beranotasi()]
    hilang = [n for n in nama if n not in sha]
    if hilang:
        raise RuntimeError(f"tidak ada di SHA256SUMS.txt: {hilang}")
    t0 = time.time()
    selesai = 0
    with ThreadPoolExecutor(workers) as ex:
        for n, status in ex.map(lambda n: _unduh_satu(n, sha[n], log=log), nama):
            selesai += 1
            log(f"[{(time.time() - t0) / 60:5.1f} mnt] {selesai}/{len(nama)} {n}: {status}")
    log(f"selesai: {len(nama)} sinyal, semua SHA256 cocok")


# ── 2. pembaca WFDB (tanpa pustaka wfdb) ───────────────────────────────
def baca_hea(base):
    """Header WFDB → fs, jumlah sampel, format, dan (gain, baseline) tiap kanal. Gain 0 = tidak dikalibrasi → 200 (seperti wfdb)."""
    base = Path(base)
    baris = [b for b in base.with_suffix(".hea").read_text().splitlines() if b.strip() and not b.startswith("#")]
    _, n_sig, fs, n_samp = baris[0].split()[:4]
    fmt, kanal = None, []
    for b in baris[1:1 + int(n_sig)]:
        q = b.split()
        fmt = q[1].split("x")[0].split(":")[0].split("+")[0]
        gb = q[2].split("/")[0]
        gain = float(gb.split("(")[0])
        base_ = int(gb.split("(")[1].rstrip(")")) if "(" in gb else int(q[4])
        kanal.append((gain if gain != 0 else 200.0, base_))
    if fmt not in ("16", "212"):
        raise ValueError(f"{base.name}: format {fmt} tidak didukung (hanya 16 dan 212)")
    return float(fs.split("/")[0]), int(n_samp), fmt, kanal


def baca_dat(base, kanal=0):
    """Sinyal fisik (mV) satu kanal; sampel tidak valid → NaN. Sama dengan wfdb.rdrecord(..., channels=[k]).p_signal."""
    fs, n, fmt, ch = baca_hea(base)
    nk = len(ch)
    if fmt == "16":
        raw = np.fromfile(Path(base).with_suffix(".dat"), dtype="<i2")[:n * nk].reshape(-1, nk)[:, kanal].astype(np.int32)
        invalid = -32768
    else:                                                  # 212: 2 sampel 12 bit dalam 3 byte, kanal berselang-seling
        b = np.fromfile(Path(base).with_suffix(".dat"), dtype=np.uint8)
        b = b[:b.size // 3 * 3].reshape(-1, 3).astype(np.int32)
        s = np.empty(b.shape[0] * 2, np.int32)
        s[0::2] = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
        s[1::2] = b[:, 2] | ((b[:, 1] & 0xF0) << 4)
        s[s > 2047] -= 4096
        raw = s[:n * nk].reshape(-1, nk)[:, kanal]
        invalid = -2048
    gain, base_ = ch[kanal]
    x = (raw.astype(np.float64) - base_) / gain
    x[raw == invalid] = np.nan
    return x, fs


def baca_atr(base):
    """
    Anotasi MIT (.atr) → (sampel, kode, aux) setiap anotasi. Format: word 16-bit little-endian, kode = 6 bit atas, interval = 10
    bit bawah; SKIP (59) = interval 32-bit di 2 word berikutnya (word tinggi dulu); NUM/SUB/CHN (60–62) tidak menambah waktu;
    AUX (63) = teks I byte (dipad genap) milik anotasi sebelumnya; kode 0 dengan interval > 0 hanya menambah waktu; word 0 = akhir.
    Catatan metadata di sampel 0 ("## time resolution: ...") dibuang seperti wfdb.rdann. Dibuktikan sama dengan wfdb.
    """
    w = np.fromfile(Path(base).with_suffix(".atr"), dtype="<u2")
    sampel, kode, aux = [], [], []
    t, i, n = 0, 0, w.size
    while i < n:
        a, iv = int(w[i]) >> 10, int(w[i]) & 1023
        if a == 0 and iv == 0:
            break
        if a == 59:                                             # SKIP
            v = (int(w[i + 1]) << 16) | int(w[i + 2])
            t += v - (1 << 32) if v >= 1 << 31 else v
            i += 3
            continue
        if a == 63:                                             # AUX
            nb = (iv + 1) // 2
            teks = w[i + 1:i + 1 + nb].tobytes()[:iv].split(b"\x00")[0].decode("ascii", "replace")
            if aux:
                aux[-1] = teks
            i += 1 + nb
            continue
        if a in (60, 61, 62):                                   # NUM, SUB, CHN
            i += 1
            continue
        if a == 0:                                              # kode 0 dengan interval > 0: menambah waktu, bukan anotasi
            t += iv
            i += 1
            continue
        t += iv
        sampel.append(t); kode.append(a); aux.append("")
        i += 1
    simpan = [j for j in range(len(sampel)) if not (sampel[j] == 0 and kode[j] == 22 and aux[j].startswith("## "))]
    return np.asarray(sampel, np.int64)[simpan], np.asarray(kode, np.int16)[simpan], [aux[j] for j in simpan]


def bukti_pembaca(shdb=("001", "002", "010", "036", "131"), afdb=("04015", "04936", "08378"), log=print):
    """baca_dat/baca_atr = wfdb, untuk SHDB-AF (format 16) dan AFDB (format 212). Butuh wfdb (`uv run --with wfdb`)."""
    import wfdb
    for base in [SHDB_DIR / r for r in shdb] + [AFDB_DIR / r for r in afdb]:
        ann = wfdb.rdann(str(base), "atr")
        s, k, a = baca_atr(base)
        assert np.array_equal(s, ann.sample), f"{base.name}: sampel anotasi beda"
        assert [x or "" for x in ann.aux_note] == a, f"{base.name}: aux beda"
        for ch in (0, 1):
            ref = wfdb.rdrecord(str(base), channels=[ch]).p_signal[:, 0]
            x, fs = baca_dat(base, ch)
            assert fs == ann.fs and x.shape == ref.shape, f"{base.name} kanal {ch}: bentuk"
            assert np.allclose(x, ref, equal_nan=True, rtol=0, atol=1e-9), f"{base.name} kanal {ch}: nilai"
        log(f"{base.name}: {len(s)} anotasi, sinyal 2 kanal = wfdb")
    log(f"pembaca .dat/.atr = wfdb pada {len(shdb)} rekaman SHDB-AF dan {len(afdb)} rekaman AFDB")


# ── 3. fitur aktivitas atrium ──────────────────────────────────────────
FITUR = ("df_hz", "organisasi", "entropi", "autokorelasi", "amp_atrium", "rasio_energi", "p_konsisten",
         "rasio_konduksi", "jarak_bulat", "kuantisasi", "n_detak")
PRA, PASCA = 5, 20                                        # templat QRST −0,10…+0,40 s
P0, P1 = 15, 5                                            # segmen gelombang P: −0,30…−0,10 s sebelum R
MIN_DETAK = 4


def _templat(xf, pk):
    idx = pk[(pk - PRA >= 0) & (pk + PASCA < xf.size)]
    if idx.size < MIN_DETAK:
        return None
    return np.median(np.stack([xf[p - PRA:p + PASCA + 1] for p in idx]), 0)


def sinyal_sisa(xf, peaks, s):
    """Potongan [s, s+500) terfilter dan sinyal sisanya setelah templat QRST median dikurangkan di setiap R-peak."""
    pk = peaks[(peaks >= s - PASCA) & (peaks < s + N_SAMP + PRA)]
    seg = xf[s:s + N_SAMP]
    if seg.size < N_SAMP:
        seg = np.pad(seg, (0, N_SAMP - seg.size))
    tpl = _templat(xf, pk[(pk >= s) & (pk < s + N_SAMP)])
    if tpl is None:
        return seg, None, pk
    sisa = seg.copy()
    for p in pk:
        a, b = p - PRA - s, p + PASCA + 1 - s
        ta, tb = max(0, -a), (b - a) - max(0, b - N_SAMP)
        if tb > ta:
            sisa[a + ta:a + tb] -= tpl[ta:tb]
    return seg, sisa, pk


UP = 5                                                    # revisi 1: 50 → 250 Hz untuk penyelarasan sub-sampel
CARI_R = 3                                                # ±10 ms → 3 sampel 250 Hz (dibulatkan ke atas)
KOSONG = (0.040, 0.060)                                   # daerah QRS yang diinterpolasi: −40…+60 ms


def sinyal_sisa_r1(xf, peaks, s):
    """
    B1 revisi 1 (RANCANGAN 14): upsample ×5, R-peak diperhalus ke puncak |x| dalam ±10 ms, templat median detak yang selaras
    dikurangkan dengan skala amplitudo per detak (kuadrat terkecil, 0,5–1,5), daerah QRS −40…+60 ms diinterpolasi linear, lalu
    sisa di-downsample kembali ke 50 Hz. Keluaran sama bentuknya dengan sinyal_sisa.
    """
    from scipy import signal as sps
    seg = xf[s:s + N_SAMP]
    if seg.size < N_SAMP:
        seg = np.pad(seg, (0, N_SAMP - seg.size))
    pk = peaks[(peaks >= s - PASCA) & (peaks < s + N_SAMP + PRA)]
    a0, a1 = max(0, s - FS), min(xf.size, s + N_SAMP + FS)          # margin 1 s agar detak di tepi ikut dikurangkan
    up = sps.resample_poly(xf[a0:a1], UP, 1)
    fu = FS * UP
    pku = []
    for p in peaks[(peaks >= a0) & (peaks < a1)]:
        c = (p - a0) * UP
        lo, hi = max(0, c - CARI_R), min(up.size, c + CARI_R + 1)
        pku.append(lo + int(np.argmax(np.abs(up[lo:hi]))))
    pku = np.asarray(pku, np.int64)
    pre, post = int(round(0.10 * fu)), int(round(0.40 * fu))
    w0, w1 = (s - a0) * UP, (s - a0 + N_SAMP) * UP
    inw = pku[(pku >= w0) & (pku < w1) & (pku - pre >= 0) & (pku + post < up.size)]
    if inw.size < MIN_DETAK:
        return seg, None, pk
    tpl = np.median(np.stack([up[p - pre:p + post + 1] for p in inw]), 0)
    sisa = up.copy()
    for p in pku:
        lo, hi = p - pre, p + post + 1
        ta, tb = max(0, -lo), (hi - lo) - max(0, hi - up.size)
        if tb <= ta:
            continue
        t_ = tpl[ta:tb]
        nn = float(t_ @ t_)
        k = min(1.5, max(0.5, float(up[lo + ta:lo + tb] @ t_) / nn)) if nn > 0 else 1.0
        sisa[lo + ta:lo + tb] -= k * t_
    b0, b1 = int(round(KOSONG[0] * fu)), int(round(KOSONG[1] * fu))
    for p in pku:
        lo, hi = max(0, p - b0), min(up.size - 1, p + b1)
        if hi > lo:
            sisa[lo:hi + 1] = np.linspace(sisa[lo], sisa[hi], hi - lo + 1)
    s50 = sps.resample_poly(sisa, 1, UP)[s - a0:s - a0 + N_SAMP]
    if s50.size < N_SAMP:
        s50 = np.pad(s50, (0, N_SAMP - s50.size))
    return seg, s50, pk


def fitur_atrium(xf, peaks, t0, revisi=0):
    """Fitur RANCANGAN bagian 5 (B2) untuk window [t0, t0+10 s). Window dengan < 4 detak: fitur NaN (kecuali n_detak).
    revisi=1: sinyal sisa dari sinyal_sisa_r1 (RANCANGAN 14); fitur B2 tidak berubah."""
    from scipy import signal as sps
    s = int(round(t0 * FS))
    seg, sisa, pk = (sinyal_sisa_r1 if revisi == 1 else sinyal_sisa)(xf, peaks, s)
    pkw = pk[(pk >= s) & (pk < s + N_SAMP)]
    out = np.full(len(FITUR), np.nan, np.float32)
    out[-1] = pkw.size
    if sisa is None:
        return out
    f, P = sps.welch(sisa - sisa.mean(), fs=FS, nperseg=256, noverlap=128, nfft=1024)
    e = lambda lo, hi: float(P[(f >= lo) & (f <= hi)].sum())  # noqa: E731
    pita = (f >= 3) & (f <= 12)
    tot = e(3, 12)
    if tot <= 0:
        return out
    cari = (f >= 3) & (f <= 10)
    df = float(f[cari][np.argmax(P[cari])])
    pn = P[pita] / P[pita].sum()
    z = sisa - sisa.mean()
    ac = np.correlate(z, z, "full")[z.size - 1:]
    lag = int(round(FS / df))
    amp_r = np.median(np.abs(seg[np.clip(pkw - s, 0, N_SAMP - 1)]))
    pseg = [seg[p - s - P0:p - s - P1] for p in pkw if p - s - P0 >= 0]
    if len(pseg) >= 3:
        c = np.corrcoef(np.stack(pseg))
        p_kons = float(c[np.triu_indices(len(pseg), 1)].mean())
    else:
        p_kons = np.nan
    rr = np.diff(pkw) / FS
    fcl = 1.0 / df
    q = rr / fcl
    rasio = float(np.median(rr) / fcl) if rr.size else np.nan
    out[:-1] = [df, (e(df - .5, df + .5) + e(2 * df - .5, 2 * df + .5)) / tot,
                float(-(pn * np.log(pn + 1e-12)).sum() / math.log(pn.size)),
                float(ac[lag] / ac[0]) if ac[0] > 0 and lag < ac.size else np.nan,
                float(sisa.std() / amp_r) if amp_r > 0 else np.nan,
                float(np.var(sisa) / np.var(seg)) if np.var(seg) > 0 else np.nan,
                p_kons, rasio, abs(rasio - round(rasio)) if rr.size else np.nan,
                float(np.mean(np.abs(q - np.round(q)) < 0.15)) if rr.size else np.nan]
    return out


def pola_rr(ibi_ms):
    """Kelompok pola RR (tabel RANCANGAN 1.1): 0 cepat-teratur, 1 antara, 2 tidak teratur, 3 lambat-teratur."""
    hr, cv = 60000.0 / ibi_ms.mean(), ibi_ms.std() / ibi_ms.mean()
    if cv >= 0.10:
        return 2
    if cv < 0.06 and hr >= 125:
        return 0
    if cv < 0.06 and hr < 110:
        return 3
    return 1


# ── 4. window SHDB-AF ──────────────────────────────────────────────────
def _modul():
    """Kode versi lama lewat sys.path (tidak disalin): icentia (v10) juga memasang path v2–v9 (hrv_core, ekg v6)."""
    p = str(REPO / "v10_icentia_wearable")
    if p not in sys.path:
        sys.path.insert(0, p)
    import icentia as I10
    return I10


def subjek_shdb():
    """{rekaman: subjek} dan {subjek: jumlah rekaman beranotasi} dari AdditionalData.csv."""
    meta = {r["Data_ID"]: r["Subject_ID"] for r in csv.DictReader(open(SHDB_DIR / "AdditionalData.csv", encoding="utf-8-sig"))}
    rek = rekaman_beranotasi()
    sub = {r: meta[r] for r in rek}
    n = {}
    for s in sub.values():
        n[s] = n.get(s, 0) + 1
    return sub, n


def proses_shdb(rec, kanal=0):
    """Satu rekaman SHDB-AF → window terpilih (RR, EKG, fitur atrium, label) → cache/v12/shdb/<rec>_k<kanal>.npz."""
    tujuan = CACHE / "shdb" / f"{rec}_k{kanal}.npz"
    if tujuan.exists():
        return rec, "sudah ada"
    I10 = _modul()
    hc, E6 = I10.hrv_core, I10.E6
    sub, n_rek = subjek_shdb()
    x, fs = baca_dat(SHDB_DIR / rec, kanal)
    s_ann, _, aux = baca_atr(SHDB_DIR / rec)
    events = [(int(s), a) for s, a in zip(s_ann, aux) if a.startswith("(") or a == ")"]
    x50, xf = E6.preprocess_record(np.nan_to_num(x), fs)
    peaks = hc.detect_rpeaks(x50, hc.FS_DEVICE)
    pk_int = np.round(np.asarray(peaks)).astype(np.int64)           # indeks sampel untuk fitur atrium
    rows = I10.label_windows(peaks, fs, events, stride=SEL["stride"])
    y = np.array([r[2] for r in rows], np.int64)
    hr = np.array([60000.0 / r[1].mean() for r in rows]) if rows else np.zeros(0)
    bagi = n_rek[sub[rec]]
    rng = np.random.default_rng(int(hashlib.sha256(f"{rec}-{kanal}".encode()).hexdigest()[:8], 16) + SEED)
    keep, cepat = [], []
    for c in range(3):
        idx = np.flatnonzero(y == c)
        k = min(math.ceil(SEL["cap"][c] / bagi), idx.size)
        pick = rng.choice(idx, k, replace=False) if k else np.zeros(0, int)
        keep += pick.tolist(); cepat += [False] * len(pick)
        if c == 0:
            sisa = np.setdiff1d(idx[hr[idx] >= SEL["hr_cepat"]], pick)
            kf = min(math.ceil(SEL["cap_cepat"] / bagi), sisa.size)
            extra = rng.choice(sisa, kf, replace=False) if kf else np.zeros(0, int)
            keep += extra.tolist(); cepat += [True] * len(extra)
    urut = np.argsort(keep, kind="stable")
    keep, cepat = np.asarray(keep, int)[urut], np.asarray(cepat, bool)[urut]
    peaks_t = peaks / hc.FS_DEVICE
    X = np.zeros((len(keep), hc.MAX_BEATS, 2), np.float32)
    M = np.zeros((len(keep), hc.MAX_BEATS), np.float32)
    ecg = np.zeros((len(keep), N_SAMP), np.float16)
    fit = np.zeros((len(keep), len(FITUR)), np.float32)
    pola = np.zeros(len(keep), np.int8)
    tidak_sejajar = 0
    for j, i in enumerate(keep):
        t0, ibi, _ = rows[i]
        X[j], M[j] = hc.build_sequence(ibi)
        s = int(round(t0 * FS))
        w = xf[s:s + N_SAMP]
        ecg[j] = E6.normalize_window(np.pad(w, (0, N_SAMP - w.size)) if w.size < N_SAMP else w).astype(np.float16)
        fit[j] = fitur_atrium(xf, pk_int, t0)
        pola[j] = pola_rr(ibi)
        n_pk = np.sum((peaks_t[1:] >= t0) & (peaks_t[1:] < t0 + hc.WINDOW_SEC))       # bukti sejajar (cara v6)
        tidak_sejajar += int(min(n_pk, hc.MAX_BEATS) != int(M[j].sum()))
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tujuan, X=X, M=M, ecg=ecg, fitur=fit, y=y[keep], hr=hr[keep].astype(np.float32), pola=pola,
                        t=np.array([rows[i][0] for i in keep], np.float32), cepat=cepat, subjek=sub[rec],
                        kandidat=np.bincount(y, minlength=3), tidak_sejajar=tidak_sejajar, n_peaks=len(peaks))
    return rec, f"{len(keep)} window (kandidat {np.bincount(y, minlength=3).tolist()}), tidak sejajar {tidak_sejajar}"


def bangun_shdb(kanal=0, workers=2, log=print):
    """Semua rekaman beranotasi (yang sinyalnya sudah ada). 2 proses: RAM PC 15 GB (pelajaran OOM 2 November 2026)."""
    rek = [r for r in rekaman_beranotasi() if (SHDB_DIR / f"{r}.dat").exists()]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, (r, status) in enumerate(ex.map(proses_shdb, rek, [kanal] * len(rek)), 1):
            log(f"[{(time.time() - t0) / 60:5.1f} mnt] {i}/{len(rek)} {r}: {status}")


# ── 5. fitur AFDB pada window dataset v2 ───────────────────────────────
def _proses_afdb(rec):
    I10 = _modul()
    hc, E6 = I10.hrv_core, I10.E6
    std = np.load(REPO / "cache/dataset_v2_detector.npz")
    recs, t0s = std["records"].astype(str), std["t_start"].astype(np.float64)
    idx = np.flatnonzero(recs == rec)
    x, fs = baca_dat(AFDB_DIR / rec, 0)                      # kanal 0, seperti v6
    x50, xf = E6.preprocess_record(np.nan_to_num(x), fs)
    peaks = hc.detect_rpeaks(x50, hc.FS_DEVICE)
    pk_int = np.round(np.asarray(peaks)).astype(np.int64)           # indeks sampel untuk fitur atrium
    fit = np.stack([fitur_atrium(xf, pk_int, t0s[i]) for i in idx])
    ecg = np.zeros((idx.size, N_SAMP), np.float16)
    for j, i in enumerate(idx):
        s = int(round(t0s[i] * FS))
        w = xf[s:s + N_SAMP]
        ecg[j] = E6.normalize_window(np.pad(w, (0, N_SAMP - w.size)) if w.size < N_SAMP else w).astype(np.float16)
    return rec, idx, fit, ecg


def bangun_afdb(workers=2, log=print):
    """Fitur atrium untuk setiap window cache/dataset_v2_detector.npz; potongan EKG harus sama dengan cache/v6/ecg_windows.npz."""
    tujuan = CACHE / "afdb_fitur.npz"
    if tujuan.exists():
        log(f"sudah ada: {tujuan}")
        return
    std = np.load(REPO / "cache/dataset_v2_detector.npz")
    recs = std["records"].astype(str)
    v6 = np.load(REPO / "cache/v6/ecg_windows.npz")["ecg"]
    fit = np.full((recs.size, len(FITUR)), np.nan, np.float32)
    beda = 0
    with ProcessPoolExecutor(workers) as ex:
        for rec, idx, f, ecg in ex.map(_proses_afdb, sorted(set(recs))):
            fit[idx] = f
            beda += int((ecg != v6[idx]).any(1).sum())
            log(f"{rec}: {idx.size} window")
    assert beda == 0, f"{beda} potongan EKG beda dengan cache v6 (front-end tidak sama)"
    X, M = std["X"], std["M"]
    pola = np.array([pola_rr(X[i, :int(M[i].sum()), 0]) for i in range(recs.size)], np.int8)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tujuan, fitur=fit, pola=pola)
    log(f"AFDB: {recs.size} window, potongan EKG = cache v6 (bit demi bit) → {tujuan}")


# ── 5b. revisi 1: fitur dihitung ulang pada window yang sama ───────────
def _fitur_r1_rekaman(arg):
    """Satu rekaman: fitur revisi 0 (harus sama dengan yang tersimpan, bukti window sama) dan revisi 1."""
    dataset, rec, t0s = arg
    I10 = _modul()
    hc, E6 = I10.hrv_core, I10.E6
    x, fs = baca_dat((AFDB_DIR if dataset == "AFDB" else SHDB_DIR) / rec, 0)
    x50, xf = E6.preprocess_record(np.nan_to_num(x), fs)
    pk = np.round(np.asarray(hc.detect_rpeaks(x50, hc.FS_DEVICE))).astype(np.int64)
    f0 = np.stack([fitur_atrium(xf, pk, t) for t in t0s])
    f1 = np.stack([fitur_atrium(xf, pk, t, revisi=1) for t in t0s])
    return dataset, rec, f0, f1


def hitung_fitur_r1(workers=2, log=print):
    """Fitur revisi 1 untuk semua window AFDB dan SHDB-AF yang sudah dipakai gerbang pertama → cache/v12/fitur_r1.npz."""
    tujuan = CACHE / "fitur_r1.npz"
    if tujuan.exists():
        log(f"sudah ada: {tujuan}")
        return
    D = muat_fitur()
    out = {}
    t_mulai = time.time()
    for nama, d in D.items():
        rek = d["rekaman"]
        f1 = np.full_like(d["fitur"], np.nan)
        tugas = [(nama, r, d["t"][rek == r].astype(np.float64)) for r in sorted(set(rek))]
        with ProcessPoolExecutor(workers) as ex:
            for i, (_, r, f0, f) in enumerate(ex.map(_fitur_r1_rekaman, tugas), 1):
                sel = rek == r
                assert np.array_equal(f0, d["fitur"][sel], equal_nan=True), f"{nama} {r}: fitur revisi 0 berubah (window beda)"
                f1[sel] = f
                log(f"[{(time.time() - t_mulai) / 60:5.1f} mnt] {nama} {i}/{len(tugas)} {r}: {sel.sum()} window")
        out[nama] = f1
    np.savez_compressed(tujuan, afdb=out["AFDB"], shdb=out["SHDB-AF"])
    log(f"fitur revisi 1 → {tujuan} (fitur revisi 0 dihitung ulang dan sama persis: window identik)")


# ── 6. gerbang (RANCANGAN 7.2 dan 7.3) ─────────────────────────────────
GERBANG_FITUR = ("df_hz", "organisasi")
GERBANG_AUROC = 0.60


def muat_fitur(revisi=0):
    """{dataset: dict(fitur, y, pola, pasien, sumber)}; window Normal ≥ 100 bpm tambahan SHDB-AF ditandai `tambahan`.
    revisi=1: fitur dari cache/v12/fitur_r1.npz (window sama)."""
    std = np.load(REPO / "cache/dataset_v2_detector.npz")
    a = np.load(CACHE / "afdb_fitur.npz")
    out = {"AFDB": dict(fitur=a["fitur"], y=std["y"].astype(np.int64), pola=a["pola"], pasien=std["records"].astype(str),
                        tambahan=np.zeros(std["y"].size, bool), t=std["t_start"].astype(np.float32),
                        rekaman=std["records"].astype(str))}
    bag = {k: [] for k in ("fitur", "y", "pola", "pasien", "tambahan", "t", "rekaman")}
    for p in sorted((CACHE / "shdb").glob("*_k0.npz")):
        z = np.load(p)
        n = z["y"].size
        bag["fitur"].append(z["fitur"]); bag["y"].append(z["y"]); bag["pola"].append(z["pola"])
        bag["pasien"].append(np.full(n, str(z["subjek"]))); bag["tambahan"].append(z["cepat"]); bag["t"].append(z["t"])
        bag["rekaman"].append(np.full(n, p.stem.split("_")[0]))
    out["SHDB-AF"] = {k: np.concatenate(v) for k, v in bag.items()}
    if revisi == 1:
        z = np.load(CACHE / "fitur_r1.npz")
        for nama, kunci in (("AFDB", "afdb"), ("SHDB-AF", "shdb")):
            assert z[kunci].shape == out[nama]["fitur"].shape, nama
            out[nama]["fitur"] = z[kunci]
    return out


def _auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not pos.size or not neg.size:
        return np.nan
    return float(roc_auc_score(np.r_[np.ones(pos.size), np.zeros(neg.size)], np.r_[pos, neg]))


def gerbang(revisi=0, log=print):
    """AUROC tiap fitur (AFL positif) per dataset; keputusan lanjut/berhenti untuk fitur gerbang → cache/v12/gerbang.json."""
    D = muat_fitur(revisi)
    hasil = {}
    for nama, d in D.items():
        m = ~d["tambahan"]
        f, y, pola = d["fitur"][m], d["y"][m], d["pola"][m]
        r = {"n": np.bincount(y, minlength=3).tolist(), "pasien_afl": int(np.unique(d["pasien"][m][y == 2]).size), "auroc": {}}
        for i, fn in enumerate(FITUR):
            afl = f[y == 2, i]
            r["auroc"][fn] = {"vs_afib": _auroc(afl, f[y == 1, i]), "vs_normal": _auroc(afl, f[y == 0, i]),
                              "afl_sulit_vs_afib": _auroc(f[(y == 2) & (pola != 0), i], f[y == 1, i]),
                              "median": [float(np.nanmedian(f[y == c, i])) for c in range(3)]}
        hasil[nama] = r
    keputusan = {}
    for fn in GERBANG_FITUR:
        for banding in ("vs_afib", "vs_normal"):
            a, b = (hasil[n]["auroc"][fn][banding] for n in ("AFDB", "SHDB-AF"))
            searah = (a - 0.5) * (b - 0.5) > 0
            kuat = max(a, 1 - a) >= GERBANG_AUROC and max(b, 1 - b) >= GERBANG_AUROC
            keputusan[f"{fn} {banding}"] = dict(afdb=a, shdb=b, searah=bool(searah), kuat=bool(kuat), lolos=bool(searah and kuat))
    lolos = all(v["lolos"] for v in keputusan.values())
    out = {"revisi": revisi, "hasil": hasil, "keputusan": keputusan, "lolos": lolos, "ambang": GERBANG_AUROC}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / ("gerbang.json" if revisi == 0 else f"gerbang_r{revisi}.json")).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for nama, r in hasil.items():
        log(f"{nama}: window {r['n']}, pasien AFL {r['pasien_afl']}")
    log(f"{'fitur':16s} " + " ".join(f"{n + ' ' + b:>22s}" for n in ("AFDB", "SHDB-AF") for b in ("vs AFIB", "vs Normal"))
        + "   AFL sulit vs AFIB (AFDB / SHDB)")
    for fn in FITUR:
        log(f"{fn:16s} " + " ".join(f"{hasil[n]['auroc'][fn][b]:22.3f}" for n in ("AFDB", "SHDB-AF") for b in ("vs_afib", "vs_normal"))
            + f"   {hasil['AFDB']['auroc'][fn]['afl_sulit_vs_afib']:.3f} / {hasil['SHDB-AF']['auroc'][fn]['afl_sulit_vs_afib']:.3f}")
    for k, v in keputusan.items():
        log(f"gerbang {k:22s} AFDB {v['afdb']:.3f} SHDB {v['shdb']:.3f} searah={v['searah']} kuat={v['kuat']} → "
            f"{'LOLOS' if v['lolos'] else 'GAGAL'}")
    log(f"KEPUTUSAN GERBANG: {'LANJUT' if lolos else 'BERHENTI (lihat RANCANGAN 7.2)'}")
    return out


def contoh_visual(n_per_kelas=3, revisi=0, log=print):
    """PNG contoh potongan EKG, sinyal sisa, dan spektrum per kelas per dataset → cache/v12/gerbang/ (dilihat sebelum lanjut)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal as sps
    I10 = _modul()
    hc, E6 = I10.hrv_core, I10.E6
    D = muat_fitur(revisi)
    rng = np.random.default_rng(SEED)
    keluar = CACHE / "gerbang"
    keluar.mkdir(parents=True, exist_ok=True)
    for nama, d in D.items():
        m = ~d["tambahan"]
        fig, axes = plt.subplots(3 * n_per_kelas, 2, figsize=(11, 2.0 * 3 * n_per_kelas),
                                 gridspec_kw=dict(width_ratios=[3, 1]))
        baris = 0
        for c in range(3):
            calon = np.flatnonzero(m & (d["y"] == c))
            for i in rng.choice(calon, min(n_per_kelas, calon.size), replace=False):
                rek = d["rekaman"][i]
                base = (AFDB_DIR if nama == "AFDB" else SHDB_DIR) / rek
                x, fs = baca_dat(base, 0)
                x50, xf = E6.preprocess_record(np.nan_to_num(x), fs)
                pk = np.round(np.asarray(hc.detect_rpeaks(x50, hc.FS_DEVICE))).astype(np.int64)
                s = int(round(float(d["t"][i]) * FS))
                seg, sisa, _ = (sinyal_sisa_r1 if revisi == 1 else sinyal_sisa)(xf, pk, s)
                tt = np.arange(N_SAMP) / FS
                ax = axes[baris, 0]
                ax.plot(tt, seg, color="#8A97A6", lw=.8, label="EKG terfilter")
                if sisa is not None:
                    ax.plot(tt, sisa, color="#2a78d6", lw=.9, label="sisa (aktivitas atrium)")
                    f, P = sps.welch(sisa - sisa.mean(), fs=FS, nperseg=256, noverlap=128, nfft=1024)
                    axes[baris, 1].semilogy(f, P, color="#eb6834")
                    axes[baris, 1].axvspan(3, 12, color="#EEF4FC")
                fi = d["fitur"][i]
                ax.set_title(f"{nama} {KELAS[c]} | {rek} t={float(d['t'][i]):.0f} s | DF {fi[0]:.1f} Hz, organisasi {fi[1]:.2f}, "
                             f"P {fi[6]:.2f}, RR/FCL {fi[7]:.2f}", fontsize=8, loc="left")
                ax.legend(fontsize=6, loc="upper right")
                axes[baris, 1].set_xlim(0, 20)
                baris += 1
        fig.tight_layout()
        p = keluar / f"contoh_{nama.replace('-', '').lower()}{'' if revisi == 0 else f'_r{revisi}'}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        log(f"{p}")


if __name__ == "__main__":
    perintah = sys.argv[1:]
    cetak = lambda m: print(m, flush=True)  # noqa: E731
    if perintah == ["unduh"]:
        unduh_shdb(log=cetak)
    elif perintah == ["bukti"]:
        bukti_pembaca(log=cetak)
    elif perintah == ["shdb"]:
        bangun_shdb(log=cetak)
    elif perintah == ["afdb"]:
        bangun_afdb(log=cetak)
    elif perintah == ["gerbang"]:
        gerbang(log=cetak)
        contoh_visual(log=cetak)
    elif perintah == ["revisi1"]:
        hitung_fitur_r1(log=cetak)
        gerbang(revisi=1, log=cetak)
        contoh_visual(revisi=1, log=cetak)
    else:
        sys.exit("pemakaian: python atrium.py unduh | bukti | shdb | afdb | gerbang | revisi1")
