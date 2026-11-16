"""
bradikardia.py — logika v16.9 (pra-registrasi di RANCANGAN.md, dikunci 10 November 2026 sebelum alarm per ritme/pita dihitung).

Evaluasi saja, dari cache: S1 = D-1 + aturan episode AF ≥ 30 s (v16.5) dan R = paket v3.1 + aturan yang sama, dinilai pada sinus
bradikardia berlabel (LTAFDB) dan pada window non-AF berdetak lambat di sepuluh dataset. Pemuat data, aturan, dan definisi episode
diimpor dari v16.3–v16.7 (tidak disalin).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "v16.7_konfirmasi_wearable")]
import konfirmasi as K  # noqa: E402
from konfirmasi import C, E, R163, U164  # noqa: E402

CACHE = REPO / "cache/v16.9"
W = E.W
LANGKAH = 5.0
PITA = ((0, 50, "< 50"), (50, 60, "50–60"), (60, 80, "60–80"), (80, 100, "80–100"), (100, np.inf, "≥ 100"))
LAMBAT, ACUAN = (0, 60), (60, 100)
DATASET = ("LTAFDB", "MIT-BIH", "CPSC 2021", "NSRDB", "EDB", "SHDB-AF", "afpdb", "incartdb", "svdb", "Icentia11k")
N_BOOT, SEED = 2000, 2026


# ── data (semua dari cache) ───────────────────────────────────────────
def _hr_cache(folder, d):
    """Detak window dari npz cache per rekaman, dicocokkan lewat (rekaman, awal)."""
    hr = np.full(d["rec"].size, np.nan)
    for r in np.unique(d["rec"]):
        z = np.load(folder / f"{r}.npz")
        m = np.flatnonzero(d["rec"] == r)
        k = np.searchsorted(z["start"], d["start"][m])
        assert np.array_equal(z["start"][k], d["start"][m])
        hr[m] = z["hr"][k]
    return hr


def muat(ds):
    """dict(rec, start, y2, unit, hr, d1, v31, [ritme]) dengan keputusan per window 1-D (bool)."""
    if ds in ("LTAFDB", "MIT-BIH"):
        Z, _ = R163.muat(ds)
        return dict(rec=Z["rekaman"], start=Z["start"].astype(float), y2=Z["y2"].astype(int), unit=Z["subjek"], hr=Z["hr"],
                    d1=Z["d1"] > R163.AMBANG, v31=Z["p31"].argmax(1) != 0, ritme=Z["ritme"])
    if ds == "CPSC 2021":
        d = E._cpsc()
        st, hr = [], []                                   # urutan sama dengan U162.bangun_jendela; hanya kolom start/hr/ok
        for rec in E.U162.K15.rekaman_cpsc():
            z = np.load(E.U162.CACHE / "rekaman_resmi" / (rec.replace("/", "__") + ".npz"))
            if bool(z["kosong"]):
                continue
            st.append(z["start"][z["ok"]])
            hr.append(z["hr"][z["ok"]])
        assert np.array_equal(np.concatenate(st).astype(float), d["start"])
        return dict(rec=d["rec"], start=d["start"], y2=d["y2"], unit=d["unit"], hr=np.concatenate(hr).astype(float), d1=d["d1"],
                    v31=d["v31"][0])
    if ds == "NSRDB":
        Z, _ = U164.muat("NSRDB")
        return dict(rec=Z["rekaman"], start=Z["start"].astype(float), y2=Z["y2"].astype(int), unit=Z["subjek"], hr=Z["hr"],
                    d1=Z["d1"] > R163.AMBANG, v31=Z["p31"].argmax(1) != 0)
    if ds == "EDB":
        d = E._edb()
        return dict(rec=d["rec"], start=d["start"], y2=d["y2"], unit=d["unit"], hr=_hr_cache(E.CACHE / "EDB", d), d1=d["d1"],
                    v31=d["v31"][0])
    if ds == "Icentia11k":
        d, _ = K.muat("Icentia")
    else:
        d, _ = C.muat(ds)
    return dict(rec=d["rec"], start=d["start"], y2=d["y2"], unit=d["unit"], hr=d["hr"], d1=d["d1"], v31=d["v31"])


def sistem(d):
    return {"S1": E.aturan(d["rec"], d["start"], d["d1"], LANGKAH), "R": E.aturan(d["rec"], d["start"], d["v31"], LANGKAH)}


# ── episode ───────────────────────────────────────────────────────────
def _runs_idx(s, on):
    """Indeks (relatif) run window aktif berurutan, kriteria sama dengan episode._run."""
    out, i = [], 0
    while i < len(s):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(s) and on[j + 1] and s[j + 1] - s[j] <= LANGKAH + 0.5:
            j += 1
        out.append(np.arange(i, j + 1))
        i = j + 1
    return out


def episode(d, alarm):
    """Episode deteksi palsu dan episode AF referensi per rekaman.

    Kembalian: palsu = list dict(rec, unit, idx global, a, b), ref = {rec: array interval AF [a, b)} dan ref_idx = list (rec, idx).
    """
    palsu, ref, ref_idx = [], {}, []
    for r in np.unique(d["rec"]):
        m = np.flatnonzero(d["rec"] == r)
        m = m[np.argsort(d["start"][m], kind="stable")]
        s = d["start"][m]
        Rr = [m[i] for i in _runs_idx(s, d["y2"][m] == 1)]
        iv = np.array([(d["start"][g[0]], d["start"][g[-1]] + W) for g in Rr], float).reshape(-1, 2)
        ref[r] = iv
        ref_idx += [(r, g) for g in Rr]
        for g in _runs_idx(s, alarm[m]):
            gi = m[g]
            a, b = d["start"][gi[0]], d["start"][gi[-1]] + W
            if E._overlap(np.array([[a, b]]), iv) == 0:
                palsu.append(dict(rec=r, unit=d["unit"][gi[0]], idx=gi, a=a, b=b))
    return palsu, ref, ref_idx


def cek_paritas(d, alarm, palsu):
    """Jumlah episode palsu sama dengan episode.metrik (definisi v16.5)."""
    m = E.metrik(d["rec"], d["start"], d["y2"], alarm, LANGKAH, d["unit"])
    n = sum(v[0] for v in m["palsu_unit"].values())
    assert n == len(palsu), (n, len(palsu))
    return m


# ── statistik ─────────────────────────────────────────────────────────
def boot(ua, ub=None, n_boot=N_BOOT, seed=SEED):
    """Rasio Σnum/Σden per sampel unit (berpasangan bila ub ada). ua/ub: {unit: [num, den]}."""
    u = sorted(set(ua) | set(ub or {}))
    A = np.array([ua.get(k, [0, 0]) for k in u], float)
    idx = np.random.default_rng(seed).integers(0, len(u), (n_boot, len(u)))
    with np.errstate(invalid="ignore", divide="ignore"):
        ra = A[idx, 0].sum(1) / A[idx, 1].sum(1)
        out = dict(nilai=float(A[:, 0].sum() / A[:, 1].sum()) if A[:, 1].sum() else float("nan"),
                   ci=np.nanpercentile(ra, [2.5, 97.5]).tolist(), n_unit=int((A[:, 1] > 0).sum()))
        if ub is not None:
            B = np.array([ub.get(k, [0, 0]) for k in u], float)
            rb = B[idx, 0].sum(1) / B[idx, 1].sum(1)
            dd = ra - rb
            out.update(nilai_b=float(B[:, 0].sum() / B[:, 1].sum()) if B[:, 1].sum() else float("nan"),
                       delta=out["nilai"] - (float(B[:, 0].sum() / B[:, 1].sum()) if B[:, 1].sum() else float("nan")),
                       ci_delta=np.nanpercentile(dd, [2.5, 97.5]).tolist(), p_lebih=float(np.nanmean(dd > 0)),
                       ci_b=np.nanpercentile(rb, [2.5, 97.5]).tolist())
    return out


def per_unit(unit, num_mask, den_mask):
    """{unit: [Σnum, Σden]} dari mask window."""
    out = {}
    for u in np.unique(unit[den_mask]):
        m = unit == u
        out[str(u)] = [float((num_mask & m).sum()), float((den_mask & m).sum())]
    return out


def jam(n_window):
    return n_window * LANGKAH / 3600.0


def pita(hr, lo, hi):
    return (hr >= lo) & (hr < hi)


# ── P1/P2: sinus bradikardia berlabel ─────────────────────────────────
def ritme_episode(d, palsu):
    """Label ritme terbanyak di antara window dalam episode."""
    out = []
    for e in palsu:
        v, c = np.unique(d["ritme"][e["idx"]], return_counts=True)
        out.append(str(v[c.argmax()]))
    return out


def laju_ritme(d, palsu, rit, ritme):
    """{unit: [episode palsu ritme itu, jam window non-AF ritme itu]}."""
    den = (d["ritme"] == ritme) & (d["y2"] == 0)
    u = {k: [0.0, jam(v[1])] for k, v in per_unit(d["unit"], den, den).items()}
    for e, r in zip(palsu, rit):
        if r == ritme:
            u.setdefault(str(e["unit"]), [0.0, 0.0])[0] += 1
    return u


def jarak_ke_af(a, b, iv):
    """Menit dari interval [a, b) ke episode AF referensi terdekat (inf bila rekaman tanpa AF)."""
    if not len(iv):
        return np.inf
    g = np.maximum(0.0, np.maximum(iv[:, 0] - b, a - iv[:, 1]))
    return float(g.min() / 60.0)


KELAS_JARAK = ((0, 10, "< 10 menit"), (10, 60, "10–60 menit"), (60, np.inf, "> 60 menit"), (np.inf, np.inf, "rekaman tanpa AF"))


def kelas_jarak(x):
    x = np.asarray(x, float)
    out = {}
    for lo, hi, nama in KELAS_JARAK:
        out[nama] = float(np.mean(np.isinf(x))) if np.isinf(lo) else float(np.mean((x >= lo) & (x < hi)))
    return out


# ── P3/P4: detak lambat lintas dataset ────────────────────────────────
def fraksi_pita(d, alarm, lo, hi):
    den = (d["y2"] == 0) & pita(d["hr"], lo, hi)
    return per_unit(d["unit"], alarm & den, den)


def laju_pita(d, palsu, lo, hi):
    den = (d["y2"] == 0) & pita(d["hr"], lo, hi)
    u = {k: [0.0, jam(v[1])] for k, v in per_unit(d["unit"], den, den).items()}
    for e in palsu:
        h = np.nanmedian(d["hr"][e["idx"]])
        if lo <= h < hi:
            u.setdefault(str(e["unit"]), [0.0, 0.0])[0] += 1
    return u


def af_lambat(d, alarm_s1, ref_idx):
    """P4: proporsi window AF < 60 bpm, episode AF ≥ 30 s dengan median < 60, dan Se episode S1 di episode itu."""
    af = d["y2"] == 1
    n_af = int(af.sum())
    ep = [(r, g) for r, g in ref_idx if d["start"][g[-1]] + W - d["start"][g[0]] >= E.PRIMER]
    lambat = [(r, g) for r, g in ep if np.nanmedian(d["hr"][g]) < 60]
    kena = []
    for r, g in lambat:
        m = np.flatnonzero(d["rec"] == r)
        a, b = d["start"][g[0]], d["start"][g[-1]] + W
        on = m[alarm_s1[m]]
        iv = np.array([[d["start"][i], d["start"][i] + W] for i in on], float).reshape(-1, 2)
        kena.append(E._overlap(np.array([[a, b]]), iv) > 0)
    return dict(window_af=n_af, frac_window_lambat=float((af & (d["hr"] < 60)).sum() / n_af) if n_af else float("nan"),
                episode_30=len(ep), episode_lambat=len(lambat), frac_episode_lambat=len(lambat) / len(ep) if ep else float("nan"),
                se_s1_lambat=float(np.mean(kena)) if kena else float("nan"))


# ── eksplorasi setelah hasil (tidak dikunci): detak referensi anotasi LTAFDB ──
KODE_DETAK = set(range(1, 14)) | {25, 30, 31, 34, 35, 38}      # kode QRS MIT (wfdb isqrs)
LTAFDB = REPO / "data/ltafdb"


def detak_referensi(rec, start):
    """Per window [awal, awal + 10 s): jumlah detak anotasi, detak (bpm) dan CV RR dari anotasi, fraksi detak bukan normal (kode ≠ 1)."""
    fs = C.A12.baca_hea(LTAFDB / rec)[0]
    s, kode, _ = C.A12.baca_atr(LTAFDB / rec)
    s, kode = np.asarray(s), np.asarray(kode)
    q = np.isin(kode, list(KODE_DETAK))
    t, k = s[q] / fs, kode[q]
    lo, hi = np.searchsorted(t, start), np.searchsorted(t, start + W)
    out = np.full((start.size, 4), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a >= 2:
            rr = np.diff(t[a:b])
            out[i] = (b - a, 60.0 / rr.mean(), rr.std() / rr.mean(), np.mean(k[a:b] != 1))
    return out


def simpan_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float, ensure_ascii=False))
