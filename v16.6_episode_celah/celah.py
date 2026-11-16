"""
celah.py — logika v16.6 (pra-registrasi di RANCANGAN.md, dikunci 10 November 2026 sebelum sinyal data validasi dibaca).

Aturan episode AF dengan toleransi celah (g window) dan durasi minimum. Tahap A memilih (min_span, g) untuk D-1 di data v16.5 dengan
kriteria yang dikunci; tahap B menguji konfigurasi itu sekali di SHDB-AF, afpdb, incartdb (dan svdb, deskriptif). Rantai prediksi per
rekaman = v16.4 `uji_baru.proses_rekaman`; metrik episode = v16.5 `episode.metrik` (diimpor, tidak disalin).
"""
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "v16.5_episode_30s")]
import episode as E  # noqa: E402
from episode import A12, R163, U164, hrv_core  # noqa: E402

CACHE = REPO / "cache/v16.6"
W = hrv_core.WINDOW_SEC
LANGKAH = 5.0
GRID_SPAN = (30, 45, 60, 90)
GRID_G = (0, 1, 2)
TARGET_PALSU, TARGET_SE = 1.0, 0.90
AF_LABEL = {"(AFIB", "(AFL", "(WPWAF"}
AKAR = {"SHDB-AF": REPO / "data/shdb_af/1.0.1", "afpdb": REPO / "data/afpdb", "incartdb": REPO / "data/incartdb",
        "svdb": REPO / "data/svdb"}
VALIDASI = ("SHDB-AF", "afpdb", "incartdb", "svdb")
SEBELUM_LABEL_NON_AF = ("incartdb", "svdb")          # RANCANGAN amandemen A


# ── aturan ─────────────────────────────────────────────────────────────
def aturan_celah(rec, start, af, langkah, min_span, g):
    """Alarm per window: run window AF dengan jarak awal antar window AF ≤ langkah·(g+1) + 0,5 s; rentang ≥ min_span → semua window
    di dalam rentang run (termasuk celah) = alarm."""
    af = np.asarray(af, bool)
    out = np.zeros(af.size, bool)
    batas = langkah * (g + 1) + 0.5
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        idx = idx[np.argsort(start[idx], kind="stable")]
        s = start[idx]
        p = np.flatnonzero(af[idx])
        if not p.size:
            continue
        putus = np.flatnonzero(np.diff(s[p]) > batas)
        awal = np.r_[0, putus + 1]
        akhir = np.r_[putus, p.size - 1]
        for a, b in zip(awal, akhir):
            if s[p[b]] + W - s[p[a]] >= min_span:
                out[idx[p[a]:p[b] + 1]] = True
    return out


def se_unit(rec, start, y2, alarm, langkah, unit):
    """{unit: [episode referensi ≥ 30 s yang terdeteksi, jumlah episode referensi ≥ 30 s]} (definisi = episode.metrik)."""
    out = {}
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        m = m[np.argsort(start[m], kind="stable")]
        s = start[m]
        R = E._run(s, y2[m] == 1, langkah)
        D = E._run(s, alarm[m], langkah)
        a = out.setdefault(str(unit[m][0]), [0, 0])
        for e in R:
            if e[1] - e[0] >= E.PRIMER:
                a[1] += 1
                a[0] += int(E._overlap(e[None], D) > 0)
    return out


def boot_rasio(ua, ub, n_boot=2000, seed=2026):
    """Δ (a − b) rasio Σpembilang/Σpenyebut, bootstrap unit berpasangan. ua/ub: {unit: [pembilang, penyebut]} (penyebut sama)."""
    u = sorted(set(ua) | set(ub))
    na = np.array([ua.get(k, [0, 0])[0] for k in u], float)
    nb = np.array([ub.get(k, [0, 0])[0] for k in u], float)
    d = np.array([(ua.get(k) or ub.get(k))[1] for k in u], float)
    idx = np.random.default_rng(seed).integers(0, len(u), (n_boot, len(u)))
    den = d[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ra, rb = na[idx].sum(1) / den, nb[idx].sum(1) / den
    dd = ra - rb
    return dict(delta=float(na.sum() / d.sum() - nb.sum() / d.sum()), ci=np.nanpercentile(dd, [2.5, 97.5]).tolist(),
                p_lebih=float(np.nanmean(dd > 0)), p_kurang=float(np.nanmean(dd < 0)), titik_a=float(na.sum() / d.sum()),
                ci_a=np.nanpercentile(ra, [2.5, 97.5]).tolist(), titik_b=float(nb.sum() / d.sum()))


def nilai(d, af, min_span, g):
    """Metrik episode (episode.metrik) + Se per unit untuk satu keputusan per window. af: (n,) atau (R, n) → rata-rata ulangan."""
    reps = af if af.ndim == 2 else af[None]
    ms, su, pu = [], None, None
    for k, a in enumerate(reps):
        al = aturan_celah(d["rec"], d["start"], a, d["langkah"], min_span, g)
        m = E.metrik(d["rec"], d["start"], d["y2"], al, d["langkah"], d["unit"])
        if k == 0:
            pu, su = m.pop("palsu_unit"), se_unit(d["rec"], d["start"], d["y2"], al, d["langkah"], d["unit"])
        else:
            m.pop("palsu_unit")
        ms.append(m)
    rata = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
    return rata | {"palsu_unit": pu, "se_unit": su}


# ── tahap A ────────────────────────────────────────────────────────────
def cek_paritas(D):
    """g = 0, 30 s harus identik dengan aturan v16.5 untuk D-1 di semua data pengembangan."""
    for ds, d in D.items():
        a = E.aturan(d["rec"], d["start"], d["d1"], d["langkah"], 30)
        b = aturan_celah(d["rec"], d["start"], d["d1"], d["langkah"], 30, 0)
        assert np.array_equal(a, b), ds
    return True


def grid(D, log=print):
    """{(span, g): {dataset: metrik D-1}}."""
    out = {}
    for sp in GRID_SPAN:
        for g in GRID_G:
            out[(sp, g)] = {ds: nilai(d, d["d1"], sp, g) for ds, d in D.items()}
            h = out[(sp, g)]
            log(f"  span {sp} g {g}: palsu NSRDB {h['NSRDB']['palsu_per_jam']:.3f} EDB {h['EDB']['palsu_per_jam']:.3f} latihan "
                f"{h['Latihan fisik']['palsu_per_jam']:.3f} | Se30 min " + f"{min(h[x]['se_episode_30'] for x in E_AF):.3f}")
    return out


E_AF = ("AFDB", "LTAFDB", "MIT-BIH", "CPSC 2021")
E_NEG = ("NSRDB", "EDB", "Latihan fisik")


def pilih(G):
    """Pemilihan dikunci (RANCANGAN bagian 3)."""
    lolos = [k for k, h in G.items() if all(h[x]["palsu_per_jam"] <= TARGET_PALSU for x in E_NEG)]
    if not lolos:
        return min(G, key=lambda k: G[k]["NSRDB"]["palsu_per_jam"]), False
    se = {k: min(G[k][x]["se_episode_30"] for x in E_AF) for k in lolos}
    top = max(se.values())
    calon = [k for k in lolos if se[k] >= top - 0.005]
    return min(calon, key=lambda k: G[k]["NSRDB"]["palsu_per_jam"] + G[k]["EDB"]["palsu_per_jam"]), True


# ── tahap B: data validasi ─────────────────────────────────────────────
def _atr_opsional(base):
    if Path(str(base) + ".atr").exists():
        return A12.baca_atr(base)
    return np.zeros(0, np.int64), np.zeros(0, np.int64), []


# afpdb tidak punya .atr; proses_rekaman v16.4 hanya memakai .atr untuk kode artefak NSRDB
U164.A12 = types.SimpleNamespace(baca_dat=A12.baca_dat, baca_atr=_atr_opsional, baca_hea=A12.baca_hea)
for _ds, _p in AKAR.items():
    U164.AKAR[_ds] = _p


def rekaman(ds):
    a = AKAR[ds]
    if ds == "SHDB-AF":
        return sorted(p.stem for p in a.glob("*.atr"))
    if ds == "afpdb":
        return [f"{h}{i:02d}" for h in "np" for i in range(1, 51) if (a / f"{h}{i:02d}.hea").exists()]
    return (a / "RECORDS").read_text().split()


def unit_rekaman(ds):
    if ds == "SHDB-AF":
        return {r["Data_ID"]: r["Subject_ID"] for r in csv.DictReader(open(AKAR[ds] / "AdditionalData.csv", encoding="utf-8-sig"))}
    if ds == "incartdb":
        out = {}
        for r in rekaman(ds):
            t = (AKAR[ds] / f"{r}.hea").read_text().splitlines()
            out[r] = next(b.split("patient")[1].strip() for b in t if b.startswith("#") and "patient" in b)
        return out
    return {r: r for r in rekaman(ds)}


def interval_ritme(ds, rec):
    """[(awal, akhir, af)] dari label ritme (aux diawali '('); berlaku sampai label ritme berikutnya. afpdb: tanpa label (None)."""
    base = AKAR[ds] / rec
    if not Path(str(base) + ".atr").exists():
        return None
    fs, n = A12.baca_hea(base)[:2]
    s, _, aux = A12.baca_atr(base)
    rr = [(si / fs, a.strip("\x00").strip()) for si, a in zip(s, aux) if a.strip("\x00").strip().startswith("(")]
    return [(t0, rr[i + 1][0] if i + 1 < len(rr) else n / fs, lab in AF_LABEL) for i, (t0, lab) in enumerate(rr)]


def label_window(ds, rec, start):
    """y (0/1) dan keep: window seluruhnya di dalam satu interval ritme; yang melintasi batas atau sebelum label pertama dibuang."""
    iv = interval_ritme(ds, rec)
    if iv is None:                                   # afpdb: semua non-AF
        return np.zeros(start.size, int), np.ones(start.size, bool)
    y = np.zeros(start.size, int)
    keep = np.zeros(start.size, bool)
    if ds in SEBELUM_LABEL_NON_AF:                   # amandemen A: sebelum label ritme pertama = non-AF
        t1 = iv[0][0] if iv else np.inf
        keep |= start + W <= t1
    for a, b, af in iv:
        inside = (start >= a) & (start + W <= b)
        keep |= inside
        if af:
            y[inside] = 1
    return y, keep


def jalankan(ds, bagian=0, n_bagian=1, log=print):
    d = CACHE / ds
    d.mkdir(parents=True, exist_ok=True)
    daftar = rekaman(ds)
    alat = None
    for j, rec in enumerate(daftar):
        if j % n_bagian != bagian:
            continue
        p = d / f"{rec}.npz"
        if p.exists():
            continue
        alat = alat or R163.Alat()
        z = U164.proses_rekaman(ds, rec, alat)
        np.savez(p.with_suffix(".tmp.npz"), **z)
        p.with_suffix(".tmp.npz").replace(p)
        log(f"  {ds}: {rec} ({j + 1}/{len(daftar)})")


def muat(ds):
    unit = unit_rekaman(ds)
    bag = {k: [] for k in ("rec", "start", "y2", "d1", "p31", "unit", "hr")}
    info = dict(rekaman=0, kosong=0, window_lolos=0, dibuang_label=0)
    for rec in rekaman(ds):
        z = np.load(CACHE / ds / f"{rec}.npz")
        info["rekaman"] += 1
        if bool(z["kosong"]):
            info["kosong"] += 1
            continue
        st = z["start"].astype(float)
        y, keep = label_window(ds, rec, st)
        info["window_lolos"] += int(st.size)
        info["dibuang_label"] += int((~keep).sum())
        for k, v in (("rec", np.full(st.size, rec)), ("start", st), ("y2", y), ("d1", z["d1"]), ("p31", z["p31"]),
                     ("unit", np.full(st.size, unit[rec])), ("hr", z["hr"])):
            bag[k].append(v[keep])
    Z = {k: np.concatenate(v) for k, v in bag.items()}
    info["window"] = int(Z["start"].size)
    info["window_AF"] = int(Z["y2"].sum())
    info["unit"] = int(np.unique(Z["unit"]).size)
    info["jam"] = float(Z["start"].size * LANGKAH / 3600)
    return dict(rec=Z["rec"], start=Z["start"], y2=Z["y2"], unit=Z["unit"], langkah=LANGKAH, hr=Z["hr"],
                d1=Z["d1"] > R163.AMBANG, v31=(Z["p31"].argmax(1) != 0)), info


def simpan_json(obj, path):
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        return o
    Path(path).write_text(json.dumps(conv(obj), indent=1, ensure_ascii=False), encoding="utf-8")
