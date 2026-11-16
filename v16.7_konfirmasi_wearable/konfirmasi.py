"""
konfirmasi.py — logika v16.7 (pra-registrasi di RANCANGAN.md, dikunci 10 November 2026 sebelum sinyal dibaca).

Uji konfirmasi S1 = D-1 + aturan episode AF ≥ 30 s (v16.5) di segmen utuh Icentia11k (wearable, satu lead) dan di nstdb (uji stres
derau). Seleksi Icentia dari pemindaian anotasi v10; parser .atr ritme Icentia dari v10 (`icentia.parse_atr_rhythm`, terbukti sama
dengan wfdb). Rantai prediksi = v16.4, aturan/metrik/bootstrap = v16.5/v16.6 (diimpor, tidak disalin).
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "v16.6_episode_celah"), str(REPO / "v10_icentia_wearable")]
import celah as C  # noqa: E402
from celah import E, R163, U164, hrv_core  # noqa: E402

CACHE = REPO / "cache/v16.7"
IC_DATA = REPO / "data/icentia_v167"
NST = REPO / "data/nstdb"
MIT = REPO / "data/mitdb"
W = hrv_core.WINDOW_SEC
LANGKAH = 5.0
SEED = 2026 + 167
N_PASIEN = 300
AF_LABEL = {"(AFIB", "(AFL"}
SNR = ("24", "18", "12", "06", "00", "_6")
TARGET_PALSU, TARGET_SE = 1.0, 0.90
U164.AKAR.update({"Icentia": IC_DATA, "nstdb": NST, "mitdb": MIT})


# ── seleksi Icentia (anotasi saja) ─────────────────────────────────────
def seleksi():
    p = CACHE / "seleksi.json"
    if p.exists():
        return json.loads(p.read_text())
    z = np.load(REPO / "cache/v10/icentia_scan.npz")
    L = list(z["labels"])
    seg, dur, lo = z["segment"], z["dur_s"], z["longest_s"]
    pas = np.array([s.split("/")[1] for s in seg])
    v10 = json.loads((REPO / "cache/v10/icentia_pilihan.json").read_text())["segmen"]
    seg_v10 = {x["segment"] for x in v10}
    pas_v10 = {x["patient"] for x in v10}
    af_dur = dur[:, L.index("(AFIB")] + dur[:, L.index("(AFL")]
    af_seg = np.maximum(lo[:, L.index("(AFIB")], lo[:, L.index("(AFL")]) >= 30
    pas_af = set(pas[af_dur > 0])
    rng = np.random.default_rng(SEED)
    out = []
    kand_af = np.flatnonzero(af_seg & ~np.isin(seg, list(seg_v10)))
    pa = np.unique(pas[kand_af])
    for p_ in sorted(rng.choice(pa, N_PASIEN, replace=False)):
        s = kand_af[pas[kand_af] == p_]
        out.append(dict(segment=str(seg[rng.choice(s)]), patient=str(p_), group="AF"))
    kand_n = np.flatnonzero(~np.isin(pas, list(pas_af | pas_v10)) & (dur[:, L.index("(N")] >= 3000))
    pn = np.unique(pas[kand_n])
    for p_ in sorted(rng.choice(pn, N_PASIEN, replace=False)):
        s = kand_n[pas[kand_n] == p_]
        out.append(dict(segment=str(seg[rng.choice(s)]), patient=str(p_), group="non-AF"))
    CACHE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=0))
    return out


def unduh_icentia(log=print, thread=12):
    import requests
    import icentia as IC
    sel = seleksi()
    s = requests.Session()
    sha = IC.ic_sha(s)
    jobs = [(x["segment"], e) for x in sel for e in ("hea", "dat", "atr")
            if not (IC_DATA / f"{x['segment']}.{e}").exists()]
    log(f"Icentia: {len(jobs)} berkas diunduh")

    def satu(j):
        rel = f"{j[0]}.{j[1]}"
        dest = IC_DATA / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        n = IC._fetch(s, rel, sha[rel], tmp)
        tmp.replace(dest)
        return n

    t0, tot = time.time(), 0
    with ThreadPoolExecutor(thread) as ex:
        for i, n in enumerate(ex.map(satu, jobs)):
            tot += n
            if (i + 1) % 150 == 0:
                log(f"  {i + 1}/{len(jobs)} berkas, {tot / 1e9:.2f} GB, {(time.time() - t0) / 60:.1f} mnt")
    log(f"Icentia siap ({tot / 1e9:.2f} GB baru)")


# ── pemrosesan per rekaman (mesin v16.4) ───────────────────────────────
def rekaman(ds):
    if ds == "Icentia":
        return [x["segment"] for x in seleksi()]
    if ds == "nstdb":
        return [f"{r}e{s}" for r in ("118", "119") for s in SNR]
    return ["118", "119"]                                     # mitdb bersih


def jalankan(ds, bagian=0, n_bagian=1, log=print):
    d = CACHE / ds
    d.mkdir(parents=True, exist_ok=True)
    daftar = rekaman(ds)
    alat = None
    for j, rec in enumerate(daftar):
        if j % n_bagian != bagian:
            continue
        p = d / (rec.replace("/", "__") + ".npz")
        if p.exists():
            continue
        alat = alat or R163.Alat()
        z = U164.proses_rekaman(ds, rec, alat)
        np.savez(p.with_suffix(".tmp.npz"), **z)
        p.with_suffix(".tmp.npz").replace(p)
        if (j + 1) % 25 == 0 or ds != "Icentia":
            log(f"  {ds}: {j + 1}/{len(daftar)}")


# ── label ──────────────────────────────────────────────────────────────
def interval_icentia(seg):
    """[(awal s, akhir s, label)] dari .atr ritme Icentia (parser v10); episode berakhir di ')' atau label ritme berikutnya."""
    import icentia as IC
    ev = IC.parse_atr_rhythm((IC_DATA / f"{seg}.atr").read_bytes())
    ev = [(s, a.strip()) for s, a in ev]
    fs = float((IC_DATA / f"{seg}.hea").read_text().split()[2])
    out = []
    for i, (s, a) in enumerate(ev):
        if a.startswith("("):
            e = ev[i + 1][0] if i + 1 < len(ev) else IC.SEG_LEN
            out.append((s / fs, e / fs, a))
    return out


def label_icentia(seg, start):
    y = np.zeros(start.size, int)
    keep = np.zeros(start.size, bool)
    for a, b, lab in interval_icentia(seg):
        if lab not in AF_LABEL and lab != "(N":
            continue
        inside = (start >= a) & (start + W <= b)
        keep |= inside
        if lab in AF_LABEL:
            y[inside] = 1
    return y, keep


def muat(ds):
    sel = {x["segment"]: x for x in seleksi()} if ds == "Icentia" else {}
    bag = {k: [] for k in ("rec", "start", "y2", "d1", "p31", "unit", "hr", "grup")}
    info = dict(rekaman=0, kosong=0, window_lolos=0, dibuang_label=0)
    for rec in rekaman(ds):
        z = np.load(CACHE / ds / (rec.replace("/", "__") + ".npz"))
        info["rekaman"] += 1
        if bool(z["kosong"]):
            info["kosong"] += 1
            continue
        st = z["start"].astype(float)
        if ds == "Icentia":
            y, keep = label_icentia(rec, st)
            unit, grup = sel[rec]["patient"], sel[rec]["group"]
        else:
            y, keep = np.zeros(st.size, int), np.ones(st.size, bool)
            unit, grup = rec[:3], (rec[4:] if ds == "nstdb" else "bersih")
        info["window_lolos"] += int(st.size)
        info["dibuang_label"] += int((~keep).sum())
        for k, v in (("rec", np.full(st.size, rec)), ("start", st), ("y2", y), ("d1", z["d1"]), ("p31", z["p31"]),
                     ("unit", np.full(st.size, unit)), ("hr", z["hr"]), ("grup", np.full(st.size, grup))):
            bag[k].append(v[keep])
    Z = {k: np.concatenate(v) for k, v in bag.items()}
    info.update(window=int(Z["start"].size), window_AF=int(Z["y2"].sum()), unit=int(np.unique(Z["unit"]).size),
                jam=float(Z["start"].size * LANGKAH / 3600))
    return dict(rec=Z["rec"], start=Z["start"], y2=Z["y2"], unit=Z["unit"], hr=Z["hr"], grup=Z["grup"], langkah=LANGKAH,
                d1=Z["d1"] > R163.AMBANG, v31=(Z["p31"].argmax(1) != 0)), info


def bagian(d, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == d["rec"].shape else v) for k, v in d.items()}


def sistem(d):
    return {"S1 = D-1 + 30 s": d["d1"], "R = v3.1 + 30 s": d["v31"], "S3 = (D-1 atau v3.1) + 30 s": d["d1"] | d["v31"]}
