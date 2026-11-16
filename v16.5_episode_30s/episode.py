"""
episode.py — logika v16.5 (pra-registrasi di RANCANGAN.md, dikunci 10 November 2026 sebelum aturan dihitung).

Aturan episode AF ≥ 30 s (definisi klinis, tidak disetel) di atas keputusan per window D-1 (ECGFounder zero-shot, d1 > −1), paket v3.1,
dan S2 = D-1 dan v3.1. Semua keputusan per window dari cache v13/v16.1–v16.4 dan v3.3; EDB (belum pernah dipakai) diproses di sini dengan
mesin v16.4.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "v16.4_nsrdb_georgia"), str(REPO / "v16_hierarki_ecgfounder")]
import uji_baru as U164  # noqa: E402
import hierarki as H16  # noqa: E402
from uji_baru import A161, A12, K33, R163, U162, hrv_core  # noqa: E402

CACHE = REPO / "cache/v16.5"
EDB = REPO / "data/edb"
W = hrv_core.WINDOW_SEC
DURASI = (0, 30, 60, 120, 360)          # 30 = primer; lainnya deskriptif (0 = tanpa aturan)
PRIMER = 30
# label ritme MIT (EDB juga memuat label ST/T di aux yang BUKAN ritme)
RITME_MIT = {"(AB", "(AFIB", "(AFL", "(B", "(BII", "(IVR", "(N", "(NOD", "(P", "(PREX", "(SBR", "(SVTA", "(T", "(VFL", "(VT"}
TARGET_PALSU, TARGET_SE = 1.0, 0.90


# ── aturan ─────────────────────────────────────────────────────────────
def aturan(rec, start, af, langkah, min_span=PRIMER):
    """Alarm per window: window AF yang termasuk run (jarak awal ≤ langkah + 0,5 s) dengan rentang ≥ min_span detik."""
    af = np.asarray(af, bool)
    out = np.zeros(af.size, bool)
    if min_span <= 0:
        return af.copy()
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        idx = idx[np.argsort(start[idx], kind="stable")]
        s, a = start[idx], af[idx]
        i = 0
        while i < len(idx):
            if not a[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(idx) and a[j + 1] and s[j + 1] - s[j] <= langkah + 0.5:
                j += 1
            if s[j] + W - s[i] >= min_span:
                out[idx[i:j + 1]] = True
            i = j + 1
    return out


def _run(s, on, langkah):
    """Interval [awal, akhir) dari run window aktif berturut-turut (akhir = awal window terakhir + 10 s)."""
    ep, i = [], 0
    while i < len(s):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(s) and on[j + 1] and s[j + 1] - s[j] <= langkah + 0.5:
            j += 1
        ep.append((s[i], s[j] + W))
        i = j + 1
    return np.array(ep, float).reshape(-1, 2)


def _gabung(iv):
    if not len(iv):
        return iv
    iv = iv[np.argsort(iv[:, 0])]
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.array(out)


def _overlap(A, B):
    A, B = _gabung(A), _gabung(B)
    return float(sum(max(0.0, min(a1, b1) - max(a0, b0)) for a0, a1 in A for b0, b1 in B))


def metrik(rec, start, y2, alarm, langkah, unit=None):
    """Episode gaya EC57 dengan interval sebenarnya: Se episode (referensi ≥ 30 s dan semua), +P, Se/+P durasi, palsu per jam."""
    tp30 = n30 = tp_all = n_all = tp_det = n_det = 0
    ov = dur_ref = dur_det = 0.0
    jam = 0.0
    palsu_unit = {}
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        m = m[np.argsort(start[m], kind="stable")]
        s = start[m]
        R = _run(s, y2[m] == 1, langkah)
        D = _run(s, alarm[m], langkah)
        jam_r = len(m) * langkah / 3600.0
        jam += jam_r
        for e in R:
            hit = _overlap(e[None], D) > 0
            n_all += 1; tp_all += hit
            if e[1] - e[0] >= PRIMER:
                n30 += 1; tp30 += hit
        palsu_r = 0
        for e in D:
            n_det += 1
            if _overlap(e[None], R) > 0:
                tp_det += 1
            else:
                palsu_r += 1
        ov += _overlap(R, D)
        dur_ref += float((_gabung(R)[:, 1] - _gabung(R)[:, 0]).sum()) if len(R) else 0.0
        dur_det += float((_gabung(D)[:, 1] - _gabung(D)[:, 0]).sum()) if len(D) else 0.0
        u = r if unit is None else unit[m][0]
        a = palsu_unit.setdefault(u, [0, 0.0])
        a[0] += palsu_r
        a[1] += jam_r
    nan = float("nan")
    return dict(se_episode_30=tp30 / n30 if n30 else nan, n_ref_30=int(n30), se_episode=tp_all / n_all if n_all else nan,
                n_ref=int(n_all), ppv_episode=tp_det / n_det if n_det else nan, n_det=int(n_det),
                se_durasi=ov / dur_ref if dur_ref else nan, ppv_durasi=ov / dur_det if dur_det else nan,
                palsu_per_jam=(n_det - tp_det) / jam if jam else nan, jam=jam,
                palsu_unit={str(k): v for k, v in palsu_unit.items()})


def boot_palsu(palsu_a, palsu_b, n_boot=2000, seed=2026):
    """Δ palsu per jam (a − b), bootstrap unit berpasangan: rasio jumlah episode palsu / jumlah jam unit hasil resampling."""
    u = sorted(palsu_a)
    ea, eb = np.array([palsu_a[k][0] for k in u], float), np.array([palsu_b[k][0] for k in u], float)
    h = np.array([palsu_a[k][1] for k in u], float)
    idx = np.random.default_rng(seed).integers(0, len(u), (n_boot, len(u)))
    da = ea[idx].sum(1) / h[idx].sum(1) - eb[idx].sum(1) / h[idx].sum(1)
    t = ea.sum() / h.sum() - eb.sum() / h.sum()
    ca = ea[idx].sum(1) / h[idx].sum(1)
    return dict(delta=float(t), ci=np.percentile(da, [2.5, 97.5]).tolist(), p_kurang=float((da < 0).mean()),
                ci_a=np.percentile(ca, [2.5, 97.5]).tolist())


# ── data ───────────────────────────────────────────────────────────────
def _afdb():
    """AFDB window detektor (dataset v2): D-1 zero-shot (cache v13), v3.1 = OOF MicroTransformer +bias cross-fit, 5 ulangan."""
    d = U162.T.load()
    z = np.load(REPO / "cache/dataset_v2_detector.npz")
    assert np.array_equal(z["records"].astype(str), d["rec"])
    d1 = A161.d1_afdb()
    assert len(d1) == len(d["y"])
    pred31 = np.stack([H16.cv.crossfit_bias(d["y"], d["probs"]["v31"][r], d["folds"][r])[0].argmax(1) != 0 for r in range(5)])
    return dict(rec=d["rec"], start=z["t_start"].astype(float), y2=(d["y"] > 0).astype(int), unit=d["rec"], langkah=5.0,
                d1=d1 > R163.AMBANG, v31=pred31)


def _ext(ds):
    Z, _ = R163.muat(ds)
    return dict(rec=Z["rekaman"], start=Z["start"].astype(float), y2=Z["y2"].astype(int), unit=Z["subjek"], langkah=5.0,
                d1=Z["d1"] > R163.AMBANG, v31=(Z["p31"].argmax(1) != 0)[None])


def _cpsc():
    J = U162.bangun_jendela(resmi=True, log=lambda *_: None)
    ok = J["ok"]
    d1 = R163.H16.tahap1(np.load(U162.CACHE / "logit_ecgfounder_resmi.npy"))[1]
    p31 = np.load(U162.CACHE / "paket_resmi.npz")["p31"]
    assert len(d1) == len(p31) == int(ok.sum())
    return dict(rec=J["rekaman"][ok], start=J["start"][ok].astype(float), y2=(J["y"][ok] > 0).astype(int), unit=J["subjek"][ok],
                langkah=5.0, d1=d1 > R163.AMBANG, v31=(p31.argmax(1) != 0)[None])


def _nsrdb():
    Z, _ = U164.muat("NSRDB")
    return dict(rec=Z["rekaman"], start=Z["start"].astype(float), y2=Z["y2"].astype(int), unit=Z["subjek"], langkah=5.0,
                d1=Z["d1"] > R163.AMBANG, v31=(Z["p31"].argmax(1) != 0)[None])


def _latihan():
    L = H16.latihan()
    J = L["J"]
    baris = np.concatenate([np.flatnonzero((J["sumber"] == s) & (J["mode"] == "perangkat") & J["ok"]) for s in np.unique(J["sumber"])])
    d1 = H16.tahap1(L["logit"])[1]
    return dict(rec=J["rekaman"][baris], start=J["start"][baris].astype(float), y2=np.zeros(len(baris), int), unit=L["subjek"],
                langkah=float(K33.STRIDE), d1=d1 > R163.AMBANG, v31=(L["p31"].argmax(1) != 0)[None], sumber=L["sumber"])


# ── EDB (belum pernah dipakai) ─────────────────────────────────────────
def segmen_af_edb(rec):
    """Interval AF (detik) dari label ritme MIT di .atr EDB (label ST/T diabaikan)."""
    fs, n = A12.baca_hea(EDB / rec)[:2]
    s, _, aux = A12.baca_atr(EDB / rec)
    rr = [(si / fs, a.strip("\x00").strip()) for si, a in zip(s, aux) if a.strip("\x00").strip() in RITME_MIT]
    out = []
    for i, (t0, lab) in enumerate(rr):
        if lab in ("(AFIB", "(AFL"):
            out.append((t0, rr[i + 1][0] if i + 1 < len(rr) else n / fs))
    return out


def jalankan_edb(log=print):
    """Mesin v16.4 (kanal 0, window lolos cek, d1/p31/v1) untuk semua rekaman EDB → cache/v16.5/EDB/."""
    U164.AKAR["EDB"] = EDB              # U164.CACHE tidak diubah: _nsrdb() tetap membaca cache/v16.4/NSRDB
    alat = None
    recs = (EDB / "RECORDS").read_text().split()
    (CACHE / "EDB").mkdir(parents=True, exist_ok=True)
    for j, rec in enumerate(recs):
        p = CACHE / "EDB" / f"{rec}.npz"
        if p.exists():
            continue
        alat = alat or R163.Alat()
        z = U164.proses_rekaman("EDB", rec, alat)
        np.savez(p.with_suffix(".tmp.npz"), **z)
        p.with_suffix(".tmp.npz").replace(p)
        if (j + 1) % 10 == 0:
            log(f"  EDB: {j + 1}/{len(recs)}")


def _edb():
    bag = {k: [] for k in ("rec", "start", "y2", "d1", "p31", "artefak")}
    buang = 0
    for rec in (EDB / "RECORDS").read_text().split():
        z = np.load(CACHE / "EDB" / f"{rec}.npz")
        if bool(z["kosong"]):
            continue
        st = z["start"]
        iv = segmen_af_edb(rec)
        y = np.zeros(st.size, int)
        keep = np.ones(st.size, bool)
        for a, b in iv:
            inside = (st >= a) & (st + W <= b)
            cross = (st < b) & (st + W > a) & ~inside
            y[inside] = 1
            keep &= ~cross
        buang += int((~keep).sum())
        for k, v in (("rec", np.full(st.size, rec)), ("start", st), ("y2", y), ("d1", z["d1"]), ("p31", z["p31"]), ("artefak", z["artefak"])):
            bag[k].append(v[keep])
    o = {k: np.concatenate(v) for k, v in bag.items()}
    return dict(rec=o["rec"], start=o["start"].astype(float), y2=o["y2"], unit=o["rec"], langkah=5.0,
                d1=o["d1"] > R163.AMBANG, v31=(o["p31"].argmax(1) != 0)[None], buang_batas=buang)


def muat_semua():
    return {"AFDB": _afdb(), "LTAFDB": _ext("LTAFDB"), "MIT-BIH": _ext("MIT-BIH"), "CPSC 2021": _cpsc(), "NSRDB": _nsrdb(),
            "EDB": _edb(), "Latihan fisik": _latihan()}


def sistem(D):
    """Keputusan per window (R, N) tiap sistem sebelum aturan: D-1, v3.1, S2 = D-1 dan v3.1."""
    d1 = D["d1"][None]
    return {"D-1": d1, "v3.1": D["v31"], "S2 (D-1 & v3.1)": d1 & D["v31"]}


def evaluasi(D):
    """{sistem: {durasi: metrik rata-rata ulangan (+ palsu per unit ulangan 0)}}."""
    out = {}
    for nama, P in sistem(D).items():
        out[nama] = {}
        for dur in DURASI:
            ms = [metrik(D["rec"], D["start"], D["y2"], aturan(D["rec"], D["start"], p, D["langkah"], dur), D["langkah"], D["unit"])
                  for p in P]
            rata = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0] if k != "palsu_unit"}
            rata["palsu_unit"] = ms[0]["palsu_unit"]
            out[nama][dur] = rata
    return out


def simpan_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=float, ensure_ascii=False))
