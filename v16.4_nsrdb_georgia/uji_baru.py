"""
uji_baru.py — logika v16.4 (pra-registrasi di RANCANGAN.md, dikunci 10 November 2026 sebelum sinyal dibaca).

D-1 (ECGFounder zero-shot, AF bila d1 > −1), paket v3.1, D-0, dan paket v1 di dua dataset yang belum pernah dipakai:
  NSRDB   — 18 subjek sehat, Holter ±24 jam: alarm AF palsu sepanjang hari (semua window non-AF).
  Georgia — 10.344 EKG 12-lead klinis (lead I): AF/AFL vs non-AF, sinus takikardia/bradikardia berlabel dokter.
Per rekaman → cache/v16.4/<dataset>/<rekaman>.npz (bisa dilanjutkan). Mesin prediksi dan evaluasi dari v16.1–v16.3 (diimpor).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "v16.3_replikasi"))
import replikasi as R163  # noqa: E402
from replikasi import A161, A12, K33, U162, hrv_core  # noqa: E402

CACHE = REPO / "cache/v16.4"
DATASET = ("NSRDB", "Georgia")
AKAR = {"NSRDB": REPO / "data/nsrdb", "Georgia": REPO / "data/challenge2021/training/georgia"}
KODE_AF = {"164889003", "164890007"}
KODE_AFL, KODE_ST, KODE_SB, KODE_SNR = "164890007", "427084000", "426177001", "426783006"
KODE_AT_SVT = {"713422000", "426761007"}
KODE_ARTEFAK = (14, 16)                       # MIT: NOISE, ARFCT
KELOMPOK = ("AF/AFL", "sinus takikardia", "sinus bradikardia", "sinus normal", "lainnya")
SISTEM = ("v3.1 paket", "D-1", "D-0", "v1 paket")


def rekaman(ds):
    if ds == "NSRDB":
        return (AKAR[ds] / "RECORDS").read_text().split()
    return sorted(str(p.relative_to(AKAR[ds]).with_suffix("")) for p in AKAR[ds].rglob("*.hea"))


def dx_georgia(base):
    t = Path(str(base) + ".hea").read_text().splitlines()
    return set([l.split(":", 1)[1].strip().split(",") for l in t if l.startswith("# Dx")][0])


def kelompok_georgia(dx):
    """Kelompok eksklusif (RANCANGAN bagian 2): AF/AFL → sinus takikardia → sinus bradikardia → sinus normal → lainnya."""
    if dx & KODE_AF:
        return "AF/AFL"
    for k, g in ((KODE_ST, "sinus takikardia"), (KODE_SB, "sinus bradikardia"), (KODE_SNR, "sinus normal")):
        if k in dx:
            return g
    return "lainnya"


def proses_rekaman(ds, rec, alat):
    """Window yang lolos cek kualitas (EKG 50 Hz, X/M) → d1 ECGFounder, p31, v1. Pola v16.3 `proses_rekaman`."""
    base = AKAR[ds] / rec
    meta = {}
    if ds == "Georgia":
        x, fs = R163.baca_cinc(base)                        # baris pertama .mat = lead I, gain dari header
        dx = dx_georgia(base)
        meta = dict(kelompok=kelompok_georgia(dx), afl=bool(KODE_AFL in dx and "164889003" not in dx),
                    at_svt=bool(dx & KODE_AT_SVT))
        y_rek = int(bool(dx & KODE_AF))
    else:
        x, fs = A12.baca_dat(base, 0)
        s, k, _ = A12.baca_atr(base)
        t_art = np.sort(s[np.isin(k, KODE_ARTEFAK)] / fs)
        y_rek = 0
    x = np.nan_to_num(np.asarray(x, np.float64))
    win, _ = hrv_core.signal_to_windows(x, fs)
    x50 = hrv_core.resample_signal(x, fs, hrv_core.FS_DEVICE)
    kol = {k: [] for k in ("ecg", "X", "M", "start", "hr", "cv", "artefak")}
    n_win = len(win)
    for w in win:
        if not w["ok"]:
            continue
        s0 = w["start_sec"]
        ibi = np.asarray(w["ibi_ms"], np.float64)
        xs, ms = hrv_core.build_sequence(ibi)
        e = x50[int(round(s0 * hrv_core.FS_DEVICE)):][:U162.N50]
        art = False
        if ds == "NSRDB":
            i0, i1 = np.searchsorted(t_art, [s0, s0 + hrv_core.WINDOW_SEC])
            art = bool(i1 > i0)
        for k, v in (("ecg", np.pad(e, (0, U162.N50 - e.size))), ("X", xs), ("M", ms), ("start", s0),
                     ("hr", 60000.0 / ibi.mean()), ("cv", ibi.std() / ibi.mean()), ("artefak", art)):
            kol[k].append(v)
    out = dict(n_window=np.array(n_win), y_rek=np.array(y_rek), **{k: np.array(v) for k, v in meta.items()})
    if not kol["start"]:
        return out | dict(kosong=np.array(True))
    ecg, X, M = np.stack(kol["ecg"]), np.stack(kol["X"]), np.stack(kol["M"])
    P = R163.POTONG
    lo = np.concatenate([R163.E13.inferensi(alat.ef, R163.E13.praproses(ecg[i:i + P], hrv_core.FS_DEVICE), alat.device)[0]
                         for i in range(0, len(ecg), P)])
    p31 = np.concatenate([alat.v31.predict_proba_sequences(X[i:i + 40000], M[i:i + 40000]) for i in range(0, len(X), 40000)])
    v1 = np.concatenate([alat.v1(list(zip(X[i:i + 40000], M[i:i + 40000]))) for i in range(0, len(X), 40000)])
    return out | dict(kosong=np.array(False), start=np.array(kol["start"]), hr=np.array(kol["hr"]), cv=np.array(kol["cv"]),
                      artefak=np.array(kol["artefak"]), d1=R163.H16.tahap1(lo)[1].astype(np.float32), p31=p31.astype(np.float32),
                      v1=np.asarray(v1, np.int8))


def jalankan(ds, alat=None, log=print):
    d = CACHE / ds
    d.mkdir(parents=True, exist_ok=True)
    daftar = rekaman(ds)
    for j, rec in enumerate(daftar):
        p = d / (rec.replace("/", "__") + ".npz")
        if p.exists():
            continue
        alat = alat or R163.Alat()
        z = proses_rekaman(ds, rec, alat)
        np.savez(p.with_suffix(".tmp.npz"), **z)
        p.with_suffix(".tmp.npz").replace(p)
        if (j + 1) % max(1, len(daftar) // 10) == 0:
            log(f"  {ds}: {j + 1}/{len(daftar)} rekaman")
    return alat


def muat(ds):
    """Gabungan window yang lolos cek + jumlah window per rekaman/kelompok (termasuk yang tidak lolos)."""
    bag, info = {}, {"rekaman": 0, "window": 0, "window_lolos": 0, "per_kelompok": {}}
    for rec in rekaman(ds):
        z = np.load(CACHE / ds / (rec.replace("/", "__") + ".npz"))
        info["rekaman"] += 1
        info["window"] += int(z["n_window"])
        g = str(z["kelompok"]) if "kelompok" in z.files else "non-AF"
        kg = info["per_kelompok"].setdefault(g, [0, 0, 0])            # rekaman, window, window lolos
        kg[0] += 1
        kg[1] += int(z["n_window"])
        if bool(z["kosong"]):
            continue
        n = z["start"].size
        info["window_lolos"] += n
        kg[2] += n
        for k in ("start", "hr", "cv", "artefak", "d1", "p31", "v1"):
            bag.setdefault(k, []).append(z[k])
        bag.setdefault("rekaman", []).append(np.full(n, rec))
        bag.setdefault("y2", []).append(np.full(n, int(z["y_rek"])))
        if ds == "Georgia":
            for k in ("kelompok", "afl", "at_svt"):
                bag.setdefault(k, []).append(np.full(n, z[k].item()))
    out = {k: np.concatenate(v) for k, v in bag.items()}
    out["y2"] = out["y2"].astype(np.int64)
    out["subjek"] = out["rekaman"]
    return out, info


def boot_alarm(unit, alarm):
    """Rata-rata alarm per window + CI bootstrap unit (pola v3.3)."""
    t, ci, _ = K33._boot_subjek(unit, alarm.astype(float))
    return dict(alarm=t, ci=ci.tolist(), n=int(len(alarm)), unit=int(np.unique(unit).size))


def evaluasi_georgia(Z):
    pred = R163.prediksi(Z)
    y2, unit = Z["y2"], Z["subjek"]
    bd = {k: A161.banding2_pasien(y2, pred[k], pred["v3.1 paket"], unit) for k in ("D-1", "D-0", "v1 paket")}
    titik = {"v3.1 paket": bd["D-1"]["titik_b"], **{k: bd[k]["titik_a"] for k in ("D-1", "D-0", "v1 paket")}}
    kel = {g: {k: boot_alarm(unit[Z["kelompok"] == g], pred[k][Z["kelompok"] == g]) for k in SISTEM}
           for g in KELOMPOK if (Z["kelompok"] == g).any()}
    sub = {}
    for nama, m in (("AFL", Z["afl"].astype(bool)), ("AF", (y2 == 1) & ~Z["afl"].astype(bool)), ("AT/SVT", Z["at_svt"].astype(bool) & (y2 == 0))):
        if m.any():
            sub[nama] = {k: boot_alarm(unit[m], pred[k][m]) for k in SISTEM}
    st = kel["sinus takikardia"]
    return dict(banding=bd, titik=titik, kelompok=kel, subkelompok=sub,
                pita={k: U162.alarm_pita(y2, p, Z["hr"]) for k, p in pred.items()},
                kriteria=dict(H1_macro_f1=bool(bd["D-1"]["Macro-F1"][3] > 0.05), H2_sensitivitas=bool(bd["D-1"]["Se"][3] > 0.05),
                              H3_sinus_takikardia=bool(st["D-1"]["alarm"] <= 0.5 * st["v3.1 paket"]["alarm"]),
                              sekunder_noninferior_se=bool(bd["D-1"]["Se"][1] > -0.02)))


def evaluasi_nsrdb(Z):
    pred = R163.prediksi(Z)
    unit = Z["subjek"]
    alarm = {k: boot_alarm(unit, p) for k, p in pred.items()}
    d = pred["D-1"].astype(float) - pred["v3.1 paket"].astype(float)
    t, ci, b = K33._boot_subjek(unit, d)
    banding = dict(delta=t, ci=ci.tolist(), p_kurang=float((b < 0).mean()))
    bersih = ~Z["artefak"].astype(bool)
    alarm_bersih = {k: boot_alarm(unit[bersih], p[bersih]) for k, p in pred.items()}
    per_subjek = {s: {k: float(pred[k][unit == s].mean()) for k in SISTEM} | {"n": int((unit == s).sum())} for s in np.unique(unit)}
    episode = {k: K33.metrik_episode(Z["rekaman"], Z["start"], Z["y2"], p.astype(np.int64), 1, langkah=5.0) for k, p in pred.items()}
    return dict(alarm=alarm, banding_d1_v31=banding, alarm_tanpa_artefak=alarm_bersih, per_subjek=per_subjek, episode=episode,
                pita={k: U162.alarm_pita(Z["y2"], p, Z["hr"]) for k, p in pred.items()},
                kriteria=dict(H4_nsrdb=bool(banding["p_kurang"] >= 0.95)))


def simpan_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=float, ensure_ascii=False))
