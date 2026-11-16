"""
replikasi.py — logika v16.3 (pra-registrasi di RANCANGAN.md, dikunci 6 November 2026 sebelum sinyal dibaca).

Replikasi detektor AF 2 kelas D-1 (ECGFounder zero-shot, AF bila d1 > −1) di CinC 2017, MIT-BIH Arrhythmia, dan LTAFDB, tanpa
penyetelan. Per rekaman: window perangkat (`hrv_core.signal_to_windows`: 50 Hz, detektor R produksi, 10 s, langkah 5 s), label
(CinC: per rekaman; MIT-BIH/LTAFDB: label ritme konvensi MIT), lalu untuk window yang lolos cek kualitas: logit ECGFounder, paket
v3.1, dan paket v1 → cache/v16.3/<dataset>/<rekaman>.npz (bisa dilanjutkan). Evaluasi memakai fungsi v16.1/v16.2.
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "v16.2_cpsc2021"))
import uji_cpsc as U162  # noqa: E402
from uji_cpsc import A161, A12, D, DK, E13, Ev, H16, K15, K33, T, hrv_core  # noqa: E402,F401

CACHE = REPO / "cache/v16.3"
AMBANG = U162.AMBANG
DATASET = ("CinC 2017", "MIT-BIH", "LTAFDB")
FOLDER = {"CinC 2017": "cinc2017", "MIT-BIH": "mitdb", "LTAFDB": "ltafdb"}
AF = ("(AFIB", "(AFL", "A")
BUANG = ("(P",)                                   # ritme pacu (seperti PACE di PTB-XL)
SUBJEK_SAMA = {"202": "201"}                       # MIT-BIH: rekaman 201 dan 202 dari subjek yang sama
POTONG = 4096


def rekaman(ds):
    """[(path relatif rekaman, label rekaman atau None)]. CinC 2017: REFERENCE-v3.csv, `~` (bising) dibuang."""
    akar = REPO / "data" / FOLDER[ds]
    if ds == "CinC 2017":
        lab = {r[0].split("/")[-1]: r[1].strip() for r in csv.reader(open(akar / "REFERENCE-v3.csv"))}
        return [(f"training/{k[:3]}/{k}", v) for k, v in sorted(lab.items()) if v != "~"]
    return [(r, None) for r in (akar / "RECORDS").read_text().split()]


def subjek(ds, rec):
    nama = rec.split("/")[-1]
    return SUBJEK_SAMA.get(nama, nama) if ds == "MIT-BIH" else nama


def baca_cinc(base):
    """CinC 2017: `val` di .mat (int) / gain header (mis. 1000/mV) → mV; fs dari header."""
    from scipy.io import loadmat
    h = base.with_suffix(".hea").read_text().splitlines()
    fs = float(h[0].split()[2])
    gain = float(h[1].split()[2].split("/")[0].split("(")[0])
    return loadmat(str(base.with_suffix(".mat")))["val"][0].astype(np.float64) / gain, fs


class Alat:
    """Model yang dimuat sekali: ECGFounder (GPU bila ada), paket v3.1, paket v1."""

    def __init__(self):
        import torch
        from predictor import ECGPredictor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ef = E13.muat_model(self.device)
        self.v31 = ECGPredictor(str(REPO / "v3.1_kalibrasi_produksi/model_export"), name="sihedaf_v3_1", device="cpu")
        self.v1 = T.load_packages()["v1"]


def proses_rekaman(ds, rec, label_rek, alat):
    """Window berlabel satu rekaman → dict (hanya window yang lolos cek) + jumlah window berlabel per ritme. None bila kosong."""
    base = REPO / "data" / FOLDER[ds] / rec
    if ds == "CinC 2017":
        x, fs = baca_cinc(base)
        seg = None
    else:
        x, fs = A12.baca_dat(base, 0)
        seg = K15.segmen_ritme(str(base))             # path absolut: CPSC / path absolut = path itu sendiri
    x = np.nan_to_num(np.asarray(x, np.float64))
    win, _ = hrv_core.signal_to_windows(x, fs)
    x50 = hrv_core.resample_signal(x, fs, hrv_core.FS_DEVICE)
    kol = {k: [] for k in ("ecg", "X", "M", "ritme", "start", "hr")}
    semua = []
    for w in win:
        s = w["start_sec"]
        if seg is None:
            lab = label_rek
        else:
            cocok = [lb for a, b, lb in seg if a <= s and s + hrv_core.WINDOW_SEC <= b]
            lab = cocok[0] if cocok else None
        if lab is None or lab in BUANG:
            continue
        semua.append((lab, bool(w["ok"])))
        if not w["ok"]:
            continue
        ibi = np.asarray(w["ibi_ms"], np.float64)
        xs, ms = hrv_core.build_sequence(ibi)
        e = x50[int(round(s * hrv_core.FS_DEVICE)):][:U162.N50]
        for k, v in (("ecg", np.pad(e, (0, U162.N50 - e.size))), ("X", xs), ("M", ms), ("ritme", lab), ("start", s),
                     ("hr", 60000.0 / ibi.mean())):
            kol[k].append(v)
    lab_all = np.array([a for a, _ in semua] or [""])
    ok_all = np.array([b for _, b in semua] or [False])
    jumlah = {str(r): [int((lab_all == r).sum()), int(((lab_all == r) & ok_all).sum())] for r in np.unique(lab_all) if r}
    if not kol["ritme"]:
        return None, jumlah
    ecg, X, M = np.stack(kol["ecg"]), np.stack(kol["X"]), np.stack(kol["M"])
    lo = np.concatenate([E13.inferensi(alat.ef, E13.praproses(ecg[i:i + POTONG], hrv_core.FS_DEVICE), alat.device)[0]
                         for i in range(0, len(ecg), POTONG)])
    p31 = np.concatenate([alat.v31.predict_proba_sequences(X[i:i + 40000], M[i:i + 40000]) for i in range(0, len(X), 40000)])
    v1 = np.concatenate([alat.v1(list(zip(X[i:i + 40000], M[i:i + 40000]))) for i in range(0, len(X), 40000)])
    ritme = np.array(kol["ritme"])
    return dict(ritme=ritme, y2=np.isin(ritme, AF).astype(np.int8), start=np.array(kol["start"]), hr=np.array(kol["hr"]),
                d1=H16.tahap1(lo)[1].astype(np.float32), p31=p31.astype(np.float32), v1=np.asarray(v1, np.int8)), jumlah


def jalankan(ds, alat=None, log=print):
    """Semua rekaman satu dataset → cache/v16.3/<folder>/<rekaman>.npz (dilewati bila sudah ada). Return Alat (dimuat bila perlu)."""
    import json
    d = CACHE / FOLDER[ds]
    d.mkdir(parents=True, exist_ok=True)
    daftar = rekaman(ds)
    for j, (rec, lab) in enumerate(daftar):
        p = d / (rec.replace("/", "__") + ".npz")
        if p.exists():
            continue
        alat = alat or Alat()
        z, jumlah = proses_rekaman(ds, rec, lab, alat)
        np.savez(p.with_suffix(".tmp.npz"), kosong=np.array(z is None), jumlah=np.array(json.dumps(jumlah)), **(z or {}))
        p.with_suffix(".tmp.npz").replace(p)
        if (j + 1) % max(1, len(daftar) // 10) == 0:
            log(f"  {ds}: {j + 1}/{len(daftar)} rekaman")
    return alat


def muat(ds):
    """Gabungan window yang lolos cek satu dataset, dengan kolom rekaman dan subjek; plus jumlah window berlabel/lolos per ritme."""
    import json
    bag, jumlah = {}, {}
    for rec, _ in rekaman(ds):
        z = np.load(CACHE / FOLDER[ds] / (rec.replace("/", "__") + ".npz"))
        for r, (n, nok) in json.loads(str(z["jumlah"])).items():
            a = jumlah.setdefault(r, [0, 0])
            a[0] += n
            a[1] += nok
        if bool(z["kosong"]):
            continue
        n = z["ritme"].size
        for k in ("ritme", "y2", "start", "hr", "d1", "p31", "v1"):
            bag.setdefault(k, []).append(z[k])
        bag.setdefault("rekaman", []).append(np.full(n, rec))
        bag.setdefault("subjek", []).append(np.full(n, subjek(ds, rec)))
    out = {k: np.concatenate(v) for k, v in bag.items()}
    out["y2"] = out["y2"].astype(np.int64)
    return out, jumlah


def prediksi(Z):
    """Keputusan AF (bool) keempat sistem."""
    return {"v3.1 paket": Z["p31"].argmax(1) != 0, "v1 paket": Z["v1"] != 0, "D-0": Z["d1"] > 0, "D-1": Z["d1"] > AMBANG}


def evaluasi(ds, Z):
    """Perbandingan berpasangan (bootstrap unit 2.000×), alarm per ritme, per pita detak, per subjek, episode (bukan CinC)."""
    pred = prediksi(Z)
    y2, unit = Z["y2"], Z["subjek"]
    bd = {k: A161.banding2_pasien(y2, pred[k], pred["v3.1 paket"], unit) for k in ("D-1", "D-0", "v1 paket")}
    bd["D-1 vs D-0"] = A161.banding2_pasien(y2, pred["D-1"], pred["D-0"], unit)
    titik = {"v3.1 paket": bd["D-1"]["titik_b"], **{k: bd[k]["titik_a"] for k in ("D-1", "D-0", "v1 paket")}}
    ritme = {str(r): dict(n=int((Z["ritme"] == r).sum()), **{k: float(p[Z["ritme"] == r].mean()) for k, p in pred.items()})
             for r in np.unique(Z["ritme"])}
    ps = {k: U162.per_subjek(y2, pred[k], unit) for k in ("v3.1 paket", "D-1")}
    turun = {s: round(ps["v3.1 paket"][s]["se"] - ps["D-1"][s]["se"], 3) for s in ps["D-1"]
             if ps["D-1"][s]["se"] is not None and ps["v3.1 paket"][s]["se"] - ps["D-1"][s]["se"] > 0.20}
    out = dict(banding=bd, titik=titik, ritme=ritme, pita={k: U162.alarm_pita(y2, p, Z["hr"]) for k, p in pred.items()},
               se_turun_0_20=turun, n=dict(window=int(len(y2)), af=int(y2.sum()), non_af=int((y2 == 0).sum()),
                                           unit=int(np.unique(unit).size), unit_af=int(np.unique(unit[y2 == 1]).size)),
               kriteria=dict(H1_macro_f1=bool(bd["D-1"]["Macro-F1"][3] > 0.05), H2_sensitivitas=bool(bd["D-1"]["Se"][3] > 0.05)))
    if ds != "CinC 2017":
        out["episode"] = {k: K33.metrik_episode(Z["rekaman"], Z["start"], y2, p.astype(np.int64), 1, langkah=5.0) for k, p in pred.items()}
    return out
