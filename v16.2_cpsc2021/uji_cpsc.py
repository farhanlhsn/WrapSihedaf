"""
uji_cpsc.py — logika v16.2 (pra-registrasi di RANCANGAN.md, dikunci 6 November 2026 sebelum sinyal CPSC 2021 dibaca).

Uji ambang t = −1 detektor AF 2 kelas ECGFounder (D-1: AF bila d1 > −1) di CPSC 2021, data yang belum pernah dinilai SIHEDAF.
Window = window perangkat (`hrv_core.signal_to_windows`: 50 Hz, detektor R produksi, 10 s, langkah 5 s) dari kanal 0 (lead I),
berlabel bila seluruhnya di dalam satu segmen ritme `(N`/`(AFIB`/`(AFL`. Himpunan uji = window berlabel yang lolos cek kualitas
detektor (punya prediksi paket v3.1). Pembanding: paket v3.1 dan v1 (AFIB + AFL digabung), D-0 (v16.1).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "v16.1_dua_kelas"))
import deteksi_af as A161  # noqa: E402
from deteksi_af import D, DK, E13, Ev, H16, K33, T  # noqa: E402,F401

K15, A12, hrv_core = H16.K15, H16.A12, E13.hrv_core
CACHE = REPO / "cache/v16.2"
AMBANG = -1.0
LABEL = {"(N": 0, "(AFIB": 1, "(AFL": 2}
N50 = int(hrv_core.WINDOW_SEC * hrv_core.FS_DEVICE)


def subjek(rec):
    """`Training_set_I/data_<subjek>_<n>` → subjek (pola v15)."""
    return rec.split("/")[1].split("_")[1]


def _label_lama(rec):
    """Aturan yang dikunci (bagian 2): label segmen `v15.segmen_ritme` bila window seluruhnya di dalam satu segmen."""
    seg = [(a, b, LABEL[lb]) for a, b, lb in K15.segmen_ritme(rec) if lb in LABEL]

    def f(s):
        lab = [lb for a, b, lb in seg if a <= s and s + hrv_core.WINDOW_SEC <= b]
        return lab[0] if lab else None
    return f


def interval_af(rec):
    """Interval AF menurut konvensi resmi CPSC 2021 (score_2021.RefInfo): `(AFIB`/`(AFL` = awal, `(N` = akhir; awal baru saat
    interval masih terbuka menutup interval sebelumnya; awal tanpa akhir → sampai akhir rekaman. Return [(mulai_s, akhir_s, kelas)]."""
    base = K15.CPSC / rec
    smp, _, aux = A12.baca_atr(base)
    fs, n = A12.baca_hea(base)[:2]
    out, buka = [], None
    for si, a in zip(smp, aux):
        a = a.strip("\x00").strip()
        if a in ("(AFIB", "(AFL"):
            if buka is not None:
                out.append((buka[0], si / fs, buka[1]))
            buka = (si / fs, LABEL[a])
        elif a == "(N" and buka is not None:
            out.append((buka[0], si / fs, buka[1]))
            buka = None
    if buka is not None:
        out.append((buka[0], n / fs, buka[1]))
    return out


def _label_resmi(rec):
    """Penyimpangan tercatat (RANCANGAN bagian 8): AF bila window di dalam interval AF resmi, Normal bila tidak bersinggungan
    dengan interval AF mana pun, None bila melintasi batas."""
    iv = interval_af(rec)

    def f(s):
        e = s + hrv_core.WINDOW_SEC
        for a, b, k in iv:
            if a <= s and e <= b:
                return k
        return None if any(s < b and e > a for a, b, _ in iv) else 0
    return f


def jendela_rekaman(rec, resmi=False):
    """Window perangkat berlabel dari satu rekaman → dict array (EKG 50 Hz, X/M paket, label, ok, detak, CV)."""
    x, fs = A12.baca_dat(K15.CPSC / rec, 0)
    x = np.nan_to_num(np.asarray(x, np.float64))
    label = (_label_resmi if resmi else _label_lama)(rec)
    win, _ = hrv_core.signal_to_windows(x, fs)
    x50 = hrv_core.resample_signal(x, fs, hrv_core.FS_DEVICE)
    kol = {k: [] for k in ("ecg", "X", "M", "y", "ok", "start", "hr", "cv", "alasan")}
    for w in win:
        s = w["start_sec"]
        lab = label(s)
        if lab is None:
            continue
        i = int(round(s * hrv_core.FS_DEVICE))
        e = x50[i:i + N50]
        ibi = np.asarray(w["ibi_ms"], np.float64)
        xs, ms = hrv_core.build_sequence(ibi) if w["ok"] else (np.zeros((hrv_core.MAX_BEATS, 2), np.float32),
                                                                 np.zeros(hrv_core.MAX_BEATS, np.float32))
        for k, v in (("ecg", np.pad(e, (0, N50 - e.size))), ("X", xs), ("M", ms), ("y", lab), ("ok", w["ok"]), ("start", s),
                     ("hr", 60000.0 / ibi.mean() if ibi.size else np.nan), ("cv", ibi.std() / ibi.mean() if ibi.size else np.nan),
                     ("alasan", w["reason"])):
            kol[k].append(v)
    if not kol["y"]:
        return None
    out = {k: np.array(v) for k, v in kol.items()}
    out["ecg"] = out["ecg"].astype(np.float16)
    return out


def bangun_jendela(resmi=False, log=print):
    """Semua rekaman yang dirilis → cache/v16.2/rekaman[_resmi]/<nama>.npz (bisa dilanjutkan) → gabungan dengan kolom
    rekaman/subjek. resmi=True: label konvensi resmi CPSC 2021 (RANCANGAN bagian 8)."""
    d = CACHE / ("rekaman_resmi" if resmi else "rekaman")
    d.mkdir(parents=True, exist_ok=True)
    rekaman = K15.rekaman_cpsc()
    for j, rec in enumerate(rekaman):
        p = d / (rec.replace("/", "__") + ".npz")
        if p.exists():
            continue
        z = jendela_rekaman(rec, resmi)
        np.savez(p.with_suffix(".tmp.npz"), kosong=np.array(z is None), **(z or {}))
        p.with_suffix(".tmp.npz").replace(p)
        if (j + 1) % 100 == 0:
            log(f"  {j + 1}/{len(rekaman)} rekaman")
    bag = {}
    for rec in rekaman:
        z = np.load(d / (rec.replace("/", "__") + ".npz"))
        if bool(z["kosong"]):
            continue
        n = z["y"].size
        for k in ("ecg", "X", "M", "y", "ok", "start", "hr", "cv", "alasan"):
            bag.setdefault(k, []).append(z[k])
        bag.setdefault("rekaman", []).append(np.full(n, rec))
        bag.setdefault("subjek", []).append(np.full(n, subjek(rec)))
    return {k: np.concatenate(v) for k, v in bag.items()}


def logit_ecgfounder(ecg50, nama="logit_ecgfounder", log=print):
    """Logit ECGFounder untuk window 50 Hz (praproses dan model v13, sama dengan ecgfounder.jalankan_eksternal) → cache."""
    import torch
    p = CACHE / f"{nama}.npy"
    if p.exists():
        return np.load(p)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E13.muat_model(device)
    out = []
    for i in range(0, len(ecg50), 4096):
        lo, _ = E13.inferensi(model, E13.praproses(ecg50[i:i + 4096], hrv_core.FS_DEVICE), device)
        out.append(lo)
    out = np.concatenate(out)
    np.save(p, out)
    log(f"  ECGFounder: {len(out)} window ({device})")
    return out


def paket(X, M, nama="paket", potong=40000):
    """Probabilitas paket v3.1 dan kelas paket v1 (v3.2.takikardia.load_packages) untuk window yang lolos cek → cache.
    Diproses per `potong` window (v1 sekaligus untuk 322 ribu window kehabisan memori); prediksi tiap window tidak bergantung
    pada window lain, dan notebook meng-assert hasilnya sama dengan run pertama pada window AF yang sama."""
    p = CACHE / f"{nama}.npz"
    if p.exists():
        z = np.load(p)
        return z["p31"], z["v1"]
    from predictor import ECGPredictor
    v31 = ECGPredictor(str(REPO / "v3.1_kalibrasi_produksi/model_export"), name="sihedaf_v3_1", device="cpu")
    run_v1 = T.load_packages()["v1"]
    p31 = np.concatenate([v31.predict_proba_sequences(X[i:i + potong], M[i:i + potong]) for i in range(0, len(X), potong)])
    v1 = np.concatenate([run_v1(list(zip(X[i:i + potong], M[i:i + potong]))) for i in range(0, len(X), potong)])
    np.savez_compressed(p, p31=p31, v1=v1)
    return p31, v1


def per_subjek(y2, pred, subj):
    """Sensitivitas dan alarm (window Normal) per subjek; None bila subjek tidak punya kelas itu."""
    out = {}
    for s in np.unique(subj):
        m = subj == s
        a, n = m & (y2 == 1), m & (y2 == 0)
        out[str(s)] = dict(n_af=int(a.sum()), n_normal=int(n.sum()), se=float(pred[a].mean()) if a.any() else None,
                           alarm=float(pred[n].mean()) if n.any() else None)
    return out


def alarm_pita(y2, pred, hr):
    """Alarm pada window Normal per pita detak (pita v11.1)."""
    return {k: dict(alarm=float(pred[(y2 == 0) & (hr >= lo) & (hr < hi)].mean()) if ((y2 == 0) & (hr >= lo) & (hr < hi)).any() else None,
                    n=int(((y2 == 0) & (hr >= lo) & (hr < hi)).sum())) for k, lo, hi in DK.PITA}
