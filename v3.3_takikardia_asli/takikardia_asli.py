"""
takikardia_asli.py — logika v3.3, dipakai notebook dan laporan.

Evaluasi saja: tidak ada training, paket produksi tidak disentuh. Menindaklanjuti riset 6 November 2026:
1. Alarm palsu pada sinus takikardia ASLI (EKG saat latihan fisik: EPFL, Wrist, EPHNOGRAM), lewat pipeline perangkat
   (resample 50 Hz → detektor R-peak → window 10 s) dan lewat R-peak manual (memisahkan kesalahan detektor dari model).
   Model: paket v1, v2, v3.1, dan model fold yang menyimpan bobot (v7, v8, v11) dengan bias fold masing-masing.
2. Aturan produk v3.2 ("≥ 110 bpm & CV IBI ≤ 0,05 → takikardia teratur, perlu dicek") diukur di data asli.
3. Keluaran abstain (conformal, LAC, cakupan 90%): window yang himpunan prediksinya memuat Normal dan kelas lain → "tidak
   yakin", bukan alarm. Ambang dipelajari cross-fit di AFDB.
4. Metrik episode gaya ANSI/AAMI EC57 (episode dan durasi: sensitivitas dan +P) untuk AFIB dan AFL di AFDB.
"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "v3.2_uji_takikardia"), str(REPO / "v3_evaluasi_robust"), str(REPO / "v2_revisi"),
                str(REPO / "v12_shdb_atrium"), str(REPO / "v11_model_tim")]
import takikardia as T  # noqa: E402  (v3.2)
import sihedaf_cv as cv  # noqa: E402
import hrv_core  # noqa: E402
import atrium as A12  # noqa: E402  (pembaca WFDB tanpa wfdb, dibuktikan sama di v12)

DATA = REPO / "data"
CACHE = REPO / "cache/v3_3"
OOF_DIR = REPO / "cache/oof_robust/detector_e30_r5_s2026"
N_REPEATS, SEED, N_BOOT = 5, 2026, 2000
CLASSES = ("Normal", "AFIB", "AFL")
PITA = [(0, 100), (100, 110), (110, 120), (120, 130), (130, 140), (140, 150), (150, 999)]
STRIDE = hrv_core.WINDOW_SEC                     # window tidak tumpang tindih di data latihan fisik
ATURAN = (110.0, 0.05)                           # aturan produk v3.2: detak ≥ 110 bpm dan CV IBI ≤ 0,05
ALPHA = 0.10                                     # conformal: cakupan 90%
KODE_DETAK = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 30, 34, 35, 38}   # kode anotasi detak MIT


# ── 1. data latihan fisik ──────────────────────────────────────────────
def rekaman_epfl():
    """EPFL high-intensity: 99 segmen 20 s, 250 Hz, sinyal mentah (µV), R-peak manual (direview dokter)."""
    akar = DATA / "epfl_hi/isi/ECG_high_intensity_exercise"
    for f in sorted((akar / "ecg_segments").glob("ecg_raw_sub*_seg*.csv")):
        sub, seg = map(int, re.match(r"ecg_raw_sub(\d+)_seg(\d+)", f.stem).groups())
        lab = akar / "manual_annotations" / f"ecg_sub{sub}_seg{seg}_labels.csv"
        if not lab.exists():
            continue
        x = np.loadtxt(f, delimiter=",", usecols=0) / 1000.0
        r = np.array([int(b.split(",")[0]) for b in lab.read_text().splitlines() if "Rpeak" in b])
        yield dict(sumber="EPFL", subjek=f"epfl{sub:02d}", rekaman=f"sub{sub}_seg{seg}", fs=250.0, ecg=x, rpeak=r,
                   aktivitas=f"bersepeda maksimal, seg{seg}")


def rekaman_wrist():
    """PhysioNet wrist 1.0.0: EKG dada 256 Hz saat jalan/lari/bersepeda, R-peak referensi (.atr)."""
    akar = DATA / "wrist"
    for nama in (akar / "RECORDS").read_text().split():
        base = akar / nama
        x, fs = A12.baca_dat(base, 0)
        s, kode, _ = A12.baca_atr(base)
        kom = [b[1:].strip() for b in base.with_suffix(".hea").read_text().splitlines() if b.startswith("#")]
        yield dict(sumber="Wrist", subjek=f"wrist{nama.split('_')[0][1:]}", rekaman=nama, fs=fs, ecg=x,
                   rpeak=s[np.isin(kode, list(KODE_DETAK))], aktivitas=(kom[0] if kom else nama.split("_", 1)[1]))


def rekaman_ephnogram():
    """EPHNOGRAM 1.0.0: EKG (kanal 'ECG', 8 kHz) + suara jantung saat istirahat dan latihan fisik, 24 subjek. Subjek dan
    skenario dari ECGPCGSpreadsheet.csv. Tanpa R-peak referensi dan tanpa label ritme (subjek sehat)."""
    import csv
    akar = DATA / "ephnogram"
    with (akar / "ECGPCGSpreadsheet.csv").open(newline="") as f:
        meta = {b["Record Name"]: b for b in csv.DictReader(f) if b.get("Record Name")}
    for hea in sorted((akar / "WFDB").glob("*.hea")):
        base = hea.with_suffix("")
        baris = hea.read_text().splitlines()
        n_sig = int(baris[0].split()[1])
        nama_kanal = [b.split()[-1] for b in baris[1:1 + n_sig]]
        if "ECG" not in nama_kanal or base.name not in meta:
            continue
        x, fs = A12.baca_dat(base, nama_kanal.index("ECG"))
        m = meta[base.name]
        yield dict(sumber="EPHNOGRAM", subjek=f"eph_{m['Subject ID']}", rekaman=base.name, fs=float(fs), ecg=x,
                   rpeak=None, aktivitas=m["Recording Scenario"])


# ── 2. window ──────────────────────────────────────────────────────────
def _potong(t_end, ibi, dur):
    """Logika window yang SAMA dengan hrv_core.signal_to_windows (dibuktikan di bukti_window)."""
    out = []
    for s in np.arange(0.0, max(dur - hrv_core.WINDOW_SEC, 0.0) + 1e-9, STRIDE):
        i0, i1 = np.searchsorted(t_end, [s, s + hrv_core.WINDOW_SEC], side="left")
        w = ibi[i0:i1]
        ok, alasan = hrv_core.window_quality(w)
        out.append((float(s), w, ok, alasan))
    return out


def bukti_window(rek):
    """Window dari _potong (dengan R-peak detektor) harus identik dengan hrv_core.signal_to_windows."""
    win, peaks = hrv_core.signal_to_windows(rek["ecg"], rek["fs"], STRIDE)
    t_end, ibi = hrv_core.peaks_to_ibi(peaks, hrv_core.FS_DEVICE)
    dur = hrv_core.resample_signal(rek["ecg"], rek["fs"]).size / hrv_core.FS_DEVICE
    mine = _potong(t_end, ibi, dur)
    assert len(mine) == len(win)
    for (s, w, ok, a), ref in zip(mine, win):
        assert s == ref["start_sec"] and ok == ref["ok"] and a == ref["reason"] and np.array_equal(w, ref["ibi_ms"])
    return len(win)


def jendela(rek):
    """Window perangkat (detektor 50 Hz) dan window R-peak manual (bila ada) untuk satu rekaman."""
    x = np.nan_to_num(np.asarray(rek["ecg"], np.float64))
    win, _ = hrv_core.signal_to_windows(x, rek["fs"], STRIDE)
    baris = [("perangkat", w["start_sec"], w["ibi_ms"], w["ok"], w["reason"]) for w in win]
    if rek.get("rpeak") is not None and len(rek["rpeak"]) > 2:
        t_end, ibi = hrv_core.peaks_to_ibi(np.sort(rek["rpeak"]), rek["fs"])
        baris += [("manual", s, w, ok, a) for s, w, ok, a in _potong(t_end, ibi, x.size / rek["fs"])]
    return baris


def bangun_jendela(sumber=("EPFL", "Wrist", "EPHNOGRAM"), log=print):
    """Semua window latihan fisik → cache/v3_3/jendela.npz (X, M siap model untuk window ok)."""
    loader = {"EPFL": rekaman_epfl, "Wrist": rekaman_wrist, "EPHNOGRAM": rekaman_ephnogram}
    kol = {k: [] for k in ("sumber", "subjek", "rekaman", "aktivitas", "mode", "start", "ok", "alasan", "hr", "cv", "n_ibi")}
    X, M = [], []
    for s in sumber:
        n0 = len(kol["sumber"])
        for rek in loader[s]():
            for mode, st, w, ok, a in jendela(rek):
                w = np.asarray(w, np.float64)
                for k, v in (("sumber", s), ("subjek", rek["subjek"] or rek["rekaman"]), ("rekaman", rek["rekaman"]),
                             ("aktivitas", rek["aktivitas"]), ("mode", mode), ("start", st), ("ok", ok), ("alasan", a),
                             ("hr", 60000.0 / w.mean() if w.size else np.nan),
                             ("cv", w.std() / w.mean() if w.size else np.nan), ("n_ibi", w.size)):
                    kol[k].append(v)
                xs, ms = hrv_core.build_sequence(w) if ok else (np.zeros((hrv_core.MAX_BEATS, 2), np.float32),
                                                                  np.zeros(hrv_core.MAX_BEATS, np.float32))
                X.append(xs)
                M.append(ms)
        log(f"{s}: {len(kol['sumber']) - n0} window")
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {k: np.array(v) for k, v in kol.items()}
    np.savez_compressed(CACHE / "jendela.npz", X=np.stack(X), M=np.stack(M), **out)
    return muat_jendela()


def muat_jendela():
    z = np.load(CACHE / "jendela.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


# ── 3. model ───────────────────────────────────────────────────────────
def _impor(nama, path):
    spec = importlib.util.spec_from_file_location(nama, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Model fold yang menyimpan bobot. Bias tiap model fold = bias cross-fit fold-nya dari OOF AFDB keluarga itu.
KELUARGA = {
    "v7_130": dict(label="v7a: + sinus takikardia sintetis 100–130 bpm", jenis="v7", varian="takikardia130"),
    "v7_160": dict(label="v7b: + sinus takikardia sintetis 100–160 bpm", jenis="v7", varian="takikardia160"),
    "v8_8a": dict(label="v8a: AFDB + Chapman (termasuk ST asli)", jenis="v8", varian="afdb_chapman"),
    "v8_8c": dict(label="v8c: AFDB + Chapman tanpa label AFL", jenis="v8", varian="afdb_chapman_bersih"),
    "v11_bigru": dict(label="v11 BiGRU bawaan", jenis="v11", varian=("bigru", "bawaan")),
    "v11_bigru_wp100": dict(label="v11 BiGRU bobot penuh", jenis="v11", varian=("bigru", "wp100")),
    "v11_tcn_wp100": dict(label="v11 TCN bobot penuh", jenis="v11", varian=("tcn", "wp100")),
}


def _oof_keluarga(kel):
    j, v = kel["jenis"], kel["varian"]
    if j == "v7":
        return np.stack([np.load(OOF_DIR / f"MicroTransformer_{v}_r{r}.npz")["oof"] for r in range(N_REPEATS)])
    if j == "v8":
        return np.stack([np.load(REPO / f"cache/v8/oof/{v}_r{r}.npz")["oof_afdb"] for r in range(N_REPEATS)])
    from bersama import runner
    h = runner.muat_hasil(REPO / "cache/v11" / v[0] / v[1])
    assert list(h.ulangan) == list(range(N_REPEATS))
    return h.oof


def _jaringan(kel, r, k, device):
    """(net, mu, sd) model fold (r, k)."""
    import torch
    from sihedaf_models import build_model
    j, v = kel["jenis"], kel["varian"]
    if j in ("v7", "v8"):
        p = REPO / f"cache/{j}/model/{v}_r{r}_k{k}.pt"
        z = torch.load(p, map_location="cpu", weights_only=True)
        net = build_model("MicroTransformer", **z["model_kwargs"])
    else:
        p = REPO / "cache/v11" / v[0] / v[1] / f"r{r}_k{k}.pt"
        z = torch.load(p, map_location="cpu", weights_only=True)
        net = _impor(f"model_{v[0]}", REPO / "v11_model_tim" / v[0] / "model.py").buat_net(z["setelan"])
    net.load_state_dict(z["state_dict"])
    return net.to(device).eval(), z["mu"].numpy(), z["sd"].numpy()


def prediksi_keluarga(kunci, X, M, y_afdb, folds, device="cpu"):
    """Probabilitas setelah bias fold, (R, 5, N, 3), untuk satu keluarga model fold pada window baru."""
    kel = KELUARGA[kunci]
    oof = _oof_keluarga(kel)
    out = np.zeros((N_REPEATS, 5, len(X), 3), np.float32)
    for r in range(N_REPEATS):
        _, b = cv.crossfit_bias(y_afdb, oof[r], folds[r])
        for k in range(5):
            net, mu, sd = _jaringan(kel, r, k, device)
            out[r, k] = cv.apply_bias(cv.predict_dl(net, X, M, mu, sd, device), b[k])
    return out


def paket():
    """Paket siap pakai: v1 (kelas), v2 dan v3.1 (probabilitas setelah bias paket)."""
    from predictor import ECGPredictor
    run = T.load_packages()
    v2 = ECGPredictor(str(REPO / "v2_revisi/model_export"), device="cpu")
    v31 = ECGPredictor(str(REPO / "v3.1_kalibrasi_produksi/model_export"), name="sihedaf_v3_1", device="cpu")
    return run["v1"], v2, v31


def prediksi_semua(J, device="cpu", log=print):
    """Kelas per window untuk semua model; keluarga fold → (R·5, N). Disimpan ke cache/v3_3/prediksi.npz."""
    ok = J["ok"]
    X, M = J["X"][ok], J["M"][ok]
    z = np.load(REPO / "cache/dataset_v2_detector.npz")
    y_afdb = z["y"]
    folds = np.load(OOF_DIR / "folds.npz")["folds"]
    run_v1, v2, v31 = paket()
    seqs = list(zip(X, M))
    hasil = {"v1": run_v1(seqs)[None], "v2_prod": v2.predict_proba_sequences(X, M).argmax(1)[None]}
    p31 = v31.predict_proba_sequences(X, M)
    hasil["v31"] = p31.argmax(1)[None]
    for kunci in KELUARGA:
        p = prediksi_keluarga(kunci, X, M, y_afdb, folds, device)
        hasil[kunci] = p.argmax(-1).reshape(-1, len(X))
        log(f"  {kunci}: {hasil[kunci].shape[0]} model fold")
    np.savez_compressed(CACHE / "prediksi.npz", p_v31=p31, **hasil)
    return hasil, p31


# ── 4. alarm palsu per pita detak, CI bootstrap subjek ─────────────────
def _boot_subjek(subj, nilai, n_boot=N_BOOT, seed=SEED):
    """Rata-rata nilai per window, bootstrap subjek (resample subjek, semua window subjek ikut)."""
    u, inv = np.unique(subj, return_inverse=True)
    s = np.bincount(inv, weights=nilai, minlength=len(u))
    n = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (n_boot, len(u)))
    b = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(s.sum() / n.sum()), np.percentile(b, [2.5, 97.5]), b


def alarm_per_pita(J, kelas, mask):
    """kelas: (model, N_ok). Fraksi window (rata-rata model) yang ditebak AFIB/AFL per pita detak, + CI subjek."""
    hr, subj = J["hr"][J["ok"]][mask], J["subjek"][J["ok"]][mask]
    alarm = (kelas[:, mask] != 0).mean(0)
    out = {}
    for lo, hi in PITA:
        m = (hr >= lo) & (hr < hi)
        if m.sum() == 0:
            continue
        t, ci, _ = _boot_subjek(subj[m], alarm[m])
        out[f"{lo}–{hi if hi < 999 else '∞'} bpm"] = dict(n=int(m.sum()), subjek=int(np.unique(subj[m]).size), alarm=t,
                                                       ci=ci.tolist())
    return out


def banding_alarm(J, kelas_a, kelas_b, mask, hr_min=120.0):
    """Selisih alarm (a − b) pada window ≥ hr_min, bootstrap subjek berpasangan. P = P(Δ < 0) (a lebih sedikit alarm)."""
    hr, subj = J["hr"][J["ok"]][mask], J["subjek"][J["ok"]][mask]
    m = hr >= hr_min
    da = (kelas_a[:, mask] != 0).mean(0)[m] - (kelas_b[:, mask] != 0).mean(0)[m]
    t, ci, b = _boot_subjek(subj[m], da)
    return dict(delta=t, ci=ci.tolist(), p_kurang=float((b < 0).mean()), n=int(m.sum()))


# ── 5. aturan produk dan abstain conformal ─────────────────────────────
def kena_aturan(hr, cvv, aturan=ATURAN):
    return (hr >= aturan[0]) & (cvv <= aturan[1])


def ambang_conformal(y, p, alpha=ALPHA):
    """LAC split-conformal: skor = 1 − p(kelas benar); ambang = kuantil (1 − α)(n + 1)/n."""
    s = 1.0 - p[np.arange(len(y)), y]
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((1 - alpha) * (n + 1)) / n), method="higher"))


def keputusan_abstain(p, q, kosong_ragu=False):
    """0/1/2 = kelas yakin; 3 = tidak yakin. Tidak yakin bila himpunan prediksi memuat Normal dan kelas lain; dengan
    kosong_ragu=True juga bila himpunan kosong (tidak ada kelas yang cukup yakin). Selain itu argmax."""
    S = (1.0 - p) <= q
    out = np.where(S.sum(1) == 1, S.argmax(1), p.argmax(1))
    ragu = ((S.sum(1) > 1) & S[:, 0]) | ((S.sum(1) == 0) if kosong_ragu else False)
    return np.where(ragu, 3, out)


def abstain_afdb(alpha=ALPHA, kosong_ragu=False):
    """v3.1 (OOF MicroTransformer + bias cross-fit): ambang conformal cross-fit per fold, keputusan per window (R, N)."""
    d = T.load()
    y, folds = d["y"], d["folds"]
    out, q_all = [], []
    for r in range(N_REPEATS):
        adj, _ = cv.crossfit_bias(y, d["probs"]["v31"][r], folds[r])
        dec = np.empty(len(y), np.int64)
        for k in range(5):
            tr, te = folds[r] != k, folds[r] == k
            q = ambang_conformal(y[tr], adj[tr], alpha)
            q_all.append(q)
            dec[te] = keputusan_abstain(adj[te], q, kosong_ragu)
        out.append(dec)
    return d, np.stack(out), np.array(q_all)


def ambang_paket(alpha=ALPHA):
    """Ambang conformal untuk paket v3.1 dari OOF AFDB dengan bias paket (proksi kalibrasi paket yang dilatih penuh)."""
    import json
    d = T.load()
    b = np.array(json.loads((REPO / "v3.1_kalibrasi_produksi/model_export/sihedaf_v3_1.json").read_text())["class_bias"])
    p = np.concatenate([cv.apply_bias(d["probs"]["v31"][r], b) for r in range(N_REPEATS)])
    return ambang_conformal(np.tile(d["y"], N_REPEATS), p, alpha)


# ── 6. metrik episode gaya EC57 ────────────────────────────────────────
def _episode(t0, t1, aktif):
    """Interval (mulai, akhir) dari run window aktif yang bersambung (celah ≤ 0,5 s dianggap bersambung)."""
    ep = []
    for a, b, on in zip(t0, t1, aktif):
        if not on:
            continue
        if ep and a - ep[-1][1] <= 0.5:
            ep[-1][1] = max(ep[-1][1], b)
        else:
            ep.append([a, b])
    return np.array(ep, float).reshape(-1, 2)


def _overlap(A, B):
    tot = 0.0
    for a0, a1 in A:
        for b0, b1 in B:
            tot += max(0.0, min(a1, b1) - max(a0, b0))
    return tot


def metrik_episode(rec, t_start, y, pred, kelas, langkah=5.0):
    """
    EC57 (episode & durasi): Se episode = episode referensi yang tumpang tindih dengan deteksi; +P episode = episode deteksi
    yang tumpang tindih dengan referensi; Se/+P durasi = durasi tumpang tindih / durasi referensi / deteksi. Tiap window
    mewakili [t + (10 − langkah)/2, t + (10 + langkah)/2] agar window bertumpuk (langkah 5 s) tidak dihitung ganda.
    Ditambah alarm palsu per jam: episode deteksi tanpa tumpang tindih referensi.
    """
    tp_e = n_ref = tp_d = n_det = ov = dur_ref = dur_det = 0.0
    jam = 0.0
    for r in np.unique(rec):
        m = rec == r
        o = np.argsort(t_start[m])
        t = t_start[m][o]
        c0 = t + (hrv_core.WINDOW_SEC - langkah) / 2
        c1 = c0 + langkah
        R = _episode(c0, c1, y[m][o] == kelas)
        D = _episode(c0, c1, pred[m][o] == kelas)
        jam += langkah * m.sum() / 3600.0
        n_ref += len(R)
        n_det += len(D)
        tp_e += sum(_overlap([e], D) > 0 for e in R)
        tp_d += sum(_overlap([e], R) > 0 for e in D)
        ov += _overlap(R, D)
        dur_ref += float((R[:, 1] - R[:, 0]).sum()) if len(R) else 0.0
        dur_det += float((D[:, 1] - D[:, 0]).sum()) if len(D) else 0.0
    return dict(se_episode=tp_e / max(n_ref, 1), ppv_episode=tp_d / max(n_det, 1), se_durasi=ov / max(dur_ref, 1e-9),
                ppv_durasi=ov / max(dur_det, 1e-9), n_ref=int(n_ref), n_det=int(n_det),
                palsu_per_jam=(n_det - tp_d) / max(jam, 1e-9))
