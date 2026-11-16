"""
kuantisasi.py — logika v15 (pra-registrasi di RANCANGAN.md, dikunci 6 November 2026 sebelum CPSC 2021 dibuka).

1. Syarat data CPSC 2021: hitung subjek dan window AFL yang lolos (≥ 5 subjek dan ≥ 200 window). Bila tidak terpenuhi, H1/H2
   ditulis "tidak cukup data" (hasil utama).
2. Penyimpangan tercatat (RANCANGAN bagian 8): H1 dan H2 diuji pada PTB-XL sebagai analisis tambahan, bukan hasil utama.
3. H2 di AFDB dengan fold resmi (latih di fold lain + SHDB-AF), berpasangan dengan v3.1.
Fitur `kuantisasi` dan `p_konsisten` dari `v12_shdb_atrium/atrium.fitur_atrium(..., revisi=1)` tanpa perubahan.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "v12_shdb_atrium"), str(REPO / "v3_evaluasi_robust"), str(REPO / "v2_revisi"),
                str(REPO / "v13_ecgfounder"), str(REPO / "v11_model_tim")]
import atrium as A12  # noqa: E402
import sihedaf_cv as cv  # noqa: E402

CACHE = REPO / "cache/v15"
CPSC = REPO / "data/cpsc2021"
FITUR_UJI = ("kuantisasi", "p_konsisten")
IDX = [A12.FITUR.index(f) for f in FITUR_UJI]
N_BOOT, SEED = 2000, 2026
SYARAT = dict(subjek=5, window=200)
BIAS_PAKET = None


# ── 1. syarat data CPSC 2021 ───────────────────────────────────────────
def rekaman_cpsc():
    """Rekaman yang benar-benar dirilis (RECORDS memuat 11 rekaman yang tidak ada di SHA256SUMS resmi)."""
    return [r for r in (CPSC / "RECORDS").read_text().split() if (CPSC / (r + ".hea")).exists()]


def segmen_ritme(rec):
    """[(mulai_s, akhir_s, label)] dari anotasi ritme MIT ('(N', '(AFIB', '(AFL'); ritme berlaku sampai label berikutnya."""
    base = CPSC / rec
    s, _, aux = A12.baca_atr(base)
    fs, n = A12.baca_hea(base)[:2]
    lab = [(int(si), a.strip("\x00").strip()) for si, a in zip(s, aux) if a.strip("\x00").strip().startswith("(")]
    return [(a / fs, (lab[j + 1][0] if j + 1 < len(lab) else n) / fs, lb) for j, (a, lb) in enumerate(lab)]


def syarat_cpsc():
    """Jumlah subjek dan window 10 s murni per ritme (window seluruhnya dalam satu segmen), sebelum cek kualitas."""
    subj = {"(AFL": set(), "(AFIB": set(), "(N": set()}
    win = {"(AFL": 0, "(AFIB": 0, "(N": 0}
    menit = {"(AFL": 0.0, "(AFIB": 0.0, "(N": 0.0}
    for r in rekaman_cpsc():
        sid = r.split("/")[1].split("_")[1]
        for a, b, lb in segmen_ritme(r):
            if lb in win:
                win[lb] += int((b - a) // 10)
                menit[lb] += (b - a) / 60
                subj[lb].add(sid)
    afl_ok = len(subj["(AFL"]) >= SYARAT["subjek"] and win["(AFL"] >= SYARAT["window"]
    return dict(rekaman=len(rekaman_cpsc()), subjek={k: len(v) for k, v in subj.items()}, window=win,
                menit={k: round(v, 1) for k, v in menit.items()}, subjek_afl=sorted(subj["(AFL"], key=int),
                syarat_terpenuhi=bool(afl_ok))


# ── 2. fitur PTB-XL (analisis tambahan) ────────────────────────────────
def fitur_ptbxl(log=print):
    """PTB-XL lead I (v13.ptbxl) → fitur atrium revisi 1 + probabilitas paket v3.1 per rekaman 10 s. Cache cache/v15/ptbxl.npz."""
    p = CACHE / "ptbxl.npz"
    if p.exists():
        return dict(np.load(p))
    import ecgfounder as E13
    A12._modul()
    import hrv_core
    import ekg as E6                                                  # v6 (dipasang di sys.path oleh atrium._modul)
    P = E13.ptbxl()
    fit = np.full((len(P["y"]), len(A12.FITUR)), np.nan, np.float32)
    pola = np.full(len(P["y"]), -1, np.int8)
    for i, x in enumerate(P["ecg"]):
        x50, xf = E6.preprocess_record(np.asarray(x, np.float64), P["fs"])
        peaks = hrv_core.detect_rpeaks(x50, hrv_core.FS_DEVICE)
        fit[i] = A12.fitur_atrium(xf, np.round(peaks).astype(np.int64), 0.0, revisi=1)
        if len(peaks) > 2:
            pola[i] = A12.pola_rr(np.diff(peaks) / hrv_core.FS_DEVICE * 1000.0)
        if (i + 1) % 5000 == 0:
            log(f"  fitur PTB-XL {i + 1}/{len(P['y'])}")
    p31 = E13.rr_paket_v31(P["ecg"], P["fs"])
    out = dict(fitur=fit, pola=pola, p31=p31, y=P["y"], st=P["st"], pasien=P["pasien"], ecg_id=P["ecg_id"])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **out)
    return out


# ── 3. H1: AUROC AFL vs AFIB dengan CI bootstrap pasien ────────────────
def _auroc(pos, neg):
    if pos.size == 0 or neg.size == 0:
        return np.nan
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1.0
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def auroc_boot(nilai, y, pasien, n_boot=N_BOOT, seed=SEED):
    """AUROC AFL (y=2) vs AFIB (y=1) dengan bootstrap pasien berstrata (pasien ber-AFL / tidak)."""
    m = np.isin(y, (1, 2)) & np.isfinite(nilai)
    v, yy, pp = nilai[m], y[m], pasien[m]
    titik = _auroc(v[yy == 2], v[yy == 1])
    u = np.unique(pp)
    ber_afl = np.array([(yy[pp == q] == 2).any() for q in u])
    grup = [np.flatnonzero(pp == q) for q in u]
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        pil = np.r_[rng.choice(np.flatnonzero(ber_afl), ber_afl.sum()), rng.choice(np.flatnonzero(~ber_afl), (~ber_afl).sum())]
        idx = np.concatenate([grup[i] for i in pil])
        b.append(_auroc(v[idx][yy[idx] == 2], v[idx][yy[idx] == 1]))
    b = np.array(b)
    return dict(auroc=titik, ci=np.nanpercentile(b, [2.5, 97.5]).tolist(), n_afl=int((yy == 2).sum()),
                n_afib=int((yy == 1).sum()), pasien_afl=int(len(np.unique(pp[yy == 2]))))


# ── 4. H2: tahap gabungan (logit v3.1 + 2 fitur) ───────────────────────
def _bias_paket():
    return np.array(json.loads((REPO / "v3.1_kalibrasi_produksi/model_export/sihedaf_v3_1.json").read_text())["class_bias"])


def masukan(logp31, fit, median_latih):
    """[log p v3.1 (3), kuantisasi, p_konsisten]; NaN fitur → median data latih."""
    f = fit[:, IDX].astype(np.float64)
    f = np.where(np.isfinite(f), f, median_latih)
    return np.column_stack([logp31, f])


def latih_gabungan(Xin, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    m.fit(Xin, y)
    return m


def data_shdb():
    """SHDB-AF (window v12 kanal 0): fitur revisi 1, label, subjek, dan log-probabilitas paket v3.1 dari RR."""
    from predictor import ECGPredictor
    f = A12.muat_fitur(revisi=1)["SHDB-AF"]
    X, M = [], []
    for p in sorted((A12.CACHE / "shdb").glob("*_k0.npz")):
        z = np.load(p)
        X.append(z["X"]); M.append(z["M"])
    P = ECGPredictor(str(REPO / "v3.1_kalibrasi_produksi/model_export"), name="sihedaf_v3_1", device="cpu")
    p = P.predict_proba_sequences(np.concatenate(X), np.concatenate(M))
    return dict(fitur=f["fitur"], y=f["y"], pasien=f["pasien"], logp=np.log(np.clip(p, 1e-7, 1)))


def _afdb_logp(r):
    """log p paket-like untuk AFDB: OOF MicroTransformer ulangan r + bias paket v3.1 (sama dengan kalibrasi paket)."""
    oof = np.load(REPO / f"cache/oof_robust/detector_e30_r5_s2026/MicroTransformer_r{r}.npz")["oof"]
    return np.log(np.clip(cv.apply_bias(oof, _bias_paket()), 1e-7, 1))


def h2_afdb(log=print):
    """OOF tahap gabungan di AFDB (R, N, 3): latih di fold lain AFDB + seluruh SHDB-AF, uji fold resmi."""
    p = CACHE / "oof_gabungan_afdb.npz"
    if p.exists():
        return np.load(p)["oof"]
    fa = A12.muat_fitur(revisi=1)["AFDB"]
    sh = data_shdb()
    folds = np.load(REPO / "cache/oof_robust/detector_e30_r5_s2026/folds.npz")["folds"]
    out = np.zeros((len(folds), len(fa["y"]), 3), np.float32)
    for r in range(len(folds)):
        lp = _afdb_logp(r)
        for k in range(5):
            tr, te = folds[r] != k, folds[r] == k
            med = np.nanmedian(np.r_[fa["fitur"][tr][:, IDX], sh["fitur"][:, IDX]], 0)
            Xtr = np.r_[masukan(lp[tr], fa["fitur"][tr], med), masukan(sh["logp"], sh["fitur"], med)]
            m = latih_gabungan(Xtr, np.r_[fa["y"][tr], sh["y"]])
            out[r, te] = m.predict_proba(masukan(lp[te], fa["fitur"][te], med))
        log(f"  gabungan AFDB ulangan {r} selesai")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, oof=out)
    return out


def h2_ptbxl(P):
    """Tahap gabungan dilatih di seluruh AFDB (ulangan 0) + SHDB-AF, diuji di PTB-XL tanpa penyetelan (argmax).
    Return (prob gabungan, mask window yang punya RR terbaca)."""
    fa = A12.muat_fitur(revisi=1)["AFDB"]
    sh = data_shdb()
    med = np.nanmedian(np.r_[fa["fitur"][:, IDX], sh["fitur"][:, IDX]], 0)
    m = latih_gabungan(np.r_[masukan(_afdb_logp(0), fa["fitur"], med), masukan(sh["logp"], sh["fitur"], med)],
                       np.r_[fa["y"], sh["y"]])
    ok = np.isfinite(P["p31"]).all(1)
    prob = np.full((len(P["y"]), 3), np.nan, np.float32)
    prob[ok] = m.predict_proba(masukan(np.log(np.clip(P["p31"][ok], 1e-7, 1)), P["fitur"][ok], med))
    return prob, ok


def banding_pasien(y, pred_a, pred_b, pasien, n_boot=N_BOOT, seed=SEED):
    """Bootstrap pasien berpasangan; fungsi dari v13 (ecgfounder.banding_pasien), tidak disalin."""
    import ecgfounder as E13
    return E13.banding_pasien(y, pred_a, pred_b, pasien, n_boot, seed)


def h2_stres():
    """Protokol butir 9 untuk tahap gabungan: set uji stres sintetis v3.2 (IBI saja, tanpa EKG) → fitur = median data latih
    (RANCANGAN bagian 4). Return {jenis variasi: ambang bpm} untuk gabungan dan paket v3.1."""
    sys.path.insert(0, str(REPO / "v3.2_uji_takikardia"))
    import takikardia as T
    from bersama import data as D
    from predictor import ECGPredictor
    fa = A12.muat_fitur(revisi=1)["AFDB"]
    sh = data_shdb()
    med = np.nanmedian(np.r_[fa["fitur"][:, IDX], sh["fitur"][:, IDX]], 0)
    m = latih_gabungan(np.r_[masukan(_afdb_logp(0), fa["fitur"], med), masukan(sh["logp"], sh["fitur"], med)],
                       np.r_[fa["y"], sh["y"]])
    uji = D.set_uji_stres()
    P = ECGPredictor(str(REPO / "v3.1_kalibrasi_produksi/model_export"), name="sihedaf_v3_1", device="cpu")
    p31 = P.predict_proba_sequences(uji.X, uji.M)
    kg = m.predict_proba(masukan(np.log(np.clip(p31, 1e-7, 1)), np.full((len(p31), len(A12.FITUR)), np.nan), med)).argmax(1)
    out = {}
    for nama, kelas in (("gabungan", kg), ("v31", p31.argmax(1))):
        out[nama] = {}
        for jenis in T.JITTERS:
            tab = np.stack([np.bincount(kelas[(uji.jenis == jenis) & (uji.bpm == b)], minlength=3) / T.N_SYNTH for b in T.BPMS])
            out[nama][jenis] = T.normal_threshold(tab)
    return out
