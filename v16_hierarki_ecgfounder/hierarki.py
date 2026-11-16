"""
hierarki.py — logika v16 (pra-registrasi di RANCANGAN.md, dikunci 6 November 2026 sebelum metrik dihitung).

Sistem bertingkat dari model yang sudah ada, tanpa melatih jaringan baru:
  tahap 1 — ECGFounder zero-shot (v13) memutuskan sinus vs AF: d1 = log((pAF + pAFL) / pN);
  tahap 2 — hanya untuk window AF: AFIB vs AFL dari RR (OOF/paket v3.1) atau regresi logistik fitur atrium (v15).
H-0 (primer): tahap 1 tanpa penyetelan (AF bila d1 > 0), tahap 2 RR, b2 cross-fit. H-A: tahap 2 atrium. H-X: skor
log [pN, pAF·qAFIB, pAF·qAFL] dengan bias 3 kelas cross-fit biasa (tahap 1 ikut disetel di AFDB).
Semua masukan dibaca dari cache v3.3, v10, v12, v13, v15; tidak ada inferensi ECGFounder baru.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "v13_ecgfounder"), str(REPO / "v15_kuantisasi_rr"), str(REPO / "v3.3_takikardia_asli"),
                str(REPO / "v11.1_dua_kelas"), str(REPO / "v11_model_tim")]
import ecgfounder as E13  # noqa: E402
import kuantisasi as K15  # noqa: E402
import takikardia_asli as K33  # noqa: E402
import dua_kelas as DK  # noqa: E402  (v11.1)
from bersama import data as D, evaluasi as Ev  # noqa: E402

cv, T, A12 = E13.cv, K33.T, E13.A12
CACHE = REPO / "cache/v16"
GRID = cv.BIAS_GRID
PASIEN_21 = "08378"                       # AFL 2:1 (±139 bpm, teratur)
_SHDB: dict = {}


# ── tahap 1 dan 2 ──────────────────────────────────────────────────────
def tahap1(logit):
    """Logit ECGFounder (N, 150) → (pE (N, 3) = skor zero-shot v13, d1 = log((pAF + pAFL) / pN))."""
    pE = E13.skor_zero_shot(logit).astype(np.float64)
    return pE, np.log(np.clip(pE[:, 1] + pE[:, 2], 1e-30, 1)) - np.log(np.clip(pE[:, 0], 1e-30, 1))


def d2_rr(p3):
    """Tahap 2 RR: log(pAFL / pAFIB) dari probabilitas 3 kelas v3.1 (clip sama dengan sihedaf_cv.apply_bias)."""
    p3 = np.asarray(p3, np.float64)
    return np.log(np.clip(p3[..., 2], 1e-9, 1)) - np.log(np.clip(p3[..., 1], 1e-9, 1))


def putuskan(d1, d2, b2):
    """0 (Normal) bila d1 ≤ 0; selain itu 2 (AFL) bila d2 + b2 > 0, 1 (AFIB) bila tidak."""
    return np.where(d1 > 0, np.where(d2 + b2 > 0, 2, 1), 0).astype(np.int64)


def crossfit_b2(y, d1, d2, folds_r):
    """b2 fold k dipilih dari fold lain dengan Macro-F1 3 kelas (kandidat pertama terbaik, aturan fit_class_bias), dipasang
    ke fold k. Return (pred (N,), b2 (5,))."""
    pred = np.empty(len(y), np.int64)
    bb = []
    for k in np.unique(folds_r):
        tr, te = folds_r != k, folds_r == k
        skor = [cv._macro_f1_fast(y[tr], putuskan(d1[tr], d2[tr], b)) for b in GRID]
        b = float(GRID[int(np.argmax(skor))])
        pred[te] = putuskan(d1[te], d2[te], b)
        bb.append(b)
    return pred, np.array(bb)


def keputusan_h(y, d1, d2, folds):
    """Sistem bertingkat per ulangan. d2 (R, N) atau (N,). Return (pred (R, N), b2 (R, 5))."""
    d2 = np.broadcast_to(d2, (len(folds), len(y)))
    out = [crossfit_b2(y, d1, d2[r], folds[r]) for r in range(len(folds))]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def skor_hx(pE, p3):
    """H-X: [pN, pAF·qAFIB, pAF·qAFL]; pAF = pAF + pAFL ECGFounder, q = AFIB/AFL v3.1 dinormalkan. p3 (N, 3) atau (R, N, 3)."""
    p3 = np.asarray(p3, np.float64)
    q = p3[..., 1:] / np.clip(p3[..., 1:].sum(-1, keepdims=True), 1e-12, None)
    pN = np.broadcast_to(pE[:, :1], q.shape[:-1] + (1,))
    return np.concatenate([pN, (pE[:, 1:2] + pE[:, 2:3]) * q], -1).astype(np.float32)


# ── tahap 2 atrium (resep v15, biner, hanya window AF) ─────────────────
def _shdb_af():
    if not _SHDB:
        sh = K15.data_shdb()
        m = np.isin(sh["y"], (1, 2))
        _SHDB.update({k: v[m] for k, v in sh.items()})
    return _SHDB


def oof_atrium(log=print):
    """d2 tahap 2 atrium di AFDB (R, N): regresi logistik biner AFL vs AFIB (v15.latih_gabungan, masukan v15.masukan), dilatih
    per (ulangan, fold) pada window AFIB/AFL fold latih AFDB + seluruh SHDB-AF. Cache cache/v16/oof_atrium.npz."""
    p = CACHE / "oof_atrium.npz"
    if p.exists():
        return np.load(p)["d2"]
    fa = A12.muat_fitur(revisi=1)["AFDB"]
    sh = _shdb_af()
    folds = np.load(REPO / "cache" / D.OOF / "folds.npz")["folds"]
    af = np.isin(fa["y"], (1, 2))
    out = np.zeros((len(folds), len(fa["y"])), np.float32)
    for r in range(len(folds)):
        lp = K15._afdb_logp(r)
        for k in range(5):
            tr, te = (folds[r] != k) & af, folds[r] == k
            med = np.nanmedian(np.r_[fa["fitur"][tr][:, K15.IDX], sh["fitur"][:, K15.IDX]], 0)
            m = K15.latih_gabungan(np.r_[K15.masukan(lp[tr], fa["fitur"][tr], med), K15.masukan(sh["logp"], sh["fitur"], med)],
                                   (np.r_[fa["y"][tr], sh["y"]] == 2).astype(np.int64))
            out[r, te] = m.decision_function(K15.masukan(lp[te], fa["fitur"][te], med))
        log(f"  atrium ulangan {r} selesai")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, d2=out)
    return out


def model_atrium_penuh():
    """Tahap 2 atrium dilatih di seluruh window AFIB/AFL AFDB (log p ulangan 0) + SHDB-AF, untuk data eksternal (pola
    v15.h2_ptbxl). Return (model, median fitur pengisi NaN)."""
    fa = A12.muat_fitur(revisi=1)["AFDB"]
    sh = _shdb_af()
    af = np.isin(fa["y"], (1, 2))
    med = np.nanmedian(np.r_[fa["fitur"][af][:, K15.IDX], sh["fitur"][:, K15.IDX]], 0)
    m = K15.latih_gabungan(np.r_[K15.masukan(K15._afdb_logp(0)[af], fa["fitur"][af], med),
                                 K15.masukan(sh["logp"], sh["fitur"], med)],
                           (np.r_[fa["y"][af], sh["y"]] == 2).astype(np.int64))
    return m, med


# ── protokol butir 10 dan sensitivitas ─────────────────────────────────
def kandidat_h(y, d1, d2, rec, pasien):
    """(R, G, P, 3, 3): confusion matrix per pasien untuk setiap b2 di grid (sistem bertingkat H-0/H-A)."""
    return np.stack([np.stack([cv.patient_cms(y, putuskan(d1, d2r, b), rec, pasien) for b in GRID]) for d2r in d2])


def bersarang3(cand, pfold, W1):
    """Macro-F1 3 kelas bersarang per sampel bootstrap (B+1,; baris 0 = titik): di tiap ulangan dan fold, parameter dipilih
    ulang pada pasien latih berbobot lalu dipasang ke pasien uji. Loop sama dengan takikardia.nested_bootstrap (paritasnya
    di-assert di notebook), tetapi mengembalikan semua sampel agar bisa dipasangkan dengan v3.1."""
    R, G, P = cand.shape[:3]
    hasil = np.zeros((W1.shape[0], R))
    for r in range(R):
        flat = cand[r].reshape(G, P, 9)
        acc = np.zeros((W1.shape[0], 9))
        for k in range(5):
            tr, te = pfold[r] != k, pfold[r] == k
            cm_tr = (W1[:, tr] @ flat[:, tr].transpose(1, 0, 2).reshape(tr.sum(), G * 9)).reshape(-1, G, 3, 3)
            g = T._macro_f1_cm(cm_tr).argmax(1)
            acc += np.einsum("bpc,bp->bc", flat[g][:, te], W1[:, te])
        hasil[:, r] = T._macro_f1_cm(acc.reshape(-1, 3, 3))
    return hasil.mean(1)


def banding_bersarang(nest_k, nest_ref):
    dd = nest_k[1:] - nest_ref[1:]
    return dict(titik=float(nest_k[0]), ci=np.percentile(nest_k[1:], [2.5, 97.5]).tolist(), delta=float(nest_k[0] - nest_ref[0]),
                ci_delta=np.percentile(dd, [2.5, 97.5]).tolist(), p=float(np.mean(dd > 0)))


def tanpa_04936(d, d1, d2_h, oofs):
    """Sensitivitas tanpa 04936, parameter cross-fit dipilih ulang di subset (pola evaluasi.tanpa_04936).
    d2_h: {nama: d2 (R, N)} sistem bertingkat; oofs: {nama: OOF (R, N, 3)} dengan bias 3 kelas, harus memuat 'v31'.
    Catatan: pairs_v31_minus di keluaran = v3.1 − sistem."""
    m = d.records != T.SUSPECT
    pr = {k: np.stack([cv.crossfit_bias(d.y[m], o[r][m], d.folds[r][m])[0].argmax(1) for r in range(len(o))])
          for k, o in oofs.items()}
    for k, d2 in d2_h.items():
        pr[k] = keputusan_h(d.y[m], d1[m], d2[:, m], d.folds[:, m])[0]
    return T.metrics_with_ci(dict(y=d.y, rec=d.records), pr, m)


def dua_kelas(d, d1, pred_v31, W, W1, pfold, cand3_v31):
    """H2b: Normal vs AF/AFL. H-0 = keputusan tahap 1 (tanpa parameter); v3.1 = bias 3 kelas lalu AFIB + AFL digabung (v11.1).
    Titik + CI + banding berpasangan, dan versi bersarang (bias v3.1 dipilih ulang di tiap sampel)."""
    y2 = (d.y > 0).astype(np.int64)
    cm_h = DK.cm_pasien(y2, np.repeat((d1 > 0).astype(np.int64)[None], len(pred_v31), 0), d.records, d.patients)
    cm_v = DK.cm_pasien(y2, (pred_v31 > 0).astype(np.int64), d.records, d.patients)
    mh, bh = DK.ringkas(cm_h, W)
    mv, bv = DK.ringkas(cm_v, W)
    nh = DK.f1_makro_cm(np.einsum("bp,rpij->brij", W1, cm_h)).mean(1)
    nv = DK.bersarang_gabung(cand3_v31, pfold, W1)
    return {"H-0": mh, "v3.1": mv, "banding": DK.banding(mh, mv, bh, bv), "bersarang": banding_bersarang(nh, nv)}


def episode(d, pred):
    """Metrik episode gaya EC57 (v3.3) untuk AFIB dan AFL, rata-rata ulangan."""
    out = {}
    for kelas, n in ((1, "AFIB"), (2, "AFL")):
        ms = [K33.metrik_episode(d.records, d.t, d.y, p, kelas) for p in pred]
        out[n] = {m: float(np.mean([x[m] for x in ms])) for m in ms[0]}
    return out


# ── H5 keluaran keempat, H6 AUROC ──────────────────────────────────────
def _bagi(alarm, cek, sel):
    return dict(alarm=float(alarm[..., sel].mean()), cek=float(cek[..., sel].mean()),
                normal=float((~alarm & ~cek)[..., sel].mean()), n=int(sel.sum()))


def h5_afdb(d, d1, pred_h0, pred_v31):
    """'Takikardia teratur, perlu dicek' = tahap 1 sinus dan v3.1 (cross-fit) AFL. Proporsi alarm AF / perlu dicek / Normal
    per kelas asli dan per pasien AFL (rata-rata ulangan)."""
    alarm = pred_h0 != 0
    cek = (d1 <= 0)[None] & (pred_v31 == 2)
    out = {"kelas": {n: _bagi(alarm, cek, d.y == c) for c, n in enumerate(("Normal", "AFIB", "AFL"))}}
    out["pasien_afl"] = {str(p): _bagi(alarm, cek, (d.records == p) & (d.y == 2)) for p in d.patients[d.strata == 2]}
    return out


def h6(logit_afdb, d, L):
    """AUROC pemisahan window AFL 08378 (AFL 2:1) dari sinus takikardia asli ≥ 120 bpm. ECGFounder: logit AFL − logit ST.
    Pembanding v3.1: log pAFL − log pN (AFDB: OOF rata-rata ulangan; latihan: paket tanpa bias paket)."""
    i = E13.indeks_label()

    def s(lo):
        return lo[:, i["AFL"]].astype(np.float64) - lo[:, i["ST"]]

    pos, neg = (d.records == PASIEN_21) & (d.y == 2), L["hr"] >= 120
    v_pos = np.mean([np.log(np.clip(o[pos, 2], 1e-9, 1)) - np.log(np.clip(o[pos, 0], 1e-9, 1)) for o in d.oof_v31], 0)
    praw = E13.tanpa_bias_paket(L["p31"])[neg]
    v_neg = np.log(np.clip(praw[:, 2], 1e-9, 1)) - np.log(np.clip(praw[:, 0], 1e-9, 1))
    ep, en = s(logit_afdb)[pos], s(L["logit"])[neg]
    return dict(ecgfounder=K15._auroc(ep, en), v31=K15._auroc(v_pos, v_neg), n_pos=int(pos.sum()), n_neg=int(neg.sum()),
                median_ecgfounder=[float(np.median(ep)), float(np.median(en))])


# ── data eksternal (semua dari cache) ──────────────────────────────────
def latihan():
    """Window latihan fisik v3.3 yang punya logit ECGFounder (mode perangkat, lolos cek), dengan urutan
    ecgfounder.jendela_latihan: sumber urut abjad, di dalam sumber urutan baris jendela.npz. Urutan dicek di notebook dengan
    mengulang angka zero-shot v13 persis."""
    J = K33.muat_jendela()
    kolom_ok = np.cumsum(J["ok"]) - 1                               # baris jendela → kolom prediksi v3.3 (window ok)
    baris = np.concatenate([np.flatnonzero((J["sumber"] == s) & (J["mode"] == "perangkat") & J["ok"])
                            for s in np.unique(J["sumber"])])
    lo = np.load(E13.CACHE / "latihan_ok_50hz.npz")["logit"]
    if len(lo) != len(baris):
        raise RuntimeError(f"cache logit latihan ({len(lo)}) tidak sejajar dengan window ({len(baris)})")
    kolom = kolom_ok[baris]
    mask = np.zeros(int(J["ok"].sum()), bool)
    mask[kolom] = True
    return dict(J=J, logit=lo, p31=np.load(K33.CACHE / "prediksi.npz")["p_v31"][kolom], kolom=kolom, mask=mask,
                hr=J["hr"][baris], cv=J["cv"][baris], subjek=J["subjek"][baris], sumber=J["sumber"][baris])


def kelas_latihan(L, pred):
    """Prediksi window latihan → array (1, N_ok) sejajar prediksi v3.3 (dipakai takikardia_asli.alarm_per_pita/banding_alarm)."""
    out = np.zeros((1, L["mask"].size), np.int64)
    out[0, L["kolom"]] = pred
    return out


def ptbxl():
    """PTB-XL dari cache v13 (logit) + v15 (fitur atrium, paket v3.1); hanya window dengan RR terbaca (sama dengan v13)."""
    z = dict(np.load(K15.CACHE / "ptbxl.npz"))
    lo = np.load(E13.CACHE / "ptbxl_50hz.npz")["logit"]
    ok = np.isfinite(z["p31"]).all(1)
    return dict(logit=lo[ok], p31=z["p31"][ok], fitur=z["fitur"][ok], y=z["y"][ok], st=z["st"][ok], pasien=z["pasien"][ok])


def icentia():
    """Icentia11k (window v10, CC BY-NC-SA): logit v13 + paket v3.1 dari RR (cache cache/v16/icentia_p31.npy)."""
    zi = np.load(REPO / "cache/v10/icentia_windows.npz")
    p = CACHE / "icentia_p31.npy"
    if not p.exists():
        from predictor import ECGPredictor
        P = ECGPredictor(str(REPO / "v3.1_kalibrasi_produksi/model_export"), name="sihedaf_v3_1", device="cpu")
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(p, P.predict_proba_sequences(zi["X"], zi["M"]))
    return dict(logit=np.load(E13.CACHE / "icentia_50hz.npz")["logit"], p31=np.load(p), y=zi["y"], pasien=zi["patient"])


def keputusan_eksternal(logit, p31_paket, b2, bias_hx, d2_atrium=None, b2_atrium=None):
    """Keputusan H-0, H-X (dan H-A bila d2 atrium diberikan) di data eksternal; pembanding = paket v3.1 (argmax).
    Tahap 2 RR memakai probabilitas paket tanpa bias paket. Return (dict prediksi, d1)."""
    pE, d1 = tahap1(logit)
    praw = E13.tanpa_bias_paket(p31_paket)
    out = {"v3.1 paket": np.asarray(p31_paket).argmax(1), "H-0": putuskan(d1, d2_rr(praw), b2),
           "H-X": cv.apply_bias(skor_hx(pE, praw), bias_hx).argmax(1)}
    if d2_atrium is not None:
        out["H-A"] = putuskan(d1, d2_atrium, b2_atrium)
    return out, d1
