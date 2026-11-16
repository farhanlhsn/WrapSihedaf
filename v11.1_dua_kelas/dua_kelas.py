"""
dua_kelas.py — v11.1: evaluasi 2 kelas (Normal vs AF/AFL, "AF₁" di literatur) untuk model yang sudah ada, tanpa training baru.

    python dua_kelas.py          # → cache/v11.1/dua_kelas.json (±25 menit CPU)

Protokol sama dengan 3 kelas (AGENTS.md), dengan satu perubahan: probabilitas AF₁ = p(AFIB) + p(AFL), dan bias kelas hanya satu
(untuk AF₁), dipilih cross-fit antar fold dari grid yang sama (−7…+3, langkah 0,25) dengan Macro-F1 2 kelas sebagai sasaran.
- Metrik pooled OOF rata-rata 5 ulangan, CI 95% bootstrap pasien 2.000× (sampel sama dengan semua evaluasi v3/v11).
- Berpasangan vs v3.1 (OOF MicroTransformer v3 + bias 2 kelas) dan vs v1 (argmax 3 kelas, digabung).
- Dua aturan keputusan dinilai setara: `ulang` = bias 2 kelas dipilih ulang (rencana awal); `gabung` = bias 3 kelas seperti
  evaluasi biasa, lalu AFIB + AFL digabung (ditambahkan setelah uji stres memperlihatkan alarm baru pada ritme teratur 85–90 bpm
  dengan aturan `ulang`). Alarm pada window Normal asli yang teratur di 80–95 bpm (CV IBI < 0,03) juga dilaporkan.
- Butir 9: % window Normal asli yang ditebak AF₁ per pita detak; ambang uji stres sintetis v3.2 dengan bias 2 kelas per fold
  (model v11 menyimpan prediksi uji stres; v3.1 diwakili kontrol v7 per fold, ulangan 0).
- Butir 10: bootstrap bersarang (bias dipilih ulang pada pasien latih berbobot di setiap sampel, pola v3.2) dan tanpa pasien 04936
  (bias dihitung ulang di 22 pasien).
Model dibaca dari cache/v11/<model>/<setelan>/ lewat bersama.runner (fold diperiksa sama dengan fold resmi).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # v11.1_dua_kelas
REPO = HERE.parent
sys.path.insert(0, str(REPO / "v11_model_tim"))
import numpy as np  # noqa: E402
from bersama import data as D, evaluasi as EV, runner  # noqa: E402

cv, T = D.cv, D.T
GRID = cv.BIAS_GRID
OUT = REPO / "cache/v11.1/dua_kelas.json"
PASIEN_TANPA = "04936"
PITA = (("< 100", 0, 100), ("100–110", 100, 110), ("110–120", 110, 120), ("≥ 120", 120, 999))
# sistem yang dievaluasi: nama tampilan → folder cache/v11 (v1 & v3.1 dari data bersama)
MODEL = {"BiGRU bawaan": "bigru/bawaan", "BiGRU wp100": "bigru/wp100", "TCN bawaan": "tcn/bawaan", "TCN wp100": "tcn/wp100",
         "XGBoost bawaan": "xgboost/bawaan", "XGBoost bobot penuh": "xgboost/bobot_penuh"}
METRIK = ("Macro-F1", "F1 AF1", "Se", "Sp", "PPV", "F1 Normal")


# ── dasar 2 kelas ──────────────────────────────────────────────────────
def ke_dua(p3):
    """(…, 3) probabilitas Normal/AFIB/AFL → (…, 2) Normal/AF₁."""
    return np.stack([p3[..., 0], p3[..., 1] + p3[..., 2]], -1)


def _logp2(p3):
    return np.log(np.clip(ke_dua(p3), 1e-9, 1))


def f1_makro_cm(cm):
    """cm (…, 2, 2) → Macro-F1 2 kelas (…)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    den = 2 * tp + (cm.sum(-2) - tp) + (cm.sum(-1) - tp)
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0).mean(-1)


def metrik(cm):
    """cm (…, 2, 2) baris = aktual → dict metrik (…)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    sup, prd = cm.sum(-1), cm.sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(prd > 0, tp / prd, 0.0)
        rec = np.where(sup > 0, tp / sup, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return {"Macro-F1": f1.mean(-1), "F1 AF1": f1[..., 1], "Se": rec[..., 1], "Sp": rec[..., 0], "PPV": prec[..., 1],
            "F1 Normal": f1[..., 0]}


def cm_pasien(y2, pred, rec, pasien):
    """(R, P, 2, 2) confusion matrix per ulangan & pasien."""
    pidx = np.searchsorted(pasien, rec)
    return np.stack([np.bincount((pidx * 2 + y2) * 2 + p, minlength=len(pasien) * 4).reshape(-1, 2, 2).astype(float)
                     for p in pred])


def keputusan2(oof3, folds, y2):
    """Bias AF₁ cross-fit per ulangan: dipilih dari fold lain (Macro-F1 2 kelas), dipasang ke fold uji. → pred (R, N), bias (R, 5)."""
    pred, bias = [], []
    for o, f in zip(oof3, folds):
        lp = _logp2(o)
        p = np.empty(len(y2), np.int64)
        bb = []
        for k in np.unique(f):
            tr, te = f != k, f == k
            cm = np.stack([np.bincount(y2[tr] * 2 + (lp[tr] + [0, b]).argmax(1), minlength=4).reshape(2, 2) for b in GRID])
            b = float(GRID[int(np.argmax(f1_makro_cm(cm.astype(float))))])
            p[te] = (lp[te] + [0, b]).argmax(1)
            bb.append(b)
        pred.append(p)
        bias.append(bb)
    return np.stack(pred), np.asarray(bias)


def ringkas(cm, W):
    """cm (R, P, 2, 2), W (B, P) → (metrik titik + CI 95%, sampel bootstrap per metrik)."""
    titik = metrik(cm.sum(1))
    boot = {k: v.mean(1) for k, v in metrik(np.einsum("bp,rpij->brij", W, cm)).items()}
    return {k: [float(titik[k].mean()), *map(float, np.percentile(boot[k], [2.5, 97.5]))] for k in METRIK}, boot


def banding(mk, mr, bk, br):
    """[Δ titik, CI bawah, CI atas, P(Δ > 0)], sampel bootstrap yang sama (pola evaluasi.banding)."""
    return {k: [float(mk[k][0] - mr[k][0]), *map(float, np.percentile(bk[k] - br[k], [2.5, 97.5])), float(np.mean(bk[k] - br[k] > 0))]
            for k in ("Macro-F1", "F1 AF1", "Se", "Sp")}


# ── butir 10: bootstrap bersarang ──────────────────────────────────────
def kandidat_cm(oof3, y2, rec, pasien):
    """(R, G, P, 2, 2): confusion matrix per pasien untuk setiap nilai bias di grid."""
    out = []
    for o in oof3:
        lp = _logp2(o)
        out.append(np.stack([cm_pasien(y2, [(lp + [0, b]).argmax(1)], rec, pasien)[0] for b in GRID]))
    return np.stack(out)


def bersarang(cand, pfold, W):
    """Macro-F1 bersarang (B,): di tiap sampel, ulangan, fold, bias dipilih pada pasien latih berbobot lalu dipasang ke pasien uji."""
    R, G, P = cand.shape[:3]
    hasil = np.zeros((W.shape[0], R))
    for r in range(R):
        flat = cand[r].reshape(G, P, 4)
        acc = np.zeros((W.shape[0], 4))
        for k in range(5):
            tr, te = pfold[r] != k, pfold[r] == k
            cm_tr = (W[:, tr] @ flat[:, tr].transpose(1, 0, 2).reshape(tr.sum(), G * 4)).reshape(-1, G, 2, 2)
            g = f1_makro_cm(cm_tr).argmax(1)                                      # (B,)
            acc += np.einsum("bpc,bp->bc", flat[g][:, te], W[:, te])
        hasil[:, r] = f1_makro_cm(acc.reshape(-1, 2, 2))
    return hasil.mean(1)


# ── butir 9: uji stres sintetis dengan bias 2 kelas ────────────────────
def ambang_stres(stres, bias2, uji):
    """stres (R, 5, S, 3) + bias AF₁ per fold (R, 5) → proporsi ditebak Normal per (jenis, bpm) dan ambang bpm."""
    normal = np.stack([((_logp2(stres[i, k]) + [0, bias2[i, k]]).argmax(1) == 0)
                       for i in range(stres.shape[0]) for k in range(stres.shape[1])])     # (model, S)
    proporsi, ambang = {}, {}
    for j in T.JITTERS:
        tab = np.array([[normal[:, (uji.jenis == j) & (uji.bpm == b)].mean()] for b in T.BPMS])
        proporsi[j] = tab[:, 0].tolist()
        ambang[j] = T.normal_threshold(tab)
    return {"bpm": list(T.BPMS), "proporsi_normal": proporsi, "ambang": ambang}


# ── aturan "gabung": bias 3 kelas, lalu AFIB + AFL digabung ───────────
def gabung_cm(cm3):
    """(…, 3, 3) → (…, 2, 2): baris/kolom AFIB dan AFL dijumlahkan."""
    a = np.concatenate([cm3[..., :1, :], cm3[..., 1:, :].sum(-2, keepdims=True)], -2)
    return np.concatenate([a[..., :1], a[..., 1:].sum(-1, keepdims=True)], -1)


def f1_makro3_cm(cm):
    """Macro-F1 3 kelas dari cm (…, 3, 3), hanya kelas yang ada (sama dengan sihedaf_cv._macro_f1_fast)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    sup, prd = cm.sum(-1), cm.sum(-2)
    den = 2 * tp + (prd - tp) + (sup - tp)
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0)
    ada = sup > 0
    return (f1 * ada).sum(-1) / np.maximum(ada.sum(-1), 1)


def kandidat_cm3(oof3, y, rec, pasien):
    """(R, G², P, 3, 3): confusion matrix 3 kelas per pasien untuk setiap pasangan bias (AFIB, AFL) di grid."""
    out = []
    for o in oof3:
        lp = np.log(np.clip(o, 1e-9, 1))
        out.append(np.stack([cv.patient_cms(y, (lp + [0.0, b1, b2]).argmax(1), rec, pasien) for b1 in GRID for b2 in GRID]))
    return np.stack(out)


def bersarang_gabung(cand3, pfold, W):
    """Seperti bersarang(), tetapi bias dipilih dengan Macro-F1 3 kelas (aturan cross-fit 3 kelas) lalu dinilai 2 kelas."""
    R, G, P = cand3.shape[:3]
    hasil = np.zeros((W.shape[0], R))
    for r in range(R):
        flat = cand3[r].reshape(G, P, 9)
        acc = np.zeros((W.shape[0], 9))
        for k in range(5):
            tr, te = pfold[r] != k, pfold[r] == k
            cm_tr = (W[:, tr] @ flat[:, tr].transpose(1, 0, 2).reshape(tr.sum(), G * 9)).reshape(-1, G, 3, 3)
            g = f1_makro3_cm(cm_tr).argmax(1)
            acc += np.einsum("bpc,bp->bc", flat[g][:, te], W[:, te])
        hasil[:, r] = f1_makro_cm(gabung_cm(acc.reshape(-1, 3, 3)))
    return hasil.mean(1)


def ambang_stres_gabung(stres, bias3, uji):
    """Uji stres dengan bias 3 kelas per fold: proporsi ditebak Normal = sama dengan keluaran 2 kelas hasil penggabungan."""
    a = EV.ambang_takikardia(stres, bias3, uji)
    return {"bpm": a["bpm"], "proporsi_normal": {j: [t[0] for t in tab] for j, tab in a["proporsi"].items()}, "ambang": a["ambang"]}


# ── evaluasi utama ─────────────────────────────────────────────────────
ATURAN = {"ulang": "bias 2 kelas dipilih ulang (rencana awal)", "gabung": "bias 3 kelas, lalu AFIB + AFL digabung (tambahan)"}


def _nested_out(nest, f_v1):
    out = {}
    for nama, v in nest.items():
        o = {"titik": float(v[0]), "ci": np.percentile(v[1:], [2.5, 97.5]).tolist()}
        for ref, rv in (("v1", f_v1), ("v31", nest["v3.1"])):
            if nama == "v3.1" and ref == "v31":
                continue
            dd = v[1:] - rv[1:]
            o[f"vs_{ref}"] = {"delta": float(v[0] - rv[0]), "ci": np.percentile(dd, [2.5, 97.5]).tolist(), "p": float(np.mean(dd > 0))}
        out[nama] = o
    return out


def _bandingkan(hasil, boot):
    for nama in hasil:
        if nama != "v3.1":
            hasil[nama]["vs_v31"] = banding(hasil[nama], hasil["v3.1"], boot[nama], boot["v3.1"])
        if nama != "v1 (argmax)" and "v1 (argmax)" in hasil:
            hasil[nama]["vs_v1"] = banding(hasil[nama], hasil["v1 (argmax)"], boot[nama], boot["v1 (argmax)"])


def evaluasi(log=print):
    d = D.muat()
    R = list(range(D.N_ULANGAN))
    folds, y2 = d.folds[R], (d.y > 0).astype(np.int64)
    W = cv.bootstrap_weights(d.strata, EV.N_BOOT, seed=D.SEED)
    W1 = np.vstack([np.ones(len(d.patients)), W])                                   # baris 0 = titik (bersarang)
    pfold = np.array([[int(folds[r][d.records == p][0]) for p in d.patients] for r in R])
    uji = D.set_uji_stres()
    ibi, msk = d.X[..., 0], d.M.astype(bool)
    cv_ibi = np.array([ibi[i][msk[i]].std() / ibi[i][msk[i]].mean() for i in range(len(ibi))])
    teratur = (d.y == 0) & (d.hr >= 80) & (d.hr < 95) & (cv_ibi < 0.03)            # Normal teratur 80–95 bpm (uji stres asli)
    H = D.lokasi_hasil()
    sumber, stres = {"v3.1": d.oof_v31[R]}, {}
    for nama, p in MODEL.items():
        h = runner.muat_hasil(H / p, d)
        assert h.ulangan == R, (p, h.ulangan)
        sumber[nama], stres[nama] = h.oof, h.stres
    p_v1 = (d.oof_v1[R].argmax(-1) > 0).astype(np.int64)
    cm_v1 = cm_pasien(y2, p_v1, d.records, d.patients)
    f_v1 = f1_makro_cm(np.einsum("bp,rpij->brij", W1, cm_v1)).mean(1)
    m = d.records != PASIEN_TANPA
    pasien_m = np.unique(d.records[m])
    W_m = cv.bootstrap_weights(cv.patient_strata(d.y[m], d.records[m], pasien_m), EV.N_BOOT, seed=D.SEED)
    alarm_pita = lambda pred: {k: float(np.mean([(p[(d.y == 0) & (d.hr >= lo) & (d.hr < hi)] == 1).mean() for p in pred]))  # noqa: E731
                               for k, lo, hi in PITA}
    alarm_sel = lambda pred, sel: float(np.mean([(p[sel] == 1).mean() for p in pred]))  # noqa: E731
    lain = (d.y == 0) & (d.hr >= 80) & (d.hr < 95) & ~teratur
    keluaran = {}
    for aturan in ATURAN:
        log(f"aturan {aturan}: {ATURAN[aturan]}")
        hasil, boot, pred_all = {}, {}, {}
        hasil["v1 (argmax)"], boot["v1 (argmax)"] = ringkas(cm_v1, W)
        hasil["v1 (argmax)"].update(alarm_normal=alarm_pita(p_v1), teratur_80_95=alarm_sel(p_v1, teratur), lain_80_95=alarm_sel(p_v1, lain))
        pred_all["v1 (argmax)"] = p_v1
        nest = {}
        tanpa, boot_m = {}, {}
        tanpa["v1 (argmax)"], boot_m["v1 (argmax)"] = ringkas(cm_pasien(y2[m], p_v1[:, m], d.records[m], pasien_m), W_m)
        for nama, oof in sumber.items():
            log(f"  {nama}")
            if aturan == "ulang":
                pred, bias = keputusan2(oof, folds, y2)
                info = dict(median=float(np.median(bias)), min=float(bias.min()), maks=float(bias.max()),
                            di_batas=float(np.mean((bias <= GRID[0]) | (bias >= GRID[-1]))))
                st = ambang_stres(stres[nama], bias, uji) if nama in stres else None
                nest[nama] = bersarang(kandidat_cm(oof, y2, d.records, d.patients), pfold, W1)
                pm, _ = keputusan2(oof[:, m], folds[:, m], y2[m])
            else:
                p3, b3 = EV.keputusan(oof, folds, d.y)
                pred = (p3 > 0).astype(np.int64)
                ba, bl = b3[..., 1], b3[..., 2]
                info = dict(median=[float(np.median(ba)), float(np.median(bl))], min=[float(ba.min()), float(bl.min())],
                            maks=[float(ba.max()), float(bl.max())],
                            di_batas=float(np.mean((b3[..., 1:] <= GRID[0]) | (b3[..., 1:] >= GRID[-1]))))
                st = ambang_stres_gabung(stres[nama], b3, uji) if nama in stres else None
                nest[nama] = bersarang_gabung(kandidat_cm3(oof, d.y, d.records, d.patients), pfold, W1)
                pm3, _ = EV.keputusan(oof[:, m], folds[:, m], d.y[m])
                pm = (pm3 > 0).astype(np.int64)
            mt, b = ringkas(cm_pasien(y2, pred, d.records, d.patients), W)
            hasil[nama] = dict(mt, bias=info, alarm_normal=alarm_pita(pred), teratur_80_95=alarm_sel(pred, teratur),
                               lain_80_95=alarm_sel(pred, lain))
            if st:
                hasil[nama]["stres"] = st
            boot[nama], pred_all[nama] = b, pred
            assert abs(nest[nama][0] - mt["Macro-F1"][0]) < 1e-12, (aturan, nama, nest[nama][0], mt["Macro-F1"][0])
            tanpa[nama], boot_m[nama] = ringkas(cm_pasien(y2[m], pm, d.records[m], pasien_m), W_m)
        # v3.1 diwakili kontrol v7 per fold (resep v3.1, ulangan 0) untuk uji stres sintetis
        if aturan == "ulang":
            _, bk = keputusan2(d.kontrol_v7.oof[None], d.folds[:1], y2)
            hasil["v3.1"]["stres_kontrol_v7"] = ambang_stres(d.kontrol_v7.stres[None], bk, uji)
        else:
            _, bk3 = EV.keputusan(d.kontrol_v7.oof[None], d.folds[:1], d.y)
            hasil["v3.1"]["stres_kontrol_v7"] = ambang_stres_gabung(d.kontrol_v7.stres[None], bk3, uji)
        _bandingkan(hasil, boot)
        _bandingkan(tanpa, boot_m)
        per_pasien = {}
        for nama, pred in pred_all.items():
            c = cm_pasien(y2, pred, d.records, d.patients).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                per_pasien[nama] = {"se": (c[:, 1, 1] / c[:, 1].sum(1)).tolist(), "alarm_normal": (c[:, 0, 1] / c[:, 0].sum(1)).tolist()}
        keluaran[aturan] = {"sistem": hasil, "bersarang": _nested_out(nest, f_v1), "tanpa_04936": tanpa, "per_pasien": per_pasien}
    # perbandingan antar aturan (sistem yang sama, sampel bootstrap sama)
    out = {"aturan": keluaran, "keterangan_aturan": ATURAN, "pasien": d.patients.tolist(), "strata": d.strata.tolist(),
           "n": {"window": int(len(y2)), "af1": int(y2.sum()), "normal": int((y2 == 0).sum()),
                 "normal_per_pita": {k: int(((d.y == 0) & (d.hr >= lo) & (d.hr < hi)).sum()) for k, lo, hi in PITA},
                 "teratur_80_95": int(teratur.sum()), "lain_80_95": int(lain.sum()),
                 "pasien": int(len(d.patients)), "pasien_tanpa": int(len(pasien_m))},
           "grid": [float(GRID[0]), float(GRID[-1]), float(GRID[1] - GRID[0])], "n_boot": EV.N_BOOT, "seed": D.SEED,
           "lingkungan": EV._lingkungan()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    log(f"selesai → {OUT}")
    return out


if __name__ == "__main__":
    evaluasi(log=lambda m: print(m, flush=True))
