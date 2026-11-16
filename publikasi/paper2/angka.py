"""Angka dan gambar paper 2 (uji stres sinus takikardia), semuanya dihitung dari cache, tidak disalin manual.

Sumber:
  cache/v3_2/ringkasan.json, prediksi.npz     uji sintetis paket P0/P1/P2, window Normal AFDB per pita detak, Macro-F1 AFDB
  cache/v3_3/ringkasan.json, jendela.npz,     EKG latihan fisik (EPFL, Wrist, EPHNOGRAM): alarm per model, mode perangkat
             prediksi.npz                     dan R-peak manual, aturan produk, abstain conformal
  cache/v7/ringkasan.json                     + sinus takikardia sintetis (7a, 7b)
  cache/v8/ringkasan.json                     + Chapman (8a, 8c); sinus takikardia klinis Chapman per pita detak
  cache/v11/<model>/<setelan>/ringkasan.json  model tim (uji sintetis, AFDB, recall AFL per pasien)
  cache/dataset_v2_detector.npz               label dan rekaman window AFDB (recall 08378 untuk P0/P1/P2)
  publikasi/paper2/penilaian_gelombang_p.json penilaian visual strip dari cek_gelombang_p.py
  cache/v16.1, v16.5, v13 ringkasan.json      mitigasi eksploratif (Tabel II): gerbang ECGFounder D-0, aturan episode 30 s;
                                              metrik episode dihitung ulang dengan v16.5 episode.py (paritas di-assert)

Jalankan dari root repo:  python publikasi/paper2/angka.py
Keluaran: publikasi/paper2/angka.json, angka.tex (makro LaTeX + baris Tabel I), gambar/*.pdf (+ .png pratinjau).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "gambar"
sys.path.insert(0, str(ROOT / "v3.3_takikardia_asli"))
import takikardia_asli as TA  # noqa: E402

BLUE, ORANGE, GREEN, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#172B43", "#526477", "#d9dde3"
AFL_PATIENTS = ["04043", "04908", "04936", "06426", "07910", "08215", "08378"]
JITTER = ["tanpa variasi", "seragam ±10 ms", "Gauss sd 10 ms", "Gauss sd 20 ms"]
JITTER_FIG = "Gauss sd 10 ms"
BANDS = ["100–110", "110–120", "120–130", "130–140", "140–150", "150–∞"]

# (kunci, grup, label naskah, sumber uji sintetis, sumber AFDB, kunci v3.3)
MODELS = [
    ("p0", "afdb", "P0 MicroTransformer, original recipe", ("v32", "v1"), ("v32", "v1"), "v1"),
    ("p1", "afdb", "P1 MicroTransformer, fixed biases", ("v32", "v2_prod"), ("v32", "v2_prod"), "v2_prod"),
    ("p2", "afdb", "P2 MicroTransformer, cross-fit biases (ref.)", ("v32", "v31"), ("v32", "v31"), "v31"),
    ("bigru_s", "afdb", "BiGRU, soft weights", ("v11", "bigru/bawaan"), ("v11", "bigru/bawaan"), "v11_bigru"),
    ("bigru_f", "afdb", "BiGRU, full weights", ("v11", "bigru/wp100"), ("v11", "bigru/wp100"), "v11_bigru_wp100"),
    ("tcn_s", "afdb", "TCN, soft weights", ("v11", "tcn/bawaan"), ("v11", "tcn/bawaan"), None),
    ("tcn_f", "afdb", "TCN, full weights", ("v11", "tcn/wp100"), ("v11", "tcn/wp100"), "v11_tcn_wp100"),
    ("it_s", "afdb", "InceptionTime, soft weights", ("v11", "inceptiontime/bawaan"), ("v11", "inceptiontime/bawaan"), None),
    ("it_f", "afdb", "InceptionTime, full weights", ("v11", "inceptiontime/wp100"), ("v11", "inceptiontime/wp100"), None),
    ("mr", "afdb", "MiniRocket + logistic regression", ("v11", "minirocket/bawaan"), ("v11", "minirocket/bawaan"), None),
    ("xgb", "afdb", "XGBoost (HRV features)", ("v11", "xgboost/bawaan"), ("v11", "xgboost/bawaan"), None),
    ("svm", "afdb", "RBF SVM (HRV features)", ("v11", "svm/bawaan"), ("v11", "svm/bawaan"), None),
    ("v7a", "st", "+ synthetic ST, 100--130 bpm", ("v7", "takikardia130"), ("v7", "takikardia130"), "v7_130"),
    ("v7b", "st", "+ synthetic ST, 100--160 bpm", ("v7", "takikardia160"), ("v7", "takikardia160"), "v7_160"),
    ("v8a", "st", "+ Chapman (incl.\\ clinical ST)", ("v8", "afdb_chapman"), ("v8", "afdb_chapman"), "v8_8a"),
    ("v8c", "st", "+ Chapman, AFL labels removed", ("v8", "afdb_chapman_bersih"), ("v8", "afdb_chapman_bersih"), "v8_8c"),
]
CHAPMAN = {"p0": "v1_paket", "p1": "v2_paket", "p2": "v31_paket", "v8a": "afdb_chapman", "v8c": "afdb_chapman_bersih"}


def load(p):
    return json.loads((ROOT / p).read_text())


def ambang(frac_normal, bpms):
    """Detak terendah di mana < 50% window ditebak Normal (= takikardia.normal_threshold v3.2)."""
    idx = np.flatnonzero(np.asarray(frac_normal) < 0.5)
    return int(bpms[idx[0]]) if idx.size else None


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kumpulkan():
    v32, v33 = load("cache/v3_2/ringkasan.json"), load("cache/v3_3/ringkasan.json")
    v7, v8 = load("cache/v7/ringkasan.json"), load("cache/v8/ringkasan.json")
    v11 = {s: load(f"cache/v11/{s}/ringkasan.json") for _, _, _, (src, s), _, _ in MODELS if src == "v11"}
    bpms = np.array(v32["bpms"])
    assert v7["bpms"] == v32["bpms"] == v8["bpms"] and all(r["takikardia"]["model"]["bpm"] == v32["bpms"] for r in v11.values())

    # ── uji sintetis ────────────────────────────────────────────────────
    def kurva(src, k):
        tab = {"v32": v32["stress"], "v7": v7["stress"], "v8": v8["stress"]}.get(src)
        prop = v11[k]["takikardia"]["model"]["proporsi"] if src == "v11" else tab[k]
        return {j: np.asarray(prop[j]) for j in JITTER}

    sint = {}
    for key, _, _, (src, k), _, _ in MODELS:
        kv = kurva(src, k)
        thr = {j: ambang(kv[j][:, 0], bpms) for j in JITTER}
        cached = {"v32": v32["threshold"], "v7": v7["threshold"], "v8": v8["threshold"]}.get(src, {}).get(k)
        if cached:
            assert thr == cached, (key, thr, cached)
        sint[key] = dict(thr=thr, normal={j: kv[j][:, 0].tolist() for j in JITTER},
                         afl_reg=float(kv["tanpa variasi"][bpms >= 130, 2].mean()),
                         af_g20=float(kv["Gauss sd 20 ms"][bpms >= 130, 1].mean()))
        # kurva tidak monoton: < 50% Normal di bawah 100 bpm lalu pulih
        n_no = kv["tanpa variasi"][:, 0]
        sint[key]["tidak_monoton"] = bool((n_no[bpms < 100] < 0.5).any() and (n_no[(bpms >= 100) & (bpms < 110)] >= 0.5).any())
    for r in v11.values():          # referensi paket v3.1 di setiap ringkasan v11 = ambang P2
        assert r["takikardia"]["paket_v31"] == sint["p2"]["thr"]["tanpa variasi"]

    # ── window Normal AFDB ≥ 120 bpm (OOF) ──────────────────────────────
    afdb120, afdb_n = {}, None
    for key, _, _, _, (src, k), _ in MODELS:
        if src == "v32":
            row = v32["by_hr"]["Normal | 120–∞ bpm"]; p = row[k]
        elif src == "v11":
            row = v11[k]["pita_detak"]["Normal | 120–∞ bpm"]; p = row["model"]
        elif src == "v7":
            row = v7["by_hr"]["Normal | 120–∞ bpm"]; p = row[k]
        else:
            row = v8["afdb"]["by_hr"]["Normal | 120–∞ bpm"]; p = row[k]
        assert afdb_n in (None, (row["n"], row["n_pasien"]))
        afdb_n = (row["n"], row["n_pasien"])
        afdb120[key] = 1.0 - p[0]
    n_normal_afdb = sum(v32["by_hr"][f"Normal | {b}"]["n"] for b in ("0–100 bpm", "100–110 bpm", "110–120 bpm", "120–∞ bpm"))

    # ── AFDB: Macro-F1 dan recall AFL 08378 ─────────────────────────────
    d = np.load(ROOT / "cache/dataset_v2_detector.npz")
    pr = np.load(ROOT / "cache/v3_2/prediksi.npz")
    m08 = (d["y"] == 2) & (d["records"] == "08378")
    rec08 = {k: float((pr["semua" + k][:, m08] == 2).mean()) for k in ("v1", "v2_prod", "v31")}
    v3t = load("v3_evaluasi_robust/laporan/_build/tables.json")
    assert abs(rec08["v31"] - v3t["afl_recall"]["MicroTransformer"]["mean"][AFL_PATIENTS.index("08378")]) < 1e-9
    afdb = {}
    for key, _, _, _, (src, k), _ in MODELS:
        if src == "v32":
            afdb[key] = dict(mf=v32["sens_all"]["rows"][k]["Macro-F1"], r08=rec08[k])
        elif src == "v11":
            afdb[key] = dict(mf=v11[k]["metrik"]["bias"]["Macro-F1"], r08=v11[k]["afl_per_pasien"]["bias"]["08378"],
                             d=v11[k]["banding_v31"]["Macro-F1"])
        elif src == "v7":
            afdb[key] = dict(mf=v7["rows"][k]["Macro-F1"], r08=v7["afl_recall"][k][6], d=v7["pairs"][k]["Macro-F1"])
        else:
            afdb[key] = dict(mf=v8["afdb"]["rows"][k]["Macro-F1"], r08=v8["afdb"]["afl_recall"][k][6],
                             d=v8["afdb"]["pairs"][k]["Macro-F1"])
    assert abs(afdb["p2"]["mf"][0] - v32["nested"]["v31_point"]) < 1e-9

    # ── sinus takikardia klinis (Chapman, label dokter) ─────────────────
    st = v8["chapman"]["st_bands"]
    assert list(st) == BANDS
    chap = {}
    for key, k in CHAPMAN.items():
        n = np.array([st[b]["n"] for b in BANDS[2:]])
        a = np.array([st[b][k][0] for b in BANDS[2:]])
        tot = float((a * n).sum() / n.sum())
        lo, hi = wilson(tot * n.sum(), n.sum())
        chap[key] = dict(alarm120=tot, ci120=[lo, hi], n120=int(n.sum()),
                         band={b: st[b][k] for b in BANDS}, all=v8["chapman"]["st_all"][k])
    chap_n = {b: st[b]["n"] for b in BANDS}

    # ── EKG latihan fisik: dihitung ulang dari prediksi per window ──────
    J = TA.muat_jendela()
    P = np.load(ROOT / "cache/v3_3/prediksi.npz")
    ok = J["ok"]
    hr, subj, mode, src_ = J["hr"][ok], J["subjek"][ok], J["mode"][ok], J["sumber"][ok]
    dev, man = mode == "perangkat", mode == "manual"
    latihan = {}
    for key, _, _, _, _, k in MODELS:
        if k is None:
            continue
        alarm = (P[k] != 0).mean(0)
        out = {}
        for nama, msk in (("dev", dev), ("man", man)):
            m = msk & (hr >= 120)
            t, ci, _ = TA._boot_subjek(subj[m], alarm[m])
            out[nama] = dict(alarm=t, ci=ci.tolist(), n=int(m.sum()), subjek=int(np.unique(subj[m]).size))
        u = v33["utama"][k]
        assert abs(out["dev"]["alarm"] - u["alarm"]) < 1e-12 and out["dev"]["n"] == u["n"] and out["dev"]["subjek"] == u["subjek"]
        out["banding"] = u["banding"]
        latihan[key] = out
    # per pita detak untuk P2 (gabungan ketiga sumber)
    alarm_p2 = (P["v31"] != 0).mean(0)
    # Bandingkan segmen EPFL/Wrist yang sama, lolos dan ≥120 bpm di kedua mode.
    kunci = np.rec.fromarrays([src_, subj, J["rekaman"][ok], J["start"][ok]],
                              names=("sumber", "subjek", "rekaman", "awal"))
    calon = (hr >= 120) & np.isin(src_, ("EPFL", "Wrist"))
    idx_dev = np.flatnonzero(calon & dev)
    idx_man = np.flatnonzero(calon & man)
    kunci_dev, kunci_man = kunci[idx_dev], kunci[idx_man]
    assert np.unique(kunci_dev).size == idx_dev.size and np.unique(kunci_man).size == idx_man.size
    pasangan, i_dev, i_man = np.intersect1d(kunci_dev, kunci_man, assume_unique=True, return_indices=True)
    assert pasangan.size and P["v31"].shape[0] == 1
    p2_berpasangan = dict(n=int(pasangan.size), subjek=int(np.unique(pasangan["subjek"]).size),
                         alarm_dev=int(alarm_p2[idx_dev[i_dev]].sum()),
                         alarm_man=int(alarm_p2[idx_man[i_man]].sum()))
    pita = {}
    for nama, msk in (("dev", dev), ("man", man)):
        pita[nama] = {}
        for lo, hi in TA.PITA:
            m = msk & (hr >= lo) & (hr < hi)
            if m.sum() < 5:
                continue
            t, ci, _ = TA._boot_subjek(subj[m], alarm_p2[m])
            pita[nama][f"{lo}–{hi if hi < 999 else '∞'}"] = dict(alarm=t, ci=ci.tolist(), n=int(m.sum()),
                                                               subjek=int(np.unique(subj[m]).size))
    per_sumber = {s: float(alarm_p2[dev & (hr >= 120) & (src_ == s)].mean()) for s in ("EPFL", "Wrist", "EPHNOGRAM")}
    data = v33["data"]
    ap = v33["alarm_pita"]
    detektor = dict(
        lolos_epfl_dev=data["EPFL / perangkat"]["% lolos"], lolos_epfl_man=data["EPFL / manual"]["% lolos"],
        cv_epfl_dev=data["EPFL / perangkat"]["CV median"], cv_epfl_man=data["EPFL / manual"]["CV median"],
        lt100_wrist_dev=ap["Wrist/perangkat"]["v31"]["0–100 bpm"]["alarm"],
        lt100_wrist_man=ap["Wrist/manual"]["v31"]["0–100 bpm"]["alarm"],
        lt100_eph_dev=ap["EPHNOGRAM/perangkat"]["v31"]["0–100 bpm"]["alarm"])
    n_subj_total = sum(int(data[f"{s} / perangkat"]["subjek"]) for s in ("EPFL", "Wrist", "EPHNOGRAM"))

    # ── mitigasi ────────────────────────────────────────────────────────
    aturan = dict(asli=v33["aturan_asli"], afdb=v33["aturan_afdb"])
    ab = v33["abstain"]
    abstain = dict(ragu=ab["kosong_ragu=False"]["asli_≥120"]["tidak_yakin"],
                   ragu_kosong=ab["kosong_ragu=True"]["asli_≥120"]["tidak_yakin"],
                   alarm_kosong=ab["kosong_ragu=True"]["asli_≥120"]["alarm"],
                   normal_ragu=ab["kosong_ragu=True"]["afdb"]["Normal"][3], afl_ragu=ab["kosong_ragu=True"]["afdb"]["AFL"][3])

    return dict(bpms=bpms.tolist(), sint=sint, afdb120=afdb120, afdb_n=afdb_n, n_normal_afdb=n_normal_afdb,
                cv_afdb=v32["cv_desc"], afdb=afdb, chap=chap, chap_n=chap_n, latihan=latihan, pita=pita,
                per_sumber=per_sumber, detektor=detektor, n_subj_total=n_subj_total, aturan=aturan, abstain=abstain,
                p2_berpasangan=p2_berpasangan, n_synth=v32["n_synth"], afl_share=v32["afl_share_suspect"])


def kumpulkan_fondasi(A):
    """
    Mitigasi eksploratif (ditambahkan 14 November 2026): gerbang 2 kelas ECGFounder 1-lead zero-shot dengan ambang bawaan
    (D-0 v16.1: AF bila pAF + pAFL > pSinus, tanpa parameter yang disetel) dan aturan episode AF ≥ 30 s (v16.5).
    Angka per window dari cache v16.1/v13; metrik episode dihitung ulang dengan v16.5 `episode.py` setelah paritas dengan
    cache v16.5 (D-1 dan v3.1) di-assert. D-0 + 30 s belum ada di cache mana pun, jadi dihitung di sini.
    """
    sys.path.insert(0, str(ROOT / "v16.5_episode_30s"))
    import episode as E  # noqa: E402
    v161, v165, v13 = load("cache/v16.1/ringkasan.json"), load("cache/v16.5/ringkasan.json"), load("cache/v13/ringkasan.json")

    # paket P2 = "v3.1 paket" di v16.1: alarm latihan ≥ 120 bpm harus sama dengan Tabel I
    lat = v161["latihan"]
    assert abs(lat["v3.1 paket"]["alarm"] - A["latihan"]["p2"]["dev"]["alarm"]) < 1e-12
    assert lat["v3.1 paket"]["n"] == lat["D-0"]["n"] == v161["n"]["latihan_120"]
    assert abs(v161["ptbxl_kelas"]["v3.1 paket"]["STACH"] - v13["ptbxl_st"]["v3.1 paket"]) < 1e-12

    def m_ep(D, p, dur):
        return E.metrik(D["rec"], D["start"], D["y2"], E.aturan(D["rec"], D["start"], p, D["langkah"], dur),
                        D["langkah"], D["unit"])

    kunci = ("se_episode_30", "palsu_per_jam")
    ep = {}
    for ds, muat, d0_fn in (("latihan", E._latihan, lambda: E.H16.tahap1(E.H16.latihan()["logit"])[1] > 0),
                            ("afdb", E._afdb, lambda: E.A161.d1_afdb() > 0)):
        D = muat()
        H = v165["hasil"]["Latihan fisik" if ds == "latihan" else "AFDB"]
        ep[ds] = {}
        for dur in (0, 30):
            x = m_ep(D, D["d1"], dur)
            assert all(abs(x[k] - H["D-1"][str(dur)][k]) < 1e-9 or (np.isnan(x[k]) and np.isnan(H["D-1"][str(dur)][k]))
                       for k in kunci)
            xs = [m_ep(D, p, dur) for p in D["v31"]]
            p2 = {k: float(np.mean([q[k] for q in xs])) for k in kunci}
            assert all(abs(p2[k] - H["v3.1"][str(dur)][k]) < 1e-9 or (np.isnan(p2[k]) and np.isnan(H["v3.1"][str(dur)][k]))
                       for k in kunci)
            d0 = m_ep(D, d0_fn(), dur)
            ep[ds][dur] = dict(p2=p2, d0={k: float(d0[k]) for k in kunci})
        ep[ds]["jam"] = float(v165["data"]["Latihan fisik" if ds == "latihan" else "AFDB"]["jam"])
        ep[ds]["unit"] = int(v165["data"]["Latihan fisik" if ds == "latihan" else "AFDB"]["unit"])
    # episode AF D-0 tanpa aturan harus sama dengan v16.1 (palsu per jam AFDB)
    assert abs(ep["afdb"][0]["d0"]["palsu_per_jam"] - v161["episode"]["D-0"]["palsu_per_jam"]) < 1e-9

    assert v161["icentia"]["D-0"]["titik_b"] == v161["icentia"]["D-L"]["titik_b"]
    afdb = v161["afdb"]

    # Alarm (AF atau AFL) pada window AFL saja. v16.1 `per_pasien[...]["08378"]["se"]` dihitung atas SEMUA window AF + AFL
    # rekaman 08378 (1.532 AF + 306 AFL), jadi bukan angka flutter 2:1 (koreksi 14 November 2026).
    dv2 = np.load(ROOT / "cache/dataset_v2_detector.npz")
    y3, rec3 = dv2["y"], dv2["records"]
    D = E._afdb()
    d0_afdb = E.A161.d1_afdb() > 0
    p2_afdb = np.asarray(D["v31"], dtype=bool)                     # 5 ulangan × window
    assert len(y3) == len(d0_afdb) == p2_afdb.shape[1] and np.array_equal(D["rec"], rec3)
    assert np.array_equal(D["y2"] == 1, y3 > 0)
    m08 = rec3 == "08378"
    for s, x in (("D-0", d0_afdb[m08 & (y3 > 0)].mean()), ("v3.1", p2_afdb[:, m08 & (y3 > 0)].mean())):
        assert abs(x - v161["per_pasien"][s]["08378"]["se"]) < 1e-9   # paritas dengan v16.1 (AF + AFL)
    afl_alarm = {"p2": dict(r08378=float(p2_afdb[:, m08 & (y3 == 2)].mean()), semua=float(p2_afdb[:, y3 == 2].mean())),
                 "d0": dict(r08378=float(d0_afdb[m08 & (y3 == 2)].mean()), semua=float(d0_afdb[y3 == 2].mean()))}
    n_afl = dict(r08378=int((m08 & (y3 == 2)).sum()), semua=int((y3 == 2).sum()))
    return dict(
        latihan={k: dict(alarm=lat[s]["alarm"], ci=lat[s]["ci"], lt100=lat[s]["pita"]["0–100 bpm"]["alarm"])
                 for k, s in (("p2", "v3.1 paket"), ("d0", "D-0"))},
        n_latihan=lat["D-0"]["n"],
        stach={"p2": v161["ptbxl_kelas"]["v3.1 paket"]["STACH"], "d0": v161["ptbxl_kelas"]["D-0"]["STACH"]},
        n_stach=v13["n"]["ptbxl_st"],
        afdb={k: dict(se=afdb[s]["Se"][0], sp=afdb[s]["Sp"][0]) for k, s in (("p2", "v3.1"), ("d0", "D-0"))},
        se_beda=v161["banding"]["D-0"]["Se"],          # [Δ, lo, hi, P(Δ>0)]
        normal110={"p2": v161["alarm_normal"]["v3.1"]["110–120"]["alarm"], "d0": v161["alarm_normal"]["D-0"]["110–120"]["alarm"]},
        afl_alarm=afl_alarm, n_afl=n_afl,
        afl08378_3kelas=v13["afl_per_pasien"]["zero_50hz"]["08378"],
        # Icentia11k: titik_a = sistem, titik_b = paket v3.1 (pembanding sama untuk D-0 dan D-L)
        icentia_se={"d0": v161["icentia"]["D-0"]["titik_a"]["Se"][0], "p2": v161["icentia"]["D-0"]["titik_b"]["Se"][0]},
        icentia_se_beda=v161["icentia"]["D-0"]["Se"],
        episode=ep)


# ── LaTeX ───────────────────────────────────────────────────────────────
def pc(x, d=0):
    return f"{100 * x:.{d}f}"


def n3(x):
    return f"{x:.3f}"


def thr_txt(t):
    v = sorted(set(t.values()))
    return f"{v[0]}" if len(v) == 1 else f"{v[0]}--{v[-1]}"


def tulis_tex(A):
    m = {}
    S, L, F = A["sint"], A["latihan"], A["afdb"]
    afdb_keys = [k for k, g, *_ in MODELS if g == "afdb"]
    thr_afdb = [t for k in afdb_keys for t in S[k]["thr"].values()]
    m["nCfgAfdb"] = str(len(afdb_keys))
    m["nFam"] = "7"
    assert len({lab.split(",")[0].split(" (")[0].replace("P0 ", "").replace("P1 ", "").replace("P2 ", "")
                for k, g, lab, *_ in MODELS if g == "afdb"}) == 7
    m["thrMin"], m["thrMax"] = str(min(thr_afdb)), str(max(thr_afdb))
    m["thrPtwo"] = thr_txt(S["p2"]["thr"])
    m["nSynth"] = str(A["n_synth"])
    m["aflRegPtwo"] = pc(S["p2"]["afl_reg"])
    m["afGtwentyPtwo"] = pc(S["p2"]["af_g20"])
    m["thrVsevenA"], m["thrVeightA"] = thr_txt(S["v7a"]["thr"]), thr_txt(S["v8a"]["thr"])
    # AFDB
    m["nNormalAfdb"] = f"{A['n_normal_afdb']:,}"
    m["nAfdbHi"], m["nAfdbHiPat"] = str(A["afdb_n"][0]), str(A["afdb_n"][1])
    m["pctAfdbHi"] = f"{100 * A['afdb_n'][0] / A['n_normal_afdb']:.1f}"
    m["cvAfdbHi"] = f"{A['cv_afdb']['Normal ≥120 bpm']['median']:.2f}"
    m["cvAflHi"] = f"{A['cv_afdb']['AFL ≥120 bpm']['median']:.2f}"
    m["afdbHiPtwo"] = pc(A["afdb120"]["p2"])
    m["afdbHiMin"] = pc(min(A["afdb120"][k] for k in afdb_keys))
    m["afdbHiMax"] = pc(max(A["afdb120"][k] for k in afdb_keys))
    # Chapman
    C = A["chap"]
    m["chapNhi"] = f"{C['p2']['n120']:,}"
    m["chapNall"] = f"{sum(A['chap_n'].values()):,}"
    m["chapPtwo"] = pc(C["p2"]["alarm120"], 1)
    m["chapPtwoCI"] = f"[{pc(C['p2']['ci120'][0], 1)}, {pc(C['p2']['ci120'][1], 1)}]"
    m["chapPkgMin"] = pc(min(C[k]["alarm120"] for k in ("p0", "p1", "p2")), 1)
    m["chapPkgMax"] = pc(max(C[k]["alarm120"] for k in ("p0", "p1", "p2")), 1)
    m["chapBandOne"] = pc(C["p2"]["band"]["100–110"][0])
    m["chapBandTwo"] = pc(C["p2"]["band"]["110–120"][0])
    m["chapVeightA"] = pc(C["v8a"]["alarm120"]); m["chapVeightC"] = pc(C["v8c"]["alarm120"])
    # latihan fisik
    m["nWinEx"] = f"{L['p2']['dev']['n']:,}"
    m["nSubjEx"] = str(L["p2"]["dev"]["subjek"])
    m["nSubjTotal"] = str(A["n_subj_total"])
    m["nWinMan"], m["nSubjMan"] = str(L["p2"]["man"]["n"]), str(L["p2"]["man"]["subjek"])
    pasangan = A["p2_berpasangan"]
    m["nPairWin"], m["nPairSubj"] = f"{pasangan['n']:,}", str(pasangan["subjek"])
    m["nPairAlarmDev"], m["nPairAlarmRef"] = f"{pasangan['alarm_dev']:,}", f"{pasangan['alarm_man']:,}"
    m["exPtwo"] = pc(L["p2"]["dev"]["alarm"], 1)
    m["exPtwoCI"] = f"[{pc(L['p2']['dev']['ci'][0], 1)}, {pc(L['p2']['dev']['ci'][1], 1)}]"
    m["manPtwo"] = pc(L["p2"]["man"]["alarm"], 1)
    ex_afdb = [L[k]["dev"]["alarm"] for k in afdb_keys if k in L]
    m["nCfgEx"] = str(len(ex_afdb))
    m["exAfdbMin"], m["exAfdbMax"] = pc(min(ex_afdb), 1), pc(max(ex_afdb), 1)
    man_afdb = [L[k]["man"]["alarm"] for k in afdb_keys if k in L]
    m["manAfdbMin"] = pc(min(man_afdb), 1)
    for s, nama in (("EPFL", "Epfl"), ("Wrist", "Wrist"), ("EPHNOGRAM", "Eph")):
        m[f"src{nama}"] = pc(A["per_sumber"][s])
    for key, nama in (("v7a", "VsevenA"), ("v7b", "VsevenB"), ("v8a", "VeightA"), ("v8c", "VeightC")):
        b = L[key]["banding"]
        m[f"ex{nama}"] = pc(L[key]["dev"]["alarm"], 1)
        m[f"d{nama}"] = f"$-${100 * abs(b['delta']):.1f}"
        m[f"r{nama}"] = f"{F[key]['r08']:.2f}"
        dm = F[key]["d"]
        m[f"mfd{nama}"] = (f"{dm[0]:+.3f}".replace("-", "$-$").replace("+", "$+$"))
        m[f"mfp{nama}"] = f"{dm[3]:.2f}" if len(dm) > 3 else ""
    m["rPtwo"] = f"{F['p2']['r08']:.2f}"
    m["rPzero"] = f"{F['p0']['r08']:.2f}"
    D = A["detektor"]
    m["lolosDev"], m["lolosMan"] = f"{D['lolos_epfl_dev']:.0f}", f"{D['lolos_epfl_man']:.0f}"
    m["cvDev"], m["cvMan"] = f"{D['cv_epfl_dev']:.3f}", f"{D['cv_epfl_man']:.3f}"
    m["ltWristDev"], m["ltWristMan"], m["ltEphDev"] = pc(D["lt100_wrist_dev"]), pc(D["lt100_wrist_man"]), pc(D["lt100_eph_dev"])
    # mitigasi
    R = A["aturan"]
    tangkap = [R["asli"][f"{s}/perangkat"]["kena_aturan"] for s in ("EPFL", "Wrist", "EPHNOGRAM")]
    m["ruleMin"], m["ruleMax"] = pc(min(tangkap)), pc(max(tangkap))
    m["ruleManEpfl"] = pc(R["asli"]["EPFL/manual"]["kena_aturan"])
    m["ruleAfl"] = pc(R["afdb"]["AFL terdeteksi dialihkan"])
    m["ruleAf"] = pc(R["afdb"]["AFIB kena aturan"], 1)
    m["ruleNormal"] = pc(R["afdb"]["Normal kena aturan"], 1)
    B = A["abstain"]
    m["abstRagu"] = pc(B["ragu"])
    m["abstKosong"], m["abstAlarm"] = pc(B["ragu_kosong"], 1), pc(B["alarm_kosong"])
    m["abstNormal"], m["abstAfl"] = pc(B["normal_ragu"], 1), pc(B["afl_ragu"], 1)
    m["aflShare"] = pc(A["afl_share"])

    # baris Tabel I
    rows, grup_lama = [], None
    for key, g, lab, *_ in MODELS:
        if g != grup_lama:
            judul = "Trained on AFDB only" if g == "afdb" else "Trained with sinus-tachycardia examples (MicroTransformer, soft weights)"
            rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{judul}}}}} \\\\")
            grup_lama = g
        thr = thr_txt(S[key]["thr"]) + ("$^{a}$" if S[key]["tidak_monoton"] else "")
        hi = pc(A["afdb120"][key])
        ch = pc(C[key]["alarm120"], 1) if key in C else "--"
        if key in ("v8a", "v8c"):
            ch += "$^{b}$"
        if key in L:
            ci_lo, ci_hi = L[key]["dev"]["ci"]
            ex = f"{pc(L[key]['dev']['alarm'], 1)} [{pc(ci_lo, 1)}, {pc(ci_hi, 1)}]"
            mn = pc(L[key]["man"]["alarm"], 1)
        else:
            ex = mn = "--"
        r8 = f"{F[key]['r08']:.2f}"     # Macro-F1 AFDB per model sengaja tidak ditabelkan: sudah dilaporkan di paper 1
        rows.append(f"{lab} & {thr} & {hi} & {ch} & {ex} & {mn} & {r8} \\\\")
    m["tabelUtama"] = "\n" + "\n".join(rows) + "\n"

    # Tabel II: mitigasi eksploratif (P2 vs gerbang ECGFounder D-0)
    G = A["fondasi"]
    epL, epA = G["episode"]["latihan"], G["episode"]["afdb"]
    ci = lambda c: f"[{pc(c[0], 1)}, {pc(c[1], 1)}]"  # noqa: E731
    baris2 = [
        ("Exercise $\\geq$120\\,bpm: alarm (\\%)", f"{pc(G['latihan']['p2']['alarm'], 1)} {ci(G['latihan']['p2']['ci'])}",
         f"{pc(G['latihan']['d0']['alarm'], 1)} {ci(G['latihan']['d0']['ci'])}"),
        ("Exercise $<$100\\,bpm: alarm (\\%)", pc(G["latihan"]["p2"]["lt100"], 1), pc(G["latihan"]["d0"]["lt100"], 1)),
        ("Exercise: false alarms/h", f"{epL[0]['p2']['palsu_per_jam']:.1f}", f"{epL[0]['d0']['palsu_per_jam']:.1f}"),
        ("\\quad with 30\\,s rule", f"{epL[30]['p2']['palsu_per_jam']:.1f}", f"{epL[30]['d0']['palsu_per_jam']:.1f}"),
        (f"PTB-XL sinus tach.\\ ($n$={G['n_stach']}): alarm (\\%)", pc(G["stach"]["p2"], 1), pc(G["stach"]["d0"], 1)),
        ("AFDB Normal 110--120\\,bpm: alarm (\\%)", pc(G["normal110"]["p2"], 1), pc(G["normal110"]["d0"], 1)),
        ("AFDB sensitivity (window)", n3(G["afdb"]["p2"]["se"]), n3(G["afdb"]["d0"]["se"])),
        ("AFDB specificity (window)", n3(G["afdb"]["p2"]["sp"]), n3(G["afdb"]["d0"]["sp"])),
        ("AFDB episodes $\\geq$30\\,s found (30\\,s rule)", n3(epA[30]["p2"]["se_episode_30"]), n3(epA[30]["d0"]["se_episode_30"])),
        ("AFDB false alarms/h (30\\,s rule)", f"{epA[30]['p2']['palsu_per_jam']:.2f}", f"{epA[30]['d0']['palsu_per_jam']:.2f}"),
        (f"AFDB AFL windows ($n$={G['n_afl']['semua']:,}): alarmed", f"{G['afl_alarm']['p2']['semua']:.2f}",
         f"{G['afl_alarm']['d0']['semua']:.2f}"),
        (f"\\quad 08378, 2:1 flutter ($n$={G['n_afl']['r08378']})", f"{G['afl_alarm']['p2']['r08378']:.2f}",
         f"{G['afl_alarm']['d0']['r08378']:.2f}"),
        ("Icentia11k sensitivity (window)", n3(G["icentia_se"]["p2"]), n3(G["icentia_se"]["d0"])),
    ]
    m["tabelMitigasi"] = "\n" + "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in baris2) + "\n"
    m["gateExHi"] = pc(G["latihan"]["d0"]["alarm"], 1)
    m["gateExHiCiHi"] = pc(G["latihan"]["d0"]["ci"][1], 1)
    m["gateStach"], m["ptwoStach"] = pc(G["stach"]["d0"], 1), pc(G["stach"]["p2"], 1)
    m["nStach"] = str(G["n_stach"])
    m["ptwoExFa"], m["ptwoExFaRule"] = f"{epL[0]['p2']['palsu_per_jam']:.1f}", f"{epL[30]['p2']['palsu_per_jam']:.1f}"
    m["gateExFaRule"] = f"{epL[30]['d0']['palsu_per_jam']:.1f}"
    m["exHours"], m["exUnits"] = f"{epL['jam']:.0f}", str(epL["unit"])
    m["gateSeDiff"] = f"{G['se_beda'][0]:+.3f}".replace("-", "$-$")
    m["gateSeDiffLo"] = f"{G['se_beda'][1]:.3f}".replace("-", "$-$")
    m["gateSeDiffHi"] = f"{G['se_beda'][2]:.3f}".replace("-", "$-$")
    m["gateSeDiffP"] = f"{G['se_beda'][3]:.2f}"
    m["gateIcentiaSe"], m["ptwoIcentiaSe"] = n3(G["icentia_se"]["d0"]), n3(G["icentia_se"]["p2"])
    A8 = G["afl_alarm"]
    m["nEightThreeAfl"], m["nAflAfdb"] = str(G["n_afl"]["r08378"]), f"{G['n_afl']['semua']:,}"
    m["gateEightThreeHit"] = str(round(A8["d0"]["r08378"] * G["n_afl"]["r08378"]))
    m["gateAflAll"], m["ptwoAflAll"] = pc(A8["d0"]["semua"]), pc(A8["p2"]["semua"])
    m["gateAfdbSe"], m["ptwoAfdbSe"] = n3(G["afdb"]["d0"]["se"]), n3(G["afdb"]["p2"]["se"])
    m["gateAfdbSp"], m["ptwoAfdbSp"] = n3(G["afdb"]["d0"]["sp"]), n3(G["afdb"]["p2"]["sp"])
    m["gateEpSe"], m["ptwoEpSe"] = n3(epA[30]["d0"]["se_episode_30"]), n3(epA[30]["p2"]["se_episode_30"])
    lines =["% Dibuat otomatis oleh angka.py dari cache. Jangan diedit manual."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (HERE / "angka.tex").write_text("\n".join(lines) + "\n")


# ── Gambar ──────────────────────────────────────────────────────────────
def gaya():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    nama = {f.name for f in font_manager.fontManager.ttflist}
    fam = next((f for f in ("Arial", "Liberation Sans") if f in nama), "DejaVu Sans")
    plt.rcParams.update({"font.family": fam, "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titlesize": 7.5})
    return plt


def simpan(fig, nama):
    # ukuran tetap selebar kolom IEEE (3,5 in), tanpa bbox "tight", supaya huruf 7 pt tidak ikut mengecil di naskah
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{nama}.pdf")
    fig.savefig(FIG / f"{nama}.png", dpi=220)


def rapikan(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def gambar_sintetis(A):
    plt = gaya()
    from matplotlib.lines import Line2D
    bpm = np.array(A["bpms"])
    fig = plt.figure(figsize=(3.5, 2.3))
    ax = fig.add_axes([0.13, 0.17, 0.84, 0.62])
    for key, g, *_ in MODELS:
        if g == "afdb" and key != "p2":
            ax.plot(bpm, 100 * np.array(A["sint"][key]["normal"][JITTER_FIG]), color=BLUE, lw=0.7, alpha=0.35, zorder=2)
    ax.plot(bpm, 100 * np.array(A["sint"]["p2"]["normal"][JITTER_FIG]), color=BLUE, lw=1.6, marker="o", ms=3, zorder=4)
    ax.plot(bpm, 100 * np.array(A["sint"]["v7a"]["normal"][JITTER_FIG]), color=ORANGE, lw=1.2, marker="s", ms=3, zorder=3)
    ax.plot(bpm, 100 * np.array(A["sint"]["v8a"]["normal"][JITTER_FIG]), color=GREEN, lw=1.2, marker="^", ms=3.4, zorder=3)
    ax.axhline(50, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(58, 172)
    ax.set_ylim(-4, 104)
    ax.set_xticks(range(60, 171, 10))
    ax.set_xlabel("Heart rate of synthetic regular rhythm (bpm)")
    ax.set_ylabel("Windows classified Normal (%)")
    rapikan(ax)
    h = [Line2D([0], [0], color=BLUE, lw=0.7, alpha=0.5, label=f"AFDB only, other {sum(g == 'afdb' for _, g, *_ in MODELS) - 1}"),
         Line2D([0], [0], color=BLUE, lw=1.6, marker="o", ms=3, label="P2 reference"),
         Line2D([0], [0], color=ORANGE, lw=1.2, marker="s", ms=3, label="+ synthetic ST 100–130"),
         Line2D([0], [0], color=GREEN, lw=1.2, marker="^", ms=3.4, label="+ Chapman")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=6,
              handletextpad=0.4, columnspacing=1.0, borderaxespad=0.1)
    simpan(fig, "sintetis")
    plt.close(fig)


def gambar_pita(A):
    plt = gaya()
    from matplotlib.lines import Line2D
    lab = ["<100", "100–110", "110–120", "120–130", "130–140", "140–150", "≥150"]
    kunci = ["0–100", "100–110", "110–120", "120–130", "130–140", "140–150", "150–∞"]
    x = np.arange(len(lab))
    fig = plt.figure(figsize=(3.5, 2.25))
    ax = fig.add_axes([0.13, 0.17, 0.84, 0.64])
    seri = [("dev", "Exercise ECG, 50 Hz detector", BLUE, "o", -0.18),
            ("man", "Exercise ECG, reference R-peaks", ORANGE, "s", 0.0),
            ("chap", "Clinical ST (Chapman), 50 Hz detector", GREEN, "^", 0.18)]
    for nama, label, c, mk, dx in seri:
        xs, ys, lo, hi = [], [], [], []
        for i, k in enumerate(kunci):
            if nama == "chap":
                if k not in A["chap"]["p2"]["band"]:
                    continue
                v = A["chap"]["p2"]["band"][k]
                t, a, b = v
            else:
                v = A["pita"][nama].get(k)
                if v is None:
                    continue
                t, (a, b) = v["alarm"], v["ci"]
            xs.append(i + dx); ys.append(100 * t); lo.append(100 * (t - a)); hi.append(100 * (b - t))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, ms=3.6, color=c, ecolor=c, elinewidth=0.9, capsize=0, zorder=3,
                    label=label)
    ax.axvspan(2.5, len(lab) - 0.5, color=GRID, alpha=0.35, lw=0, zorder=0)
    ax.text(4.5, 55, f"≥120 bpm: {100 * A['afdb_n'][0] / A['n_normal_afdb']:.1f}% of AFDB\nNormal windows",
            ha="center", va="center", fontsize=6, color=MUTED)
    ax.set_xticks(x, lab)
    ax.set_xlim(-0.5, len(lab) - 0.5)
    ax.set_ylim(-4, 104)
    ax.set_xlabel("Heart rate (bpm)")
    ax.set_ylabel("Windows alarmed as AF/AFL (%)")
    rapikan(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False, fontsize=6, handletextpad=0.3,
              borderaxespad=0.1)
    simpan(fig, "pita")
    plt.close(fig)


def gambar_tradeoff(A):
    plt = gaya()
    L, F = A["latihan"], A["afdb"]
    fig = plt.figure(figsize=(3.5, 2.3))
    ax = fig.add_axes([0.13, 0.17, 0.84, 0.70])
    nama = {"p0": "P0", "p1": "P1", "p2": "P2", "bigru_s": "BiGRU s", "bigru_f": "BiGRU f", "tcn_f": "TCN f",
            "v7a": "ST 100–130", "v7b": "ST 100–160", "v8a": "Chapman", "v8c": "Chapman, no AFL"}
    geser = {"p0": (4, 3), "p1": (4, -8), "p2": (-12, -9), "bigru_s": (-30, 2), "bigru_f": (-30, 5), "tcn_f": (-22, -8),
             "v7a": (5, 2), "v7b": (6, 4), "v8a": (5, -2), "v8c": (-40, 5)}
    for key, g, *_ in MODELS:
        if key not in L:
            continue
        c, mk = (BLUE, "o") if g == "afdb" else ((ORANGE, "s") if key.startswith("v7") else (GREEN, "^"))
        x, y = 100 * L[key]["dev"]["alarm"], F[key]["r08"]
        lo, hi = L[key]["dev"]["ci"]
        ax.errorbar(x, y, xerr=[[x - 100 * lo], [100 * hi - x]], fmt=mk, ms=4, color=c, ecolor=c, elinewidth=0.8,
                    capsize=0, zorder=3)
        ax.annotate(nama[key], (x, y), xytext=geser[key], textcoords="offset points", fontsize=6, color=INK)
    ax.set_xlim(50, 104)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("Alarm on exercise ECG ≥120 bpm (%, 95% CI)")
    ax.set_ylabel("AFL recall, record 08378 (2:1)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(51, 1.03, "desired: low alarm, high recall", fontsize=6, color=MUTED, va="top")
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=BLUE, marker="o", ls="", ms=4, label="AFDB only"),
         Line2D([0], [0], color=ORANGE, marker="s", ls="", ms=4, label="+ synthetic ST"),
         Line2D([0], [0], color=GREEN, marker="^", ls="", ms=4, label="+ Chapman")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, fontsize=6,
              handletextpad=0.2, columnspacing=1.0, borderaxespad=0.1)
    simpan(fig, "tradeoff")
    plt.close(fig)


if __name__ == "__main__":
    A = kumpulkan()
    A["fondasi"] = kumpulkan_fondasi(A)
    (HERE / "angka.json").write_text(json.dumps(A, indent=1, ensure_ascii=False, default=float))
    tulis_tex(A)
    gambar_sintetis(A)
    gambar_pita(A)
    gambar_tradeoff(A)
    print({k: v["thr"] for k, v in A["sint"].items()})
    print({k: round(v["dev"]["alarm"], 3) for k, v in A["latihan"].items()})
