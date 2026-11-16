"""
cek_gelombang_p.py — sampel strip EKG latihan fisik ≥ 120 bpm untuk cek visual gelombang P (paper 2).

Data latihan fisik (v3.3) tidak punya label ritme; subjek sehat diasumsikan sinus. Skrip ini memilih sampel acak bertingkat
(sumber × subjek, seed tetap) dari window ≥ 120 bpm yang lolos cek kualitas, lalu menggambar potongan 4 s pada frekuensi asli
(filter 0,5–40 Hz) dengan R-peak yang dipakai window. Penilaian (gelombang P di depan setiap QRS, interval PR tetap) dicatat
manual di `publikasi/paper2/README.md`; idealnya dikonfirmasi klinisi.

Keluaran: cache/publikasi/paper2/gelombang_p/strip_*.png dan sampel.json. Jalankan dari root repo:
    python publikasi/paper2/cek_gelombang_p.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v3.3_takikardia_asli"))
import takikardia_asli as TA  # noqa: E402

OUT = REPO / "cache/publikasi/paper2/gelombang_p"
SEED = 2026
PER_SUMBER = {"EPFL": 8, "Wrist": 4, "EPHNOGRAM": 8}
MODE = {"EPFL": "manual", "Wrist": "manual", "EPHNOGRAM": "perangkat"}   # EPHNOGRAM tidak punya R-peak referensi
DETIK = 4.0


def pilih_sampel(J):
    rng = np.random.default_rng(SEED)
    sampel = []
    for src, n in PER_SUMBER.items():
        idx = np.flatnonzero(J["ok"] & (J["hr"] >= 120) & (J["sumber"] == src) & (J["mode"] == MODE[src]))
        subjek = rng.permutation(sorted(set(J["subjek"][idx])))[:n]
        for s in subjek:
            i = rng.choice(idx[J["subjek"][idx] == s])
            sampel.append(dict(i=int(i), sumber=src, subjek=str(s), rekaman=str(J["rekaman"][i]),
                               start=float(J["start"][i]), hr=float(J["hr"][i]), cv=float(J["cv"][i]),
                               aktivitas=str(J["aktivitas"][i]), mode=MODE[src]))
    return sampel


def baca(src, rekaman):
    """Sinyal EKG (asli) dan R-peak referensi (bila ada) untuk satu rekaman, tanpa membaca seluruh dataset."""
    if src == "EPFL":
        return next(r for r in TA.rekaman_epfl() if r["rekaman"] == rekaman)
    if src == "Wrist":
        return next(r for r in TA.rekaman_wrist() if r["rekaman"] == rekaman)
    base = TA.DATA / "ephnogram/WFDB" / rekaman
    baris = base.with_suffix(".hea").read_text().splitlines()
    nama = [b.split()[-1] for b in baris[1:1 + int(baris[0].split()[1])]]
    x, fs = TA.A12.baca_dat(base, nama.index("ECG"))
    return dict(ecg=np.asarray(x, float), fs=float(fs), rpeak=None)


def r_peak(rek, s):
    """R-peak yang dipakai window (indeks sampel pada frekuensi asli)."""
    if s["mode"] == "manual":
        return np.sort(rek["rpeak"])
    _, peaks = TA.hrv_core.signal_to_windows(np.nan_to_num(rek["ecg"]), rek["fs"], TA.STRIDE)
    return np.round(np.asarray(peaks) * rek["fs"] / TA.hrv_core.FS_DEVICE).astype(int)


def gambar(sampel, sinyal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    per = 5
    for h in range(0, len(sampel), per):
        bag = sampel[h:h + per]
        fig, axes = plt.subplots(len(bag), 1, figsize=(11, 2.1 * len(bag)))
        for ax, s in zip(np.atleast_1d(axes), bag):
            t, y, rp = sinyal[s["no"]]
            ax.plot(t, y, color="#172B43", lw=0.8)
            ax.plot(rp, np.interp(rp, t, y), "v", color="#eb6834", ms=5)
            ax.set_xlim(t[0], t[-1])
            ax.set_title(f"#{s['no']}  {s['sumber']} {s['subjek']} {s['rekaman']} t={s['start']:.0f}s  "
                         f"{s['hr']:.0f} bpm  CV {s['cv']:.3f}  ({s['mode']})  {s['aktivitas'][:40]}", fontsize=8, loc="left")
            ax.set_xticks(np.arange(np.ceil(t[0] * 5) / 5, t[-1], 0.2), minor=True)
            ax.grid(which="both", axis="x", color="#e6e6e6", lw=0.5)
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(OUT / f"strip_{h // per + 1}.png", dpi=130)
        plt.close(fig)


def main():
    J = TA.muat_jendela() if hasattr(TA, "muat_jendela") else np.load(TA.CACHE / "jendela.npz")
    sampel = pilih_sampel(J)
    sinyal = {}
    sos_cache = {}
    for no, s in enumerate(sampel, 1):
        s["no"] = no
        rek = baca(s["sumber"], s["rekaman"])
        fs = rek["fs"]
        x = np.nan_to_num(np.asarray(rek["ecg"], float))
        rp = r_peak(rek, s)
        if fs not in sos_cache:
            sos_cache[fs] = butter(3, [0.5, 40.0], btype="band", fs=fs, output="sos")
        i0, i1 = int(s["start"] * fs), int((s["start"] + DETIK) * fs)
        lo, hi = max(i0 - int(2 * fs), 0), min(i1 + int(2 * fs), x.size)
        y = sosfiltfilt(sos_cache[fs], x[lo:hi])[i0 - lo:i1 - lo]
        step = max(int(fs // 500), 1)                     # EPHNOGRAM 8 kHz → ±500 Hz untuk gambar
        t = (np.arange(i0, i1) / fs)[::step]
        y = y[::step]
        r = rp[(rp >= i0) & (rp < i1)] / fs
        sinyal[no] = (t, y, r)
    gambar(sampel, sinyal)
    (OUT / "sampel.json").write_text(json.dumps(sampel, indent=1, ensure_ascii=False))
    print(f"{len(sampel)} strip → {OUT}")


if __name__ == "__main__":
    main()
