"""
data.py — data, fold, OOF pembanding, dan set uji stres v11 (dipakai semua folder model).

Data TIDAK dibuat ulang. Semua berkas diambil dari cache tim (atau dari zip GitHub Release di Kaggle/Colab) lalu
dicocokkan dengan sidik jari SHA-256 yang di-commit (`v11_model_tim/sidik_jari.json`). Hash tidak cocok → error,
karena OOF pembanding v1/v3.1 hanya sejajar dengan window cache yang persis sama.

Lokasi:
- data: `$SIHEDAF_DATA` jika diset, selain itu `<repo>/cache` (strukturnya sama dengan `cache/`);
- hasil: `$SIHEDAF_HASIL` jika diset (mis. folder Google Drive), selain itu `<repo>/cache/v11`. Saat SIHEDAF_QUICK=1
  hasil masuk `cache/v11_quick` atau `$SIHEDAF_HASIL/_quick`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent          # v11_model_tim/bersama
V11 = HERE.parent                               # v11_model_tim


def _cari_repo(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "AGENTS.md").exists():
            return p
    raise RuntimeError(f"Root repo (folder yang berisi AGENTS.md) tidak ditemukan dari {start}")


REPO = _cari_repo(HERE)
# Kode versi lama dipakai lewat impor, tidak disalin (AGENTS.md).
for _d in ("v2_revisi", "v3_evaluasi_robust", "v4_smoothing_ensemble", "v3.2_uji_takikardia"):
    if str(REPO / _d) not in sys.path:
        sys.path.insert(0, str(REPO / _d))
import sihedaf_cv as cv  # noqa: E402
import hrv_core  # noqa: E402
import takikardia as T  # noqa: E402

QUICK = os.environ.get("SIHEDAF_QUICK") == "1"          # uji alur saja, bukan hasil
N_ULANGAN, N_FOLD, SEED = 5, 5, 2026
OOF = "oof_robust/detector_e30_r5_s2026"
BERKAS = (
    "dataset_v2_detector.npz",
    f"{OOF}/folds.npz",
    *(f"{OOF}/MicroTransformer_r{r}.npz" for r in range(N_ULANGAN)),         # v3.1 = OOF ini + bias cross-fit
    *(f"{OOF}/versi_lama/oof_r{r}.npz" for r in range(N_ULANGAN)),           # v1, argmax
    "v7/MicroTransformer_kontrol_r0.npz",                                      # kontrol per fold uji takikardia
    "v7/uji_kontrol_r0.npz",
)
SIDIK_JARI = V11 / "sidik_jari.json"
REPO_UNDUHAN = REPO / "cache" / "v11_unduhan"
PESAN_MANUAL = (
    "Data v11 tidak ditemukan atau tidak cocok, dan unduhan dari {url} gagal ({err}).\n"
    "Cara manual: unduh v11_data.zip dari GitHub Release repo (tag v11-data), ekstrak ke sebuah folder "
    "(di Kaggle: tambahkan sebagai dataset), lalu set os.environ['SIHEDAF_DATA'] = '<folder itu>' "
    "sebelum menjalankan sel ini lagi. Di Kaggle, pastikan Internet aktif untuk unduhan otomatis."
)
_TERPERIKSA: set[str] = set()


def seed_latih(r: int, k: int) -> int:
    """Seed latih protokol (sama dengan v3/v5): 2026 + 100·r + k."""
    return SEED + 100 * r + k


def lokasi_data() -> Path:
    v = os.environ.get("SIHEDAF_DATA")
    return Path(v) if v else REPO / "cache"


def lokasi_hasil() -> Path:
    """Hasil QUICK selalu di folder terpisah, juga bila $SIHEDAF_HASIL diset (mis. Google Drive)."""
    v = os.environ.get("SIHEDAF_HASIL")
    if v:
        return Path(v) / "_quick" if QUICK else Path(v)
    return REPO / "cache" / ("v11_quick" if QUICK else "v11")


# ── sidik jari ─────────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def buat_sidik_jari(akar: Path | None = None) -> dict:
    """Hitung SHA-256 + ukuran semua BERKAS di `akar` dan tulis ke sidik_jari.json (hanya pemilik repo)."""
    akar = Path(akar) if akar else lokasi_data()
    sj = {"berkas": {}, "ukuran": {}}
    for rel in BERKAS:
        p = akar / rel
        sj["berkas"][rel] = _sha256(p)
        sj["ukuran"][rel] = p.stat().st_size
    SIDIK_JARI.write_text(json.dumps(sj, indent=2) + "\n", encoding="utf-8")
    return sj


def periksa_sidik_jari(akar: Path | None = None) -> None:
    """Error (RuntimeError) bila ada berkas yang hilang atau isinya berbeda dari sidik jari yang di-commit."""
    akar = Path(akar) if akar else lokasi_data()
    if str(akar.resolve()) in _TERPERIKSA:
        return
    if not SIDIK_JARI.exists():
        raise RuntimeError(f"{SIDIK_JARI} tidak ada; repo tidak lengkap.")
    sj = json.loads(SIDIK_JARI.read_text(encoding="utf-8"))
    salah = []
    for rel, h in sj["berkas"].items():
        p = akar / rel
        if not p.exists():
            salah.append(f"hilang  : {rel}")
        elif p.stat().st_size != sj["ukuran"][rel] or _sha256(p) != h:
            salah.append(f"berbeda : {rel}")
    if salah:
        raise RuntimeError(f"Data di {akar} tidak cocok dengan sidik jari v11:\n  " + "\n  ".join(salah))
    _TERPERIKSA.add(str(akar.resolve()))


# ── data ───────────────────────────────────────────────────────────────
def muat() -> SimpleNamespace:
    """Dataset detektor + fold + OOF pembanding. Setiap array npz dibaca sekali saja."""
    akar = lokasi_data()
    periksa_sidik_jari(akar)
    z = np.load(akar / BERKAS[0])
    X, M, F, y = z["X"], z["M"], z["F"], z["y"]
    records, t = z["records"].astype(str), z["t_start"].astype(np.float64)
    patients = np.unique(records)
    starts = np.flatnonzero(np.r_[True, records[1:] != records[:-1]])
    oofd = akar / OOF
    return SimpleNamespace(
        X=X, M=M, F=F, y=y, records=records, t=t, patients=patients,
        strata=cv.patient_strata(y, records, patients),
        folds=np.load(oofd / "folds.npz")["folds"],
        hr=(60000.0 / ((X[:, :, 0] * M).sum(1) / M.sum(1))).astype(np.float64),
        bounds=list(zip(starts, np.r_[starts[1:], len(y)])),
        oof_v31=np.stack([np.load(oofd / f"MicroTransformer_r{r}.npz")["oof"] for r in range(N_ULANGAN)]),
        oof_v1=np.stack([np.load(oofd / "versi_lama" / f"oof_r{r}.npz")["det"] for r in range(N_ULANGAN)]),
        kontrol_v7=SimpleNamespace(oof=np.load(akar / "v7/MicroTransformer_kontrol_r0.npz")["oof"],
                                   stres=np.load(akar / "v7/uji_kontrol_r0.npz")["stress"]),
        akar=akar,
    )


def set_uji_stres() -> SimpleNamespace:
    """
    Set uji stres v3.2: ritme teratur 60–170 bpm × 4 jenis variasi × 50 window (S = 4.600), urutan dan seed sama
    dengan v3.2/v7. Fitur HRV dihitung dari IBI float64 sintetis, BUKAN dari X float32 (hasilnya berbeda).
    """
    ibi, jenis, bpm = [], [], []
    for j, (nama, (kind, scale)) in enumerate(T.JITTERS.items()):
        for b in T.BPMS:
            w = T.synthetic_ibi(b, kind, scale, T.N_SYNTH, T.SEED + 1000 * j + b)
            ibi += w
            jenis += [nama] * len(w)
            bpm += [b] * len(w)
    seqs = [hrv_core.build_sequence(v) for v in ibi]
    return SimpleNamespace(
        ibi=ibi,
        X=np.stack([s[0] for s in seqs]).astype(np.float32),
        M=np.stack([s[1] for s in seqs]).astype(np.float32),
        F=np.stack([hrv_core.hrv_features(v) for v in ibi]),
        jenis=np.array(jenis), bpm=np.array(bpm),
    )


def bobot_sampel(y: np.ndarray, pangkat: float = 0.5) -> np.ndarray:
    """Bobot kelas per sampel: class_weights(y) ** pangkat, dari data latih itu sendiri. Bawaan 0,5 = lunak (AGENTS.md);
    1,0 = bobot seimbang penuh."""
    return (cv.class_weights(y) ** pangkat)[y].astype(np.float64)


# ── zip untuk Kaggle/Colab ─────────────────────────────────────────────
def buat_zip(tujuan: Path, akar: Path | None = None) -> Path:
    """v11_data.zip mandiri (BERKAS dengan path relatif yang sama + sidik_jari.json), untuk GitHub Release."""
    akar = Path(akar) if akar else lokasi_data()
    periksa_sidik_jari(akar)
    tujuan = Path(tujuan)
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    tmp = tujuan.with_name(tujuan.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in BERKAS:
            zf.write(akar / rel, rel)
        zf.write(SIDIK_JARI, "sidik_jari.json")
    tmp.replace(tujuan)
    return tujuan


def _lolos(akar: Path) -> bool:
    try:
        periksa_sidik_jari(akar)
        return True
    except RuntimeError:
        return False


def siapkan_data(url: str) -> Path:
    """
    Pastikan data v11 tersedia dan cocok, lalu kembalikan lokasinya.
    1. `$SIHEDAF_DATA` diset → hanya folder itu yang diperiksa (tidak cocok → error, tidak mengunduh).
    2. `<repo>/cache` atau unduhan sebelumnya (`cache/v11_unduhan`) cocok → dipakai.
    3. Selain itu zip diunduh dari `url`, diekstrak ke `cache/v11_unduhan`, diperiksa, dan `$SIHEDAF_DATA` diset.
    """
    if os.environ.get("SIHEDAF_DATA"):
        periksa_sidik_jari(lokasi_data())
        return lokasi_data()
    for akar in (REPO / "cache", REPO_UNDUHAN):
        if _lolos(akar):
            os.environ["SIHEDAF_DATA"] = str(akar)
            return akar
    REPO_UNDUHAN.mkdir(parents=True, exist_ok=True)
    zpath = REPO_UNDUHAN / "v11_data.zip"
    tmp = REPO_UNDUHAN / "v11_data.zip.tmp"
    try:
        with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
    except (OSError, ValueError) as e:                       # URLError/HTTPError turunan OSError
        raise RuntimeError(PESAN_MANUAL.format(url=url, err=e)) from e
    tmp.replace(zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(REPO_UNDUHAN)
    zpath.unlink()
    periksa_sidik_jari(REPO_UNDUHAN)
    os.environ["SIHEDAF_DATA"] = str(REPO_UNDUHAN)
    return REPO_UNDUHAN
