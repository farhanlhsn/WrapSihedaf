# v2_revisi

## Purpose
Revisi pipeline v1: praproses bersama (`hrv_core.py`), detektor R-peak 50 Hz, ekstraksi dua sumber R-peak,
benchmark 3 arsitektur DL + LightGBM, kalibrasi bias kelas, model produksi, dan `predictor.py` untuk backend.
Evaluasi yang lebih ketat **bukan** di sini, melainkan di `../v3_evaluasi_robust/`.

## Entry Points
- `revisi_sihedaf_v2.ipynb` — seluruh pipeline (cwd = folder ini).
- `predictor.py` → `ECGPredictor("model_export")`: `predict_ecg(sinyal, fs)` / `predict_ibi(ibi_ms)`.
- `laporan/build_report.py` — laporan PDF v2 (membaca output notebook).

## Contracts & Invariants
- **Notebook adalah sumber kebenaran modul.** Sel `%%writefile` menulis ulang `hrv_core.py`, `sihedaf_models.py`,
  `predictor.py`, dan `requirements.txt` setiap notebook dijalankan. Ubah kodenya di sel notebook, bukan di file
  `.py`, atau perubahan Anda akan tertimpa.
- Sel training final **mengekspor ulang** `model_export/sihedaf_v2.{pt,json}`, termasuk pada `SIHEDAF_QUICK=1`
  (model 2 epoch). Jangan jalankan notebook ini tanpa persetujuan pemilik repo; backend memakai file tersebut.
- `predictor.py` menolak bekerja jika `preprocessing` di JSON ≠ `hrv_core.preprocessing_config()` atau checksum
  bobot berbeda. Mengubah konstanta `hrv_core` berarti model harus dilatih & diekspor ulang.
- `../cache/dataset_v2_{annotation,detector}.npz` dibuat di sini dan dipakai v3. Mengubah ekstraksi
  membatalkan semua OOF di `../cache/oof_robust/`.
- `class_bias` v2 `[0.0, 2.0, -3.0]` terpotong grid. **Paket produksi aktif sekarang `../v3.1_kalibrasi_produksi/model_export/`**
  (bobot sama, bias `[0.0, -0.5, -4.5]`). Paket v2 dipertahankan sebagai titik rollback.

## Anti-patterns
- Menyalin kelas model ke file lain. Impor dari `sihedaf_models` (lihat cara v3).
- Memanggil `joblib.load` / pickle untuk model. Format v2 adalah `state_dict` + JSON.
