# v2_revisi — revisi pipeline & model produksi

Merevisi v1: training–serving skew (R-peak anotasi 250 Hz vs `find_peaks` 50 Hz), fallback diam-diam,
asal-usul model yang tidak tertelusuri, stride AFL yang tumpang tindih, dan 1D-ResNet yang mengabaikan mask.
Detail tiap perbaikan ada di sel pertama notebook.

| File | Isi |
|---|---|
| `revisi_sihedaf_v2.ipynb` | Pipeline lengkap: unduh/ekstraksi AFDB (anotasi & detektor 50 Hz), benchmark, kalibrasi bias, eksperimen pergeseran domain, latensi, training final, ekspor, uji predictor |
| `hrv_core.py` | Praproses bersama training & produksi (detektor R-peak, IBI, cek kualitas window, fitur HRV) |
| `sihedaf_models.py` | MicroTransformer, MaskedResNet1D, BiGRUAttention |
| `predictor.py` | `ECGPredictor` untuk backend (tanpa pickle, cek checksum & konstanta praproses) |
| `model_export/` | Model produksi: `sihedaf_v2.pt` (state_dict) + `sihedaf_v2.json` (metadata) |
| `laporan/` | `laporan_hasil_sihedaf_v2.pdf` + `build_report.py` |

Tiga file `.py` di atas **ditulis oleh notebook** (`%%writefile`); ubah di notebook. Lihat `AGENTS.md`.

## Hasil (dari notebook, satu partisi pasien)

| Model | Macro-F1 (data anotasi, argmax) | Macro-F1 (+ bias) |
|---|---|---|
| LightGBM | 0,753 | 0,739 |
| MicroTransformer | 0,681 | 0,838 |
| MaskedResNet1D | 0,764 | 0,813 |
| BiGRUAttention | 0,732 | 0,842 |

Model produksi: MicroTransformer, dilatih pada data detektor 50 Hz. **Paket aktif sekarang ada di
`../v3.1_kalibrasi_produksi/`** (bobot sama, bias baru); paket di sini disimpan sebagai rollback. Metadata melaporkan Macro-F1 0,792
(bias cross-fit). Evaluasi v3 menunjukkan selisih antar model di tabel ini sebagian besar noise, dan grid
bias −3…+3 memotong optimum. Lihat `../v3_evaluasi_robust/`.

## Lingkungan hasil

Hasil di atas dan `model_export/` dibuat 12 Oktober 2026 di WSL2 (Debian), PC desktop dengan GPU NVIDIA 8 GB, Python 3.13, torch 2.11 (detail versi ada di `model_export/sihedaf_v2.json`). Notebook penuh ±1 jam.

## Menjalankan

```bash
python tools/run_notebook.py v2_revisi/revisi_sihedaf_v2.ipynb      # dari root repo, ±1 jam (GPU)
```

⚠ Ini **mengekspor ulang `model_export/`**, termasuk pada `SIHEDAF_QUICK=1`. Minta persetujuan dulu.

Laporan: `python v2_revisi/laporan/build_report.py --extract` (WSL), lalu tanpa argumen (Python + ReportLab).
