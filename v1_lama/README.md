# v1_lama — versi sebelum revisi (arsip)

Pipeline awal SIHEDAF. Disimpan apa adanya sebagai pembanding. **Jangan dikembangkan lagi**;
penggantinya adalah `../v2_revisi/`.

| File | Isi |
|---|---|
| `ekstraksi_hrv.py` | Ekstraksi AFDB: R-peak dari anotasi 250 Hz, window 10 s, stride N 5 s / AFIB 10 s / **AFL 1 s** |
| `model.ipynb` | Benchmark LightGBM (± SMOTE), 1D-ResNet, MicroTransformer (weighted loss / sampler) |
| `main.ipynb` | EDA dataset + eksperimen RandomForest/HistGB awal |
| `final.joblib` | Model MicroTransformer yang dulu dipakai backend (dilatih pada semua 23 pasien) |
| `predictor.py` | Predictor lama (memuat `final.joblib`) |
| `dokumentasi.md`, `cek_lama/` | Dokumentasi backend lama (front-end `find_peaks`) |
| `arahan.md` | Brief tugas benchmark awal |

## Hasil

| Sumber | Macro-F1 MicroTransformer |
|---|---|
| Dilaporkan `model.ipynb` (rata-rata per fold, satu partisi, data lab) | 0,826 ± 0,084 |
| Diukur ulang di v3, data lab, cara lapor yang sama | 0,803 ± 0,039 |
| Diukur ulang di v3, data uji detektor 50 Hz (kondisi perangkat) | 0,744 [0,64–0,81] |
| Diukur ulang di v3, front-end `find_peaks` lama (sistem utuh) | 0,418 [0,36–0,48] |

Penjelasan lengkap ada di laporan v3, halaman 2–3: `../v3_evaluasi_robust/laporan/laporan_evaluasi_robust.pdf`.

## Menjalankan (cwd = folder ini, WSL)

```bash
python ekstraksi_hrv.py          # → ../cache/v1/dataset_hrv_{tabular.csv,sequence_23p.npz}
python ../tools/run_notebook.py model.ipynb
```

## Perubahan dari commit asli

Hanya path, supaya cocok dengan struktur repo: data dibaca dari `../data/afdb` (dulu `./files`), dan
keluaran ekstraksi ditulis ke `../cache/v1/` (dulu root repo). `model.ipynb` dan `main.ipynb` membaca
dari path baru itu. Logika tidak diubah.

Catatan yang diketahui:
- Sel inferensi `model.ipynb` (sel 11) memakai bobot acak dan scaler dummy jika file tidak ada, dan IBI dalam detik.
- `main.ipynb` merujuk `dataset_hrv.csv` dan `raw_71.json` yang tidak ada di repo.
