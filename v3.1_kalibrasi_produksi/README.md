# v3.1_kalibrasi_produksi — menerapkan keputusan v3

Versi minor: **tanpa training atau eksperimen baru**. Menerapkan lima keputusan dari laporan v3 dan menghasilkan
paket produksi aktif.

| Keputusan v3 | Status di v3.1 |
|---|---|
| 1. Ganti sistem lama (`final.joblib` + `find_peaks`) | Paket baru + panduan migrasi [`BACKEND.md`](BACKEND.md) |
| 2. Tetap MicroTransformer | Bobot v2 dipakai apa adanya (checksum sama) |
| 3. Stop pencarian arsitektur RR-only | Dicatat di `README.md` dan `AGENTS.md` root |
| 4. Kalibrasi ulang `class_bias` (grid −7…+3) | **Selesai:** `[0.0, 2.0, -3.0]` → `[0.0, -0.5, -4.5]` |
| 5. Protokol v3 untuk eksperimen berikutnya | Semua angka di sini memakai fold, seed, dan bootstrap v3 |

| File | Isi |
|---|---|
| `kalibrasi_produksi.ipynb` | Pemilihan bias, perbandingan dengan v2 produksi & v1, cek model final, ekspor paket, uji paritas, demo |
| `kalibrasi.py` | Logika perhitungan (dipakai notebook dan pembuat laporan) |
| `model_export/sihedaf_v3_1.{pt,json}` | **Paket produksi aktif.** `.pt` identik dengan v2; JSON berisi bias baru, asal-usul, metrik protokol v3 |
| `BACKEND.md` | Panduan integrasi & migrasi backend |
| `laporan/` | `laporan_kalibrasi_produksi.pdf` + `build_report.py` |

## Hasil (data uji detektor 50 Hz, OOF v3, CI 95% bootstrap pasien)

| Sistem | Macro-F1 | AFL F1 | AFL P | AFL R | Normal R |
|---|---|---|---|---|---|
| v1 (sebelum revisi, diukur ulang) | 0,744 [0,64–0,81] | 0,399 | 0,275 | 0,728 | 0,903 |
| v2 produksi, bias `[0.0, 2.0, -3.0]` | 0,779 [0,63–0,82] | 0,536 | 0,598 | 0,487 | 0,847 |
| **v3.1, bias `[0.0, -0.5, -4.5]`** (cross-fit) | **0,832** [0,67–0,92] | **0,647** | **0,715** | 0,592 | **0,924** |

- v3.1 vs v2 produksi: Δ Macro-F1 +0,053 [+0,01…+0,12], P(Δ>0) = 1,00 → lebih baik secara konsisten.
- v2 produksi vs v1: tidak terbedakan (P = 0,90). Bias v2 (AFIB +2) membuat banyak window Normal ditebak AFIB.
- AFL palsu pada pasien tanpa AFL: 72 → 25 window per ulangan.

## Lingkungan hasil

Dijalankan 13 Oktober 2026 di WSL2 (Debian), PC desktop dengan GPU NVIDIA 8 GB. Notebook ±2 menit (tanpa training).

## Menjalankan (dari root repo)

```bash
python tools/run_notebook.py v3.1_kalibrasi_produksi/kalibrasi_produksi.ipynb   # menulis ulang model_export/ folder ini
python v3.1_kalibrasi_produksi/laporan/build_report.py --extract
python v3.1_kalibrasi_produksi/laporan/build_report.py
```

Prasyarat: cache v2 (`dataset_v2_detector.npz`) dan v3 (OOF MicroTransformer + `versi_lama/`), serta `../data/afdb`
untuk demo.
