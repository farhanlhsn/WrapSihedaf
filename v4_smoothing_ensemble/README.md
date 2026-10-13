# v4_smoothing_ensemble — pasca-proses tanpa training

Pertanyaan: apakah smoothing antar-window atau ensemble MicroTransformer + BiGRU bisa memperbaiki sistem produksi v3.1?
Semua dikerjakan di atas OOF v3 (fold & seed resmi). Tidak ada training, dan paket produksi tidak disentuh.

| File | Isi |
|---|---|
| `smoothing_ensemble.ipynb` | Uji fungsi smoothing, prediksi semua varian (cross-fit), hasil window & episode, konfigurasi terpilih, per pasien |
| `pascaproses.py` | Smoothing (rata-rata bergerak, HMM; offline & real-time), ensemble, seleksi cross-fit, metrik episode |
| `laporan/` | `laporan_smoothing_ensemble.pdf` (6 halaman) + `build_report.py` |

**Metode.** Untuk tiap fold uji, konfigurasi smoothing (termasuk "tanpa smoothing") dan bias kelas dipilih dari fold lain.
Varian utama ditetapkan sebelum melihat hasil. Selain metrik per window, dihitung metrik episode: alarm AFL palsu per jam, dan
episode AFL yang terdeteksi (13 episode dalam 234 jam rekaman).

## Hasil (data uji detektor 50 Hz, CI 95% bootstrap pasien, Δ berpasangan vs v3.1)

| Varian | Macro-F1 | Alarm AFL palsu/jam | Episode AFL terdeteksi | Vonis Macro-F1 |
|---|---|---|---|---|
| v3.1 (referensi) | 0,832 [0,67–0,92] | 0,51 | 74% | — |
| Ensemble MT + BiGRU | 0,835 | 0,45 | 74% | tidak terbedakan (P = 0,72) |
| MT + smoothing offline | 0,810 | **0,09** | **38%** | tidak terbedakan (P = 0,11) |
| MT + smoothing real-time | 0,806 | 0,19 | 45% | lebih buruk (P = 0,03) |
| Ensemble + smoothing offline | 0,803 | 0,08 | 38% | tidak terbedakan |
| Ensemble + smoothing real-time | 0,794 | 0,23 | 48% | lebih buruk |

**Kesimpulan.** Tidak ada varian yang lebih baik dari v3.1. Paket produksi tetap v3.1, dan v4 tidak dikemas (tidak ada v4.1).
Smoothing adalah tukar-menukar: alarm palsu turun sekitar 6×, tetapi separuh episode AFL hilang, terutama episode pendek.
Smoothing bisa menjadi opsi produk ("hemat alarm") jika pemilik produk memutuskan demikian.

## Lingkungan hasil

Dijalankan 13 Oktober 2026 di WSL2 (Debian), PC desktop. Cross-fit paralel 5 proses CPU ±14 menit; tidak memakai GPU.

## Menjalankan (dari root repo)

```bash
python tools/run_notebook.py v4_smoothing_ensemble/smoothing_ensemble.ipynb   # ±14 menit; hasil di ../cache/v4/
python v4_smoothing_ensemble/laporan/build_report.py --extract
python v4_smoothing_ensemble/laporan/build_report.py
```

Untuk mengulang cross-fit dari awal, hapus `../cache/v4/pred.npz`. Prasyarat: OOF v3 di `../cache/oof_robust/`.
Di mesin dengan CPU sedikit (Colab/Kaggle), kurangi `workers` di `pascaproses.run_predictions()`; waktunya lebih lama.
