# v5_training_afl — dua resep training untuk AFL

Pertanyaan: apakah resep training MicroTransformer bisa diubah supaya AFL terdeteksi lebih baik daripada v3.1?
Fold, seed latih, hiperparameter, dan window uji identik dengan referensi; yang berubah hanya resep training.

| Varian | Yang diubah | Hipotesis | Status |
|---|---|---|---|
| 5a. AFL rapat | Window AFL di data **latih** digeser tiap 1 s (5.733 vs 1.152 window AFL); uji tetap 5 s | AFL recall/F1 naik, Macro-F1 tidak turun | **Ditolak** |
| 5b. Bobot lunak | Bobot kelas di loss = akar bobot standar (AFL 6,9× bukan 48×) | Tidak butuh bias ekstrem, Macro-F1 tidak turun | **Terbukti** |

| File | Isi |
|---|---|
| `training_afl.ipynb` | Verifikasi data & paritas training, training 2 × 5 × 5 fold, hasil, kalibrasi, per pasien |
| `latih.py` | Ekstraksi window AFL rapat (salinan ekstraktor v2, dibuktikan identik), training berbobot, evaluasi |
| `laporan/` | `laporan_training_afl.pdf` (5 halaman) + `build_report.py` |

## Hasil (data uji detektor 50 Hz, bias cross-fit grid −7…+3, Δ berpasangan vs v3.1)

| | Macro-F1 | AFL F1 | Deteksi episode AFL | Bias AFL (median) | Tanpa bias: Macro-F1 |
|---|---|---|---|---|---|
| v3.1 (referensi) | 0,832 [0,67–0,92] | 0,647 | 74% | −4,5 | 0,717 |
| 5a. AFL rapat | 0,836 (Δ +0,004, P = 0,78) | 0,660 | 69% (turun konsisten) | −4,75 | 0,709 |
| 5b. Bobot lunak | 0,830 (Δ −0,001, P = 0,48) | 0,637 | 78% | **−2,25** | **0,793** (P = 1,00) |

**Kesimpulan.** Tidak ada resep yang lebih baik dari v3.1, jadi paket produksi tetap v3.1 dan v5 tidak dikemas.
Bobot lunak dijadikan resep bawaan untuk model baru: skornya setara, koreksi biasnya setengah, dan tanpa bias jauh lebih baik.
Tapi 5b tidak lebih tahan bila bias terpasang lalu meleset. Semua tuas berbasis interval RR kini sudah dicoba; langkah
berikutnya (v6) adalah cabang EKG mentah atau data AFL tambahan.

## Lingkungan hasil

Dijalankan 13 Oktober 2026 di WSL2 (Debian), PC desktop dengan GPU NVIDIA 8 GB: training ±58 menit (10 job × 5 fold),
evaluasi ±3 menit.

## Menjalankan (dari root repo)

```bash
python tools/run_notebook.py v5_training_afl/training_afl.ipynb   # ±60 menit GPU; job yang sudah ada dilewati
python v5_training_afl/laporan/build_report.py --extract
python v5_training_afl/laporan/build_report.py
```

OOF tersimpan di `../cache/oof_robust/detector_e30_r5_s2026/MicroTransformer_{aflrapat,bobotlunak}_r*.npz`; data latih
AFL rapat di `../cache/v5/`. Prasyarat: `../data/afdb`, cache v2, dan OOF v3/v1.
