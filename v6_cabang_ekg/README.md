# v6_cabang_ekg — cabang sinyal EKG mentah 50 Hz

Pertanyaan: setelah semua tuas berbasis interval RR habis (v3–v5), apakah **bentuk gelombang EKG** menambah informasi,
terutama untuk pasien AFL yang gagal di semua versi (07910, 08215, 06426)?

| Varian | Input | Hipotesis | Hasil |
|---|---|---|---|
| 6a. CNN EKG saja | EKG 10 s @50 Hz (500 sampel), bandpass 0,5–20 Hz | Morfologi saja memuat info AFL | Sebagian: ada info, tapi Macro-F1 jauh lebih buruk |
| 6b. Dua cabang EKG + RR (utama) | CNN EKG + badan MicroTransformer v2 | Lebih baik dari v3.1 | **Ditolak** |
| Late fusion (eksploratif) | rata-rata probabilitas v3.1 dan 6a | — | Setara; perlu konfirmasi |

MiniRocket direncanakan sebagai baseline, tapi pustakanya (aeon/sktime) tidak terpasang. Posisinya digantikan CNN EKG saja (6a).

| File | Isi |
|---|---|
| `cabang_ekg.ipynb` | Potongan EKG + bukti sejajar, spektrum per kelas, training 2 × 5 × 5 fold, hasil, per pasien |
| `ekg.py` | Prapemrosesan EKG, arsitektur (CNN, dua cabang), training (bobot lunak, augmentasi polaritas), evaluasi |
| `laporan/` | `laporan_cabang_ekg.pdf` (5 halaman) + `build_report.py` |

## Hasil (data uji detektor 50 Hz, bias cross-fit grid −7…+3, Δ berpasangan vs v3.1)

| | Macro-F1 | AFL F1 | Recall Normal | AFL palsu/ulangan | Vonis |
|---|---|---|---|---|---|
| v3.1 (RR saja) | 0,832 [0,67–0,92] | 0,647 | 0,924 | 274 | referensi |
| 6a. CNN EKG saja | 0,651 | 0,380 | 0,751 | 1.160 | lebih buruk (P = 0,00) |
| 6b. Dua cabang | 0,747 | 0,583 | 0,800 | 464 | tidak terbedakan, cenderung buruk (P = 0,07) |
| Late fusion | 0,838 | 0,632 | **0,965** | **156** | tidak terbedakan (P = 0,62) |

**Kesimpulan.** Paket produksi tetap v3.1; v6 tidak dikemas.
- CNN berbasis EKG tidak tergeneralisasi ke pasien baru: bias yang dibutuhkan melompat antar fold (sampai −7…+3).
- Pasien 07910 dan 08215 tetap 0 pada semua model.
- Late fusion menaikkan recall Normal secara konsisten dan mengurangi AFL palsu, tapi Macro-F1 setara.
- Batas proyek sekarang adalah jumlah pasien AFL (7), bukan model. Langkah berikutnya: data dengan lebih banyak pasien AFL (v7).

## Lingkungan hasil

Dijalankan 13 Oktober 2026 di WSL2 (Debian), PC desktop dengan GPU NVIDIA 8 GB. Training ±2 jam (122 menit)
(CNN 132 s/fold, dua cabang 160 s/fold).

## Menjalankan (dari root repo)

```bash
python tools/run_notebook.py v6_cabang_ekg/cabang_ekg.ipynb   # ±2 jam GPU; job yang sudah ada dilewati
python v6_cabang_ekg/laporan/build_report.py --extract
python v6_cabang_ekg/laporan/build_report.py
```

Potongan EKG di `../cache/v6/ecg_windows.npz` (float16, ±150 MB); OOF di
`../cache/oof_robust/detector_e30_r5_s2026/{EKG_CNN,EKG_RR_DualBranch}_r*.npz`. Prasyarat: `../data/afdb`, cache v2–v5.
