# v3_evaluasi_robust — evaluasi yang bisa dipercaya + perbandingan dengan v1

Pertanyaan: apakah selisih antar model di v1/v2 itu nyata, dan seberapa jauh sistem sekarang lebih baik dari
versi sebelum revisi? Folder ini tidak melatih atau mengekspor model produksi.

| File | Isi |
|---|---|
| `evaluasi_robust.ipynb` | 4 kandidat (LightGBM, MicroTransformer, ResNet1D, BiGRU) × 5 ulangan × 5 fold pasien, CI bootstrap, perbandingan berpasangan, bias kelas, per pasien |
| `sihedaf_cv.py` | Fold berulang, training GPU, bias cross-fit, metrik per pasien, bootstrap |
| `eval_versi_lama.py` | Latih ulang resep v1 pada fold yang sama; uji di data lab, data detektor, dan front-end lama |
| `laporan/` | `laporan_evaluasi_robust.pdf` (11 halaman) + `build_report.py` |

## Hasil utama (data uji detektor 50 Hz, rata-rata 5 ulangan, CI 95% bootstrap pasien)

| Sistem | Macro-F1 | AFL F1 | AFL precision | AFL recall |
|---|---|---|---|---|
| v1 (diukur ulang) | 0,744 [0,64–0,81] | 0,399 | 0,275 | 0,728 |
| v1 + front-end lama | 0,418 [0,36–0,48] | 0,051 | 0,027 | 0,410 |
| **Produksi v2 + bias grid lebar** | **0,832** [0,67–0,92] | 0,647 [0,17–0,89] | 0,715 | 0,592 |
| BiGRU + bias (setara produksi) | 0,829 [0,68–0,92] | 0,621 | 0,692 | 0,564 |

Temuan:
- Sistem sekarang lebih baik dari v1 secara konsisten (P(Δ>0) = 0,98). Sekitar 80% kenaikan berasal dari bias
  kelas. Harganya: recall AFL turun.
- Mengganti arsitektur berbasis RR tidak membantu. Batas yang tersisa adalah 7 pasien AFL; 07910 dan 08215
  gagal di semua model.
- Bias AFL optimal −4…−5,5. Bias produksi `[0.0, 2.0, -3.0]` terpotong grid lama.

## Lingkungan hasil

Dijalankan 12–13 Oktober 2026 di WSL2 (Debian), PC desktop dengan GPU NVIDIA 8 GB: evaluasi kandidat ±1 jam 40 menit, versi lama ±25 menit. Di lingkungan lain angka bisa sedikit berbeda (GPU non-deterministik); untuk perbandingan, pakai cache OOF yang sama (lihat AGENTS.md root).

## Menjalankan (dari root repo)

```bash
python tools/run_notebook.py v3_evaluasi_robust/evaluasi_robust.ipynb   # ±1 j 40 m pertama kali; ±5 m dari cache
cd v3_evaluasi_robust && python eval_versi_lama.py                      # ±25 m, bisa dilanjutkan
python v3_evaluasi_robust/laporan/build_report.py --extract             # angka + grafik dari cache
python v3_evaluasi_robust/laporan/build_report.py                       # PDF (Python + ReportLab)
```

Prasyarat: `../cache/dataset_v2_detector.npz` dari notebook v2, dan `../data/afdb` untuk `eval_versi_lama.py`.

## Langkah berikutnya (v4)

Smoothing antar-window, ensemble MicroTransformer + BiGRU, window AFL rapat khusus training, bobot kelas
lebih lunak. Semua dibandingkan dengan protokol ini (lihat `AGENTS.md`).
