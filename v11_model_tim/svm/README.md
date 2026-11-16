# svm — SVM kernel RBF (aproksimasi Nystroem) pada fitur HRV

**Model.** Pipeline: standardisasi 16 fitur HRV → aproksimasi kernel RBF dengan `Nystroem` (2.000 komponen) → SVM linear
(`LinearSVC`, *squared hinge loss*, bawaan scikit-learn, diselesaikan dalam bentuk primal yang cepat untuk data sebesar
ini), dengan bobot sampel lunak. Fiturnya sama dengan XGBoost/LightGBM.

**Kenapa aproksimasi.** SVM RBF eksak (`sklearn.svm.SVC`) waktunya tumbuh kira-kira kuadratik terhadap jumlah data;
dengan ±130 ribu window latih × 25 fold, itu tidak sanggup. Nystroem memetakan fitur ke ruang yang meniru kernel RBF,
lalu SVM linear memisahkannya. Hasilnya mendekati SVM RBF, tetapi tidak identik; tulis "SVM RBF (aproksimasi Nystroem)"
di laporan.

**Probabilitas.** SVM tidak menghasilkan probabilitas. Skor keputusan tiap kelas diubah ke pseudo-probabilitas dengan
softmax. Skalanya tidak terkalibrasi, tetapi bias kelas cross-fit (evaluasi standar) menyesuaikannya. Lihat hasil
"+ bias", bukan argmax.

**Setelan bawaan** (`model.py`): `komponen` 2.000, `gamma` None (= 1/jumlah fitur), `C` 1,0, `max_iter` 5.000. Kalau
LinearSVC belum konvergen, notebook mencetak peringatan dan `info.konvergen` = false.

**RAM.** Puncaknya **±7–8 GB per fold** dengan 2.000 komponen. Rinciannya: matriks Nystroem (±130 ribu × 2.000),
salinan float64-nya, dan salinan format internal liblinear (16 byte per nilai). Aman di Kaggle (±30 GB). Di Colab gratis
(±12 GB) masih cukup, tetapi tutup notebook lain. RAM habis = sesi mati dan hasil yang belum disimpan hilang (lihat
`HASIL_DI` di README tim).

**Ukuran hasil.** Bobot tiap fold menyimpan matriks Nystroem (±16 MB), jadi zip 25 fold ±375 MB (folder hasil 404 MB).

**Waktu.** 218 s per fold di Colab CPU High-RAM (8 core), dengan 5 proses paralel (satu per ulangan) berbagi core itu.
25 fold ±18 menit. Evaluasi ±2,5 menit. Di PC dengan RAM 14–15 GB, satu proses dengan batas memori 7 GB dimatikan OOM:
latih di Kaggle atau Colab High-RAM.

**Ide eksperimen** (ganti `NAMA_SETELAN` tiap kali): `{"C": 0.1}` / `{"C": 10.0}`, `{"gamma": 0.02}`,
`{"komponen": 1000}`. `{"komponen": 4000}` butuh RAM ±15 GB: **hanya Kaggle**.

**Balancing** (arahan pembimbing, lihat `../README.md` "Opsi balancing"): `{"balancing": "under"}` (±3 ribu window latih,
cepat) dan `{"balancing": "smote"}`. SMOTE menaikkan data latih ke ±240 ribu window per fold, kira-kira 1,8× bawaan, jadi
RAM-nya diperkirakan ±13–15 GB dengan 2.000 komponen: **hanya Kaggle**, atau pakai `{"balancing": "smote", "komponen": 1000}`.
Latih juga `bawaan`, lalu bandingkan di `gabungan.ipynb`.

## Hasil (3 November 2026)

Setelan `bawaan`, 5 ulangan × 5 fold pasien (protokol v3), seed `2026 + 100·r + k`. Pembanding v1 dan v3.1 memakai ulangan
dan sampel bootstrap yang sama. Angka dari `jalankan.ipynb` (output tersimpan) dan `cache/v11/svm/bawaan/ringkasan.json`.
`+ bias` = bias kelas cross-fit (grid −7…+3), CI 95% bootstrap pasien 2.000×.

| Sistem | Macro-F1 | AFL F1 | AFL P | AFL R | AFIB F1 | Normal R |
|---|---|---|---|---|---|---|
| v1 (argmax) | 0,744 [0,64–0,81] | 0,399 [0,09–0,59] | 0,275 | 0,728 [0,37–0,98] | 0,898 | 0,903 [0,82–0,95] |
| v3.1 (referensi) | **0,832** [0,67–0,92] | **0,647** [0,17–0,89] | 0,715 | 0,592 [0,13–0,91] | 0,909 | 0,924 [0,84–0,97] |
| SVM argmax (tanpa bias) | 0,734 [0,62–0,90] | 0,360 [0,03–0,83] | 0,256 | 0,685 [0,21–0,97] | 0,912 | 0,904 [0,82–0,96] |
| **SVM + bias** (protokol) | 0,773 [0,64–0,92] | 0,487 [0,09–0,90] | 0,441 | 0,601 [0,14–0,93] | 0,901 | 0,929 [0,86–0,97] |

Bias [AFIB, AFL] median −0,5 / −1,25 (rentang −0,75…+0,5 / −1,75…−0,25). Tidak ada yang menyentuh batas grid. Bias AFL-nya
yang paling kecil di antara model v11 bawaan (BiGRU −2, TCN −3, InceptionTime −4).

**Perbandingan berpasangan** (Δ [CI 95%], P(Δ > 0); vonis P ≥ 0,95 lebih baik, ≤ 0,05 lebih buruk)

| Kandidat vs referensi | Δ Macro-F1 | P | Vonis | Catatan |
|---|---|---|---|---|
| SVM + bias vs v3.1 | −0,059 [−0,12…+0,00] | 0,072 | tidak terbedakan | AFL P −0,275 (P = 0,007, lebih buruk); AFL F1 −0,161 (P = 0,09); AFL R +0,009 (P = 1,00); Normal R +0,005 (P = 0,71) |
| SVM + bias vs v1 | +0,029 [−0,01…+0,14] | 0,907 | tidak terbedakan | AFL P +0,165 (P = 0,99); AFL R −0,127 (P = 0,00); Normal R +0,025 (P = 0,99) |

**Recall AFL per pasien dan AFL palsu**

| Sistem | 04043 | 04908 | 04936 | 06426 | 07910 | 08215 | 08378 | AFL palsu / ulangan (di pasien tanpa AFL) |
|---|---|---|---|---|---|---|---|---|
| v3.1 | 0,40 | 0,92 | 0,51 | 0,17 | 0,00 | 0,00 | 0,96 | 274 (25) |
| SVM argmax | 0,80 | 1,00 | 0,63 | 0,47 | 0,03 | 0,00 | 1,00 | 2.801 (156) |
| SVM + bias | 0,80 | 0,92 | 0,51 | 0,20 | 0,00 | 0,00 | 0,98 | 1.223 (37) |

Asal AFL palsu SVM + bias: 958 window dari ritme Normal dan 265 dari AFIB (v3.1: 55 dan 219).

**Uji stres takikardia (protokol butir 9).** Ambang = detak terendah di mana < 50% window sintetis ditebak Normal, per jenis
variasi (tanpa variasi / seragam ±10 ms / Gauss 10 ms / Gauss 20 ms); makin tinggi makin baik. Pembanding: kontrol v7 per fold
120/120/120/115, paket v3.1 115. Kolom pita = % window Normal asli yang ditebak AFIB/AFL (OOF).

| Sistem | Ambang (bpm) | 0–100 bpm | 100–110 bpm | 110–120 bpm | ≥ 120 bpm (100 window) |
|---|---|---|---|---|---|
| v3.1 | 115 (paket) | 6,7% | 12,7% | 59,0% | 95,2% |
| SVM + bias | 140 / 125 / 125 / 120 | 5,5% | 19,8% | 50,7% | 78,4% |

**Kesimpulan**
- **SVM bawaan tidak terbedakan dari v3.1** (Δ Macro-F1 −0,059, P = 0,07) **maupun dari v1** (+0,029, P = 0,91). Ia bukan
  kandidat pengganti paket produksi.
- **Recall AFL setara v3.1** (0,601 vs 0,592), termasuk AFL 2:1 pasien 08378 (0,98). Bias AFL-nya kecil, jadi 08378 tidak
  terhapus seperti pada TCN bawaan.
- **Harganya presisi AFL**: 0,441 vs 0,715 (P = 0,007). AFL palsu 1.223 window per ulangan, 4,5× v3.1, dan 78% di antaranya
  berasal dari ritme Normal (v3.1: 20%). Fitur HRV ringkasan tampaknya sulit membedakan AFL 2:1 dari ritme Normal yang teratur
  dan cepat. Ini dugaan, belum diuji.
- **Takikardia: trade-off.** Ambang untuk ritme tanpa variasi 140 bpm (tertinggi di v11), tetapi turun ke 120–125 bpm begitu
  detak sedikit bervariasi. Alarm pada Normal ≥ 120 bpm 78,4% (v3.1 95,2%), sedangkan di pita 100–110 bpm justru naik
  (19,8% vs 12,7%).
- **Yang tidak bisa disimpulkan.** Hanya setelan bawaan; `C`, `gamma`, dan `komponen` belum di-tuning. Bootstrap bersarang dan
  sensitivitas tanpa 04936 tidak dijalankan karena SVM bukan kandidat. Training (scikit-learn 1.6.1) dan evaluasi
  (scikit-learn 1.9.1) memakai versi pustaka berbeda. Evaluasi hanya membaca prediksi yang tersimpan, jadi angkanya tidak
  terpengaruh.

**Lingkungan.**
- **Training:** Google Colab lewat Colab CLI, runtime CPU High-RAM (8 core, RAM 51 GB, tanpa GPU), 3 November 2026.
  - Pustaka: Python 3.13.18, scikit-learn 1.6.1, numpy 2.4.6, scipy 1.16.3.
  - Kode: commit `b18d6bc` + perubahan v11 yang belum di-commit, diunggah sebagai arsip.
  - 5 proses `runner.jalankan(..., ulangan=[r])` berjalan paralel dengan `OMP_NUM_THREADS=1`.
- **Evaluasi:** di PC.
  - Folder hasil diunduh ke `cache/v11/svm/bawaan/`, lalu `jalankan.ipynb` dijalankan lewat `tools/run_notebook.py` (±3 menit).
  - Runner melewati 25 fold yang sudah ada, jadi notebook hanya mengevaluasi. Karena itu, `lingkungan` di `ringkasan.json`
    mencatat PC (scikit-learn 1.9.1), bukan Colab.
  - Bobot `.joblib` dibuat dengan scikit-learn 1.6.1. Muat dengan versi yang sama; versi lain bisa memberi peringatan atau gagal.
