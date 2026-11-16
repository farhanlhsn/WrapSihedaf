# xgboost — XGBoost pada fitur HRV

**Model.** Gradient boosting pohon keputusan (XGBoost, `hist`) pada **16 fitur HRV** per window: rata-rata/median/min/maks
IBI, SDNN, RMSSD, pNN50, CV, SD1, SD2, SD1/SD2, TPR, SampEn, COSEn, detak rata-rata, dan jumlah detak. Fiturnya sama
dengan LightGBM v1/v3 (`dataset_v2_detector.npz["F"]`), dihitung oleh `v2_revisi/hrv_core.hrv_features`.

**Beda dengan LightGBM v1** (yang tidak boleh dipakai lagi): pustaka dan algoritma pertumbuhan pohon berbeda. Bobot
kelasnya lunak (bobot sampel = akar bobot seimbang), bukan `class_weight="balanced"` penuh. Sebagai patokan, LightGBM v3
mendapat Macro-F1 0,770 tanpa bias; bias malah menurunkannya ke 0,742.

**Setelan bawaan** (`model.py`): 400 pohon, `learning_rate` 0,05, `max_depth` 6, `tree_method` "hist",
`perangkat` "cpu" (ubah ke "cuda" untuk GPU), `weight_power` 0,5 (bobot sampel = bobot kelas seimbang ** 0,5; 1,0 = bobot
penuh).

**Waktu** (CPU 12 core): latih 2,3 s per fold (bawaan), 3,0 s (`weight_power` 1,0), 4,4 s (smote), 0,4 s (under). Satu setelan
25 fold ±1–2,5 menit, ditambah evaluasi ±2,5 menit.

**Catatan.** Folder ini bernama `xgboost` tetapi tidak punya `__init__.py`, jadi `import xgboost` tetap memuat pustaka
aslinya. Jangan menambah `__init__.py` di sini.

**Ide eksperimen** (ganti `NAMA_SETELAN` tiap kali): `{"max_depth": 4}`, `{"n_estimators": 800, "learning_rate": 0.025}`,
`{"perangkat": "cuda"}`.

**Balancing** (arahan pembimbing, lihat `../README.md` "Opsi balancing"): `{"balancing": "smote"}` dengan
`NAMA_SETELAN = "smote"`, dan `{"balancing": "under"}` dengan `NAMA_SETELAN = "under"`. Latih juga `bawaan`, lalu bandingkan di
`gabungan.ipynb`. Dengan SMOTE, data latih menjadi ±240 ribu window per fold (3 × jumlah Normal), jadi training lebih lama.

## Hasil: balancing vs pembobotan (23 Oktober 2026)

Empat setelan, masing-masing 5 ulangan × 5 fold pasien (protokol v3). Hasilnya pooled OOF rata-rata ulangan, dengan CI 95%
bootstrap pasien 2.000×. `+ bias` = bias kelas cross-fit (grid −7…+3). Pertanyaannya: apakah balancing (arahan pembimbing)
lebih baik daripada pembobotan kelas? `bobot penuh` adalah kontrol: kelas minoritas diberi bobot lebih besar tanpa membuang
atau membuat data.

| Setelan | `SETELAN` | Window latih per kelas (fold r0 k0) |
|---|---|---|
| bawaan | `{}` (bobot lunak, `weight_power` 0,5) | 82.012 / 46.765 / 1.143 |
| bobot penuh | `{"weight_power": 1.0}` | 82.012 / 46.765 / 1.143 |
| smote | `{"balancing": "smote"}` | 82.012 × 3 |
| under | `{"balancing": "under"}` | 1.143 × 3 (458–1.143 antar fold) |

**Metrik (+ bias)**

| Sistem | Macro-F1 | AFL F1 | AFL P | AFL R | AFIB F1 | Normal R |
|---|---|---|---|---|---|---|
| v1 (argmax) | 0,744 [0,64–0,81] | 0,399 [0,09–0,59] | 0,275 | 0,728 [0,37–0,98] | 0,898 | 0,903 |
| v3.1 (referensi) | **0,832** [0,67–0,92] | **0,647** [0,17–0,89] | 0,715 | 0,592 [0,13–0,91] | 0,909 | 0,924 |
| XGBoost bawaan | 0,730 [0,62–0,78] | 0,363 [0,04–0,51] | 0,394 | 0,342 [0,05–0,53] | 0,895 | 0,931 |
| XGBoost bobot penuh | 0,744 [0,63–0,79] | 0,401 [0,08–0,53] | 0,449 | 0,364 [0,07–0,54] | 0,897 | 0,929 |
| XGBoost smote | 0,743 [0,62–0,79] | 0,396 [0,05–0,54] | 0,430 | 0,370 [0,04–0,57] | 0,899 | 0,926 |
| XGBoost under | 0,794 [0,65–0,84] | 0,554 [0,13–0,69] | 0,626 | 0,508 [0,11–0,67] | 0,899 | 0,904 |

Tanpa bias (argmax), Macro-F1: bawaan 0,774, bobot penuh 0,769, smote 0,755, under 0,701. Bias cross-fit merugikan XGBoost
kecuali `under` (pola yang sama dengan LightGBM v3). Bias AFL bawaan juga melompat antar fold (−6 … +1,75).

**Perbandingan berpasangan** (Δ Macro-F1 [CI 95%], P(Δ > 0); vonis P ≥ 0,95 lebih baik, ≤ 0,05 lebih buruk)

| Kandidat vs referensi | Δ Macro-F1 | P | Vonis | Catatan |
|---|---|---|---|---|
| bobot penuh vs bawaan | +0,013 [−0,00…+0,03] | 0,97 | lebih baik (kecil) | AFL P +0,055 |
| smote vs bawaan | +0,013 [+0,00…+0,02] | 0,98 | lebih baik (kecil) | Normal R −0,005 (P = 0,00) |
| **smote vs bobot penuh** | −0,001 [−0,02…+0,01] | 0,42 | tidak terbedakan | |
| under vs bawaan | +0,063 [+0,00…+0,18] | 0,98 | lebih baik | AFL F1 +0,19 |
| **under vs bobot penuh** | +0,050 [+0,00…+0,15] | 0,98 | lebih baik | AFL F1 +0,15 (P = 0,99); Normal R −0,026 (P = 0,00) |
| under vs v3.1 | −0,038 [−0,08…−0,01] | 0,00 | lebih buruk | |
| bobot penuh / smote / bawaan vs v3.1 | −0,088 / −0,089 / −0,101 | ≤ 0,004 | lebih buruk | |
| under vs v1 | +0,050 [−0,00…+0,09] | 0,97 | lebih baik | |
| bobot penuh / smote / bawaan vs v1 | +0,000 / −0,001 / −0,013 | 0,36–0,48 | tidak terbedakan | |

**Recall AFL per pasien (+ bias) dan AFL palsu**

| Sistem | 04043 | 04908 | 04936 | 06426 | 07910 | 08215 | 08378 | AFL palsu / ulangan (di pasien tanpa AFL) |
|---|---|---|---|---|---|---|---|---|
| v3.1 | 0,40 | 0,92 | 0,51 | 0,17 | 0,00 | 0,00 | **0,96** | 274 (25) |
| bawaan | 0,20 | 1,00 | 0,49 | 0,08 | 0,04 | 0,00 | 0,05 | 600 (252) |
| bobot penuh | 0,40 | 0,98 | 0,49 | 0,05 | 0,01 | 0,00 | 0,13 | 544 (193) |
| smote | 0,40 | 0,99 | 0,53 | 0,02 | 0,01 | 0,00 | 0,07 | 586 (192) |
| under | 0,20 | 0,89 | 0,50 | 0,11 | 0,00 | 0,00 | 0,67 | 350 (74) |

**Uji stres takikardia (protokol butir 9).** Ambang bpm di mana model berhenti menebak Normal, per jenis variasi detak
(tanpa variasi / seragam ±10 ms / Gauss 10 ms / Gauss 20 ms). Pembanding: kontrol v7 per fold 120/120/120/115, paket v3.1 115.

| Setelan | Ambang (bpm) | % Normal asli ≥ 120 bpm ditebak AFIB/AFL (100 window) | % Normal 0–100 bpm ditebak AFIB/AFL |
|---|---|---|---|
| v3.1 | 115 (paket) | 95,2 | 6,7 |
| bawaan | 125 / 120 / 115 / 115 | 85,0 | 6,0 |
| bobot penuh | 125 / 120 / 120 / 115 | 82,8 | 6,2 |
| smote | 130 / 120 / 115 / 115 | 86,8 | 6,5 |
| under | 125 / 115 / 115 / 115 | 94,8 | 8,7 |

**Tanpa pasien 04936** (protokol butir 10, bias dihitung ulang di subset, 22 pasien, 6 pasien AFL), Macro-F1: v3.1 0,859,
under 0,779, bobot penuh 0,661, bawaan 0,647. v3.1 tetap lebih baik dari under: Δ +0,080 [−0,00…+0,10], P = 0,97.

**Kesimpulan**
- **SMOTE tidak memberi apa pun di luar efek pembobotan.** smote dan bobot penuh tidak terbedakan (P = 0,42). Keduanya hanya
  sedikit lebih baik dari bobot lunak (+0,013).
- **Undersampling punya efek tambahan pada XGBoost** (+0,050 vs bobot penuh, P = 0,98). Hampir seluruhnya berasal dari satu pasien
  AFL 2:1 (08378: 0,13 → 0,67), ditambah AFL palsu yang turun (544 → 350).
  - Harganya: Normal recall turun (0,929 → 0,904), dan alarm pada ritme normal naik di semua pita detak (≥ 120 bpm: 83% → 95%).
  - Penyebabnya **belum diketahui**. Dugaan: 400 pohon dengan kedalaman 6 berperilaku berbeda pada ±3 ribu window dibanding ±130
    ribu window. Belum diuji, dan hiperparameter tidak di-tuning di setelan mana pun.
- **Tidak ada setelan XGBoost yang mendekati v3.1.** Semuanya lebih buruk (P ≤ 0,004), juga tanpa 04936. Recall 08378 di semua
  setelan jauh di bawah v3.1 (0,96).
- **Yang tidak bisa disimpulkan.** Hasil ini untuk XGBoost pada 16 fitur HRV saja. Untuk model urutan (DL), undersampling hanya
  menyisakan 458–1.143 window per kelas, dan efeknya belum diuji.
- **Batas grid.** Bias AFL under menempel ke batas: median −6, 6% nilai di −7. Ini wajar, karena proporsi AFL berubah dari ±0,7%
  menjadi 33%, tetapi grid −7…+3 mungkin sedikit memotong optimum.

**Lingkungan.** PC desktop, CPU 12 core (GPU tidak dipakai). Venv sementara: Python 3.13, xgboost 3.4.1,
imbalanced-learn 0.14.2, scikit-learn 1.9.1, numpy 2.5.4. Kode: commit `b18d6bc` + perubahan v11 yang belum di-commit.
Dijalankan lewat `bersama.runner` (sama dengan sel training notebook), bukan lewat `jalankan.ipynb` (notebook itu dijalankan
3 November 2026 untuk evaluasi saja; angkanya identik, sel Kesimpulan terisi). Hasil di
`cache/v11/xgboost/{bawaan,bobot_penuh,smote,under}/`; perbandingan di `cache/v11/xgboost/banding_{balancing,bobot_penuh}.json`.
