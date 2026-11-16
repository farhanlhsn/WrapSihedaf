# Rancangan v13: model fondasi EKG ECGFounder 1-lead pada window SIHEDAF

Tanggal: 6 November 2026. Status: **dikunci sebelum metrik dihitung.** Yang sudah dilihat sebelum dikunci: uji asap 600 window
pasien 04936 (kecepatan dan arah skor zero-shot). Bagian 2–5 tidak diubah setelah hasil keluar; penyimpangan di bagian 7.

## 1. Latar

- Model RR punya plafon (recall AFL ±0,6; alarm sinus takikardia ±100% di atas 120 bpm, juga pada EKG latihan fisik asli, v3.3).
  Cabang EKG yang dilatih sendiri gagal: overfit 23 pasien (v6) atau pergeseran domain (v9, v10).
- ECGFounder (PKUDigitalHealth, MIT): Net1D 76 juta parameter, pralatih 10,8 juta EKG / 1,8 juta pasien (HEEDB), 150 label
  termasuk sinus takikardia dan atrial flutter; checkpoint 1-lead. Terbaik di benchmark FOUND-AF (arXiv 2608.03597) untuk AF di
  AFDB, CinC2017, CPSC2021, LTAFDB, tetapi AF dan AFL digabung dan window uji tumpang tindih.
- Ini informasi baru (pengetahuan pralatih dari jutaan pasien), bukan resep RR baru.

## 2. Data dan jalur sinyal

- AFDB kanal 0, window detektor yang sama dengan semua versi (165.978 window). Dua jalur: **50hz** (resample ke 50 Hz dengan
  `hrv_core.resample_signal`, seperti perangkat) dan **250hz** (kontrol, sinyal asli). Keduanya → 500 Hz → filter dan z-score
  ECGFounder.
- Uji eksternal (tidak dipakai untuk memilih apa pun): EKG latihan fisik v3.3 (EPFL, Wrist, EPHNOGRAM; mode perangkat), PTB-XL
  1.0.3 lead I 100 Hz → 50 Hz (sinus termasuk STACH, AFIB, AFLT), Icentia11k (window v10, akademik saja).

## 3. Cara pakai (tetap)

- **zero-shot:** logit head `[maks(NSR, SR, ST), AF, AFL]` → softmax → bias cross-fit.
- **probe:** fitur 1.024 dimensi beku → regresi logistik multinomial (C = 1, standardisasi dan bobot kelas lunak dari data latih
  fold) per (ulangan, fold) resmi → bias cross-fit.
- **fusi:** rata-rata log-probabilitas probe dan OOF v3.1 → bias cross-fit.
- Untuk data eksternal: probe dan fusi dilatih di seluruh AFDB; bias = median bias cross-fit.

## 4. Hipotesis dan kriteria

- **H1 (AFDB, protokol penuh):** Macro-F1 probe-50hz dan fusi-50hz vs v3.1 berpasangan; vonis protokol. Juga recall AFL per pasien
  dan tanpa 04936.
- **H2 (sinus takikardia asli):** alarm AFIB/AFL pada window latihan fisik ≥ 120 bpm, dibandingkan dengan paket v3.1 pada window
  yang sama, CI bootstrap subjek. Bermakna bila alarm turun paling sedikit separuh.
- **H3 (biaya 50 Hz):** selisih Macro-F1 jalur 50hz − 250hz untuk zero-shot dan probe.
- **H4 (eksternal):** PTB-XL dan Icentia: Macro-F1 3 kelas dan recall AFL, berdampingan dengan v3.1 (pelajaran v9: laporkan
  AFDB dan eksternal bersama). Tanda bahaya: bias menyentuh dua batas grid.
- Protokol butir 9 sintetis tidak bisa dipakai untuk model bentuk gelombang; diganti H2 (EKG asli).

## 5. Yang tidak dilakukan

- Tidak fine-tune seluruh jaringan (risiko overfit 23 pasien, pelajaran v6). Tidak memilih hiperparameter dari fold uji atau data
  eksternal. Tidak mengubah paket produksi.

## 6. Keluaran

`v13_ecgfounder/`: `ecgfounder.py`, `ecgfounder.ipynb`, `README.md`, `laporan/`. Cache `../cache/v13/`. Bobot dan kode ECGFounder di
`../data/ecgfounder/` (tidak di-commit; diunduh dengan `tools/unduh_berkas.py`, SHA256 dari API Hugging Face).

## 7. Penyimpangan dari rancangan

(belum ada)
