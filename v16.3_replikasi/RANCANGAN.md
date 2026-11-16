# Rancangan v16.3: replikasi detektor AF 2 kelas D-1 (ECGFounder, ambang −1) di tiga dataset yang belum pernah dipakai

Tanggal: 6 November 2026. Status: **dikunci sebelum sinyal ketiga dataset dibaca oleh kode apa pun** (sha256 dan waktu di
`logs/v16.3_praregistrasi.log`). Yang sudah dilihat sebelum dikunci: daftar berkas, header, berkas label CinC 2017, dan anotasi ritme
MIT-BIH/LTAFDB (jumlah jam per label, posisi label pertama). Bagian 2–6 tidak diubah setelah hasil keluar; penyimpangan di bagian 8.

## 1. Latar

v16.2 menguji D-1 (AF bila `d1 = log((pAF + pAFL) / pN) > −1`, skor zero-shot ECGFounder 1-lead) di CPSC 2021 dan lolos: Macro-F1
2 kelas lebih baik dari paket v3.1 dengan sensitivitas setara. Dua kelemahan tersisa: sensitivitas lebih rendah di data wearable
(Icentia11k, data yang sudah dipakai) dan alarm 25% pada window non-AF ≥ 120 bpm di CPSC (label hanya "bukan AF"). v16.3 mengulang uji
yang sama, tanpa penyetelan apa pun, di tiga dataset yang belum pernah dipakai SIHEDAF.

## 2. Data (tetap)

| Dataset | Isi | Kanal | Label 2 kelas |
|---|---|---|---|
| **CinC 2017** (PhysioNet `challenge-2017` 1.0.0, `training/`) | 8.528 rekaman EKG genggam satu lead (AliveCor), 300 Hz, 9–61 s | satu-satunya | per rekaman dari `REFERENCE-v3.csv` (versi akhir; salinan di `training/` labelnya sama): AF = `A`; non-AF = `N` dan `O` (ritme lain); `~` (bising) dibuang |
| **MIT-BIH Arrhythmia** (`mitdb` 1.0.0) | 48 rekaman 30 menit, 360 Hz, 47 subjek (201 dan 202 subjek yang sama) | kanal 0 (umumnya MLII) | label ritme anotasi |
| **LTAFDB** (`ltafdb` 1.0.0) | 84 rekaman ±24 jam, 128 Hz, subjek AF paroksismal/persisten | kanal 0 | label ritme anotasi |

- Label ritme MIT-BIH dan LTAFDB mengikuti konvensi WFDB/MIT (sama dengan AFDB): label `(XXX` menandai awal ritme yang berlaku sampai
  label ritme berikutnya. Window sebelum label ritme pertama tidak berlabel dan dibuang. AF = `(AFIB` atau `(AFL`; non-AF = semua label
  ritme lain kecuali `(P` (ritme pacu, dibuang seperti PACE di PTB-XL).
- Window perangkat: `hrv_core.signal_to_windows(x, fs)` (resample 50 Hz, detektor R produksi, 10 s, langkah 5 s); berlabel bila seluruh
  10 s di dalam satu segmen ritme (CinC: semua window rekaman). EKG untuk ECGFounder = 500 sampel 50 Hz yang sama.
- **Himpunan uji = window berlabel yang lolos cek kualitas detektor** (punya prediksi paket v3.1).
- Unit bootstrap: subjek (MIT-BIH: 201 dan 202 digabung; LTAFDB: rekaman = subjek); CinC 2017 tidak punya ID pasien → rekaman.

## 3. Sistem (tetap, sama dengan v16.2)

D-1 (primer, `d1 > −1`), **paket v3.1 (referensi)**, D-0 (`d1 > 0`), paket v1. AFIB + AFL digabung. Tidak ada yang dilatih atau dipilih.

## 4. Hipotesis dan kriteria

- **Per dataset:** H1 Macro-F1 2 kelas D-1 bukan "lebih buruk" dari v3.1 (P(Δ>0) > 0,05); H2 sensitivitas D-1 bukan "lebih buruk"
  (P > 0,05). Bootstrap unit berpasangan 2.000×, seed 2026 (`v16.1.banding2_pasien`).
- **Vonis replikasi:** berhasil bila H1 dan H2 lolos di ketiga dataset; sebagian bila di satu atau dua; gagal bila tidak di satu pun.
- **Deskriptif:** spesifisitas dan PPV (vonis protokol); alarm per jenis ritme non-AF (MIT-BIH, LTAFDB: per label; CinC: `N` vs `O`);
  alarm non-AF per pita detak (pita v11.1); sensitivitas per subjek (sebutkan yang turun > 0,20 vs v3.1); episode AF gaya EC57 (MIT-BIH,
  LTAFDB; langkah 5 s); D-0 dan v1 sebagai konteks.

## 5. Keputusan yang ditetapkan sebelum hasil

- Bila replikasi berhasil: D-1 tetap kandidat detektor AF 2 kelas untuk uji perangkat, dengan bukti dari empat dataset independen
  (CPSC 2021 + tiga ini). Bila sebagian atau gagal: tulis dataset dan kriteria yang gagal; status kandidat ditinjau ulang di README
  (bukan dengan mengubah ambang).
- Bukan pengganti paket v3.1 dalam keadaan apa pun.

## 6. Yang tidak dilakukan

- Tidak memilih ambang, lead, atau aturan label dari hasil. Tidak melatih apa pun. Tidak mengubah paket produksi.

## 7. Keluaran

`v16.3_replikasi/`: `replikasi.py`, `replikasi.ipynb`, `README.md`, `laporan/`. Cache `../cache/v16.3/` (hasil per rekaman, bisa
dilanjutkan).

## 8. Penyimpangan dari rancangan

(belum ada)
