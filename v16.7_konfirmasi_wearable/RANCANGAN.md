# Rancangan v16.7: uji konfirmasi S1 (D-1 + episode AF ≥ 30 s) di wearable kontinu (Icentia11k) dan uji stres derau (nstdb)

Tanggal: 10 November 2026. Status: **dikunci sebelum sinyal segmen Icentia yang dipilih dan sinyal nstdb dibaca** (sha256 dan waktu di
`logs/v16.7_praregistrasi.log`). Bagian 2–6 tidak diubah setelah hasil keluar; penyimpangan di bagian 8.

Yang sudah dilihat sebelum dikunci:
- semua hasil v16.2–v16.6. S1 memenuhi target di SHDB-AF, afpdb, incartdb, tetapi hanya sebagai **analisis sekunder** v16.6;
- D-1 per window di Icentia (window acak v10, 2.420 pasien): sensitivitas lebih rendah dari v3.1 (0,843 vs ±0,87, v16.1/v16.3);
- hasil pemindaian anotasi Icentia v10 (`cache/v10/icentia_scan.npz`: durasi dan episode per ritme per segmen, tanpa sinyal). Semua
  1.225 pasien ber-AF ada di seleksi v10, tetapi v10 hanya memproses 1–6 segmen per pasien. 8.575 pasien tanpa AF tidak pernah dipakai;
- header nstdb (tanpa sinyal).

## 1. Pertanyaan

S1 = D-1 (ECGFounder 1-lead zero-shot, d1 > −1) + aturan episode AF ≥ 30 s (v16.5, tanpa celah) adalah kandidat terbaik untuk deteksi AF
2 kelas. Buktinya di data Holter. Kelemahan D-1 yang tercatat ada di EKG non-Holter/wearable. Apakah S1 memenuhi target produk di
**rekaman wearable kontinu** (patch satu lead), dan seberapa tahan terhadap **derau gerak/elektroda**?

## 2. Sistem (tetap)

| Sistem | Keputusan |
|---|---|
| **S1 (primer)** | d1 > −1, lalu aturan v16.5 (run window AF berjarak ≤ langkah + 0,5 s, rentang ≥ 30 s) |
| R (referensi) | paket v3.1 (argmax ≠ Normal), aturan yang sama |
| S3 (deskriptif saja) | (D-1 **atau** v3.1), aturan yang sama: apakah menggabungkan keduanya menutup kekurangan sensitivitas? Tidak dipakai untuk keputusan |

## 3. Data (tetap)

**A. Icentia11k** (`icentia11k-continuous-ecg` 1.0, CC BY-NC-SA, hanya evaluasi; patch CardioSTAT satu lead, 250 Hz, segmen ±70 menit).
Seleksi dari hasil pemindaian anotasi, seed 2026 + 167:
- **Kelompok AF:** 300 pasien acak dari pasien yang punya segmen dengan episode `(AFIB` atau `(AFL` terpanjang ≥ 30 s. Per pasien satu
  segmen acak dari segmen seperti itu yang **tidak termasuk** seleksi v10.
- **Kelompok non-AF:** 300 pasien acak dari pasien yang tidak punya `(AFIB`/`(AFL` di segmen mana pun dan **tidak termasuk** seleksi v10.
  Per pasien satu segmen acak dengan durasi `(N` ≥ 3.000 s.
- Label: `(AFIB`, `(AFL` = AF; `(N` = non-AF; berlaku sampai label ritme berikutnya atau `)`. Waktu tanpa label, label lain, dan
  window yang melintasi batas dibuang. Unit bootstrap = pasien.

**B. nstdb** (MIT-BIH Noise Stress Test, 1.0.0): 12 rekaman 30 menit (118/119 MIT-BIH + derau elektroda/gerak, SNR 24, 18, 12, 6, 0,
−6 dB; derau dalam blok 2 menit mulai menit ke-5), kanal 0, 360 Hz. Semua non-AF. Pembanding bersih: 118 dan 119 MIT-BIH (sinyal sama
tanpa derau), diproses ulang dengan mesin yang sama.

Rantai prediksi untuk keduanya = v16.4 `uji_baru.proses_rekaman` (50 Hz, detektor perangkat, cek kualitas, ECGFounder, paket v3.1).

## 4. Kriteria (dikunci)

- **Target produk di Icentia (S1):** (a) Se episode AF (referensi ≥ 30 s) ≥ **0,90**; (b) alarm palsu per jam ≤ **1,0** pada kelompok
  non-AF; titik.
- **H1:** alarm palsu per jam kelompok non-AF S1 < R, bootstrap pasien berpasangan 2.000×, seed 2026, P(Δ < 0) ≥ 0,95.
- **H2:** Se episode S1 bukan lebih buruk dari R: P(Δ > 0) > 0,05.
- **H3 (nstdb):** alarm palsu per jam S1 ≤ 1,0 pada SNR ≥ 12 dB (gabungan 118 + 119 per SNR, titik). SNR 6, 0, −6 dB: deskriptif.
- **Sekunder (deskriptif):** alarm palsu per jam di waktu non-AF kelompok AF; Se/+P durasi; alarm non-AF per pita detak; S3.
- **Keputusan:** target (a), (b), H1, H2 lolos → S1 **terkonfirmasi** sebagai kandidat detektor AF 2 kelas untuk wearable kontinu. H3
  menentukan apakah perlu gerbang kualitas sinyal tambahan sebelum uji perangkat. Ada yang gagal → laporkan; ambang dan aturan tidak diubah.
- Bukan pengganti paket v3.1; tidak ada `model_export/` yang dibuat atau diubah.

## 5. Yang tidak dilakukan

Tidak menyetel ambang d1, durasi, atau aturan; tidak melatih apa pun; tidak memakai data ini untuk memilih apa pun.

## 6. Keluaran

`v16.7_konfirmasi_wearable/`: `konfirmasi.py`, `konfirmasi.ipynb`, `README.md`, `laporan/`. Cache `../cache/v16.7/`. Sinyal Icentia
terpilih di `data/icentia_v167/` (tidak di-commit).

## 8. Penyimpangan dari rancangan

(belum ada)
