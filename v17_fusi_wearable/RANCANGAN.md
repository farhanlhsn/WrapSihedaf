# Rancangan v17: fusi ECGFounder + RR v3.1 yang dilatih dengan EKG genggam, untuk sensitivitas AF di data wearable

Tanggal: 6 November 2026. Status: **dikunci sebelum model dilatih atau metrik dihitung** (sha256 dan waktu di `logs/v17_praregistrasi.log`).
Bagian 2–6 tidak diubah setelah hasil keluar; penyimpangan ditulis di bagian 8.

## 1. Latar

D-1 (ECGFounder zero-shot, AF bila `d1 > −1`) adalah kandidat detektor AF 2 kelas (v16.2–v16.3): lebih spesifik dari paket v3.1 di tujuh
dataset, tetapi pada EKG genggam/wearable satu lead sensitivitasnya lebih rendah (CinC 2017: 0,830 vs v3.1 0,884; Icentia11k: 0,843 vs
0,871). Model RR v3.1 lebih sensitif di data itu tetapi mengalarmkan sinus takikardia. Keduanya membaca informasi berbeda (bentuk gelombang
vs ritme). Pertanyaan v17: apakah gabungan kecil keduanya, yang **dilatih dengan EKG genggam**, menaikkan sensitivitas wearable tanpa
kehilangan spesifisitas D-1?

Yang sudah diketahui sebelum dikunci: d1 dan keputusan paket v3.1 di semua dataset di bawah (v13–v16.3). Belum pernah dihitung: model fusi
apa pun yang dilatih di CinC 2017, dan hasilnya di dataset mana pun.

## 2. Data (tetap)

- **Data latih: CinC 2017** (PhysioNet `challenge-2017`, lisensi **CC BY 3.0**, boleh dipakai produk dengan atribusi): window perangkat
  yang lolos cek dari v16.3 (`cache/v16.3/cinc2017/`), 43.465 window dari 8.172 rekaman; AF = `A`, non-AF = `N` + `O`.
- **Uji utama (wearable): Icentia11k** (window v10, 115.722 window, 2.420 pasien; CC BY-NC-SA, **hanya evaluasi**, tidak dipakai untuk
  melatih atau memilih apa pun). AF = AFIB atau AFL. Semua window dipakai seperti di v13–v16.2 (termasuk window Normal cepat tambahan v10;
  hasil tanpa window itu dilaporkan juga). Catatan: EKG window Icentia dari v10 sudah difilter 0,5–20 Hz (sama untuk semua sistem).
- **Penjaga (Holter/klinis):** CPSC 2021 (label resmi, v16.2), MIT-BIH dan LTAFDB (v16.3). **Sinus takikardia asli:** latihan fisik v3.3
  ≥ 120 bpm. **Deskriptif:** AFDB (5 ulangan OOF v3.1 + bias paket, seperti `v15._afdb_logp`), PTB-XL. Validasi silang CinC 2017 per
  rekaman (5 fold, `GroupKFold`) sebagai pendukung.
- Semua masukan sudah ada di cache (v13, v16, v16.2, v16.3); tidak ada inferensi ECGFounder baru.

## 3. Sistem (tetap)

| Sistem | Keputusan AF | Parameter |
|---|---|---|
| **W-F (primer)** | regresi logistik pada `[d1, r]`, `r = log((pAFIB + pAFL) / pN)` paket v3.1; StandardScaler + `LogisticRegression(C=1, class_weight="balanced")` (resep `v15.latih_gabungan`), AF bila p ≥ 0,5 | 3, dilatih di seluruh CinC 2017 |
| D-1 (kandidat sekarang) | `d1 > −1` | – |
| **v3.1 paket (produksi)** | argmax paket, AFIB + AFL digabung | – |
| D-1 ∨ v3.1 (pembanding naif) | AF bila salah satu menyebut AF | – |

## 4. Hipotesis dan kriteria (W-F)

Bootstrap unit berpasangan 2.000×, seed 2026 (`v16.1.banding2_pasien`; Icentia/CPSC/MIT-BIH/LTAFDB per pasien, latihan fisik per
subjek).

- **K1:** Icentia11k, sensitivitas W-F vs D-1: **lebih baik** (P ≥ 0,95).
- **K2:** Icentia11k, Macro-F1 2 kelas W-F vs D-1: bukan lebih buruk (P > 0,05).
- **K3:** Icentia11k, Macro-F1 dan sensitivitas W-F vs v3.1 paket: bukan lebih buruk (P > 0,05).
- **K4:** sinus takikardia asli ≥ 120 bpm: alarm W-F ≤ 10% (D-1 0,46%, v3.1 98,4%).
- **K5:** CPSC 2021, MIT-BIH, LTAFDB: Macro-F1 W-F vs D-1 bukan lebih buruk (P > 0,05) di ketiganya.
- **Pendukung (bukan kriteria):** validasi silang CinC 2017, AFDB, PTB-XL, alarm per ritme non-AF (MIT-BIH, LTAFDB), koefisien W-F.

## 5. Keputusan yang ditetapkan sebelum hasil

- K1–K5 lolos semua → **W-F menggantikan D-1 sebagai kandidat detektor AF 2 kelas untuk uji perangkat.** Selain itu D-1 tetap kandidat;
  tulis kriteria yang gagal. Bukan pengganti paket v3.1 dalam keadaan apa pun.
- Bila W-F lolos, rekaman perangkat tetap penentu (seperti D-1).

## 6. Yang tidak dilakukan

- Tidak melatih atau memilih apa pun di Icentia11k, CPSC 2021, MIT-BIH, LTAFDB, AFDB, PTB-XL, atau data latihan fisik. Tidak mengubah ambang
  0,5, masukan, atau resep setelah melihat hasil. Tidak fine-tune ECGFounder. Tidak mengubah paket produksi.

## 7. Keluaran

`v17_fusi_wearable/`: `fusi.py`, `fusi.ipynb`, `README.md`, `laporan/`. Cache `../cache/v17/` (model W-F dan ringkasan).

## 8. Penyimpangan dari rancangan

(belum ada)
