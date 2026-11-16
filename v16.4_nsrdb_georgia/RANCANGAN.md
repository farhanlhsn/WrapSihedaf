# Rancangan v16.4: D-1 dan v3.1 di dua dataset yang belum pernah dipakai (NSRDB, Georgia 12-lead)

Tanggal: 10 November 2026. Status: **dikunci sebelum sinyal kedua dataset dibaca oleh kode apa pun** (sha256 dan waktu di
`logs/v16.4_praregistrasi.log`). Yang sudah dilihat sebelum dikunci: daftar berkas, header (fs, durasi, kode diagnosis Georgia), dan
anotasi detak NSRDB (jumlah per kode, tanpa label ritme). Bagian 2–6 tidak diubah setelah hasil keluar; penyimpangan di bagian 8.

## 1. Latar

D-1 (AF bila `d1 = log((pAF + pAFL) / pN) > −1`, ECGFounder 1-lead zero-shot) adalah kandidat detektor AF 2 kelas untuk uji perangkat
(v16.2 CPSC 2021; v16.3 replikasi 2/3). Ketujuh data publik sebelumnya sudah dipakai. Dua pertanyaan produk belum terjawab:

1. Berapa alarm AF palsu pada **orang sehat sepanjang hari** (aktivitas harian, tidur, artefak), bukan hanya latihan fisik singkat (v3.3)?
2. Apakah keunggulan D-1 bertahan di populasi klinis lain (AS) dengan label dokter untuk **sinus takikardia**, sinus bradikardia
   (v16.3: D-1 mengalarmkan 28% sinus bradikardia LTAFDB), dan **AFL**?

## 2. Data (tetap)

| Dataset | Isi | Kanal | Label 2 kelas | Unit bootstrap |
|---|---|---|---|---|
| **NSRDB** (PhysioNet `nsrdb` 1.0.0, ODC-BY) | 18 subjek tanpa aritmia berarti, Holter ±24 jam (437,5 jam), 128 Hz | kanal 0 (ECG1) | semua window **non-AF** (tidak ada label ritme) | subjek |
| **Georgia** (PhysioNet `challenge-2021` 1.0.3, `training/georgia/`, CC BY 4.0) | 10.344 EKG 12-lead istirahat, 10 s (52 rekaman 5 s), 500 Hz, Emory (AS) | **lead I** (baris pertama, sama dengan v16.2 dan ECGFounder 1-lead) | AF bila kode Dx memuat 164889003 (AF) atau 164890007 (AFL); selain itu non-AF | rekaman (tidak ada ID pasien) |

- Window perangkat: `hrv_core.signal_to_windows(x, fs)` (resample 50 Hz, detektor R produksi, 10 s, langkah 5 s). Georgia 10 s = satu
  window; rekaman 5 s tidak menghasilkan window. EKG untuk ECGFounder = 500 sampel 50 Hz yang sama (pola v16.2/v16.3).
- **Himpunan uji = window yang lolos cek kualitas detektor** (punya prediksi paket v3.1), sama dengan v16.2/v16.3. Jumlah yang
  dibuang dilaporkan per kelompok.
- Kelompok Georgia (eksklusif, urutan prioritas): AF/AFL → sinus takikardia (427084000) → sinus bradikardia (426177001) → sinus normal
  (426783006) → lainnya. Subkelompok deskriptif: AFL tanpa AF, AF, takikardia atrium/SVT (713422000, 426761007).
- NSRDB: window yang memuat anotasi artefak/derau (kode MIT 14 atau 16) ditandai untuk analisis sensitivitas.

## 3. Sistem (tetap, sama dengan v16.2/v16.3)

**D-1** (primer, `d1 > −1`), **paket v3.1** (referensi; AF bila argmax ≠ Normal), D-0 (`d1 > 0`), paket v1. AFIB + AFL digabung. Tidak
ada yang dilatih atau dipilih. Kode: `v16.3.replikasi.Alat` (ECGFounder, paket v3.1, paket v1), `v16.2.uji_cpsc`, `v16.1.deteksi_af`.

## 4. Hipotesis dan kriteria

- **H1 (Georgia):** Macro-F1 2 kelas D-1 bukan "lebih buruk" dari v3.1 (P(Δ>0) > 0,05), bootstrap rekaman berpasangan 2.000×, seed
  2026 (`v16.1.banding2_pasien`).
- **H2 (Georgia):** sensitivitas D-1 bukan "lebih buruk" dari v3.1 (P > 0,05).
- **H3 (Georgia):** alarm pada rekaman sinus takikardia D-1 ≤ separuh v3.1 (titik), dengan CI bootstrap rekaman.
- **H4 (NSRDB):** proporsi window non-AF yang dialarmkan D-1 lebih rendah dari v3.1, P(Δ < 0) ≥ 0,95, bootstrap subjek berpasangan.
- **Sekunder (pelajaran v3.4, tidak ikut keputusan):** non-inferioritas sensitivitas dengan margin −0,02: batas bawah CI 95% Δ Se > −0,02.
- **Deskriptif:** spesifisitas, PPV; alarm per kelompok Georgia (sinus normal, sinus bradikardia, lainnya, AT/SVT) dan sensitivitas AFL
  dan AF terpisah; alarm non-AF per pita detak (< 100, 100–110, 110–120, ≥ 120 bpm); NSRDB: alarm per subjek, episode AF palsu per jam
  (`takikardia_asli.metrik_episode`, langkah 5 s), alarm per jam-dalam-hari tidak dihitung (waktu mulai rekaman tidak seragam),
  sensitivitas tanpa window berartefak; D-0 dan v1 sebagai konteks.

## 5. Keputusan yang ditetapkan sebelum hasil

- **H1–H4 lolos:** D-1 tetap kandidat detektor AF 2 kelas untuk uji perangkat, dengan bukti dari enam dataset independen (CPSC 2021,
  CinC 2017, MIT-BIH, LTAFDB, NSRDB, Georgia).
- Ada yang gagal: tulis hipotesis dan dataset yang gagal; status kandidat ditinjau ulang di README, **bukan dengan mengubah ambang**.
- Bukan pengganti paket v3.1 dalam keadaan apa pun; tidak ada `model_export/` yang dibuat atau diubah.

## 6. Yang tidak dilakukan

- Tidak memilih ambang, lead, kanal, aturan label, atau kelompok dari hasil. Tidak melatih apa pun. Tidak mengubah paket produksi.
- Front-end v3.4 (F1/F2) tidak dipakai (tidak direkomendasikan).

## 7. Keluaran

`v16.4_nsrdb_georgia/`: `uji_baru.py`, `uji_baru.ipynb`, `README.md`, `laporan/`. Cache `../cache/v16.4/` (hasil per rekaman, bisa
dilanjutkan). Data: `data/nsrdb/` (mirror S3), `data/challenge2021/training/georgia/` (physionet.org, dicek SHA256SUMS.txt; tidak ada di
mirror S3, lewat opsi baru `tools/unduh_physionet.py --sumber physionet`).

## 8. Penyimpangan dari rancangan

(belum ada)
