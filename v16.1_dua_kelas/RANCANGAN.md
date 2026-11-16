# Rancangan v16.1: detektor AF 2 kelas dari tahap 1 v16 (ECGFounder tanpa penyetelan)

Tanggal: 6 November 2026. Status: **dikunci sebelum metrik baru dihitung** (sha256 dan waktu di `logs/v16.1_praregistrasi.log`).
Bagian 2–6 tidak diubah setelah hasil keluar; penyimpangan ditulis di bagian 8.

Versi minor dari v16: tidak ada training dan tidak ada parameter yang disetel. Keluarannya hanya **Normal vs AF** (AFIB atau AFL
digabung), atas permintaan pemilik repo setelah v16 menunjukkan bahwa sistem 3 kelas gagal tetapi keputusan sinus vs AF setara v3.1.

Yang **sudah diketahui** dari v16 (tidak dihitung ulang sebagai temuan baru): di AFDB, tahap 1 tanpa penyetelan memberi Macro-F1 2 kelas
0,951 vs v3.1 0,926 (P = 0,92; bersarang P = 0,92), sensitivitas 0,893 vs 0,934 (lebih buruk, P = 0,017), spesifisitas 0,995 vs
0,924; alarm sinus takikardia asli ≥ 120 bpm 0,1% vs 98,4%; keluaran "perlu dicek" (tahap 1 sinus dan v3.1 AFL) menampung 58% AFL AFDB
dan 74% sinus takikardia asli.

Yang **belum** dihitung: 2 kelas di PTB-XL dan Icentia, tanpa 04936 untuk 2 kelas, sensitivitas dan alarm per pasien, alarm Normal per
pita detak di AFDB, episode AF gabungan, sistem D-L (bagian 3), kurva operasi ambang, paritas fungsi prediksi, dan latensi.

## 1. Latar

Bila produk SIHEDAF cukup melaporkan "AF ya/tidak", masalah AFL vs AFIB hilang dan tinggal satu pertanyaan: apakah ECGFounder sebagai
detektor AF lebih aman daripada v3.1 (alarm palsu jauh lebih sedikit) tanpa kehilangan terlalu banyak AF? Harga yang sudah terlihat
adalah AFL: tahap 1 menyebut 85% window AFL AFDB sebagai sinus.

## 2. Data (semua dari cache, tanpa inferensi ulang kecuali bagian 5)

Sama dengan v16: AFDB window detektor 50 Hz (fold resmi 5×5), EKG latihan fisik v3.3 (mode perangkat), PTB-XL lead I 50 Hz, Icentia11k
(akademik saja). Label 2 kelas: 0 = Normal, 1 = AF (AFIB atau AFL).

## 3. Sistem (tetap)

- **D-0 (primer):** AF bila `d1 = log((pAF + pAFL) / pN) > 0` dari skor zero-shot ECGFounder (`v16.hierarki.tahap1`). Tanpa parameter.
- **D-L (sekunder):** D-0, ditambah AF bila v3.1 menyebut AFL (AFDB: OOF v3.1 dengan bias 3 kelas cross-fit; eksternal: paket v3.1).
  Ini keluaran "perlu dicek" v16 yang dihitung sebagai alarm.
- **Pembanding:** v3.1 (bias 3 kelas lalu AFIB + AFL digabung, aturan v11.1; eksternal: paket v3.1 argmax digabung), v1 (argmax
  digabung), dan versi 2 kelas terbaik sebelumnya, **BiGRU bawaan v11.1** (aturan yang sama; tidak punya model penuh, jadi hanya
  AFDB dan latihan fisik lewat model fold v3.3).

## 4. Hipotesis dan kriteria

- **H1 (eksternal 2 kelas, utama):** D-0 vs paket v3.1 di PTB-XL dan Icentia11k: Macro-F1 2 kelas, sensitivitas, spesifisitas,
  bootstrap pasien berpasangan (2.000×, seed 2026). Vonis protokol.
- **H2 (tanpa 04936, utama):** Macro-F1 2 kelas D-0 vs v3.1 pada AFDB tanpa 04936 (bias v3.1 dipilih ulang di subset). Vonis protokol.
- **H3 (per pasien):** sensitivitas per pasien ber-AF dan alarm pada window Normal per pasien, D-0 vs v3.1. Deskriptif; sebutkan pasien
  dengan sensitivitas turun > 0,20.
- **H4 (D-L):** sensitivitas dan spesifisitas D-L vs D-0 dan v3.1 di AFDB dan PTB-XL, dan alarm sinus takikardia asli ≥ 120 bpm.
- **H5 (kurva operasi, deskriptif):** ambang `d1 > t` untuk t ∈ {−4, −3, −2, −1, 0, 1, 2, 3, 4}: sensitivitas dan spesifisitas AFDB,
  alarm sinus takikardia asli ≥ 120 bpm, alarm STACH PTB-XL, sensitivitas PTB-XL. **Tidak ada ambang yang dipilih**; D-0 tetap t = 0.
- **H6 (fungsi prediksi, butir 9):** fungsi `DetektorAF` (EKG 50 Hz mentah → keputusan) dijalankan ulang di CPU float32 pada window
  08378 (semua window AFL + window non-AFL berjarak tetap, ±600) dan dibandingkan dengan logit cache v13 (GPU bf16): kesesuaian
  keputusan dan selisih maksimum `d1`. Latensi per window di CPU (batch 1) dan GPU (batch 64). Deskriptif.
- **Protokol:** butir 1–4 untuk 2 kelas (pola v11.1). Butir 5: per pasien (H3). Butir 6: D-0 tanpa parameter; v3.1 dan BiGRU memakai
  bias 3 kelas cross-fit. Butir 7: episode AF gabungan (`takikardia_asli.metrik_episode`, kelas AF). Butir 9: sinus takikardia asli
  per pita detak (latihan fisik) dan alarm Normal AFDB per pita detak serta Normal teratur 80–95 bpm (pola v11.1); uji stres RR
  sintetis tidak berlaku untuk D-0. Butir 10: bootstrap bersarang (dari v16, ditulis ulang di laporan) dan H2.

## 5. Keputusan yang ditetapkan sebelum hasil

- D-0 disebut **kandidat detektor AF 2 kelas untuk diuji di perangkat** bila: Macro-F1 2 kelas tidak "lebih buruk" dari v3.1 di AFDB
  (sudah diketahui lolos), tanpa 04936 (H2), PTB-XL dan Icentia11k (H1); dan alarm sinus takikardia asli ≤ separuh v3.1 (sudah
  diketahui lolos). Penurunan sensitivitas dilaporkan apa adanya; tidak ada margin non-inferioritas yang ditambahkan.
- Bukan pengganti paket v3.1. Tidak ada paket `model_export/` yang dibuat atau diubah; `DetektorAF` adalah fungsi prediksi untuk uji.

## 6. Yang tidak dilakukan

- Tidak memilih ambang D-0 dari AFDB, data latihan fisik, atau data eksternal (pelajaran v16 H-X). Tidak melatih apa pun. Tidak
  mengubah paket produksi.

## 7. Keluaran

`v16.1_dua_kelas/`: `deteksi_af.py`, `deteksi_af.ipynb`, `README.md`, `laporan/`. Cache `../cache/v16.1/`.

## 8. Penyimpangan dari rancangan

- **Tambahan setelah hasil (6 November 2026, setelah run pertama):** Icentia11k adalah satu-satunya kriteria yang gagal, dan kurva
  operasi (H5) menunjukkan sensitivitas pulih pada ambang lebih rendah. Karena itu kurva yang sama juga dihitung untuk Icentia11k
  (sensitivitas, spesifisitas, Macro-F1 2 kelas per t). Deskriptif saja: kriteria bagian 5 dan D-0 (t = 0) tidak berubah, dan
  tidak ada ambang yang dipilih. Ambang lain hanya boleh diuji di versi berikutnya dengan rancangan baru dan data yang tidak
  dipakai untuk melihat kurva ini.
