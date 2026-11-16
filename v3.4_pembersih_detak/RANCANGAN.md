# Rancangan v3.4: pembersih detak (penyelarasan R-peak ke templat QRS + gerbang morfologi)

Tanggal: 10 November 2026. Status: **dikunci sebelum model dilatih dan sebelum metrik AFDB apa pun dihitung dengan front-end baru**
(sha256 dan waktu di `logs/v3.4_praregistrasi.log`). Bagian 2–7 tidak diubah setelah hasil keluar; penyimpangan ditulis di bagian 9.

## 1. Latar

v3.3: detektor R-peak 50 Hz menjadi sumber alarm palsu saat aktivitas. EPFL hanya 68% window lolos cek kualitas (R-peak manual 95%),
CV IBI median 0,028 (manual 0,009), alarm v3.1 pada window < 100 bpm 21% (EPHNOGRAM) dan 29% (Wrist; R-peak manual 9%), dan aturan
produk hanya menangkap 61–83% alarm ≥ 120 bpm karena jitter menaikkan CV.

Sumber ide: pembaca EKG di Tugas Akhir pemilik repo (`asd_ecg/measurement.py`, `_dominant_beats`): setiap QRS dibandingkan dengan
templat median QRS di sekitarnya (korelasi ≥ 0,8, ukuran 0,6–1,67 kali templat, minimal 60% detak dominan).

Pengamatan `hrv_core.detect_rpeaks`: refinement memakai puncak `|sinyal|`, jadi pada sebagian detak titiknya pindah dari R ke lembah S.

**Yang sudah dilihat sebelum rancangan ini (probe eksploratif 10 November 2026, scratchpad):**
- Penyelarasan ke templat bertanda (templat per potongan 60 s, satu putaran) menurunkan jitter vs R-peak manual: EPFL 10,7 → 2,6 ms.
- Window lolos QC EPFL 68% → 91%; CV median 0,028 → 0,012; aturan produk 73% → 89%.
- Alarm paket v3.1 < 100 bpm: Wrist 29,2% → 22,4%, EPHNOGRAM 21,4% → 18,6%.
- AFDB: jitter dan RMSSD per ritme hampir tidak berubah (AFIB RMSSD 232 → 232 ms).

Jadi hipotesis data latihan fisik (H2, H3) **bukan uji independen**: arahnya sudah terlihat. Yang belum pernah dilihat: metrik AFDB
model yang dilatih dengan front-end baru, efek gerbang morfologi, dan uji drop-in.

## 2. Front-end (tetap)

Semua di 50 Hz, setelah `hrv_core.resample_signal` dan `hrv_core.detect_rpeaks` (detektor produksi tidak diubah).

- **Sinyal acuan** `y`: sinyal bandpass yang sama dengan di dalam `detect_rpeaks` (clip persentil 0,5/99,5, Butterworth orde 2,
  5–15 Hz, `filtfilt`). Salinan dibuktikan identik: detektor yang dibangun ulang dari salinan ini harus menghasilkan R-peak yang sama
  bit demi bit dengan `hrv_core.detect_rpeaks`.
- **Segmen QRS:** ±0,10 s di sekitar `round(peak)` (11 sampel).
- **Templat per detak:** median segmen semua detak dengan |t − t_detak| ≤ 5 s (konteks 10 s, sepanjang window; tidak bergantung
  pada batas window). Butuh ≥ 3 segmen lengkap; bila tidak, detak tidak diubah dan dianggap dominan (pola TA: ujung rekaman).
- **F1 — penyelarasan:** korelasi (Pearson) segmen dengan templat pada lag −3…+3 sampel; ambil lag terbaik, geser sub-sampel dengan
  interpolasi parabolik puncak korelasi (offset dibatasi ±0,5). Posisi baru = `round(peak) + lag + offset`. Satu putaran.
- **F2 — F1 + gerbang morfologi:** detak dominan = korelasi terbaik ≥ 0,8 dan ptp(segmen)/ptp(templat) dalam 0,6–1,67. Window
  (sesudah F1) **ditolak** bila < 60% detak penutup IBI-nya dominan. Window ditolak = keluaran "sinyal buruk": tidak ada alarm.
- Aturan lain di TA (detak terlewat 1,7–2,3× median, pembuangan detak prematur) **tidak dipakai**: keduanya bisa menghapus
  ketidakteraturan AF.
- `hrv_core.window_quality`, panjang window, langkah, dan `build_sequence` tidak berubah.

## 3. Data dan model (tetap)

- **AFDB:** ekstraktor v2 (sumber "detector") dibangun ulang dengan pembaca `v12.atrium` (tanpa wfdb). Dengan front-end lama ia harus
  identik bit demi bit dengan `cache/dataset_v2_detector.npz` (X, M, y, t_start, records). Dataset F1 = ekstraktor yang sama dengan
  R-peak F1. Fold: pasien → fold dari `folds.npz` resmi (pembagian pasien sama; window boleh berbeda).
- **Model F1:** MicroTransformer, resep v3 = resep v3.1 (`sihedaf_cv.train_dl`, bobot kelas penuh, 30 epoch, batch 256, lr 1e-3,
  wd 1e-4, seed `2026 + 100·r + k`), 5 ulangan × 5 fold, dilatih dan diuji pada window F1. Bobot kelas penuh (bukan lunak) dipakai
  **sengaja** supaya satu-satunya perbedaan dengan referensi adalah front-end. Bobot setiap model fold disimpan.
- **Uji paritas (protokol 8):** fungsi job v3.4 pada dataset lama ulangan 0 harus mereproduksi OOF `MicroTransformer_r0.npz`
  (selisih maks ≤ 1e-5; bila GPU non-deterministik, argmax sama ≥ 99,9% window dan dicatat).
- **Bias:** cross-fit grid −7…+3 (`sihedaf_cv.crossfit_bias`) pada OOF F1. Untuk F2, model dan bias F1 dipakai apa adanya; window
  yang ditolak gerbang dihitung sebagai prediksi Normal (tidak ada alarm), sehingga himpunan window F2 = himpunan F1.
- **Latihan fisik:** EPFL, Wrist, EPHNOGRAM lewat pembaca v3.3, grid window yang sama dengan v3.3 (langkah 10 s, tidak tergantung QC).
  Front-end lama harus mereproduksi `cache/v3_3/jendela.npz` mode perangkat.

## 4. Hipotesis dan kriteria

- **H1 (utama, AFDB):** model F1 +bias vs v3.1 (`MicroTransformer +bias` OOF v3), bootstrap pasien berpasangan 2.000× (seed 2026,
  strata dari dataset lama): Macro-F1 **dan** recall AFL bukan "lebih buruk" (P(Δ>0) > 0,05).
- **H2 (utama, aktivitas):** paket v3.1 dengan front-end F1 vs front-end lama, window Wrist + EPHNOGRAM mode perangkat dengan detak
  < 100 bpm menurut front-end lama (himpunan window yang sama untuk keduanya; window gagal QC = tidak ada alarm): alarm lebih sedikit,
  P(Δ < 0) ≥ 0,95 dengan bootstrap subjek berpasangan.
- **H3 (aktivitas):** (a) % window lolos QC EPFL naik (deskriptif, per subjek); (b) alarm yang tersisa sesudah aturan produk
  (≥ 110 bpm & CV ≤ 0,05) pada window ≥ 120 bpm (front-end lama), ketiga sumber: lebih sedikit, P(Δ < 0) ≥ 0,95.
- **H4 (gerbang F2):** vs F1 dengan paket v3.1: alarm < 100 bpm aktivitas lebih sedikit (P ≥ 0,95, himpunan H2). Biaya di AFDB
  (model F1, gated = Normal): recall AFIB dan recall AFL masing-masing turun ≤ 0,02 (titik) dibanding F1, dan Macro-F1 F2 vs v3.1
  bukan "lebih buruk".
- **H5 (drop-in, sekunder):** model yang dilatih dengan front-end lama (`cache/v7/model/kontrol_r0_k*.pt`, resep v5b, ulangan 0,
  bias fold dari OOF lamanya) diuji pada window F1 vs OOF-nya sendiri pada window lama. Δ Macro-F1 dan recall AFL dengan CI pasien;
  "lolos" bila bukan lebih buruk. Ini menjawab apakah bobot paket v3.1 bisa dipakai ulang tanpa latih ulang.
- **Deskriptif:** jitter vs R-peak manual (EPFL, Wrist) dan vs anotasi `.qrs` AFDB per ritme; RMSSD per ritme; jumlah window per
  kelas AFDB; recall AFL per pasien (7) dan asal AFL palsu; window Normal AFDB per pita detak (`takikardia.by_hr_bin`); metrik episode
  AFIB/AFL gaya EC57; alarm per pita (detak front-end sendiri) seperti v3.3; analisis sensitivitas pada window AFDB yang sama
  (rekaman, t_start, label sama di kedua dataset).
- **Protokol 9:** uji stres sintetis 60–170 bpm pada model fold F1 (front-end tidak berperan pada IBI sintetis; dilaporkan untuk
  kelengkapan), ambang bpm; alarm sinus takikardia asli ≥ 120 bpm untuk paket v3.1 + F1/F2 dan model fold F1 dengan bias fold.
- **Protokol 10:** bootstrap bersarang (bias dipilih ulang tiap sampel untuk F1 dan v3.1, bobot pasien sama) dan sensitivitas tanpa
  04936, untuk H1.

## 5. Keputusan yang ditetapkan sebelum hasil

- **F1 direkomendasikan sebagai front-end paket v3.1** bila H1 lolos, H2 atau H3(b) lolos, dan H5 lolos. Selain itu: tidak
  direkomendasikan; tulis bagian yang gagal.
- **F2 direkomendasikan sebagai tambahan** bila F1 direkomendasikan dan H4 lolos (keduanya: alarm turun dan biaya AFDB dalam batas).
- Rekomendasi bukan pemasangan: `model_export/` mana pun tidak dibuat atau diubah di v3.4. Memasang front-end di paket butuh
  persetujuan pemilik repo (versi berikutnya).

## 6. Yang tidak dilakukan

- Tidak menyetel parameter front-end (lag, konteks, ambang 0,8 / 0,6–1,67 / 60%) pada data mana pun; nilainya dari TA dan probe.
- Tidak menjalankan notebook v2 atau v3.1, tidak mengubah paket. Tidak melatih model selain MicroTransformer resep v3 di atas.

## 7. Keluaran

`v3.4_pembersih_detak/`: `pembersih.py`, `pembersih_detak.ipynb`, `test_pembersih.py`, `README.md`, `laporan/`. Cache
`../cache/v3_4/` (dataset, OOF, bobot model fold, prediksi, `ringkasan.json`).

## 8. Lingkungan

PC desktop, RTX 5060 8 GB (training ±30 menit diperkirakan; Colab bila jauh lebih lama).

## 9. Penyimpangan dari rancangan

### Amandemen A (10 November 2026, ±04:05 WIB): data eksternal

Diminta pemilik repo dan ditulis **sebelum** metrik AFDB, prediksi, atau sinyal data eksternal apa pun dengan front-end baru dilihat.
Yang sudah terlihat saat amandemen ini ditulis: hanya bukti kendali mutu (dataset lama identik dengan v2, uji paritas training
identik, window latihan fisik lama identik dengan v3.3) dan jumlah window F1 per kelas AFDB.

- **Data:** CinC 2017, MIT-BIH Arrhythmia, LTAFDB (pembaca dan label v16.3: CinC label rekaman, `~` dibuang, AF = `A`; MIT-BIH dan
  LTAFDB label ritme konvensi MIT, ritme pacu `(P` dibuang, AF = `(AFIB`/`(AFL`; subjek MIT-BIH 201 = 202) dan CPSC 2021 (pembaca
  dan **label resmi** v16.2 `_label_resmi`, subjek dari nama rekaman). Kanal 0.
- **Window:** grid `hrv_core.signal_to_windows` (10 s, langkah 5 s, tidak bergantung QC), sama untuk front-end lama, F1, F2, sehingga
  window berpasangan. Front-end lama di-assert sama dengan `signal_to_windows`.
- **Sistem:** paket v3.1 (2 kelas: AF bila argmax ≠ Normal) dengan front-end lama, F1, dan F2. Tidak ada yang dilatih atau disetel.
- **Himpunan uji utama:** semua window berlabel; window gagal cek kualitas = tidak ada alarm (perilaku produk). Sensitivitas: hanya
  window yang lolos QC di front-end lama **dan** F1.
- **Metrik:** Se, Sp, PPV, Macro-F1 2 kelas, bootstrap subjek berpasangan 2.000× (`v16.1.deteksi_af.banding2_pasien`, seed 2026);
  alarm pada window non-AF per pita detak (< 100, 100–110, 110–120, ≥ 120 bpm, detak front-end lama); % window lolos QC per kelas.
- **H6 (ditambahkan ke keputusan bagian 5):** F1 hanya direkomendasikan bila, di **setiap** dataset eksternal, sensitivitas F1 vs
  front-end lama bukan "lebih buruk" (P(Δ>0) > 0,05). F2 hanya direkomendasikan bila hal yang sama berlaku untuk F2 vs front-end lama.
  Spesifisitas dan Macro-F1 dilaporkan dengan vonis protokol tanpa syarat tambahan.
- Dijalankan di PC (CPU 12 core): pekerjaannya pemrosesan sinyal CPU; prediksi paket kecil. GPU/Colab tidak mempercepat.

### Penyimpangan B (10 November 2026, ±04:35 WIB, sesudah run pertama notebook): pelaporan saja

- **Vonis "identik".** Pada F2 vs F1, recall AFL sama persis di semua sampel bootstrap (gerbang tidak menolak satu pun window AFL),
  sehingga P(Δ>0) = 0 dan vonis otomatis tertulis "lebih buruk". `pembersih.banding` sekarang menulis "identik" bila Δ = 0 di semua
  sampel. Tidak ada keputusan bagian 5 yang memakai perbandingan ini.
- **Tabel Se per subjek ber-AF** di data eksternal ditambahkan (deskriptif), setelah H6 F1 gagal karena MIT-BIH (Δ Se −0,007, P = 0,038;
  seri hanya 4% sampel bootstrap, jadi bukan artefak). Aturan H6 dan definisi P tidak diubah; keputusan tetap.
