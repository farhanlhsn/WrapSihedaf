# Rancangan v12: aktivitas atrium dari EKG + data Holter SHDB-AF

Tanggal: 2 November 2026. Status: **disetujui pemilik repo** (2 November 2026), termasuk unduhan sinyal 98 rekaman beranotasi
SHDB-AF (6,71 GB).

**Hasil (2 November 2026): berhenti di gerbang.** Gerbang 7.2 gagal pada rancangan awal (bagian 13) dan setelah revisi 1
(bagian 14). Hasil dan kesimpulan di [README.md](README.md).

Ringkasnya: model RR resep v3.1 dipertahankan apa adanya, lalu ditambah cabang kecil yang mengukur **aktivitas atrium**
(gelombang flutter vs gelombang-f AF vs gelombang P sinus) dari EKG yang sama. Dua cabang digabung di akhir oleh model fusi
kecil. Data latih ditambah pasien Holter dari SHDB-AF, yaitu domain yang sama dengan AFDB.

## 1. Latar dan alasan

### 1.1 Batas model RR sudah terukur

Model RR-only (v3.1 dan TCN `wp100` v11) di AFDB, dengan window AFL dikelompokkan menurut pola RR-nya:

| Window AFL AFDB | Jumlah | Recall v3.1 | Recall TCN wp100 |
|---|---|---|---|
| cepat-teratur (≥ 125 bpm, CV IBI < 0,06), seperti AFL 2:1 | 695 (60%) | 0,92 | 0,92 |
| antara | 128 | 0,21 | 0,15 |
| tidak teratur (CV ≥ 0,10), mirip AF | 208 | 0,07 | 0,11 |
| lambat-teratur (< 110 bpm, CV < 0,06), mirip sinus | 121 | 0,00 | 0,00 |

Window **non-AFL** yang cepat-teratur (134 window) ditebak AFL 84–88%. Jadi model RR pada dasarnya belajar aturan "cepat dan
teratur = AFL". Plafon recall AFL ±0,6, dan Macro-F1 3 kelas RR-only di AFDB ±0,83–0,85. Tuning BiGRU dan TCN, ensemble
(0,833), dan data AFL publik untuk model RR (v8, v10) tidak menembusnya.

### 1.2 Literatur

- ArNet2 (RR 60 detak, 2.147 pasien Holter UVAF) hanya mendeteksi 54% window pasien AFL (F1 0,70), padahal AF dan AFL sudah
  digabung. Model EKG mentah pada window yang sama: 0,97 (F1 0,98). Alasannya: RR pada AFL relatif teratur; informasinya ada di
  gelombang flutter (Ben-Moshe dkk., CinC 2022; Biton dkk., npj Digit Med 2023).
- Literatur AF berbasis Holter menggabungkan AF + AFL ("AF₁"); F1 RR-only di data eksternal 0,89–0,92 (RawECGNet, arXiv
  2401.05411; SHDB-AF, arXiv 2406.16974).

### 1.3 Kenapa cabang EKG sebelumnya gagal, dan apa yang beda sekarang

| Versi | Cabang EKG | Data | Macro-F1 AFDB | Penyebab gagal |
|---|---|---|---|---|
| v6 6b | CNN 4 blok residual (32→128 kanal) + MicroTransformer | AFDB | 0,747 | CNN besar overfit per pasien; bias −7…+3 |
| v6 late fusion | rata-rata v3.1 + CNN EKG | AFDB | **0,838** (setara) | satu-satunya varian stabil; recall Normal 0,965 |
| v9 9a | CNN v6 | + Chapman (EKG klinis 10 s) | 0,687 | domain beda |
| v10 10a | CNN v6 | + Chapman + Icentia (patch) | 0,646 | batas AFL/AFIB Icentia beda |

Yang beda di v12:
1. **Cabang atrium berkapasitas sangat kecil** (fitur fisiologis + model fusi linear), bukan CNN ratusan ribu parameter.
2. **Cabang RR tidak dilatih ulang dan tidak digabung di dalam satu jaringan**; penggabungan di akhir (pola yang stabil di v6).
3. **Data tambahan dari domain yang sama**: SHDB-AF adalah Holter 24 jam rawat jalan, seperti AFDB.

### 1.4 SHDB-AF (sudah diperiksa dari anotasi)

PhysioNet `shdb-af` 1.0.1 (Tsutsui dkk. 2024), Holter Jepang (Fukuda, lead CC5 dan NASA, direkam 125 Hz, dirilis 200 Hz),
lisensi **ODC-BY 1.0** (sama dengan AFDB; boleh komersial dengan atribusi). 128 rekaman; **98 beranotasi per detak oleh seorang
fellow kardiologi** (PhysioZoo), dari **93 subjek** (5 subjek punya lebih dari satu rekaman). Label: (N, (AFIB, (AFL, (AT,
(PAT/(NOD, (AB. Setiap detak bersimbol `"`; label ritme ada di `aux_note` saat ritme berganti.

Anotasi dan header sudah di `data/shdb_af/1.0.1/` (232 file, 21 MB, SHA256 cocok). Fenotipe window 10 s (langkah 5 s, label
bila semua detak satu ritme, dari detak anotasi):

| Kelas | Pasien / rekaman | Window | Detak | CV IBI | cepat-teratur | tidak teratur | lambat-teratur |
|---|---|---|---|---|---|---|---|
| AFL AFDB | 7 pasien | 1.152 | 139 | 0,030 | **60%** | 18% | 11% |
| AFL SHDB-AF | 12 subjek | 25.683 | 70 | 0,125 | 18% | **62%** | 9% |
| AFIB AFDB | 23 | 65.826 | 90 | 0,213 | 0% | 97% | 0% |
| AFIB SHDB-AF | 76 | 326.037 | 86 | 0,199 | 1% | 93% | 4% |
| Normal AFDB | 21 | 99.000 | 70 | 0,025 | 0% | 22% | 73% |
| Normal SHDB-AF | 97 | 1.313.844 | 69 | 0,017 | 0% | 20% | 76% |

AFIB dan Normal kedua dataset mirip. AFL SHDB-AF kebanyakan **tidak teratur** (blok AV berubah-ubah); satu subjek (rekaman
010, 12.479 window, 60 bpm, CV 0,18) menyumbang 49%. AFL cepat-teratur ±4.600 window dari 5 subjek. Dari RR saja, AFL SHDB-AF
yang tidak teratur identik dengan AF; justru bagian inilah yang perlu dibedakan oleh cabang atrium.

## 2. Tujuan

1. Menguji apakah informasi aktivitas atrium dari EKG satu lead 50 Hz (kondisi perangkat) mendeteksi AFL **di luar pola
   cepat-teratur**, tanpa menambah AFL palsu.
2. Mengukur sumbangan pasien Holter SHDB-AF dibanding AFDB saja.
3. Mengevaluasi di dua domain berdampingan (AFDB dan SHDB-AF), sesuai pelajaran v9.

## 3. Bukan tujuan

- Melatih ulang atau mengganti model RR. Model RR resep v3.1 dipakai apa adanya.
- CNN EKG besar end-to-end (sudah gagal di v6, v9, v10).
- Mengubah paket produksi. Hasil v12 baru jadi kandidat bila lolos protokol butir 9–10.
- Mengejar angka dengan mengubah definisi tugas. Macro-F1 2 kelas (AF/AFL vs Normal) dilaporkan **berdampingan**, bukan
  menggantikan 3 kelas.

## 4. Anti-tujuan (dianggap gagal walaupun kodenya jalan)

- Bias kelas cross-fit menyentuh batas grid (−7 atau +3) di > 10% fold, atau melompat lebar antar fold (tanda overfit, v6/v9).
- AFL palsu naik dari window AFIB pasien tertentu (pola 04936/06995 di v10).
- Recall AFL 08378 (AFL 2:1) turun di bawah 0,78.
- Hasil baik di SHDB-AF tetapi buruk di AFDB, atau sebaliknya, tanpa dilaporkan berdampingan.

## 5. Arsitektur

```
EKG mentah satu lead (AFDB: kanal 0 seperti v6; SHDB-AF: ECG1/CC5, ECG2/NASA untuk analisis sensitivitas)
   │ front-end produksi (dipakai ulang, tidak disalin): hrv_core.resample_signal → 50 Hz; v6 preprocess_record → bandpass
   │ 0,5–20 Hz; hrv_core.detect_rpeaks → window 10 s, langkah 5 s (sama dengan dataset v2)
   ├─► [A] Cabang RR: MicroTransformer resep v3.1 (model per fold dari v8, cache/v8/model/v31_r*_k*.pt) → 3 logit
   └─► [B] Cabang aktivitas atrium (baru, tanpa jaringan besar)
          B1. Pembatalan QRST: templat = median detak di window, sejajar R-peak (−0,10…+0,40 s = −5…+20 sampel);
              templat dikurangkan dari tiap detak → sinyal sisa (aktivitas atrium). Window < 4 detak: templat dari window
              tetangga; tetap < 4: fitur ditandai hilang.
          B2. Fitur (±12, ditetapkan sekarang):
              • frekuensi atrium dominan DF (Welch 3–10 Hz) → siklus flutter FCL = 1/DF
              • indeks organisasi: energi di DF±0,5 Hz dan 2·DF±0,5 Hz / energi 3–12 Hz (AFL teratur, AF kacau)
              • entropi spektral 3–12 Hz; tinggi puncak autokorelasi sinyal sisa di jeda FCL
              • amplitudo sinyal sisa relatif terhadap amplitudo R (gelombang flutter/f)
              • konsistensi gelombang P: korelasi rata-rata segmen −0,30…−0,10 s sebelum R antar detak (tinggi pada sinus)
              • rasio konduksi: median(RR) / FCL, jaraknya ke bilangan bulat terdekat, dan fraksi RR yang dekat
                kelipatan bulat FCL (ide "kuantisasi" paten US10925508, kini bisa diuji karena FCL diukur dari EKG)
          B3. (varian 12b) 20 bin log-PSD sinyal sisa 3–12 Hz
   [C] Fusi: input = 3 logit RR + fitur atrium (distandardisasi dari data latih fold)
          12a: regresi logistik multinomial, L2, C = 1, bobot kelas seimbang penuh (alasan di 6.4)
          12b: HistGradientBoosting kedalaman 3, 200 iterasi, + bin log-PSD
   [D] Bias kelas cross-fit (protokol, grid −7…+3) → Normal / AFIB / AFL
```

Kenapa 50 Hz cukup: gelombang flutter ±4–6 Hz dan gelombang-f AF 4–9 Hz jauh di bawah Nyquist 25 Hz. Pembatalan QRST di 50 Hz
memang kasar; kualitasnya diperiksa visual per kelas di gerbang (7.2) sebelum dipakai.

## 6. Data dan pembagian

### 6.1 AFDB

Window dan potongan EKG yang sudah ada dipakai ulang: `cache/dataset_v2_detector.npz` dan `cache/v6/ecg_windows.npz` (sudah
dibuktikan sejajar di v6). Fold resmi 5×5 (`folds.npz`).

### 6.2 SHDB-AF (aturan `SEL`, seed tetap, ditetapkan sebelum melihat hasil)

1. Unduh sinyal 98 rekaman beranotasi (6,71 GB) dari mirror S3 PhysioNet, dicek dengan `SHA256SUMS.txt`, satu rekaman per
   waktu (RAM PC 15 GB). Lisensi dan sitasi dicatat di README.
2. Front-end produksi pada ECG1 (dan ECG2 untuk analisis sensitivitas). Label window = ritme anotasi bila semua detak anotasi di
   window satu ritme; R-peak untuk fitur dari detektor produksi (bukan dari anotasi), dicek kecocokannya dengan anotasi.
3. Kelas: (N → Normal, (AFIB → AFIB, (AFL → AFL. (AT, (PAT, (NOD, (AB tidak dilatih; dilaporkan deskriptif.
4. Batas per subjek: AFL ≤ 1.500 window (mengekang rekaman 010), AFIB ≤ 600, Normal ≤ 400 acak + Normal ≥ 100 bpm ≤ 400
   (sinus takikardia asli saat beraktivitas). Window Normal ≥ 100 bpm tambahan ikut dilatih dan dipakai untuk uji takikardia,
   tetapi tidak masuk metrik 3 kelas (pola v10).
5. Fold **per subjek** (bukan per rekaman; 5 subjek punya dua rekaman), 5 ulangan × 5 fold, distratifikasi kelompok subjek
   (punya AFL / punya AFIB / lainnya), seed 2026 + 3000.

### 6.3 Pembagian latih/uji

Fold (r, k): latih = AFDB fold ≠ k ∪ SHDB-AF fold ≠ k; uji = AFDB fold k dan SHDB-AF fold k. 25 model fusi melayani dua domain.

### 6.4 Logit RR untuk data latih fusi (CV bersarang, wajib)

OOF RR yang ada untuk pasien latih berasal dari model yang pernah melihat pasien uji fold k (kebocoran). Karena itu:
- pasien AFDB di data latih fold (r, k): logit dari model RR CV dalam (4 fold di dalam fold latih, resep v3.1, seed
  `2026 + 100·r + 10·k + j`), total 100 model kecil;
- pasien SHDB-AF: logit dari model RR luar fold (r, k) (dilatih AFDB saja, tidak pernah melihat SHDB-AF);
- uji: model RR luar fold (r, k) dari v8. Uji paritas: OOF model luar harus sama dengan `cache/v8/oof/v31_r*.npz`.

Bobot kelas fusi **penuh** (menyimpang dari aturan bobot lunak AGENTS.md, dengan alasan tercatat): di BiGRU dan TCN v11 bobot
lunak membuat skor AFL 2:1 tipis sehingga 08378 hilang setelah bias, sedangkan bobot penuh memulihkannya; model fusi linear
berkapasitas kecil sehingga risiko overfit dari bobot besar rendah.

## 7. Gerbang sebelum training fusi (ditetapkan sekarang)

1. **Cek fenotipe RR** (sudah, 1.4): AFIB dan Normal kedua dataset konsisten.
2. **Cek fitur atrium tanpa training**: untuk DF dan indeks organisasi, hitung AUROC AFL-vs-AFIB dan AFL-vs-Normal per dataset.
   Lanjut bila arah pemisahannya sama di AFDB dan SHDB-AF dan AUROC ≥ 0,60 di keduanya. Bila arahnya berlawanan (batas AFL/AFIB
   beda, seperti Icentia di v10), berhenti dan laporkan.
3. **Cek visual pembatalan QRST**: contoh sinyal sisa dan spektrum per kelas per dataset (PNG), dilihat sebelum lanjut.

## 8. Varian (ditetapkan sebelum hasil)

| Varian | Fusi | Data latih fusi | Hipotesis |
|---|---|---|---|
| **12a (utama)** | logistik: logit RR + fitur atrium | AFDB + SHDB-AF | AFDB: Macro-F1 > v31 (P ≥ 0,95); recall AFL di luar pola cepat-teratur ≥ 0,40 (v31 ≤ 0,21) |
| 12b | HGB: + bin log-PSD | AFDB + SHDB-AF | eksploratif: apakah spektrum penuh menambah di atas fitur? |
| 12c (kontrol data) | sama dengan 12a | AFDB saja | 12a > 12c → sumbangan SHDB-AF |
| 12d (kontrol fusi) | logistik: logit RR saja | AFDB + SHDB-AF | ≈ v31 (P > 0,05) → kenaikan 12a berasal dari informasi atrium, bukan mekanisme fusi |

## 9. Evaluasi

- **AFDB (utama):** protokol butir 1–8; berpasangan vs v31 dan vs 12d; recall AFL per pasien dan **per pola RR** (tabel 1.1);
  asal AFL palsu; metrik episode (butir 7).
- **SHDB-AF:** metrik 3 kelas per subjek pada fold uji, berdampingan dengan AFDB; pembanding v31 (model RR saja) pada window
  yang sama.
- **Uji takikardia:** uji sintetis v3.2 hanya berisi RR, jadi tidak berlaku untuk cabang EKG. Penggantinya: % window Normal asli
  per pita detak yang ditebak AFIB/AFL, di AFDB dan di sinus takikardia asli SHDB-AF (≥ 100, ≥ 120 bpm).
- **Kandidat (butir 10):** bootstrap bersarang dan tanpa 04936.
- Juga dilaporkan: Macro-F1 2 kelas (AF/AFL vs Normal), dan hasil ECG2 vs ECG1 di SHDB-AF (kepekaan terhadap lead).

### Kriteria kandidat produksi untuk 12a (ditetapkan sekarang)

1. AFDB: lebih baik dari v31 (P ≥ 0,95), tidak ada pasien AFL turun ≥ 0,20, AFL palsu tidak naik > 20%, bias tidak di batas grid.
2. SHDB-AF: AFL F1 lebih tinggi dari v31 pada window yang sama.
3. Sinus takikardia asli SHDB-AF ≥ 120 bpm: alarm tidak lebih tinggi dari v31.
4. Bila lolos: protokol butir 10, lalu keputusan pemilik repo soal pengemasan (backend butuh EKG, bukan hanya IBI).

## 10. Langkah dan biaya

| Langkah | Tempat | Perkiraan |
|---|---|---|
| 1. Persetujuan rancangan ini | – | – |
| 2. Unduh sinyal SHDB-AF (6,71 GB, minta izin dulu) + window + EKG | PC, CPU, per rekaman | ±1–2 jam |
| 3. Fitur atrium + gerbang (7.2, 7.3) | PC | ±1 jam |
| 4. 100 model RR CV dalam | Colab L4, 6 proses paralel | ±1 jam, ±2 compute unit |
| 5. Fusi 12a–12d (25 fold × 4) + evaluasi | PC, CPU | ±1 jam |
| 6. README, notebook ter-eksekusi, laporan PDF | PC | ±2 jam |

## 11. Berkas yang direncanakan

| Berkas | Isi |
|---|---|
| `atrium.py` | Front-end SHDB-AF (pakai `hrv_core`, `v6_cabang_ekg/ekg.py` lewat `sys.path`), pembatalan QRST, fitur atrium, CV dalam RR, fusi, evaluasi |
| `atrium.ipynb` | Data, gerbang, training fusi, hasil AFDB / SHDB-AF, uji takikardia, kesimpulan |
| `laporan/build_report.py` | PDF dua tahap (`--extract`, lalu rakit) |
| `README.md` | Tujuan, berkas, cara menjalankan, hasil, lingkungan |

Keluaran ke `cache/v12/` (window SHDB-AF, fitur, logit RR, OOF fusi). Data SHDB-AF di `data/shdb_af/1.0.1/`.

## 12. Batasan yang sudah diketahui

- Label AFL SHDB-AF dibuat satu orang; label AFDB 04936 juga diperdebatkan (v10). Batas AFL/AFIB antar pelabel bisa berbeda;
  gerbang 7.2 mengujinya sejauh mungkin.
- Pasien AFL total 19 (7 AFDB + 12 SHDB-AF); satu subjek SHDB-AF mendominasi (dibatasi di 6.2).
- Lead AFDB, SHDB-AF (CC5/NASA), dan perangkat SIHEDAF berbeda. Fitur frekuensi/organisasi relatif tidak bergantung lead,
  tetapi amplitudo dan bentuk gelombang P bergantung lead.
- 50 Hz membatasi pembatalan QRST dan bentuk gelombang P.
- Bila berhasil, backend produksi perlu mengirim potongan EKG, bukan hanya IBI.

## 13. Hasil gerbang (2 November 2026)

Data: AFDB 99.000/65.826/1.152 window Normal/AFIB/AFL (7 pasien AFL); SHDB-AF 36.403/33.943/6.609 (12 subjek AFL), tanpa window Normal ≥ 100 bpm tambahan.
Pembaca `.dat` (format 16 dan 212) dan `.atr` dibuktikan sama dengan wfdb; potongan EKG AFDB sama bit demi bit dengan cache v6;
98 rekaman SHDB-AF, 0 window tidak sejajar.

AUROC (AFL positif; < 0,5 = nilai fitur lebih kecil pada AFL). "AFL sulit" = window AFL di luar pola cepat-teratur.

| Fitur | AFDB vs AFIB | AFDB vs Normal | SHDB-AF vs AFIB | SHDB-AF vs Normal | AFL sulit vs AFIB (AFDB / SHDB-AF) |
|---|---|---|---|---|---|
| **`df_hz`** | 0,586 | 0,387 | 0,418 | 0,359 | 0,581 / 0,416 |
| **`organisasi`** | 0,578 | 0,766 | 0,740 | 0,778 | 0,571 / 0,687 |
| `entropi` | 0,262 | 0,214 | 0,237 | 0,226 | 0,363 / 0,295 |
| `autokorelasi` | 0,341 | 0,294 | 0,536 | 0,517 | 0,431 / 0,641 |
| `amp_atrium` | 0,476 | 0,569 | 0,531 | 0,669 | 0,404 / 0,433 |
| `rasio_energi` | 0,428 | 0,479 | 0,590 | 0,683 | 0,392 / 0,504 |
| `p_konsisten` | 0,965 | 0,653 | 0,797 | 0,355 | 0,916 / 0,740 |
| `rasio_konduksi` | 0,329 | 0,121 | 0,483 | 0,238 | 0,412 / 0,576 |
| `jarak_bulat` | 0,193 | 0,375 | 0,297 | 0,426 | 0,206 / 0,353 |
| `kuantisasi` | 0,845 | 0,682 | 0,747 | 0,588 | 0,859 / 0,691 |
| `n_detak` | 0,905 | 0,966 | 0,465 | 0,708 | 0,773 / 0,321 |

Keputusan menurut aturan 7.2 (fitur gerbang ditebalkan):

- `df_hz vs_afib`: AFDB 0,586, SHDB-AF 0,418, searah tidak, ≥ 0,60 di keduanya tidak → **gagal**
- `df_hz vs_normal`: AFDB 0,387, SHDB-AF 0,359, searah ya, ≥ 0,60 di keduanya ya → **lolos**
- `organisasi vs_afib`: AFDB 0,578, SHDB-AF 0,740, searah ya, ≥ 0,60 di keduanya tidak → **gagal**
- `organisasi vs_normal`: AFDB 0,766, SHDB-AF 0,778, searah ya, ≥ 0,60 di keduanya ya → **lolos**

**Keputusan gerbang: BERHENTI.** Training fusi tidak dijalankan.

Cek visual 7.3 (`cache/v12/gerbang/contoh_*.png`): pembatalan QRST di 50 Hz menyisakan lonjakan QRS besar di setiap R-peak, sehingga
spektrum sinyal sisa didominasi harmonik detak jantung (mis. AFL 2:1 08378 terbaca DF 7–8 Hz, padahal siklus flutter ±4,6 Hz).
Gelombang flutter tetap terlihat di sebagian contoh (SHDB-AF 036: DF 5,3 Hz, organisasi 0,57, rasio konduksi 2,02).
Fitur di luar gerbang (`p_konsisten`, `kuantisasi`, `entropi`, `jarak_bulat`) memisahkan AFL dari AFIB searah di kedua dataset,
tetapi fitur itu dilihat **setelah** hasil, jadi tidak boleh dipakai untuk meloloskan gerbang tanpa revisi rancangan yang tercatat.

## 14. Revisi 1: pembatalan QRST (ditulis 2 November 2026, sebelum dijalankan; disetujui pemilik repo)

Alasan: cek visual 7.3 menunjukkan lonjakan QRS tersisa di setiap R-peak. Penyebab yang paling mungkin: R-peak 50 Hz hanya tepat
sampai ±10 ms, sedangkan lereng QRS curam, sehingga templat yang digeser setengah sampel meninggalkan sisa besar; amplitudo QRS juga
berubah antar detak. Revisi ini **hanya mengubah B1** (cara menghitung sinyal sisa). Fitur B2, window, data, dan gerbang 7.2 (fitur
`df_hz` dan `organisasi`, AFL vs AFIB dan AFL vs Normal, searah dan ≥ 0,60 di kedua dataset) **tidak berubah**.

B1 revisi 1, per window (input tetap EKG 50 Hz, kondisi perangkat):
1. Potongan window ± margin di-upsample ×5 ke 250 Hz (`scipy.signal.resample_poly`), tanpa menambah informasi di atas 25 Hz.
2. Setiap R-peak diperhalus ke puncak |x| terdekat dalam ±10 ms (penyelarasan sub-sampel 50 Hz).
3. Templat = median detak yang sudah diselaraskan (−0,10…+0,40 s); dikurangkan dengan skala amplitudo per detak (kuadrat terkecil,
   dibatasi 0,5–1,5).
4. Daerah QRS −40…+60 ms di sekitar setiap R-peak pada sinyal sisa diganti interpolasi linear (sisa QRS tidak masuk spektrum).
5. Sinyal sisa di-downsample kembali ke 50 Hz; fitur B2 dihitung dengan kode yang sama.

Aturan:
- Hanya **satu** revisi. Perbaikan bug yang membuat kode sesuai teks di atas boleh; perubahan metode lain tidak.
- Cek visual 7.3 diulang sebelum keputusan, hanya untuk memastikan implementasi benar (bukan untuk menyetel).
- Bila gerbang 7.2 gagal lagi, **v12 berhenti**; hasil ditulis di README v12.
- Fitur yang terlihat baik setelah hasil gerbang pertama (`p_konsisten`, `kuantisasi`) tetap bukan fitur gerbang.

**Hasil revisi 1 (2 November 2026):** implementasi bekerja (lonjakan QRS hilang; AFL 2:1 08378 DF 4,5–4,7 Hz), tetapi gerbang
tetap gagal pada AFL vs AFIB (`df_hz` 0,458 / 0,571, arah berlawanan; `organisasi` 0,486 / 0,608). Sesuai aturan, v12 berhenti.
Rincian di README.md.
