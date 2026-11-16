# Kerangka paper: evaluasi kritis klasifikasi AF/AFL berbasis interval RR

> **Keputusan 6 November 2026: Jalur B, dua paper.** C1–C3 + C5 ke ICIMPACT 2027 (30 Nov 2026), C4 ke MeMeA atau EMBC 2027
> (±24 Jan 2027). Pembagian isi, aturan anti tumpang tindih, dan jadwal: [rencana_dua_paper.md](rencana_dua_paper.md).

Status: **draf kerangka (6 November 2026)**. Semua angka diambil dari README/laporan versi yang disebut di kolom "Sumber"; sebelum
masuk naskah, angka dihitung ulang dari `cache/` (aturan laporan di AGENTS.md). Daftar pustaka di bagian 9 belum diverifikasi satu per
satu.

## 1. Judul kerja dan pesan utama

**Judul kerja (Inggris):**
*Passing the AFDB benchmark is not enough: evaluation pitfalls and sinus-tachycardia false alarms in RR-interval atrial
fibrillation/flutter classifiers for a 50 Hz single-lead device*

Alternatif lebih pendek: *What the MIT-BIH AFDB benchmark misses: lessons from developing an RR-interval AF/AFL classifier*.

**Pesan utama dalam satu kalimat:** angka performa klasifikasi AF/AFL dari interval RR di AFDB mudah menggelembung, perbedaan antar
arsitektur tenggelam dalam noise, dan model yang lolos protokol AFDB tetap mengalarmkan hampir semua sinus takikardia asli, kegagalan
yang tidak bisa terlihat di AFDB.

**Bukan klaim paper ini:** model baru yang lebih baik. Jangan dijual sebagai "novel architecture".

## 2. Kontribusi (yang diklaim baru)

| # | Kontribusi | Bukti utama | Sumber |
|---|---|---|---|
| C1 | Anatomi angka yang menggelembung: klaim 0,826 bisa direproduksi, lalu selisihnya dengan sistem yang dipakai (0,418) didekomposisi per faktor dengan model yang sama | window uji −0,114; R-peak 50 Hz −0,016; front-end −0,325 | `publikasi/dekomposisi_v1.py` |
| C2 | Arsitektur bukan pembatas: 10 model (9 arsitektur) dengan protokol berpasangan yang sama tidak ada yang mengalahkan referensi; bobot kelas dan kalibrasi bias lebih menentukan | v3: 4 kandidat; v11: 6 model | `v3_evaluasi_robust/`, `v11_model_tim/` |
| C3 | Kerapuhan kesimpulan: keunggulan bergantung satu pasien (04936, 58% window AFL), P turun saat bias ikut tidak pasti (bootstrap bersarang) | P 0,98 → 0,94 tanpa 04936; P 1,00 → 0,993 bersarang | `v3.2_uji_takikardia/` |
| C4 | Uji stres sinus takikardia: semua model RR yang dilatih di AFDB saja mengalarmkan 89–100% sinus takikardia asli ≥ 120 bpm; AFDB hanya punya 0,1% window Normal > 120 bpm sehingga protokol standar buta terhadapnya | sintetis + EKG latihan 41 subjek | `v3.2_uji_takikardia/`, `v3.3_takikardia_asli/` |
| C5 | Daftar periksa evaluasi yang bisa dipakai ulang (protokol 10 butir) + kode | `sihedaf_cv.py`, `takikardia.py` | AGENTS.md "Protokol evaluasi" |

Posisi terhadap literatur (perlu tinjauan sistematis sebelum klaim "belum ada"):
- Split per pasien dan kebocoran data sudah dibahas (de Chazal 2004; Domazetoski & Gligorić 2022; arXiv 1804.02543; Kapoor &
  Narayanan 2023). Jadi C1 adalah studi kasus terkuantifikasi, bukan temuan konsep baru.
- AF/AFL dari RR dengan DL sudah ada (Ivanović 2019; Faust 2021). Paper ini tidak bersaing di akurasi.
- Bahwa AFL 2:1 teratur dan mirip sinus takikardia dari RR sudah dikenal secara fisiologis. Yang belum ditemukan dalam pencarian
  singkat: kuantifikasi alarm palsu sinus takikardia pada model AFL berbasis RR, termasuk bukti bahwa protokol AFDB tidak bisa
  menangkapnya. Ini inti kebaruan (C4).

## 3. Draf abstrak (Inggris, ±250 kata, angka sementara)

> **Background.** RR-interval classifiers for atrial fibrillation (AF) and atrial flutter (AFL) are commonly developed and
> evaluated on the MIT-BIH Atrial Fibrillation Database (AFDB). We report lessons from iteratively developing such a classifier
> for a 50 Hz single-lead device.
> **Methods.** Ten-second windows from 23 AFDB records were classified as normal, AF or AFL. Ten models spanning nine
> architectures (transformer, recurrent, convolutional, kernel and tree-based) were compared with 5×5 repeated patient-level
> cross-validation, patient-bootstrap confidence intervals, paired comparisons, cross-fitted class-bias calibration and nested
> bootstrap. Models were stress-tested with synthetic regular rhythms (60–170 bpm) and with exercise ECG from 41 healthy
> subjects processed through the device pipeline.
> **Results.** An initially reported macro-F1 of 0.826 was reproducible (0.828) but fell to 0.759 (95% CI 0.66–0.83) once test
> windows were sampled uniformly (densely overlapping AFL windows had doubled AFL precision), to 0.744 (0.64–0.81) with 50 Hz
> detector R-peaks, and to 0.418 (0.36–0.48) with the originally deployed R-peak front-end. After recalibration the reference
> system reached 0.832 (0.67–0.92); no architecture improved on it, and class weighting mattered more than architecture. Its
> advantage over the initial pipeline depended on one patient contributing 58% of AFL windows. All RR-only models trained on AFDB
> labelled regular rhythms above 110–140 bpm as AF/AFL and raised alarms on 89–100% of real sinus-tachycardia windows ≥120 bpm,
> a failure invisible in AFDB, where 0.1% of normal windows exceed 120 bpm. Conformal abstention flagged none of these errors.
> **Conclusions.** AFDB-only evaluation of RR-interval AF/AFL classifiers overstates readiness for wearable use. We propose a
> reporting checklist including patient-level uncertainty, single-patient sensitivity and stress tests for conditions absent from
> the test set.

## 4. Peta nama versi → nama di naskah

Nama folder internal (v1, v3.1, …) jangan muncul di naskah.

| Repo | Nama di naskah | Isi |
|---|---|---|
| v1 | P0, *initial pipeline* | R-peak anotasi 250 Hz, MicroTransformer, front-end `find_peaks` |
| v2 | P1, *revised pipeline* | R-peak detektor 50 Hz, praproses bersama, bias grid −3…+3 (terpotong) |
| v3.1 | P2, *reference* | bobot P1, bias cross-fit grid −7…+3 |
| v3 (4 kandidat) + v11 (6 model) | *model zoo* | LightGBM, MicroTransformer, ResNet1D, BiGRU; BiGRU, TCN, InceptionTime, MiniRocket, XGBoost, SVM |
| v7a, v8c | *+synthetic ST*, *+external ST* | pembanding: data latih yang memuat sinus takikardia |
| v4 | (suplemen) | smoothing antar-window: contoh metrik window menyembunyikan hilangnya episode |

## 5. Struktur naskah

### 5.1 Introduction (±1 halaman)
1. Deteksi AF di wearable; AFL penting tetapi jarang; RR-interval murah dan cocok untuk 50 Hz.
2. AFDB adalah benchmark de facto; 23 pasien, AFL hanya di 7 pasien.
3. Masalah: angka tinggi di literatur sering dari split non-pasien; kondisi pemakaian (aktivitas fisik) tidak ada di AFDB.
4. Pertanyaan penelitian:
   - RQ1: seberapa besar angka berubah saat evaluasi diperketat? (C1)
   - RQ2: apakah pilihan arsitektur penting di bawah evaluasi berpasangan? (C2, C3)
   - RQ3: bagaimana perilaku model pada sinus takikardia, yang tidak ada di AFDB? (C4)
5. Daftar kontribusi C1–C5.

### 5.2 Related work (±0,5 halaman)
- Detektor AF berbasis RR klasik: Tateno & Glass 2001, Lake & Moorman 2011 (COSEn), Petrėnas 2015.
- DL berbasis RR: Faust 2018, Andersen 2019, Ivanović 2019 (AF+AFL), Faust 2021 (3 kelas).
- Evaluasi & kebocoran: de Chazal 2004 (paradigma antar pasien), Domazetoski & Gligorić 2022, arXiv 1804.02543, Kapoor &
  Narayanan 2023, TRIPOD+AI 2024.
- Alarm palsu wearable: Apple Heart Study (Perez 2019), studi PPG/jam pintar saat aktivitas.

### 5.3 Methods (±2 halaman)
**Data**
- AFDB: 23 rekaman (00735, 03665 dikeluarkan), window 10 s, stride 5 s, label Normal/AFIB/AFL, dua sumber R-peak (anotasi 250 Hz dan
  detektor 50 Hz). Jumlah window detektor: 99.000 / 65.826 / 1.152. Audit detektor: sensitivitas rata-rata per rekaman 0,944, PPV
  0,940; 07162 menyimpang (0,205). Sumber: `v2_revisi/model_export/sihedaf_v2.json`, laporan v2 hal. 2 dan 7.
- Data uji stres (subjek sehat, diasumsikan sinus): EPFL high-intensity (20 subjek, CC BY 4.0), PhysioNet Wrist (8 subjek, ODC-BY),
  EPHNOGRAM (24 subjek, ODC-BY). 2.189 window ≥ 120 bpm dari 41 subjek. Sumber: `v3.3_takikardia_asli/README.md`.
- (Opsional) Chapman–Shaoxing: 6.937 pasien sinus takikardia; paket P2 mengalarmkan 42% (98–100% di ≥ 120 bpm). Sumber:
  `v8_chapman_shaoxing/README.md`.

**Model**
- Input urutan: IBI dan ΔIBI, maksimum 30 detak, mask padding. Input tabular: 16 fitur HRV.
- 10 model, resep training seragam; bobot kelas lunak (`class_weights ** 0.5`) dan penuh (`wp100`).

**Protokol evaluasi** (inti metode, beri satu gambar alur)
- 5 ulangan × 5 fold pasien, pembagian round-robin per strata pasien (bukan `StratifiedGroupKFold(shuffle=True)`, lihat 5.4).
- Pooled OOF dirata-rata antar ulangan; CI 95% bootstrap pasien 2.000×, stratified.
- Perbandingan berpasangan dengan sampel bootstrap yang sama; vonis P(Δ>0) ≥ 0,95 / ≤ 0,05.
- Bias kelas cross-fit antar fold, grid −7…+3; bootstrap bersarang (bias dipilih ulang tiap sampel).
- Sensitivitas tanpa 04936; recall AFL per pasien; metrik episode gaya ANSI/AAMI EC57.

**Uji stres takikardia**
- Sintetis: ritme teratur 60–170 bpm × 4 jenis variasi (tanpa, seragam ±10 ms, Gauss 10, Gauss 20), 50 window per titik; ukuran =
  bpm terendah di mana < 50% window ditebak Normal.
- Window Normal AFDB asli per pita detak.
- EKG latihan asli lewat pipeline perangkat (detektor 50 Hz) dan lewat R-peak manual; CI bootstrap subjek berpasangan.
- Abstain conformal (LAC, cakupan 90%) dan aturan produk (≥ 110 bpm & CV IBI ≤ 0,05) sebagai mitigasi.

### 5.4 Results

**R1. Anatomi angka yang menggelembung (C1)** — Tabel 2, Gambar 2

Dekomposisi dihitung 6 November 2026 dengan `publikasi/dekomposisi_v1.py`. Hasilnya ada di `cache/publikasi/dekomposisi_v1/ringkasan.json`,
log di `logs/dekomposisi_v1.log`. Resep P0 (MicroTransformer, weighted CE, argmax) dilatih di window ekstraksi P0, lalu **model fold yang
sama** diuji di setiap set uji. Jadi selisih antar langkah hanya berasal dari cara mengukur.

Langkah A–D memakai rata-rata per fold (± = SD antar fold). Langkah E–J memakai pooled OOF rata-rata 5 ulangan dengan CI 95% bootstrap
pasien, dan Δ terhadap langkah sebelumnya dihitung berpasangan.

| Langkah | Kondisi | Macro-F1 | Δ vs langkah sebelumnya | AFL P | AFL R |
|---|---|---|---|---|---|
| A | Klaim `model.ipynb`: partisi P0, rata-rata per fold, window P0 | 0,826 ± 0,084 | – | – | – |
| B | Diulang: partisi dan resep yang sama | 0,828 ± 0,086 | +0,002 | – | – |
| D | 5×5 fold pasien, rata-rata per fold | 0,803 ± 0,130 | −0,025 | – | – |
| E | Pooled OOF, window P0 | 0,873 [0,72–0,92] | +0,070 | 0,745 | 0,815 |
| F | Window AFL uji dijarangkan dari stride 1 s ke ≥ 5 s | 0,786 [0,66–0,85] | −0,087 [−0,115; −0,052], P = 0,000 | 0,371 | 0,812 |
| G | Window AFIB uji diberi bobot 2 (emulasi stride 5 s; P0 memakai 10 s) | 0,759 [0,66–0,83] | −0,026 [−0,038; +0,002], P = 0,033 | 0,264 | 0,812 |
| H | Ekstraksi P1 dengan R-peak anotasi 250 Hz (stride seragam 5 s) | 0,759 [0,66–0,83] | −0,000 (P = 0,39) | 0,263 | 0,812 |
| I | Ekstraksi P1 dengan R-peak detektor 50 Hz | 0,744 [0,64–0,81] | −0,016 [−0,028; −0,006], P = 0,000 | 0,275 | 0,728 |
| J | Front-end lama `find_peaks` di sinyal 50 Hz, tanpa cek kualitas | 0,418 [0,36–0,48] | −0,325 [−0,427; −0,200], P = 0,000 | 0,027 | 0,410 |

Temuan:
- **Klaimnya bisa direproduksi (B) dan resepnya jujur**: tidak ada pasien yang bocor antar fold.
- **Partisi tunggal**: berganti ke 5×5 fold pasien menurunkan 0,025, dan SD antar fold 0,13. Satu partisi tidak cukup.
- **Pooled vs rata-rata per fold**: +0,070. Ini bukan inflasi, melainkan estimator yang berbeda. Fold yang hanya punya sedikit window
  AFL membuat F1 AFL per fold tidak stabil. Pelajarannya: angka bisa bergeser 0,07 hanya karena cara merata-rata, jadi cara itu wajib
  ditulis.
- **Inflasi terbesar datang dari konstruksi window uji (−0,114)**:
  - Window AFL yang tumpang tindih 90% melipatgandakan presisi AFL (0,371 → 0,745) tanpa mengubah recall (0,812 vs 0,815). Ini efek
    prevalensi: satu episode AFL dihitung ±5 kali.
  - Stride AFIB 10 s menyumbang −0,026.
- **Koreksi window terbukti tepat**: setelah dua koreksi itu, hasilnya sama persis dengan ekstraksi P1 (H, Δ 0,000). Jumlah window AFL
  juga identik: 991 (510 dari 04936) di kedua set.
- **Sumber R-peak hanya −0,016**: R-peak anotasi 250 Hz vs detektor 50 Hz, untuk window yang lolos cek kualitas. Yang turun terutama
  recall AFL (0,812 → 0,728).
- **Front-end adalah kerugian terbesar (−0,325)**: `find_peaks` tanpa filter dan cek kualitas membuat presisi AFL tinggal 0,027. Jarak
  antara angka offline dan sistem yang benar-benar dipakai paling banyak berasal dari sini, bukan dari model.
- **Contoh kedua**: P1 melaporkan 0,792 dari satu partisi; dengan protokol penuh hasilnya 0,779 [0,63–0,82].

**Demo partisi.** Dengan 100 seed, `StratifiedGroupKFold(shuffle=True)` menghasilkan **satu** partisi pasien yang sama, baik di window P0
maupun P1 (scikit-learn 1.9.1). Mekanismenya terlihat di kode sumber: `shuffle` hanya mengacak urutan pasien yang SD distribusi kelasnya
sama persis (`argsort` stabil). Karena jumlah window per pasien selalu berbeda, ikatan seperti itu tidak pernah terjadi. Akibatnya
"ulangan dengan seed berbeda" sebenarnya partisi yang sama. Pembagian round-robin per strata (v3) menghasilkan 5 partisi berbeda untuk
5 ulangan.

Perilaku ini **bergantung versi**, jadi versinya wajib ditulis di naskah:
- Batasan ini memang terdokumentasi ("can only shuffle groups that have approximately the same y class distribution"), tetapi kata
  "approximately" menyesatkan.
- Di scikit-learn ≤ 1.7 ada bug: hitungan kelas diacak tanpa ikut mengacak grup. Akibatnya partisi memang berganti antar seed, tetapi
  acak dan tidak terstratifikasi. Bug ini diperbaiki di 1.8.0 (issue #32478, PR #32540).

Langkah B terbukti memakai partisi asli P0. Pola skor per fold cocok dengan log `model.ipynb`:
- Macro-F1: 0,758 / 0,901 / 0,775 / 0,940 / 0,765 vs 0,743 / 0,895 / 0,782 / 0,953 / 0,755.
- Recall AFL: 0,24 / 1,00 / 0,87 / 0,97 / 0,79 vs 0,20 / 1,00 / 0,86 / 0,98 / 0,78.

**R2. Arsitektur bukan pembatas (C2)** — Gambar 3 (forest plot), Tabel 3

- Referensi P2: 0,832 [0,67–0,92], AFL F1 0,647 [0,17–0,89].
- Resep sama: BiGRU 0,821 (setara); TCN, InceptionTime, XGBoost, MiniRocket lebih buruk; SVM tidak terbedakan.
- Bobot kelas penuh: BiGRU dan TCN 0,830 (setara referensi, juga tanpa 04936 dan bersarang).
- ±80% kenaikan P0 → P2 berasal dari kalibrasi bias kelas.
- Balancing under/over lebih buruk dari bobot kelas (BiGRU `under` −0,223).
- Jebakan grid bias: grid −3…+3 memotong optimum (bias AFL optimal −4…−5,5); pencarian mengompensasi lewat AFIB +2 dan recall
  Normal turun ke ±0,85.
- Per pasien: 07910 dan 08215 gagal di semua model (Gambar 5).

Sumber: `v3_evaluasi_robust/README.md`, `v11_model_tim/README.md` "Hasil", AGENTS.md "Anti-pattern".

**R3. Kerapuhan kesimpulan (C3)** — Tabel 4

| Uji | P2 vs P1 | P2 vs P0 |
|---|---|---|
| Bias tetap, semua pasien | P = 1,00 | P = 0,98 |
| Bootstrap bersarang | Δ +0,053, P = 0,993 | Δ +0,088, P = 0,978 |
| Tanpa 04936 | P = 1,00 | P = 0,94 (tidak terbedakan) |

Sumber: `v3.2_uji_takikardia/README.md`.

**R4. Sinus takikardia (C4)** — Gambar 6, Gambar 7, Tabel 5
- AFDB: 100 window Normal ≥ 120 bpm dari 99.000 (0,1%), dan itu pun tidak teratur (CV median 0,18).
- Sintetis: ambang P0/P1/P2 110–115 bpm; model zoo v11 115–140 bpm.
- Window Normal AFDB ≥ 120 bpm: 94% / 100% / 95% dialarmkan (P0/P1/P2).
- EKG latihan asli (perangkat, ≥ 120 bpm): P0 98,0% [96–100], P1 99,9%, P2 98,4% [97–100]; BiGRU/TCN 88,9–96,0%.
- Hanya data latih dengan sinus takikardia yang menurunkan alarm (+synthetic ST 63,9%, +external ST 60,3%), dengan harga AFL 2:1
  (08378: 0,96 → 0,71 untuk +synthetic ST).
- Mitigasi: abstain conformal 0% "tidak yakin" (model yakin dalam kesalahannya); aturan produk menangkap 61–83% alarm tetapi
  mengalihkan 86% AFL yang terdeteksi.
- Detektor 50 Hz saat aktivitas: EPFL 68% window lolos cek kualitas (R-peak manual 95%); alarm pada < 100 bpm 21–29% vs 9% dengan
  R-peak manual.

Sumber: `v3.2_uji_takikardia/README.md`, `v3.3_takikardia_asli/README.md`, `v7_augmentasi_takikardia/`, `v8_chapman_shaoxing/`.

**R5 (suplemen). Metrik window menyembunyikan episode:** smoothing menurunkan alarm palsu 0,51 → 0,09/jam dengan Macro-F1
"tidak terbedakan", padahal deteksi episode AFL turun 74% → 38%. Sumber: `v4_smoothing_ensemble/README.md`.

### 5.5 Discussion (±1,5 halaman)
1. Mengapa protokol AFDB buta: masalah cakupan data uji, bukan model. RR saja tidak memisahkan AFL 2:1 dari sinus takikardia.
2. Implikasi untuk literatur: angka tinggi tanpa split pasien, CI pasien, dan uji stres tidak bisa dibandingkan.
3. Daftar periksa (Tabel 6): split pasien berulang; pooled OOF + CI bootstrap pasien; perbandingan berpasangan; bias cross-fit
   dengan grid yang dicek batasnya; bootstrap bersarang; sensitivitas tanpa pasien dominan; recall per pasien; metrik episode; uji
   dengan front-end yang benar-benar dipakai; uji stres untuk kondisi yang tidak ada di data uji.
4. Jalan keluar yang terlihat (singkat, rujuk pekerjaan lanjutan): data latih dengan sinus takikardia (ada trade-off AFL 2:1),
   informasi bentuk gelombang (model fondasi EKG; v16.2, paper terpisah).

### 5.6 Limitations
- AFDB 23 pasien, AFL dari 7 pasien; CI lebar (AFL F1 0,17–0,89).
- Data latihan fisik: subjek sehat tanpa label ritme, diasumsikan sinus.
- Belum ada rekaman perangkat SIHEDAF berlabel.
- Studi kasus satu tim; generalisasi dibantu oleh model zoo 9 arsitektur, tetapi resep data sama.
- Hasil antar mesin bisa sedikit berbeda (GPU non-deterministik); semua perbandingan memakai cache OOF yang sama.

### 5.7 Conclusion (1 paragraf)

## 6. Gambar dan tabel

| No | Isi | Data |
|---|---|---|
| Gambar 1 | Alur sistem 50 Hz + protokol evaluasi | skema |
| Gambar 2 | Waterfall langkah A–J (0,826 → 0,418) dengan CI, lalu P2 0,832 sebagai pembanding | `cache/publikasi/dekomposisi_v1/ringkasan.json` |
| Gambar 3 | Forest plot Δ Macro-F1 vs P2 + CI, marker berbeda per vonis | v3, v11 |
| Gambar 4 | Permukaan Macro-F1 terhadap bias AFIB/AFL, kotak grid lama −3…+3 | OOF v3 |
| Gambar 5 | Heatmap recall AFL per pasien (7 pasien × model) | v3, v11 |
| Gambar 6 | % ditebak Normal vs bpm (sintetis, 4 jenis variasi) + alarm per pita detak EKG latihan | v3.2, v3.3 |
| Gambar 7 | Trade-off: alarm sinus takikardia ≥ 120 bpm vs recall AFL 08378 per model | v3.3, v7, v8, v11 |
| Tabel 1 | Dataset & lisensi | |
| Tabel 2 | Anatomi angka (R1) | |
| Tabel 3 | Model zoo | |
| Tabel 4 | Ketahanan kesimpulan (R3) | |
| Tabel 5 | Alarm sinus takikardia per model & sumber | |
| Tabel 6 | Daftar periksa evaluasi | |

Palet & aturan grafik mengikuti AGENTS.md "Gaya laporan" (warna `#2a78d6`, `#eb6834`, `#1baf7a` + bentuk marker).

## 7. Yang masih harus dikerjakan sebelum menulis

1. ~~Dekomposisi R1 per faktor~~ **Selesai 6 November 2026** (`publikasi/dekomposisi_v1.py`, lihat R1). Prediksi run baru identik
   100% dengan OOF cache v3 (argmax sama), dan angka E/I/J sama dengan laporan v3.
2. ~~Demonstrasi `StratifiedGroupKFold(shuffle=True)`~~ **Selesai**: 100 seed menghasilkan 1 partisi (scikit-learn 1.9.1), lihat R1.
3. **Satu skrip ekstraksi angka paper** dari `cache/` dengan `assert` terhadap ringkasan tiap versi (pola `extract()` v3).
4. **Tinjauan pustaka sistematis** (Scopus, PubMed, IEEE Xplore) untuk menguatkan klaim C4: kata kunci "sinus tachycardia", "false
   positive", "RR interval", "atrial flutter", "exercise".
5. **Cek lisensi & sitasi dataset**: AFDB (ODC-BY), EPFL (CC BY 4.0), Wrist & EPHNOGRAM (ODC-BY), Chapman (cek). Semua data publik,
   tanpa subjek baru, jadi kemungkinan tidak perlu ethical clearance; konfirmasi ke pembimbing.
6. **Ketersediaan kode**: putuskan apakah `sihedaf_cv.py`, `takikardia.py`, `takikardia_asli.py` dirilis (pemilik repo).
7. **Daftar periksa TRIPOD+AI** diisi sebagai lampiran.
8. Sepakati penulis & urutan, pembimbing, dan target venue.

## 8. Target venue (cek tenggat dan ketentuan terbaru)

> **Diperbarui 6 November 2026:** hasil riset kelayakan dan daftar konferensi EDAS ada di
> [riset_kelayakan_dan_venue.md](riset_kelayakan_dan_venue.md). Kandidat utama: IEEE MeMeA 2027 (tenggat 24 Januari 2027) atau EMBC 2027,
> dan ICIMPACT 2027 Telkom (30 November 2026) bila memilih dua paper. Tabel di bawah adalah rencana awal.

| Pilihan | Format | Catatan |
|---|---|---|
| Computing in Cardiology (CinC) | 4 halaman, konferensi | komunitas AF/EKG; umpan balik cepat; muat C1 + C4 saja |
| Physiological Measurement (IOP) | jurnal penuh | cocok untuk paper metodologi evaluasi |
| PLOS Digital Health / JMIR Cardio | jurnal akses terbuka | cek biaya APC |
| Jurnal nasional terakreditasi SINTA | jurnal penuh, bisa Bahasa Indonesia | versi sistem + evaluasi; cek peringkat SINTA |

Saran: versi CinC (C1 + C4) dulu, lalu versi jurnal penuh dengan C2, C3, C5.

## 9. Pustaka awal (verifikasi sebelum dikutip)

- Moody GB, Mark RG. A new method for detecting atrial fibrillation using R-R intervals. Computers in Cardiology 1983 (AFDB).
- Goldberger AL dkk. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 2000.
- Tateno K, Glass L. Automatic detection of atrial fibrillation using the coefficient of variation and density histograms of RR
  and ΔRR intervals. Med Biol Eng Comput 2001.
- Lake DE, Moorman JR. Accurate estimation of entropy in very short physiological time series. Am J Physiol Heart 2011.
- Petrėnas A dkk. Low-complexity detection of atrial fibrillation in continuous long-term monitoring. Comput Biol Med 2015.
- Faust O dkk. Automated detection of atrial fibrillation using LSTM network with RR interval signals. Comput Biol Med 2018.
- Andersen RS dkk. A deep learning approach for real-time detection of atrial fibrillation. Expert Syst Appl 2019.
- Ivanović MD dkk. Deep learning approach for highly specific atrial fibrillation and flutter detection based on RR intervals.
  EMBC 2019. https://www.ncbi.nlm.nih.gov/pubmed/31946242
- Faust O dkk. (ResNet RR 3 kelas AFIB/AFL/NSR). Diagnostics 2021;11(8):1446. https://www.mdpi.com/2075-4418/11/8/1446
- de Chazal P dkk. Automatic classification of heartbeats using ECG morphology and heartbeat interval features. IEEE TBME 2004.
- Domazetoski V, dkk. The influence of atrial flutter in automated detection of atrial arrhythmias – are we ready to go into clinical
  practice? Comput Methods Programs Biomed 2022;221:106901. doi:10.1016/j.cmpb.2022.106901 (link arXiv 2206.14200 yang dulu ditulis
  di sini salah; itu paper lain)
- Butkuvienė M, dkk. Considerations on performance evaluation of atrial fibrillation detectors. IEEE TBME 2021;68(11):3250–3260.
  doi:10.1109/TBME.2021.3067698 (karya terdekat untuk C1; wajib dibahas)
- Not quite unreasonable effectiveness of machine learning algorithms. https://arxiv.org/abs/1804.02543
- Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns 2023.
- Collins GS dkk. TRIPOD+AI statement. BMJ 2024.
- Perez MV dkk. Large-scale assessment of a smartwatch to identify atrial fibrillation. NEJM 2019.
- Angelopoulos AN, Bates S. A gentle introduction to conformal prediction. 2021.
- Dataset uji stres: EPFL high-intensity (Zenodo 5727800); PhysioNet `wrist`; PhysioNet `ephnogram`; Chapman–Shaoxing
  (PhysioNet `ecg-arrhythmia`).
