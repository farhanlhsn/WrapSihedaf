# Riset kelayakan paper evaluasi kritis + pilihan konferensi EDAS

Tanggal: **6 November 2026**. Pendamping [kerangka_paper_evaluasi.md](kerangka_paper_evaluasi.md).

Cara riset:
- Dua penelusuran literatur paralel: ±70 pencarian web, arXiv, PubMed, dan Exa, ditambah ±40 halaman paper, GitHub, dan IFU produk yang
  dibaca langsung.
- Daftar konferensi diambil dari EDAS "Conferences accepting submissions", dibuka lewat akun EDAS pemilik repo: 210 konferensi dan jurnal.
- Ini **bukan tinjauan sistematis**. Scopus, WoS, dan teks lengkap berbayar (IEEE Xplore, ScienceDirect) tidak bisa diakses. Sebelum menulis
  "belum ada yang meneliti X", cek ulang di Scopus atau PubMed.

## 1. Vonis singkat

**Layak dipublikasikan, asalkan dibingkai sebagai paper evaluasi atau kuantifikasi, bukan model baru.** Kekuatan tiap klaim:

| Klaim | Kebaruan | Catatan |
|---|---|---|
| **C4 sinus takikardia** | **Paling kuat.** Tidak ditemukan studi yang mengukur alarm sinus takikardia (ST) pada model RR-only hasil latih AFDB per pita bpm, dengan sapuan ritme teratur sintetis dan EKG latihan asli, lalu mengaitkannya dengan kebutaan protokol AFDB. | Bahan dasarnya sudah dikenal (lihat 2.1). Bingkai sebagai kuantifikasi dan kritik benchmark, bukan konsep baru. |
| **C1 dekomposisi angka** | **Sedang–kuat.** Kebocoran split pasien sudah banyak dibahas. Yang baru: dekomposisi dengan **model fold yang sama**, inflasi **tanpa** kebocoran pasien (konstruksi window uji per kelas), dan front-end produk −0,325 vs sumber R-peak 50 Hz −0,016. | Butkuvienė 2021 dan Domazetoski 2022 wajib dibahas di awal Related Work. |
| C2 arsitektur bukan pembatas | Rendah (konsepnya sudah diketahui) | Konfirmasi di setting RR 3 kelas. Dengan 23 pasien, "tidak ada yang lebih baik" ≠ setara. |
| C3 kerapuhan kesimpulan | Rendah–sedang | Contoh konkret yang berguna. Bootstrap bersarang = konsep BBC-CV (Tsamardinos 2018), jadi jangan diklaim sebagai metode baru. |
| C5 daftar periksa | Sedang | Perluasan rekomendasi Butkuvienė 2021 dan TRIPOD+AI, dengan butir yang belum ada di sana. |

**Kekuatan yang sudah dimiliki:**
- Semua data publik dengan lisensi yang membolehkan publikasi: AFDB ODC-BY, EPFL CC BY 4.0, Wrist dan EPHNOGRAM ODC-BY, Chapman CC BY 4.0.
- Kode dan cache bisa direproduksi. Dekomposisi run ulang identik 100% dengan cache v3.
- 10 model dari 9 arsitektur diuji dengan protokol berpasangan yang sama.
- EKG latihan asli dari 41 subjek.
- Pemisahan efek model dari efek detektor sudah ada: dengan R-peak manual (EPFL direview dokter, 188 window ≥ 120 bpm; Wrist referensi, 114
  window), paket v3.1 tetap mengalarmkan **100%**. Sumber: `cache/v3_3/ringkasan.json`, `aturan_asli/*/manual`.

## 2. Posisi terhadap literatur

### 2.1 Untuk C4 (sinus takikardia)

> **Koreksi 6 November 2026** (verifikasi sumber untuk paper 2, detail di `paper2/catatan_literatur.md`):
> - Genzoni 2020 memakai **PPG pergelangan**, bukan EKG. Teks lengkapnya tidak bisa diakses.
> - Lindow 2019 "34%" adalah label AFL komputer yang salah (27/80, kebanyakan SVT), **bukan** AFL 2:1 yang dibaca sinus takikardia.
> - Klaim AFL 2:1 di Littmann 2023 belum terverifikasi.
> - Kebijakan perangkat:
>   - Kardia: "Tachycardia" berarti 100–140 bpm, dan di atas 140 bpm hasilnya "Unclassified".
>   - Samsung: sejak v1.3 (K240909, 2024) punya kelas detak tinggi 100–150 bpm.
>   - Withings: kelas 100–150 bpm baru ada di K240795 (2025).
>   - Fitbit: ambang 120 bpm hanya tercantum di halaman bantuan Google.
> - Halvaei 2021 dan Eerikäinen 2019: alarm palsunya dari noise dan AFL, bukan dari latihan fisik.
> - Studi terdekat tambahan: Sanjana 2020 (19–28% dari 118 sampel sinus takikardia) dan Tse 1999.

**Sudah diketahui (wajib disitasi):**
- **Genzoni dkk., EMBC 2020** (doi:10.1109/EMBC44109.2020.9176418) adalah preseden terdekat. Pengklasifikasi AF berbasis IBI pada 37 pasien:
  spesifisitas turun 0,966 → 0,853 saat ada aritmia lain, dan "sinus tachycardia led to false positives". Ambil teks lengkapnya untuk angka
  ST mereka.
- Detak tinggi memang dipakai sebagai petunjuk AF oleh detektor klasik: Lake & Moorman 2011 (COSEn), Petrėnas dkk. 2015.
- AFL 2:1 sering dibaca sebagai sinus takikardia oleh software EKG: Littmann 2023; Lindow dkk. 2019 (diagnosis AFL komputer salah 34%).
- Alarm palsu AF pada EKG aktivitas, terutama karena noise: Halvaei dkk. 2021, Kumar dkk. 2023, Eerikäinen dkk. 2019/2020.
- Kritik terhadap variasi ritme AFDB dan uji stres sintetis (dengan APB dan noise): **Butkuvienė dkk., IEEE TBME 2021**
  (doi:10.1109/TBME.2021.3067698).
- **Produk komersial menghindari masalah ini lewat desain** (dari IFU masing-masing):
  - Apple ECG 2.0 punya kelas "High Heart Rate" 100–150 bpm dan tidak mengklasifikasi di atas 150 bpm.
  - Samsung: inconclusive bila > 100 bpm dan bukan AF.
  - Fitbit/Google: inconclusive bila > 120 bpm.
  - Withings: kelas "High HR" 100–150 bpm.
  - Kardia: "Tachycardia" bila > 100 bpm.
  - Algoritma PPG Apple dan Fitbit hanya menganalisis saat pengguna diam.
- Conformal gagal diam-diam saat data bergeser: Abbasi dkk., Heart Rhythm 2026 (doi:10.1016/j.hrthm.2026.10.064); Tibshirani 2019; Barber 2023.

**Belum ditemukan (inti kebaruan C4):**
1. Angka alarm ST per pita bpm untuk model RR-only hasil latih AFDB, termasuk 3 kelas dengan AFL, pada EKG latihan orang sehat.
2. Uji perilaku berupa sapuan bpm ritme teratur sintetis yang menghasilkan "ambang bpm" per model.
3. Bukti kuantitatif bahwa protokol AFDB buta terhadap masalah ini (0,1% window Normal > 120 bpm).
4. Kegagalan yang konsisten di 10 model dari 9 arsitektur.
5. Trade-off terukur: menambah ST ke data latih menurunkan alarm ke ±60%, tetapi menghilangkan deteksi AFL 2:1.
6. Conformal yang menandai 0% kesalahan pada kasus ini.

Kalimat klaim yang aman:

> We found no prior study that quantified sinus-tachycardia false alarms of RR-interval AF/AFL classifiers trained on the MIT-BIH AFDB.
> Using a synthetic regular-rhythm heart-rate sweep and real exercise ECG from 41 healthy subjects, all ten models labelled regular
> rhythms above 110–140 bpm as AF/AFL and flagged 89–100% of sinus-tachycardia windows ≥120 bpm, a failure that AFDB-based evaluation
> cannot reveal because only 0.1% of its normal windows exceed 120 bpm. This extends earlier observations that a photoplethysmography-based inter-beat-interval
> classifier can misclassify sinus tachycardia (Genzoni et al., 2020) and explains why commercial devices abstain or report a separate
> high-heart-rate category above 100–150 bpm.

Jangan menulis "first" tanpa "to our knowledge".

### 2.2 Untuk C1–C3, C5

- **Split pasien vs per segmen sudah dikuantifikasi.**
  - Butkuvienė 2021: hanya 4 dari 14 detektor DL di AFDB yang memisahkan pasien. HAN-ECG Se/Sp turun 0,991/0,985 → 0,905/0,797.
  - **Domazetoski dkk., CMPB 2022** (doi:10.1016/j.cmpb.2022.106901): split pasien menurunkan Se AFL 71% → 5,4%.
  - Illarionov & Khudorozhkov 2018 (arXiv:1804.02543); Seo dkk., Sci Rep 2021.
- **Window tumpang tindih.** Belum ada angka untuk EKG. Analoginya: HAR (Dehghani 2019, +16%) dan EEG (Brookshire 2024, 99,8% vs 53,0%).
- **Stride berbeda per kelas di data uji:** tidak ditemukan paper yang mengkuantifikasinya. Mekanismenya = Kapoor & Narayanan 2023, L3.3
  "sampling bias in test distribution". **Angka kita (−0,114) tampaknya baru.**
- **Pooled vs rata-rata per fold:** Forman & Scholz 2010 menunjukkan F1 rata-rata per fold bias negatif. Cocok dengan temuan pooled +0,070.
- **Front-end dan laju sampel.**
  - Oster & Clifford 2015: akurasi AF berbasis RR turun seiring noise.
  - Kwon dkk. 2018: 50 Hz tidak cukup untuk HRV.
  - Ellis dkk. 2015: dengan interpolasi R-peak, di bawah 100 Hz masih bisa.
  - **Tidak ditemukan studi yang memakai model yang sama untuk membandingkan front-end produk dengan front-end tervalidasi.** Angka −0,325
    tampaknya baru. Bingkai sebagai *training–serving skew* (Sculley dkk. 2015) agar tidak dianggap strawman.
- **Arsitektur setara dalam error bar:** Strodthoff dkk., JBHI 2021 (PTB-XL); Clifford dkk., CinC 2017. Pembanding yang wajib disebut:
  FOUND-AF 2026 (ECGFounder konsisten terbaik).
- **Statistik.**
  - Bootstrap per pasien: Rutter 2000.
  - Seleksi di dalam bootstrap: Tsamardinos 2018 (BBC-CV); Steyerberg & Harrell 2016.
  - Cutoff yang dipilih dari data: Leeflang 2008.
  - Uji tanpa pasien dominan belum ada di pedoman ML, jadi tulis sebagai praktik yang diusulkan.
- **Pedoman pelaporan.** **TRIPOD+AI** (Collins dkk., BMJ 2024) paling cocok dan diisi sebagai lampiran. Untuk metrik episode, tulis
  "EC57-style", jangan "sesuai EC57".
- **`StratifiedGroupKFold(shuffle=True)`.**
  - Batasannya terdokumentasi, tetapi kalimatnya menyesatkan.
  - Di ≤ 1.7 ada bug yang membuat partisi acak dan tidak terstratifikasi. Diperbaiki oleh PR #32540 (digabung 29 Oktober 2025, rilis 1.8).
  - Demo kita (1 partisi dari 100 seed) berlaku untuk ≥ 1.8. Langkah B dibuktikan memakai partisi asli P0, karena skor per fold cocok
    dengan log `model.ipynb`.
- **Contoh paper sejenis di venue target:** Butkuvienė (TBME), Domazetoski (CMPB), Gilon dkk. (CinC 2022, reproduksi gagal), Biton dkk.
  (npj Digit Med 2023: model RR ArNet2 melewatkan 69,68% window AFL).

### 2.3 Sitasi yang harus diperbaiki di kerangka

- Link arXiv 2206.14200 bukan Domazetoski (sudah diperbaiki di kerangka).
- Faust 2021 memakai 4.051 subjek, bukan AFDB.

## 3. Risiko reviewer dan cara menjawabnya

| Risiko | Jawaban / tindakan |
|---|---|
| "Semua ini sudah diketahui" | Tampilkan Butkuvienė, Domazetoski, dan Genzoni sejak awal. Jual sebagai kuantifikasi dengan model tetap dan bukti benchmark buta. |
| "Produk sudah membatasi detak, jadi trivial" | Justru menjelaskan **mengapa** produk perlu abstain. Model riset yang dilatih dan dievaluasi di AFDB tidak melihatnya. Usulkan kelas "takikardia teratur, perlu dicek". |
| "Alarm karena noise detektor saat olahraga" | Sudah terjawab: R-peak manual tetap 100% alarm. Uji sintetis bebas noise memberi ambang 110–140 bpm. |
| Label "sinus" pada data latihan tidak ada | Subjek sehat, detak naik bertahap mengikuti beban, CV IBI rendah (median 0,009 dengan R-peak manual EPFL). Tambahkan pengecekan visual sampel P-wave sebagai penguat (pekerjaan baru, ±1 hari). |
| 23 pasien, 7 AFL → daya uji rendah | Laporkan CI Δ dan selisih terkecil yang bisa terdeteksi. Jangan klaim "setara". |
| Front-end −0,325 dianggap bug/strawman | Bingkai sebagai training–serving skew yang nyata terjadi di sistem yang dipakai, dengan sitasi Sculley dkk. dan Breck dkk. |
| Bootstrap bersarang dianggap metode baru | Sebut sebagai penerapan BBC-CV, bukan metode baru. |
| Versi scikit-learn | Tulis versinya. Jelaskan bug ≤ 1.7, dan bahwa partisi B diverifikasi terhadap log asli. |
| Self-critique pipeline sendiri | Bingkai sebagai studi kasus pengembangan perangkat nyata. Justru ini nilai tambahnya. |
| Etika | Semua data publik dan anonim, tanpa subjek baru, jadi kemungkinan besar bebas ethical clearance. **Konfirmasi ke pembimbing / komite etik Telkom.** |

## 4. Pekerjaan yang tersisa sebelum submit

| Pekerjaan | Perkiraan |
|---|---|
| Related Work dengan sitasi wajib (bagian 2), termasuk teks lengkap Genzoni 2020 | 2–3 hari |
| Gambar dari cache: waterfall C1, forest plot, kurva sapuan bpm, trade-off alarm ST vs recall AFL 08378 | 2–3 hari |
| Pengecekan visual P-wave sampel window ≥ 120 bpm (penguat label sinus) | ±1 hari |
| Skrip ekstraksi angka paper dari `cache/` dengan `assert` | ±1 hari |
| Naskah bahasa Inggris 5–6 halaman format IEEE, plus lampiran TRIPOD+AI | ±1 minggu |
| Review internal tim dan pembimbing, revisi | ±1 minggu |
| **Total untuk satu paper konferensi 6 halaman** | **±3–4 minggu kerja** |

## 5. Konferensi di EDAS (dicek 6 November 2026)

Dari 210 entri yang terbuka, yang cocok dengan topik (biosinyal, pengukuran medis, kesehatan digital) dan tenggatnya masih realistis:

| Konferensi | Tempat & tanggal | Tenggat (WIB) | Format & publikasi | Kecocokan |
|---|---|---|---|---|
| **IEEE MeMeA 2027** (Medical Measurements and Applications) | Roma, Italia, 10–13 Jun 2027 | **24–25 Jan 2027** (full paper); notifikasi 14 Mar | 5–6 halaman; IEEE Xplore; versi diperluas ke special section **IEEE Trans. Instrum. Meas.** | **Tinggi.** Topik: "AI & ML for biosignal measurement", "Clinical validation", "Evaluation methodologies and real-world deployments". Sesi khusus SS7 *Wearable Sensing*, SS8 *Data Quality in Real-World Biomonitoring*, SS3 *Measurement Systems for LMICs* |
| **ICIMPACT 2027** (Telkom University) | Bandung, 3–5 Feb 2027 | **30 Nov 2026** (24 hari lagi) | Selektivitas 50%. Situs resmi `icimpact.telkomuniversity.ac.id` (aktif, dicek 6 Nov sore): maks. 6 halaman termasuk referensi, template konferensi IEEE, *blind review*, edisi pertama, kebijakan *no-show* IEEE (naskah diajukan ke IEEE Xplore; ID PDF Express belum ada) | **Tinggi** secara topik. Track *Biomedical Multimedia Intelligence & Digital Health Systems* menyebut biosinyal EKG/PPG, wearable/edge, dan *robustness/domain shift, calibration/uncertainty*. Lokal dan murah |
| **IMEKO 2027** World Congress | Rimini, Italia, 30 Agu–3 Sep 2027 | **15–16 Jan 2027** (extended abstract min. 4 halaman); full paper maks. 6 halaman Mei 2027 | Proceedings open access di situs IMEKO, plus jalur jurnal. Indeksasi Scopus belum terverifikasi | Tinggi untuk sudut "validasi". TC13 *Biology & Medicine*, SS#25 *Trustworthy Medical Technologies*, SS#38 *AI in Measurement Science: Metrological Validation* |
| **IEEE TENSYMP 2027** (Region 10) | Kandy, Sri Lanka, 12–14 Mei 2027 | 13 Jan 2027 | IEEE Xplore | Sedang. Track *Technology for Health, Humanity, and Society* (luas) |
| **IEEE I2MTC 2027** | Chongqing, Tiongkok, 17–21 Mei 2027 | 5 Des 2026 | Flagship IEEE I&M; batas halaman belum dicek | Sedang–tinggi. Track *I&M in Medical, Biomedical and Healthcare Systems*. Waktu sempit |
| IEEE AI4IM 2027 | Xi'an, Tiongkok, 13–15 Mei 2027 | 9 Jan 2027 | 5–6 halaman; selektivitas 80% | Sedang |
| IEEE SAS 2027 | Ho Chi Minh City, 26–29 Apr 2027 | 16 Nov 2026 | – | Terlalu cepat |

**Tidak disarankan:**
- Tenggat hari ini atau besok: ICGHIT, MEBS, ICDSG.
- Tanpa track kesehatan: ISRITI.
- Konferensi umum yang topiknya jauh: ICISET, ICETISI, dan lain-lain.
- Jurnal-jurnal di EDAS dengan reputasi tidak jelas: seri "The J…" UK, JBSIR, dan sejenisnya.

**Pembanding di luar EDAS** (komunitas yang paling pas, tanggal belum diverifikasi dari situs resmi):
- **IEEE EMBC 2027**: Singapura, 11–15 Jul 2027, tenggat full paper ±24 Jan 2027 (menurut agregator). Paling bergengsi di rekayasa
  biomedis, dan lokasinya dekat sehingga biaya jauh lebih murah daripada Italia.
- **Computing in Cardiology (CinC) 2027**: Auckland. Abstrak biasanya 15 April. Ini komunitas inti deteksi AF, formatnya 4 halaman.

## 6. Rekomendasi strategi

Satu paper tidak boleh dikirim ke dua venue sekaligus. Ada dua jalur yang masuk akal:

**Jalur A, satu paper kuat (disarankan bila tim kecil).** Isinya C4 + C1 (+ C3 singkat), 6 halaman, dikirim ke **MeMeA 2027** (24 Januari)
atau **EMBC 2027** (±24 Januari, Singapura). Waktu ±11 minggu cukup untuk pekerjaan ±3–4 minggu di bagian 4. MeMeA lebih cocok topiknya dan
punya jalur ke IEEE TIM. EMBC lebih bergengsi dan lebih dekat. Pilih satu.

**Jalur B, dua paper dengan isi berbeda.**
- **ICIMPACT 2027** (30 November, Telkom): **C1 + C2 + C3**, yaitu pitfall evaluasi dan dekomposisi. Angkanya sudah lengkap, tinggal menulis
  dalam ±3 minggu.
- **EMBC 2027** (dipilih 6 Nov 2026; tenggat 24 Jan 2027, Singapura 11–15 Juli 2027, 4+1 halaman, maks. 20 referensi, hadir langsung): **C4** (uji stres sinus takikardia), temuan terkuat untuk venue yang lebih kuat.
- Risikonya: dua paper dari satu proyek harus jelas tidak tumpang tindih (bukan *salami slicing*), dan ICIMPACT lebih dulu dicek indeksasi
  serta batas halamannya. Hubungi panitia karena situsnya sedang tidak bisa dibuka.

Setelah konferensi, versi jurnal bisa dibuat (special section IEEE TIM bila lewat MeMeA, atau Physiological Measurement) dengan konten baru
yang cukup dan sitasi ke paper konferensinya. Cek kebijakan masing-masing penerbit.

## 7. Biaya pendaftaran dan kebijakan presentasi (dicek 6 November 2026)

Sebagian besar konferensi 2027 belum merilis biaya. Kalau begitu, yang ditampilkan adalah **biaya edisi 2026 sebagai patokan** dan ditandai.
Semua angka di bawah adalah biaya pendaftaran saja, belum termasuk tiket, visa, dan penginapan.

| Konferensi | Biaya pendaftaran penulis (yang menanggung paper) | Tarif mahasiswa menanggung paper? | Presentasi online? |
|---|---|---|---|
| **IEEE MeMeA** | 2026 (patokan): anggota IEEE USD 649 (awal) / 822 (reguler); non-anggota 772 / 920; anggota IMS 600 / 748. Termasuk PPN 22%. Biaya 2027 belum terbit; batas awal 11 Apr 2027 | **Tidak.** Tarif mahasiswa (USD 379/453) tidak bisa menanggung paper; mahasiswa harus bayar tarif anggota atau non-anggota | **Tidak.** Penulis harus "registered AND attends the Conference to present" |
| **IEEE EMBC** | 2026 (patokan): anggota EMBS USD 995 (awal) / 1.195; anggota IEEE 1.095 / 1.295; non-anggota 1.350 / 1.550. Tarif LLMI USD 500–650 hanya untuk negara berpendapatan rendah atau menengah-bawah; Indonesia kemungkinan **tidak** termasuk karena masuk kategori menengah-atas Bank Dunia sejak 2023 (cek) | **Tidak.** Hanya Full Registration; satu pendaftaran menanggung maksimal 2 paper | **Tidak.** "no virtual, remote, or pre-recorded option". Tidak hadir berarti paper dihapus dari Xplore, bahkan bila visa ditolak. Tidak hadir tanpa dokumen: masuk daftar no-show EMBS minimal 2 tahun |
| **IEEE I2MTC** | 2026 (patokan): anggota IEEE USD 880 (awal) / 980; non-anggota 1.045 / 1.145; anggota IMS 845 / 945. Paper tambahan USD 100 | **Tidak** | **Tidak.** Wajib presentasi langsung; bila tidak, ditandai "no show" dan dihapus dari IEEE Xplore |
| IEEE AI4IM | Halaman biaya 2026 tidak ditemukan | Kemungkinan sama dengan konferensi IMS lain (MeMeA, I2MTC), belum terverifikasi | Kemungkinan wajib hadir (pola IMS), belum terverifikasi |
| **IEEE TENSYMP 2027** (sudah terbit) | Internasional: anggota IEEE USD 300 (sebelum 15 Mar 2027) / 360; non-anggota 400 / 480. Satu pendaftaran non-mahasiswa menanggung 2 paper | **Ya.** Presenter mahasiswa anggota IEEE USD 200 / 240, non-anggota 300 / 360, menanggung 1 paper | **Hanya darurat ekstrem**, dengan persetujuan panitia; pada dasarnya wajib hadir di Kandy |
| IMEKO 2027 | Belum terbit. Patokan IMEKO 2024 (tentatif): EUR 850 awal, 1.100 reguler, mahasiswa 490 | Belum diketahui | Belum diketahui. Tidak ada informasi presentasi jarak jauh |
| **ICIMPACT 2027** (Telkom) | **Terverifikasi dari situs resmi 6 Nov 2026.** Mahasiswa: Rp2.800.000 (anggota IEEE) / Rp3.500.000 (non-anggota); reguler Rp3.200.000 / Rp4.000.000; *early bird* semua kategori Rp3.000.000 sampai 18 Des 2026; halaman tambahan > 6 Rp600.000/halaman | Ya, konferensi **hybrid** | **Kemungkinan ya.** Situs menulis "Bandung (Hybrid)". Aturan *no-show*: salah satu penulis atau wakilnya wajib presentasi. Mekanisme online belum dirinci; tanyakan ke `icimpact@telkomuniversity.ac.id` |
| CinC 2027 | Belum dicek | – | Belum terverifikasi. Halaman CinC menyebut "remote attendance", tetapi diskon mahasiswa mensyaratkan hadir langsung |

**Implikasi untuk strategi (bagian 6):**
- **Hanya ICIMPACT yang realistis untuk presentasi online.** MeMeA, EMBC, dan I2MTC wajib hadir langsung, dan paper dihapus dari IEEE Xplore
  bila tidak dipresentasikan. TENSYMP hanya mengizinkan online untuk darurat.
- **Di konferensi IEEE IMS dan EMBC, tarif mahasiswa tidak menanggung paper.** Minimal satu penulis membayar tarif penuh:
  - ±USD 650–820 (MeMeA, patokan 2026);
  - ±USD 1.000–1.300 (EMBC, patokan 2026).
  
  TENSYMP pengecualian: tarif presenter mahasiswa (USD 200–240) sudah menanggung 1 paper.
- **Biaya perjalanan.**
  - Ke Italia (MeMeA, IMEKO) perlu visa Schengen dan tiket jarak jauh.
  - Ke Singapura (EMBC) tidak perlu visa bagi WNI dan tiketnya jauh lebih murah.
  - Ke Sri Lanka (TENSYMP) dan Tiongkok (I2MTC, AI4IM): cek visa dan tiket.
  - Harga tiket belum dicek.
- **Bila tim tidak bisa berangkat ke luar negeri**, jalur yang aman: ICIMPACT (hybrid, Rp2,8–3,5 juta untuk mahasiswa; Rp3 juta bila *early bird*), atau jurnal
  (tanpa presentasi, mis. Physiological Measurement). Kebijakan APC jurnal perlu dicek terpisah.

## Sumber utama

- EDAS: daftar "Conferences accepting submissions" dan halaman detail/track MeMeA (N35766), ICIMPACT (N35057), I2MTC (N35663),
  SAS (N35706), TENSYMP (N35167), IMEKO (N35447), AI4IM (N35687), ISRITI (N35654). Semua dibuka 6 November 2026.
- [MeMeA 2027 Call for Papers](https://memea.ieee-ims.org/2027/authors/call-for-papers) · [IMEKO 2027 Call for Papers](https://imeko2027.org/call-for-papers/)
  · [TENSYMP 2027](https://tensymp27.ieee.lk/)
- [scikit-learn issue #32478](https://github.com/scikit-learn/scikit-learn/issues/32478) (ditutup oleh PR #32540, 29 Okt 2025)
- Biaya & kebijakan: [MeMeA 2026 registration](https://memea.ieee-ims.org/2026/registration-information) dan flyer biayanya ·
  [MeMeA 2027 FAQ](https://memea.ieee-ims.org/2027/general-faq) · [EMBC 2026 registration](https://embc.embs.org/2026/registration/) ·
  [EMBC 2026 Author Non-Attendance Policy](https://embc.embs.org/2026/wp-content/uploads/sites/118/2026/06/EMBC2026_Author_NonAttendance_Policy_Table.pdf) ·
  [I2MTC 2026 registration](https://i2mtc2026.ieee-ims.org/registration) · [TENSYMP 2027 registration](https://tensymp27.ieee.lk/registration) ·
  [ICoICT 2026 registration (Telkom)](https://icoict.telkomuniversity.ac.id/registration/) · biaya tentatif IMEKO 2024 dari halaman PTB
  ([energy.ptb.de](https://www.energy.ptb.de/world-congress))
- [Auckland tuan rumah CinC 2027](https://www.ttgmice.com/?p=59884) · [EMBC 2027 (agregator)](https://www.myhuiban.com/conference/5339)
- Sitasi literatur lengkap dengan DOI ada di bagian 2. Daftar wajib-sitasi gabungan: Butkuvienė 2021, Domazetoski 2022, Genzoni 2020,
  de Chazal 2004, Luz 2016, Kapoor & Narayanan 2023, Forman & Scholz 2010, Illarionov 2018, Oster & Clifford 2015, Kwon 2018, Ellis 2015,
  Strodthoff 2021, Clifford 2017, Tsamardinos 2018, Cawley & Talbot 2010, Varoquaux 2018, Rutter 2000, Leeflang 2008, TRIPOD+AI 2024,
  Biton 2023, FOUND-AF 2026, Seo 2021, Brookshire 2024, Dehghani 2019, Lake & Moorman 2011, Petrėnas 2015, Littmann 2023, Lindow 2019,
  Halvaei 2021, Kumar 2023, Eerikäinen 2019/2020, Abbasi 2026, Tibshirani 2019, Barber 2023, Perez 2019 (Apple Heart Study), Lubitz 2022,
  IFU Apple/Samsung/Fitbit/Withings/Kardia (atau Abu-Alrub 2022), Ivanović 2019, Faust 2021.
