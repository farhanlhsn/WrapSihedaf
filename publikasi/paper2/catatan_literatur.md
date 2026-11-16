# Catatan literatur paper 2 (uji stres sinus takikardia)

Diverifikasi 6 November 2026. Entri BibTeX ada di `refs_kandidat.bib` (folder ini). Uji kompilasi dengan IEEEtran:
48 entri, tanpa warning BibTeX. IEEEtran tidak mencetak field `doi`. Karena itu entri dataset diberi `url` https://doi.org/….
Agar garis bawah di URL FDA tercetak benar, `main.tex` perlu `\usepackage{url}` (atau hyperref).

**Sumber verifikasi** (disebut per butir):

- **Crossref**: `api.crossref.org/works/<DOI>`.
- **PubMed**: E-utilities, untuk abstrak dan daftar penulis.
- **PMC**: teks lengkap lewat Europe PMC `fullTextXML`.
- **arXiv**: API.
- **DataCite / Zenodo**: API.
- **FDA**: PDF ringkasan 510(k)/De Novo di `accessdata.fda.gov`, nomor dicari lewat openFDA.
- **IFU / halaman resmi produsen**.
- **Halaman PhysioNet**.

Kutipan langsung paling banyak satu per referensi, masing-masing di bawah 15 kata.

---

## 0. Koreksi terhadap draf (`riset_kelayakan_dan_venue.md`, `main.tex`)

1. **Genzoni 2020.** Teks lengkap **tidak bisa diakses** (detail di §1.1), jadi yang terverifikasi hanya abstrak. Ada tiga hal
   yang perlu diketahui:
   - Datanya **PPG** (optical heart rate monitor di pergelangan), bukan EKG.
   - Data direkam dari 37 pasien selama ablasi kateter.
   - Menurut abstrak, **AFL tidak mengganggu** klasifikasi mereka. Di model kita AFL justru menjadi kelas yang ikut
     mengalarmkan sinus takikardia.
2. **Lindow 2019, "34%".** Angka ini adalah **AFL palsu dari komputer**: 27 dari 80 EKG yang oleh komputer disebut AFL
   ternyata bukan AFL. Penyebab terbanyak di kelompok ini adalah SVT/takikardia atrium (9 dari 27). Studinya hanya memeriksa
   EKG yang sudah dilabel AF/AFL oleh komputer, jadi **tidak mengukur AFL 2:1 yang dibaca sinus takikardia**. Jangan
   mensitasi Lindow untuk klaim itu.
3. **Littmann 2023.** Klaim "AFL 2:1 dibaca sinus takikardia" **belum terverifikasi**. Abstrak Part II maupun Part III tidak
   menyebut flutter, dan teks lengkapnya tertutup (Elsevier, status "closed" di Unpaywall/Semantic Scholar). Periksa lewat
   akses institusi sebelum dipakai.
4. **Samsung.** "Inconclusive > 100 bpm bila bukan AF" benar untuk ECG Monitor v1, yaitu K201168 (2020) dan K230292 (2023).
   Namun **Samsung ECG App v1.3 (K240909, 2 Agustus 2024)** sudah menambah kelas *high heart rate (non-AFib)* dan *AFib with high
   heart rate* untuk 100–150 bpm. Tulis versinya.
5. **Kardia.** Kelas "Tachycardia" berarti ritme normal **100–140 bpm**, bukan "> 100 bpm". Di atas 140 bpm hasilnya
   "Unclassified". Kalimat `main.tex` "no classification above 100–150 bpm" masih benar sebagai rentang gabungan
   (batas atas Kardia 140, Apple 150).
6. **Withings.** Kelas *High Heart Rate (No AFib)* 100–150 bpm tercatat di **Withings ECG App K240795 (15 Juni 2025)**. Pada
   Withings Move ECG yang diuji tahun 2022 (Abu-Alrub 2022), ritme sinus > 100 bpm masih tidak diklasifikasi.
7. **Fitbit/Google.** Ringkasan FDA K200948 **tidak menyebut** ambang bpm. Sumber ambang 120 bpm adalah halaman bantuan
   resmi Google (`google_ecg_help`). Nama aplikasinya sekarang "Google ECG app".
8. **Halvaei 2021** membahas noise transien pada EKG genggam (skrining StrokeStop I), **bukan** aktivitas fisik.
   **Eerikäinen 2019 (JBHI)** menemukan alarm palsu detektor IBI terutama dari **AFL**, juga bukan aktivitas. Efek gerak
   dibahas di Eerikäinen 2018 (Physiol Meas). Alarm palsu yang terkait konteks aktivitas dibahas oleh **Kumar 2023**.
9. **EPFL di `main.tex`: "cardiologist-reviewed R-peaks".** Zenodo hanya menyebut R-peak direview oleh seorang dokter
   CHUV (tanpa menyebut kardiolog), dan versi arXiv makalahnya menyebut tenaga ahli medis. Sarankan diganti menjadi
   "physician-reviewed".
10. **"Healthy subjects".** Hanya EPHNOGRAM yang eksplisit menyebut "24 healthy adults" (abstrak Physiol Meas 2024 dan
    halaman PhysioNet). Halaman PhysioNet Wrist dan deskripsi Zenodo EPFL tidak menyebut status kesehatan, dan teks MDPI
    Jarchi diblokir (HTTP 403), jadi status itu belum terverifikasi. Tulis misalnya "subjects without known arrhythmia
    performing exercise", atau periksa teks lengkapnya.
11. **Sitasi standar PhysioNet berubah.** Setiap halaman database kini meminta Pollard dkk., *Nature Health* 2026
    (`pollard2026physionet`), bukan Goldberger 2000. Goldberger 2000 tetap sah sebagai sitasi klasik.
12. **Apple ECG 2.0** tetap bisa mengklasifikasi AF sampai 150 bpm (*Atrial Fibrillation High Heart Rate*, 100–150 bpm).
    Kelas *High Heart Rate (No AFib)* 100–150 bpm adalah hasil terpisah, dan > 150 bpm hanya dilaporkan sebagai detak tinggi.
    Versi pertama (DEN180044, 2018) tidak mengklasifikasi detak > 120 bpm.

---

## 1. Per referensi

### 1.1 Preseden langsung dan detektor klasik

- **genzoni2020**: pengklasifikasi tradisional berbasis IBI untuk AF vs ritme sinus, diuji pada 37 pasien yang memakai
  optical heart rate monitor selama ablasi kateter.
  - AF vs sinus murni: akurasi, sensitivitas, dan spesifisitas 0,979; 1,000; 0,966.
  - Bila window boleh berisi aritmia non-AF sampai 50%, angkanya turun menjadi 0,886; 0,998; 0,853.
  - Kutipan: "While sinus tachycardia led to false positives". Ekstrasistol, bigemini, bradikardia, dan AFL tidak mengganggu.
  - Sumber: abstrak PubMed 33019109, sama dengan abstrak di Infoscience.
  - **Teks lengkap tidak tersedia**:
    - Infoscience EPFL: record `datacite.rights = metadata-only`, tanpa bitstream PDF. Klaim "green OA" di
      Unpaywall/OpenAlex ternyata keliru.
    - IRIS/Serval UNIL: halaman di balik CAPTCHA (tidak dilewati).
    - IEEE: berbayar. Europe PMC/PMC dan arXiv tidak punya salinan.
  - Akibatnya, jumlah episode/subjek sinus takikardia, angka positif palsu sinus takikardia, jenis classifier, dan rentang
    detak **tidak diketahui**. Perlu akses IEEE institusi atau meminta PDF ke penulis (CSEM/EPFL).
- **lake2011** (COSEn):
  - Kutipan: "the mean heart beat interval each contributed to accurate detection of AF".
  - Artinya detak tinggi (interval rata-rata pendek) ikut menjadi petunjuk AF.
  - Algoritme dikembangkan di database MIT-BIH (abstrak menyebutnya kanonik) dan divalidasi pada Holter UVA.
  - Kesalahan tersering: ektopi, dan AFL yang entropinya bisa rendah atau tinggi.
  - Sumber: abstrak PubMed 21037227.
- **petrenas2015**:
  - Detektor AF berjendela 8 detak dengan penyaringan detak ektopik, penekanan bigemini, karakterisasi ketidakteraturan RR,
    dan fusi sinyal.
  - Sensitivitas 97,1% dan spesifisitas 98,3% di AFDB.
  - Klaim "detak tinggi dipakai sebagai petunjuk AF" **tidak disebut di abstrak**; tidak terverifikasi. Untuk klaim itu,
    pakai lake2011 saja, atau cek teks lengkap Petrėnas.
  - Sumber: abstrak PubMed 25666902.
- **tse1999**:
  - 5 pasien dengan defibrilator atrium implan (METRIX) menjalani uji treadmill. Detak puncak rata-rata 137 ± 26 bpm.
  - Ada 1 deteksi AF positif palsu saat sinus takikardia pada setelan nominal. Syok dicegah oleh kriteria RR minimum, dan
    masalah hilang setelah *Quiet Interval* diprogram ulang.
  - Sumber: abstrak PubMed 9990640.
- **sanjana2020**:
  - Model DL (RNN, LSTM, GRU, CNN, RSCNN) dilatih dengan fitur HRV dari data AF CinC 2017, lalu diuji pada 118 sampel sinus
    takikardia dari MIT-BIH Arrhythmia DB.
  - Sampel ST yang disebut positif: 18,6–28,0%. Untuk CNN, teks menyebut 18,64% sedangkan Tabel 9 menyebut 26,50% (tidak
    konsisten).
  - Kerangka penulis: kegagalan *mendeteksi* ST sebagai penyakit takikardia. Sumber: teks lengkap PMC7787999.
- **butkuviene2021** (sudah ada di paper 1):
  - Tiga jenis detektor AF diuji pada EKG nyata dan simulasi.
  - Kinerja dipengaruhi cara pembandingan (detak, segmen, episode), laju APB, morfologi, dan noise.
  - Tidak ada sapuan detak sinus. Sumber: abstrak PubMed 33750686.

### 1.2 Interpretasi EKG komputer untuk AF/AFL

- **lindow2019**:
  - Dari 988 EKG berlabel AF/AFL oleh komputer, 89 (9,0%) salah dan 36 di antaranya tidak dikoreksi dokter.
  - AFL salah 27/80 (34%) vs AF 62/908 (7%). 12 pasien mendapat antikoagulan yang tidak perlu.
  - Di kelompok AFL palsu, SVT berkelanjutan (termasuk takikardia atrium ektopik) 9/27.
  - Sumber: abstrak PubMed 31684791 dan Tabel 1 teks lengkap PMC6883419. Lihat koreksi §0.2.
- **littmann2023** (Part II, hlm. 277–280) dan **littmann2023iii** (Part III, hlm. 281–284): seri tentang kesalahan software
  interpretasi EKG. Hanya abstrak yang terverifikasi, dan abstrak tidak menyebut AFL atau sinus takikardia (PubMed 37633808,
  37599144). Klaim AFL 2:1 belum terverifikasi (§0.3).

### 1.3 Alarm AF palsu di luar klinik

- **halvaei2021**:
  - CNN untuk mengenali noise transien sebelum detektor AF berkompleksitas rendah, pada EKG genggam 30 s (StrokeStop I).
  - Deteksi AF palsu turun 22,5% tanpa kehilangan sensitivitas. Ini soal noise, bukan aktivitas.
  - Sumber: abstrak PubMed 34149452.
- **kumar2023**:
  - Model DL yang kinerjanya *state-of-the-art* di tiga dataset aritmia PhysioNet diterapkan ke 215 hari EKG satu kanal di
    kondisi bebas (free-living).
  - ±62% segmen yang ditandai AF berdurasi ≤ 50 s, dan 99,9% di antaranya positif palsu.
  - ±78% positif palsu pendek itu terkait perubahan aktivitas, perubahan posisi tubuh, atau akselerasi mendadak.
  - Nama ketiga dataset PhysioNet tidak disebut di abstrak.
  - Sumber: abstrak via OpenAlex (artikel OA CC BY; ScienceDirect memblokir unduhan).
- **eerikainen2019** (JBHI, terbit online 2019, nomor jurnal 2020):
  - Random forest PPG + IPI + akselerometer, 40 pasien, monitoring 24 jam.
  - Sensitivitas AF/AFL/lainnya 97,6/84,5/98,1%.
  - Pada classifier AF berbasis IPI, mayoritas deteksi palsu disebabkan AFL.
  - Sumber: abstrak PubMed 31689222.
- **eerikainen2020**:
  - Tinjauan AF dengan PPG pergelangan. Kebanyakan studi dilakukan di rumah sakit, dan dataset publik yang memadai kurang.
  - Sumber: abstrak PubMed 35265873.
  - Alternatif: surat JACC 2020 (doi 10.1016/j.jacc.2019.10.064) tentang Huawei Heart Study. Hanya judul dan kalimat
    pembukanya yang terverifikasi, jadi tidak dimasukkan ke bib.
- **eerikainen2018** (opsional):
  - Pada 27 pasien, CosEn adalah fitur EKG terbaik (κ 0,95). PPG menyamai bila segmen bergerak dibuang, tetapi cakupan PPG
    hanya 24,0–57,6% vs 92,1% untuk EKG.
  - Sumber: abstrak PubMed 29995641.
- **torressoto2020** (opsional, DeepBeat):
  - Data pengembangan PPG memuat kohort uji latih jantung tanpa AF: 41 subjek di Tabel 1, sedangkan teks menyebut 42 dipantau.
  - Ini contoh "memasukkan data latihan ke data latih", tanpa laporan alarm per detak.
  - Sumber: teks lengkap PMC7481177.

### 1.4 Studi wearable skala besar dan validasi perangkat

- **perez2019** (Apple Heart Study):
  - 419.297 peserta. 2.161 (0,52%) menerima notifikasi pulsa tidak teratur.
  - AF ada pada 34% patch EKG yang bisa dianalisis.
  - PPV notifikasi untuk AF bersamaan dengan EKG: 0,84 (95% CI 0,76–0,92).
  - Sumber: abstrak PubMed 31722151.
- **lubitz2022** (Fitbit Heart Study):
  - 455.699 peserta, algoritme tachogram 5 menit. IHRD = 11 tachogram tidak teratur berturut-turut.
  - IHRD pada 4.728 peserta (1%). PPV IHRD 98,2% (95% CI 95,5–99,5%).
  - Abstrak tidak menyebut syarat diam; syarat itu ada di FDA K212372 (§2).
  - Sumber: abstrak Crossref/PubMed 36148649.
- **abualrub2022**:
  - 100 pasien AF dan 100 sinus, membandingkan Apple Watch 5, Samsung Galaxy Watch Active 3, dan Withings Move ECG.
  - Withings: 1 EKG sinus tidak diklasifikasi karena detak > 100/menit. Apple: 1 EKG AF tidak diklasifikasi karena > 150/menit.
  - Penulis mencatat ambang Apple naik dari 120 ke 150 bpm.
  - Sumber: teks lengkap PMC8854369.
- **mannhart2023**:
  - 201 pasien, 5 perangkat.
  - Tracing inconclusive: Apple 18%, Samsung 17%, Withings 24%, Fitbit 21%, AliveCor 26%.
  - Sumber: abstrak PubMed 36858690.
- **fu2021** (opsional, contoh tandingan):
  - Perekam EKG wearable + AI pada 114 peserta, posisi berbaring, berdiri, dan setelah latihan 60 s.
  - Spesifisitas 100% saat berdiri; setelah latihan "serupa". Detak tidak dilaporkan.
  - Sumber: abstrak PubMed 34800984.

### 1.5 Conformal dan pergeseran distribusi

- **tibshirani2019**:
  - Conformal berbobot memberi interval valid bila distribusi kovariat latih dan uji berbeda, asalkan rasio likelihood-nya
    diketahui atau bisa diestimasi.
  - Sumber: abstrak arXiv 1904.06019; metadata dan halaman (2530–2540) dari proceedings.neurips.cc.
- **barber2023**:
  - Validitas conformal bergantung pada asumsi *exchangeability*, yang sering dilanggar saat deployment.
  - Kuantil berbobot mengurangi hilangnya cakupan ketika data bergeser.
  - Sumber: abstrak arXiv 2202.13415; metadata dari Crossref.
- **angelopoulos2023**:
  - Pengantar conformal prediction, termasuk pergeseran distribusi dan model yang abstain.
  - Sumber: abstrak Crossref.
- **abbasi2026**: **ada.** Crossref dan PubMed 42982386 mencatatnya, terbit online 4 November 2026, tanpa volume/halaman.
  - Classifier ritme 12-lead 4 kelas dilatih di PTB-XL.
  - Cakupan conformal 90% turun dari 0,901 menjadi 0,758 (Chapman-Shaoxing) dan 0,722 (Georgia).
  - AUC AF tetap 0,998 di China, tetapi turun ke 0,936 di AS.
  - Koreksi tanpa label gagal. Rekalibrasi lokal dengan 50 EKG berlabel memulihkan cakupan rata-rata ke 0,90, tetapi
    variabilitasnya lebar.
  - Kutipan: "AI-ECG uncertainty can fail silently despite high AUC."
  - Sumber: abstrak PubMed. Catatan: 12-lead dan pergeseran antar-negara, bukan sinus takikardia.

### 1.6 Dataset

- **epfl_hi** (Zenodo 10.5281/zenodo.5727800, CC BY 4.0, 25 November 2021):
  - Penulis: De Giovanni, Teijeiro, Meier, Millet, Atienza.
  - 22 subjek menjalani uji kardiopulmoner maksimal di sepeda ergometer. 20 subjek dilaporkan; subjek 5 dan 12 dibuang.
  - EKG BIOPAC satu lead, 500 Hz diturunkan ke 250 Hz.
  - 5 segmen 20 s per subjek (99 segmen). R-peak manual direview seorang dokter CHUV.
  - Makalah terkait: **degiovanni2023** (IEEE TBME 70(3):941–953). Versi arXiv 2112.04369 merujuk dataset Zenodo ini
    sebagai ref. [30].
  - Sumber: Zenodo API, DataCite, arXiv PDF.
- **jarchi2016wrist** (Data 2(1):1; online 24 Desember 2016, MDPI menulis 2017) dan **physionet_wrist**
  (10.13026/C2PQ1X, v1.0.0, 20 Oktober 2017, ODC-BY):
  - 8 peserta (3 pria, 5 wanita), usia 22–32.
  - Jalan dan lari di treadmill, sepeda beban ringan dan berat. EKG dada 256 Hz dengan R-peak ditandai manual.
  - Halaman PhysioNet meminta sitasi makalah Data.
  - Sumber: halaman PhysioNet, Crossref, DataCite.
- **kazemnejad2021ephnogram** (PhysioNet 10.13026/tjtq-5911, v1.0.0, 11 Juni 2021, ODC-BY) dan **kazemnejad2024**
  (Physiol Meas 45(5):055005):
  - 24 pria sehat, usia 23–29. 69 rekaman: istirahat, jalan, lari, sepeda statis. Sampling 8 kHz.
  - Halaman PhysioNet juga meminta sitasi preprint bioRxiv 10.1101/2021.05.17.444563 (kini terbit sebagai kazemnejad2024)
    dan toolbox OSET.
  - Sumber: halaman PhysioNet, DataCite, Crossref, PubMed 38663430.
- **zheng2020chapman** (Sci Data 7:48), **zheng2022ecgarrhythmia** (PhysioNet 10.13026/wgex-er52, v1.0.0, 24 Agustus 2022,
  CC BY 4.0), dan **zheng2020scirep** (Sci Rep 10:2898): halaman PhysioNet `ecg-arrhythmia` meminta sitasi dataset dan
  *original publication* Sci Rep 2020. Sumber: halaman PhysioNet, Crossref, DataCite.
- **moody1983afdb** dan **physionet_afdb** (10.13026/C2MW2D): halaman PhysioNet meminta sitasi Moody & Mark 1983 (Computers in
  Cardiology 10:227–230, tanpa DOI). DataCite mencatat tahun 1992; PhysioNet menulis "Published Nov. 4, 2000".
- **goldberger2000physionet** dan **pollard2026physionet** (Nature Health 1(8):792–795): lihat §0.11.

### 1.7 Standar

- **aami_ec57**: ANSI/AAMI EC57:2012/(R)2020. Hanya judul dan status dari katalog ANSI/BSI yang terverifikasi; teks lengkap
  tidak dibaca. Ringkasan FDA Apple K201525 menyebut pengujian database "per EC57" di bawah 21 CFR 870.2345.

---

## 2. Perilaku perangkat komersial pada detak tinggi

| Perangkat | Dokumen | Yang terverifikasi | Sumber |
|---|---|---|---|
| Apple ECG 2.0 | FDA 510(k) **K201525**, 8 Okt 2020 (`apple_ecg`) | Kelas: Low HR < 50; Sinus 50–99; **High Heart Rate (No AFib) 100–150**; AF 50–99; AF High HR 100–150; Inconclusive; Poor Recording; **High Heart Rate > 150** (tanpa klasifikasi ritme). Kutipan: "high heart rate (no detected AF with heart rate 100-150 bpm)". | Ringkasan FDA |
| Apple ECG 1.0 | De Novo **DEN180044**, 2018 | Ritme > 120 atau < 50 bpm, dan ritme teratur > 100 bpm, masuk "Unclassified" di studi klinis. Tabel di K201525 mencatat High HR > 120 bpm. | Ringkasan FDA |
| Apple ECG 2.0, spesifisitas di detak tinggi | Tabel pembanding di **K240909** (Samsung) | Spesifisitas ritme sinus Apple ECG 2.0 pada **100–150 bpm: 83% (95% CI 77,8–88%)**, vs 99,3% untuk 50–150 bpm (K201525). | Ringkasan FDA |
| Samsung ECG Monitor v1 | **K201168** (4 Agu 2020), **K230292** (2 Mei 2023) | AF 50–120; Sinus 50–100; Inconclusive termasuk "(SR and HR >100 BPM)" dan AF > 120. | Ringkasan FDA |
| Samsung ECG App v1.3 | **K240909** (2 Agu 2024) | Kelas baru *high heart rate (non-AFib)* dan *AFib with high heart rate* (100–150 bpm). Spesifisitas sinus 100–150 bpm 96,3% (93,5–98,9%). | Ringkasan FDA |
| Fitbit/Google ECG | FDA **K200948** (11 Sep 2020) + halaman bantuan Google (`google_ecg_help`) | FDA hanya mencatat kelas AF / NSR / Inconclusive, tanpa ambang. Halaman Google: "over 120 bpm, the Google ECG app can't assess your heart rhythm"; juga < 50 bpm. | Ringkasan FDA, halaman resmi |
| Withings ECG App (ScanWatch) | **K240795** (15 Jun 2025) | **High Heart Rate (No AFib) 100–150**, High HR > 150, AF 50–99, AF High HR 50–150. Pemakaian "while at rest". | Ringkasan FDA |
| AliveCor Kardia | Kardia App User Manual **00LB17.16** (Agu 2022, Amerika Utara) | Kutipan: "Heart rate between 100-140 bpm: The Normal Algorithm declares the EKG as “Tachycardia”". Unclassified < 40 atau > 140 bpm; Normal hanya 50–100 bpm. Ringkasan FDA (K181823, K201985, K210753, K220350) menyebut kelas takikardia tanpa angka. | IFU resmi (tautan dari halaman AliveCor Support "User Manuals") |

**PPG hanya saat diam:**

- **Apple IRN, DEN180042 (2018):** "These data are only captured when the user is still." Kalimat yang sama ada di K212516
  dan K231173.
- **Fitbit IRN, K212372 (8 Apr 2022):** kalimat yang sama. Tabel perbandingannya menyebut analisis tachogram "during periods
  of stillness or sleep". Halaman Google juga menulis bahwa notifikasi ini mengumpulkan data pulsa di latar belakang saat pengguna diam atau istirahat.
- **Samsung IHRN, K230292 dan K240909:** "when the user is still".

---

## 3. Karya terdekat dan penilaian kebaruan

**Pencarian yang dilakukan:**

- PubMed, dengan kombinasi "atrial fibrillation" × "sinus tachycardia" × detect/classif × false positive, serta AF detect ×
  exercise/treadmill/stress test × false positive/specificity × heart rate.
- Crossref.
- Semantic Scholar (sebagian kena rate limit).
- arXiv API.
- Pencarian web, beberapa kueri termasuk `site:cinc.org`.

**Yang tidak dicari:**

- Google Scholar secara langsung.
- Teks lengkap IEEE Xplore.
- Scopus/WoS.
- Arsip CinC secara sistematis.

Sebelum klaim final, ulangi di Google Scholar dengan kueri seperti "AFDB" "sinus tachycardia" "exercise" "RR interval".

**Lima karya terdekat:**

| Karya | Model / data latih | Sapuan bpm sintetis | EKG latihan orang sehat | Alarm per pita detak | 3 kelas dengan AFL |
|---|---|---|---|---|---|
| genzoni2020 | Classifier IBI tradisional; PPG 37 pasien ablasi | Tidak | Tidak (pasien ablasi) | Tidak diketahui (teks lengkap tak terakses) | Tidak (AF vs SR) |
| tse1999 | Algoritme defibrilator atrium implan (elektrogram), 5 pasien | Tidak | Treadmill, pasien implan | Tidak (1 positif palsu) | Tidak |
| sanjana2020 | DL dengan fitur HRV, latih CinC 2017 | Tidak | Tidak (ST dari MIT-BIH Arrhythmia) | Tidak | Tidak |
| kumar2023 | DL, latih 3 dataset PhysioNet; uji 215 hari free-living | Tidak | Tidak (pasien, ambulatori) | Tidak (per konteks aktivitas) | Tidak |
| butkuviene2021 | 3 jenis detektor AF; EKG nyata dan simulasi | Simulasi APB dan noise, bukan sapuan bpm | Tidak | Tidak | Tidak |

Tambahan: torressoto2020 memasukkan data uji latih ke data latih PPG tanpa melaporkan alarm per detak. fu2021 melaporkan
spesifisitas tinggi setelah latihan singkat 60 s, tanpa data detak. Di sisi produk, ringkasan FDA menunjukkan produsen
memisahkan kelas detak tinggi, dan bahkan Apple ECG 2.0 hanya mencapai spesifisitas sinus 83% pada 100–150 bpm (K240909).

**Kesimpulan jujur:** dari pencarian ini **tidak ditemukan** studi yang melakukan apa yang dilakukan paper kita, yaitu:

- model RR-only 3 kelas (dengan AFL) hasil latih AFDB;
- sapuan ritme teratur sintetis per bpm yang menghasilkan ambang bpm per model;
- EKG latihan asli dari subjek tanpa aritmia, dengan alarm per pita detak;
- kuantifikasi kebutaan protokol AFDB.

Bahan dasarnya sudah dikenal. Sinus takikardia sebagai sumber positif palsu detektor AF berbasis interval sudah dilaporkan
(Genzoni 2020, Tse 1999), dan produk komersial menghindarinya lewat desain. Klaim kebaruan sebaiknya dibingkai sebagai
**kuantifikasi dan kritik benchmark**, bukan penemuan fenomena. Kalimat yang aman: "we found no prior study that
quantified…". Kalimat seperti "first to show that sinus tachycardia causes false AF alarms" tidak aman.

---

## 4. Yang belum atau tidak bisa diverifikasi

1. Detail sinus takikardia di Genzoni 2020: jumlah episode, angka positif palsu, classifier, dan detak (teks lengkap tak
   terakses).
2. Isi Littmann 2023 Part II/III terkait AFL 2:1 (teks lengkap tertutup).
3. Peran detak dalam detektor Petrėnas 2015 (abstrak tidak menyebutnya).
4. Nama tiga dataset PhysioNet yang dipakai melatih model Kumar 2023.
5. Ambang Withings sebelum 2025 dan untuk pasar non-AS. Ambang Kardia di IFU luar Amerika Utara (00LB33.1) tidak dibaca.
6. Status "sehat" subjek Wrist dan EPFL (lihat §0.10).
7. Teks lengkap ANSI/AAMI EC57.
