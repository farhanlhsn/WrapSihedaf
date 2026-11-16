# AGENTS.md — panduan untuk AI agent dan anggota tim

Tujuan file ini: siapa pun (manusia atau agent) yang melanjutkan eksperimen SIHEDAF menghasilkan
output dengan bentuk, protokol, dan kualitas yang sama. Ringkasan proyek & cara setup: [README.md](README.md).

## Intent Layer

**Sebelum mengubah apa pun di folder versi, baca `AGENTS.md` / `README.md` folder itu dulu.**

- **v2_revisi**: `v2_revisi/AGENTS.md` — notebook adalah sumber kebenaran modul; `model_export/` dipakai backend.
- **v3_evaluasi_robust**: `v3_evaluasi_robust/AGENTS.md` — protokol evaluasi wajib, layout cache OOF.
- **v3.1_kalibrasi_produksi**: `v3.1_kalibrasi_produksi/README.md` + `BACKEND.md` — paket produksi aktif, panduan backend.
- **v3.2_uji_takikardia**: `v3.2_uji_takikardia/README.md` — uji stres takikardia (wajib), bootstrap bersarang, sensitivitas tanpa 04936.
- **v4_smoothing_ensemble**: `v4_smoothing_ensemble/README.md` — pasca-proses di atas OOF v3; pola seleksi cross-fit konfigurasi + bias.
- **v5_training_afl**: `v5_training_afl/README.md` — pola melatih ulang varian resep pada fold resmi + uji paritas training.
- **v6_cabang_ekg**: `v6_cabang_ekg/README.md` — potongan EKG per window (sejajar dengan RR), CNN/dua cabang, late fusion.
- **v7_augmentasi_takikardia**: `v7_augmentasi_takikardia/README.md` — urutan detak per rekaman (terbukti identik dengan v2),
  pembuat sinus takikardia sintetis, bobot model per fold di `cache/v7/model/`.
- **v8_chapman_shaoxing**: `v8_chapman_shaoxing/README.md` — data eksternal Chapman–Shaoxing (PhysioNet `ecg-arrhythmia`),
  notebook Kaggle-first (clone repo, sidik jari dataset & fold AFDB, hasil dibawa pulang lewat `v8_hasil.zip`).
- **v9_gelombang_ekg**: `v9_gelombang_ekg/README.md` — EKG Chapman sejajar dengan window RR v8, model v6 dilatih dengan
  AFDB + Chapman, pembanding dari cache v8 (dataset Kaggle `sihedaf-v8-cache`).
- **v10_icentia_wearable**: `v10_icentia_wearable/README.md` — data wearable Icentia11k (CC BY-NC-SA, akademik saja): pemindaian
  semua anotasi `.atr` lewat S3 (pembaca vektor, dibuktikan sama dengan wfdb), pemilihan pasien tetap, dua kernel Kaggle (CPU data →
  GPU), pembanding dari dataset `sihedaf-v8-cache` + `sihedaf-v9-cache`, eksplorasi AUROC/kalibrasi AFDB.
- **v11_model_tim**: `v11_model_tim/README.md` + `RANCANGAN.md` — permintaan pembimbing: 6 model berbasis RR yang berbeda dari
  v1, **satu folder per anggota tim** (`<model>/model.py` + `jalankan.ipynb`), kode bersama di `bersama/` (data + sidik jari,
  loop DL identik v3, runner 5×5 yang bisa dilanjutkan, evaluasi protokol). Pengecualian dari aturan folder versi: notebook
  berjalan dengan cwd = **folder model** (satu tingkat lebih dalam), path dicari dari `bersama/data.REPO`, bukan `../cache`.
  Data di Kaggle/Colab dari GitHub Release `v11-data` (`cache/v11_data.zip`, dicek `sidik_jari.json`). aeon (MiniRocket)
  sengaja di luar `requirements.txt`. Bawaan tanpa balancing data (keputusan pemilik repo; alasan di README). Opsi setelan
  `balancing` (`under` dan `over` semua model, `smote` hanya model fitur) dijalankan runner **di dalam setiap fold, hanya pada window latih**,
  untuk menguji arahan pembimbing secara berpasangan vs `bawaan`. Jangan me-balance data di luar fold atau memakai CSV balancing
  turunan v1. Tes pytest dan uji alur QUICK lolos untuk keenam model (22–23 Oktober 2026). MiniRocket butuh `numpy<2.4` (numba
  stabil; numba beta gagal di `np.trapz`). **Selesai (3 November 2026).** Keenam model sudah dilatih penuh,
  `gabungan.ipynb` sudah dijalankan (menulis `cache/v11/gabungan.json`), dan laporan gabungan ada di
  `laporan/laporan_v11_model_tim.pdf`. Hasilnya:
  - tidak ada yang lebih baik dari v3.1;
  - dengan resep yang sama BiGRU terbaik;
  - dengan bobot kelas penuh, BiGRU dan TCN setara v3.1;
  - balancing tidak lebih baik dari bobot kelas. SVM butuh RAM ±7–8 GB per fold (latih di Colab/Kaggle High-RAM). Regresi logistik MiniRocket belum konvergen di 11/25 fold, dan
  `info` di `ringkasan.json` hanya memuat fold terakhir. BiGRU
  (2 November 2026): balancing `under`/`over` lebih buruk daripada bobot kelas; bobot penuh (`weight_power` 1,0) memulihkan AFL 2:1
  (08378) dan menghilangkan ketergantungan pada 04936; regularisasi di atasnya tidak menambah apa-apa (tuning selesai, lihat
  `bigru/eksperimen.py` dan PDF di `bigru/laporan/`). TCN (2 November 2026, Colab): bawaan lebih buruk daripada v3.1 dan BiGRU
  karena margin skor AFL di 08378 tipis sehingga bias cross-fit menghapusnya; bobot penuh (`wp100`) memulihkannya dan setara v3.1
  dan BiGRU `wp100` (pelajaran sama: bobot kelas yang menentukan, bukan arsitektur). Lihat `tcn/eksperimen.py`, PDF di `tcn/laporan/`.
  InceptionTime `wp100` (3 November 2026) adalah pengecualian: 08378 pulih, tetapi margin AFL di window non-AFL juga naik dua kali lipat.
  Akibatnya bias AFL menempel di −7 pada 8 dari 25 fold, dan Macro-F1 tidak naik. Jadi periksa juga margin non-AFL, bukan hanya
  08378 (`inceptiontime/eksperimen.py`).
- **v12_shdb_atrium**: `v12_shdb_atrium/RANCANGAN.md` + `atrium.py` — cabang aktivitas atrium (pembatalan QRST + fitur
  fisiologis) yang digabung di akhir dengan model RR resep v3.1, data Holter AFDB + SHDB-AF (PhysioNet, ODC-BY; 98 rekaman di
  `data/shdb_af/1.0.1/`). Pembaca WFDB format 16/212 dan `.atr` tanpa pustaka wfdb (dibuktikan sama). Dasar: model RR hanya
  mendeteksi AFL yang cepat dan teratur (recall 0,92 vs ≤ 0,21 untuk pola lain). **Berhenti di gerbang (2 November 2026)**:
  frekuensi atrium dominan dan indeks organisasi dari EKG 50 Hz tidak memisahkan AFL dari AFIB secara konsisten antar dataset, juga
  setelah satu revisi pembatalan QRST yang ditetapkan sebelumnya; training tidak dijalankan. `p_konsisten`/`kuantisasi` terlihat
  memisahkan, tetapi dilihat setelah hasil (hipotesis untuk versi berikutnya). Lihat `v12_shdb_atrium/README.md`.
- **v11.1_dua_kelas**: `v11.1_dua_kelas/README.md` + `dua_kelas.ipynb` (logika di `dua_kelas.py`) — evaluasi saja: model yang ada dinilai sebagai 2 kelas
  (Normal vs AF/AFL) dengan protokol lengkap (butir 9–10). Macro-F1 2 kelas 0,916–0,938; BiGRU bawaan terbaik. **Ambil keputusan
  2 kelas dengan bias 3 kelas lalu menggabungkan AFIB + AFL**, bukan satu bias AF/AFL yang dipilih ulang: yang terakhir kehilangan
  penekan AFL dan membuat alarm pada ritme normal teratur 80–95 bpm (sampai 22%), karena AFDB punya AFL teratur ±85–92 bpm
  (hampir semuanya 04936).
- **v3.3_takikardia_asli**: `v3.3_takikardia_asli/README.md` — evaluasi saja: EKG latihan fisik asli (EPFL, Wrist, EPHNOGRAM)
  lewat detektor perangkat dan R-peak manual; model fold v7/v8/v11 dengan bias fold; aturan produk; abstain conformal; **metrik
  episode gaya EC57 (`takikardia_asli.metrik_episode`, dipakai v13–v15)**. v3.1 mengalarmkan 98,4% sinus ≥ 120 bpm; hanya model yang
  pernah melihat sinus takikardia membaik (v8c 60%). Detektor 50 Hz saat aktivitas memicu alarm AF juga di < 100 bpm (21%).
- **v3.4_pembersih_detak**: `v3.4_pembersih_detak/README.md` + `RANCANGAN.md` (dikunci) — front-end di atas detektor produksi: `pembersih.selaraskan` menggeser R-peak ke templat QRS median ±5 s (detektor lama memilih |sinyal| dan melompat R→S), gerbang
  morfologi opsional. Ekstraktor AFDB tanpa wfdb (identik bit demi bit dengan dataset v2), job training resep v3.1 dengan bobot tersimpan (paritas 0,0). Jitter EPFL 10,7 → 2,8 ms dan AFDB setara, tetapi **tidak direkomendasikan**: Se AF turun tipis di
  MIT-BIH (F1) dan 1–3 poin di 4 data eksternal (F2, gerbang). Ide berikutnya diuji di data perangkat, bukan data publik yang sudah dipakai.
- **v13_ecgfounder**: `v13_ecgfounder/README.md` + `RANCANGAN.md` (dikunci) — ECGFounder 1-lead (PKUDigitalHealth, MIT) beku; bobot dan
  kode di `data/ecgfounder/` (unduh dengan SHA256, muat `weights_only` + izin skalar numpy). Zero-shot menurunkan alarm sinus
  takikardia asli 98% → 5% dan lebih baik dari v3.1 di PTB-XL dan Icentia, tetapi AFL 2:1 (08378) hilang dan AFDB 3 kelas tidak lebih
  baik; probe yang dilatih di AFDB kehilangan keunggulan eksternal. Biaya 50 Hz tidak terlihat (kontrol 250 Hz).
- **v14_konteks_menit**: `v14_konteks_menit/README.md` + `RANCANGAN.md` (dikunci) — tahap kedua XGBoost dengan dinamika detak 2–10
  menit di atas OOF v3.1. **Keempat hipotesis tidak lolos** (Macro-F1 lebih buruk, P = 0,033; alarm sinus asli 98% → 96%); hanya
  episode AFIB palsu turun ±4×. AFDB tidak punya episode sinus takikardia untuk dicontoh.
- **v15_kuantisasi_rr**: `v15_kuantisasi_rr/README.md` + `RANCANGAN.md` (pra-registrasi; hash di `logs/`) — fitur v12 `kuantisasi`
  dan `p_konsisten`. Hasil utama **tidak cukup data** (CPSC 2021: AFL hanya 2 subjek). Tambahan PTB-XL: AUROC AFL vs AFIB 0,92/0,86
  (lolos ambang), tetapi tahap gabungan 3 kelas memperburuk AFDB (P = 0,037). Usulan: pakai hanya untuk AFIB vs AFL.
- **v16_hierarki_ecgfounder**: `v16_hierarki_ecgfounder/README.md` + `RANCANGAN.md` (dikunci) — ECGFounder zero-shot **tanpa
  penyetelan** memutuskan sinus vs AF, lalu RR v3.1 (H-0) atau fitur atrium v15 (H-A) memutuskan AFIB vs AFL; semua dari cache.
  Alarm sinus takikardia asli 98,4% → 0,1%, episode AFIB palsu 11,9 → 1,2/jam, 2 kelas setara v3.1 (0,951 vs 0,926; Sp 0,995).
  Tetapi ECGFounder 1-lead 50 Hz menyebut 85% window AFL AFDB (termasuk 08378) sebagai sinus, jadi 3 kelas lebih buruk (0,650) dan
  tahap 2 tidak berpengaruh. Menyetel tahap 1 di AFDB (H-X) membuat bias menempel di batas grid (82%). Keluaran "perlu dicek"
  (tahap 1 sinus + v3.1 AFL) menampung AFL 2:1 tetapi juga 74% sinus takikardia. `hierarki.bersarang3` = bootstrap bersarang
  berpasangan untuk parameter apa pun (paritas dengan v3.2 di-assert).
- **v16.1_dua_kelas**: `v16.1_dua_kelas/README.md` + `RANCANGAN.md` (dikunci) — versi minor tanpa training: tahap 1 v16 sebagai
  detektor AF 2 kelas (D-0). Sangat spesifik (alarm sinus takikardia 0,1%, episode AF palsu 11,4 → 1,0/jam), setara v3.1 di AFDB
  (juga tanpa 04936), lebih baik di PTB-XL, tetapi sensitivitas lebih rendah dan Icentia11k lebih buruk → belum kandidat. Kurva
  ambang (dilihat setelah hasil) menunjukkan t = −1 lebih seimbang; harus dikunci dan diuji di data baru (CPSC 2021 belum pernah
  dinilai). `deteksi_af.DetektorAF` = fungsi prediksi EKG 50 Hz → keputusan (paritas 100% dengan cache v13; ±29 ms/window CPU).
- **v16.2_cpsc2021**: `v16.2_cpsc2021/README.md` + `RANCANGAN.md` (dikunci sebelum sinyal dibaca) — ambang t = −1 (D-1) diuji di
  CPSC 2021. **Kandidat detektor AF 2 kelas untuk uji perangkat**: Macro-F1 0,953 vs v3.1 0,881, sensitivitas sama (0,955 vs 0,951),
  spesifisitas 0,958 vs 0,853; lolos juga di analisis yang dikunci. Label CPSC 2021: `(AFIB`/`(AFL` = awal episode, `(N` = akhir,
  rekaman non-AF tanpa label ritme (`uji_cpsc.interval_af`, sesuai `score_2021.py`); `v15.segmen_ritme` salah untuk kelas Normal.
  Ambang lain yang diuji ulang di CPSC 2021 tidak lagi independen.
- **v16.3_replikasi**: `v16.3_replikasi/README.md` + `RANCANGAN.md` (dikunci) — D-1 diuji ulang di CinC 2017, MIT-BIH, LTAFDB:
  **replikasi sebagian (2/3)**. Macro-F1 dan spesifisitas lebih baik di ketujuh dataset publik; sensitivitas setara/lebih baik di
  Holter, tetapi lebih rendah 3–5 poin di EKG genggam/wearable satu lead (CinC 2017, Icentia11k). Sinus bradikardia LTAFDB 28%
  dialarmkan. Ketujuh dataset publik sudah terpakai untuk D-1; ambang baru hanya dari data perangkat. `tools/unduh_physionet.py`
  memakai MD5 dari ETag S3 bila database tidak punya SHA256SUMS (mis. challenge-2017).
- **v16.4_nsrdb_georgia**: `v16.4_nsrdb_georgia/README.md` + `RANCANGAN.md` (dikunci) — D-1 di NSRDB (orang sehat 24 jam) dan Georgia
  (Challenge 2021, lead I, label SNOMED per rekaman). Alarm palsu orang sehat 15,0% → 5,6%, sinus takikardia klinis 50% → 2%, tetapi
  **Se lebih rendah di EKG klinis 10 s** (0,706 vs 0,770): kekurangan Se D-1 konsisten di data non-Holter (CinC 2017, Icentia, Georgia),
  setara di Holter. Challenge 2021 tidak ada di S3 (`unduh_physionet.py --sumber physionet`, ≤ 6 thread).
- **v16.5_episode_30s**: `v16.5_episode_30s/README.md` + `RANCANGAN.md` (dikunci) — aturan episode AF ≥ 30 s (run window AF, rentang
  ≥ 30 s) dan metrik episode dengan interval sebenarnya (`episode.metrik`: Se ≥ 30 s, +P, durasi, palsu/jam). D-1 + 30 s: alarm palsu
  orang sehat 1,23/jam (target 1,0, gagal tipis), olahraga 0,13 (v3.1: 8,8), Se episode 0,91–0,98. EDB diproses dengan mesin v16.4.
  Jangan mengubah `U164.CACHE` dari modul lain (bug yang pernah membuat NSRDB dicari di folder yang salah).
- **v16.6_episode_celah**: `v16.6_episode_celah/README.md` + `RANCANGAN.md` (dikunci + amandemen A) — grid durasi × celah di data v16.5,
  validasi di SHDB-AF (seluruh 98 Holter), afpdb, incartdb. Celah menaikkan alarm palsu; 45 s terpilih tapi Se SHDB-AF 0,70 (gagal).
  **D-1 + 30 s (S1) lolos di data baru** (Se 0,942, alarm palsu 0,77–0,89/jam): sistem terbaik saat ini untuk deteksi AF 2 kelas
  (episode). Pelajaran: jangan memilih durasi aturan dengan Se untuk episode yang lebih pendek dari aturan itu. Data publik Holter yang
  relevan kini terpakai; titik kerja berikutnya hanya dari data perangkat. `celah.jalankan(ds, bagian, n_bagian)` membagi pemrosesan.
- **v16.7_konfirmasi_wearable**: `v16.7_konfirmasi_wearable/README.md` + `RANCANGAN.md` (dikunci) — uji konfirmasi primer S1 (D-1 + 30 s)
  di segmen utuh Icentia11k (600 pasien; seleksi dari `cache/v10/icentia_scan.npz`, di luar seleksi v10; unduh S3 + SHA256) dan nstdb
  (tidak ada di S3). **S1 terkonfirmasi**: Se episode 0,940, alarm palsu 0,50/jam (v3.1 + 30 s: 0,899; 2,33). Derau SNR ≤ 6 dB tetap
  memicu 5–8 alarm/jam (gerbang kualitas berikutnya: jangan disetel di nstdb lalu diklaim di nstdb).
- **v16.8_gerbang_kualitas**: `v16.8_gerbang_kualitas/README.md` + `RANCANGAN.md` (dikunci + amandemen A) — gerbang SQI templat QRS
  (`pembersih.selaraskan` v3.4) untuk S1. **Gagal**: derau otot (afpdb + `ma` nstdb, tidak dipakai memilih) tidak turun; hanya derau
  elektroda (jenis derau tahap A). q terpilih di batas grid. afpdb berderau dibuat dengan `gerbang.bising` (SNR setelah bandpass
  0,5–40 Hz). Gerbang derau berikutnya butuh fitur frekuensi tinggi pada sinyal asli dan data derau perangkat.
- **v16.9_sinus_bradikardia**: `v16.9_sinus_bradikardia/README.md` + `RANCANGAN.md` (dikunci) — evaluasi saja dari cache v16.2–v16.7
  (`bradikardia.muat` memuat 10 dataset dengan detak per window). Sinus bradikardia LTAFDB: S1 8,89 alarm palsu/jam vs v3.1 0,48, tetapi 90%
  dari 3 subjek; detak lambat (< 50 bpm) juga menaikkan alarm S1 di dataset lain (pola U). Window yang dialarmkan teratur dan tanpa ektopik
  (anotasi .atr), jadi penyebabnya bentuk gelombang ECGFounder, bukan RR. Aturan detak tidak bisa (11–16% window AF < 60 bpm). Uji perangkat
  wajib memuat rekaman tidur/detak istirahat rendah.
- **v17_fusi_wearable**: `v17_fusi_wearable/README.md` + `RANCANGAN.md` (dikunci) — regresi logistik 3 parameter pada [d1, log-odds AF
  v3.1] dilatih di CinC 2017 (CC BY 3.0; Icentia CC BY-NC-SA hanya untuk evaluasi). Sensitivitas Icentia naik 0,843 → 0,921, tetapi alarm
  sinus takikardia asli 57% dan Holter lebih buruk dari D-1 → **tidak menggantikan D-1**. Titik kerja wearable vs aktivitas hanya bisa
  dipilih dari data perangkat. Data latih untuk produk harus berlisensi komersial (cek lisensi sebelum melatih).
- **v1_lama**: `v1_lama/README.md` — arsip beku, hanya path yang disesuaikan.

### Invariant global

- Folder versi yang sudah selesai **tidak diubah logikanya**. Penyesuaian path dan dokumentasi boleh.
  - Eksperimen atau training baru → `vN_<nama_singkat>/` (N berikutnya, sekarang v18).
  - Tindak lanjut tanpa eksperimen baru (kalibrasi, pengemasan, perbaikan) → versi minor `vN.M_<nama>/` (contoh: v3.1).
- Notebook dijalankan dengan **folder versinya sebagai cwd**. Data lewat `../data/afdb`, cache lewat `../cache`.
  Jangan menulis keluaran ke root repo.
- Versi baru memakai kode versi lama lewat `sys.path` (contoh: `v3_evaluasi_robust/sihedaf_cv.py` mengimpor
  `sihedaf_models` dari `v2_revisi`). Jangan menyalin definisi model atau praproses.
- `data/`, `cache/`, `logs/`, `**/laporan/_build/` tidak di-commit. Notebook, `model_export/`, laporan PDF,
  dan pembuat laporannya di-commit.
- **Paket produksi aktif: `v3.1_kalibrasi_produksi/model_export/`.** Paket v2 adalah titik rollback. Jangan mengubah
  paket mana pun tanpa persetujuan eksplisit pemilik repo. Menjalankan notebook v2 (termasuk `SIHEDAF_QUICK=1`)
  mengekspor ulang model v2; menjalankan notebook v3.1 menulis ulang paket v3.1.
- Arsitektur atau resep baru yang inputnya hanya interval RR **tidak** dicoba lagi (v3–v5 sudah menghabiskannya).
  Pengecualian tercatat: v11 (permintaan pembimbing, studi perbandingan antar model tim; protokol tetap sama).
  Bentuk gelombang EKG dari 23 pasien AFDB juga sudah dicoba (v6) dan overfit per pasien. Sinus takikardia sintetis (v7)
  hanya menggeser batas alarm palsu ke ±135 bpm; di atasnya interval RR tidak cukup. Data eksternal dengan sinus takikardia
  asli (v8) menurunkan alarm ST 42% → 9%, tetapi AFL 2:1 dan SVT ikut hilang. Bentuk gelombang EKG dengan 34 ribu pasien
  Chapman (v9) menurunkannya ke 2% di Chapman, tetapi gagal di AFDB: masalahnya kini pergeseran domain antar-dataset. Menambah 510
  pasien AFL wearable (v10, Icentia) memperbaiki Icentia dan Chapman tetapi memperburuk AFDB: batas AFL/AFIB tiap dataset berbeda.
  Data publik lain dengan resep yang sama kemungkinan tidak memutuskan; yang dibutuhkan rekaman perangkat SIHEDAF berlabel. Eksperimen baru harus menambah
  **pasien** (data eksternal) atau informasi yang benar-benar baru. Model DL baru memakai **bobot kelas lunak** (`class_weights(y) ** 0.5`,
  lihat `v5_training_afl/latih.train_dl_weighted`) kecuali ada alasan tercatat.
- Satuan IBI selalu **milidetik**. Label: 0 Normal, 1 AFIB, 2 AFL.

## Lingkungan

Anggota tim memakai setup berbeda. Kode tidak boleh bergantung pada mesin tertentu: tanpa path absolut,
tanpa nama GPU, tanpa asumsi jumlah core.

- **Disarankan: WSL2 / Linux dengan GPU NVIDIA (CUDA)** + venv dari `requirements.txt`. Notebook dijalankan dengan
  `python tools/run_notebook.py <notebook>`: output tersimpan per sel, status ke `logs/`. Kernel yang dipakai:
  `$SIHEDAF_KERNEL`, lalu `sihedaf-v2` jika terpasang, selain itu `python3`.
- **Alternatif: Kaggle / Google Colab (GPU).** Langkahnya:
  1. Clone repo, `pip install -r requirements.txt`.
  2. Unduh AFDB dengan `python tools/download_afdb.py`, atau `wfdb.dl_database("afdb", "data/afdb")`. Data PhysioNet lain
     diunduh dari mirror resmi S3 (`https://physionet-open.s3.amazonaws.com/<db>/<versi>/`, dicek dengan `SHA256SUMS.txt`),
     **bukan** zip dari physionet.org: dari Kaggle kecepatannya terukur ±0,05 MB/s, sehingga notebook terlihat macet (v8).
  3. Pindah ke folder versi (`%cd vN_...`) sebelum menjalankan notebook. Path `../data` dan `../cache` harus berlaku.
  4. Sesi bisa terputus. Simpan `cache/` ke Google Drive / output Kaggle; job yang sudah ada di cache dilewati.
  5. CPU di sana sedikit (2–4 core). Di v3, set `ecfg.cpu_models = ()` agar BiGRU juga jalan di GPU.
  6. Untuk laporan PDF: `apt-get install -y fonts-liberation` (metrik sama dengan Arial), lalu jalankan dua tahap
     `build_report.py` dengan Python yang sama.
- **CPU saja** bisa, tapi training DL lebih lambat (terukur 1,5× untuk BiGRU s.d. ±11× untuk ResNet1D).
  `SIHEDAF_QUICK=1` hanya untuk uji alur, bukan untuk hasil.
- Windows + WSL: dari Git Bash, panggil dengan `MSYS_NO_PATHCONV=1 wsl.exe -- python ...` (tanpa variabel itu path rusak).
- Job panjang (> 2 menit) dijalankan di background dan **bisa dilanjutkan**: simpan hasil per (model, ulangan)
  ke cache dan lewati yang sudah ada (pola `sihedaf_cv.run_dl_job`).
- Model DL di sini kecil, jadi bottleneck-nya overhead Python per batch, bukan GPU. Satu proses GPU paling
  efisien; menambah worker paralel tidak menambah kecepatan secara linear.
- **Hasil antar mesin bisa sedikit berbeda** (GPU non-deterministik, versi pustaka). Perbandingan berpasangan hanya
  sah jika model baru dan referensi memakai OOF dari cache yang sama. Untuk itu, bagikan
  `cache/oof_robust/` (±90 MB) dan `cache/dataset_v2_*.npz` (±30 MB) antar anggota tim (Drive / Kaggle Dataset),
  atau hitung ulang referensi di lingkungan Anda dengan notebook v3.

## Protokol evaluasi (wajib untuk semua klaim "lebih baik")

Detail dan kodenya ada di `v3_evaluasi_robust/`. Ringkasnya:

1. Pembagian pasien memakai `cache/oof_robust/detector_e30_r5_s2026/folds.npz` (5 ulangan × 5 fold, seed 2026).
   Jangan memakai `StratifiedGroupKFold(shuffle=True)`: dengan 23 pasien, partisinya sama untuk seed berapa pun.
2. Data uji = window detektor 50 Hz (`cache/dataset_v2_detector.npz`), yaitu kondisi produksi.
3. Laporkan metrik **pooled OOF** rata-rata ulangan + **CI 95% bootstrap pasien** (2.000×, stratified),
   bukan rata-rata per fold.
4. Bandingkan **berpasangan** terhadap referensi **v3.1** (= `MicroTransformer +bias` di OOF v3: bias cross-fit grid −7…+3)
   dengan sampel bootstrap yang sama.
   Vonis: P(Δ>0) ≥ 0,95 lebih baik; ≤ 0,05 lebih buruk; selebihnya tidak terbedakan.
5. Selalu sertakan recall AFL per pasien (7 pasien AFL) dan asal AFL palsu.
6. Bias kelas dipelajari cross-fit antar fold (`sihedaf_cv.crossfit_bias`, grid −7…+3). Hiperparameter pasca-proses
   (mis. lebar smoothing) dipilih dengan cara yang sama, bersama bias (`v4_smoothing_ensemble/pascaproses.crossfit_select`).
7. Untuk ide yang mengubah perilaku antar-window, laporkan juga metrik episode (alarm AFL palsu/jam, deteksi episode).
8. Varian training baru: pakai seed latih yang sama dengan referensi (`2026 + 100·r + k`) dan buktikan dulu bahwa fungsi
   training Anda identik dengan referensi pada pengaturan standar (uji paritas, lihat notebook v5).
9. **Uji stres takikardia (wajib sejak v3.2).** AFDB hampir tidak punya ritme normal yang cepat (hanya 0,1% window Normal
   di atas 120 bpm), jadi metrik 1–8 tidak menangkap alarm palsu pada sinus takikardia. Untuk setiap model kandidat:
   - jalankan `v3.2_uji_takikardia/takikardia.stress_test` (ritme teratur sintetis 60–170 bpm, 4 jenis variasi) dan laporkan
     ambang bpm di mana model berhenti menebak Normal. Referensi v3.1: 115 bpm;
   - laporkan juga % window Normal asli yang ditebak AFIB/AFL per pita detak (`takikardia.by_hr_bin`, OOF).
   Karena itu, model kandidat harus **menyimpan bobotnya** (atau fungsi prediksi). v4–v6 tidak menyimpannya, sehingga
   tidak bisa diuji di data sintetis.
10. Klaim "lebih baik" yang akan mengganti paket produksi: sertakan juga **bootstrap bersarang** (bias dipilih ulang di setiap
    sampel, `takikardia.nested_bootstrap`) dan sensitivitas **tanpa pasien 04936** (58% window AFL berasal dari satu pasien).
    Contoh: v3.1 vs v1 P = 0,98 turun menjadi 0,94 tanpa 04936.

## Definition of Done satu versi ("selesai mendapatkan hasil")

1. Notebook versi ter-eksekusi penuh dari atas ke bawah (lewat `tools/run_notebook.py`, Jupyter, Colab, atau
   Kaggle), output tersimpan, tanpa error, dengan sel **Kesimpulan** berisi angka nyata. Catat di README versi
   di mana hasilnya dijalankan (lingkungan, GPU, waktu).
2. `vN_.../README.md` berisi tujuan, file, cara menjalankan, dan tabel hasil.
3. Laporan PDF di `vN_.../laporan/` dibuat oleh `vN_.../laporan/build_report.py` (dua tahap: `--extract` menghitung
   angka & grafik dari cache, lalu tanpa argumen merakit PDF dengan ReportLab). Template: salin
   `v3_evaluasi_robust/laporan/build_report.py`; font lewat `tools/report_fonts.py`.
4. Laporan **wajib** memuat perbandingan dengan versi sebelumnya (pada data uji & fold yang sama) dan dengan v1, serta
   hasil uji stres takikardia (protokol butir 9) untuk model yang diusulkan.
5. Angka laporan diverifikasi otomatis terhadap ringkasan notebook (`assert`, lihat `extract()` v3), lalu setiap
   halaman dirender ke PNG (`pypdfium2`) dan **dilihat** sebelum diserahkan.
6. Baris baru di tabel "Versi dan hasil" di `README.md` root.

## Gaya laporan

- Bahasa Indonesia yang sederhana. Pembaca adalah mahasiswa/tim, bukan reviewer jurnal; jelaskan istilah (OOF, CI, bias).
- Urutan halaman: ringkasan (kartu angka + jawaban singkat + temuan) → perbandingan dengan versi sebelumnya →
  metode → hasil → per pasien → kesimpulan & langkah berikutnya → batasan → lampiran teknis (cara reproduksi).
- Angka memakai koma desimal, pemisah ribuan titik, dan tanda minus "−". Selalu tulis CI untuk metrik AFL.
- A4, font Arial atau Liberation Sans (lewat `tools/report_fonts.py`), warna tata letak `#008B83` / `#172B43`. Warna data memakai palet tervalidasi
  `#2a78d6`, `#eb6834`, `#1baf7a` (validator dataviz: semua PASS). Bedakan juga dengan bentuk marker, bukan warna saja.
- Jujur soal trade-off. Contoh dari v3: sistem sekarang lebih baik, tetapi recall AFL turun.
  Tulis juga hal yang tidak bisa disimpulkan.

## Anti-pattern yang sudah pernah terjadi

- Menyimpulkan model terbaik dari satu partisi atau rata-rata per fold (v1 dan v2): selisihnya noise.
- Grid bias terlalu sempit sehingga optimum terpotong (v2: −3…+3, optimum −4…−5,5). Pencarian lalu "mengakali"
  batas itu lewat kelas lain (AFIB +2), yang diam-diam menurunkan recall Normal ke 0,85. Selalu cek bias di batas grid.
- Window AFL tumpang tindih 90% (stride 1 s) **di data uji**, sehingga skor terlihat tinggi (v1). Sebagai augmentasi
  data latih, stride rapat justru membantu (temuan v3).
- Front-end `find_peaks` tanpa filter di backend (v1): IBI terlihat wajar tetapi urutannya salah.
- Fallback diam-diam (bobot acak, scaler dummy) saat file hilang. Harus error.
- `*.json` di `.gitignore` sempat membuat `model_export/sihedaf_v2.json` tidak ter-commit.
- Melatih model berkapasitas besar (CNN EKG ±400 ribu parameter) bersama cabang kecil yang sudah kuat (v6): cabang besar
  overfit per pasien dan merusak cabang kecil. Tanda bahayanya: bias cross-fit yang melompat antar fold.
- Menyalin kode yang hanya ada di sel notebook (mis. ekstraktor v2) tanpa membuktikan salinannya identik. v5 meng-`assert`
  dataset hasil salinan sama bit demi bit dengan cache v2 sebelum memakainya.
- Menilai smoothing hanya dari Macro-F1 window (v4): ia terlihat "setara", padahal menghapus separuh episode AFL. Selalu
  lihat metrik episode dan recall AFL per pasien; episode pendek (1–20 window) paling mudah hilang.
- Data uji yang tidak memuat kondisi pemakaian penting (v1–v6). AFDB hampir tidak punya sinus takikardia, sehingga semua
  versi lolos protokol padahal menebak ritme teratur ≥ ±115 bpm sebagai AFIB/AFL. Masalah ini ditemukan reviewer eksternal
  (v3.2). Sebelum klaim siap produksi, tanyakan kondisi nyata apa yang tidak ada di data uji, lalu buat uji stresnya.
- Memakai label data eksternal tanpa memeriksa kewajarannya secara fisiologis (v8, run pertama). Di PhysioNet
  `ecg-arrhythmia`, bagian Ningbo tidak punya AFIB, tetapi "AFL"-nya berdetak dan tidak teratur seperti AFIB. Model yang
  dilatih dengan label itu kehilangan deteksi AFL di AFDB (08378: 0,98 → 0,00). Sebelum melatih, bandingkan detak & CV IBI
  per label dan per sumber dengan AFDB (`v8_chapman_shaoxing/chapman.afl_label_check`).
- Menyimpulkan dari satu domain saja (v9). Model bentuk gelombang sangat baik di Chapman tetapi rusak di AFDB. Tanda bahayanya:
  bias cross-fit menyentuh **kedua** batas grid (−7 dan +3), dan AFL palsu menumpuk di window non-AFL milik pasien AFL. Selalu
  evaluasi di AFDB **dan** di data eksternal, dan laporkan keduanya berdampingan.
- Menganggap kelas dengan nama sama berarti sama di setiap dataset (v10). AFL Icentia kebanyakan blok bervariasi (86 bpm, CV 0,20),
  AFL AFDB hampir semuanya 2:1 (139 bpm, CV 0,03). Model yang belajar AFL dari Icentia menyebut AF "kasar" AFDB sebagai AFL. Selain cek
  detak & CV (v8), bandingkan fenotipe per kelas antar dataset, dan saat AFDB memburuk, cek AUROC per kelas dan asal AFL palsu per
  kelas asli (`v10_icentia_wearable/icentia.eksplorasi_afdb`) sebelum menyalahkan kalibrasi.
- Mengandalkan abstain/conformal untuk kondisi yang tidak ada di data kalibrasi (v3.3). Jaminan cakupan hanya berlaku bila data
  uji mirip data kalibrasi; pada sinus takikardia model yakin dalam kesalahannya, sehingga 0% window menjadi "tidak yakin".
- Membaca label data eksternal tanpa mencocokkan dengan kode penilaian resminya (v15/v16.2). Di CPSC 2021 label ritme hanya menandai
  awal dan akhir episode AF; aturan "segmen setelah label" membuang semua rekaman non-AF. Sebelum memakai dataset baru, baca kode
  penilaian/pemuat label resminya dan cek jumlah rekaman per kelas header vs per label.
- Menyetel ambang "sinus vs AF" di AFDB (v16 H-X), atau melatih fusi ECGFounder + RR di CinC 2017 (v17, 1,4% window non-AF ≥ 120 bpm). AFDB tidak punya sinus takikardia, jadi pencarian bias hanya melihat AFL yang
  hilang dan mendorong AF ke batas grid; alarm sinus takikardia ikut naik dan hasilnya tidak berpindah ke data lain. Ambang tahap
  sinus vs AF dari model fondasi dibiarkan bawaan, atau dipilih dengan data yang memuat sinus takikardia.
- CI dengan bias yang sudah dipilih dianggap tetap (v3.1 melaporkan P = 1,00). Pemilihan bias sendiri juga tidak pasti;
  bootstrap bersarang memberi P = 0,993.
