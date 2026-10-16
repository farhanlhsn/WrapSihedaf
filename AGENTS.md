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
- **v1_lama**: `v1_lama/README.md` — arsip beku, hanya path yang disesuaikan.

### Invariant global

- Folder versi yang sudah selesai **tidak diubah logikanya**. Penyesuaian path dan dokumentasi boleh.
  - Eksperimen atau training baru → `vN_<nama_singkat>/` (N berikutnya, sekarang v8).
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
  Bentuk gelombang EKG dari 23 pasien AFDB juga sudah dicoba (v6) dan overfit per pasien. Sinus takikardia sintetis (v7)
  hanya menggeser batas alarm palsu ke ±135 bpm; di atasnya interval RR tidak cukup. Eksperimen baru harus menambah
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
  2. Unduh AFDB dengan `python tools/download_afdb.py`, atau `wfdb.dl_database("afdb", "data/afdb")`.
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
- CI dengan bias yang sudah dipilih dianggap tetap (v3.1 melaporkan P = 1,00). Pemilihan bias sendiri juga tidak pasti;
  bootstrap bersarang memberi P = 0,993.
