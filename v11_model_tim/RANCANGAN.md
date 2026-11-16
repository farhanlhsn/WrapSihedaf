# Rancangan v11: enam model berbeda dari v1, satu folder per anggota tim

Tanggal: 20 Oktober 2026. Status: **disetujui pemilik repo** (20 Oktober 2026).

**Revisi pelaksanaan (20 Oktober 2026, keputusan pemilik repo):** WSL akan dihapus, jadi kode **ditulis tanpa
dijalankan**. Tes pytest, uji paritas, dan uji jalan (6.9) tetap ditulis, tetapi baru dijalankan nanti oleh pemilik repo
atau tim. Venv MiniRocket terpisah di PC (6.8) tidak dibuat. Notebook model ditulis langsung (isinya sama, kecuali baris
`MODEL`); tidak ada generator notebook.

**Revisi 22 Oktober 2026 (permintaan pemilik repo): opsi balancing per fold.** Bawaan tetap tanpa balancing. Ditambah
kunci setelan bersama `balancing` (`None` / `"under"` / `"over"` / `"smote"`, `bersama/balancing.py`; `"over"` ditambahkan
2 November 2026). Runner menerapkannya pada window
latih setiap fold, sehingga arahan pembimbing bisa diuji dengan protokol yang sama dan dibandingkan berpasangan dengan
setelan `bawaan` (`evaluasi.banding_setelan`, sel opsional di `gabungan.ipynb`). SMOTE hanya untuk model fitur
(`INPUT = "fitur"`, kontrak baru `model.py`). Pemicunya dataset balancing yang dibuat di luar repo (turunan v1, satu partisi
14/9 pasien, tanpa kolom pasien), sehingga tidak bisa dipakai langsung. Detail di README bagian "Opsi balancing". Butir
"Bukan tujuan: balancing dataset" di bawah kini berarti: balancing bukan bawaan dan bukan rekomendasi.

## 1. Latar dan alasan

- Arahan pembimbing (lewat catatan teman; pemilik repo tidak hadir di bimbingan):
  1. dataset di-balance dulu;
  2. tiap anggota tim (6 orang) mencoba satu model yang **berbeda dari model v1**, yaitu LightGBM, 1D-ResNet,
     MicroTransformer, serta RandomForest/HistGradientBoosting di EDA v1. Tim melanjutkan penelitian v1.
- **Keputusan pemilik repo: data tidak di-balance.** Alasannya:
  - AFL hanya 0,7% window dan berasal dari 7 pasien; balancing tidak menambah pasien;
  - SMOTE, sampler, dan oversampling AFL sudah dicoba (v1, v5a) dan tidak mengalahkan bobot kelas;
  - undersampling membuang ±90% data, termasuk hampir semua dari ±100 window Normal > 120 bpm.
  Ketidakseimbangan tetap ditangani dengan bobot kelas dan bias cross-fit. Pemilik repo membawa alasan ini ke pembimbing.
- Tujuan v11: kode siap-jalan untuk 6 model, satu folder per model, supaya tiap anggota mudah bereksperimen dan
  hasilnya bisa disandingkan secara adil.

## 2. Tujuan

1. Enam folder model. Masing-masing jalan dengan "Run All" di PC, Kaggle, atau Colab.
2. Data, fold, seed, bobot kelas, dan evaluasi sama untuk semua model (protokol AGENTS.md butir 1–6, 8, 9).
3. Untuk bereksperimen, anggota cukup mengubah **satu tempat**: setelan/arsitektur di `model.py` foldernya.
4. Tabel gabungan otomatis untuk model yang sudah punya hasil, berdampingan dengan v1 dan v3.1.
5. Kode terbukti jalan (uji jalan singkat) sebelum dibagikan.

## 3. Bukan tujuan

- Training penuh atau hasil penelitian. Anggota menjalankan modelnya sendiri. Satu-satunya yang dijalankan Claude
  adalah uji jalan singkat (6.9), dan hasilnya dibuang.
- Balancing dataset.
- Laporan PDF. Dibuat setelah hasil terkumpul; v11 belum "selesai" menurut Definition of Done sampai saat itu.
- Model berbasis sinyal EKG.
- Mengubah paket produksi, folder versi lama, atau `cache/oof_robust/`.
- Commit/push dan unggah ke Kaggle (dilakukan pemilik repo).

## 4. Anti-tujuan (dianggap gagal walaupun kodenya jalan)

- Hasil antar model tidak bisa dibandingkan karena data, fold, seed, atau evaluasinya berbeda.
- Hiperparameter dipilih dengan melihat fold uji.
- Loop training baru tanpa bukti identik dengan referensi (protokol butir 8).
- Definisi model/praproses lama disalin, bukan diimpor.
- Fallback diam-diam (berkas hilang, hash tidak cocok, pustaka tidak ada). Harus error.
- Model tidak menyimpan bobotnya, sehingga tidak bisa diuji takikardia.
- Eksperimen anggota saling menimpa hasil.
- Kode error di tangan teman (Kaggle/Colab).

## 5. Keputusan pemilik repo

- Tanpa balancing: data asli + bobot kelas.
- Enam model, semuanya berbasis interval RR (urutan IBI atau 16 fitur HRV), bukan model v1:
  SVM (RBF), XGBoost, MiniRocket, **BiGRU + Attention (pemilik repo)**, TCN, InceptionTime.
- Satu folder per model. Claude menyiapkan seluruh kode tetapi tidak menjalankan training penuh.
- Uji jalan singkat diizinkan (±10–15 menit, hasil dibuang).

## 6. Keputusan Claude (boleh diganti saat review)

### 6.1 Struktur folder

```
v11_model_tim/
  README.md          panduan tim: pembagian model, cara menjalankan (PC/Kaggle/Colab), cara berbagi hasil
  RANCANGAN.md       dokumen ini
  bersama/           kode bersama, tidak diubah anggota
    data.py          memuat data + fold + OOF pembanding, cek hash, menemukan folder data (repo atau Kaggle/Colab)
    latih_dl.py      loop training DL umum (dibuktikan identik dengan v3/v5)
    runner.py        25 job (5 ulangan × 5 fold), bisa dilanjutkan, simpan OOF + bobot + prediksi uji stres
    evaluasi.py      metrik standar, bias cross-fit, perbandingan berpasangan, uji takikardia, seleksi cross-fit
  svm/  xgboost/  minirocket/  bigru/  tcn/  inceptiontime/
    model.py         definisi model + SETELAN; satu-satunya tempat yang diubah anggota
    jalankan.ipynb   Run All: training → evaluasi → ringkasan
    README.md        modelnya apa, setelan bawaan & alasannya, perkiraan waktu, ide eksperimen
  gabungan.ipynb     tabel semua model yang hasilnya ada, + v1 + v3.1
```

- Notebook dijalankan dengan cwd = folder modelnya. Ini satu tingkat lebih dalam dari aturan AGENTS.md ("cwd = folder
  versi, `../cache`"). Karena itu `bersama/data.py` mencari root repo (folder yang berisi `AGENTS.md`) dan tidak memakai
  path relatif tetap. Entri Intent Layer v11 di AGENTS.md mencatat pengecualian ini.
- Notebook dan kode diserahkan **tanpa output**. *Diubah setelah v11 selesai (3 November 2026):* notebook di repo kini
  menyimpan run `bawaan` pemilik repo dan sel Kesimpulan sebagai catatan hasil (lihat README, "Status").
- **Cara teman mendapat dan mengubah kode** (keputusan Claude): sel pertama `jalankan.ipynb` berisi `REPO_URL`,
  `BRANCH`, `NAMA_SETELAN`, dan kamus `SETELAN` yang menimpa bawaan `model.py`. Di Kaggle/Colab, sel itu meng-clone
  repo pada branch tersebut (pola notebook v8). Eksperimen setelan cukup di sel itu. Mengubah arsitektur dilakukan di
  `model.py` pada fork/branch milik anggota sendiri, lalu `BRANCH`/`REPO_URL` diarahkan ke sana. `ringkasan.json`
  mencatat commit hash dan apakah ada perubahan yang belum di-commit.

### 6.2 Data dan pembanding

- Data: `cache/dataset_v2_detector.npz` (X = IBI + ΔIBI, 30 × 2; M = mask; F = 16 fitur HRV; y; records) dan
  `cache/oof_robust/detector_e30_r5_s2026/folds.npz`. Setiap fold k: latih di pasien fold ≠ k, uji di fold k.
- Pembanding dari cache yang sama (tidak dilatih ulang):
  - v3.1 = OOF `MicroTransformer_r*.npz` + bias cross-fit;
  - v1 = `versi_lama/oof_r*.npz`, argmax.
- `v11_model_tim/sidik_jari.json` (**di-commit**) berisi SHA-256 semua berkas di atas. Hash tidak cocok → error.
- Kaggle/Colab: fungsi pembantu membuat `v11_data.zip` mandiri (dataset, folds, OOF v3.1 & v1, kontrol v7, sidik
  jari) untuk diunggah pemilik repo. `data.py` menerima lokasi lewat variabel lingkungan `SIHEDAF_DATA` (folder hasil
  ekstrak), atau memakai `cache/` repo jika tidak diset.
- **Hosting (keputusan pemilik repo): GitHub Release** di `farhanlhsn/WrapSihedaf`. Pemilik repo yang membuat release
  dan mengunggah zip; Claude menyiapkan zip dan perintah `gh release create`-nya. Di Kaggle/Colab, notebook mengunduh
  zip dari URL release (bisa diubah di sel pertama), mengekstrak, lalu memeriksa hash. Kaggle butuh internet aktif.
  Tanpa internet, anggota menambahkan zip sebagai dataset Kaggle dan mengisi `SIHEDAF_DATA`. Ini langkah manual yang
  tertulis di README, bukan fallback otomatis. README mencantumkan atribusi AFDB (ODC-BY).

### 6.3 Resep training (sama untuk semua model)

- Seed latih `2026 + 100·r + k`. 5 ulangan × 5 fold = 25 model per setelan.
- **Bobot kelas lunak** `class_weights(y) ** 0.5` (aturan AGENTS.md). Model DL memakainya di loss; model klasik
  sebagai `sample_weight`.
- Bawaan model DL = resep v3: z-score bermask dari data latih, AdamW, cosine, 30 epoch, batch 256, lr 1e-3,
  weight decay 1e-4. Boleh diubah per model di `model.py`.
- `latih_dl.py` menerima model apa pun (bukan hanya `build_model` v2). Uji paritas wajib: dengan MicroTransformer,
  bobot penuh, dan seed yang sama, hasilnya identik dengan `sihedaf_cv.train_dl` (pola uji paritas notebook v5).
- Ukuran model DL dijaga kecil (≤ ±100 ribu parameter, sekelas model yang ada) untuk menghindari overfit per pasien
  (pelajaran v6). Jumlah parameter dicetak di ringkasan.

### 6.4 Setelan bawaan per model

| Model | Input | Bawaan | Catatan |
|---|---|---|---|
| SVM | 16 fitur HRV, standardisasi dari data latih | Aproksimasi kernel RBF (`Nystroem`, 2.000 komponen) + SVM linear (`LinearSVC`, squared hinge) | SVM RBF biasa tidak sanggup ±130 ribu window × 25 fold. Skor keputusan diubah ke pseudo-probabilitas dengan softmax; bias cross-fit menyesuaikan skalanya. |
| XGBoost | 16 fitur HRV | `hist`, 400 pohon, lr 0,05, kedalaman 6 | CPU; GPU opsional. |
| MiniRocket | urutan IBI + ΔIBI (30 × 2, nol di luar mask) | `aeon` MiniRocket multivariat, ±2.000 kernel → standardisasi → regresi logistik, C tetap | Kernel dikurangi dari 10.000 agar fitur muat di RAM ±7,6 GB (WSL). Tanpa CV internal. |
| BiGRU + Attention | urutan IBI | `BiGRUAttention(hidden=32)` diimpor dari `v2_revisi` | Dilatih ulang dengan bobot lunak di GPU dan **disimpan bobotnya**; hasil v3 (bobot penuh, tanpa bobot tersimpan) hanya jadi pembanding. |
| TCN | urutan IBI | Blok residual konvolusi dilatasi (1, 2, 4, 8), kernel 3, 32 kanal, dropout 0,1, global average pooling bermask | Baru. |
| InceptionTime | urutan IBI | Satu jaringan (bukan ensemble 5), modul Inception dengan kernel disesuaikan panjang 30, residual, pooling bermask | Baru. Ensemble 5 jaringan versi asli terlalu mahal untuk 25 fold. |

### 6.5 Evaluasi (`bersama/evaluasi.py`, memakai ulang kode v3/v3.2/v4 lewat impor)

Untuk tiap setelan model:
- Macro-F1, F1/precision/recall per kelas, pooled OOF rata-rata 5 ulangan, CI 95% bootstrap pasien 2.000×
  (stratified).
- Hasil argmax dan +bias cross-fit (grid −7…+3), dengan peringatan bila bias menyentuh batas grid.
- Perbandingan berpasangan vs v3.1 dan vs v1 dengan sampel bootstrap yang sama; vonis P ≥ 0,95 / ≤ 0,05.
- Recall AFL per pasien (7 pasien) dan asal AFL palsu.
- Uji takikardia: setiap model fold memprediksi set uji stres v3.2 (60–170 bpm × 4 jenis variasi × 50 window) dengan
  bias fold-nya, lalu dicari ambang bpm di mana model berhenti menebak Normal. Referensinya dua: paket produksi v3.1
  (115 bpm, satu model dari 23 pasien) dan kontrol MicroTransformer per fold dari v7 (`cache/v7/uji_kontrol_r0.npz`,
  bobot lunak, ulangan 0, 115–120 bpm), yang sebanding dengan model fold v11. Berkas kontrol v7 ikut sidik jari dan zip;
  tidak ada training tambahan. Model fitur HRV menghitung fiturnya dari IBI sintetis float64 dengan
  `hrv_core.hrv_features`, **bukan** dari `X` float32 (fitur dari `X` berbeda dari `F` tersimpan). Ditambah % window
  Normal asli yang ditebak AFIB/AFL per pita detak (`by_hr_bin`).
- Bootstrap bersarang dan sensitivitas tanpa 04936 (protokol butir 10) tersedia sebagai opsi di `gabungan.ipynb`,
  hanya bila sebuah model akan diusulkan mengganti produksi.

### 6.6 Keluaran dan eksperimen tanpa saling menimpa

- Setiap setelan punya nama (mis. `bawaan`, `hidden64`). Hasil ke `cache/v11/<model>/<nama_setelan>/`:
  - per (ulangan, fold): bobot model (`.pt` untuk torch, berkas native untuk XGBoost, `joblib` untuk scikit-learn),
    prediksi OOF fold itu, dan prediksi uji stres;
  - `ringkasan.json`: metrik, setelan lengkap, jumlah parameter, waktu, lingkungan (GPU, versi pustaka).
- Checkpoint per (ulangan, fold), jadi sesi Kaggle/Colab yang terputus bisa dilanjutkan. Mengubah setelan tanpa
  mengganti nama → error (setelan disimpan dan dicocokkan), supaya hasil lama tidak tercampur.
- Untuk dibagikan ke tim: fungsi pembantu membuat zip dari satu folder setelan.

### 6.7 Tuning (opsional)

Grid kecil didefinisikan di `model.py`. Setiap titik grid adalah setelan bernama yang dilatih penuh di 25 fold. Pemilihan
memakai cara cross-fit protokol butir 6 (`v4_smoothing_ensemble/pascaproses.crossfit_select`): untuk tiap fold uji,
dipilih setelan terbaik di fold lain. Hasil tuning hanya dilaporkan dengan cara ini.

### 6.8 Dependensi

- `xgboost` ditambahkan ke `requirements.txt` dan dipasang di venv WSL.
- `aeon` (MiniRocket) **tidak** masuk `requirements.txt` utama. aeon 1.6.0 mensyaratkan numpy < 2.5 dan scipy < 1.18,
  sedangkan venv WSL memakai numpy 2.5.3 dan scipy 1.18.1, yang dipinjam dari venv proyek lain lewat `.pth`. Memasangnya
  di venv utama akan menurunkan numpy/scipy untuk seluruh repo dan bisa merusak pemeriksaan bit-identik v5/v7/v8.
- Keputusan pemilik repo: aeon di `v11_model_tim/minirocket/requirements-minirocket.txt`. Di Kaggle/Colab dipasang
  biasa di sel pertama notebook MiniRocket. Di PC pemilik repo, uji jalan memakai **venv terpisah khusus MiniRocket**
  di luar repo. Venv ini memakai ulang paket venv utama (torch, dll.) kecuali numpy/scipy/aeon, dan venv utama tidak
  disentuh. README MiniRocket menjelaskan cara memilih kernelnya (`SIHEDAF_KERNEL`).

### 6.9 Uji jalan singkat (diizinkan pemilik repo)

- `SIHEDAF_QUICK=1`: tiap model 1 ulangan × 5 fold (bias cross-fit butuh semua fold), 1–2 epoch atau sedikit
  pohon/kernel, sebagian kecil data latih.
  Keluaran ke folder sementara di luar `cache/v11/`, lalu dihapus. Notebook dibersihkan dari output setelahnya.
- Termasuk: uji paritas `latih_dl.py`, jalur evaluasi lengkap (termasuk uji takikardia), checkpoint/lanjut, error saat
  setelan berubah tanpa ganti nama, dan jalur `SIHEDAF_DATA` dengan `v11_data.zip` yang diekstrak ke scratchpad.
- Total ±10–15 menit di WSL.

### 6.10 Dokumentasi

- `v11_model_tim/README.md` dan README per model (bahasa sederhana, untuk teman yang belum mengenal repo).
- AGENTS.md: entri Intent Layer v11, termasuk pengecualian cwd subfolder dan pengecualian invariant "arsitektur RR-only
  tidak dicoba lagi" dengan alasan: permintaan pembimbing, studi perbandingan antar model.
- Baris v11 di tabel README root ditambahkan setelah hasil ada.

## 7. Diserahkan ke pembangun

Struktur internal kode, detail arsitektur TCN/InceptionTime dalam batas 6.3–6.4, nama fungsi di luar yang disebut di
sini, dan bentuk grafik (mengikuti palet AGENTS.md).

## 8. Batasan yang sudah diketahui

- Kemungkinan besar semua model berkumpul di sekitar Macro-F1 0,83, karena batasnya 7 pasien AFL (v3–v6).
- Hasil antar mesin bisa sedikit berbeda (GPU non-deterministik). Perbandingan berpasangan tetap sah karena referensi
  diambil dari cache yang sama.
- SVM memakai aproksimasi kernel RBF, bukan SVM RBF eksak.
- MiniRocket memakai lebih sedikit kernel daripada bawaan aslinya.
