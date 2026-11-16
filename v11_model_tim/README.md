# v11_model_tim — enam model berbeda dari v1, satu folder per anggota tim

**Tujuan.** Arahan pembimbing: setiap anggota tim (6 orang) mencoba satu model yang **berbeda dari model v1**
(LightGBM, 1D-ResNet, MicroTransformer, RandomForest/HistGB). Semua model memakai data, pembagian pasien, seed, bobot
kelas, dan evaluasi yang sama, jadi hasilnya bisa disandingkan secara adil dengan v1, v3.1, dan satu sama lain.
Rancangan: [RANCANGAN.md](RANCANGAN.md). Rencana implementasi: [RENCANA.md](RENCANA.md).

> **Status: selesai (3 November 2026).** Keenam model sudah dilatih penuh. `gabungan.ipynb` sudah dijalankan, dan laporan
> gabungan ada di [laporan/laporan_v11_model_tim.pdf](laporan/laporan_v11_model_tim.pdf). Hasil dan kesimpulan ada di "Hasil" di
> bawah. **Tidak ada model yang lebih baik dari v3.1; paket produksi tetap v3.1.**
> - Venv sementara, CPU saja: Python 3.13, torch 2.14.1+cpu, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1,
>   imbalanced-learn 0.14.2, xgboost 3.4.1, lightgbm 4.7.0.
> - 22 Oktober 2026: tes pytest **71 lolos, 2 di-skip** (MiniRocket, aeon tidak terpasang), 4 mnt 39 dtk.
> - 23 Oktober 2026: uji alur `SIHEDAF_QUICK=1` lewat `tools/run_notebook.py` COMPLETE untuk svm, xgboost, bigru, tcn, dan
>   inceptiontime (49–76 dtk per notebook). Opsi balancing juga diuji alurnya (xgboost dan svm `smote`/`under`, bigru `under`, termasuk
>   `banding_setelan` vs `bawaan`), dan bigru `smote` ditolak dengan benar. Setelah itu notebook dipulihkan tanpa output dan
>   `cache/v11_quick/` dihapus.
> - 23 Oktober 2026, MiniRocket di venv terpisah (aeon 1.6.0, numpy 2.3.5, numba 0.63.1, scipy 1.17.1, torch CPU): tes **74 lolos,
>   0 di-skip**, dan uji alur QUICK COMPLETE (51 dtk). Ditemukan dan diperbaiki: tanpa batas `numpy<2.4` di
>   `minirocket/requirements-minirocket.txt`, resolver memilih numba beta yang gagal (`np.trapz`). Detail di `minirocket/README.md`.
> - Juga 23 Oktober 2026: seluruh tes lolos di GPU (RTX 5060, torch 2.11.0+cu128; 72 lolos, 2 di-skip MiniRocket).
> - 23 Oktober 2026: `gabungan.ipynb` dijalankan dengan hasil nyata xgboost dan bigru (6 setelan), dengan semua flag opsional
>   aktif, lewat salinan sementara yang lalu dihapus. Hasilnya COMPLETE (±16 menit, terutama bootstrap bersarang). Format tampilan
>   diperbaiki: teks vonis tidak lagi terpotong, dan sel tuning, tanpa 04936, dan bersarang tampil sebagai angka + CI.
> - 2 November 2026: opsi balancing `over` (oversampling acak, juga untuk model urutan) ditambahkan; tes 76 lolos, 2 di-skip. BiGRU
>   dilatih dengan 8 setelan tambahan (bobot kelas, kapasitas, lr, epoch, `over`) lalu 6 regularisasi di atas bobot penuh, sebagian
>   di Kaggle (T4 ×2). Kandidat BiGRU: `{"weight_power": 1.0}`; tuning selesai. Hasil di `bigru/README.md` ("Hasil 2", "Hasil 3"),
>   PDF per eksperimen di `bigru/laporan/`.
> - 2 November 2026: TCN setelan `bawaan` dilatih penuh oleh pemilik repo di Google Colab (GPU T4, 107 s per fold). Hasilnya
>   lebih buruk daripada v3.1 (Δ Macro-F1 −0,080, P = 0,035) dan BiGRU bawaan (−0,070, P = 0,003), karena bias cross-fit menghapus
>   AFL 2:1 (08378). Lalu 6 setelan tuning dilatih bersamaan di Colab L4 (68 menit): bobot kelas penuh (`{"weight_power": 1.0}`)
>   memulihkan 08378 (0,92) dan setara v3.1 (P = 0,38) dan BiGRU `wp100` (P = 0,47); kapasitas, dropout, dan dilatasi tidak menolong.
>   Kandidat TCN: `wp100`. Hasil di `tcn/README.md`, PDF per eksperimen di `tcn/laporan/`.
> - 3 November 2026: svm, inceptiontime, dan minirocket setelan `bawaan` dilatih penuh oleh pemilik repo (bukan oleh anggota tim).
>   InceptionTime dan MiniRocket dijalankan di PC lewat `jalankan.ipynb`. SVM dilatih di Colab CPU High-RAM, karena RAM PC tidak cukup,
>   lalu dievaluasi di PC dengan notebook yang sama. Tidak ada yang lebih baik daripada v3.1: InceptionTime (P = 0,006) dan MiniRocket
>   (P = 0,000) lebih buruk, SVM tidak terbedakan (P = 0,07). MiniRocket kehilangan AFL 2:1 (08378) bahkan tanpa bias, dan regresi
>   logistiknya belum konvergen di 11 dari 25 fold. Sel Kesimpulan ketiga notebook sudah diisi.
> - 3 November 2026: InceptionTime `wp100` (bobot kelas penuh) dilatih di PC (`inceptiontime/eksperimen.py e2`). Hasilnya:
>   - 08378 pulih (0,99), tetapi Macro-F1 sama dengan `bawaan` (P = 0,53), karena AFL palsu ikut naik;
>   - bias AFL menempel di batas −7 pada 8 dari 25 fold;
>   - lebih buruk daripada BiGRU `wp100` (P = 0,028). Bukan kandidat.
>
>   PDF per eksperimen (e1 bawaan, e2 wp100) ada di `inceptiontime/laporan/`.
> - 3 November 2026: `gabungan.ipynb` diperluas lalu dijalankan di PC (55 menit). Isinya antar model berpasangan, balancing vs bawaan,
>   tuning cross-fit per model, kandidat tanpa 04936 + bootstrap bersarang, dan `cache/v11/gabungan.json`. Laporan gabungan dibuat
>   oleh `laporan/build_report.py`.
> - 3 November 2026: `jalankan.ipynb` bigru, tcn, dan xgboost dijalankan di PC lewat `tools/run_notebook.py`. Isinya evaluasi
>   saja: fold di cache dilewati, ±3 menit per notebook, dan angkanya identik dengan `ringkasan.json` sebelumnya. Sejak itu keenam
>   notebook model dan `gabungan.ipynb` menyimpan output run pemilik repo sebagai catatan hasil, dengan sel Kesimpulan masing-masing.
>   `tests/test_notebook.py` disesuaikan (4 November 2026): output boleh ada, dan keenam notebook harus identik kecuali baris
>   `MODEL` dan sel Kesimpulan (8 lolos).

## Pembagian model

| Folder | Model | Input | Perangkat | Anggota |
|---|---|---|---|---|
| [`bigru/`](bigru/) | BiGRU + Attention | urutan IBI | GPU | pemilik repo |
| [`tcn/`](tcn/) | Temporal Convolutional Network | urutan IBI | GPU | |
| [`inceptiontime/`](inceptiontime/) | InceptionTime (1 jaringan) | urutan IBI | GPU | |
| [`minirocket/`](minirocket/) | MiniRocket + regresi logistik | urutan IBI | CPU | |
| [`xgboost/`](xgboost/) | XGBoost | 16 fitur HRV | CPU (GPU opsional) | |
| [`svm/`](svm/) | SVM RBF (aproksimasi Nystroem) | 16 fitur HRV | CPU | |

## Kenapa data tidak di-balance

Pembimbing juga meminta dataset di-balance. Setelah dibahas, data **tidak** di-balance:
- AFL hanya 0,7% window (1.152 dari 165.978) dan berasal dari **7 pasien**; 58% dari satu pasien (04936).
  Balancing tidak menambah pasien;
- SMOTE dan sampler (v1) serta oversampling AFL (v5a) sudah dicoba, dan tidak mengalahkan bobot kelas;
- undersampling membuang ±90% data, termasuk hampir semua dari ±100 window Normal > 120 bpm, padahal alarm palsu pada
  detak cepat adalah masalah terbesar model sekarang.

Ketidakseimbangan tetap ditangani dengan **bobot kelas lunak** (akar bobot seimbang) di loss/bobot sampel, ditambah
**bias kelas cross-fit**. Data uji tetap dengan distribusi asli.

### Opsi balancing (untuk menguji arahan pembimbing)

Bawaan tetap tanpa balancing. Supaya arahan pembimbing bisa **diuji**, bukan hanya diperdebatkan, ada kunci setelan
`balancing` untuk semua model (22 Oktober 2026). Runner menerapkannya **di dalam setiap fold, hanya pada window latih**.
Window uji, set uji stres takikardia, fold, dan seed tetap sama, jadi hasilnya bisa dibandingkan berpasangan.

| `SETELAN` | Isi | Model |
|---|---|---|
| `{}` (bawaan) | tanpa balancing, bobot kelas lunak | semua |
| `{"balancing": "under"}` | undersampling acak: tiap kelas dipotong ke jumlah AFL fold latih (458–1.143 window per kelas) | semua |
| `{"balancing": "over"}` | oversampling acak: AFIB & AFL disalin acak (dengan pengembalian) sampai sama dengan jumlah Normal (±80 ribu per kelas). Salinan utuh window asli, jadi juga untuk model urutan (2 November 2026) | semua |
| `{"balancing": "smote"}` | SMOTE (imbalanced-learn, k = 5) pada 16 fitur HRV: AFIB & AFL dinaikkan ke jumlah Normal (±80 ribu per kelas). Fitur distandardisasi dulu dari data latih fold | `svm`, `xgboost` |

- SMOTE tidak tersedia untuk model urutan (BiGRU, TCN, InceptionTime, MiniRocket): menginterpolasi dua urutan IBI dengan
  panjang berbeda tidak menghasilkan ritme yang bermakna. Runner berhenti dengan pesan bila dicoba. Padanan oversampling untuk
  model urutan adalah `over`.
- Setelah balancing, bobot kelas lunak otomatis bernilai 1 (data sudah seimbang), jadi tidak ada koreksi ganda. Bias
  cross-fit tetap dipelajari seperti biasa.
- Jumlah window latih per kelas tiap fold dicatat di `info["n_latih"]` dan dicetak saat training.
- Pakai `NAMA_SETELAN` yang jelas, mis. `"smote"` atau `"under"`, dan latih juga `"bawaan"` untuk model yang sama.
  `gabungan.ipynb` (`JALANKAN_BANDING_BALANCING = True`) membandingkan setiap setelan ber-balancing dengan `bawaan`
  model itu: Δ Macro-F1/AFL + CI + vonis, dan recall AFL per pasien.

**Kenapa CSV balancing di luar repo tidak dipakai langsung.** Dataset balancing yang dibuat di luar repo
(`dataset_hrv_{over,under}sampled.csv`, Oktober 2026) diperiksa dan tidak bocor: dibagi per pasien 14/9, scaler dari
pasien latih, balancing hanya di data latih. Namun:
- sumbernya dataset v1 (`cache/v1/dataset_hrv_tabular.csv`, identik), yaitu anotasi 250 Hz, 6 fitur, dan window AFL stride
  1 s (tumpang tindih 90%);
- pembagiannya hanya satu partisi, dan 98% AFL ujinya dari 04936;
- kolom pasien sudah hilang, sehingga tidak bisa disejajarkan dengan fold v11.

Opsi `balancing` di atas mengulang caranya (SMOTE / undersampling acak, imbalanced-learn) di data dan protokol v11.

## Yang sama untuk semua model (jangan diubah)

- Data uji: window detektor 50 Hz (`cache/dataset_v2_detector.npz`), kondisi produksi.
- Pembagian pasien: `cache/oof_robust/detector_e30_r5_s2026/folds.npz`, 5 ulangan × 5 fold. Seed latih `2026 + 100·r + k`.
- Evaluasi (`bersama/evaluasi.py`):
  - Macro-F1 pooled OOF + CI 95% bootstrap pasien;
  - hasil argmax dan + bias cross-fit;
  - perbandingan berpasangan vs v3.1 dan v1;
  - recall AFL per pasien dan asal AFL palsu;
  - uji stres takikardia.
- Hiperparameter **tidak** boleh dipilih dengan melihat hasil fold uji. Untuk tuning, latih beberapa setelan lalu pakai
  seleksi cross-fit di `gabungan.ipynb`.

## Cara menjalankan

Satu-satunya yang perlu dibuka adalah `jalankan.ipynb` di folder model Anda. Satu-satunya sel yang diubah adalah sel
**Setelan**. Notebook di repo menyimpan output run setelan `bawaan` oleh pemilik repo, sebagai catatan hasil. *Run All* menimpa
output itu. Sel Kesimpulan tetap berisi angka `bawaan` sampai Anda menulis ulang dengan hasil Anda.

**Kaggle / Colab (disarankan untuk anggota tim)**
1. Unggah `v11_model_tim/<model>/jalankan.ipynb` ke Kaggle atau Colab. Aktifkan GPU untuk model DL; di Kaggle, aktifkan
   juga Internet.
2. *Run All*. Sel pertama akan:
   - meng-clone repo;
   - memasang pustaka tambahan model (`requirements-<model>.txt`);
   - mengunduh `v11_data.zip` dari GitHub Release dan memeriksa hash-nya.
3. **Jangan sampai hasil hilang.** File di sesi Colab (`/content`) dan sesi interaktif Kaggle (`/kaggle/working`) **terhapus**
   saat sesi berakhir (idle, batas waktu, atau koneksi putus). Fitur "lanjut dari fold terakhir" hanya berguna kalau
   hasilnya disimpan di tempat yang bertahan:
   - **Colab:** isi `HASIL_DI = "/content/drive/MyDrive/sihedaf_v11"` di sel Setelan. Google Drive tersambung otomatis.
     Kalau sesi putus, buka lagi notebook yang sama dan *Run All*; fold yang sudah ada di Drive dilewati.
   - **Kaggle:** jalankan lewat **Save Version → Save & Run All (Commit)**. Notebook berjalan di latar (maks. ±12 jam) dan
     isi `/kaggle/working` tersimpan sebagai output versi itu. Hasil ada di `WrapSihedaf/cache/v11/` dalam output.
   - **Kaggle, kalau berhenti di tengah** (mis. kena batas waktu):
     1. buat dataset dari output versi tadi, lalu tambahkan sebagai input;
     2. di awal sel Setelan, salin dulu: `import shutil; shutil.copytree("/kaggle/input/<dataset>/WrapSihedaf/cache/v11",
        "/kaggle/working/hasil_v11", dirs_exist_ok=True)`;
     3. isi `HASIL_DI = "/kaggle/working/hasil_v11"`, lalu Commit lagi.
4. Kaggle **wajib** Internet aktif, karena repo di-clone (akun harus terverifikasi nomor HP). Kalau tidak bisa, pakai
   Colab.
5. Kalau hanya unduhan data yang gagal (mis. release belum ada): tambahkan `v11_data.zip` sebagai dataset Kaggle (atau
   ekstrak di Drive), lalu isi `DATA_DI` di sel Setelan dengan folder hasil ekstraknya.

**PC (repo + cache sudah ada)**

```bash
pip install -r requirements.txt
python tools/run_notebook.py v11_model_tim/<model>/jalankan.ipynb
```

Untuk MiniRocket, lihat [minirocket/README.md](minirocket/README.md) (aeon di venv terpisah bila perlu).

**Bereksperimen**
- Ubah `SETELAN`, mis. `{"epochs": 40}`, **dan ganti `NAMA_SETELAN`**. Setiap setelan punya folder hasil sendiri
  (`cache/v11/<model>/<NAMA_SETELAN>/`). Kalau `SETELAN` diubah tanpa mengganti nama, notebook berhenti dengan pesan.
- Mengubah arsitektur: edit `model.py` di fork/branch Anda sendiri, lalu isi `REPO_URL`/`BRANCH` di sel pertama.
  `ringkasan.json` mencatat commit yang dipakai.
- `SIHEDAF_QUICK=1`: uji alur cepat (1 ulangan, data kecil, 1 epoch). Hasilnya disimpan terpisah di `cache/v11_quick/`
  dan **bukan** hasil penelitian.

**Berbagi hasil.** Sel terakhir notebook membuat `cache/v11/<model>/<model>_<setelan>.zip`, berisi bobot, prediksi, dan
`ringkasan.json`. Kirim ke pemilik repo; zip diekstrak di `cache/v11/`, lalu jalankan `gabungan.ipynb`.

## Pemeriksaan pertama (pemilik repo, sebelum dibagikan)

```bash
python -m pytest v11_model_tim/tests -q
```

```bash
SIHEDAF_QUICK=1 python tools/run_notebook.py v11_model_tim/bigru/jalankan.ipynb
```

Tes MiniRocket di-skip bila aeon belum terpasang. Uji alur ulangi untuk keenam model. Setelah itu, kembalikan notebook
dengan `git checkout -- v11_model_tim/*/jalankan.ipynb` dan hapus `cache/v11_quick/`. Jangan pakai `--clear-output`, karena
notebook menyimpan hasil run penuh.

**Data untuk Kaggle/Colab** (sekali, oleh pemilik repo). `cache/v11_data.zip` sudah dibuat; isinya dicocokkan dengan
[sidik_jari.json](sidik_jari.json). Unggah sebagai GitHub Release:

```bash
gh release create v11-data cache/v11_data.zip --title "Data v11" --notes "Data uji SIHEDAF v11: turunan MIT-BIH AFDB (PhysioNet, ODC-BY 1.0) + OOF pembanding v1/v3.1/v7"
```

Kalau cache berubah, buat ulang sidik jari dan zip:

```bash
python -c "import sys; sys.path.insert(0, 'v11_model_tim'); from bersama import data; data.buat_sidik_jari(); print(data.buat_zip(data.REPO / 'cache/v11_data.zip'))"
```

## OOF hasil v11 dan v3 (GitHub Release `v11-oof`)

Prediksi out-of-fold (OOF) semua run penuh tersedia di Release
[`v11-oof`](https://github.com/farhanlhsn/WrapSihedaf/releases/tag/v11-oof) (16 November 2026). Dengan zip ini, perbandingan
berpasangan, `gabungan.ipynb`, dan v11.1 bisa diulang tanpa melatih ulang, dan semua orang memakai OOF referensi yang sama.

| Berkas | Ukuran | SHA256 |
|---|---|---|
| `v11_oof.zip` | 493,5 MB (724 MB setelah diekstrak, 910 berkas) | `2077002fb837e613627facd474b7f4b3b20407698fdd539d9bf2df4e976f4a39` |
| `v11_oof_sidik_jari.json` | 130 KB, SHA256 dan ukuran setiap berkas | `7213f8d2dd6255e200c0e2292bd1ef83a0cf1d52a38a368076e39c2bbc2d6017` |

Isi zip (path relatif terhadap `cache/`):

- `dataset_v2_detector.npz`: label dan pasien window uji (sama dengan yang di `v11-data`), supaya zip bisa dipakai sendiri.
- `oof_robust/detector_e30_r5_s2026/`: seluruh OOF v3, termasuk `folds.npz`, v3.1 (`MicroTransformer_r*.npz`), varian v5 dan
  v7, BiGRUAttention, LightGBM, MaskedResNet1D, EKG_CNN, EKG_RR_DualBranch, `ringkasan.csv`, `banding_vs_referensi.csv`,
  serta OOF v1 di `versi_lama/`.
- `v11/<model>/<setelan>/`: 31 setelan (BiGRU 16, TCN 7, XGBoost 4, InceptionTime 2, MiniRocket 1, SVM 1). Setiap setelan
  berisi 25 `r{r}_k{k}.npz` (`oof`, `idx`, `stres` = prediksi uji stres takikardia, `waktu`, `info`), `setelan.json`, dan
  `ringkasan.json`.
- `v11/gabungan.json` dan `v11/<model>/banding_*.json`.

**Tidak** ikut: bobot model (`.pt`, `.joblib`, dump XGBoost `r*_k*.json`), zip hasil per anggota, dan cache versi lain
(v13–v17; sebagian berasal dari Icentia11k yang berlisensi CC BY-NC-SA). Isi zip hanya turunan MIT-BIH AFDB (ODC-BY 1.0).
Karena bobot tidak ikut, uji stres takikardia baru dan uji di data eksternal (v3.3) tetap butuh bobot dari pemilik repo.

Unduh dan ekstrak dari root repo, lalu cek setiap berkas dengan sidik jarinya:

```bash
gh release download v11-oof --repo farhanlhsn/WrapSihedaf --dir cache
```

```bash
python -c "import zipfile; zipfile.ZipFile('cache/v11_oof.zip').extractall('cache')"
```

```bash
python -c "import hashlib, json, pathlib; s = json.load(open('cache/v11_oof_sidik_jari.json'))['berkas']; bad = [r for r, v in s.items() if hashlib.sha256(pathlib.Path('cache', r).read_bytes()).hexdigest() != v['sha256']]; print('cocok semua' if not bad else bad)"
```

Tanpa `gh`: unduh kedua berkas dari halaman release dengan browser. Ekstraksi menimpa berkas dengan nama sama di `cache/`.
Kalau Anda punya hasil run sendiri yang belum dibagikan, ekstrak dulu ke folder lain.

## Berkas

| Berkas | Isi |
|---|---|
| `bersama/data.py` | Data, fold, OOF pembanding, sidik jari, zip/unduhan, set uji stres |
| `bersama/latih_dl.py` | Loop training DL (identik dengan v3, bobot kelas lunak), simpan bobot |
| `bersama/runner.py` | 25 job per setelan, bisa dilanjutkan, penjaga setelan; kontrak `model.py` ada di docstring |
| `bersama/balancing.py` | Opsi `balancing` per fold (`under`, `over`, `smote`) |
| `bersama/evaluasi.py` | Metrik + CI, perbandingan berpasangan, uji takikardia, seleksi cross-fit, protokol butir 10 |
| `<model>/model.py`, `<model>/jalankan.ipynb` | Model dan notebook (enam notebook identik kecuali baris `MODEL` dan sel Kesimpulan; output = run `bawaan` pemilik repo) |
| `gabungan.ipynb` | Tabel semua model + v1 + v3.1, antar model (bawaan, berpasangan), balancing vs bawaan, tuning cross-fit, sensitivitas tanpa 04936, bootstrap bersarang; menulis `cache/v11/gabungan.json` |
| `laporan/build_report.py`, `laporan/laporan_v11_model_tim.pdf` | Laporan gabungan v11 (dua tahap: `--extract`, lalu tanpa argumen) |
| `<model>/eksperimen.py`, `<model>/laporan/` | Eksperimen dan PDF per model (BiGRU, TCN, InceptionTime) |
| `tests/` | Tes pytest |
| `sidik_jari.json` | SHA-256 data yang dipakai (di-commit) |

**Atribusi data.** MIT-BIH Atrial Fibrillation Database (Moody & Mark, 1983), PhysioNet, lisensi ODC-BY 1.0.

## Hasil (3 November 2026, v11 selesai)

Semua angka dari `gabungan.ipynb`, yang menulis `cache/v11/gabungan.json`, dan `ringkasan.json` tiap setelan. Laporan:
[laporan/laporan_v11_model_tim.pdf](laporan/laporan_v11_model_tim.pdf), dibuat dari `laporan/build_report.py` dua tahap; angkanya
dicek otomatis terhadap OOF. `gabungan.ipynb` dijalankan di PC (CPU) lewat `tools/run_notebook.py`: 55 menit untuk 31 setelan,
terlama tuning cross-fit BiGRU (32 menit). Laporan per eksperimen: `bigru/laporan/`, `tcn/laporan/`, `inceptiontime/laporan/`.

Semua angka memakai bias cross-fit dan CI 95% bootstrap pasien. Pembanding v3.1: Macro-F1 0,832 [0,67–0,92], AFL F1 0,647
[0,17–0,89], recall 08378 0,96, alarm Normal ≥ 120 bpm 95,2%, ambang paket 115 bpm. Ambang takikardia ditulis per jenis
variasi (tanpa / seragam ±10 ms / Gauss 10 / Gauss 20); makin tinggi makin baik.

| Model (setelan) | Macro-F1 | AFL F1 | Δ Macro-F1 vs v3.1 | P vs v1 | Recall 08378 | Ambang (bpm) | Alarm Normal ≥ 120 bpm |
|---|---|---|---|---|---|---|---|
| BiGRU (`bawaan`) | 0,821 [0,68–0,87] | 0,592 [0,16–0,73] | −0,010 (P = 0,34, tidak terbedakan) | 0,99 | 0,74 | 120/120/120/120 | 86,6% |
| TCN (`bawaan`) | 0,752 [0,63–0,80] | 0,391 [0,02–0,53] | −0,080 (P = 0,035, lebih buruk) | 0,53 | 0,01 | 120/125/130/130 | 71,8% |
| InceptionTime (`bawaan`) | 0,767 [0,64–0,86] | 0,446 [0,06–0,72] | −0,065 (P = 0,006, lebih buruk) | 0,86 | 0,79 | 120/130/130/125 | 72,2% |
| MiniRocket (`bawaan`) | 0,709 [0,62–0,76] | 0,306 [0,04–0,46] | −0,122 (P = 0,000, lebih buruk) | 0,21 | 0,11 | 130/135/130/120 | 92,4% |
| XGBoost (`bawaan`) | 0,730 [0,62–0,78] | 0,363 [0,04–0,51] | −0,101 (P = 0,004, lebih buruk) | 0,36 | 0,05 | 125/120/115/115 | 85,0% |
| SVM (`bawaan`) | 0,773 [0,64–0,92] | 0,487 [0,09–0,90] | −0,059 (P = 0,072, tidak terbedakan) | 0,91 | 0,98 | 140/125/125/120 | 78,4% |
| BiGRU (`wp100`, kandidat) | 0,830 [0,69–0,93] | 0,625 [0,21–0,92] | −0,002 (P = 0,47, tidak terbedakan) | 0,99 | 1,00 | 120/120/120/115 | 90,2% |
| TCN (`wp100`, kandidat) | 0,830 [0,67–0,91] | 0,623 [0,14–0,86] | −0,002 (P = 0,38, tidak terbedakan) | 0,99 | 0,92 | 125/125/125/120 | 84,4% |
| InceptionTime (`wp100`) | 0,766 [0,64–0,92] | 0,444 [0,06–0,88] | −0,066 (P = 0,089, tidak terbedakan) | 0,81 | 0,99 | 125/130/130/120 | 76,0% |

**Antar model dengan resep yang sama** (setelan `bawaan` keenam model, berpasangan terhadap model bawaan terbaik):

| Model (`bawaan`) vs BiGRU `bawaan` | Δ Macro-F1 [CI] | P | Vonis | Δ AFL F1 (P) | Δ AFL R (P) |
|---|---|---|---|---|---|
| SVM | −0,048 [−0,08…+0,05] | 0,162 | tidak terbedakan | −0,106 (0,25) | +0,057 (0,66) |
| InceptionTime | −0,055 [−0,08…−0,00] | 0,022 | lebih buruk | −0,146 (0,03) | −0,077 (0,12) |
| TCN | −0,070 [−0,21…−0,01] | 0,003 | lebih buruk | −0,201 (0,00) | −0,204 (0,00) |
| XGBoost | −0,091 [−0,21…−0,03] | 0,000 | lebih buruk | −0,229 (0,00) | −0,202 (0,04) |
| MiniRocket | −0,112 [−0,20…−0,04] | 0,000 | lebih buruk | −0,287 (0,00) | −0,248 (0,00) |

**Balancing (arahan pembimbing)**, berpasangan terhadap setelan `bawaan` model yang sama:

| Balancing vs `bawaan` model yang sama | Δ Macro-F1 [CI] | P | Vonis | Δ AFL R (P) | Δ Normal R (P) | Recall 08378 |
|---|---|---|---|---|---|---|
| BiGRU `over` | −0,045 [−0,07…−0,02] | 0,000 | lebih buruk | −0,103 (0,00) | −0,006 (0,00) | 0,55 (bawaan 0,74) |
| BiGRU `under` | −0,223 [−0,33…−0,14] | 0,000 | lebih buruk | −0,132 (0,26) | −0,184 (0,00) | 0,03 (bawaan 0,74) |
| XGBoost `smote` | +0,013 [+0,00…+0,02] | 0,977 | lebih baik | +0,028 (0,86) | −0,005 (0,00) | 0,07 (bawaan 0,05) |
| XGBoost `under` | +0,063 [+0,00…+0,18] | 0,984 | lebih baik | +0,167 (0,85) | −0,028 (0,00) | 0,67 (bawaan 0,05) |

**Tuning cross-fit per model.** Untuk tiap fold uji, setelan dipilih dari fold lain, jadi ini perkiraan jujur untuk "model yang
di-tuning". Setelan ber-balancing tidak ikut:

| Seleksi cross-fit | Setelan | Macro-F1 [CI] | vs v3.1 |
|---|---|---|---|
| BiGRU | 14 | 0,818 [0,67–0,88] | −0,013 (P = 0,08, tidak terbedakan) |
| TCN | 7 | 0,782 [0,65–0,82] | −0,050 (P = 0,25, tidak terbedakan) |
| InceptionTime | 2 | 0,766 [0,64–0,86] | −0,066 (P = 0,01, lebih buruk) |
| XGBoost | 2 | 0,745 [0,63–0,79] | −0,087 (P = 0,00, lebih buruk) |

**Kandidat: tanpa pasien 04936 dan bootstrap bersarang** (protokol butir 10):

| Kandidat (protokol butir 10) | Tanpa 04936: Macro-F1 | v3.1 − kandidat (P v3.1 lebih baik) | Bersarang: Macro-F1 [CI] | Bersarang vs v1 |
|---|---|---|---|---|
| v3.1 | 0,859 [0,62–0,92] | – | – | – |
| BiGRU `wp100` | 0,852 [0,63–0,90] | +0,007 (0,60) | 0,830 [0,68–0,91] | +0,086, P = 0,980 (lebih baik) |
| TCN `wp100` | 0,840 [0,63–0,90] | +0,019 (0,77) | 0,830 [0,67–0,90] | +0,086, P = 0,984 (lebih baik) |


**Kesimpulan**
- **Tidak ada model yang lebih baik daripada v3.1. Paket produksi tetap v3.1.** Dengan resep yang sama, BiGRU terbaik (0,821,
  setara v3.1). TCN, InceptionTime, XGBoost, dan MiniRocket lebih buruk dari BiGRU; SVM tidak terbedakan.
- **Bobot kelas lebih menentukan daripada arsitektur.** Dengan bobot kelas penuh, BiGRU dan TCN naik ke 0,830, setara v3.1
  pada semua pasien, tanpa 04936, dan bootstrap bersarang. Pada bootstrap bersarang keduanya juga lebih baik dari v1.
  - Mekanismenya: bobot penuh memperlebar jarak antara keyakinan AFL di AFL 2:1 (08378) dan di window non-AFL. Akibatnya bias
    cross-fit tidak lagi menghapus 08378.
  - InceptionTime pengecualian. Jaraknya hilang, karena keyakinan AFL di window non-AFL ikut naik; bias AFL menempel di −7.
    Lihat halaman 6 laporan.
- **Balancing bukan jalan keluar.**
  - Pada BiGRU, balancing lebih buruk daripada bobot kelas. Undersampling hanya menyisakan ±1.000 window per kelas.
  - Pada XGBoost, SMOTE sebesar efek bobot penuh. Undersampling naik lebih banyak, tetapi tetap di bawah v3.1 dan menambah alarm
    pada detak cepat (lihat `xgboost/README.md`).
  - Bawaan v11 tetap tanpa balancing.
- **Seleksi cross-fit lebih rendah daripada setelan terbaik di tabel** (BiGRU 0,818 vs `wp100` 0,830). Setelan terbaik
  dipilih setelah melihat hasil; angka cross-fit adalah perkiraan yang jujur.
- **Takikardia belum terpecahkan.** Alarm pada Normal ≥ 120 bpm 72–92% di semua model, dan ambang sintetis 115–140 bpm. Sama
  dengan pelajaran v7–v10: interval RR saja tidak cukup; yang dibutuhkan rekaman sinus takikardia asli dari perangkat SIHEDAF.
- **Per pasien.** AFL 2:1 (08378) membedakan model. AFL lambat atau tidak teratur (07910, 08215) tidak terdeteksi oleh v3.1
  maupun model v11 mana pun.
- **Yang tidak bisa disimpulkan.**
  - Model dilatih di mesin berbeda (PC RTX 5060 / CPU, Colab T4 / L4 / CPU High-RAM, Kaggle T4). Perbandingannya tetap sah karena
    OOF, fold, dan sampel bootstrap sama, tetapi selisih kecil bisa berasal dari variasi training.
  - MiniRocket belum konvergen di 11 dari 25 fold.
  - SVM, InceptionTime, dan MiniRocket hanya punya 1–2 setelan.
  - Rincian ada di bagian "Lingkungan" tiap README.
