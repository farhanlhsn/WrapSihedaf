# v8_chapman_shaoxing — data eksternal Chapman–Shaoxing

Pertanyaan:
1. Seberapa sering paket produksi v3.1 membunyikan alarm AFIB/AFL pada **sinus takikardia asli**? v3.2 dan v7 hanya bisa
   mengukurnya pada data sintetis.
2. Apakah menambah ribuan pasien dari luar AFDB ke data latih memperbaiki masalah takikardia **dan** deteksi AFL? v3–v7 tidak
   berhasil dengan 23 pasien AFDB saja.

**Data:** PhysioNet [`ecg-arrhythmia` 1.0.0](https://physionet.org/content/ecg-arrhythmia/1.0.0/) (Chapman University, Shaoxing
People's Hospital, Ningbo First Hospital; CC BY 4.0).
- 45.152 pasien; setiap pasien punya satu EKG 12-lead 10 detik, 500 Hz, dengan label SNOMED CT.
- Hanya lead II yang dipakai, diproses dengan front-end produksi (`hrv_core.signal_to_windows`), jadi satu window 10 s per pasien.
- Pemetaan kelas:
  - Normal = SR, SB, ST, SA (irama sinus);
  - AFIB = AFIB;
  - AFL = AF (atrial flutter).
- SVT, AT, AVNRT, AVRT, WAVN, dan pasien dengan lebih dari satu kode ritme tidak dipakai untuk training. Prediksi untuk mereka
  tetap dilaporkan.

| Varian | Data latih | Hipotesis (ditetapkan sebelum hasil) | Status |
|---|---|---|---|
| `v31` referensi | AFDB, resep v3 (bobot penuh) | Dilatih ulang di lingkungan yang sama agar perbandingan berpasangan sah | Macro-F1 0,830 (= v3.1) |
| `afdb` ablasi | AFDB, bobot lunak (= v5b) | Setara v31 | Terbukti (P = 0,68) |
| **`afdb_chapman` 8a (utama)** | AFDB + Chapman, bobot lunak | AFDB tidak lebih buruk dari v31; di Chapman alarm palsu sinus takikardia jauh lebih rendah dan AFL lebih baik daripada paket v3.1 | **Ditolak**: label AFL Ningbo merusak AFL di AFDB |
| `chapman` 8b (eksploratif) | Chapman saja | Apakah model dari pasien lain menggeneralisasi ke AFDB? | Tidak (Macro-F1 0,718) |
| `afdb_chapman_bersih` 8c (ditambahkan setelah run pertama) | AFDB + Chapman tanpa label AFL (Normal + AFIB) | Alarm palsu ST turun seperti 8a, AFL di AFDB tidak lebih buruk dari v31 | **Sebagian**: alarm ST 42% → 9%, tetapi AFL 2:1 dan SVT hilang |

**Label AFL bagian Ningbo tidak andal** (ditemukan di run pertama, `chapman.afl_label_check`):
- Record JS00001–JS10646 (Chapman–Shaoxing): 1.696 AFIB dan 416 AFL. AFL-nya berdetak median 113 bpm.
- Record berikutnya (Ningbo): **0 AFIB**, tetapi 7.192 "AFL" berdetak median 88 bpm dan 86% tidak teratur (CV IBI > 0,10).
  Itu persis AFIB (AFDB: AFIB 90 bpm, 97% tidak teratur; AFL 139 bpm, 18% tidak teratur). Kemungkinan besar "AF" (atrial
  fibrillation) Ningbo dipetakan ke kode SNOMED atrial flutter.
- Akibatnya 8a dilatih dengan ±7.000 AFIB berlabel AFL. Varian 8c ditambahkan dengan hanya label yang tidak diragukan, dan
  metrik 3 kelas di Chapman memakai AFL Shaoxing saja. Hasil 8a tetap dilaporkan apa adanya.

**Kriteria kandidat produksi untuk 8a** (ditetapkan sebelum hasil; berlaku juga untuk 8c):
1. AFDB: Macro-F1 vs v31 tidak lebih buruk (P > 0,05), dan tidak ada pasien AFL yang recall-nya turun ≥ 0,20.
2. Chapman: alarm palsu pada sinus takikardia paling banyak separuh paket v3.1, dan recall AFL tidak lebih rendah dari paket v3.1.
3. Uji stres: ambang Normal ≥ 135 bpm.

| File | Isi |
|---|---|
| `chapman_kaggle.ipynb` | Notebook untuk Kaggle (juga jalan di WSL/Colab): clone repo, unduh AFDB + Chapman, training, evaluasi, zip hasil |
| `laporan/` | `laporan_chapman_shaoxing.pdf` + `build_report.py` |
| `chapman.py` | Unduhan Chapman (mirror S3 resmi, dicek SHA256) & ekstraksi, sidik jari AFDB & fold, training, evaluasi AFDB + Chapman + uji stres |

## Hasil

**Sinus takikardia asli (6.937 pasien Chapman)**, % ditebak AFIB/AFL:

| Detak | Pasien | paket v1 | paket v3.1 | 8a | **8c** |
|---|---|---|---|---|---|
| 100–110 bpm | 3.415 | 8% | 6% | 4% | 4% |
| 110–120 bpm | 1.639 | 82% | 55% | 7% | 6% |
| 120–130 bpm | 783 | 98% | **98%** | 12% | **9%** |
| 130–140 bpm | 484 | 100% | **100%** | 25% | 20% |
| 140–150 bpm | 199 | 100% | 100% | 58% | 59% |
| ≥ 150 bpm | 277 | 100% | 100% | 41% | 27% |
| **semua ST** | 6.937 | 49% | **42%** | 11% | **9%** |

**AFDB** (protokol v3; window detektor 50 Hz, 5×5 fold, bias cross-fit, Δ berpasangan vs `v31`):

| | Macro-F1 | AFL recall | 08378 | AFL palsu dari pasien tanpa AFL | Vonis vs v31 |
|---|---|---|---|---|---|
| v31 (resep v3.1) | 0,830 [0,67–0,92] | 0,600 | 0,98 | 33 | referensi |
| afdb (= v5b) | 0,835 | 0,607 | 0,88 | 18 | tidak terbedakan |
| 8a AFDB + Chapman | 0,715 | 0,320 | 0,00 | 829 | lebih buruk (P = 0,001) |
| 8b Chapman saja | 0,718 | 0,351 | 0,04 | 183 | lebih buruk (P = 0,000) |
| **8c tanpa label AFL** | 0,799 [0,66–0,83] | 0,479 | **0,51** | 94 | tidak terbedakan (P = 0,23); vs afdb lebih buruk (P = 0,02) |

**Chapman, label andal** (32.025 Normal, 1.696 AFIB, 416 AFL Shaoxing):

| | Macro-F1 | Normal R | AFIB R | AFL R (Shaoxing) | SVT → Normal |
|---|---|---|---|---|---|
| paket v3.1 | 0,520 | 0,865 | 0,953 | 0,243 | 1% (91% → AFL) |
| **8c** | 0,583 | 0,956 | 0,942 | **0,078** | **74%** |

Uji stres sintetis (ambang Normal): paket v3.1 115 bpm; 8a 140/140/140/120; **8c 135/135/135/120**.

**Kesimpulan.** Paket produksi tetap v3.1; v8 tidak dikemas.
- Data nyata membuktikan masalah takikardia: paket v3.1 membunyikan alarm pada 98–100% sinus takikardia ≥ 120 bpm.
- Menambah sinus takikardia asli ke data latih (8c) menurunkan alarm itu dari 42% ke 9%, tetapi model ikut menyebut AFL 2:1 dan
  SVT sebagai Normal. 08378 turun 0,98 → 0,51 dan AFL cepat di Chapman hampir tak terdeteksi.
- Ini batas interval RR: ritme cepat yang teratur tidak bisa dipisahkan tanpa bentuk gelombang. Langkah berikutnya (v9) adalah
  cabang bentuk gelombang EKG yang dilatih dengan 42 ribu pasien.
- Label AFL bagian Ningbo tidak boleh dipakai (lihat atas).

## Lingkungan hasil

Dijalankan 17 Oktober 2026 di **Google Colab** (1× Tesla T4, 2 CPU, Python 3.13, torch 2.11):
- unduh dan ekstraksi data ±7 menit;
- training 125 fold ±2 jam (59–76 s/fold);
- evaluasi 43 menit.

Run Kaggle sebelumnya (2× T4, tanpa 8c) memberi angka yang identik untuk empat varian yang sama. Notebook di folder ini adalah
hasil run Colab. Hasilnya diekstrak ke `../cache/v8/`, sedangkan hasil run pertama disimpan di `../cache/v8_run1/`.

## Menjalankan di Kaggle (disarankan; PC tidak dipakai)

1. Pastikan folder ini sudah di-push ke GitHub. Notebook meng-clone repo publik `farhanlhsn/WrapSihedaf`, jadi yang dijalankan
   adalah commit terakhir di `main`.
2. Kaggle → **Create → New Notebook → File → Import Notebook**, lalu unggah `chapman_kaggle.ipynb`.
3. **Settings**:
   - Accelerator **GPU T4 x2** (P100 tidak didukung PyTorch versi baru);
   - **Internet: On** (butuh akun Kaggle yang terverifikasi nomor HP).
4. **Save Version → Save & Run All (Commit)**. Notebook berjalan di server Kaggle sekitar 3–4 jam:
   - ±5–10 menit unduh (AFDB + Chapman dari mirror S3 resmi PhysioNet);
   - ±15 menit ekstraksi;
   - ±2,5–3 jam training 100 model fold;
   - ±15 menit evaluasi.
5. Setelah selesai, buka versi itu → tab **Output** → unduh `v8_hasil.zip` dan notebook beserta output-nya.
   - `v8_hasil.zip` diekstrak di root repo; isinya mengisi `cache/v8/` dan `folds.npz`.
   - Notebook disalin ke folder ini, menimpa `chapman_kaggle.ipynb`.

Isi `v8_hasil.zip` (±80 MB):
- `cache/v8/ringkasan.json`;
- OOF AFDB + Chapman per varian/ulangan;
- `chapman_windows.npz` (IBI lead II semua pasien);
- bobot 100 model fold.

Laporan PDF dibuat di repo setelah hasil dibawa pulang (dari root repo):

```bash
python v8_chapman_shaoxing/laporan/build_report.py --extract   # evaluasi ulang dari cache/v8 + cocokkan dengan ringkasan notebook (±20 menit CPU)
python v8_chapman_shaoxing/laporan/build_report.py             # rakit PDF
```

`--charts` menggambar ulang grafik dari `laporan/_build/tables.json` yang sudah diverifikasi, tanpa evaluasi ulang.

## Menjalankan di WSL / Colab

```bash
python tools/run_notebook.py v8_chapman_shaoxing/chapman_kaggle.ipynb
```

Di mesin lokal, notebook memakai `../cache/dataset_v2_detector.npz` dan `folds.npz` yang sudah ada (diverifikasi dengan sidik
jari), lalu mengunduh Chapman ±5 GB ke `../data/chapman/`. Unduhan butuh persetujuan pemilik repo.
