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
| `v31` referensi | AFDB, resep v3 (bobot penuh) | Dilatih ulang di lingkungan yang sama agar perbandingan berpasangan sah | _(menunggu run Kaggle)_ |
| `afdb` ablasi | AFDB, bobot lunak (= v5b) | Setara v31 | _(menunggu run Kaggle)_ |
| **`afdb_chapman` 8a (utama)** | AFDB + Chapman, bobot lunak | AFDB tidak lebih buruk dari v31; di Chapman alarm palsu sinus takikardia jauh lebih rendah dan AFL lebih baik daripada paket v3.1 | _(menunggu run Kaggle)_ |
| `chapman` 8b (eksploratif) | Chapman saja | Apakah model dari pasien lain menggeneralisasi ke AFDB? | _(menunggu run Kaggle)_ |
| `afdb_chapman_bersih` 8c (ditambahkan setelah run pertama) | AFDB + Chapman tanpa label AFL (Normal + AFIB) | Alarm palsu ST turun seperti 8a, AFL di AFDB tidak lebih buruk dari v31 | _(menunggu run Kaggle)_ |

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
| `chapman.py` | Unduhan Chapman (mirror S3 resmi, dicek SHA256) & ekstraksi, sidik jari AFDB & fold, training, evaluasi AFDB + Chapman + uji stres |

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

Laporan PDF dibuat di repo setelah hasil dibawa pulang.

## Menjalankan di WSL / Colab

```bash
python tools/run_notebook.py v8_chapman_shaoxing/chapman_kaggle.ipynb
```

Di mesin lokal, notebook memakai `../cache/dataset_v2_detector.npz` dan `folds.npz` yang sudah ada (diverifikasi dengan sidik
jari), lalu mengunduh Chapman ±5 GB ke `../data/chapman/`. Unduhan butuh persetujuan pemilik repo.
