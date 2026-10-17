# SIHEDAF — klasifikasi ritme jantung dari interval RR

Klasifikasi window EKG 10 detik menjadi **Normal / AFIB / AFL** dari urutan interval RR (IBI),
untuk perangkat EKG single-lead 50 Hz. Data latih: MIT-BIH Atrial Fibrillation Database (AFDB,
23 pasien). Repo ini menyimpan setiap iterasi eksperimen sebagai folder versi tersendiri
beserta laporannya.

> Untuk AI agent / anggota tim yang ingin menghasilkan output serupa: baca **[AGENTS.md](AGENTS.md)**.

## Versi dan hasil

| Versi | Isi | Hasil utama (data uji detektor 50 Hz)* | Laporan |
|---|---|---|---|
| [v1_lama](v1_lama/) | Pipeline awal: `ekstraksi_hrv.py`, `model.ipynb`, `final.joblib`, backend `find_peaks` | Diklaim Macro-F1 0,826. Diukur ulang: **0,744**; dengan front-end aslinya **0,418** | dibahas di laporan v3 (hal. 2–3) |
| [v2_revisi](v2_revisi/) | Modul bersama `hrv_core.py`, detektor R-peak 50 Hz + cek kualitas, 3 arsitektur DL, bias kelas, `predictor.py` + `model_export/` | MicroTransformer, bias `[0.0, 2.0, -3.0]`: **0,779** [0,63–0,82] (notebook melaporkan 0,792, satu partisi) | [PDF](v2_revisi/laporan/laporan_hasil_sihedaf_v2.pdf) |
| [v3_evaluasi_robust](v3_evaluasi_robust/) | Protokol evaluasi: 5×5 fold pasien, CI bootstrap pasien, perbandingan berpasangan; v1 dilatih ulang untuk dibandingkan | Kandidat model sama kuat; batasnya data AFL; bias v2 terpotong grid | [PDF](v3_evaluasi_robust/laporan/laporan_evaluasi_robust.pdf) |
| [v3.1_kalibrasi_produksi](v3.1_kalibrasi_produksi/) | Menerapkan keputusan v3 tanpa training: bias baru, **paket produksi aktif**, panduan backend | Bias `[0.0, -0.5, -4.5]`: **0,832** [0,67–0,92], AFL F1 0,647; lebih baik dari v2 produksi (P = 1,00; bersarang 0,993, lihat v3.2) | [PDF](v3.1_kalibrasi_produksi/laporan/laporan_kalibrasi_produksi.pdf) |
| [v3.2_uji_takikardia](v3.2_uji_takikardia/) | Evaluasi saja (menindaklanjuti review eksternal): uji stres ritme teratur 60–170 bpm, bootstrap bersarang, tanpa 04936, simulasi aturan produk | **Semua versi menebak ritme teratur ≥ ±115 bpm sebagai AFIB/AFL** (95% Normal ≥120 bpm jadi alarm). v3.1 vs v2 tetap lebih baik (P = 0,993 bersarang); vs v1 tidak terbedakan tanpa 04936 (P = 0,94) | [PDF](v3.2_uji_takikardia/laporan/laporan_uji_takikardia.pdf) |
| [v4_smoothing_ensemble](v4_smoothing_ensemble/) | Smoothing antar-window & ensemble di atas OOF v3, tanpa training | Tidak ada yang lebih baik dari v3.1. Smoothing: alarm palsu 0,51 → 0,09/jam tapi episode AFL terdeteksi 74% → 38% | [PDF](v4_smoothing_ensemble/laporan/laporan_smoothing_ensemble.pdf) |
| [v5_training_afl](v5_training_afl/) | Dua resep training MicroTransformer: window AFL rapat, bobot kelas lunak | Keduanya setara dengan v3.1. Bobot lunak: bias AFL −2,25 (bukan −4,5), tanpa bias 0,793 vs 0,717 | [PDF](v5_training_afl/laporan/laporan_training_afl.pdf) |
| [v6_cabang_ekg](v6_cabang_ekg/) | Cabang sinyal EKG mentah 50 Hz: CNN EKG saja, dua cabang EKG + RR, late fusion | EKG saja 0,651 (lebih buruk); dua cabang 0,747 (cenderung buruk); late fusion 0,838 (setara). 07910 & 08215 tetap 0 | [PDF](v6_cabang_ekg/laporan/laporan_cabang_ekg.pdf) |

| [v7_augmentasi_takikardia](v7_augmentasi_takikardia/) | MicroTransformer + sinus takikardia sintetis (detak Normal asli pasien latih yang dipercepat): 100–130 bpm (7a), 100–160 bpm (7b) | 7a: ambang alarm 115 → **135 bpm**, Macro-F1 0,825 (setara), tetapi recall AFL 0,592 → 0,555 (08378: 0,96 → 0,71). 7b rusak (0,700). Belum dikemas | [PDF](v7_augmentasi_takikardia/laporan/laporan_augmentasi_takikardia.pdf) |

| [v8_chapman_shaoxing](v8_chapman_shaoxing/) | Data eksternal PhysioNet `ecg-arrhythmia` (Chapman–Shaoxing & Ningbo, 42 ribu pasien, lead II); dijalankan di Colab/Kaggle | **Sinus takikardia asli: paket v3.1 alarm 42% (98–100% di ≥120 bpm).** 8c (+ Chapman tanpa label AFL): alarm ST 9%, AFDB 0,799 (setara), tetapi AFL 2:1 & SVT hilang (08378: 0,98 → 0,51). Label AFL Ningbo tidak andal. Belum dikemas | [PDF](v8_chapman_shaoxing/laporan/laporan_chapman_shaoxing.pdf) |

\*Rata-rata out-of-fold, pasien uji tidak pernah ikut dilatih. Ini bukan uji pada perangkat asli.

## Keputusan proyek yang berlaku

- **Paket produksi aktif: `v3.1_kalibrasi_produksi/model_export/sihedaf_v3_1.{pt,json}`.** Paket v2 dipertahankan sebagai
  titik rollback (bobot sama, hanya bias berbeda).
- Sistem lama (`final.joblib` + `find_peaks`) diganti. Panduan migrasi: [BACKEND.md](v3.1_kalibrasi_produksi/BACKEND.md).
- Model produksi tetap MicroTransformer.
- **Pencarian arsitektur yang inputnya hanya interval RR dihentikan.** Empat arsitektur gagal di pasien AFL yang sama.
  Arsitektur baru hanya layak dicoba jika menambah informasi input (mis. sinyal EKG mentah).
- Semua eksperimen berikutnya memakai protokol v3 dan dibandingkan berpasangan dengan referensi **v3.1**.
- Ensemble MT + BiGRU dan smoothing antar-window **tidak** dipakai sebagai default (v4). Smoothing offline tersedia sebagai
  opsi produk "hemat alarm" jika pemilik produk memilih menukar deteksi AFL dengan lebih sedikit alarm palsu.
- Resep training (v5): window AFL rapat **tidak** dipakai; **bobot kelas lunak (akar) jadi resep bawaan untuk model baru**
  (setara, koreksi bias setengahnya). Model produksi tidak diganti.
- Semua tuas yang inputnya hanya interval RR sudah dicoba (v3–v5) dan berhenti di Macro-F1 ±0,83.
- Input bentuk gelombang EKG 50 Hz (v6) juga tidak memperbaiki: CNN overfit per pasien; late fusion setara (kandidat, perlu
  konfirmasi). **Batas proyek sekarang adalah jumlah pasien AFL (7 di AFDB).** Eksperimen berikutnya (v7) butuh data dengan
  lebih banyak pasien AFL (mis. Chapman–Shaoxing di PhysioNet) — unduhan perlu persetujuan pemilik repo.
- **Batasan produksi yang diketahui (v3.2): detak cepat yang teratur (sinus takikardia, mulai ±115 bpm) dilaporkan sebagai
  AFIB/AFL.** Penyebabnya, AFDB hampir tidak punya ritme normal yang cepat. Peringatan untuk backend ada di BACKEND.md.
  Aturan produk "takikardia teratur, perlu dicek" menunggu keputusan pemilik produk: aturan ini juga mengalihkan 86% AFL yang
  terdeteksi. Data v7 harus memuat sinus takikardia, tidak hanya pasien AFL tambahan.
- Uji stres takikardia wajib untuk setiap model kandidat (AGENTS.md, protokol butir 9).
- **Data eksternal (v8) mengonfirmasi batas interval RR.** Dengan sinus takikardia asli sebagai data latih, alarm palsu ST turun
  42% → 9%, tetapi AFL 2:1 dan SVT ikut disebut Normal. Langkah berikutnya (v9) adalah cabang bentuk gelombang EKG yang
  dilatih dengan AFDB + Chapman (label andal). Label AFL bagian Ningbo di `ecg-arrhythmia` tidak boleh dipakai.
- Augmentasi sinus takikardia sintetis (v7) hanya memindahkan batas alarm palsu ke ±135 bpm, dengan harga sebagian deteksi
  AFL 2:1. Varian 100–130 bpm adalah kandidat yang **menunggu keputusan pemilik produk**. Di 135–160 bpm, interval RR saja
  tidak bisa membedakan sinus takikardia dari AFL 2:1; ini butuh data eksternal dengan bentuk gelombang EKG.

## Struktur folder

```
README.md                 dokumentasi untuk manusia
AGENTS.md, CLAUDE.md      aturan untuk agent & tim (CLAUDE.md = penunjuk ke AGENTS.md untuk Claude Code)
requirements.txt          dependensi lingkungan tim
tools/                    run_notebook.py (runner), download_afdb.py (unduh AFDB), report_fonts.py (font laporan)
data/afdb/                AFDB mentah            ← tidak di-commit, unduh dulu
cache/                    hasil ekstraksi & OOF  ← tidak di-commit, dibuat ulang oleh notebook
logs/                     log runner             ← tidak di-commit
v1_lama/                  versi sebelum revisi (arsip, beku)
v2_revisi/                revisi pipeline + model produksi + laporan/
v3_evaluasi_robust/       evaluasi robust + perbandingan dengan v1 + laporan/
v3.1_kalibrasi_produksi/  bias baru + paket produksi aktif + BACKEND.md + laporan/
v3.2_uji_takikardia/      uji stres takikardia, bootstrap bersarang, sensitivitas 04936 (evaluasi saja) + laporan/
v4_smoothing_ensemble/    pasca-proses (smoothing, ensemble) tanpa training + laporan/
v5_training_afl/          dua resep training untuk AFL (window rapat, bobot lunak) + laporan/
v6_cabang_ekg/            cabang sinyal EKG mentah 50 Hz (CNN, dua cabang, late fusion) + laporan/
v7_augmentasi_takikardia/ sinus takikardia sintetis sebagai data latih (100–130, 100–160 bpm) + laporan/
v8_chapman_shaoxing/      data eksternal Chapman–Shaoxing (notebook Kaggle/Colab-first) + laporan/
```

Setiap folder versi berisi `README.md` (apa, hasil, cara menjalankan) dan `laporan/`
(PDF + `build_report.py` pembuatnya). Notebook dijalankan dengan **folder versinya sebagai
direktori kerja**; data dan cache diakses lewat `../data` dan `../cache`.

## Setup

Disarankan **WSL2 / Linux dengan GPU NVIDIA**. Kaggle dan Google Colab (GPU) juga bisa; langkah khususnya
ada di [AGENTS.md](AGENTS.md) bagian "Lingkungan". CPU saja bisa, tapi lambat.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name sihedaf-v2     # opsional; runner memakai python3 jika tidak ada
```

Unduh AFDB (±600 MB) ke `data/afdb/`. Skrip ini bisa dilanjutkan jika koneksi terputus:

```bash
python tools/download_afdb.py
```

Hasil yang sudah ada di `cache/` (±120 MB, tidak ada di git) bisa disalin dari anggota tim lain supaya tidak
perlu training ulang, dan supaya perbandingan memakai OOF referensi yang sama.

## Menjalankan

Semua perintah dari root repo dengan venv di atas. Waktu di tabel diukur pada PC desktop dengan GPU NVIDIA 8 GB; di mesin lain
bisa berbeda.

| Tujuan | Perintah | Waktu |
|---|---|---|
| Dataset v1 (lama) | `cd v1_lama && python ekstraksi_hrv.py` | ±20 s |
| Pipeline v2 lengkap (ekstraksi, benchmark, **ekspor model**) | `python tools/run_notebook.py v2_revisi/revisi_sihedaf_v2.ipynb` | ±1 jam |
| Evaluasi robust v3 | `python tools/run_notebook.py v3_evaluasi_robust/evaluasi_robust.ipynb` | ±1 jam 40 mnt (pertama kali), ±5 mnt jika OOF sudah ada |
| Ukur ulang v1 dengan protokol v3 | `cd v3_evaluasi_robust && python eval_versi_lama.py` | ±25 mnt |
| Kalibrasi & paket produksi v3.1 | `python tools/run_notebook.py v3.1_kalibrasi_produksi/kalibrasi_produksi.ipynb` | ±2 mnt |
| Uji stres takikardia v3.2 | `python tools/run_notebook.py v3.2_uji_takikardia/uji_takikardia.ipynb` | ±10 mnt (CPU) pertama kali, lalu dari cache |
| Smoothing & ensemble v4 | `python tools/run_notebook.py v4_smoothing_ensemble/smoothing_ensemble.ipynb` | ±14 mnt (CPU) |
| Resep training v5 | `python tools/run_notebook.py v5_training_afl/training_afl.ipynb` | ±60 mnt (GPU) |
| Cabang EKG v6 | `python tools/run_notebook.py v6_cabang_ekg/cabang_ekg.ipynb` | ±2 jam (GPU) |
| Data eksternal v8 (Kaggle/Colab, GPU) | lihat `v8_chapman_shaoxing/README.md` | ±1,5 jam (Kaggle 2× T4) / ±3 jam (Colab 1× T4) |
| Augmentasi takikardia v7 | `python tools/run_notebook.py v7_augmentasi_takikardia/augmentasi_takikardia.ipynb` | ±45 mnt (GPU) + ±12 mnt evaluasi |
| Laporan versi N | `python vN_.../laporan/build_report.py --extract`, lalu `python vN_.../laporan/build_report.py` | ±3 mnt |

Urutan dari nol: unduh AFDB → v2 (membuat `cache/dataset_v2_*.npz`) → v3 (+ `eval_versi_lama.py`) → v3.1 → laporan.
Laporan butuh ReportLab dan font Arial atau Liberation Sans (`apt-get install -y fonts-liberation`);
lihat `tools/report_fonts.py`.

## Memakai model di backend

```python
import sys; sys.path.insert(0, "v2_revisi")
from predictor import ECGPredictor
P = ECGPredictor("v3.1_kalibrasi_produksi/model_export", name="sihedaf_v3_1")
hasil = P.predict_ecg(sinyal_mentah, fs=50)   # atau P.predict_ibi(ibi_ms)  — IBI dalam MILIDETIK
```

Backend menyalin `v2_revisi/{hrv_core.py, sihedaf_models.py, predictor.py}` dan
`v3.1_kalibrasi_produksi/model_export/`. Panduan lengkap, format keluaran, checklist migrasi, dan rollback ada di
[BACKEND.md](v3.1_kalibrasi_produksi/BACKEND.md). `final.joblib` dan `predictor.py` di `v1_lama/` sudah usang;
front-end lamanya membuat recall Normal hanya ±34%.

## Menambah eksperimen baru

Eksperimen baru → `vN_<nama_singkat>/` (berikutnya v9). Tindak lanjut tanpa eksperimen/training baru (kalibrasi, pengemasan,
perbaikan) → versi minor `vN.M_<nama>/`, seperti v3.1. Ikuti checklist "Definition of Done" di [AGENTS.md](AGENTS.md):
notebook ter-eksekusi, kesimpulan, `README.md` versi, laporan PDF di `v4_.../laporan/`, dan baris baru
di tabel "Versi dan hasil" di atas.
