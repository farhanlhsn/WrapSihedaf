# v16 — hierarki: ECGFounder memutuskan sinus vs AF, RR atau fitur atrium memutuskan AFIB vs AFL

> **Status: selesai (6 November 2026). Bukan kandidat sistem 3 kelas.** Tahap 1 ECGFounder menghapus alarm palsu sinus takikardia
> (98,4% → 0,1%) dan sangat spesifik untuk AFIB, tetapi menyebut 85% window AFL sebagai sinus. Akibatnya Macro-F1 3 kelas AFDB lebih
> buruk dari v3.1, sedangkan untuk 2 kelas setara (0,951 vs 0,926, P = 0,92). Laporan:
> [laporan/laporan_v16_hierarki_ecgfounder.pdf](laporan/laporan_v16_hierarki_ecgfounder.pdf).

## Tujuan

ECGFounder (v13) mengenali sinus takikardia tetapi mencampur AFIB/AFL. Model RR (v3.1) memisahkan AFIB dari AFL tetapi menyebut
sinus takikardia sebagai AF. v16 menguji pembagian tugas: **tahap 1** ECGFounder zero-shot memutuskan sinus vs AF, **tahap 2** RR v3.1
atau fitur atrium v15 memutuskan AFIB vs AFL pada window AF. Ini kombinasi model yang sudah ada; tidak ada jaringan yang dilatih.
Rancangan dikunci sebelum metrik dihitung: [RANCANGAN.md](RANCANGAN.md) (sha256 di `logs/v16_praregistrasi.log`).

| Sistem | Tahap 1 (sinus vs AF) | Tahap 2 (AFIB vs AFL) |
|---|---|---|
| **H-0 (primer)** | ECGFounder, **tanpa penyetelan** (AF bila log((pAF + pAFL)/pN) > 0) | RR v3.1, `b2` cross-fit |
| **H-A** | sama dengan H-0 | regresi logistik `[log p v3.1, kuantisasi, p_konsisten]` (v15), `b2` cross-fit |
| **H-X** | skor `log [pN, pAF·qAFIB, pAF·qAFL]`, bias 3 kelas cross-fit (tahap 1 ikut disetel di AFDB) | RR v3.1 |

Tahap 1 sengaja tidak disetel di AFDB, karena AFDB hampir tidak punya sinus takikardia sehingga ambangnya tidak melihat biaya alarm
palsu. H-X mengukur apa yang terjadi bila ambang itu tetap disetel.

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Sistem, hipotesis H1–H6, kriteria keputusan (dikunci) |
| `hierarki.py` | Tahap 1/2, cross-fit `b2`, tahap 2 atrium per fold, bootstrap bersarang berpasangan, tanpa 04936, 2 kelas, episode, H5/H6, pemuat data eksternal dari cache |
| `hierarki.ipynb` | Notebook utama; menulis `../cache/v16/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v16_hierarki_ecgfounder.pdf` | Laporan dua tahap |

Kode lama dipakai lewat impor: `v13_ecgfounder/ecgfounder.py` (skor zero-shot, data eksternal, bootstrap pasien),
`v15_kuantisasi_rr/kuantisasi.py` (masukan dan resep regresi logistik atrium, SHDB-AF), `v3.3_takikardia_asli/takikardia_asli.py`
(alarm per pita, metrik episode), `v11.1_dua_kelas/dua_kelas.py` (2 kelas), `v11_model_tim/bersama` (protokol).

## Cara menjalankan

```bash
python tools/run_notebook.py v16_hierarki_ecgfounder/hierarki.ipynb
python v16_hierarki_ecgfounder/laporan/build_report.py --extract && python v16_hierarki_ecgfounder/laporan/build_report.py
```

Tidak butuh GPU. Butuh cache yang sudah ada: v13 (`cache/v13/afdb_50hz/`, `latihan_ok_50hz.npz`, `ptbxl_50hz.npz`,
`icentia_50hz.npz`, `ringkasan.json`), v15 (`ptbxl.npz`, `oof_gabungan_afdb.npz`, `ringkasan.json`), v12 (`fitur_r1.npz`,
`shdb/`), v3.3 (`jendela.npz`, `prediksi.npz`), v10 (`icentia_windows.npz`), dan `cache/oof_robust/`. Notebook meng-assert bahwa angka
v3.1, v13, dan v15 yang dihitung ulang sama dengan ringkasan versinya, dan bahwa urutan window latihan fisik dan PTB-XL sejajar
(alarm dan Macro-F1 zero-shot v13 terulang persis).

## Hasil

**AFDB 3 kelas** (pooled OOF 5 ulangan, CI 95% bootstrap pasien; vonis berpasangan vs v3.1):

| Sistem | Macro-F1 [CI] | AFL F1 | AFL R | Normal R | vs v3.1 (P) |
|---|---|---|---|---|---|
| v1 (argmax) | 0,744 [0,64–0,81] | 0,399 | 0,728 | 0,903 | 0,015, lebih buruk |
| v3.1 (referensi) | 0,832 [0,67–0,92] | 0,647 | 0,592 | 0,924 | – |
| v13 ECGFounder zero-shot | 0,663 [0,64–0,68] | 0,087 | 0,256 | 0,973 | 0,026, lebih buruk |
| v15 gabungan | 0,756 [0,66–0,86] | 0,413 | 0,346 | 0,927 | 0,037, lebih buruk |
| **H-0 (primer)** | 0,650 [0,62–0,67] | 0,044 | 0,034 | **0,995** | 0,020, **lebih buruk** |
| H-A | 0,649 [0,63–0,68] | 0,043 | 0,037 | 0,995 | 0,026, lebih buruk |
| H-X | 0,709 [0,63–0,75] | 0,250 | 0,364 | 0,925 | 0,015, lebih buruk |

Tahap 1 di AFDB: AFIB terdeteksi 90,6%, AFL 15,3%, Normal benar 99,5%. Recall AFL per pasien H-0: 04043 0,00 · 04908 0,00 · 04936
0,05 · 06426 0,10 · 07910 0,00 · 08215 0,00 · 08378 0,00. Bootstrap bersarang H-0 vs v3.1: Δ −0,182, P = 0,0195. Tanpa 04936: H-0
0,637 vs v3.1 0,859. H-X: 82% nilai bias AFIB/AFL menempel di batas grid (median +3; tanda bahaya: penyetelan di AFDB mendorong AF setinggi mungkin
untuk mengejar AFL).

**AFDB 2 kelas (Normal vs AF/AFL; v3.1 = bias 3 kelas lalu digabung, aturan v11.1):**

| | Macro-F1 [CI] | Sensitivitas | Spesifisitas | PPV |
|---|---|---|---|---|
| v3.1 | 0,926 [0,88–0,95] | 0,934 | 0,924 | 0,892 |
| H-0 (tahap 1 saja) | 0,951 [0,92–0,98] | 0,893 | 0,995 | 0,992 |
| H-0 − v3.1 | +0,026, P = 0,92 (tidak terbedakan; bersarang P = 0,92) | −0,040, P = 0,017 | +0,071, P = 1,00 | |

Metrik episode (gaya EC57): episode AFIB palsu 11,9 → 1,2 per jam, +P episode AFIB 0,56 → 0,94; episode AFL Se 0,69 → 0,23.

**Sinus takikardia asli** (latihan fisik v3.3, mode perangkat 50 Hz, ≥ 120 bpm, 2.189 window, 41 subjek):

| Sistem | Alarm AFIB/AFL [CI subjek] | < 100 bpm | PTB-XL STACH (811) |
|---|---|---|---|
| v3.1 paket | 98,4% [97–100] | 22% | 29,7% |
| **H-0 / H-A** | **0,1% [0–0]** | 1% | 1,6% |
| H-X | 7,3% [4–11] | 11% | 5,1% |

**Data eksternal** (bias/`b2` = median cross-fit AFDB; bootstrap pasien berpasangan vs paket v3.1):

| Data | Sistem | Macro-F1 sistem / v3.1 | P | Recall AFL sistem / v3.1 |
|---|---|---|---|---|
| PTB-XL (20.311; AFL 50 dari 39 pasien) | H-0 | 0,687 / 0,622 | 1,00, lebih baik | 0,14 / 0,50 |
| | H-A | 0,703 / 0,622 | 1,00, lebih baik | 0,18 / 0,50 |
| | H-X | 0,578 / 0,622 | 0,001, lebih buruk | 0,56 / 0,50 |
| Icentia11k (115.722; 2.420 pasien) | H-0 | 0,449 / 0,476 | 0,000, lebih buruk | 0,02 / 0,05 |
| | H-X | 0,488 / 0,476 | 0,989, lebih baik | 0,08 / 0,05 |

H-A vs H-0: AFDB Δ −0,001 (P = 0,46), PTB-XL Δ +0,015 (P = 0,83), tidak terbedakan.

**H5, keluaran keempat "takikardia teratur, perlu dicek"** (tahap 1 sinus dan v3.1 AFL): menampung 96% window AFL 08378 dan 58%
seluruh window AFL AFDB, 0,1% Normal AFDB, tetapi juga 74% [61–85] sinus takikardia asli ≥ 120 bpm dan 17% STACH PTB-XL.

**H6:** AUROC window AFL 08378 vs sinus takikardia asli ≥ 120 bpm: logit ECGFounder AFL − ST 0,78; skor RR v3.1 (log pAFL − log pN)
0,90. Satu pasien AFL dan dua dataset berbeda, jadi hanya petunjuk.

## Kesimpulan

- ECGFounder 1-lead pada EKG 50 Hz adalah detektor AFIB yang sangat spesifik (alarm palsu sinus takikardia hampir nol, juga saat
  bergerak dengan detektor 50 Hz), tetapi AFL, terutama AFL 2:1 dan AFL AFDB pada umumnya, dibacanya sebagai sinus. Hierarki tidak bisa
  memperbaiki AFL yang sudah dibuang di tahap 1, jadi tahap 2 atrium tidak berpengaruh (H3).
- Menyetel tahap 1 di AFDB (H-X) memulihkan sebagian AFL tetapi biasnya menempel di batas grid dan alarm sinus takikardia naik lagi; AFDB
  tidak bisa dipakai untuk memilih ambang ini.
- Untuk produk 2 kelas (AF ya/tidak), H-0 setara v3.1 dengan spesifisitas jauh lebih tinggi dan alarm palsu jauh lebih sedikit, dengan
  harga AFL. Keluaran "perlu dicek" menampung AFL 2:1, tetapi juga sebagian besar sinus takikardia. Apakah trade-off ini diterima adalah
  keputusan produk/klinis dan harus diuji pada rekaman perangkat berlabel.

## Lingkungan

PC desktop, RTX 5060 8 GB (tidak dipakai; semua dari cache), CPU 12 core. Python 3.13.16, numpy 2.5.4, scipy 1.18.1, scikit-learn
1.9.1, torch 2.11.0+cu128 (paket v3.1 untuk Icentia). Kode: commit `b18d6bc` + perubahan yang belum di-commit. Notebook COMPLETE lewat
`tools/run_notebook.py`, 6 November 2026 (±12 menit).
