# v15 — kuantisasi RR dan konsistensi pra-QRS untuk AFL konduksi bervariasi (pra-registrasi)

> **Status: selesai (6 November 2026). Hasil utama: tidak cukup data** (CPSC 2021 hanya memuat AFL dari 2 subjek). Analisis tambahan
> di PTB-XL mendukung bahwa kedua fitur memisahkan AFL dari AFIB, tetapi tahap gabungan sederhana memperburuk AFDB.
> Laporan: [laporan/laporan_v15_kuantisasi_rr.pdf](laporan/laporan_v15_kuantisasi_rr.pdf).

## Tujuan

Menguji hipotesis eksploratif v12 (`p_konsisten` dan `kuantisasi` memisahkan AFL dari AFIB, juga AFL yang lolos dari model RR) pada
data yang belum pernah dilihat, dengan rancangan yang dikunci lebih dulu: [RANCANGAN.md](RANCANGAN.md) (ditulis sebelum CPSC 2021
selesai diunduh; sha256 dan waktu di `logs/v15_praregistrasi.log`; penyimpangan di bagian 8).

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Hipotesis H1/H2, ambang, syarat data (dikunci), penyimpangan tercatat |
| `kuantisasi.py` | Syarat data CPSC 2021, fitur PTB-XL, AUROC dengan bootstrap pasien, tahap gabungan (AFDB fold resmi, PTB-XL), uji stres |
| `kuantisasi_rr.ipynb` | Notebook utama; menulis `../cache/v15/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v15_kuantisasi_rr.pdf` | Laporan dua tahap |

Fitur dari `v12_shdb_atrium/atrium.fitur_atrium(..., revisi=1)` tanpa perubahan; PTB-XL dibaca lewat `v13_ecgfounder/ecgfounder.ptbxl`.

## Cara menjalankan

```bash
python tools/unduh_physionet.py cpsc2021 1.0.0 --ke cpsc2021                                   # 1,4 GB
python tools/unduh_physionet.py ptb-xl 1.0.3 --ke ptbxl --pola "*.csv" "records100/*"         # 0,5 GB
python tools/run_notebook.py v15_kuantisasi_rr/kuantisasi_rr.ipynb
python v15_kuantisasi_rr/laporan/build_report.py --extract && python v15_kuantisasi_rr/laporan/build_report.py
```

Butuh cache v12 (`cache/v12/fitur_r1.npz`, `afdb_fitur.npz`, `shdb/`) dan SHDB-AF di `data/shdb_af/`.

## Hasil

**Hasil utama (CPSC 2021, 1.425 rekaman yang dirilis):** AFL hanya dari 2 subjek (25 dan 79), 26,3 menit, 84 window 10 s murni.
Syarat ≥ 5 subjek dan ≥ 200 window tidak terpenuhi → **tidak cukup data**. AFIB 53 subjek (9.867 menit), Normal 54 subjek. Catatan:
11 rekaman di `RECORDS` (subjek 51 dan sebagian 50) tidak ada di `SHA256SUMS.txt` resmi.

> Catatan (6 November 2026, v16.2): di CPSC 2021 `(AFIB`/`(AFL` menandai awal episode dan `(N` menandai akhirnya; rekaman non-AF
> tidak punya label ritme (`score_2021.py` resmi). Jumlah Normal di atas (dari segmen setelah `(N`) karena itu terlalu kecil;
> jumlah AFL dan kesimpulan "tidak cukup data" tidak berubah. Lihat `v16.2_cpsc2021/uji_cpsc.interval_af`.

**Analisis tambahan, PTB-XL (lead I 100 → 50 Hz; 20.500 rekaman; AFL 50 dari 39 pasien; bukan hasil utama):**

| Fitur | AUROC AFL vs AFIB [CI 95%] | AFL sulit (bukan cepat-teratur) | Ambang H1 (≥ 0,70, CI bawah > 0,5) |
|---|---|---|---|
| konsistensi pra-QRS (`p_konsisten`) | 0,924 [0,87–0,97] | 0,855 [0,77–0,93] | lolos |
| kuantisasi RR (`kuantisasi`) | 0,858 [0,77–0,93] | 0,748 [0,61–0,87] | lolos |

Tahap gabungan (log-probabilitas v3.1 + 2 fitur, dilatih di AFDB + SHDB-AF) di PTB-XL: recall AFL 0,50 → 0,76 (P = 1,00); Macro-F1
0,622 → 0,603 (P = 0,072, tidak terbedakan).

**H2 di AFDB (fold resmi 5×5, + SHDB-AF, bias cross-fit):**

| Sistem | Macro-F1 [CI] | AFL F1 [CI] | AFL R | AFL P |
|---|---|---|---|---|
| v1 (argmax) | 0,744 [0,64–0,81] | 0,399 [0,09–0,59] | 0,728 | 0,275 |
| v3.1 (referensi) | 0,832 [0,67–0,92] | 0,647 [0,17–0,89] | 0,592 | 0,715 |
| v15 gabungan | 0,756 [0,66–0,86] | 0,413 [0,18–0,71] | 0,346 | 0,521 |

Gabungan vs v3.1: Macro-F1 −0,076 (P = 0,037, **lebih buruk**). Recall AFL per pasien (gabungan / v3.1): 04043 0,20/0,40 · 04908
0,69/0,92 · 04936 0,16/0,51 · **06426 0,47/0,17** · **07910 0,10/0,00** · 08215 0,00/0,00 · 08378 0,78/0,96. Uji stres takikardia
(fitur = median): ambang 120/120/115/115 bpm vs v3.1 115.

## Kesimpulan

- Uji yang direncanakan tidak bisa dijalankan: data publik dengan AFL kontinu dari banyak subjek hampir tidak ada (CPSC 2021: 2 subjek).
- Informasi atrium dari EKG 50 Hz, terutama konsistensi pra-QRS, membedakan AFL dari AFIB di data yang belum pernah dilihat
  (analisis tambahan), termasuk AFL yang lolos dari model RR.
- Penggabungan sederhana dengan v3.1 untuk tiga kelas sekaligus merusak pola AFL yang sudah tertangkap (terutama 04936). Usulan:
  pakai fitur ini hanya untuk memisahkan AFIB vs AFL pada window yang sudah diputuskan AF, dan uji pada rekaman perangkat berlabel
  dengan rancangan baru yang dikunci lebih dulu.

## Lingkungan

PC desktop, CPU 12 core. Python 3.13, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, torch 2.11.0+cu128 (paket v3.1). Kode: commit
`b18d6bc` + perubahan yang belum di-commit. Notebook COMPLETE lewat `tools/run_notebook.py`, 6 November 2026 (±4 menit dengan cache).
