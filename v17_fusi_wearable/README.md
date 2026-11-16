# v17 — fusi ECGFounder + RR v3.1 yang dilatih dengan EKG genggam, untuk sensitivitas AF di data wearable

> **Status: selesai (6 November 2026). Tidak menggantikan D-1.** W-F (regresi logistik 3 parameter pada d1 ECGFounder dan log-odds AF v3.1,
> dilatih di CinC 2017) menutup celah sensitivitas di Icentia11k (0,921 vs D-1 0,843), tetapi mengalarmkan 57% sinus takikardia asli
> dan lebih buruk dari D-1 di CPSC 2021 dan MIT-BIH. D-1 tetap kandidat detektor AF 2 kelas; paket v3.1 tetap produksi.
> Laporan: [laporan/laporan_v17_fusi_wearable.pdf](laporan/laporan_v17_fusi_wearable.pdf).

## Tujuan

D-1 (ECGFounder zero-shot, d1 > −1) lebih spesifik dari v3.1 di tujuh dataset, tetapi kurang sensitif pada EKG genggam/wearable satu
lead (v16.3). Model RR v3.1 lebih sensitif di sana tetapi mengalarmkan sinus takikardia. v17 menguji gabungan kecil keduanya yang dilatih
dengan EKG genggam. Rancangan dikunci sebelum model dilatih: [RANCANGAN.md](RANCANGAN.md) (sha256 di `logs/v17_praregistrasi.log`).

**Lisensi:** data latih CinC 2017 berlisensi CC BY 3.0 (boleh untuk produk dengan atribusi). Icentia11k (CC BY-NC-SA) hanya dipakai
untuk evaluasi, tidak untuk melatih atau memilih apa pun.

| Sistem | Keputusan AF |
|---|---|
| **W-F (primer)** | regresi logistik pada `[d1, r]`, `r = log((pAFIB + pAFL)/pN)` paket v3.1; StandardScaler + LR(C = 1, balanced), p ≥ 0,5; dilatih di seluruh CinC 2017 |
| D-1 | d1 > −1 |
| v3.1 paket | argmax, AFIB + AFL digabung |
| D-1 ∨ v3.1 | salah satu menyebut AF |

Model yang dipelajari: **AF bila 0,629·d1 + 0,272·r + 1,616 ≥ 0** (`cache/v17/wf.json`), kira-kira d1 > −2,6 − 0,43·r.

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Data, sistem, kriteria K1–K5 (dikunci) |
| `fusi.py` | Masukan, pelatihan (resep `v15.latih_gabungan`), koefisien, validasi silang CinC per rekaman, pemuat semua dataset dari cache, evaluasi |
| `fusi.ipynb` | Notebook utama; menulis `../cache/v17/ringkasan.json` dan `wf.json` |
| `laporan/build_report.py`, `laporan/laporan_v17_fusi_wearable.pdf` | Laporan dua tahap |

## Cara menjalankan

```bash
python tools/run_notebook.py v17_fusi_wearable/fusi.ipynb
python v17_fusi_wearable/laporan/build_report.py --extract && python v17_fusi_wearable/laporan/build_report.py
```

Tidak butuh GPU: semua masukan dari cache v10, v13, v16, v16.2, v16.3 (±1 menit).

## Hasil

| Data | Peran | Sistem | Macro-F1 | Sensitivitas | Spesifisitas |
|---|---|---|---|---|---|
| Icentia11k (2.420 pasien) | uji utama | **W-F** | **0,913** | **0,921** | 0,934 |
| | | D-1 | 0,870 | 0,843 | 0,972 |
| | | v3.1 paket | 0,858 | 0,871 | 0,882 |
| | | D-1 ∨ v3.1 | 0,902 | 0,937 | 0,870 |
| CPSC 2021 (105 subjek) | penjaga | W-F / D-1 / v3.1 | 0,931 / 0,953 / 0,881 | 0,993 / 0,955 / 0,951 | 0,905 / 0,958 / 0,853 |
| MIT-BIH (46 subjek) | penjaga | W-F / D-1 / v3.1 | 0,867 / 0,967 / 0,754 | 0,998 / 0,983 / 0,916 | 0,929 / 0,987 / 0,857 |
| LTAFDB (84 subjek) | penjaga | W-F / D-1 / v3.1 | 0,971 / 0,967 / 0,933 | 0,989 / 0,974 / 0,948 | 0,949 / 0,960 / 0,915 |
| CinC 2017 (OOF per rekaman) | pendukung | W-F / D-1 / v3.1 | 0,823 / 0,867 / 0,757 | 0,918 / 0,830 / 0,884 | 0,926 / 0,965 / 0,886 |
| AFDB (5 ulangan) | deskriptif | W-F / D-1 / v3.1 | 0,980 / 0,968 / 0,929 | 0,984 / 0,935 / 0,939 | 0,978 / 0,992 / 0,926 |
| PTB-XL | deskriptif | W-F / D-1 / v3.1 | 0,906 / 0,939 / 0,817 | 0,964 / 0,912 / 0,917 | 0,971 / 0,988 / 0,935 |

**Sinus takikardia asli ≥ 120 bpm:** W-F 56,6% [45–68], D-1 0,5%, v3.1 98,4%, D-1 ∨ v3.1 98,4%. Per pita: 120–130 bpm 27%, ≥ 130 bpm
73–78%. PTB-XL STACH: W-F 6,0%, D-1 2,5%, v3.1 29,7%.

**Kriteria yang dikunci (W-F):**

| | Hasil |
|---|---|
| K1 sensitivitas Icentia lebih baik dari D-1 | lolos (+0,078, P = 1,00) |
| K2 Macro-F1 Icentia bukan lebih buruk dari D-1 | lolos (+0,043, lebih baik) |
| K3 Macro-F1 dan sensitivitas Icentia bukan lebih buruk dari v3.1 | lolos (+0,056; +0,050) |
| K4 alarm sinus takikardia asli ≤ 10% | **gagal (56,6%)** |
| K5 Macro-F1 CPSC 2021, MIT-BIH, LTAFDB bukan lebih buruk dari D-1 | **gagal** (CPSC −0,022, MIT-BIH −0,100; LTAFDB +0,004 tidak terbedakan) |

Tanpa window Normal cepat tambahan v10, Icentia W-F 0,912 vs D-1 0,862 (sama arahnya).

## Kesimpulan

- Celah sensitivitas D-1 di data wearable **bisa** ditutup: model yang dilatih di EKG genggam menurunkan ambang ECGFounder dan menambah
  suara RR, dan di Icentia11k jadi lebih baik dari D-1 dan v3.1 di semua metrik utama kecuali spesifisitas.
- Harganya persis kelemahan lama model RR: sinus takikardia saat aktivitas kembali dialarmkan (57%), dan spesifisitas di Holter turun.
  Di CinC 2017 hanya 1,4% window non-AF yang ≥ 120 bpm, jadi model yang dilatih di sana hampir tidak melihat biaya itu (pelajaran v1–v6
  dan v16 H-X berulang).
- Titik kerja antara sensitivitas wearable dan alarm saat aktivitas tidak bisa dipilih dari data publik yang ada. Itu harus dipilih dari
  rekaman perangkat SIHEDAF yang memuat aktivitas fisik. D-1 tetap kandidat; W-F tercatat sebagai alternatif bila uji perangkat
  menunjukkan sensitivitas D-1 tidak cukup **dan** alarm saat aktivitas bisa ditangani lain (mis. sensor gerak).

## Lingkungan

PC desktop, CPU 12 core (tanpa GPU). Python 3.13.16, numpy 2.5.4, scikit-learn 1.9.1. Kode: commit `b18d6bc` + perubahan yang belum
di-commit. Notebook COMPLETE lewat `tools/run_notebook.py`, 6 November 2026 (±1 menit).
