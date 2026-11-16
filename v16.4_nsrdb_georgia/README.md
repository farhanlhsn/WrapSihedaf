# v16.4 — D-1 dan v3.1 di NSRDB (Holter sehat 24 jam) dan Georgia (EKG klinis 12-lead)

> **Status: selesai (10 November 2026). 3 dari 4 kriteria lolos; H2 (sensitivitas) gagal.** D-1 jauh lebih spesifik daripada paket v3.1
> di dua data yang belum pernah dipakai: alarm palsu pada orang sehat 15,0% → 5,6% window, alarm sinus takikardia klinis 50% → 2%,
> Macro-F1 Georgia 0,691 → 0,843. Tetapi sensitivitas AF di EKG klinis 10 s lebih rendah (0,706 vs 0,770, P = 0,000). Kekurangan
> sensitivitas 3–6 poin kini konsisten di tiga data non-Holter (CinC 2017, Icentia11k, Georgia), sedangkan di keempat data Holter D-1 setara
> atau lebih baik. Laporan: [laporan/laporan_v16_4_nsrdb_georgia.pdf](laporan/laporan_v16_4_nsrdb_georgia.pdf).

## Tujuan

D-1 (ECGFounder 1-lead zero-shot, AF bila d1 > −1) adalah kandidat detektor AF 2 kelas. Ketujuh data publik sebelumnya sudah dipakai untuk
D-1. Dua data baru menjawab dua pertanyaan produk:

1. Berapa alarm AF palsu pada orang sehat sepanjang hari? (NSRDB, 18 subjek, Holter ±24 jam)
2. Apakah keunggulan D-1 bertahan di EKG klinis lain dengan label dokter untuk sinus takikardia, sinus bradikardia, dan AFL? (Georgia,
   PhysioNet/CinC Challenge 2021, 10.344 EKG 12-lead 10 s, lead I)

Rancangan dikunci sebelum sinyal dibaca: [RANCANGAN.md](RANCANGAN.md) (sha256 di `logs/v16.4_praregistrasi.log`).

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Data, kelompok Georgia (prioritas AF/AFL → sinus takikardia → sinus bradikardia → sinus normal → lainnya), H1–H4 (dikunci) |
| `uji_baru.py` | Daftar rekaman, Dx SNOMED Georgia, rantai prediksi per rekaman ke cache (bisa dilanjutkan), evaluasi Georgia dan NSRDB |
| `uji_baru.ipynb` | Notebook utama; menulis `../cache/v16.4/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v16_4_nsrdb_georgia.pdf` | Laporan dua tahap (6 halaman) |

## Cara menjalankan

```bash
python tools/unduh_physionet.py nsrdb 1.0.0 --ke nsrdb --pola "*.hea" "*.dat" "*.atr" RECORDS
python tools/unduh_physionet.py challenge-2021 1.0.3 --ke challenge2021 --sumber physionet --thread 6 --pola "training/georgia/*"
python tools/run_notebook.py v16.4_nsrdb_georgia/uji_baru.ipynb
cd v16.4_nsrdb_georgia && python laporan/build_report.py --extract && python laporan/build_report.py
```

- `nsrdb`: `SHA256SUMS.txt` memuat berkas `17453.hea-` yang tidak ada (404), jadi unduh dengan `--pola`.
- Challenge 2021 tidak ada di mirror S3; `--sumber physionet` (baru di `tools/unduh_physionet.py`) mengunduh dari physionet.org. Pakai
  ≤ 6 thread; 16 thread memicu timeout. Georgia 1,24 GB, ±36 menit.
- Butuh GPU untuk ECGFounder dan bobot di `data/ecgfounder/`. Pemrosesan NSRDB ±6 menit, Georgia ±4 menit (RTX 5060).

Dijalankan 10 November 2026 di PC desktop (RTX 5060, Python 3.13), notebook ±3,5 menit (termasuk pemrosesan Georgia).

## Hasil

**Georgia (lead I, satu window 10 s per EKG, 10.249 lolos cek kualitas; CI 95% bootstrap rekaman 2.000×):**

| Sistem | Macro-F1 | Sensitivitas | Spesifisitas | PPV |
|---|---|---|---|---|
| v1 paket | 0,664 [0,652–0,677] | 0,786 [0,755–0,817] | 0,850 | 0,285 |
| v3.1 paket (referensi) | 0,691 [0,677–0,704] | 0,770 [0,736–0,799] | 0,877 | 0,323 |
| D-0 (t = 0) | 0,860 [0,845–0,873] | 0,646 [0,609–0,678] | 0,992 | 0,857 |
| **D-1 (t = −1)** | **0,843 [0,828–0,856]** | 0,706 [0,673–0,739] | **0,978** | **0,709** |

D-1 − v3.1: Macro-F1 +0,152 [+0,138; +0,166] P = 1,000; Se −0,063 [−0,093; −0,035] P = 0,000; Sp +0,101 P = 1,000.

| Kelompok Georgia (alarm AF) | EKG | v3.1 | D-1 |
|---|---|---|---|
| sinus normal | 1.734 | 2,0% | 0,5% |
| sinus takikardia | 1.233 | 50,4% | **2,0%** |
| sinus bradikardia | 1.648 | 5,0% | 5,6% |
| lainnya | 4.909 | 8,8% | 1,7% |
| AF (sensitivitas) | 555 | 80,0% | 73,7% |
| AFL (sensitivitas) | 170 | 67,1% | 60,6% |
| AT/SVT | 52 | 80,8% | 17,3% |

**NSRDB (18 subjek, 294.162 window, ±409 jam, semua non-AF):**

| Sistem | Window dialarmkan [CI] | Tanpa window berartefak | Episode AF palsu/jam (tanpa aturan durasi) |
|---|---|---|---|
| v1 paket | 22,17% [16,59–28,57] | 22,37% | 70,7 |
| v3.1 paket | 15,00% [10,90–19,96] | 15,30% | 51,9 |
| D-0 | 2,08% [1,63–2,58] | 2,30% | 10,6 |
| **D-1** | **5,64% [4,30–7,15]** | 5,57% | 20,6 |

D-1 − v3.1: −9,36 poin [−14,59; −5,32], P(Δ < 0) = 1,000; D-1 lebih rendah pada 18 dari 18 subjek.

**Kriteria yang dikunci:** H1 Macro-F1 lolos, **H2 sensitivitas gagal**, H3 sinus takikardia lolos (2,0% vs 50,4%), H4 NSRDB lolos.
Sekunder (Se non-inferior, margin −0,02): gagal.

**Pita detak (non-AF):** v3.1 mengalarmkan 97–98% ritme ≥ 120 bpm di kedua data. D-1: Georgia 7,5%, NSRDB 25% (110–120) dan 22%
(≥ 120); Holter ambulatori lebih sulit daripada EKG istirahat.

**Eksplorasi setelah hasil (bukan uji yang dikunci):** pada AF Georgia yang RR-nya tidak teratur (CV IBI ≥ 0,08, n = 558), D-1 0,864 vs
v3.1 0,912, jadi kekurangan sensitivitas bukan hanya soal label. 167 EKG berlabel AF/AFL justru teratur (CV < 0,08); keduanya hampir tidak
menangkapnya (v3.1 0,293, D-1 0,180): AFL dengan konduksi tetap atau label tingkat rekaman.

## Keputusan

Sesuai rancangan, ambang tidak diubah. D-1 tetap kandidat untuk **perangkat yang merekam terus-menerus** (Holter/patch), di mana AF yang
terlewat di satu window bisa tertangkap di window berikutnya. Untuk pemeriksaan sesaat 10–60 s (EKG genggam), D-1 belum cukup sensitif.
Tanpa aturan durasi, D-1 masih memberi ±21 episode AF palsu per jam pada orang sehat. Itu diuji di
[v16.5_episode_30s](../v16.5_episode_30s/). Bukan pengganti paket v3.1; tidak ada `model_export/` yang dibuat atau diubah.
