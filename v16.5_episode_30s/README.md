# v16.5 — aturan episode AF ≥ 30 detik: alarm palsu per jam vs episode AF yang terdeteksi

> **Status: selesai (10 November 2026). Tidak ada sistem yang memenuhi target yang dikunci; S1 (D-1 + 30 s) gagal tipis.** Aturan klinis
> "AF ≥ 30 s" memangkas alarm palsu D-1 di orang sehat (NSRDB) dari 20,6 ke 1,23 per jam. Saat latihan fisik turun ke 0,13 per jam (v3.1
> + 30 s: 8,8), dan Se episode AF ≥ 30 s tetap 0,91–0,98 di empat data Holter. Target: ≤ 1,0 alarm palsu per jam di NSRDB, EDB, dan
> latihan, serta Se ≥ 0,90. S1 hanya gagal di NSRDB. H1 (S1 < R di NSRDB) dan H2 (Se durasi LTAFDB tidak turun > 0,05) lolos.
> Laporan: [laporan/laporan_v16_5_episode_30s.pdf](laporan/laporan_v16_5_episode_30s.pdf).

## Tujuan

Tanpa aturan durasi, D-1 memberi ±21 episode AF palsu per jam pada orang sehat (v16.4), v3.1 ±52; tidak layak untuk perangkat. Pedoman
klinis (ESC/HRS) menyebut episode AF yang tercatat sebagai AF ≥ 30 detik. v16.5 menerapkan aturan itu (tetap, tidak disetel) di atas
keputusan per window. Rancangan dikunci sebelum aturan dihitung dan sebelum sinyal EDB dibaca: [RANCANGAN.md](RANCANGAN.md).

- **Aturan:** run = window AF berturut-turut (jarak awal ≤ langkah + 0,5 s; window gagal cek kualitas memutus run). Alarm bila rentang
  run ≥ 30 s.
- **Sistem:** S1 = D-1 + 30 s (primer), S2 = (D-1 dan v3.1) + 30 s, R = v3.1 + 30 s (referensi).
- **Data:** AFDB, LTAFDB, MIT-BIH, CPSC 2021 (sensitivitas); NSRDB, **EDB** (European ST-T, belum pernah dipakai, 90 × 2 jam), latihan fisik
  v3.3 (alarm palsu). Semua keputusan per window dari cache v13/v16.1–v16.4/v3.3; EDB diproses dengan mesin v16.4.

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Aturan, sistem, data, target, H1/H2 (dikunci) |
| `episode.py` | Aturan run, metrik episode gaya EC57 dengan interval sebenarnya (Se ≥ 30 s, +P, Se/+P durasi, palsu/jam), bootstrap subjek, pemuat 7 dataset, EDB |
| `episode.ipynb` | Notebook utama; menulis `../cache/v16.5/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v16_5_episode_30s.pdf` | Laporan dua tahap (5 halaman) |

## Cara menjalankan

```bash
python tools/unduh_physionet.py edb 1.0.0 --ke edb --pola "*.hea" "*.dat" "*.atr" RECORDS
python tools/run_notebook.py v16.5_episode_30s/episode.ipynb       # butuh cache v13, v16.1–v16.4, v3.3
cd v16.5_episode_30s && python laporan/build_report.py --extract && python laporan/build_report.py
```

EDB diproses sekali ke `cache/v16.5/EDB/` (±2,5 menit GPU). Notebook ±7 menit (memuat LTAFDB 1,27 juta window). Dijalankan 10 November 2026
di PC desktop (RTX 5060, Python 3.13).

## Hasil (aturan 30 s)

| Data | Sistem | Palsu/jam tanpa aturan | Palsu/jam 30 s | Se episode ≥ 30 s | Se durasi |
|---|---|---|---|---|---|
| AFDB | R / **S1** / S2 | 11,4 / 1,8 / 1,3 | 1,59 / **0,12** / 0,10 | 0,947 / **0,927** / 0,886 | 0,911 / 0,908 / 0,838 |
| LTAFDB | R / **S1** / S2 | 12,5 / 5,7 / 2,7 | 0,94 / **0,46** / 0,19 | 0,922 / **0,941** / 0,885 | 0,925 / 0,955 / 0,908 |
| MIT-BIH | R / **S1** / S2 | 37,2 / 4,9 / 4,0 | 3,71 / **0,20** / 0,10 | 0,886 / **0,977** / 0,864 | 0,831 / 0,928 / 0,813 |
| CPSC 2021 | R / **S1** / S2 | 25,7 / 7,0 / 5,2 | 2,98 / **0,81** / 0,62 | 0,892 / **0,911** / 0,835 | 0,913 / 0,912 / 0,859 |
| NSRDB | R / **S1** / S2 | 51,9 / 20,6 / 16,3 | 3,57 / **1,23** / 0,45 | – | – |
| EDB | R / **S1** / S2 | 15,5 / 5,2 / 1,1 | 1,02 / **0,18** / 0,05 | 1 episode | – |
| Latihan fisik | R / **S1** / S2 | 40,0 / 5,6 / 4,3 | 8,82 / **0,13** / 0,03 | – | – |

- **Target (≤ 1,0 palsu/jam di NSRDB, EDB, latihan; Se ≥ 0,90 di 4 data AF):** R gagal keduanya, S1 gagal hanya NSRDB (1,23), S2 gagal
  Se (0,83–0,89). Tidak ada sistem terpilih.
- **H1** S1 vs R, palsu/jam NSRDB: Δ −2,34 [−4,52; −0,92], P(Δ < 0) = 1,000 → lolos (EDB: Δ −0,85, P = 1,000).
- **H2** Se durasi LTAFDB S1 vs D-1 tanpa aturan: turun 0,029 → lolos.
- NSRDB per subjek (S1): 10 dari 18 subjek > 1 alarm palsu/jam (tertinggi 17052: 2,45); tersebar, bukan satu subjek.
- Kurva durasi (deskriptif): D-1 di NSRDB 0,039 palsu/jam pada 60 s, tetapi Se episode ≥ 30 s LTAFDB 0,941 → 0,592 (sebagian karena
  definisi: aturan 60 s tidak mengalarmkan episode 30–60 s). Eksplorasi setelah hasil: run juga terputus oleh satu window yang terlewat
  → [v16.6_episode_celah](../v16.6_episode_celah/).

## Keputusan

Aturan episode ≥ 30 s adalah langkah paling berpengaruh sejauh ini untuk alarm palsu, dan D-1 jauh lebih baik daripada v3.1 dengan
aturan yang sama (terutama saat olahraga). Target produk belum terpenuhi; konfigurasi durasi/celah dipilih dan divalidasi di v16.6 dengan
rancangan baru. Bukan pengganti paket v3.1; tidak ada `model_export/` yang dibuat atau diubah.
