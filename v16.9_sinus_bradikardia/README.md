# v16.9 — sinus bradikardia dan detak lambat pada S1 (D-1 + episode AF ≥ 30 s)

> **Status: selesai (10 November 2026).** Detak lambat adalah **kelemahan nyata S1**, tetapi sempit. Selama sinus bradikardia di LTAFDB,
> S1 memberi 8,89 alarm AF palsu/jam [2,32–14,93] (v3.1 + 30 s: 0,48; P(Δ > 0) = 0,998), dan 90% alarm itu berasal dari 3 dari 25 subjek.
> Pola yang sama, lebih lemah, muncul di EDB dan Icentia11k. Aturan detak tidak bisa dipakai karena 11–16% window AF sungguhan berdetak
> < 60 bpm. S1 tetap sistem terbaik untuk deteksi AF 2 kelas. Laporan:
> [laporan/laporan_v16_9_sinus_bradikardia.pdf](laporan/laporan_v16_9_sinus_bradikardia.pdf).

## Tujuan

Di v16.3, D-1 menyebut 27,9% window sinus bradikardia LTAFDB sebagai AF (v3.1: 4,2%). Versi minor ini menjawab, **tanpa data baru dan
tanpa training** (semua dari cache v16.2–v16.7):

- P1: apakah S1 masih memberi alarm palsu setelah aturan 30 s?
- P2: apakah alarm itu terkonsentrasi atau dekat episode AF?
- P3: apakah detak lambat memicu alarm S1 di dataset lain?
- P4: berapa banyak AF sungguhan yang berdetak lambat?

Rancangan dikunci sebelum alarm per ritme/pita dihitung: [RANCANGAN.md](RANCANGAN.md) (sha256 di `logs/v16.9_praregistrasi.log`).

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Pertanyaan P1–P4, metrik, kriteria (dikunci); bagian 8 mencatat eksplorasi setelah hasil |
| `bradikardia.py` | Pemuat 10 dataset dari cache (detak per window), episode palsu dan referensi (paritas dengan `episode.metrik`), laju per ritme/pita, bootstrap subjek berpasangan, AF lambat, detak dari anotasi LTAFDB |
| `bradikardia.ipynb` | Notebook utama; menulis `../cache/v16.9/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v16_9_sinus_bradikardia.pdf` | Laporan dua tahap (5 halaman) |

## Cara menjalankan

```bash
python tools/run_notebook.py v16.9_sinus_bradikardia/bradikardia.ipynb   # butuh cache v16.2–v16.7 dan anotasi data/ltafdb
cd v16.9_sinus_bradikardia && python laporan/build_report.py --extract && python laporan/build_report.py
```

Notebook ±3 menit di PC desktop tanpa GPU (10 November 2026), COMPLETE lewat `tools/run_notebook.py`.

## Hasil

**P1–P2, LTAFDB** (S1 vs v3.1 + 30 s; bootstrap subjek 2.000×):

| Ritme | Jam | Subjek | Palsu/jam S1 [CI] | Palsu/jam v3.1 [CI] | Waktu dialarmkan S1 / v3.1 |
|---|---|---|---|---|---|
| sinus bradikardia | 35,4 | 25 | **8,89 [2,32–14,93]** | 0,48 [0,00–2,15] | 12,1% / 0,5% |
| sinus normal | 746,6 | 50 | 0,66 [0,12–1,57] | 2,16 [1,04–3,60] | 1,0% / 2,8% |

- Konsentrasi: subjek 53, 13, dan 37 menyumbang 90% episode palsu S1. Hanya 7 dari 25 subjek yang punya setidaknya satu.
- Jarak ke AF: 27% episode palsu S1 terjadi dalam 10 menit dari AF (semua window SBR: 14%), tetapi 42% lebih dari 1 jam dari AF.

**P3, window non-AF di 10 dataset** (fraksi waktu dialarmkan, < 60 vs 60–100 bpm):
- S1 lebih sering alarm pada detak lambat menurut kriteria terkunci di LTAFDB, EDB, dan Icentia11k, sehingga lolos sebagai **kelemahan umum**.
- Polanya berbentuk U: alarm S1 paling sedikit di 60–100 bpm.
- Pada < 50 bpm, laju palsu S1 > 1,0/jam di 8 dari 10 dataset. Contoh Icentia11k: 1,08 (< 50) dan 1,41 (50–60) vs 0,07 (60–80).
- Pada ≥ 60 bpm v3.1 lebih sering alarm di semua dataset. Pada detak lambat S1 kadang lebih buruk (LTAFDB, NSRDB, SHDB-AF, afpdb, svdb).

**P4, AF lambat:** 10,8% (LTAFDB), 15,2% (CPSC 2021), 11,8% (SHDB-AF), dan 16,0% (Icentia11k) window AF berdetak < 60 bpm. S1
mendeteksi episode AF lambat dengan Se 0,87–1,00, jadi aturan "abaikan AF bila detak lambat" akan membuang banyak AF sungguhan.

**Eksplorasi (setelah hasil):** window SBR yang dialarmkan dibandingkan dengan yang tidak, memakai detak dari anotasi .atr. Keduanya tidak
berbeda:
- detak: 45,0 vs 45,5 bpm;
- CV RR: 0,019 di keduanya;
- detak ektopik: 6% vs 8% window;
- detak detektor sama dengan anotasi.

Jadi alarm bukan dari ritme tidak teratur atau R-peak yang salah, tetapi dari bentuk gelombang yang dilihat ECGFounder pada subjek tertentu.

## Kesimpulan

- S1 tetap sistem terbaik untuk deteksi AF 2 kelas. Kelemahan keduanya kini terukur: alarm AF palsu pada detak lambat (terutama < 50 bpm),
  terkonsentrasi pada sebagian kecil orang. v3.1 hampir tidak punya kelemahan ini karena hanya melihat RR yang teratur.
- Uji perangkat harus memuat rekaman tidur/malam dan orang dengan detak istirahat rendah, dengan kriteria alarm palsu per jam pada < 50 bpm.
- Hipotesis untuk data perangkat (belum diuji): pada detak lambat dengan RR sangat teratur, alarm D-1 diturunkan menjadi "perlu dicek".
  Versi global (S2 = D-1 dan v3.1, v16.5) kehilangan sensitivitas.

## Lingkungan

PC desktop, CPU saja (semua dari cache). Python 3.13.16, numpy 2.5.4. Kode: commit `b18d6bc` + perubahan yang belum di-commit.
