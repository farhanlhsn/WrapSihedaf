# v16.6 — aturan episode AF dengan toleransi celah: dipilih di data lama, divalidasi di SHDB-AF, afpdb, incartdb

> **Status: selesai (10 November 2026). Vonis utama gagal; analisis sekunder: aturan klinis 30 s (S1) lolos di data baru.** Tahap A
> memilih durasi 45 s tanpa celah. Di SHDB-AF konfigurasi itu memenuhi batas alarm palsu (0,38/jam) tetapi sensitivitas episode AF ≥ 30 s
> hanya 0,700 (target 0,90). Toleransi celah menaikkan alarm palsu jauh lebih cepat daripada sensitivitas. Sebaliknya, S1 = D-1 + 30 s (aturan
> v16.5, ditetapkan sebelum data ini dilihat) memenuhi kedua target pada nilai titik: SHDB-AF Se 0,942 [0,898–0,976], alarm palsu
> 0,87/jam [0,41–1,44]; afpdb 0,89; incartdb 0,77. v3.1 + 30 s: 1,80 / 2,21 / 5,98. Laporan:
> [laporan/laporan_v16_6_episode_celah.pdf](laporan/laporan_v16_6_episode_celah.pdf).

## Tujuan

v16.5: D-1 + aturan episode ≥ 30 s hampir memenuhi target produk (NSRDB 1,23 > 1,0 alarm palsu/jam). Eksplorasi setelah hasil
menunjukkan run AF terputus oleh window tunggal yang terlewat. v16.6:

- **Tahap A** memilih durasi minimum {30, 45, 60, 90} s × celah {0, 1, 2} window untuk D-1 di data v16.5, dengan kriteria yang dikunci.
- **Tahap B** menguji konfigurasi itu sekali di data yang belum pernah dipakai untuk D-1 maupun aturan episode.

Rancangan dikunci sebelum sinyal data validasi dibaca: [RANCANGAN.md](RANCANGAN.md) (sha256 di `logs/v16.6_praregistrasi.log`). Amandemen
A juga dibuat sebelum sinyal dibaca: incartdb dan svdb hanya menandai perubahan dari ritme normal, jadi waktu sebelum label pertama dihitung non-AF.

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Aturan celah, grid, pemilihan, data validasi, target, H1/H2 (dikunci + amandemen A) |
| `celah.py` | `aturan_celah` (paritas dengan v16.5 di-assert), Se episode per unit, bootstrap rasio berpasangan, grid & pemilihan, pemuat SHDB-AF/afpdb/incartdb/svdb (label ritme, unit subjek/pasien), pemrosesan per rekaman (mesin v16.4, bisa dilanjutkan, bisa dibagi ke beberapa proses) |
| `celah.ipynb` | Notebook utama; menulis `../cache/v16.6/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v16_6_episode_celah.pdf` | Laporan dua tahap (6 halaman) |

## Cara menjalankan

```bash
python tools/unduh_physionet.py svdb 1.0.0 --ke svdb --pola "*.dat" "*.hea" "*.atr" RECORDS
python tools/unduh_physionet.py afpdb 1.0.0 --ke afpdb --pola "*.dat" "*.hea" "*.qrs" RECORDS
python tools/unduh_physionet.py incartdb 1.0.0 --ke incartdb --pola "*.dat" "*.hea" "*.atr" RECORDS
cd v12_shdb_atrium && python atrium.py unduh && cd ..          # SHDB-AF (6,7 GB), bila belum ada
cd v16.6_episode_celah
python -c "import celah as C; C.jalankan('SHDB-AF', 0, 2)" &   # dua proses: ±22 menit (RTX 5060)
python -c "import celah as C; C.jalankan('SHDB-AF', 1, 2)"
cd .. && python tools/run_notebook.py v16.6_episode_celah/celah.ipynb   # ±9 menit; afpdb/incartdb/svdb diproses di sel pertama
cd v16.6_episode_celah && python laporan/build_report.py --extract && python laporan/build_report.py
```

Dijalankan 10 November 2026 di PC desktop (RTX 5060, Python 3.13), dengan penjaga RAM (proses dihentikan bila RAM tersedia < 1,5 GB).

## Hasil

**Tahap A (data pengembangan, D-1):**

| Durasi | Celah | Palsu/jam NSRDB | EDB | Latihan | Se ep. ≥ 30 s (min 4 data AF) |
|---|---|---|---|---|---|
| 30 s | 0 | 1,234 | 0,176 | 0,133 | 0,911 |
| 30 s | 1 | 3,622 | 0,358 | 1,131 | 0,958 |
| 30 s | 2 | 8,102 | 0,575 | 1,630 | 0,968 |
| **45 s** | **0** | **0,208** | **0,051** | **0,000** | **0,726** ← terpilih |
| 45 s | 1 | 1,327 | 0,182 | 0,333 | 0,773 |
| 60 s | 0 | 0,039 | 0,006 | 0,000 | 0,592 |
| 60 s | 1 | 0,497 | 0,068 | 0,266 | 0,682 |
| 90 s | 0–2 | 0,002–1,273 | ≤ 0,051 | ≤ 0,266 | 0,444–0,703 |

**Tahap B (data baru):**

| Data | Jam | D-1 45 s (palsu/jam, Se) | v3.1 45 s | **S1 = D-1 30 s** | v3.1 30 s |
|---|---|---|---|---|---|
| SHDB-AF (93 subjek, 1.648 episode AF ≥ 30 s) | 2.305 | 0,38; 0,700 | 0,80; 0,660 | **0,87; 0,942** | 1,80; 0,865 |
| afpdb (100 rekaman, non-AF) | 49 | 0,35 | 0,65 | **0,89** | 2,21 |
| incartdb (32 pasien, 4 episode AF) | 35 | 0,20; 1,000 | 2,72 | **0,77; 1,000** | 5,98 |
| svdb (deskriptif, ritme tidak dianotasi) | 39 | 0,93 | 2,98 | 2,07 | 6,37 |

- **Kriteria dikunci:** target Se SHDB-AF **gagal** (0,700 [0,622–0,801]); target alarm palsu lolos; H1 lolos (Δ −0,43/jam, P = 0,985);
  H2 lolos (Δ Se +0,039, P(Δ > 0) = 0,678). Keputusan: konfigurasi terpilih **bukan** sistem kandidat.
- **Sekunder, S1 vs v3.1 + 30 s:** alarm palsu lebih rendah di keempat data (SHDB-AF Δ −0,93, P(Δ < 0) = 0,991; afpdb P = 0,961;
  incartdb dan svdb P = 1,000). Se episode SHDB-AF +0,078 [−0,068; +0,231].
- Waktu non-AF SHDB-AF ≥ 120 bpm: S1 mengalarmkan 34% window, v3.1 + 30 s 95%. 14 dari 93 subjek di atas 1 alarm palsu/jam dengan S1.

**Pelajaran:** kriteria pemilihan mengukur Se untuk episode ≥ 30 s, padahal aturan ≥ 45 s tidak bisa mengalarmkan episode 30–45 s. Batas
alarm palsu yang ketat di NSRDB (18 subjek) mendorong pemilihan ke sana. Titik kerja yang dipilih dari definisi klinis lebih tahan
dibanding yang disetel di data negatif kecil.

## Keputusan

Vonis utama: gagal. **Sistem terbaik saat ini untuk deteksi AF 2 kelas (episode)** adalah D-1 + aturan episode ≥ 30 s (S1). Bukti: CPSC 2021,
MIT-BIH, LTAFDB, NSRDB, Georgia, EDB, latihan fisik, SHDB-AF, afpdb, incartdb. Status S1 di v16.6 adalah analisis sekunder; nilai titik
memenuhi target, tetapi CI menyentuh batas. Kelemahan tercatat: Se lebih rendah di EKG pendek non-Holter, alarm palsu ±1–2/jam di NSRDB
dan svdb, tidak membedakan AFIB dari AFL. Langkah berikutnya: uji S1 di rekaman perangkat SIHEDAF berlabel dengan rancangan yang dikunci.
Paket 3 kelas tetap v3.1; tidak ada `model_export/` yang dibuat atau diubah.
