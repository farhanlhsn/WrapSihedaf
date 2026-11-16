# v16.2 — uji ambang t = −1 detektor AF 2 kelas (ECGFounder) di CPSC 2021

> **Status: selesai (6 November 2026). Kandidat detektor AF 2 kelas untuk uji perangkat.** Pada data yang belum pernah dinilai
> (CPSC 2021, 105 subjek), D-1 (ECGFounder zero-shot, AF bila d1 > −1) lebih baik dari paket v3.1 (Macro-F1 2 kelas 0,953 vs 0,881)
> dengan sensitivitas yang sama (0,955 vs 0,951) dan alarm palsu jauh lebih sedikit. Bukan pengganti paket v3.1 dan tidak dikemas.
> Laporan: [laporan/laporan_v16_2_cpsc2021.pdf](laporan/laporan_v16_2_cpsc2021.pdf).

> **Tindak lanjut:** replikasi di CinC 2017, MIT-BIH, LTAFDB → [v16.3_replikasi](../v16.3_replikasi/): sebagian (2/3); sensitivitas lebih rendah di EKG genggam satu lead (CinC 2017).

## Tujuan

v16.1 menunjukkan bahwa ECGFounder sebagai detektor AF 2 kelas sangat spesifik tetapi kurang sensitif pada ambang 0, dan kurva ambang
yang dilihat **setelah** hasil menyarankan t = −1. v16.2 mengunci t = −1 lalu mengujinya di CPSC 2021, yang sinyalnya belum pernah
dinilai model SIHEDAF mana pun (v15 hanya membaca anotasinya). Tidak ada yang dilatih atau disetel. Rancangan:
[RANCANGAN.md](RANCANGAN.md) (dikunci sebelum sinyal dibaca; sha256 di `logs/v16.2_praregistrasi.log`; satu penyimpangan di bagian 8).

| Sistem | Keputusan AF |
|---|---|
| **D-1 (primer)** | ECGFounder zero-shot 1-lead: `log((pAF + pAFL) / pN) > −1` |
| D-0 (v16.1) | `... > 0` |
| **v3.1 paket (referensi)**, v1 paket | argmax paket pada RR window, AFIB + AFL digabung |

Data: kanal 0 (lead I) setiap rekaman, window perangkat (`hrv_core.signal_to_windows`: resample 50 Hz, detektor R produksi, 10 s,
langkah 5 s) yang lolos cek kualitas. Bootstrap subjek berpasangan 2.000×.

## Penyimpangan: label Normal CPSC 2021

Aturan label yang dikunci memakai `v15.segmen_ritme`, yang hanya menganggap bagian setelah label `(N` sebagai Normal. Kode penilaian
resmi (`data/cpsc2021/python_entry/score_2021.py`) memakai `(AFIB`/`(AFL` sebagai **awal** episode AF dan `(N` sebagai **akhir**;
rekaman "non atrial fibrillation" tidak punya label ritme. Akibatnya analisis yang dikunci membuang 721 rekaman non-AF (window Normal
hanya dari 22 subjek). Kedua analisis dilaporkan: window AF identik (di-assert), dan kandidat hanya bila kriteria lolos di keduanya.

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Sistem, ambang, hipotesis, kriteria (dikunci), penyimpangan |
| `uji_cpsc.py` | Window perangkat berlabel per rekaman (dua aturan label, bisa dilanjutkan), interval AF resmi, logit ECGFounder, paket v3.1/v1 per potongan, per subjek, alarm per pita |
| `uji_cpsc.ipynb` | Notebook utama; menulis `../cache/v16.2/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v16_2_cpsc2021.pdf` | Laporan dua tahap |

## Cara menjalankan

```bash
python tools/unduh_physionet.py cpsc2021 1.0.0 --ke cpsc2021          # 1,4 GB (sudah ada sejak v15)
python tools/run_notebook.py v16.2_cpsc2021/uji_cpsc.ipynb
python v16.2_cpsc2021/laporan/build_report.py --extract && python v16.2_cpsc2021/laporan/build_report.py
```

Butuh GPU untuk ECGFounder (±3 menit untuk 322 ribu window di RTX 5060; CPU ±30 ms/window), bobot di `data/ecgfounder/`, dan cache
v3.3, v10, v13, v15, v16, v16.1 untuk bagian H4. Membaca dan memotong 1.425 rekaman ±3 menit per aturan label; hasil per rekaman di
`cache/v16.2/rekaman[_resmi]/`.

## Hasil

**CPSC 2021, 2 kelas** (CI 95% bootstrap subjek; P = P(Δ > 0) berpasangan vs v3.1 paket):

| Analisis | Sistem | Macro-F1 [CI] | Sensitivitas | Spesifisitas | PPV | vs v3.1 (P) |
|---|---|---|---|---|---|---|
| Dikunci (124.959 window, 54 subjek; Normal 13.495 dari 22 subjek) | v3.1 paket | 0,883 [0,78–0,93] | 0,951 | 0,926 | 0,991 | – |
| | **D-1** | 0,908 [0,81–0,95] | 0,955 | 0,994 | 0,999 | 0,964, **lebih baik** |
| | D-0 | 0,835 [0,70–0,91] | 0,906 | 0,998 | 1,000 | 0,004, lebih buruk |
| | v1 paket | 0,892 [0,79–0,93] | 0,957 | 0,925 | 0,991 | 0,904 |
| Label resmi (321.818 window, 105 subjek; Normal 210.354 dari 76 subjek) | v3.1 paket | 0,881 [0,83–0,91] | 0,951 | 0,853 | 0,774 | – |
| | **D-1** | 0,953 [0,92–0,97] | 0,955 | 0,958 | 0,923 | 1,00, **lebih baik** |
| | D-0 | 0,946 [0,92–0,96] | 0,906 | 0,976 | 0,951 | 1,00, lebih baik |
| | v1 paket | 0,867 [0,82–0,90] | 0,957 | 0,829 | 0,748 | 0,001, lebih buruk |

D-1 − v3.1, sensitivitas: +0,003, P = 0,69 (dikunci) dan P = 0,70 (label resmi), tidak terbedakan. D-1 − D-0, sensitivitas: +0,049 (P = 1,00)
di kedua analisis.

**Kriteria yang dikunci (D-1):** Macro-F1 bukan lebih buruk ✔, sensitivitas bukan lebih buruk ✔, alarm sinus takikardia asli ≤ separuh
v3.1 ✔ (0,46% [0,1–0,9] vs 98,4%), di kedua analisis → **kandidat uji perangkat**.

**Alarm palsu (label resmi):** window non-AF dialarmkan 4,2% (v3.1 14,7%); per pita detak < 100 / 100–110 / 110–120 / ≥ 120 bpm: 4,1% /
5,3% / 7,0% / 25,1% (v3.1 14,2% / 18,2% / 48,9% / 93,8%). Episode AF palsu 25,7 → 7,0 per jam; Se episode 0,96 → 0,97; +P episode
0,34 → 0,66.

**Data yang sudah dipakai (H4, bukan uji independen):** AFDB 0,968 vs 0,926 (Se 0,935 vs 0,934); PTB-XL 0,939 vs 0,817 (Se 0,912 vs
0,917); Icentia11k 0,870 vs 0,858 (P = 0,98), tetapi Se 0,843 vs 0,871 (lebih buruk).

**Per subjek:** sensitivitas D-1 turun > 0,20 hanya di subjek 61 (−0,33). AFL: 144 window dari 2 subjek, teralarm 92% (v3.1 98%).

Kendali mutu: window AF kedua aturan label identik; prediksi paket pada window itu identik (paket diproses per potongan setelah
prediksi v1 sekaligus kehabisan memori); keputusan D-1 sama 100% (selisih d1 maks 0,03 karena batch GPU berbeda).

## Kesimpulan

- Ambang t = −1 yang dipilih dari kurva v16.1 bertahan di data baru: D-1 sama sensitifnya dengan v3.1 dan jauh lebih spesifik. Ini
  sistem pertama di seri v13–v16.2 yang lolos kriteria kandidat yang dikunci, di data yang belum pernah dinilai, untuk keluaran
  **2 kelas**.
- Untuk 3 kelas, tidak ada yang berubah: ECGFounder tidak membedakan AFL (v16). Paket v3.1 tetap paket produksi.
- Langkah berikutnya: uji D-1 (`v16.1_dua_kelas/deteksi_af.DetektorAF` dengan ambang −1) pada rekaman perangkat SIHEDAF berlabel, dan
  putuskan apakah produk cukup 2 kelas. Alarm pada ritme non-AF cepat (≥ 120 bpm, 25% di Holter CPSC) perlu diperiksa di sana.

## Lingkungan

PC desktop, GPU NVIDIA RTX 5060 8 GB, CPU 12 core. Python 3.13.16, numpy 2.5.4, scikit-learn 1.9.1, torch 2.11.0+cu128. Kode: commit
`b18d6bc` + perubahan yang belum di-commit. Notebook COMPLETE lewat `tools/run_notebook.py`, 6 November 2026 (run ketiga: run pertama
dengan aturan yang dikunci, run kedua kehabisan memori di prediksi paket v1, run ketiga lengkap dengan cache per rekaman).
