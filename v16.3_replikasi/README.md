# v16.3 — replikasi detektor AF 2 kelas D-1 (ECGFounder, ambang −1) di CinC 2017, MIT-BIH, LTAFDB

> **Status: selesai (6 November 2026). Replikasi sebagian (2 dari 3 dataset).** D-1 lebih baik dari paket v3.1 (Macro-F1 2 kelas) di
> ketiga dataset dan sensitivitasnya lebih tinggi di MIT-BIH dan LTAFDB. Tetapi di **CinC 2017** (EKG genggam satu lead) sensitivitasnya
> lebih rendah (0,830 vs 0,884), seperti di Icentia11k. D-1 tetap kandidat uji perangkat, dengan sensitivitas di perangkat sebagai kriteria
> utama. Laporan: [laporan/laporan_v16_3_replikasi.pdf](laporan/laporan_v16_3_replikasi.pdf).

## Tujuan

v16.2 menguji ambang t = −1 di satu dataset baru (CPSC 2021, Holter) dan lolos. v16.3 mengulang uji yang sama, tanpa penyetelan, di tiga
dataset yang belum pernah dipakai SIHEDAF, untuk menjawab dua kelemahan yang tersisa: sensitivitas di data genggam/wearable dan alarm pada
aritmia non-AF. Rancangan dikunci sebelum sinyal dibaca: [RANCANGAN.md](RANCANGAN.md) (sha256 di `logs/v16.3_praregistrasi.log`).

| Dataset | Isi | Unit bootstrap | Window uji (AF) |
|---|---|---|---|
| CinC 2017 (`challenge-2017`, `REFERENCE-v3.csv`) | 8.249 rekaman genggam AliveCor satu lead, 300 Hz; AF = `A`, non-AF = `N` + `O`, `~` dibuang | rekaman (tanpa ID pasien) | 43.465 (3.898) |
| MIT-BIH Arrhythmia (`mitdb`) | 48 rekaman 30 menit, 360 Hz, kanal 0; ritme pacu `(P` dibuang | 46 subjek (201 = 202) | 14.152 (1.514) |
| LTAFDB (`ltafdb`) | 84 rekaman ±24 jam, 128 Hz, kanal 0 | 84 subjek | 1.272.328 (706.582) |

Window perangkat (50 Hz, detektor R produksi, 10 s, langkah 5 s) yang lolos cek kualitas; label ritme MIT-BIH/LTAFDB konvensi MIT
(label berlaku sampai label berikutnya, sama dengan AFDB). Sistem sama dengan v16.2: **D-1** (AF bila d1 > −1), **paket v3.1**
(referensi), D-0, paket v1.

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Data, label, kriteria per dataset, vonis replikasi (dikunci) |
| `replikasi.py` | Pembaca CinC (.mat) dan MIT, window + label per rekaman, ECGFounder/paket v3.1/v1 per rekaman ke cache (bisa dilanjutkan), evaluasi berpasangan, alarm per ritme, episode |
| `replikasi.ipynb` | Notebook utama; menulis `../cache/v16.3/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v16_3_replikasi.pdf` | Laporan dua tahap |

## Cara menjalankan

```bash
python tools/unduh_physionet.py challenge-2017 1.0.0 --ke cinc2017 --pola "training/*" "REFERENCE-v3.csv"   # MD5 dari ETag S3
python tools/unduh_physionet.py mitdb 1.0.0 --ke mitdb --pola "[0-9]*.dat" "[0-9]*.hea" "[0-9]*.atr" RECORDS ANNOTATORS
python tools/unduh_physionet.py ltafdb 1.0.0 --ke ltafdb --pola "*.dat" "*.hea" "*.atr" RECORDS ANNOTATORS        # 3,6 GB
python tools/run_notebook.py v16.3_replikasi/replikasi.ipynb
python v16.3_replikasi/laporan/build_report.py --extract && python v16.3_replikasi/laporan/build_report.py
```

`challenge-2017` tidak punya `SHA256SUMS.txt`; `tools/unduh_physionet.py` sekarang memakai MD5 dari ETag listing S3 untuk database seperti
itu (objek satu-bagian; `training/MD5SUMS` resmi berisi nilai yang sama). Butuh GPU untuk ECGFounder dan bobot di `data/ecgfounder/`.
Hasil per rekaman di `cache/v16.3/<dataset>/` (LTAFDB ±40 menit dengan dua proses; dua proses aman karena penulisan atomik).

## Hasil

**Per dataset, 2 kelas** (CI 95% bootstrap unit; P = P(Δ > 0) berpasangan vs paket v3.1):

| Dataset | Sistem | Macro-F1 [CI] | Sensitivitas | Spesifisitas | PPV | H1 Macro-F1 | H2 Sensitivitas |
|---|---|---|---|---|---|---|---|
| CinC 2017 | v3.1 paket | 0,757 [0,74–0,77] | 0,884 | 0,886 | 0,433 | | |
| | **D-1** | 0,867 [0,86–0,88] | 0,830 | 0,965 | 0,702 | lolos (P = 1,00) | **gagal** (Δ −0,054, P = 0,000) |
| MIT-BIH | v3.1 paket | 0,754 [0,60–0,84] | 0,916 | 0,857 | 0,435 | | |
| | **D-1** | 0,967 [0,90–0,99] | 0,983 | 0,987 | 0,902 | lolos (P = 1,00) | lolos (lebih baik, P = 0,997) |
| LTAFDB | v3.1 paket | 0,933 [0,91–0,95] | 0,948 | 0,915 | 0,933 | | |
| | **D-1** | 0,967 [0,95–0,98] | 0,974 | 0,960 | 0,968 | lolos (P = 1,00) | lolos (lebih baik, P = 1,00) |

v1 paket lebih buruk dari v3.1 (Macro-F1) di CinC 2017 dan MIT-BIH, tidak terbedakan di LTAFDB. D-0 lebih spesifik tetapi kurang sensitif
dari D-1 di ketiga dataset.

**Alarm pada ritme non-AF** (proporsi window yang dialarmkan AF, v3.1 → D-1):
- CinC 2017: normal 6,9% → 1,5%; ritme lain (`O`) 20,2% → 7,4%; non-AF ≥ 120 bpm 98,2% → 12,1%.
- MIT-BIH: sinus normal 12,7% → 0,9%; bigemini ventrikel 57% → 5%; SVTA 100% → 21% (19 window); takikardia ventrikel 100% → 60% (15
  window); ritme idioventrikel 5% → 52% (21 window).
- LTAFDB: sinus normal 8,5% → 2,8%; bigemini atrium 58% → 9%; SVTA 27% → 10%; **sinus bradikardia 4,2% → 27,9%** (25.501 window).

**Episode AF** (EC57): MIT-BIH episode palsu 37,2 → 4,9 per jam (Se episode 0,95 → 0,99); LTAFDB 12,5 → 5,7 per jam, Se episode 0,97 →
0,90 (yang terlewat episode pendek: Se durasi justru naik 0,95 → 0,97).

**Per unit:** LTAFDB 2 dari 83 subjek ber-AF turun sensitivitasnya > 0,20 (116, 16); MIT-BIH tidak ada; CinC 2017 99 dari 740 rekaman AF
(rekaman pendek, 1–11 window).

**Lintas dataset (D-1 − v3.1, sensitivitas):** CPSC 2021 +0,003 (P = 0,70), MIT-BIH +0,067, LTAFDB +0,025, CinC 2017 −0,054; data yang
sudah dipakai: AFDB +0,002, PTB-XL −0,005, Icentia11k −0,029. Macro-F1 dan spesifisitas D-1 lebih baik di ketujuh dataset.

## Kesimpulan

- Keunggulan D-1 sebagai detektor AF 2 kelas bertahan di tiga dataset baru: jauh lebih sedikit alarm palsu (termasuk pada aritmia lain dan
  detak cepat), Macro-F1 lebih baik, dan pada Holter sensitivitasnya setara atau lebih baik.
- Kelemahannya konsisten dan sekarang jelas: pada **EKG genggam/wearable satu lead** (CinC 2017, Icentia11k) D-1 melewatkan ±3–5 poin
  lebih banyak AF daripada v3.1. Perangkat SIHEDAF wearable, jadi ini risiko utama. Kelemahan baru: sinus bradikardia di LTAFDB.
- Vonis yang dikunci: replikasi sebagian. Ambang tidak diubah. D-1 tetap kandidat uji perangkat, tetapi uji itu harus mengunci sensitivitas
  (bukan hanya Macro-F1) sebagai kriteria utama, dan memeriksa sinus bradikardia.

## Lingkungan

PC desktop, GPU NVIDIA RTX 5060 8 GB, CPU 12 core. Python 3.13.16, numpy 2.5.4, scikit-learn 1.9.1, torch 2.11.0+cu128. Kode: commit
`b18d6bc` + perubahan yang belum di-commit. Hasil per rekaman ±40 menit (dua proses: CinC + MIT-BIH, LTAFDB maju dan mundur); notebook
COMPLETE lewat `tools/run_notebook.py`, 6 November 2026.
