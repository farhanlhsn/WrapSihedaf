# v13 — model fondasi EKG ECGFounder 1-lead pada window SIHEDAF

> **Status: selesai (6 November 2026).** ECGFounder zero-shot menurunkan alarm sinus takikardia asli dari 98,4% (v3.1) ke 5,4% dan
> lebih baik dari v3.1 di PTB-XL dan Icentia11k, tetapi kehilangan AFL 2:1 (08378) dan tidak mengalahkan v3.1 di AFDB 3 kelas.
> Bukan pengganti paket. Laporan: [laporan/laporan_v13_ecgfounder.pdf](laporan/laporan_v13_ecgfounder.pdf).

## Tujuan

Riset 6 November 2026: ECGFounder (PKUDigitalHealth, MIT; Net1D 1-lead 30,8 juta parameter (dihitung dari checkpoint; RANCANGAN yang dikunci menulis 76 juta, keliru); 10,8 juta EKG dari 1,8 juta pasien; head 150
label termasuk SINUS TACHYCARDIA dan ATRIAL FLUTTER) terbaik di benchmark FOUND-AF untuk AF di AFDB. Pertanyaan: apakah pengetahuan
pralatih ini menembus batas model RR (alarm sinus takikardia, AFL), dan berapa yang hilang karena perangkat hanya 50 Hz? Rancangan
dikunci sebelum metrik dihitung: [RANCANGAN.md](RANCANGAN.md) (sha256 di `logs/v13_praregistrasi.log`).

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Jalur sinyal, tiga cara pakai, hipotesis H1–H4 (dikunci) |
| `ecgfounder.py` | Praproses (filter ECGFounder), muat model (SHA256 + weights_only), inferensi AFDB per rekaman (bisa dilanjutkan), zero-shot, probe per fold, fusi, PTB-XL, data latihan fisik, Icentia |
| `ecgfounder.ipynb` | Notebook utama (GPU); menulis `../cache/v13/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v13_ecgfounder.pdf` | Laporan dua tahap |

Bobot dan kode ECGFounder **tidak** di-commit (`data/ecgfounder/`).

## Cara menjalankan

```bash
python tools/unduh_berkas.py https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/1_lead_ECGFounder.pth \
    --ke ecgfounder --sha256 f863a38897fb49a27fec7e44008ea3c7bdbd29c77fa4a02ecbb8c56df4f37603
# kode resmi (MIT, cabang master) ke data/ecgfounder/kode/: net1d.py, tasks.txt, LICENSE
python tools/run_notebook.py v13_ecgfounder/ecgfounder.ipynb
python v13_ecgfounder/laporan/build_report.py --extract && python v13_ecgfounder/laporan/build_report.py
```

Butuh GPU (±2.000 window/detik di RTX 5060), PTB-XL (`tools/unduh_physionet.py ptb-xl 1.0.3 …`), cache v3.3 (window latihan fisik)
dan v10 (window Icentia, CC BY-NC-SA, akademik saja). Checkpoint memuat skalar numpy; dimuat dengan `weights_only=True` dan hanya
tipe numpy yang diizinkan.

## Hasil

**AFDB (pooled OOF 5 ulangan, CI 95% bootstrap pasien, bias cross-fit):**

| Sistem | Macro-F1 [CI] | AFL F1 | AFL R | Normal R | vs v3.1 (P) |
|---|---|---|---|---|---|
| v1 (argmax) | 0,744 [0,64–0,81] | 0,399 | 0,728 | 0,903 | – |
| v3.1 (referensi) | 0,832 [0,67–0,92] | 0,647 | 0,592 | 0,924 | – |
| zero-shot 50 Hz | 0,663 [0,64–0,68] | 0,087 | 0,256 | 0,973 | 0,026, lebih buruk |
| probe 50 Hz | 0,746 [0,62–0,81] | 0,379 | 0,366 | 0,944 | 0,087 |
| fusi probe + v3.1, 50 Hz | 0,775 [0,66–0,82] | 0,423 | 0,468 | 0,973 | 0,187 |
| zero-shot / probe / fusi 250 Hz (kontrol) | 0,649 / 0,661 / 0,750 | – | – | – | 0,021 / 0,007 / 0,052 |

Recall AFL 08378 (AFL 2:1): v3.1 0,96, zero-shot 0,01, probe 0,00, fusi 0,10. Zero-shot menyebut banyak AFIB sebagai AFL (AFL palsu
5.320 window/ulangan) dan biasnya menyentuh batas grid di 42% fold. Tanpa 04936: v3.1 0,859, probe 0,629, fusi 0,676. Biaya 50 Hz
tidak terlihat (50 − 250 Hz: zero-shot +0,015, probe +0,086, fusi +0,025).

**Sinus takikardia asli (EKG latihan fisik v3.3, mode perangkat, ≥ 120 bpm, 2.189 window):**

| Sistem | Alarm AFIB/AFL [CI subjek] | EPFL | EPHNOGRAM | Wrist |
|---|---|---|---|---|
| v3.1 paket | 98,4% [97–100] | 100% | 98% | 100% |
| **ECGFounder zero-shot** | **5,4% [3–9]** | 5% | 6% | 0% |
| probe | 30,7% [23–40] | 10% | 33% | 2% |
| fusi | 62,0% [52–71] | 53% | 64% | 8% |

**Data eksternal berlabel (model eksternal dilatih di seluruh AFDB; bias = median cross-fit):**

| Data | Sistem | Macro-F1 model / v3.1 | P | Recall AFL model / v3.1 |
|---|---|---|---|---|
| PTB-XL (20.311 window) | zero-shot | 0,642 / 0,622 | 0,954 | 0,62 / 0,50 |
| | probe / fusi | 0,559 / 0,658 | 0,004 / 0,950 | 0,08 / 0,24 |
| Icentia11k (115.722 window) | zero-shot | 0,549 / 0,476 | 1,000 | 0,14 / 0,05 |
| | probe / fusi | 0,445 / 0,481 | 0,000 / 0,970 | 0,02 / 0,05 |

PTB-XL sinus takikardia (STACH, 811): alarm v3.1 29,7%, zero-shot 3,5%, probe 7,0%, fusi 5,9%.

## Kesimpulan

- Pengetahuan pralatih adalah informasi baru pertama yang menurunkan alarm sinus takikardia secara besar tanpa melatih dengan data
  takikardia, juga di 50 Hz, dan lebih tahan antar-dataset daripada model RR.
- Harganya: AFL 2:1 di ±150 bpm, yang mirip sinus takikardia, ikut disebut sinus (AFDB 08378), dan zero-shot mencampur AFIB/AFL di
  AFDB. Probe yang dilatih di 23 pasien AFDB kehilangan keunggulan eksternal (menghafal AFDB, pelajaran v6/v9).
- Bukan pengganti paket v3.1 untuk 3 kelas. Bila produk cukup 2 kelas atau AFL 2:1 ditangani terpisah, ECGFounder zero-shot di backend
  layak diuji pada rekaman perangkat. Usulan rancangan berikutnya (dikunci dulu): ECGFounder memutuskan sinus vs AF, model RR atau
  fitur atrium (v15) hanya untuk AFIB vs AFL.

## Lingkungan

PC desktop, GPU NVIDIA RTX 5060 8 GB, CPU 12 core. Python 3.13, torch 2.11.0+cu128 (bf16 autocast untuk inferensi), scikit-learn
1.9.1, numpy 2.5.4, scipy 1.18.1. Kode: commit `b18d6bc` + perubahan yang belum di-commit. Notebook COMPLETE lewat
`tools/run_notebook.py`, 6 November 2026 (±20 menit dengan cache inferensi; inferensi AFDB dua jalur ±8 menit; probe 25 fold × 2 jalur
±35 menit saat CPU dibagi dengan job lain). Catatan: `cache/v13/latihan_50hz.npz` adalah cache dari run pertama yang salah memasukkan
window tak lolos cek; tidak dipakai (diganti `latihan_ok_50hz.npz`) dan boleh dihapus.
