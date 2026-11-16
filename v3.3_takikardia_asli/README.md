# v3.3 — sinus takikardia asli, aturan produk, abstain, dan metrik episode

> **Status: selesai (6 November 2026). Evaluasi saja; paket produksi tidak disentuh.** Paket v3.1 menyebut AFIB/AFL pada 98,4%
> window sinus takikardia asli ≥ 120 bpm. Laporan: [laporan/laporan_v3_3_takikardia_asli.pdf](laporan/laporan_v3_3_takikardia_asli.pdf).

## Tujuan

Menindaklanjuti riset 6 November 2026 dan v3.2 (yang hanya punya ritme sintetis):
1. alarm palsu pada **EKG latihan fisik asli** lewat pipeline perangkat (50 Hz) dan lewat R-peak manual;
2. aturan produk v3.2 ("≥ 110 bpm & CV IBI ≤ 0,05 → takikardia teratur, perlu dicek") di data asli;
3. keluaran abstain dengan conformal prediction;
4. metrik episode gaya ANSI/AAMI EC57 (dipakai v13–v15).

Versi minor dari v3: tidak ada eksperimen training baru, hanya evaluasi model yang sudah ada (paket v1/v2/v3.1 dan model fold
v7, v8, v11 yang menyimpan bobot).

## Berkas

| Berkas | Isi |
|---|---|
| `takikardia_asli.py` | Pembaca EPFL, Wrist, EPHNOGRAM; window perangkat dan manual (dibuktikan identik dengan `hrv_core.signal_to_windows`); prediksi semua model; alarm per pita detak dengan CI subjek; aturan produk; conformal; metrik episode |
| `takikardia_asli.ipynb` | Notebook utama; menulis `../cache/v3_3/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v3_3_takikardia_asli.pdf` | Laporan dua tahap |

## Data (diunduh 6 November 2026, dicek checksum resmi)

| Sumber | Lisensi | Isi | Cara unduh |
|---|---|---|---|
| EPFL high-intensity (Zenodo 5727800) | CC BY 4.0 | 20 subjek, 99 segmen 20 s saat uji bersepeda maksimal, 250 Hz, R-peak manual direview dokter | `python tools/unduh_berkas.py <url> --ke epfl_hi --md5 1aa4af41…` (lalu unzip ke `data/epfl_hi/isi/`) |
| PhysioNet `wrist` 1.0.0 | ODC-BY | 8 subjek, EKG dada 256 Hz saat jalan, lari, bersepeda, R-peak referensi | `python tools/unduh_physionet.py wrist 1.0.0 --ke wrist` |
| PhysioNet `ephnogram` 1.0.0 | ODC-BY | 24 subjek, 69 rekaman EKG 8 kHz (istirahat, berjalan, treadmill Bruce, bersepeda) | `python tools/unduh_physionet.py ephnogram 1.0.0 --ke ephnogram` (6,4 GB) |

Subjek sehat tanpa label ritme; diasumsikan sinus.

## Cara menjalankan

```bash
python tools/run_notebook.py v3.3_takikardia_asli/takikardia_asli.ipynb
python v3.3_takikardia_asli/laporan/build_report.py --extract
python v3.3_takikardia_asli/laporan/build_report.py
```

Butuh `cache/oof_robust/`, bobot model fold di `cache/v7/model/`, `cache/v8/model/`, `cache/v11/<model>/<setelan>/` dan GPU (atau CPU,
lebih lambat). Prediksi disimpan di `cache/v3_3/prediksi.npz`.

## Hasil

**Alarm AFIB/AFL pada sinus ≥ 120 bpm** (mode perangkat, ketiga sumber, 2.189 window, 41 subjek; CI dan P dari bootstrap subjek
berpasangan):

| Model | Alarm [CI 95%] | Δ vs v3.1 |
|---|---|---|
| v1 paket | 98,0% [96–100] | −0,4 poin (P = 0,89) |
| v2 paket | 99,9% [100–100] | +1,5 poin |
| **v3.1 paket (produksi)** | **98,4% [97–100]** | – |
| v7a: + sinus takikardia sintetis 100–130 bpm | 63,9% [58–70] | −34,5 poin |
| v7b: + sinus takikardia sintetis 100–160 bpm | 70,9% [66–75] | −27,5 poin |
| v8a: AFDB + Chapman (termasuk sinus takikardia asli) | 63,8% [57–71] | −34,6 poin |
| v8c: AFDB + Chapman tanpa label AFL | 60,3% [53–68] | −38,1 poin |
| v11 BiGRU bawaan / bobot penuh / TCN bobot penuh | 91,0% / 96,0% / 88,9% | −7,4 / −2,4 / −9,5 poin |

Per sumber, v3.1: EPFL (bersepeda maksimal) 100%, Wrist 100%, EPHNOGRAM 98%. v7a sangat baik sampai 130 bpm (Wrist 120–130 bpm:
8%), lalu kembali ±100% di atas rentang latihnya.

**Detektor R-peak 50 Hz saat latihan:** EPFL hanya 68% window lolos cek kualitas (R-peak manual 95%); CV IBI median 0,028 vs 0,009.
Akibatnya v3.1 juga mengalarmkan window < 100 bpm saat beraktivitas: 21% (EPHNOGRAM), 29% (Wrist perangkat) vs 9% (Wrist, R-peak
manual).

**Aturan produk** menangkap 73% (EPFL), 83% (Wrist), 61% (EPHNOGRAM) alarm ≥ 120 bpm; sisanya lolos karena jitter detektor menaikkan
CV. Biaya di AFDB: 86% AFL yang terdeteksi ikut dialihkan (v3.2).

**Abstain conformal** (LAC, cakupan 90%, ambang ±0,33): 0% "tidak yakin" pada sinus takikardia, karena model yakin dalam
kesalahannya. Bila himpunan kosong juga dianggap ragu: 12,9% menjadi "tidak yakin" (alarm 86%), dengan biaya 3,9% Normal dan 9,7% AFL
di AFDB ikut ragu.

**Metrik episode v3.1 di AFDB (gaya EC57):** AFL Se episode 0,686, +P 0,403, 0,52 episode palsu/jam (14 episode referensi); AFIB
Se 0,970, +P 0,562, 11,9 episode palsu/jam.

## Kesimpulan

- Temuan v3.2 dari data sintetis terkonfirmasi di EKG asli: model produksi menyebut sinus takikardia sebagai AFIB/AFL hampir selalu.
- Hanya data latih yang memuat sinus takikardia (v7 sintetis, v8 Chapman asli) yang menurunkan alarm, dengan trade-off AFL 2:1 di
  AFDB (lihat laporan v7 dan v8). Abstain conformal tidak berguna untuk masalah ini; aturan produk menolong tetapi mengorbankan AFL 2:1.
- Detektor R-peak 50 Hz menjadi sumber kesalahan tambahan saat aktivitas (alarm AF pada detak normal). Ini perlu diuji di perangkat
  asli, dan detektor yang lebih tahan gerak (mis. NeuroKit2/UNSW, riset 6 November 2026) layak dibandingkan.

## Lingkungan

PC desktop, GPU NVIDIA RTX 5060 8 GB, CPU 12 core. Python 3.13, torch 2.11.0+cu128, numpy 2.5.4, scikit-learn 1.9.1. Kode: commit
`b18d6bc` + perubahan yang belum di-commit. Notebook COMPLETE lewat `tools/run_notebook.py`, 6 November 2026 (±9 menit dengan cache;
prediksi semua model ±19 menit, termasuk membaca 69 rekaman EPHNOGRAM 8 kHz).
