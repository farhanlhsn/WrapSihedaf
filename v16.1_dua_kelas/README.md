# v16.1 — detektor AF 2 kelas dari tahap 1 v16 (ECGFounder tanpa penyetelan)

> **Status: selesai (6 November 2026). Belum kandidat.** Sebagai detektor "AF ya/tidak", ECGFounder hampir menghapus alarm palsu
> (sinus takikardia asli 98,4% → 0,1%, episode AF palsu 11,4 → 1,0 per jam) dan setara v3.1 di AFDB, lebih baik di PTB-XL. Tetapi
> sensitivitasnya lebih rendah di semua data, dan di Icentia11k (wearable) Macro-F1 lebih buruk, sehingga kriteria yang dikunci tidak
> terpenuhi. Laporan: [laporan/laporan_v16_1_dua_kelas.pdf](laporan/laporan_v16_1_dua_kelas.pdf).

> **Tindak lanjut:** ambang t = −1 dikunci dan diuji di CPSC 2021 → [v16.2_cpsc2021](../v16.2_cpsc2021/) (kandidat uji perangkat, 2 kelas).

## Tujuan

Permintaan pemilik repo setelah v16: versi yang keluarannya hanya **Normal vs AF** (AFIB atau AFL digabung). Versi minor: tidak ada
training dan tidak ada parameter yang disetel; keputusan AF diambil dari tahap 1 v16 apa adanya. Rancangan dikunci sebelum metrik baru
dihitung: [RANCANGAN.md](RANCANGAN.md) (sha256 di `logs/v16.1_praregistrasi.log`; satu tambahan setelah hasil di bagian 8).

| Sistem | Keputusan AF |
|---|---|
| **D-0 (primer)** | ECGFounder zero-shot 1-lead: `log((pAF + pAFL) / pN) > 0` |
| **D-L** | D-0, atau v3.1 menyebut AFL ("perlu dicek" v16 dihitung sebagai alarm) |
| v3.1, BiGRU bawaan (v11.1), v1 | bias 3 kelas cross-fit (v1: argmax), lalu AFIB + AFL digabung (aturan v11.1) |

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Sistem, hipotesis H1–H6, kriteria (dikunci), penyimpangan |
| `deteksi_af.py` | Pembanding 2 kelas, tanpa 04936, per pasien, alarm Normal per pita, episode AF, bootstrap pasien 2 kelas untuk data eksternal, kurva ambang, **`DetektorAF`** (fungsi prediksi: EKG 10 s 50 Hz → keputusan), paritas, latensi |
| `deteksi_af.ipynb` | Notebook utama; menulis `../cache/v16.1/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v16_1_dua_kelas.pdf` | Laporan dua tahap |

Kode lama lewat impor: `v16_hierarki_ecgfounder/hierarki.py` (tahap 1, data eksternal), `v13_ecgfounder/ecgfounder.py` (praproses,
model), `v11.1_dua_kelas/dua_kelas.py` (metrik 2 kelas), `v3.3_takikardia_asli/takikardia_asli.py` (alarm per pita, episode).

## Cara menjalankan

```bash
python tools/run_notebook.py v16.1_dua_kelas/deteksi_af.ipynb
python v16.1_dua_kelas/laporan/build_report.py --extract && python v16.1_dua_kelas/laporan/build_report.py
```

Butuh cache v3.3, v10, v11 (`bigru/bawaan`), v11.1, v13, v15, v16 dan `cache/oof_robust/`. Bobot ECGFounder di `data/ecgfounder/`
hanya untuk paritas dan latensi (bagian 6 notebook). Contoh pemakaian fungsi prediksi:

```python
from deteksi_af import DetektorAF
det = DetektorAF("cpu")            # atau "cuda"
d1, af = det(ekg_50hz)             # ekg_50hz: array (N, 500), window 10 s pada 50 Hz (mV)
```

## Hasil

**AFDB 2 kelas** (pooled OOF 5 ulangan, CI 95% bootstrap pasien; berpasangan vs v3.1):

| Sistem | Macro-F1 [CI] | Sensitivitas | Spesifisitas | vs v3.1 (P) |
|---|---|---|---|---|
| v1 (argmax) | 0,922 [0,88–0,95] | 0,954 | 0,903 | 0,195 |
| v3.1 (referensi) | 0,926 [0,88–0,95] | 0,934 | 0,924 | – |
| BiGRU bawaan (v11.1) | 0,938 [0,91–0,96] | 0,926 | 0,950 | 0,996, lebih baik |
| **D-0 (primer)** | 0,951 [0,92–0,98] | 0,893 | **0,995** | 0,92, tidak terbedakan |
| D-L | 0,956 [0,93–0,98] | 0,905 | 0,995 | 0,953, lebih baik |

D-0 − v3.1: sensitivitas −0,040 (P = 0,017, lebih buruk), spesifisitas +0,071 (P = 1,00). Bersarang (v16): P = 0,92. Tanpa 04936:
0,954 vs 0,927 (P = 0,93). D-0 vs BiGRU: P = 0,83. Sensitivitas per pasien turun > 0,20 di 05091 (9 window AF), 05121, 08378.

**Alarm palsu** (butir 9):

| | v3.1 | BiGRU (v11.1) | D-0 | D-L |
|---|---|---|---|---|
| Sinus takikardia asli ≥ 120 bpm (2.189 window, 41 subjek) | 98,4% | 91,0% | **0,1%** | 74,6% |
| Normal AFDB 100–110 bpm / 110–120 bpm | 12,7% / 59,0% | 10,0% / 52,6% | 0,9% / 12,0% | 1,0% / 19,6% |
| Episode AF palsu per jam (AFDB) | 11,4 | 9,3 | **1,0** | 1,1 |

Se episode AF 0,97 → 0,91, +P episode 0,57 → 0,95 (v3.1 → D-0).

**Data eksternal** (bootstrap pasien berpasangan vs paket v3.1):

| Data | Sistem | Macro-F1 sistem / v3.1 | P | Se sistem / v3.1 | Sp sistem / v3.1 |
|---|---|---|---|---|---|
| PTB-XL (20.311) | D-0 | 0,944 / 0,817 | 1,00, lebih baik | 0,863 / 0,917 | 0,995 / 0,935 |
| | D-L | 0,926 / 0,817 | 1,00, lebih baik | 0,879 / 0,917 | 0,988 / 0,935 |
| Icentia11k (115.722) | D-0 | 0,823 / 0,858 | 0,000, **lebih buruk** | 0,767 / 0,871 | 0,987 / 0,882 |
| | D-L | 0,839 / 0,858 | 0,000, lebih buruk | 0,798 / 0,871 | 0,973 / 0,882 |

PTB-XL: AFL terdeteksi 36% (v3.1 92%), STACH dialarmkan 1,6% (v3.1 29,7%).

**Kurva ambang** (deskriptif; AF bila `d1 > t`; Icentia ditambahkan setelah hasil):

| t | Se AFDB | Sp AFDB | Sinus ≥ 120 bpm | Se PTB-XL | Macro-F1 Icentia |
|---|---|---|---|---|---|
| −2 | 0,962 | 0,971 | 2,0% | 0,943 | 0,883 |
| −1 | 0,935 | 0,992 | 0,5% | 0,912 | 0,870 |
| **0 (D-0)** | 0,893 | 0,995 | 0,1% | 0,863 | 0,823 |
| 1 | 0,818 | 0,997 | 0,0% | 0,746 | 0,725 |

Ambang lain terlihat lebih seimbang, tetapi ini dilihat setelah hasil pada data yang sama. Ambang itu harus dikunci dan diuji di data
baru sebelum diklaim.

**Fungsi prediksi:** `DetektorAF` di CPU float32 memberi keputusan yang 100% sama dengan cache v13 (GPU bf16) pada 606 window 08378;
selisih `d1` maksimum 0,50. Latensi praproses + inferensi ±29 ms/window di CPU (6 thread, batch 1), 0,73 ms/window di GPU (RTX 5060,
batch 64).

## Kesimpulan

- ECGFounder sebagai detektor AF 2 kelas menukar sensitivitas dengan spesifisitas: alarm palsu ±10× lebih sedikit, AF yang terlewat
  lebih banyak (terutama AFL). Di AFDB dan PTB-XL hasil bersihnya setara atau lebih baik; di Icentia11k, yang AF-nya banyak, lebih buruk.
- Kriteria yang dikunci tidak terpenuhi (Icentia11k), jadi D-0 belum menjadi kandidat uji perangkat. Paket v3.1 tetap.
- Langkah berikutnya yang jujur: kunci ambang lain (mis. t = −1) lalu uji pada data yang belum dipakai: CPSC 2021 (sudah diunduh di
  v15, sinyalnya belum pernah dinilai) dan rekaman perangkat SIHEDAF berlabel.

## Lingkungan

PC desktop, GPU NVIDIA RTX 5060 8 GB (hanya paritas dan latensi), CPU 12 core (torch 6 thread). Python 3.13.16, numpy 2.5.4, scikit-learn
1.9.1, torch 2.11.0+cu128. Kode: commit `b18d6bc` + perubahan yang belum di-commit. Notebook COMPLETE lewat `tools/run_notebook.py`,
6 November 2026 (±3 menit; run kedua setelah tambahan kurva Icentia).
