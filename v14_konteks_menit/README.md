# v14 — konteks beberapa menit di atas keluaran per window

> **Status: selesai (6 November 2026). Keempat hipotesis yang dikunci tidak lolos; bukan kandidat.** Satu-satunya perbaikan:
> episode AFIB palsu turun ±4×. Laporan: [laporan/laporan_v14_konteks_menit.pdf](laporan/laporan_v14_konteks_menit.pdf).

## Tujuan

Riset 6 November 2026 (S4ECG, arXiv 2510.17406; monitor implan) menunjukkan konteks 10–20 menit menurunkan positif palsu AF dan
membantu AFL. v14 menguji apakah **dinamika detak beberapa menit** (cara mulai, kedataran, lama keadaan cepat-teratur) di atas
keluaran v3.1 per window bisa memisahkan sinus takikardia (naik bertahap) dari AFL 2:1 (mendadak, datar ±150 bpm), tanpa merusak
deteksi AFL. Rancangan dan hipotesis dikunci sebelum training: [RANCANGAN.md](RANCANGAN.md) (sha256 di `logs/v14_praregistrasi.log`).

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Fitur, model, hipotesis H1–H4 dan kriterianya (dikunci) |
| `konteks.py` | Fitur kausal per rekaman, XGBoost/regresi logistik per fold, model penuh, uji EPHNOGRAM (H3), uji stres konteks sintetis (H4) |
| `konteks_menit.ipynb` | Notebook utama; menulis `../cache/v14/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v14_konteks_menit.pdf` | Laporan dua tahap |

Metrik episode memakai `v3.3_takikardia_asli/takikardia_asli.metrik_episode` (gaya EC57); evaluasi protokol memakai
`v11_model_tim/bersama/evaluasi.py`.

## Cara menjalankan

```bash
python tools/unduh_physionet.py ephnogram 1.0.0 --ke ephnogram     # 6,4 GB, untuk H3
python tools/run_notebook.py v14_konteks_menit/konteks_menit.ipynb
python v14_konteks_menit/laporan/build_report.py --extract
python v14_konteks_menit/laporan/build_report.py
```

Butuh `cache/oof_robust/` (OOF v3 dan fold) dan `cache/dataset_v2_detector.npz`. Langkah berat disimpan di `cache/v14/`.

## Hasil (AFDB, pooled OOF 5 ulangan, CI 95% bootstrap pasien, bias cross-fit)

| Sistem | Macro-F1 [CI] | AFL F1 [CI] | AFL R | AFL P | Normal R |
|---|---|---|---|---|---|
| v3.1 (referensi) | 0,832 [0,67–0,92] | 0,647 [0,17–0,89] | 0,592 | 0,715 | 0,924 |
| **v14 XGBoost** | 0,781 [0,64–0,87] | 0,473 [0,05–0,73] | 0,477 | 0,498 | 0,949 |
| v14 regresi logistik | 0,736 [0,65–0,82] | 0,338 [0,11–0,57] | 0,282 | 0,438 | 0,946 |

| Hipotesis (dikunci) | Hasil |
|---|---|
| H1: Macro-F1 tidak lebih buruk dari v3.1 | **tidak lolos**: Δ −0,051, P = 0,033 (lebih buruk) |
| H2: Se episode AFL turun ≤ 0,10 dan AFL palsu/jam turun | **tidak lolos**: 0,686 → 0,443; 0,52 → 0,76 per jam |
| H3: alarm sinus takikardia asli (EPHNOGRAM ≥ 120 bpm) turun ≥ separuh | **tidak lolos**: 98,3% → 95,8% (3.944 window, 17 subjek) |
| H4: uji stres konteks: alarm sinus datar turun ≥ separuh, AFL ≥ 0,8× v3.1 | **tidak lolos**: sinus ±100% (kecuali 120 bpm jitter 20 ms: 59%), AFL 99% |

Episode (gaya EC57): AFIB palsu 11,9 → 3,1 per jam, +P episode AFIB 0,56 → 0,71. AFL palsu 274 → 936 window per ulangan (dari
Normal 55 → 710). Recall AFL per pasien (XGBoost / v3.1): 04043 0,00/0,40 · 04908 0,21/0,92 · 04936 0,39/0,51 · 06426 0,09/0,17 ·
07910 0,21/0,00 · 08215 0,00/0,00 · 08378 0,83/0,96.

## Kesimpulan

- Tahap kedua yang dilatih hanya dengan AFDB tidak bisa belajar bahwa detak yang naik bertahap adalah sinus, karena AFDB tidak
  punya episode sinus takikardia. Dengan 7 pasien AFL, fitur konteks juga menghafal pola per pasien.
- Konteks berguna untuk AF yang terpecah-pecah (episode AFIB palsu turun ±4×); layak dibawa ke logika alarm AF saat uji perangkat.
- Langkah berikutnya: latih ulang dengan data yang memuat episode sinus takikardia berlabel (idealnya rekaman perangkat; atau
  EPHNOGRAM + Wrist sebagai negatif latih dengan uji subjek terpisah), dengan rancangan baru yang dikunci dulu.

## Lingkungan

PC desktop, GPU NVIDIA RTX 5060 8 GB (hanya untuk paket v3.1 pada data eksternal), CPU 12 core. Python 3.13, xgboost 3.4.1,
scikit-learn 1.9.1, numpy 2.5.4, torch 2.11.0+cu128. Kode: commit `b18d6bc` + perubahan yang belum di-commit. Notebook COMPLETE
lewat `tools/run_notebook.py`, 6 November 2026 (±7 menit dengan cache; OOF XGBoost ±1,5 menit, H3/H4 ±7 menit).
