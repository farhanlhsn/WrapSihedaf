# v9_gelombang_ekg — cabang bentuk gelombang EKG dengan AFDB + Chapman

Pertanyaan: v7–v8 membuktikan bahwa dari interval RR saja, ritme cepat yang teratur tidak bisa dipisahkan. Yang dimaksud adalah
sinus takikardia, AFL 2:1, dan SVT. Akibatnya model harus memilih salah satu:
- alarm palsu pada sinus takikardia (paket v3.1), atau
- kehilangan AFL 2:1 dan SVT (8c).

Pembedanya ada di bentuk gelombang, yaitu gelombang P sinus versus gelombang flutter. v6 mencoba CNN EKG dengan 23 pasien dan
overfit per pasien. **Apakah arsitektur yang sama, dilatih dengan ±42 ribu pasien, bisa memisahkan ritme-ritme itu?**

**Data latih** (sama untuk semua varian): AFDB fold latih dan Chapman fold latih berlabel andal.
- Label: Normal (SR/SB/ST/SA), AFIB, dan AFL bagian Shaoxing. AFL Ningbo tidak andal (v8).
- Total Chapman berlabel andal: 32.025 Normal, 1.696 AFIB, 416 AFL.
- Bentuk gelombang memakai praproses v6: lead II Chapman atau kanal 0 AFDB, 50 Hz, bandpass 0,5–20 Hz, 500 sampel per window.
  Window ini dibuktikan sejajar dengan window interval RR.

| Varian | Model | Hipotesis (ditetapkan sebelum hasil) | Status |
|---|---|---|---|
| `rr` kontrol | MicroTransformer interval RR (resep v5b) | Mirip 8c: alarm ST rendah, AFL 2:1 & SVT hilang | Terbukti |
| **`dual` 9a (utama)** | CNN EKG + RR (DualBranch v6, 417 ribu parameter) | Dibanding `rr`: recall AFL Shaoxing & alarm SVT naik (P ≥ 0,95), alarm ST tidak naik > 5 poin; 08378 di AFDB kembali ≥ 0,78 | **Sebagian**: Chapman ya; AFDB tidak (08378 → 0,00) |
| `ekg` 9b | CNN EKG saja (ECGOnly v6) | Morfologi saja memisahkan ST dari AFL lebih baik daripada `rr` | Terbukti di Chapman; AFDB lebih buruk |
| `fusi` 9c | rata-rata `rr` + `ekg` | Eksploratif, tanpa training | Di antara rr dan ekg |

**Pembanding** diambil dari cache v8 (lingkungan T4 dan fold Chapman yang sama): v31, afdb, 8c, serta paket v1/v2/v3.1.

**Kriteria kandidat produksi untuk 9a** (ditetapkan sebelum hasil):
1. AFDB: Macro-F1 vs v31 tidak lebih buruk (P > 0,05), dan tidak ada pasien AFL yang turun ≥ 0,20.
2. Chapman: alarm sinus takikardia ≤ separuh paket v3.1, dan recall AFL (Shaoxing) tidak lebih rendah dari paket v3.1.
3. Alarm pada sinus takikardia asli ≥ 120 bpm ≤ 25%.

**Uji stres (protokol butir 9).** Ritme sintetis v3.2 hanya berisi interval RR, jadi uji itu hanya berlaku untuk model `rr`. Untuk
model EKG, uji stresnya memakai 6.937 sinus takikardia **asli** Chapman per pita detak.

| File | Isi |
|---|---|
| `gelombang_ekg.ipynb` | Data (EKG AFDB v6 + EKG Chapman sejajar), contoh bentuk gelombang, training, hasil AFDB & Chapman, uji stres |
| `laporan/` | `laporan_gelombang_ekg.pdf` + `build_report.py` |
| `gelombang.py` | Ekstraksi EKG Chapman (dibuktikan sejajar), training (fungsi v6/v5 diimpor), evaluasi |

## Hasil

**Chapman** (34 ribu pasien uji eksternal, label andal; bias dari AFDB):

| | Macro-F1 | alarm sinus takikardia | … ST 140–150 bpm | alarm SVT (bagus) | AFL recall (Shaoxing) | Normal R |
|---|---|---|---|---|---|---|
| paket v3.1 | 0,520 | 42% | 100% | 99% | 0,243 | 0,865 |
| kontrol rr | 0,594 | 11% | 70% | 39% | 0,136 | 0,951 |
| **9a EKG + RR** | **0,739** | **2%** | **3%** | 56% | 0,225 | **0,993** |
| 9b EKG saja | 0,762 | 2% | 3% | 73% | 0,309 | 0,992 |
| 9c fusi | 0,707 | 3% | 19% | 53% | 0,206 | 0,988 |

**AFDB** (protokol v3, Δ vs v31):

| | Macro-F1 | AFL recall | 08378 | AFL palsu/ulangan | bias AFL min…maks | Vonis |
|---|---|---|---|---|---|---|
| v31 | 0,830 [0,67–0,92] | 0,600 | 0,98 | 293 | −5,25…−3,75 | referensi |
| kontrol rr | 0,783 | 0,436 | 0,35 | 313 | −2,50…−0,25 | tidak terbedakan (P = 0,16) |
| **9a EKG + RR** | 0,687 [0,64–0,77] | 0,404 | **0,00** | **2.878** | **−7…+3** | tidak terbedakan (P = 0,07) |
| 9b EKG saja | 0,651 | 0,452 | 0,03 | 3.809 | −7…+3 | lebih buruk (P = 0,02) |
| 9c fusi | 0,716 | 0,465 | 0,03 | 2.771 | −2,25…+0,50 | tidak terbedakan (P = 0,10) |

**Kesimpulan.** Paket produksi tetap v3.1; v9 tidak dikemas.
- Bentuk gelombang memuat informasi yang hilang dari interval RR: pada pasien Chapman yang tidak ikut dilatih, alarm palsu sinus
  takikardia turun 42% → 2% (juga di 140–150 bpm), Normal R 0,99, dan semua metrik lebih baik dari kontrol rr (P = 1,00).
- Tetapi di AFDB, model bentuk gelombang membunyikan AFL palsu di window non-AFL milik pasien AFL, kehilangan 08378 dan 04043, dan
  bias kelasnya tidak stabil (kedua batas grid). Batas proyek sekarang adalah **pergeseran domain** (Holter vs EKG klinis).
- Langkah berikutnya (v10): validasi dengan rekaman perangkat SIHEDAF berlabel, training yang tahan domain, dan data AFL 2:1 tambahan.

## Lingkungan hasil

Dijalankan 17–18 Oktober 2026 di **Kaggle** (2× Tesla T4, 4 CPU, Python 3.13, torch 2.11), satu sesi berisi:
- data ±7,5 menit;
- uji cepat 21 menit;
- training 4 jam 19 menit (9a ±580 s/fold, 9b ±560 s/fold, rr ±88 s/fold, dua proses paralel);
- evaluasi 65 menit.

Pembanding v8 dipasang dari dataset Kaggle privat `farhanlhsn/sihedaf-v8-cache`. Hasil diekstrak ke `../cache/v9/`.

## Menjalankan

Prasyarat:
- hasil v8 di `../cache/v8/` (window Chapman, OOF v31/afdb/8c) dan fold resmi;
- data mentah AFDB dan Chapman, yang diunduh ulang dari mirror S3 resmi bila belum ada (±5 menit).

Model CNN jauh lebih berat dari MicroTransformer, jadi GPU yang kuat sangat membantu. Perkiraan waktu training 75 fold:
- terukur di Kaggle 2× T4: training 4 jam 19 menit, ditambah evaluasi 65 menit;
- 1× T4: ±8,5 jam;
- A100 (Colab Pro): kemungkinan ±2–3 jam.

**Colab** (cache v8 ada di Google Drive):
- setup sama dengan run v8: clone repo, symlink `cache` → `MyDrive/sihedaf_v8_colab/cache`, lalu `tools/run_notebook.py` di
  background;
- uji alur cepat dulu dengan `SIHEDAF_QUICK=1` (cache terpisah `cache/v9_quick`).

**WSL / Linux:**

```bash
SIHEDAF_QUICK=1 python tools/run_notebook.py v9_gelombang_ekg/gelombang_ekg.ipynb   # uji alur (±15 menit)
python tools/run_notebook.py v9_gelombang_ekg/gelombang_ekg.ipynb                   # run penuh
```
