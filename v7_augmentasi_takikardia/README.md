# v7_augmentasi_takikardia — augmentasi sinus takikardia sintetis

Pertanyaan: v3.2 menemukan bahwa semua model menebak ritme teratur ≥ ±115 bpm sebagai AFIB/AFL. Bisakah ini diperbaiki
**tanpa data baru**, dengan menambahkan contoh "Normal cepat" sintetis ke data latih?

Class weighting dan bias tidak menolong (v3.2: alarm palsu pada Normal ≥120 bpm tetap 95% untuk v3.1 dan v5b). SMOTE juga tidak
punya bahan: hanya ada 100 window Normal > 120 bpm, dan semuanya tidak teratur. Jadi v7 membuat contoh baru dari urutan detak
Normal **asli** pasien latih. Waktunya dipadatkan (IBI × s, s < 1) ke detak target, lalu dipotong 10 detik seperti window asli.
Window sintetis harus teratur (CV IBI ≤ 0,10), karena sinus takikardia itu teratur; segmen berdenyut ektopik dibuang.

Alasan eksperimen berbasis interval RR ini tetap dijalankan, meski AGENTS.md melarang resep RR-only baru: tujuannya berbeda
(alarm palsu takikardia, bukan Macro-F1), dan ia menambah pengetahuan klinis yang tidak ada di AFDB (sinus takikardia ada dan
teratur). Disetujui pemilik repo 16 Oktober 2026.

| Varian | Data latih | Hipotesis (ditetapkan sebelum melihat hasil) | Status |
|---|---|---|---|
| 7a. sintetis 100–130 bpm (utama) | + 6.000 window sintetis per fold | Ambang Normal ≥ 125 bpm, Macro-F1 tidak lebih buruk dari v3.1 | **Terbukti, dengan harga AFL** |
| 7b. sintetis 100–160 bpm (eksploratif) | + 6.000 window sintetis per fold | Ambang naik lebih jauh, recall AFL turun (AFL 2:1 di 130–155 bpm) | **Ditolak** (rusak) |
| kontrol | tanpa sintetis (= v5b), ulangan 0 | Uji paritas terhadap OOF v5b | Terbukti (selisih 0,0) |

| File | Isi |
|---|---|
| `augmentasi_takikardia.ipynb` | Bukti urutan detak identik dengan v2, contoh sintetis, training, hasil, uji stres, per pasien, sensitivitas |
| `augmentasi.py` | Ekstraksi urutan detak, pembuat window sintetis, training (fungsi v5 diimpor), evaluasi |
| `laporan/` | `laporan_augmentasi_takikardia.pdf` (6 halaman) + `build_report.py` |

## Hasil

Data uji detektor 50 Hz, bias cross-fit grid −7…+3, Δ berpasangan vs v3.1.

| | Macro-F1 | AFL recall | Normal recall | Ambang uji stres | Sintetis pasien uji 120–130 bpm → Normal | Vonis Macro-F1 |
|---|---|---|---|---|---|---|
| v3.1 (referensi) | 0,832 [0,67–0,92] | 0,592 [0,13–0,91] | 0,924 | 115 bpm (paket) | 0% (paket) | — |
| v5b (ablasi) | 0,830 | 0,595 | 0,929 | 115–120 bpm (kontrol) | 1% (kontrol) | tidak terbedakan |
| **7a. 100–130 bpm** | 0,825 [0,68–0,88] | 0,555 [0,14–0,71] | **0,937** | **135 bpm** | **98%** | tidak terbedakan (P = 0,41) |
| 7b. 100–160 bpm | 0,700 [0,62–0,75] | 0,454 | 0,876 | 85 bpm (rusak) | 50% | lebih buruk (P = 0,00) |

Rincian 7a:
- **Recall AFL:** vs v3.1 tidak terbedakan (P = 0,37), vs v5b lebih buruk (Δ −0,039, P = 0,01). Kerugian terbesar di 08378
  (AFL 2:1 sangat teratur, ±138 bpm): 0,96 → 0,71.
- **AFL palsu dari pasien tanpa AFL:** 25 → 58 per ulangan. Deteksi episode AFL tetap 74%.
- **Window Normal asli ≥ 120 bpm yang memicu alarm:** 95% → 69%.
- **Tanpa 04936:** 0,805 vs 0,859 (tidak terbedakan, P = 0,67).
- **Kriteria kandidat:** Macro-F1 dan ambang lolos; per pasien tidak lolos (08378 turun 0,25).

**Kesimpulan.** Paket produksi tetap v3.1; v7 tidak dikemas.
- Augmentasi memindahkan batas alarm palsu dari 115 ke 135 bpm, tetapi tidak menghilangkan masalahnya.
- Di 135–160 bpm, sinus takikardia dan AFL 2:1 tidak bisa dibedakan dari interval RR. Inilah yang ditunjukkan 7b.
- 7a adalah kandidat bila pemilik produk menerima harganya, yaitu sebagian deteksi AFL 2:1 (08378). Sebelum dikemas sebagai v7.1:
  latih ulang di semua pasien, bootstrap bersarang, dan validasi pada sinus takikardia asli.
- Data eksternal (sinus takikardia asli + bentuk gelombang EKG) tetap dibutuhkan.

## Lingkungan hasil

Dijalankan 16 Oktober 2026 di WSL2 (Debian), PC desktop dengan GPU NVIDIA 8 GB.
- Training: 11 job × 5 fold, ±53–60 s/fold setelah PC di-restart; 4 job pertama lebih lambat (±118 s/fold) karena PC bermasalah.
- Evaluasi: ±12 menit (CPU).

## Menjalankan (dari root repo)

```bash
python tools/run_notebook.py v7_augmentasi_takikardia/augmentasi_takikardia.ipynb   # ±45 menit GPU; job yang sudah ada dilewati
python v7_augmentasi_takikardia/laporan/build_report.py --extract
python v7_augmentasi_takikardia/laporan/build_report.py
```

Prasyarat: `../data/afdb` (ekstraksi ulang urutan detak), `../cache/dataset_v2_detector.npz`, OOF v3/v5 dan `folds.npz` di
`../cache/oof_robust/`, hasil v3.2 (`../cache/v3_2/ringkasan.json`), dan paket `v3.1_kalibrasi_produksi/model_export/`
(hanya dibaca).

Keluaran:
- OOF varian: `../cache/oof_robust/detector_e30_r5_s2026/MicroTransformer_takikardia{130,160}_r*.npz`;
- urutan detak: `../cache/v7/detak_afdb.npz`;
- prediksi uji stres dan uji sintetis pasien uji: `../cache/v7/uji_*.npz`;
- bobot setiap model fold: `../cache/v7/model/*.pt` (berisi `state_dict`, `mu`, `sd`).
