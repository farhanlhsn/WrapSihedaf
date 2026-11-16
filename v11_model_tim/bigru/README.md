# bigru — BiGRU + Attention (pemilik repo)

**Model.** GRU dua arah membaca urutan IBI + ΔIBI (maks. 30 detak), lalu *attention pooling* bermask memilih detak yang
penting. 7.172 parameter (docstring di v2 menulis "~15 ribu", itu keliru). Arsitekturnya **diimpor** dari
`v2_revisi/sihedaf_models.py` (`BiGRUAttention`), tidak
disalin. Detak padding diabaikan lewat `pack_padded_sequence`.

**Beda dengan BiGRU di v3** (Macro-F1 + bias 0,829, setara v3.1):

| | v3 | v11 |
|---|---|---|
| Bobot kelas di loss | penuh (AFL ±48×) | lunak, akar (AFL ±6,9×), aturan AGENTS.md |
| Bobot model fold | tidak disimpan | disimpan → bisa diuji takikardia |
| Perangkat | CPU | GPU bila ada |

**Setelan bawaan** (`model.py`): `hidden` 32, resep v3 (30 epoch, batch 256, lr 1e-3, weight decay 1e-4), `weight_power` 0,5.

**Waktu.** GPU RTX 5060 (8 GB): 83,8 s per fold, 35 menit untuk 25 fold, ditambah evaluasi ±3 menit. Sebagai pembanding, BiGRU v3
butuh ±430 s per fold di CPU. Dengan `{"balancing": "under"}`: 1,8 s per fold. GPU hanya terpakai ±30% (bottleneck-nya overhead
Python per batch), jadi dua proses di satu GPU hampir menggandakan kecepatan (89–91 s per fold masing-masing). Kaggle GPU T4 ×2
dengan tiga proses: ±175 s per fold.

**Ringkasan tiga eksperimen** (detail di bawah, PDF di `laporan/`):

| Eksperimen | Isi | Jawaban |
|---|---|---|
| 1 ("Hasil", 23 Okt) | bobot lunak vs undersampling | bawaan setara v3.1; undersampling jauh lebih buruk |
| 2 ("Hasil 2", 2 Nov) | 8 setelan: bobot kelas, kapasitas, lr, epoch, oversampling, ulangan | bobot penuh (`weight_power` 1,0) paling tahan uji; balancing lebih buruk |
| 3 ("Hasil 3", 2 Nov) | 6 regularisasi di atas bobot penuh | tidak ada yang lebih baik; tuning selesai |

**Kandidat BiGRU: `{"weight_power": 1.0}`** (Macro-F1 0,830, setara v3.1; tanpa 04936 0,852). Alasan menyimpang dari bobot lunak
bawaan (AGENTS.md): AFL 2:1 di 08378 dan sensitivitas tanpa 04936. `model.py` sengaja tidak diubah, supaya folder
`cache/v11/bigru/bawaan/` tetap sesuai `setelan.json`-nya.

**File eksperimen.** `eksperimen.py` berisi definisi ketiga eksperimen (`e1`–`e3`), training lewat `bersama.runner`
(`python eksperimen.py latih e3 [setelan]`, bisa dilanjutkan), dan evaluasi lewat `bersama.evaluasi`
(`python eksperimen.py evaluasi e3`). `laporan/build_report.py` membuat satu PDF per eksperimen (`--extract eN`, lalu tanpa argumen).
`presentasi/build_presentasi.py` membuat `presentasi/presentasi_bigru.html`, ringkasan ketiga eksperimen dalam bahasa sederhana untuk
pembimbing. Angkanya dibaca dari `laporan/_build/e1–e3.json`, jadi jalankan `--extract` dulu bila hasil berubah. Halamannya bisa dibuka
tanpa internet.
`ekspor_sampling.py` menulis CSV hasil sampling untuk dibagikan (`cache/v11/bigru/sampling_csv.zip`, ±17 MB): semua window + fold,
indeks undersampling/oversampling per (ulangan, fold), jumlah per kelas, dan ringkasan hasil. Indeksnya diambil dari
`bersama.balancing.terapkan` dengan seed latih protokol, lalu di-assert sama dengan data latih training dan `n_latih` di cache.
Penjelasan kolom dan `contoh_pakai.py` ada di `sampling_csv/`.

## Hasil (23 Oktober 2026)

Dua setelan, masing-masing 5 ulangan × 5 fold pasien (protokol v3), seed latih `2026 + 100·r + k`. Hasilnya pooled OOF rata-rata
ulangan, dengan CI 95% bootstrap pasien 2.000×. `+ bias` = bias kelas cross-fit (grid −7…+3). Kontrol **bobot penuh** = OOF BiGRU v3
(`cache/oof_robust/.../BiGRUAttention_r*.npz`): arsitektur, fold, seed, dan resep sama, hanya bobot kelas penuh. Bobotnya tidak
tersimpan, jadi tidak bisa diuji takikardia.

| Setelan | `SETELAN` | Window latih per kelas (fold r0 k0) |
|---|---|---|
| bawaan | `{}` (bobot lunak, `weight_power` 0,5) | 82.012 / 46.765 / 1.143 |
| under | `{"balancing": "under"}` | 1.143 × 3 (458–1.143 antar fold) |

**Metrik (+ bias)**

| Sistem | Macro-F1 | AFL F1 | AFL P | AFL R | AFIB F1 | Normal R |
|---|---|---|---|---|---|---|
| v1 (argmax) | 0,744 [0,64–0,81] | 0,399 [0,09–0,59] | 0,275 | 0,728 [0,37–0,98] | 0,898 | 0,903 |
| v3.1 (referensi) | **0,832** [0,67–0,92] | **0,647** [0,17–0,89] | 0,715 | 0,592 [0,13–0,91] | 0,909 | 0,924 |
| BiGRU v3, bobot penuh | 0,829 [0,68–0,92] | 0,621 [0,17–0,90] | 0,692 | 0,564 [0,15–0,92] | 0,921 | 0,934 |
| **BiGRU v11 bawaan** | 0,821 [0,68–0,87] | 0,592 [0,16–0,73] | 0,652 | 0,544 [0,16–0,75] | 0,922 | **0,950** [0,90–0,98] |
| BiGRU v11 under | 0,599 [0,49–0,66] | 0,302 [0,01–0,49] | 0,240 | 0,412 [0,02–0,64] | 0,702 | 0,766 |

Tanpa bias (argmax), Macro-F1: bawaan 0,795, under 0,579. Bias AFL bawaan berkisar −4 … −1,25 (median −2), dan tidak ada nilai
yang menyentuh batas grid.

**Perbandingan berpasangan** (Δ Macro-F1 [CI 95%], P(Δ > 0); vonis P ≥ 0,95 lebih baik, ≤ 0,05 lebih buruk)

| Kandidat vs referensi | Δ Macro-F1 | P | Vonis | Catatan |
|---|---|---|---|---|
| bawaan vs v3.1 | −0,010 [−0,06…+0,02] | 0,34 | tidak terbedakan | Normal R +0,026 (P = 1,00, lebih baik); AFL F1 −0,055 (P = 0,11) |
| bawaan vs BiGRU v3 bobot penuh | −0,008 [−0,06…+0,02] | 0,36 | tidak terbedakan | Normal R +0,016 (P = 1,00); AFL P −0,040 (P = 0,14) |
| bawaan vs v1 | +0,078 [+0,02…+0,13] | 0,99 | lebih baik | bootstrap bersarang: P = 0,91 → tidak terbedakan |
| under vs bawaan | −0,223 [−0,33…−0,14] | 0,00 | lebih buruk | AFL P −0,41, Normal R −0,18 |
| under vs v3.1 / v1 | −0,233 / −0,145 | 0,00 | lebih buruk | |

**Bootstrap bersarang** (protokol butir 10, bias dipilih ulang di setiap sampel): bawaan 0,821 [0,67–0,87]. Selisih dengan v1
+0,078 [−0,05…+0,13], P = 0,91: tidak terbedakan. Sebagai perbandingan, v3.1 vs v1 dengan cara yang sama: P = 0,978 (v3.2).

**Recall AFL per pasien (+ bias) dan AFL palsu**

| Sistem | 04043 | 04908 | 04936 | 06426 | 07910 | 08215 | 08378 | AFL palsu / ulangan (di pasien tanpa AFL) |
|---|---|---|---|---|---|---|---|---|
| v3.1 | 0,40 | 0,92 | 0,51 | 0,17 | 0,00 | 0,00 | **0,96** | 274 (25) |
| BiGRU v3 bobot penuh | 0,60 | 0,87 | 0,46 | 0,28 | 0,01 | 0,00 | **0,97** | 291 (29) |
| bawaan | 0,80 | 0,98 | 0,52 | 0,29 | 0,00 | 0,00 | 0,74 | 332 (44) |
| under | 0,00 | 0,99 | 0,62 | 0,16 | 0,00 | 0,00 | 0,03 | 1.937 (1.188) |

**Uji stres takikardia (protokol butir 9).** Ambang bpm di mana model berhenti menebak Normal, per jenis variasi detak
(tanpa variasi / seragam ±10 ms / Gauss 10 ms / Gauss 20 ms). Pembanding: kontrol v7 per fold 120/120/120/115, paket v3.1 115.
Ditambah % window Normal asli yang ditebak AFIB/AFL per pita detak (OOF).

| Sistem | Ambang (bpm) | 0–100 bpm | 100–110 bpm | 110–120 bpm | ≥ 120 bpm (100 window) |
|---|---|---|---|---|---|
| v3.1 | 115 (paket) | 6,7% | 12,7% | 59,0% | 95,2% |
| bawaan | 120 / 120 / 120 / 120 | 4,1% | 10,0% | 52,6% | 86,6% |
| under | 110 / 110 / 110 / 110 | 18,2% | 71,6% | 98,4% | 100,0% |

**Tanpa pasien 04936** (protokol butir 10, bias dihitung ulang di subset, 22 pasien, 6 pasien AFL), Macro-F1: v3.1 0,859
[0,62–0,92], BiGRU v3 bobot penuh 0,860 [0,63–0,92], **bawaan 0,721** [0,63–0,81], under 0,533 [0,48–0,60]. v3.1 − bawaan =
+0,138 [−0,02…+0,21], P = 0,68: tidak terbedakan karena CI lebar, tetapi turunnya bawaan besar. Penyebab yang paling mungkin adalah
08378: tanpa 04936, pasien ini menjadi bagian terbesar AFL yang tersisa, dan recall-nya 0,74 (v3.1 0,96).

**Kesimpulan**
- **BiGRU bobot lunak setara dengan v3.1 dan dengan BiGRU v3 bobot penuh** (P = 0,34 dan 0,36). Normal recall-nya lebih tinggi
  (0,950, P = 1,00), dan alarm pada ritme normal lebih sedikit di semua pita detak. Ambang takikardianya 120 bpm, sedikit di atas
  paket v3.1 (115).
- **Trade-off bobot lunak:** deteksi AFL 2:1 (08378) turun dari 0,97 ke 0,74. Karena itu hasilnya bergantung pada 04936: tanpa
  pasien ini Macro-F1 turun ke 0,721, sedangkan v3.1 dan BiGRU v3 tetap ±0,86.
- **Undersampling merusak BiGRU** (−0,223, P = 0,00). Dengan ±3 ribu window, model menebak AFIB/AFL di 72% ritme normal
  100–110 bpm, dan AFL palsu naik ±6×. Ini kebalikan dari XGBoost (`../xgboost/README.md`), di mana undersampling membantu.
  Jadi efek balancing **bergantung pada model**, dan tidak bisa dijadikan aturan umum.
- **Yang tidak bisa disimpulkan.** Belum ada tuning, dan satu kali training per (ulangan, fold) di GPU tidak sepenuhnya
  deterministik. BiGRU v3 dilatih di CPU, di mesin lain. Pembandingnya tetap sah karena memakai OOF dan fold yang sama, tetapi
  sebagian kecil selisih bisa berasal dari variasi training.
  *Koreksi 2 November 2026:* di mesin yang sama, training ini ternyata deterministik (setelan `bawaan_ulang` identik bit demi bit
  dengan `bawaan`), dan resep yang sama di CPU v3 dan GPU v11 memberi Macro-F1 yang sama (Δ +0,000). Lihat "Hasil 2".

**Lingkungan.** PC desktop, GPU NVIDIA GeForce RTX 5060 8 GB (compute capability 12.0), CPU 12 core. Venv sementara: Python 3.13,
torch 2.11.0+cu128, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, imbalanced-learn 0.14.2. Kode: commit `b18d6bc` + perubahan v11
yang belum di-commit. Seluruh tes v11 lolos di lingkungan ini (72 lolos, 2 di-skip MiniRocket), termasuk uji paritas `latih_dl`
di CUDA. Dijalankan lewat `bersama.runner` (sama dengan sel training notebook), bukan lewat `jalankan.ipynb` (notebook itu
dijalankan 3 November 2026 untuk evaluasi saja; angkanya identik). Hasil dan bobot per
fold di `cache/v11/bigru/{bawaan,under}/`; perbandingan di `cache/v11/bigru/banding_gpu.json`.

## Hasil 2: bobot kelas, kapasitas, resep, dan oversampling (2 November 2026)

**Pertanyaan.** Kelemahan setelan bawaan adalah AFL 2:1 di pasien 08378 (0,74) dan ketergantungan pada 04936. Apakah setelan lain
memperbaikinya? Dan apakah oversampling (arahan pembimbing) membantu model urutan? SMOTE tidak bisa dipakai untuk urutan IBI, jadi
ditambahkan opsi `{"balancing": "over"}` (`../bersama/balancing.py`): window AFIB dan AFL disalin acak sampai jumlahnya sama dengan
Normal, hanya di data latih tiap fold.

Delapan setelan baru, protokol sama dengan "Hasil" di atas (5 ulangan × 5 fold pasien, seed `2026 + 100·r + k`, + bias cross-fit
grid −7…+3, CI bootstrap pasien 2.000×), dibandingkan **berpasangan** dengan `bawaan`. `under` dari 23 Oktober diikutkan sebagai
pembanding.

| Setelan | `SETELAN` | Dilatih di | Parameter | Window latih per kelas (fold r0 k0) |
|---|---|---|---|---|
| bawaan_ulang | `{}` (persis sama dengan bawaan) | PC, RTX 5060 | 7.172 | 82.012 / 46.765 / 1.143 |
| wp075 | `{"weight_power": 0.75}` | PC, RTX 5060 | 7.172 | 82.012 / 46.765 / 1.143 |
| wp100 | `{"weight_power": 1.0}` (bobot penuh = resep BiGRU v3) | PC, RTX 5060 | 7.172 | 82.012 / 46.765 / 1.143 |
| ep50 | `{"epochs": 50}` | PC, RTX 5060 | 7.172 | 82.012 / 46.765 / 1.143 |
| over | `{"balancing": "over"}` | PC, RTX 5060 | 7.172 | 82.012 × 3 |
| h16 | `{"hidden": 16}` | Kaggle, T4 | 2.052 | 82.012 / 46.765 / 1.143 |
| h64 | `{"hidden": 64}` | Kaggle, T4 | 26.628 | 82.012 / 46.765 / 1.143 |
| lr5e4 | `{"lr": 0.0005}` | Kaggle, T4 | 7.172 | 82.012 / 46.765 / 1.143 |

**Noise training.** `bawaan_ulang` **identik bit demi bit** dengan `bawaan` (OOF dan prediksi uji stres, 25 fold). Di mesin yang
sama, training ini deterministik, jadi selisih antar setelan lokal murni berasal dari setelannya. Lintas mesin: `wp100` (GPU v11)
vs BiGRU v3 bobot penuh (CPU v3, mesin lain, resep sama) memberi Δ Macro-F1 +0,000 [−0,01…+0,02]. Recall per pasien bisa sedikit
berbeda (06426: 0,28 vs 0,41). Tiga setelan Kaggle dibaca dengan catatan itu.

**Metrik (+ bias)**

| Sistem | Macro-F1 | AFL F1 | AFL P | AFL R | AFIB F1 | Normal R | Macro-F1 argmax |
|---|---|---|---|---|---|---|---|
| v3.1 (referensi) | 0,832 [0,67–0,92] | 0,647 [0,17–0,89] | 0,715 | 0,592 [0,13–0,91] | 0,909 | 0,924 | |
| BiGRU v3, bobot penuh | 0,829 [0,68–0,92] | 0,621 [0,17–0,90] | 0,692 | 0,564 [0,15–0,92] | 0,921 | 0,934 | |
| bawaan | 0,821 [0,68–0,87] | 0,592 [0,16–0,73] | 0,652 | 0,544 [0,16–0,75] | 0,922 | 0,950 | 0,795 |
| wp075 | 0,821 [0,68–0,89] | 0,590 [0,16–0,79] | 0,675 | 0,528 [0,14–0,81] | 0,924 | 0,942 | 0,756 |
| **wp100** | 0,830 [0,69–0,93] | 0,625 [0,21–0,92] | 0,691 | 0,572 [0,18–0,96] | 0,919 | 0,935 | 0,719 |
| lr5e4 | 0,833 [0,68–0,91] | 0,633 [0,17–0,86] | 0,674 | 0,601 [0,18–0,89] | 0,920 | 0,943 | 0,806 |
| h16 | 0,824 [0,68–0,90] | 0,601 [0,17–0,81] | 0,674 | 0,544 [0,16–0,83] | 0,922 | 0,941 | 0,803 |
| h64 | 0,813 [0,67–0,84] | 0,564 [0,14–0,64] | 0,649 | 0,502 [0,13–0,61] | 0,925 | 0,949 | 0,796 |
| ep50 | 0,797 [0,67–0,83] | 0,519 [0,14–0,60] | 0,559 | 0,489 [0,15–0,60] | 0,921 | 0,953 | 0,789 |
| over | 0,776 [0,65–0,82] | 0,465 [0,10–0,58] | 0,505 | 0,441 [0,12–0,57] | 0,917 | 0,944 | 0,772 |
| under | 0,599 [0,49–0,66] | 0,302 [0,01–0,49] | 0,240 | 0,412 [0,02–0,64] | 0,702 | 0,766 | 0,579 |

**Perbandingan berpasangan** (Δ Macro-F1 [CI 95%], P(Δ > 0); vonis P ≥ 0,95 lebih baik, ≤ 0,05 lebih buruk)

| Setelan | vs bawaan | P | Vonis | Δ AFL F1 (P) | Δ Normal R (P) | vs v3.1 | P |
|---|---|---|---|---|---|---|---|
| wp075 | −0,000 [−0,01…+0,02] | 0,47 | tidak terbedakan | −0,002 (0,40) | −0,008 (0,00) | −0,010 [−0,03…+0,01] | 0,12 |
| **wp100** | +0,008 [−0,02…+0,07] | 0,65 | tidak terbedakan | +0,032 (0,67) | −0,015 (0,00) | −0,002 [−0,01…+0,02] | 0,47 |
| lr5e4 | +0,012 [−0,01…+0,04] | 0,87 | tidak terbedakan | +0,041 (0,94) | −0,007 (0,00) | +0,002 [−0,03…+0,02] | 0,61 |
| h16 | +0,002 [−0,01…+0,03] | 0,61 | tidak terbedakan | +0,008 (0,63) | −0,009 (0,00) | −0,008 [−0,03…+0,01] | 0,20 |
| h64 | −0,008 [−0,03…+0,00] | 0,18 | tidak terbedakan | −0,029 (0,14) | −0,001 (0,37) | −0,018 [−0,08…+0,02] | 0,28 |
| ep50 | −0,025 [−0,06…−0,00] | 0,00 | lebih buruk | −0,073 (0,00) | +0,003 (0,96) | −0,035 [−0,12…+0,01] | 0,15 |
| over | −0,045 [−0,07…−0,02] | 0,00 | lebih buruk | −0,128 (0,00) | −0,006 (0,00) | −0,056 [−0,13…−0,01] | 0,01 |
| under | −0,223 [−0,33…−0,14] | 0,00 | lebih buruk | −0,291 (0,00) | −0,184 (0,00) | −0,233 [−0,38…−0,13] | 0,00 |
| wp100 vs BiGRU v3 bobot penuh | +0,000 [−0,01…+0,02] | 0,53 | tidak terbedakan | +0,004 (0,62) | +0,001 (0,76) | | |

**Recall AFL per pasien (+ bias), AFL palsu, dan bias AFL**

| Sistem | 04043 | 04908 | 04936 | 06426 | 07910 | 08215 | 08378 | AFL palsu / ulangan (di pasien tanpa AFL) | bias AFL median [min…maks] |
|---|---|---|---|---|---|---|---|---|---|
| v3.1 | 0,40 | 0,92 | 0,51 | 0,17 | 0,00 | 0,00 | 0,96 | 274 (25) | |
| BiGRU v3 bobot penuh | 0,60 | 0,87 | 0,46 | 0,28 | 0,01 | 0,00 | 0,97 | 291 (29) | |
| bawaan | 0,80 | 0,98 | 0,52 | 0,29 | 0,00 | 0,00 | 0,74 | 332 (44) | −2,00 [−4,00…−1,25] |
| wp075 | 0,80 | 0,94 | 0,46 | 0,20 | 0,00 | 0,00 | 0,83 | 296 (41) | −3,25 [−5,50…−2,25] |
| **wp100** | 0,60 | 0,95 | 0,45 | 0,41 | 0,00 | 0,00 | 1,00 | 300 (32) | −4,00 [−5,75…−3,00] |
| lr5e4 | 0,60 | 0,99 | 0,53 | 0,28 | 0,03 | 0,00 | 0,92 | 338 (58) | −1,75 [−4,00…−1,00] |
| h16 | 0,80 | 0,97 | 0,47 | 0,27 | 0,00 | 0,00 | 0,84 | 299 (35) | −2,25 [−3,75…−1,00] |
| h64 | 0,40 | 0,77 | 0,52 | 0,26 | 0,00 | 0,00 | 0,61 | 311 (52) | −2,50 [−5,25…−1,50] |
| ep50 | 0,60 | 0,97 | 0,52 | 0,26 | 0,00 | 0,02 | 0,52 | 454 (58) | −2,25 [−4,25…−1,50] |
| over | 0,40 | 0,81 | 0,44 | 0,15 | 0,00 | 0,00 | 0,55 | 525 (88) | −5,75 [−7,00…−3,25], 8% di batas |
| under | 0,00 | 0,99 | 0,62 | 0,16 | 0,00 | 0,00 | 0,03 | 1937 (1188) | −2,25 [−5,25…−0,25] |

**Uji stres takikardia** (ambang per jenis variasi detak: tanpa variasi / seragam ±10 ms / Gauss 10 ms / Gauss 20 ms; % window Normal asli yang ditebak AFIB/AFL per pita detak, OOF)

| Sistem | Ambang (bpm) | 0–100 bpm | 100–110 bpm | 110–120 bpm | ≥ 120 bpm (100 window) |
|---|---|---|---|---|---|
| v3.1 | 115 (paket) | 6,7% | 12,7% | 59,0% | 95,2% |
| bawaan | 120 / 120 / 120 / 120 | 4,1% | 10,0% | 52,6% | 86,6% |
| wp075 | 120 / 120 / 120 / 115 | 4,9% | 11,1% | 55,9% | 87,0% |
| **wp100** | 120 / 120 / 120 / 115 | 5,5% | 12,2% | 60,5% | 90,2% |
| lr5e4 | 120 / 120 / 120 / 115 | 4,7% | 11,0% | 59,4% | 93,2% |
| h16 | 120 / 120 / 120 / 120 | 5,0% | 10,3% | 56,2% | 93,4% |
| h64 | 125 / 125 / 125 / 120 | 4,3% | 9,6% | 48,1% | 83,8% |
| ep50 | 120 / 120 / 125 / 120 | 3,8% | 10,3% | 49,3% | 83,2% |
| over | 120 / 125 / 125 / 120 | 4,6% | 12,1% | 51,4% | 79,2% |
| under | 110 / 110 / 110 / 110 | 18,2% | 71,6% | 98,4% | 100,0% |

**Tanpa pasien 04936** (22 pasien, 6 pasien AFL, 479 window AFL; bias dihitung ulang di subset)

| Sistem | Macro-F1 | AFL F1 | v3.1 − sistem | P(v3.1 lebih baik) |
|---|---|---|---|---|
| v3.1 | 0,859 [0,62–0,92] | 0,725 | | |
| BiGRU v3 bobot penuh | 0,860 [0,63–0,92] | 0,708 | −0,001 [−0,03…+0,01] | 0,32 |
| bawaan | 0,721 [0,63–0,81] | 0,289 | +0,138 [−0,02…+0,21] | 0,68 |
| wp075 | 0,818 [0,63–0,87] | 0,578 | +0,041 [−0,01…+0,06] | 0,74 |
| **wp100** | 0,852 [0,63–0,90] | 0,690 | +0,007 [−0,04…+0,03] | 0,60 |
| lr5e4 | 0,721 [0,63–0,81] | 0,301 | +0,138 [−0,03…+0,21] | 0,67 |
| h16 | 0,777 [0,63–0,84] | 0,455 | +0,083 [−0,04…+0,13] | 0,66 |
| h64 | 0,748 [0,63–0,80] | 0,375 | +0,111 [−0,01…+0,16] | 0,84 |
| ep50 | 0,736 [0,63–0,79] | 0,342 | +0,123 [−0,01…+0,17] | 0,92 |
| over | 0,768 [0,63–0,83] | 0,437 | +0,091 [−0,01…+0,12] | 0,93 |
| under | 0,533 [0,48–0,60] | 0,054 | +0,326 [+0,10…+0,40] | 1,00 |

**Bootstrap bersarang** (bias dipilih ulang di setiap sampel; tiga setelan dengan Macro-F1 tertinggi) vs v1

| Setelan | Macro-F1 [CI] | Δ vs v1 [CI] | P | Vonis |
|---|---|---|---|---|
| lr5e4 | 0,833 [0,66–0,90] | +0,090 [−0,05…+0,13] | 0,92 | tidak terbedakan |
| **wp100** | 0,830 [0,68–0,91] | +0,086 [+0,01…+0,16] | 0,98 | lebih baik |
| h16 | 0,824 [0,67–0,89] | +0,080 [−0,01…+0,13] | 0,97 | lebih baik |

**Seleksi cross-fit** dari 8 setelan (bawaan, wp075, wp100, over, h16, h64, lr5e4, ep50): Macro-F1 0,816 [0,67–0,88], AFL F1 0,567 [0,14–0,76]; vs v3.1 −0,016 [−0,04…+0,01], P = 0,06. Setelan terpilih per fold (25): lr5e4 6, h16 6, wp075 5, h64 4, ep50 2, wp100 2.

**Kesimpulan**
- **Tidak ada setelan yang lebih baik dari `bawaan` pada Macro-F1** (semua P < 0,95). Tiga setelan lebih buruk: `ep50`, `over`, dan
  `under`. Karena training-nya deterministik, ini bukan soal noise run. Masalahnya, dengan 23 pasien CI-nya lebar.
- **Bobot kelas penuh (`wp100`) adalah pilihan paling tahan uji.** Macro-F1 0,830 setara v3.1 (P = 0,47) dan setara BiGRU v3
  (Δ +0,000).
  - 08378 kembali terdeteksi (0,74 → 1,00).
  - **Ketergantungan pada 04936 hilang**: tanpa 04936 Macro-F1 0,852, sedangkan `bawaan` 0,721 (v3.1 0,859).
  - Bootstrap bersarang vs v1: P = 0,98 (lebih baik). Untuk `bawaan` nilainya 0,91 (tidak terbedakan).
  - Harganya: recall Normal 0,950 → 0,935 (P = 0,00, lebih buruk), dan alarm pada Normal ≥ 120 bpm 87% → 90% (v3.1 95%). Ambang
    takikardia 120 / 120 / 120 / 115, sama dengan paket v3.1 pada variasi terbesar. Recall 04043 juga turun 0,80 → 0,60, tetapi
    pasien itu hanya punya **1 window AFL** (tertebak di 3 dari 5 ulangan, bukan 4), jadi angka 04043 di semua tabel tidak bermakna.
  - Kalau BiGRU dijadikan kandidat, pakai `{"weight_power": 1.0}`. Alasan menyimpang dari bobot lunak (AGENTS.md): AFL 2:1 dan
    sensitivitas tanpa 04936.
  - Karena `wp100` = resep BiGRU v3 dengan bobot tersimpan, BiGRU v3 kini juga sudah diuji takikardia.
- **`lr5e4` punya Macro-F1 tertinggi (0,833), tetapi tidak terbedakan dari `bawaan`** (P = 0,87; AFL F1 +0,041, P = 0,94). Tanpa
  04936 ia jatuh ke 0,721, sama dengan `bawaan`. Keunggulannya datang dari window 04936.
- **Kapasitas dan lama training.**
  - `h16` (2.052 parameter, 3,5× lebih kecil) setara `bawaan` (+0,002), jadi model bisa lebih kecil tanpa rugi.
  - `h64` (26.628 parameter) sedikit lebih buruk di AFL (04908 0,77; 08378 0,61), walaupun ambang takikardianya paling tinggi
    (125 bpm).
  - `ep50` lebih buruk (−0,025, P = 0,00): AFL palsu naik 332 → 454 per ulangan, tanda overfit.
  - Resep v3 (30 epoch, `hidden` 32) sudah dekat optimum di rentang yang dicoba.
- **Oversampling lebih buruk daripada bobot kelas**: `over` −0,045 vs `bawaan` (P = 0,00), dan jauh di bawah `wp100`, padahal
  keduanya memberi AFL bobot total yang sama.
  - Menyalin 1.143 window AFL ±70× membuat model menghafalnya.
  - 8% nilai bias kelas menyentuh batas grid (AFL −7), tanda bahaya (AGENTS.md).
  - Bersama `under` (−0,223), jawabannya untuk BiGRU: **balancing data lebih buruk daripada pembobotan kelas**. Di XGBoost justru
    `under` membantu (`../xgboost/README.md`), jadi efek balancing bergantung pada model.
- **Seleksi cross-fit dari 8 setelan tidak menolong.** Hasilnya 0,816, di bawah `bawaan` 0,821 (vs v3.1 P = 0,06). Pilihan per fold
  berpindah-pindah antara lr5e4, h16, wp075, dan h64. Tidak ada pemenang yang stabil, jadi tuning di rentang ini tidak memberi lebih
  dari variasi antar fold.

**Yang tidak bisa disimpulkan**
- `wp100` dipilih setelah melihat hasil. Ia bukan nilai hasil tuning (resepnya sudah ada sejak v3), tetapi pada data penuh ia tetap
  tidak terbedakan dari `bawaan` (P = 0,65). Keunggulannya terutama di uji tanpa 04936, dan selisih itu terhadap v3.1 ber-CI lebar
  (P = 0,60).
- `h16`, `h64`, dan `lr5e4` dilatih di GPU lain (T4). Determinisme di sana tidak diperiksa.
- Kombinasi setelan (mis. `wp100` + `lr5e4`) belum dicoba di sini; dicoba di "Hasil 3", dengan seleksi cross-fit.
- Semua angka dari AFDB saja. Masalah domain dan sinus takikardia asli (v8–v10) tidak berubah oleh tuning ini.

**Lingkungan.**
- PC: GPU RTX 5060 8 GB, dua proses berbagi GPU. Venv sementara: Python 3.13, torch 2.11.0+cu128, numpy 2.5.4, scipy 1.18.1,
  scikit-learn 1.9.1, imbalanced-learn 0.14.2.
- Kaggle: kernel privat `farhanlhsn/sihedaf-v11-bigru-tuning`, GPU T4 ×2, torch 2.11.0+cu128, tiga proses, ±73 menit. Kode v11
  dibawa di dalam script, karena belum ada di GitHub. Data dari GitHub Release `v11-data` dan dicek sidik jarinya.
- Kode: commit `b18d6bc` + perubahan v11 yang belum di-commit (termasuk opsi `over`). Seluruh tes v11 lolos.
- Dijalankan lewat `bersama.runner.jalankan` (sama dengan sel training notebook).
- `jalankan.ipynb` juga dijalankan penuh lewat salinan untuk setelan `bawaan` (COMPLETE, angka sama). *Pembaruan 3 November
  2026:* notebook di repo sekarang juga dijalankan (evaluasi dari cache, tanpa training ulang, angka identik) dan menyimpan output
  serta sel Kesimpulan sebagai catatan hasil; `tests/test_notebook.py` disesuaikan.
- Hasil per fold di `cache/v11/bigru/<setelan>/`, ringkasan per setelan di `ringkasan.json`, dan semua perbandingan di
  `cache/v11/bigru/banding_tuning.json`.

## Hasil 3: regularisasi di atas bobot penuh (2 November 2026)

**Pertanyaan.** Di "Hasil 2", setelan yang berlatih lebih lama atau berkapasitas lebih besar justru lebih buruk, tanda overfit. Model
ini juga tidak punya regularisasi selain weight decay kecil. Apakah regularisasi di atas bobot kelas penuh (`wp100`) memperbaiki BiGRU?

Enam setelan baru, semuanya dengan `"weight_power": 1.0`, protokol sama, dibandingkan **berpasangan** dengan `wp100`. `bawaan` ikut
sebagai pembanding. Definisi, training, dan evaluasinya ada di `eksperimen.py` (`e3`), dan hasil perbandingan di
`cache/v11/bigru/banding_putaran3.json`.

| Setelan | `SETELAN` | Dilatih di | Parameter |
|---|---|---|---|
| w1_ep20 | `{"weight_power": 1.0, "epochs": 20}` | PC, RTX 5060 | 7.172 |
| w1_bs512 | `{"weight_power": 1.0, "batch_size": 512}` | PC, RTX 5060 | 7.172 |
| w1_h16 | `{"weight_power": 1.0, "hidden": 16}` | PC, RTX 5060 | 2.052 |
| w1_wd1e3 | `{"weight_power": 1.0, "weight_decay": 0.001}` | Kaggle, T4 | 7.172 |
| w1_wd1e2 | `{"weight_power": 1.0, "weight_decay": 0.01}` | Kaggle, T4 | 7.172 |
| w1_lr5e4 | `{"weight_power": 1.0, "lr": 0.0005}` | Kaggle, T4 | 7.172 |

**Metrik (+ bias)**

| Sistem | Macro-F1 | AFL F1 | AFL P | AFL R | AFIB F1 | Normal R | Macro-F1 argmax |
|---|---|---|---|---|---|---|---|
| v3.1 (referensi) | 0,832 [0,67–0,92] | 0,647 [0,17–0,89] | 0,715 | 0,592 [0,13–0,91] | 0,909 | 0,924 | |
| BiGRU v3, bobot penuh | 0,829 [0,68–0,92] | 0,621 [0,17–0,90] | 0,692 | 0,564 [0,15–0,92] | 0,921 | 0,934 | |
| **wp100** | 0,830 [0,69–0,93] | 0,625 [0,21–0,92] | 0,691 | 0,572 [0,18–0,96] | 0,919 | 0,935 | 0,719 |
| w1_wd1e3 | 0,833 [0,68–0,93] | 0,636 [0,17–0,91] | 0,695 | 0,589 [0,15–0,94] | 0,917 | 0,934 | 0,715 |
| w1_wd1e2 | 0,832 [0,68–0,93] | 0,634 [0,17–0,91] | 0,699 | 0,581 [0,15–0,94] | 0,917 | 0,934 | 0,715 |
| w1_bs512 | 0,828 [0,67–0,92] | 0,631 [0,17–0,91] | 0,696 | 0,581 [0,15–0,94] | 0,912 | 0,929 | 0,710 |
| w1_ep20 | 0,827 [0,69–0,93] | 0,625 [0,20–0,91] | 0,697 | 0,570 [0,17–0,96] | 0,914 | 0,931 | 0,715 |
| w1_lr5e4 | 0,824 [0,68–0,92] | 0,627 [0,18–0,89] | 0,698 | 0,572 [0,16–0,92] | 0,907 | 0,923 | 0,710 |
| w1_h16 | 0,782 [0,68–0,87] | 0,500 [0,17–0,76] | 0,587 | 0,455 [0,17–0,79] | 0,908 | 0,924 | 0,716 |
| bawaan | 0,821 [0,68–0,87] | 0,592 [0,16–0,73] | 0,652 | 0,544 [0,16–0,75] | 0,922 | 0,950 | 0,795 |

**Perbandingan berpasangan** (Δ Macro-F1 [CI 95%], P(Δ > 0); vonis P ≥ 0,95 lebih baik, ≤ 0,05 lebih buruk)

| Setelan | vs wp100 | P | Vonis | Δ AFL F1 (P) | Δ Normal R (P) | vs v3.1 | P |
|---|---|---|---|---|---|---|---|
| w1_wd1e3 | +0,003 [−0,01…+0,01] | 0,65 | tidak terbedakan | +0,011 (0,67) | −0,001 (0,04) | +0,001 [−0,01…+0,01] | 0,63 |
| w1_wd1e2 | +0,002 [−0,01…+0,01] | 0,64 | tidak terbedakan | +0,010 (0,68) | −0,001 (0,08) | +0,001 [−0,01…+0,01] | 0,57 |
| w1_bs512 | −0,001 [−0,02…+0,00] | 0,33 | tidak terbedakan | +0,007 (0,65) | −0,007 (0,00) | −0,003 [−0,01…+0,01] | 0,32 |
| w1_ep20 | −0,003 [−0,01…+0,00] | 0,06 | tidak terbedakan | +0,000 (0,54) | −0,004 (0,00) | −0,004 [−0,02…+0,02] | 0,40 |
| w1_lr5e4 | −0,006 [−0,02…+0,00] | 0,13 | tidak terbedakan | +0,003 (0,54) | −0,012 (0,00) | −0,008 [−0,02…+0,01] | 0,07 |
| w1_h16 | −0,048 [−0,07…−0,01] | 0,00 | lebih buruk | −0,125 (0,00) | −0,011 (0,00) | −0,050 [−0,06…+0,01] | 0,05 |
| bawaan | −0,008 [−0,07…+0,02] | 0,35 | tidak terbedakan | −0,032 (0,33) | +0,015 (1,00) | −0,010 [−0,06…+0,02] | 0,34 |

**Recall AFL per pasien (+ bias), AFL palsu, dan bias AFL**

| Sistem | 04043 | 04908 | 04936 | 06426 | 07910 | 08215 | 08378 | AFL palsu / ulangan (di pasien tanpa AFL) | bias AFL median [min…maks] |
|---|---|---|---|---|---|---|---|---|---|
| v3.1 | 0,40 | 0,92 | 0,51 | 0,17 | 0,00 | 0,00 | 0,96 | 274 (25) | |
| BiGRU v3 bobot penuh | 0,60 | 0,87 | 0,46 | 0,28 | 0,01 | 0,00 | 0,97 | 291 (29) | |
| **wp100** | 0,60 | 0,95 | 0,45 | 0,41 | 0,00 | 0,00 | 1,00 | 300 (32) | −4,00 [−5,75…−3,00] |
| w1_wd1e3 | 0,60 | 0,90 | 0,48 | 0,28 | 0,00 | 0,00 | 0,99 | 304 (39) | −4,50 [−5,75…−3,00] |
| w1_wd1e2 | 0,60 | 0,94 | 0,47 | 0,25 | 0,00 | 0,00 | 0,99 | 291 (26) | −4,50 [−5,75…−3,25] |
| w1_bs512 | 0,80 | 0,87 | 0,47 | 0,31 | 0,00 | 0,00 | 1,00 | 302 (37) | −4,50 [−6,25…−3,25] |
| w1_ep20 | 0,60 | 0,96 | 0,44 | 0,40 | 0,00 | 0,00 | 1,00 | 294 (40) | −3,75 [−5,75…−3,25] |
| w1_lr5e4 | 0,40 | 0,91 | 0,47 | 0,35 | 0,00 | 0,00 | 0,96 | 293 (37) | −4,00 [−5,50…−3,25] |
| w1_h16 | 0,60 | 0,93 | 0,34 | 0,38 | 0,00 | 0,00 | 0,80 | 286 (46) | −4,00 [−6,75…−3,00] |
| bawaan | 0,80 | 0,98 | 0,52 | 0,29 | 0,00 | 0,00 | 0,74 | 332 (44) | −2,00 [−4,00…−1,25] |

**Uji stres takikardia** (ambang per jenis variasi detak: tanpa variasi / seragam ±10 ms / Gauss 10 ms / Gauss 20 ms; % window Normal asli yang ditebak AFIB/AFL per pita detak, OOF)

| Sistem | Ambang (bpm) | 0–100 bpm | 100–110 bpm | 110–120 bpm | ≥ 120 bpm (100 window) |
|---|---|---|---|---|---|
| v3.1 | 115 (paket) | 6,7% | 12,7% | 59,0% | 95,2% |
| **wp100** | 120 / 120 / 120 / 115 | 5,5% | 12,2% | 60,5% | 90,2% |
| w1_wd1e3 | 120 / 120 / 120 / 115 | 5,6% | 12,0% | 59,4% | 90,8% |
| w1_wd1e2 | 125 / 125 / 120 / 115 | 5,6% | 12,2% | 60,8% | 89,8% |
| w1_bs512 | 120 / 120 / 120 / 115 | 6,1% | 13,4% | 66,6% | 93,4% |
| w1_ep20 | 120 / 120 / 120 / 115 | 5,8% | 13,0% | 66,1% | 92,6% |
| w1_lr5e4 | 120 / 120 / 120 / 115 | 6,5% | 14,3% | 68,8% | 95,2% |
| w1_h16 | 120 / 120 / 120 / 115 | 6,5% | 13,8% | 67,6% | 96,2% |
| bawaan | 120 / 120 / 120 / 120 | 4,1% | 10,0% | 52,6% | 86,6% |

**Tanpa pasien 04936** (22 pasien, 6 pasien AFL, 479 window AFL; bias dihitung ulang di subset)

| Sistem | Macro-F1 | AFL F1 | v3.1 − sistem | P(v3.1 lebih baik) |
|---|---|---|---|---|
| v3.1 | 0,859 [0,62–0,92] | 0,725 | | |
| BiGRU v3 bobot penuh | 0,860 [0,63–0,92] | 0,708 | −0,001 [−0,03…+0,01] | 0,32 |
| **wp100** | 0,852 [0,63–0,90] | 0,690 | +0,007 [−0,04…+0,03] | 0,60 |
| w1_wd1e3 | 0,842 [0,63–0,90] | 0,663 | +0,017 [−0,02…+0,04] | 0,66 |
| w1_wd1e2 | 0,851 [0,63–0,91] | 0,691 | +0,008 [−0,03…+0,02] | 0,64 |
| w1_bs512 | 0,845 [0,62–0,90] | 0,675 | +0,015 [−0,02…+0,03] | 0,69 |
| w1_ep20 | 0,844 [0,63–0,90] | 0,675 | +0,015 [−0,02…+0,03] | 0,67 |
| w1_lr5e4 | 0,848 [0,62–0,91] | 0,699 | +0,011 [−0,02…+0,02] | 0,68 |
| w1_h16 | 0,775 [0,62–0,82] | 0,478 | +0,085 [−0,01…+0,12] | 0,81 |
| bawaan | 0,721 [0,63–0,81] | 0,289 | +0,138 [−0,02…+0,21] | 0,68 |

**Bootstrap bersarang** (bias dipilih ulang di setiap sampel; tiga setelan dengan Macro-F1 tertinggi) vs v1

| Setelan | Macro-F1 [CI] | Δ vs v1 [CI] | P | Vonis |
|---|---|---|---|---|
| w1_wd1e3 | 0,833 [0,67–0,91] | +0,089 [+0,01…+0,15] | 0,98 | lebih baik |
| w1_wd1e2 | 0,832 [0,67–0,92] | +0,089 [+0,01…+0,16] | 0,98 | lebih baik |
| w1_bs512 | 0,828 [0,66–0,92] | +0,085 [+0,00…+0,16] | 0,98 | lebih baik |

**Seleksi cross-fit** dari ketujuh setelan berbobot penuh (wp100 + enam di atas): Macro-F1 0,830 [0,68–0,93], AFL F1 0,629
[0,18–0,92]; vs v3.1 −0,002 [−0,01…+0,02], P = 0,46. Setelan terpilih per fold (25): w1_bs512 9, wp100 8, w1_wd1e3 4, w1_h16 2,
w1_ep20 1, w1_wd1e2 1.

**Kesimpulan**
- **Tidak ada regularisasi yang lebih baik dari `wp100`.** Lima setelan berada dalam −0,006 … +0,003 Macro-F1 dari `wp100`, dan
  semuanya tidak terbedakan. `w1_h16` lebih buruk (−0,048, P = 0,00).
- **Seleksi cross-fit memberi 0,830**, sama dengan `wp100` dan v3.1 (P = 0,46). Pilihan per fold berpindah-pindah, jadi tidak ada
  pemenang yang stabil. Resep bobot penuh sudah di titik jenuh.
- **Keunggulan bobot penuh tetap ada.**
  - Tanpa 04936 semua setelan di 0,842–0,851 (v3.1 0,859, `bawaan` 0,721), kecuali `w1_h16` 0,775.
  - Bootstrap bersarang vs v1: P = 0,98 untuk `w1_wd1e3`, `w1_wd1e2`, dan `w1_bs512` (lebih baik), sama dengan `wp100`.
- **`hidden` 16 dengan bobot penuh rusak**: AFL F1 −0,125 (P = 0,00), 08378 0,80. Padahal di "Hasil 2" `hidden` 16 dengan bobot
  lunak setara `bawaan`. Jadi model kecil tidak cocok dengan bobot AFL yang besar. Penyebab pastinya belum diuji.
- **Alarm pada detak cepat tidak membaik.**
  - Semua setelan sedikit menurunkan recall Normal dibanding `wp100` (−0,012 … −0,001).
  - Alarm pada Normal ≥ 120 bpm 89,8–96,2% (`wp100` 90,2%).
  - `w1_wd1e2` sedikit terbaik (ambang 125 / 125 / 120 / 115), tetapi selisihnya kecil.
- **Kesimpulan akhir BiGRU: tuning selesai.** Kandidat BiGRU = `{"weight_power": 1.0}`, resep paling sederhana di antara yang
  setara. Perbaikan berikutnya butuh data (pasien baru, rekaman perangkat SIHEDAF berlabel), bukan hiperparameter.

**Yang tidak bisa disimpulkan**
- `w1_wd1e3`, `w1_wd1e2`, dan `w1_lr5e4` dilatih di Kaggle T4, lalu dibandingkan dengan `wp100` dari GPU lokal. Selisih sekecil
  ±0,003 bisa berasal dari beda mesin.
- Seleksi cross-fit hanya di antara tujuh setelan berbobot penuh (`bawaan` tidak ikut).
- Dropout dan GRU 2 lapis tidak dicoba (butuh perubahan `model.py`). Karena regularisasi lain tidak memberi apa-apa, peluangnya kecil.

**Laporan PDF** (satu per eksperimen, dibuat oleh `laporan/build_report.py` dari cache; angka dicek otomatis terhadap OOF dan
`ringkasan.json`, setiap halaman dirender dan diperiksa):
- `laporan/laporan_bigru_e1_balancing.pdf`: "Hasil" (bobot lunak vs undersampling);
- `laporan/laporan_bigru_e2_tuning.pdf`: "Hasil 2";
- `laporan/laporan_bigru_e3_regularisasi.pdf`: "Hasil 3".

**Lingkungan.** Sama dengan "Hasil 2". Kernel Kaggle privat `farhanlhsn/sihedaf-v11-bigru-putaran3` (T4 ×2, torch 2.11.0+cu128,
tiga proses, ±75 menit). Training lokal sempat terhenti karena sesi berhenti, lalu dilanjutkan dari fold terakhir (runner
melewati fold yang sudah ada).
