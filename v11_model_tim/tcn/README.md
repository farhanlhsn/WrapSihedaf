# tcn — Temporal Convolutional Network

**Model.** Empat blok residual konvolusi 1D dengan dilatasi 1, 2, 4, 8 (dua konvolusi per blok, kernel 3, BatchNorm,
ReLU, dropout). Jangkauan pandang ±61 detak, cukup untuk seluruh window 30 detak. Konvolusinya non-kausal (melihat
detak sebelum dan sesudah), karena prediksi dibuat per window utuh. Input dan aktivasi setiap blok dikalikan mask,
sehingga isi posisi padding tidak memengaruhi hasil (`tests/test_model_dl.py`). ±23 ribu parameter.

**Input.** Urutan IBI + ΔIBI (30 × 2), z-score bermask dari data latih (sama dengan model DL v3).

**Setelan bawaan** (`model.py`): `kanal` 32, `kernel` 3 (harus ganjil), `dilasi` (1, 2, 4, 8), `dropout` 0,1, resep v3
(30 epoch, batch 256, lr 1e-3, weight decay 1e-4), bobot kelas lunak.

**Waktu.** Colab GPU Tesla T4: 107 s per fold, ±45 menit untuk 25 fold, ditambah evaluasi ±3 menit (2 November 2026). GPU yang
lebih besar hampir tidak mempercepat satu proses, karena bottleneck-nya overhead Python per batch. Yang mempercepat adalah
beberapa proses di satu GPU: 6 setelan sekaligus di Colab L4 (12 core, 53 GB RAM) selesai dalam 68 menit ("Hasil 2").

**Ringkasan dua eksperimen** (pemilik repo; definisi di `eksperimen.py`, PDF di `laporan/`):

| Eksperimen | Isi | Jawaban |
|---|---|---|
| 1 ("Hasil") | setelan bawaan | lebih buruk dari v3.1 dan BiGRU: AFL 2:1 (08378) hilang setelah bias |
| 2 ("Hasil 2") | bobot kelas, kapasitas, dropout, dilatasi | bobot penuh (`{"weight_power": 1.0}`) setara v3.1 dan BiGRU `wp100`; kandidat TCN |

**Ide eksperimen** (ganti `NAMA_SETELAN` tiap kali; kanal 16, dropout 0,2, dan dilatasi 1–4 sudah dicoba di "Hasil 2"):
- `{"kanal": 64}`: jaga jumlah parameter ≤ ±100 ribu (dicetak di info hasil);
- `{"dilasi": [1, 2, 4]}` (jangkauan lebih pendek) atau `{"kernel": 5}`;
- `{"dropout": 0.2}`.

## Hasil (2 November 2026)

Setelan `bawaan` (tanpa balancing, bobot kelas lunak), dijalankan pemilik repo sebagai uji coba arsitektur selain BiGRU.
Protokol sama dengan model lain: 5 ulangan × 5 fold pasien, seed `2026 + 100·r + k`, + bias cross-fit grid −7…+3, CI
bootstrap pasien 2.000×. Pembanding BiGRU memakai OOF di `cache/v11/bigru/`, dengan fold dan sampel bootstrap yang sama.
PDF: `laporan/laporan_tcn_e1_bawaan.pdf` (`eksperimen.py` `e1`).

| Sistem | Macro-F1 | AFL F1 | AFL P | AFL R | AFIB F1 | Normal R |
|---|---|---|---|---|---|---|
| v1 (argmax) | 0,744 [0,64–0,81] | 0,399 [0,09–0,59] | 0,275 | 0,728 [0,37–0,98] | 0,898 | 0,903 [0,82–0,95] |
| v3.1 (referensi) | 0,832 [0,67–0,92] | 0,647 [0,17–0,89] | 0,715 | 0,592 [0,13–0,91] | 0,909 | 0,924 [0,84–0,97] |
| BiGRU v11 bawaan | 0,821 [0,68–0,87] | 0,592 [0,16–0,73] | 0,652 | 0,544 [0,16–0,75] | 0,922 | 0,950 [0,90–0,98] |
| TCN argmax (tanpa bias) | 0,790 [0,65–0,86] | 0,489 [0,08–0,67] | 0,379 | 0,694 [0,23–0,97] | 0,930 | 0,932 [0,88–0,97] |
| **TCN + bias** (protokol) | 0,752 [0,63–0,80] | 0,391 [0,02–0,53] | 0,473 | 0,340 [0,01–0,55] | 0,916 | 0,959 [0,92–0,98] |

Bias AFL berkisar −6,5 … −2,25 (median −3), dan tidak ada yang menyentuh batas grid. Rentangnya sedikit lebih lebar daripada
BiGRU (bawaan −4 … −1,25, `wp100` −5,75 … −3), terutama karena satu fold bernilai −6,5. Bias AFIB median −1,5.

**Perbandingan berpasangan** (Δ Macro-F1 [CI 95%], P(Δ > 0); vonis P ≥ 0,95 lebih baik, ≤ 0,05 lebih buruk)

| Kandidat vs referensi | Δ Macro-F1 | P | Vonis | Catatan |
|---|---|---|---|---|
| TCN + bias vs v3.1 | −0,080 [−0,27…+0,00] | 0,035 | lebih buruk | AFL R −0,251 (P = 0,19); AFL F1 −0,256 (P = 0,00); Normal R +0,036 (P = 1,00) |
| TCN + bias vs BiGRU bawaan | −0,070 [−0,21…−0,01] | 0,003 | lebih buruk | AFL R −0,204 (P = 0,00); AFL F1 −0,201 (P = 0,00); Normal R +0,009 (P = 1,00) |
| TCN + bias vs BiGRU `wp100` | −0,078 [−0,28…+0,01] | 0,070 | tidak terbedakan | AFL R −0,232 (P = 0,24); AFL F1 −0,233 (P = 0,08); Normal R +0,024 (P = 1,00) |
| TCN + bias vs v1 | +0,008 [−0,16…+0,09] | 0,529 | tidak terbedakan | AFL R −0,388 (P = 0,00); AFL F1 −0,007 (P = 0,45); Normal R +0,056 (P = 1,00) |
| TCN argmax vs v3.1 | −0,042 [−0,12…+0,00] | 0,035 | lebih buruk | AFL R +0,102 (P = 1,00); AFL F1 −0,158 (P = 0,02); Normal R +0,008 (P = 0,68) |
| TCN argmax vs BiGRU bawaan | −0,032 [−0,09…+0,00] | 0,047 | lebih buruk | AFL R +0,150 (P = 1,00); AFL F1 −0,103 (P = 0,03); Normal R −0,018 (P = 0,00) |

**Recall AFL per pasien dan AFL palsu**

| Sistem | 04043 | 04908 | 04936 | 06426 | 07910 | 08215 | 08378 | AFL palsu / ulangan (di pasien tanpa AFL) |
|---|---|---|---|---|---|---|---|---|
| v3.1 | 0,40 | 0,92 | 0,51 | 0,17 | 0,00 | 0,00 | 0,96 | 274 (25) |
| BiGRU bawaan | 0,80 | 0,98 | 0,52 | 0,29 | 0,00 | 0,00 | 0,74 | 332 (44) |
| TCN argmax | 1,00 | 1,00 | 0,64 | 0,42 | 0,06 | 0,02 | 1,00 | 1.317 (494) |
| TCN + bias | 0,60 | 0,94 | 0,51 | 0,08 | 0,00 | 0,00 | 0,01 | 410 (74) |

**Kenapa bias merusak TCN.**
- Tanpa bias, TCN menebak AFL terlalu sering: 1.317 window AFL palsu per ulangan (4,8× v3.1), 713 dari ritme Normal dan 604
  dari AFIB.
- Bias cross-fit menekannya ke 410, tetapi ikut menghapus AFL 2:1 pasien 08378 (1,00 → 0,01) dan 06426 (0,42 → 0,08). AFL
  04936 bertahan (0,64 → 0,51).
- Kemampuan memisahkan AFL secara keseluruhan tidak jauh berbeda. AUROC AFL-vs-lainnya (rata-rata 5 ulangan): TCN 0,922,
  v3.1 0,929, BiGRU bawaan 0,938, BiGRU `wp100` 0,920.
- Masalahnya ada di **margin skor AFL** (log p(AFL) − log p kelas lain terbesar, median per pasien; satuannya sama dengan
  bias). Di 08378 (AFL 2:1) margin TCN hanya 1,72, jauh di bawah 04936 (3,77). BiGRU `wp100` dan v3.1 punya 5,23 dan 5,65 di
  08378, jadi tetap menangkapnya walau bias AFL-nya −4. AFL hanya ditebak bila margin kira-kira melebihi besarnya bias, sehingga
  bias −3 menghapus hampir seluruh 08378 tetapi menyisakan 04936. TCN belajar AFL terutama dari 04936 (58% window AFL),
  bukan dari AFL 2:1 yang teratur. (Margin dihitung di `laporan/build_report.py`, halaman diagnosa PDF e1.)

**Uji stres takikardia (protokol butir 9).** Ambang bpm di mana model berhenti menebak Normal, per jenis variasi detak
(tanpa variasi / seragam ±10 ms / Gauss 10 ms / Gauss 20 ms). Pembanding: kontrol v7 per fold 120/120/120/115, paket v3.1 115.
Ditambah % window Normal asli yang ditebak AFIB/AFL per pita detak (OOF).

| Sistem | Ambang (bpm) | 0–100 bpm | 100–110 bpm | 110–120 bpm | ≥ 120 bpm (100 window) |
|---|---|---|---|---|---|
| v3.1 | 115 (paket) | 6,7% | 12,7% | 59,0% | 95,2% |
| BiGRU bawaan | 120 / 120 / 120 / 120 | 4,1% | 10,0% | 52,6% | 86,6% |
| TCN + bias | 120 / 125 / 130 / 130 | 3,4% | 7,7% | 43,1% | 71,8% |

**Kesimpulan**
- **TCN bawaan lebih buruk daripada v3.1 dan BiGRU.** Δ Macro-F1 vs v3.1 −0,080 (P = 0,035), vs BiGRU bawaan −0,070
  (P = 0,003). Hasilnya setara v1 (P = 0,53), jadi bukan kandidat pengganti paket produksi.
- **Tanpa bias, TCN menemukan AFL lebih banyak daripada v3.1 dan BiGRU** (recall 0,694; 08378 1,00), tetapi alarm AFL palsunya
  4,8× v3.1. Keyakinannya pada AFL 2:1 tipis (margin 1,72), jadi bias cross-fit yang menekan AFL palsu ikut menghapus 08378.
- **Ambang takikardianya tertinggi** di antara yang dibandingkan (120–130 bpm; v3.1 115, BiGRU 120). Alarm pada ritme Normal
  ≥ 120 bpm juga turun dari 95,2% (v3.1) ke 71,8%. Sebagian karena bias menekan AFL secara umum, jadi ini trade-off, bukan
  pemahaman takikardia yang lebih baik.
- Sejalan dengan AGENTS.md: mengganti arsitektur dengan input RR saja tidak menembus batas yang sama.
- **Tanpa pasien 04936** (protokol butir 10, bias dihitung ulang di 22 pasien) Macro-F1 TCN 0,671, terendah dari semua sistem
  (v3.1 0,859, BiGRU bawaan 0,721, `wp100` 0,852). Penyebabnya 08378, yang tanpa 04936 menjadi bagian terbesar AFL tersisa.
- **Yang tidak bisa disimpulkan.** Hanya setelan bawaan, tanpa tuning. Bootstrap bersarang tidak dijalankan karena TCN bukan
  kandidat. TCN dilatih di T4, BiGRU di RTX 5060; perbandingannya tetap
  sah karena OOF, fold, dan sampel bootstrap sama, tetapi sebagian kecil selisih bisa berasal dari variasi training.
- **Untuk anggota tim yang melanjutkan:** setelan baru dianggap berhasil kalau 08378 tetap terdeteksi **setelah** bias tanpa
  menambah AFL palsu (margin AFL 08378 naik). Periksa itu, bukan Macro-F1 saja.

**Lingkungan.** Google Colab Pro, GPU Tesla T4: 107 s per fold, ±45 menit untuk 25 fold, ditambah evaluasi ±3 menit; ±0,9
compute unit. Python 3.13.16, torch 2.11.0+cu130, numpy 2.1.3, scipy 1.16.3, scikit-learn 1.6.1.
Kode: commit `b18d6bc` + perubahan v11 yang belum di-commit. Karena v11 belum ada di GitHub, kode dibawa sebagai arsip base64
di satu sel tambahan di depan `jalankan.ipynb` (sisanya sama), sehingga `ringkasan.json` saat itu mencatat commit "tidak diketahui".
Data dari GitHub Release `v11-data`, dicek sidik jarinya. Alur notebook Colab diuji dulu di PC dengan `SIHEDAF_QUICK=1`
(COMPLETE). Hasil disimpan di disk sesi Colab, lalu zip-nya diunduh dan diekstrak ke `cache/v11/tcn/bawaan/`.
Evaluasi dihitung ulang di PC (numpy 2.5.4) dari prediksi yang sama, dan kedelapan bagian `ringkasan.json` identik.
Pada 3 November 2026 `jalankan.ipynb` di repo dijalankan di PC (evaluasi saja, ±3 menit, sel Kesimpulan terisi). Semua angka
identik, dan `ringkasan.json` kini mencatat lingkungan evaluasi PC (RTX 5060, commit `b18d6bc`), bukan T4.
Perbandingan dengan BiGRU dan AUROC ada di `cache/v11/tcn/banding_tcn.json`.

## Hasil 2: bobot kelas, kapasitas, dan regularisasi (2 November 2026)

**Pertanyaan.** Kelemahan TCN bawaan adalah AFL 2:1 pasien 08378 yang hilang setelah bias (margin skor AFL tipis). Apakah
setelan training memperbaikinya, dan apakah TCN yang diperbaiki setara BiGRU v11 dan v3.1? Definisi di `eksperimen.py` (`e2`),
PDF di `laporan/laporan_tcn_e2_tuning.pdf`.

Enam setelan dilatih **bersamaan di satu GPU L4** (Google Colab Pro, 12 core, 53 GB RAM, 6 proses, 68 menit, ±2,3 compute
unit), lewat salinan notebook yang menjalankan `python eksperimen.py latih e2 <setelan>` per proses. `bawaan_ulang` = setelan
bawaan yang dilatih ulang di L4, jadi referensi berasal dari mesin yang sama. TCN bawaan dari "Hasil" (T4) ikut sebagai
pembanding. `wp100` ditetapkan sebagai hipotesis utama sebelum training, karena resep ini yang memulihkan 08378 di BiGRU.

| Setelan | `SETELAN` | Param. | s / fold (6 proses berbagi GPU) |
|---|---|---|---|
| `bawaan_ulang` | `{}` | 22.659 | 161 (L4) |
| `wp100` | `{"weight_power": 1.0}` | 22.659 | 161 (L4) |
| `wp025` | `{"weight_power": 0.25}` | 22.659 | 160 (L4) |
| `k16` | `{"kanal": 16}` | 5.955 | 122 (L4) |
| `d02` | `{"dropout": 0.2}` | 22.659 | 161 (L4) |
| `dil124` | `{"dilasi": [1, 2, 4]}` | 16.323 | 130 (L4) |
| `bawaan` | `{}` | 22.659 | 107 (T4) |

| Sistem | Macro-F1 | AFL F1 | AFL R | Normal R | 08378 | tanpa 04936 | margin AFL 08378 |
|---|---|---|---|---|---|---|---|
| v3.1 | 0,832 [0,67–0,92] | 0,647 | 0,592 | 0,924 | 0,96 | 0,859 | 5,65 |
| BiGRU bawaan | 0,821 [0,68–0,87] | 0,592 | 0,544 | 0,950 | 0,74 | 0,721 | 1,82 |
| BiGRU wp100 | 0,830 [0,69–0,93] | 0,625 | 0,572 | 0,935 | 1,00 | 0,852 | 5,23 |
| **TCN bawaan_ulang** | 0,783 [0,65–0,82] | 0,482 | 0,439 | 0,959 | 0,18 | 0,703 | 1,71 |
| **TCN wp100** | 0,830 [0,67–0,91] | 0,623 | 0,591 | 0,950 | 0,92 | 0,840 | 5,13 |
| TCN wp025 | 0,785 [0,65–0,83] | 0,487 | 0,437 | 0,959 | 0,11 | 0,694 | 0,77 |
| TCN k16 | 0,812 [0,68–0,84] | 0,573 | 0,539 | 0,953 | 0,58 | 0,742 | 2,21 |
| TCN d02 | 0,770 [0,63–0,82] | 0,461 | 0,396 | 0,961 | 0,02 | 0,671 | 1,47 |
| TCN dil124 | 0,762 [0,63–0,81] | 0,420 | 0,383 | 0,956 | 0,04 | 0,709 | 1,66 |
| TCN bawaan | 0,752 [0,63–0,80] | 0,391 | 0,340 | 0,959 | 0,01 | 0,671 | 1,72 |

**Perbandingan berpasangan** (Δ Macro-F1 [CI 95%], P(Δ > 0)). Kolom terakhir: Δ Macro-F1 vs BiGRU `wp100` (P).

| Setelan | vs `bawaan_ulang` | P | Vonis | vs v3.1 (P) | vs BiGRU `wp100` (P) |
|---|---|---|---|---|---|
| `bawaan_ulang` | – | – | – | −0,049 (0,253) | −0,047 (0,255) |
| `wp100` | +0,047 [−0,01…+0,21] | 0,718 | tidak terbedakan | −0,002 (0,384) | 0,000 (0,469) |
| `wp025` | +0,002 [−0,01…+0,01] | 0,603 | tidak terbedakan | −0,046 (0,279) | −0,044 (0,262) |
| `k16` | +0,029 [0,00…+0,11] | 0,897 | tidak terbedakan | −0,020 (0,295) | −0,018 (0,297) |
| `d02` | −0,012 [−0,05…+0,01] | 0,229 | tidak terbedakan | −0,061 (0,254) | −0,059 (0,249) |
| `dil124` | −0,021 [−0,04…0,00] | 0,007 | lebih buruk | −0,069 (0,111) | −0,068 (0,206) |
| `bawaan` | −0,031 [−0,05…0,00] | 0,021 | lebih buruk | −0,080 (0,035) | −0,078 (0,070) |

**Seleksi cross-fit** (setelan tiap fold uji dipilih dari fold lain, 6 setelan): Macro-F1 0,782 [0,65–0,82], vs v3.1 −0,050 (P = 0,25). Terpilih: `wp100` 9×, `k16` 6×, `wp025` 5×, `d02` 3×, `bawaan_ulang` 2× dari 25 fold.

**Bootstrap bersarang vs v1** (bias dipilih ulang di setiap sampel): `wp100` P = 0,984, `k16` P = 0,935, `wp025` P = 0,795 (BiGRU `wp100` di eksperimen BiGRU: 0,98).

**Kesimpulan**
- **Bobot kelas penuh (`wp100`) memperbaiki TCN.** Macro-F1 0,830, setara v3.1 (−0,002, P = 0,38) dan BiGRU `wp100` (0,000, P = 0,47). 08378 kembali terdeteksi (0,18 → 0,92), tanpa 04936 0,840, dan bootstrap bersarang vs v1 P = 0,984.
- **Mekanismenya sama dengan BiGRU.** Margin skor AFL 08378 naik 1,71 → 5,13, jadi bias yang lebih kuat (median −5,25, −6,50 … −4,00, tidak di batas grid) tidak lagi menghapusnya. AUROC AFL hampir tidak berubah (0,920 → 0,925). **Yang menentukan adalah bobot kelas, bukan arsitektur**: TCN dan BiGRU dengan bobot penuh memberi Macro-F1 yang sama.
- **Setelan lain tidak menolong.** `k16` memulihkan 08378 sebagian (0,58); `wp025`, `d02` tidak berbeda; `dil124` lebih buruk (P = 0,007).
- **Variasi run besar dengan bobot lunak.** Setelan bawaan yang sama: 0,752 di T4 dan 0,783 di L4 (P = 0,02), 08378 0,01 vs 0,18. Karena itu `wp100` vs `bawaan_ulang` (+0,047) belum pasti (P = 0,72).
- **Harga bobot penuh: alarm detak cepat.** Ambang 125 / 125 / 125 / 120 bpm; alarm Normal ≥ 120 bpm 76,6% → 84,4% (BiGRU `wp100` 90,2%, v3.1 95,2%).
- **Yang tidak bisa disimpulkan.** `wp100` adalah satu dari 6 setelan, walau ditetapkan sebelum training; perkiraan jujur untuk "TCN yang di-tuning" adalah seleksi cross-fit (0,782), yang tidak konsisten memilih `wp100`. Setiap setelan dilatih sekali.
- **Untuk tim:** kandidat TCN di `gabungan.ipynb` adalah `wp100`. Tuning TCN tidak diprioritaskan lagi; InceptionTime sebaiknya
  langsung dicoba juga dengan `{"weight_power": 1.0}`.

**Lingkungan.** Training: Google Colab Pro, GPU NVIDIA L4, 12 core, 53 GB RAM, torch 2.11.0+cu130; kode v11
(commit `b18d6bc` + perubahan yang belum di-commit) dibawa sebagai arsip base64, data dari GitHub Release `v11-data`. Jumlah proses
dibatasi core **dan** RAM (`MemAvailable / 3,5 GB`). Alur notebook diuji dulu di PC dengan `SIHEDAF_QUICK=1` (COMPLETE).
Zip hasil (`cache/v11/tcn/tcn_e2.zip`, 312 file, CRC OK) diekstrak ke `cache/v11/tcn/<setelan>/`; fold dicek sama dengan fold
resmi. Evaluasi di PC: `python eksperimen.py evaluasi e2` (46 menit, 2 proses) → `cache/v11/tcn/banding_e2_tuning.json`.
PDF: `python laporan/build_report.py --extract e2` lalu `python laporan/build_report.py e2`; semua halaman dirender dan dilihat.
