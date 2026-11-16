# inceptiontime — InceptionTime (satu jaringan)

**Model.** Enam modul Inception. Tiap modul: bottleneck 1×1, tiga konvolusi paralel dengan kernel 3, 5, 9, dan cabang
max-pool → 1×1, digabung jadi 64 kanal (BatchNorm, ReLU). Ada shortcut residual setiap 3 modul, lalu rata-rata bermask
dan lapisan linear. ±42 ribu parameter.

Bedanya dengan InceptionTime asli (Ismail Fawaz dkk., 2020):
- **kernel diperkecil** dari 10/20/40 menjadi 3/5/9, karena window hanya 30 detak;
- **satu jaringan**, bukan ensemble 5 jaringan: ensemble 5 × 25 fold terlalu mahal. Ensemble bisa dicoba sendiri;
- semua cabang dikalikan mask, dan max-pool mengabaikan posisi padding (`tests/test_model_dl.py`).

**Input.** Urutan IBI + ΔIBI (30 × 2), z-score bermask dari data latih.

**Setelan bawaan** (`model.py`): `filter` 16, `kernel` (3, 5, 9) (harus ganjil), `kedalaman` 6, `dropout` 0, resep v3,
bobot kelas lunak.

**Waktu.** 106 s per fold di GPU RTX 5060 (25 fold ±44 menit), ditambah evaluasi ±2,5 menit. Zip hasil 15 MB.

**Ide eksperimen** (ganti `NAMA_SETELAN` tiap kali): `{"kedalaman": 3}`, `{"filter": 32}` (cek jumlah parameter),
`{"kernel": [3, 7, 11]}`, `{"dropout": 0.2}`.

## Hasil (3 November 2026)

Setelan `bawaan`, 5 ulangan × 5 fold pasien (protokol v3), seed `2026 + 100·r + k`. Pembanding v1 dan v3.1 memakai ulangan
dan sampel bootstrap yang sama. Angka dari `jalankan.ipynb` (output tersimpan) dan `cache/v11/inceptiontime/bawaan/ringkasan.json`.
`+ bias` = bias kelas cross-fit (grid −7…+3), CI 95% bootstrap pasien 2.000×.
PDF: `laporan/laporan_inceptiontime_e1_bawaan.pdf`, dibuat oleh `laporan/build_report.py` e1. Isinya ditambah
perbandingan dengan kandidat v11 dan diagnosa margin.

| Sistem | Macro-F1 | AFL F1 | AFL P | AFL R | AFIB F1 | Normal R |
|---|---|---|---|---|---|---|
| v1 (argmax) | 0,744 [0,64–0,81] | 0,399 [0,09–0,59] | 0,275 | 0,728 [0,37–0,98] | 0,898 | 0,903 [0,82–0,95] |
| v3.1 (referensi) | **0,832** [0,67–0,92] | **0,647** [0,17–0,89] | 0,715 | 0,592 [0,13–0,91] | 0,909 | 0,924 [0,84–0,97] |
| InceptionTime argmax (tanpa bias) | 0,746 [0,64–0,90] | 0,373 [0,05–0,82] | 0,281 | 0,657 [0,19–0,95] | 0,926 | 0,915 [0,85–0,96] |
| **InceptionTime + bias** (protokol) | 0,767 [0,64–0,86] | 0,446 [0,06–0,72] | 0,441 | 0,467 [0,08–0,74] | 0,914 | 0,947 [0,91–0,98] |

Bias [AFIB, AFL] median −1,75 / −4 (rentang −4,25…0 / −6,75…−1,75). Belum ada yang menyentuh batas grid, tetapi bias AFL
terendah (−6,75) sudah dekat −7.

**Perbandingan berpasangan** (Δ [CI 95%], P(Δ > 0); vonis P ≥ 0,95 lebih baik, ≤ 0,05 lebih buruk)

| Kandidat vs referensi | Δ Macro-F1 | P | Vonis | Catatan |
|---|---|---|---|---|
| InceptionTime + bias vs v3.1 | −0,065 [−0,12…−0,02] | 0,006 | lebih buruk | AFL F1 −0,201 (P = 0,002); AFL R −0,124 (P = 0,005); AFL P −0,274 (P = 0,13); Normal R +0,023 (P = 0,93) |
| InceptionTime + bias vs v1 | +0,023 [−0,01…+0,09] | 0,858 | tidak terbedakan | AFL P +0,166 (P = 0,98); AFL R −0,261 (P = 0,00); Normal R +0,044 (P = 1,00) |

**Recall AFL per pasien dan AFL palsu**

| Sistem | 04043 | 04908 | 04936 | 06426 | 07910 | 08215 | 08378 | AFL palsu / ulangan (di pasien tanpa AFL) |
|---|---|---|---|---|---|---|---|---|
| v3.1 | 0,40 | 0,92 | 0,51 | 0,17 | 0,00 | 0,00 | 0,96 | 274 (25) |
| InceptionTime argmax | 0,80 | 0,86 | 0,59 | 0,35 | 0,05 | 0,00 | 1,00 | 2.470 (155) |
| InceptionTime + bias | 0,80 | 0,60 | 0,40 | 0,07 | 0,01 | 0,00 | 0,79 | 937 (36) |

Asal AFL palsu InceptionTime + bias: 783 window dari ritme Normal dan 155 dari AFIB (v3.1: 55 dan 219).

**Uji stres takikardia (protokol butir 9).** Ambang = detak terendah di mana < 50% window sintetis ditebak Normal, per jenis
variasi (tanpa variasi / seragam ±10 ms / Gauss 10 ms / Gauss 20 ms); makin tinggi makin baik. Pembanding: kontrol v7 per fold
120/120/120/115, paket v3.1 115. Kolom pita = % window Normal asli yang ditebak AFIB/AFL (OOF).

| Sistem | Ambang (bpm) | 0–100 bpm | 100–110 bpm | 110–120 bpm | ≥ 120 bpm (100 window) |
|---|---|---|---|---|---|
| v3.1 | 115 (paket) | 6,7% | 12,7% | 59,0% | 95,2% |
| InceptionTime + bias | 120 / 130 / 130 / 125 | 4,1% | 13,7% | 44,6% | 72,2% |

**Kesimpulan**
- **InceptionTime bawaan lebih buruk daripada v3.1** (Δ Macro-F1 −0,065, P = 0,006) dan setara v1 (+0,023, P = 0,86). Ia bukan
  kandidat pengganti paket produksi.
- **Polanya sama dengan TCN bawaan, tetapi lebih ringan.** Tanpa bias, AFL 2:1 pasien 08378 terdeteksi penuh (1,00), tetapi AFL
  palsunya 2.470 window per ulangan. Bias cross-fit AFL (median −4) menekan AFL palsu ke 937 (3,4× v3.1, 84% dari ritme Normal),
  tetapi ikut menurunkan 08378 ke 0,79 dan 04908 ke 0,60. Recall AFL turun ke 0,467 (v3.1 0,592, P = 0,005). Margin skor AFL
  di 08378 per ulangan +2,5…+5,7 (dihitung terpisah dari `r{r}_k{k}.npz`). Itu lebih besar daripada TCN bawaan (1,72), tetapi
  di bawah v3.1 (5,65). Jadi bias −4 menghapus sebagian 08378, tidak seluruhnya.
- **Normal R lebih tinggi daripada v3.1** (0,947 vs 0,924), dan alarm pada Normal ≥ 120 bpm 72,2% (v3.1 95,2%; TCN bawaan 71,8%).
  Ambang takikardia 120–130 bpm. Ini trade-off: bias yang kuat menekan AFL secara umum, bukan pemahaman takikardia yang lebih baik.
- **Yang tidak bisa disimpulkan.** Hanya setelan bawaan, satu jaringan (bukan ensemble 5). Bootstrap bersarang dan sensitivitas
  tanpa 04936 tidak dijalankan karena InceptionTime bukan kandidat.
- **Untuk anggota tim yang melanjutkan:** bobot kelas penuh (`wp100`) sudah dicoba; lihat "Hasil 2". Setelan dianggap berhasil
  kalau 08378 tetap terdeteksi **setelah** bias tanpa menambah AFL palsu. Pantau juga bias AFL: kalau menyentuh −7, grid
  memotong optimum.

**Lingkungan.**
- PC desktop, GPU RTX 5060, 3 November 2026, 02:02–02:48.
- Pustaka: Python 3.13.16, torch 2.11.0+cu128, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1.
- Kode: commit `b18d6bc` + perubahan v11 yang belum di-commit.
- `jalankan.ipynb` dijalankan lewat `tools/run_notebook.py`, di dalam scope `systemd-run` dengan batas memori 5 GB (tidak
  ada OOM). Total 47 menit: training 44 menit, evaluasi ±2,5 menit.

## Hasil 2: bobot kelas penuh (3 November 2026)

Pertanyaannya: apakah bobot kelas penuh (`{"weight_power": 1.0}`, setelan `wp100`) memulihkan AFL 2:1 pasien 08378 seperti pada
BiGRU dan TCN? Kodenya `eksperimen.py` (`latih e2`, `evaluasi e2`), hasilnya di `cache/v11/inceptiontime/banding_e2_wp100.json`.
`bawaan` adalah referensi berpasangan. Keduanya dilatih di GPU yang sama (RTX 5060) dengan seed dan fold yang sama.
`eksperimen.py evaluasi e1` menghitung analisis yang sama untuk `bawaan` saja (`banding_e1_bawaan.json`).
PDF: `laporan/laporan_inceptiontime_e2_wp100.pdf` (`laporan/build_report.py` e2).

| Sistem (+ bias) | Macro-F1 | AFL F1 | AFL P | AFL R | AFIB F1 | Normal R |
|---|---|---|---|---|---|---|
| v3.1 (referensi) | **0,832** [0,67–0,92] | **0,647** [0,17–0,89] | 0,715 | 0,592 [0,13–0,91] | 0,909 | 0,924 [0,84–0,97] |
| InceptionTime `bawaan` | 0,767 [0,64–0,86] | 0,446 [0,06–0,72] | 0,441 | 0,467 [0,08–0,74] | 0,914 | 0,947 [0,91–0,98] |
| InceptionTime `wp100` | 0,766 [0,64–0,92] | 0,444 [0,06–0,88] | 0,427 | 0,527 [0,10–0,92] | 0,914 | 0,943 [0,90–0,98] |
| `wp100` argmax (tanpa bias) | 0,715 [0,62–0,85] | 0,288 [0,03–0,68] | 0,187 | 0,682 [0,28–0,97] | 0,924 | 0,898 [0,82–0,96] |

**Perbandingan berpasangan** (Δ [CI 95%], P(Δ > 0); vonis P ≥ 0,95 lebih baik, ≤ 0,05 lebih buruk)

| Kandidat vs referensi | Δ Macro-F1 | P | Vonis | Catatan |
|---|---|---|---|---|
| `wp100` vs `bawaan` | −0,001 [−0,01…+0,06] | 0,53 | tidak terbedakan | AFL R +0,059 (P = 1,00); AFL P −0,014 (P = 0,31); Normal R −0,004 (P = 0,11) |
| `wp100` vs v3.1 | −0,066 [−0,12…+0,01] | 0,089 | tidak terbedakan | AFL F1 −0,203 (P = 0,06); AFL R −0,065 (P = 0,14) |
| `wp100` vs v1 | +0,022 [−0,02…+0,14] | 0,81 | tidak terbedakan | AFL R −0,201 (P = 0,00); Normal R +0,040 (P = 1,00) |
| `wp100` vs BiGRU `wp100` | −0,064 [−0,13…+0,00] | 0,028 | lebih buruk | AFL F1 −0,181 (P = 0,01); AFL R −0,046 (P = 0,00) |
| `wp100` vs TCN `wp100` | −0,064 [−0,11…+0,01] | 0,093 | tidak terbedakan | AFL F1 −0,179 (P = 0,11) |
| `bawaan` vs BiGRU `wp100` | −0,063 [−0,13…−0,02] | 0,001 | lebih buruk | |
| `bawaan` vs TCN `wp100` | −0,063 [−0,10…−0,02] | 0,002 | lebih buruk | |

Bootstrap bersarang (bias dipilih ulang di setiap sampel): `wp100` 0,766 [0,64–0,92], vs v1 Δ +0,022, P = 0,69.

**Recall AFL per pasien (+ bias) dan AFL palsu**

| Sistem | 04043 | 04908 | 04936 | 06426 | 07910 | 08215 | 08378 | AFL palsu / ulangan (di pasien tanpa AFL) |
|---|---|---|---|---|---|---|---|---|
| v3.1 | 0,40 | 0,92 | 0,51 | 0,17 | 0,00 | 0,00 | 0,96 | 274 (25) |
| `bawaan` | 0,80 | 0,60 | 0,40 | 0,07 | 0,01 | 0,00 | 0,79 | 937 (36) |
| `wp100` | 0,60 | 0,67 | 0,40 | 0,11 | 0,01 | 0,00 | **0,99** | 1.288 (26) |

Asal AFL palsu `wp100`: 1.105 window dari ritme Normal dan 183 dari AFIB (`bawaan`: 783 dan 155).

**Diagnosa: keyakinan AFL dan bias.** Margin skor AFL = log p(AFL) − log p kelas lain terbesar, satuannya sama dengan bias.
Dihitung median di window AFL dan persentil 99,5 di window non-AFL (calon AFL palsu), definisi sama dengan laporan TCN.

| Sistem | AUROC AFL | Bias AFL median [min…maks] | Margin 08378 | Margin 04936 | Margin non-AFL (p99,5) |
|---|---|---|---|---|---|
| v3.1 | 0,929 | – | 5,65 | 4,60 | 2,54 |
| BiGRU `wp100` | 0,920 | – | 5,23 | 3,42 | 2,47 |
| TCN `wp100` | 0,925 | – | 5,13 | 4,83 | 2,96 |
| InceptionTime `bawaan` | 0,923 | −4 [−6,75…−1,75] | 2,98 | 2,34 | 2,72 |
| InceptionTime `wp100` | 0,928 | **−6 [−7…−3,75]** | 5,33 | 2,75 | **5,36** |

**Bias AFL `wp100` menyentuh batas grid −7 di 8 dari 25 fold** (32%; `di_batas` di `ringkasan.json` = 16% karena dihitung dari
bias AFIB dan AFL bersama). Ini tanda bahaya (AGENTS.md). `bawaan`: 0.

**Uji stres takikardia** (ambang per jenis variasi; % window Normal asli ditebak AFIB/AFL per pita)

| Sistem | Ambang (bpm) | 0–100 bpm | 100–110 bpm | 110–120 bpm | ≥ 120 bpm |
|---|---|---|---|---|---|
| v3.1 | 115 (paket) | 6,7% | 12,7% | 59,0% | 95,2% |
| `bawaan` | 120 / 130 / 130 / 125 | 4,1% | 13,7% | 44,6% | 72,2% |
| `wp100` | 125 / 130 / 130 / 120 | 4,3% | 16,9% | 45,1% | 76,0% |

**Tanpa pasien 04936** (protokol butir 10, bias dihitung ulang di 22 pasien, 6 pasien AFL), Macro-F1: v3.1 0,859, BiGRU `wp100`
0,852, TCN `wp100` 0,840, `wp100` 0,785, `bawaan` 0,749. v3.1 − `wp100` = +0,074 (P = 0,75, tidak terbedakan); v3.1 − `bawaan`
= +0,110 (P = 0,98).

**Kesimpulan**
- **Bobot penuh memulihkan AFL 2:1 pasien 08378** (0,79 → 0,99; margin 2,98 → 5,33, setara BiGRU/TCN `wp100` dan v3.1). Recall AFL
  naik +0,059 (P = 1,00).
- **Tetapi Macro-F1 tidak berubah** (0,767 → 0,766, P = 0,53), karena InceptionTime `wp100` juga makin yakin AFL di window
  non-AFL. Margin non-AFL p99,5 naik 2,72 → 5,36, dua kali BiGRU/TCN `wp100` (2,5–3,0). Akibatnya:
  - bias AFL harus lebih kuat (median −6), dan 8 dari 25 fold menempel di batas grid −7;
  - AFL palsu naik 937 → 1.288 per ulangan; kenaikannya hampir semuanya dari ritme Normal (783 → 1.105).
  - (Recall 04043 turun 0,80 → 0,60, tetapi pasien ini hanya punya 1 window AFL, jadi itu tidak bermakna.)
- **Pelajaran BiGRU/TCN ("bobot kelas yang menentukan, bukan arsitektur") hanya berlaku sebagian di sini.** Bobot penuh menaikkan
  keyakinan AFL yang benar, sama seperti di BiGRU/TCN. Namun hanya InceptionTime yang ikut menaikkan keyakinan AFL yang salah. Dugaan
  yang belum diuji: modul Inception (kernel 3/5/9 + max-pool) lebih mudah mengenali "urutan teratur dan cepat" tanpa membedakan
  AFL 2:1 dari sinus yang cepat.
- **Tidak terbedakan dari v3.1** (P = 0,089; tanpa 04936 P = 0,75), tetapi lebih buruk daripada BiGRU `wp100` (P = 0,028). Bukan
  kandidat; kandidat v11 tetap BiGRU dan TCN `wp100`.
- **Yang tidak bisa disimpulkan.** Bias di batas grid berarti grid −7…+3 mungkin memotong optimum `wp100`. Grid itu bagian dari
  protokol, jadi tidak diperlebar di sini. Hanya dua setelan; kapasitas, dropout, dan kernel belum dicoba.

**Lingkungan.**
- PC desktop, GPU RTX 5060, 3 November 2026, 10:41–11:37.
- Pustaka sama dengan `bawaan`.
- `eksperimen.py latih e2` lalu `evaluasi e2` di scope `systemd-run` (batas memori 5 GB, tidak ada OOM). Training 107 s per fold
  (45 menit), evaluasi 10,5 menit (4 proses).
