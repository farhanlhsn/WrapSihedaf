# minirocket — MiniRocket + regresi logistik

**Model.** MiniRocket (Dempster dkk., 2021; implementasi pustaka `aeon`) membuat ribuan konvolusi acak berbobot tetap
pada urutan IBI + ΔIBI. Dari tiap konvolusi diambil proporsi nilai positif (PPV) sebagai fitur. Fitur itu
distandardisasi, lalu dipisahkan oleh regresi logistik dengan bobot sampel lunak. Bagian yang "dilatih" hanya
regresi logistiknya, jadi cocok dijalankan di CPU.

**Input.** Urutan IBI + ΔIBI (2 kanal × 30 detak), z-score bermask dari data latih, sehingga posisi padding selalu 0.
MiniRocket tidak mendukung panjang berbeda, jadi urutan berpadding dipakai apa adanya.

**Setelan bawaan** (`model.py`):
- `kernel` 2.016, dibulatkan ke kelipatan 84 oleh aeon. Bawaan aslinya 10.000; dikurangi agar matriks fitur
  (±130 ribu × 2.016 float32 ≈ 1 GB) muat di RAM;
- `C` 1,0, `max_iter` 1.000. Peringatan konvergensi dicatat di `info`;
- fitur dihitung per batch 8.192 window.

**Pemasangan aeon.** aeon membatasi versi numpy/scipy, jadi **tidak** masuk `requirements.txt` utama.
- Kaggle/Colab: dipasang otomatis oleh sel pertama notebook (`requirements-minirocket.txt`).
- PC: `pip install -r requirements-minirocket.txt`. Kalau itu akan menurunkan numpy/scipy lingkungan utama, buat venv
  terpisah, daftarkan kernel Jupyter-nya, lalu jalankan dengan `SIHEDAF_KERNEL=<nama kernel>`.
- Berkas itu juga membatasi `numpy<2.4`. aeon 1.6.0 mengizinkan numpy < 2.5 dan numba < 0.64, padahal numba stabil di bawah 0.64
  hanya mendukung numpy < 2.4. Tanpa batas ini, resolver memilih numpy 2.4 + numba 0.63.0b1 (beta), dan `latih()` gagal dengan
  `AttributeError: module 'numpy' has no attribute 'trapz'`.
- Teruji 23 Oktober 2026 (venv terpisah, Python 3.13): aeon 1.6.0, numpy 2.3.5, numba 0.63.1, scipy 1.17.1, scikit-learn 1.9.1,
  torch 2.14.1+cpu. Seluruh tes v11 lolos (74), dan uji alur `SIHEDAF_QUICK=1` COMPLETE.

**Waktu.** 211 s per fold di CPU 12 core (25 fold ±88 menit), ditambah evaluasi ±2,5 menit. Zip hasil 13 MB. Uji alur QUICK
(84 kernel, 3.000 window, 5 fold): 51 detik untuk seluruh notebook.

**Ide eksperimen** (ganti `NAMA_SETELAN` tiap kali): `{"kernel": 5040}` (cek RAM), `{"C": 0.1}`, `{"max_iter": 3000}`.

## Hasil (3 November 2026)

Setelan `bawaan`, 5 ulangan × 5 fold pasien (protokol v3), seed `2026 + 100·r + k`. Pembanding v1 dan v3.1 memakai ulangan
dan sampel bootstrap yang sama. Angka dari `jalankan.ipynb` (output tersimpan) dan `cache/v11/minirocket/bawaan/ringkasan.json`.
`+ bias` = bias kelas cross-fit (grid −7…+3), CI 95% bootstrap pasien 2.000×.

| Sistem | Macro-F1 | AFL F1 | AFL P | AFL R | AFIB F1 | Normal R |
|---|---|---|---|---|---|---|
| v1 (argmax) | 0,744 [0,64–0,81] | 0,399 [0,09–0,59] | 0,275 | 0,728 [0,37–0,98] | 0,898 | 0,903 [0,82–0,95] |
| v3.1 (referensi) | **0,832** [0,67–0,92] | **0,647** [0,17–0,89] | 0,715 | 0,592 [0,13–0,91] | 0,909 | 0,924 [0,84–0,97] |
| MiniRocket argmax (tanpa bias) | 0,723 [0,62–0,79] | 0,343 [0,05–0,53] | 0,287 | 0,444 [0,15–0,59] | 0,900 | 0,907 [0,84–0,95] |
| **MiniRocket + bias** (protokol) | 0,709 [0,62–0,76] | 0,306 [0,04–0,46] | 0,331 | 0,296 [0,07–0,45] | 0,897 | 0,905 [0,83–0,95] |

Bias [AFIB, AFL] median 0 / −2,5 (rentang −0,5…+1,25 / −7…0). Satu dari 50 nilai bias (2%) menyentuh batas grid −7, jadi ada
tanda bahaya (AGENTS.md, pelajaran v2/v9). Bias juga menurunkan Macro-F1 (0,723 → 0,709), seperti pada XGBoost dan LightGBM v3.

**Perbandingan berpasangan** (Δ [CI 95%], P(Δ > 0); vonis P ≥ 0,95 lebih baik, ≤ 0,05 lebih buruk)

| Kandidat vs referensi | Δ Macro-F1 | P | Vonis | Catatan |
|---|---|---|---|---|
| MiniRocket + bias vs v3.1 | −0,122 [−0,25…−0,03] | 0,000 | lebih buruk | AFL F1 −0,342 (P = 0,002); AFL P −0,384 (P = 0,004); AFL R −0,296 (P = 0,02); Normal R −0,019 (P = 0,00) |
| MiniRocket + bias vs v1 | −0,034 [−0,15…+0,03] | 0,212 | tidak terbedakan | AFL F1 −0,093 (P = 0,24); AFL R −0,432 (P = 0,00); Normal R +0,002 (P = 0,64) |

**Recall AFL per pasien dan AFL palsu**

| Sistem | 04043 | 04908 | 04936 | 06426 | 07910 | 08215 | 08378 | AFL palsu / ulangan (di pasien tanpa AFL) |
|---|---|---|---|---|---|---|---|---|
| v3.1 | 0,40 | 0,92 | 0,51 | 0,17 | 0,00 | 0,00 | 0,96 | 274 (25) |
| MiniRocket argmax | 1,00 | 0,94 | 0,56 | 0,20 | 0,03 | 0,00 | 0,28 | 1.359 (319) |
| MiniRocket + bias | 0,80 | 0,86 | 0,39 | 0,12 | 0,01 | 0,00 | 0,11 | 796 (145) |

Asal AFL palsu MiniRocket + bias: 562 window dari ritme Normal dan 235 dari AFIB (v3.1: 55 dan 219). AFL palsu di pasien tanpa
AFL 145 per ulangan, 5,7× v3.1.

**Uji stres takikardia (protokol butir 9).** Ambang = detak terendah di mana < 50% window sintetis ditebak Normal, per jenis
variasi (tanpa variasi / seragam ±10 ms / Gauss 10 ms / Gauss 20 ms); makin tinggi makin baik. Pembanding: kontrol v7 per fold
120/120/120/115, paket v3.1 115. Kolom pita = % window Normal asli yang ditebak AFIB/AFL (OOF).

| Sistem | Ambang (bpm) | 0–100 bpm | 100–110 bpm | 110–120 bpm | ≥ 120 bpm (100 window) |
|---|---|---|---|---|---|
| v3.1 | 115 (paket) | 6,7% | 12,7% | 59,0% | 95,2% |
| MiniRocket + bias | 130 / 135 / 130 / 120 | 8,1% | 18,8% | 78,2% | 92,4% |

**Konvergensi.** Regresi logistik belum konvergen di **11 dari 25 fold** (`max_iter` 1.000): r0 k1–k2, r1 k0 dan k4, r2 k0 dan
k4, r3 k0–k2, r4 k0 dan k3. Status per fold ada di `info` tiap `r{r}_k{k}.npz`. `info` di `ringkasan.json` (dan sel terakhir
notebook) hanya memuat fold terakhir, yang kebetulan konvergen, jadi jangan dibaca sebagai "semua konvergen".

**Diagnosa 08378** (dihitung terpisah dari `r{r}_k{k}.npz`, belum masuk notebook). Pasien 08378 ada di satu fold uji per
ulangan. Recall argmax per ulangan 0,08 / 0,20 / 0,68 / 0,30 / 0,12. Margin skor AFL (median log p(AFL) − log p kelas lain
terbesar, satuannya sama dengan bias) −7,6 / −4,3 / +2,8 / −3,0 / −8,1. Pembanding: InceptionTime +2,5…+5,7 (argmax 1,00 di
semua ulangan); dari `tcn/README.md`, TCN bawaan 1,72 dan v3.1 5,65. Fold 08378 yang belum konvergen (r0, r3, r4) memberi
0,08 / 0,30 / 0,12. Yang konvergen memberi 0,20 (r1) dan 0,68 (r2), jadi konvergensi tidak menjelaskan semuanya.

**Kesimpulan**
- **MiniRocket bawaan adalah model v11 terburuk.** Ia lebih buruk daripada v3.1 di Macro-F1 (Δ −0,122, P = 0,000), semua metrik
  AFL, dan Normal R, serta setara v1 (P = 0,21). Ia bukan kandidat pengganti paket produksi.
- **Kehilangan AFL 2:1 (08378) bukan karena bias.** Pada TCN dan InceptionTime, model yakin pada AFL lalu bias menghapusnya.
  MiniRocket sudah tidak yakin sejak awal: margin negatif di 4 dari 5 ulangan, recall argmax 0,28. Setelah bias, recall 0,11.
- **Hasilnya tidak stabil antar ulangan.** Recall 08378 berkisar 0,08–0,68, dan 11 fold belum konvergen. Penyebabnya belum
  diketahui. Dugaan yang belum diuji:
  - optimasi belum selesai (`max_iter`);
  - jumlah kernel dipotong dari 10.000 ke 2.016;
  - fitur PPV dari urutan yang sudah di-z-score kurang peka terhadap detak yang cepat dan sangat teratur.
- **Takikardia: ambang sintetis menyesatkan bila dibaca sendiri.** Ambangnya 120–135 bpm, lebih tinggi dari v3.1. Namun pada window
  Normal asli, alarm MiniRocket lebih tinggi daripada v3.1 di tiga pita di bawah 120 bpm, terutama 110–120 bpm (78,2% vs 59,0%).
  Ritme sintetis teratur, sedangkan Normal asli punya ektopik dan variasi yang tidak ada di uji stres.
- **Yang tidak bisa disimpulkan.** Hanya setelan bawaan. Bootstrap bersarang dan sensitivitas tanpa 04936 tidak dijalankan karena
  MiniRocket bukan kandidat.
- **Untuk anggota tim yang melanjutkan:**
  - urutan yang disarankan: `{"max_iter": 3000}` dulu (konvergensi), lalu `{"kernel": 5040}` (cek RAM);
  - model ini belum punya kunci `weight_power`, karena `latih()` memanggil `D.bobot_sampel(data.y)` dengan pangkat bawaan 0,5.
    Untuk mencoba bobot penuh seperti BiGRU/TCN `wp100`, tambahkan kunci itu di `model.py` (pola `xgboost/model.py`);
  - ukur keberhasilan dari recall 08378 **setelah** bias dan dari margin AFL-nya, bukan Macro-F1 saja.

**Lingkungan.**
- PC desktop, CPU 12 core (GPU tidak dipakai), 3 November 2026, 02:48–04:19.
- Venv terpisah `sihedaf-minirocket`: Python 3.13.16, aeon 1.6.0, numpy 2.3.5, scipy 1.17.1, scikit-learn 1.9.1, torch 2.15.0.
- Kode: commit `b18d6bc` + perubahan v11 yang belum di-commit.
- `jalankan.ipynb` dijalankan lewat `tools/run_notebook.py` dengan Python venv itu (kernel `python3` venv tersebut), di dalam
  scope `systemd-run` dengan batas memori 7 GB (tidak ada OOM). Total 91 menit: training 88 menit, evaluasi ±2,5 menit.
