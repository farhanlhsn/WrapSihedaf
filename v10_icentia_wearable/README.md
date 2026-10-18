# v10_icentia_wearable — data wearable Icentia11k untuk cabang bentuk gelombang EKG

Pertanyaan: v9 membuktikan bahwa bentuk gelombang memisahkan sinus takikardia dari AFL di Chapman (EKG klinis istirahat). Tetapi
model yang sama gagal di AFDB (Holter rawat jalan):
- AFL palsu melonjak;
- 08378 hilang;
- bias kelas melompat ke kedua batas grid.

Batasnya **pergeseran domain**. Perangkat SIHEDAF adalah wearable satu lead, jadi domain yang penting adalah rekaman rawat jalan.
**Apakah data wearable rawat jalan dari banyak pasien membuat model bentuk gelombang bekerja di AFDB, tanpa kehilangan
keunggulannya di Chapman?**

**Data baru: Icentia11k** (PhysioNet `icentia11k-continuous-ecg` 1.0)
- Rekaman patch EKG satu lead (CardioSTAT, 250 Hz) yang dipakai sehari-hari, dari 11.000 pasien. Tiap pasien punya
  sampai 50 segmen ±70 menit.
- Label ritme: Normal, AFIB, AFL. Dibuat semi-otomatis lalu diperiksa teknisi.
- **Lisensi CC BY-NC-SA 4.0: hanya untuk akademik/riset.** Model yang dilatih dengan data ini tidak boleh dipakai komersial
  tanpa izin. Sitasi: Tan dkk., *Icentia11K: An Unsupervised Representation Learning Dataset for Arrhythmia Subtype Discovery*
  (2019), dan PhysioNet (Goldberger dkk., 2000).

**Pemilihan data** (aturan `SEL` di `icentia.py`, seed tetap, ditetapkan sebelum melihat hasil model):
1. Semua file anotasi (±541 ribu segmen) dibaca untuk menghitung durasi tiap ritme per segmen. File tidak disimpan. Pembaca
   anotasi dibuktikan sama dengan `wfdb.rdann`.
2. Kelompok pasien:
   - AFL: punya episode AFL ≥ 10 s;
   - AFIB: punya episode AFIB ≥ 10 s, tanpa AFL;
   - Normal: tanpa AFIB/AFL sama sekali, dan punya ritme Normal ≥ 5 menit dalam satu segmen.
3. Semua pasien AFL dipakai, ditambah 800 pasien AFIB dan 1.200 pasien Normal acak. Segmen yang dipakai: AFL sampai 6 segmen
   (AFL terbanyak), AFIB 2, Normal 1.
4. Segmen diproses seperti AFDB:
   - front-end produksi (50 Hz, detektor R-peak v2), window 10 s langkah 5 s, label = ritme anotasi (semua detak sama);
   - potongan EKG v6 (bandpass 0,5–20 Hz), dibuktikan sejajar dengan window RR.
5. Batas per pasien: 120 window AFL, 40 AFIB, 20 Normal acak + 12 Normal tambahan ≥ 100 bpm (sinus takikardia asli saat
   beraktivitas). Window tambahan ikut dilatih dan dipakai untuk analisis per pita detak, tetapi tidak masuk metrik 3 kelas.

**Cek label (sebelum training, pelajaran v8).** AFL Icentia berbeda fenotipe dengan AFL AFDB:

| | detak median | CV IBI median | window CV > 0,10 |
|---|---|---|---|
| AFL AFDB (7 pasien) | 139 bpm | 0,03 | 18% |
| AFL Icentia (509 pasien) | 86 bpm | 0,20 | 82% |
| AFIB Icentia (729 pasien) | 75 bpm | 0,22 | 97% |

AFL Icentia kebanyakan berupa AFL dengan blok AV yang berubah-ubah. Dari interval RR saja ia mirip AFIB. Hanya 75 dari 509
pasien yang sebagian besar window AFL-nya teratur.

Label tetap dipakai karena dua alasan:
- berbeda dengan Ningbo (v8), Icentia juga memberi label AFIB terpisah dengan pola RR yang sama, jadi pelabel membedakan keduanya
  dari aktivitas atrium;
- contoh bentuk gelombang memperlihatkan gelombang atrium yang teratur.

Uji yang ditetapkan sebelum hasil: bila 10a memisahkan AFL dari AFIB di pasien Icentia uji jauh lebih baik daripada kontrol rr,
label itu memuat informasi atrium yang nyata.

| Varian | Model & data latih | Hipotesis (ditetapkan sebelum hasil) |
|---|---|---|
| `rr` kontrol | MicroTransformer RR; AFDB + Chapman + Icentia | Ambang uji stres sintetis naik (> 115 bpm); AFL AFDB tidak membaik dibanding kontrol rr v9 |
| **`dual` 10a (utama)** | CNN EKG + RR (DualBranch v6); AFDB + Chapman + Icentia | vs 9a di AFDB: Macro-F1 naik (P ≥ 0,95), AFL palsu turun, bias AFL tidak di batas grid, 08378 ≥ 0,78. Di Icentia: lebih baik dari 9a & paket v3.1. Di Chapman: alarm ST ≤ 5% |
| `dual_tanpa_afdb` 10b | DualBranch; Chapman + Icentia saja (1 ulangan × 5 fold) | Eksploratif: AFDB menjadi domain yang tidak pernah dilihat, seperti perangkat SIHEDAF nanti. Tidak lebih buruk dari v31 (P > 0,05)? |

Resep training sama dengan v9 (fungsi v6/v5 diimpor tanpa diubah). Pembagian data:
- AFDB: fold resmi 5×5;
- Chapman: fold v8;
- Icentia: fold pasien baru (diacak per kelompok, seed 2026 + 2000).

Pembanding:
- v31 (v8);
- 9a dan kontrol rr v9. Bobot model v9 dan v8 dipakai ulang untuk memprediksi pasien Icentia; pasien Icentia fold k diprediksi
  model fold k;
- paket v3.1.

**Kriteria kandidat produksi untuk 10a** (ditetapkan sebelum hasil):
1. AFDB: Macro-F1 vs v31 tidak lebih buruk (P > 0,05) dan tidak ada pasien AFL yang turun ≥ 0,20.
2. Chapman: alarm sinus takikardia ≤ separuh paket v3.1; recall AFL Shaoxing tidak lebih rendah dari paket v3.1.
3. Icentia: alarm pada window Normal ≥ 120 bpm ≤ 25%; recall AFL tidak lebih rendah dari paket v3.1.
4. Bila lolos: protokol butir 10 (bootstrap bersarang, tanpa 04936) sebelum dikemas, dan keputusan lisensi non-komersial oleh
   pemilik produk.

| File | Isi |
|---|---|
| `icentia_wearable.ipynb` | Data (AFDB, Chapman, Icentia), cek label, contoh bentuk gelombang, training, hasil AFDB / Icentia / Chapman, uji stres |
| `icentia.py` | Pemindaian anotasi, pemilihan, ekstraksi window Icentia, training (fungsi v6/v5 diimpor), evaluasi |
| `laporan/` | `laporan_icentia_wearable.pdf` + `build_report.py` |

## Hasil

**Icentia** (2.420 pasien wearable yang tidak ikut dilatih; bias dari AFDB; metrik 3 kelas pada window acak):

| | Macro-F1 | AFL recall | AFL F1 | Normal R | alarm Normal ≥ 120 bpm | AFL tidak teratur → AFL |
|---|---|---|---|---|---|---|
| paket v3.1 | 0,477 [0,47–0,49] | 0,051 | 0,097 | 0,905 | 83% | 1% |
| 9a (v9) | 0,473 | 0,027 | 0,052 | 0,952 | 16% | 1% |
| kontrol rr | 0,482 | 0,047 | 0,089 | 0,898 | 44% | 1% |
| **10a EKG + RR** | **0,666 [0,65–0,68]** | **0,359** | **0,503** | **0,972** | **13%** | **33%** |
| 10b tanpa AFDB | 0,662 | 0,338 | 0,486 | 0,974 | 13% | 30% |

**Chapman** (label andal, fold v8):

| | Macro-F1 | AFL recall (Shaoxing) | Normal R | alarm sinus takikardia | alarm SVT (bagus) |
|---|---|---|---|---|---|
| paket v3.1 | 0,520 | 0,243 | 0,865 | 42% | 99% |
| 9a (v9) | 0,739 | 0,225 | 0,993 | 2% | 56% |
| kontrol rr | 0,581 | 0,122 | 0,943 | 11% | 45% |
| **10a EKG + RR** | **0,769 [0,76–0,78]** | **0,348** | **0,994** | **1%** | 70% |
| 10b tanpa AFDB | 0,790 | 0,430 | 0,994 | 1% | 74% |

**AFDB** (protokol v3, Δ vs v31):

| | Macro-F1 | AFL recall | 08378 | 07910 | AFL palsu/ulangan | alarm AFL palsu/jam | bias AFL min…maks | Vonis |
|---|---|---|---|---|---|---|---|---|
| v31 | 0,830 [0,67–0,92] | 0,600 | 0,98 | 0,00 | 293 | 0,56 | −5,25…−3,75 | referensi |
| 9a (v9) | 0,687 | 0,404 | 0,00 | 0,07 | 2.878 | 2,66 | −7…+3 | tidak terbedakan (P = 0,07) |
| kontrol rr | 0,789 | 0,426 | 0,23 | 0,03 | 239 | 0,49 | −2,50…−0,25 | tidak terbedakan (P = 0,33) |
| 10a EKG + RR | 0,646 [0,61–0,67] | 0,421 | 0,04 | **0,39** | 8.245 | 13,5 | −7…−7 | **lebih buruk (P = 0,01)** |
| 10b tanpa AFDB | 0,627 | 0,411 | 0,05 | **0,57** | 13.084 | 26,6 | −7…−7 | lebih buruk (P = 0,006) |

10a vs 9a di AFDB: Δ Macro-F1 −0,041 (P = 0,02, lebih buruk). Kontrol rr: ambang uji stres sintetis 135–140 bpm (v31: 115).

**Eksplorasi setelah hasil** (bukan protokol, `icentia.eksplorasi_afdb`):
- grid bias yang diperlebar ke −15 tidak menolong (10a 0,645);
- AUROC AFL-lawan-lain di AFDB turun 0,937 (v31) → 0,905 (9a) → 0,830 (10a), sedangkan AUROC AFIB tetap 0,95–0,98;
- 95% AFL palsu 10a berasal dari window AFIB, 70% dari pasien 04936 dan 06995.

Jadi masalahnya pemisahan AFL dari AFIB, bukan kalibrasi.

**Kesimpulan.** Paket produksi tetap v3.1; v10 tidak dikemas.
- Data wearable membuat model bentuk gelombang jauh lebih baik di dua domain eksternal:
  - Icentia: Macro-F1 0,48 → 0,67, AFL terdeteksi untuk pertama kalinya, alarm sinus cepat 83% → 13%;
  - Chapman: lebih baik dari 9a, P = 1,00.
- Di AFDB, model ini lebih buruk. Uji label lolos: model memisahkan AFL dari AFIB di Icentia, jadi label AFL Icentia memuat
  informasi atrium yang nyata. Tetapi batas AFL/AFIB yang dipelajarinya tidak cocok dengan pelabelan AFDB. Model menyebut banyak AF
  "kasar" AFDB sebagai AFL.
- Batas proyek sekarang adalah definisi AFL vs AFIB yang berbeda antar dataset, dan belum adanya data dari perangkat SIHEDAF sendiri.
- Langkah berikutnya:
  1. validasi dengan rekaman perangkat yang dilabeli dokter;
  2. tinjauan ahli atas label AFDB 04936/06995 dan contoh AFL Icentia;
  3. keputusan produk: apakah AFL perlu dibedakan dari AFIB, atau cukup satu alarm "AF/AFL".

## Lingkungan hasil

Dijalankan 18 Oktober 2026 di **Kaggle**, dua kernel:
- **Data** (CPU, 4 core, ±35 menit): pindai anotasi 18 menit, ekstraksi window 14 menit, 10,6 GB segmen diunduh.
- **Notebook** (2× Tesla T4, Python 3.13, torch 2.11, ±5 jam):
  - uji cepat 18 menit;
  - training 3 jam 50 menit (10a ±870 s/fold, 10b ±410 s/fold, rr ±125 s/fold, dua proses paralel);
  - evaluasi 50 menit.

Kode v10 dipasang di atas commit `51287fd` (folder v10 belum di-commit saat itu). Pembanding dari dataset Kaggle privat
`farhanlhsn/sihedaf-v8-cache` dan `farhanlhsn/sihedaf-v9-cache` (OOF & bobot v9 9a/rr, bobot v31 v8). Hasil diekstrak ke
`../cache/v10/`.

## Menjalankan

Prasyarat:
- `../cache/v8/`: window Chapman, OOF dan bobot v31;
- `../cache/v9/`: OOF dan bobot 9a/rr;
- fold resmi.

Data mentah AFDB dan Chapman diunduh ulang dari mirror S3 resmi bila belum ada.

**Tahap 1, data Icentia** (sekali; butuh internet, CPU saja, ±1 jam). Hasilnya `../cache/v10/icentia_scan.npz`,
`icentia_pilihan.json`, dan `icentia_windows.npz` (±150 MB). Bagikan ketiga file ini ke anggota tim agar tidak perlu mengulang.

```python
import icentia as I          # cwd = v10_icentia_wearable
I.scan_icentia()             # baca ±23 GB anotasi (tidak disimpan)
I.select_segments()
I.build_icentia_windows()    # unduh ±8 GB segmen terpilih (dicek SHA256, dihapus setelah diproses)
```

**Tahap 2, notebook** (GPU). Data Icentia yang sudah ada dilewati.

```bash
SIHEDAF_QUICK=1 python tools/run_notebook.py v10_icentia_wearable/icentia_wearable.ipynb   # uji alur
python tools/run_notebook.py v10_icentia_wearable/icentia_wearable.ipynb                   # run penuh
```
