# v3_evaluasi_robust

## Purpose
Protokol evaluasi robust (fold pasien berulang, CI bootstrap pasien, perbandingan berpasangan) untuk kandidat
model v2, plus pengukuran ulang v1 dengan protokol yang sama. Folder ini tidak melatih atau mengekspor model
produksi.

## Entry Points
- `evaluasi_robust.ipynb` — evaluasi 4 kandidat × 5 ulangan × 5 fold (cwd = folder ini).
- `sihedaf_cv.py` — semua utilitas: `repeated_patient_folds`, `train_dl`, `run_dl_job` (resumable),
  `crossfit_bias`, `patient_cms`, `bootstrap_weights`, `boot_metrics`.
- `eval_versi_lama.py` — latih ulang resep v1 pada fold yang sama, uji di 3 kondisi (lab / detektor / front-end lama).
- `laporan/build_report.py` — laporan PDF v3. **Template untuk laporan versi berikutnya.**

## Contracts & Invariants
- OOF disimpan di `../cache/oof_robust/detector_e30_r5_s2026/{Model}_r{ulangan}.npz` (`oof` (N,3), `times`)
  dan `versi_lama/oof_r{ulangan}.npz` (`lab`, `det`, `fe`). Nama tag = `{sumber}_e{epoch}_r{ulangan}_s{seed}`;
  konfigurasi lain menghasilkan folder lain, jadi hasil tidak tercampur.
- `folds.npz` adalah pembagian pasien resmi. Eksperimen versi berikutnya **memakai file ini**, tidak membuat ulang.
- Seed latih fold *k* ulangan *r* = `2026 + 100·r + k`. Hiperparameter DL sama dengan v2.
- Referensi perbandingan = `MicroTransformer +bias` (= prosedur paket produksi v3.1). Bootstrap pakai `seed=2026`, `n_boot=2000`,
  strata pasien dari `patient_strata`, supaya angka laporan dapat direproduksi persis.
- `build_report.py --extract` meng-`assert` angka utama sama dengan `ringkasan.csv` notebook. Jika assert gagal,
  notebook dan cache tidak sinkron; jalankan notebook ulang, jangan melonggarkan assert.

## Patterns
Menambah kandidat model baru (di folder versi baru):
1. Tulis OOF ke folder tag yang sama dengan format `{Model}_r{r}.npz`, memakai `folds.npz` dan seed di atas.
2. Muat bersama OOF yang ada, lalu hitung `patient_cms` → `boot_metrics` dengan bobot bootstrap yang sama.
3. Laporkan Δ berpasangan terhadap referensi + recall AFL per pasien.

## Anti-patterns
- Memakai `StratifiedGroupKFold(shuffle=True)` untuk "ulangan" (partisinya identik).
- Membandingkan model dari CI tunggal yang tumpang tindih. Gunakan Δ berpasangan.
- Menjalankan beberapa proses GPU paralel untuk model kecil (lebih lambat, lihat log v3).
