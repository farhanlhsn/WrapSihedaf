# v3.4 — pembersih detak: penyelarasan R-peak ke templat QRS + gerbang morfologi

> **Status: selesai (10 November 2026). Tidak direkomendasikan untuk dipasang; paket produksi tetap v3.1 dan tidak disentuh.**
> Laporan: [laporan/laporan_v3_4_pembersih_detak.pdf](laporan/laporan_v3_4_pembersih_detak.pdf). Rancangan dikunci sebelum hasil:
> [RANCANGAN.md](RANCANGAN.md) (sha256 di `../logs/v3.4_praregistrasi.log`).

## Tujuan

v3.3 menunjukkan detektor R-peak 50 Hz menjadi sumber alarm palsu saat aktivitas. Penyebab yang ditemukan: `hrv_core.detect_rpeaks`
memilih puncak `|sinyal|`, sehingga pada sebagian detak titiknya melompat dari puncak R ke lembah S. Ide perbaikannya dari pembaca
EKG di Tugas Akhir pemilik repo (`asd_ecg/measurement.py`, `_dominant_beats`): bandingkan setiap QRS dengan templat median QRS di
sekitarnya.

- **F1:** setiap R-peak dari detektor produksi digeser −3…+3 sampel ke posisi dengan korelasi tertinggi terhadap templat QRS
  median ±5 s, dengan sub-sampel dari parabola.
- **F2:** F1 + window ditolak ("sinyal buruk", tanpa alarm) bila < 60% detaknya dominan (korelasi ≥ 0,8, ukuran 0,6–1,67 kali
  templat). Ambang dari Tugas Akhir, tidak disetel.

Detektor produksi tidak diubah. Ini versi minor (perbaikan front-end). Tidak ada arsitektur atau resep RR baru: model F1 memakai
resep v3.1 persis, untuk mengisolasi efek front-end.

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Pra-registrasi, amandemen A (data eksternal, sebelum hasil), penyimpangan B (pelaporan) |
| `pembersih.py` | Front-end (bandpass yang dibuktikan identik dengan detektor, `selaraskan`, gerbang), ekstraktor AFDB tanpa wfdb, training per fold (bobot disimpan), evaluasi AFDB/aktivitas/eksternal |
| `test_pembersih.py` | 7 tes: salinan detektor = `detect_rpeaks` (sintetis + AFDB), lompatan R/S berkurang, AF tetap tidak teratur, gerbang menolak artefak |
| `eksplorasi_ektopik.py` | Analisis pasca-hasil: penyebab kegagalan MIT-BIH (detak V) dan varian F1b (eksploratif) |
| `pembersih_detak.ipynb` | Notebook utama; menulis `../cache/v3_4/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v3_4_pembersih_detak.pdf` | Laporan dua tahap (9 halaman) |

## Cara menjalankan (dari root repo)

```bash
cd v3.4_pembersih_detak
python -m pytest test_pembersih.py -q
python -c "import pembersih as P; P.bangun_afdb(); P.jalankan_semua(); P.jalankan_eksternal(workers=4)"   # bisa dilanjutkan
cd .. && python tools/run_notebook.py v3.4_pembersih_detak/pembersih_detak.ipynb
python v3.4_pembersih_detak/laporan/build_report.py --extract
python v3.4_pembersih_detak/laporan/build_report.py
```

Butuh `data/afdb`, data latihan fisik v3.3 (`data/epfl_hi`, `data/wrist`, `data/ephnogram`), `data/cinc2017`, `data/mitdb`,
`data/ltafdb`, `data/cpsc2021`, dan cache v3 (`oof_robust/`), v3.2, v3.3, v7 (`model/kontrol_r0_k*.pt`). Keluaran di `../cache/v3_4/`:
`dataset_f1.npz`, `oof/`, `model/f1_r*_k*.pt` (25 model fold, `state_dict` + `mu`/`sd`), `latihan.npz`, `eksternal/`, `ringkasan.json`.

## Hasil

**Bukti kendali mutu:** ekstraktor v3.4 dengan front-end lama = `dataset_v2_detector.npz` bit demi bit (165.978 window); uji paritas
training selisih maksimum 0,0 vs OOF MicroTransformer v3; window latihan fisik lama = v3.3; front-end lama = `signal_to_windows` di
setiap rekaman eksternal.

| Kriteria (dikunci) | Hasil | Angka |
|---|---|---|
| H1 AFDB: F1 vs v3.1, Macro-F1 dan recall AFL bukan lebih buruk | lolos | Macro-F1 0,832 → **0,842** (P = 0,96; bersarang P = 0,95, tidak terbedakan); recall AFL 0,592 → 0,606 |
| H2 alarm < 100 bpm saat aktivitas (Wrist + EPHNOGRAM) | lolos | 20,4% → 17,7% (P = 1,00) |
| H3 EPFL lolos QC; alarm ≥ 120 bpm sesudah aturan produk | lolos | 68% → 90%; 30,0% → 23,3% (P = 1,00) |
| H4 gerbang F2: alarm turun, biaya recall AFDB ≤ 0,02 | lolos | 17,7% → **12,0%**; recall AFIB −0,016, AFL 0 |
| H5 drop-in (bobot front-end lama + window F1) bukan lebih buruk | lolos | Macro-F1 +0,005 (P = 0,92) |
| H6 Se di CinC 2017, MIT-BIH, LTAFDB, CPSC 2021 bukan lebih buruk | **gagal** (F1 dan F2) | F1: MIT-BIH −0,7 poin (P = 0,038; subjek 210, 217); F2: −1,1 s.d. −3,1 poin di keempatnya |
| **Keputusan** | **F1 dan F2 tidak direkomendasikan** | |

Rincian lain:
- **Jitter vs R-peak manual:** EPFL 10,7 → 2,8 ms, Wrist 14,7 → 10,4 ms. CV IBI median EPFL 0,028 → 0,012.
- **AFDB tidak berubah secara berarti:** ketidakteraturan AFIB sama (CV 0,213 → 0,213), 98–100% window lolos gerbang.
- **Spesifisitas naik di keempat data eksternal:** F1 +1,0 s.d. +1,6 poin, F2 +1,8 s.d. +3,0 poin. Contoh alarm non-AF 100–110 bpm
  di LTAFDB: 23,5% → 16,3% (F1) → 5,3% (F2).
- **Sinus takikardia ≥ 120 bpm tetap dialarmkan** (v3.1 98,4%, + F1 97,6%, + F2 95,7%). Front-end tidak menyelesaikan masalah
  utama v3.2/v3.3.
- Uji stres sintetis, model fold F1: 120 bpm (v3.1 paket 115 bpm). AFL palsu dari Normal pasien tanpa AFL 4,0 → 15,2 per ulangan
  (kecil absolut, ±4× relatif). Episode AFL dan AFIB setara; F2 menurunkan Se episode AFIB 0,97 → 0,94.

## Kesimpulan

- Mekanisme dari Tugas Akhir terbukti: lompatan R/S adalah sumber jitter, dan penyelarasan ke templat menghapusnya tanpa merusak
  AFDB. Bobot paket v3.1 bisa dipakai ulang dengan front-end baru.
- Kriteria yang dikunci tetap tidak lolos. F1 gagal sangat tipis di MIT-BIH: dengan seri bootstrap dihitung separuh, P ≈ 0,06. Ini
  bukan bukti F1 merugikan, tetapi juga bukan bukti aman. F2 jelas menukar sensitivitas AF dengan spesifisitas.
- Langkah berikutnya: periksa window AF MIT-BIH 210/217 yang berubah keputusan. Bila ada perbaikan yang jelas (mis. tidak menggeser
  detak berkorelasi rendah), kunci rancangan baru dan uji di rekaman perangkat SIHEDAF, bukan di data publik yang sudah dipakai. Gerbang
  F2 lebih cocok sebagai keluaran "sinyal buruk" terpisah dengan ambang dari data perangkat.

## Analisis pasca-hasil (eksploratif, 10 November 2026)

`eksplorasi_ektopik.py`, halaman 8 laporan. Dilakukan **sesudah** hasil dan pada data yang sudah dipakai, jadi bukan uji independen.

- **Penyebab F1 gagal di MIT-BIH:** F1 menggeser detak ektopik ventrikel (V/PVC) ±50–60 ms ke batas lag, karena templatnya median
  detak normal. Di 9 dari 10 window AF yang hilang (210, 217) ada detak V seperti itu. Galat posisi detak V > 30 ms: 18% (lama) → 34% (F1).
- **F1b** (geser hanya detak dominan: korelasi ≥ 0,8, ukuran 0,6–1,67; satu varian): detak V 22%, Se MIT-BIH pulih (+0,13 poin,
  P = 0,58). Tetapi Se CPSC 2021 −0,19 poin dengan P = 0,005, jitter EPFL 4,7 ms (F1 2,8), alarm < 100 bpm 18,3% (F1 17,7%).
- **Pelajaran:** (1) jangan menggeser detak yang bentuknya berbeda dari templat; (2) untuk data sebesar LTAFDB/CPSC, kriteria
  "bukan lebih buruk" tanpa margin menggagalkan selisih −0,2 poin. Rancangan berikutnya memakai margin non-inferioritas yang dikunci
  (mis. −1 poin) dan diuji di data perangkat.

## Lingkungan

PC desktop, GPU NVIDIA RTX 5060 8 GB, CPU 12 core, Python 3.13.16, numpy 2.5.4, torch 2.11.0+cu128. Kode: commit `b18d6bc` + perubahan
yang belum di-commit. 10 November 2026:
- dataset AFDB 3 menit (6 proses);
- training 21 menit (paritas + 25 fold, 34–50 s/fold);
- data eksternal 5 menit (4 proses);
- notebook COMPLETE lewat `tools/run_notebook.py`, ±10 menit.
