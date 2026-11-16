# Paper 2: uji stres sinus takikardia (draf, ICIMPACT 2027)

*Sinus Tachycardia Is a Blind Spot of RR-Interval Atrial Fibrillation and Flutter Classifiers Trained on the MIT-BIH AFDB: A Stress
Test with Synthetic, Clinical and Exercise ECG.* Isi: C4 dari [../kerangka_paper_evaluasi.md](../kerangka_paper_evaluasi.md); aturan
anti tumpang tindih dengan paper 1 ada di [../rencana_dua_paper.md](../rencana_dua_paper.md). Venue: **ICIMPACT 2027**, sama dengan paper 1 (dipilih 6 November 2026 setelah EMBC dibatalkan
karena biaya). Tenggat EDAS **30 November 2026**. Aturannya: template IEEE, maks. 6 halaman termasuk referensi, review blind.

| Berkas | Isi |
|---|---|
| `main.tex` | Naskah (IEEEtran conference, A4, bahasa Inggris). Semua angka lewat makro dari `angka.tex`. `\blindtrue` = versi anonim untuk submisi |
| `angka.py` | Menghitung semua angka dari cache v3.2, v3.3, v7, v8, v11 dengan `assert` terhadap ringkasan tiap versi, lalu menulis `angka.tex` (makro + baris Tabel I), `angka.json`, dan `gambar/*.pdf|png` |
| `cek_gelombang_p.py` | Sampel acak 20 strip EKG latihan ≥ 120 bpm (seed 2026, satu per subjek) untuk cek visual gelombang P; gambar ke `cache/publikasi/paper2/gelombang_p/` |
| `penilaian_gelombang_p.json` | Arsip penilaian visual awal untuk pemeriksaan eksploratif |
| `refs.bib` | Referensi yang dipakai naskah: `refs_kandidat.bib` + `domazetoski2022` (paper 1) + `anon_paper1` |
| `refs_kandidat.bib`, `catatan_literatur.md` | 48 entri terverifikasi (Crossref/PubMed/arXiv/FDA/IFU) dan catatan apa yang tiap sumber dukung, termasuk koreksi klaim lama |
| `IEEEtran.cls`, `IEEEtran.bst` | Dari CTAN (V1.8b), sama dengan paper 1 |

## Membangun ulang (dari root repo)

```bash
python publikasi/paper2/angka.py
cd publikasi/paper2 && latexmk -pdf -outdir=_build main.tex && cp _build/main.pdf paper2_draft.pdf
```

Butuh cache `v3_2`, `v3_3` (termasuk `jendela.npz`, `prediksi.npz`), `v7`, `v8`, `v11`, `v13`, `v16.1`, `v16.5`,
`oof_robust`, `dataset_v2_detector.npz`, dan
`v3_evaluasi_robust/laporan/_build/tables.json`.

## Revisi 8 November 2026

- Hasil utama adalah 99,2% alarm P2 pada 1.743 EKG sinus takikardia klinis berlabel dokter. Hasil latihan dilaporkan
  sebagai alarm pada EKG latihan ≥ 120 bpm karena label ritmenya tidak tersedia.
- `angka.py` memasangkan window EPFL dan Wrist berdasarkan dataset, subjek, rekaman, dan waktu mulai. Di kedua mode,
  P2 mengalarmkan seluruh 192 window berpasangan dari 24 subjek yang lolos penyaringan dan mencapai ≥ 120 bpm.
- Klaim tentang data latih ditulis sebagai interpretasi hasil. Trade-off recall flutter 2:1 dibatasi pada rekaman 08378.
- Penilaian awal gelombang P dipindahkan dari naskah ke catatan eksploratif di bawah.

## Cek visual gelombang P (penilaian awal, 6 November 2026)

Asisten AI melakukan penilaian awal ini. Hasilnya memerlukan pemeriksaan klinisi sebelum dapat dipakai sebagai bukti.
Untuk membuat ulang gambar strip, jalankan `python publikasi/paper2/cek_gelombang_p.py` dengan data di `data/epfl_hi`,
`data/wrist`, dan `data/ephnogram`.

| Kategori | Strip |
|---|---|
| Gelombang P jelas di depan setiap QRS | #4, #9, #10, #13, #18 |
| Kemungkinan ada P (kecil atau menyatu dengan T) | #1, #7, #14, #15, #19 |
| Tidak bisa dinilai | #2, #3, #6, #11, #12, #17 (noise gerak/baseline); #5, #8, #16, #20 (fusi P–T pada detak tinggi) |
| Gelombang flutter atau RR tidak teratur | tidak ada |

Pada 6 dari 8 strip EPHNOGRAM, penilaian awal menandai dugaan posisi R-peak pada gelombang T atau di awal QRS.

## Status (6 November 2026)

- Draf sesuai aturan ICIMPACT: 5 halaman A4 termasuk 26 referensi (batas 6), anonim, 3 gambar, 1 tabel. Kompilasi tanpa warning;
  setiap halaman sudah dirender dan diperiksa.
- Tumpang tindih dengan paper 1 dikurangi:
  - kolom Macro-F1 AFDB per model dibuang dari Tabel I;
  - paper 1 dan paper 2 saling mensitasi secara anonim (`anon_paper1` / `anon_paper2`).
- Data tambahan: sinus takikardia klinis berlabel dokter dari Chapman–Shaoxing dan Ningbo (1.743 EKG ≥ 120 bpm).

*(Butir terakhir di atas terpotong pada berkas lama dan direkonstruksi dari naskah pada 13 November 2026.)*

## Pemeriksaan pra-submisi (13 November 2026)

- `angka.py` dijalankan ulang dari cache. Kolom "AFDB Normal" di Tabel I pada `angka.tex` lama masih berisi Macro-F1 mentah
  (mis. 0.9953945943432708), padahal headernya "% alarm ≥ 120 bpm". Kodenya sudah dibetulkan 8 November, tetapi `angka.tex` belum
  dibuat ulang. Sekarang kolom itu benar (P2: 95%) dan sesuai dengan teks bagian III-A.
- Semua sitasi yang dikutip dicek otomatis: DOI lewat CrossRef/DataCite; dokumen FDA dan IFU Kardia bisa dibuka (HTTP 200). Hasilnya
  tidak ada referensi palsu. Tahun Jarchi (2017 di bib, 2016 di CrossRef) sesuai sitasi resmi MDPI *Data* 2017, 2, 1.
- Penilaian gelombang P tidak ada di naskah lagi (lihat catatan eksploratif di atas), jadi tidak menghambat submisi.
- Kompilasi: **4 halaman** A4 termasuk 26 referensi (batas 6), anonim, log tanpa warning. Semua halaman dirender dan dilihat.
- Sisa (keputusan pemilik repo): daftar penulis dan afiliasi, pernyataan kode, review pembimbing, apakah sisa 2 halaman dipakai,
  unggah ke EDAS dengan paper 1 disebut sebagai submisi terkait.

## Tambahan mitigasi eksploratif (14 November 2026)

Atas keputusan pemilik repo (13 Nov), sisa halaman dipakai untuk bagian **Exploratory mitigation** (Metode II-D, Hasil III-E,
Tabel II, satu paragraf diskusi, batasan, abstrak, dan kesimpulan). Naskah jadi **5 halaman** dengan 30 referensi.

- **Sistem:** P2 dibandingkan dengan gerbang ECGFounder 1-lead zero-shot **D-0** (v16.1: AF bila pAF + pAFL > pSinus, ambang bawaan,
  tanpa parameter yang disetel), plus aturan episode AF ≥ 30 s (v16.5). D-1 (t = −1) sengaja tidak dipakai: ambangnya dipilih setelah
  melihat kurva yang memuat data latihan fisik yang sama (v16.1).
- **Angka** dihitung `kumpulkan_fondasi()` di `angka.py` dari cache v16.1/v16.5/v13. Assert yang dipasang:
  - alarm P2 latihan sama dengan Tabel I;
  - metrik episode D-1 dan v3.1 yang dihitung ulang lewat `v16.5_episode_30s/episode.py` sama dengan cache v16.5 (latihan dan AFDB);
  - alarm palsu per jam D-0 tanpa aturan sama dengan v16.1.
- **Angka baru** (belum ada di cache mana pun): D-0 + 30 s memberi 0,0 alarm palsu/jam pada 30 jam latihan (52 subjek; P2 + 30 s: 8,8),
  dan di AFDB Se episode ≥ 30 s 0,866 (P2 0,947), alarm palsu 0,08/jam (P2 1,59).
- **Biaya yang ditulis jujur:**
  - Se window AFDB 0,893 vs 0,934 (Δ −0,040 [−0,086; −0,002]);
  - 34% window AFL 2:1 di 08378 disebut sinus;
  - Se Icentia11k 0,767 vs 0,871.
- **Pembanding klinis gerbang:** PTB-XL STACH (811 rekaman), bukan Chapman. EKG Chapman tidak tersimpan dalam bentuk gelombang 50 Hz,
  jadi ECGFounder belum pernah dijalankan di sana.
- **Referensi baru** (DOI/arXiv dicek di CrossRef dan API arXiv):
  - Li 2025 (ECGFounder, NEJM AI);
  - Wagner 2020 (PTB-XL);
  - Van Gelder 2024 (pedoman ESC, definisi AF ≥ 30 s);
  - Tan 2019 (Icentia11k).
- **Catatan:** temuan ECGFounder dan aturan 30 s kini "terpakai" di paper 2. Paper lanjutan tentang D-1/S1 harus mensitasi paper ini
  dan menonjolkan hasil v16.2–v16.7 (data yang tidak dipakai di sini).

## Pemeriksaan ulang (14 November 2026, sore)

- **Koreksi angka 08378 di bagian mitigasi.** Baris Tabel II "08378 (2:1 flutter): windows alarmed" (P2 0,93, gerbang 0,66) dan
  kalimat "missed 34% of 2:1 flutter windows" memakai `per_pasien[...]["08378"]["se"]` v16.1, yang dihitung atas **semua**
  window AF + AFL rekaman 08378 (1.532 AF + 306 AFL), bukan flutter 2:1 saja. Pada 306 window AFL 08378, gerbang D-0 mengalarmkan
  **0/306** (P2: 1,00); pada seluruh 1.152 window AFL AFDB, gerbang 15% vs P2 85%. `angka.py` kini menghitungnya per kelas dari
  `cache/dataset_v2_detector.npz` + keputusan D-0/P2 v16.5, dengan assert paritas terhadap angka AF + AFL v16.1. Abstrak, hasil
  III-E, diskusi, dan kesimpulan disesuaikan (temuan ini sejalan dengan v16: ECGFounder 50 Hz menyebut 85% window AFL sinus).
- Sitasi dicek ulang: 29 entri yang dikutip (tanpa `anon_paper1`) cocok dengan CrossRef/DataCite (judul, penulis pertama, tahun,
  volume, halaman) atau API arXiv; dokumen FDA/IFU sudah dicek sebelumnya. Rujukan Abbasi 2026 kini ditulis sesuai isinya
  (deteksi AF AI-ECG di bawah pergeseran antar-negara), bukan "ECG rhythm classifiers" umum.
- Gaya bahasa dirapikan (abstrak berakhir dengan rekomendasi konkret, kalimat "Its cost was sensitivity." dan "point to where the
  missing information lies" diganti). 99.000 window Normal AFDB memang tepat 99.000 (89.566 + 8.766 + 568 + 100).
- Kompilasi: 5 halaman A4, tanpa warning; halaman 1 dan 4 dirender dan dilihat.
