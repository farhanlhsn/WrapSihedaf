# Paper 1: ICIMPACT 2027 (draf)

*Where Did the F1 Go? A Fixed-Model Decomposition of Performance Inflation in RR-Interval Atrial Fibrillation and Flutter
Classification.* Isi: C1–C3 + C5 dari [../kerangka_paper_evaluasi.md](../kerangka_paper_evaluasi.md); rencana dan aturan anti tumpang
tindih ada di [../rencana_dua_paper.md](../rencana_dua_paper.md). Tenggat EDAS: **30 November 2026**.

| Berkas | Isi |
|---|---|
| `main.tex` | Naskah (IEEEtran conference, A4, bahasa Inggris). Semua angka lewat makro dari `angka.tex`. `\blindtrue` = versi anonim untuk submisi, `\blindfalse` = versi bernama |
| `angka.py` | Menghitung semua angka dari cache dengan `assert` terhadap ringkasan tiap versi, lalu menulis `angka.tex` (makro + baris Tabel I), `angka.json`, dan `gambar/*.pdf|png` |
| `refs.bib` | 24 referensi, metadata diverifikasi lewat Crossref/PubMed/arXiv. Lebih dari enam penulis dicetak "et al." (entri `IEEEtranBSTCTL`) |
| `IEEEtran.cls`, `IEEEtran.bst` | Dari CTAN (V1.8b, LPPL). Panitia meminta template konferensi IEEE standar, jadi ini sudah sesuai |
| `paper1_draft.pdf` | Hasil kompilasi terakhir |

## Membangun ulang (dari root repo)

```bash
python publikasi/paper1/angka.py
cd publikasi/paper1 && latexmk -pdf -outdir=_build main.tex && cp _build/main.pdf paper1_draft.pdf
```

Butuh cache `publikasi/dekomposisi_v1`, `oof_robust`, `v3_2`, `v11`, serta `v3_evaluasi_robust/laporan/_build/tables.json` (dari
`build_report.py --extract` v3). `_build/` jangan di-commit.

## Revisi 8 November 2026

- Abstrak dan kesimpulan membedakan reproduksi partisi tunggal (A–B) dari dekomposisi dengan 25 model fold tetap (C–I).
  Rujukan langkah di teks sudah sesuai dengan Tabel I.
- Teks membedakan efek memilih ulang bias pada bootstrap P2 vs P1 dari perubahan vonis P2 vs P0 saat rekaman 04936
  dikeluarkan dari evaluasi dan pemilihan bias.
- Metode menjelaskan sumber anotasi detak. Batasan mencatat bahwa urutan dekomposisi dan jendela yang diloloskan setiap
  front-end ikut menentukan selisih.

## Status (6 November 2026)

- Draf lengkap: 4 halaman isi + referensi (5 halaman; batas panitia 6 halaman **termasuk** referensi), 2 gambar, 3 tabel.
  Kompilasi tanpa warning. Setiap halaman sudah dirender dan diperiksa.
- Review internal pertama sudah dikerjakan. Perbaikannya:
  - huruf langkah dibuat berurutan A–I (cache tidak punya langkah C);
  - klaim "model tetap" dibatasi ke langkah C–I (A adalah skor yang dilaporkan dan B memakai lima model latih ulang);
  - P0/P1/P2 didefinisikan sebagai pipeline pada window uji detektor yang sama (P1 = bias tetap v2 0/+2/−3);
  - jumlah window per kelas ditambahkan;
  - forest plot ditulis "12 konfigurasi dari 9 model";
  - P disebut probabilitas bootstrap (fraksi resample dengan Δ > 0), bukan p-value.

*(Butir terakhir di atas terpotong pada berkas lama dan direkonstruksi dari naskah pada 13 November 2026.)*

## Pemeriksaan pra-submisi (13 November 2026)

- `angka.py` dijalankan ulang dari cache: `angka.tex` identik dengan versi 8 November.
- Label P di Gambar 2 kini "P < 0.01" bila P < 0,005 (sebelumnya tercetak "P = 0.00").
- Referensi [24] diperbaiki: judul dan penulis sesuai issue asli (Pau Folch, scikit-learn #32478); perbaikan PR #32540 dipastikan
  masuk changelog scikit-learn 1.8.0, jadi klaim "versi sebelum 1.8" di naskah benar.
- Semua sitasi yang dikutip dicek otomatis: DOI lewat CrossRef/DataCite (judul, tahun, penulis pertama, volume), arXiv lewat API
  arXiv. Hasilnya tidak ada referensi palsu atau metadata salah.
- Anonimitas: tidak ada nama, afiliasi, atau tautan repo di PDF; metadata PDF kosong. Issue scikit-learn dibuat orang lain.
- Kompilasi: 5 halaman A4 (batas 6), log tanpa warning. Semua halaman dirender dan dilihat.
- Sisa (keputusan pemilik repo): daftar penulis dan afiliasi, pernyataan kode untuk versi bernama, review pembimbing, unggah ke
  EDAS dengan paper 2 disebut sebagai submisi terkait.

## Pemeriksaan ulang (14 November 2026)

- Sitasi dicek ulang: 24 entri yang dikutip (tanpa `anon_paper2`) cocok dengan CrossRef/DataCite atau API arXiv; entri tanpa DOI
  (Moody 1983, Cawley 2010 JMLR, issue scikit-learn) sesuai sumbernya.
- Pendahuluan kini menyebut "twelve configurations of nine RR-interval models", sama dengan bagian hasil dan Gambar 2
  (sebelumnya "ten models from nine families", yang menghitung referensi).
- Gaya bahasa dirapikan di tiga kalimat (pitfall scikit-learn, batas grid bias, paragraf front-end di diskusi). Angka tidak berubah.
- Kompilasi: 5 halaman A4, tanpa warning; halaman 4 dirender dan dilihat.
