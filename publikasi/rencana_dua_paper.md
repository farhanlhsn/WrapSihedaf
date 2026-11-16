# Rencana dua paper (Jalur B)

Diputuskan pemilik repo **6 November 2026**. Isi sumber: [kerangka_paper_evaluasi.md](kerangka_paper_evaluasi.md) (C1–C5). Alasan pemilihan
venue: [riset_kelayakan_dan_venue.md](riset_kelayakan_dan_venue.md).

| | Paper 1 | Paper 2 |
|---|---|---|
| Isi | **C1 + C2 + C3 + C5**: dekomposisi angka, arsitektur bukan pembatas, kerapuhan kesimpulan, daftar periksa | **C4**: uji stres sinus takikardia (sintetis + EKG latihan asli) |
| Venue | **ICIMPACT 2027** (Telkom University, Bandung, 3–5 Feb 2027), track *Biomedical Multimedia Intelligence & Digital Health Systems* | **ICIMPACT 2027 juga** (dipilih 6 Nov 2026 setelah EMBC dibatalkan karena biaya ±USD 2.000 dengan perjalanan) |
| Tenggat | **30 Nov 2026** (EDAS, WIB) | **30 Nov 2026** (EDAS, WIB) |
| Presentasi | Hybrid (situs resmi); penulis atau wakilnya wajib presentasi | Sama (hybrid) |
| Status angka | Lengkap di cache. Dekomposisi selesai 6 Nov (`publikasi/dekomposisi_v1.py`) | Lengkap di cache v3.2/v3.3/v7/v8/v11. Perlu cek P-wave sampel |

## Aturan supaya kedua paper tidak tumpang tindih

1. Paper 1 **tidak** memuat hasil sinus takikardia. Cukup satu kalimat bahwa uji di luar distribusi AFDB dilaporkan terpisah.
2. Paper 2 tidak mengulang tabel dekomposisi. Protokol evaluasi cukup diringkas dan disitasi ke paper 1 ("under review" bila belum terbit).
3. Model zoo boleh muncul di keduanya, tetapi dengan hasil berbeda: paper 1 berisi metrik AFDB, paper 2 berisi metrik uji stres.
4. Tidak ada gambar atau tabel yang sama persis.
5. Bila formulir EDAS menanyakan submisi terkait, sebutkan paper lainnya.

## Paper 1: ICIMPACT 2027

> **Status 6 November 2026 (sore):** draf lengkap ada di [paper1/](paper1/) (`main.tex`, `paper1_draft.pdf`; angka dan gambar dari
> `paper1/angka.py`). Sudah: review internal pertama dan perbaikannya, referensi diverifikasi lewat Crossref/PubMed/arXiv, versi
> anonim untuk *blind review* (`\blindtrue`). Sisa pekerjaan: penulis, pernyataan kode, review tim dan pembimbing.

**Judul kerja:** *Where Did the F1 Go? A Fixed-Model Decomposition of Performance Inflation in RR-Interval Atrial Fibrillation and
Flutter Classification on the MIT-BIH AFDB*

Alternatif: *How Evaluation Choices Inflate RR-Interval AF/AFL Classification: Lessons from a 50 Hz Single-Lead Device Pipeline*

**Draf abstrak (±250 kata):**

> Performance figures for RR-interval atrial fibrillation (AF) and atrial flutter (AFL) classifiers on the MIT-BIH Atrial Fibrillation
> Database (AFDB) are hard to compare because evaluation choices vary. For a 50 Hz single-lead device pipeline, we held the trained fold
> models fixed and changed only how they were evaluated: partition, averaging (per-fold vs pooled), test-window construction, R-peak
> source and front-end. We then compared ten RR-interval models from nine architecture families under 5×5 repeated patient-level
> cross-validation with patient-bootstrap confidence intervals and paired comparisons, and tested the robustness of conclusions with a
> nested bootstrap and a leave-dominant-patient-out analysis. A reported macro-F1 of 0.826 was reproducible (0.828). Sampling test windows
> uniformly lowered it to 0.759 (95% CI 0.66–0.83); 1-s-stride AFL test windows had doubled AFL precision. Replacing annotation R-peaks
> with a 50 Hz detector cost only 0.016, whereas the originally deployed peak-finding front-end without quality gating cost 0.325
> (macro-F1 0.418). No architecture outperformed the recalibrated reference (0.832, 0.67–0.92); class weighting and decision-bias
> calibration mattered more than architecture. The reference's advantage over the original pipeline (P = 0.98) became indistinguishable
> after removing the patient who contributed 58% of AFL windows (P = 0.94). In scikit-learn ≥1.8, StratifiedGroupKFold with shuffling
> produced a single patient partition for 100 seeds. We provide a checklist that extends existing AF-detector evaluation recommendations.

**Anggaran halaman** (situs resmi: maks. 6 halaman **termasuk referensi**, template konferensi IEEE; draf sekarang 5 halaman):

| Bagian | Halaman | Isi | Angka / sumber |
|---|---|---|---|
| I. Introduction | 0,75 | Masalah perbandingan angka AF/AFL di AFDB; RQ1–RQ3; kontribusi C1, C2, C3, C5 | – |
| II. Related Work | 0,5 | Butkuvienė 2021, Domazetoski 2022, de Chazal 2004, Forman & Scholz 2010, Kapoor & Narayanan 2023, Strodthoff 2021, Tsamardinos 2018 (BBC-CV), Oster & Clifford 2015, Kwon 2018 | `riset_kelayakan_dan_venue.md` bagian 2.2 |
| III. Methods | 1,5 | A. Data AFDB dan dua sumber R-peak. B. Model (10 model, 9 arsitektur). C. Protokol (5×5 fold pasien, pooled OOF, bootstrap pasien, berpasangan, bias cross-fit). D. Desain dekomposisi langkah A–J. E. Uji ketahanan (bersarang, tanpa 04936) | kerangka 5.3 |
| IV-A. Decomposition | 0,8 | Tabel I (langkah A–J) + Gambar 1 (waterfall). Bukti B memakai partisi asli; demo `StratifiedGroupKFold` | `cache/publikasi/dekomposisi_v1/ringkasan.json` |
| IV-B. Model comparison | 0,7 | Gambar 2 (forest plot Δ Macro-F1 vs referensi, 10 model) + bobot kelas vs arsitektur + jebakan grid bias | `v3_evaluasi_robust/`, `cache/v11/gabungan.json` |
| IV-C. Robustness | 0,4 | Tabel II (bersarang, tanpa 04936) + recall AFL per pasien (07910, 08215 gagal) | `v3.2_uji_takikardia/` (bagian 3–4 saja) |
| V. Discussion + checklist | 0,75 | Tabel III (daftar periksa C5); batasan (23 pasien, daya uji rendah, satu dataset) | kerangka 5.5–5.6 |
| VI. Conclusion | 0,2 | – | – |
| References | 0,4 | ±20 sitasi | – |

**Gambar dan tabel paper 1:**
- Gambar 1: waterfall langkah A–J dengan CI (langkah E–J), referensi P2 0,832 sebagai garis.
- Gambar 2: forest plot Δ Macro-F1 vs referensi + CI, marker berbeda per vonis (lebih baik / setara / lebih buruk).
- Tabel I: dekomposisi. Tabel II: ketahanan kesimpulan. Tabel III: daftar periksa.
- Palet data `#2a78d6`, `#eb6834`, `#1baf7a`, ditambah pembeda bentuk marker (AGENTS.md "Gaya laporan").

**Pekerjaan paper 1:**
1. Skrip `publikasi/paper1/angka.py`: ambil semua angka dari cache dengan `assert` terhadap ringkasan tiap versi, lalu buat Gambar 1–2.
2. Naskah IEEE (LaTeX/Overleaf atau Word, sesuai template panitia) dalam bahasa Inggris.
3. Daftar penulis, afiliasi, dan pembimbing (keputusan tim).

## Paper 2: ICIMPACT 2027 (sama dengan paper 1)

> **Keputusan 6 November 2026:** EMBC dibatalkan karena biaya. Paper 2 dikirim ke **ICIMPACT 2027** bersama paper 1, dengan tenggat
> EDAS yang sama (30 Nov 2026). Draf `paper2/` sudah diubah ke A4, anonim (`\blindtrue`), 5 halaman (batas 6), 26 referensi.
>
> **Supaya dua paper di satu konferensi tetap sah:**
> - Isinya tidak tumpang tindih:
>   - kolom Macro-F1 AFDB per model dibuang dari Tabel I paper 2, karena sudah ada di paper 1;
>   - paper 2 hanya memuat metrik uji stres dan recall 08378.
> - Kedua paper saling mensitasi secara anonim (*under review*).
> - Di formulir EDAS, sebutkan paper lainnya sebagai submisi terkait.
>
> **Biaya:** registrasi per paper, kemungkinan Rp2,8–3,5 jt × 2 (*early bird* Rp3 jt × 2 sampai 18 Des 2026). Tanyakan ke panitia
> apakah satu registrasi bisa menanggung dua paper.


### Alternatif Indonesia untuk paper 2 (dicek 6 November 2026 dari EDAS dan situs resmi)

| Konferensi | Tenggat | Track yang cocok | Biaya penulis mahasiswa | Online? | Catatan |
|---|---|---|---|---|---|
| **ICoDSA 2027** (CommDIS; ketua dan panitia banyak dari Telkom University), Bali, 21–22 Sep 2027 | 21 Mar 2027 (ada juga 23 Mei dan 31 Jul) | **Health and Humanity** | Online Rp2,5 jt (anggota IEEE/APTIKOM) / Rp3 jt; onsite Rp4–4,5 jt | **Ya** (tarif online terpisah) | 4–6 halaman termasuk referensi. Edisi 3–9 ada di IEEE Xplore; sponsor 2027 belum diumumkan |
| **ICIMPACT 2027** (Telkom), Bandung, 3–5 Feb 2027 | **30 Nov 2026** | Biomedical Multimedia & Digital Health | Rp2,8 jt (anggota IEEE) / Rp3,5 jt; *early bird* Rp3 jt | Hybrid | Sama dengan paper 1, jadi dua paper dari satu proyek di satu konferensi edisi pertama. Isinya tetap terpisah (lihat aturan di atas) |
| **SIML 2027** (UMS), Surakarta/hybrid, 8–10 Jul 2027 | 11 Jan 2027 | Tidak ada track kesehatan (paling dekat: bioinformatics; EDAS menyebut signal processing) | Awal Rp2 jt (IEEE) / Rp2,5 jt; standar Rp2,5 / 3 jt | **Ya** (Zoom) | Tercatat di IEEE Conference Publication Program #74641 |
| ICERA 2027 (STMIK El Rahma), Malang, 24 Jul 2027 | 30 Apr 2027 | Tidak ada track kesehatan | Rp2,75 jt (IEEE) / Rp3 jt | Ya | 2025–2026 di IEEE Xplore. Situs memuat tautan tersembunyi dan rekening bank yang berbeda-beda; kurang disarankan |
| iBioMed 2027 (UGM + IEEE Indonesia Section) | Belum diumumkan (pola 2026: Mei–Juli, konferensi Oktober) | **Biomedical Signal Processing** (paling cocok) | Belum diketahui | Belum diketahui | Edisi 2027 belum ada di situs |
| IBITeC 2027 (UII, dua tahunan; 2025 IEEE) | Belum diumumkan (pola: November) | Biomedical instrumentation | Belum diketahui | 2023 hybrid | Edisi 2027 belum ada di situs |

ICONIQ 2027 (Bali, tenggat 12 Mar 2027) tidak bisa dicek: sertifikat situsnya tidak valid.


## Info resmi EMBC 2027 (diverifikasi 6 November 2026 dari embc.embs.org/2027)

| Butir | Isi |
|---|---|
| Acara | 49th IEEE EMBC, **11–15 Juli 2027**, Suntec Singapore Convention & Exhibition Centre |
| Tenggat | Full paper **24 Januari 2027** (23.59 AoE, "will not be extended"); notifikasi 16 April 2027; versi final 30 April 2027. Portal submisi dibuka Desember 2026 |
| Format | Full contributed paper **4+1 halaman**: halaman 1–4 isi; halaman 5 hanya referensi dan ucapan terima kasih; **maks. 20 referensi**. Template konferensi IEEE (ukuran kertas tidak disebut; draf memakai letter). Pelanggaran format: *desk reject* |
| Review | Jenis review (single/double blind) tidak disebut |
| Presentasi | **Hadir langsung di Singapura**; tidak ada presentasi online. Corresponding author adalah presenter; pengganti harus diajukan lewat formulir EMBS sebelumnya, kalau tidak dianggap *no show* |
| Biaya | Belum diumumkan (registrasi dibuka Maret 2027). Patokan EMBC 2026: anggota IEEE ±USD 1.095–1.295 |
| Lain | Surat undangan visa diterbitkan setelah registrasi dibayar. Track yang cocok: *Wearable Biomedical Sensors and Systems* atau *Biomedical Signal Processing* |

Draf `paper2/` sudah disesuaikan: 4 halaman (letter) termasuk 20 referensi.

> **Status 6 November 2026:** draf awal ada di [paper2/](paper2/) (`main.tex`, `paper2_draft.pdf`; angka dan gambar dari
> `paper2/angka.py`), 4 halaman termasuk referensi. Tambahan dari rencana: sinus takikardia **klinis berlabel dokter** dari Chapman
> (1.743 EKG ≥ 120 bpm; P2 alarm 99,2%) dan cek visual gelombang P pada 20 strip (`paper2/cek_gelombang_p.py`; penilaian awal oleh
> asisten AI, perlu konfirmasi klinisi). Referensi sudah diverifikasi (`paper2/catatan_literatur.md`; Genzoni 2020 ternyata PPG). Venue: EMBC 2027 dipilih lalu dibatalkan (biaya); dicari konferensi Indonesia. Sisa: konfirmasi cek gelombang P,
> teks lengkap Genzoni 2020, penulis, review.

**Judul kerja:** *RR-Interval AF/AFL Classifiers Trained on MIT-BIH AFDB Alarm on Sinus Tachycardia: A Stress Test with Synthetic Rhythms
and Real Exercise ECG*

**Isi:**
- Sapuan ritme teratur sintetis 60–170 bpm: ambang 110–140 bpm untuk 10 model.
- Window Normal AFDB per pita detak: hanya 0,1% yang > 120 bpm.
- EKG latihan asli (EPFL, Wrist, EPHNOGRAM; 41 subjek, 2.189 window ≥ 120 bpm): alarm 89–100%, dengan R-peak perangkat maupun manual.
- Trade-off data latih dengan sinus takikardia (v7a 63,9%, v8c 60,3%) melawan hilangnya AFL 2:1 (08378).
- Mitigasi: conformal 0%, aturan produk menangkap 61–83% tetapi mengalihkan 86% AFL.
- Opsional: Chapman 6.937 pasien (alarm 42%).

**Pekerjaan tambahan paper 2:**
1. Cek visual P-wave pada sampel window ≥ 120 bpm untuk menguatkan label sinus.
2. Teks lengkap Genzoni 2020.
3. Gambar: kurva % Normal vs bpm, alarm per pita dengan CI subjek, scatter trade-off.
4. Pilih venue:
   - MeMeA cocok topik, punya jalur IEEE TIM, dan tarif pendaftarannya lebih rendah (±USD 650–820). Kekurangannya: ke Italia perlu visa
     Schengen dan tiket jarak jauh.
   - EMBC lebih bergengsi dan Singapura bebas visa. Kekurangannya: tarif penuhnya lebih mahal (±USD 1.000–1.300).

## Jadwal

| Tanggal | Paper 1 (ICIMPACT) | Paper 2 |
|---|---|---|
| 6–9 Nov | Tanya panitia ICIMPACT (lihat daftar di bawah); sepakati penulis; siapkan template | – |
| 9–14 Nov | Skrip angka + Gambar 1–2 + Tabel I–II; tulis Methods dan Results | – |
| 15–20 Nov | Introduction, Related Work, Discussion + daftar periksa, abstrak | – |
| 21–25 Nov | Review internal dan pembimbing | – |
| 26–29 Nov | Revisi, cek format, daftarkan paper di EDAS (track Digital Health) | – |
| **30 Nov** | **Tenggat** | – |
| 1–12 Des | – | Konfirmasi cek P-wave oleh pembimbing/dokter, teks lengkap Genzoni; portal EMBC dibuka (cek template & ukuran kertas) |
| 13 Des–10 Jan | – | Gambar dan naskah |
| 11–20 Jan | – | Review internal dan pembimbing |
| 21–24 Jan | – | Revisi dan submit |

## Info resmi ICIMPACT (situs aktif, dicek 6 November 2026)

Sumber: `https://icimpact.telkomuniversity.ac.id` (EDAS N35057). Kontak `icimpact@telkomuniversity.ac.id`.

| Butir | Isi |
|---|---|
| Format | Maks. 6 halaman termasuk referensi, bahasa Inggris, template konferensi IEEE, submisi lewat EDAS |
| Review | *Blind peer review*. Naskah submisi dibuat anonim (`\blindtrue` di `paper1/main.tex`) |
| Tanggal | Submisi 30 Nov 2026, notifikasi 21 Des 2026, *early bird* 18 Des 2026, camera-ready 5 Jan 2027, registrasi normal 24 Jan 2027, konferensi 3–5 Feb 2027 |
| Presentasi | Hybrid (Bandung). Aturan *no-show* IEEE: penulis atau wakilnya wajib presentasi, atau paper tidak dikirim ke IEEE Xplore |
| Publikasi | Diajukan ke IEEE Xplore (ID PDF Express belum diumumkan). Edisi pertama, jadi belum ada riwayat indeksasi |
| Biaya (per paper) | Mahasiswa Rp2.800.000 (anggota IEEE) / Rp3.500.000 (non-anggota); reguler Rp3.200.000 / Rp4.000.000; *early bird* Rp3.000.000; halaman ke-7 dst. Rp600.000/halaman |

## Pertanyaan yang masih perlu ditanyakan ke panitia

1. Apakah *blind* berarti *double-blind* (penulis harus anonim)? Draf sudah dibuat anonim untuk berjaga-jaga.
2. Mekanisme presentasi online (rekaman atau langsung) dan apakah tarifnya sama.
3. Kepastian IEEE Xplore (nomor konferensi IEEE) dan indeksasi Scopus.
4. Kebijakan versi jurnal yang diperluas setelah konferensi.
