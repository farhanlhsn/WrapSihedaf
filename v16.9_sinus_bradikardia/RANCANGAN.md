# Rancangan v16.9: sinus bradikardia dan detak lambat pada S1 (D-1 + episode AF ≥ 30 s)

Tanggal: 10 November 2026. Status: **dikunci sebelum alarm S1 atau R dihitung per ritme atau per pita detak di tingkat episode** (sha256 dan
waktu di `logs/v16.9_praregistrasi.log`). Bagian 2–6 tidak diubah setelah hasil keluar; penyimpangan di bagian 8.

Versi minor, evaluasi saja: semua keputusan per window dari cache v16.2–v16.7. Tidak ada data baru, sinyal tidak dibaca ulang, tidak ada
training, tidak ada ambang atau aturan yang dipilih.

Yang sudah dilihat sebelum dikunci:
- v16.3 (tingkat window, tanpa aturan episode): sinus bradikardia LTAFDB dialarmkan AF 4,2% (v3.1) → 27,9% (D-1), 25.501 window. Tabel
  alarm per pita detak tingkat window v16.2–v16.4 ada di ringkasan JSON masing-masing; tidak dipakai menyusun rancangan ini.
- v16.4 Georgia (EKG klinis 10 s, label dokter): sinus bradikardia 5,0% (v3.1) vs 5,6% (D-1).
- v16.5: S1 di NSRDB 1,23 alarm palsu/jam (tidak dipecah per detak).
- Survei jumlah data (tanpa melihat d1 atau p31): label `(SBR` hanya ada di LTAFDB (25 subjek, 35,4 jam window) dan MIT-BIH (1 subjek,
  0,46 jam); SHDB-AF, incartdb, svdb tidak punya `(SBR`. LTAFDB punya 98.833 window `(N` dengan detak < 60 bpm dari 49 subjek.

## 1. Pertanyaan

- **P1 (primer):** apakah S1 membuat alarm AF palsu yang relevan selama sinus bradikardia berlabel (LTAFDB), setelah aturan episode 30 s?
- **P2:** apakah alarm itu terkonsentrasi di sedikit subjek, dan apakah terjadi dekat episode AF sungguhan (pola takikardia-bradikardia)?
- **P3:** apakah detak lambat secara umum (window non-AF dengan detak < 60 bpm, apa pun label ritmenya) lebih sering memicu alarm S1 di
  dataset lain?
- **P4 (deskriptif):** berapa banyak AF sungguhan yang berdetak lambat, yaitu berapa mahal aturan berbasis detak bila kelak dipertimbangkan?

## 2. Sistem dan aturan (tetap)

- **S1** = D-1 (d1 > −1) + aturan episode ≥ 30 s v16.5 (`episode.aturan`, tanpa celah). **R** = paket v3.1 (argmax ≠ Normal) + aturan yang
  sama. Aturan dijalankan pada seluruh rekaman, bukan per segmen ritme (sama dengan v16.5–v16.8).
- Detak window = `hr` di cache (60.000 / rata-rata IBI detektor perangkat 50 Hz).

## 3. Metrik

- **Episode palsu** = episode deteksi setelah aturan yang tidak tumpang tindih dengan episode AF referensi (definisi `episode.metrik`).
- **Ritme episode palsu** = label ritme terbanyak di antara window di dalam episode itu (LTAFDB, MIT-BIH). **Pita episode palsu** = pita
  dari median detak window di dalam episode.
- **Laju palsu per ritme / pita** = jumlah episode palsu ritme / pita itu ÷ jam window non-AF ritme / pita itu (window × 5 s / 3600).
- **Fraksi waktu dialarmkan** = proporsi window non-AF ritme / pita itu yang berada di dalam episode alarm.
- Pita detak: < 50, 50–60, 60–80, 80–100, ≥ 100 bpm. "Lambat" = < 60, "acuan" = 60–100.
- Bootstrap unit (subjek bila ada, selain itu rekaman) berpasangan, 2.000×, seed 2026; laju = rasio jumlah (episode ÷ jam) per sampel.

## 4. Kriteria dan analisis (dikunci)

- **P1:** LTAFDB, window `(SBR`. Sinus bradikardia disebut **kelemahan nyata S1** bila laju palsu S1 per jam `(SBR` > 1,0 **dan** lebih tinggi
  dari R (P(Δ > 0) ≥ 0,95, bootstrap atas subjek yang punya window `(SBR`). Pembanding deskriptif: laju yang sama untuk `(N` LTAFDB, dan
  fraksi waktu `(SBR` dialarmkan.
- **P2 (deskriptif):** episode palsu `(SBR` per subjek; porsi dari 3 subjek teratas. Jarak dari episode palsu `(SBR` ke episode AF referensi
  terdekat di rekaman yang sama (< 10 menit, 10–60 menit, > 60 menit, rekaman tanpa AF), dibandingkan dengan sebaran jarak yang sama untuk
  semua window `(SBR` (garis dasar).
- **P3:** dataset: LTAFDB, MIT-BIH, CPSC 2021, NSRDB, EDB, SHDB-AF, afpdb, incartdb, svdb, Icentia11k (600 segmen v16.7). Hanya dataset dengan
  ≥ 1 jam window non-AF di detak < 60 bpm yang dinilai. Untuk tiap dataset: Δ = fraksi waktu dialarmkan (lambat) − (acuan), untuk S1 dan R.
  Detak lambat disebut **kelemahan umum S1** bila di **≥ 2 dataset selain LTAFDB** S1 punya P(Δ > 0) ≥ 0,95 **dan** fraksi lambat ≥ 2 × fraksi
  acuan. Juga dilaporkan per pita dan laju palsu per jam per pita.
- **P4 (deskriptif):** untuk dataset dengan AF: proporsi window AF dengan detak < 60 bpm, proporsi episode AF referensi ≥ 30 s dengan median
  detak < 60, dan Se episode S1 pada episode lambat itu.
- **Vonis:** P1 dan P3 dilaporkan apa adanya. Tidak ada aturan berbasis detak yang dipilih atau diklaim; bila P1 atau P3 positif, itu menjadi
  kriteria yang harus diperiksa di uji perangkat.

## 5. Yang tidak dilakukan

Tidak mengubah ambang d1, aturan episode, atau batas pita setelah melihat hasil. Tidak memakai data atau sinyal baru. Tidak melatih apa pun.
Tidak mengubah paket. AFDB, CinC 2017, Georgia (rekaman ≤ 60 s atau tanpa label ritme berkelanjutan untuk aturan 30 s), nstdb (derau
buatan), dan data latihan fisik (detak cepat) tidak dipakai.

## 6. Keluaran

`v16.9_sinus_bradikardia/`: `bradikardia.py`, `bradikardia.ipynb`, `README.md`, `laporan/`. Ringkasan di `../cache/v16.9/ringkasan.json`.

## 8. Penyimpangan dari rancangan

Tidak ada penyimpangan dari bagian 2–6.

**Analisis tambahan setelah hasil (10 November 2026, bukan bagian kriteria).** Setelah P1–P4 dilihat, notebook bagian 5 menambah eksplorasi
deskriptif: untuk window `(SBR` LTAFDB, detak, CV RR, dan detak ektopik dibaca dari anotasi .atr, lalu dibandingkan antara window di dalam
alarm S1 dan yang tidak. Hasilnya tidak dipakai memilih apa pun.
