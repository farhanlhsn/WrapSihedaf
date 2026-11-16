# Rancangan v16.2: uji ambang t = −1 detektor AF 2 kelas (ECGFounder) di CPSC 2021 yang belum pernah dinilai

Tanggal: 6 November 2026. Status: **dikunci sebelum sinyal CPSC 2021 dibaca oleh kode apa pun** (sha256 dan waktu di
`logs/v16.2_praregistrasi.log`). Bagian 2–6 tidak diubah setelah hasil keluar; penyimpangan ditulis di bagian 8.

## 1. Latar

v16.1 menilai tahap 1 v16 (ECGFounder zero-shot, AF bila `d1 = log((pAF + pAFL) / pN) > t`) sebagai detektor AF 2 kelas dengan t = 0
(D-0). D-0 sangat spesifik tetapi sensitivitasnya lebih rendah dari v3.1 dan gagal di Icentia11k. Kurva ambang v16.1, yang dilihat
**setelah** hasil pada data yang sama, menunjukkan t = −1 lebih seimbang. Klaim itu hanya sah bila diuji di data yang belum dipakai.

Yang sudah diketahui untuk t = −1 (v16.1, titik saja, tanpa uji berpasangan): AFDB Se 0,935, Sp 0,992; sinus takikardia asli ≥ 120 bpm
0,5%; PTB-XL Se 0,912; Icentia11k Macro-F1 0,870.

**CPSC 2021** (PhysioNet `cpsc2021` 1.0.0, 1.425 rekaman yang dirilis): sejauh ini hanya anotasinya yang dibaca (v15: jumlah subjek dan
window per ritme). Sinyalnya belum pernah dinilai oleh model SIHEDAF mana pun. ECGFounder dipralatih di HEEDB, bukan CPSC 2021
(benchmark FOUND-AF memakainya sebagai data uji).

## 2. Data dan window (tetap)

- Kanal 0 setiap rekaman (lead I menurut header; ECGFounder 1-lead juga lead I). Sinyal dibaca dengan `v12.atrium.baca_dat`.
- Window perangkat: `hrv_core.signal_to_windows(x, fs)` (resample ke 50 Hz, detektor R produksi, window 10 s, langkah 5 s, sama dengan
  window detektor AFDB). EKG window untuk ECGFounder = sinyal 50 Hz yang sama (`hrv_core.resample_signal`), 500 sampel dari awal window.
- Label: ritme anotasi (`v15.kuantisasi.segmen_ritme`) bila window 10 s seluruhnya di dalam satu segmen `(N`, `(AFIB`, atau `(AFL`.
  Window yang melintasi pergantian ritme dibuang. 2 kelas: Normal = `(N`, AF = `(AFIB` atau `(AFL`.
- **Himpunan uji = window berlabel yang lolos cek kualitas detektor** (window yang punya prediksi paket v3.1). Jumlah window yang
  dibuang karena kualitas dilaporkan.
- Subjek = angka kedua nama rekaman (`data_<subjek>_<n>`, pola v15). Bootstrap per subjek.

## 3. Sistem (tetap)

| Sistem | Keputusan AF |
|---|---|
| **D-1 (primer)** | ECGFounder zero-shot: `d1 > −1` (ambang dikunci di sini, tidak disetel ulang) |
| D-0 (v16.1, versi sebelumnya) | `d1 > 0` |
| **v3.1 paket (referensi)** | argmax paket produksi pada RR window, AFIB + AFL digabung |
| v1 paket | kelas paket v1 (`v3.2.takikardia.load_packages`), AFIB + AFL digabung |

Tidak ada model yang dilatih dan tidak ada parameter yang dipilih di CPSC 2021.

## 4. Hipotesis dan kriteria

- **H1 (utama):** D-1 vs v3.1 paket di CPSC 2021, 2 kelas, bootstrap subjek berpasangan 2.000× (seed 2026, `v16.1.banding2_pasien`):
  (a) Macro-F1 2 kelas bukan "lebih buruk" (P(Δ>0) > 0,05) dan (b) sensitivitas bukan "lebih buruk" (P > 0,05).
- **H2:** spesifisitas dan PPV D-1 vs v3.1 (vonis protokol). Dugaan: lebih baik.
- **H3:** D-1 vs D-0 (Macro-F1, Se, Sp) di CPSC 2021: apakah ambang −1 juga lebih seimbang di data baru?
- **H4 (sekunder, sebagian sudah dilihat):** D-1 vs v3.1 berpasangan di AFDB (aturan 2 kelas v11.1), PTB-XL, Icentia11k, dan alarm
  sinus takikardia asli ≥ 120 bpm dengan CI subjek. Ditandai di laporan sebagai data yang sudah dipakai untuk melihat kurva.
- **Deskriptif:** alarm pada window Normal CPSC per pita detak (< 100, 100–110, 110–120, ≥ 120 bpm), proporsi alarm per kelas asli
  (AFL hanya 2 subjek, tanpa uji), sensitivitas dan alarm per subjek (sebutkan subjek dengan Se turun > 0,20 vs v3.1), metrik episode
  AF gaya EC57 (`takikardia_asli.metrik_episode`, langkah 5 s), dan v1 paket.
- **Protokol:** butir 1–2 tidak berlaku (data eksternal; tidak ada fold). Butir 3–4: metrik pooled + CI dan perbandingan berpasangan
  bootstrap subjek. Butir 5: per subjek. Butir 6: tidak ada parameter. Butir 7: episode. Butir 9: sinus takikardia asli (H4) dan
  alarm Normal per pita. Butir 10: tidak ada pemilihan parameter yang perlu dibootstrap ulang.

## 5. Keputusan yang ditetapkan sebelum hasil

- D-1 disebut **kandidat detektor AF 2 kelas untuk uji perangkat** bila H1 (a) dan (b) lolos dan alarm sinus takikardia asli ≥ 120 bpm
  ≤ separuh v3.1 (sudah diketahui 0,5% vs 98,4%; dihitung ulang dengan CI). Selain itu: bukan kandidat; tulis bagian yang gagal.
- Bukan pengganti paket v3.1 dalam keadaan apa pun; tidak ada `model_export/` yang dibuat atau diubah.

## 6. Yang tidak dilakukan

- Tidak memilih ambang, bias, atau lead dari CPSC 2021. Tidak melatih apa pun. Tidak mengubah paket produksi.

## 7. Keluaran

`v16.2_cpsc2021/`: `uji_cpsc.py`, `uji_cpsc.ipynb`, `README.md`, `laporan/`. Cache `../cache/v16.2/`.

## 8. Penyimpangan dari rancangan

- **Label Normal tidak mengikuti konvensi CPSC 2021 (ditemukan setelah run pertama, 6 November 2026).** Aturan di bagian 2 memakai
  `v15.segmen_ritme`, yang hanya menganggap bagian setelah label `(N` sebagai Normal. Menurut kode penilaian resmi
  (`data/cpsc2021/python_entry/score_2021.py`, `RefInfo._load_ref`), `(AFIB`/`(AFL` menandai **awal** episode AF, `(N` menandai
  **akhir**-nya, dan rekaman "non atrial fibrillation" tidak punya label ritme sama sekali. Akibatnya run pertama membuang 721
  rekaman non-AF dan bagian Normal sebelum episode AF pertama: window Normal hanya dari 22 subjek. Window AF tidak terpengaruh
  (interval AF sama di kedua aturan), jadi sensitivitas tidak berubah.
- **Penanganan.** Hasil run pertama tetap dilaporkan sebagai hasil yang dikunci. Ditambah analisis **label resmi**: window AF bila
  seluruhnya di dalam interval [awal, akhir) AF (kelas dari label awal; awal tanpa akhir → sampai akhir rekaman), Normal bila tidak
  bersinggungan dengan interval AF mana pun, dibuang bila melintasi batas. Sistem, ambang, metrik, dan kriteria bagian 4–5 sama;
  tidak ada parameter yang dipilih. **D-1 hanya disebut kandidat bila kriteria lolos di kedua analisis.**
