# Rancangan v16.5: aturan episode AF ≥ 30 detik untuk D-1 dan v3.1 (alarm palsu per jam)

Tanggal: 10 November 2026. Status: **dikunci sebelum aturan episode dihitung di data mana pun dan sebelum sinyal EDB dibaca** (sha256 dan
waktu di `logs/v16.5_praregistrasi.log`). Yang sudah dilihat: keputusan per window D-1 dan v3.1 di semua data di bawah (v16.1–v16.4,
v3.3), termasuk alarm per window NSRDB (v16.4: D-1 5,6%, v3.1 15,0%; episode palsu per jam 20,6 dan 51,9 tanpa aturan), dan anotasi
ritme EDB (180 jam, 1 rekaman ber-AF). Bagian 2–6 tidak diubah setelah hasil keluar; penyimpangan di bagian 8.

## 1. Latar

v16.4 (NSRDB, orang sehat 24 jam): tanpa aturan durasi, D-1 menghasilkan ±20 episode AF palsu per jam, v3.1 ±52. Itu tidak layak untuk
perangkat. Pedoman klinis (ESC 2020/2024, HRS) mendefinisikan episode AF yang tercatat sebagai AF yang berlangsung **≥ 30 detik**.
Perangkat komersial juga tidak mengalarmkan satu window. Pertanyaan: dengan aturan itu, berapa alarm palsu per jam dan berapa episode
AF yang tetap terdeteksi?

## 2. Aturan (tetap, dari definisi klinis, tidak disetel)

- Keputusan per window sistem (AF ya/tidak) diurutkan per rekaman. **Run** = window AF berturut-turut dengan jarak awal antar window ≤
  langkah window rekaman itu (+0,5 s); window yang tidak lolos cek kualitas memutus run.
- **Alarm** dikeluarkan untuk run yang rentangnya (awal window terakhir + 10 s − awal window pertama) **≥ 30 s**. Window di run yang
  lebih pendek menjadi "tidak ada alarm". Langkah 5 s → ≥ 5 window; langkah 10 s (latihan fisik v3.3) → ≥ 3 window.
- Deskriptif (tidak ikut keputusan): rentang ≥ 60 s, ≥ 120 s, ≥ 360 s, dan tanpa aturan.

## 3. Sistem (tetap)

| Sistem | Keputusan AF per window |
|---|---|
| **S1 = D-1 + 30 s (primer)** | d1 > −1 (ECGFounder zero-shot, v16.2), lalu aturan bagian 2 |
| S2 = (D-1 dan v3.1) + 30 s | AF hanya bila D-1 dan paket v3.1 (argmax ≠ Normal) sama-sama AF, lalu aturan |
| R = v3.1 + 30 s (referensi) | paket v3.1 (AFDB: OOF MicroTransformer +bias cross-fit, 5 ulangan), lalu aturan |
| Tanpa aturan | D-1 dan v3.1 per window (konteks, = v16.1–v16.4) |

## 4. Data (tetap)

| Peran | Data | Sumber keputusan |
|---|---|---|
| Sensitivitas episode AF | AFDB (23 rekaman), LTAFDB (84), MIT-BIH (48), CPSC 2021 (label resmi) | cache v13/v16.1, v16.3, v16.2 |
| Alarm palsu, orang sehat 24 jam | NSRDB (18 subjek) | cache v16.4 |
| **Alarm palsu, belum pernah dipakai** | **EDB** (European ST-T, 90 rekaman × 2 jam, 250 Hz, kanal 0): semua waktu non-AF; segmen `(AFIB`/`(AFL` di satu rekaman dihitung AF | diproses di v16.5 dengan mesin v16.4 |
| Alarm palsu saat latihan fisik | EPFL, Wrist, EPHNOGRAM (mode perangkat, langkah 10 s) | cache v3.3 (v31) dan v13 (d1) |

- Window = window perangkat yang lolos cek kualitas (sama dengan sumbernya). CinC 2017 dan Georgia (≤ 60 s per rekaman) tidak dipakai.

## 5. Metrik dan kriteria

- **Episode** (gaya EC57, `takikardia_asli.metrik_episode` diperluas): episode referensi AF = run window berlabel AF; deteksi = run alarm.
  - Se episode untuk episode referensi **≥ 30 s** (primer) dan semua episode (sekunder); +P episode; Se/+P durasi (beban AF).
  - **Alarm palsu per jam** = episode deteksi tanpa tumpang tindih referensi / jam pengamatan.
- **Kriteria kandidat produk (ditetapkan sekarang):** sistem disebut **memenuhi target** bila (a) alarm palsu per jam ≤ **1,0** di NSRDB
  dan di EDB (titik), (b) Se episode AF ≥ 30 s ≥ **0,90** di setiap dataset AF (AFDB, LTAFDB, MIT-BIH, CPSC 2021), dan (c) alarm palsu
  per jam saat latihan fisik ≤ 1,0.
- **H1:** S1 vs R: alarm palsu per jam NSRDB lebih rendah, bootstrap subjek berpasangan 2.000× (seed 2026), P(Δ<0) ≥ 0,95.
- **H2:** S1 vs D-1 tanpa aturan: Se durasi (beban AF) di LTAFDB tidak turun lebih dari 0,05 (titik).
- **Pemilihan "sistem terbaik" v16.5:** di antara S1, S2, R yang memenuhi target, pilih alarm palsu per jam NSRDB + EDB terendah; bila
  tidak ada yang memenuhi, laporkan kriteria mana yang gagal untuk tiap sistem. Durasi lain (60–360 s) hanya kurva deskriptif; bila
  dipakai kemudian, harus dikunci ulang dan diuji di data baru.

## 6. Yang tidak dilakukan

- Tidak menyetel ambang d1, durasi, atau aturan run dari hasil. Tidak melatih apa pun. Tidak mengubah paket produksi.

## 7. Keluaran

`v16.5_episode_30s/`: `episode.py`, `episode.ipynb`, `README.md`, `laporan/`. Cache `../cache/v16.5/` (EDB per rekaman, ringkasan).

## 8. Penyimpangan dari rancangan

(belum ada)
