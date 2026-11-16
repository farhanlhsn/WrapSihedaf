# Rancangan v16.8: gerbang kualitas sinyal (SQI templat QRS) untuk S1 (D-1 + episode AF ≥ 30 s)

Tanggal: 10 November 2026. Status: **dikunci sebelum SQI dihitung di data mana pun dan sebelum versi berderau afpdb dibuat** (sha256 dan
waktu di `logs/v16.8_praregistrasi.log`). Bagian 2–6 tidak diubah setelah hasil keluar; penyimpangan di bagian 8.

Yang sudah dilihat: hasil v16.2–v16.7, termasuk nstdb di v16.7. Di sana S1 memberi 1,01 alarm palsu/jam pada SNR 12 dB dan 5–8/jam pada
≤ 6 dB, karena window berderau lolos cek kualitas perangkat. ECGFounder tidak punya label artefak/derau (tasks.txt).

## 1. Pertanyaan

Apakah gerbang kualitas sederhana (window dengan bentuk QRS yang tidak konsisten dianggap "tidak terbaca") menekan alarm AF palsu saat
derau berat, tanpa kehilangan sensitivitas episode AF di rekaman bersih dan wearable nyata?

## 2. SQI dan gerbang (tetap)

- Sinyal kanal 0 → 50 Hz (`hrv_core.resample_signal`), R-peak detektor perangkat (`hrv_core.detect_rpeaks`) di seluruh rekaman, lalu
  `pembersih.selaraskan(pembersih.bandpass(x), peaks)` v3.4: korelasi tiap detak dengan templat QRS median detak dalam ±5 s.
- **SQI window** = median korelasi detak yang R-peak-nya berada di [awal, awal + 10 s). Tanpa korelasi yang valid → NaN.
- **Gerbang q:** window dengan SQI < q atau NaN dianggap tidak terbaca dan dikeluarkan (sama dengan window yang gagal cek kualitas: tidak
  dinilai, memutus run). Lalu S1 seperti biasa (d1 > −1, aturan episode ≥ 30 s v16.5). S1G = S1 + gerbang terpilih.
- Grid q ∈ {0,5; 0,6; 0,7; 0,8; 0,9}.

## 3. Tahap A, pengembangan (data yang sudah dipakai)

- Derau: nstdb (118/119 + derau elektroda `em`, SNR 24 … −6 dB) dan 118/119 bersih (cache v16.7).
- Biaya sensitivitas: LTAFDB dan MIT-BIH (cache v16.3). Bersih: NSRDB (cache v16.4).
- **Pemilihan (dikunci):** q yang memenuhi syarat = penurunan Se episode AF ≥ 30 s (S1 − S1 + gerbang) ≤ 0,01 di LTAFDB **dan** MIT-BIH.
  Di antaranya, pilih q dengan alarm palsu per jam gabungan nstdb SNR 12, 6, 0, −6 dB terendah; selisih ≤ 0,05/jam → q lebih kecil.
  Tanpa q yang memenuhi syarat → tidak ada gerbang; dilaporkan.

## 4. Tahap B, validasi (sekali, konfigurasi terkunci)

- **afpdb + derau**: 100 rekaman n01–n50, p01–p50 (non-AF, kanal 0, 128 Hz). Derau dari rekaman derau nstdb kanal 0 (360 Hz → 128 Hz).
  **Derau otot `ma` (tidak pernah dipakai, primer)** dan derau elektroda `em` (sekunder; sama dengan jenis derau di tahap A), dijumlahkan
  ke seluruh rekaman pada SNR 12, 6, 0 dB. SNR = 10·log10(var(bandpass(sinyal)) / var(bandpass(derau terskala))), bandpass 0,5–40 Hz
  (`pembersih.bandpass` pada fs asli); derau mulai di offset acak per rekaman (seed 2026 + 168). Unit = rekaman.
- **Icentia11k** (600 segmen v16.7, wearable nyata): biaya sensitivitas dan alarm palsu.
- **Kriteria (dikunci):**
  - **V1:** afpdb + `ma` SNR 6 dB: alarm palsu S1G ≤ 1,0/jam **dan** lebih rendah dari S1 (bootstrap rekaman berpasangan 2.000×, seed
    2026, P(Δ < 0) ≥ 0,95).
  - **V2:** Icentia pasien AF: Se episode ≥ 30 s S1G ≥ 0,90 dan penurunan vs S1 ≤ 0,02 (titik).
  - **V3:** Icentia pasien tanpa AF: alarm palsu S1G ≤ 1,0/jam.
  - Sekunder (deskriptif): afpdb + `ma` 12/0 dB, afpdb + `em` semua SNR, afpdb bersih, fraksi window yang digerbang per data, R (v3.1)
    dengan gerbang yang sama.
- **Keputusan:** V1–V3 lolos → S1G menggantikan S1 sebagai sistem terbaik untuk deteksi AF 2 kelas. Selain itu, S1 tetap; laporkan.

## 5. Yang tidak dilakukan

Tidak menyetel d1, aturan episode, definisi SQI, atau grid setelah melihat data validasi. Tidak melatih apa pun. Tidak mengubah paket.

## 6. Keluaran

`v16.8_gerbang_kualitas/`: `gerbang.py`, `gerbang.ipynb`, `README.md`, `laporan/`. Cache `../cache/v16.8/` (SQI per rekaman, afpdb
berderau per rekaman, ringkasan).

## 8. Penyimpangan dari rancangan

**Amandemen A (10 November 2026, sebelum SQI atau SNR dihitung; hash baru di log).** Bagian 4 menyebut bandpass SNR "0,5–40 Hz
(`pembersih.bandpass`)". Padahal `pembersih.bandpass` adalah filter detektor 5–15 Hz dengan clipping persentil. Yang berlaku: SNR memakai
Butterworth orde 2, 0,5–40 Hz, `filtfilt`, pada fs asli rekaman, untuk sinyal dan derau. `pembersih.bandpass` hanya dipakai untuk SQI
(bagian 2), sesuai v3.4.
