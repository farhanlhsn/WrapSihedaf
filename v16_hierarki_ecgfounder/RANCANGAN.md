# Rancangan v16: hierarki ECGFounder (sinus vs AF) lalu RR atau fitur atrium (AFIB vs AFL)

Tanggal: 6 November 2026. Status: **dikunci sebelum metrik apa pun dihitung** (sha256 dan waktu di `logs/v16_praregistrasi.log`).
Bagian 2–6 tidak diubah setelah hasil keluar; penyimpangan ditulis di bagian 8.

Yang sudah diketahui sebelum dikunci (dari v13 dan v15, bukan dari v16):
- ECGFounder zero-shot 3 kelas dengan bias median cross-fit: alarm sinus takikardia asli ≥ 120 bpm 5,4%, tetapi recall AFL 08378
  (AFL 2:1) 0,01 dan Macro-F1 AFDB 0,663 karena AFIB dan AFL tercampur.
- Fitur `p_konsisten` dan `kuantisasi` memisahkan AFL dari AFIB di PTB-XL (AUROC 0,92 dan 0,86). Tahap gabungan 3 kelas di v15
  memperburuk AFDB (04936 0,51 → 0,16).

Yang **belum** pernah dihitung: keputusan sinus vs AF ECGFounder tanpa penyetelan, sistem bertingkat di bagian 3, keluaran keempat
(H5), dan pemisahan AFL 2:1 vs sinus takikardia oleh logit ECGFounder (H6).

## 1. Latar

ECGFounder dan model RR gagal di tempat yang berbeda. ECGFounder mengenali sinus takikardia tetapi mencampur AFIB/AFL. Model RR
memisahkan AFIB dari AFL (terutama AFL cepat-teratur), tetapi menyebut sinus takikardia sebagai AF. Pertanyaan v16: apakah membagi
tugas, ECGFounder untuk **sinus vs AF** dan RR atau fitur atrium untuk **AFIB vs AFL**, mempertahankan keunggulan masing-masing?
Ini rancangan kombinasi dari model yang sudah ada. Tidak ada jaringan yang dilatih ulang; tahap 2 atrium adalah regresi logistik
kecil.

## 2. Data (semua dari cache versi lama, tidak dihitung ulang)

- **AFDB:** window detektor 50 Hz (165.978), fold resmi 5×5. Logit ECGFounder jalur 50 Hz dari `cache/v13/afdb_50hz/`. OOF v3.1
  (MicroTransformer) dari `cache/oof_robust/`. Fitur atrium revisi 1 dari `cache/v12/fitur_r1.npz`.
- **SHDB-AF:** window AF/AFL v12 (kanal 0), hanya untuk melatih tahap 2 atrium (pola v15).
- **Sinus takikardia asli:** window latihan fisik v3.3 (EPFL, Wrist, EPHNOGRAM; mode perangkat, lolos cek kualitas). Logit
  ECGFounder dari `cache/v13/latihan_ok_50hz.npz`, probabilitas paket v3.1 dari `cache/v3_3/prediksi.npz`.
- **PTB-XL:** lead I 100 → 50 Hz (`cache/v13/ptbxl_50hz.npz`, fitur atrium dan paket v3.1 dari `cache/v15/ptbxl.npz`).
- **Icentia11k:** window v10 (`cache/v10/icentia_windows.npz`, logit `cache/v13/icentia_50hz.npz`). Fitur atrium tidak tersedia,
  jadi H-A tidak dinilai di Icentia.

## 3. Sistem (tetap)

Notasi: `pE = [pN, pAF, pAFL]` = skor zero-shot v13 (`ecgfounder.skor_zero_shot`, softmax logit `[maks(NSR, SR, ST), AF, AFL]`).
Tahap 1: `d1 = log((pAF + pAFL) / pN)`.

Tahap 2 (hanya menentukan AFIB vs AFL pada window yang diputuskan AF):
- **RR:** `d2 = log(pAFL / pAFIB)` dari OOF v3.1 (data eksternal: probabilitas paket v3.1 tanpa bias paket).
- **atrium:** regresi logistik biner (StandardScaler + `LogisticRegression(C=1, class_weight="balanced")`, resep
  `v15.latih_gabungan`) dengan masukan `[log p v3.1 (3), kuantisasi, p_konsisten]` (`v15.masukan`, NaN → median data latih). Dilatih
  per (ulangan, fold) pada window AFIB/AFL **fold latih AFDB + seluruh SHDB-AF**; `d2` = logit-nya.

| Sistem | Tahap 1 | Tahap 2 | Parameter yang dipilih cross-fit |
|---|---|---|---|
| **H-0 (primer)** | AF bila `d1 > 0` (**tanpa penyetelan**) | RR | `b2` (AFL bila `d2 + b2 > 0`) |
| **H-A** | sama dengan H-0 | atrium | `b2` |
| **H-X** | skor `log [pN, pAF·qAFIB, pAF·qAFL]` (`pAF` = peluang AF ECGFounder, `q` = RR dinormalkan) | | bias 3 kelas standar |

- `b2` dipilih dari grid `sihedaf_cv.BIAS_GRID` (−7…+3, langkah 0,25) dengan Macro-F1 3 kelas pada fold latih (aturan yang sama
  dengan `fit_class_bias`; kandidat pertama terbaik). H-X memakai `sihedaf_cv.crossfit_bias` apa adanya.
- **Alasan tahap 1 tanpa penyetelan pada sistem primer:** AFDB hampir tidak punya sinus takikardia (0,1% window Normal > 120 bpm),
  sehingga ambang yang dipilih di AFDB tidak melihat biaya alarm palsu (anti-pattern v1–v6). H-X mengukur apa yang terjadi bila
  ambang itu tetap disetel di AFDB.
- Data eksternal: tahap 1 H-0/H-A tanpa parameter; `b2` dan bias H-X = median cross-fit 25 fold; tahap 2 atrium dilatih di seluruh
  AFDB (ulangan 0) + SHDB-AF (pola `v15.h2_ptbxl`).

## 4. Hipotesis dan kriteria

- **H1 (sinus takikardia asli, utama):** alarm AFIB/AFL H-0 pada window latihan fisik ≥ 120 bpm vs paket v3.1, CI dan Δ bootstrap
  subjek berpasangan (`takikardia_asli.banding_alarm`). **Lolos** bila alarm ≤ separuh v3.1 dan batas atas CI Δ < 0. Juga dilaporkan
  per pita detak (< 100 bpm termasuk, karena detektor 50 Hz saat bergerak memicu alarm di sana).
- **H2 (AFDB 3 kelas, utama, protokol penuh):** Macro-F1 H-0 vs v3.1 berpasangan; vonis protokol. **Lolos** bila bukan "lebih buruk"
  (P(Δ>0) > 0,05). Selalu dengan recall AFL per pasien, asal AFL palsu, Normal R. Dugaan sebelum dihitung: 08378 hilang.
- **H2b (AFDB 2 kelas):** Normal vs AF/AFL. Keputusan tahap 1 H-0 vs v3.1 (bias 3 kelas lalu AFIB + AFL digabung, aturan v11.1);
  Macro-F1 2 kelas, Se, Sp, berpasangan; vonis protokol.
- **H3 (tahap 2 atrium):** H-A vs H-0 berpasangan di AFDB (Macro-F1, AFL F1, AFL R) dan PTB-XL (Macro-F1, recall AFL). Lebih baik bila
  P ≥ 0,95.
- **H4 (eksternal):** H-0 dan H-X di PTB-XL dan Icentia vs paket v3.1 (Macro-F1 3 kelas, recall AFL, bootstrap pasien
  berpasangan), dan alarm pada sinus takikardia PTB-XL (STACH). H-A hanya di PTB-XL.
- **H5 (keluaran keempat, deskriptif):** window yang tahap 1 sebut sinus tetapi v3.1 sebut AFL diberi keluaran "takikardia teratur,
  perlu dicek" (bukan alarm AF). Laporkan proporsi keluaran ini pada AFL per pasien AFDB (terutama 08378), Normal AFDB, sinus
  takikardia asli ≥ 120 bpm, dan STACH PTB-XL. Tidak ada ambang yang disetel.
- **H6 (eksploratif):** apakah logit ECGFounder memisahkan AFL 2:1 (window AFL 08378) dari sinus takikardia asli ≥ 120 bpm? AUROC
  `logit AFL − logit ST` (dan pembanding: `log pAFL − log pN` paket v3.1). Satu pasien AFL → tanpa CI pasien; hanya untuk versi
  berikutnya.
- **Protokol:** butir 1–6 seperti biasa. Butir 7: metrik episode (`takikardia_asli.metrik_episode`) H-0 vs v3.1. Butir 9: uji
  stres RR sintetis tidak bisa dipakai karena tahap 1 membaca EKG; diganti H1 (EKG asli), sama dengan v13. Butir 10: bootstrap
  bersarang (`b2` atau bias dipilih ulang di setiap sampel, berpasangan dengan v3.1 yang biasnya juga dipilih ulang) dan
  sensitivitas tanpa 04936 untuk H-0 dan H-X.

## 5. Keputusan yang ditetapkan sebelum hasil

- v16 disebut **kandidat untuk diuji di perangkat** bila H1 lolos, H2 lolos, dan H4 tidak "lebih buruk" di PTB-XL maupun Icentia.
  Selain itu: bukan kandidat; tulis bagian mana yang gagal.
- Dalam keadaan apa pun v16 **bukan pengganti paket** v3.1 tanpa data perangkat berlabel (ECGFounder 76 juta parameter juga butuh
  inferensi di backend).

## 6. Yang tidak dilakukan

- Tidak fine-tune ECGFounder, tidak melatih jaringan baru, tidak memilih parameter dari fold uji, data latihan fisik, atau data
  eksternal. Tidak menambah aturan atau ambang untuk H5/H6. Tidak mengubah paket produksi.

## 7. Keluaran

`v16_hierarki_ecgfounder/`: `hierarki.py`, `hierarki.ipynb`, `README.md`, `laporan/`. Cache `../cache/v16/`.

## 8. Penyimpangan dari rancangan

(belum ada)
