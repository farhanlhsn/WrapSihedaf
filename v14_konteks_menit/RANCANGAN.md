# Rancangan v14: konteks beberapa menit di atas keluaran per window

Tanggal: 6 November 2026. Status: **dikunci sebelum training.** Bagian 2–5 tidak diubah setelah hasil keluar; penyimpangan
ditulis di bagian 7.

## 1. Latar

- Riset 6 November 2026: S4ECG (arXiv 2510.17406) membaca 10–20 menit EKG dan menurunkan positif palsu AF 3–10×; AUROC AFL di AFDB
  naik dengan konteks lebih panjang. Monitor implan memakai durasi dan cara mulai detak cepat (LUX-Dx: takikardia atrium > 110 bpm
  selama 2–4 jam; ICD: "onset ratio" menolak 98% percepatan sinus pada VT vs sinus).
- Fisiologi yang dimanfaatkan: sinus takikardia naik bertahap mengikuti aktivitas dan detaknya terus berubah; AFL 2:1 mulai
  mendadak dan detaknya mendatar di sekitar setengah frekuensi flutter (±140–150 bpm).
- Beda dengan v4: v4 hanya merata-ratakan probabilitas antar window (dan menghapus episode AFL pendek). v14 memakai dinamika detak
  (cara mulai, kelandaian, lama berlangsung) sebagai informasi baru. Invariant "tidak ada resep RR-only baru" (AGENTS.md):
  pengecualian ini dicatat karena informasinya lintas window, bukan dari window 10 s.

## 2. Fitur (kausal: hanya window sekarang dan sebelumnya, supaya bisa jalan waktu nyata)

Per window i pada rekaman yang sama (AFDB: langkah 5 s), dengan keluaran tahap pertama v3.1:
1. log p(Normal), log p(AFIB), log p(AFL) window i.
2. Rata-rata p(AFIB) dan p(AFL) 2 menit dan 10 menit terakhir.
3. Detak window i; rata-rata detak 10 menit; simpangan baku detak 2 menit (kedataran); kemiringan detak 2 menit (regresi linear,
   bpm/menit); lompatan detak terbesar antar window berurutan dalam 2 menit (cara mulai).
4. Lama keadaan "cepat-teratur" (detak ≥ 110 bpm dan CV IBI ≤ 0,05) yang sedang berlangsung, menit (maks 30).
5. Bagian window cepat-teratur dalam 10 menit terakhir.
6. CV IBI window i dan rata-rata CV 2 menit.
Window tanpa riwayat cukup memakai riwayat yang ada (rata-rata dari window yang tersedia).

## 3. Model dan data

- Tahap pertama: OOF v3.1 (MicroTransformer v3, `cache/oof_robust`, 5 ulangan). Untuk data eksternal: paket v3.1.
- Tahap kedua (utama): XGBoost (kedalaman 4, 300 pohon, laju 0,05, bobot kelas lunak `class_weights ** 0,5`), dilatih per (ulangan,
  fold) pada window latih fold itu, lalu bias cross-fit (grid −7…+3) seperti protokol. Sekunder: regresi logistik pada fitur yang
  sama.
- Fold resmi 5×5, seed `2026 + 100·r + k`.

## 4. Hipotesis dan kriteria (dikunci)

- **H1 (AFDB, tidak lebih buruk):** Macro-F1 v14 vs v3.1 berpasangan: tidak lebih buruk (P(Δ > 0) > 0,05).
- **H2 (episode):** sensitivitas episode AFL (EC57, `takikardia_asli.metrik_episode`) turun paling banyak 0,10 dibanding v3.1,
  dan episode AFL palsu per jam turun.
- **H3 (sinus takikardia asli):** di EPHNOGRAM (rekaman kontinu latihan fisik, window langkah 5 s), alarm AFIB/AFL pada window
  ≥ 120 bpm turun paling sedikit separuh dibanding paket v3.1, CI bootstrap subjek.
- **H4 (uji stres konteks sintetis):** urutan 15 menit: (A) sinus naik bertahap 80 → 120/140/160 bpm dalam 5 menit lalu mendatar
  10 menit, jitter Gauss 10 dan 20 ms; (B) sinus 80 bpm 5 menit lalu AFL 2:1 mendadak 150 bpm (CV 0,03) 10 menit. Alarm di
  bagian datar (A) turun paling sedikit separuh, dan deteksi (B) paling sedikit 0,8 × v3.1.
- Protokol butir 9 sintetis per window tidak berlaku langsung (v14 butuh urutan); diganti H4 dan dilaporkan.

## 5. Yang tidak dilakukan

- Tidak menyetel hiperparameter dengan melihat fold uji, EPHNOGRAM, atau uji stres.
- Tidak mengubah paket produksi. Klaim produksi butuh butir 10 dan persetujuan pemilik repo.

## 6. Keluaran

`v14_konteks_menit/`: `konteks.py`, `konteks_menit.ipynb`, `README.md`, `laporan/`. Cache `../cache/v14/`.

## 7. Penyimpangan dari rancangan

(belum ada)
