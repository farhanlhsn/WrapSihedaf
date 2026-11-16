# Rancangan v15: kuantisasi RR dan konsistensi pra-QRS untuk AFL konduksi bervariasi (pra-registrasi)

Tanggal: 6 November 2026. Status: **dikunci sebelum data uji dibuka.** Ditulis saat CPSC 2021 belum selesai diunduh; belum ada
satu pun berkas CPSC 2021 yang dibaca. Setelah ditulis, isi bagian 2–6 tidak diubah. Penyimpangan, kalau ada, ditulis di
bagian 8 dengan alasan, dan hasil tetap dilaporkan menurut rancangan asli.

## 1. Latar

- Model RR-only hanya mendeteksi AFL yang cepat dan teratur (AFL 2:1). Recall AFL "tidak teratur" 0,07–0,11, "lambat-teratur"
  0,00 (v12 RANCANGAN 1.1). Ini kegagalan 2 di riset 6 November 2026.
- v12 berhenti di gerbang, tetapi dua fitur di luar gerbang memisahkan AFL dari AFIB searah di AFDB dan SHDB-AF, juga untuk AFL
  sulit: `p_konsisten` (AUROC 0,965 / 0,797) dan `kuantisasi` (0,837 / 0,743). Keduanya dilihat **setelah** hasil, jadi angka itu
  optimistis (v12 README).
- Literatur mendukung arah ini: model blok AV dua tingkat pada 16–22 interval RR membedakan AFL dari AF (Scholz dkk. 2014:
  sensitivitas 84%, spesifisitas 74%; Sager dkk. 2021: akurasi 82,8%, AUC ±0,9). Peringatan: RR saat AF juga bisa mengelompok
  di kelipatan siklus atrium (Climent dkk. 2008).
- Pengecualian dari invariant "tidak ada resep RR-only baru" (AGENTS.md): `p_konsisten` memakai bentuk gelombang EKG (segmen
  sebelum QRS), `kuantisasi` memakai siklus flutter dari sinyal sisa EKG, dan data ujinya pasien baru (CPSC 2021).

## 2. Data

- **Pengembangan (sudah dilihat):** AFDB (23 pasien) dan SHDB-AF (98 rekaman), fitur v12 revisi 1 (`cache/v12/fitur_r1.npz`,
  `cache/v12/afdb_fitur.npz`). Dipakai untuk melatih tahap gabungan di H2.
- **Uji (belum dilihat):** CPSC 2021 (PhysioNet `cpsc2021` 1.0.0, CC BY 4.0), rekaman Holter dan wearable 200 Hz dengan anotasi
  ritme. Diproses dengan pipeline produksi: kanal 0 di-resample ke 50 Hz (`hrv_core.resample_signal`), R-peak
  `hrv_core.detect_rpeaks`, window 10 detik tidak tumpang tindih, cek kualitas detektor yang sama dengan AFDB.
- Label window: Normal / AFIB / AFL hanya bila **seluruh** 10 detik berada dalam satu anotasi ritme. Window campuran dibuang.
  Label lain (mis. ritme yang bukan N/AFIB/AFL) dibuang.
- Kanal 1 hanya dilaporkan sebagai sensitivitas, bukan hasil utama.

## 3. Fitur (tetap, dari kode v12)

`v12_shdb_atrium/atrium.fitur_atrium(xf, peaks, t0, revisi=1)`, tanpa perubahan kode: `kuantisasi` (bagian RR yang rasionya
terhadap siklus flutter 1/DF dalam ±0,15 dari bilangan bulat) dan `p_konsisten` (rata-rata korelasi antar-detak segmen
−0,30…−0,10 s sebelum R). Window dengan < 4 detak: fitur NaN, dikeluarkan dari H1 dan diberi nilai median data latih di H2.

## 4. Hipotesis dan ambang (dikunci)

- **Cek kewajaran label dulu (pelajaran v8).** Detak dan CV IBI per label CPSC 2021 dibandingkan dengan AFDB dan SHDB-AF dan
  dilaporkan. Cek ini tidak dipakai untuk membuang window.
- **Syarat data cukup.** H1 dan H2 hanya dinilai bila CPSC 2021 berisi ≥ 5 subjek AFL dan ≥ 200 window AFL yang lolos. Kalau
  tidak, hasil ditulis "tidak cukup data", bukan lulus atau gagal.
- **H1 (utama): pemisahan AFL vs AFIB di data baru.** Untuk masing-masing fitur, AUROC AFL-vs-AFIB di CPSC 2021, arah AFL > AFIB,
  CI 95% bootstrap subjek (2.000×, strata subjek ber-AFL/tidak). **Lulus** bila AUROC ≥ 0,70 **dan** batas bawah CI > 0,50,
  untuk kedua fitur. H1 juga dihitung pada subkelompok AFL sulit (`atrium.pola_rr` ≠ 0) sebagai hasil sekunder (H1b), tanpa ambang.
- **H2 (sekunder): manfaat sebagai tahap gabungan.** Regresi logistik multinomial (scikit-learn, C = 1,0, bobot kelas seimbang)
  dengan masukan logit v3.1 (paket produksi, 3 nilai) + `kuantisasi` + `p_konsisten` (distandardisasi dari data latih).
  Dilatih di AFDB + SHDB-AF (logit v3.1 di AFDB dari OOF v3; di SHDB-AF dari paket v3.1), lalu diuji di CPSC 2021 dibandingkan
  dengan paket v3.1 saja, berpasangan, bootstrap subjek: Macro-F1 3 kelas dan recall AFL, vonis protokol (P ≥ 0,95 lebih baik,
  ≤ 0,05 lebih buruk). Tanpa penyetelan apa pun di CPSC 2021.
- Di AFDB, H2 juga dilaporkan dengan fold resmi 5×5 (latih di fold lain + SHDB-AF), berpasangan dengan v3.1, ditambah uji stres
  takikardia (protokol butir 9). Untuk uji stres sintetis tidak ada EKG, jadi fitur diberi nilai median data latih; ini dicatat.

## 5. Yang tidak dilakukan

- Tidak memilih fitur, ambang, atau hiperparameter dengan melihat CPSC 2021.
- Tidak membuang subjek atau window berdasarkan hasil.
- Tidak mengklaim produksi dari H1 saja. Penggantian paket butuh protokol butir 9–10 dan persetujuan pemilik repo.

## 6. Keluaran

`v15_kuantisasi_rr/`: `kuantisasi.py` (pemroses CPSC 2021 + evaluasi), `kuantisasi_rr.ipynb`, `README.md`, `laporan/`. Cache di
`../cache/v15/`.

## 7. Catatan

- Lisensi: CPSC 2021 CC BY 4.0; SHDB-AF ODC-BY; AFDB ODC-BY.
- Kalau CPSC 2021 tidak memisahkan AFL dari AFIB di anotasinya, v15 berhenti di syarat data dan dilaporkan begitu.

## 8. Penyimpangan dari rancangan

1. **(6 November 2026, sebelum fitur CPSC 2021 dihitung)** Anotasi ritme CPSC 2021 hanya memuat AFL dari **2 subjek** (±26 menit).
   Syarat data bagian 4 (≥ 5 subjek, ≥ 200 window AFL) tidak terpenuhi, jadi **hasil utama H1/H2 = "tidak cukup data"**. Selain itu
   11 rekaman di `RECORDS` (subjek 51 dan sebagian subjek 50) tidak ada di `SHA256SUMS.txt` resmi, jadi memang tidak dirilis.
2. Sebagai **analisis tambahan** (bukan hasil utama, dan tidak dipakai untuk klaim), H1 dan H2 dijalankan pada PTB-XL 1.0.3
   (lead I 100 Hz → 50 Hz, rekaman 10 s istirahat; AFLT 73 rekaman / 56 pasien, AFIB 1.514). Fitur, ambang, dan model tidak diubah.
   Batasan: PTB-XL bukan Holter/wearable, setiap rekaman hanya satu window, dan sebagian label berasal dari pembacaan otomatis
   perangkat yang dicek manusia.
3. H2 di AFDB (fold resmi + SHDB-AF) tetap dijalankan sesuai bagian 4.
