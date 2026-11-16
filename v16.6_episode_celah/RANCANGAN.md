# Rancangan v16.6: aturan episode AF dengan toleransi celah, dipilih di data lama dan divalidasi di data baru

Tanggal: 10 November 2026. Status: **dikunci sebelum sinyal SHDB-AF (seluruh rekaman), afpdb, incartdb, dan svdb dibaca untuk D-1 atau
aturan episode** (sha256 dan waktu di `logs/v16.6_praregistrasi.log`). Bagian 2–6 tidak diubah setelah hasil keluar; penyimpangan di
bagian 8.

Yang sudah dilihat sebelum dikunci:
- semua hasil v16.5 (aturan ≥ 30 s tanpa celah: D-1 NSRDB 1,23 palsu/jam, Se episode ≥ 30 s 0,91–0,98);
- eksplorasi deskriptif v16.5 setelah hasil: aturan ≥ 60 s menurunkan alarm palsu D-1 ke ≤ 0,22/jam di semua data negatif, tetapi Se
  episode ≥ 60 s turun ke 0,87–0,92. Dugaan: satu window yang terlewat memutus run;
- SHDB-AF: di v12 hanya sampel window (fitur atrium, tanpa D-1 dan tanpa aturan episode); jumlah label ritme per rekaman (77/98 rekaman
  punya `(AFIB`, 12 `(AFL`);
- header afpdb, incartdb, svdb (jumlah rekaman, komentar diagnosis), tanpa sinyal.

## 1. Latar

v16.5 menunjukkan D-1 + aturan episode ≥ 30 s hampir memenuhi target produk (gagal hanya karena NSRDB 1,23 > 1,0 palsu/jam). Aturan
yang lebih panjang menekan alarm palsu, tetapi run terputus oleh window tunggal yang terlewat. Perangkat komersial biasanya menjembatani
celah pendek. v16.6 memilih **durasi minimum dan toleransi celah** dari grid kecil di data yang sudah dipakai, dengan kriteria yang dikunci,
lalu menguji konfigurasi terpilih **satu kali** di data yang belum pernah dipakai.

## 2. Aturan dengan celah (tetap)

- Keputusan AF per window (sistem) diurutkan per rekaman. Run = rangkaian window AF di mana jarak awal antar window AF berurutan
  ≤ langkah × (g + 1) + 0,5 s. Artinya hingga g window di antaranya boleh tidak-AF atau tidak lolos cek kualitas (celah dijembatani).
- Alarm untuk run dengan rentang (awal window AF terakhir + 10 s − awal window AF pertama) ≥ min_span. **Semua window** di dalam rentang
  run itu (termasuk window celah) ditandai alarm, sehingga satu run = satu episode deteksi.
- g = 0 dan min_span = 30 s sama persis dengan aturan v16.5 (diperiksa dengan assert).

## 3. Tahap A, pengembangan (data yang sudah dipakai di v16.5)

- Data: AFDB, LTAFDB, MIT-BIH, CPSC 2021 (sensitivitas), NSRDB, EDB, latihan fisik (alarm palsu); sama persis dengan v16.5.
- Grid: min_span ∈ {30, 45, 60, 90} s × g ∈ {0, 1, 2} = 12 konfigurasi; sistem D-1 (d1 > −1).
- **Pemilihan (dikunci):** di antara konfigurasi dengan alarm palsu per jam ≤ 1,0 di NSRDB, EDB, dan latihan fisik, pilih yang
  **Se episode (referensi ≥ 30 s) minimum di empat data AF** terbesar. Bila selisih ≤ 0,005, pilih yang alarm palsu NSRDB + EDB lebih
  rendah. Bila tidak ada yang memenuhi batas alarm palsu, pilih yang alarm palsu NSRDB terendah dan laporkan itu.
- Referensi: paket v3.1 dengan konfigurasi yang sama (dan v3.1 + 30 s tanpa celah, = R v16.5).

## 4. Tahap B, validasi (data yang belum pernah dipakai untuk D-1 maupun aturan episode)

| Data | Isi | Peran | Unit bootstrap |
|---|---|---|---|
| **SHDB-AF** (`shdb-af` 1.0.1, ODC-BY) | 98 Holter 24 jam, 200 Hz, ECG1, 93 subjek, Jepang | Se episode + alarm palsu di waktu non-AF pasien AF | subjek (`AdditionalData.csv`) |
| **afpdb** (`afpdb` 1.0.0) | n01–n50 (tanpa PAF) dan p01–p50 (pasien PAF, di luar episode), 30 menit, 128 Hz, kanal 0; rekaman `c` (kelanjutan 5 menit, sebagian berisi awal AF) dan `t` (uji, label terpisah) **tidak dipakai** | alarm palsu (semua non-AF) | rekaman |
| **incartdb** (`incartdb` 1.0.0) | 75 rekaman 30 menit, 12-lead 257 Hz, lead I, 32 pasien | alarm palsu; AF hanya di beberapa rekaman (deskriptif) | pasien (komentar header) |
| svdb (`svdb` 1.0.0) | 78 rekaman 30 menit, aritmia supraventrikular, 128 Hz, kanal 0; **ritme tidak dianotasi** | deskriptif saja (alarm per jam; tidak dihitung "palsu" pasti) | rekaman |

- Label: label ritme = anotasi dengan aux diawali `(`; label berlaku sampai label ritme berikutnya. AF = `(AFIB`, `(AFL`, `(WPWAF`.
  Window yang seluruhnya di dalam interval AF = AF; seluruhnya di dalam interval non-AF = non-AF; yang melintasi batas atau sebelum label
  ritme pertama dibuang. afpdb: semua window non-AF (tidak ada `.atr`).
- Rantai prediksi sama dengan v16.4 (`uji_baru.proses_rekaman`: 50 Hz, detektor perangkat, cek kualitas, ECGFounder d1, paket v3.1).

## 5. Kriteria dan keputusan (dikunci)

- **Target produk** untuk D-1 + konfigurasi terpilih: (a) Se episode (referensi ≥ 30 s) di SHDB-AF ≥ **0,90**; (b) alarm palsu per jam
  ≤ **1,0** di SHDB-AF (waktu non-AF), afpdb, dan incartdb (titik).
- **H1:** alarm palsu per jam di SHDB-AF lebih rendah daripada v3.1 + konfigurasi yang sama: bootstrap subjek berpasangan 2.000×, seed 2026,
  P(Δ < 0) ≥ 0,95.
- **H2:** Se episode SHDB-AF tidak lebih buruk daripada v3.1 + konfigurasi yang sama: P(Δ > 0) > 0,05 (bootstrap subjek berpasangan).
- **Sekunder:** konfigurasi terpilih vs aturan v16.5 (30 s, g = 0) untuk D-1 di SHDB-AF: alarm palsu per jam dan Se episode (deskriptif +
  bootstrap); Se/+P durasi; alarm per jam svdb; alarm per pita detak non-AF.
- **Keputusan:** target (a) dan (b) terpenuhi serta H1 dan H2 lolos → "D-1 + konfigurasi terpilih" disebut **sistem kandidat terbaik untuk
  uji perangkat** (detektor AF 2 kelas, episode). Selain itu, laporkan kriteria yang gagal; konfigurasi tidak diubah di data validasi.
- Bukan pengganti paket v3.1 (3 kelas); tidak ada `model_export/` yang dibuat atau diubah tanpa persetujuan pemilik repo.

## 6. Yang tidak dilakukan

- Tidak menyetel ambang d1 (tetap −1), grid, atau aturan label setelah melihat data validasi. Tidak melatih apa pun.
- Data validasi tidak dipakai untuk memilih apa pun.

## 7. Keluaran

`v16.6_episode_celah/`: `celah.py`, `celah.ipynb`, `README.md`, `laporan/`. Cache `../cache/v16.6/` (per rekaman, ringkasan).

## 8. Penyimpangan dari rancangan

**Amandemen A (10 November 2026, sebelum sinyal data validasi dibaca; hash baru di log).** Hitungan label ritme yang sudah dilihat sebelum
dikunci menunjukkan incartdb hanya menandai perubahan dari ritme normal (`(AFIB` 2 rekaman, `(PREX` 4, `(WPWAF` 1; rekaman lain tanpa
label ritme), dan svdb tidak menganotasi ritme. Aturan "window sebelum label ritme pertama dibuang" akan menghapus hampir seluruh incartdb.
Perubahan: untuk **incartdb dan svdb**, waktu sebelum label ritme pertama (termasuk seluruh rekaman tanpa label ritme) dihitung **non-AF**
(konvensi MIT: ritme bawaan normal sampai ada label). Untuk SHDB-AF aturan tetap (setiap rekaman diawali `(N`). Tidak ada kriteria lain
yang berubah.
