# v12_shdb_atrium — aktivitas atrium dari EKG + data Holter SHDB-AF

**Status: berhenti di gerbang (2 November 2026).** Gerbang yang ditetapkan sebelum hasil (RANCANGAN 7.2) gagal dua kali: pada
rancangan awal dan setelah satu revisi pembatalan QRST yang juga ditetapkan sebelum dijalankan (RANCANGAN 14). Sesuai aturan,
training fusi dan CV bersarang tidak dijalankan. Tidak ada model yang dikemas.

**Pertanyaan.** Model RR hanya mendeteksi AFL yang cepat dan teratur: recall 0,92 untuk pola itu, 0,00–0,21 untuk pola lain
(v3.1 dan TCN `wp100` v11). Apakah aktivitas atrium dari EKG satu lead 50 Hz, ditambah pasien Holter SHDB-AF, bisa membedakan
AFL yang tidak cepat-teratur dari AF? Rancangan lengkap: [RANCANGAN.md](RANCANGAN.md).

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Rancangan, aturan gerbang, revisi 1, hasil gerbang (bagian 13) |
| `atrium.py` | Unduh SHDB-AF (S3 + SHA256), pembaca WFDB format 16/212 dan `.atr` (dibuktikan sama dengan wfdb), window SHDB-AF dengan front-end produksi, pembatalan QRST (awal dan revisi 1), 11 fitur atrium, gerbang, gambar contoh |

Kode versi lama dipakai lewat `sys.path`: `hrv_core` (v2), `ekg.preprocess_record`/`normalize_window` (v6), `icentia.label_windows`
(v10, logika ekstraktor AFDB v2). Keluaran di `cache/v12/` (tidak di-commit): `afdb_fitur.npz`, `shdb/<rekaman>_k0.npz`,
`fitur_r1.npz`, `gerbang.json`, `gerbang_r1.json`, `gerbang/contoh_*.png`. Data di `data/shdb_af/1.0.1/`.

## Menjalankan (dari folder ini)

```bash
python atrium.py unduh      # 98 sinyal beranotasi SHDB-AF (6,71 GB), dicek SHA256, bisa dilanjutkan
uv run --no-project --with "wfdb>=4.1" --with numpy python atrium.py bukti   # pembaca = wfdb (wfdb tidak ada di venv)
python atrium.py afdb        # fitur atrium untuk 165.978 window AFDB (potongan EKG harus sama dengan cache v6)
python atrium.py shdb        # window SHDB-AF (2 proses; RAM PC 15 GB)
python atrium.py gerbang     # gerbang rancangan awal + gambar contoh
python atrium.py revisi1     # fitur revisi 1 pada window yang sama + gerbang + gambar
```

## Data

- **AFDB**: window dataset v2 (`cache/dataset_v2_detector.npz`) dan potongan EKG v6; fitur dihitung ulang dari sinyal kanal 0, potongan
  EKG sama bit demi bit dengan cache v6.
- **SHDB-AF** (PhysioNet `shdb-af` 1.0.1, Tsutsui dkk. 2024, ODC-BY 1.0): Holter 24 jam Jepang, 98 rekaman beranotasi per detak oleh
  seorang fellow kardiologi, 93 subjek. ECG1 (CC5), front-end produksi 50 Hz, label window cara ekstraktor AFDB v2, batas per subjek
  (RANCANGAN 6.2). 0 window tidak sejajar.
- Window gerbang: AFDB 99.000/65.826/1.152 Normal/AFIB/AFL (7 pasien AFL); SHDB-AF 36.403/33.943/6.609 (12 subjek AFL).

## Hasil gerbang

Aturan (ditetapkan sebelum hasil): fitur `df_hz` (frekuensi atrium dominan) dan `organisasi` harus memisahkan AFL dari AFIB **dan**
dari Normal dengan arah yang sama di AFDB dan SHDB-AF, AUROC ≥ 0,60 (atau ≤ 0,40) di keduanya. AUROC dengan AFL sebagai kelas positif.

| Fitur gerbang | Perbandingan | Awal: AFDB / SHDB-AF | Revisi 1: AFDB / SHDB-AF |
|---|---|---|---|
| `df_hz` | AFL vs AFIB | 0,586 / 0,418 → **gagal** | 0,458 / 0,571 → **gagal** |
| `df_hz` | AFL vs Normal | 0,387 / 0,359 → lolos | 0,638 / 0,657 → lolos |
| `organisasi` | AFL vs AFIB | 0,578 / 0,740 → **gagal** | 0,486 / 0,608 → **gagal** |
| `organisasi` | AFL vs Normal | 0,766 / 0,778 → lolos | 0,618 / 0,677 → lolos |

Keduanya gagal pada perbandingan yang penting, AFL vs AFIB. Pemisahan dari Normal lolos.

**Revisi 1** (penyelarasan sub-sampel di 250 Hz, skala amplitudo per detak, daerah QRS diinterpolasi) bekerja seperti yang dimaksud:
lonjakan QRS hilang dari sinyal sisa, dan AFL 2:1 pasien 08378 kini terbaca DF 4,5–4,7 Hz (siklus flutter ±4,6 Hz; rasio konduksi
1,97–2,07), dibanding 7–8 Hz sebelumnya. Tetapi AF di 50 Hz juga terbaca ±5 Hz, dan indeks organisasinya tumpang tindih dengan AFL,
jadi kedua fitur spektral itu tidak memisahkan AFL dari AFIB secara konsisten antar dataset.

**Semua fitur** (eksploratif, hanya dua fitur pertama yang menjadi gerbang):

| Fitur | AFL vs AFIB awal (AFDB / SHDB-AF) | AFL vs AFIB revisi 1 | AFL sulit vs AFIB revisi 1 |
|---|---|---|---|
| `df_hz` | 0,586 / 0,418 | 0,458 / 0,571 | 0,432 / 0,559 |
| `organisasi` | 0,578 / 0,740 | 0,486 / 0,608 | 0,472 / 0,553 |
| `entropi` | 0,262 / 0,237 | 0,443 / 0,247 | 0,434 / 0,268 |
| `autokorelasi` | 0,341 / 0,536 | 0,386 / 0,575 | 0,468 / 0,673 |
| `amp_atrium` | 0,476 / 0,531 | 0,255 / 0,398 | 0,290 / 0,429 |
| `rasio_energi` | 0,428 / 0,590 | 0,205 / 0,403 | 0,265 / 0,470 |
| `p_konsisten` | 0,965 / 0,797 | 0,965 / 0,797 | 0,916 / 0,740 |
| `rasio_konduksi` | 0,329 / 0,483 | 0,190 / 0,585 | 0,266 / 0,684 |
| `jarak_bulat` | 0,193 / 0,297 | 0,198 / 0,303 | 0,228 / 0,355 |
| `kuantisasi` | 0,845 / 0,747 | 0,837 / 0,743 | 0,825 / 0,691 |
| `n_detak` | 0,905 / 0,465 | 0,905 / 0,465 | 0,773 / 0,321 |

"AFL sulit" = window AFL di luar pola RR cepat-teratur, yaitu bagian yang gagal dideteksi model RR.

## Kesimpulan

- **v12 berhenti sesuai aturan.** Fitur spektral aktivitas atrium (frekuensi dominan, organisasi) dari EKG satu lead 50 Hz tidak
  memisahkan AFL dari AFIB secara konsisten di AFDB dan SHDB-AF, bahkan setelah pembatalan QRST diperbaiki.
- **Infrastruktur yang berguna untuk versi berikutnya:** pembaca WFDB tanpa wfdb (terbukti sama), 98 rekaman Holter SHDB-AF terunduh
  dan terverifikasi, window SHDB-AF dengan front-end produksi, dan pembatalan QRST revisi 1 yang terbukti bekerja secara visual.
- **Temuan eksploratif (dilihat setelah hasil, bukan hasil gerbang):** dua fitur di luar gerbang memisahkan AFL dari AFIB searah di
  kedua dataset, juga untuk AFL sulit: konsistensi segmen sebelum QRS `p_konsisten` (0,965 / 0,797) dan kuantisasi RR terhadap siklus flutter `kuantisasi` (0,837 / 0,743).
  Karena dipilih setelah melihat data yang sama, angka ini optimistis. Kalau mau diuji, jadikan hipotesis baru yang ditetapkan
  sebelumnya, idealnya dengan data yang belum dilihat (dataset Holter lain atau rekaman perangkat SIHEDAF).
- **Yang tidak bisa disimpulkan:** apakah cabang EKG gagal secara umum. Yang gagal adalah fitur spektral pada EKG 50 Hz; model
  bentuk gelombang yang dilatih (CNN) pada data Holter yang lebih banyak tidak diuji di sini. Label AFL SHDB-AF dibuat satu orang.
- Target Macro-F1 3 kelas ≥ 0,90 tetap tidak tercapai dengan informasi yang ada. Macro-F1 2 kelas (AF/AFL vs Normal) ±0,93 (perkiraan,
  belum dengan bias 2 kelas dan CI) adalah jalur yang sejalan dengan literatur.

## Lingkungan

PC CachyOS, RAM 15 GB, venv `~/.venvs/sihedaf-v11` (Python 3.13, numpy 2.5.4, scipy 1.18.1); bukti pembaca lewat `uv run --with wfdb`.
Unduhan 16 menit, window SHDB-AF 2,6 menit, fitur AFDB 1,5 menit, fitur revisi 1 6,5 menit (termasuk menghitung ulang fitur awal
sebagai bukti window identik), semuanya dengan 2 proses. Tidak ada GPU yang dipakai.
Kode: commit `b18d6bc` + perubahan yang belum di-commit.

**Atribusi data.** MIT-BIH Atrial Fibrillation Database (Moody & Mark, 1983), PhysioNet, ODC-BY 1.0. SHDB-AF: Tsutsui K, Biton Brimer S,
Ben-Moshe N, dkk. SHDB-AF: a Japanese Holter ECG database of atrial fibrillation. arXiv:2406.16974 (2024); PhysioNet,
doi:10.13026/10mk-y852, ODC-BY 1.0.
