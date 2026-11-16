# v16.7 — uji konfirmasi S1 (D-1 + episode AF ≥ 30 s) di wearable kontinu (Icentia11k) dan uji stres derau (nstdb)

> **Status: selesai (10 November 2026). S1 terkonfirmasi sebagai hipotesis primer untuk wearable kontinu.** Di 600 pasien Icentia11k
> (patch satu lead, 534 jam), S1 menangkap 94,0% episode AF ≥ 30 s [91,6–96,0] dengan 0,50 alarm palsu per jam [0,21–0,86].
> v3.1 + 30 s: 89,9% dan 2,33 per jam. Keempat kriteria primer lolos. Uji derau (H3, tidak ikut keputusan) gagal tipis: 1,01 alarm
> palsu/jam pada SNR 12 dB, 5–8 per jam pada ≤ 6 dB. Laporan: [laporan/laporan_v16_7_konfirmasi_wearable.pdf](laporan/laporan_v16_7_konfirmasi_wearable.pdf).

## Tujuan

S1 adalah kandidat terbaik untuk deteksi AF 2 kelas, tetapi buktinya dari data Holter, dan di v16.6 hanya analisis sekunder. Kelemahan
D-1 yang tercatat ada di EKG wearable. v16.7 menguji S1 **sebagai hipotesis primer** di rekaman wearable kontinu, plus ketahanan terhadap
derau. Rancangan dikunci sebelum sinyal dibaca: [RANCANGAN.md](RANCANGAN.md) (sha256 di `logs/v16.7_praregistrasi.log`).

- **Icentia11k** (CC BY-NC-SA, hanya evaluasi):
  - 300 pasien AF: satu segmen ±70 menit dengan episode AF ≥ 30 s yang tidak pernah diproses v10;
  - 300 pasien tanpa AF yang tidak pernah dipakai.
- **nstdb:** 118/119 MIT-BIH + derau elektroda/gerak, SNR 24 … −6 dB.

## Berkas

| Berkas | Isi |
|---|---|
| `RANCANGAN.md` | Sistem, seleksi Icentia, label, target, H1–H3 (dikunci) |
| `konfirmasi.py` | Seleksi segmen dari pemindaian anotasi v10, unduh Icentia (S3 + SHA256), pemrosesan per rekaman (mesin v16.4, bisa dibagi ke beberapa proses), label ritme Icentia (parser v10), pemuat nstdb/mitdb |
| `konfirmasi.ipynb` | Notebook utama; menulis `../cache/v16.7/ringkasan.json` |
| `laporan/build_report.py`, `laporan/laporan_v16_7_konfirmasi_wearable.pdf` | Laporan dua tahap (5 halaman) |

## Cara menjalankan

```bash
cd v16.7_konfirmasi_wearable && python -c "import konfirmasi as K; K.unduh_icentia()" && cd ..     # 600 segmen, 1,28 GB, ±1,2 menit
python tools/unduh_physionet.py nstdb 1.0.0 --ke nstdb --sumber physionet --thread 4 --pola "*.dat" "*.hea" "*.atr" RECORDS
python tools/run_notebook.py v16.7_konfirmasi_wearable/konfirmasi.ipynb   # pemrosesan juga bisa dibagi: K.jalankan('Icentia', b, 2)
cd v16.7_konfirmasi_wearable && python laporan/build_report.py --extract && python laporan/build_report.py
```

nstdb tidak ada di mirror S3 (pakai `--sumber physionet`). Pemrosesan 600 segmen ±5 menit dengan 2 proses (RTX 5060), notebook ±1 menit.
Dijalankan 10 November 2026 di PC desktop.

## Hasil

**Icentia11k** (bootstrap pasien berpasangan 2.000×):

| Kelompok | Sistem | Alarm palsu/jam | Se episode ≥ 30 s | Se durasi | +P durasi |
|---|---|---|---|---|---|
| 300 pasien AF (245 jam, 3.975 episode ≥ 30 s) | **S1 = D-1 + 30 s** | 0,03 | **0,940** [0,916–0,960] | 0,856 | 1,000 |
| | R = v3.1 + 30 s | 0,40 | 0,899 | 0,813 | 0,994 |
| | S3 = (D-1 atau v3.1) + 30 s (deskriptif) | 0,42 | 0,970 | 0,894 | 0,994 |
| 300 pasien tanpa AF (289 jam) | **S1** | **0,50** [0,21–0,86] | – | – | – |
| | R | 2,33 | – | – | – |
| | S3 | 2,84 | – | – | – |

- Target Se ≥ 0,90: **lolos**. Target alarm palsu ≤ 1,0/jam: **lolos**.
- H1 (alarm palsu S1 < R): Δ −1,84 [−2,66; −1,00], P = 1,000 → **lolos**.
- H2 (Se S1 bukan lebih buruk): Δ +0,041 [+0,021; +0,062] → **lolos** (bahkan lebih baik).
- Alarm palsu S1 terkumpul di sedikit pasien: 26 dari 300 pasien tanpa AF punya ≥ 1 alarm, 20 di atas 1/jam, tertinggi 32,7/jam.
- Window non-AF ≥ 120 bpm: S1 0,7% vs v3.1 + 30 s 52%.

**nstdb** (alarm palsu per jam; ±1 jam per SNR):

| SNR | bersih | 24 dB | 18 dB | 12 dB | 6 dB | 0 dB | −6 dB |
|---|---|---|---|---|---|---|---|
| S1 | 0 | 0 | 0 | 1,01 | 7,42 | 7,62 | 5,45 |
| R (v3.1) | 0 | 0 | 1,00 | 3,02 | 7,42 | 7,62 | 10,91 |

H3 (S1 ≤ 1,0/jam pada SNR ≥ 12 dB) **gagal tipis**: satu alarm dalam ±1 jam pada 12 dB. Derau berat lolos cek kualitas perangkat.

## Keputusan

**S1 = D-1 + aturan episode AF ≥ 30 s terkonfirmasi sebagai sistem terbaik untuk deteksi AF 2 kelas** di Holter (v16.2–v16.6) dan
wearable kontinu (v16.7). Kelemahan yang tersisa:
- derau berat: butuh gerbang kualitas sinyal, dirancang dan diuji dengan data derau baru atau rekaman perangkat;
- EKG pendek sesaat (10–60 s);
- tidak membedakan AFIB dari AFL.

Klasifikasi 3 kelas tetap paket v3.1. Mengemas S1 sebagai paket uji perangkat (versi minor) perlu persetujuan pemilik repo; tidak ada
`model_export/` yang dibuat atau diubah.
