# v16.8 — gerbang kualitas sinyal (SQI templat QRS) untuk S1

> **Status: selesai (10 November 2026).** Gerbang **gagal**:
> di afpdb + derau otot `ma` pada SNR 6 dB alarm palsu tidak turun (S1 1,36 → S1G 1,34/jam, P(Δ < 0) = 0,65; target ≤ 1,0). S1 tetap
> sistem terbaik untuk deteksi AF 2 kelas. Laporan: [laporan/laporan_v16_8_gerbang_kualitas.pdf](laporan/laporan_v16_8_gerbang_kualitas.pdf).

Rancangan dikunci sebelum SQI/SNR dihitung: [RANCANGAN.md](RANCANGAN.md) (+ amandemen A; sha256 di `logs/v16.8_praregistrasi.log`).
Logika di `gerbang.py`, notebook `gerbang.ipynb` (menulis `../cache/v16.8/ringkasan.json`), laporan dua tahap di
`laporan/build_report.py` (5 halaman).

## Cara menjalankan

```bash
python tools/run_notebook.py v16.8_gerbang_kualitas/gerbang.ipynb   # butuh cache v16.3–v16.7 + data afpdb, nstdb, ltafdb, mitdb, nsrdb, icentia_v167
cd v16.8_gerbang_kualitas && python laporan/build_report.py --extract && python laporan/build_report.py
```

Pemrosesan afpdb berderau (600 rekaman) dan SQI ±6 menit di PC desktop (RTX 5060, 10 November 2026).

## Hasil

- **Tahap A** memilih q = 0,9 (SQI = median korelasi detak ke templat QRS lokal, `pembersih.selaraskan` v3.4): di nstdb alarm palsu
  5,30 → 0,00/jam, tetapi q ada di batas atas grid dan mengeluarkan 33% window nstdb.
- **V1 gagal**: afpdb + `ma` 6 dB: 1,36 → 1,34 alarm palsu/jam (9,5% window digerbang).
- **V2 lolos**: Se episode Icentia 0,940 → 0,931.
- **V3 lolos**: alarm palsu Icentia non-AF 0,50 → 0,46/jam.

| Data | S1 palsu/jam | S1G palsu/jam | Window digerbang |
|---|---|---|---|
| afpdb bersih | 0,89 | 0,88 | 4,9% |
| afpdb + `ma` 12 / 6 / 0 dB | 1,19 / 1,36 / 1,92 | 1,23 / 1,34 / 1,91 | 6,7 / 9,5 / 19,0% |
| afpdb + `em` 12 / 6 / 0 dB (sekunder) | 1,21 / 2,18 / 6,05 | 0,87 / 1,34 / 3,11 | 8,5 / 14,5 / 35,4% |
| Icentia pasien tanpa AF | 0,50 | 0,46 | 7,5% |

Templat QRS menangkap derau elektroda (`em`, jenis derau yang dipakai memilih q) tetapi tidak derau otot (`ma`). Gerbang derau berikutnya
butuh fitur lain (mis. daya frekuensi tinggi) dan data derau perangkat.
