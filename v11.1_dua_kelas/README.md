# v11.1_dua_kelas — evaluasi 2 kelas (Normal vs AF/AFL)

Versi minor, **evaluasi saja**: tidak ada training baru, dan tidak ada paket model yang diubah. Pertanyaannya: kalau AFL tidak
dibedakan dari AFIB, seberapa baik model yang sudah ada membedakan aritmia dari ritme normal? Pertanyaan ini muncul dari
target pemilik repo (Macro-F1 ≥ 0,90). Pada 3 kelas, target itu tidak tercapai oleh model berbasis interval RR, karena hanya AFL
yang cepat dan teratur yang bisa dikenali (lihat v12). Literatur AF berbasis Holter juga lazim menggabungkan AFIB + AFL ("AF₁").

Model yang dinilai: v1 (argmax) dan v3.1 dari OOF v3, serta enam setelan v11 (BiGRU, TCN, XGBoost; masing-masing bobot kelas
lunak = `bawaan` dan bobot penuh = `wp100` / `bobot_penuh`) dari `cache/v11/<model>/<setelan>/`.

| File | Isi |
|---|---|
| `dua_kelas.ipynb` | Menjalankan evaluasi, tabel & grafik per butir protokol, cek angka kesimpulan (`assert`), Kesimpulan |
| `dua_kelas.py` | Logika evaluasi (kedua aturan keputusan, butir 9 dan 10); menulis `../cache/v11.1/dua_kelas.json` |
| `laporan/` | `laporan_dua_kelas.pdf` (10 halaman) + `build_report.py` (dua tahap) |

## Cara menjalankan

Butuh venv v11 (`v11_model_tim/bersama`) dan cache v11 untuk keenam setelan di atas, serta `cache/oof_robust/` dan
`cache/dataset_v2_detector.npz`. Dari root repo:

```bash
python tools/run_notebook.py v11.1_dua_kelas/dua_kelas.ipynb   # ±45 menit CPU → cache/v11.1/dua_kelas.json
```

Lalu dari folder ini:

```bash
python laporan/build_report.py --extract  # angka & grafik → laporan/_build/ (assert terhadap OOF di cache)
python laporan/build_report.py            # → laporan/laporan_dua_kelas.pdf
```

`python dua_kelas.py` menghasilkan JSON yang sama tanpa notebook (±25 menit).

Notebook dijalankan 3 November 2026 di PC lokal (Linux, CPU saja; Python 3.13, numpy 2.5.4) lewat `tools/run_notebook.py`, 44 menit, tanpa error. Hasilnya identik dengan run skrip 2 November (4.268 nilai dicocokkan, 0 berbeda). Commit dasar `b18d6bc` + perubahan v11/v11.1 yang belum di-commit.

## Protokol

Sama dengan 3 kelas (AGENTS.md): 5 ulangan × 5 fold pasien, data detektor 50 Hz, metrik pooled OOF rata-rata ulangan, CI 95%
bootstrap pasien 2.000× (sampel yang sama dengan evaluasi v3/v11), perbandingan berpasangan vs v3.1 dan v1, bootstrap bersarang,
tanpa 04936, uji stres takikardia. Dua aturan keputusan dinilai:

| Aturan | Cara | Status |
|---|---|---|
| `ulang` | p(AF/AFL) = p(AFIB) + p(AFL); **satu** bias AF/AFL dipilih cross-fit (grid −7…+3) untuk Macro-F1 2 kelas | rencana awal |
| `gabung` | bias 3 kelas dipilih cross-fit seperti evaluasi biasa, lalu tebakan AFIB dan AFL disatukan | ditambahkan setelah uji stres aturan `ulang` memperlihatkan alarm baru (analisis tambahan) |

Pemeriksaan otomatis: Macro-F1 2 kelas v3.1 dan TCN bawaan untuk kedua aturan dihitung ulang dari OOF di `extract()` dan
harus sama dengan JSON. Selain itu, Macro-F1 bersarang dengan bobot semua-satu harus sama dengan cross-fit biasa (semua sistem,
`assert` di `dua_kelas.py`).

## Hasil

Angka berikut dicetak dari `laporan/_build/data.json`. Kolom "2 kelas" memakai aturan `gabung` kecuali disebut lain. P = peluang
lebih baik pada sampel bootstrap yang sama (≥ 0,95 lebih baik, ≤ 0,05 lebih buruk).

| Sistem | Macro-F1 3 kelas | 2 kelas, gabung [CI 95%] | Se / Sp | vs v3.1 Δ (P) | bersarang vs v1 (P) | tanpa 04936 | 2 kelas, ulang | alarm Normal teratur 80–95 bpm (gabung / ulang) | alarm Normal ≥ 120 bpm |
|---|---|---|---|---|---|---|---|---|---|
| v1 (argmax) | 0,744 | 0,922 [0,88–0,95] | 0,954 / 0,903 | −0,004 (0,20) | – | 0,922 | 0,922 | 0,3% / 0,3% | 94,4% |
| v3.1 | 0,832 | 0,926 [0,88–0,95] | 0,934 / 0,924 | – | +0,004 (0,52) | 0,927 | 0,916 | 0,1% / 18,6% | 95,2% |
| **BiGRU bawaan** | 0,821 | **0,938** [0,91–0,96] | 0,926 / 0,950 | +0,012 (1,00) | +0,016 (0,98) | 0,938 | 0,938 | 0,1% / 6,2% | 86,6% |
| BiGRU wp100 | 0,830 | 0,934 [0,90–0,96] | 0,938 / 0,935 | +0,009 (1,00) | +0,012 (0,96) | 0,933 | 0,924 | 0,0% / 22,4% | 90,2% |
| TCN bawaan | 0,752 | 0,933 [0,91–0,95] | 0,902 / 0,959 | +0,008 (0,74) | +0,011 (0,89) | 0,937 | 0,938 | 0,9% / 6,9% | 71,8% |
| TCN wp100 | 0,830 | 0,935 [0,91–0,96] | 0,919 / 0,950 | +0,010 (0,86) | +0,013 (0,94) | 0,936 | 0,932 | 0,5% / 18,7% | 84,4% |
| XGBoost bawaan | 0,730 | 0,916 [0,87–0,95] | 0,901 / 0,931 | −0,010 (0,07) | −0,006 (0,12) | 0,918 | 0,919 | 1,6% / 2,2% | 85,0% |
| XGBoost bobot penuh | 0,744 | 0,916 [0,87–0,95] | 0,905 / 0,929 | −0,009 (0,05) | −0,005 (0,12) | 0,919 | 0,918 | 1,8% / 3,6% | 82,8% |

- **Target ≥ 0,90 tercapai untuk 2 kelas** di semua sistem (0,916–0,938). Selisih dengan 3 kelas (+0,09 s.d. +0,19) hampir
  seluruhnya berasal dari kelas AFL.
- **Lebih baik dari v3.1:** BiGRU bawaan dan BiGRU wp100 (P = 1,00), terutama karena spesifisitas lebih tinggi (Δ Sp +0,026 /
  +0,011). TCN tidak terbedakan (P = 0,74 / 0,86), XGBoost cenderung lebih buruk (P = 0,07 / 0,05).
- **Bootstrap bersarang (butir 10):** BiGRU bawaan lebih baik dari v1 (P = 0,98) dan v3.1 (P = 0,96); BiGRU wp100 0,96 / 0,96.
  TCN wp100 vs v1 P = 0,94 (tidak terbedakan).
- **Tanpa 04936:** Macro-F1 semua sistem tetap 0,918–0,938, jadi tingginya angka 2 kelas tidak bergantung pada satu pasien.
  Namun keunggulan atas v3.1 tidak lagi pasti (BiGRU bawaan P = 0,92, BiGRU wp100 P = 0,91).
- **Alarm pada ritme normal teratur 80–95 bpm (temuan setelah melihat hasil).** Dengan aturan `ulang`, 18,6% (v3.1) dan 22,4%
  (BiGRU wp100) window Normal yang teratur (CV IBI < 0,03; 4.968 window) di 80–95 bpm menjadi alarm. Di uji stres sintetis,
  ambang BiGRU wp100 turun ke 85 bpm. Penyebabnya: AFDB punya 113 window AFL teratur di 84–92 bpm (109 dari 04936, 4 dari 08215).
  Bias 3 kelas menekan AFL kuat (bias AFL median −4,5 untuk v3.1), sedangkan bias 2 kelas tunggal kehilangan penekan itu. Aturan
  `gabung` memberi Macro-F1 yang hampir sama dengan alarm 0,0–1,8%.
- **Takikardia (butir 9) tetap masalah.** Window Normal ≥ 120 bpm yang menjadi alarm: 95% (v3.1), 87% (BiGRU bawaan), 72%
  (TCN bawaan). Ambang uji stres sintetis dengan aturan `gabung`: 115–130 bpm (v3.1, diwakili kontrol v7: 120 / 120 / 120 / 115).
  Menggabungkan kelas tidak memisahkan ritme cepat-teratur dari aritmia.

## Kesimpulan

1. Laporkan Macro-F1 2 kelas (AF/AFL vs Normal) berdampingan dengan 3 kelas. Ini sebanding dengan cara literatur menilai deteksi
   AF, dan sesuai batas informasi interval RR (AFL yang tidak cepat-teratur tidak terpisah dari AFIB; v8–v12).
2. Kalau keluaran 2 kelas dipakai, ambil keputusan **3 kelas lalu digabung**, bukan bias 2 kelas yang dipilih ulang.
3. BiGRU bawaan adalah kandidat keluaran 2 kelas terbaik (0,938; lebih baik dari v3.1 dan v1 pada uji bersarang), tetapi
   keunggulannya atas v3.1 tidak pasti tanpa 04936 dan belum dikemas. Paket produksi tetap v3.1.

## Batasan

- Validasi silang internal AFDB (23 pasien). Angka literatur (F1 AF₁ 0,89–0,92) berasal dari data uji eksternal, jadi tidak
  sebanding langsung.
- Aturan `gabung` dan pemilihan "terbaik" dilakukan setelah melihat hasil. Karena itu vonis memakai uji bersarang.
- Model dilatih untuk 3 kelas; model yang dilatih langsung untuk 2 kelas tidak diuji. Metrik episode (butir 7) belum dihitung.
- Uji stres sintetis v3.1 memakai kontrol v7 (resep v3.1, ulangan 0), bukan paket produksi. Ambang paket produksi 3 kelas di v3.2
  adalah 115 bpm.
