# v3.2_uji_takikardia — uji stres takikardia & analisis sensitivitas

Versi minor, **evaluasi saja**: tidak ada training dan paket produksi tidak diubah. Versi ini menindaklanjuti review
eksternal atas paket v3.1. Review itu menemukan bahwa ritme cepat yang teratur (sinus takikardia) ditebak AFIB/AFL.
Protokol v3 tidak menangkap masalah ini, karena data uji AFDB hampir tidak punya ritme normal yang cepat.

| # | Analisis | Model |
|---|---|---|
| 1 | Ritme teratur sintetis 60–170 bpm × 4 jenis variasi detak (50 window per titik) | v1 (`final.joblib`), v2, v3.1: hanya yang menyimpan bobot |
| 2 | Window Normal/AFL **asli** per pita detak jantung | semua model yang punya OOF (v1–v6) |
| 3 | Sensitivitas tanpa pasien 04936 (58% window AFL; reviewer meragukan labelnya) | v1, v2, v3.1 |
| 4 | Bootstrap bersarang: bias dipilih ulang di setiap sampel bootstrap | v3.1 vs v2, v1 |
| 5 | Simulasi aturan produk "detak cepat & teratur → takikardia teratur, perlu dicek" | v3.1 |

| File | Isi |
|---|---|
| `uji_takikardia.ipynb` | Menjalankan kelima analisis, tabel & grafik, kesimpulan; menulis `../cache/v3_2/ringkasan.json` |
| `takikardia.py` | Logika: IBI sintetis, pemuat paket v1/v2/v3.1, metrik per pita detak, bootstrap bersarang, aturan produk |
| `laporan/` | `laporan_uji_takikardia.pdf` (6 halaman) + `build_report.py` |

## Hasil

**1. Uji sintetis.** Detak terendah di mana kurang dari separuh window ditebak Normal:

| Model | tanpa variasi | seragam ±10 ms | Gauss sd 10 ms | Gauss sd 20 ms |
|---|---|---|---|---|
| v1 | 115 bpm | 110 bpm | 110 bpm | 110 bpm |
| v2 produksi | 115 bpm | 115 bpm | 115 bpm | 110 bpm |
| v3.1 produksi | 115 bpm | 115 bpm | 115 bpm | 115 bpm |

Masalah ini sudah ada sejak v1 dan tidak dibuat oleh revisi. Ritme yang sangat teratur ditebak AFL, sedangkan ritme
dengan variasi sedikit lebih besar ditebak AFIB.

**2. Window asli.** Bagian window Normal yang ditebak AFIB/AFL:

| Pita detak | Window | v1 | v2 | v3.1 | v6 late fusion |
|---|---|---|---|---|---|
| 100–110 bpm | 8.766 | 14% | 18% | 13% | 7% |
| 110–120 bpm | 568 | 66% | 78% | 59% | 50% |
| ≥ 120 bpm | 100 | 94% | 100% | 95% | 86% |

Window Normal cepat di AFDB sendiri tidak teratur (koefisien variasi median 0,18). Karena itu, skala masalah untuk sinus
takikardia sungguhan belum bisa diukur dengan data ini.

**3–4. Ketahanan kesimpulan v3.1.**

| Uji | v3.1 vs v2 | v3.1 vs v1 |
|---|---|---|
| v3 (bias tetap, semua pasien) | P = 1,00 | P = 0,98 |
| Bootstrap bersarang | Δ +0,053 [+0,007 … +0,112], **P = 0,993** | Δ +0,088 [+0,005 … +0,149], **P = 0,978** |
| Tanpa 04936 | P = 1,00 | **P = 0,94 (tidak terbedakan)** |

Tanpa 04936, nilai Macro-F1 v3.1 menjadi 0,859 [0,62–0,92], v2 0,763, dan v1 0,720.

**5. Aturan produk** (≥110 bpm & CV IBI ≤ 0,05). Hampir semua ritme sintetis teratur tertangkap aturan ini, dan AFIB
hampir tidak tersentuh (0,2%). Namun **86% AFL yang berhasil dideteksi ikut dialihkan**, karena AFL 2:1 juga cepat dan
teratur. Manfaat aturan ini tidak bisa diukur di AFDB.

**Kesimpulan.**
- Paket produksi tetap v3.1. `BACKEND.md` diberi peringatan takikardia.
- Uji stres takikardia jadi wajib di protokol (AGENTS.md).
- Klaim "P = 1,00" di v3.1 dikoreksi menjadi P = 0,993 (bootstrap bersarang).
- Keunggulan atas v1 bergantung pada 04936.
- Aturan produk menunggu keputusan pemilik produk.
- Perbaikan sesungguhnya butuh data latih yang memuat sinus takikardia (v7, mis. Chapman–Shaoxing).

## Lingkungan hasil

Dijalankan 16 Oktober 2026 di WSL2 (Debian), PC desktop. Semua analisis memakai CPU; pertama kali ±10 menit
(prediksi sintetis + kandidat bias bootstrap bersarang), lalu dari cache `../cache/v3_2/prediksi.npz` dalam hitungan detik.

## Menjalankan (dari root repo)

```bash
python tools/run_notebook.py v3.2_uji_takikardia/uji_takikardia.ipynb
python v3.2_uji_takikardia/laporan/build_report.py --extract
python v3.2_uji_takikardia/laporan/build_report.py
```

Prasyarat:
- `../cache/oof_robust/` berisi OOF v3–v6 dan `folds.npz`.
- `../cache/dataset_v2_detector.npz`.
- Untuk ukur ulang v1: OOF dari `v3_evaluasi_robust/eval_versi_lama.py`.
- Paket model `v1_lama/final.joblib`, `v2_revisi/model_export/`, `v3.1_kalibrasi_produksi/model_export/`. Paket-paket ini
  hanya dibaca.

### Uji stres untuk model baru (wajib, lihat AGENTS.md)

```python
import sys; sys.path.insert(0, "../v3.2_uji_takikardia")
import takikardia as T
# prediksi(seqs) -> id kelas (n,); seqs = daftar (X, M) dari hrv_core.build_sequence, praproses yang sama dengan produksi
tab = T.stress_test({"model_baru": prediksi})
print({j: T.normal_threshold(t) for j, t in tab["model_baru"].items()})   # v3.1: 115 bpm di semua jenis variasi
```
