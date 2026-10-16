# Panduan backend — migrasi ke paket v3.1

Dokumen ini menggantikan `v1_lama/dokumentasi.md` (sistem lama `final.joblib` + `find_peaks`).
Laporan v3 menunjukkan sistem lama, bila dipakai utuh dengan front-end aslinya, hanya mencapai Macro-F1 ±0,42
dengan recall Normal ±34%. Paket v3.1 mencapai Macro-F1 0,832 [0,67–0,92] pada data uji detektor 50 Hz.

## 1. File yang disalin ke server

```
hrv_core.py          ← v2_revisi/hrv_core.py          (praproses: deteksi R-peak, IBI, cek kualitas)
sihedaf_models.py    ← v2_revisi/sihedaf_models.py    (arsitektur)
predictor.py         ← v2_revisi/predictor.py         (ECGPredictor)
model_export/
  sihedaf_v3_1.pt    ← v3.1_kalibrasi_produksi/model_export/
  sihedaf_v3_1.json  ← v3.1_kalibrasi_produksi/model_export/
```

Dependensi: `numpy`, `scipy`, `torch` (CPU cukup; latensi ±0,3 ms per window 10 s). Tidak perlu `joblib`, `wfdb`,
`lightgbm`, atau GPU.

## 2. Pemakaian

```python
from predictor import ECGPredictor

P = ECGPredictor("model_export", name="sihedaf_v3_1")        # muat sekali saat server start

hasil = P.predict_ecg(sinyal_ekg, fs=50)    # sinyal mentah 1-D, fs = sampling rate ASLI perangkat
hasil = P.predict_ibi(ibi_ms)               # jika perangkat sudah mengirim interval RR, dalam MILIDETIK
```

`predict_ecg` memotong sinyal menjadi window 10 detik (geser 5 detik), memprediksi tiap window, lalu meringkas
(rata-rata probabilitas window yang valid):

```json
{
  "summary": {"status": "ok", "class_id": 2, "class_name": "AFL (Atrial Flutter)", "confidence": 0.89,
              "probabilities": {"Normal": 0.04, "AFIB (Atrial Fibrillation)": 0.07, "AFL (Atrial Flutter)": 0.89},
              "duration_sec": 60.0, "n_rpeaks": 147, "heart_rate_bpm": 146.7, "n_windows": 11, "n_windows_valid": 11},
  "windows": [{"status": "ok", "start_sec": 0.0, "class_name": "AFL (Atrial Flutter)", "confidence": 0.9, "...": "..."},
              {"status": "tidak_terbaca", "start_sec": 5.0, "reason": "2 IBI di luar 300–2000 ms"}]
}
```

Angka contoh di atas ilustratif. Formatnya sama dengan keluaran nyata, dan contoh nyata ada di sel demo
`kalibrasi_produksi.ipynb`.

Status yang harus ditangani:

| Kondisi | Hasil |
|---|---|
| Sinyal/IBI terbaca | `status: "ok"` + kelas & probabilitas |
| Window terlalu sedikit detak atau IBI tidak fisiologis | `status: "tidak_terbaca"` + `reason`. Tampilkan "sinyal tidak terbaca", **jangan** diganti tebakan |
| Semua window tidak terbaca | `summary.status = "tidak_terbaca"` |
| IBI dikirim dalam detik (median < 10) | `ValueError`. Perbaiki di sisi pengirim |
| File model hilang / checksum salah / konstanta praproses beda | Exception saat `ECGPredictor(...)`. Server harus gagal start, bukan jalan dengan model acak |

## 3. Perbedaan dengan sistem lama (v1)

| | Lama (`final.joblib`) | Baru (v3.1) |
|---|---|---|
| Input | Matriks (30, 2) yang disiapkan backend sendiri | Sinyal EKG mentah + `fs`, atau IBI (ms) |
| Deteksi R-peak | `find_peaks` polos di backend, tanpa filter, maksimal 150 bpm | `hrv_core.detect_rpeaks` (Pan-Tompkins 50 Hz, sampai 240 bpm), **kode yang sama dengan training** |
| Cek kualitas | Tidak ada | Window ditolak bila IBI tidak fisiologis / terlalu sedikit |
| Panjang sinyal | Hanya 10 detik pertama | Seluruh sinyal, per window 10 s + ringkasan |
| Format model | pickle (`joblib`) | `state_dict` + JSON, checksum SHA-256 |
| Bias kelas | Tidak ada (argmax) | `[0.0, -0.5, -4.5]` dari metadata |
| Jika file hilang | Fallback diam-diam ke bobot acak | Error |

## 4. Checklist migrasi

- [ ] Hapus pemakaian `final.joblib`, `cek lama/predictor.py`, dan kode `find_peaks` di backend.
- [ ] Kirim sinyal mentah beserta `fs` aslinya. Resampling ke 50 Hz dilakukan `hrv_core`, jangan di-resample sendiri.
- [ ] Jika perangkat mengirim IBI, pastikan satuannya milidetik.
- [ ] Tangani `status: "tidak_terbaca"` di UI.
- [ ] Log `P.meta["version"]` (= `"3.1"`) dan `P.meta["weights_sha256"]` di setiap respons atau saat start, supaya hasil bisa ditelusuri.
- [ ] Uji cepat setelah deploy: `P.predict_ibi([800, 810, 790, 805, 800, 795, 810])` → `Normal`.

## 5. Rollback

Bobot v3.1 identik dengan v2, hanya metadata yang berbeda. Untuk kembali ke perilaku v2, ganti file model menjadi
`v2_revisi/model_export/sihedaf_v2.{pt,json}` dan `name="sihedaf_v2"`. Kode Python tidak perlu diubah.

## 6. Batasan yang perlu diketahui tim backend

> **⚠ Detak cepat yang teratur dilaporkan sebagai AFIB/AFL.** Pada uji ritme teratur sintetis (variasi detak kecil),
> paket ini memprediksi **Normal sampai ±110 bpm**, tetapi **AFIB atau AFL mulai ±115 bpm** (20 dari 20 window).
> Penyebabnya, data latih (AFDB) hampir tidak punya ritme normal yang cepat: hanya 0,1% window Normal di atas 120 bpm,
> sedangkan 78% window AFL di atas 120 bpm. Jadi model belajar "cepat berarti tidak normal". Akibatnya, sinus takikardia
> (naik tangga, olahraga, demam, cemas) kemungkinan menjadi **sumber alarm palsu terbesar** di pemakaian sehari-hari.
> Sampai ada model yang dilatih dengan contoh sinus takikardia, jangan tampilkan hasil AFIB/AFL pada detak cepat
> yang teratur sebagai diagnosis. Analisis lengkap dan simulasi aturan produk ("takikardia teratur, perlu dicek")
> ada di `../v3.2_uji_takikardia/`.

- Model dilatih dan diuji pada AFDB (23 pasien) yang disimulasikan ke 50 Hz, **belum** pada data perangkat asli
  yang dilabeli dokter.
- Deteksi AFL masih lemah: recall AFL 0,59 dengan CI 0,13–0,91. Hasil AFL sebaiknya ditampilkan sebagai "indikasi,
  perlu konfirmasi", bukan diagnosis.
