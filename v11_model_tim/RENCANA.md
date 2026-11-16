# v11 Enam Model Tim: Rencana Implementasi

> **Untuk agent pelaksana:** WAJIB memakai sub-skill superpowers:subagent-driven-development (disarankan) atau
> superpowers:executing-plans untuk menjalankan rencana ini per tugas. Langkah memakai checkbox (`- [ ]`).

**Tujuan:** Kode siap-jalan untuk 6 model berbasis interval RR (SVM, XGBoost, MiniRocket, BiGRU + Attention, TCN,
InceptionTime), satu folder per model. Semua memakai data, fold, seed, bobot kelas, dan evaluasi yang sama, sehingga
hasil tiap anggota tim bisa disandingkan dengan v1 dan v3.1.

**Arsitektur:**
- `v11_model_tim/bersama/` berisi pemuat data, loop training DL, runner 25 job yang bisa dilanjutkan, dan evaluasi.
  Semua kode lama dipakai lewat impor (`sihedaf_cv`, `hrv_core`, `sihedaf_models`, `takikardia`, `pascaproses`).
- Tiap folder model berisi `model.py` dengan kontrak yang sama, plus notebook hasil generator.
- Tidak ada training penuh. Yang dijalankan hanya uji paritas dan uji jalan `SIHEDAF_QUICK=1`.

**Teknologi:** Python ≥ 3.10, numpy, PyTorch, scikit-learn, xgboost, aeon (venv terpisah), nbformat/nbclient,
pytest.

**Spesifikasi:** [`v11_model_tim/RANCANGAN.md`](RANCANGAN.md), disetujui pemilik repo 20 Oktober 2026.

> **Revisi pelaksanaan (20 Oktober 2026, keputusan pemilik repo): jangan jalankan WSL, dan jangan jalankan kode.**
> - Semua langkah "jalankan tes", "uji paritas", "uji jalan QUICK", dan "pasang pustaka" dilewati. Kode dan tes tetap
>   ditulis, lalu diperiksa secara statis (review).
> - Tugas 10 Langkah 1 (venv MiniRocket) dibatalkan.
> - `buat_notebook.py` tidak dibuat. Notebook ditulis langsung sebagai JSON: enam notebook model identik kecuali baris
>   `MODEL = "<nama>"`, dan tes `test_notebook.py` memeriksa keidentikannya.
> - `sidik_jari.json` dihitung dengan `Get-FileHash` PowerShell (tidak menjalankan kode proyek).
> - Versi minimum di `requirements.txt`: `xgboost>=2.0`, `pytest>=8.0`, dan `aeon>=1.0` di `requirements-minirocket.txt`.

**Cara menjalankan perintah.** Dari root repo, dengan Python lingkungan proyek. Di bawah, `$PY` berarti interpreter itu:

```bash
PY=python
eval "$PY -m pytest v11_model_tim/tests -q"
```

> Catatan riwayat: rencana ini awalnya ditulis untuk venv WSL di PC pemilik repo. Perintah yang menyebut WSL atau venv
> khusus di bawah sudah **dibatalkan** (lihat kotak revisi di atas). Perintah itu disimpan hanya sebagai riwayat dan
> tidak berlaku untuk anggota tim.

## Batasan global

- **Jangan commit/push dan jangan mengunggah apa pun.** Pemilik repo yang melakukannya. Langkah "commit" di bawah
  sengaja diganti "serahkan".
- Jangan mengubah folder versi lama, `cache/oof_robust/`, `cache/v5/`, `cache/v7/`, atau `*/model_export/`.
- Tidak ada training penuh. Yang boleh hanya uji paritas (Tugas 2) dan uji jalan `SIHEDAF_QUICK=1` (±10–15 menit total).
- Fold: `cache/oof_robust/detector_e30_r5_s2026/folds.npz` (5 ulangan × 5 fold). Seed latih `2026 + 100·r + k`.
- Bobot kelas lunak `cv.class_weights(y) ** 0.5`: model DL di loss, model klasik sebagai `sample_weight`.
- Bias cross-fit `cv.crossfit_bias`, grid `cv.BIAS_GRID` (−7…+3). Bootstrap pasien 2.000×, stratified, seed 2026.
- Bawaan DL = resep v3: 30 epoch, batch 256, lr 1e-3, weight decay 1e-4. Model DL ≤ 100.000 parameter.
- IBI dalam milidetik. Label 0 Normal, 1 AFIB, 2 AFL.
- Tanpa path absolut, nama GPU, atau jumlah core di berkas yang di-commit. Path dicari dari `data.REPO`.
- Tanpa fallback diam-diam: berkas hilang, hash tidak cocok, atau pustaka tidak ada → error dengan pesan jelas.
- Definisi model/praproses lama diimpor, tidak disalin.
- numpy 2.5.3 dan scipy 1.18.1 di venv utama **tidak boleh berubah**. Cek sebelum dan sesudah setiap `pip install`.
- Notebook diserahkan tanpa output. Teks notebook dan README dalam Bahasa Indonesia sederhana.

## Fokus review

Lima kondisi yang tidak diatur spesifikasi tetapi paling mungkin menggigit tim, dengan perilaku yang diharapkan:

1. **Nilai di posisi padding bocor ke model konvolusi.** Mengubah nilai `X` di posisi `M == 0` tidak boleh mengubah
   output TCN, InceptionTime, atau BiGRU. (Tes di Tugas 5, 6, 7.)
2. **Sesi Kaggle mati di tengah fold.** Berkas `.tmp` sisa diabaikan, fold yang sudah selesai dilewati, dan fold yang
   belum selesai diulang. (Tes di Tugas 3.)
3. **Hasil belum lengkap dievaluasi.** Ulangan yang belum lengkap 5 fold dilewati. Perbandingan berpasangan memakai
   ulangan yang sama untuk referensi. Kalau tidak ada ulangan yang lengkap → error. (Tes di Tugas 3 dan 4.)
4. **Folder `xgboost/` menutupi paket `xgboost`.** Dari cwd `v11_model_tim/xgboost`, `import xgboost` harus memuat
   pustaka asli. (Tes di Tugas 8.)
5. **Fitur HRV uji stres dihitung dari `X` float32.** `F` set uji stres harus sama dengan `hrv_features` dari IBI
   float64. (Tes di Tugas 1.)

---

### Tugas 1: Lingkungan + `bersama/data.py` (data, sidik jari, zip, set uji stres)

**Berkas:**
- Ubah: `requirements.txt`. Tambah `xgboost>=<versi terpasang>` di bagian eksperimen, dan bagian baru `# ── tes ──`
  berisi `pytest>=<versi terpasang>`.
- Buat: `v11_model_tim/bersama/__init__.py` berisi `MODEL_TIM = ("svm", "xgboost", "minirocket", "bigru", "tcn",
  "inceptiontime")`.
- Buat: `v11_model_tim/bersama/data.py`, `v11_model_tim/sidik_jari.json` (hasil `buat_sidik_jari()`).
- Buat: `v11_model_tim/tests/conftest.py`. Menyisipkan `v11_model_tim` ke `sys.path` dan menyediakan fixture sesi
  `d = data.muat()`.
- Tes: `v11_model_tim/tests/test_data.py`.

**Antarmuka:**
- Memakai: `sihedaf_cv` (v3), `hrv_core` (v2), `takikardia` (v3.2: `synthetic_ibi`, `JITTERS`, `BPMS`, `N_SYNTH`, `SEED`).
- Menghasilkan untuk tugas lain:
  - `REPO: Path` (folder yang berisi `AGENTS.md`, dicari naik dari berkas ini), `QUICK: bool`
    (`SIHEDAF_QUICK == "1"`), `N_ULANGAN = 5`, `N_FOLD = 5`, `SEED = 2026`.
  - `seed_latih(r: int, k: int) -> int`.
  - `BERKAS: tuple[str, ...]`, path relatif terhadap akar data:
    - `dataset_v2_detector.npz`;
    - `oof_robust/detector_e30_r5_s2026/folds.npz`;
    - `oof_robust/detector_e30_r5_s2026/MicroTransformer_r{0..4}.npz`;
    - `oof_robust/detector_e30_r5_s2026/versi_lama/oof_r{0..4}.npz`;
    - `v7/MicroTransformer_kontrol_r0.npz`;
    - `v7/uji_kontrol_r0.npz`.
  - `lokasi_data() -> Path`: `$SIHEDAF_DATA` jika diset, selain itu `REPO/"cache"`.
  - `lokasi_hasil() -> Path`: `$SIHEDAF_HASIL` jika diset, selain itu `REPO/"cache"/("v11_quick" if QUICK else "v11")`.
  - `buat_sidik_jari() -> dict`: menulis `v11_model_tim/sidik_jari.json`
    (`{"berkas": {rel: sha256}, "ukuran": {rel: byte}}`).
  - `periksa_sidik_jari(akar: Path | None = None) -> None`: `RuntimeError` yang menyebut semua berkas hilang/berbeda.
  - `muat() -> SimpleNamespace` dengan field:
    - `X (N,30,2) f32`, `M (N,30) f32`, `F (N,16) f32`, `y (N,) int64`, `records (N,) str`, `t (N,) f64`;
    - `patients`, `strata` (`cv.patient_strata`), `folds (5,N) int8`;
    - `hr (N,) f64` = 60000 / rata-rata IBI bermask;
    - `bounds` (sama dengan `pascaproses.load`);
    - `oof_v31 (5,N,3)`, `oof_v1 (5,N,3)` (kunci `det`);
    - `kontrol_v7 = SimpleNamespace(oof (N,3), stres (5,S,3))`.
    Memanggil `periksa_sidik_jari()` dulu. Setiap array npz dibaca **sekali** (mengindeks `NpzFile` di loop membaca
    ulang seluruh array).
  - `set_uji_stres() -> SimpleNamespace(ibi: list[np.ndarray], X, M, F, jenis: np.ndarray[str], bpm: np.ndarray[int])`.
    Urutannya: jenis variasi → bpm → window, dengan seed `T.SEED + 1000·j + bpm`. `F` dari `hrv_core.hrv_features(ibi)`
    float64. S = 4 × 23 × 50 = 4.600.
  - `bobot_sampel(y: np.ndarray) -> np.ndarray` = `(cv.class_weights(y) ** 0.5)[y]`.
  - `buat_zip(tujuan: Path) -> Path`: zip berisi `BERKAS` dengan path relatif yang sama + `sidik_jari.json`.
  - `REPO_UNDUHAN: Path = REPO/"cache"/"v11_unduhan"`.
  - `siapkan_data(url: str) -> Path`:
    - jika `periksa_sidik_jari(lokasi_data())` lolos, kembalikan path itu;
    - jika tidak, unduh `url` (`urllib`, mendukung `file://` untuk tes) ke `REPO_UNDUHAN`, ekstrak, set
      `os.environ["SIHEDAF_DATA"]`, lalu periksa;
    - unduhan gagal → `RuntimeError` dengan langkah manual (dataset Kaggle + `SIHEDAF_DATA`).

- [ ] **Langkah 1: Pasang pustaka dan pastikan numpy/scipy tidak berubah**

```bash
eval "$PY -c 'import numpy,scipy;print(numpy.__version__,scipy.__version__)'" < /dev/null   # harus 2.5.3 1.18.1
eval "$PY -m pip install pytest xgboost" < /dev/null
eval "$PY -c 'import numpy,scipy,xgboost,pytest;print(numpy.__version__,scipy.__version__,xgboost.__version__,pytest.__version__)'" < /dev/null
```
Harapan: numpy 2.5.3 dan scipy 1.18.1 tetap sama. Jika pip mengubahnya, hentikan pekerjaan dan laporkan ke pemilik repo.
Catat versi xgboost/pytest untuk `requirements.txt`.

- [ ] **Langkah 2: Tulis tes yang gagal** (`tests/test_data.py`)

```python
def test_repo_dan_bentuk(d):
    assert (data.REPO / "AGENTS.md").exists()
    assert d.X.shape == (165978, 30, 2) and d.F.shape == (165978, 16) and d.folds.shape == (5, 165978)
    assert d.oof_v31.shape == d.oof_v1.shape == (5, 165978, 3)
    assert d.kontrol_v7.stres.shape == (5, 4600, 3)

def test_seed_latih():
    assert data.seed_latih(0, 0) == 2026 and data.seed_latih(4, 3) == 2429

def test_sidik_jari_rusak_error(tmp_path, monkeypatch):
    # salin semua BERKAS ke tmp_path, ubah satu byte dataset → periksa_sidik_jari(tmp_path) RuntimeError
    # yang pesannya memuat "dataset_v2_detector.npz"
    ...

def test_zip_bolak_balik(tmp_path):
    z = data.buat_zip(tmp_path / "v11_data.zip"); zipfile.ZipFile(z).extractall(tmp_path / "x")
    data.periksa_sidik_jari(tmp_path / "x")          # tidak error

def test_siapkan_data_dari_url(tmp_path, monkeypatch):   # jalur Kaggle/Colab, URL file:// menggantikan release
    z = data.buat_zip(tmp_path / "v11_data.zip"); monkeypatch.setenv("SIHEDAF_DATA", str(tmp_path / "kosong"))
    monkeypatch.setattr(data, "REPO_UNDUHAN", tmp_path / "unduh")   # konstanta tujuan unduhan, bawaan REPO/cache/v11_unduhan
    akar = data.siapkan_data(z.as_uri())
    assert akar == tmp_path / "unduh" and os.environ["SIHEDAF_DATA"] == str(akar)

def test_siapkan_data_url_gagal(tmp_path, monkeypatch):
    monkeypatch.setenv("SIHEDAF_DATA", str(tmp_path / "kosong")); monkeypatch.setattr(data, "REPO_UNDUHAN", tmp_path / "u")
    with pytest.raises(RuntimeError, match="SIHEDAF_DATA"):
        data.siapkan_data((tmp_path / "tidak_ada.zip").as_uri())

def test_uji_stres_fitur_dari_ibi_float64():
    u = data.set_uji_stres()
    assert len(u.ibi) == 4600 and u.X.shape == (4600, 30, 2)
    assert np.array_equal(u.F[7], hrv_core.hrv_features(u.ibi[7]))
    assert u.bpm[0] == 60 and u.bpm[-1] == 170 and u.jenis[0] == "tanpa variasi"

def test_uji_stres_sama_dengan_v7():
    import augmentasi                                  # v7, lewat sys.path
    Xv7, Mv7, _ = augmentasi.stress_set(); u = data.set_uji_stres()
    assert np.array_equal(u.X, Xv7) and np.array_equal(u.M, Mv7)
```

- [ ] **Langkah 3: Jalankan, pastikan gagal** dengan `eval "$PY -m pytest v11_model_tim/tests/test_data.py -q" < /dev/null`.
  Harapan: FAIL, `ModuleNotFoundError: bersama`.

- [ ] **Langkah 4: Implementasikan `bersama/data.py`** sesuai antarmuka, lalu jalankan `buat_sidik_jari()` sekali untuk
  menulis `sidik_jari.json`.

- [ ] **Langkah 5: Jalankan tes, pastikan lolos.** Harapan: semua PASS.

- [ ] **Langkah 6: Serahkan.** Jangan commit. Catat berkas yang berubah untuk ringkasan akhir.

---

### Tugas 2: `bersama/latih_dl.py` (loop training DL umum + uji paritas)

**Berkas:** Buat `v11_model_tim/bersama/latih_dl.py`. Tes: `v11_model_tim/tests/test_latih_dl.py`.

**Antarmuka:**
- Memakai: `cv.set_seed`, `cv.zscore_fit`, `cv.zscore_apply`, `cv.class_weights`, `cv.predict_dl`. Data latih
  berupa `SimpleNamespace(X, M, F, y)` yang diberikan runner (Tugas 3).
- Menghasilkan:
  - `latih_dl(buat_net: Callable[[], nn.Module], X, M, y, *, seed: int, device, epochs=30, batch_size=256, lr=1e-3,
    weight_decay=1e-4, weight_power=0.5) -> tuple[nn.Module, np.ndarray, np.ndarray, float]`.
    Urutan operasinya **sama persis** dengan `sihedaf_cv.train_dl`: `set_seed` → z-score → tensor → criterion →
    `buat_net()` → AdamW → cosine → generator → loop. Satu-satunya beda: bobot loss `class_weights(y) ** weight_power`.
  - `PaketDL` (dataclass): `net`, `mu`, `sd`, `setelan: dict`.
  - `latih_paket_dl(buat_net, data, setelan: dict, seed: int, device) -> PaketDL`. Memakai kunci setelan `epochs`,
    `batch_size`, `lr`, `weight_decay` (bawaan = resep v3), dan `weight_power=0.5`.
  - `prediksi_paket_dl(paket: PaketDL, X, M, F, device) -> np.ndarray` (n,3), lewat `cv.predict_dl`.
  - `simpan_paket_dl(paket: PaketDL, path: Path) -> Path`: menulis `path.with_suffix(".pt")` berisi `state_dict` (CPU),
    `mu`, `sd`, `setelan`.
  - `jumlah_parameter(net) -> int`.

- [ ] **Langkah 1: Tulis tes paritas yang gagal**

```python
def test_paritas_dengan_v3(d):   # pola notebook v5 sel 4: 2 epoch, latih = window idx % 5 != 0, uji 5.000 window
    tr = np.arange(len(d.y)) % 5 != 0; te = np.flatnonzero(~tr)[:5000]
    kw = dict(input_dim=2, d_model=32, nhead=2, dim_ff=64, num_classes=3, dropout=0.1)
    a = latih_dl.latih_dl(lambda: build_model("MicroTransformer", **kw), d.X[tr], d.M[tr], d.y[tr],
                          seed=7, device=dev, epochs=2, weight_power=1.0)
    b = cv.train_dl("MicroTransformer", kw, d.X[tr], d.M[tr], d.y[tr], epochs=2, batch_size=256, lr=1e-3,
                    weight_decay=1e-4, seed=7, device=dev)
    pa = cv.predict_dl(a[0], d.X[te], d.M[te], a[1], a[2], dev); pb = cv.predict_dl(b[0], d.X[te], d.M[te], b[1], b[2], dev)
    assert np.abs(pa - pb).max() < 1e-4

def test_simpan_muat_pt(tmp_path, d): ...   # simpan_paket_dl → .pt ada; state_dict dimuat ke net baru → prediksi sama persis
```

- [ ] **Langkah 2: Jalankan, pastikan gagal.** Harapan: FAIL (modul belum ada).
- [ ] **Langkah 3: Implementasikan `latih_dl.py`.**
- [ ] **Langkah 4: Jalankan, pastikan lolos.** Harapan: PASS, selisih maks < 1e-4. Tes ini ±1 menit di GPU.
- [ ] **Langkah 5: Serahkan** (tanpa commit).

---

### Tugas 3: `bersama/runner.py` (25 job, bisa dilanjutkan, penjaga setelan, mode QUICK)

**Berkas:** Buat `v11_model_tim/bersama/runner.py`. Tes: `v11_model_tim/tests/test_runner.py`, memakai modul model
palsu yang cepat (tanpa training sungguhan).

**Antarmuka:**
- Kontrak `model.py` (didokumentasikan di docstring `runner.py`):
  - `NAMA: str`, `SETELAN: dict`, `SETELAN_QUICK: dict`;
  - `latih(data, setelan, seed, device) -> object`;
  - `prediksi(model, X, M, F, device) -> np.ndarray` (n,3), baris berjumlah 1;
  - `simpan(model, path: Path) -> Path` (path tanpa ekstensi);
  - `info(model) -> dict`.
- Memakai: `data.lokasi_hasil`, `data.muat`, `data.set_uji_stres`, `data.seed_latih`, `data.QUICK`.
- Menghasilkan:
  - `jalankan(mdl, nama_setelan: str, setelan: dict, *, ulangan=None, log=print) -> Path`. Folder hasil:
    `lokasi_hasil()/mdl.NAMA/nama_setelan`.
    - Setelan efektif = `{**setelan, **mdl.SETELAN_QUICK}` bila QUICK.
    - `setelan.json` (`{"model", "setelan", "quick"}`) ditulis saat pertama kali. Isinya berbeda saat dijalankan ulang
      → `ValueError` yang menyuruh mengganti `NAMA_SETELAN`.
    - Ulangan bawaan `range(5)`; saat QUICK `[0]`.
    - Per (r, k): bila `r{r}_k{k}.npz` ada, lewati. Bila tidak: latih di window `folds[r] != k` (QUICK: `subset_quick`),
      prediksi fold uji dan set uji stres, simpan bobot `r{r}_k{k}.*`, **lalu** tulis npz secara atomik (`.tmp` →
      `replace`). Isi npz: `oof (n_te,3) f32`, `idx (n_te,) int64`, `stres (S,3) f32`, `waktu f8`, `info` (str JSON).
  - `subset_quick(idx, y, seed, n=3000, min_afl=300) -> np.ndarray`.
  - `muat_hasil(folder: Path) -> SimpleNamespace(oof (R,N,3), stres (R,5,S,3), waktu (R,5), ulangan: list[int],
    setelan: dict, quick: bool, info: list)`. Hanya ulangan yang lengkap 5 fold; tidak ada satu pun → `RuntimeError`.
  - `zip_hasil(folder: Path) -> Path`.

- [ ] **Langkah 1: Tulis tes yang gagal**

```python
def test_lanjut_setelah_putus(tmp_path, monkeypatch, d):
    monkeypatch.setenv("SIHEDAF_HASIL", str(tmp_path))
    f = runner.jalankan(palsu, "a", {"p": 1}, ulangan=[0])          # 5 fold
    (f / "r0_k3.npz").unlink(); (f / "r0_k1.npz.tmp").write_bytes(b"rusak")
    palsu.panggilan.clear(); runner.jalankan(palsu, "a", {"p": 1}, ulangan=[0])
    assert palsu.panggilan == [(0, 3)]                                # hanya fold yang hilang dilatih ulang

def test_setelan_berubah_error(tmp_path, monkeypatch):
    ...; runner.jalankan(palsu, "a", {"p": 1}, ulangan=[0])
    with pytest.raises(ValueError, match="NAMA_SETELAN"):
        runner.jalankan(palsu, "a", {"p": 2}, ulangan=[0])

def test_hasil_tidak_lengkap(tmp_path, monkeypatch):
    # ulangan 0 lengkap, ulangan 1 hanya fold 0 → muat_hasil().ulangan == [0] dan oof.shape == (1, N, 3)
    # folder kosong → RuntimeError

def test_oof_menutup_semua_window(...):   # oof ulangan lengkap tanpa NaN, setiap baris tepat dari fold ujinya

def test_subset_quick():
    s = runner.subset_quick(np.arange(len(d.y)), d.y, 1)
    assert len(s) == 3000 and (d.y[s] == 2).sum() >= 300
```

- [ ] **Langkah 2: Jalankan, pastikan gagal.**
- [ ] **Langkah 3: Implementasikan `runner.py`.**
- [ ] **Langkah 4: Jalankan, pastikan lolos.**
- [ ] **Langkah 5: Serahkan** (tanpa commit).

---

### Tugas 4: `bersama/evaluasi.py` (metrik, berpasangan, takikardia, seleksi cross-fit)

**Berkas:** Buat `v11_model_tim/bersama/evaluasi.py`. Tes: `v11_model_tim/tests/test_evaluasi.py`, memakai OOF v3
yang sudah ada sebagai "model" (tanpa training).

**Antarmuka:**
- Memakai: `cv.crossfit_bias`, `cv.apply_bias`, `cv.patient_cms`, `cv.boot_metrics`, `cv.point_metrics`,
  `cv.bootstrap_weights`, `cv.METRICS`, `T.normal_threshold`, `T.by_hr_bin`, `T.metrics_with_ci`,
  `T.candidate_cms`, `T.nested_bootstrap`, `pascaproses.crossfit_select`; `runner.muat_hasil`; `data.*`.
- Menghasilkan:
  - Konstanta `AMBANG_V31_PAKET = 115` (sumber: v3.2) dan `BATAS_GRID = (-7.0, 3.0)`.
  - `keputusan(oof (R,N,3), folds (R,N), y) -> tuple[pred (R,N) int, bias (R,5,3)]`.
  - `tabel_metrik(d, pred: dict[str, (R,N)]) -> dict[str, dict[str, [titik, lo, hi]]]`.
  - `banding(d, pred_kandidat (R,N), pred_ref (R,N)) -> dict[str, [Δ, lo, hi, P]]`, untuk Macro-F1, AFL F1, AFL P,
    AFL R, Normal R. Memakai bobot bootstrap yang sama untuk keduanya.
  - `afl_per_pasien(d, pred) -> dict[str, float]` (7 pasien) dan `afl_palsu(d, pred) -> {"total", "dari_normal",
    "dari_afib", "pasien_tanpa_afl"}` (rata-rata per ulangan).
  - `bias_di_batas(bias (R,5,3)) -> float`: proporsi bias AFIB/AFL yang bernilai −7 atau +3.
  - `ambang_takikardia(stres (R,5,S,3), bias (R,5,3), uji) -> {"proporsi": {jenis: (23,3)}, "ambang": {jenis: bpm|None}}`.
  - `laporan_model(folder: Path, d) -> dict`:
    - Isi: `model`, `setelan`, `quick`, `ulangan`;
      `metrik` (v1, v31, `argmax`, `+bias`);
      `banding_v31`, `banding_v1`, `afl_per_pasien`, `afl_palsu`;
      `bias` (median/min/maks/di_batas);
      `takikardia` (model, `kontrol_v7`, `paket_v31: 115`);
      `pita_detak` (`T.by_hr_bin`), `waktu_per_fold`, `info`;
      `lingkungan` (`commit` dan `ada_perubahan` dari `git`, atau `"tidak diketahui"` bila git tidak ada; `gpu`;
      `pustaka` berisi versi python/numpy/scipy/torch/sklearn/xgboost/aeon yang terpasang).
    - Pembanding v1 dan v3.1 dipotong ke ulangan yang sama dengan kandidat.
    - Hasilnya ditulis ke `folder/"ringkasan.json"`.
  - `pilih_crossfit(d, folders: list[Path]) -> dict`: per ulangan bersama, `crossfit_select` atas OOF tiap setelan →
    `pred`, `terpilih` per fold, `metrik`, `banding_v31`.
  - `tanpa_04936(d, oofs: dict[str, (R,N,3)]) -> dict`. Bias dihitung ulang di subset; pola v7. Kunci wajib `"v31"`.
  - `bootstrap_bersarang(d, oof (5,N,3)) -> dict`: kandidat vs v1 (`T.candidate_cms` + `T.nested_bootstrap`).
    Butuh 5 ulangan lengkap; kalau tidak → `ValueError`.

- [ ] **Langkah 1: Tulis tes yang gagal**

```python
def test_v31_dari_oof_cocok_dengan_v3(d):       # MicroTransformer OOF sebagai kandidat → Macro-F1 +bias 0,832 (ringkasan v3)
    pred, _ = evaluasi.keputusan(d.oof_v31, d.folds, d.y)
    assert round(evaluasi.tabel_metrik(d, {"k": pred})["k"]["Macro-F1"][0], 3) == 0.832

def test_banding_diri_sendiri(d):                # kandidat = referensi → Δ = 0 dan P = 0
    pred, _ = evaluasi.keputusan(d.oof_v31, d.folds, d.y)
    assert evaluasi.banding(d, pred, pred)["Macro-F1"][0] == 0 and evaluasi.banding(d, pred, pred)["Macro-F1"][3] == 0

def test_ambang_kontrol_v7(d):                   # angka v7: ambang kontrol 115–120 bpm
    _, b = evaluasi.keputusan(d.kontrol_v7.oof[None], d.folds[:1], d.y)
    a = evaluasi.ambang_takikardia(d.kontrol_v7.stres[None], b, data.set_uji_stres())
    assert a["ambang"]["tanpa variasi"] in (115, 120)

def test_bias_di_batas():
    b = np.zeros((1, 5, 3)); b[0, 0, 2] = -7.0
    assert evaluasi.bias_di_batas(b) == 0.1          # 1 dari 10 nilai AFIB/AFL

def test_laporan_model_ulangan_terpotong(tmp_path, d):
    # buat folder hasil palsu dari OOF BiGRUAttention_r0 + stres kontrol v7 (1 ulangan) →
    # laporan_model: ulangan == [0], ringkasan.json ada, banding_v31 memakai v31 ulangan 0 saja,
    # dan r["lingkungan"] punya kunci "commit", "ada_perubahan", "gpu", "pustaka"

def test_bersarang_butuh_5_ulangan(d):
    with pytest.raises(ValueError): evaluasi.bootstrap_bersarang(d, d.oof_v31[:1])
```

- [ ] **Langkah 2: Jalankan, pastikan gagal.**
- [ ] **Langkah 3: Implementasikan `evaluasi.py`.**
- [ ] **Langkah 4: Jalankan, pastikan lolos.** Harapan: PASS. Jika 0,832 tidak cocok, cari bedanya dengan
  `pascaproses.evaluate`; jangan mengubah angka harapannya.
- [ ] **Langkah 5: Serahkan** (tanpa commit).

---

### Tugas 5: Generator notebook + folder `bigru/` (milik pemilik repo) + uji jalan ujung ke ujung

**Berkas:**
- Buat: `v11_model_tim/bersama/buat_notebook.py`, `v11_model_tim/bigru/model.py`, `v11_model_tim/bigru/README.md`.
- Hasil generator: `v11_model_tim/bigru/jalankan.ipynb`.
- Tes: `v11_model_tim/tests/test_model_bigru.py`, `v11_model_tim/tests/test_notebook.py`.

**Antarmuka:**
- `bigru/model.py`:
  - `NAMA = "bigru"`;
  - `SETELAN = {"hidden": 32, "epochs": 30, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4, "weight_power": 0.5}`;
  - `SETELAN_QUICK = {"epochs": 1}`;
  - `buat_net(setelan) -> nn.Module` = `build_model("BiGRUAttention", input_dim=2, hidden=..., num_classes=3)`, diimpor
    dari `v2_revisi/sihedaf_models`;
  - `latih`/`prediksi`/`simpan`/`info` membungkus `latih_dl.*_paket_dl`.
- `buat_notebook.py`:
  - `buat_notebook_model(nama: str) -> Path` dan `buat_gabungan() -> Path`;
  - `buat_semua() -> list[Path]` untuk semua folder di `MODEL_TIM` yang sudah punya `model.py`.
- Isi sel notebook model, berurutan:
  1. **Markdown:** judul, nama model, cara pakai singkat.
  2. **Setup:**
     - `REPO_URL = "https://github.com/farhanlhsn/WrapSihedaf.git"`, `BRANCH = "main"`;
     - `DATA_URL = "https://github.com/farhanlhsn/WrapSihedaf/releases/download/v11-data/v11_data.zip"`;
     - `NAMA_SETELAN = "bawaan"`, `SETELAN = {}`;
     - jika `model.py` tidak ada di cwd: clone `--depth 1 -b BRANCH` (pola notebook v8) ke `/kaggle/working/WrapSihedaf`
       atau `./WrapSihedaf`, lalu `chdir` ke `v11_model_tim/<model>`;
     - `sys.path` diisi folder induk dan cwd;
     - di Kaggle/Colab saja: `pip install -r` berkas `requirements-<model>.txt` bila ada;
     - `data.siapkan_data(DATA_URL)`;
     - cetak commit, GPU, dan QUICK.
  3. **Muat:** `import model as MDL`, `d = data.muat()`.
  4. **Latih:** `folder = runner.jalankan(MDL, NAMA_SETELAN, {**MDL.SETELAN, **SETELAN})`.
  5. **Evaluasi:** `r = evaluasi.laporan_model(folder, d)`. Tabel pandas: metrik vs v1/v3.1 dengan vonis,
     recall AFL per pasien, AFL palsu, ambang takikardia vs `kontrol_v7` dan 115, pita detak, peringatan bias di batas.
  6. **Kesimpulan** (markdown): diisi anggota setelah hasil ada.
  7. **Berbagi:** `runner.zip_hasil(folder)`.

- [ ] **Langkah 1: Tulis tes yang gagal**

```python
def test_bigru_bentuk_dan_mask():
    net = M.buat_net(M.SETELAN); x = torch.randn(4, 30, 2); m = torch.zeros(4, 30); m[:, :13] = 1
    y1 = net(x * m[..., None], m); x2 = x.clone(); x2[:, 13:] = 99.0
    assert y1.shape == (4, 3) and torch.allclose(y1, net(x2, m))     # nilai padding tidak bocor

def test_notebook_valid():
    p = buat_notebook.buat_notebook_model("bigru"); nb = nbformat.read(p, 4); nbformat.validate(nb)
    src = "".join(c.source for c in nb.cells)
    assert 'NAMA_SETELAN = "bawaan"' in src and "siapkan_data(DATA_URL)" in src
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
```

- [ ] **Langkah 2: Jalankan, pastikan gagal.**
- [ ] **Langkah 3: Implementasikan `bigru/model.py` dan `buat_notebook.py`**, lalu buat notebook-nya.
- [ ] **Langkah 4: Jalankan tes, pastikan lolos.**
- [ ] **Langkah 5: Uji jalan ujung ke ujung**

```bash
SIHEDAF_QUICK=1 python tools/run_notebook.py v11_model_tim/bigru/jalankan.ipynb
```
Harapan: `logs/jalankan.status.txt` tidak FAILED, `cache/v11_quick/bigru/bawaan/ringkasan.json` ada dengan
`"quick": true`, dan semua tabel tercetak. Lalu jalankan `buat_notebook.buat_notebook_model("bigru")` lagi supaya
notebook kembali tanpa output, dan hapus `cache/v11_quick/`.

- [ ] **Langkah 6: Tulis `bigru/README.md`:**
  - modelnya (Bi-GRU + attention, 7.172 parameter, diimpor dari v2);
  - setelan bawaan;
  - perbedaannya dengan hasil v3 (bobot penuh, tanpa bobot tersimpan);
  - perkiraan waktu (±75 menit GPU untuk 25 fold);
  - ide eksperimen (`hidden`, epoch, tuning cross-fit).
- [ ] **Langkah 7: Serahkan** (tanpa commit).

---

### Tugas 6: Folder `tcn/`

**Berkas:**
- Buat: `v11_model_tim/tcn/model.py`, `v11_model_tim/tcn/README.md`.
- Hasil generator: `tcn/jalankan.ipynb`.
- Tes: `v11_model_tim/tests/test_model_tcn.py`.

**Antarmuka:**
- `NAMA = "tcn"`.
- `SETELAN = {"kanal": 32, "kernel": 3, "dilasi": (1, 2, 4, 8), "dropout": 0.1, "epochs": 30, "batch_size": 256,
  "lr": 1e-3, "weight_decay": 1e-4, "weight_power": 0.5}`; `SETELAN_QUICK = {"epochs": 1}`.
- `class TCN(nn.Module)`, `forward(x (B,30,2), mask (B,30)) -> (B,3)`:
  - blok residual konvolusi dilatasi non-kausal (padding "same");
  - input dan aktivasi setiap blok dikalikan mask;
  - global average pooling bermask → linear.
- `buat_net(setelan) -> TCN`. Pembungkus `latih`/`prediksi`/`simpan`/`info` sama dengan bigru.

- [ ] **Langkah 1: Tulis tes yang gagal:** bentuk (B,3); invarian padding (pola tes bigru, nilai 99 di posisi mask 0);
  `latih_dl.jumlah_parameter(net) <= 100_000`; satu urutan sepanjang 4 detak (panjang minimum, `hrv_core.MIN_IBI`)
  tidak menghasilkan NaN.
- [ ] **Langkah 2: Jalankan, pastikan gagal.**
- [ ] **Langkah 3: Implementasikan `TCN`**, lalu jalankan `buat_notebook.buat_notebook_model("tcn")`.
- [ ] **Langkah 4: Jalankan tes, pastikan lolos.**
- [ ] **Langkah 5: Uji jalan QUICK** dengan perintah Tugas 5 untuk `v11_model_tim/tcn/jalankan.ipynb`. Harapan: tidak
  FAILED. Setelah itu buat ulang notebook tanpa output dan hapus `cache/v11_quick/`.
- [ ] **Langkah 6: Tulis `tcn/README.md`** (isi seperti bigru), lalu **serahkan** (tanpa commit).

---

### Tugas 7: Folder `inceptiontime/`

**Berkas:**
- Buat: `v11_model_tim/inceptiontime/model.py`, `README.md`.
- Hasil generator: notebook.
- Tes: `tests/test_model_inceptiontime.py`.

**Antarmuka:**
- `NAMA = "inceptiontime"`.
- `SETELAN = {"filter": 16, "kernel": (3, 5, 9), "kedalaman": 6, "dropout": 0.0, "epochs": 30, "batch_size": 256,
  "lr": 1e-3, "weight_decay": 1e-4, "weight_power": 0.5}`; `SETELAN_QUICK = {"epochs": 1}`.
- `class InceptionTime(nn.Module)`, `forward(x, mask) -> (B,3)`:
  - modul Inception: bottleneck 1×1 → tiga konvolusi (kernel 3, 5, 9), ditambah cabang max-pool → konvolusi 1×1,
    lalu concat 4×`filter`, BatchNorm, ReLU;
  - shortcut residual setiap 3 modul;
  - masking dan pooling bermask seperti TCN;
  - **satu jaringan**, bukan ensemble 5.

- [ ] **Langkah 1–6:** sama dengan Tugas 6 (tes bentuk, invarian padding, ≤ 100.000 parameter, 4 detak tanpa NaN →
  implementasi → notebook → uji jalan QUICK → README → serahkan). Catatan untuk README: max-pool di dekat posisi padding
  harus memakai nilai yang sudah dimask. Tes invarian padding akan menangkapnya.

---

### Tugas 8: Folder `xgboost/`

**Berkas:**
- Buat: `v11_model_tim/xgboost/model.py`, `README.md`.
- Hasil generator: notebook.
- Tes: `tests/test_model_xgboost.py`.

**Antarmuka:**
- `NAMA = "xgboost"`.
- `SETELAN = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "tree_method": "hist", "perangkat": "cpu"}`;
  `SETELAN_QUICK = {"n_estimators": 20}`.
- `latih`: `XGBClassifier(objective="multi:softprob", random_state=seed, n_jobs=-1, device=setelan["perangkat"], ...)`
  dilatih pada `data.F` dengan `sample_weight=data.bobot_sampel(y)`.
- `prediksi`: `predict_proba(F)`.
- `simpan`: `save_model(path.with_suffix(".json"))`.
- `info`: `{"pohon": n}`.

- [ ] **Langkah 1: Tulis tes yang gagal**

```python
def test_import_xgboost_asli_dari_folder_model(monkeypatch):
    monkeypatch.chdir(data.REPO / "v11_model_tim/xgboost"); monkeypatch.syspath_prepend(".")
    monkeypatch.syspath_prepend(str(data.REPO / "v11_model_tim"))
    import importlib, sys; sys.modules.pop("xgboost", None)
    assert hasattr(importlib.import_module("xgboost"), "XGBClassifier")

def test_bobot_sampel_dipakai(d): ...   # latih 500 window kecil; booster param/fit menerima sample_weight (cek lewat spy)
def test_proba(d): ...                    # prediksi (n,3), baris berjumlah 1 (toleransi 1e-6), deterministik untuk seed sama
```

- [ ] **Langkah 2–6:** jalankan, pastikan gagal → implementasi → lolos → uji jalan QUICK notebook → README (16 fitur HRV,
  bobot sampel lunak, GPU opsional lewat `perangkat`) → serahkan (tanpa commit).

---

### Tugas 9: Folder `svm/`

**Berkas:**
- Buat: `v11_model_tim/svm/model.py`, `README.md`.
- Hasil generator: notebook.
- Tes: `tests/test_model_svm.py`.

**Antarmuka:**
- `NAMA = "svm"`.
- `SETELAN = {"komponen": 2000, "gamma": None, "C": 1.0, "max_iter": 5000}`; `SETELAN_QUICK = {"komponen": 200}`.
- `latih` → `Pipeline(StandardScaler(), Nystroem(kernel="rbf", gamma=..., n_components=..., random_state=seed),
  LinearSVC(C=..., max_iter=..., random_state=seed))`. Dilatih pada `F` dengan `sample_weight` diteruskan ke langkah
  `linearsvc`. `gamma=None` berarti bawaan Nystroem (1/jumlah fitur).
- `prediksi` → softmax dari `decision_function` (pseudo-probabilitas; bias cross-fit menyesuaikan skalanya).
- `simpan` → `joblib.dump(path.with_suffix(".joblib"))`.

- [ ] **Langkah 1: Tulis tes yang gagal:** proba (n,3) dengan baris berjumlah 1; deterministik untuk seed sama;
  `ConvergenceWarning` tidak dibungkam diam-diam (dicatat ke `info`).
- [ ] **Langkah 2–6:** sama dengan Tugas 8. README menjelaskan kenapa memakai aproksimasi RBF (SVM RBF eksak tidak
  sanggup ±130 ribu window × 25 fold) dan bahwa outputnya pseudo-probabilitas.

---

### Tugas 10: Folder `minirocket/` + venv terpisah

**Berkas:**
- Buat: `v11_model_tim/minirocket/model.py`, `README.md`, `requirements-minirocket.txt`
  (`aeon>=1.6,<1.7`, `numpy<2.5`, `scipy<1.18`).
- Hasil generator: notebook.
- Tes: `tests/test_model_minirocket.py` (di-skip dengan pesan jelas bila `aeon` tidak terpasang).

**Antarmuka:**
- `NAMA = "minirocket"`.
- `SETELAN = {"kernel": 2016, "C": 1.0, "max_iter": 1000}`; `SETELAN_QUICK = {"kernel": 84}`.
- `latih`:
  - input `zscore_apply(X, M, mu, sd)` (mu/sd dari data latih), diubah ke bentuk aeon `(n, 2, 30)`;
  - aeon `MiniRocket(n_kernels=..., random_state=seed)` (multivariat);
  - `StandardScaler` → `LogisticRegression(C, max_iter)` dengan `sample_weight`.
- `prediksi` → `predict_proba`, ditransformasi per batch 8.192 agar hemat RAM.
- `simpan` → `joblib` (berisi mu, sd, transformer, scaler, classifier).

- [ ] **Langkah 1 (DIBATALKAN 20 Oktober 2026, WSL dihapus):** venv MiniRocket khusus di PC pemilik repo. Penggantinya:
  `pip install -r v11_model_tim/minirocket/requirements-minirocket.txt` di lingkungan yang menjalankan MiniRocket
  (otomatis di Kaggle/Colab), atau venv terpisah bila aeon akan menurunkan numpy/scipy lingkungan utama.

- [ ] **Langkah 2: Tulis tes yang gagal:** proba (n,3) dengan baris berjumlah 1; fitur dari mask yang sama
  deterministik; `prediksi` pada 20.000 window tidak membuat matriks fitur penuh sekaligus (batch). Jalankan tes dengan
  python venv MiniRocket.
- [ ] **Langkah 3: Implementasikan, lalu jalankan tes sampai lolos.**
- [ ] **Langkah 4: Uji jalan QUICK** dengan `env SIHEDAF_QUICK=1 SIHEDAF_KERNEL=sihedaf-minirocket ... tools/run_notebook.py
  v11_model_tim/minirocket/jalankan.ipynb`. Setelah itu buat ulang notebook tanpa output dan hapus `cache/v11_quick/`.
- [ ] **Langkah 5: Tulis `minirocket/README.md`:**
  - jumlah kernel dikurangi karena RAM;
  - di Kaggle/Colab aeon terpasang otomatis dari `requirements-minirocket.txt`;
  - di PC pakai kernel `sihedaf-minirocket` (`SIHEDAF_KERNEL`) dan cara membuat venv-nya, tanpa path absolut.

  Lalu **serahkan** (tanpa commit).

---

### Tugas 11: `gabungan.ipynb`, dokumentasi tim, AGENTS.md, zip data, uji jalan akhir

**Berkas:**
- Hasil generator: `v11_model_tim/gabungan.ipynb`.
- Buat: `v11_model_tim/README.md`.
- Ubah: `AGENTS.md` (Intent Layer + nomor versi berikutnya v12).
- Tes: `tests/test_notebook.py` (ditambah).

**Antarmuka:**
- Sel `gabungan.ipynb`:
  1. Setup, sama dengan notebook model, tanpa `NAMA_SETELAN`.
  2. Pindai `lokasi_hasil()/*/*/ringkasan.json` yang `quick` = false.
  3. Tabel per baris (model, setelan): Macro-F1 +bias [CI], P vs v3.1, P vs v1, AFL F1, AFL R, Normal R, ambang
     takikardia, bias di batas, waktu per fold. Ditambah v1 dan v3.1, diurutkan menurut Macro-F1.
  4. Opsional (flag `False` bawaan): `pilih_crossfit` per model, `tanpa_04936`, `bootstrap_bersarang`.
- Isi `v11_model_tim/README.md`:
  - tujuan dan alasan tanpa balancing;
  - pembagian model (BiGRU = pemilik repo, 5 lainnya = teman);
  - cara jalan di PC/Kaggle/Colab dan cara eksperimen (`NAMA_SETELAN`/`SETELAN`, fork/branch untuk arsitektur);
  - cara berbagi hasil (`zip_hasil`) dan atribusi AFDB (ODC-BY);
  - perintah untuk pemilik repo:
    ```bash
    gh release create v11-data cache/v11_data.zip --title "Data v11" --notes "Data uji SIHEDAF v11 (turunan AFDB, ODC-BY)"
    ```
  - tabel hasil yang dikosongkan (diisi setelah hasil ada).
- `AGENTS.md`: entri Intent Layer v11, berisi:
  - satu folder per model, cwd = folder model, path dari `bersama/data.REPO`;
  - pengecualian invariant "arsitektur RR-only tidak dicoba lagi": permintaan pembimbing, studi perbandingan antar model;
  - aeon di venv terpisah.

  Ubah juga "sekarang v11" → "sekarang v12". README root **tidak** diubah sekarang.

- [ ] **Langkah 1: Tulis tes yang gagal:** `buat_gabungan()` valid dan tanpa output; pemindaian mengabaikan ringkasan
  dengan `quick` = true (folder palsu di tmp lewat `SIHEDAF_HASIL`).
- [ ] **Langkah 2: Implementasikan `buat_gabungan`, lalu jalankan tes sampai lolos.**
- [ ] **Langkah 3: Buat zip data dan uji jalur Kaggle/Colab secara lokal**

```bash
eval "$PY -c 'import sys;sys.path.insert(0,\"v11_model_tim\");from bersama import data;print(data.buat_zip(data.REPO/\"cache/v11_data.zip\"))'" < /dev/null
```
Lalu, dengan `SIHEDAF_DATA` menunjuk ke hasil ekstrak zip di scratchpad, jalankan `bigru/jalankan.ipynb` dalam mode
QUICK. Harapan: lolos, dan hash cocok. Ukuran zip dicatat untuk README.

- [ ] **Langkah 4: Uji jalan QUICK semua notebook model, satu per satu, lalu `gabungan.ipynb`.** Harapan: semua tidak
  FAILED, total ±10–15 menit. Setelah itu `buat_notebook.buat_semua()` (semua tanpa output), hapus `cache/v11_quick/`,
  dan jalankan `eval "$PY -m pytest v11_model_tim/tests -q" < /dev/null`. Harapan: semua PASS; tes MiniRocket di-skip
  di venv utama.
- [ ] **Langkah 5: Tulis README dan entri AGENTS.md**, lalu `git status --short`. Harapan: hanya berkas `v11_model_tim/`,
  `requirements.txt`, dan `AGENTS.md` yang berubah.
- [ ] **Langkah 6: Serahkan ke pemilik repo**, dengan:
  - daftar berkas;
  - perintah `gh release create`;
  - catatan bahwa zip di `cache/v11_data.zip` belum diunggah, dan unduhan dari URL release baru teruji lewat
    `file://` (release belum ada);
  - hal yang belum dikerjakan: training penuh, laporan PDF, baris README root.
