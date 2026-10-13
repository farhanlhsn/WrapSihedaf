# Dokumentasi Pipeline Ekstraksi Data Baru & Model Micro-Transformer (`cek lama/model.ipynb`)

Dokumentasi ini disusun secara lengkap dan terstruktur berdasarkan implementasi pada folder **`cek lama/`** khususnya notebook **`cek lama/model.ipynb`**, script ekstraksi **`cek lama/ekstraksi_hrv.py`**, pelatihan **`cek lama/train_and_save_model.py`**, serta modul inferensi **`cek lama/predictor.py`**.

---

## 📌 1. Struktur Repositori & File Acuan (`cek lama/`)

Seluruh sistem klasifikasi irama jantung berbasis Micro-Transformer dibangun di atas berkas-berkas berikut:

| Nama File / Path | Fungsi & Deskripsi Utama |
| :--- | :--- |
| **`cek lama/model.ipynb`** | Notebook utama eksperimen, arsitektur `MicroTransformer`, `ResNet1D`, pengujian benchmark cross-validation, visualisasi confusion matrix, dan pipeline inferensi raw JSON (`process_and_infer`). |
| **`cek lama/final.joblib`** | File artefak model terbaik (Skenario B1 Weighted Loss) berisi `state_dict` bobot PyTorch, `mu` & `sigma` normalisasi, `config`, dan `labels_map`. |
| **`cek lama/predictor.py`** | Modul kelas `ECGPredictor` produksi untuk memuat `final.joblib` dan menjalankan inferensi. |
| **`cek lama/train_and_save_model.py`** | Script Python untuk melatih ulang Micro-Transformer pada `dataset_hrv_sequence_23p.npz` dan mengekspor hasilnya ke `.joblib`. |
| **`cek lama/ekstraksi_hrv.py`** | Pipeline ekstraksi dataset AFDB (23 pasien PhysioNet) dari file sinyal EKG (`.dat`, `.hea`, `.qrsc`/`.qrs`, `.atr`) menjadi tabular CSV dan sequence NPZ. |
| **`cek lama/dataset_hrv_sequence_23p.npz`** | Dataset urutan HRV berbentuk numpy compressed array (`X`: shape $(N, 30, 2)$, `mask`: shape $(N, 30)$, `y`: shape $(N,)$). |
| **`cek lama/raw_71.json`** | Contoh sampel data sinyal EKG mentah format JSON untuk pengujian inferensi. |

---

## 🏗️ 2. Arsitektur Model Micro-Transformer (`cek lama/model.ipynb`)

Model yang dilatih pada `model.ipynb` adalah **Micro-Transformer Encoder** yang sangat ringan (8,611 parameter), dirancang khusus untuk memproses urutan variabel waktu antar detak jantung (HRV).

### Spesifikasi Arsitektur (`cek lama/model.ipynb` Cell 5)
- **Input Dimension (`input_dim`)**: 2 (Fitur per detak: `IBI` dalam ms dan `delta_IBI` dalam ms).
- **Hidden Dimension (`d_model`)**: 32.
- **Attention Heads (`nhead`)**: 2.
- **Feedforward Dimension (`dim_ff`)**: 64.
- **Activation Function**: GELU (`activation="gelu"`).
- **Transformer Layer**: 1 `nn.TransformerEncoderLayer` dengan `dropout=0.0`.
- **Pooling Layer**: Masked Mean Pooling (menghitung rata-rata hanya pada detak valid/non-padded).
- **Linear Output**: Classifier 3 Kelas (`0: Normal`, `1: AFIB`, `2: AFL`).

---

## 🔄 3. Pipeline Ekstraksi Data Baru (End-to-End Ingestion)

Sebagaimana didefinisikan dalam `cek lama/ekstraksi_hrv.py` dan Cell 11 `cek lama/model.ipynb`, pengolahan data EKG baru wajib melewati 6 langkah berikut:

```
Sinyal ECG Raw Baru (JSON / Array / WFDB)
   │
   ▼
[1. Resampling & Windowing 10s] ──> Resample ke fs (250 Hz) & potong ke window 10 detik
   │
   ▼
[2. Deteksi R-Peak]              ──> Ekstraksi indeks sampel puncak R (find_peaks / Pan-Tompkins)
   │
   ▼
[3. Kalkulasi IBI & Delta IBI]   ──> IBI = diff(peaks)/fs * 1000 (ms), Delta IBI = diff(IBI) (ms)
   │
   ▼
[4. Formatting Tensor (30, 2)]   ──> Pad/Truncate ke max 30 detak, buat Masking (1 = valid, 0 = pad)
   │
   ▼
[5. Z-Score Normalization]       ──> X_norm = (X - mu) / sigma * mask (mu & sigma dari final.joblib)
   │
   ▼
[6. Inferensi MicroTransformer]  ──> Output logit / probabilitas softmax via ECGPredictor
```

### Detail Langkah Ekstraksi Data Baru

#### Langkah 1: Resampling & Cut Window 10 Detik
Sinyal mentah dikalibrasi ke **$f_s = 250\text{ Hz}$** (atau diproses pada sampling rate target). Window analisis berdurasi **10 detik** (2500 sampel pada 250 Hz).

#### Langkah 2: Deteksi R-Peak
Mencari posisi puncak R dari sinyal EKG. Pada `cek lama/ekstraksi_hrv.py` digunakan anotasi QRS (`.qrsc`/`.qrs`), sedangkan pada Cell 11 `cek lama/model.ipynb` digunakan detektor puncak ringan `scipy.signal.find_peaks` dengan jarak minimum `target_fs * 0.4` sampel:
```python
peaks, _ = find_peaks(windowed_signal, distance=target_fs * 0.4, height=np.mean(windowed_signal))
```

#### Langkah 3: Kalkulasi Fitur IBI & Delta IBI (Dalam Milidetik)
> ⚠️ **CATATAN PENTING**: Berdasarkan statistik pada `final.joblib`, rerata nilai $\mu_{\text{IBI}} = 737.7807$ dan $\sigma_{\text{IBI}} = 204.97357$. Ini membuktikan bahwa satuan IBI yang digunakan pada pelatihan model di `cek lama/model.ipynb` adalah **MILIDETIK (ms)**.

Formulasi:
$$\text{IBI}_i = \left( \frac{\text{peaks}[i+1] - \text{peaks}[i]}{f_s} \right) \times 1000 \quad [\text{ms}]$$
$$\Delta \text{IBI}_i = \text{IBI}_i - \text{IBI}_{i-1} \quad [\text{ms}]$$
*(Untuk detak pertama $i=0$, $\Delta \text{IBI}_0 = 0.0$)*.

#### Langkah 4: Format Sequence & Masking Tensor
Urutan dibatasi maksimum **30 detak** per window 10 detik ($L_{max} = 30$):
- Matrix $X$: Shape `(1, 30, 2)` $\rightarrow$ Fitur 0: `IBI`, Fitur 1: `Delta_IBI`.
- Matrix $M$ (Mask): Shape `(1, 30)` $\rightarrow$ `1.0` untuk detak valid, `0.0` untuk padding.

#### Langkah 5: Z-Score Normalization
Memakai nilai `mu` dan `sigma` persis seperti tersimpan di `final.joblib`:
```python
X_norm = (X - artifact["mu"]) / artifact["sigma"] * mask[..., np.newaxis]
```

---

## 💻 4. Kode Implementasi Backend Berdasarkan `cek lama/predictor.py`

Berikut adalah kode modul produksi backend `predictor.py` (yang tersimpan di `cek lama/predictor.py`) untuk memuat `final.joblib` dan mengeksekusi inferensi:

```python
"""
Script Inference Produksi: cek lama/predictor.py
===============================================
Memuat artefak final.joblib dan menjalankan inferensi MicroTransformer.
"""

import joblib
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (Sesuai model.ipynb Cell 5)."""
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MicroTransformer(nn.Module):
    """Arsitektur Micro-Transformer Encoder (Sesuai model.ipynb Cell 5)."""
    def __init__(self, input_dim: int = 2, d_model: int = 32, nhead: int = 2, dim_ff: int = 64, num_classes: int = 3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, dropout=0.0, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pos_enc(self.embedding(x))
        key_pad = (mask == 0) if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=key_pad)
        if mask is not None:
            m = mask.unsqueeze(-1)
            x = (x * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return self.fc(x)


class ECGPredictor:
    """Class Inference Produksi untuk ECG / HRV Classification."""
    def __init__(self, model_path: str = "cek lama/final.joblib", device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        try:
            self.artifact = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Gagal memuat file model dari '{model_path}': {e}")

        cfg = self.artifact.get("config", {
            "input_dim": 2, "d_model": 32, "nhead": 2, "dim_ff": 64, "num_classes": 3
        })
        self.mu = np.array(self.artifact["mu"], dtype=np.float32)
        self.sigma = np.where(np.array(self.artifact["sigma"]) == 0, 1e-8, np.array(self.artifact["sigma"]))
        self.labels_map = self.artifact.get("labels_map", {
            0: "Normal", 1: "AFIB (Atrial Fibrillation)", 2: "AFL (Atrial Flutter)"
        })

        self.model = MicroTransformer(
            input_dim=cfg.get("input_dim", 2),
            d_model=cfg.get("d_model", 32),
            nhead=cfg.get("nhead", 2),
            dim_ff=cfg.get("dim_ff", 64),
            num_classes=cfg.get("num_classes", 3)
        ).to(self.device)

        state_dict = self.artifact.get("state_dict", self.artifact.get("model_state_dict"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _preprocess_input(self, X: np.ndarray, mask: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        if mask is None:
            mask = np.ones((X.shape[0], X.shape[1]), dtype=np.float32)
        else:
            mask = np.asarray(mask, dtype=np.float32)
            if mask.ndim == 1:
                mask = np.expand_dims(mask, axis=0)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_norm = (X - self.mu) / self.sigma * mask[..., np.newaxis]

        return torch.tensor(X_norm, dtype=torch.float32).to(self.device), \
               torch.tensor(mask, dtype=torch.float32).to(self.device)

    def predict(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        Xt, Mt = self._preprocess_input(X, mask)
        with torch.no_grad():
            logits = self.model(Xt, Mt)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predicted_ids = logits.argmax(dim=1).cpu().numpy()

        results = []
        for pid, prob in zip(predicted_ids, probabilities):
            results.append({
                "class_id": int(pid),
                "class_name": self.labels_map.get(int(pid), "Unknown"),
                "confidence": float(prob[pid]),
                "probabilities": {self.labels_map[i]: float(p) for i, p in enumerate(prob)}
            })

        return results
```

---

## 🛠️ 5. Pipeline Ekstraksi & Prediksi Data Mentah Baru (`process_and_infer`)

Fungsi berikut mencerminkan Cell 11 `cek lama/model.ipynb` untuk menerima data sinyal mentah EKG baru (JSON / Array) dan mengembalikan hasil prediksi:

```python
"""
Ekstraksi Data Sinyal Mentah Baru (Inisialisasi seperti Cell 11 model.ipynb)
"""

import json
import numpy as np
from scipy.signal import find_peaks, resample
from predictor import ECGPredictor

def process_raw_signal_and_predict(
    raw_input: str, 
    model_path: str = "cek lama/final.joblib", 
    fs_orig: int = 250
) -> dict:
    target_fs = 250
    window_sec = 10
    max_beats = 30

    # 1. Parse JSON / Raw Input
    if isinstance(raw_input, str):
        with open(raw_input, 'r') as f:
            raw_data = json.load(f)
        signal = np.array(raw_data)
    else:
        signal = np.array(raw_input)

    signal = signal.flatten()

    # 2. Resample jika fs_orig != 250 Hz
    if fs_orig != target_fs:
        num_samples = int(len(signal) * target_fs / fs_orig)
        signal = resample(signal, num_samples)

    target_samples = window_sec * target_fs
    if len(signal) > target_samples:
        windowed_signal = signal[:target_samples]
    else:
        windowed_signal = np.pad(signal, (0, target_samples - len(signal)), 'constant')

    # 3. Deteksi R-Peaks (find_peaks)
    peaks, _ = find_peaks(windowed_signal, distance=target_fs * 0.4, height=np.mean(windowed_signal))

    if len(peaks) < 2:
        return {"error": "Kurang dari 2 R-peak terdeteksi pada sinyal EKG."}

    # 4. Kalkulasi IBI (ms) & Delta IBI (ms)
    diff_samples = np.diff(peaks)
    ibi_ms = (diff_samples / float(target_fs)) * 1000.0
    delta_ibi_ms = np.zeros_like(ibi_ms)
    if len(ibi_ms) > 1:
        delta_ibi_ms[1:] = np.diff(ibi_ms)

    # 5. Format Sequence ke Shape (30, 2)
    N = min(len(ibi_ms), max_beats)
    X = np.zeros((30, 2), dtype=np.float32)
    M = np.zeros((30,), dtype=np.float32)

    X[:N, 0] = ibi_ms[:N]
    X[:N, 1] = delta_ibi_ms[:N]
    M[:N] = 1.0

    # 6. Predict via ECGPredictor
    predictor = ECGPredictor(model_path=model_path)
    res = predictor.predict(X, M)

    return res[0]
```

---

## ❓ 6. Penyebab Error Utama & Solusi untuk Tim Backend

| Potensi Error | Penyebab di `cek lama/` | Solusi Backend |
| :--- | :--- | :--- |
| **`AttributeError: 'dict' object has no attribute 'predict'`** | Menggunakan `joblib.load('final.joblib').predict()` secara langsung. | `final.joblib` berisi artefak dictionary PyTorch. Gunakan class `ECGPredictor` dari `cek lama/predictor.py`. |
| **Hasil Prediksi Berubah Mengacak** | Nilai IBI dimasukkan dalam satuan detik ($\sim 0.75\text{ s}$). | Pada `cek lama/model.ipynb`, `mu` IBI adalah $737.78\text{ ms}$. Wajib gunakan satuan **milidetik (ms)**! |
| **`FileNotFoundError` pada file `.npz` atau `.csv`** | Mengimpor script training lama yang mencari `dataset_hrv_sequence_23p.npz`. | Untuk backend inferensi, tidak perlu file `.npz` atau `.csv`. Cukup perlukan `cek lama/final.joblib` dan `cek lama/predictor.py`. |
| **Model Incompatible Shape Error** | Mengirimkan matriks dengan shape selain `(30, 2)`. | Pastikan urutan selalu berupa matriks padded 30 detak dan 2 kolom fitur (`[IBI, delta_IBI]`). |
