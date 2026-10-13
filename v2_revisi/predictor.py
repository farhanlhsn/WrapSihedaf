"""
predictor.py — inferensi produksi SIHEDAF v2.

Perbedaan dari versi lama:
- Memuat bobot dari .pt (state_dict, torch.load weights_only=True) + metadata
  .json. Tidak ada pickle/joblib → aman dimuat dan tidak terikat versi pickle.
- Verifikasi checksum SHA-256 bobot terhadap metadata.
- Praproses memakai hrv_core (kode yang sama dengan training).
- TIDAK ADA fallback diam-diam: file hilang / sinyal buruk → error atau
  status eksplisit, bukan prediksi dari bobot acak / scaler dummy.
- Bias per kelas hasil kalibrasi CV ikut diterapkan.
"""
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import hrv_core
from sihedaf_models import build_model


class ECGPredictor:
    def __init__(self, model_dir: str = "model_export", name: str = "sihedaf_v2",
                 device: Optional[str] = None):
        model_dir = Path(model_dir)
        meta_path = model_dir / f"{name}.json"
        weight_path = model_dir / f"{name}.pt"
        if not meta_path.exists() or not weight_path.exists():
            raise FileNotFoundError(f"Butuh {meta_path} dan {weight_path}")

        self.meta = json.loads(meta_path.read_text())
        digest = hashlib.sha256(weight_path.read_bytes()).hexdigest()
        if digest != self.meta["weights_sha256"]:
            raise ValueError("Checksum bobot tidak cocok dengan metadata — file tertukar/rusak.")

        pre = self.meta["preprocessing"]
        if pre != hrv_core.preprocessing_config():
            raise ValueError(
                "Konstanta hrv_core berbeda dari saat training. "
                f"Training: {pre} | Sekarang: {hrv_core.preprocessing_config()}"
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mu = np.asarray(self.meta["mu"], dtype=np.float32)
        self.sigma = np.asarray(self.meta["sigma"], dtype=np.float32)
        self.class_bias = np.asarray(self.meta.get("class_bias", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.labels_map = {int(k): v for k, v in self.meta["labels_map"].items()}

        self.model = build_model(self.meta["model_name"], **self.meta["model_config"]).to(self.device)
        state = torch.load(weight_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    # ── level sekuens ──────────────────────────────────────────────────
    @torch.no_grad()
    def predict_proba_sequences(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """X: (N, MAX_BEATS, 2) dalam ms, mask: (N, MAX_BEATS) → prob (N, 3)."""
        X = np.asarray(X, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if X.ndim == 2:
            X, mask = X[None], mask[None]
        Xn = (X - self.mu) / self.sigma * mask[..., None]
        logits = self.model(torch.from_numpy(Xn).to(self.device),
                            torch.from_numpy(mask).to(self.device)).cpu().numpy()
        logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        adj = logp + self.class_bias
        adj = np.exp(adj - adj.max(axis=1, keepdims=True))
        return adj / adj.sum(axis=1, keepdims=True)

    def _format(self, prob: np.ndarray) -> dict:
        cid = int(np.argmax(prob))
        return {
            "status": "ok",
            "class_id": cid,
            "class_name": self.labels_map[cid],
            "confidence": float(prob[cid]),
            "probabilities": {self.labels_map[i]: float(p) for i, p in enumerate(prob)},
        }

    # ── level IBI (jika perangkat sudah mengirim IBI) ──────────────────
    def predict_ibi(self, ibi_ms) -> dict:
        ibi_ms = np.asarray(ibi_ms, dtype=np.float64)
        if np.nanmedian(ibi_ms) < 10:
            raise ValueError("IBI tampak dalam DETIK. Kirim dalam MILIDETIK (mis. 800, bukan 0.8).")
        ok, reason = hrv_core.window_quality(ibi_ms)
        if not ok:
            return {"status": "tidak_terbaca", "reason": reason}
        X, m = hrv_core.build_sequence(ibi_ms)
        return self._format(self.predict_proba_sequences(X, m)[0])

    # ── level sinyal EKG mentah ────────────────────────────────────────
    def predict_ecg(self, ecg, fs: float, stride_sec: float = 5.0) -> dict:
        """
        EKG mentah → prediksi per window 10 detik + ringkasan (rata-rata
        probabilitas window yang valid). fs = sampling rate ASLI sinyal.
        """
        windows, peaks = hrv_core.signal_to_windows(np.asarray(ecg), fs, stride_sec)
        valid = [w for w in windows if w["ok"]]
        per_window = []
        if valid:
            seqs = [hrv_core.build_sequence(w["ibi_ms"]) for w in valid]
            X = np.stack([s[0] for s in seqs]); M = np.stack([s[1] for s in seqs])
            probs = self.predict_proba_sequences(X, M)
            for w, p in zip(valid, probs):
                r = self._format(p); r["start_sec"] = w["start_sec"]
                per_window.append(r)
        for w in windows:
            if not w["ok"]:
                per_window.append({"status": "tidak_terbaca", "start_sec": w["start_sec"], "reason": w["reason"]})
        per_window.sort(key=lambda r: r["start_sec"])

        dur = len(ecg) / fs
        hr = 60.0 * (len(peaks) - 1) / ((peaks[-1] - peaks[0]) / hrv_core.FS_DEVICE) if len(peaks) > 2 else None
        summary = {"status": "tidak_terbaca", "reason": "tidak ada window valid"}
        if valid:
            summary = self._format(np.mean(probs, axis=0))
        summary.update({
            "duration_sec": round(dur, 1), "n_rpeaks": int(len(peaks)),
            "heart_rate_bpm": round(hr, 1) if hr else None,
            "n_windows": len(windows), "n_windows_valid": len(valid),
        })
        return {"summary": summary, "windows": per_window}
