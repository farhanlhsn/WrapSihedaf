import joblib
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding."""
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MicroTransformer(nn.Module):
    """Micro-Transformer Encoder."""
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

    def forward(self, x, mask=None):
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
    def __init__(self, model_path: str = "model_terbaik.joblib", device: str = None):
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
        self.mu = self.artifact["mu"]
        self.sigma = np.where(self.artifact["sigma"] == 0, 1e-8, self.artifact["sigma"])
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

    def _preprocess_input(self, X: np.ndarray, mask: np.ndarray = None):
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

    def predict(self, X: np.ndarray, mask: np.ndarray = None):
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
