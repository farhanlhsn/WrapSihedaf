"""
sihedaf_models.py — definisi arsitektur, dipakai notebook DAN predictor.
Satu definisi → tidak ada lagi salinan kelas yang bisa berbeda diam-diam.
"""
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: (B, S, C), mask: (B, S) → (B, C), rata-rata hanya detak valid."""
    m = mask.unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


class MicroTransformer(nn.Module):
    """Sama persis dengan versi lama (nama parameter kompatibel)."""
    def __init__(self, input_dim=2, d_model=32, nhead=2, dim_ff=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, dropout=dropout, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask):
        h = self.pos_enc(self.embedding(x))
        h = self.encoder(h, src_key_padding_mask=(mask == 0))
        return self.fc(masked_mean(h, mask))


class _ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.skip = nn.Conv1d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.act = nn.SiLU()  # = Swish

    def forward(self, x, m):          # x: (B, C, S), m: (B, 1, S)
        out = self.act(self.bn1(self.conv1(x))) * m
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x)) * m


class MaskedResNet1D(nn.Module):
    """
    1D-ResNet yang MEMAKAI mask (versi lama mengabaikannya sehingga
    posisi padding ikut dirata-rata di global pooling).
    """
    def __init__(self, input_dim=2, num_classes=3, channels=(32, 64, 128)):
        super().__init__()
        chans = (input_dim,) + tuple(channels)
        self.blocks = nn.ModuleList(_ResBlock(chans[i], chans[i + 1]) for i in range(len(channels)))
        self.fc = nn.Linear(chans[-1], num_classes)

    def forward(self, x, mask):
        m = mask.unsqueeze(1)                 # (B, 1, S)
        h = x.permute(0, 2, 1) * m
        for blk in self.blocks:
            h = blk(h, m)
        return self.fc(masked_mean(h.permute(0, 2, 1), mask))


class BiGRUAttention(nn.Module):
    """Bi-GRU + attention pooling bermask. Ringan (~15 ribu parameter)."""
    def __init__(self, input_dim=2, hidden=32, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x, mask):
        lengths = mask.sum(1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        score = self.att(h).squeeze(-1).masked_fill(mask == 0, -1e9)
        w = torch.softmax(score, dim=1).unsqueeze(-1)
        return self.fc((h * w).sum(1))


MODEL_REGISTRY = {
    "MicroTransformer": MicroTransformer,
    "MaskedResNet1D": MaskedResNet1D,
    "BiGRUAttention": BiGRUAttention,
}


def build_model(name: str, **kwargs) -> nn.Module:
    return MODEL_REGISTRY[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
