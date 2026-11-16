"""
Tes v11. Jalankan dari root repo: `python -m pytest v11_model_tim/tests -q`.

Butuh cache tim (`cache/`, cocok dengan sidik_jari.json) atau `SIHEDAF_DATA`. Tes MiniRocket di-skip bila aeon tidak
terpasang. Tidak ada training penuh: hanya model kecil/sebentar.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

V11 = Path(__file__).resolve().parents[1]
if str(V11) not in sys.path:
    sys.path.insert(0, str(V11))


@pytest.fixture(scope="session")
def d():
    from bersama import data
    return data.muat()


@pytest.fixture(scope="session")
def dev():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def muat_model():
    """Impor <folder>/model.py dengan nama modul unik (semua folder punya model.py)."""
    def _muat(nama):
        spec = importlib.util.spec_from_file_location(f"model_{nama}", V11 / nama / "model.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return _muat


def cek_invarian_padding(net):
    """Nilai X di posisi mask 0 tidak boleh mengubah output (pelajaran v2: ResNet yang mengabaikan mask)."""
    import torch
    net.eval()
    torch.manual_seed(0)
    x = torch.randn(4, 30, 2)
    m = torch.zeros(4, 30)
    m[0, :4] = 1                                    # window terpendek (hrv_core.MIN_IBI = 4)
    m[1, :13] = 1
    m[2, :27] = 1
    m[3, :] = 1
    with torch.no_grad():
        y1 = net(x * m[..., None], m)
        x2 = x.clone()
        x2[m == 0] = 99.0
        y2 = net(x2, m)
    assert y1.shape == (4, 3)
    assert torch.isfinite(y1).all()
    assert torch.allclose(y1, y2, atol=1e-5)
