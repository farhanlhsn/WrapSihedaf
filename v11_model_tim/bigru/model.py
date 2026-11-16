"""
BiGRU + Attention (pemilik repo). Arsitektur diimpor dari v2_revisi/sihedaf_models.py, tidak disalin.

Beda dengan BiGRU di v3: bobot kelas lunak (bukan penuh), dilatih di GPU bila ada, dan bobot tiap model fold
disimpan (supaya bisa diuji takikardia). Kontrak modul: lihat bersama/runner.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # v11_model_tim → paket `bersama`
from bersama import latih_dl as L  # noqa: E402
from sihedaf_models import build_model  # noqa: E402  (v2_revisi, sys.path diatur bersama.data)

NAMA = "bigru"
INPUT = "urutan"
SETELAN = {"hidden": 32, **L.BAWAAN}
SETELAN_QUICK = {"epochs": 1}


def buat_net(setelan):
    return build_model("BiGRUAttention", input_dim=2, hidden=setelan["hidden"], num_classes=3)


def latih(data, setelan, seed, device):
    return L.latih_paket_dl(lambda: buat_net(setelan), data, setelan, seed, device)


def prediksi(model, X, M, F, device):
    return L.prediksi_paket_dl(model, X, M, F, device)


def simpan(model, path):
    return L.simpan_paket_dl(model, path)


def info(model):
    return {"parameter": L.jumlah_parameter(model.net)}
