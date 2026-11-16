"""Notebook ditulis langsung (tanpa generator): enam notebook model harus identik kecuali baris MODEL dan sel Kesimpulan.

Output boleh ada: sejak v11 selesai (3 November 2026) notebook di repo menyimpan run pemilik repo sebagai catatan hasil
(Definition of Done butir 1), dan sel Kesimpulan berisi angka model masing-masing.
"""
import re

import nbformat
import pytest

from bersama import MODEL_TIM, data

POLA_MODEL = re.compile(r'^MODEL = "(\w+)"', re.M)
JUDUL_KESIMPULAN = "## 5. Kesimpulan"


def _baca(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    return nb


def _kesimpulan(c):
    return c.cell_type == "markdown" and c.source.startswith(JUDUL_KESIMPULAN)


@pytest.mark.parametrize("nama", MODEL_TIM)
def test_notebook_model_valid(nama):
    nb = _baca(data.V11 / nama / "jalankan.ipynb")
    src = "\n".join(c.source for c in nb.cells)
    assert POLA_MODEL.findall(src) == [nama]
    assert 'NAMA_SETELAN = "bawaan"' in src and "data.siapkan_data(DATA_URL)" in src
    assert sum(_kesimpulan(c) for c in nb.cells) == 1
    assert (data.V11 / nama / "model.py").exists() and (data.V11 / nama / "README.md").exists()


def test_notebook_model_identik():
    isi = {}
    for n in MODEL_TIM:
        nb = nbformat.read(data.V11 / n / "jalankan.ipynb", as_version=4)
        sel = [c.cell_type + "\n" + (JUDUL_KESIMPULAN if _kesimpulan(c) else c.source) for c in nb.cells]
        isi[n] = POLA_MODEL.sub('MODEL = "X"', "\n␞\n".join(sel))
    assert len(set(isi.values())) == 1, "notebook model berbeda selain baris MODEL dan sel Kesimpulan; samakan keenamnya"


def test_gabungan_valid():
    _baca(data.V11 / "gabungan.ipynb")
