"""Unduh satu berkas (Zenodo, Hugging Face, dll.) ke data/<folder>/ dengan checksum resmi; gagal bila tidak cocok.

Pemakaian (dari root repo):
    python tools/unduh_berkas.py URL --ke epfl_hi --md5 1aa4af41...
    python tools/unduh_berkas.py URL --ke ecgfounder --sha256 f863a388...

Checksum wajib diisi dari halaman/API resmi sumbernya (bukan dihitung dari berkas yang diunduh).
"""
import argparse
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent


def unduh(url, ke, sha256=None, md5=None, nama=None, log=print):
    if not (sha256 or md5):
        raise ValueError("isi --sha256 atau --md5 dari sumber resmi")
    dest = ROOT / "data" / ke / (nama or Path(urlparse(url).path).name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256() if sha256 else hashlib.md5()
    harap = (sha256 or md5).lower()

    def cek(p):
        hh = hashlib.sha256() if sha256 else hashlib.md5()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                hh.update(b)
        return hh.hexdigest() == harap

    if dest.exists() and cek(dest):
        log(f"sudah ada dan cocok: {dest}")
        return dest
    part = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=(20, 120)) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for b in r.iter_content(1 << 20):
                f.write(b)
                h.update(b)
    if h.hexdigest() != harap:
        part.unlink()
        raise RuntimeError(f"checksum tidak cocok untuk {url}")
    part.replace(dest)
    log(f"siap: {dest} ({dest.stat().st_size:,} byte)")
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url")
    ap.add_argument("--ke", required=True)
    ap.add_argument("--nama")
    ap.add_argument("--sha256")
    ap.add_argument("--md5")
    a = ap.parse_args()
    unduh(a.url, a.ke, a.sha256, a.md5, a.nama, log=lambda s: print(s, flush=True))
