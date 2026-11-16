"""Unduh database PhysioNet dari mirror resmi S3 ke data/<folder>/, dicek dengan SHA256SUMS.txt, bisa dilanjutkan.

Pemakaian (dari root repo):
    python tools/unduh_physionet.py ephnogram 1.0.0 --ke ephnogram
    python tools/unduh_physionet.py ptb-xl 1.0.3 --ke ptbxl --pola "*.csv" "records100/*"

File yang sudah ada dan cocok checksum-nya dilewati. File besar diunduh bertahap ke .part (Range), jadi koneksi putus
tidak mengulang dari awal. Sesuai AGENTS.md: mirror S3, bukan zip physionet.org.

Database tanpa SHA256SUMS.txt (mis. challenge-2017): daftar berkas dan MD5 diambil dari ETag listing S3. ETag objek
satu-bagian = MD5 isinya; objek multipart (ETag berakhiran -N) tidak bisa dicek sehingga ditolak.
"""
import xml.etree.ElementTree as ET
import argparse
import fnmatch
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
S3 = "https://physionet-open.s3.amazonaws.com/{db}/{versi}/"
# Database yang tidak dicerminkan di S3 (mis. challenge-2021): physionet.org, tetap dicek dengan SHA256SUMS.txt
PHYSIONET = "https://physionet.org/files/{db}/{versi}/"
CHUNK = 1 << 20


def daftar_sha(base, sesi):
    r = sesi.get(base + "SHA256SUMS.txt", timeout=60)
    r.raise_for_status()
    return [(q[1].strip(), q[0]) for q in (b.split(maxsplit=1) for b in r.text.splitlines()) if len(q) == 2]


def daftar_etag(db, versi, sesi):
    """[(path relatif, "md5:<hex>")] dari ListObjectsV2 mirror S3 (dengan continuation token)."""
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    awal, out, token = f"{db}/{versi}/", [], None
    while True:
        par = {"list-type": "2", "prefix": awal, **({"continuation-token": token} if token else {})}
        r = sesi.get("https://physionet-open.s3.amazonaws.com/", params=par, timeout=60)
        r.raise_for_status()
        akar = ET.fromstring(r.content)
        for c in akar.findall("s:Contents", ns):
            kunci, etag = c.find("s:Key", ns).text, c.find("s:ETag", ns).text.strip('"')
            out.append((kunci[len(awal):], "md5:" + etag if "-" not in etag else "multipart:" + etag))
        if akar.findtext("s:IsTruncated", namespaces=ns) != "true":
            return out
        token = akar.findtext("s:NextContinuationToken", namespaces=ns)


def sha_file(path, algo="sha256"):
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for blok in iter(lambda: f.read(CHUNK), b""):
            h.update(blok)
    return h.hexdigest()


def unduh(db, versi, ke, pola=None, thread=16, coba=10, log=print, sumber="s3"):
    base = (S3 if sumber == "s3" else PHYSIONET).format(db=db, versi=versi)
    tujuan = ROOT / "data" / ke
    tujuan.mkdir(parents=True, exist_ok=True)
    lokal = threading.local()

    def sesi():
        if not hasattr(lokal, "s"):
            lokal.s = requests.Session()
        return lokal.s

    try:
        berkas = daftar_sha(base, requests.Session())
    except requests.HTTPError:
        if sumber != "s3":
            raise
        berkas = daftar_etag(db, versi, requests.Session())
        log(f"{db} {versi}: SHA256SUMS.txt tidak ada, dicek dengan MD5 dari ETag S3")
    if pola:
        berkas = [(p, h) for p, h in berkas if any(fnmatch.fnmatch(p, q) for q in pola)]
    if not berkas:
        raise RuntimeError(f"tidak ada berkas yang cocok di {base}")
    multi = [p for p, h in berkas if h.startswith("multipart:")]
    if multi:
        raise RuntimeError(f"objek multipart tanpa checksum yang bisa dicek: {multi[:5]} (persempit --pola)")

    def cocok(path, cek):
        algo, nilai = cek.split(":", 1) if ":" in cek else ("sha256", cek)
        return sha_file(path, algo) == nilai

    def ambil(item):
        rel, sha = item
        dest = tujuan / rel
        if dest.exists() and cocok(dest, sha):
            return dest.stat().st_size, 0
        part = dest.with_name(dest.name + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        for n in range(coba):
            try:
                awal = part.stat().st_size if part.exists() else 0
                kepala = {"Range": f"bytes={awal}-"} if awal else {}
                with sesi().get(base + rel, headers=kepala, stream=True, timeout=(20, 60)) as r:
                    if r.status_code == 416:            # .part sudah lengkap
                        pass
                    else:
                        r.raise_for_status()
                        mode = "ab" if awal and r.status_code == 206 else "wb"
                        with part.open(mode) as f:
                            for blok in r.iter_content(CHUNK):
                                f.write(blok)
                if not cocok(part, sha):
                    part.unlink()
                    raise OSError("checksum tidak cocok")
                part.replace(dest)
                return dest.stat().st_size, 1
            except (requests.RequestException, OSError) as e:
                if n == coba - 1:
                    raise RuntimeError(f"gagal mengunduh {rel}: {e}") from e
                time.sleep(min(2 * (n + 1), 20))

    t0, nbyte, nbaru = time.perf_counter(), 0, 0
    log(f"{db} {versi}: {len(berkas)} berkas → data/{ke}/ ({thread} thread)")
    with ThreadPoolExecutor(thread) as ex:
        for i, (b, baru) in enumerate(ex.map(ambil, berkas)):
            nbyte += b
            nbaru += baru
            if (i + 1) % max(1, len(berkas) // 10) == 0 or i + 1 == len(berkas):
                dt = time.perf_counter() - t0
                log(f"  {i + 1}/{len(berkas)} berkas, {nbyte / 1e9:.2f} GB, {nbyte / 1e6 / max(dt, 1e-9):.1f} MB/s")
    (tujuan / ".lengkap").write_text(f"{db} {versi} {len(berkas)} berkas {nbyte} byte, pola={pola}\n")
    log(f"{db} siap: {nbyte / 1e9:.2f} GB, {nbaru} berkas baru, {(time.perf_counter() - t0) / 60:.1f} mnt")
    return tujuan


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("db")
    ap.add_argument("versi")
    ap.add_argument("--ke", required=True, help="nama folder di data/")
    ap.add_argument("--pola", nargs="*", help="glob path relatif yang diambil (bawaan: semua)")
    ap.add_argument("--thread", type=int, default=16)
    ap.add_argument("--sumber", choices=("s3", "physionet"), default="s3",
                    help="s3 = mirror resmi (bawaan, AGENTS.md); physionet = physionet.org/files untuk database yang tidak ada di S3")
    a = ap.parse_args()
    unduh(a.db, a.versi, a.ke, a.pola, a.thread, log=lambda s: print(s, flush=True), sumber=a.sumber)
