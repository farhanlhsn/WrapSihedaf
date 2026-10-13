"""Unduh AFDB (PhysioNet) ke data/afdb/, bisa dilanjutkan jika terputus.

Pemakaian (WSL, dari root repo):  python tools/download_afdb.py
Alternatif tanpa skrip: sel unduh di v2_revisi/revisi_sihedaf_v2.ipynb (wfdb.dl_database).
"""
import concurrent.futures
import re
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://physionet-open.s3.amazonaws.com/afdb/1.0.0/"
DATA = ROOT / "data/afdb"
DATA.mkdir(parents=True, exist_ok=True)
index = (Path(__file__).resolve().parent / "afdb_index.html").read_text()
files = []
for line in index.splitlines():
    match = re.search(r'href="([0-9]{5}\.(?:dat|hea|atr|qrs|qrsc))".*?\s(\d+)\s*$', line)
    if match:
        files.append((match[1], int(match[2])))
if not files:
    raise RuntimeError("No AFDB records found in official directory index")
local = threading.local()


def download(item):
    if not hasattr(local, "session"):
        local.session = requests.Session()
    name, size = item
    target = DATA / name
    partial = DATA / (name + ".part")
    if target.exists() and target.stat().st_size == size:
        return name
    for attempt in range(12):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            partial.replace(target)
            print(f"DONE {name} ({size:,} bytes)", flush=True)
            return name
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with local.session.get(BASE + name, headers=headers, stream=True, timeout=(20, 30)) as response:
                response.raise_for_status()
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with partial.open(mode) as output:
                    for chunk in response.iter_content(16384):
                        output.write(chunk)
            if partial.stat().st_size != size:
                raise RuntimeError(f"Incomplete {name}: {partial.stat().st_size}/{size}")
            partial.replace(target)
            print(f"DONE {name} ({size:,} bytes)", flush=True)
            return name
        except (requests.RequestException, RuntimeError) as error:
            print(f"RETRY {name} ({attempt + 1}/12): {error}", flush=True)
            time.sleep(min(attempt + 1, 10))
    (DATA / "download_status.txt").write_text(f"FAILED: {name}\n")
    raise RuntimeError(f"Unable to finish {name}")


print(f"AFDB: {len(files)} files, {sum(s for _, s in files) / 1e6:.1f} MB", flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(download, files))
records = sorted({name[:5] for name, _ in files if name.endswith(".hea")})
(DATA / "RECORDS").write_text("\n".join(records) + "\n")
(DATA / "download_status.txt").write_text("COMPLETE\n")
print("COMPLETE: AFDB ready, including RECORDS and corrected annotations", flush=True)
