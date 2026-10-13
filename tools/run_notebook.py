"""Jalankan notebook tanpa Jupyter UI dan simpan output tiap sel ke notebook itu.

Pemakaian (dari root repo; WSL/Linux, Colab, atau Kaggle):
    python tools/run_notebook.py v3_evaluasi_robust/evaluasi_robust.ipynb
    SIHEDAF_QUICK=1 python tools/run_notebook.py <notebook>     # uji alur cepat (jika notebook mendukung)

Direktori kerja = folder notebook (sama seperti membuka notebook di Jupyter), jadi path
relatif di notebook (../data, ../cache, model_export/) berlaku. Log progres ke stdout;
status akhir ke logs/<nama-notebook>.status.txt.
"""
import datetime
import os
import sys
import traceback
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(sys.argv[1]).resolve()
WORKDIR = SOURCE.parent
(ROOT / "logs").mkdir(exist_ok=True)
STATUS = ROOT / "logs" / f"{SOURCE.stem}.status.txt"
os.chdir(WORKDIR)
os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
os.environ["PYTHONPATH"] = str(WORKDIR) + os.pathsep + os.environ.get("PYTHONPATH", "")


# Kernel: SIHEDAF_KERNEL jika di-set; "sihedaf-v2" jika terpasang (setup WSL); selain itu "python3" (Colab/Kaggle).
KERNEL = os.environ.get("SIHEDAF_KERNEL") or (
    "sihedaf-v2" if "sihedaf-v2" in KernelSpecManager().find_kernel_specs() else "python3")


def log(msg):
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


nb = nbformat.read(SOURCE, as_version=4)
for c in nb.cells:
    if c.cell_type == "code":
        c.outputs, c.execution_count = [], None


def checkpoint():
    tmp = SOURCE.with_suffix(".ipynb.tmp")
    nbformat.write(nb, tmp)
    tmp.replace(SOURCE)


class StreamingClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg.get("msg_type") == "stream":
            print(msg["content"].get("text", ""), end="", flush=True)
        return super().process_message(msg, cell, cell_index)


def on_start(cell, cell_index, **_):
    log(f"START sel {cell_index}: {cell.source.splitlines()[0][:80] if cell.source else ''}")
    STATUS.write_text(f"RUNNING sel {cell_index}\n")


def on_done(cell, cell_index, **_):
    checkpoint()
    log(f"DONE sel {cell_index}")


client = StreamingClient(nb, timeout=None, kernel_name=KERNEL, allow_errors=False,
                         resources={"metadata": {"path": str(WORKDIR)}},
                         on_cell_start=on_start, on_cell_executed=on_done)
try:
    log(f"Mulai {SOURCE.relative_to(ROOT)} | cwd={WORKDIR.relative_to(ROOT)} | kernel={KERNEL} | "
        f"QUICK={os.environ.get('SIHEDAF_QUICK') == '1'}")
    client.execute()
except BaseException:
    checkpoint()
    STATUS.write_text("FAILED\n" + traceback.format_exc())
    traceback.print_exc()
    raise
else:
    checkpoint()
    STATUS.write_text("COMPLETE\n")
    log("COMPLETE")
