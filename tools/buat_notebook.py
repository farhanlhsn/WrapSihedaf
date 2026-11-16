"""Tulis notebook .ipynb dari daftar sel [("md"|"code", teks)] tanpa output (dijalankan nanti lewat tools/run_notebook.py)."""
import sys

import nbformat


def tulis(path, sel):
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(t) if k == "md" else nbformat.v4.new_code_cell(t) for k, t in sel]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.validate(nb)
    nbformat.write(nb, path)
    return path


if __name__ == "__main__":
    print(__doc__, file=sys.stderr)
