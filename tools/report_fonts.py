"""Cari font TTF untuk laporan PDF di berbagai lingkungan (Windows, WSL, Linux, Colab, Kaggle).

Urutan: Arial (Windows / WSL / msttcorefonts) → Liberation Sans (metrik sama dengan Arial,
`apt install fonts-liberation`) → DejaVu Sans bawaan matplotlib (cadangan; huruf lebih lebar,
tata letak bisa bergeser sedikit).

Pemakaian di build_report.py:
    sys.path.insert(0, str(REPO / "tools")); from report_fonts import font_path
    TTFont("Arial", font_path("regular"))
"""
from pathlib import Path

_CANDIDATES = {
    "regular": ["arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"],
    "bold": ["arialbd.ttf", "Arial_Bold.ttf", "LiberationSans-Bold.ttf"],
    "italic": ["ariali.ttf", "Arial_Italic.ttf", "LiberationSans-Italic.ttf"],
}
_DIRS = [
    "C:/Windows/Fonts", "/mnt/c/Windows/Fonts",
    "/usr/share/fonts/truetype/msttcorefonts",
    "/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/liberation2",
    "/usr/share/fonts/liberation-sans",
]
_DEJAVU = {"regular": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf", "italic": "DejaVuSans-Oblique.ttf"}


def font_path(style: str = "regular") -> str:
    for name in _CANDIDATES[style]:
        for d in _DIRS:
            p = Path(d) / name
            if p.exists():
                return str(p)
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts/ttf" / _DEJAVU[style]
        if p.exists():
            print(f"PERINGATAN: Arial/Liberation Sans tidak ada, memakai {p.name}. Huruf lebih lebar; "
                  "laporan bisa gagal assert 'lewat footer'. Pasang: apt-get install -y fonts-liberation")
            return str(p)
    except ImportError:
        pass
    raise FileNotFoundError(f"Font '{style}' tidak ditemukan. Pasang fonts-liberation atau matplotlib.")
