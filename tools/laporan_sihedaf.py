"""Kerangka PDF laporan SIHEDAF (ReportLab, A4) dengan gaya AGENTS.md "Gaya laporan": Arial/Liberation Sans lewat
report_fonts, warna tata letak #008B83 / #172B43, palet data tervalidasi #2a78d6 / #eb6834 / #1baf7a.

Dipakai build_report.py versi v3.3, v13, v14, v15. Gaya disalin dari v11_model_tim/laporan/build_report.py.
Setiap penempatan memeriksa (assert) bahwa isi tidak melewati footer, supaya halaman yang terpotong ketahuan saat build.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_fonts import font_path  # noqa: E402

TEAL, INK, GRAY = "#008B83", "#172B43", "#526477"
BIRU, ORANYE, HIJAU = "#2a78d6", "#eb6834", "#1baf7a"


def fnum(v, nd=3):
    return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")


def sg(v, nd=3):
    return ("+" if v >= 0 else "") + fnum(v, nd) if round(v, nd) != 0 else fnum(abs(v), nd)


def pct(v, nd=1):
    return f"{v * 100:.{nd}f}%".replace(".", ",")


def thou(v):
    return f"{int(round(v)):,}".replace(",", ".")


def ci(a, b, nd=2):
    return f"[{fnum(a, nd)}–{fnum(b, nd)}]"


def vonis(p):
    return "lebih baik" if p >= .95 else "lebih buruk" if p <= .05 else "tidak terbedakan"


def daftar(v, nd=2):
    """[a; b; c] dengan koma desimal (mis. bias kelas)."""
    return "[" + "; ".join(fnum(x, nd) for x in v) + "]"


LABEL_PALSU = {"total": "total", "dari_normal": "dari ritme Normal", "dari_afib": "dari AFIB",
               "pasien_tanpa_afl": "di pasien tanpa AFL"}


def sumbu_koma(ax, sumbu="y"):
    """Angka sumbu matplotlib dengan koma desimal (gaya laporan) dan grid di belakang data."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
    (ax.yaxis if sumbu == "y" else ax.xaxis).set_major_formatter(f)
    ax.set_axisbelow(True)


class Laporan:
    def __init__(self, out, judul, kicker_atas, sumber, tanggal):
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
        for name, style in (("Arial", "regular"), ("Arial-Bold", "bold"), ("Arial-Italic", "italic")):
            pdfmetrics.registerFont(TTFont(name, font_path(style)))
        pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
        self.HexColor, self.white, self.pdfmetrics = HexColor, white, pdfmetrics
        self.S = {
            "body": ParagraphStyle("body", fontName="Arial", fontSize=10, leading=14.3, textColor=HexColor(INK)),
            "small": ParagraphStyle("small", fontName="Arial", fontSize=8.6, leading=11.8, textColor=HexColor(GRAY)),
            "table": ParagraphStyle("table", fontName="Arial", fontSize=8.1, leading=10.6, textColor=HexColor(INK)),
            "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=8.0, leading=10.6, textColor=white),
            "box": ParagraphStyle("box", fontName="Arial", fontSize=8.7, leading=11.7, textColor=HexColor(INK)),
        }
        self.c = canvas.Canvas(str(out), pagesize=A4, pageCompression=1)
        self.c.setTitle(judul)
        self.c.setAuthor("SIHEDAF (pemilik repo dan tim)")
        self.W, self.H = A4
        self.M = 40
        self.CW = self.W - 2 * self.M
        self.page_no = 0
        self.kicker_atas, self.sumber, self.tanggal = kicker_atas, sumber, tanggal

    def para(self, txt, y, style="body", x=None, width=None, gap=8):
        from reportlab.platypus import Paragraph
        p = Paragraph(txt, self.S[style])
        _, h = p.wrap(width or self.CW, self.H)
        assert y - h >= 51, f"hal {self.page_no}: paragraf lewat footer: {txt[:60]}"
        p.drawOn(self.c, x if x is not None else self.M, y - h)
        return y - h - gap

    def page(self, title, kicker):
        c, H = self.c, self.H
        if self.page_no:
            c.showPage()
        self.page_no += 1
        n = self.page_no
        c.setFillColor(self.HexColor(TEAL)); c.rect(self.M, H - 41, 32, 4, fill=1, stroke=0)
        c.setFillColor(self.HexColor(GRAY)); c.setFont("Arial-Bold", 8)
        c.drawString(self.M + 43, H - 42, self.kicker_atas)
        fs = 20
        while self.pdfmetrics.stringWidth(title, "Arial-Bold", fs) > self.CW and fs > 13:
            fs -= .5
        c.setFillColor(self.HexColor(INK)); c.setFont("Arial-Bold", fs); c.drawString(self.M, H - 78, title)
        c.setFillColor(self.HexColor(GRAY)); c.setFont("Arial", 9); c.drawString(self.M, H - 97, kicker)
        c.setStrokeColor(self.HexColor("#D8E1E9")); c.line(self.M, 43, self.W - self.M, 43)
        c.setFont("Arial", 8)
        c.drawString(self.M, 29, f"{self.tanggal} | Sumber: {self.sumber}")
        c.drawRightString(self.W - self.M, 29, f"{n:02d}")
        c.bookmarkPage(f"p{n}"); c.addOutlineEntry(title, f"p{n}", 0, False)
        return H - 118

    def section(self, title, y):
        self.c.setFont("Arial-Bold", 11.5); self.c.setFillColor(self.HexColor(INK)); self.c.drawString(self.M, y - 12, title)
        return y - 24

    def table(self, headers, rows, y, widths=None, sorot=()):
        from reportlab.platypus import Paragraph, Table, TableStyle
        HexColor = self.HexColor
        widths = widths or [self.CW / len(headers)] * len(headers)
        cells = [[Paragraph(str(v), self.S["head"]) for v in headers]]
        cells += [[Paragraph(str(v), self.S["table"]) for v in row] for row in rows]
        tb = Table(cells, colWidths=widths, hAlign="LEFT")
        st = [("BACKGROUND", (0, 0), (-1, 0), HexColor(INK)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
              ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
              ("LINEBELOW", (0, -1), (-1, -1), .5, HexColor("#D8E1E9")),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [self.white, HexColor("#F1F5F8")])]
        for r in sorot:
            st.append(("BACKGROUND", (0, r), (-1, r), HexColor("#E3F1EF")))
        tb.setStyle(TableStyle(st))
        _, h = tb.wrap(self.CW, self.H)
        assert y - h >= 51, f"hal {self.page_no}: tabel lewat footer ({y - h:.0f})"
        tb.drawOn(self.c, self.M, y - h)
        return y - h - 11

    def chart(self, path, y, maxh):
        from PIL import Image
        iw, ih = Image.open(path).size
        w = self.CW; h = w * ih / iw
        if h > maxh:
            h = maxh; w = h * iw / ih
        assert y - h >= 51, (str(path), y, h)
        self.c.drawImage(str(path), self.M + (self.CW - w) / 2, y - h, width=w, height=h, mask="auto")
        return y - h - 9

    def kartu(self, items, y, h=64):
        """Baris kartu angka: [(angka, label, catatan)]."""
        c, HexColor = self.c, self.HexColor
        n = len(items)
        gap = 10
        w = (self.CW - gap * (n - 1)) / n
        for i, (angka, label, catatan) in enumerate(items):
            x = self.M + i * (w + gap)
            c.setFillColor(HexColor("#F1F5F8")); c.rect(x, y - h, w, h, fill=1, stroke=0)
            c.setFillColor(HexColor(TEAL)); c.rect(x, y - 3, w, 3, fill=1, stroke=0)
            c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold", 17); c.drawString(x + 8, y - 26, angka)
            c.setFont("Arial-Bold", 8.2); c.drawString(x + 8, y - 41, label)
            c.setFillColor(HexColor(GRAY)); c.setFont("Arial", 7.6); c.drawString(x + 8, y - 54, catatan)
        return y - h - 12

    def selesai(self):
        self.c.save()
