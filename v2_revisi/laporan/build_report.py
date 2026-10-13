"""Build the SIHEDAF result report without rerunning training.

Run --extract with the WSL SIHEDAF Python (pandas/matplotlib); then run without
arguments with the bundled Windows Python (ReportLab). Source outputs are read
from the executed notebook; raw OOF predictions were not saved by that notebook.
"""
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # = v2_revisi/
sys.path.insert(0, str(ROOT.parent / "tools"))
from report_fonts import font_path  # noqa: E402  (Arial → Liberation Sans → DejaVu)
TMP = ROOT / "laporan/_build"
OUT = ROOT / "laporan/laporan_hasil_sihedaf_v2.pdf"
NBPATH = ROOT / "revisi_sihedaf_v2.ipynb"
META = ROOT / "model_export/sihedaf_v2.json"
BLUE, TEAL, INK, GRAY, ORANGE = "#2455A4", "#008B83", "#172B43", "#526477", "#C76B27"
ALIASES = ["A1 LightGBM", "A2 LGBM + SMOTE", "B1 Transformer", "B1 ResNet", "B1 BiGRU", "B2 Transformer"]


def extract():
    import numpy as np
    import pandas as pd
    from bs4 import BeautifulSoup
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TMP.mkdir(parents=True, exist_ok=True)
    nb = json.loads(NBPATH.read_text(encoding="utf-8"))
    payload = {}
    for idx in (10, 12, 14, 18, 19, 21, 23, 26, 32):
        for o in nb["cells"][idx].get("outputs", []):
            html = o.get("data", {}).get("text/html")
            if html:
                soup = BeautifulSoup("".join(html), "html.parser")
                headers = [x.get_text(" ", strip=True) or "index" for x in soup.select("thead tr")[0].find_all("th")]
                records = []
                for row in soup.select("tbody tr"):
                    values = [x.get_text(" ", strip=True) for x in row.find_all(["th", "td"], recursive=False)]
                    assert len(values) == len(headers), (idx, values, headers)
                    item = {}
                    for key, value in zip(headers, values):
                        if key != "index" and re.fullmatch(r"-?\d+(?:\.\d+)?", value):
                            value = float(value) if "." in value else int(value)
                        item[key] = value
                    records.append(item)
                payload[str(idx)] = records
    assert len(payload) == 9, payload.keys()
    assert all(c.get("execution_count") is not None for c in nb["cells"] if c["cell_type"] == "code")
    assert not any(o.get("output_type") == "error" for c in nb["cells"] for o in c.get("outputs", []))
    payload["code_cells"] = sum(c["cell_type"] == "code" for c in nb["cells"])
    payload["notebook_sha256"] = hashlib.sha256(NBPATH.read_bytes()).hexdigest()
    (TMP / "tables.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for idx, name in ((18, "confusion_benchmark"), (23, "confusion_domain")):
        for o in nb["cells"][idx]["outputs"]:
            img = o.get("data", {}).get("image/png")
            if img:
                (TMP / f"{name}.png").write_bytes(base64.b64decode("".join(img)))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#C4CED8",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    def save(fig, name):
        fig.savefig(TMP / f"{name}.png", dpi=210, bbox_inches="tight", pad_inches=.12)
        plt.close(fig)
    main = payload["18"]
    y = np.arange(6)
    fig, ax = plt.subplots(figsize=(7.0, 3.05))
    vals = [r["Macro-F1 (pooled)"] for r in main]
    ax.barh(y, vals, color=[BLUE, BLUE, TEAL, TEAL, TEAL, TEAL], height=.65)
    ax.set_yticks(y, ALIASES); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 pooled OOF, argmax (sumber anotasi)")
    ax.grid(axis="x", alpha=.16); ax.set_axisbelow(True)
    for i, v in enumerate(vals): ax.text(v+.012, i, f"{v:.3f}", va="center", fontsize=10)
    fig.tight_layout(); save(fig, "main_f1")

    patients = payload["19"]
    cols = [r["Model"] for r in main]
    matrix = np.array([[r[c] for c in cols] for r in patients])
    labels = [f"{int(r[next(iter(r))]):05d}  |  n={int(r['jumlah window AFL'])}" for r in patients]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(7), labels)
    ax.set_xticks(range(6), ["A1 LGBM", "A2 SMOTE", "B1 Transf.", "B1 ResNet", "B1 BiGRU", "B2 Transf."])
    ax.tick_params(axis="both", length=0); ax.tick_params(axis="x", labelsize=9)
    for i in range(7):
        for j in range(6): ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="white" if matrix[i,j]>.6 else INK, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.03); cb.set_label("Recall AFL")
    fig.tight_layout(); save(fig, "afl_patients")

    calib = payload["21"]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for i, r in enumerate(calib):
        before, after = r["Macro-F1 sebelum"], r["Macro-F1 sesudah"]
        ax.plot([before, after], [i, i], color=TEAL if after>=before else ORANGE, lw=3)
    ax.scatter([r["Macro-F1 sebelum"] for r in calib], y, color=GRAY, label="Argmax", s=40, zorder=3)
    ax.scatter([r["Macro-F1 sesudah"] for r in calib], y, color=TEAL, label="Bias cross-fit", s=45, zorder=3)
    ax.set_yticks(y, ALIASES); ax.invert_yaxis(); ax.set_xlim(.62, .90); ax.set_xlabel("Macro-F1 pooled OOF (anotasi)")
    ax.legend(loc="lower right", frameon=False); ax.grid(axis="x", alpha=.16)
    fig.tight_layout(); save(fig, "calibration")

    shift = payload["23"]
    fig, ax = plt.subplots(figsize=(7.0, 2.65))
    for start, name, color in ((0,"LightGBM", BLUE),(3,"MicroTransformer",TEAL)):
        vv=[r["Macro-F1 (pooled)"] for r in shift[start:start+3]]
        ax.plot(range(3), vv, "o-", lw=2.4, color=color, label=name)
        for i, v in enumerate(vv): ax.annotate(f"{v:.3f}", (i,v), xytext=(0,9), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(range(3), ["S1: anotasi → anotasi", "S2: anotasi → detektor", "S3: detektor → detektor"])
    ax.set_ylim(.60,.80); ax.set_ylabel("Macro-F1, argmax"); ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower left", frameon=False); fig.tight_layout(); save(fig,"domain_shift")

    stats=pd.read_csv(ROOT.parent/"cache/dataset_v2_detector.stats.csv", dtype={"record":str}).sort_values("sensitivitas")
    payload["detector_stats"]=stats.to_dict("records")
    (TMP / "tables.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, ax=plt.subplots(figsize=(7.0,5.15))
    yy=np.arange(len(stats))
    ax.barh(yy-.16,stats.sensitivitas,height=.31,color=TEAL,label="Sensitivitas")
    ax.barh(yy+.16,stats.PPV,height=.31,color=BLUE,label="PPV")
    ax.set_yticks(yy,stats.record); ax.invert_yaxis(); ax.set_xlim(0,1.06)
    ax.tick_params(axis="y", labelsize=8.5); ax.set_xlabel("Kecocokan R-peak, toleransi 100 ms")
    ax.grid(axis="x",alpha=.16); ax.legend(loc="upper center",bbox_to_anchor=(.5,1.075),ncol=2,frameon=False,fontsize=9)
    fig.tight_layout(); save(fig,"detector_quality")

    lat=payload["26"]
    fig, ax=plt.subplots(figsize=(7.0,2.45))
    names=["LightGBM", "MicroTransformer", "MaskedResNet1D", "BiGRUAttention"]
    vv=[r["1.000 window (ms)"] for r in lat]
    ax.barh(range(4),vv,color=[BLUE,TEAL,TEAL,TEAL],height=.62)
    ax.set_yticks(range(4),names); ax.invert_yaxis(); ax.set_xlim(0,360)
    ax.set_xlabel("Latensi CPU per 1.000 window (ms), termasuk praproses HRV/sekuens")
    for i,v in enumerate(vv): ax.text(v+4,i,f"{v:.2f}",va="center")
    ax.grid(axis="x",alpha=.16); fig.tight_layout(); save(fig,"latency")
    print("Extracted source tables and generated six charts")


def build():
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor, Color, white
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image
    from xml.sax.saxutils import escape

    OUT.parent.mkdir(parents=True, exist_ok=True)
    d=json.loads((TMP/"tables.json").read_text(encoding="utf-8"))
    meta=json.loads(META.read_text(encoding="utf-8"))
    weights=ROOT/"model_export/sihedaf_v2.pt"
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==meta["weights_sha256"]
    for name,style in (("Arial","regular"),("Arial-Bold","bold"),("Arial-Italic","italic")):
        pdfmetrics.registerFont(TTFont(name,font_path(style)))
    pdfmetrics.registerFontFamily("Arial",normal="Arial",bold="Arial-Bold",italic="Arial-Italic")
    styles={
        "body":ParagraphStyle("body",fontName="Arial",fontSize=10.1,leading=14.6,textColor=HexColor(INK)),
        "small":ParagraphStyle("small",fontName="Arial",fontSize=8.7,leading=12,textColor=HexColor(GRAY)),
        "table":ParagraphStyle("table",fontName="Arial",fontSize=8.6,leading=11.5,textColor=HexColor(INK)),
        "head":ParagraphStyle("head",fontName="Arial-Bold",fontSize=8.5,leading=11.5,textColor=white),
    }
    c=canvas.Canvas(str(OUT),pagesize=A4,pageCompression=1)
    c.setTitle("Laporan Hasil SIHEDAF v2 - Eksekusi 12 Oktober 2026")
    c.setAuthor("Laporan analisis hasil notebook SIHEDAF")
    c.setSubject("Benchmark pasien, AFL, bias kelas, pergeseran domain, dan model final")
    page_num=0; W,H=A4; M=40; CW=W-2*M
    def para(txt,y,style="body",x=M,width=None):
        p=Paragraph(txt,styles[style]); _,h=p.wrap(width or CW, H)
        assert y-h >= 51, f"Page {page_num}: paragraph below footer: {txt[:60]} -> {y-h}"
        p.drawOn(c,x,y-h); return y-h-9
    def page(title,kicker):
        nonlocal page_num
        if page_num: c.showPage()
        c.setPageSize(A4); page_num+=1
        c.setFillColor(HexColor(TEAL)); c.rect(M,H-41,32,4,fill=1,stroke=0)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial-Bold",8)
        c.drawString(M+43,H-42,"SIHEDAF v2  /  HASIL EKSPERIMEN")
        c.setFillColor(HexColor(INK)); c.setFont("Arial-Bold",21); c.drawString(M,H-79,title)
        c.setFillColor(HexColor(GRAY)); c.setFont("Arial",9); c.drawString(M,H-99,kicker)
        c.setStrokeColor(HexColor("#D8E1E9"));c.line(M,43,W-M,43)
        c.setFont("Arial",8); c.drawString(M,29,"Eksekusi 12 Oktober 2026 | Sumber: notebook dan metadata lokal")
        c.drawRightString(W-M,29,f"{page_num:02d}")
        c.bookmarkPage(f"page{page_num}"); c.addOutlineEntry(title,f"page{page_num}",0,False)
        return H-121
    def section(title,y):
        c.setFont("Arial-Bold",12); c.setFillColor(HexColor(INK)); c.drawString(M,y-12,title)
        return y-25
    def table(headers,rows,y,widths=None,fs=None):
        widths=widths or [CW/len(headers)]*len(headers)
        cells=[[Paragraph(escape(str(v)),styles["head"]) for v in headers]]
        cells.extend([[Paragraph(escape(str(v)),styles["table"]) for v in row] for row in rows])
        t=Table(cells,colWidths=widths,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HexColor(INK)),("ROWBACKGROUNDS",(0,1),(-1,-1),[white,HexColor("#F1F5F8")]),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
            ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
            ("LINEBELOW",(0,-1),(-1,-1),.5,HexColor("#D8E1E9"))]))
        _,h=t.wrap(CW,H); assert y-h>=51,f"Page {page_num}: table overflow {y-h}"
        t.drawOn(c,M,y-h);return y-h-13
    def chart(name,y,maxh):
        p=TMP/f"{name}.png"; iw,ih=Image.open(p).size
        width=CW; height=width*ih/iw
        if height>maxh: height=maxh; width=height*iw/ih
        assert y-height>=51,(name,y,height)
        c.drawImage(str(p),M+(CW-width)/2,y-height,width=width,height=height,mask="auto")
        return y-height-10
    def callout(title,txt,y,tint="#EDF6F5"):
        p=Paragraph(f"<b>{title}</b><br/>{txt}",styles["body"]); _,h=p.wrap(CW-24,H)
        assert y-h-20>=51,(title,y,h)
        c.setFillColor(HexColor(tint)); c.roundRect(M,y-h-20,CW,h+20,6,fill=1,stroke=0)
        p.drawOn(c,M+12,y-h-10);return y-h-31
    def n(v): return f"{int(v):,}".replace(",",".")
    def f(v): return f"{float(v):.3f}".replace(".",",")

    raw=meta["cv_metrics"]["argmax"]; cal=meta["cv_metrics"]["dengan_bias_crossfit"]
    derived={}
    counts=meta['training']['class_counts']; total=sum(counts)
    for name, metrics in meta['cv_metrics'].items():
        tp=[round(metrics[k]*count) for k,count in zip(['Normal R','AFIB R','AFL R'],counts)]
        predicted=[0,round(tp[1]/metrics['AFIB P']),round(tp[2]/metrics['AFL P'])]
        predicted[0]=total-predicted[1]-predicted[2]
        f1=[2*t/(support+pred) for t,support,pred in zip(tp,counts,predicted)]
        assert all(t<=s and t<=p for t,s,p in zip(tp,counts,predicted))
        assert abs(sum(f1)/3-metrics['Macro-F1 (pooled)'])<1e-12
        derived[name]={'Weighted-F1':sum(v*count for v,count in zip(f1,counts))/total,'Accuracy':sum(tp)/total}
    y=page("Laporan hasil SIHEDAF v2","Ringkasan untuk analisis | Notebook lengkap, QUICK=False")
    cardw=(CW-24)/4
    for i,(value,label) in enumerate(((f(cal['Macro-F1 (pooled)']),"Macro-F1 final*"),(f"{cal['AFL R']*100:.1f}%".replace('.',','),"Recall AFL final*"),(f"{cal['AFL P']*100:.1f}%".replace('.',','),"Precision AFL final*"),("23","Rekaman / grup"))):
        x=M+i*(cardw+8); c.setFillColor(HexColor("#EDF3F8"));c.roundRect(x,y-88,cardw,88,6,fill=1,stroke=0)
        c.setFillColor(HexColor(TEAL));c.setFont("Arial-Bold",24);c.drawString(x+11,y-37,value)
        c.setFillColor(HexColor(GRAY));c.setFont("Arial",8.7);c.drawString(x+11,y-61,label)
    y-=105
    y=para("*Metrik OOF model MicroTransformer pada sumber detektor setelah bias cross-fit. Ini <b>bukan akurasi dari pengujian perangkat independen</b>; hasil penyesuaian bias perlu audit evaluasi nested (hal. 5).",y,"small")
    y=callout("Hasil utama","Eksekusi berhasil: 20 sel kode, 30 epoch per training DL, 5 fold berdasarkan rekaman. Model final dan predictor sudah diekspor serta lolos uji paritas. Performa AFL tetap menjadi titik analisis utama.",y)
    y=section("Empat temuan yang perlu dibaca bersama",y)
    for title,txt in [
        ("1. Model terbaik bergantung protokol.","Pada anotasi tanpa bias, ResNet tertinggi (Macro-F1 0,764). Setelah bias, BiGRU tertinggi (0,842; eksploratif). Model final tetap Transformer sesuai konfigurasi awal, bukan pilihan otomatis dari ranking."),
        ("2. Training sesuai detektor belum memulihkan skor mentah.","Macro-F1 Transformer: 0,681 pada S1, 0,667 pada S2, dan 0,657 pada S3. Jadi dugaan pemulihan domain tidak didukung hasil argmax pada run ini."),
        ("3. AFL sedikit dan tidak merata.","Hanya 991 window AFL pada anotasi (0,606%). Sebanyak 510 berasal dari satu rekaman; performa per pasien berbeda jauh."),
        ("4. Uji alur belum menunjukkan ketepatan klasifikasi yang konsisten.","Contoh 60 detik berlabel Normal diprediksi AFIB; contoh berlabel AFL diprediksi Normal. Uji perangkat asli dilewati karena raw_71.json tidak tersedia.")]:
        y=para(f"<b>{title}</b> {txt}",y)
    y=para("Panduan baca: data & fold (2), benchmark (3), AFL per pasien (4), bias kelas (5), domain (6), detektor (7), efisiensi (8), model final (9), batas interpretasi & sumber (10).",y,"small")

    y=page("Data dan protokol evaluasi","Window 10 detik | Stride seragam 5 detik | Dua sumber R-peak")
    counts=[("Anotasi",97183,65452,991,163626),("Detektor 50 Hz",99000,65826,1152,165978)]
    y=table(["Sumber","Normal","AFIB","AFL","Total","Proporsi AFL"],[[a,n(b),n(cc),n(dd),n(e),f"{dd/e*100:.3f}%".replace('.',',')] for a,b,cc,dd,e in counts],y,[110,80,80,65,85,CW-420])
    y=para("Kedua dataset memuat 23 rekaman. Rekaman 00735 dan 03665 dikecualikan sesuai blacklist. Jumlah window berbeda karena kualitas IBI dan hasil deteksi berbeda; satu window bukan satu pasien independen.",y)
    y=section("Pembagian pasien/rekaman pada sumber anotasi",y)
    y=table(["Fold","Grup","Grup AFL","Rekaman AFL","Normal","AFIB","AFL"],[[r['fold'],r['pasien'],r['pasien_AFL'],f"{r['rekaman_AFL']:05d}" if isinstance(r['rekaman_AFL'],int) else r['rekaman_AFL'],n(r['N']),n(r['AFIB']),n(r['AFL'])] for r in d['14']],y,[40,44,56,145,80,80,CW-445])
    y=callout("Mengapa pooled OOF berbeda dari rata-rata fold?","Pooled OOF menghitung metrik setelah seluruh prediksi uji digabung. Rata-rata fold memberi bobot sama pada setiap fold, meskipun jumlah AFL berbeda (22 sampai 519 window). Kedua angka menjawab pertanyaan yang berbeda.",y)
    y=section("Prosedur yang dijalankan",y)
    for txt in [
        "<b>Split grup:</b> StratifiedGroupKFold (5, shuffle=True, seed=42) dibentuk dari anotasi; pemetaan rekaman yang sama digunakan pada dataset detektor.",
        "<b>Cabang tabular:</b> 16 fitur HRV. A1 menggunakan class_weight='balanced'; A2 memakai StandardScaler + SMOTE + LightGBM di dalam train fold.",
        "<b>Cabang DL:</b> IBI dan delta-IBI, maksimum 30 beat, mask padding. Mean/std dihitung hanya dari train fold; B1 weighted loss dan B2 sampler untuk Transformer.",
        "<b>Training:</b> 30 epoch, batch 256, AdamW, learning rate 0,001, weight decay 0,0001; scheduler cosine, tanpa early stopping. Model final dilatih pada semua data detektor."]:
        y=para(txt,y)
    y=para("Fold berdasarkan ID rekaman sebagaimana kode notebook; identitas pasien di luar ID tersebut tidak diaudit ulang dalam laporan ini. Window overlap 50% berada dalam grup yang sama.",y,"small")

    y=page("Benchmark model utama","Sumber anotasi | Pooled out-of-fold | Argmax tanpa penyesuaian bias")
    y=table(["Model","Macro-F1","N R","AFIB P","AFIB R","AFL P","AFL R","AFL F1","s/fold"],[[ALIASES[i]]+[f(r[k]) for k in ['Macro-F1 (pooled)','Normal R','AFIB P','AFIB R','AFL P','AFL R','AFL F1']]+[str(r['Train/fold (s)']).replace('.',',')] for i,r in enumerate(d['18'])],y,[135]+[(CW-135)/8]*8)
    y=para("P = precision; R = recall. A1: balanced class weight. A2: SMOTE in-fold. B1: weighted loss. B2: WeightedRandomSampler. Transf. = MicroTransformer, ResNet = MaskedResNet1D, BiGRU = BiGRUAttention.",y,"small")
    y=chart("main_f1",y,225)
    y=callout("Interpretasi","ResNet memiliki Macro-F1 dan AFL F1 tertinggi tanpa bias (0,764 dan 0,390). Transformer mempunyai recall AFL tertinggi (0,809), tetapi precision hanya 0,102: sekitar 89,8% prediksi AFL tidak sesuai label AFL pada evaluasi ini.",y)
    y=para("SMOTE menurunkan Macro-F1 LightGBM dari 0,753 menjadi 0,744, walau recall AFL naik dari 0,394 ke 0,417. Sampler menaikkan Macro-F1 Transformer dari 0,681 ke 0,704, namun recall AFL turun dari 0,809 ke 0,770. Selisih ini berasal dari satu run; signifikansi statistik tidak dihitung.",y)

    y=page("AFL: lihat pasien, bukan hanya pooled","Recall AFL per rekaman | Sumber anotasi | Argmax")
    y=chart("afl_patients",y,335)
    y=para("Warna dan angka menunjukkan proporsi window AFL yang diprediksi AFL untuk setiap rekaman. n adalah jumlah window AFL; angka dibaca dari tabel notebook yang dibulatkan dua desimal.",y,"small")
    y=callout("Konsentrasi sampel","Rekaman 04936 menyumbang 510 dari 991 window AFL (51,5%); bersama 08378 (307 window), keduanya menyumbang 82,4%. Maka skor pooled terutama mencerminkan dua rekaman tersebut.",y)
    y=section("Apa yang tampak dari heatmap",y)
    y=para("<b>Generalisasi tidak seragam.</b> Pada 07910, recall terbaik hanya 0,25 (BiGRU); pada 08215, maksimum 0,22. ResNet yang terbaik secara pooled tetap memiliki recall 0,05 pada 07910 dan 0,00 pada 08215.",y)
    y=para("<b>Nilai 1,00 belum tentu kuat.</b> Rekaman 04043 hanya memiliki satu window AFL. Angka 1,00 di baris itu tidak setara kekuatan bukti dengan ratusan window, terlebih window saling overlap.",y)
    y=para("<b>Analisis lanjutan:</b> telaah sinyal dan anotasi pada 07910, 08215, dan 06426; bandingkan pola interval dan kualitas deteksi antarrekaman. Interval ketidakpastian sebaiknya dihitung dengan unit resampling pasien/rekaman, bukan window individual.",y)

    y=page("Penyesuaian bias kelas","Sumber anotasi | Sebelum dan sesudah bias cross-fit")
    y=table(["Model","F1 awal","F1 bias","Delta F1","AFL P bias","AFL R bias"],[[ALIASES[i],f(r['Macro-F1 sebelum']),f(r['Macro-F1 sesudah']),f(r['Macro-F1 sesudah']-r['Macro-F1 sebelum']),f(r['AFL P sesudah']),f(r['AFL R sesudah'])] for i,r in enumerate(d['21'])],y,[145]+[(CW-145)/5]*5)
    y=chart("calibration",y,205)
    y=para("BiGRU mencapai Macro-F1 0,842 dan Transformer 0,838 setelah bias pada sumber anotasi. Peningkatan biasanya dibayar dengan turunnya recall AFL, sementara precision naik. Angka ini tidak dapat dibandingkan langsung dengan model final pada sumber detektor.",y)
    y=callout("Catatan evaluasi: hasil bias bersifat eksploratif","Kode mempelajari bias fold k dari OOF fold lain. Tetapi model pembentuk OOF fold lain dapat pernah melatih data pasien fold k. Karena itu pipeline bias bukan nested patient-wise CV yang sepenuhnya terisolasi. Potensi pengaruh tidak langsung ini merupakan analisis terhadap kode, bukan error eksekusi. Evaluasi ulang dengan inner-OOF dari outer-train atau holdout pasien diperlukan untuk klaim generalisasi pipeline. [M1]",y,tint="#FFF4E8")
    y=para("<b>Bias kelas tidak membuktikan confidence terkalibrasi.</b> Bias dioptimalkan untuk Macro-F1, bukan untuk kecocokan probabilitas dengan frekuensi aktual. Notebook tidak melaporkan Brier score, log loss, ECE, atau reliability diagram. Nilai confidence predictor perlu ditafsirkan sebagai skor model yang sudah disesuaikan. [M2]",y,"small")

    y=page("Pergeseran domain: hasil yang terjadi","Argmax | Pasien uji sama antar-sumber | S1, S2, S3")
    tags=["S1 anotasi/anotasi","S2 anotasi/detektor","S3 detektor/detektor"]
    y=table(["Model dan skenario","Macro-F1","Normal R","AFL P","AFL R","AFL F1"],[[('LGBM ' if i<3 else 'Transf. ')+tags[i%3]]+[f(r[k]) for k in ['Macro-F1 (pooled)','Normal R','AFL P','AFL R','AFL F1']] for i,r in enumerate(d['23'])],y,[190]+[(CW-190)/5]*5)
    y=chart("domain_shift",y,205)
    y=callout("Training pada detektor belum memulihkan Macro-F1 mentah","LightGBM: 0,753 → 0,732 → 0,722. Transformer: 0,681 → 0,667 → 0,657. Dari S2 ke S3, skor justru turun masing-masing 0,010. Jadi hasil run ini tidak mendukung kesimpulan bahwa retraining detektor memulihkan performa argmax.",y)
    y=para("Recall AFL Transformer juga turun: 0,809 → 0,731 → 0,712. Selisih S1 dan S2 menunjukkan sensitivitas terhadap perubahan input, tetapi S2 dan S3 tetap rendah pada precision AFL (0,108 dan 0,089).",y)
    y=para("<b>Perbandingan yang belum ada:</b> bias cross-fit pada semua model/sumber detektor, serta ResNet/BiGRU pada sumber detektor. Maka 0,842 BiGRU anotasi bukan bukti bahwa BiGRU pasti terbaik untuk perangkat. Model final dipilih oleh cfg.final_model, bukan optimasi otomatis hasil benchmark.",y,"small")

    y=page("Audit detektor R-peak 50 Hz","Kecocokan terhadap anotasi | Toleransi 100 ms | 23 rekaman")
    stats=d['detector_stats']; mean_s=sum(r['sensitivitas'] for r in stats)/len(stats); mean_p=sum(r['PPV'] for r in stats)/len(stats)
    y=para(f"Rata-rata per rekaman: sensitivitas <b>{f(mean_s)}</b>, PPV <b>{f(mean_p)}</b>. Median antarrekaman dari galat median peak cocok: <b>8,2 ms</b>. Ringkasan ini bukan sensitivitas pooled seluruh detak.",y)
    y=chart("detector_quality",y,378)
    y=table(["Rekaman prioritas","Sensitivitas","PPV","Galat median (ms)"],[[r['record'],f(r['sensitivitas']),f(r['PPV']),f"{r['galat_median_ms']:.2f}".replace('.',',')] for r in stats[:3]],y,[145,100,100,CW-345])
    y=para("<b>07162 paling menyimpang:</b> sensitivitas 0,205 dan PPV 0,216. Ini menandakan kecocokan detektor/anotasi buruk, tetapi tidak menentukan penyebabnya. Perlu melihat lead, sinyal, timing anotasi, dan peak sebelum menyimpulkan kegagalan algoritme.",y)
    y=para("Keterbatasan implementasi: pasangan peak ditentukan dengan nearest-neighbor dua arah, bukan matching satu-ke-satu; banyak deteksi dapat cocok pada satu referensi. Galat median hanya dihitung untuk peak dalam toleransi, sehingga tidak merangkum seluruh kegagalan deteksi.",y,"small")

    y=page("Efisiensi komputasi","Training DL di CUDA | Latensi inferensi di CPU satu thread")
    lat=d['26']; names=['LightGBM','MicroTransformer','MaskedResNet1D','BiGRUAttention']; params=['-','8.739','107.011','7.172']
    y=table(["Model","Parameter DL","1 window (ms)","1.000 window (ms)"],[[names[i],params[i],f"{r['1 window (ms)']:.2f}".replace('.',','),f"{r['1.000 window (ms)']:.2f}".replace('.',',')] for i,r in enumerate(lat)],y,[155,105,120,CW-380])
    y=chart("latency",y,190)
    y=callout("Trade-off yang terlihat","Transformer tercepat untuk satu window (0,28 ms); BiGRU tercepat pada batch 1.000 window (17,86 ms). BiGRU juga paling lambat dilatih pada benchmark anotasi (178,2 s/fold), sekitar 2,46 kali waktu Transformer (72,4 s/fold).",y)
    y=section("Batas benchmark latensi",y)
    y=para("Angka median berasal dari perf_counter dengan warmup. CPU diatur satu thread. Jalur tabular mencakup ekstraksi fitur HRV; jalur DL mencakup pembentukan sekuens dan z-score. Keduanya mulai dari daftar IBI.",y)
    y=para("<b>Latensi bukan end-to-end dari EKG mentah.</b> Resampling, deteksi R-peak, pembentukan window, I/O, dan transport tidak termasuk. LightGBM memakai model full-fit; DL memakai arsitektur dengan bobot baru untuk timing. Karena itu ini benchmark biaya jalur komputasi, bukan pengukuran langsung predictor produksi yang diekspor.",y)
    y=para("Epoch: 30 per training DL. Model final: 88,9 detik. Eksekusi notebook yang berhasil tercatat 15:39:02 sampai 16:38:10 (59 menit 8 detik), termasuk menunggu unduhan; waktu setup dependensi dan percobaan awal berada di luar interval tersebut.",y,"small")

    y=page("Model final dan uji predictor","MicroTransformer | Data detektor | Artefak sihedaf_v2.pt + .json")
    final_rows=[[label,f(raw[key]),f(cal[key])] for label,key in [('Macro-F1 pooled','Macro-F1 (pooled)'),('Recall Normal','Normal R'),('Precision AFIB','AFIB P'),('Recall AFIB','AFIB R'),('Precision AFL','AFL P'),('Recall AFL','AFL R'),('F1 AFL','AFL F1')]]
    final_rows += [[name+'*',f(derived['argmax'][name]),f(derived['dengan_bias_crossfit'][name])] for name in ['Weighted-F1','Accuracy']]
    y=table(["Metrik OOF detektor","Argmax","Bias cross-fit"],final_rows,y,[CW-200,100,100])
    y=para("*Weighted-F1 dan accuracy direkonstruksi dari jumlah kelas dan P/R presisi penuh metadata, bukan output langsung notebook; Macro-F1 rekonstruksi cocok dengan metadata. Bias final log-probabilitas [0, +2, -3] di-fit pada seluruh OOF: skor relatif AFIB ×7,39 dan AFL ×0,050, lalu dinormalisasi. Kolom evaluasi bias memakai cross-fit, bukan bias full-OOF export.",y,"small")
    y=section("Uji alur EKG 60 detik dari data AFDB",y)
    y=table(["Label awal","Rekaman / t0","Prediksi","Confidence","Window valid"],[[r['label asli'],f"{int(r['rekaman']):05d} / {int(r['mulai (s)'])} s",'AFIB' if 'AFIB' in r['prediksi'] else r['prediksi'],f(r['confidence']),r['window valid']] for r in d['32']],y,[82,118,93,102,CW-395])
    y=para("Dua dari tiga contoh berbeda dari label awal: Normal → AFIB dan AFL → Normal. Ini bukan estimasi akurasi 33%: hanya tiga contoh, berasal dari data training, dan label awal diambil dari satu window sedangkan predictor merangkum segmen 60 detik yang bisa berisi beberapa ritme.",y,"small")
    y=callout("Verifikasi teknis yang lolos","Selisih probabilitas notebook vs predictor maksimum 1,25 × 10^-7 (&lt; 10^-4). IBI dalam detik ditolak; IBI terlalu pendek menghasilkan status tidak_terbaca. Checksum bobot cocok dengan metadata. Uji raw_71.json dilewati karena file tidak ada.",y)
    y=para("<b>Makna praktis:</b> alur ekspor dan inferensi berfungsi. Ketepatan kelas serta confidence masih perlu ditelaah melalui label konsisten per segmen, evaluasi pasien independen, dan data perangkat yang tersedia.",y)

    y=page("Batas interpretasi dan tindak lanjut","Hasil satu run | Angka tabel dibulatkan sesuai output sumber")
    y=section("Prioritas analisis berdasarkan temuan",y)
    for title,txt in [
        ("1. Audit contoh inferensi yang berbeda label.","Plot EKG/IBI, anotasi ritme sepanjang 60 detik, peak, serta prediksi tiap window pada Normal 04015 t=0 dan AFL 04043 t=30677. Pisahkan salah deteksi, transisi ritme, agregasi, dan keputusan bias."),
        ("2. Audit peak yang paling tidak cocok.","Mulai dari 07162, kemudian 07859 dan 06426. Cek keselarasan anotasi, lead, serta matching peak satu-ke-satu sebelum mengubah detektor."),
        ("3. Evaluasi pipeline bias dengan isolasi pasien lengkap.","Dalam setiap outer fold, buat inner-OOF hanya dari outer-train untuk mempelajari bias. Uji model dan bias hanya pada outer-test. Simpan OOF mentah, ID rekaman, dan fold."),
        ("4. Lengkapi perbandingan sumber produksi.","Jalankan ResNet dan BiGRU pada detektor dengan fold sama; bandingkan argmax dan bias dengan protokol evaluasi yang sama. Pemilihan model jangan hanya mengandalkan skor anotasi."),
        ("5. Lengkapi metrik dan bukti yang belum disimpan.","Weighted-F1 semua benchmark, confidence interval berbasis pasien, reliability diagram, dan evaluasi perangkat asli belum tersedia. Weighted-F1 final pada hal. 9 dapat direkonstruksi dari metadata presisi penuh; jangan memperlakukan matriks dibulatkan sebagai data mentah.")]:
        y=para(f"<b>{title}</b> {txt}",y)
    y=section("Sumber dan ketertelusuran",y)
    y=para("<b>[S1]</b> revisi_sihedaf_v2.ipynb: sel 10 (data), 12 (detektor), 14 (fold), 18-19 (benchmark/pasien), 21 (bias), 23 (domain), 26 (latensi), 28 dan 31-33 (final/uji). Seluruh 20 sel kode selesai tanpa output error.",y,"small")
    y=para("<b>[S2]</b> model_export/sihedaf_v2.json dan .pt; cache/dataset_v2_detector.stats.csv; .sihedaf_run/execution.log. Tabel benchmark dibaca dari HTML keluaran notebook; angka final memakai metadata presisi penuh. OOF mentah tidak diekspor. Weighted-F1/accuracy final dapat direkonstruksi dari total kelas, TP = recall × support, dan jumlah prediksi = TP / precision; prediksi Normal diperoleh dari sisa total.",y,"small")
    y=para(f"Python {meta['versions']['python']} | PyTorch {meta['versions']['torch']} | NumPy {meta['versions']['numpy']} | WSL Debian | GPU RTX 5060. Export: {meta['created']}. Commit yang direkam metadata: {meta['git_commit']}.",y,"small")
    y=para("SHA-256 bobot:<br/>"+meta['weights_sha256'][:32]+"<br/>"+meta['weights_sha256'][32:],y,"small")
    y=para('<b>[M1]</b> scikit-learn, Nested versus non-nested cross-validation. Dasar pemisahan tuning dan evaluasi. <link href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html" color="#2455A4">Buka dokumentasi resmi</link>.<br/><b>[M2]</b> scikit-learn, Probability calibration. Dasar membedakan skor kelas dan kualitas probabilitas. <link href="https://scikit-learn.org/stable/modules/calibration.html" color="#2455A4">Buka dokumentasi resmi</link>. Diakses 12 Oktober 2026.',y,"small")

    # Preserve notebook confusion matrices at readable size. The wide original
    # benchmark is viewed in two groups of three using PDF clipping, without
    # changing the source image or reconstructing rounded cell values.
    c.showPage(); page_num+=1; LW,LH=landscape(A4);c.setPageSize((LW,LH))
    c.setFillColor(HexColor(INK));c.setFont("Arial-Bold",19);c.drawString(40,LH-47,"Lampiran: confusion matrix benchmark")
    c.setFillColor(HexColor(GRAY));c.setFont("Arial",9);c.drawString(40,LH-67,"Sumber anotasi, argmax | Normalisasi per baris; diagonal = recall | Gambar asli notebook")
    p=TMP/"confusion_benchmark.png"; iw,ih=Image.open(p).size
    for group in range(2):
        boxw=LW-100; factor=boxw/(iw/2); crop_top=32
        boxh=(ih-crop_top)*factor; x=50; top=LH-89-group*233
        c.saveState();clip=c.beginPath();clip.rect(x,top-boxh,boxw,boxh);c.clipPath(clip,stroke=0)
        c.drawImage(str(p),x-group*boxw,top-(ih-crop_top)*factor,width=iw*factor,height=ih*factor,mask="auto");c.restoreState()
    c.setStrokeColor(HexColor("#D8E1E9"));c.line(40,43,LW-40,43);c.setFont("Arial",8)
    c.drawString(40,29,"Urutan atas: A1 LightGBM, A2 SMOTE, B1 Transformer | Bawah: B1 ResNet, B1 BiGRU, B2 Transformer")
    c.drawRightString(LW-40,29,f"{page_num:02d}")
    c.bookmarkPage(f"page{page_num}");c.addOutlineEntry("Lampiran: confusion matrix benchmark",f"page{page_num}",0,False)

    c.showPage();page_num+=1;c.setPageSize((LW,LH))
    c.setFillColor(HexColor(INK));c.setFont("Arial-Bold",19);c.drawString(40,LH-47,"Lampiran: confusion matrix pergeseran domain")
    c.setFillColor(HexColor(GRAY));c.setFont("Arial",9);c.drawString(40,LH-67,"MicroTransformer, argmax | S1 anotasi/anotasi, S2 anotasi/detektor, S3 detektor/detektor")
    p=TMP/"confusion_domain.png";iw,ih=Image.open(p).size;ww=LW-80;hh=ww*ih/iw
    c.drawImage(str(p),40,LH-100-hh,width=ww,height=hh,mask="auto")
    ptxt=Paragraph("Gambar asli notebook, angka dibulatkan dua desimal. Baris = aktual, kolom = prediksi; diagonal = recall. Ini bukan confusion matrix setelah bias. Untuk model final setelah bias, yang tersedia adalah metrik metadata pada halaman 9.",styles['body'])
    _,ph=ptxt.wrap(ww,100);ptxt.drawOn(c,40,LH-120-hh-ph)
    c.setStrokeColor(HexColor("#D8E1E9"));c.line(40,43,LW-40,43);c.setFont("Arial",8)
    c.drawString(40,29,"Gambar sumber: output sel 23, revisi_sihedaf_v2.ipynb");c.drawRightString(LW-40,29,f"{page_num:02d}")
    c.bookmarkPage(f"page{page_num}");c.addOutlineEntry("Lampiran: confusion matrix domain",f"page{page_num}",0,False)
    c.save()
    print(f"Created {OUT} ({page_num} pages)")


if __name__=="__main__":
    extract() if "--extract" in sys.argv else build()
