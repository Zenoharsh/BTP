"""
Word (.docx) version of paper/main.tex, for co-authors who edit in Word.

    python scripts/make_docx.py                  # -> paper/BTP_paper.docx

Uses pandoc (pip: pypandoc_binary) with the IEEE CSL style for numbered references, PNG renders of the
figure PDFs (pip: pymupdf), Times New Roman text, and a two-column body after the abstract like the
IEEE conference layout. The LaTeX file stays the master copy: regenerate the .docx after editing it.
"""
import io
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile

import pypandoc
import pymupdf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(PAPER, "BTP_paper.docx")
CSL_URL = "https://raw.githubusercontent.com/citation-style-language/styles/master/ieee.csl"

AUTHORS = [  # name, number, email, ORCID
    ("Harsh Raj", 1, "harsh.raj.ug23@nsut.ac.in", "0009-0000-3091-1946"),
    ("Saurav Kumar", 2, "saurav.kumar_ug23@nsut.ac.in", "0009-0003-2525-1316"),
    ("Rumeshi Mittal", 3, "rumeshi.mittal.ug23@nsut.ac.in", "0009-0001-0925-6542"),
    ("Vicky Suri", 4, "vickysuri@nsut.ac.in", None),
]
AFFIL = ["Instrumentation & Control Engineering Department", "Netaji Subhas University of Technology",
         "New Delhi, India-110078"]


def latex_body(tex):
    title = re.search(r"\\title\{(.*?)\}\s*\n", tex, re.S).group(1)
    body = tex[tex.index(r"\maketitle") + len(r"\maketitle"):tex.index(r"\bibliographystyle")]
    for a, b in [(r"\textminus", "\u2212"), (r"\texttimes{}", "\u00d7"), (r"\texttimes", "\u00d7")]:
        body = body.replace(a, b)
    body = re.sub(r"\\tbd\{([^}]*)\}", r"\1", body)
    # PNG renders, every figure one column wide (Word clips images wider than the column)
    body = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{figs/([^}]+)\.pdf\}",
                  r"\\includegraphics[width=3.35in]{figs/\1.png}", body)
    body = body.replace(r"\begin{IEEEkeywords}", r"\noindent\textbf{\textit{Index Terms}}---")
    body = body.replace(r"\end{IEEEkeywords}", "")
    return title, number_parts(body) + "\n\\section*{References}\n"


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def number_parts(body):
    """IEEE numbering written into the text (pandoc's docx writer does not number them):
    sections I, II, ..., subsections A, B, ..., captions 'Fig. n.' and 'TABLE n.' in order of appearance."""
    out, sec, sub = [], 0, 0
    for tok in re.split(r"(\\(?:sub)?section\{)", body):
        if tok == "\\section{":
            sec, sub = sec + 1, 0
            out.append(tok + ROMAN[sec - 1] + ". ")
        elif tok == "\\subsection{":
            sub += 1
            out.append(tok + chr(64 + sub) + ". ")
        else:
            out.append(tok)
    body = "".join(out)
    counts = {"figure": 0, "table": 0}

    def cap(m):
        kind = "table" if m.group(1).startswith("table") else "figure"
        counts[kind] += 1
        label = f"TABLE {ROMAN[counts[kind] - 1]}. " if kind == "table" else f"Fig. {counts[kind]}. "
        return m.group(0).replace("\\caption{", "\\caption{" + label, 1)

    return re.sub(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", cap, body, flags=re.S)


def render_figures(tex, figdir):
    os.makedirs(figdir, exist_ok=True)
    for name in sorted(set(re.findall(r"\{figs/([^}]+)\.pdf\}", tex))):
        doc = pymupdf.open(os.path.join(PAPER, "figs", name + ".pdf"))
        doc[0].get_pixmap(dpi=300).save(os.path.join(figdir, name + ".png"))


def reference_docx(path):
    """pandoc's default reference.docx with IEEE-like fonts: Times New Roman 10 pt body, 24 pt title."""
    tmp = path + ".src.docx"
    pypandoc.convert_text("x", "docx", format="markdown", outputfile=tmp)
    with zipfile.ZipFile(tmp) as z:
        files = {n: z.read(n) for n in z.namelist()}
    os.remove(tmp)
    st = files["word/styles.xml"].decode("utf-8")
    st = re.sub(r'w:ascii="[^"]*"', 'w:ascii="Times New Roman"', st)
    st = re.sub(r'w:hAnsi="[^"]*"', 'w:hAnsi="Times New Roman"', st)
    st = re.sub(r'w:eastAsia="[^"]*"', 'w:eastAsia="Times New Roman"', st)
    st = re.sub(r'w:cs="[^"]*"', 'w:cs="Times New Roman"', st)
    st = re.sub(r"w:asciiTheme=\"[^\"]*\"|w:hAnsiTheme=\"[^\"]*\"|w:eastAsiaTheme=\"[^\"]*\"|w:cstheme=\"[^\"]*\"", "", st)
    st = re.sub(r'(<w:docDefaults>.*?<w:sz w:val=")\d+(")', r"\g<1>20\2", st, flags=re.S)      # 10 pt
    st = re.sub(r'(<w:style [^>]*w:styleId="Title".*?<w:sz w:val=")\d+(")', r"\g<1>48\2", st, flags=re.S)

    def restyle(style_id, rpr_extra, ppr_extra=""):
        nonlocal st
        m = re.search(r'<w:style [^>]*w:styleId="' + style_id + r'".*?</w:style>', st, re.S)
        if not m:
            return
        blk = m.group(0)
        blk = re.sub(r"<w:color [^>]*/>", '<w:color w:val="000000"/>', blk)
        blk = re.sub(r'<w:sz w:val="\d+"\s*/>', '<w:sz w:val="20"/>', blk)
        blk = re.sub(r'<w:szCs w:val="\d+"\s*/>', '<w:szCs w:val="20"/>', blk)
        # schema order inside rPr: rFonts first, then b/i/smallCaps ...
        blk = re.sub(r"(<w:rPr>\s*(?:<w:rFonts[^>]*/>)?)", lambda m: m.group(1) + rpr_extra, blk, count=1)
        if ppr_extra:
            blk = blk.replace("<w:pPr>", "<w:pPr>" + ppr_extra, 1)
        st = st.replace(m.group(0), blk)

    # IEEE look: 10 pt black headings, sections in small caps and centred, subsections in italics
    restyle("Heading1", "<w:smallCaps/>", '<w:jc w:val="center"/>')
    restyle("Heading2", "<w:i/>")
    restyle("Abstract", "<w:b/>")          # IEEE abstracts are set in bold
    # smaller text inside tables (pandoc puts table cells in the "Compact" style)
    st = re.sub(r'(<w:style [^>]*w:styleId="Compact".*?)(</w:style>)',
                r'\1<w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>\2', st, count=1, flags=re.S)
    files["word/styles.xml"] = st.encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)


def two_column_body(docx_path):
    """Section break after the abstract/keywords: front matter in one column, body in two."""
    with zipfile.ZipFile(docx_path) as z:
        files = {n: z.read(n) for n in z.namelist()}
    doc = files["word/document.xml"].decode("utf-8")
    m = re.search(r"<w:p>(?:(?!<w:p>).)*?Introduction</w:t>", doc, re.S)
    if m:
        brk = ('<w:p><w:pPr><w:sectPr><w:type w:val="continuous"/>'
               '<w:pgSz w:w="12240" w:h="15840"/>'
               '<w:pgMar w:top="1080" w:right="1022" w:bottom="1440" w:left="1022" w:header="720" '
               'w:footer="720" w:gutter="0"/><w:cols w:space="360"/></w:sectPr></w:pPr></w:p>')
        doc = doc[:m.start()] + brk + doc[m.start():]
    final = ('<w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="12240" w:h="15840"/>'
             '<w:pgMar w:top="1080" w:right="1022" w:bottom="1440" w:left="1022" w:header="720" '
             'w:footer="720" w:gutter="0"/><w:cols w:num="2" w:space="360"/></w:sectPr>')
    # replace only the document's LAST section properties (not the break inserted above)
    doc = re.sub(r"<w:sectPr\b(?:(?!<w:sectPr\b).)*?</w:sectPr>\s*</w:body>", final + "</w:body>", doc,
                 flags=re.S)
    files["word/document.xml"] = doc.encode("utf-8")
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)


def main():
    tex = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    title, body = latex_body(tex)
    with tempfile.TemporaryDirectory() as d:
        render_figures(tex, os.path.join(d, "figs"))
        csl = os.path.join(d, "ieee.csl")
        with urllib.request.urlopen(CSL_URL, timeout=30) as r, open(csl, "wb") as f:
            f.write(r.read())
        shutil.copy(os.path.join(PAPER, "refs.bib"), os.path.join(d, "refs.bib"))
        meta = os.path.join(d, "meta.yaml")
        with open(meta, "w", encoding="utf-8") as f:
            f.write("---\n" + f"title: \"{title}\"\nauthor:\n" + "\n".join(
                f"  - \"{n}{'\u00b9\u00b2\u00b3\u2074'[k - 1]}, " + ", ".join(AFFIL) + f", {e}"
                + (f", ORCID {o}" if o else "") + "\"" for n, k, e, o in AUTHORS) + "\n---\n")
        src = os.path.join(d, "paper.tex")
        with open(src, "w", encoding="utf-8") as f:
            f.write("\\documentclass{article}\n\\begin{document}\n" + body + "\n\\bibliography{refs}\n\\end{document}\n")
        ref = os.path.join(d, "reference.docx")
        reference_docx(ref)
        pypandoc.convert_file(src, "docx", format="latex", outputfile=OUT,
                              extra_args=["--citeproc", f"--bibliography={os.path.join(d, 'refs.bib')}",
                                          f"--csl={csl}", f"--metadata-file={meta}", f"--resource-path={d}",
                                          f"--reference-doc={ref}", "--number-sections=false"],
                              cworkdir=d)
    two_column_body(OUT)
    print("->", OUT)


if __name__ == "__main__":
    main()
