"""Convierte medium/article_medium.md -> medium/article_medium.html (autocontenido,
figuras incrustadas) para leerlo/previsualizarlo como se verá en Medium."""
import re, base64, os
STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"


def b64(name):
    # el artículo es en inglés: preferir siempre la figura _en si existe
    cands = []
    if name.endswith(".png"):
        en = name[:-4] + "_en.png"
        cands += [f"{STUDY}/{en}", f"{STUDY}/medium/assets/{en}"]
    cands += [f"{STUDY}/{name}", f"{STUDY}/medium/assets/{name}"]
    for p in cands:
        if os.path.exists(p):
            return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
    return None


def inline(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", t)
    return t


def convert(md):
    lines = md.split("\n")
    html, i, n = [], 0, len(lines)
    while i < n:
        ln = lines[i]
        s = ln.strip()
        # figura: *[FIGURE N — file.png]*  o  [FIGURE N — file.png]
        mfig = re.search(r"\[(?:FIGURE|TABLE)\s*\d+[a-z]?\s*[—-]\s*([\w.\-]+\.png)\]", s)
        if mfig:
            d = b64(mfig.group(1))
            if d:
                html.append(f'<figure><img src="{d}" alt="figure"/></figure>')
            i += 1; continue
        if not s:
            i += 1; continue
        if s.startswith("# "):
            html.append(f"<h1>{inline(s[2:])}</h1>"); i += 1; continue
        if s.startswith("## "):
            html.append(f"<h2>{inline(s[3:])}</h2>"); i += 1; continue
        if s.startswith("### "):
            html.append(f"<h3>{inline(s[4:])}</h3>"); i += 1; continue
        if s.startswith("---"):
            html.append("<hr/>"); i += 1; continue
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip()); i += 1
            html.append(f"<blockquote>{inline(' '.join(buf))}</blockquote>"); continue
        if s.startswith("|"):   # tabla
            buf = []
            while i < n and lines[i].strip().startswith("|"):
                buf.append(lines[i].strip()); i += 1
            rows = [r for r in buf if not re.match(r"^\|[\s|:-]+\|$", r)]
            trs = []
            for k, r in enumerate(rows):
                cells = [c.strip() for c in r.strip("|").split("|")]
                tag = "th" if k == 0 else "td"
                trs.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
            html.append("<div class='tw'><table>" + "".join(trs) + "</table></div>"); continue
        cont = lambda t: t and not re.match(r"^(#|>|\||-\s|\d+\.\s|---)", t) and not re.search(r"\[(FIGURE|TABLE)", t)
        if re.match(r"^\d+\.\s", s):   # lista ordenada (con líneas de continuación)
            buf = []
            while i < n and re.match(r"^\d+\.\s", lines[i].strip()):
                item = re.sub(r"^\d+\.\s", "", lines[i].strip()); i += 1
                while i < n and cont(lines[i].strip()):
                    item += " " + lines[i].strip(); i += 1
                buf.append(item)
            html.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in buf) + "</ol>"); continue
        if s.startswith("- "):
            buf = []
            while i < n and lines[i].strip().startswith("- "):
                item = lines[i].strip()[2:]; i += 1
                while i < n and cont(lines[i].strip()):
                    item += " " + lines[i].strip(); i += 1
                buf.append(item)
            html.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in buf) + "</ul>"); continue
        # párrafo (juntar líneas hasta blanco)
        buf = [s]; i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#|>|\||-\s|\d+\.\s|---)", lines[i].strip()) \
                and not re.search(r"\[(FIGURE|TABLE)", lines[i]):
            buf.append(lines[i].strip()); i += 1
        para = " ".join(buf)
        cls = ' class="dek"' if para.startswith("*") and para.endswith("*") and "**" not in para else ""
        html.append(f"<p{cls}>{inline(para)}</p>")
    return "\n".join(html)


import sys
_inp = sys.argv[1] if len(sys.argv) > 1 else f"{STUDY}/medium/article_medium.md"
_out = sys.argv[2] if len(sys.argv) > 2 else f"{STUDY}/medium/article_medium.html"
md = open(_inp).read()
CSS = open(f"{STUDY}/_article_css.html").read()
body = "<article>" + convert(md) + "</article>"
open(_out, "w", encoding="utf-8").write(
    '<meta charset="utf-8">\n' + CSS + body)
print(_out, "escrito")
