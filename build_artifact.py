"""Convert article.md -> self-contained styled HTML fragment for the Artifact.
Inlines the 5 PNG figures as base64 data URIs. Emits only <style> + content
(no doctype/html/head/body — the Artifact host wraps it)."""
import base64, re, markdown, pathlib, sys

OUT = pathlib.Path("/home/agomez/proyectos/precision-at-k-study")

LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
SRC = {"en": "article.md", "es": "article_es.md"}[LANG]
DST = {"en": "article.html", "es": "article_es.html"}[LANG]
EYEBROW = {"en": "Machine Learning &middot; Model Evaluation",
           "es": "Machine Learning &middot; Evaluaci&oacute;n de modelos"}[LANG]
BYLINE = {"en": "A study across 90 real + 3 synthetic imbalanced datasets",
          "es": "Un estudio sobre 90 datasets reales + 3 sint&eacute;ticos desbalanceados"}[LANG]

md_text = (OUT / SRC).read_text()

# markdown -> html (tables, fenced code, attr lists)
html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])

# inline images as data URIs
def datauri(fn):
    b = (OUT / fn).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()

def repl_img(m):
    src = m.group("src")
    alt = m.group("alt")
    return (f'<figure><img src="{datauri(src)}" alt="{alt}" loading="lazy">'
            f'<figcaption>{alt}</figcaption></figure>')

html = re.sub(r'<img alt="(?P<alt>[^"]*)" src="(?P<src>[^"]+\.png)"\s*/?>',
              repl_img, html)
# markdown may order attrs src then alt; handle both
html = re.sub(r'<img src="(?P<src>[^"]+\.png)" alt="(?P<alt>[^"]*)"\s*/?>',
              repl_img, html)

# wrap tables for horizontal scroll
html = re.sub(r'(<table>.*?</table>)', r'<div class="table-wrap">\1</div>',
              html, flags=re.S)

# The first <h1> becomes the masthead title; split it out with the following
# emphasized subtitle lines (### rendered as h3 right after h1).
STYLE = """
<style>
  :root{
    --paper:#F7F8FA; --ink:#161A20; --ink-soft:#3C4655; --muted:#6B7688;
    --rule:#E4E7EC; --card:#FFFFFF; --card-brd:#E6E9EF;
    --accent:#C0392B; --accent-soft:#F5E4E1; --good:#1E874B;
    --max:44rem;
    --serif:Georgia,"Iowan Old Style","Times New Roman",serif;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme:dark){
    :root{
      --paper:#0F1216; --ink:#E7EAF0; --ink-soft:#B7C0CE; --muted:#8A94A6;
      --rule:#242A33; --card:#F7F8FA; --card-brd:#20262E;
      --accent:#E86050; --accent-soft:#2A1C1A; --good:#4CC47D;
    }
  }
  :root[data-theme="light"]{
    --paper:#F7F8FA; --ink:#161A20; --ink-soft:#3C4655; --muted:#6B7688;
    --rule:#E4E7EC; --card:#FFFFFF; --card-brd:#E6E9EF;
    --accent:#C0392B; --accent-soft:#F5E4E1; --good:#1E874B;
  }
  :root[data-theme="dark"]{
    --paper:#0F1216; --ink:#E7EAF0; --ink-soft:#B7C0CE; --muted:#8A94A6;
    --rule:#242A33; --card:#F7F8FA; --card-brd:#20262E;
    --accent:#E86050; --accent-soft:#2A1C1A; --good:#4CC47D;
  }
  *{box-sizing:border-box}
  body{margin:0}
  .wrap{background:var(--paper);color:var(--ink);
    font-family:var(--sans);font-size:19px;line-height:1.68;
    -webkit-font-smoothing:antialiased;padding:0 1.25rem 6rem;}
  .col{max-width:var(--max);margin:0 auto}

  /* masthead */
  header.masthead{max-width:var(--max);margin:0 auto;padding:4.5rem 0 2rem;
    border-bottom:1px solid var(--rule)}
  .eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.18em;
    text-transform:uppercase;color:var(--accent);margin:0 0 1.4rem;
    display:flex;align-items:center;gap:.7rem}
  .eyebrow::before{content:"";width:2.2rem;height:2px;background:var(--accent)}
  header.masthead h1{font-family:var(--serif);font-weight:700;
    font-size:clamp(2.1rem,6vw,3.15rem);line-height:1.08;letter-spacing:-.01em;
    margin:0 0 1.1rem;text-wrap:balance;color:var(--ink)}
  .dek{font-family:var(--serif);font-style:italic;font-size:1.3rem;
    line-height:1.5;color:var(--ink-soft);margin:0 0 1.8rem;text-wrap:balance}
  .byline{font-family:var(--mono);font-size:.8rem;color:var(--muted);
    letter-spacing:.02em}

  article{max-width:var(--max);margin:0 auto}

  /* headings inside body */
  article h2{font-family:var(--serif);font-weight:700;
    font-size:clamp(1.5rem,3.6vw,2rem);line-height:1.18;letter-spacing:-.01em;
    margin:3.4rem 0 .4rem;padding-top:2.2rem;position:relative;text-wrap:balance}
  article h2::before{content:"";position:absolute;top:0;left:0;
    width:3rem;height:3px;background:var(--accent)}
  article h3{font-family:var(--sans);font-weight:700;font-size:1.16rem;
    letter-spacing:.005em;margin:2.4rem 0 .3rem;color:var(--ink)}
  article p{margin:1.05rem 0}
  article a{color:var(--accent);text-decoration:none;
    border-bottom:1px solid color-mix(in srgb,var(--accent) 40%,transparent)}
  article a:hover{border-bottom-color:var(--accent)}
  strong{font-weight:700;color:var(--ink)}
  em{font-style:italic}

  /* lists */
  article ul,article ol{margin:1.05rem 0;padding-left:1.3rem}
  article li{margin:.4rem 0;padding-left:.25rem}
  article li::marker{color:var(--accent)}

  /* blockquote = the central tension */
  blockquote{margin:2rem 0;padding:1.1rem 0 1.1rem 1.6rem;
    border-left:3px solid var(--accent);
    font-family:var(--serif);font-style:italic;font-size:1.28rem;line-height:1.5;
    color:var(--ink)}
  blockquote strong{font-style:normal}

  /* inline code + fenced */
  code{font-family:var(--mono);font-size:.86em;
    background:var(--accent-soft);color:var(--ink);
    padding:.08em .38em;border-radius:4px;
    border:1px solid color-mix(in srgb,var(--accent) 18%,transparent)}
  pre code{display:block;padding:1rem;overflow-x:auto;border-radius:8px}

  /* figures — always a light card so charts read in both themes */
  figure{margin:2.4rem 0;background:var(--card);
    border:1px solid var(--card-brd);border-radius:12px;padding:.8rem .8rem .2rem;
    box-shadow:0 1px 3px rgba(16,20,30,.06),0 8px 24px rgba(16,20,30,.05)}
  figure img{display:block;width:100%;height:auto;border-radius:6px}
  figcaption{font-family:var(--mono);font-size:.76rem;line-height:1.5;
    color:#5A6472;padding:.7rem .3rem .8rem;letter-spacing:.01em}

  /* table */
  .table-wrap{overflow-x:auto;margin:2rem 0;
    border:1px solid var(--rule);border-radius:10px}
  table{border-collapse:collapse;width:100%;font-size:.92rem;
    font-variant-numeric:tabular-nums}
  th,td{text-align:right;padding:.7rem .9rem;border-bottom:1px solid var(--rule)}
  th:first-child,td:first-child{text-align:left;font-family:var(--sans)}
  thead th{font-family:var(--mono);font-size:.72rem;letter-spacing:.06em;
    text-transform:uppercase;color:var(--muted);background:transparent}
  tbody tr:last-child td{border-bottom:none}

  /* equation */
  .eq{display:flex;align-items:center;justify-content:center;gap:1rem;
    flex-wrap:wrap;margin:2rem 0;font-family:var(--serif);font-size:1.25rem}
  .eq-eq{color:var(--muted)}
  .frac{display:inline-flex;flex-direction:column;text-align:center;
    vertical-align:middle}
  .frac-num{padding:0 .7rem .25rem}
  .frac-den{padding:.25rem .7rem 0;
    border-top:2px solid var(--ink)}
  .eq em{font-style:italic}

  hr{border:none;border-top:1px solid var(--rule);margin:3rem 0}

  article > p:last-of-type{margin-top:2.5rem;padding-top:1.4rem;
    border-top:1px solid var(--rule);font-size:.85rem;color:var(--muted);
    font-family:var(--mono);line-height:1.6}

  @media (max-width:560px){ .wrap{font-size:17.5px} }
  @media (prefers-reduced-motion:no-preference){
    header.masthead,article>*{animation:rise .6s cubic-bezier(.2,.7,.2,1) both}
    article>*:nth-child(n+6){animation:none}
    @keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
  }
</style>
"""

# Split masthead: first <h1> + immediately following <h3>(s) styled as dek.
m = re.search(r'<h1>(.*?)</h1>\s*<h3>(.*?)</h3>', html, flags=re.S)
if m:
    title = m.group(1).strip()
    dek = m.group(2).strip()
    masthead = (f'<header class="masthead">'
                f'<p class="eyebrow">{EYEBROW}</p>'
                f'<h1>{title}</h1>'
                f'<p class="dek">{dek}</p>'
                f'<p class="byline">{BYLINE}</p>'
                f'</header>')
    html = masthead + '<article>' + html[m.end():] + '</article>'
else:
    html = '<article>' + html + '</article>'

page = STYLE + '<div class="wrap">' + html + '</div>'
# Encode every non-ASCII char as an HTML numeric entity so the page renders
# correctly regardless of the charset the viewer applies (fixes em-dash/accents).
page = page.encode("ascii", "xmlcharrefreplace").decode("ascii")
(OUT / DST).write_text(page)
print(f"wrote {DST}", len(page), "bytes")
print("figures inlined:", page.count("data:image/png"))
