"""Render the gradient worked-example table as a PNG for Medium (no table support)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rows=[["A","0.92","fraud (1)","-0.08","0"],
      ["B","0.85","legit (0)","+0.85","0"],
      ["C","0.80","legit (0)","+0.80","0"],
      ["D","0.78","fraud (1)","-0.22","0"],
      ["E","0.60","fraud (1)","-0.40","0"],
      ["F","0.55","legit (0)","+0.55","0"],
      ["G","0.30","fraud (1)","-0.70","0"],
      ["H","0.10","legit (0)","+0.10","0"]]
cols=["item","score p","label y","log-loss grad. (p - y)","precision@3 grad."]
RED="#C0392B"; INK="#161A20"
fig,ax=plt.subplots(figsize=(9.6,3.4)); ax.axis("off")
t=ax.table(cellText=rows,colLabels=cols,cellLoc="center",loc="center")
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1,1.55)
for (r,cc),cell in t.get_celld().items():
    cell.set_edgecolor("#DDE1E7")
    if r==0:
        cell.set_facecolor("#F2F4F7"); cell.set_text_props(weight="bold",color=INK)
    else:
        lab=rows[r-1]
        if lab[2].startswith("fraud"): cell.set_facecolor("#FBEEEC" if cc in (0,2) else "#FFFFFF")
        # emphasise the two big false-positive gradients B,C in the gradient column
        if cc==3 and rows[r-1][0] in ("B","C"):
            cell.set_text_props(weight="bold",color=RED)
fig.tight_layout()
fig.savefig("/home/agomez/proyectos/precision-at-k-study/table_gradient_en.png",dpi=150,bbox_inches="tight")
print("saved table_gradient_en.png")
