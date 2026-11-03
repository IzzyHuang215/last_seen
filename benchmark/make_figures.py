"""Figures for the model comparison (run after compare_models.py)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

BENCH = Path(__file__).resolve().parent
OUT = BENCH / "results"
GT = json.load(open(BENCH / "ground_truth.json"))
ITEMS = GT["items"]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK})

# --- 1. per-item AP heatmap -------------------------------------------------
ap = pd.read_csv(OUT / "per_item_ap.csv", index_col=0)
order = [i for i in ITEMS if ITEMS[i]["coco"]] + [i for i in ITEMS if not ITEMS[i]["coco"]]
ap = ap.loc[order]
cmap = LinearSegmentedColormap.from_list("blue", RAMP)
fig, ax = plt.subplots(figsize=(1.55 * ap.shape[1] + 2.6, 7.4), facecolor=SURF)
ax.set_facecolor(SURF)
ax.imshow(ap.values.astype(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
for (r, c), v in np.ndenumerate(ap.values):
    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=10,
            color="white" if v >= 0.6 else INK)
ax.set_xticks(range(ap.shape[1]), [c.replace(" (baseline)", "\n(baseline)").replace(" @", "\n@") for c in ap.columns], fontsize=10)
ax.set_yticks(range(len(order)), order, fontsize=10)
ax.xaxis.tick_top()
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-.5, ap.shape[1]), minor=True)
ax.set_yticks(np.arange(-.5, len(order)), minor=True)
ax.grid(which="minor", color=SURF, linewidth=2)
ax.axhline(7.5, color=INK, linewidth=1.2)
ax.text(-0.62, 3.5, "inside COCO", rotation=90, ha="right", va="center", fontsize=10, color=INK2,
        transform=ax.transData, clip_on=False)
ax.text(-0.62, 11.5, "outside COCO", rotation=90, ha="right", va="center", fontsize=10, color=INK2,
        transform=ax.transData, clip_on=False)
ax.yaxis.set_tick_params(pad=26)
fig.suptitle("Per-item AP, 9 desk photos (box location hand-checked)", fontsize=13, x=0.02, ha="left", y=0.995)
fig.text(0.02, 0.005, "Per image: best correctly-placed box = hit, best wrongly-placed box = false positive (IoU ≥ 0.5 with a hand-checked box). "
         "0.00 = never correctly detected.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig(OUT / "per_item_ap_heatmap.png", dpi=180, facecolor=SURF)
print("saved heatmap")

# --- 2. qualitative example on 9.jpg ----------------------------------------
def load_font(size):
    import matplotlib.font_manager as fm
    for name in ("DejaVuSans-Bold.ttf", fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def panel(model_key, title, thr, img="9.jpg", width=900):
    rec = json.load(open(OUT / "raw" / f"{model_key}.json"))
    im = Image.open(BENCH.parent / "images" / img).convert("RGB")
    s = width / im.width
    im = im.resize((width, int(im.height * s)))
    d = ImageDraw.Draw(im)
    f = load_font(22)
    for det in rec["detections"][img]:
        if det["score"] < thr:
            continue
        x1, y1, x2, y2 = [v * s for v in det["box"]]
        col = (37, 106, 191) if ITEMS[det["item"]]["coco"] else (196, 61, 33)
        d.rectangle([x1, y1, x2, y2], outline=col, width=4)
        t = f"{det['item']} {det['score']:.2f}"
        tw = d.textlength(t, font=f)
        d.rectangle([x1, max(0, y1 - 28), x1 + tw + 8, max(28, y1)], fill=col)
        d.text((x1 + 4, max(0, y1 - 27)), t, fill="white", font=f)
    return im, f"{title}  (conf ≥ {thr})"

panels = [panel("yolo11s", "YOLO11s (baseline)", 0.3), panel("yoloe_l", "YOLOE-11L", 0.3),
          panel("owlv2", "OWLv2", 0.3)]
fig, axs = plt.subplots(1, 3, figsize=(15, 7.4), facecolor=SURF)
for a, (im, t) in zip(axs, panels):
    a.imshow(im); a.set_title(t, fontsize=12, loc="left"); a.axis("off")
fig.suptitle("Our real photo, three detectors — blue = COCO item, red = non-COCO item", fontsize=13, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "example_9jpg.png", dpi=130, facecolor=SURF)
print("saved example")
