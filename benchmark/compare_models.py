"""
LastSeen — detector comparison on the shared image set.

Runs every candidate detector on the same images, maps detections to a fixed
list of 16 query items (8 inside COCO, 8 outside), and scores them against the
hand-labeled image-level ground truth in ground_truth.json.

Usage (from the repo root):
    python benchmark/compare_models.py                 # run all models that can load
    python benchmark/compare_models.py --models yolo11s rtdetr
    python benchmark/compare_models.py --report-only   # rebuild tables from cached raw results

Raw detections are cached in benchmark/results/raw/<model>.json, so models can be
run on different machines (e.g. some here, some in Colab) and the report merges them.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"
IMAGE_DIR = ROOT / "images"
RAW_DIR = BENCH / "results" / "raw"
VIS_DIR = BENCH / "results" / "vis"
GT = json.load(open(BENCH / "ground_truth.json"))
ITEMS = GT["items"]
LABELS = GT["labels"]
LOW_CONF = 0.01  # collect low-score detections so AP can sweep thresholds

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "yolo11s": dict(display="YOLO11s (baseline)", arch="CNN, one-stage", open_vocab=False,
                    weights="yolo11s.pt", default_conf=0.25),
    "fasterrcnn": dict(display="Faster R-CNN R50-FPN v2", arch="CNN, two-stage", open_vocab=False,
                       weights="torchvision DEFAULT", default_conf=0.5),
    "rtdetr": dict(display="RT-DETR-L", arch="Transformer (DETR)", open_vocab=False,
                   weights="rtdetr-l.pt", default_conf=0.25),
    "yoloworld": dict(display="YOLO-World v2-S", arch="CNN + CLIP text encoder", open_vocab=True,
                      weights="yolov8s-worldv2.pt", default_conf=0.25),
    "yoloe": dict(display="YOLOE-11S", arch="CNN + MobileCLIP text encoder", open_vocab=True,
                  weights="yoloe-11s-seg.pt", default_conf=0.25),
    "yoloe_1280": dict(display="YOLOE-11S @1280px", arch="CNN + MobileCLIP text encoder", open_vocab=True,
                       weights="yoloe-11s-seg.pt", default_conf=0.25, imgsz=1280),
    "yoloe_l": dict(display="YOLOE-11L", arch="CNN + MobileCLIP text encoder", open_vocab=True,
                    weights="yoloe-11l-seg.pt", default_conf=0.25),
    "owlv2": dict(display="OWLv2 B/16", arch="ViT (Transformer)", open_vocab=True,
                  weights="google/owlv2-base-patch16-ensemble", default_conf=0.1),
}


def item_names():
    return list(ITEMS.keys())


def prompts():
    return [ITEMS[i]["prompt"] for i in item_names()]


def coco_to_item():
    m = {}
    for item, cfg in ITEMS.items():
        for c in cfg["coco_classes"]:
            m[c] = item
    return m


def n_params(module):
    return int(sum(p.numel() for p in module.parameters()))


def weight_path(name):
    """Look for weights in repo root, benchmark/weights, then let the library download."""
    for d in (ROOT, BENCH / "weights"):
        p = d / name
        if p.exists():
            return str(p)
    return name


# ---------------------------------------------------------------------------
# Loaders: each returns (predict_fn, n_params). predict_fn(rgb_np) -> list of
# (item_name, score, [x1,y1,x2,y2]) in original image pixels.
# ---------------------------------------------------------------------------
def load_ultralytics_closed(key):
    from ultralytics import YOLO, RTDETR
    cls = RTDETR if key == "rtdetr" else YOLO
    model = cls(weight_path(MODELS[key]["weights"]))
    imgsz = MODELS[key].get("imgsz", 640)
    mapping = coco_to_item()
    names = model.names

    def predict(rgb):
        r = model(rgb[:, :, ::-1], conf=LOW_CONF, imgsz=imgsz, verbose=False, device="cpu")[0]
        out = []
        for b in r.boxes:
            label = names[int(b.cls[0])]
            if label in mapping:
                out.append((mapping[label], float(b.conf[0]), [float(x) for x in b.xyxy[0]]))
        return out, r

    return predict, n_params(model.model)


def load_ultralytics_open(key):
    from ultralytics import YOLO
    imgsz = MODELS[key].get("imgsz", 640)
    if key.startswith("yoloe"):
        from ultralytics import YOLOE
        model = YOLOE(weight_path(MODELS[key]["weights"]))
        p = prompts()
        try:
            model.set_classes(p, model.get_text_pe(p))
        except TypeError:
            model.set_classes(p)
    else:
        model = YOLO(weight_path(MODELS[key]["weights"]))
        model.set_classes(prompts())
    names = item_names()

    def predict(rgb):
        r = model(rgb[:, :, ::-1], conf=LOW_CONF, imgsz=imgsz, verbose=False, device="cpu")[0]
        out = [(names[int(b.cls[0])], float(b.conf[0]), [float(x) for x in b.xyxy[0]]) for b in r.boxes]
        return out, r

    return predict, n_params(model.model)


def load_fasterrcnn(key):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=LOW_CONF).eval()
    cats = weights.meta["categories"]
    mapping = coco_to_item()
    tf = weights.transforms()

    def predict(rgb):
        x = tf(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
        with torch.no_grad():
            o = model([x])[0]
        out = []
        for box, lab, s in zip(o["boxes"], o["labels"], o["scores"]):
            label = cats[int(lab)]
            if label in mapping:
                out.append((mapping[label], float(s), [float(v) for v in box]))
        return out, None

    return predict, n_params(model)


def load_owlv2(key):
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    mid = MODELS[key]["weights"]
    processor = Owlv2Processor.from_pretrained(mid)
    model = Owlv2ForObjectDetection.from_pretrained(mid).eval()
    names = item_names()
    texts = [[f"a photo of a {p}" for p in prompts()]]

    def predict(rgb):
        h, w = rgb.shape[:2]
        inputs = processor(text=texts, images=Image.fromarray(rgb), return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        side = max(h, w)  # OWLv2 pads to a square; boxes are relative to the padded image
        sizes = torch.tensor([[side, side]])
        if hasattr(processor, "post_process_grounded_object_detection"):  # transformers >= 4.48
            res = processor.post_process_grounded_object_detection(
                outputs, threshold=LOW_CONF, target_sizes=sizes)[0]
        else:
            res = processor.post_process_object_detection(
                outputs, threshold=LOW_CONF, target_sizes=sizes)[0]
        out = [(names[int(l)], float(s), [float(v) for v in b])
               for b, s, l in zip(res["boxes"], res["scores"], res["labels"])]
        return out, None

    return predict, n_params(model)


LOADERS = {
    "yolo11s": load_ultralytics_closed,
    "rtdetr": load_ultralytics_closed,
    "fasterrcnn": load_fasterrcnn,
    "yoloworld": load_ultralytics_open,
    "yoloe": load_ultralytics_open,
    "yoloe_1280": load_ultralytics_open,
    "yoloe_l": load_ultralytics_open,
    "owlv2": load_owlv2,
}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def load_images():
    files = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return {p.name: np.array(Image.open(p).convert("RGB")) for p in files}


def draw(rgb, dets, conf, path):
    from PIL import ImageDraw, ImageFont
    im = Image.fromarray(rgb)
    scale = 1600 / max(im.size)
    im = im.resize((int(im.width * scale), int(im.height * scale)))
    d = ImageDraw.Draw(im)
    try:
        import matplotlib.font_manager as fm
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold")), 22)
    except Exception:
        font = ImageFont.load_default()
    for item, s, b in dets:
        if s < conf:
            continue
        x1, y1, x2, y2 = [v * scale for v in b]
        color = (46, 160, 67) if ITEMS[item]["coco"] else (219, 68, 55)
        d.rectangle([x1, y1, x2, y2], outline=color, width=4)
        d.text((x1 + 4, max(0, y1 - 26)), f"{item} {s:.2f}", fill=color, font=font)
    im.save(path, quality=85)


def run_model(key, images):
    import torch
    torch.set_grad_enabled(False)
    print(f"\n=== {key}: loading")
    predict, params = LOADERS[key](key)
    first = next(iter(images.values()))
    predict(first)  # warm-up
    per_image, times = {}, []
    (VIS_DIR / key).mkdir(parents=True, exist_ok=True)
    for name, rgb in images.items():
        t0 = time.perf_counter()
        dets, _ = predict(rgb)
        times.append(time.perf_counter() - t0)
        per_image[name] = [dict(item=i, score=s, box=b) for i, s, b in dets]
        draw(rgb, dets, MODELS[key]["default_conf"], VIS_DIR / key / name)
        print(f"  {name}: {len(dets)} raw dets, {times[-1]*1000:.0f} ms")
    rec = dict(model=key, params=params, ms_per_image=float(np.mean(times) * 1000),
               device=f"CPU ({os.cpu_count()} threads)", torch_threads=torch.get_num_threads(),
               detections=per_image)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(RAW_DIR / f"{key}.json", "w"))
    return rec


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
VERIFIED = json.load(open(BENCH / "verified_boxes.json"))["boxes"]
MATCH_IOU = 0.5


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def entries_for_item(rec, item, needs_review):
    """Per image: best correctly-localized score (TP entry) and best wrong score (FP entry).

    A detection is correct only on an image where the item is present AND its box has
    IoU >= 0.5 with a hand-verified correct box. Everything else is a false positive.
    """
    entries, n_pos = [], 0
    for name, labels in LABELS.items():
        lab = labels[item]
        if lab == -1:
            continue
        n_pos += lab
        ver = VERIFIED.get(name, {}).get(item, {"correct": [], "wrong": []})
        best_tp = best_fp = 0.0
        for d in rec["detections"].get(name, []):
            if d["item"] != item:
                continue
            ok = lab == 1 and any(iou(d["box"], b) >= MATCH_IOU for b in ver["correct"])
            if ok:
                best_tp = max(best_tp, d["score"])
            else:
                best_fp = max(best_fp, d["score"])
                if lab == 1 and d["score"] >= 0.1 and not any(
                        iou(d["box"], b) >= MATCH_IOU for b in ver["wrong"]):
                    needs_review.append((rec["model"], name, item, round(d["score"], 2),
                                         [round(v) for v in d["box"]]))
        if best_tp > 0:
            entries.append((best_tp, True))
        if best_fp > 0:
            entries.append((best_fp, False))
    return entries, n_pos


def average_precision(entries, n_pos):
    if n_pos == 0:
        return float("nan")
    tp = fp = 0
    ap = 0.0
    for _, is_tp in sorted(entries, key=lambda e: -e[0]):
        if is_tp:
            tp += 1
            ap += tp / (tp + fp)
        else:
            fp += 1
    return ap / n_pos


def score_model(rec):
    conf = MODELS[rec["model"]]["default_conf"]
    per_item = {}
    TP = FP = FN = 0
    group = {"coco": [0, 0], "non": [0, 0]}  # [tp, positives]
    needs_review = []
    for item, cfg in ITEMS.items():
        entries, pos = entries_for_item(rec, item, needs_review)
        ap = average_precision(entries, pos)
        tp = sum(1 for sc, t in entries if t and sc >= conf)
        fp = sum(1 for sc, t in entries if not t and sc >= conf)
        TP, FP, FN = TP + tp, FP + fp, FN + pos - tp
        g = "coco" if cfg["coco"] else "non"
        group[g][0] += tp
        group[g][1] += pos
        per_item[item] = dict(ap=ap, recall=tp / pos if pos else float("nan"), tp=tp, fp=fp, pos=pos)
    if needs_review:
        print(f"\n[{rec['model']}] {len(needs_review)} detection(s) on positive images were never hand-checked "
              "and are counted as wrong. Inspect them and add to verified_boxes.json if correct:")
        for r in needs_review:
            print("   ", r)
    coco_items = [i for i in ITEMS if ITEMS[i]["coco"]]
    non_items = [i for i in ITEMS if not ITEMS[i]["coco"]]
    n_dets = sum(1 for dets in rec["detections"].values() for d in dets if d["score"] >= conf)
    summary = dict(
        mAP_all=float(np.nanmean([per_item[i]["ap"] for i in ITEMS])),
        mAP_coco=float(np.nanmean([per_item[i]["ap"] for i in coco_items])),
        mAP_non=float(np.nanmean([per_item[i]["ap"] for i in non_items])),
        recall_coco=group["coco"][0] / group["coco"][1],
        recall_non=group["non"][0] / group["non"][1],
        recall_all=TP / (TP + FN),
        precision=TP / (TP + FP) if TP + FP else float("nan"),
        items_found=sum(1 for i in ITEMS if per_item[i]["tp"] > 0),
        n_dets=n_dets, conf=conf,
    )
    return summary, per_item


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report():
    recs = {}
    for key in MODELS:
        p = RAW_DIR / f"{key}.json"
        if p.exists():
            recs[key] = json.load(open(p))
    if not recs:
        print("No results yet.")
        return
    rows, item_rows = [], []
    for key, rec in recs.items():
        s, per_item = score_model(rec)
        m = MODELS[key]
        rows.append({
            "Model": m["display"], "Architecture": m["arch"],
            "Open-vocab": "yes" if m["open_vocab"] else "no",
            "Params (M)": round(rec["params"] / 1e6, 1),
            "CPU ms/img": round(rec["ms_per_image"]),
            "mAP (all 16)": round(s["mAP_all"], 2),
            "mAP COCO (8)": round(s["mAP_coco"], 2),
            "mAP non-COCO (8)": round(s["mAP_non"], 2),
            "Recall COCO": round(s["recall_coco"], 2),
            "Recall non-COCO": round(s["recall_non"], 2),
            "Precision": round(s["precision"], 2),
            "Items found (/16)": s["items_found"],
            "Conf thr": s["conf"],
        })
        for item, v in per_item.items():
            item_rows.append({"Model": m["display"], "Item": item,
                              "COCO": "yes" if ITEMS[item]["coco"] else "no",
                              "Positives": v["pos"], "AP": round(v["ap"], 2),
                              "Recall": round(v["recall"], 2), "FP images": v["fp"]})
    import pandas as pd
    df = pd.DataFrame(rows)
    di = pd.DataFrame(item_rows)
    out = BENCH / "results"
    df.to_csv(out / "summary.csv", index=False)
    di.to_csv(out / "per_item.csv", index=False)
    pivot = di.pivot(index="Item", columns="Model", values="AP").reindex(list(ITEMS.keys()))
    pivot = pivot[[MODELS[k]["display"] for k in recs]]
    pivot.to_csv(out / "per_item_ap.csv")
    print("\n" + df.to_string(index=False))
    print("\nPer-item AP\n" + pivot.to_string())
    return df, di, pivot, recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="don't re-run models that already have cached raw results")
    args = ap.parse_args()
    if not args.report_only:
        images = load_images()
        for key in args.models:
            if args.skip_existing and (RAW_DIR / f"{key}.json").exists():
                print(f"=== {key}: cached, skipping")
                continue
            try:
                run_model(key, images)
            except Exception as e:  # keep going if one model can't load (e.g. no internet for weights)
                print(f"!! {key} failed: {type(e).__name__}: {e}")
    build_report()
