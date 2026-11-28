"""
Score the segmentation pipeline against hand-labelled ground truth
(a Roboflow "COCO Segmentation" export).

Two steps, so a slow pipeline run is done once and can be re-scored:

    # 1. run a version of segmentation.py on every labelled image
    python evaluate_segmentation.py predict GT_DIR --tag current
    python evaluate_segmentation.py predict GT_DIR --tag develop --seg path/to/old/segmentation.py

    # 2. score one or more runs
    python evaluate_segmentation.py score GT_DIR current develop

GT_DIR is one split of the export (the folder holding _annotations.coco.json
and its images). Predictions go to compare/eval/<tag>/.

How a prediction is paired with a ground-truth object: one-to-one, maximising
mask IoU (Hungarian), and a pair counts only at IoU >= 0.5. Labels play no
part in the pairing, so a found-but-misnamed piece is a detection with a
wrong label, not a miss plus a false positive. Reported:

  detection   precision / recall / F1 of the pairing (did it find the piece?)
  type acc.   share of pairs whose labels map to the same product-matching
              type (label_to_type: floor_lamp and table_lamp are both "lamp").
              This is the label that decides which products are searched.
  subclass    share of pairs with the same fine class (floor_lamp vs table_lamp)
  mask IoU    mean IoU of the pairs (how good the mask is, once found)
  usable      pairs with the right type, as precision and recall: a card that
              product matching can use, out of all cards / all real pieces
  box         the same detection score from boxes alone (what DINO is
              trained on), pairing by box IoU >= 0.5
"""

import argparse
import asyncio
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from dotenv import load_dotenv

# Like main.py, so the GEMINI_API_KEY naming step runs as in the app
# (SEGMENTATION_LABELER=clip to score the offline rule instead).
for _env_name in (".env", "backend.env"):
    load_dotenv(Path(__file__).with_name(_env_name))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.services.furniture_types import label_to_type  # noqa: E402

EVAL_DIR = Path("compare/eval")
IOU_MIN = 0.5

# Detection labels -> the fine class names used in the ground truth.
_SUBCLASS_ALIASES = {
    "couch": "sofa",
    "bedside table": "nightstand",
    "end table": "side_table",
    "console table": "side_table",
    "sideboard": "cabinet",
    "tv stand": "tv_bench",
    "media console": "tv_bench",
    "bookcase": "bookshelf",
    "chandelier": "pendant_lamp",
    "carpet": "rug",
    "area rug": "rug",
    "curtains": "curtain",
    "plant": "plant_pot",
    "potted plant": "plant_pot",
    "planter": "plant_pot",
    "painting": "artwork",
    "picture": "artwork",
    "framed picture": "artwork",
    "headboard": "bed",
    "dining chair": "chair",
    "office chair": "chair",
    "sconce": "wall_lamp",
}


def subclass(label: str) -> str:
    key = " ".join(label.lower().replace("_", " ").split())
    return _SUBCLASS_ALIASES.get(key, key.replace(" ", "_"))


def ptype(label: str) -> str | None:
    return label_to_type(label)


# ── Ground truth ────────────────────────────────────────────────────────────

def load_ground_truth(gt_dir: Path) -> list[dict]:
    coco = json.loads((gt_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    # Roboflow adds the project name as category 0, the parent of every class.
    names = {c["id"]: c["name"] for c in coco["categories"] if c.get("supercategory") != "none"}
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in names:
            by_image[ann["image_id"]].append(ann)

    images = []
    for im in coco["images"]:
        W, H = im["width"], im["height"]
        objects = []
        for ann in by_image[im["id"]]:
            canvas = Image.new("L", (W, H), 0)
            draw = ImageDraw.Draw(canvas)
            for poly in ann["segmentation"]:
                if len(poly) >= 6:
                    draw.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=255)
            x, y, w, h = ann["bbox"]
            objects.append({
                "label": names[ann["category_id"]],
                "mask": np.asarray(canvas) > 127,
                "box": (x, y, x + w, y + h),
            })
        images.append({"file": im["file_name"], "W": W, "H": H, "objects": objects})
    return images


# ── Step 1: predictions ─────────────────────────────────────────────────────

def load_segmentation(path: Path | None):
    if path is None:
        import segmentation
        return segmentation
    spec = importlib.util.spec_from_file_location("segmentation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_mask(data_url: str, W: int, H: int) -> np.ndarray:
    import base64
    raw = base64.b64decode(data_url.split(",", 1)[1])
    mask = Image.open(BytesIO(raw)).convert("L")
    if mask.size != (W, H):
        mask = mask.resize((W, H), Image.NEAREST)
    return np.asarray(mask) > 127


def predict(gt_dir: Path, tag: str, seg_path: Path | None) -> None:
    seg = load_segmentation(seg_path)
    seg._persist_result = lambda result: None  # keep segmentation_results/ clean
    out_dir = EVAL_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for im in load_ground_truth(gt_dir):
        path = gt_dir / im["file"]
        result = asyncio.run(seg.run_segmentation(path.read_bytes(), "image/jpeg"))
        if result.method != "grounding_dino_sam2":
            raise SystemExit(f"{im['file']}: local pipeline did not run ({result.fallback_reason})")
        W, H = im["W"], im["H"]
        items = []
        masks = []
        for item in result.items:
            b = item.bbox
            items.append({
                "label": item.label,
                "confidence": item.confidence,
                "box": [b.x_min * W, b.y_min * H, b.x_max * W, b.y_max * H],
                "mask_precise": item.mask_precise,
            })
            masks.append(decode_mask(item.mask_image, W, H))
        stem = Path(im["file"]).stem
        np.savez_compressed(out_dir / f"{stem}.npz", masks=np.array(masks, dtype=bool).reshape(len(masks), H, W))
        (out_dir / f"{stem}.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
        print(f"{tag} {im['file'][:40]}: {len(items)} items")


def load_predictions(tag: str, file_name: str) -> list[dict]:
    stem = Path(file_name).stem
    items = json.loads((EVAL_DIR / tag / f"{stem}.json").read_text(encoding="utf-8"))
    masks = np.load(EVAL_DIR / tag / f"{stem}.npz")["masks"]
    for item, mask in zip(items, masks):
        item["mask"] = mask
    return items


# ── Step 2: scoring ─────────────────────────────────────────────────────────

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pair(gts: list[dict], preds: list[dict], iou) -> list[tuple[int, int, float]]:
    """One-to-one pairs (gt index, pred index, IoU) with IoU >= IOU_MIN."""
    if not gts or not preds:
        return []
    m = np.array([[iou(g, p) for p in preds] for g in gts])
    rows, cols = linear_sum_assignment(-m)
    return [(r, c, float(m[r, c])) for r, c in zip(rows, cols) if m[r, c] >= IOU_MIN]


def ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def score(gt_dir: Path, tags: list[str]) -> None:
    images = load_ground_truth(gt_dir)
    n_gt = sum(len(im["objects"]) for im in images)
    report = {}
    per_class_all = {}

    for tag in tags:
        tp = n_pred = box_tp = type_ok = sub_ok = 0
        ious = []
        per_class = defaultdict(lambda: {"n": 0, "found": 0, "type_ok": 0, "iou_sum": 0.0})
        confusion = Counter()
        rows = []
        for im in images:
            gts = im["objects"]
            preds = load_predictions(tag, im["file"])
            n_pred += len(preds)
            pairs = pair(gts, preds, lambda g, p: mask_iou(g["mask"], p["mask"]))
            box_tp += len(pair(gts, preds, lambda g, p: box_iou(g["box"], p["box"])))
            matched = {gi: (pi, iou) for gi, pi, iou in pairs}
            for gi, g in enumerate(gts):
                c = per_class[g["label"]]
                c["n"] += 1
                if gi not in matched:
                    rows.append([im["file"], g["label"], "", "", "missed"])
                    continue
                pi, iou = matched[gi]
                p = preds[pi]
                same_type = ptype(p["label"]) is not None and ptype(p["label"]) == ptype(g["label"])
                tp += 1
                ious.append(iou)
                type_ok += same_type
                sub_ok += subclass(p["label"]) == g["label"]
                c["found"] += 1
                c["type_ok"] += same_type
                c["iou_sum"] += iou
                if not same_type:
                    confusion[(g["label"], p["label"])] += 1
                rows.append([im["file"], g["label"], p["label"], f"{iou:.3f}", "ok" if same_type else "wrong type"])
            for pi, p in enumerate(preds):
                if pi not in {m[0] for m in matched.values()}:
                    rows.append([im["file"], "", p["label"], "", "extra"])

        precision, recall = ratio(tp, n_pred), ratio(tp, n_gt)
        report[tag] = {
            "ground truth objects": n_gt,
            "predicted cards": n_pred,
            "detection precision": precision,
            "detection recall": recall,
            "detection F1": ratio(2 * precision * recall, precision + recall),
            "type accuracy (of found)": ratio(type_ok, tp),
            "subclass accuracy (of found)": ratio(sub_ok, tp),
            "mean mask IoU (of found)": float(np.mean(ious)) if ious else float("nan"),
            "usable precision": ratio(type_ok, n_pred),
            "usable recall": ratio(type_ok, n_gt),
            "box precision": ratio(box_tp, n_pred),
            "box recall": ratio(box_tp, n_gt),
        }
        per_class_all[tag] = per_class
        out = EVAL_DIR / tag
        with open(out / "matches.csv", "w", encoding="utf-8") as f:
            f.write("image,gt_label,pred_label,mask_iou,result\n")
            f.writelines(",".join(r) + "\n" for r in rows)
        (out / "confusion.json").write_text(
            json.dumps([{"gt": g, "pred": p, "count": n} for (g, p), n in confusion.most_common()], indent=1),
            encoding="utf-8",
        )

    # Overall table
    width = max(len(k) for k in report[tags[0]])
    print(f"\n{'':{width}}  " + "  ".join(f"{t:>10}" for t in tags))
    for key in report[tags[0]]:
        vals = [report[t][key] for t in tags]
        fmt = (lambda v: f"{v:>10d}") if isinstance(vals[0], int) else (lambda v: f"{v:>10.3f}")
        print(f"{key:{width}}  " + "  ".join(fmt(v) for v in vals))

    # Per class: recall, type accuracy and mean IoU of the found ones
    classes = sorted({c for t in tags for c in per_class_all[t]}, key=lambda c: -per_class_all[tags[0]][c]["n"])
    print(f"\n{'class':14} {'n':>3}  " + "  ".join(f"{t[:8]:>8} rec / type / IoU" for t in tags))
    for cls in classes:
        cells = []
        for t in tags:
            c = per_class_all[t][cls]
            cells.append(
                f"{ratio(c['found'], c['n']):>8.2f} / {ratio(c['type_ok'], c['found']):>4.2f} / {ratio(c['iou_sum'], c['found']):>4.2f}"
            )
        print(f"{cls:14} {per_class_all[tags[0]][cls]['n']:>3}  " + "  ".join(cells))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary = EVAL_DIR / f"summary_{'_vs_'.join(tags)}.json"
    summary.write_text(json.dumps({
        "overall": report,
        "per_class": {t: {k: dict(v) for k, v in per_class_all[t].items()} for t in tags},
    }, indent=1), encoding="utf-8")
    print(f"\nsaved {summary} and compare/eval/<tag>/matches.csv, confusion.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("gt_dir", type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--seg", type=Path, help="segmentation.py to test (default: this folder's)")
    s = sub.add_parser("score")
    s.add_argument("gt_dir", type=Path)
    s.add_argument("tags", nargs="+")
    args = ap.parse_args()
    if args.cmd == "predict":
        predict(args.gt_dir, args.tag, args.seg)
    else:
        score(args.gt_dir, args.tags)


if __name__ == "__main__":
    main()
