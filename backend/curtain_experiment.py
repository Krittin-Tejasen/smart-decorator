"""
Curtain recall experiment: run one config of the local pipeline over a folder
of room photos, then compare configs with a curtain-focused report.

    # 1. one run per config (outputs go to compare/eval/<tag>/)
    python curtain_experiment.py predict DIR --tag before --no-curtain-pass
    python curtain_experiment.py predict DIR --tag after_tiny
    python curtain_experiment.py predict DIR --tag canon --no-curtain-pass --canon-curtain
    python curtain_experiment.py predict DIR --tag base_30 --model IDEA-Research/grounding-dino-base

    # 2. compare
    python curtain_experiment.py report DIR before after_tiny canon base_30 --html curtain_report.html
    python evaluate_segmentation.py score DIR before after_tiny canon base_30   # only with ground truth

DIR is either a Roboflow COCO export split (holds _annotations.coco.json:
the report then scores curtains against the answer key) or a plain folder of
.jpg/.png photos (no ground truth: counts plus a crop gallery to check by eye).

Predictions are written in evaluate_segmentation.py's format, so that
script's `score` works on the same tags. Each tag also gets _run.json with:
  - every raw Grounding DINO box per pass (main / opening / curtain), with
    DINO's raw text label before canonicalize_label and the label after it
  - which pass and raw text each final card came from
  - seconds per image (the first image is run once untimed as a warm-up, so
    model loading and downloads are not in the average)

Config switches (all monkeypatched on the loaded module, segmentation.py
itself is unchanged):
  --no-curtain-pass   drop the curtain/blinds-only third DINO pass (= the
                      code before e082afc)
  --canon-curtain     in canonicalize_label, a raw label with curtain/blinds
                      and "window" (but no door) stays a curtain instead of
                      collapsing to "window" (the cheap alternative fix)
  --model             GROUNDING_DINO_MODEL (default: tiny)
  --box-threshold     SAM_BOX_THRESHOLD for the main pass (default 0.30)
  --curtain-threshold SAM_CURTAIN_BOX_THRESHOLD (default 0.20)
  --labeler           SEGMENTATION_LABELER (default none: DINO's own label,
                      so the runs compare DINO, not Gemini)
"""

import argparse
import asyncio
import html
import json
import os
import sys
import time
from base64 import b64encode
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE / "compare" / "eval"
CURTAIN_LABELS = {"curtain", "blinds"}
GT_CURTAIN_LABELS = {"curtain", "curtains", "blinds"}
# --wall-pass prompt words that are another name for a main-vocabulary piece.
WALL_SYNONYMS = {"painting": "artwork", "picture frame": "artwork", "picture": "artwork", "frame": "artwork"}


def list_images(d: Path) -> tuple[list[str], bool]:
    coco = d / "_annotations.coco.json"
    if coco.exists():
        return [im["file_name"] for im in json.loads(coco.read_text(encoding="utf-8"))["images"]], True
    files = sorted(p.name for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    return files, False


def mime_of(name: str) -> str:
    ext = Path(name).suffix.lower()
    return {".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")


# ── Step 1: predict ─────────────────────────────────────────────────────────

def load_patched_segmentation(args):
    os.environ["SEGMENTATION_LABELER"] = args.labeler
    os.environ["SEGMENTATION_CLIP_RELABEL"] = "0"
    if args.model:
        os.environ["GROUNDING_DINO_MODEL"] = args.model
    if args.box_threshold is not None:
        os.environ["SAM_BOX_THRESHOLD"] = str(args.box_threshold)
    if args.curtain_threshold is not None:
        os.environ["SAM_CURTAIN_BOX_THRESHOLD"] = str(args.curtain_threshold)

    os.chdir(HERE)  # checkpoints/ and configs are relative to backend/
    sys.path.insert(0, str(HERE))
    from dotenv import load_dotenv
    for name in (".env", "backend.env"):
        load_dotenv(HERE / name)

    import segmentation as seg

    seg._persist_result = lambda result: None
    # _CURTAIN_PROMPT only exists in segmentation.py from e082afc (removed after it).
    pass_of_prompt = {seg._FURNITURE_PROMPT: "main", seg._OPENING_PROMPT: "opening",
                      getattr(seg, "_CURTAIN_PROMPT", "curtain . blinds ."): "curtain",
                      getattr(seg, "_WALL_PROMPT", "-"): "wall"}
    raw_log: list[dict] = []          # every DINO box of the current image
    current_raws: list[str] = []

    orig_canon = seg.canonicalize_label

    def canon(raw: str) -> str:
        out = orig_canon(raw)
        if args.canon_curtain and out == "window":
            words = raw.lower().split()
            if "door" not in words and "doorway" not in words:
                for w in ("curtain", "blinds"):
                    if w in words:
                        out = w
                        break
        current_raws.append(raw)
        return out

    seg.canonicalize_label = canon
    orig_run = seg._run_grounding_dino

    def run_dino(img, prompt, box_threshold=None):
        current_raws.clear()
        dets = orig_run(img, prompt, box_threshold)
        name = pass_of_prompt.get(prompt, "other")
        W, H = img.size
        for d, raw in zip(dets, current_raws):
            x1, y1, x2, y2 = d["box"]
            raw_log.append({
                "pass": name, "raw": raw, "label": d["label"], "confidence": d["confidence"],
                "box": [x1 / W, y1 / H, x2 / W, y2 / H],
            })
        return dets

    seg._run_grounding_dino = run_dino

    if args.wall_pass:
        pass_of_prompt[args.wall_pass] = "wall"
    if args.no_curtain_pass or args.wall_pass:
        orig_detect = seg._detect_with_grounding_dino_local

        def passes(img):
            if args.no_curtain_pass:
                openings = [
                    {**d, "label": "doorway"}
                    for d in seg._run_grounding_dino(img, seg._OPENING_PROMPT, seg._OPENING_BOX_THRESHOLD)
                ]
                dets = seg._run_grounding_dino(img, seg._FURNITURE_PROMPT) + openings
            else:
                dets = orig_detect(img)
            if args.wall_pass:
                # Its synonyms name the main vocabulary's piece; a box with no
                # text over the text threshold ("furniture") is dropped.
                for d in seg._run_grounding_dino(img, args.wall_pass, args.wall_threshold):
                    label = WALL_SYNONYMS.get(d["label"], d["label"])
                    if label in seg._VOCAB_SET and label != "furniture":
                        dets.append({**d, "label": label})
            return dets
        seg._detect_with_grounding_dino_local = passes

    return seg, raw_log


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def source_of(item_conf: float, item_box, raws: list[dict]) -> dict | None:
    """The raw DINO box a card came from: same confidence (rounded like the
    card's), the closest box among ties. Dedupe keeps the most confident box's
    confidence, so this is the box that won."""
    same = [r for r in raws if round(r["confidence"], 3) == item_conf]
    return max(same, key=lambda r: box_iou(r["box"], item_box), default=None)


def predict(args) -> None:
    import numpy as np
    from evaluate_segmentation import decode_mask

    d = args.dir.resolve()
    files, _has_gt = list_images(d)
    if args.limit:
        files = files[: args.limit]
    seg, raw_log = load_patched_segmentation(args)
    out_dir = EVAL_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up: load every model before timing.
    asyncio.run(seg.run_segmentation((d / files[0]).read_bytes(), mime_of(files[0])))

    run = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "model": os.getenv("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny"), "images": {}}
    for i, name in enumerate(files):
        raw_log.clear()
        data = (d / name).read_bytes()
        t0 = time.perf_counter()
        result = asyncio.run(seg.run_segmentation(data, mime_of(name)))
        dt = time.perf_counter() - t0
        if result.method != "grounding_dino_sam2":
            raise SystemExit(f"{name}: local pipeline did not run ({result.fallback_reason})")
        from PIL import Image
        W, H = Image.open(BytesIO(data)).size
        items, masks, cards = [], [], []
        for item in result.items:
            b = item.bbox
            box_n = [b.x_min, b.y_min, b.x_max, b.y_max]
            items.append({"label": item.label, "confidence": item.confidence,
                          "box": [b.x_min * W, b.y_min * H, b.x_max * W, b.y_max * H],
                          "mask_precise": item.mask_precise})
            masks.append(decode_mask(item.mask_image, W, H))
            src = source_of(item.confidence, box_n, raw_log)
            cards.append({"label": item.label, "confidence": item.confidence, "box": box_n,
                          "pass": src["pass"] if src else None, "raw": src["raw"] if src else None})
        stem = Path(name).stem
        np.savez_compressed(out_dir / f"{stem}.npz", masks=np.array(masks, dtype=bool).reshape(len(masks), H, W))
        (out_dir / f"{stem}.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
        run["images"][name] = {"seconds": round(dt, 2), "raw_dino": list(raw_log), "cards": cards}
        n_cur = sum(c["label"] in CURTAIN_LABELS for c in cards)
        print(f"[{args.tag} {i + 1}/{len(files)}] {name[:40]}: {len(cards)} cards, {n_cur} curtain, {dt:.1f}s", flush=True)
        (out_dir / "_run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    secs = [v["seconds"] for v in run["images"].values()]
    print(f"done {args.tag}: {len(files)} images, mean {sum(secs) / len(secs):.2f}s/image (warm-up excluded)")


# ── Step 2: report ──────────────────────────────────────────────────────────

def report(args) -> None:
    import numpy as np

    sys.path.insert(0, str(HERE))
    os.chdir(HERE)
    d = args.dir.resolve()
    files, has_gt = list_images(d)
    sys.path.insert(0, str(HERE))
    import segmentation as seg
    vocab = seg._VOCAB_SET

    gt = {}
    if has_gt:
        from evaluate_segmentation import load_ground_truth, load_predictions, mask_iou, pair
        gt = {im["file"]: im for im in load_ground_truth(d)}

    runs = {t: json.loads((EVAL_DIR / t / "_run.json").read_text(encoding="utf-8")) for t in args.tags}
    files = [f for f in files if all(f in r["images"] for r in runs.values())]
    out: dict = {"images": len(files), "has_ground_truth": has_gt, "tags": {}}
    gallery: dict[str, list] = {t: [] for t in args.tags}

    for t, run in runs.items():
        s = {"cards": 0, "curtain cards": 0, "curtain cards by pass": {}, "seconds mean": 0.0,
             "garbage labels": {}, "main-pass raw curtain+window/door": 0,
             "  ...collapsed to window/door": 0, "raw examples": {},
             "curtain-pass cards on a main-pass window box": 0}
        secs = []
        if has_gt:
            s.update({"gt curtains": 0, "gt curtains found (any label)": 0, "gt curtains found as curtain": 0,
                      "curtain cards matching a gt curtain": 0, "curtain cards matching other gt": 0,
                      "curtain cards matching nothing": 0, "detection tp": 0, "detection preds": 0, "gt objects": 0})
        for f in files:
            rec = run["images"][f]
            secs.append(rec["seconds"])
            cards = rec["cards"]
            s["cards"] += len(cards)
            for r in rec["raw_dino"]:
                words = r["raw"].lower().split()
                if r["pass"] == "main" and ({"curtain", "blinds"} & set(words)) and ({"window", "door", "doorway"} & set(words)):
                    s["main-pass raw curtain+window/door"] += 1
                    if r["label"] in {"window", "door", "doorway"}:
                        s["  ...collapsed to window/door"] += 1
                    s["raw examples"][r["raw"]] = s["raw examples"].get(r["raw"], 0) + 1
            for c in cards:
                if c["label"] not in vocab:
                    s["garbage labels"][c["label"]] = s["garbage labels"].get(c["label"], 0) + 1
            windows = [r["box"] for r in rec["raw_dino"] if r["pass"] == "main" and r["label"] == "window"]
            main_curtains = [r["box"] for r in rec["raw_dino"] if r["pass"] == "main" and r["label"] in CURTAIN_LABELS]

            verdicts = {}
            if has_gt:
                im = gt[f]
                preds = load_predictions(t, f)
                pairs = pair(im["objects"], preds, lambda g, p: mask_iou(g["mask"], p["mask"]))
                s["detection tp"] += len(pairs)
                s["detection preds"] += len(preds)
                s["gt objects"] += len(im["objects"])
                by_pred = {pi: im["objects"][gi]["label"] for gi, pi, _ in pairs}
                for gi, g in enumerate(im["objects"]):
                    if g["label"] in GT_CURTAIN_LABELS:
                        s["gt curtains"] += 1
                        hit = [pi for gj, pi, _ in pairs if gj == gi]
                        if hit:
                            s["gt curtains found (any label)"] += 1
                            s["gt curtains found as curtain"] += preds[hit[0]]["label"] in CURTAIN_LABELS
                for pi, p in enumerate(preds):
                    if p["label"] in CURTAIN_LABELS:
                        g = by_pred.get(pi)
                        verdicts[pi] = "gt curtain" if g in GT_CURTAIN_LABELS else (f"gt {g}" if g else "nothing")
                        key = ("curtain cards matching a gt curtain" if g in GT_CURTAIN_LABELS else
                               "curtain cards matching other gt" if g else "curtain cards matching nothing")
                        s[key] += 1

            for ci, c in enumerate(cards):
                if c["label"] not in CURTAIN_LABELS:
                    continue
                s["curtain cards"] += 1
                p = c["pass"] or "?"
                s["curtain cards by pass"][p] = s["curtain cards by pass"].get(p, 0) + 1
                on_window = c["pass"] == "curtain" and any(box_iou(c["box"], w) > 0.5 for w in windows) \
                    and not any(box_iou(c["box"], m) > 0.5 for m in main_curtains)
                s["curtain-pass cards on a main-pass window box"] += on_window
                gallery[t].append({"file": f, "box": c["box"], "pass": p, "conf": c["confidence"],
                                   "raw": c["raw"], "on_window": on_window, "verdict": verdicts.get(ci)})
        s["seconds mean"] = round(sum(secs) / len(secs), 2) if secs else None
        s["raw examples"] = dict(sorted(s["raw examples"].items(), key=lambda kv: -kv[1])[:10])
        if has_gt:
            p_, r_ = s["detection tp"] / max(s["detection preds"], 1), s["detection tp"] / max(s["gt objects"], 1)
            s["detection P/R/F1"] = [round(p_, 3), round(r_, 3), round(2 * p_ * r_ / (p_ + r_), 3) if p_ + r_ else 0]
        out["tags"][t] = s
    out["curtain_cards"] = gallery

    print(json.dumps(out, indent=1, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.html:
        write_gallery(Path(args.html), d, gallery, args.tags)
        print(f"gallery: {args.html}")


def write_gallery(path: Path, d: Path, gallery: dict, tags: list[str]) -> None:
    """Crops of every curtain card per tag, to check by eye whether each is a
    curtain (crop padded 15% so the window around it shows)."""
    from PIL import Image, ImageDraw

    cache: dict[str, Image.Image] = {}
    parts = ["<!doctype html><meta charset=utf-8><title>Curtain cards</title><style>"
             "body{font:13px system-ui;margin:16px;background:#fff;color:#111}"
             "figure{display:inline-block;margin:4px;width:220px;vertical-align:top}"
             "img{width:220px;height:auto;border:1px solid #ccc}.bad{color:#b00}.ok{color:#070}</style>"]
    for t in tags:
        parts.append(f"<h2>{html.escape(t)} — {len(gallery[t])} curtain cards</h2>")
        for g in gallery[t]:
            if g["file"] not in cache:
                cache[g["file"]] = Image.open(d / g["file"]).convert("RGB")
            img = cache[g["file"]]
            W, H = img.size
            x1, y1, x2, y2 = g["box"]
            px, py = (x2 - x1) * 0.15, (y2 - y1) * 0.15
            crop_box = (max(0, (x1 - px)) * W, max(0, (y1 - py)) * H, min(1, (x2 + px)) * W, min(1, (y2 + py)) * H)
            crop = img.crop(tuple(int(v) for v in crop_box)).copy()
            ImageDraw.Draw(crop).rectangle(
                (int(x1 * W - crop_box[0]), int(y1 * H - crop_box[1]), int(x2 * W - crop_box[0]), int(y2 * H - crop_box[1])),
                outline=(255, 0, 0), width=3)
            crop.thumbnail((440, 440))
            buf = BytesIO()
            crop.save(buf, "JPEG", quality=80)
            v = g["verdict"]
            cls = "ok" if v == "gt curtain" else ("bad" if v else "")
            note = f"<span class={cls}>{html.escape(v)}</span>" if v else ""
            warn = " <b class=bad>on window box</b>" if g["on_window"] else ""
            parts.append(
                f"<figure><img src='data:image/jpeg;base64,{b64encode(buf.getvalue()).decode()}'>"
                f"<figcaption>{html.escape(g['file'][:28])}<br>pass={g['pass']} conf={g['conf']}"
                f"{warn}<br>raw: {html.escape(str(g['raw']))}<br>{note}</figcaption></figure>")
    path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("dir", type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--no-curtain-pass", action="store_true")
    p.add_argument("--canon-curtain", action="store_true")
    p.add_argument("--model")
    p.add_argument("--box-threshold", type=float)
    p.add_argument("--curtain-threshold", type=float)
    p.add_argument("--wall-pass", help='extra DINO pass prompt, e.g. "artwork . painting . picture frame . mirror . vase ."')
    p.add_argument("--wall-threshold", type=float, default=0.25)
    p.add_argument("--labeler", default="none")
    p.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("dir", type=Path)
    r.add_argument("tags", nargs="+")
    r.add_argument("--html")
    r.add_argument("--json")
    args = ap.parse_args()
    predict(args) if args.cmd == "predict" else report(args)


if __name__ == "__main__":
    main()
