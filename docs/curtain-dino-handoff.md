# ม่านหายจาก segmentation — handoff ไปรันบนเครื่องที่มี GPU

เขียนไว้ 2026-11-23 บน branch `fix/sam2-mask-quality` (เครื่องไม่มี GPU เลยต้องส่งต่อ)
บริบทเพิ่มเติม (pipeline ทั้งหมด, การตัดสินใจอื่นๆ): [segmentation-notes.md](segmentation-notes.md)

## ปัญหา

จากชุดทดสอบ 24 ภาพจริง (Roboflow, ไม่อยู่ใน repo) ม่าน (`curtain`) เป็นหมวดที่ detect หลุดมากที่สุด
24 จาก 46 ชิ้น (~52%) หายไป มากกว่าหมวดอื่นชัดเจน

## สาเหตุที่เจอ (อ่านโค้ด ยังไม่ได้วัดจริงเพราะไม่มี GPU)

ม่านหายตั้งแต่ตอน `canonicalize_label` (`backend/segmentation.py`) ไม่ใช่ตัวกรองหลัง SAM 2:

- โค้ดมีกฎ "architecture wins": ถ้าคำดิบจาก DINO มีคำว่า `door`/`window` ติดมาด้วย จะคืนคำนั้นทันที
  (กันไม่ให้ตู้เสื้อผ้าที่ DINO เห็นเป็น `wardrobe door` โผล่เป็นการ์ดสินค้า — ของเดิมตั้งใจไว้แบบนี้)
- ปัญหาคือม่านแขวนอยู่บนหน้าต่างเสมอ ภาพเดียวกัน ตำแหน่งเดียวกัน DINO เลย glue คำ `window` ติดกับ
  `curtain`/`blinds` บ่อยกว่าคู่อื่นมาก (มีเทสยืนยันคำดิบจริงจาก DINO: `canonicalize_label("blinds window") == "window"`
  ใน `backend/tests/test_segmentation_helpers.py`)
- พอกลายเป็น `window` มันถูกตัดทิ้งเป็น non-furniture ทันที (`is_candidate_label`) — ไม่เหลือ label furniture
  ให้ตัวกรองอื่น (group-box, dedupe, swallowed-mask) เห็นเลย คือหายตั้งแต่ต้นทาง

## สิ่งที่แก้ไปแล้ว (ยังไม่ commit — อยู่ใน working tree ของเครื่องนี้)

เพิ่ม DINO pass ที่ 3 เฉพาะ curtain/blinds ที่ **ไม่มีคำ `window`/`door` อยู่ใน prompt เลย**
(เลียน pattern เดิมของ doorway-only pass) — DINO ตอบคำที่ไม่อยู่ใน prompt ไม่ได้ จึงไม่มีทางถูก
collapse เป็น `window` ในรอบนี้ ไม่ได้แก้กฎ architecture-wins เดิมเลย (ยังจำเป็นสำหรับเคสตู้เสื้อผ้า/ประตู)

ถ้า clone branch นี้แล้วไม่เห็น diff นี้ใน `backend/segmentation.py` (เผื่อยังไม่ได้ push/commit) ให้ apply patch นี้เอง:

```diff
--- a/backend/segmentation.py
+++ b/backend/segmentation.py
@@ -72,6 +72,10 @@ Env vars
   SEGMENTATION_DEVICE       Force "cuda" or "cpu" (default: auto-detect via torch.cuda.is_available())
   SAM_BOX_THRESHOLD         Confidence threshold for detection boxes (default: 0.30)
   SAM_TEXT_THRESHOLD        Text-matching threshold for Grounding DINO (default: 0.25)
+  SAM_CURTAIN_BOX_THRESHOLD Confidence threshold for the curtain/blinds-only DINO pass
+                            (default: 0.20; lower than furniture — curtains . blinds . has
+                            no "window"/"door" in it, so it can't be collapsed to architecture
+                            by canonicalize_label the way the main pass's "curtain window" can)
   SAM_BOX_PAD               Grow each DINO box by this share of its size before SAM 2 (default: 0,
                             DINO's own box; padding mostly pulled in wall and floor)
   SAM_MULTIMASK             1 = ask SAM 2 for 3 candidate masks and pick the largest likely one,
@@ -966,13 +970,27 @@ _OPENING_PROMPT = "doorway . open door ."
 # the sill check) catch what a lower bar lets in.
 _OPENING_BOX_THRESHOLD = 0.25

+# A third, curtain/blinds-only pass, with no "window" or "door" in its prompt.
+# In the full prompt, DINO's grounded text for a curtain box is often glued
+# with "window" ("curtain window", like "blinds window" below) because the
+# curtain hangs right on the window it was matched beside; canonicalize_label
+# then has to prefer the architecture word (a wardrobe boxed with "door" must
+# come back as "door", not a shoppable wardrobe), so the curtain is dropped as
+# architecture too. Asking for just "curtain . blinds ." removes "window" from
+# the vocabulary DINO can even answer with, so this pass can't glue it on.
+_CURTAIN_PROMPT = "curtain . blinds ."
+_CURTAIN_BOX_THRESHOLD_ENV = "SAM_CURTAIN_BOX_THRESHOLD"
+_CURTAIN_BOX_THRESHOLD_DEFAULT = 0.20
+

 def _detect_with_grounding_dino_local(img: Image.Image) -> list[dict]:
     openings = [
         {**d, "label": "doorway"}
         for d in _run_grounding_dino(img, _OPENING_PROMPT, _OPENING_BOX_THRESHOLD)
     ]
-    return _run_grounding_dino(img, _FURNITURE_PROMPT) + openings
+    curtain_threshold = float(os.getenv(_CURTAIN_BOX_THRESHOLD_ENV, str(_CURTAIN_BOX_THRESHOLD_DEFAULT)))
+    curtains = _run_grounding_dino(img, _CURTAIN_PROMPT, curtain_threshold)
+    return _run_grounding_dino(img, _FURNITURE_PROMPT) + openings + curtains


 def _run_grounding_dino(img: Image.Image, prompt: str, box_threshold: float | None = None) -> list[dict]:
```

รัน `python -m unittest discover -s tests` ผ่านหมดแล้ว (121 ตัว อีกตัวที่ fail คือ `test_product_matching`
ขาด `pydantic_settings` ไม่เกี่ยวกับการแก้นี้)

## งานที่ต้องทำต่อบนเครื่อง GPU

**คำถามที่ต้องตอบ:** (1) การแก้ curtain pass ข้างบนช่วยจริงไหม (ก่อน/หลัง) (2) ถ้าเปลี่ยน
`GROUNDING_DINO_MODEL` จาก `grounding-dino-tiny` เป็น `grounding-dino-base` จะช่วยเรื่อง detect/label/
อื่นๆ เพิ่มอีกไหม คุ้มกับ (ขนาดโมเดลโต ~2 เท่า, inference ช้าลง) ไหม หรือมีทางที่ดีกว่า

**ชุดภาพ:** `D:\Downloads\images\images\` บนเครื่องเดิม มีห้อง ~70 ห้อง **ไม่มี ground truth
(ไม่มี `_annotations.coco.json`)** — เป็นภาพดิบล้วนๆ ดังนั้นรันนี้คือการเทียบจำนวน/label ที่เจอ
ก่อน-หลัง (ไม่ใช่ scored eval แบบ precision/recall จริงจากไฟล์ `evaluate_segmentation.py`) ถ้าอยาก
ได้ตัวเลข precision/recall จริง ต้องมีชุด Roboflow export (ดู `segmentation-notes.md` หัวข้อ "ที่ยังค้าง")
แยกออกไปทำอีกรอบ

ย้าย/copy โฟลเดอร์ภาพไปเครื่อง GPU ด้วย (หรือ map ไดรฟ์เน็ตเวิร์ก) แล้วแก้ path ในสคริปต์ด้านล่างให้ตรง

### 1. setup เครื่อง GPU

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# torch แบบ CUDA (เช็ก CUDA version ของเครื่องก่อน เช่น RTX 50-series ใช้ cu128):
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

checkpoint SAM 2 ต้องมีที่ `backend/checkpoints/sam2.1_hiera_small.pt` (ดูวิธีโหลดใน docstring บนสุดของ
`segmentation.py`) — ถ้า copy ทั้งโฟลเดอร์ `backend/` มาจากเครื่องนี้ก็ติดมาด้วยอยู่แล้ว

**สำคัญ:** `GEMINI_API_KEY` ที่ตั้งไว้ใน `backend.env` โควต้าหมดแล้ว (เจอ `402 Payment Required` ตอนรัน
unit test) — สคริปต์ด้านล่างปิด Gemini ไว้ (`SEGMENTATION_LABELER=none`) เพื่อวัดแค่ผลของ DINO เอง
(ทั้ง detect และ label ดิบ) ไม่ปนกับ Gemini/CLIP ถ้าจะเติม key ใหม่ก่อนก็ได้ แต่ไม่ใช่เรื่องจำเป็นสำหรับ
คำถามนี้ (DINO vs DINO ไม่เกี่ยวกับ Gemini)

`grounding-dino-base` ยังไม่เคยโหลดบนเครื่องนั้น จะโดนดาวน์โหลดจาก HF Hub รอบแรกที่ใช้ (~1GB, เน็ตต้องถึง
huggingface.co)

### 2. ไฟล์สคริปต์ที่ใช้ (วางในโฟลเดอร์ไหนก็ได้ แก้ `BACKEND_DIR` ให้ตรง)

`run_batch.py` — รัน pipeline ทั้งหมดบนภาพทีละ config แล้ว dump label/confidence/area ต่อชิ้นเป็น JSON
(ไม่ใช้ Gemini, ไม่ persist ผลลงดิสก์แบบปกติ):

```python
"""
Run the local segmentation pipeline over a list of room images with one
config (which segmentation.py file, which GROUNDING_DINO_MODEL) and dump a
per-item JSON. No ground truth here -- this is a before/after + DINO-model
comparison on unlabeled photos, not a scored eval.

Usage:
    python run_batch.py --out results_X.json --images img1.jpg img2.jpg ...
      [--seg-file path/to/segmentation.py] [--model IDEA-Research/grounding-dino-base]
"""
import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(r"C:\dev\smart-decorator\backend")  # <-- แก้ตามเครื่อง GPU


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--seg-file", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    os.environ["SEGMENTATION_LABELER"] = "none"   # isolate DINO itself; Gemini quota is out anyway
    os.environ["SEGMENTATION_CLIP_RELABEL"] = "0"
    if args.model:
        os.environ["GROUNDING_DINO_MODEL"] = args.model

    os.chdir(BACKEND_DIR)
    sys.path.insert(0, str(BACKEND_DIR))

    from dotenv import load_dotenv
    for name in (".env", "backend.env"):
        load_dotenv(BACKEND_DIR / name)

    if args.seg_file:
        spec = importlib.util.spec_from_file_location("segmentation_variant", args.seg_file)
        seg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(seg)
    else:
        import segmentation as seg  # noqa

    seg._persist_result = lambda result: None

    results = {}
    t_start = time.time()
    for i, p in enumerate(args.images):
        path = Path(p)
        t0 = time.time()
        try:
            data = path.read_bytes()
            result = asyncio.run(seg.run_segmentation(data, "image/jpeg"))
            dt = time.time() - t0
            results[path.name] = {
                "method": result.method,
                "fallback_reason": result.fallback_reason,
                "seconds": round(dt, 1),
                "items": [
                    {
                        "label": it.label,
                        "dino_label": it.dino_label,
                        "confidence": it.confidence,
                        "area_pct": it.features.area_pct,
                        "mask_precise": it.mask_precise,
                    }
                    for it in result.items
                ],
            }
            print(f"[{i+1}/{len(args.images)}] {path.name}: {len(result.items)} items in {dt:.1f}s", flush=True)
        except Exception as exc:
            results[path.name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[{i+1}/{len(args.images)}] {path.name}: ERROR {exc}", flush=True)

        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"done: {len(args.images)} images in {time.time() - t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
```

### 3. config ที่ต้องรัน (3 รอบ บนภาพชุดเดียวกันทั้งหมด)

| tag | seg-file | GROUNDING_DINO_MODEL | คืออะไร |
|---|---|---|---|
| `before` | เวอร์ชันก่อนแก้ (ดูวิธีได้ `segmentation_before.py` ด้านล่าง) | ค่า default (tiny) | baseline ก่อนแก้ม่าน |
| `after_tiny` | เวอร์ชันปัจจุบัน (มี curtain pass) | ค่า default (tiny) | ผลของการแก้ม่านล้วนๆ |
| `after_base` | เวอร์ชันปัจจุบัน (มี curtain pass) | `IDEA-Research/grounding-dino-base` | ผลของการแก้ม่าน **+** เปลี่ยนโมเดล |

เอาไฟล์ `segmentation.py` เวอร์ชันก่อนแก้มาจาก git (รันบนเครื่องที่มี repo นี้):
`git log --oneline -- backend/segmentation.py` หาคอมมิตก่อนที่จะมี diff ข้างบน แล้ว
`git show <commit>:backend/segmentation.py > segmentation_before.py` (ถ้า diff นี้ยังไม่ commit ที่ไหน
เลยให้เอา `backend/segmentation.py` ปัจจุบันไปลบ diff ข้างบนออกเองแทน)

ตัวอย่างคำสั่งรัน (PowerShell, ปรับ path ภาพ/จำนวนภาพได้ — เอาแค่ครึ่งหนึ่งก่อนก็พอถ้านานไป):

```powershell
$images = Get-ChildItem "D:\Downloads\images\images\*.jpg" | Sort-Object Name | Select-Object -First 35 | % FullName

.\.venv\Scripts\python.exe run_batch.py --out results_before.json `
  --seg-file segmentation_before.py --images $images

.\.venv\Scripts\python.exe run_batch.py --out results_after_tiny.json `
  --images $images

.\.venv\Scripts\python.exe run_batch.py --out results_after_base.json `
  --model IDEA-Research/grounding-dino-base --images $images
```

GPU ควรเร็วกว่า CPU มาก (เครื่องนี้วัดได้ tiny ~32s/ภาพ, base ~41s/ภาพ บน CPU ล้วนๆ — บน GPU ควรลดลง
เหลือหลักวินาที) ลองรันทั้ง 70 ภาพได้เลยถ้าเร็วพอ

### 4. วิเคราะห์ผล — สิ่งที่ต้องดู

1. **นับจำนวน `curtain`/`blinds`** ต่อภาพในแต่ละ tag (`label` หรือ `dino_label` ที่เป็น curtain/blinds)
   เทียบ `before` vs `after_tiny` — ควรเห็นจำนวนเพิ่มขึ้นชัดถ้า fix ได้ผลจริง
2. **เทียบ `after_tiny` vs `after_base`**: จำนวนรวมต่อภาพเปลี่ยนไหม, label ที่ DINO เดาเปลี่ยนไหม (ดู
   field `dino_label` ก่อน Gemini เปลี่ยน — แต่ตอนนี้ปิด Gemini ไว้แล้ว `label` คือคำดิบของ DINO เลย),
   confidence เฉลี่ยสูงขึ้นไหม, เจอหมวดที่ tiny พลาดไปเพิ่มไหม (เช่นม่านที่ tiny ยังพลาดอยู่)
3. **เช็คของปลอม/เสีย**: label แปลกๆ ที่ไม่อยู่ใน vocabulary (เช่นตัวอย่างที่เจอบนเครื่องนี้ base เคยตอบ
   `"furniture"` ซึ่งเป็น fallback ของ `canonicalize_label` แปลว่า DINO กลึงคำแปลกมา — ถ้า base ให้คำ
   แบบนี้บ่อยกว่าหรือพอๆกับ tiny ก็เป็นข้อเสียที่ต้องเอามาเทียบกับข้อดีด้าน recall)
4. **เวลา/ขนาด**: เทียบ `seconds` เฉลี่ยต่อภาพของ tiny vs base บน GPU จริง (ไม่ใช่เลข CPU จากเครื่องนี้) —
   ถ้าต่างกันน้อยบน GPU ก็เป็นเหตุผลสนับสนุนเปลี่ยนได้ง่ายขึ้น ถ้าต่างมากต้องถามว่า latency ต่อ request
   ของระบบจริงรับได้ไหม
5. ถ้าอยากดูภาพจริงแบบ side-by-side (ไม่ใช่แค่ตัวเลข) ใช้ `compare_segmentation.py` ที่มีอยู่แล้ว —
   เช็คสัก 5-10 ภาพที่ `before`→`after_tiny` ต่างกันมากที่สุดในแง่จำนวนม่าน

### สิ่งที่คาดว่าจะได้คำตอบ

- การแก้ curtain pass ช่วยเรื่องม่านจริงไหม แค่ไหน (ตัวเลขนับจำนวนม่านต่อภาพ)
- เปลี่ยนเป็น `grounding-dino-base` ช่วยต่อจากนั้นอีกไหม ในแง่ detect (เจอของมากขึ้น/น้อยลง), label
  ดิบของ DINO (แม่นขึ้นไหม หรือมี garbage label มากขึ้น), เวลาที่ใช้จริงบน GPU — แล้วสรุปว่าคุ้มเปลี่ยน
  ไหม หรือทางอื่น (เช่น threshold เฉพาะหมวด, prompt แยกเพิ่มสำหรับหมวดอื่นที่มีปัญหาเดียวกับม่าน) จะดีกว่า
