# ม่านหายจาก segmentation — ผลการทดลอง + งานที่เหลือ

เขียนครั้งแรก 2026-11-23, อัปเดต 2026-11-24 หลังรันจริงบน GPU (RTX 5060, เครื่องเดิม — ตอนนี้มี CUDA แล้ว
ไม่ต้องย้ายไปเครื่องอื่น) บน branch `fix/sam2-mask-quality`
บริบทเพิ่มเติม (pipeline ทั้งหมด, การตัดสินใจอื่นๆ): [segmentation-notes.md](segmentation-notes.md)

## สรุปสั้น

- **ชุดเฉลย 24 ภาพ (Roboflow) อยู่ในเครื่องนี้** ที่ `C:\Users\PCHERO\Downloads\Smart Decorator Thesis.v1i.coco-segmentation\train`
  — เลยวัดเป็น precision/recall จริงได้ ไม่ต้องพึ่งการนับจำนวนบนภาพไม่มีเฉลย
- **สมมติฐานเดิมอธิบายได้แค่ส่วนน้อย**: คำดิบ "curtain/blinds + window" ที่โดน collapse เป็น `window`
  มีแค่ 8 กล่องใน 24 ภาพ แต่ม่านหาย 29 จาก 52 ชิ้น สาเหตุหลักคือ DINO-tiny ไม่ให้กล่องม่านเลย
  (คะแนนต่ำกว่า threshold / แพ้ decoy `window`)
- **curtain pass (e082afc) เพิ่ม recall ม่านจริง แต่ precision ตกเท่ากัน F1 แทบไม่ขยับ** (0.800 → 0.802) และสร้าง
  การ์ด `furniture` ขยะ 15 ใบ
- **ทางแก้ถูก (`--canon-curtain`) ไม่ช่วย**: F1 0.788 แย่กว่าเดิม
- **ดีที่สุดคือเปลี่ยนเป็น `grounding-dino-base` ที่ box threshold 0.25 และไม่มี curtain pass**: F1 0.843,
  ม่าน 40/52 ผิดแค่ 3 ใบ แต่ artwork / mirror / vase ลดลง (ดูข้อเสียด้านล่าง)
- **คำแนะนำ**: ถอด curtain pass ออก แล้วเปลี่ยน default เป็น base@0.25 **ต่อเมื่อ**แก้ artwork/mirror ที่หายได้แล้ว
  หรือยอมรับ trade-off นี้ (ยังไม่ได้เปลี่ยนโค้ด — รอตัดสินใจ)

## ผลบนชุดเฉลย 24 ภาพ (293 ชิ้น, ม่าน+มู่ลี่ 52 ชิ้น)

ทุก run ใช้ `SEGMENTATION_LABELER=none` (label ดิบของ DINO, ไม่มี Gemini/CLIP) เพื่อเทียบ DINO ล้วนๆ
เวลา = วินาทีต่อภาพบน RTX 5060 ไม่นับภาพ warm-up

| config | P | R | F1 | ม่านที่เจอ /52 | การ์ดม่านผิด | การ์ด `furniture` | s/ภาพ |
|---|---|---|---|---|---|---|---|
| `c_before` tiny, ไม่มี curtain pass (= 064a5ea) | 0.839 | 0.765 | 0.800 | 23 | 1 | 0 | 1.22 |
| `c_after_tiny` tiny + curtain pass @0.20 (= ปัจจุบัน) | 0.770 | 0.836 | 0.802 | 42 (เป็น curtain 35) | 23 | 15 | 1.75 |
| `c_after_t30` tiny + curtain pass @0.30 | 0.795 | 0.805 | 0.800 | 33 | 20 | 0 | 1.60 |
| `c_canon` tiny + canonicalize ไม่ collapse ม่าน | 0.817 | 0.761 | 0.788 | 21 | 8 | 0 | 1.25 |
| `c_base_before` base @0.30, ไม่มี curtain pass | **0.933** | 0.758 | 0.836 | 37 | 1 | 0 | 1.36 |
| **`c_base_nc_25` base @0.25, ไม่มี curtain pass** | 0.893 | 0.799 | **0.843** | **40** | 3 | 0 | 1.57 |
| `c_base_nc_20` base @0.20, ไม่มี curtain pass | 0.652 | 0.805 | 0.721 | 37 | 3 | 0 | 1.95 |
| `c_base_30` base @0.30 + curtain pass | 0.753 | 0.792 | 0.772 | 45 (เป็น curtain 39) | 37 | 30 | 2.06 |
| `c_base_25` / `c_base_35` base + curtain pass | 0.739 / 0.756 | 0.829 / 0.730 | 0.781 / 0.743 | 47 / 45 | 37 / 37 | 28 / 30 | 2.31 / 1.95 |

recall ต่อหมวดที่เปลี่ยนมาก (tiny เดิม → base@0.25 ไม่มี curtain pass):

| หมวด (n) | curtain (46) | blinds (6) | artwork (27) | vase (16) | mirror (7) | table_lamp (29) | coffee_table type acc. |
|---|---|---|---|---|---|---|---|
| `c_before` | 0.48 | 0.17 | 0.74 | 0.81 | 0.71 | 0.86 | 0.20 |
| `c_base_nc_25` | 0.83 | 0.33 | **0.56** | **0.69** | **0.57** | 0.90 | 0.83 |

ภาพรวม "usable" (การ์ดที่ product matching ใช้ได้): precision 0.742 → 0.798, recall 0.676 → 0.713

## คำตอบของข้อกังวลแต่ละข้อ

1. **นับจำนวนเฉยๆ จะให้รางวัลของปลอม** — ยืนยันแล้ว จำนวนการ์ดม่านของ `c_after_tiny` เพิ่ม 24 → 58 แต่ 23 ใบผิด
   (21 ใบไม่ตรงกับอะไรในเฉลยเลย) ถ้าดูแค่จำนวนจะคิดว่าดีขึ้นมาก ทั้งที่ F1 เท่าเดิม
2. **สาเหตุยังไม่ได้วัด** — วัดแล้ว ใน main pass ของ tiny มีคำดิบ `blinds window` 4 ครั้ง + `curtain blinds window`
   4 ครั้ง ทั้ง 8 โดน collapse เป็น `window` จริง แต่ม่านหาย 29 ชิ้น จึงเป็นสาเหตุรองเท่านั้น
   base เจอคำดิบแบบนี้แค่ 2 ครั้ง และให้กล่องม่านใน main pass 44 กล่อง (tiny ให้ 26)
3. **ทางแก้ถูกกว่า (canonicalize)** — ไม่ได้ผล ได้ม่าน 21 (น้อยกว่าเดิม) การ์ดผิด 8 ใบ เพราะกล่องที่
   คำดิบมี "window" ส่วนใหญ่เป็นหน้าต่างจริงที่มีม่าน ไม่ใช่กล่องม่าน
4. **เทียบ base ต้องลองหลาย threshold** — base@0.30 precision สูงสุดแต่ recall ต่ำ, @0.25 F1 ดีสุด, @0.20 ของปลอม
   ท่วม (P 0.652) ส่วน curtain pass ทำให้ base แย่ลงทุก threshold
5. **เรื่องในเอกสารเดิมที่ล้าสมัย**
   - diff ถูก commit แล้วใน e082afc → ตอนนี้ baseline ได้ด้วย flag `--no-curtain-pass` ไม่ต้องดึงไฟล์เก่าจาก git
   - เวลา: สคริปต์ใหม่รันภาพแรก 1 รอบเป็น warm-up แล้วค่อยจับเวลา ทุก run ข้างบนไม่รวมเวลาโหลดโมเดล
   - **curtain pass เพิ่มเวลา ~0.5 s/ภาพ (+43%)** บน tiny; base แทน tiny เพิ่มแค่ 0.14 s (@0.30) / 0.35 s (@0.25)
   - mime type: ตอนนี้เดาจากนามสกุลไฟล์แล้ว (เดิมเป็น jpeg เสมอ)
   - `sam2-mask-fixes.patch` ที่ root ของ repo: patch เก่า (2026-10-24) ที่ apply ไม่ได้ทั้งไปและกลับ ฟีเจอร์ในนั้น
     (`SAM_MASK_SCORE_MARGIN`, `compare_segmentation.py`, …) อยู่ในโค้ดแล้วแบบใหม่กว่า **ไม่ต้องใช้ ลบทิ้งได้**
6. **อื่นๆ**
   - **การ์ด `furniture`**: มาจาก curtain pass ทั้งหมด คือกล่องที่ DINO คืน text label ว่าง (`""`) 31 จาก 123 กล่องของ
     pass นี้ ถ้า force เป็น `curtain` ได้ม่านเพิ่ม 7 แต่การ์ดผิดเพิ่ม 8 ถ้าทิ้งก็เสียม่านที่เจอ 7 — ยังไงก็ไม่คุ้ม
   - **กล่องบนหน้าต่าง**: curtain-pass card ที่ทับกล่อง `window` ของ main pass (IoU > 0.5) โดยไม่มีม่านจาก main pass
     ซ้อนอยู่ มี 12 ใบ (tiny) ผิด 10 ถูก 2 ถ้าตัดทิ้ง F1 ขึ้นเป็น 0.811 แต่ยังแพ้ base
   - **ม่านคู่**: การ์ดม่านที่ "ไม่ตรงอะไรเลย" ส่วนที่ไม่ได้ทับหน้าต่าง (11 ใบใน tiny, 26 ใน base) ยังไม่ได้ดูด้วยตาว่า
     เป็นกล่องคลุมม่านทั้งคู่ที่ `drop_group_boxes` ตัดไม่ออก หรือเป็นอย่างอื่น — ดู gallery (ข้างล่าง)
   - **Gemini**: key ยังโดน 402 (เห็นตอนรัน unit test) ผลทั้งหมดข้างบนคือ DINO ล้วน ก่อนขึ้น production ต้องเติม
     key แล้วรัน `evaluate_segmentation.py predict` แบบเปิด Gemini อีกรอบ
   - unit test: `python -m unittest discover -s tests` ผ่านครบ 161 ตัว

## งานที่เหลือ

1. **ตัดสินใจเรื่องโค้ด**: (ก) ถอด curtain pass (e082afc) ออก (ข) เปลี่ยน default เป็น
   `GROUNDING_DINO_MODEL=IDEA-Research/grounding-dino-base` + `SAM_BOX_THRESHOLD=0.25` — ถ้าทำ (ข) ต้อง
   tune filter ที่ปรับมากับ tiny ใหม่ด้วย (`_MIN_AREA_PCT_BY_LABEL`, decoy, TV threshold ฯลฯ)
2. **หาว่าทำไม base เสีย artwork/mirror/vase**: main pass ของ base ให้กล่อง artwork 14 (tiny 26), mirror 4 (10),
   vase 11 (23) ลองดูว่าเป็นเพราะคะแนนต่ำกว่า threshold หรือโดน decoy (`window`/`tv`) แย่งไป อาจแก้ด้วย threshold
   รายหมวด หรือ pass แยกสำหรับของแขวนผนัง (แบบเดียวกับ opening pass)
3. **blinds** (n=6): tiny เดิม 1/6, base ไม่มี curtain pass 2/6, base + curtain pass 6/6 — pass มู่ลี่แยก (`"blinds ."` อย่างเดียว) อาจคุ้มถ้าคุมของปลอมได้ แต่ n น้อยเกินจะสรุป
4. ดู gallery ด้วยตาสำหรับการ์ดม่านผิด แล้วรันบนชุด ~70 ภาพไม่มีเฉลย (`D:\Downloads\images\images\` — **ไม่มีในเครื่องนี้แล้ว**
   ต้องหาโฟลเดอร์ใหม่) เพื่อเช็คว่าผลบน 24 ภาพไม่ใช่ความบังเอิญ
5. รันซ้ำแบบเปิด Gemini หลังเติม key

## วิธีรัน — `backend/curtain_experiment.py`

แทน `run_batch.py` เดิม (ไม่ต้อง copy สคริปต์จากเอกสารนี้แล้ว อยู่ใน repo) ทุก config เป็น flag ไม่ต้องแก้
`segmentation.py` หรือดึงไฟล์เก่าจาก git:

| flag | ความหมาย |
|---|---|
| `--no-curtain-pass` | ตัด curtain/blinds pass (= โค้ดก่อน e082afc) |
| `--canon-curtain` | canonicalize ไม่ collapse "curtain/blinds + window" เป็น window (ถ้าไม่มี door) |
| `--model` | `GROUNDING_DINO_MODEL` (default tiny) |
| `--box-threshold` / `--curtain-threshold` | `SAM_BOX_THRESHOLD` / `SAM_CURTAIN_BOX_THRESHOLD` |
| `--labeler` | `SEGMENTATION_LABELER` (default `none`) |
| `--limit N` | รันแค่ N ภาพแรก |

ผลไปที่ `backend/compare/eval/<tag>/` (gitignored) ในรูปแบบเดียวกับ `evaluate_segmentation.py` แถม
`_run.json` ที่เก็บ **ทุกกล่องดิบของ DINO แยกตาม pass (main/opening/curtain) พร้อมคำดิบก่อน
`canonicalize_label`**, การ์ดแต่ละใบมาจาก pass ไหน/คำดิบอะไร, และเวลาต่อภาพ

```powershell
cd backend
$GT = "C:\Users\PCHERO\Downloads\Smart Decorator Thesis.v1i.coco-segmentation\train"
$B  = "IDEA-Research/grounding-dino-base"
.\.venv\Scripts\python.exe curtain_experiment.py predict $GT --tag c_before --no-curtain-pass
.\.venv\Scripts\python.exe curtain_experiment.py predict $GT --tag c_after_tiny
.\.venv\Scripts\python.exe curtain_experiment.py predict $GT --tag c_canon --no-curtain-pass --canon-curtain
.\.venv\Scripts\python.exe curtain_experiment.py predict $GT --tag c_base_nc_25 --model $B --no-curtain-pass --box-threshold 0.25

# รายงานเฉพาะม่าน (+ gallery รูปตัดของการ์ดม่านทุกใบ ไว้ดูด้วยตา)
.\.venv\Scripts\python.exe curtain_experiment.py report $GT c_before c_after_tiny c_canon c_base_nc_25 `
  --json curtain.json --html curtain_gallery.html
# คะแนนรวม + รายหมวด
.\.venv\Scripts\python.exe evaluate_segmentation.py score $GT c_before c_after_tiny c_canon c_base_nc_25
```

ถ้า `DIR` ไม่มี `_annotations.coco.json` (ชุด ~70 ภาพดิบ) สคริปต์จะรันกับทุก .jpg/.png ในโฟลเดอร์ และ `report`
จะให้แค่จำนวน, pass ที่มา, ตัวนับ "กล่องม่านที่ทับหน้าต่าง", label ขยะ, เวลา และ gallery (ไม่มีคะแนน)

ทั้ง 24 ภาพใช้ ~1–2 วินาทีต่อภาพต่อ config บน RTX 5060 (ทุก config ข้างบนรวมกัน < 10 นาที) `grounding-dino-base`
อยู่ใน HF cache ของเครื่องนี้แล้ว

## อัปเดต 2026-11-24 (2): ทำไม base ทำของติดผนังหาย (ตอบงานที่เหลือข้อ 2)

**สาเหตุ: prompt หลักยาวเกินไปสำหรับ base ไม่ใช่ว่า base มองไม่เห็น** วัดคะแนนกล่องที่ดีที่สุดที่ทับชิ้นในเฉลย
(IoU ≥ 0.5) โดยรัน DINO ที่ threshold 0.05:

| หมวด (n) | tiny, prompt หลัก: ≥0.25 / ค่ากลาง | base, prompt หลัก: ≥0.25 / ค่ากลาง | base, prompt สั้นของติดผนัง: ≥0.25 / ค่ากลาง |
|---|---|---|---|
| artwork (27) | 26 / 0.42 | **19 / 0.30** | 26 / 0.43 |
| mirror (7) | 7 / 0.58 | 7 / 0.37 | 6 / 0.37 |
| vase (16) | 13 / 0.47 | 11 / 0.33 | 13 / 0.34 |

- base กระจายคะแนนไปทั่ว prompt ที่มี ~50 คำ คะแนนของ artwork/vase จึงตกลงมาใกล้ threshold และ 8 กล่องของ artwork
  ไม่มีคำไหนผ่าน text threshold เลย (label ว่าง → `furniture`)
- ชิ้นที่ tiny เจอแต่ base@0.25 หาย: artwork 5 + vase 2 = ไม่มีกล่องผ่าน threshold, mirror 1 = base เรียกว่า `window` (0.29)

**ทางแก้ที่ลอง: pass เสริมสำหรับของติดผนัง** (flag `--wall-pass` ใน `curtain_experiment.py`, คำพ้อง `painting`/
`picture frame` → `artwork`, กล่องที่ไม่มี label ทิ้ง) บน base@0.25, ไม่มี curtain pass:

| config | P | R | F1 | artwork | vase | mirror | table_lamp | usable P / R | s/ภาพ |
|---|---|---|---|---|---|---|---|---|---|
| `c_before` tiny (064a5ea) | 0.839 | 0.765 | 0.800 | 0.74 | 0.81 | 0.71 | 0.86 | 0.742 / 0.676 | 1.22 |
| `c_base_nc_25` base@0.25 | 0.893 | 0.799 | 0.843 | 0.56 | 0.69 | 0.57 | 0.90 | 0.798 / 0.713 | 1.57 |
| **`w_full_30`** + `"artwork . painting . picture frame . mirror . vase ."` @0.30 | 0.888 | 0.816 | **0.851** | 0.70 | 0.75 | 0.57 | 0.90 | 0.796 / 0.730 | 1.97 |
| `w_full_25` prompt เดียวกัน @0.25 | 0.870 | 0.819 | 0.844 | 0.78 | 0.81 | 0.57 | 0.83 | 0.779 / 0.734 | 2.01 |
| `w_short_25` + `"artwork . mirror . vase ."` @0.25 | 0.828 | 0.836 | 0.832 | 0.81 | 0.94 | 0.71 | 0.79 | 0.736 / 0.744 | 2.02 |
| `w_short_30` prompt เดียวกัน @0.30 | 0.853 | 0.833 | 0.843 | 0.74 | 0.94 | 0.71 | 0.86 | 0.759 / 0.741 | 2.01 |

ม่านเท่ากันทุกแถวของ base (0.83)

- `w_full_30`: การ์ดจาก wall pass ถูก 25 ใบ ผิด 3 ใบ (vase ทั้งหมด) และไม่ทำให้ชิ้นที่ base@0.25 เคยเจอหายไปเลย
- `w_short_25`: เจอเยอะสุด แต่มี vase ผิด 20 ใบ (ส่วนใหญ่คือฐานโคมไฟ) และโคมไฟตั้งโต๊ะหาย 3 ชิ้น เพราะการ์ด vase แย่ง
  pixel ฐานโคมไป mask ของโคมจึงไม่ผ่าน IoU 0.5 — prompt สั้น = ไม่มี `table lamp` ให้ DINO เลือก (ปัญหาแบบเดียวกับม่าน/หน้าต่าง)

**ข้อสรุป**: config ที่ดีที่สุดตอนนี้คือ base, main @0.25, ไม่มี curtain pass, + wall pass แบบเต็ม @0.30 → F1 0.800 → 0.851,
ม่าน 0.48 → 0.83, ยังเสีย artwork/vase/mirror นิดหน่อยเมื่อเทียบกับ tiny (0.74→0.70, 0.81→0.75, 5/7→4/7) เวลา 1.22 → 1.97 s/ภาพ

**ข้อควรระวัง**: config นี้เลือกบนชุด 24 ภาพเดียวกับที่ใช้วัดผล จึงมีโอกาสฟิตกับชุดนี้เกินจริง ต้องยืนยันบนชุดภาพอื่น (ชุด ~70 ภาพ
หรือ label เพิ่ม) ก่อนเปลี่ยน default ใน `segmentation.py`

```powershell
.\.venv\Scripts\python.exe curtain_experiment.py predict $GT --tag w_full_30 --model $B --no-curtain-pass `
  --box-threshold 0.25 --wall-pass "artwork . painting . picture frame . mirror . vase ." --wall-threshold 0.30
```

## สถานะโค้ด (2026-11-24, ยังไม่ commit)

`backend/segmentation.py` ตอนนี้:
- `GROUNDING_DINO_MODEL` default = `IDEA-Research/grounding-dino-base`, `SAM_BOX_THRESHOLD` default = 0.25
- ถอด curtain pass (e082afc) ออกแล้ว
- เพิ่ม wall pass: `_WALL_PROMPT = "artwork . painting . picture frame . mirror . vase ."` ที่ `_WALL_BOX_THRESHOLD = 0.30`
  ชื่อแปลงด้วย `wall_pass_label` (painting / picture frame → artwork, กล่องไม่มีชื่อทิ้ง) มีเทสใน `test_segmentation_helpers.py`

รันค่า default ใหม่บนชุดเฉลย (tag `e_wall_default`) ได้ตรงกับ `w_full_30` ทุกตัวเลข: F1 0.851, P 0.888, R 0.816, 2.01 s/ภาพ
unit test 162 ตัวผ่านหมด รายงานภาพเทียบ: https://claude.ai/artifact/VyVmt737rBvTZ4SYFP3n6k

## แก้กระจก/รูปภาพถูกรวมเข้าเตียง (2026-11-25, ยังไม่ commit)

`merge_parts_into_hosts` เดิมรวม `mirror`/`artwork`/`painting` ทุกชิ้นที่กล่องอยู่ในกล่องเตียงหรือโซฟา ≥ 85% เข้าไปเป็นหัวเตียงเสมอ
บนชุดเฉลย 24 ภาพ + 56 ห้อง (base) กฎนี้ทำงาน 10 ครั้ง **ผิดทั้ง 10**: กระจกบนผนังหลังเตียงสี่เสา, รูปเหนือหัวเตียงที่อยู่ในกล่องเตียง,
กรอบรูปบนตู้ข้างเตียง ทุกชิ้นกว้าง 0.06–0.15 ของความกว้างเตียง และ mask ไม่ทับเตียงเลย

ตอนนี้รวมเฉพาะเมื่อ host เป็นเตียง และ `looks_like_headboard`: กว้าง ≥ 0.25 ของกล่องเตียง และขอบบนห่างจากขอบบนกล่องเตียง ≤ 0.1
ของความสูงเตียง (หัวเตียงที่ DINO เรียก mirror ในเคสเดิม กว้าง ~0.3 และเริ่มที่ขอบบนเตียงพอดี — เทสเดิมยังผ่าน) เพิ่มเทส 2 ตัว

| | `e_wall_default` | `f_mirror_fix` |
|---|---|---|
| F1 | 0.851 | **0.865** |
| precision / recall | 0.888 / 0.816 | 0.891 / 0.840 |
| artwork | 19/27 | **24/27** |
| mirror | 4/7 | **6/7** |
| usable P / R | 0.796 / 0.730 | 0.801 / 0.754 |

เวลาต่อภาพวัดเทียบไม่ได้รอบนี้: มีโปรแกรมอื่นใช้ GPU ~67% ระหว่างรัน (กฎเดิมรันช่วงเดียวกันได้ 4.41 s ทั้งที่เมื่อวาน 2.01 s) ต้องวัดใหม่ตอน GPU ว่าง
