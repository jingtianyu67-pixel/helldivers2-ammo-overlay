# -*- coding: utf-8 -*-
"""量出弹匣外框的「上两个端点 + 底部两点 + 底部斜线」探针坐标（v2）。

姿态基准改用 0918_213544（满弹、外框最完整），排除桌面截图 220005。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ammo_detect import RegionDetector, largest_cc   # noqa: E402

AST = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "_out")
SHOTS = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
D = cfg["detect"]
WIN = np.load(os.path.join(AST, "interior_mask.npy"))

det = RegionDetector("B", WIN, None,
                     bg_margin=D["bg_margin"], lum_floor=D["lum_floor"],
                     chroma_min=D["chroma_min"], r_min=D["r_min"],
                     red_ratio=D.get("red_ratio"), edge=D["edge"],
                     shape_min_iou=D["shape_min_iou"],
                     red_shape_min_iou=D["red_shape_min_iou"],
                     min_mask_px=D["min_mask_px"], min_fill_px=D["min_fill_px"],
                     red_min_px=D["red_min_px"], near_dilate=D["near_dilate"])


def outer_sil(rgb, margin=12):
    a = rgb.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    e = max(1, min(5, rgb.shape[1] // 3))
    bg = np.median(np.concatenate([lum[:, :e], lum[:, -e:]], axis=1), axis=1)
    return largest_cc(np.abs(lum - bg[:, None]) > margin, 20)


samples = []
for fp in sorted(glob.glob(os.path.join(SHOTS, "*_1x.png"))):
    a = np.array(Image.open(fp).convert("RGB"))
    if a.shape[:2] == WIN.shape:
        samples.append((os.path.basename(fp), a))

REAL = [s for s in samples if not s[0].startswith("0918_2200")]
ref_name, ref_img = [s for s in REAL if "213544" in s[0]][0]
sil = outer_sil(ref_img)
ys, xs = np.nonzero(sil)
y0, y1 = int(ys.min()), int(ys.max())
top_xs = xs[ys <= y0 + 2]
print(f"姿态基准 {ref_name}: 外框 Y{y0}-{y1}  顶边 y{y0}~{y0 + 2} x{top_xs.min()}~{top_xs.max()}")

print("\n底部轮廓（每列最低点）:")
bot_pts = []
for x in range(int(xs.min()), int(xs.max()) + 1):
    col = np.nonzero(sil[:, x])[0]
    if len(col):
        bot_pts.append((x, int(col.max())))
for x, y in bot_pts:
    print(f"   x{x:>3} y{y}")

# --- 顶部两个端点 / 底部两个端点 -------------------------------------------- #
TL = (int(top_xs.min()), y0)
TR = (int(top_xs.max()), y0)
bx = [p[0] for p in bot_pts]
BL = (min(bx), max(p[1] for p in bot_pts if p[0] == min(bx)))
BR = (max(bx), max(p[1] for p in bot_pts if p[0] == max(bx)))
CORNERS = [("顶左", *TL), ("顶右", *TR), ("底左", *BL), ("底右", *BR)]
print(f"\n四角: {CORNERS}")

# 底部斜线探针：在底左~底右之间等距取点，y 取该列最低外框像素
xs_line = np.linspace(BL[0] + 1, BR[0] - 1, 4).round().astype(int)
SLANT = [(int(x), int(max(p[1] for p in bot_pts if p[0] == x))) for x in xs_line]
print(f"底斜线: {SLANT}")


def count(mask, x, y, r=1):
    return int(mask[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].sum())


print("\n=== 四角白色探针（3x3）===")
print(f"{'样本':<22}" + "".join(f"{n:>10}" for n, _, _ in CORNERS))
for name, a in samples:
    w, _ = det.extract(a)
    print(f"{name:<22}" + "".join(f"{count(w, x, y):>10}" for _, x, y in CORNERS))

print("\n=== 底斜线红色探针（3x3）===")
print(f"{'样本':<22}" + "".join(f"{x:>8}" for x, _ in SLANT))
for name, a in samples:
    _, rd = det.extract(a)
    print(f"{name:<22}" + "".join(f"{count(rd, x, y):>8}" for x, y in SLANT))

print("\n=== 底斜线白色计数（对照组）===")
print(f"{'样本':<22}" + "".join(f"{x:>8}" for x, _ in SLANT))
for name, a in samples:
    w, _ = det.extract(a)
    print(f"{name:<22}" + "".join(f"{count(w, x, y):>8}" for x, y in SLANT))

sc = 10
big = Image.fromarray(ref_img).resize((WIN.shape[1] * sc, WIN.shape[0] * sc), Image.NEAREST)
d = ImageDraw.Draw(big)
for label, x, y in CORNERS:
    d.rectangle([(x - 1) * sc, (y - 1) * sc, (x + 1) * sc + sc, (y + 1) * sc + sc],
                outline=(255, 0, 255))
    d.text((x * sc, y * sc - 14), label, fill=(255, 0, 255))
for i, (x, y) in enumerate(SLANT):
    d.rectangle([(x - 1) * sc, (y - 1) * sc, (x + 1) * sc + sc, (y + 1) * sc + sc],
                outline=(0, 255, 0))
    d.text((x * sc, y * sc + 14), f"R{i}", fill=(0, 255, 0))
big.save(os.path.join(OUT, "probe_map.png"))
print(f"\n探针图: {OUT}\\probe_map.png")

json.dump({"corners": [[n, x, y] for n, x, y in CORNERS], "slant": SLANT},
          open(os.path.join(OUT, "probe_candidates.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
