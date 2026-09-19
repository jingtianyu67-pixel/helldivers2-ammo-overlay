# -*- coding: utf-8 -*-
"""决定性测量：把你这两张图还原到生产区域坐标系，看现有检测器到底读多少。

方法：把裁剪图缩放到生产尺度（图标高 40px），再按「窗口质心对齐」放进 40x55 的画布，
得到的就是 live_ammo.py 实际会看到的那块图。然后跑真检测器。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ammo_detect import (RegionDetector, convex_hull_mask, erode,   # noqa: E402
                        largest_cc)

CROPS = [
    ("近满", r"C:\Users\Administrator\.workbuddy\clipboard-images\clipboard-2026-09-18T14-06-45-451Z-68eb550e.png"),
    ("较少", r"C:\Users\Administrator\.workbuddy\clipboard-images\clipboard-2026-09-18T14-06-45-459Z-7e2a1a63.png"),
]
ICON_H = 40

cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
D = cfg["detect"]
WIN = np.load(os.path.join(HERE, "assets", "interior_mask.npy"))
print(f"生产窗口 {WIN.shape} 面积 {int(WIN.sum())}")

det = RegionDetector("B", WIN, None,
                     bg_margin=D["bg_margin"], lum_floor=D["lum_floor"],
                     chroma_min=D["chroma_min"], r_min=D["r_min"],
                     red_ratio=D.get("red_ratio"), edge=D["edge"],
                     shape_min_iou=D["shape_min_iou"],
                     red_shape_min_iou=D["red_shape_min_iou"],
                     min_mask_px=D["min_mask_px"], min_fill_px=D["min_fill_px"],
                     red_min_px=D["red_min_px"], near_dilate=D["near_dilate"])

# 窗口质心（生产坐标系）
Wy, Wx = np.nonzero(WIN)
wcy, wcx = Wy.mean(), Wx.mean()

rows = []
for label, fp in CROPS:
    im = Image.open(fp).convert("RGB")
    a = np.array(im)
    white, red = det.extract(a)
    hot = white | red
    cc = largest_cc(hot, 1)
    ys, xs = np.nonzero(cc)
    h = ys.max() - ys.min() + 1
    s = ICON_H / h
    new_size = (max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s)))
    im2 = im.resize(new_size, Image.LANCZOS)
    a2 = np.array(im2)
    b = det.extract(a2)
    hot2 = b[0] | b[1]
    cc2 = largest_cc(hot2, 1)
    ys2, xs2 = np.nonzero(cc2)
    k = max(1, round(2 * (ys2.max() - ys2.min() + 1) / ICON_H))
    win2 = erode(convex_hull_mask(cc2), k)
    w2y, w2x = np.nonzero(win2)
    ccy, ccx = w2y.mean(), w2x.mean()

    # 按质心对齐，把缩放后的图平移进 40x55 画布
    canvas = np.zeros((WIN.shape[0], WIN.shape[1], 3), np.uint8)
    oy = int(round(wcy - ccy))
    ox = int(round(wcx - ccx))
    for y in range(canvas.shape[0]):
        sy = y - oy
        if not (0 <= sy < a2.shape[0]):
            continue
        for x in range(canvas.shape[1]):
            sx = x - ox
            if 0 <= sx < a2.shape[1]:
                canvas[y, x] = a2[sy, sx]

    r = det.analyze(canvas)
    rows.append((label, r))
    print(f"\n{label}  缩放 {h}→{ICON_H}px (x{s:.3f})  平移到 ({ox},{oy})")
    print(f"   → 真检测器读数: valid={r.valid} {r.fraction * 100:5.1f}% state={r.state} "
          f"fill={r.fill}/{r.area} IoU={r.shape_iou:.2f} {r.reason}")
    Image.fromarray(canvas).resize((WIN.shape[1] * 8, WIN.shape[0] * 8), Image.NEAREST).save(
        os.path.join(HERE, "_out", f"align_{label}.png"))

print("\n" + "=" * 62)
if len(rows) == 2 and all(r.valid for _, r in rows):
    lo = min(r.fraction for _, r in rows)
    hi = max(r.fraction for _, r in rows)
    print(f"两图生产读数: {rows[0][0]}={rows[0][1].fraction * 100:.1f}%  "
          f"{rows[1][0]}={rows[1][1].fraction * 100:.1f}%")
    print(f"→ snap 阈值应落在 ({lo:.3f}, {hi:.3f}] 之间，建议 "
          f"{(lo + hi) / 2:.2f}，或取整 {round((lo + hi) / 2 * 100) / 100}")
