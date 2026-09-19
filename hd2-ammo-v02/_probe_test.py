# -*- coding: utf-8 -*-
"""候选探针点逐张实测。

候选来自 0918_213544（满弹）与 213537（红低弹）的逐行白/红分布：
  顶边 y8~11  x10~20        底边 y42~45  x17~30
  红低弹底斜线 y44~46  x19~24
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ammo_detect import RegionDetector   # noqa: E402

AST = os.path.join(HERE, "assets")
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

# 候选：四角（顶左 顶右 底左 底右）+ 左右中框（可选加固）
CORNERS = [("顶左", 11, 10), ("顶右", 19, 10), ("底左", 18, 43), ("底右", 29, 43)]
SIDES = [("左中", 12, 28), ("右中", 29, 33)]
SLANT = [(20, 45), (22, 46), (25, 44), (23, 45)]

samples = []
for fp in sorted(glob.glob(os.path.join(SHOTS, "*_1x.png"))):
    a = np.array(Image.open(fp).convert("RGB"))
    if a.shape[:2] == WIN.shape:
        samples.append((os.path.basename(fp)[5:11], a))


def cnt(m, x, y, r):
    return int(m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].sum())


for r in (1, 2):
    print(f"\n{'=' * 78}\n半径 r={r}（patch {(2 * r + 1)}x{(2 * r + 1)}）")
    print(f"{'样本':<10}" + "".join(f"{n:>8}" for n, _, _ in CORNERS + SIDES)
          + f"{'白合计':>9}" + "".join(f"{str(p):>10}" for p in SLANT) + f"{'红合计':>9}")
    for name, a in samples:
        w, rd = det.extract(a)
        wc = [cnt(w, x, y, r) for _, x, y in CORNERS]
        ws = [cnt(w, x, y, r) for _, x, y in SIDES]
        rc = [cnt(rd, x, y, r) for x, y in SLANT]
        print(f"{name:<10}" + "".join(f"{c:>8}" for c in wc + ws)
              + f"{sum(wc + ws):>9}" + "".join(f"{c:>10}" for c in rc)
              + f"{sum(rc):>9}")
