# -*- coding: utf-8 -*-
"""A 区域样本像素级定位：外框 bbox、逐行范围、填充区形状。"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ammo_detect import (build_detector, convex_hull_mask, dilate, erode,
                         largest_cc)

D = r"C:\Users\Administrator\Desktop\hd2_ammo_shots\A"
cfg = json.load(open("config.json", encoding="utf-8"))
det = [s for s in build_detector(cfg).specs if s.name == "A"][0].detector

files = sorted(glob.glob(os.path.join(D, "*_1x.png")))
print(f"样本 {len(files)} 张，区域 {files and Image.open(files[0]).size}\n")

for fp in files:
    tag = os.path.basename(fp)[:3]
    a = np.array(Image.open(fp).convert("RGB"))
    w, r = det.extract(a)
    hot = w | r
    cc = largest_cc(hot, 12)
    if cc.sum() == 0:
        print(f"{tag}: 无掩膜  white={int(w.sum())} red={int(r.sum())}")
        continue
    ys, xs = np.nonzero(cc)
    print(f"{tag}: hot={int(cc.sum()):4d}px 白={int(w.sum()):4d} 红={int(r.sum()):4d}  "
          f"bbox x{xs.min()}-{xs.max()} y{ys.min()}-{ys.max()}  "
          f"({xs.max()-xs.min()+1}x{ys.max()-ys.min()+1})")

print("\n=== 满弹样本 A14/A15 白色填充区逐行 ===")
for tag in ("A14", "A15"):
    fp = [f for f in files if os.path.basename(f).startswith(tag)][0]
    a = np.array(Image.open(fp).convert("RGB"))
    w, r = det.extract(a)
    cc = largest_cc(w | r, 12)
    ys, xs = np.nonzero(cc)
    print(f"-- {tag}  (填充 {int(cc.sum())}px)")
    for y in range(ys.min(), ys.max() + 1):
        row = np.nonzero(cc[y])[0]
        if len(row):
            print(f"   y{y:>3}: x{row.min():>3}-{row.max():>3}  n={len(row)}")

print("\n=== 低弹红色样本 A09/A11 红色掩膜逐行 ===")
for tag in ("A09", "A11"):
    fp = [f for f in files if os.path.basename(f).startswith(tag)][0]
    a = np.array(Image.open(fp).convert("RGB"))
    w, r = det.extract(a)
    cc = largest_cc(r, 8)
    ys, xs = np.nonzero(cc)
    print(f"-- {tag}  (红 {int(cc.sum())}px) bbox x{xs.min()}-{xs.max()} y{ys.min()}-{ys.max()}")
    for y in range(ys.min(), ys.max() + 1):
        row = np.nonzero(cc[y])[0]
        if len(row):
            print(f"   y{y:>3}: x{row.min():>3}-{row.max():>3}  n={len(row)}")
