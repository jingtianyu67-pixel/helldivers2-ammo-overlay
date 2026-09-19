# -*- coding: utf-8 -*-
"""A 区域：实测白/红探针候选 + 生成窗口掩膜。"""
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
files = {os.path.basename(f)[:3]: f for f in sorted(glob.glob(os.path.join(D, "*_1x.png")))}

WHITE_GRP = ["A01", "A02", "A03", "A14", "A15"]      # 白色弹匣（半满+满）
RED_GRP = ["A05", "A06", "A07"]                       # 红色低弹
EMPTY_GRP = ["A08", "A09", "A10", "A11", "A13"]       # 空弹匣
BAD_GRP = ["A12"]                                     # 极暗，单独看

CACHE = {}
for t, fp in files.items():
    a = np.array(Image.open(fp).convert("RGB"))
    CACHE[t] = det.extract(a)


def probe(x, y, rad=1):
    xs = slice(max(0, x - rad), x + rad + 1)
    ys = slice(max(0, y - rad), y + rad + 1)
    out = []
    for nm, grp, idx in (("白", WHITE_GRP, 0), ("红", RED_GRP, 1), ("空", EMPTY_GRP, 0)):
        wsum = sum(int(CACHE[t][0][ys, xs].sum()) for t in grp)
        rsum = sum(int(CACHE[t][1][ys, xs].sum()) for t in grp)
        out.append(f"{nm}组({len(grp)}) W{wsum:>3} R{rsum:>3}")
    return f"({x:>2},{y:>2}) " + " | ".join(out)


print("=== 上部端点候选 ===")
for pt in [(23, 4), (31, 4), (23, 5), (31, 5), (24, 5), (32, 5),
           (23, 7), (38, 7), (23, 8), (38, 8), (24, 3), (30, 3)]:
    print("  " + probe(*pt))

print("\n=== 底部端点候选 ===")
for pt in [(29, 35), (45, 35), (30, 36), (44, 36), (29, 37), (43, 37),
           (31, 39), (40, 39), (32, 40), (39, 40), (33, 41), (36, 41)]:
    print("  " + probe(*pt))

print("\n=== 红探针（底斜线/底部填充）候选 ===")
for pt in [(37, 34), (38, 35), (39, 36), (40, 34), (34, 36),
           (36, 35), (41, 35), (35, 36), (33, 34), (38, 37)]:
    print("  " + probe(*pt))

# ---------- 生成窗口掩膜 ----------
print("\n=== 生成窗口掩膜 ===")
acc = None
for t in ("A14", "A15"):
    w, r = CACHE[t]
    cc = largest_cc(w | r, 20)
    acc = cc if acc is None else (acc | cc)
sil = convex_hull_mask(acc)
for e in (1, 2, 3):
    win = erode(sil, e)
    print(f"  腐蚀 {e}: 剪影 {int(sil.sum())}px → 窗口 {int(win.sum())}px")
