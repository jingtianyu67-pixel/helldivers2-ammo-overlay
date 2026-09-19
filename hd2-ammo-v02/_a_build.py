# -*- coding: utf-8 -*-
"""A 区域标定：生成窗口掩膜 + 验证探针分离度。"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ammo_detect as A
from ammo_detect import RegionDetector, convex_hull_mask, erode, largest_cc

D = r"C:\Users\Administrator\Desktop\hd2_ammo_shots\A"
B_DIR = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
cfg = json.load(open("config.json", encoding="utf-8"))
DV = cfg["detect"]

# ---------- 1. 窗口掩膜 ----------
raw = A.RegionDetector("tmp", None, None, **{k: DV[k] for k in
                                            ("bg_margin", "lum_floor", "chroma_min",
                                             "r_min", "red_ratio", "edge")})
acc = None
for t in ("A14", "A15"):
    fp = glob.glob(os.path.join(D, t + "_*_1x.png"))[0]
    a = np.array(Image.open(fp).convert("RGB"))
    w, r = raw.extract(a)
    cc = largest_cc(w | r, 20)
    acc = cc if acc is None else (acc | cc)
sil = convex_hull_mask(acc)
WIN = erode(sil, 2)
np.save("assets/a_interior_mask.npy", WIN)
ys, xs = np.nonzero(WIN)
print(f"A 窗口掩膜: {int(WIN.sum())}px  bbox x{xs.min()}-{xs.max()} y{ys.min()}-{ys.max()}")

# ---------- 2. 探针 ----------
PW = [(23, 5), (38, 8), (29, 35), (45, 35), (23, 16), (40, 22)]
PR = [(37, 34), (38, 35), (35, 36), (40, 34)]

kw = {k: DV[k] for k in ("bg_margin", "lum_floor", "chroma_min", "r_min", "red_ratio",
                         "edge", "shape_min_iou", "red_shape_min_iou", "min_mask_px",
                         "min_fill_px", "red_min_px", "near_dilate", "snap_full")}
det = RegionDetector("A", WIN, None, probe_white=PW, probe_red=PR,
                     probe_radius=1, white_need=24, red_need=10, **kw)
det.WhiteList, det.RedList = PW, PR


def analyze(a):
    return det.analyze(a)


def hit(a):
    """返回白/红探针原始命中数。"""
    w, r = det.extract(a)
    rad = 1
    wh = rr = 0
    for x, y in PW:
        wh += int(w[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1].sum())
    for x, y in PR:
        rr += int(r[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1].sum())
    return wh, rr


print(f"\n白探针 {len(PW)} 点(满分 {len(PW)*9}) | 红探针 {len(PR)} 点(满分 {len(PR)*9})")
print("\n=== A 区域 15 张样本 ===")
for fp in sorted(glob.glob(os.path.join(D, "*_1x.png"))):
    tag = os.path.basename(fp)[:3]
    a = np.array(Image.open(fp).convert("RGB"))
    wh, rr = hit(a)
    res = analyze(a)
    print(f"  {tag}: 白{wh:>3} 红{rr:>3} | valid={int(res.valid)} "
          f"{res.fraction * 100:5.1f}% {res.state:5s} hint={res.hint:12s} "
          f"fill={res.fill:>3}/{res.area} {res.reason}")

print("\n=== 桌面/游戏内截图切 A 区域（负样本，应拒绝）===")
for fp in sorted(glob.glob(os.path.join(B_DIR, "*_full.png"))):
    im = np.array(Image.open(fp).convert("RGB"))
    l, t, r_, b_ = 330, 1250, 390, 1300
    P = 4
    a = im[t + P:b_ - P, l + P:r_ - P]
    wh, rr = hit(a)
    res = analyze(a)
    print(f"  {os.path.basename(fp)}: 白{wh:>3} 红{rr:>3} | valid={int(res.valid)} "
          f"{res.fraction * 100:5.1f}% {res.state:5s} "
          f"{'✓ 误报' if res.valid else '× 正确拒绝'} {res.reason}")
