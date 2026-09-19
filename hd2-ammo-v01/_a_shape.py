# -*- coding: utf-8 -*-
"""A 区域形状 IoU 测：验证凸包形状能否区分弹匣图标与枪身剪影。"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ammo_detect import (RegionDetector, convex_hull_mask, erode, iou,
                         largest_cc, normalize_shape)

D = r"C:\Users\Administrator\Desktop\hd2_ammo_shots\A"
B = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
cfg = json.load(open("config.json", encoding="utf-8"))
DV = cfg["detect"]
EK = ("bg_margin", "lum_floor", "chroma_min", "r_min", "red_ratio", "edge")
det = RegionDetector("A", None, None, **{k: DV[k] for k in EK})
PW = [(23, 5), (38, 8), (29, 35), (45, 35), (23, 16), (40, 22)]
PR = [(37, 34), (38, 35), (35, 36), (40, 34)]


def stats(a):
    w, r = det.extract(a)
    hot = largest_cc(w | r, 20)
    hred = largest_cc(r, 8)
    hull = convex_hull_mask(hot if hot.sum() >= 30 else hred)
    v = iou(normalize_shape(hull), det.template)
    wh = sum(int(w[max(0, y - 1):y + 2, max(0, x - 1):x + 2].sum()) for x, y in PW)
    rr = sum(int(r[max(0, y - 1):y + 2, max(0, x - 1):x + 2].sum()) for x, y in PR)
    return int(hot.sum()), int(hred.sum()), wh, rr, v


print("=== 正样本（A 区域）===")
print(f"{'样本':<5} {'掩膜':>6} {'红':>4} {'白探':>5} {'红探':>5} {'形状IoU':>7}")
pos = []
for fp in sorted(glob.glob(os.path.join(D, "*_1x.png"))):
    tag = os.path.basename(fp)[:3]
    a = np.array(Image.open(fp).convert("RGB"))
    m, rp, wh, rr, v = stats(a)
    pos.append((tag, v))
    print(f"{tag:<5} {m:>6} {rp:>4} {wh:>5} {rr:>5} {v:>7.2f}")

print("\n=== 负样本（枪身剪影 / 桌面壁纸）===")
neg = []
for fp in sorted(glob.glob(os.path.join(B, "*_full.png"))):
    im = np.array(Image.open(fp).convert("RGB"))
    a = im[1250:1300, 330:390].copy()
    a[:3, :] = 0
    a[-3:, :] = 0
    a[:, :3] = 0
    a[:, -3:] = 0
    tag = os.path.basename(fp)[5:11]
    m, rp, wh, rr, v = stats(a)
    neg.append((tag, v))
    print(f"{tag:<5} {m:>6} {rp:>4} {wh:>5} {rr:>5} {v:>7.2f}")

pv = [v for _, v in pos]
nv = [v for _, v in neg]
print(f"\n正样本 IoU: min={min(pv):.2f} 中位={np.median(pv):.2f}  max={max(pv):.2f}")
print(f"负样本 IoU: min={min(nv):.2f} 中位={np.median(nv):.2f}  max={max(nv):.2f}")
