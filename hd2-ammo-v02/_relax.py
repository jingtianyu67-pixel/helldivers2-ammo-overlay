# -*- coding: utf-8 -*-
"""放宽阈值方案测算。

用户要求：接近满的（图1）显示 100%；只有像图2 那样明显少了才开始报百分比。

做法：把两张图当作真值样本，量出
  a) 满弹时弹匣内部窗口实际被填了多少（→ 修正分母 full_ref）
  b) 填充的「高度进度」是否比面积占比更直观
  c) 套回既有 6 张样张，看会不会把别的武器读数改坏
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
from ammo_detect import (RegionDetector, convex_hull_mask, erode,   # noqa: E402
                        largest_cc)

CROPS = [
    ("近满(应=100%)", r"C:\Users\Administrator\.workbuddy\clipboard-images\clipboard-2026-09-18T14-06-45-451Z-68eb550e.png"),
    ("较少(应<100%)", r"C:\Users\Administrator\.workbuddy\clipboard-images\clipboard-2026-09-18T14-06-45-459Z-7e2a1a63.png"),
]
SHOTS = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"

cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
D = cfg["detect"]
PROD_WIN = np.load(os.path.join(HERE, "assets", "interior_mask.npy"))
PROD_AREA = int(PROD_WIN.sum())
ICON_H = 40          # 生产尺度下图标高度（由视频 18x30 x1.3333 推得）

det = RegionDetector("X", None, None,
                     bg_margin=D["bg_margin"], lum_floor=D["lum_floor"],
                     chroma_min=D["chroma_min"], r_min=D["r_min"],
                     red_ratio=D.get("red_ratio"), edge=D["edge"],
                     shape_min_iou=D["shape_min_iou"],
                     red_shape_min_iou=D["red_shape_min_iou"],
                     min_mask_px=D["min_mask_px"], min_fill_px=D["min_fill_px"],
                     red_min_px=D["red_min_px"], near_dilate=D["near_dilate"])


def measure(rgb):
    """返回 (window, fill, 高度进度) —— 全部尺度无关。"""
    white, red = det.extract(rgb)
    hot = white | red
    cc = largest_cc(hot, 1)
    ys, xs = np.nonzero(cc)
    h_icon = ys.max() - ys.min() + 1
    k = max(1, round(2 * h_icon / ICON_H))
    hull = convex_hull_mask(cc)
    win = erode(hull, k)
    fill = int((hot & win).sum())
    area = int(win.sum())
    # 高度进度：填充最高处 vs 窗口顶/底
    wys, wxs = np.nonzero(win)
    top, bot = wys.min(), wys.max()
    fill_ys, _ = np.nonzero(hot & win)
    ftop = fill_ys.min() if len(fill_ys) else bot
    vprog = (bot - ftop) / max(1, bot - top)
    return dict(h_icon=h_icon, k=k, win=win, area=area, fill=fill,
                ratio=fill / max(1, area), vprog=vprog, hot=hot)


print(f"生产窗口面积 {PROD_AREA} px（分母）")
print("=" * 70)
res = {}
for label, fp in CROPS:
    a = np.array(Image.open(fp).convert("RGB"))
    m = measure(a)
    res[label] = m
    print(f"{label}  {a.shape[1]}x{a.shape[0]}  图标高 {m['h_icon']}px  erode k={m['k']}")
    print(f"    窗口 {m['area']}px  填充 {m['fill']}px  → 面积占比 {m['ratio'] * 100:.1f}%"
          f"   高度进度 {m['vprog'] * 100:.1f}%")
    cap = int((m["win"] & ~m["hot"]).sum())
    print(f"    窗口内未填充（顶部空腔）{cap}px = 占窗口 {cap / m['area'] * 100:.1f}%")
    # 可视化
    ov = a.copy().astype(np.uint8)
    ov[m["win"]] = (ov[m["win"]] * 0.45 + np.array([255, 0, 255]) * 0.55)
    ov[m["hot"] & m["win"]] = (ov[m["hot"] & m["win"]] * 0.25 + np.array([0, 255, 0]) * 0.75)
    Image.fromarray(ov).resize((a.shape[1] * 6, a.shape[0] * 6), Image.NEAREST).save(
        os.path.join(HERE, "_out", f"relax_{label[:2]}.png"))
    print()

near = res["近满(应=100%)"]
new_full = round(near["ratio"] * PROD_AREA)
print("=" * 70)
print(f"方案 A：把 full_ref 从 {PROD_AREA} 改成 {new_full}"
      f"（= 满弹时窗口填充量）→ 满弹读 {new_full / PROD_AREA * 100:.0f}%x 归一后 = 100%")
print(f"方案 B：加 snap 死区，fraction >= {near['ratio']:.2f} 直接当 100%\n")

print("套回既有样张（B 区域 40x55）:")
det_b = RegionDetector("B", PROD_WIN, None,
                       bg_margin=D["bg_margin"], lum_floor=D["lum_floor"],
                       chroma_min=D["chroma_min"], r_min=D["r_min"],
                       red_ratio=D.get("red_ratio"), edge=D["edge"],
                       shape_min_iou=D["shape_min_iou"],
                       red_shape_min_iou=D["red_shape_min_iou"],
                       min_mask_px=D["min_mask_px"], min_fill_px=D["min_fill_px"],
                       red_min_px=D["red_min_px"], near_dilate=D["near_dilate"])
for fp in sorted(glob.glob(os.path.join(SHOTS, "*_1x.png"))):
    a = np.array(Image.open(fp).convert("RGB"))
    if a.shape[:2] != PROD_WIN.shape:
        continue
    r = det_b.analyze(a)
    old = r.fraction
    new = min(1.0, r.fill / new_full) if r.valid else 0.0
    snap = 1.0 if new >= near["ratio"] else new
    print(f"  {os.path.basename(fp)}: fill={r.fill:4d}  "
          f"旧 {old * 100:5.1f}%  →  新 {new * 100:5.1f}%  →  snap后 {snap * 100:5.1f}%")
