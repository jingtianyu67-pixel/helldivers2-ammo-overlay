# -*- coding: utf-8 -*-
"""诊断：为什么满弹被读成 70%+。

对用户刚贴的这张图：
  1. 报尺寸
  2. 用检测规则提掩膜，量出图标真实包围盒
  3. 与 B 组窗口掩膜 (55x40 / 内部 526px) 对比
  4. 把窗口叠到图上画出来，看差在哪
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ammo_detect import (RegionDetector, build_detector, largest_cc,   # noqa: E402
                         convex_hull_mask, normalize_shape, iou, erode, dilate)

SRC = r"C:\Users\Administrator\.workbuddy\clipboard-images\clipboard-2026-09-18T14-04-46-667Z-3b92caaf.png"
OUT = os.path.join(HERE, "_out")
os.makedirs(OUT, exist_ok=True)

im = Image.open(SRC).convert("RGB")
a = np.array(im)
print(f"图尺寸: {im.size[0]}x{im.size[1]}  shape={a.shape}")

# --- 用 B 组检测器的提膜逻辑 ------------------------------------------------ #
bwin = np.load(os.path.join(HERE, "assets", "interior_mask.npy"))
print(f"B 掩膜: {bwin.shape}  内部像素 {int(bwin.sum())}")

cfg = __import__("json").load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
D = cfg["detect"]
det = RegionDetector("B", bwin, None,
                     bg_margin=D["bg_margin"], lum_floor=D["lum_floor"],
                     chroma_min=D["chroma_min"], r_min=D["r_min"],
                     red_ratio=D.get("red_ratio"), edge=D["edge"],
                     shape_min_iou=D["shape_min_iou"],
                     red_shape_min_iou=D["red_shape_min_iou"],
                     min_mask_px=D["min_mask_px"], min_fill_px=D["min_fill_px"],
                     red_min_px=D["red_min_px"], near_dilate=D["near_dilate"])

white, red = det.extract(a)
hot = white | red
print(f"\n提膜: white={int(white.sum())}px  red={int(red.sum())}px")

cc = largest_cc(hot, 1)
ys, xs = np.nonzero(cc)
if len(ys):
    print(f"最大连通域 bbox: x {xs.min()}-{xs.max()} ({xs.max()-xs.min()+1}px)  "
          f"y {ys.min()}-{ys.max()} ({ys.max()-ys.min()+1}px)  面积 {int(cc.sum())}")
    hull = convex_hull_mask(cc)
    print(f"凸包面积 {int(hull.sum())}  归一化形状 IoU vs 模板 = "
          f"{iou(normalize_shape(hull), det.template):.3f}")

# --- 若图正好是 B 区域切片，直接跑检测器 ------------------------------------- #
if a.shape[:2] == bwin.shape:
    r = det.analyze(a)
    print(f"\n[当作 B 区域切片] valid={r.valid} {r.fraction*100:.1f}% {r.state} "
          f"fill={r.fill}/{r.area} IoU={r.shape_iou:.2f} {r.reason}")
else:
    print(f"\n尺寸 {a.shape[1]}x{a.shape[0]} ≠ B 区域 {bwin.shape[1]}x{bwin.shape[0]}，"
          f"不是 B 区域的原尺寸切片")

# --- 叠加可视化 -------------------------------------------------------------- #
sc = max(1, 400 // max(1, max(im.size)))
big = im.resize((im.size[0] * sc * 6, im.size[1] * sc * 6), Image.NEAREST)
d = ImageDraw.Draw(big)
for y, x in zip(*np.nonzero(hot)):
    d.point((x * sc * 6 + 1, y * sc * 6 + 1), fill=(0, 255, 255))
    d.point((x * sc * 6, y * sc * 6), fill=(0, 255, 255))
big.save(os.path.join(OUT, "why70_hot.png"))

ov = im.copy()
arr = np.array(ov)
arr[hot] = (arr[hot] * 0.4 + np.array([0, 255, 255]) * 0.6).astype(np.uint8)
Image.fromarray(arr).resize((im.size[0] * 4, im.size[1] * 4), Image.NEAREST).save(
    os.path.join(OUT, "why70_overlay.png"))
print(f"\n输出: {OUT}\\why70_overlay.png / why70_hot.png")
