# -*- coding: utf-8 -*-
"""复现空弹：把实机满弹帧的弹匣内部填充抹掉、只留外框轮廓，看现行链路判什么。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from PIL import Image

import ammo_detect as ad

cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
det = ad.build_detector(cfg, Path("."))
spec = det.specs[0]
d = spec.detector
print("区域", spec.name, spec.box)
print("window.sum() =", int(d.window.sum()), " mask_floor =", d.mask_floor,
      " shape_min_iou =", d.shape_min_iou)

im = Image.open("_out/prod_screen.png").convert("RGB")
k = im.width / 2560.0
l, t, r_, b_ = spec.box
box = (int(round(l * k)), int(round(t * k)), int(round(r_ * k)), int(round(b_ * k)))
rgb = np.array(im.crop(box))

print("\n=== 原帧（满弹）===")
res = d.analyze(rgb)
print(f"  valid={res.valid} {res} state={res.state} white={res.white_px} "
      f"body={res.area} fill={res.fill} IoU={res.shape_iou:.2f} hint={res.hint}")

# 构造「空弹壳」：把弹匣体内的白像素替换成周围背景色（内部无填充）
white = d.extract(rgb)
hot = ad.largest_cc(white, d.min_mask_px)
hull = ad.convex_hull_mask(hot)
body = ad.erode(hull, d.fill_inset)
inset_white = white & body
shell = rgb.copy()
# 内部填充去掉：用该区域的暗背景（取窗口内非白像素的中位色）填充
mask_bg = d.window & ~white if d.window is not None else ~white
fill_col = np.median(rgb[mask_bg], axis=0) if mask_bg.sum() > 10 else np.array([60, 70, 72])
shell[inset_white] = fill_col.astype(np.uint8)
print(f"\n  构造空壳：抹掉 body 内 {int(inset_white.sum())}px，填色 {fill_col.astype(int)}")
Image.fromarray(shell).resize((shell.shape[1] * 8, shell.shape[0] * 8),
                             Image.NEAREST).save("_out/shell_A_8x.png")

print("\n=== 构造的空弹壳 ===")
res2 = d.analyze(shell)
print(f"  valid={res2.valid} {res2} state={res2.state} white={res2.white_px} "
      f"body={res2.area} fill={res2.fill} IoU={res2.shape_iou:.2f} "
      f"hint={res2.hint} reason={res2.reason}")

sw = d.extract(shell)
print(f"\n  空壳里白掩膜 = {int(sw.sum())}px（门槛 mask_floor={d.mask_floor}）")
hot2 = ad.largest_cc(sw, d.min_mask_px)
print(f"  最大连通域 = {int(hot2.sum())}px，凸包 = {int(ad.convex_hull_mask(hot2).sum())}px")
if hot2.sum() > 0:
    print(f"  形状 IoU = {ad.iou(ad.normalize_shape(ad.convex_hull_mask(hot2)), d.template):.3f}")
    print(f"  落在窗口邻域内 = {bool((hot2 & ad.dilate(d.window, d.near_dilate)).any())}")

# 用户给的那张空弹小图，缩放到本区域尺度再跑一遍
P = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images"
         r"\clipboard-2026-09-18T16-28-11-427Z-403561d7.png")
u = Image.open(P).convert("RGB")
print(f"\n=== 用户空弹样本 {u.size} → 缩放到 {rgb.shape[1]}x{rgb.shape[0]} ===")
u2 = np.array(u.resize((rgb.shape[1], rgb.shape[0]), Image.LANCZOS))
res3 = d.analyze(u2)
print(f"  valid={res3.valid} {res3} state={res3.state} white={res3.white_px} "
      f"body={res3.area} fill={res3.fill} IoU={res3.shape_iou:.2f} "
      f"hint={res3.hint} reason={res3.reason}")
