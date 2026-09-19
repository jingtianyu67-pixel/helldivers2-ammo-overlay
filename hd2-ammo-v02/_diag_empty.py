# -*- coding: utf-8 -*-
"""空弹样本（红色描边、内部空）的量测：白通道 / 硬红 / 软红分别能捞到多少像素。"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from PIL import Image

import ammo_detect as ad

P = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images"
         r"\clipboard-2026-09-18T16-28-11-427Z-403561d7.png")
im = Image.open(P).convert("RGB")
print("尺寸", im.size)
a = np.array(im).astype(np.int16)
r, g, b = a[..., 0], a[..., 1], a[..., 2]
chroma = r - np.maximum(g, b)

print("\n== 全图统计 ==")
print("R 分位", [int(np.percentile(r, q)) for q in (50, 90, 99, 100)])
print("chroma(R-maxGB) 分位", [int(np.percentile(chroma, q)) for q in (50, 90, 99, 100)])
bg = np.median(np.concatenate([a[:, :5].reshape(-1, 3), a[:, -5:].reshape(-1, 3)], 0), 0)
print("背景中位 RGB", bg)

print("\n== 各阈值下的掩膜 ==")
for tag, m in [
    ("硬红 R-max>=55 & R>=120", (chroma >= 55) & (r >= 120)),
    ("软红 R-max>=30 & R>=110", (chroma >= 30) & (r >= 110)),
    ("软红 R-max>=20 & R>=100", (chroma >= 20) & (r >= 100)),
    ("更软 R-max>=12 & R>=95", (chroma >= 12) & (r >= 95)),
]:
    print(f"  {tag:26s} {int(m.sum()):6d} px")

# 白通道（现行口径，区域=整图）
tmpl0 = np.load("assets/shape_template.npy")
det = ad.RegionDetector("empty", None, tmpl0,
                        bg_margin=18, lum_floor=55, edge=5, min_mask_px=30)
white = det.extract(np.array(im))
print("\n现行白掩膜 %d px" % int(white.sum()))

for tag, m in [("硬红", (chroma >= 55) & (r >= 120)),
               ("软红30", (chroma >= 30) & (r >= 110)),
               ("软红20", (chroma >= 20) & (r >= 100))]:
    hot = ad.largest_cc(m, 20)
    n = int(hot.sum())
    if n < 20:
        print(f"\n{tag}: 连通域仅 {n}px，放弃")
        continue
    hull = ad.convex_hull_mask(hot)
    inner = ad.erode(hull, 2)
    area = int(inner.sum())
    print(f"\n{tag}: 最大连通域 {n}px  hull {int(hull.sum())}px  body {area}px")
    print(f"    body 内红 {int((m & inner).sum())}px = {100.0 * (m & inner).sum() / max(1, area):.1f}%")
    print(f"    轮廓线占比 (n/area) = {100.0 * n / max(1, area):.1f}%")

# 模板 IoU
tmpl = np.load("assets/shape_template.npy")
for tag, m in [("硬红", (chroma >= 55) & (r >= 120)),
               ("软红30", (chroma >= 30) & (r >= 110)),
               ("软红20", (chroma >= 20) & (r >= 100)),
               ("白", white)]:
    hot = ad.largest_cc(m, 20)
    if hot.sum() < 20:
        print(f"{tag}: 空")
        continue
    hull = ad.convex_hull_mask(hot)
    print(f"{tag}: IoU = {ad.iou(ad.normalize_shape(hull), tmpl):.3f}")

# 存一份放大图 + 掩膜着色
vis = np.array(im).copy()
vis[white] = [0, 255, 0]
m2 = (chroma >= 20) & (r >= 100)
vis[m2 & ~white] = [255, 0, 255]
Z = max(1, 900 // max(im.width, 1))
Image.fromarray(vis).resize((im.width * Z, im.height * Z), Image.NEAREST).save(
    "_out/empty_mask.png")
print("\n出图 _out/empty_mask.png  绿=白 品红=软红")
