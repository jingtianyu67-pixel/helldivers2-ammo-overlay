# -*- coding: utf-8 -*-
"""诊断：用户截图里白像素到底怎么数成 50% 的。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

import ammo_detect as ad

IMG = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images\clipboard-2026-09-18T15-47-56-207Z-92054776.png")
rgb = np.array(Image.open(IMG).convert("RGB"))
print("图像尺寸 (h,w) =", rgb.shape)

det = ad.RegionDetector("diag", None, np.load(Path("assets/shape_template.npy")),
                        bg_margin=18, lum_floor=55, edge=5, min_mask_px=30)
white = det.extract(rgb)
print("白掩膜总像素 =", int(white.sum()))
hot = ad.largest_cc(white, 30)
print("最大连通域 =", int(hot.sum()))
hull = ad.convex_hull_mask(hot)
print("凸包面积 =", int(hull.sum()))
core = ad.opening(white, 2)
print("开运算(k=2)后 =", int(core.sum()))

ys, xs = np.where(hot)
print("弹匣包围盒 y[%d,%d] x[%d,%d]  ->  %dx%d"
      % (ys.min(), ys.max(), xs.min(), xs.max(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))

# 量外框线粗细：取图标中部若干行，看白色连续段长度
for y in [ys.min() + 3, (ys.min() + ys.max()) // 2, ys.max() - 3]:
    row = hot[y]
    runs, cur = [], 0
    for v in row:
        if v:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    print("  y=%d 白段长度序列 %s" % (y, runs))

# 逐 k 对比：分母与分子都用同一 k
print("\n  k | 内缩凸包(分母) | erode(白,k)∩分区(分子) | 占比")
for k in range(0, 9):
    inner = ad.erode(hull, k) if k else hull
    num = ad.erode(white, k) if k else white
    fill = int((num & inner).sum())
    a = int(inner.sum())
    print("%3d | %14d | %22d | %6.1f%%" % (k, a, fill, 100.0 * fill / max(1, a)))

# 现行口径（ring_open 开运算 + fill_inset 内缩）
inner = ad.erode(hull, 2)
core = ad.opening(white, 2)
fill = int((core & inner).sum())
print("\n现行口径 fill_inset=2 ring_open=2 -> %d/%d = %.1f%%" % (fill, int(inner.sum()), 100.0 * fill / max(1, int(inner.sum()))))

# 目视：实心填充块单独量（把外框线按“细”过滤，用半径5开运算只留大块）
big = ad.opening(white, 4)
print("开运算(k=4)后 =", int(big.sum()))

Image.fromarray((white * 255).astype(np.uint8)).save("_out/diag_white.png")
Image.fromarray((core * 255).astype(np.uint8)).save("_out/diag_core2.png")
Image.fromarray((big * 255).astype(np.uint8)).save("_out/diag_core4.png")
