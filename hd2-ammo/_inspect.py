# -*- coding: utf-8 -*-
"""定位截图里的红色标注框，反推用户在 2K 屏幕上给出的坐标是否一致。"""
import numpy as np
from PIL import Image
from pathlib import Path

SRC = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images\clipboard-2026-09-18T13-18-21-264Z-ac430eed.jpg")
OUT = Path(r"C:\Users\Administrator\WorkBuddy\2026-09-18-21-15-56\hd2-ammo\_out")
OUT.mkdir(parents=True, exist_ok=True)

im = Image.open(SRC).convert("RGB")
a = np.asarray(im).astype(np.int16)
H, W = a.shape[:2]
print(f"image: {W}x{H}")

R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
red = (R > 170) & (G < 80) & (B < 80)
print("red pixel count:", int(red.sum()))

rows = red.sum(axis=1)
cols = red.sum(axis=0)
print("\n-- rows with most red (top 12) --")
for y in np.argsort(rows)[::-1][:12]:
    print(f"  y={int(y):4d}  count={int(rows[y])}")
print("-- cols with most red (top 12) --")
for x in np.argsort(cols)[::-1][:12]:
    print(f"  x={int(x):4d}  count={int(cols[x])}")

# 连通域粗定位：用红像素的行/列投影找外接矩形
ys, xs = np.where(red)
if len(ys):
    print(f"\nred bbox: x[{xs.min()}..{xs.max()}] y[{ys.min()}..{ys.max()}]")

# 按用户给的 2K 坐标 (330,1250)-(390,1300) 反推截图坐标
scale = W / 2560.0
print(f"\nscale = {scale:.4f} (截图宽 / 2560)")
for (x, y) in [(330, 1250), (390, 1300)]:
    print(f"  2K({x},{y}) -> 截图({x*scale:.0f},{y*scale:.0f})")

# 裁出候选区域 + 放大保存
bx0, by0, bx1, by1 = [int(v * scale) for v in (330, 1250, 390, 1300)]
pad = 14
crop = im.crop((max(0, bx0 - pad), max(0, by0 - pad), min(W, bx1 + pad), min(H, by1 + pad)))
crop.resize((crop.width * 8, crop.height * 8), Image.NEAREST).save(OUT / "crop_scaled.png")
print(f"saved crop_scaled.png from box ({bx0},{by0})-({bx1},{by1})")

# 同时存一张整图的左下角区域，便于人工对照
crop2 = im.crop((0, int(H * 0.6), int(W * 0.35), H))
crop2.resize((crop2.width * 3, crop2.height * 3), Image.NEAREST).save(OUT / "bottomleft.png")
print("saved bottomleft.png")
