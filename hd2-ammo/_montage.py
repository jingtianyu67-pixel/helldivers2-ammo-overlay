# -*- coding: utf-8 -*-
"""把桌面 hd2_ammo_shots 里的 1x 样张拼成对比图，并打印纵向/横向剖面，用于标定弹匣图形。"""
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SHOTS = Path(r"C:\Users\Administrator\Desktop\hd2_ammo_shots")
OUT = Path(r"C:\Users\Administrator\WorkBuddy\2026-09-18-21-15-56\hd2-ammo\_out")
OUT.mkdir(parents=True, exist_ok=True)
ZOOM = 8

files = sorted(SHOTS.glob("*_1x.png"))
print(f"找到 {len(files)} 张 1x 样张\n")

panels = []
for f in files:
    img = Image.open(f).convert("RGB")
    a = np.asarray(img).astype(np.int16)
    lum = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    bright = lum > 165
    reddish = (r > 120) & (r - g > 40) & (r - b > 40)
    icon = bright | reddish

    tag = f.stem.split("_")[1]
    print(f"--- {tag}  {img.width}x{img.height} ---")
    print(f"  亮度均值 {lum.mean():6.1f}  min {lum.min():5.0f}  max {lum.max():5.0f}")
    print(f"  亮像素 {bright.sum():4d}  红像素 {reddish.sum():4d}  图标像素 {icon.sum():4d}"
          f"  ({icon.sum() / lum.size * 100:.1f}%)")
    rows = icon.sum(axis=1)
    print("  逐行图标像素数(自上而下): " + " ".join(f"{int(v):2d}" for v in rows))
    cols = icon.sum(axis=0)
    print("  逐列图标像素数(自左而右): " + " ".join(f"{int(v):2d}" for v in cols))

    big = img.resize((img.width * ZOOM, img.height * ZOOM), Image.NEAREST)
    panel = Image.new("RGB", (big.width, big.height + 22), (20, 20, 20))
    panel.paste(big, (0, 22))
    ImageDraw.Draw(panel).text((4, 6), f"{tag}  icon={icon.sum() / lum.size * 100:.0f}%",
                               fill=(255, 220, 80))
    panels.append(panel)

gap = 16
W = sum(p.width for p in panels) + gap * (len(panels) + 1)
H = max(p.height for p in panels) + gap * 2
sheet = Image.new("RGB", (W, H), (0, 0, 0))
x = gap
for p in panels:
    sheet.paste(p, (x, gap))
    x += p.width + gap
sheet.save(OUT / "montage_1x.png")
print(f"\n拼图 -> {OUT / 'montage_1x.png'}  ({sheet.width}x{sheet.height})")
