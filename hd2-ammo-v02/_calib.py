# -*- coding: utf-8 -*-
"""标定：找出能把「填充」和「背景」分开的判据，并输出掩膜预览。"""
from pathlib import Path

import numpy as np
from PIL import Image

SHOTS = Path(r"C:\Users\Administrator\Desktop\hd2_ammo_shots")
OUT = Path(r"C:\Users\Administrator\WorkBuddy\2026-09-18-21-15-56\hd2-ammo\_out")
OUT.mkdir(parents=True, exist_ok=True)
ZOOM = 8

files = sorted(SHOTS.glob("*_1x.png"))


def masks(a):
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lo = np.minimum(np.minimum(r, g), b)
    hi = np.maximum(np.maximum(r, g), b)
    out = {}
    for t in (200, 215, 225, 235, 245):
        out[f"min>{t}"] = lo >= t
    out["red"] = (r >= 110) & ((r - np.maximum(g, b)) >= 35)
    out["colored"] = out["min>225"] | out["red"]
    return out


panels = []
for f in files:
    img = Image.open(f).convert("RGB")
    a = np.asarray(img).astype(np.int16)
    m = masks(a)
    tag = f.stem.split("_")[1]

    corners = np.concatenate([
        a[0:6, 0:6].reshape(-1, 3), a[0:6, -6:].reshape(-1, 3),
        a[-6:, 0:6].reshape(-1, 3), a[-6:, -6:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)

    print(f"--- {tag} ---")
    print(f"  背景(四角中位) RGB={bg.astype(int).tolist()}  亮像素峰值={int(a.max())}")
    print("  各判据命中像素数: " + "  ".join(
        f"{k}={int(v.sum())}" for k, v in m.items()))

    # 填充向上延伸到第几行（彩色像素的最上一行）
    col = m["colored"]
    rows = np.where(col.any(axis=1))[0]
    if rows.size:
        print(f"  填充/轮廓纵向范围: y{rows.min()} ~ y{rows.max()}  "
              f"(共 {rows.max() - rows.min() + 1} 行, 占比 {(rows.max() - rows.min() + 1) / 55 * 100:.0f}%)")

    # 预览：原图 + 每个判据的彩色覆盖
    big = np.asarray(img.resize((img.width * ZOOM, img.height * ZOOM), Image.NEAREST)).copy()
    ov = big.copy()
    sel = np.kron(col, np.ones((ZOOM, ZOOM), bool))
    ov[sel] = [0, 255, 0]
    combo = np.concatenate([big, ov], axis=1)
    p = Image.fromarray(combo)
    panels.append((tag, p, int(col.sum())))

gap = 14
W = sum(p.width for _, p, _ in panels) + gap * (len(panels) + 1)
H = max(p.height for _, p, _ in panels) + gap * 2
sheet = Image.new("RGB", (W, H), (10, 10, 10))
x = gap
from PIL import ImageDraw
for tag, p, n in panels:
    sheet.paste(p, (x, gap))
    ImageDraw.Draw(sheet).text((x + 2, 2), f"{tag} n={n}", fill=(255, 220, 80))
    x += p.width + gap
sheet.save(OUT / "mask_preview.png")
print(f"\n预览 -> {OUT / 'mask_preview.png'} ({sheet.width}x{sheet.height})")
