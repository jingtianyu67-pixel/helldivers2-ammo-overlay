# -*- coding: utf-8 -*-
"""用内部窗口掩膜 + 逐行自适应背景，验证 6 张样张算出来的余量百分比。"""
from pathlib import Path

import numpy as np
from PIL import Image

SHOTS = Path(r"C:\Users\Administrator\Desktop\hd2_ammo_shots")
OUT = Path(r"C:\Users\Administrator\WorkBuddy\2026-09-18-21-15-56\hd2-ammo\_out")
MASK = np.load(OUT / "interior_mask.npy")
AREA = int(MASK.sum())
ROWS = np.where(MASK.any(axis=1))[0]
MARGIN = 25


def analyze(a):
    a = a.astype(np.int16)
    L = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    h, w = L.shape

    white = np.zeros_like(MASK)
    for y in ROWS:
        xs = np.where(MASK[y])[0]
        x0, x1 = xs.min(), xs.max()
        cand = []
        if x0 - 3 > 0:
            cand.append(L[y, max(0, x0 - 9):x0 - 3])
        if x1 + 4 < w:
            cand.append(L[y, x1 + 4:min(w, x1 + 10)])
        cand = [c for c in cand if c.size]
        ref = float(np.median(np.concatenate(cand))) if cand else 90.0
        white[y] = MASK[y] & (L[y] > ref + MARGIN) & (L[y] > 120)

    red = MASK & ((r - np.maximum(g, b)) > 30) & (r > 100)
    fill = white | red
    return white, red, fill


print(f"内部窗口面积 = {AREA} px\n")
print(f"{'样张':<10}{'白':>6}{'红':>6}{'合计':>7}{'百分比':>10}   填充顶端")
print("-" * 56)
for f in sorted(SHOTS.glob("*_1x.png")):
    tag = f.stem.split("_")[1]
    a = np.asarray(Image.open(f).convert("RGB"))
    white, red, fill = analyze(a)
    n = int(fill.sum())
    ys = np.where(fill.any(axis=1))[0]
    top = f"y{ys.min()}" if ys.size else "-"
    print(f"{tag:<10}{int(white.sum()):>6}{int(red.sum()):>6}{n:>7}"
          f"{n / AREA * 100:>9.1f}%   {top}")

# 可视化最后一张
f = sorted(SHOTS.glob("*_1x.png"))[-1]
img = Image.open(f).convert("RGB")
a = np.asarray(img)
white, red, fill = analyze(a)
Z = 8
big = np.asarray(img.resize((img.width * Z, img.height * Z), Image.NEAREST)).copy()
ov = big.copy()
selw = np.kron(white, np.ones((Z, Z), bool))
selr = np.kron(red, np.ones((Z, Z), bool))
selm = np.kron(MASK & ~fill, np.ones((Z, Z), bool))
ov[selm] = (ov[selm] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(np.uint8)
ov[selw] = [0, 255, 0]
ov[selr] = [255, 80, 0]
Image.fromarray(np.concatenate([big, ov], axis=1)).save(OUT / "validate_last.png")
Image.fromarray(ov).save(OUT / "validate_last_only.png")
print(f"\n预览 -> {OUT / 'validate_last.png'}  (左原图 / 右掩膜: 绿=白填充 橙=红填充 紫=判定为空)")
