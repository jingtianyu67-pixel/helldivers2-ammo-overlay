# -*- coding: utf-8 -*-
"""标定产物：从样张提取弹匣「内部窗口」静态掩膜，作为余量百分比的分母。

思路：用暗背景样张，阈值取出「轮廓 + 填充」的连通块 = 弹匣外轮廓剪影；
对剪影做形态学腐蚀（腐蚀量 = 描边厚度），剩下的就是填充可达的内部窗口。
"""
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

SHOTS = Path(r"C:\Users\Administrator\Desktop\hd2_ammo_shots")
OUT = Path(r"C:\Users\Administrator\WorkBuddy\2026-09-18-21-15-56\hd2-ammo\_out")
OUT.mkdir(parents=True, exist_ok=True)


def erode(m, k=1):
    out = m.copy()
    for _ in range(k):
        e = out.copy()
        e[1:, :] &= out[:-1, :]
        e[:-1, :] &= out[1:, :]
        e[:, 1:] &= out[:, :-1]
        e[:, :-1] &= out[:, 1:]
        out = e
    return out


def dilate(m, k=1):
    out = m.copy()
    for _ in range(k):
        e = out.copy()
        e[1:, :] |= out[:-1, :]
        e[:-1, :] |= out[1:, :]
        e[:, 1:] |= out[:, :-1]
        e[:, :-1] |= out[:, 1:]
        out = e
    return out


def fill_holes(wall):
    h, w = wall.shape
    reach = np.zeros_like(wall)
    dq = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not wall[y, x] and not reach[y, x]:
                reach[y, x] = True
                dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not wall[y, x] and not reach[y, x]:
                reach[y, x] = True
                dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not wall[ny, nx] and not reach[ny, nx]:
                reach[ny, nx] = True
                dq.append((ny, nx))
    return ~reach


def lum(a):
    return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]


# 用暗背景样张
src = next(SHOTS.glob("*_213544_1x.png"))
img = Image.open(src).convert("RGB")
a = np.asarray(img).astype(np.int16)
L = lum(a)
print(f"样张 {src.name}  亮度 min={L.min():.0f} max={L.max():.0f} 均值={L.mean():.1f}")

for th in (110, 115, 120, 130):
    body = L > th
    solid = fill_holes(dilate(body, 1))          # 剪影（补掉描边缺口）
    print(f"\nthreshold lum>{th}: 命中 {body.sum():4d} px, 剪影 {solid.sum():4d} px")
    for k in (1, 2, 3):
        m = erode(solid, k)
        ys, xs = np.where(m)
        if ys.size:
            print(f"   腐蚀{k} -> 内部窗口 {m.sum():4d} px  "
                  f"bbox x[{xs.min()}..{xs.max()}] y[{ys.min()}..{ys.max()}]")

# 取一组参数落地
body = L > 115
solid = fill_holes(dilate(body, 1))
interior = erode(solid, 2)
print(f"\n采用: lum>115 + dilate1 + erode2 -> 内部窗口 {interior.sum()} px")

# 逐行窗口宽度，检查形状是否合理
print("\n逐行窗口宽度(自上而下):")
for y in range(interior.shape[0]):
    xs = np.where(interior[y])[0]
    if xs.size:
        print(f"  y{y:2d}: x{xs.min():2d}..{xs.max():2d}  宽 {xs.size}")

np.save(OUT / "interior_mask.npy", interior)

# 预览
ZOOM = 8
big = np.asarray(img.resize((img.width * ZOOM, img.height * ZOOM), Image.NEAREST)).copy()
sel = np.kron(interior, np.ones((ZOOM, ZOOM), bool))
ov = big.copy()
ov[sel] = (ov[sel] * 0.35 + np.array([0, 255, 0]) * 0.65).astype(np.uint8)
comb = np.concatenate([big, ov], axis=1)
Image.fromarray(comb).save(OUT / "interior_preview.png")
print(f"\n掩膜 -> {OUT / 'interior_mask.npy'}")
print(f"预览 -> {OUT / 'interior_preview.png'}")
