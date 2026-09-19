# -*- coding: utf-8 -*-
"""对比不同阈值下红色低弹帧的掩膜质量。"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

SHOTS = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"


def render(fp, tag, chroma_min, r_min, bg_margin, lum_floor, edge=5):
    a = np.array(Image.open(fp).convert("RGB")).astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)
    side = np.concatenate([lum[:, :edge], lum[:, -edge:]], axis=1)
    bg = np.median(side, axis=1)
    white = (lum > bg[:, None] + bg_margin) & (lum > lum_floor) & (chroma < 25)
    red = (chroma > chroma_min) & (r > r_min)
    m = white | red
    print(f"\n--- {tag}  chroma>{chroma_min} R>{r_min} bg+{bg_margin} lum>{lum_floor} "
          f"→ 白 {int(white.sum())} 红 {int(red.sum())} 合 {int(m.sum())} ---")
    for y in range(a.shape[0]):
        row = ""
        for x in range(a.shape[1]):
            if red[y, x]:
                row += "R"
            elif white[y, x]:
                row += "#"
            else:
                row += "."
        print(f"{y:3d} {row}")


F = os.path.join(SHOTS, "0918_213537_1x.png")
render(F, "213537 旧阈值", 30, 100, 25, 120)
render(F, "213537 新阈值-保守", 20, 80, 18, 60)
render(F, "213537 新阈值-激进", 14, 60, 12, 45)

F2 = os.path.join(SHOTS, "0918_213541_1x.png")
render(F2, "213541 新阈值-保守", 20, 80, 18, 60)
