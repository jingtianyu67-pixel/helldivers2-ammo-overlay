# -*- coding: utf-8 -*-
"""诊断红色低弹帧：为什么掩膜只有 120 像素。"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

SHOTS = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"


def show(fp, tag):
    a = np.array(Image.open(fp).convert("RGB")).astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)
    print(f"\n===== {tag}  {os.path.basename(fp)}  {a.shape[1]}x{a.shape[0]} =====")
    print(f"亮度: min={lum.min():.0f} 中位={np.median(lum):.0f} max={lum.max():.0f}")
    print(f"chroma(R-maxGB): min={chroma.min():.0f} 中位={np.median(chroma):.0f} "
          f"max={chroma.max():.0f}")
    for th in (18, 25, 30, 40, 50):
        m = chroma > th
        rr = (chroma > th) & (r > 80)
        print(f"  chroma>{th:3d}: {int(m.sum()):4d} px   (且 R>80: {int(rr.sum()):4d} px)")
    # 叠加约束
    for cth in (18, 25, 30):
        for rth in (70, 85, 100):
            m = (chroma > cth) & (r > rth)
            print(f"  chroma>{cth} & R>{rth}: {int(m.sum()):4d} px", end="")
        print()

    # 白色判据
    bg = np.median(np.concatenate([lum[:, :3], lum[:, -3:]], axis=1), axis=1)
    for mg in (10, 20, 25, 30, 40):
        w = (lum > bg[:, None] + mg) & (chroma <= 30)
        print(f"  white lum>bg+{mg:2d}: {int(w.sum()):4d} px")

    # ASCII
    print("ASCII ('R'=红chroma>25, '#'=亮, '.'=暗):")
    red = (chroma > 25) & (r > 80)
    white = (lum > bg[:, None] + 20) & (chroma <= 30)
    for y in range(a.shape[0]):
        row = ""
        for x in range(a.shape[1]):
            if red[y, x]:
                row += "R"
            elif white[y, x]:
                row += "#"
            elif lum[y, x] > np.median(lum) + 15:
                row += "+"
            else:
                row += "."
        print(f"{y:3d} {row}")


for f, t in [("0918_213537_1x.png", "红色低弹 A"), ("0918_213541_1x.png", "红色低弹 B"),
             ("0918_213535_1x.png", "白色半弹"), ("0918_213544_1x.png", "白色满弹")]:
    show(os.path.join(SHOTS, f), t)
