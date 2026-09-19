# -*- coding: utf-8 -*-
"""把样张按 1 像素 = 1 字符打印，用来确定弹匣内部窗口的精确边界。"""
from pathlib import Path

import numpy as np
from PIL import Image

SHOTS = Path(r"C:\Users\Administrator\Desktop\hd2_ammo_shots")
RAMP = " .:-=+*#%@"


def show(tag, a, marks=None):
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    lo = np.minimum(np.minimum(r, g), b)
    red = (r >= 110) & ((r - np.maximum(g, b)) >= 35)
    white = lo >= 225
    fill = red | white

    h, w = lum.shape
    print(f"===== {tag}   {w}x{h} =====")
    print("    " + "".join(str(x % 10) for x in range(w)))
    for y in range(h):
        line = []
        for x in range(w):
            if marks == "fill" and fill[y, x]:
                line.append("#")
            else:
                idx = min(len(RAMP) - 1, max(0, int(lum[y, x] / 256 * len(RAMP))))
                line.append(RAMP[idx])
        print(f"{y:3d} " + "".join(line))
    print()


for tag in ("213537", "213544", "213525"):
    f = next(SHOTS.glob(f"*_{tag}_1x.png"))
    a = np.asarray(Image.open(f).convert("RGB")).astype(np.int16)
    show(tag, a, marks="fill")
