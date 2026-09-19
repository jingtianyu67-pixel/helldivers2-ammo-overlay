# -*- coding: utf-8 -*-
"""A 区域探针搜索：统计 15 张样本的白/红像素命中率，挑出外框端点与底斜线。"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ammo_detect import build_detector

D = r"C:\Users\Administrator\Desktop\hd2_ammo_shots\A"
cfg = json.load(open("config.json", encoding="utf-8"))
det = [s for s in build_detector(cfg).specs if s.name == "A"][0].detector

files = {os.path.basename(f)[:3]: f for f in sorted(glob.glob(os.path.join(D, "*_1x.png")))}
WHITE_GRP = ["A01", "A02", "A03", "A04", "A14", "A15"]
RED_GRP = ["A05", "A06", "A07"]
EMPTY_GRP = ["A08", "A09", "A10", "A11", "A12", "A13"]

H, W = 50, 60
XMAX = 48   # 右侧 x52-59 是另一个 HUD 元素，排除


def stack(group):
    wc = np.zeros((H, W), np.int16)
    rc = np.zeros((H, W), np.int16)
    for t in group:
        a = np.array(Image.open(files[t]).convert("RGB"))
        w, r = det.extract(a)
        wc += w.astype(np.int16)
        rc += r.astype(np.int16)
    return wc, rc


wW, rW = stack(WHITE_GRP)
wR, rR = stack(RED_GRP)
wE, rE = stack(EMPTY_GRP)


def show(mat, thr, label):
    print(f"\n--- {label}（>= {thr} 命中）---")
    hdr = "     " + "".join(str(x % 10) for x in range(0, XMAX))
    print(hdr)
    for y in range(0, 46):
        line = "".join("#" if mat[y, x] >= thr else ("+" if mat[y, x] > 0 else ".")
                       for x in range(0, XMAX))
        if "#" in line:
            print(f"y{y:>3} {line}")


show(wW, 5, "白组(6张) 白色命中 —— 外框恒白")
show(rR, 2, "红组(3张) 红色命中 —— 底部填充转红")
show(wE, 5, "空组(6张) 白色命中")

print("\n--- 分组统计：各候选像素的表现 ---")


def probe_report(x, y, rad=1):
    out = []
    for name, mat_w, mat_r, n in (("白", wW, rW, len(WHITE_GRP)),
                                  ("红", wR, rR, len(RED_GRP)),
                                  ("空", wE, rE, len(EMPTY_GRP))):
        xs = slice(max(0, x - rad), x + rad + 1)
        ys = slice(max(0, y - rad), y + rad + 1)
        out.append(f"{name}:W{int(mat_w[ys, xs].sum())} R{int(mat_r[ys, xs].sum())}")
    return f"({x:>2},{y:>2}) " + "  ".join(out)


print("\n外框端点候选（左上/右上/左下/右下 边缘）:")
for pt in [(23, 6), (23, 10), (39, 10), (39, 14), (29, 36), (45, 35), (30, 40), (34, 41),
           (23, 16), (39, 19), (24, 22), (40, 22)]:
    print("  " + probe_report(*pt))

print("\n底部填充区候选（红探针）:")
for pt in [(37, 34), (38, 35), (39, 36), (36, 37), (35, 38), (34, 36),
           (40, 34), (42, 35), (33, 35), (37, 38)]:
    print("  " + probe_report(*pt))
