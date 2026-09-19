# -*- coding: utf-8 -*-
"""A 区域探针候选定位：逐行打印白/红像素分布，用来挑外框端点与底斜线。"""
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


def dump(tag, which="both", ymax=None):
    a = np.array(Image.open(files[tag]).convert("RGB"))
    w, r = det.extract(a)
    print(f"\n===== {tag}  白{int(w.sum())}px 红{int(r.sum())}px =====")
    ys, xs = np.nonzero(w | r)
    lo, hi = ys.min(), ys.max()
    if ymax is not None:
        hi = min(hi, ymax)
    for y in range(lo, hi + 1):
        seg = []
        for name, m in (("W", w), ("R", r)):
            idx = np.nonzero(m[y])[0]
            if not len(idx):
                continue
            # 压缩成连续段
            runs, s = [], idx[0]
            for i in range(1, len(idx)):
                if idx[i] != idx[i - 1] + 1:
                    runs.append((s, idx[i - 1]))
                    s = idx[i]
            runs.append((s, idx[-1]))
            seg.append(name + " " + " ".join(f"{p}-{q}" if p != q else f"{p}" for p, q in runs))
        if seg:
            print(f"  y{y:>3} | " + " | ".join(seg))


for t in ("A14", "A01", "A05", "A09", "A13", "A12"):
    dump(t)
