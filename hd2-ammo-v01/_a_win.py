# -*- coding: utf-8 -*-
"""A 区域标定 v2：选窗口腐蚀量、定 empty_ref、验证负样本。"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ammo_detect import (RegionDetector, convex_hull_mask, erode, largest_cc)

D = r"C:\Users\Administrator\Desktop\hd2_ammo_shots\A"
B_DIR = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
cfg = json.load(open("config.json", encoding="utf-8"))
DV = cfg["detect"]
EK = ("bg_margin", "lum_floor", "chroma_min", "r_min", "red_ratio", "edge")
raw = RegionDetector("tmp", None, None, **{k: DV[k] for k in EK})

acc = None
for t in ("A14", "A15"):
    fp = glob.glob(os.path.join(D, t + "_*_1x.png"))[0]
    w, r = raw.extract(np.array(Image.open(fp).convert("RGB")))
    cc = largest_cc(w | r, 20)
    acc = cc if acc is None else (acc | cc)
sil = convex_hull_mask(acc)

samples = {}
for fp in sorted(glob.glob(os.path.join(D, "*_1x.png"))):
    samples[os.path.basename(fp)[:3]] = np.array(Image.open(fp).convert("RGB"))

# 负样本：从 _full.png 切 A 区域，最外 3px 置黑以抹掉标注框线
neg = {}
for fp in sorted(glob.glob(os.path.join(B_DIR, "*_full.png"))):
    im = np.array(Image.open(fp).convert("RGB"))
    a = im[1250:1300, 330:390].copy()
    a[:3, :] = 0
    a[-3:, :] = 0
    a[:, :3] = 0
    a[:, -3:] = 0
    neg[os.path.basename(fp)[:11]] = a

print(f"剪影 {int(sil.sum())}px\n")
for e in (2, 3, 4):
    win = erode(sil, e)
    det = RegionDetector("A", win, None,
                         probe_white=[(23, 5), (38, 8), (29, 35), (45, 35), (23, 16), (40, 22)],
                         probe_red=[(37, 34), (38, 35), (35, 36), (40, 34)],
                         probe_radius=1, white_need=24, red_need=10, **{k: DV[k] for k in EK})
    fills = {}
    for t, a in samples.items():
        w, r = det.extract(a)
        cc = largest_cc(w | r, 20)
        fills[t] = int((cc & win).sum())
    empty = [fills[t] for t in ("A08", "A09", "A10", "A11", "A12", "A13")]
    print(f"腐蚀 {e}px → 窗口 {int(win.sum()):>3}px | "
          f"满弹 A14={fills['A14']:>3} A15={fills['A15']:>3} | "
          f"空弹 {empty} (max {max(empty)})")
    print(f"    其余: " + " ".join(f"{t}={fills[t]}" for t in
                                  ("A01", "A02", "A03", "A04", "A05", "A06", "A07")))
    bad = {t: det.analyze(a).valid for t, a in neg.items()}
    print(f"    负样本误报: {sum(bad.values())}/{len(bad)}  {[t for t, v in bad.items() if v]}")
