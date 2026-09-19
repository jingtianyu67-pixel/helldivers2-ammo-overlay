# -*- coding: utf-8 -*-
"""对比口径：分子去掉开运算，直接「白像素 ∩ 内缩凸包」。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image

import ammo_detect as ad

BASE = Path(__file__).resolve().parent
cfg = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
d = cfg["detect"]
tmpl = np.load(BASE / "assets/shape_template.npy")
CLIP = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images")

CASES = [
    ("特写(红线)", CLIP / "clipboard-2026-09-18T15-47-56-207Z-92054776.png", None),
    ("白弧图A", CLIP / "clipboard-2026-09-18T15-37-25-493Z-295e67be.jpg", 0.75),
    ("红弧图A", CLIP / "clipboard-2026-09-18T15-37-25-495Z-2dab9498.jpg", 0.75),
    ("白弧图B", CLIP / "clipboard-2026-09-18T15-37-25-493Z-295e67be.jpg", 0.75),
]

for tag, p, k in CASES:
    im = Image.open(p).convert("RGB")
    print("\n=== %s %s ===" % (tag, im.size))
    if k is None:
        sub = np.array(im)
        det = ad.RegionDetector("x", None, tmpl, bg_margin=d["bg_margin"],
                                lum_floor=d["lum_floor"], edge=d["edge"], min_mask_px=30)
        items = [{"name": "特写", "w": True}]
        crops = [sub]
    else:
        crops, items = [], []
        for item in cfg["regions"]:
            l, t, r, b = item["region"]
            box = (int(round(l * k)), int(round(t * k)), int(round(r * k)), int(round(b * k)))
            crops.append(np.array(im.crop(box)))
            items.append(item)
        det = ad.RegionDetector("x", None, tmpl, bg_margin=d["bg_margin"],
                                lum_floor=d["lum_floor"],
                                edge=max(1, int(round(d["edge"] * k))), min_mask_px=30)

    for it, sub in zip(items, crops):
        nm = it["name"]
        white = det.extract(sub)
        hot = ad.largest_cc(white, 30)
        if hot.sum() < 30:
            print("  %s 空" % nm); continue
        hull = ad.convex_hull_mask(hot)
        v = ad.iou(ad.normalize_shape(hull), tmpl)
        line = "  %s 白=%d 凸包=%d IoU=%.2f |" % (nm, int(white.sum()), int(hull.sum()), v)
        for kk in (1, 2, 3):
            inner = ad.erode(hull, kk)
            a = int(inner.sum())
            num = int((white & inner).sum())
            old = int((ad.opening(white, 2) & inner).sum())
            line += "  k=%d:%d/%d=%.1f%%(旧%.1f%%)" % (kk, num, a, 100.0 * num / a, 100.0 * old / a)
        print(line)
