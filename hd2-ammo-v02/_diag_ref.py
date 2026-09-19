# -*- coding: utf-8 -*-
"""参考图：直接量「白像素 / 弹匣内缩面积」，绕开窗口尺寸校验。"""
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
SHOTS = [
    ("白弧(正常)", CLIP / "clipboard-2026-09-18T15-37-25-493Z-295e67be.jpg"),
    ("红弧(红线)", CLIP / "clipboard-2026-09-18T15-37-25-495Z-2dab9498.jpg"),
    ("弹匣特写(红线)", CLIP / "clipboard-2026-09-18T15-47-56-207Z-92054776.png"),
]

for tag, p in SHOTS:
    im = Image.open(p).convert("RGB")
    W, H = im.size
    k = W / 2560.0
    print("\n=== %s  %dx%d  scale=%.3f ===" % (tag, W, H, k))
    for item in cfg["regions"]:
        l, t, r, b = item["region"]
        box = (int(round(l * k)), int(round(t * k)), int(round(r * k)), int(round(b * k)))
        sub = np.array(im.crop(box))
        det = ad.RegionDetector(item["name"], None, tmpl,
                                bg_margin=d["bg_margin"], lum_floor=d["lum_floor"],
                                edge=max(1, int(round(d["edge"] * k))),
                                min_mask_px=30,
                                fill_inset=max(1, int(round(d["fill_inset"]))),
                                ring_open=max(0, int(round(d["ring_open"]))))
        white = det.extract(sub)
        hot = det._collect(white)
        hull = ad.convex_hull_mask(hot)
        inner = ad.erode(hull, det.fill_inset)
        core = ad.opening(white, det.ring_open)
        fill = int((core & inner).sum())
        v = ad.iou(ad.normalize_shape(hull), tmpl)
        a = int(inner.sum())
        print("  %s %s 白=%d 凸包=%d 分母=%d 分子=%d -> %.1f%%  IoU=%.2f"
              % (item["name"], box, int(white.sum()), int(hull.sum()), a, fill,
                 100.0 * fill / max(1, a), v))
