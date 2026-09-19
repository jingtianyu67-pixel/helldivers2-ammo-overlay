# -*- coding: utf-8 -*-
"""质心法精定位移：让弹匣在区域内的相对位置在两种布局下一致。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from PIL import Image

import ammo_detect as ad

cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
det = ad.build_detector(cfg, Path("."))
C = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images")
WITH = C / "clipboard-2026-09-18T17-04-06-946Z-02f9b08a.jpg"
WITHOUT = C / "clipboard-2026-09-18T17-04-06-944Z-a0b0e66b.jpg"
img_w = np.array(Image.open(WITH).convert("RGB").resize((2560, 1440), Image.LANCZOS))
img_o = np.array(Image.open(WITHOUT).convert("RGB").resize((2560, 1440), Image.LANCZOS))


def blob(rgb, d):
    wm = d.extract(rgb)
    hot = ad.largest_cc(wm, d.min_mask_px)
    if hot.sum() < 50:
        return None, 0
    ys, xs = np.nonzero(hot)
    return (float(xs.mean()), float(ys.mean()), int(hot.sum())), int(wm.sum())


for sp in det.specs:
    l, t, r, b = sp.box
    d = sp.detector
    rgb_w = img_w[t:b, l:r]
    cw, nw = blob(rgb_w, d)
    print(f"\n===== 区域 {sp.name} {sp.box} =====")
    print(f"  有背包：弹匣质心(局部) {cw}  连通域 {nw}px")
    if cw is None:
        continue
    cx_w, cy_w, _ = cw

    rows = []
    for sx in range(-150, 81):
        bl, br = l + sx, r + sx
        if bl < 0 or br > 2560:
            continue
        rgb_o = img_o[t:b, bl:br]
        ret, _ = blob(rgb_o, d)
        if ret is None:
            continue
        cx, cy, n = ret
        dist = ((cx - cx_w) ** 2 + (cy - cy_w) ** 2) ** 0.5
        rows.append((dist, sx, cx, cy, n))
    rows.sort()
    print(f"  无背包：按「弹匣在区域内的相对位置最一致」排序前 6：")
    for dist, sx, cx, cy, n in rows[:6]:
        print(f"      shift_x={sx:+4d} → 区域({l+sx},{t},{r+sx},{b})  "
              f"质心({cx:5.1f},{cy:5.1f}) 偏离{dist:5.2f}px  掩膜{n:4d}px")
