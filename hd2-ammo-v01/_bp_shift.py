# -*- coding: utf-8 -*-
"""在「无背包」截图上滑动搜索弹匣图标的真实位置，定出左移量。"""
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
WITH = C / "clipboard-2026-09-18T17-04-06-946Z-02f9b08a.jpg"     # 有背包（弹匣在现坐标）
WITHOUT = C / "clipboard-2026-09-18T17-04-06-944Z-a0b0e66b.jpg"  # 无背包（弹匣左移）

img_w = np.array(Image.open(WITH).convert("RGB").resize((2560, 1440), Image.LANCZOS))
img_o = np.array(Image.open(WITHOUT).convert("RGB").resize((2560, 1440), Image.LANCZOS))
k = 1.0
print(f"两图原始 1920x1079 → 归一化到 2560x1440 再裁切")

for sp in det.specs:
    l, t, r, b = sp.box
    print(f"\n===== 区域 {sp.name} {sp.box} =====")
    # 有背包：按现坐标裁（应该就是弹匣）
    box_w = (int(round(l * k)), int(round(t * k)), int(round(r * k)), int(round(b * k)))
    rw = sp.detector.analyze(np.array(Image.fromarray(img_w).crop(box_w)))
    print(f"  有背包@现坐标: valid={rw.valid} {rw} state={rw.state} "
          f"白={rw.white_px} 体={rw.area} IoU={rw.shape_iou:.2f}")

    # 无背包：横向滑动找最佳位置
    best = []
    for sx in range(-140, 61, 1):
        bl = l + sx
        if bl < 0 or bl + (r - l) > 2560:
            continue
        bb = (int(round(bl * k)), int(round(t * k)),
              int(round((bl + r - l) * k)), int(round(b * k)))
        rr = sp.detector.analyze(np.array(Image.fromarray(img_o).crop(bb)))
        if rr.valid and rr.state == "white":
            best.append((rr.fraction, rr.shape_iou, rr.white_px, sx))
    if not best:
        print("  无背包：滑动范围内没有命中")
        continue
    best.sort(reverse=True)
    print(f"  无背包 命中 {len(best)} 个位移；按占比排序前 8：")
    for frac, v, wp, sx in best[:8]:
        print(f"      shift_x={sx:+4d} → 区域({l+sx},{t},{r+sx},{b}) "
              f"占比{frac * 100:5.1f}% 白={wp:3d} IoU={v:.2f}")
