# -*- coding: utf-8 -*-
"""A 区域探针标定验收：正样本、负样本、以及对 B 区域与视频回归的影响。"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ammo_detect import build_detector

D = r"C:\Users\Administrator\Desktop\hd2_ammo_shots\A"
B = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
cfg = json.load(open("config.json", encoding="utf-8"))
md = build_detector(cfg)
detA = [s for s in md.specs if s.name == "A"][0].detector
detB = [s for s in md.specs if s.name == "B"][0].detector

print("=" * 78)
print("A 区域正样本（15 张，用户游戏内实拍）")
print("=" * 78)
ok = 0
n = 0
for fp in sorted(glob.glob(os.path.join(D, "*_1x.png"))):
    tag = os.path.basename(fp)[:3]
    a = np.array(Image.open(fp).convert("RGB"))
    r = detA.analyze(a)
    n += 1
    ok += r.valid
    print(f"  {tag}: {'✓' if r.valid else '×'} {r.fraction * 100:5.1f}% "
          f"{r.state:5s} hint={r.hint:12s} IoU={r.shape_iou:.2f} "
          f"fill={r.fill:>3}/{r.area} 探针W{r.probe_white:>2}/R{r.probe_red:>2} {r.reason}")
print(f"  → 识别 {ok}/{n}")

print()
print("=" * 78)
print("A 区域负样本（持枪枪身剪影 / 桌面壁纸，应全部拒绝）")
print("=" * 78)
rej = 0
for fp in sorted(glob.glob(os.path.join(B, "*_full.png"))):
    im = np.array(Image.open(fp).convert("RGB"))
    a = im[1250:1300, 330:390].copy()
    a[:3, :] = 0
    a[-3:, :] = 0
    a[:, :3] = 0
    a[:, -3:] = 0
    r = detA.analyze(a)
    rej += (not r.valid)
    print(f"  {os.path.basename(fp)}: {'✓ 误报!' if r.valid else '× 正确拒绝'} "
          f"IoU={r.shape_iou:.2f} 探针W{r.probe_white:>2}/R{r.probe_red:>2} {r.reason}")
print(f"  → 正确拒绝 {rej}/{len(glob.glob(os.path.join(B, '*_full.png')))}")

print()
print("=" * 78)
print("B 区域回归（确认没被改动影响）")
print("=" * 78)
for fp in sorted(glob.glob(os.path.join(B, "*_1x.png"))):
    a = np.array(Image.open(fp).convert("RGB"))
    if a.shape[:2] != (55, 40):
        print(f"  {os.path.basename(fp)}: 跳过（{a.shape[1]}x{a.shape[0]}，非 B 区域）")
        continue
    r = detB.analyze(a)
    print(f"  {os.path.basename(fp)}: {'✓' if r.valid else '×'} {r.fraction * 100:5.1f}% "
          f"{r.state:5s} hint={r.hint:12s} IoU={r.shape_iou:.2f} {r.reason}")
