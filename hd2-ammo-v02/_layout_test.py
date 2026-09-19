# -*- coding: utf-8 -*-
"""布局自适应端到端验证：有背包 / 无背包 / 无 HUD 三种输入。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from PIL import Image

import ammo_detect as ad

cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
C = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images")


def make_grab(img):
    return lambda box: img[box[1]:box[3], box[0]:box[2]].copy()


def load(p, size=(2560, 1440)):
    return np.array(Image.open(p).convert("RGB").resize(size, Image.LANCZOS))


cases = [
    ("有背包（用户图2）",
     load(C / "clipboard-2026-09-18T17-04-06-946Z-02f9b08a.jpg"), True),
    ("无背包（用户图1）",
     load(C / "clipboard-2026-09-18T17-04-06-944Z-a0b0e66b.jpg"), False),
    ("无 HUD（纯背景）", np.full((1440, 2560, 3), 70, np.uint8), None),
    ("实机满弹帧 prod_screen", load(Path("_out/prod_screen.png")), None),
]

for tag, img, expect in cases:
    det = ad.build_detector(cfg, Path("."))
    grab = make_grab(img)
    for _ in range(4):                    # hold_frames=3 的时间去抖要先喂满
        r = det.analyze(grab)
    run = det.last_run
    got = det.has_backpack
    mark = "" if expect is None else ("✓" if got == expect else "✗ 期望 " + str(expect))
    print(f"\n===== {tag} =====")
    print(f"  判据竖线最长竖直连续 = {run}px（门槛 {det.layout.min_run}）"
          f" → 布局 {det.mode} {mark}")
    l, t, rr, bb = det.specs[0].box
    l += det.shift[0]; rr += det.shift[0]
    print(f"  实际使用区域 ({l},{t})-({rr},{bb})  shift={det.shift}")
    print(f"  读数 valid={r.valid} {r} state={r.state} 白={r.white_px} "
          f"体={r.area} IoU={r.shape_iou:.2f} hint={r.hint} {r.reason}")
    cl, ct, cr, cb = det.canvas
    print(f"  抓取范围 ({cl},{ct})-({cr},{cb}) {cr - cl}x{cb - ct}")

# 判据鲁棒性：连续 30 帧同一张图，看有没有抖动
print("\n===== 抖动检查（同图连跑 12 帧）=====")
for tag, p in [("有背包", C / "clipboard-2026-09-18T17-04-06-946Z-02f9b08a.jpg"),
               ("无背包", C / "clipboard-2026-09-18T17-04-06-944Z-a0b0e66b.jpg")]:
    img = load(p)
    det = ad.build_detector(cfg, Path("."))
    grab = make_grab(img)
    seq = []
    for _ in range(12):
        det.analyze(grab)
        seq.append((det.last_run, det.has_backpack))
    runs = [s[0] for s in seq]
    bags = [s[1] for s in seq]
    print(f"  {tag}: 竖线段 {min(runs)}~{max(runs)}px  布局{'稳定' if len(set(bags)) == 1 else '抖动!'}"
          f"  ({'有背包' if bags[0] else '无背包'})")
