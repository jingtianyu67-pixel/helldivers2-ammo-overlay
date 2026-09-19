# -*- coding: utf-8 -*-
"""验证布局判据改成「按秒」节流后：不同采样率下每秒判几次、布局结论是否正确。"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
import ammo_detect as ad

cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
C = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images")
CASES = [("有背包", C / "clipboard-2026-09-18T17-04-06-946Z-02f9b08a.jpg", True),
         ("无背包", C / "clipboard-2026-09-18T17-04-06-944Z-a0b0e66b.jpg", False)]

# 判据这次真的跑了吗 —— 给 longest_run 套一层计数
calls = {"n": 0}
_orig = ad.LayoutProbe.longest_run


def counted(self, rgb):
    calls["n"] += 1
    return _orig(self, rgb)


ad.LayoutProbe.longest_run = counted

img = None
for tag, p, _ in CASES:
    if p.exists():
        img = np.array(Image.open(p).convert("RGB").resize((2560, 1440), Image.LANCZOS))
        break
if img is None:
    sys.exit("找不到参考截图")

box = (224, 1250, 390, 1300)


def grab(_):
    l, t, r, b = box
    return img[t:b, l:r]


for hz in (50, 20, 5):
    det = ad.build_detector(cfg, Path("."))
    calls["n"] = 0
    dur, n = 2.5, 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < dur:
        det.analyze(grab)
        n += 1
        time.sleep(1.0 / hz)
    el = time.monotonic() - t0
    print(f"{hz:>3}Hz：{n} 帧 / {el:.2f}s → 判据实跑 {calls['n']} 次"
          f"（{calls['n'] / el:.2f} 次每秒，期望 ≈{1 / det.layout.interval:.2f}）"
          f" | 结论 {det.mode}")
