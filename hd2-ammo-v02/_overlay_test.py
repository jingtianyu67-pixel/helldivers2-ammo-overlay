# -*- coding: utf-8 -*-
"""覆盖层端到端自检：窗口真的贴到屏幕上了吗？

创建分层窗口 → 依次显示几个余量 → 用 mss 抓屏回读 → 检查弧线像素是否存在。
这一步不能省：UpdateLayeredWindow 失败时不会报错，只是屏幕上什么都没有。
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mss                                    # noqa: E402
from capture import set_dpi_aware             # noqa: E402
from overlay import AmmoOverlay, screen_size  # noqa: E402

cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
set_dpi_aware()
scr = screen_size()
print("屏幕", scr)

ov = AmmoOverlay(cfg, scr)
x0, y0, x1, y1 = ov.geo.bbox()
print("弧线 bbox", (x0, y0, x1, y1), f"{x1 - x0}x{y1 - y0}")
pad = 6
box = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

ctx = mss.mss()
mon = {"left": box[0], "top": box[1],
       "width": box[2] - box[0], "height": box[3] - box[1]}


def grab() -> np.ndarray:
    a = np.array(ctx.grab(mon))
    return a[:, :, 2::-1].copy()          # BGRA -> RGB


base = grab()
base_lum = base.mean()

ok_all = True
for frac in (1.0, 0.6, 0.25):
    ov.show(frac)
    time.sleep(0.25)
    shot = grab()
    # 覆盖层像素：与「隐藏时」的截图相比明显变亮
    diff = shot.astype(np.int16) - base.astype(np.int16)
    changed = int((np.abs(diff).max(axis=2) > 40).sum())
    lum = shot.mean()
    ok = changed > 300
    ok_all &= ok
    print(f"  show({frac:.2f}) -> 屏幕变化像素 {changed:>6d}  平均亮度 {lum:6.1f}"
          f"（背景 {base_lum:6.1f}）  {'OK' if ok else '失败：屏幕上没看到弧线'}")
    from PIL import Image as _I
    _I.fromarray(shot).save(os.path.join(HERE, "_out", f"overlay_on_{int(frac * 100)}.png"))

ov.hide()
time.sleep(0.25)
after = grab()
changed = int((np.abs(after.astype(np.int16) - base.astype(np.int16)).max(axis=2) > 40).sum())
print(f"  hide()   -> 与初始背景差异 {changed}  {'OK' if changed < 300 else '失败：没隐藏干净'}")
ok_all &= changed < 300

print(f"\n重绘次数 {ov.redraws}（show 4 次 + hide）")
ov.close()
ctx.close()

# 关闭后窗口应已销毁
print("窗口句柄已释放:", ov.hwnd is None)
print("结论:", "全部通过" if ok_all else "有失败项")
