# -*- coding: utf-8 -*-
"""mss 抓取耗时 vs 区域尺寸：决定要不要为省时间做动态 canvas。"""
import sys
import time

sys.path.insert(0, ".")
import mss

from capture import grab_rgb, set_dpi_aware

set_dpi_aware()
ctx = mss.MSS()
boxes = {
    "60x50  (区域A 原位置)": (330, 1250, 390, 1300),
    "113x50 (无背包区域+判据)": (224, 1250, 337, 1300),
    "166x50 (当前 canvas)": (224, 1250, 390, 1300),
    "300x300 (参考)": (200, 1200, 500, 1500),
}
N = 200
for tag, box in boxes.items():
    grab_rgb(ctx, box)                                   # 预热
    t0 = time.perf_counter()
    for _ in range(N):
        grab_rgb(ctx, box)
    dt = (time.perf_counter() - t0) / N * 1000
    print(f"  {tag:26s} {dt:6.2f} ms/次")
ctx.close()
