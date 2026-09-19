# -*- coding: utf-8 -*-
"""细量：判据矩形内的竖直白线 + 两张图的弹匣图标精确位置。"""
from pathlib import Path

import numpy as np
from PIL import Image

C = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images")
IMGS = [("图1(弹匣靠左)", C / "clipboard-2026-09-18T17-04-06-944Z-a0b0e66b.jpg"),
        ("图2(弹匣靠右)", C / "clipboard-2026-09-18T17-04-06-946Z-02f9b08a.jpg")]

for tag, p in IMGS:
    im = Image.open(p).convert("RGB")
    k = im.width / 2560.0
    a = np.array(im).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)
    print(f"\n===== {tag}  {im.size} k={k:.3f} =====")

    # 判据矩形 (330,1250)-(337,1295)：亮度分布 + 每列竖直连续亮段
    l, t, rr, bb = 330, 1250, 337, 1295
    x0, y0 = int(round(l * k)), int(round(t * k))
    x1, y1 = int(round(rr * k)), int(round(bb * k))
    sub = lum[y0:y1, x0:x1]
    print(f"判据矩形像素 {sub.shape}  lum 分位50/90/99/max = "
          f"{[int(np.percentile(sub, q)) for q in (50, 90, 99)]}/{int(sub.max())}")
    for th in (110, 130, 150):
        m = sub > th
        # 每列最长竖直连续段
        best = 0
        for c in range(m.shape[1]):
            col = m[:, c]
            run = mx = 0
            for v in col:
                run = run + 1 if v else 0
                mx = max(mx, run)
            best = max(best, mx)
        print(f"   阈值{th}: 亮像素 {int(m.sum()):3d}  最长竖直连续 {best}px "
              f"(矩形高 {sub.shape[0]})")

    # 左移另一侧同样量一遍（怀疑判据矩形在两张图里角色相反）
    for name, (cl, ct, cr, cb) in [("判据矩形", (330, 1250, 337, 1295)),
                                   ("左移候选", (230, 1250, 237, 1295))]:
        cx0, cy0 = int(round(cl * k)), int(round(ct * k))
        cx1, cy1 = int(round(cr * k)), int(round(cb * k))
        s2 = lum[cy0:cy1, cx0:cx1]
        m2 = s2 > 130
        best = 0
        for c in range(m2.shape[1]):
            col = m2[:, c]
            run = mx = 0
            for v in col:
                run = run + 1 if v else 0
                mx = max(mx, run)
            best = max(best, mx)
        print(f"   [{name}] x{cl}-{cr}: >130 亮像素 {int(m2.sum()):3d} 最长竖直 {best}px")

    # 左下 HUD 条带里的白色连通域（找弹匣/枪剪影以定位位移）
    hb0, hb1, hb2, hb3 = (int(190 * k), int(1230 * k), int(470 * k), int(1310 * k))
    strip = lum[hb1:hb3, hb0:hb2]
    mw = (strip > 130) & (chroma[hb1:hb3, hb0:hb2] < 40)
    lab = np.zeros(mw.shape, np.int32)
    cur = 0
    boxes = []
    H, W = mw.shape
    for yy in range(H):
        for xx in range(W):
            if mw[yy, xx] and lab[yy, xx] == 0:
                cur += 1
                stack = [(yy, xx)]
                lab[yy, xx] = cur
                miny = maxy = yy
                minx = maxx = xx
                n = 0
                while stack:
                    cy, cx = stack.pop()
                    n += 1
                    miny, maxy = min(miny, cy), max(maxy, cy)
                    minx, maxx = min(minx, cx), max(maxx, cx)
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W and mw[ny, nx] and lab[ny, nx] == 0:
                                lab[ny, nx] = cur
                                stack.append((ny, nx))
                if n >= 30:
                    boxes.append((n, minx / k + 190, miny / k + 1230,
                                  maxx / k + 190, maxy / k + 1230))
    print("   HUD 条带内连通域 (2560 坐标, 像素数>=30):")
    for n, bx0, by0, bx1, by1 in sorted(boxes, key=lambda z: z[1]):
        print(f"      {n:5d}px  x {bx0:6.0f}~{bx1:6.0f} (宽{bx1-bx0:4.0f})  "
              f"y {by0:6.0f}~{by1:6.0f} (高{by1-by0:4.0f})")
