# -*- coding: utf-8 -*-
"""从参考图里提取弧条的中心线、粗细、端点与配色。

思路：弧条是「灰度带 + 白色末端块」，逐行取连通段并做连续性跟踪，
排除背景里同色但不相连的元素（漂浮的世界物件、白色箭头等）。
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

SRC = r"C:\Users\Administrator\.workbuddy\clipboard-images\clipboard-2026-09-18T13-18-21-263Z-c10ca407.png"
OUT = "_out"


def segs(mask_row: np.ndarray) -> list[tuple[int, int]]:
    """把一行的布尔掩膜切成连续段。"""
    x = np.nonzero(mask_row)[0]
    if not len(x):
        return []
    out, s, p = [], x[0], x[0]
    for v in x[1:]:
        if v != p + 1:
            out.append((int(s), int(p)))
            s = v
        p = v
    out.append((int(s), int(p)))
    return out


def main() -> None:
    im = Image.open(SRC).convert("RGB")
    a = np.array(im).astype(np.int16)
    H, W, _ = a.shape
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    band = (chroma < 30) & (lum > 88)
    tip = (chroma < 30) & (lum >= 205)

    print(f"图幅 {W}x{H}  灰度带 {int(band.sum())}px  白块 {int(tip.sum())}px")
    print()
    print("逐行跟踪（取与上一行中心最接近的段）：")
    rows: list[tuple[int, float, int, int, bool]] = []
    prev_cx: float | None = None
    for y in range(H):
        cand = segs(band[y])
        if not cand:
            prev_cx = None
            continue
        if prev_cx is None:
            # 起始行：选最靠右的段（弧条从右上开始，左上那颗是图标）
            s, e = max(cand, key=lambda t: t[1])
        else:
            s, e = min(cand, key=lambda t: abs((t[0] + t[1]) / 2 - prev_cx))
            if abs((s + e) / 2 - prev_cx) > 14:      # 跳太远 = 不是同一条
                prev_cx = None
                continue
        cx = (s + e) / 2
        is_tip = bool(tip[y, s:e + 1].mean() > 0.5)
        rows.append((y, cx, s, e, is_tip))
        prev_cx = cx

    print(f"  跟踪到 {len(rows)} 行，y {rows[0][0]}..{rows[-1][0]}")
    for y, cx, s, e, t in rows[::4]:
        print(f"    y{y:>3}  x{s:>3}-{e:>3} 宽{e - s + 1:>2} 中心{cx:6.1f}{'  <白末端>' if t else ''}")

    ys = np.array([q[0] for q in rows], float)
    xs = np.array([q[1] for q in rows], float)
    wid = np.array([q[3] - q[2] + 1 for q in rows], float)

    # 最小二乘圆拟合
    A = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    bb = xs ** 2 + ys ** 2
    sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
    cx0, cy0 = sol[0], sol[1]
    R = float(np.sqrt(sol[2] + cx0 ** 2 + cy0 ** 2))
    res = np.abs(np.hypot(xs - cx0, ys - cy0) - R)
    print()
    print(f"圆拟合：圆心 ({cx0:.1f}, {cy0:.1f})  半径 {R:.1f}  残差 中位{np.median(res):.2f} 最大{res.max():.2f}")
    ang = np.degrees(np.arctan2(ys - cy0, xs - cx0))
    print(f"角度范围 {ang.min():.1f}° .. {ang.max():.1f}° （0°=正右，90°=正下）")
    print(f"粗细：中位 {np.median(wid):.1f}  最小 {wid.min():.0f}  最大 {wid.max():.0f}")

    # 配色：轨道灰、末端白
    tip_rows = [q for q in rows if q[4]]
    print()
    if tip_rows:
        print(f"白末端：y {tip_rows[0][0]}..{tip_rows[-1][0]}（{len(tip_rows)} 行）"
              f"，中心 x {min(q[1] for q in tip_rows):.1f}..{max(q[1] for q in tip_rows):.1f}")
    gm, wm = [], []
    for y, cx, s, e, t in rows:
        seg = a[y, s:e + 1]
        (wm if t else gm).append(seg.mean(0))
    gm = np.array(gm).mean(0) if gm else np.zeros(3)
    wm = np.array(wm).mean(0) if wm else np.zeros(3)
    print(f"轨道色 RGB {gm.round(0)} （亮度 {0.299 * gm[0] + 0.587 * gm[1] + 0.114 * gm[2]:.0f}）"
          f"  白末端 RGB {wm.round(0)}")

    # 套合验证图：把拟合圆与跟踪到的中心线画回原图
    chk = im.copy().convert("RGB").resize((W * 3, H * 3), Image.NEAREST)
    px = chk.load()
    for t in np.linspace(np.radians(ang.min()) - 0.3, np.radians(ang.max()) + 0.3, 900):
        X = (cx0 + R * np.cos(t)) * 3
        Y = (cy0 + R * np.sin(t)) * 3
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                xi, yi = int(X) + ox, int(Y) + oy
                if 0 <= xi < W * 3 and 0 <= yi < H * 3:
                    px[xi, yi] = (0, 255, 0)
    for y, cx, s, e, t in rows:
        for ox in (-1, 0, 1):
            ix = int(cx * 3) + ox
            if 0 <= ix < W * 3:
                px[ix, y * 3] = (255, 0, 255)
    chk.save(os.path.join(OUT, "arc_fit.png"))
    print("-> _out/arc_fit.png（绿=拟合圆，品红=实测中心线）")

    # 端点与「相对准星」的几何（准星假定在裁剪图右侧 x=W-3 的中偏下处）
    print()
    print("端点（裁剪图局部坐标）：")
    print(f"  上端 y{rows[0][0]}  x{rows[0][1]:.1f}")
    print(f"  下端 y{rows[-1][0]}  x{rows[-1][1]:.1f}")
    print(f"  最右点 x{xs.max():.1f} @ y{ys[xs.argmax()]:.0f}")

    np.save(os.path.join(OUT, "arc_centerline.npy"),
            np.c_[xs, ys].astype(np.float32))
    with open(os.path.join(OUT, "arc_geo.json"), "w", encoding="utf-8") as f:
        json.dump({"size": [W, H], "center": [cx0, cy0], "radius": R,
                   "ang_deg": [float(ang.min()), float(ang.max())],
                   "width_med": float(np.median(wid)),
                   "track_rgb": gm.tolist(), "tip_rgb": wm.tolist(),
                   "top": [float(rows[0][1]), rows[0][0]],
                   "bottom": [float(rows[-1][1]), rows[-1][0]]},
                  f, ensure_ascii=False, indent=2)
    print("\n-> _out/arc_centerline.npy  _out/arc_geo.json")


if __name__ == "__main__":
    main()
