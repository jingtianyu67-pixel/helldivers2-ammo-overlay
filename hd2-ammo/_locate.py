# -*- coding: utf-8 -*-
"""在视频帧里自动定位左下角 HUD 的弹药图标。

输出：亮像素的 ASCII 图 + 连通域候选列表（bbox / 尺寸 / 面积）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
FF = Path(open(Path("/tmp/ffpath.txt").read().strip(), encoding="utf-8").read().strip()) \
    if Path("/tmp/ffpath.txt").exists() else None

VIDEO = r"E:\record\HELLDIVERS 2\HELLDIVERS 2 2026.08.24 - 00.56.02.02.DVR.mp4"


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def grab(t: float, box: tuple[int, int, int, int]) -> Image.Image:
    """box = (x, y, w, h)"""
    x, y, w, h = box
    out = HERE / "_vid" / f"_loc_{t}.png"
    cmd = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", str(t), "-i", VIDEO,
           "-vf", f"crop={w}:{h}:{x}:{y}", "-frames:v", "1", str(out)]
    subprocess.run(cmd, check=True)
    return Image.open(out).convert("RGB")


def connected_components(mask: np.ndarray, min_px: int = 6):
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    cur = 0
    res = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or lab[y0, x0]:
                continue
            cur += 1
            stack = [(y0, x0)]
            lab[y0, x0] = cur
            pts = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = cur
                            stack.append((ny, nx))
            if len(pts) >= min_px:
                ys = [p[0] for p in pts]
                xs = [p[1] for p in pts]
                res.append(dict(area=len(pts), y0=min(ys), y1=max(ys),
                                x0=min(xs), x1=max(xs)))
    return res, lab


def main() -> None:
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
    bx, by, bw, bh = 60, 860, 220, 160
    im = grab(t, (bx, by, bw, bh))
    a = np.array(im).astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)

    white = lum > 140
    red = (chroma > 30) & (r > 100)
    hot = white | red

    # ASCII：每 2x2 合并成一个字符
    print(f"=== t={t}s  窗口 ({bx},{by}) {bw}x{bh}  亮度中位 {np.median(lum):.0f} ===")
    print("图例: '#'=亮(W)  'R'=红  '.'=暗   列: 每字符=2px，行首数字=该行视频 y 坐标")
    hh, ww = hot.shape
    for yy in range(0, hh, 2):
        row = ""
        for xx in range(0, ww, 2):
            blk_h = hot[yy:yy + 2, xx:xx + 2]
            blk_r = red[yy:yy + 2, xx:xx + 2]
            if blk_r.any():
                row += "R"
            elif blk_h.sum() >= 2:
                row += "#"
            elif blk_h.any():
                row += "+"
            else:
                row += "."
        print(f"{by + yy:4d} {row}")
    # 列标尺
    ruler = "     "
    for xx in range(0, ww, 2):
        vx = bx + xx
        ruler += str((vx // 10) % 10) if vx % 10 == 0 else " "
    print(ruler)
    print("     每 10px 一个数字位，读作 x 坐标的十位")

    comps, _ = connected_components(hot, min_px=8)
    comps.sort(key=lambda c: -c["area"])
    print(f"\n连通域 top12（视频坐标）:")
    for c in comps[:12]:
        print(f"  area={c['area']:5d}  x {bx + c['x0']}-{bx + c['x1']}  "
              f"y {by + c['y0']}-{by + c['y1']}  "
              f"尺寸 {c['x1'] - c['x0'] + 1}x{c['y1'] - c['y0'] + 1}")


if __name__ == "__main__":
    main()
