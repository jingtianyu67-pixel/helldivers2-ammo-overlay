# -*- coding: utf-8 -*-
"""按时间轴列出每帧类别，并识别「方形凹槽」这类非目标图标出现的时段。"""
from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.join(HERE, "_vid", "seq")
FPS = 2.0


def load(fp):
    return np.array(Image.open(fp).convert("RGB")).astype(np.int16)


def feats(a):
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)
    white = (lum > 140) & (chroma <= 30)
    red = (chroma > 30) & (r > 100)
    hot = white | red
    # 亮像素的垂直范围（判断是否整屏亮）
    ys, xs = np.where(hot)
    if len(ys) == 0:
        return dict(white=0, red=0, hot=0, span_y=0, span_x=0, fill_ratio=0.0)
    return dict(white=int(white.sum()), red=int(red.sum()), hot=int(hot.sum()),
                span_y=int(ys.max() - ys.min() + 1), span_x=int(xs.max() - xs.min() + 1),
                fill_ratio=float(hot.mean()))


def cls(f):
    if f["hot"] < 8:
        return "NONE"          # 无亮内容
    if f["fill_ratio"] > 0.80 and f["span_x"] >= 28 and f["span_y"] >= 38:
        return "FULLBG"        # 背景整体亮（雪地等）
    if f["red"] > f["white"]:
        return "RED"
    return "WHT"


def main():
    files = sorted(glob.glob(os.path.join(SEQ, "*.png")))
    seq = []
    for i, fp in enumerate(files):
        f = feats(load(fp))
        f["i"] = i
        f["t"] = i / FPS
        f["cls"] = cls(f)
        seq.append(f)

    # 时间轴直方：每 10 秒一格的类别分布
    print("时间轴（每格 10s，字母=该时段主导类别）:")
    line = []
    for b in range(0, len(seq), int(10 * FPS)):
        chunk = seq[b:b + int(10 * FPS)]
        from collections import Counter
        c = Counter(x["cls"] for x in chunk)
        line.append(f"{int(chunk[0]['t']):3d}s:{c.most_common(1)[0][0][:4]}")
    for i in range(0, len(line), 8):
        print("   " + "  ".join(line[i:i + 8]))

    for k in ["NONE", "FULLBG", "RED", "WHT"]:
        sub = [x for x in seq if x["cls"] == k]
        print(f"\n{k} 共 {len(sub)} 帧，时间点:", end=" ")
        ts = [f"{x['t']:.0f}" for x in sub[:40]]
        print(", ".join(ts) + (" ..." if len(sub) > 40 else ""))

    # WHT 里再按 fill_ratio 分布看看有几种形态
    wht = [x for x in seq if x["cls"] == "WHT"]
    fr = np.array([x["fill_ratio"] for x in wht])
    print(f"\nWHT 的 fill_ratio: min={fr.min():.2f} 中位={np.median(fr):.2f} max={fr.max():.2f}")
    for lo, hi in [(0, .10), (.10, .20), (.20, .30), (.30, .45), (.45, .60), (.60, 1.0)]:
        sub = [x for x in wht if lo <= x["fill_ratio"] < hi]
        if sub:
            print(f"  fill {lo:.2f}-{hi:.2f}: {len(sub):3d} 帧  例: "
                  + ", ".join(f"{x['t']:.1f}s" for x in sub[:8]))


if __name__ == "__main__":
    main()
