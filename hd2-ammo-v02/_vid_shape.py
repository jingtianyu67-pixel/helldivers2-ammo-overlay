# -*- coding: utf-8 -*-
"""按形状特征给视频序列分类：倾斜长方体（目标）vs 方形凹槽（非目标）vs 其他。"""
from __future__ import annotations

import glob
import os
from collections import Counter

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.join(HERE, "_vid", "seq")
FPS = 2.0


def mask_of(a):
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)
    # 逐行背景参照：每行取区域左右边缘 3px 的亮度中位作为背景
    bg = np.median(np.concatenate([lum[:, :3], lum[:, -3:]], axis=1), axis=1)
    return (lum > bg[:, None] + 30) | ((chroma > 30) & (r > 100))


def main():
    files = sorted(glob.glob(os.path.join(SEQ, "*.png")))
    rows = []
    for i, fp in enumerate(files):
        a = np.array(Image.open(fp).convert("RGB")).astype(np.int16)
        m = mask_of(a)
        ys, xs = np.where(m)
        if len(ys) < 10:
            rows.append(dict(i=i, t=i / FPS, kind="none", n=len(ys)))
            continue
        # 去掉离群点：用分位数裁
        y0, y1 = int(np.percentile(ys, 2)), int(np.percentile(ys, 98))
        x0, x1 = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
        h, w = y1 - y0 + 1, x1 - x0 + 1
        rows.append(dict(i=i, t=i / FPS, kind=None, n=len(ys), x0=x0, x1=x1, y0=y0, y1=y1,
                         w=w, h=h, ar=w / max(1, h)))

    # 分类：宽高比 0.5~0.75 且高 24~36 → 倾斜长方体；宽高比 0.85~1.2 → 方形
    for r in rows:
        if r["kind"] == "none":
            continue
        ar, h = r["ar"], r["h"]
        if 0.45 <= ar <= 0.80 and 22 <= h <= 40:
            r["kind"] = "slab"      # 倾斜长方体
        elif 0.80 < ar <= 1.35 and 18 <= h <= 34:
            r["kind"] = "square"    # 方形
        else:
            r["kind"] = f"other_{ar:.2f}x{h}"

    c = Counter(r["kind"].split("_")[0] if r["kind"].startswith("other") else r["kind"]
                for r in rows)
    print("形状分类:", dict(c))

    print("\n时间轴（每 10s 一格的形状分布，S=slab Q=square n=none）:")
    line = []
    for b in range(0, len(rows), int(10 * FPS)):
        chunk = rows[b:b + int(10 * FPS)]
        cc = Counter(x["kind"] for x in chunk)
        s = cc.get("slab", 0)
        q = cc.get("square", 0)
        n = cc.get("none", 0) + sum(v for k, v in cc.items() if k.startswith("other"))
        top = max([("S", s), ("Q", q), ("n", n)], key=lambda z: z[1])[0]
        line.append(f"{int(chunk[0]['t']):3d}:{top}({s}/{q}/{n})")
    for i in range(0, len(line), 6):
        print("  " + " ".join(line[i:i + 6]))

    # slab 的宽高比分布
    slabs = [r for r in rows if r["kind"] == "slab"]
    print(f"\nslab {len(slabs)} 帧，ar 中位={np.median([r['ar'] for r in slabs]):.2f} "
          f"高 中位={np.median([r['h'] for r in slabs]):.0f} "
          f"宽 中位={np.median([r['w'] for r in slabs]):.0f}")
    sq = [r for r in rows if r["kind"] == "square"]
    if sq:
        print(f"square {len(sq)} 帧，ar 中位={np.median([r['ar'] for r in sq]):.2f} "
              f"高 中位={np.median([r['h'] for r in sq]):.0f} 时间例: "
              + ", ".join(f"{r['t']:.0f}" for r in sq[:20]))

    # 导出 slab 帧列表
    out = os.path.join(HERE, "_out", "slab_frames.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(",".join(f"{r['i']:04d}" for r in slabs))
    print(f"\nslab 帧号 -> {out}")


if __name__ == "__main__":
    main()
