# -*- coding: utf-8 -*-
"""扫描视频序列，统计每帧弹匣区域的特征，输出分布与代表帧拼图。"""
from __future__ import annotations

import glob
import os
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.join(HERE, "_vid", "seq")


def load(fp):
    return np.array(Image.open(fp).convert("RGB")).astype(np.int16)


def feats(a):
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)
    white = (lum > 140) & (chroma <= 30)
    red = (chroma > 30) & (r > 100)
    return dict(lum_max=float(lum.max()), lum_mean=float(lum.mean()),
                white=int(white.sum()), red=int(red.sum()),
                hot=int((white | red).sum()))


def main():
    files = sorted(glob.glob(os.path.join(SEQ, "*.png")))
    print(f"共 {len(files)} 帧")
    rows = []
    for fp in files:
        f = feats(load(fp))
        f["name"] = os.path.basename(fp)[:-4]
        rows.append(f)

    # 分档统计
    def bucket(f):
        if f["hot"] < 8:
            return "A_几乎无亮(可能无武器/离屏)"
        if f["red"] > f["white"]:
            return "B_红"
        return "C_白"

    cnt = Counter(bucket(f) for f in rows)
    for k in sorted(cnt):
        print(f"  {k}: {cnt[k]}")

    hot = np.array([f["hot"] for f in rows])
    print(f"\nhot 像素: min={hot.min()} 中位={np.median(hot):.0f} max={hot.max()}")
    print(f"lum_max: min={min(f['lum_max'] for f in rows):.0f} "
          f"中位={np.median([f['lum_max'] for f in rows]):.0f} "
          f"max={max(f['lum_max'] for f in rows):.0f}")

    # 每档挑代表：按 hot 分位取
    picks = []
    for k in sorted(cnt):
        sub = [f for f in rows if bucket(f) == k]
        sub.sort(key=lambda f: f["hot"])
        idx = np.linspace(0, len(sub) - 1, min(12, len(sub))).astype(int)
        picks.append((k, [sub[i] for i in idx]))

    # 拼图
    SC = 8
    W, H = 30 * SC, 41 * SC
    per_row = 12
    total_rows = sum((len(p) + per_row - 1) // per_row for _, p in picks) + len(picks)
    canvas = Image.new("RGB", (W * per_row + 10, H * total_rows + 40), (25, 25, 28))
    d = ImageDraw.Draw(canvas)
    cy = 4
    for k, plist in picks:
        d.text((4, cy), f"--- {k} ({len(plist)} shown) ---", fill=(120, 255, 120))
        cy += 18
        for i, f in enumerate(plist):
            im = Image.open(os.path.join(SEQ, f["name"] + ".png")).convert("RGB")
            im = im.resize((W, H), Image.NEAREST)
            x = (i % per_row) * W + 5
            y = cy + (i // per_row) * H
            canvas.paste(im, (x, y))
            d.text((x + 2, y + 2), f"{f['name']}", fill=(0, 255, 0))
            d.text((x + 2, y + H - 12), f"h{f['hot']} r{f['red']}", fill=(255, 220, 0))
        cy += ((len(plist) + per_row - 1) // per_row) * H
    out = os.path.join(HERE, "_out", "vid_survey.png")
    canvas.save(out)
    print(f"\n拼图 -> {out}  {canvas.size}")


if __name__ == "__main__":
    main()
