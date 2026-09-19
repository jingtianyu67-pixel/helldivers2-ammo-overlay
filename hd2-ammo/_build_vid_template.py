# -*- coding: utf-8 -*-
"""从视频 slab 帧中提取「倾斜长方体」剪影模板与内部窗口掩膜。

产出：
  assets/vid_slab_silhouette.npy   剪影（含轮廓）
  assets/vid_slab_interior.npy     内部窗口（腐蚀后）
  _out/vid_template.png            可视化
"""
from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.join(HERE, "_vid", "seq")
OUT = os.path.join(HERE, "_out")
AST = os.path.join(HERE, "assets")
FPS = 2.0


def mask_of(a, margin=30):
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)
    bg = np.median(np.concatenate([lum[:, :3], lum[:, -3:]], axis=1), axis=1)
    return (lum > bg[:, None] + margin) | ((chroma > 30) & (r > 100))


def shift(m, dy, dx):
    out = np.zeros_like(m)
    h, w = m.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def erode(m, k=1):
    out = m.copy()
    for _ in range(k):
        e = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                e &= shift(out, dy, dx)
        out = e
    return out


def largest_cc(m):
    h, w = m.shape
    lab = np.zeros((h, w), bool)
    best = None
    seen = np.zeros((h, w), bool)
    for y0 in range(h):
        for x0 in range(w):
            if not m[y0, x0] or seen[y0, x0]:
                continue
            stack, pts = [(y0, x0)], []
            seen[y0, x0] = True
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if best is None or len(pts) > len(best):
                best = pts
    if best:
        for y, x in best:
            lab[y, x] = True
    return lab


def main():
    allf = sorted(glob.glob(os.path.join(SEQ, "*.png")))
    ids = [int(x) for x in
           open(os.path.join(OUT, "slab_frames.txt"), encoding="utf-8").read().strip().split(",")]
    print(f"slab 帧 {len(ids)}")
    masks = []
    for i in ids:
        a = np.array(Image.open(allf[i]).convert("RGB")).astype(np.int16)
        masks.append(mask_of(a))
    masks = np.array(masks)
    print("掩膜堆叠:", masks.shape, "每帧像素数中位:", int(np.median(masks.sum(axis=(1, 2)))))

    # 多数投票 + 取最大连通域 → 剪影
    vote = masks.sum(axis=0) >= max(2, int(len(masks) * 0.4))
    sil = largest_cc(vote)
    ys, xs = np.where(sil)
    print(f"剪影 bbox: x {xs.min()}-{xs.max()}  y {ys.min()}-{ys.max()}  "
          f"({xs.max() - xs.min() + 1}x{ys.max() - ys.min() + 1})  面积 {sil.sum()}")

    interior_all = erode(sil, 1)
    interior = largest_cc(interior_all)
    print(f"腐蚀1次后面积 {interior.sum()}")

    # 充填率
    print("\n各 slab 帧的填充率（按帧序）:")
    fr = []
    for i, m in zip(ids, masks):
        fill = int((m & interior).sum())
        fr.append((int(i), int(i) / FPS, fill, fill / max(1, interior.sum()),
                   int((m & ~sil).sum())))
    fr_sorted = sorted(fr, key=lambda z: z[3])
    print("  最低 8:", [f"{t:.1f}s:{r * 100:.0f}%" for _, t, _, r, _ in fr_sorted[:8]])
    print("  最高 8:", [f"{t:.1f}s:{r * 100:.0f}%" for _, t, _, r, _ in fr_sorted[-8:]])
    outside = [o for _, _, _, _, o in fr]
    print(f"  落在剪影外的像素: 中位 {int(np.median(outside))} max {max(outside)} "
          f"（越小说明剪影覆盖越完整）")

    os.makedirs(AST, exist_ok=True)
    np.save(os.path.join(AST, "vid_slab_silhouette.npy"), sil)
    np.save(os.path.join(AST, "vid_slab_interior.npy"), interior)

    # 可视化：剪影 / 内部窗口 / 一帧实例
    SC = 14
    h, w = sil.shape
    def paint(m, base=None):
        im = np.zeros((h, w, 3), np.uint8)
        im[:] = (30, 30, 34)
        if base is not None:
            im[base] = (90, 90, 95)
        im[m] = (80, 230, 120)
        return Image.fromarray(im).resize((w * SC, h * SC), Image.NEAREST)
    panels = [paint(sil), paint(interior, sil)]
    a0 = np.array(Image.open(allf[ids[5]]).convert("RGB"))
    panels.append(Image.fromarray(a0).resize((w * SC, h * SC), Image.NEAREST))
    m0 = mask_of(np.array(Image.open(allf[ids[5]]).convert("RGB")).astype(np.int16))
    panels.append(paint(m0 & sil, sil))
    W = w * SC * 4 + 30
    canvas = Image.new("RGB", (W, h * SC), (18, 18, 20))
    for k, p in enumerate(panels):
        canvas.paste(p, (k * (w * SC + 10), 0))
    canvas.save(os.path.join(OUT, "vid_template.png"))
    print(f"\n模板与预览 -> assets/vid_slab_*.npy, _out/vid_template.png  {canvas.size}")


if __name__ == "__main__":
    main()
