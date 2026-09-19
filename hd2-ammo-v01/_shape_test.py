# -*- coding: utf-8 -*-
"""形状判别验证：用归一化剪影 IoU 区分「倾斜长方体（目标）」与「方形凹槽（非目标）」。

同时对比视频帧与用户屏幕样张（40x55 区域），确认尺度无关性。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.join(HERE, "_vid", "seq")
OUT = os.path.join(HERE, "_out")
AST = os.path.join(HERE, "assets")
SHOTS = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
NORM = 32


def mask_of(a, margin=30):
    """逐行背景参照 + 红色判别，得到候选掩膜。"""
    a = a.astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)
    bg = np.median(np.concatenate([lum[:, :3], lum[:, -3:]], axis=1), axis=1)
    return (lum > bg[:, None] + margin) | ((chroma > 30) & (r > 100))


def fill_holes(m):
    """从边界泛洪，剩下的闭合区域就是洞。"""
    h, w = m.shape
    bg = np.zeros((h, w), bool)
    stack = []
    for x in range(w):
        for y in (0, h - 1):
            if not m[y, x] and not bg[y, x]:
                bg[y, x] = True
                stack.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not m[y, x] and not bg[y, x]:
                bg[y, x] = True
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not m[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True
                stack.append((ny, nx))
    return ~bg


def bbox_crop(m, pad=1):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + 1 + pad)
    return m[y0:y1, x0:x1]


def to_norm(m, n=NORM):
    """剪影 → 等比缩放到 n x n 画布居中，保持长宽比。"""
    c = bbox_crop(m)
    if c is None:
        return np.zeros((n, n), bool)
    im = Image.fromarray((c * 255).astype(np.uint8))
    scale = n / max(c.shape)
    nw, nh = max(1, int(round(c.shape[1] * scale))), max(1, int(round(c.shape[0] * scale)))
    im = im.resize((nw, nh), Image.BILINEAR)
    arr = np.array(im) > 127
    out = np.zeros((n, n), bool)
    oy, ox = (n - nh) // 2, (n - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = arr
    return out


def iou(a, b):
    i = (a & b).sum()
    u = (a | b).sum()
    return i / u if u else 0.0


def main():
    # 模板：视频 slab 剪影
    sil = np.load(os.path.join(AST, "vid_slab_silhouette.npy"))
    T = to_norm(fill_holes(sil))
    print(f"模板: 原始 {sil.shape} 面积 {sil.sum()} → 归一化 {T.shape} 面积 {T.sum()}")
    imT = Image.fromarray((T * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST)
    imT.save(os.path.join(OUT, "norm_template.png"))

    allf = sorted(glob.glob(os.path.join(SEQ, "*.png")))
    ids = [int(x) for x in
           open(os.path.join(OUT, "slab_frames.txt"), encoding="utf-8").read().strip().split(",")]
    slab_set = set(ids)

    # 抽检：slab 帧（正）与其余帧（负）
    rng = np.random.default_rng(0)
    negs = [i for i in range(len(allf)) if i not in slab_set]
    negs_sample = sorted(rng.choice(negs, size=min(200, len(negs)), replace=False).tolist())

    def score(i):
        a = np.array(Image.open(allf[i]).convert("RGB"))
        m = mask_of(a)
        if m.sum() < 20:
            return None, 0, m.sum()
        s = fill_holes(m)
        c = bbox_crop(s)
        ar = c.shape[1] / max(1, c.shape[0])
        return iou(to_norm(s), T), ar, int(m.sum())

    pos = [score(i) for i in ids]
    neg = [score(i) for i in negs_sample]
    pv = np.array([s for s, _, _ in pos if s is not None])
    nv = np.array([s for s, _, _ in neg if s is not None])
    print(f"\n正样本 (slab) n={len(pv)}  IoU: min={pv.min():.3f} "
          f"p5={np.percentile(pv, 5):.3f} 中位={np.median(pv):.3f} max={pv.max():.3f}")
    print(f"负样本 (其他) n={len(nv)}  IoU: min={nv.min():.3f} "
          f"中位={np.median(nv):.3f} p95={np.percentile(nv, 95):.3f} max={nv.max():.3f}")
    none_cnt = sum(1 for s, _, _ in pos + neg if s is None)
    print(f"掩膜过少被判空: {none_cnt}")

    # 阈值扫描
    print("\n阈值扫描 (正/负 通过率):")
    for th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        tp = (pv >= th).mean()
        fp = (nv >= th).mean()
        print(f"  >={th:.2f}: 正 {tp * 100:5.1f}%  负 {fp * 100:5.1f}%")

    # 用户屏幕样张
    print("\n用户屏幕样张（B 区域 40x55）:")
    for fp in sorted(glob.glob(os.path.join(SHOTS, "*_1x.png"))):
        a = np.array(Image.open(fp).convert("RGB"))
        m = mask_of(a)
        if m.sum() < 20:
            print(f"  {os.path.basename(fp)}: 掩膜过少 ({m.sum()})")
            continue
        s = fill_holes(m)
        c = bbox_crop(s)
        v = iou(to_norm(s), T)
        print(f"  {os.path.basename(fp)}: IoU={v:.3f} 掩膜={int(m.sum())} "
              f"bbox={c.shape[1]}x{c.shape[0]} ar={c.shape[1] / max(1, c.shape[0]):.2f}")


if __name__ == "__main__":
    main()
