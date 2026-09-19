# -*- coding: utf-8 -*-
"""调参诊断：分离正/负样本，找形状阈值与红色判据的最佳取值。"""
from __future__ import annotations

import glob
import os
from collections import Counter

import numpy as np
from PIL import Image

from ammo_detect import (convex_hull_mask, dilate, erode, iou, largest_cc,
                         normalize_shape)

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.join(HERE, "_vid", "seq")
OUT = os.path.join(HERE, "_out")
AST = os.path.join(HERE, "assets")
FPS = 2.0


def masks(rgb, window, *, bg_margin=18, lum_floor=55, chroma_min=18, r_min=70,
          edge=5, near_dilate=3, red_ratio=None, red_r_min=70):
    a = rgb.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)
    e = max(1, min(edge, rgb.shape[1] // 3))
    side = np.concatenate([lum[:, :e], lum[:, -e:]], axis=1)
    bg = np.median(side, axis=1)
    white = (lum > bg[:, None] + bg_margin) & (lum > lum_floor) & (chroma < 25)

    red = (chroma > chroma_min) & (r > r_min)
    if red_ratio is not None:
        red &= (r > g * red_ratio) & (r > b * red_ratio)
    red = red.astype(bool)

    # 不裁剪，只取最大连通域（裁剪会毁掉形状可分性）
    hot = largest_cc(white | red, 1)
    hred = largest_cc(red, 1)
    return white, red, hot, hred


def probe(red_ratio, red_shape_min_iou, shape_min_iou, near_dilate=2):
    allf = sorted(glob.glob(os.path.join(SEQ, "*.png")))
    ids = set(int(x) for x in
              open(os.path.join(OUT, "slab_frames.txt"), encoding="utf-8").read().strip().split(","))
    sil = np.load(os.path.join(AST, "vid_slab_silhouette.npy"))
    win = erode(sil, 1)
    T = np.load(os.path.join(AST, "shape_template.npy"))

    pos_ok = neg_bad = 0
    pos_ious, neg_ious = [], []
    pos_detail = Counter()
    for i, fp in enumerate(allf):
        a = np.array(Image.open(fp).convert("RGB"))
        white, red, hot, hred = masks(a, win, near_dilate=near_dilate, red_ratio=red_ratio)
        mp, rp = int(hot.sum()), int(hred.sum())
        hull = convex_hull_mask(hot if mp > 0 else hred)
        v = iou(normalize_shape(hull), T) if mp or rp else 0.0
        is_slab = i in ids
        (pos_ious if is_slab else neg_ious).append(v)
        # 判定
        valid = False
        tag = ""
        if mp < 30 and rp < 15:
            tag = "掩膜少"
        elif rp >= 15 and v >= red_shape_min_iou:
            valid = True
            tag = "红"
        elif mp >= 30 and v >= shape_min_iou:
            valid = True
            tag = "白"
        else:
            tag = "形状"
        if is_slab:
            pos_ok += valid
            if not valid:
                pos_detail[tag] += 1
        else:
            neg_bad += valid
    return pos_ok, neg_bad, np.array(pos_ious), np.array(neg_ious), pos_detail


sil_window = None
if __name__ == "__main__":
    print("配置: near_dilate=2（视频尺度）")
    print(f"{'red_ratio':>9} {'red_iou':>7} {'white_iou':>9} | {'正通过':>6} {'负误报':>6} | "
          f"正IoU中位  负IoU中位  正p5")
    for red_ratio in (None, 1.4, 1.6):
        for red_iou in (0.30, 0.40):
            for white_iou in (0.50, 0.55, 0.60):
                p, n, pi, ni, det = probe(red_ratio, red_iou, white_iou)
                print(f"{str(red_ratio):>9} {red_iou:>7.2f} {white_iou:>9.2f} | "
                      f"{p:>4}/123 {n:>4}/398 | {np.median(pi):>8.3f}  {np.median(ni):>8.3f}  "
                      f"{np.percentile(pi, 5):>6.3f}")
    print("\n最终候选的无效原因分布:")
    for red_ratio in (1.6,):
        for red_iou in (0.40,):
            for white_iou in (0.50,):
                p, n, pi, ni, det = probe(red_ratio, red_iou, white_iou)
                print(f"  red_ratio={red_ratio} red_iou={red_iou} white_iou={white_iou}: "
                      f"正通过 {p}/123, 负误报 {n}/398, 正样本失败原因 {dict(det)}")
