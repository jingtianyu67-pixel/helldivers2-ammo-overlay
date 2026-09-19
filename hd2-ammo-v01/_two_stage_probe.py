# -*- coding: utf-8 -*-
"""两级重构的前置测量：轮廓（存在性）掩膜能不能独立于「填充量」工作。

现状问题：存在性判定和余量测量共用 (white|red) 掩膜 —— 弹少时填充变细，
凸包形状跟着塌，IoU 掉到 0.46 附近，还被迫给红色状态单开一条宽松通道。

候选方案：轮廓 = 与「逐行局部背景」的**绝对偏差**超过阈值（不管比背景亮还是暗），
这样弹匣外框永远可见，形状校验与弹量彻底解耦。

本脚本对三组样本量这个新判据：
  A 用户 6 张样张（B 区域生产尺度）
  B 视频 slab 正样本（1080p 尺度）
  C 视频干扰帧（应为负）
并顺带量出对齐基准（连通域 bbox 的中心/底边 vs 窗口 bbox）。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ammo_detect import (RegionDetector, convex_hull_mask, erode,   # noqa: E402
                        iou, largest_cc, normalize_shape)

SEQ = os.path.join(HERE, "_vid", "seq")
OUT = os.path.join(HERE, "_out")
AST = os.path.join(HERE, "assets")
SHOTS = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
FPS = 2.0

TM = np.load(os.path.join(AST, "shape_template.npy"))
print(f"模板 {TM.shape}")


def masks(rgb, sil_margin):
    a = rgb.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)
    e = max(1, min(5, rgb.shape[1] // 3))
    bg = np.median(np.concatenate([lum[:, :e], lum[:, -e:]], axis=1), axis=1)
    dev = np.abs(lum - bg[:, None])
    sil = dev > sil_margin
    red = (chroma > 18) & (r > 70) & (r > g * 1.4) & (r > b * 1.4)
    return sil, red


def iou_of(rgb, sil_margin, win, red_ok=True):
    sil, red = masks(rgb, sil_margin)
    m = sil | red if red_ok else sil
    if win is not None:
        m = m  # 不做裁剪，只用位置约束否决
    cc = largest_cc(m, 12)
    if cc.sum() < 12:
        return None
    if win is not None:
        from ammo_detect import dilate
        if not bool((cc & dilate(win, 3)).any()):
            return None
    hull = convex_hull_mask(cc)
    ys, xs = np.nonzero(cc)
    bb = (xs.min(), ys.min(), xs.max(), ys.max())
    return dict(iou=iou(normalize_shape(hull), TM), bb=bb, n=int(cc.sum()))


bwin = np.load(os.path.join(AST, "interior_mask.npy"))
sil_vid = np.load(os.path.join(AST, "vid_slab_silhouette.npy"))
vwin = erode(sil_vid, 1)

# ---- 样本集 ---------------------------------------------------------------- #
pos_u, pos_v, neg_v = [], [], []
for fp in sorted(glob.glob(os.path.join(SHOTS, "*_1x.png"))):
    a = np.array(Image.open(fp).convert("RGB"))
    if a.shape[:2] == bwin.shape:
        pos_u.append((os.path.basename(fp), a))
ids = set(int(x) for x in open(os.path.join(OUT, "slab_frames.txt"),
                               encoding="utf-8").read().strip().split(","))
for i, fp in enumerate(sorted(glob.glob(os.path.join(SEQ, "*.png")))):
    a = np.array(Image.open(fp).convert("RGB"))
    (pos_v if i in ids else neg_v).append((f"{i / FPS:.1f}s", a))
print(f"样本: 用户白/红 {len(pos_u)} 张, 视频正 {len(pos_v)} 帧, 视频负 {len(neg_v)} 帧\n")

print(f"{'sil_margin':>10} | {'用户样张 IoU':>28} | {'视频正 pass/中位IoU':>22} | {'视频负 误报率':>12}")
print("-" * 92)
best = None
for sm in (6, 8, 10, 12, 15, 20):
    u = [iou_of(a, sm, bwin) for _, a in pos_u]
    uiou = [x["iou"] for x in u if x]
    upass = sum(1 for x in u if x and x["iou"] >= 0.46)
    v = [iou_of(a, sm, vwin) for _, a in pos_v]
    viou = [x["iou"] for x in v if x]
    vpass = sum(1 for x in v if x and x["iou"] >= 0.46)
    n = [iou_of(a, sm, vwin) for _, a in neg_v]
    nfp = sum(1 for x in n if x and x["iou"] >= 0.46)
    s_u = ",".join(f"{x['iou']:.2f}" if x else "--" for x in u)
    print(f"{sm:>10} | {s_u:>28} | {vpass:>4}/{len(v):<4} {np.median(viou) if viou else 0:>8.3f}"
          f" ({upass}/{len(u)}) | {nfp / len(n) * 100:>10.1f}%")
    if sm == 10:
        best = (u, v, n)

print("\n=== sil_margin=10 明细：用户样张（对齐基准则用这组）===")
u, v, n = best
for (name, _), x in zip(pos_u, u):
    if x:
        x0, y0, x1, y1 = x["bb"]
        print(f"  {name}: IoU={x['iou']:.2f} bbox=({x0},{y0})-({x1},{y1}) "
              f"W{x1 - x0 + 1}xH{y1 - y0 + 1} 中心x={(x0 + x1) / 2:.1f} 底y={y1} 面积{x['n']}")

wy, wx = np.nonzero(bwin)
print(f"\n窗口 bbox=({wx.min()},{wy.min()})-({wx.max()},{wy.max()}) "
      f"W{wx.max() - wx.min() + 1}xH{wy.max() - wy.min() + 1} "
      f"中心x={(wx.min() + wx.max()) / 2:.1f} 底y={wy.max()} 面积{int(bwin.sum())}")

# 对齐基准 = 用户样张里 IoU 最高那张的 bbox（对齐最好的一张当参考姿态）
ok = [(x["iou"], x["bb"]) for x in u if x]
if ok:
    ok.sort(key=lambda t: -t[0])
    x0, y0, x1, y1 = ok[0][1]
    print(f"\n参考姿态（取 IoU 最高的样张）: 中心x={(x0 + x1) / 2:.1f} 底y={y1} "
          f"W{x1 - x0 + 1}xH{y1 - y0 + 1}")
