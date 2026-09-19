# -*- coding: utf-8 -*-
"""回归测试 + 参数扫描。

注意：_full.png 上画了红色标注框，直接切片会把框线带进来（判成 red）。
所有 full 切片一律内缩 PAD 像素规避。
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

import numpy as np
from PIL import Image

from ammo_detect import RegionDetector, erode

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.join(HERE, "_vid", "seq")
OUT = os.path.join(HERE, "_out")
AST = os.path.join(HERE, "assets")
SHOTS = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
FPS = 2.0
PAD = 4                      # 规避 full 图上的标注框线
BOX_B = (230, 1245, 270, 1300)
BOX_A = (330, 1250, 390, 1300)

BASE = dict(bg_margin=18, lum_floor=55, chroma_min=18, r_min=70, edge=5,
            shape_min_iou=0.46, red_shape_min_iou=0.32, min_mask_px=30,
            min_fill_px=4, red_min_px=15, near_dilate=3)


def make(name, window, **over):
    kw = dict(BASE)
    kw.update(over)
    return RegionDetector(name, window, None, **kw)


def load_video():
    files = sorted(glob.glob(os.path.join(SEQ, "*.png")))
    ids = set(int(x) for x in
              open(os.path.join(OUT, "slab_frames.txt"), encoding="utf-8").read().strip().split(","))
    arrs = [np.array(Image.open(f).convert("RGB")) for f in files]
    return files, ids, arrs


def main():
    files, ids, arrs = load_video()
    sil = np.load(os.path.join(AST, "vid_slab_silhouette.npy"))
    vwin = erode(sil, 1)
    bwin = np.load(os.path.join(AST, "interior_mask.npy"))

    user_1x = sorted(glob.glob(os.path.join(SHOTS, "*_1x.png")))
    print(f"视频 {len(arrs)} 帧 / slab 正样本 {len(ids)} / 用户样张 {len(user_1x)} 张\n")
    print(f"{'red_ratio':>9} {'red_iou':>7} | {'视频正通过':>10} {'视频负误报':>10} | "
          f"{'用户白帧':>8} {'用户红帧':>8}")
    print("-" * 72)

    best = None
    for red_ratio in (None, 1.2, 1.3, 1.4):
        for red_iou in (0.32, 0.40):
            det_v = make("VID", vwin, red_ratio=red_ratio, red_shape_min_iou=red_iou,
                         near_dilate=2)
            pos_ok = neg_bad = 0
            for a, i in zip(arrs, range(len(arrs))):
                r = det_v.analyze(a)
                if i in ids:
                    pos_ok += r.valid
                else:
                    neg_bad += r.valid
            det_b = make("B", bwin, red_ratio=red_ratio, red_shape_min_iou=red_iou)
            res = []
            for fp in user_1x:
                a = np.array(Image.open(fp).convert("RGB"))
                res.append(det_b.analyze(a))
            w_ok = sum(r.valid and r.state != "red" for r in res)
            r_ok = sum(r.valid and r.state == "red" for r in res)
            print(f"{str(red_ratio):>9} {red_iou:>7.2f} | {pos_ok:>6}/123 "
                  f"{neg_bad:>6}/398 | {w_ok:>6}/4 {r_ok:>6}/2")
            if best is None or (pos_ok - neg_bad * 2) > best[0]:
                best = (pos_ok - neg_bad * 2, red_ratio, red_iou)

    print(f"\n最优: red_ratio={best[1]} red_iou={best[2]}\n")

    rr, ri = best[1], best[2]
    print("=" * 72)
    print(f"详细报告 @ red_ratio={rr} red_iou={ri}")
    det_v = make("VID", vwin, red_ratio=rr, red_shape_min_iou=ri, near_dilate=2)
    rows = [(i / FPS, i in ids, det_v.analyze(a)) for i, a in enumerate(arrs)]
    pos = [(t, r) for t, s, r in rows if s]
    neg = [(t, r) for t, s, r in rows if not s]
    ok = [r for _, r in pos if r.valid]
    print(f"视频正样本 valid {len(ok)}/{len(pos)} = {len(ok) / len(pos) * 100:.1f}% "
          f"(红 {sum(r.state == 'red' for r in ok)} / 白 {sum(r.state != 'red' for r in ok)})")
    print("  失败原因:", dict(Counter(r.reason.split("(")[0] for _, r in pos if not r.valid)))
    fr = np.array([r.fraction for r in ok])
    if len(fr):
        print(f"  百分比 min={fr.min() * 100:.0f}% 中位={np.median(fr) * 100:.0f}% "
              f"max={fr.max() * 100:.0f}%")
    bad = [(t, r) for t, r in neg if r.valid]
    print(f"视频负样本误报 {len(bad)}/{len(neg)} = {len(bad) / len(neg) * 100:.1f}%")
    if bad:
        print("  误报:", [(f"{t:.1f}s", r.state, f"{r.shape_iou:.2f}", r.mask_px, r.red_px)
                          for t, r in bad[:10]])

    print("\n时间轴（每 10s：w=白 r=红 n=未检测）:")
    line = []
    for b in range(0, len(rows), int(10 * FPS)):
        chunk = rows[b:b + int(10 * FPS)]
        rv = [r for _, _, r in chunk if r.valid]
        red = sum(r.state == "red" for r in rv)
        line.append(f"{int(chunk[0][0]):3d}:{len(rv) - red}w/{red}r/{len(chunk) - len(rv)}n")
    for i in range(0, len(line), 8):
        print("   " + " ".join(line[i:i + 8]))

    print("\n--- 用户屏幕样张 B 区域（干净区域图）---")
    det_b = make("B", bwin, red_ratio=rr, red_shape_min_iou=ri)
    for fp in user_1x:
        a = np.array(Image.open(fp).convert("RGB"))
        r = det_b.analyze(a)
        print(f"  {os.path.basename(fp)}: {'✓' if r.valid else '×'} {r.fraction * 100:5.1f}% "
              f"{r.state:5s} fill={r.fill:4d}/{r.area} IoU={r.shape_iou:.2f} "
              f"掩膜={r.mask_px:4d} 红={r.red_px:4d} {r.reason}")

    print("\n--- 用户全屏样张切 B（窗口同步内缩，规避框线）---")
    bwin_in = bwin[PAD:-PAD, PAD:-PAD]
    det_bin = make("B", bwin_in, red_ratio=rr, red_shape_min_iou=ri)
    for fp in sorted(glob.glob(os.path.join(SHOTS, "*_full.png"))):
        im = np.array(Image.open(fp).convert("RGB"))
        l, t, r_, b_ = BOX_B
        r = det_bin.analyze(im[t + PAD:b_ - PAD, l + PAD:r_ - PAD])
        print(f"  {os.path.basename(fp)}: {'✓' if r.valid else '×'} {r.fraction * 100:5.1f}% "
              f"{r.state:5s} IoU={r.shape_iou:.2f} {r.reason}")

    print("\n--- 用户全屏样张切 A（auto 模式，步枪剪影 → 应拒绝）---")
    det_a = make("A", None, red_ratio=rr, red_shape_min_iou=ri)
    for fp in sorted(glob.glob(os.path.join(SHOTS, "*_full.png"))):
        im = np.array(Image.open(fp).convert("RGB"))
        l, t, r_, b_ = BOX_A
        r = det_a.analyze(im[t + PAD:b_ - PAD, l + PAD:r_ - PAD])
        print(f"  {os.path.basename(fp)}: {'✓ 误报!' if r.valid else '× 正确拒绝'} "
              f"IoU={r.shape_iou:.2f} 掩膜={r.mask_px} {r.reason}")




if __name__ == "__main__":
    main()
