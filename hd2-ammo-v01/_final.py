# -*- coding: utf-8 -*-
"""用 config.json 的最终参数复验全部样本。

指标：
  视频 slab 正样本通过率 / 视频其它帧误报率
  用户屏幕 6 张样张（4 白 2 红）逐张结果
  A 区域（步枪剪影）是否被正确拒绝
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ammo_detect import RegionDetector, erode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.join(HERE, "_vid", "seq")
OUT = os.path.join(HERE, "_out")
AST = os.path.join(HERE, "assets")
SHOTS = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
FPS = 2.0
PAD = 4
BOX_A = (330, 1250, 390, 1300)

CFG = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
D = CFG["detect"]


def make(name, window, **over):
    kw = dict(bg_margin=D["bg_margin"], lum_floor=D["lum_floor"], edge=D["edge"],
              shape_min_iou=D["shape_min_iou"],
              min_mask_px=D["min_mask_px"], near_dilate=D["near_dilate"],
              fill_inset=D.get("fill_inset", 2), empty_frac=D.get("empty_frac", 0.06))
    kw.update(over)
    return RegionDetector(name, window, None, **kw)


def main():
    print("=== 用 config.json 实际参数复验 ===")
    print(f"    shape_iou={D['shape_min_iou']}  "
          f"bg_margin={D['bg_margin']}  "
          f"lum_floor={D['lum_floor']}  hold={D['hold_frames']}")

    files = sorted(glob.glob(os.path.join(SEQ, "*.png")))
    ids = set(int(x) for x in
              open(os.path.join(OUT, "slab_frames.txt"), encoding="utf-8").read().strip().split(","))
    arrs = [np.array(Image.open(f).convert("RGB")) for f in files]
    bwin = np.load(os.path.join(AST, "interior_mask.npy"))
    sil = np.load(os.path.join(AST, "vid_slab_silhouette.npy"))

    det_v = make("VID", erode(sil, 1), near_dilate=2)
    rows = [(i / FPS, i in ids, det_v.analyze(a)) for i, a in enumerate(arrs)]
    pos = [r for _, s, r in rows if s]
    neg = [r for _, s, r in rows if not s]
    ok = [r for r in pos if r.valid]
    bad = [r for r in neg if r.valid]
    print(f"\n[视频 1080p 尺度] 正样本 {len(ok)}/{len(pos)} = {len(ok) / len(pos) * 100:.1f}%   "
          f"负样本误报 {len(bad)}/{len(neg)} = {len(bad) / len(neg) * 100:.1f}%")
    fr = np.array([r.fraction for r in ok])
    if len(fr):
        print(f"    百分比 min={fr.min() * 100:.0f}%  中位={np.median(fr) * 100:.0f}%  "
              f"max={fr.max() * 100:.0f}%")
    if bad:
        print(f"    误报样本 IoU: {[round(r.shape_iou, 2) for r in bad[:10]]}")

    print(f"\n[用户屏幕尺度] B 区域样张（仅取 {bwin.shape[0]}x{bwin.shape[1]} 的切片）:")
    det_b = make("B", bwin)
    n_ok = n_all = 0
    skipped = []
    for fp in sorted(glob.glob(os.path.join(SHOTS, "*_1x.png"))):
        a = np.array(Image.open(fp).convert("RGB"))
        if a.shape[:2] != bwin.shape:      # 别的区域抓的样张，与本检测器窗口不符
            skipped.append(f"{os.path.basename(fp)} {a.shape[1]}x{a.shape[0]}")
            continue
        n_all += 1
        r = det_b.analyze(a)
        n_ok += r.valid
        print(f"    {os.path.basename(fp)}: {'✓' if r.valid else '×'} {r.fraction * 100:5.1f}% "
              f"{r.state:5s} fill={r.fill:4d}/{r.area} IoU={r.shape_iou:.2f} {r.reason}")
    print(f"    → {n_ok}/{n_all} 识别成功"
          + (f"（跳过 {len(skipped)} 张非本区域: {', '.join(skipped)}）" if skipped else ""))

    print("\n[用户屏幕尺度] A 区域（步枪剪影，应全部拒绝）:")
    det_a = make("A", None)
    n_rej = n_a = 0
    for fp in sorted(glob.glob(os.path.join(SHOTS, "*_full.png"))):
        im = np.array(Image.open(fp).convert("RGB"))
        l, t, r_, b_ = BOX_A
        r = det_a.analyze(im[t + PAD:b_ - PAD, l + PAD:r_ - PAD])
        n_a += 1
        n_rej += not r.valid
        print(f"    {os.path.basename(fp)}: {'✓ 误报!' if r.valid else '× 正确拒绝'} "
              f"IoU={r.shape_iou:.2f} {r.reason}")
    print(f"    → 正确拒绝 {n_rej}/{n_a}")


if __name__ == "__main__":
    main()
