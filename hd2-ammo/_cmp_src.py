# -*- coding: utf-8 -*-
"""比对「全屏样张切片」与「F1 独立截取的区域图」是否一致，并测不同 red_ratio 对红帧的影响。"""
from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image

SHOTS = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
BOX = (230, 1245, 270, 1300)


def main():
    print("=== 1. full 图切片 vs 1x 图 像素比对 ===")
    for ff in sorted(glob.glob(os.path.join(SHOTS, "*_full.png"))):
        ts = os.path.basename(ff)[:-9]
        xf = os.path.join(SHOTS, ts + "_1x.png")
        if not os.path.exists(xf):
            continue
        full = np.array(Image.open(ff).convert("RGB"))
        one = np.array(Image.open(xf).convert("RGB"))
        l, t, r, b = BOX
        sub = full[t:b, l:r]
        same_shape = sub.shape == one.shape
        if same_shape:
            diff = np.abs(sub.astype(int) - one.astype(int))
            print(f"  {ts}: shape 一致 {sub.shape}  最大差={diff.max()} 均值差={diff.mean():.2f}  "
                  f"完全相同={bool((diff == 0).all())}")
            if diff.max() > 0:
                # 找最佳偏移
                best = None
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        y0, x0 = t + dy, l + dx
                        if y0 < 0 or x0 < 0 or y0 + 55 > full.shape[0] or x0 + 40 > full.shape[1]:
                            continue
                        s2 = full[y0:y0 + 55, x0:x0 + 40]
                        d = np.abs(s2.astype(int) - one.astype(int)).mean()
                        if best is None or d < best[0]:
                            best = (d, dy, dx)
                print(f"      最佳对齐偏移 dy={best[1]} dx={best[2]} 均值差={best[0]:.2f}"
                      f"（当前偏移 0,0 差值 {diff.mean():.2f}）")
        else:
            print(f"  {ts}: shape 不同 full切片{sub.shape} vs 1x{one.shape}")

    print("\n=== 2. 红色判据在不同 red_ratio 下的表现（用户屏红帧） ===")
    reds = [(os.path.join(SHOTS, "0918_213537_1x.png"), "极暗红"),
            (os.path.join(SHOTS, "0918_213541_1x.png"), "暗红"),
            (os.path.join(SHOTS, "0918_213535_1x.png"), "白(对照)")]
    for fp, tag in reds:
        a = np.array(Image.open(fp).convert("RGB")).astype(np.int16)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        chroma = r - np.maximum(g, b)
        base = (chroma > 18) & (r > 70)
        print(f"\n  {tag} {os.path.basename(fp)}:")
        print(f"    chroma>18&R>70: {int(base.sum())} px")
        for rr in (None, 1.2, 1.3, 1.4, 1.5, 1.6):
            m = base.copy()
            if rr:
                m &= (r > g * rr) & (r > b * rr)
            print(f"    red_ratio={str(rr):>4}: {int(m.sum()):4d} px", end="")
        print()
        # 红色像素的 r/g 比值分布
        if base.sum():
            ratio = (r[base] / np.maximum(1, g[base])).astype(float)
            print(f"    红色像素 r/g 比值: min={ratio.min():.2f} 中位={np.median(ratio):.2f} "
                  f"p90={np.percentile(ratio, 90):.2f} max={ratio.max():.2f}")


if __name__ == "__main__":
    main()
