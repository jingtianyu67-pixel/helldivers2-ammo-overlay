# -*- coding: utf-8 -*-
"""生成「倾斜长方体弹匣」的归一化形状模板（32x32，尺度无关）。

素材：
  - 视频 123 帧 slab 正样本
  - 用户屏幕 4 张白色样张
产出 assets/shape_template.npy + 校验报告
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
SHOTS = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
NORM = 32


def extract_mask(rgb, bg_margin=18, chroma_min=18, r_min=70, lum_floor=55, edge=5):
    a = rgb.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)
    side = np.concatenate([lum[:, :edge], lum[:, -edge:]], axis=1)
    bg = np.median(side, axis=1)
    white = (lum > bg[:, None] + bg_margin) & (lum > lum_floor) & (chroma < 25)
    red = (chroma > chroma_min) & (r > r_min)
    return white | red


def fill_holes(m):
    from collections import deque
    h, w = m.shape
    bg = np.zeros((h, w), bool)
    dq = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not m[y, x] and not bg[y, x]:
                bg[y, x] = True
                dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not m[y, x] and not bg[y, x]:
                bg[y, x] = True
                dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not m[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True
                dq.append((ny, nx))
    return ~bg


def cleanup(m, it=1):
    def sh(a, dy, dx):
        o = np.zeros_like(a)
        h, w = a.shape
        ys0, ys1 = max(0, dy), min(h, h + dy)
        xs0, xs1 = max(0, dx), min(w, w + dx)
        o[ys0:ys1, xs0:xs1] = a[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        return o
    out = m.copy()
    for _ in range(it):
        # 闭运算：先膨胀后腐蚀，补断线
        d = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                d |= sh(out, dy, dx)
        e = d.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                e &= sh(d, dy, dx)
        out = e
    return out


def largest_cc(m):
    from collections import deque
    h, w = m.shape
    seen = np.zeros((h, w), bool)
    best = []
    for y0 in range(h):
        for x0 in range(w):
            if not m[y0, x0] or seen[y0, x0]:
                continue
            dq, pts = deque([(y0, x0)]), []
            seen[y0, x0] = True
            while dq:
                y, x = dq.popleft()
                pts.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            dq.append((ny, nx))
            if len(pts) > len(best):
                best = pts
    out = np.zeros((h, w), bool)
    for y, x in best:
        out[y, x] = True
    return out


def norm_shape(m, n=NORM):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return np.zeros((n, n), bool)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = m[y0:y1, x0:x1]
    im = Image.fromarray((c * 255).astype(np.uint8))
    sc = n / max(c.shape)
    nw = max(1, int(round(c.shape[1] * sc)))
    nh = max(1, int(round(c.shape[0] * sc)))
    arr = np.array(im.resize((nw, nh), Image.BILINEAR)) > 127
    out = np.zeros((n, n), bool)
    oy, ox = (n - nh) // 2, (n - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = arr
    return out


def iou(a, b):
    i = (a & b).sum()
    u = (a | b).sum()
    return float(i / u) if u else 0.0


def silhouette_of(rgb):
    m = extract_mask(rgb)
    if m.sum() < 15:
        return None
    m = largest_cc(cleanup(m, 1))
    if m.sum() < 15:
        return None
    return fill_holes(m)


def main():
    allf = sorted(glob.glob(os.path.join(SEQ, "*.png")))
    ids = [int(x) for x in
           open(os.path.join(OUT, "slab_frames.txt"), encoding="utf-8").read().strip().split(",")]

    vid_norms = []
    for i in ids:
        a = np.array(Image.open(allf[i]).convert("RGB"))
        s = silhouette_of(a)
        if s is None or s.sum() < 30:
            continue
        vid_norms.append(norm_shape(s))
    print(f"视频有效正样本 {len(vid_norms)}/{len(ids)}")

    user_norms = []
    for fn in ["0918_213525_1x.png", "0918_213530_1x.png", "0918_213535_1x.png",
               "0918_213544_1x.png"]:
        fp = os.path.join(SHOTS, fn)
        if not os.path.exists(fp):
            continue
        a = np.array(Image.open(fp).convert("RGB"))
        s = silhouette_of(a)
        if s is None:
            print(f"  {fn}: 剪影提取失败")
            continue
        user_norms.append(norm_shape(s))
        print(f"  {fn}: 剪影 {s.sum()} px → 归一化 {norm_shape(s).sum()} px")

    # 中位形状（多数投票）
    T_vid = np.array(vid_norms).mean(axis=0) >= 0.5
    print(f"\n视频中位模板面积 {T_vid.sum()}/{NORM * NORM}")
    if user_norms:
        T_usr = np.array(user_norms).mean(axis=0) >= 0.5
        print(f"用户中位模板面积 {T_usr.sum()}  视频↔用户 模板 IoU={iou(T_vid, T_usr):.3f}")
        T = (np.array(vid_norms + user_norms).mean(axis=0) >= 0.5)
    else:
        T = T_vid
    print(f"合并模板面积 {T.sum()}")

    # 正样本自检
    pos = np.array([iou(x, T) for x in vid_norms])
    print(f"\n视频正样本对合并模板 IoU: min={pos.min():.3f} p5={np.percentile(pos, 5):.3f} "
          f"中位={np.median(pos):.3f}")
    if user_norms:
        up = np.array([iou(x, T) for x in user_norms])
        print(f"用户正样本 IoU: {[f'{v:.2f}' for v in up]}")

    # 负样本
    rng = np.random.default_rng(1)
    neg_ids = [i for i in range(len(allf)) if i not in set(ids)]
    neg_ids = rng.choice(neg_ids, size=min(250, len(neg_ids)), replace=False)
    negs = []
    for i in sorted(neg_ids):
        a = np.array(Image.open(allf[i]).convert("RGB"))
        s = silhouette_of(a)
        if s is None or s.sum() < 30:
            continue
        negs.append(norm_shape(s))
    nv = np.array([iou(x, T) for x in negs])
    print(f"\n负样本 n={len(nv)} IoU: 中位={np.median(nv):.3f} p90={np.percentile(nv, 90):.3f} "
          f"p99={np.percentile(nv, 99):.3f} max={nv.max():.3f}")

    print("\n阈值扫描（正通过率 / 负误报率）:")
    for th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        print(f"  >={th:.2f}: 正 {(pos >= th).mean() * 100:5.1f}%   负 {(nv >= th).mean() * 100:5.1f}%")

    np.save(os.path.join(AST, "shape_template.npy"), T)
    # 可视化
    canvas = Image.new("RGB", (NORM * 10 * 3 + 20, NORM * 10), (18, 18, 20))
    for k, M in enumerate([T_vid, T, (np.array(vid_norms).mean(axis=0) * 255).astype(np.uint8)
                           if False else T]):
        im = Image.fromarray((M * 255).astype(np.uint8)).resize((NORM * 10, NORM * 10), Image.NEAREST)
        canvas.paste(im.convert("RGB"), (k * (NORM * 10 + 10), 0))
    canvas.save(os.path.join(OUT, "shape_template.png"))
    print(f"\n模板 -> assets/shape_template.npy, _out/shape_template.png")


if __name__ == "__main__":
    main()
