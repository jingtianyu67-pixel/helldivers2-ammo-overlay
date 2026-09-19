# -*- coding: utf-8 -*-
"""红像素普查：真实 HUD 里红色出现在哪、弹匣区域内有多少。

只做统计，不改任何代码。
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CLIP = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images")

CASES = [
    ("prod_screen", ROOT / "_out/prod_screen.png", 2560),
    ("demo_screen", ROOT / "_out/demo_screen.png", 2560),
    ("iso_base", ROOT / "_out/iso_base.png", 2560),
    ("orig_full", ROOT / "_out/orig_full.png", 2560),
    ("hud_low", ROOT / "_out/hud_low.png", None),
    ("ref_w(23:37)", CLIP / "clipboard-2026-09-18T15-37-25-493Z-295e67be.jpg", 1920),
    ("ref_r(23:37)", CLIP / "clipboard-2026-09-18T15-37-25-495Z-2dab9498.jpg", 1920),
    ("shot(23:23)", CLIP / "clipboard-2026-09-18T15-23-39-864Z-bffed756.jpg", None),
    ("shot(22:29)", CLIP / "clipboard-2026-09-18T14-29-14-820Z-3e8079e0.jpg", None),
]

cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
REG = [(it["name"], tuple(it["region"])) for it in cfg["regions"]]


def red_mask(a: np.ndarray, chroma=40, rmin=110) -> np.ndarray:
    a = a.astype(np.int16)
    return ((a[..., 0] - np.maximum(a[..., 1], a[..., 2])) >= chroma) & (a[..., 0] >= rmin)


def clusters(m: np.ndarray, min_px=6):
    """红像素簇（8连通，迭代式，区域小所以够快）。"""
    from collections import deque
    h, w = m.shape
    seen = np.zeros((h, w), bool)
    out = []
    for y0 in range(h):
        for x0 in range(w):
            if not m[y0, x0] or seen[y0, x0]:
                continue
            dq = deque([(y0, x0)])
            seen[y0, x0] = True
            pts = []
            while dq:
                y, x = dq.popleft()
                pts.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            dq.append((ny, nx))
            if len(pts) >= min_px:
                ys = [p[0] for p in pts]
                xs = [p[1] for p in pts]
                out.append((len(pts), min(xs), min(ys), max(xs), max(ys)))
    out.sort(reverse=True)
    return out


for tag, p, base_w in CASES:
    if not p.exists():
        print(f"{tag:14s} 缺文件")
        continue
    im = Image.open(p).convert("RGB")
    a = np.array(im)
    k = im.width / base_w if base_w else 1.0
    print(f"=== {tag}  {im.width}x{im.height}  k={k:.3f}")
    for name, (l, t, r, b) in REG:
        if base_w and abs(k - 1.0) > 1e-6:
            box = (int(round(l * k)), int(round(t * k)), int(round(r * k)), int(round(b * k)))
        else:
            box = (l, t, r, b)
        if box[2] > im.width or box[3] > im.height:
            print(f"   {name}: 区域越界 {box}")
            continue
        sub = a[box[1]:box[3], box[0]:box[2]]
        rm = red_mask(sub)
        cl = clusters(rm)
        print(f"   {name} {box} 红像素={int(rm.sum())} 簇={len(cl)} "
              f"{('最大簇 n=%d bbox=%s' % (cl[0][0], cl[0][1:])) if cl else ''}")
        # 整帧 HUD 下三分之一有多少红
    hud = red_mask(a[int(a.shape[0] * 0.68):])
    hcl = clusters(hud, 12)
    print(f"   全帧下部红像素={int(hud.sum())} 簇(>=12px)={len(hcl)} "
          f"前3={[(c[0], c[1], c[2] + int(a.shape[0] * 0.68)) for c in hcl[:3]]}")
