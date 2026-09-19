# -*- coding: utf-8 -*-
"""红色通道验证：真实素材误报率 + 合成低弹红帧命中率。"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ammo_detect as ad  # noqa: E402

CLIP = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images")
cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
D = cfg["detect"]
TMPL = np.load(ROOT / "assets/shape_template.npy")
BOX_A = tuple(cfg["regions"][0]["region"])
BOX_B = tuple(cfg["regions"][1]["region"])


def mk(name="T"):
    return ad.RegionDetector(
        name, None, TMPL, bg_margin=D["bg_margin"], lum_floor=D["lum_floor"],
        edge=D["edge"], shape_min_iou=D["shape_min_iou"],
        min_mask_px=D["min_mask_px"], near_dilate=D["near_dilate"],
        snap_full=D["snap_full"], max_mask_ratio=D["max_mask_ratio"],
        fill_inset=D["fill_inset"], empty_frac=D["empty_frac"],
        red_chroma=D["red_chroma"], red_min=D["red_min"],
        red_min_px=D["red_min_px"], red_body_frac=D["red_body_frac"],
        red_shape_min_iou=D["red_shape_min_iou"])


def crop(im, box, base=2560):
    k = im.width / base
    return np.array(im.crop(tuple(int(round(v * k)) for v in box)))


FILES = [
    ("prod_screen", ROOT / "_out/prod_screen.png"),
    ("iso_base", ROOT / "_out/iso_base.png"),
    ("demo_screen(火山红)", ROOT / "_out/demo_screen.png"),
    ("orig_full(带红框)", ROOT / "_out/orig_full.png"),
    ("ref_w23:37", CLIP / "clipboard-2026-09-18T15-37-25-493Z-295e67be.jpg"),
    ("ref_r23:37", CLIP / "clipboard-2026-09-18T15-37-25-495Z-2dab9498.jpg"),
    ("shot23:23", CLIP / "clipboard-2026-09-18T15-23-39-864Z-bffed756.jpg"),
    ("shot22:29", CLIP / "clipboard-2026-09-18T14-29-14-820Z-3e8079e0.jpg"),
]

print("=" * 74)
print("A. 真实素材误报检查（期望：全部不出现 low）")
print("=" * 74)
bad = 0
for tag, p in FILES:
    if not p.exists():
        continue
    im = Image.open(p).convert("RGB")
    for bname, box in (("A", BOX_A), ("B", BOX_B)):
        sub = crop(im, box)
        r = mk(bname).analyze(sub)
        flag = ""
        if r.state == "low":
            flag = "   <<< 误判为低弹红！"
            bad += 1
        print(f"  {tag:20s} {bname} valid={str(r.valid):5s} state={r.state:6s} "
              f"{r.fraction * 100:5.1f}%  白={r.white_px:4d} 红={r.red_px:4d} "
              f"IoU={r.shape_iou:.2f} {r.reason}{flag}")
print(f"  → 误报 {bad} 例")

print()
print("=" * 74)
print("B. 合成低弹红帧（拿真实满弹帧把弹匣体染红，模拟游戏告警配色）")
print("=" * 74)
src = Image.open(ROOT / "_out/prod_screen.png").convert("RGB")
for bname, box in (("A", BOX_A), ("B", BOX_B)):
    sub = crop(src, box)
    det = mk(bname)
    white = det.extract(sub)
    hot = ad.largest_cc(white, det.min_mask_px)
    hull = ad.convex_hull_mask(hot)
    body = ad.erode(hull, det.fill_inset)
    for mode, label in (("fill", "外框白+填充红"), ("all", "整个计量器转红")):
        v = sub.copy()
        if mode == "fill":
            v[body] = (218, 46, 42)          # 只有填充区染红
        else:
            v[hull] = (218, 46, 42)          # 整块（含外框）染红
        r = mk(bname).analyze(v)
        ok = "命中" if r.state == "low" else "未命中 <<<"
        print(f"  区域{bname} {label:16s} -> valid={str(r.valid):5s} state={r.state:6s} "
              f"{r.fraction * 100:5.1f}%  红={r.red_px:4d} IoU={r.shape_iou:.2f} "
              f"hint={r.hint} {ok}")
        if mode == "all":
            Image.fromarray(v).resize((v.shape[1] * 8, v.shape[0] * 8),
                                      Image.NEAREST).save(
                ROOT / f"_out/red_synth_{bname}.png")
