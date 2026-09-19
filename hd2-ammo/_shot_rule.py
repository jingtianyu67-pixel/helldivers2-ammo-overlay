# -*- coding: utf-8 -*-
"""出图：新版口径「弹匣体内白像素 / 弹匣体面积」在参考帧与满弹帧上的可视化。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ammo_detect as ad

BASE = Path(__file__).resolve().parent
cfg = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
d = cfg["detect"]
tmpl = np.load(BASE / "assets/shape_template.npy")
CLIP = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images")
FONT = r"C:\Windows\Fonts\msyh.ttc"


def panel(rgb, k, title, box=None):
    det = ad.RegionDetector("v", None, tmpl, bg_margin=d["bg_margin"],
                            lum_floor=d["lum_floor"],
                            edge=max(1, int(round(d["edge"] * k))),
                            min_mask_px=30, fill_inset=d["fill_inset"])
    white = det.extract(rgb)
    hot = ad.largest_cc(white, 30)
    hull = ad.convex_hull_mask(hot)
    body = ad.erode(hull, d["fill_inset"])
    num = int((white & body).sum())
    area = int(body.sum())
    frac = num / max(1, area)
    if frac >= d["snap_full"]:
        frac = 1.0
    elif frac <= d["empty_frac"]:
        frac = 0.0

    vis = rgb.copy()
    vis[body & ~white] = (vis[body & ~white] * 0.45
                          + np.array([0, 70, 190], np.uint8) * 0.55).astype(np.uint8)
    vis[white & body] = np.array([0, 220, 60], np.uint8)      # 计入分子的白像素
    vis[white & ~body] = np.array([150, 150, 150], np.uint8)  # 外框线：不计
    z = 10 if max(rgb.shape[:2]) < 100 else 2
    img = Image.fromarray(vis).resize((vis.shape[1] * z, vis.shape[0] * z), Image.NEAREST)
    f = ImageFont.truetype(FONT, 22)
    head = ImageFont.truetype(FONT, 26)
    bar = 62
    out = Image.new("RGB", (img.width, img.height + bar + 30), (24, 24, 28))
    out.paste(img, (0, bar))
    dr = ImageDraw.Draw(out)
    dr.text((8, 8), title, font=head, fill=(240, 240, 240))
    dr.text((8, bar - 26), "白像素 %d / 弹匣体 %d = %.1f%%   (外框线已排除)"
            % (num, area, 100.0 * num / max(1, area)), font=f, fill=(120, 230, 140))
    dr.text((8, img.height + bar + 4), "读数 %.0f%%" % (frac * 100),
            font=f, fill=(255, 210, 90))
    return out


# 1) 用户参考帧（红线触发点）
im1 = Image.open(CLIP / "clipboard-2026-09-18T15-47-56-207Z-92054776.png").convert("RGB")
p1 = panel(np.array(im1), 1.0, "你的参考帧（红线的触发点）")

# 2) 实机满弹帧 region A
im2 = Image.open(BASE / "_out" / "prod_screen.png").convert("RGB")
l, t, r, b = cfg["regions"][0]["region"]
p2 = panel(np.array(im2.crop((l, t, r, b))), 1.0, "实机满弹（2560x1440，区域 A）")

W = p1.width + p2.width + 24
H = max(p1.height, p2.height)
canvas = Image.new("RGB", (W, H), (24, 24, 28))
canvas.paste(p1, (0, 0))
canvas.paste(p2, (p1.width + 24, 0))
out = BASE / "_out" / "白像素口径.png"
canvas.save(out)
print("saved", out, canvas.size)
