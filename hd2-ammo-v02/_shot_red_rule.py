# -*- coding: utf-8 -*-
"""红色弹药检测规则的对照图：识别端判定 + 显示端弧线，走同一份 decide_show。"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ammo_detect as ad                      # noqa: E402
from overlay import ArcGeometry, decide_show  # noqa: E402

cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
D, O = cfg["detect"], cfg["overlay"]
BOX_A = tuple(cfg["regions"][0]["region"])
font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 19)
head = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 23)
tiny = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 16)

src = Image.open(ROOT / "_out/prod_screen.png").convert("RGB")
k = src.width / 2560.0
sub = np.array(src.crop(tuple(int(round(v * k)) for v in BOX_A)))

det = ad.RegionDetector(
    "A", None, np.load(ROOT / "assets/shape_template.npy"),
    bg_margin=D["bg_margin"], lum_floor=D["lum_floor"], edge=D["edge"],
    shape_min_iou=D["shape_min_iou"], min_mask_px=D["min_mask_px"],
    near_dilate=D["near_dilate"], snap_full=D["snap_full"],
    max_mask_ratio=D["max_mask_ratio"], fill_inset=D["fill_inset"],
    empty_frac=D["empty_frac"], red_chroma=D["red_chroma"], red_min=D["red_min"],
    red_min_px=D["red_min_px"], red_body_frac=D["red_body_frac"],
    red_shape_min_iou=D["red_shape_min_iou"])

white = det.extract(sub)
body = ad.erode(ad.convex_hull_mask(ad.largest_cc(white, det.min_mask_px)),
                det.fill_inset)
h = sub.shape[0]
ys = np.where(body.any(axis=1))[0]
bg_col = np.median(sub[body & ~white], axis=0).astype(np.uint8) \
    if (body & ~white).any() else np.array([40, 40, 44], np.uint8)

# ---- 四种状态：真实满弹帧 + 按需改造成低弹 / 红弹 / 空弹 ---------------------- #
frames = [("满弹（白）", sub.copy())]

v = sub.copy()
cut = ys.min() + int(0.85 * (ys.max() - ys.min()))
v[body & white & (np.arange(h)[:, None] < cut)] = bg_col
frames.append(("低弹（白，按占比触红）", v))

v = sub.copy()
v[body & white] = (218, 46, 42)
frames.append(("红色弹药（检测驱动）", v))

v = sub.copy()
v[body & white] = bg_col
frames.append(("空弹匣（不画条）", v))

geo = ArcGeometry((2560, 1440), O)
tiles = []
for label, px in frames:
    r = det.analyze(px)
    disp, red = decide_show(r.fraction, r.fill, r.state == "low", O)

    arc = geo.render(disp, O["low_rgb"] if red else O["fill_rgb"],
                     O["track_rgb"], O["track_alpha"], O["alpha"])
    arc_bg = Image.new("RGB", arc.size, (46, 46, 52))
    arc_bg.paste(arc, (0, 0), arc)

    Z = 5
    vis = px.astype(np.float32).copy()
    wm = det.extract(px)
    rm = det.extract_red(px)
    vis[wm] = vis[wm] * 0.55 + np.array([40, 220, 90]) * 0.45
    vis[rm] = vis[rm] * 0.45 + np.array([255, 40, 220]) * 0.55
    mag = Image.fromarray(vis.astype(np.uint8)).resize(
        (px.shape[1] * Z, px.shape[0] * Z), Image.NEAREST)

    colw = mag.width
    pad, bar = 34, 92
    out = Image.new("RGB", (colw, pad + arc_bg.height + mag.height + bar),
                    (26, 26, 30))
    out.paste(arc_bg, ((colw - arc_bg.width) // 2, pad))
    out.paste(mag, (0, pad + arc_bg.height))
    d = ImageDraw.Draw(out)
    y0 = pad + arc_bg.height
    d.text((8, 6), label, font=head, fill=(240, 240, 240))
    d.text((8, pad + arc_bg.height - 26),
           "弧线预览（准星右侧）", font=tiny, fill=(150, 150, 158))
    d.text((8, y0 + 4), "绿=白像素（进分子）　品红=红像素（不进分子）",
           font=tiny, fill=(170, 170, 175))
    d.text((8, y0 + mag.height + 4),
           f"识别 state={r.state}  白{r.white_px}px  红{r.red_px}px", font=font,
           fill=(250, 130, 130) if r.state == "low" else (150, 225, 165))
    emit = "整条泛红" if red else ("不画实心条" if disp <= 0 else "白条")
    d.text((8, y0 + mag.height + 32),
           f"显示 → {emit}  长度{disp * 100:.0f}%", font=font,
           fill=(255, 140, 140) if red else (200, 200, 205))
    if r.state == "low":
        d.text((8, y0 + mag.height + 60), "← 这一档旧版是整窗隐藏",
               font=tiny, fill=(255, 190, 90))
    tiles.append(out)

W = sum(t.width for t in tiles) + 10 * (len(tiles) - 1)
H = max(t.height for t in tiles)
canvas = Image.new("RGB", (W, H), (26, 26, 30))
x = 0
for t in tiles:
    canvas.paste(t, (x, 0))
    x += t.width + 10
p = ROOT / "_out" / "红色弹药检测规则.png"
canvas.save(p)
print("saved", p, canvas.size)
