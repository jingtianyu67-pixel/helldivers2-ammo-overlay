# -*- coding: utf-8 -*-
"""三种终局状态对照：打空（空壳）/ 低弹红 / 满弹。

上半行 = 识别端看到了什么（绿=被计为白的像素，蓝=弹匣体，灰=轮廓线被排除）
下半行 = 覆盖层画成什么（走 decide_show，与运行时同一份判定）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ammo_detect as ad
from overlay import ArcGeometry, decide_show

cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
D, O = cfg["detect"], cfg["overlay"]
det = ad.build_detector(cfg, Path("."))
sp = det.specs[0]
d = sp.detector
W, H = sp.box[2] - sp.box[0], sp.box[3] - sp.box[1]

font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 19)
head = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 23)
tiny = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 16)

# ---- 素材 ---------------------------------------------------------------- #
base = np.array(Image.open("_out/prod_screen.png").convert("RGB")
                .crop((sp.box[0], sp.box[1], sp.box[2], sp.box[3])))
# 空弹：用户给的真实样本，缩放到本区域尺度
U = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images"
         r"\clipboard-2026-09-18T16-28-11-427Z-403561d7.png")
empty_rgb = np.array(Image.open(U).convert("RGB").resize((W, H), Image.LANCZOS))


def synth_low_body(rgb):
    """把弹匣体染成游戏告警红（外框留白）→ 模拟低弹红。"""
    w = d.extract(rgb)
    hull = ad.convex_hull_mask(ad.largest_cc(w, d.min_mask_px))
    body = ad.erode(hull, d.fill_inset)
    out = rgb.copy()
    out[body] = [214, 44, 40]
    return out


cases = [
    ("① 打空（空壳）", empty_rgb, "游戏：填充清空，只剩一圈轮廓"),
    ("② 低弹（红填充）", synth_low_body(base), "游戏：计量器染红"),
    ("③ 满弹（白填充）", base, "游戏：白色实心条"),
]

# ---- 识别端特写 ---------------------------------------------------------- #
Z = 8
tiles = []
for label, rgb, note in cases:
    r = d.analyze(rgb)
    w = d.extract(rgb)
    hot = ad.largest_cc(w, d.min_mask_px)
    hull = ad.convex_hull_mask(hot)
    body = ad.erode(hull, d.fill_inset)
    vis = rgb.astype(np.float32).copy()
    m = w.copy()
    vis[m] = vis[m] * 0.45 + np.array([40, 230, 100]) * 0.55      # 白像素=绿
    vis[body & ~m] = vis[body & ~m] * 0.6 + np.array([0, 90, 255]) * 0.4  # 弹匣体空=蓝
    mag = Image.fromarray(vis.astype(np.uint8)).resize(
        (W * Z, H * Z), Image.NEAREST)
    tiles.append((label, note, r, mag))

CW = tiles[0][3].width
AH = 430
pad, bar = 30, 118


GEO = ArcGeometry((2560, 1440), O)


def arc_panel(fraction, fill, low, mask, title, sub, color):
    """出：真实 layered 渲染的弧线（叠在深色底上）+ 说明。"""
    disp, red = decide_show(fraction, fill, low, O)
    img = GEO.render(disp, O["low_rgb"] if red else O["fill_rgb"], O["track_rgb"],
                     O["track_alpha"], 1.0)
    bg = Image.new("RGB", (mask.width, mask.height + 46), (44, 44, 50))
    bg.paste(img, ((bg.width - img.width) // 2, (mask.height - img.height) // 2), img)
    dd = ImageDraw.Draw(bg)
    emit = ("整条泛红" if red else
            ("不画实心条，只留灰轨" if disp <= 0 else f"白条 {disp * 100:.0f}%"))
    dd.text((10, mask.height + 8), f"覆盖层 → {emit}", font=font, fill=color)
    dd.text((10, mask.height + 32), "弧线预览（准星右侧）", font=tiny, fill=(150, 150, 158))
    return bg


panels = []
for label, note, r, mag in tiles:
    low = (r.state == "low")
    title = f"{label}  识别：{ '空弹匣' if r.state == 'empty' else ('低弹红' if low else f'{r.fraction * 100:.0f}%') }"
    sub = (f"白 {r.white_px}px  红 {r.red_px}px  弹匣体 {r.area}px  IoU {r.shape_iou:.2f}  "
           f"via {r.hint or '-'}")
    col = (255, 150, 150) if low else (150, 225, 165)
    panels.append((title, note, sub, mag, arc_panel(r.fraction, r.fill, low, mag, "", sub, col), col))

TOP, PAD = 60, 26
ROW_H = TOP + PAD + 24 + tiles[0][3].height + 30
TH = ROW_H + panels[0][4].height + 20
out = Image.new("RGB", (CW * 3 + 40, TH), (26, 26, 30))
dr = ImageDraw.Draw(out)
dr.text((10, 8), "打空 / 低弹 / 满弹 —— 上=识别端（绿=计入分子的白像素，蓝=弹匣体），下=覆盖层实际画面",
        font=head, fill=(240, 240, 240))
dr.text((10, 32), "空壳通道：白像素只剩轮廓线（< mask_floor）时改看形状 IoU，判定为空弹匣 → 覆盖层画 0% 空轨，不再整窗隐藏",
        font=tiny, fill=(170, 170, 178))
for i, (t, note, sub, mag, arc, col) in enumerate(panels):
    x = i * (CW + 20)
    dr.text((x + 8, TOP), t, font=head, fill=(240, 240, 240))
    dr.text((x + 8, TOP + 28), note, font=tiny, fill=(165, 165, 172))
    out.paste(mag, (x, TOP + PAD + 24))
    dr.text((x + 8, TOP + PAD + 24 + mag.height + 4), sub, font=tiny, fill=col)
    out.paste(arc, (x, ROW_H))

out.save("_out/空弹显示.png")
print("出图 _out/空弹显示.png", out.size)
for t, note, r, _ in tiles:
    print(f"  {t}: valid={r.valid} state={r.state} frac={r.fraction:.3f} "
          f"white={r.white_px} body={r.area} IoU={r.shape_iou:.2f} hint={r.hint}")
