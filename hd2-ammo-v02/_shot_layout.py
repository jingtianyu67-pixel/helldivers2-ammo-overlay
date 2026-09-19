# -*- coding: utf-8 -*-
"""有无背包两种布局的判据 + 检测框对照图。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ammo_detect as ad

cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
C = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images")
font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 20)
head = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 24)
tiny = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 17)

CASES = [
    ("有背包", C / "clipboard-2026-09-18T17-04-06-946Z-02f9b08a.jpg", (0, 200, 90)),
    ("无背包", C / "clipboard-2026-09-18T17-04-06-944Z-a0b0e66b.jpg", (0, 190, 255)),
]
L, T, R, B = 190, 1225, 520, 1310          # HUD 展示窗（2560 坐标）
Z = 3

rows = []
for tag, p, col in CASES:
    img = np.array(Image.open(p).convert("RGB").resize((2560, 1440), Image.LANCZOS))
    det = ad.build_detector(cfg, Path("."))
    grab = lambda box: img[box[1]:box[3], box[0]:box[2]].copy()
    for _ in range(4):
        r = det.analyze(grab)
    pl, pt, pr, pb = det.layout.rect
    l, t, rr, bb = det.specs[0].box
    dx = det.shift[0]
    used = (l + dx, t, rr + dx, bb)

    big = Image.fromarray(img[T:B, L:R].copy()).resize(
        ((R - L) * Z, (B - T) * Z), Image.NEAREST)
    d = ImageDraw.Draw(big)
    d.rectangle([(pl - L) * Z, (pt - T) * Z, (pr - L) * Z, (pb - T) * Z],
                outline=(255, 60, 255), width=3)
    d.rectangle([(used[0] - L) * Z, (used[1] - T) * Z,
                 (used[2] - L) * Z, (used[3] - T) * Z], outline=col, width=3)
    d.text(((used[0] - L) * Z + 4, (used[1] - T) * Z - 26), "检测框",
           font=tiny, fill=col)
    d.text(((pl - L) * Z - 40, (pt - T) * Z - 26), "判据", font=tiny,
           fill=(255, 60, 255))

    stripe = img[pt:pb, pl - 4:pr + 4]
    sj = 6
    s_img = Image.fromarray(stripe.copy()).resize(
        (stripe.shape[1] * sj, stripe.shape[0] * sj), Image.NEAREST)
    rows.append(dict(tag=tag, big=big, stripe=s_img, r=r, det=det, used=used, col=col))

W = max(rows[0]["big"].width, 1) + 40 + rows[0]["stripe"].width + 24
ROWH = max(rows[0]["big"].height, rows[0]["stripe"].height) + 96
out = Image.new("RGB", (W + 20, ROWH * 2 + 66), (26, 26, 30))
dr = ImageDraw.Draw(out)
dr.text((12, 10), "HUD 布局自适应：判据矩形里那条白色竖线决定用哪套坐标", font=head,
        fill=(240, 240, 240))
dr.text((12, 40), f"判据 = 矩形 {cfg['layout']['probe_rect']} 内最长竖直连续亮段 "
                  f">= {cfg['layout']['min_run']}px（亮度 > {cfg['layout']['lum']}）；"
                  f"无竖线时检测框左移 {cfg['layout']['shift'][0]}px",
        font=tiny, fill=(168, 168, 176))

for i, row in enumerate(rows):
    y = 66 + i * ROWH
    r = row["r"]
    dr.text((12, y), f"{row['tag']} —— 竖线 {row['det'].last_run}px → "
                     f"{row['det'].mode}　使用区域 {row['used'][0]}~{row['used'][2]}　"
                     f"读数 {'未检测到' if not r.valid else f'{r.fraction * 100:.0f}%'}",
            font=font, fill=row["col"])
    out.paste(row["big"], (12, y + 30))
    x = 12 + row["big"].width + 30
    out.paste(row["stripe"], (x, y + 30))
    dr.text((x, y + 30 + row["stripe"].height + 6), "判据矩形 6x 放大",
            font=tiny, fill=(160, 160, 168))

out.save("_out/背包布局判据.png")
print("出图 _out/背包布局判据.png", out.size)
for row in rows:
    r = row["r"]
    print(f"  {row['tag']}: 竖线={row['det'].last_run}px 布局={row['det'].mode} "
          f"区域={row['used']} 读数={'未检测到' if not r.valid else f'{r.fraction * 100:.0f}%'}")
