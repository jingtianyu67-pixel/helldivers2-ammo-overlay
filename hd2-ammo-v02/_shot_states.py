# -*- coding: utf-8 -*-
"""把「低弹整条泛红 / 空弹无实心条」这条规则画成对照图。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from overlay import ArcGeometry

BASE = Path(__file__).resolve().parent
cfg = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
O = cfg["overlay"]
font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 20)
head = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 24)

# 复刻 AmmoOverlay.show_value 的判定
def decide(f):
    if f <= 0.0:
        return 0.0, False
    red = f < float(O["low_threshold"])
    return (1.0 if (red and O["low_full_bar"]) else f), red

STATES = [(1.00, "满弹"), (0.50, "半弹"), (0.22, "刚好在阈值上"),
          (0.15, "低弹（阈值以下）"), (0.05, "只剩一点点"), (0.00, "空弹匣")]

geo = ArcGeometry((2560, 1440), O)
tiles = []
for f, label in STATES:
    disp, red = decide(f)
    img = geo.render(disp, O["low_rgb"] if red else O["fill_rgb"],
                     O["track_rgb"], O["track_alpha"], O["alpha"])
    bg = Image.new("RGB", img.size, (46, 46, 52))
    bg.paste(img, (0, 0), img)
    top, bar = 32, 62
    out = Image.new("RGB", (img.width, img.height + top + bar), (26, 26, 30))
    out.paste(bg, (0, top))
    d = ImageDraw.Draw(out)
    d.text((10, 5), f"余量 {f * 100:.0f}%  {label}", font=head, fill=(235, 235, 235))
    d.text((10, img.height + top + 4),
           f"显示：{disp * 100:.0f}%  颜色：{'红' if red else '白'}"
           f"{'   ← 整条泛红' if red and O['low_full_bar'] else ''}"
           f"{'   ← 不画实心条' if f <= 0 else ''}",
           font=font, fill=(255, 120, 120) if red else (140, 230, 160))
    tiles.append(out)

W = sum(t.width for t in tiles) + 10 * (len(tiles) - 1)
H = max(t.height for t in tiles)
canvas = Image.new("RGB", (W, H), (26, 26, 30))
x = 0
for t in tiles:
    canvas.paste(t, (x, 0))
    x += t.width + 10
p = BASE / "_out" / "低弹与空弹显示规则.png"
canvas.save(p)
print("saved", p, canvas.size)
