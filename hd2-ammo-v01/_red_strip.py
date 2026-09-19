# -*- coding: utf-8 -*-
"""把各张实机截图的武器 HUD 区裁出来并排，看红框是不是条件性出现。"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
CLIP = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images")

# (标签, 路径, 标定分辨率宽度)  区域 = (230,1230,470,1320) @2560 便于包含红框
CASES = [
    ("prod_screen", ROOT / "_out/prod_screen.png", 2560),
    ("iso_base", ROOT / "_out/iso_base.png", 2560),
    ("demo_screen", ROOT / "_out/demo_screen.png", 2560),
    ("orig_full", ROOT / "_out/orig_full.png", 2560),
    ("shot22:29", CLIP / "clipboard-2026-09-18T14-29-14-820Z-3e8079e0.jpg", 2560),
    ("shot23:23", CLIP / "clipboard-2026-09-18T15-23-39-864Z-bffed756.jpg", 2560),
    ("ref_w23:37", CLIP / "clipboard-2026-09-18T15-37-25-493Z-295e67be.jpg", 2560),
    ("ref_r23:37", CLIP / "clipboard-2026-09-18T15-37-25-495Z-2dab9498.jpg", 2560),
    ("shot21:18", CLIP / "clipboard-2026-09-18T13-18-21-264Z-ac430eed.jpg", 2560),
]

BOX = (230, 1225, 470, 1320)     # 含弹匣图标 + 红框 + 右侧数字
Z = 4
tiles = []
for tag, p, base in CASES:
    if not p.exists():
        print("缺", tag)
        continue
    im = Image.open(p).convert("RGB")
    k = im.width / base
    box = tuple(int(round(v * k)) for v in BOX)
    c = im.crop(box)
    c = c.resize((c.width * Z, c.height * Z), Image.NEAREST)
    d = ImageDraw.Draw(c)
    d.text((6, 4), f"{tag}  k={k:.3f}", fill=(0, 255, 0))
    tiles.append(c)
    print(f"{tag:12s} {im.size} k={k:.3f} crop={box}")

W = max(t.width for t in tiles)
H = sum(t.height + 8 for t in tiles)
out = Image.new("RGB", (W, H), (20, 20, 24))
y = 0
for t in tiles:
    out.paste(t, (0, y))
    y += t.height + 8
out.save(ROOT / "_out/red_hud_strip.png")
print("->", out.size)
