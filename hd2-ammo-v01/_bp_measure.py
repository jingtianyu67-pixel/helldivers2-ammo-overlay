# -*- coding: utf-8 -*-
"""有无背包两种 HUD 布局：量竖线判据矩形 + 找弹药图标实际位置。"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

C = Path(r"C:\Users\Administrator\.workbuddy\clipboard-images")
IMGS = {
    "带包": C / "clipboard-2026-09-18T17-04-06-944Z-a0b0e66b.jpg",
    "无包": C / "clipboard-2026-09-18T17-04-06-946Z-02f9b08a.jpg",
}
font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 20)
tiny = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 16)

for tag, p in IMGS.items():
    im = Image.open(p).convert("RGB")
    k = im.width / 2560.0
    print(f"{tag}: {p.name} 尺寸 {im.size}  k={k:.4f}")
    a = np.array(im).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = r - np.maximum(g, b)
    white = (lum > 150) & (chroma < 25)

    # 用户给的竖线判据矩形（2560 坐标）
    l, t, rr, bb = 330, 1250, 337, 1295
    box = (int(l * k), int(t * k), int(rr * k), int(bb * k))
    sub_w = white[box[1]:box[3], box[0]:box[2]]
    print(f"   竖线矩形{box} 宽{box[2]-box[0]}px 高{box[3]-box[1]}px → 白像素 {int(sub_w.sum())}"
          f"  每行白数 max={int(sub_w.sum(1).max()) if sub_w.size else 0}")
    crop = im.crop(box)
    crop.resize((crop.width * 10, crop.height * 10), Image.NEAREST).save(
        f"_out/bp_line_{tag}_10x.png")

    # 左下 HUD 全貌（2560 坐标 190..520 x 1200..1400）
    hb = (int(190 * k), int(1200 * k), int(520 * k), int(1400 * k))
    h = im.crop(hb)
    h = h.resize((h.width * 4, h.height * 4), Image.NEAREST)
    d = ImageDraw.Draw(h)
    # 叠上现有区域 A / B 的框
    for name, (rl, rt, r2, r3), col in [
            ("A", (330, 1250, 390, 1300), (0, 255, 0)),
            ("B", (230, 1245, 270, 1300), (0, 180, 255))]:
        x0 = (rl * k - hb[0]) * 4
        y0 = (rt * k - hb[1]) * 4
        x1 = (r2 * k - hb[0]) * 4
        y1 = (r3 * k - hb[1]) * 4
        d.rectangle([x0, y0, x1, y1], outline=col, width=2)
        d.text((x0, y0 - 22), name, font=font, fill=col)
    # 竖线矩形
    x0 = (l * k - hb[0]) * 4
    y0 = (t * k - hb[1]) * 4
    d.rectangle([x0, y0, (rr * k - hb[0]) * 4, (bb * k - hb[1]) * 4],
                outline=(255, 80, 255), width=2)
    h.save(f"_out/bp_hud_{tag}_4x.png")
    print(f"   出图 _out/bp_hud_{tag}_4x.png {h.size}")
