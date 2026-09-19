# -*- coding: utf-8 -*-
"""在整屏样张上标出两组候选坐标，裁出左下角放大比对。"""
from __future__ import annotations

import glob
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
OUT = os.path.join(HERE, "_out")

REGIONS = {
    "A": (330, 1250, 390, 1300),   # 第一组（优先）
    "B": (230, 1245, 270, 1300),   # 第二组（备选）
}


def main():
    files = sorted(glob.glob(os.path.join(SHOTS, "*_full.png")))
    print("整屏样张:", [os.path.basename(f) for f in files])
    im = Image.open(files[-1]).convert("RGB")
    print("尺寸:", im.size)

    d = ImageDraw.Draw(im)
    colors = {"A": (255, 60, 60), "B": (60, 255, 60)}
    for k, (x0, y0, x1, y1) in REGIONS.items():
        d.rectangle([x0, y0, x1, y1], outline=colors[k], width=2)
        d.text((x0, y0 - 14), k, fill=colors[k])

    # 裁左下角
    cx0, cy0, cx1, cy1 = 150, 1180, 520, 1360
    crop = im.crop((cx0, cy0, cx1, cy1))
    SC = 4
    crop = crop.resize((crop.width * SC, crop.height * SC), Image.LANCZOS)
    cd = ImageDraw.Draw(crop)
    for k, (x0, y0, x1, y1) in REGIONS.items():
        cd.rectangle([(x0 - cx0) * SC, (y0 - cy0) * SC, (x1 - cx0) * SC, (y1 - cy0) * SC],
                     outline=colors[k], width=2)
        cd.text(((x0 - cx0) * SC + 3, (y0 - cy0) * SC + 3), k, fill=colors[k])
    crop.save(os.path.join(OUT, "regions_cmp.png"))
    print("-> _out/regions_cmp.png", crop.size)

    # 各自单独裁一份大图
    for k, (x0, y0, x1, y1) in REGIONS.items():
        pad = 25
        sub = im.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad))
        SC2 = 10
        sub = sub.resize((sub.width * SC2, sub.height * SC2), Image.LANCZOS)
        sd = ImageDraw.Draw(sub)
        sd.rectangle([pad * SC2, pad * SC2, (pad + x1 - x0) * SC2, (pad + y1 - y0) * SC2],
                     outline=colors[k], width=2)
        sd.text((4, 4), f"region {k} {(x0, y0, x1, y1)}", fill=colors[k])
        sub.save(os.path.join(OUT, f"region_{k}.png"))
        print(f"-> _out/region_{k}.png", sub.size)


if __name__ == "__main__":
    main()
