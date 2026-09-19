"""同一个现场帧，扫分子/分母口径，并画出被计入的像素。"""
import sys
sys.path.insert(0, '.')
import numpy as np, mss
from PIL import Image
from capture import set_dpi_aware, load_config, resolve_regions, grab_rgb
from ammo_detect import (build_detector, largest_cc, convex_hull_mask, erode,
                         dilate, opening)

set_dpi_aware()
cfg = load_config()
d = cfg["detect"]
det = build_detector(cfg)
rd = det.specs[0].detector
box = dict(resolve_regions(cfg, verbose=False))["A"]
rgb = grab_rgb(mss.mss(), box)
white, red = rd.extract(rgb)

wcc = largest_cc(white, d.get("min_mask_px", 30))
rcc = largest_cc(red, d.get("min_mask_px", 30))
hot = wcc if int(wcc.sum()) >= d.get("min_mask_px", 30) else rcc
hull = convex_hull_mask(hot)
print(f"白CC={int(wcc.sum())} 红CC={int(rcc.sum())} 用={('白' if hot is wcc else '红')} "
      f"hull={int(hull.sum())}")

print("--- 分子口径 / 分母内缩 ---")
for k in (1, 2, 3):
    inner = erode(hull, k)
    a = int(inner.sum())
    core = opening(white, k)
    for label, num in (("core∩inner", int((core & inner).sum())),
                       ("core∩hull", int((core & hull).sum())),
                       ("white∩inner", int((white & inner).sum()))):
        print(f"  open={k} inset={k} {label:12s} {num:4d}/{a:4d} = {num/a:.3f}")

# 画出来：绿=计入的分子(core∩hull, open=2)，紫=白色外框线(core 去掉的部分)
core = opening(white, 2)
ov = np.array(Image.fromarray(rgb)).copy()
ov[hull & ~core] = [90, 90, 90]
ov[core & hull] = [0, 255, 0]
ov[white & ~core] = [255, 0, 255]
im = Image.fromarray(ov)
im.resize((im.width * 8, im.height * 8), Image.NEAREST).save("_out/chk_A_mark.png")
print("绿=计入分子, 紫=被开运算抹掉的白色外框线, 已存 _out/chk_A_mark.png")
