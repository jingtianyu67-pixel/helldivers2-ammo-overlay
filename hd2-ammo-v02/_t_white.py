"""验证：① 现场读数只数白像素；② 低弹时 overlay 是否整条转红。"""
import sys
sys.path.insert(0, '.')
import mss
from capture import set_dpi_aware, load_config, grab_rgb
from ammo_detect import build_detector, AmmoReading

set_dpi_aware()
cfg = load_config()
det = build_detector(cfg)
ctx = mss.mss()

print("=== 现场连续帧（只数白像素）===")
for i in range(6):
    cache = {}
    r = det.analyze(lambda b: cache.setdefault(tuple(b), grab_rgb(ctx, b)))
    print(f"  第{i+1}帧 {r.fraction*100:5.1f}% state={r.state:6s} "
          f"region={r.region} hint={r.hint} fill={r.fill}/{r.area} 红像素={r.red_px}")

print("=== overlay 低弹配色 ===")
from overlay import AmmoOverlay, screen_size
ov = AmmoOverlay(cfg, screen_size())
called = []
ov.show = lambda f, red=False: called.append((round(f, 3), red))
for frac in (1.0, 0.62, 0.16, 0.14, 0.0):
    ov.update(AmmoReading(True, frac, 100, 200, "empty" if frac == 0 else "white"))
    print(f"  余量 {frac*100:5.1f}% -> show(fraction={called[-1][0]:.3f}, red={called[-1][1]})")
print(f"  low_threshold={ov.low_threshold} low_full_bar={ov.low_full_bar}")
ov.close()
