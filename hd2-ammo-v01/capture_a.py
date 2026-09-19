# -*- coding: utf-8 -*-
"""HD2 A 区域取景工具（F1 热键）。

用途：在游戏里按 F1，把 A 区域的原始像素原样落到桌面，用来给 A 区域标定探针。
样本单独存到 hd2_ammo_shots\\A\\，不与 B 区域混放（两组区域尺寸不同，
混在一个目录里跑检测器会报「区域尺寸与窗口不符」）。

每次按 F1 存 3 张：
    A01_<时间>_1x.png      区域原图，物理像素无损 —— 唯一能拿去做标定的图
    A01_<时间>_grid.png    12x 放大 + 每 5px 坐标网格 —— 用来读探针坐标
    A01_<时间>_full.png    整屏 + 红框标区域 + 绿十字标准星

用法：
    python capture_a.py                     # 常驻，按 F1 截图，输入 q 或 Ctrl+C 退出
    python capture_a.py --once              # 立刻截一次然后退出（自检）
    python capture_a.py --zoom 16           # 改网格图放大倍数
    python capture_a.py --region 330 1250 390 1300 --save-region   # 改 A 区域坐标并写回配置
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

import mss
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from capture import (desktop_dir, grab_rgb, load_config, resolve_regions,
                     save_config, screen_size, set_dpi_aware)
from capture_probe import region_stats

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
user32.RegisterHotKey.restype = wintypes.BOOL
user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                               wintypes.UINT, wintypes.UINT]
user32.GetMessageW.restype = wintypes.BOOL

VK_F1 = 0x70
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
GRID_STEP = 5


def _font(size: int = 13):
    for name in ("consola.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_grid(rgb: np.ndarray, zoom: int, step: int = GRID_STEP) -> Image.Image:
    """放大并叠加坐标网格。网格线只画在这张「给人看」的图上，1x 原图保持干净。"""
    h, w = rgb.shape[:2]
    big = Image.fromarray(rgb).resize((w * zoom, h * zoom), Image.NEAREST)
    d = ImageDraw.Draw(big)
    f = _font(max(11, zoom))
    for x in range(0, w, step):
        X = x * zoom
        d.line([(X, 0), (X, big.height)], fill=(0, 255, 0), width=1)
        d.text((X + 2, 2), str(x), fill=(0, 255, 0), font=f)
    for y in range(0, h, step):
        Y = y * zoom
        d.line([(0, Y), (big.width, Y)], fill=(0, 180, 255), width=1)
        d.text((2, Y + 2), str(y), fill=(0, 180, 255), font=f)
    return big


class ProbeA:
    def __init__(self, region, out_dir: Path, zoom: int):
        self.region = tuple(region)
        self.out_dir = out_dir
        self.zoom = zoom
        self.count = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def shoot(self) -> list[Path]:
        self.count += 1
        tag = f"A{self.count:02d}_{datetime.now().strftime('%H%M%S')}"
        sw, sh = screen_size()
        left, top, right, bottom = self.region
        with mss.mss() as sct:
            rgb = grab_rgb(sct, self.region)
            full = np.array(sct.grab({"left": 0, "top": 0, "width": sw, "height": sh}))
            full = full[:, :, 2::-1]

        saved: list[Path] = []

        p1 = self.out_dir / f"{tag}_1x.png"
        Image.fromarray(rgb).save(p1)
        saved.append(p1)

        pg = self.out_dir / f"{tag}_grid.png"
        draw_grid(rgb, self.zoom).save(pg)
        saved.append(pg)

        pf = self.out_dir / f"{tag}_full.png"
        canvas = Image.fromarray(full).convert("RGB")
        d = ImageDraw.Draw(canvas)
        d.rectangle([left - 1, top - 1, right, bottom], outline=(255, 0, 0), width=2)
        cx, cy = sw // 2, sh // 2
        d.line([cx - 18, cy, cx + 18, cy], fill=(0, 255, 0), width=2)
        d.line([cx, cy - 18, cx, cy + 18], fill=(0, 255, 0), width=2)
        canvas.save(pf)
        saved.append(pf)

        with open(self.out_dir / "shots.log", "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')}\t{tag}\t"
                     f"region={self.region}\tscreen={sw}x{sh}\t{region_stats(rgb)}\n")

        print(f"[{self.count:02d}] {tag}  区域 {self.region}  "
              f"尺寸 {right - left}x{bottom - top}")
        print(f"     {region_stats(rgb)}")
        return saved


def hotkey_loop(probe: ProbeA, stop: threading.Event) -> None:
    tid = ctypes.windll.kernel32.GetCurrentThreadId()
    if not user32.RegisterHotKey(None, 1, MOD_NOREPEAT, VK_F1):
        err = ctypes.get_last_error()
        print(f"[错误] F1 注册失败（错误码 {err}），可能被别的程序占用。")
        stop.set()
        return
    print("[就绪] F1 已注册，切到游戏里按 F1 截图。\n")
    msg = wintypes.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam == 1:
                try:
                    for p in probe.shoot():
                        print(f"     -> {p}")
                except Exception as exc:  # 截屏失败不要弄死监听
                    print(f"[错误] 截图失败: {exc!r}")
    finally:
        user32.UnregisterHotKey(None, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="HD2 A 区域取景工具")
    ap.add_argument("--name", default="A", help="区域名，默认 A")
    ap.add_argument("--region", nargs=4, type=int, metavar=("L", "T", "R", "B"),
                    help="临时指定区域，不写入 config.json")
    ap.add_argument("--save-region", action="store_true", help="把 --region 写入 config.json")
    ap.add_argument("--zoom", type=int, default=12, help="网格图放大倍数，默认 12")
    ap.add_argument("--once", action="store_true", help="截一次就退出")
    args = ap.parse_args()

    set_dpi_aware()
    cfg = load_config()
    region_list = resolve_regions(cfg)

    if args.region:
        region = tuple(args.region)
    else:
        hit = [b for n, b in region_list if n == args.name]
        if not hit:
            print(f"[错误] config.json 里没有名为 {args.name} 的区域，"
                  f"可选 {[n for n, _ in region_list]}", file=sys.stderr)
            return 2
        region = hit[0]

    if args.save_region:
        regions = cfg.get("regions") or []
        for item in regions:
            if item.get("name", "main") == args.name:
                item["region"] = list(region)
                break
        else:
            regions.append({"name": args.name, "region": list(region)})
        cfg["regions"] = regions
        cfg["calibrated_for"] = list(screen_size())
        save_config(cfg)
        print(f"[配置] 已写入 config.json: {args.name} region={list(region)}")

    out_dir = desktop_dir() / cfg.get("shots_dir", "hd2_ammo_shots") / args.name
    probe = ProbeA(region, out_dir, args.zoom)

    sw, sh = screen_size()
    print("=" * 68)
    print(f"屏幕 {sw}x{sh} | {args.name} 区域 {region} "
          f"({region[2] - region[0]}x{region[3] - region[1]} px)")
    print(f"输出目录 {out_dir}")
    print("=" * 68)

    if args.once:
        probe.shoot()
        return 0

    stop = threading.Event()
    worker = threading.Thread(target=hotkey_loop, args=(probe, stop), daemon=True)
    worker.start()
    time.sleep(0.3)
    if stop.is_set():
        return 1

    print("结束方式：在本窗口输入 q 回车，或按 Ctrl+C。\n")
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            while not stop.wait(0.5):
                pass
        else:
            while True:
                line = sys.stdin.readline()
                if not line or line.strip().lower() in ("q", "quit", "exit"):
                    break
    except KeyboardInterrupt:
        print("\n[退出] Ctrl+C")
    stop.set()          # 热键线程是 daemon，主线程退出后随之结束
    print(f"[退出] 共截图 {probe.count} 次，文件在 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
