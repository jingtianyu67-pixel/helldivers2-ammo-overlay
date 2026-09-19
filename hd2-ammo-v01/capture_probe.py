# -*- coding: utf-8 -*-
"""HD2 弹药区域取景工具。

用途：在游戏内按 F1，把「准备截取的区域」原样存到桌面，用来核对坐标是否正确。
每次按 F1 会存 3 张图：
    <时间>_1x.png        区域原图（物理像素，无损 PNG）
    <时间>_8x.png        区域放大 8 倍（NEAREST，方便肉眼看清弹匣格子）
    <时间>_full.png      整屏截图，红框标出区域、绿十字标出准星（屏幕中心）

用法：
    python capture_probe.py            # 常驻，按 F1 截图，按 q 或 Ctrl+C 退出
    python capture_probe.py --once     # 立刻截一次然后退出（自检用）
    python capture_probe.py --region 330 1250 390 1300    # 临时指定区域
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
from PIL import Image, ImageDraw

from capture import (BASE_DIR, desktop_dir, grab_rgb, load_config, resolve_regions,
                     save_config, screen_size, set_dpi_aware)

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


def region_stats(rgb: np.ndarray) -> str:
    """给区域图打个简要的像素统计，方便判断取景对不对。"""
    f = rgb.astype(np.int16)
    lum = (0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2])
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    bright = lum > 165
    red = (r > 140) & (r - g > 45) & (r - b > 45)
    total = lum.size
    rows = bright.sum(axis=1)
    top = np.argsort(rows)[::-1][:6]
    row_txt = ", ".join(f"y{int(i)}({int(rows[i])})" for i in top)
    return (f"亮度均值 {lum.mean():5.1f} | 亮像素 {bright.sum() * 100 / total:5.1f}% | "
            f"红像素 {red.sum() * 100 / total:5.1f}% | 最亮行 {row_txt}")


class Probe:
    def __init__(self, region, out_dir: Path, zoom: int):
        self.region = region
        self.out_dir = out_dir
        self.zoom = zoom
        self.count = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def shoot(self) -> list[Path]:
        self.count += 1
        ts = datetime.now().strftime("%m%d_%H%M%S")
        sw, sh = screen_size()
        with mss.mss() as sct:
            rgb = grab_rgb(sct, self.region)
            full = np.array(sct.grab({"left": 0, "top": 0, "width": sw, "height": sh}))
            full = full[:, :, 2::-1]

        left, top, right, bottom = self.region
        saved: list[Path] = []

        p1 = self.out_dir / f"{ts}_1x.png"
        Image.fromarray(rgb).save(p1)
        saved.append(p1)

        pz = self.out_dir / f"{ts}_{self.zoom}x.png"
        img = Image.fromarray(rgb)
        Image.fromarray(np.asarray(
            img.resize((img.width * self.zoom, img.height * self.zoom), Image.NEAREST)
        )).save(pz)
        saved.append(pz)

        pfull = self.out_dir / f"{ts}_full.png"
        canvas = Image.fromarray(full).convert("RGB")
        d = ImageDraw.Draw(canvas)
        d.rectangle([left - 1, top - 1, right, bottom], outline=(255, 0, 0), width=2)
        cx, cy = sw // 2, sh // 2
        d.line([cx - 18, cy, cx + 18, cy], fill=(0, 255, 0), width=2)
        d.line([cx, cy - 18, cx, cy + 18], fill=(0, 255, 0), width=2)
        canvas.save(pfull)
        saved.append(pfull)

        with open(self.out_dir / "shots.log", "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')}\t"
                     f"#{self.count}\tregion={self.region}\tscreen={sw}x{sh}\t"
                     f"{region_stats(rgb)}\n")

        print(f"[{self.count:02d}] 区域 {self.region}  尺寸 {right - left}x{bottom - top}")
        print(f"     {region_stats(rgb)}")
        for p in saved:
            print(f"     -> {p}")
        return saved


def hotkey_loop(probe: Probe, stop: threading.Event) -> None:
    tid = ctypes.windll.kernel32.GetCurrentThreadId()
    probe.thread_id = tid
    if not user32.RegisterHotKey(None, 1, MOD_NOREPEAT, VK_F1):
        err = ctypes.get_last_error()
        print(f"[错误] F1 注册失败（错误码 {err}）。可能有别的程序占用了 F1，"
              f"改 capture_probe.py 里的 VK_F1 换一个键。")
        stop.set()
        return
    print("[就绪] F1 已注册，切到游戏里按 F1 即可截图。\n")
    msg = wintypes.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam == 1:
                try:
                    probe.shoot()
                except Exception as exc:  # 截屏失败不要弄死监听
                    print(f"[错误] 截图失败: {exc!r}")
    finally:
        user32.UnregisterHotKey(None, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="HD2 弹药区域取景工具")
    ap.add_argument("--name", default=None,
                    help="按 config.json 里的区域名取坐标（如 A / B），默认取第一个")
    ap.add_argument("--once", action="store_true", help="截一次就退出")
    ap.add_argument("--region", nargs=4, type=int, metavar=("L", "T", "R", "B"),
                    help="临时指定区域，不写入 config.json")
    ap.add_argument("--save-region", action="store_true",
                    help="把 --region 写入 config.json")
    ap.add_argument("--zoom", type=int, default=None, help="放大倍数，默认取 config")
    args = ap.parse_args()

    set_dpi_aware()
    cfg = load_config()

    region_list = resolve_regions(cfg)
    if args.region:
        region = tuple(args.region)
    elif args.name:
        hit = [b for n, b in region_list if n == args.name]
        if not hit:
            print(f"[错误] config.json 里没有名为 {args.name} 的区域，"
                  f"可选 {[n for n, _ in region_list]}", file=sys.stderr)
            return 2
        region = hit[0]
    else:
        region = region_list[0][1]

    if args.save_region:
        want = args.name or region_list[0][0]
        regions = cfg.get("regions") or [{"name": "main", "region": list(region)}]
        for item in regions:
            if item.get("name", "main") == want:
                item["region"] = list(region)
                break
        else:
            regions.append({"name": want, "region": list(region)})
        cfg["regions"] = regions
        cfg["calibrated_for"] = list(screen_size())
        save_config(cfg)
        print(f"[配置] 已写入 config.json: {want} region={list(region)} "
              f"calibrated_for={list(screen_size())}")

    out_dir = desktop_dir() / cfg.get("shots_dir", "hd2_ammo_shots")
    zoom = args.zoom or int(cfg.get("zoom", 8))
    probe = Probe(region, out_dir, zoom)

    sw, sh = screen_size()
    print("=" * 64)
    print(f"屏幕 {sw}x{sh} | 截取区域 {region} "
          f"({region[2] - region[0]}x{region[3] - region[1]} px)")
    print(f"输出目录 {out_dir}")
    print("=" * 64)

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
    tid = getattr(probe, "thread_id", None)
    if tid:
        user32.PostThreadMessageW(tid, WM_QUIT, 0, 0)
    worker.join(timeout=1.5)
    print(f"[退出] 共截图 {probe.count} 次，文件在 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
