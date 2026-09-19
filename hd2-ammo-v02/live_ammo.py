# -*- coding: utf-8 -*-
"""实时弹药余量识别 + 准星旁弧线覆盖层。

HUD 布局（背包 / 无背包）：
  游戏里背不背补给背包，左下角弹药 HUD 会整体平移（实测左移 106px）。
  每隔 `layout.interval` 秒（默认 1 秒）从判据矩形里找一条白色竖线来判：
  有竖线 = 有背包 = 用 config 里标定的坐标；无竖线 = 无背包 = 区域坐标整体加
  `layout.shift`。判据与区域共用一次截图，非检查帧沿用上一次结论。

热键：
  F1  开启 / 关闭（关闭时停止截图与检测，覆盖层隐藏）

用法：
    python live_ammo.py                  # 常驻，实时刷新一行，Ctrl+C 退出
    python live_ammo.py --duration 10    # 跑 10 秒自动退出并打印统计
    python live_ammo.py --scroll         # 每帧新起一行（便于重定向到文件）
    python live_ammo.py --hz 25          # 改采样率
    python live_ammo.py --diag           # 行尾附带判定原因
    python live_ammo.py --layout nobag   # 强制按「无背包」坐标跑（标定用）
    python live_ammo.py --no-hotkey      # 不注册热键
    python live_ammo.py --no-overlay     # 不显示弧线覆盖层
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import mss
import numpy as np

from ammo_detect import AmmoReading, build_detector
from capture import (app_dir, grab_rgb, load_config, resolve_layout,
                     resolve_regions, set_dpi_aware)
from hotkeys import HotkeyThread

try:  # Windows 定时器精度：默认 15.6ms 会让 50Hz 排不满
    _winmm = ctypes.WinDLL("winmm")
except Exception:  # pragma: no cover
    _winmm = None


def make_mss():
    try:
        return mss.MSS()
    except AttributeError:
        return mss.mss()


def render(r: AmmoReading, hz_now: float, ms: float, scan: str,
           diag: bool, mode: str, paused: bool = False) -> str:
    if paused:
        mid = "[ 已暂停 ]"
    elif r.valid:
        body = f"{r.fraction * 100:5.1f}%"
        tag = {"white": "白", "low": "红", "empty": "空"}.get(r.state, "?")
        via = {"shell": "空壳通道"}.get(
            r.hint, "探针" if not r.hint.startswith("shape") else "形状")
        if r.state == "low":
            body = f"红{r.red_px:3d}px"
            via = "红色通道"
        mid = (f"[{body}] {r.bar[1:-1]} {r.region} {tag} {via}"
               f" IoU{r.shape_iou:.2f}")
    else:
        mid = "[ 未检测到 ]"
    tail = f"  | {scan}" if diag else ""
    return f"{mid}  [{mode}]  {hz_now:5.1f}Hz {ms:4.1f}ms{tail}"


def main() -> int:
    ap = argparse.ArgumentParser(description="HD2 弹药余量实时识别（实验脚本）")
    ap.add_argument("--hz", type=float, default=None, help="采样率，默认取 config")
    ap.add_argument("--cal", action="store_true", help="统计原始填充像素极值（标定用）")
    ap.add_argument("--scroll", action="store_true", help="每帧新起一行")
    ap.add_argument("--diag", action="store_true", help="行尾显示各区域判定原因")
    ap.add_argument("--duration", type=float, default=None, help="运行秒数，到点自动退出")
    ap.add_argument("--no-hotkey", action="store_true", help="不注册全局热键")
    ap.add_argument("--no-overlay", action="store_true", help="不显示弧线覆盖层")
    ap.add_argument("--overlay-demo", type=float, default=None, metavar="F",
                    help="诊断：无视识别结果，始终按固定余量画弧线（0-1，或写百分数如 50）")
    ap.add_argument("--log", action="store_true",
                    help="把输出写进 logs/live_*.log（无控制台启动时用）")
    ap.add_argument("--layout", choices=["auto", "bag", "nobag"], default="auto",
                    help="启动时的 HUD 布局：auto = 每帧按判据自动（默认）；"
                         "bag / nobag = 强制按有/无背包的坐标（标定用）")
    args = ap.parse_args()

    if args.log or sys.stdout is None:
        # 无控制台启动（exe / pythonw / 被启动器派生）时自己落盘，否则输出白丢
        d = app_dir() / "logs"
        os.makedirs(d, exist_ok=True)
        sys.stdout = open(d / time.strftime("live_%Y%m%d_%H%M%S.log"),
                          "w", encoding="utf-8", buffering=1)
        print(f"日志 -> {sys.stdout.name}")

    set_dpi_aware()
    cfg = load_config()
    regions = resolve_regions(cfg)
    hz = args.hz or float(cfg.get("hz", 50))

    det = build_detector(cfg)
    # 用换算后的屏幕坐标覆盖配置里的 box
    for spec, (name, box) in zip(det.specs, regions):
        spec.box = box
    # 布局判据矩形与位移也要按分辨率换算，然后重算抓取范围
    lay = resolve_layout(cfg)
    if lay is not None and det.layout is not None:
        det.layout.rect, det.layout.shift = lay
        det.update_canvas()

    req: dict = {"v": None}

    print("=" * 76)
    for spec in det.specs:
        l, t, r, b = spec.box
        w = spec.detector.window
        print(f"区域 {spec.name}: ({l},{t})-({r},{b})  {r - l}x{b - t} px  "
              f"窗口 {'固定 ' + str(int(w.sum())) + ' px' if w is not None else '自动'}")
    print(f"形状模板 {det.specs[0].detector.template.shape} | 采样率 {hz:g} Hz")
    if det.layout is not None:
        pl, pt, pr, pb = det.layout.rect
        print(f"布局判据：({pl},{pt})-({pr},{pb})  {pr - pl}x{pb - pt} px 里找白色竖线"
              f"（竖直连续 >= {det.layout.min_run}px、亮度 > {det.layout.lum:g}）")
        sx, sy = det.layout.shift
        print(f"          有竖线 = 有背包 = 用上面坐标；无竖线 → 区域平移 "
              f"({sx:+d},{sy:+d})")
        print(f"          每 {det.layout.interval:g} 秒重判一次")
        if det.layout_forced is not None:
            print(f"          启动强制为「{'有背包' if det.layout_forced else '无背包'}」")
    cl, ct, cr, cb = det.canvas
    print(f"抓取范围 ({cl},{ct})-({cr},{cb}) {cr - cl}x{cb - ct} px"
          f"（一次截图，布局判据与区域都从这里切）")
    print("热键：F1 = 开启 / 关闭（关闭时停止截图与检测、覆盖层隐藏）")
    if args.layout != "auto":
        det.force_layout(args.layout == "bag")

    ov = None
    demo = None
    if args.overlay_demo is not None:
        demo = args.overlay_demo / 100.0 if args.overlay_demo > 1.0 else args.overlay_demo
        demo = max(0.0, min(1.0, demo))
    if not args.no_overlay and cfg.get("overlay", {}).get("enabled", True):
        from overlay import AmmoOverlay, screen_size
        ov = AmmoOverlay(cfg, screen_size())
        ov.on_show = lambda rct: print(
            f"[覆盖层] 已上屏 矩形=({rct[0]},{rct[1]})-({rct[2]},{rct[3]})",
            flush=True)
        x0, y0, x1, y1 = ov.geo.bbox()
        print(f"覆盖层：弧线 bbox ({x0},{y0})-({x1},{y1}) {x1 - x0}x{y1 - y0} px  "
              f"半径 {ov.geo.r:g} 圆心 ({ov.geo.cx:g},{ov.geo.cy:g})")
        if demo is not None:
            print(f"覆盖层：演示模式，固定显示 {demo * 100:.0f}%（忽略识别结果）")
    else:
        print("覆盖层：已关闭")
    print("=" * 76)
    sys.stdout.flush()

    hk: HotkeyThread | None = None
    if not args.no_hotkey:
        hk = HotkeyThread([("F1", lambda: req.__setitem__("v", "toggle"))])
        hk.start()
        hk.wait_ready()
        if hk.failed:
            print(f"[热键] 注册失败：{', '.join(hk.failed)}（可能被其它程序占用）")
        if hk.ok:
            print(f"[热键] 已注册：{', '.join(hk.ok)}")

    if _winmm:
        _winmm.timeBeginPeriod(1)
    ctx = make_mss()
    period = 1.0 / hz
    nxt = time.perf_counter()
    count = 0
    t_start = time.perf_counter()
    fills: list[int] = []
    cap_total = 0.0
    hz_now = 0.0
    last_report = t_start
    stats = {"white": 0, "low": 0, "empty": 0, "none": 0}

    paused = False
    reading = AmmoReading(False)
    try:
        while True:
            nxt += period
            # 热键回调在别的线程里，只投递请求，真正的切换在主循环执行
            if req["v"] is not None:
                want = req["v"]
                req["v"] = None
                if want == "toggle":
                    paused = not paused
                    if paused and ov is not None:
                        ov.hide()
                    print(("\n" if not args.scroll else "")
                          + f"[热键] {'关闭（停止截图与检测）' if paused else '开启'}"
                          + ("\n" if not args.scroll else ""))

            if paused:
                now = time.perf_counter()
                if now - last_report >= 0.5:
                    hz_now = (count / (now - t_start)) if now > t_start else 0.0
                    last_report = now
                line = render(reading, hz_now, 0.0, "", args.diag, det.mode, paused=True)
                print(("\n" + line) if args.scroll else line,
                      end="" if args.scroll else "\r", flush=True)
                if args.duration and now - t_start >= args.duration:
                    break
                time.sleep(0.05)
                nxt = time.perf_counter()
                continue

            t0 = time.perf_counter()
            reading = det.analyze(lambda box: grab_rgb(ctx, box))
            t1 = time.perf_counter()
            if ov is not None:
                if demo is not None:
                    ov.show_value(demo, 0)
                else:
                    ov.update(reading)

            count += 1
            cap_total += t1 - t0
            stats["none" if not reading.valid else reading.state] = \
                stats.get("none" if not reading.valid else reading.state, 0) + 1
            if reading.valid:
                fills.append(reading.fill)
            now = time.perf_counter()
            if now - last_report >= 0.5:
                hz_now = (count / (now - t_start)) if now > t_start else 0.0
                last_report = now

            scan = " ".join(det.last_scan)
            if det.layout is not None:
                scan = (f"竖线{det.last_run}px("
                        f"{'有包' if det.has_backpack else '无包'}) " + scan)
            line = render(reading, hz_now, (t1 - t0) * 1000,
                          scan, args.diag, det.mode)
            print(("\n" + line) if args.scroll else line, end="" if args.scroll else "\r",
                  flush=True)

            if args.duration and now - t_start >= args.duration:
                break

            slack = nxt - time.perf_counter()
            if slack > 0.002:
                time.sleep(slack - 0.0015)
            while time.perf_counter() < nxt:
                pass
            if time.perf_counter() - nxt > period * 3:      # 落后太多就重新对时
                nxt = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        if hk is not None:
            hk.stop()
        if ov is not None:
            ov.close()
        if _winmm:
            _winmm.timeEndPeriod(1)
        ctx.close()

    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 76)
    cap_avg = cap_total / max(1, count)
    print(f"运行 {elapsed:.1f}s，采样 {count} 帧，平均 {count / elapsed:.1f} Hz")
    print(f"单帧总耗时均值 {cap_avg * 1000:.2f} ms（含截图+识别，"
          f"50Hz 下单核约 {cap_avg * 50 * 100:.0f}% 上限）")
    if ov is not None:
        print(f"覆盖层重绘 {ov.redraws} 次（仅余量变化超过 "
              f"{ov.min_step * 100:g}% 时重画）")
        print(f"覆盖层状态：{ov.status()}")
        if ov.redraws == 0:
            print("提示：整段运行没有一次有效读数，覆盖层从未显示。"
                  "跑 `--overlay-demo 50` 可以撇开识别单独验证窗口能否盖住游戏。")
    print("状态分布: " + "  ".join(f"{k}={v}" for k, v in stats.items()))
    if fills:
        arr = np.array(fills)
        print(f"有效帧填充像素 min={arr.min()} max={arr.max()} 中位={np.median(arr):.0f}")
        for p in (0, 5, 50, 95, 100):
            print(f"    P{p:<3d} = {np.percentile(arr, p):7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
