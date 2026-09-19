# -*- coding: utf-8 -*-
"""实时弹药余量识别 + 准星旁弧线覆盖层。

区域选择：
  - **自动**：粘性 fallback —— 认准一组就一直用，失效才换另一组，
    两组都识别不到则输出「未检测到」。
  - **手动**：F1 = 靠左的那组区域，F2 = 靠右的那组，F4 = 回到自动。
    手动模式下只截锁定区域，不做 fallback，识别不到就报「未检测到」。

热键：
  F1  锁定靠左区域        F2  锁定靠右区域
  F3  暂停 / 恢复（暂停时停止截图与检测，覆盖层隐藏）
  F4  回到自动模式

用法：
    python live_ammo.py                  # 常驻，实时刷新一行，Ctrl+C 退出
    python live_ammo.py --duration 10    # 跑 10 秒自动退出并打印统计
    python live_ammo.py --scroll         # 每帧新起一行（便于重定向到文件）
    python live_ammo.py --hz 25          # 改采样率
    python live_ammo.py --diag           # 行尾附带判定原因
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
from capture import grab_rgb, load_config, resolve_regions, set_dpi_aware
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
        tag = {"white": "白", "empty": "空"}.get(r.state, "?")
        via = "探针" if not r.hint.startswith("shape") else "形状"
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
    ap.add_argument("--lock", choices=["F1", "F2", "auto"], default="F2",
                    help="启动时锁定的区域，默认 F2（靠右）；auto = 自动 fallback")
    args = ap.parse_args()

    if args.log or sys.stdout is None:
        # 无控制台启动（pythonw / 被启动器派生）时自己落盘，否则输出白丢
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(d, exist_ok=True)
        sys.stdout = open(os.path.join(d, time.strftime("live_%Y%m%d_%H%M%S.log")),
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

    # 热键绑到「屏幕最左 / 次左」的区域，不依赖 config 里的先后顺序
    by_x = sorted(range(len(det.specs)), key=lambda i: det.specs[i].box[0])
    hot = {"F1": by_x[0], "F2": by_x[1] if len(by_x) > 1 else by_x[0]}
    req: dict = {"v": None}

    print("=" * 76)
    for spec in det.specs:
        l, t, r, b = spec.box
        w = spec.detector.window
        print(f"区域 {spec.name}: ({l},{t})-({r},{b})  {r - l}x{b - t} px  "
              f"窗口 {'固定 ' + str(int(w.sum())) + ' px' if w is not None else '自动'}")
    print(f"形状模板 {det.specs[0].detector.template.shape} | 采样率 {hz:g} Hz")
    if det.union:
        ul, ut, ur, ub = det.union
        print(f"并集截图 ({ul},{ut})-({ur},{ub}) {ur - ul}x{ub - ut} px（未锁定时一次抓取再切片）")
    print("模式：启动默认锁定 F2（靠右区域），可随时用热键切换")
    print(f"热键：F1 = 锁定靠左区域（{det.specs[hot['F1']].name}） | "
          f"F2 = 锁定靠右区域（{det.specs[hot['F2']].name}）")
    print("      F3 = 暂停/恢复 | F4 = 回到自动")

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

    det.force(None if args.lock == "auto" else hot[args.lock])

    hk: HotkeyThread | None = None
    if not args.no_hotkey:
        hk = HotkeyThread([("F1", lambda: req.__setitem__("v", hot["F1"])),
                           ("F2", lambda: req.__setitem__("v", hot["F2"])),
                           ("F3", lambda: req.__setitem__("v", -2)),
                           ("F4", lambda: req.__setitem__("v", -1))])
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
    stats = {"white": 0, "empty": 0, "none": 0}

    paused = False
    reading = AmmoReading(False)
    try:
        while True:
            nxt += period
            # 热键回调在别的线程里，只投递请求，真正的切换在主循环执行
            if req["v"] is not None:
                want = req["v"]
                req["v"] = None
                if want == -2:                                  # F3：开关
                    paused = not paused
                    if paused and ov is not None:
                        ov.hide()
                    print(("\n" if not args.scroll else "")
                          + f"[热键] {'暂停（停止截图与检测）' if paused else '恢复'}"
                          + ("\n" if not args.scroll else ""))
                else:
                    if paused:                                  # 切区域时自动恢复
                        paused = False
                    det.force(None if want < 0 else want)
                    print(("\n" if not args.scroll else "")
                          + f"[热键] 区域模式 -> {det.mode}"
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

            line = render(reading, hz_now, (t1 - t0) * 1000,
                          " ".join(det.last_scan), args.diag, det.mode)
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
