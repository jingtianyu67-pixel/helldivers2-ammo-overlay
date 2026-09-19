# -*- coding: utf-8 -*-
"""环境探测：分辨率 / DPI / tkinter / 截图通道可用性。"""
import ctypes
from ctypes import wintypes

u32 = ctypes.windll.user32
shcore = ctypes.windll.shcore

# 进程 DPI 感知
try:
    shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
    dpi_mode = "per-monitor-aware"
except Exception as e:
    dpi_mode = f"fail({e})"

u32.SetProcessDPIAware()

print(f"dpi awareness: {dpi_mode}")
print(f"primary screen: {u32.GetSystemMetrics(0)} x {u32.GetSystemMetrics(1)}")

try:
    print(f"virtual screen: {u32.GetSystemMetrics(78)} x {u32.GetSystemMetrics(79)}"
          f"  origin=({u32.GetSystemMetrics(76)},{u32.GetSystemMetrics(77)})")
except Exception:
    pass

# 每显示器 DPI
try:
    hdc = u32.GetDC(0)
    print("GetDpiForSystem:", u32.GetDpiForSystem())
    u32.ReleaseDC(0, hdc)
except Exception as e:
    print("dpi q fail", e)

# 显示器枚举
monitors = []
MonitorEnumProc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                                     ctypes.POINTER(wintypes.RECT), ctypes.c_double)
def _cb(hmon, hdc, lprc, data):
    r = lprc.contents
    monitors.append((r.left, r.top, r.right, r.bottom))
    return 1
u32.EnumDisplayMonitors(0, None, MonitorEnumProc(_cb), 0)
for i, m in enumerate(monitors):
    print(f"monitor[{i}]: L{m[0]} T{m[1]} R{m[2]} B{m[3]}  ({m[2]-m[0]}x{m[3]-m[1]})")

# tkinter
try:
    import tkinter
    r = tkinter.Tk()
    r.withdraw()
    print(f"tkinter: OK  TkVersion={tkinter.TkVersion}  scaling={r.tk.call('tk', 'scaling')}")
    r.destroy()
except Exception as e:
    print(f"tkinter: FAIL {e}")

# mss 截一帧测速
try:
    import time
    import mss
    with mss.mss() as sct:
        full = sct.monitors[0]
        print(f"mss virtual monitor: {full}")
        t0 = time.perf_counter()
        for _ in range(20):
            sct.grab(full)
        dt = (time.perf_counter() - t0) / 20
        print(f"mss full-screen grab: {dt*1000:.2f} ms/frame  ({1/dt:.0f} fps max)")
        t0 = time.perf_counter()
        box = {"left": 400, "top": 1200, "width": 120, "height": 100}
        for _ in range(200):
            sct.grab(box)
        dt = (time.perf_counter() - t0) / 200
        print(f"mss small-region grab: {dt*1000:.3f} ms/frame  ({1/dt:.0f} fps max)")
except Exception as e:
    print(f"mss: FAIL {e}")
