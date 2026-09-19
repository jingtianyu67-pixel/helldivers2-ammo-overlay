# -*- coding: utf-8 -*-
"""跨进程热键探针 v2：对比「纯 sleep 主循环」与「忙等主循环」下热键是否还能触发。

用法：python _hkprobe2.py <秒数> <sleep|busy|busy_sleep>
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hotkeys import HotkeyThread   # noqa: E402

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
mode = sys.argv[2] if len(sys.argv) > 2 else "sleep"
LOG = os.path.join(HERE, "_out", f"hkprobe2_{mode}.log")

fired: list[str] = []
with open(LOG, "w", encoding="utf-8") as f:
    f.write(f"start mode={mode}\n")


def mk(nm):
    def cb():
        fired.append(nm)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"fired {nm} @ {time.time():.3f}\n")
    return cb


hk = HotkeyThread([("F1", mk("F1")), ("F2", mk("F2")),
                   ("F3", mk("F3")), ("F4", mk("F4"))])
hk.start()
hk.wait_ready()
with open(LOG, "a", encoding="utf-8") as f:
    f.write(f"ok={hk.ok} failed={hk.failed}\n")

t0 = time.time()
t0p = time.perf_counter()
if mode == "sleep":
    while time.time() - t0 < secs:
        time.sleep(0.02)
elif mode == "busy":
    period = 1.0 / 50
    nxt = time.perf_counter()
    while time.perf_counter() - t0p < secs:
        nxt += period
        while time.perf_counter() < nxt:
            pass
elif mode == "busy_sleep":
    period = 1.0 / 50
    nxt = time.perf_counter()
    while time.perf_counter() - t0p < secs:
        nxt += period
        slack = nxt - time.perf_counter()
        if slack > 0.002:
            time.sleep(slack - 0.0015)
        while time.perf_counter() < nxt:
            pass

with open(LOG, "a", encoding="utf-8") as f:
    f.write(f"end, total {len(fired)}\n")
hk.stop()
