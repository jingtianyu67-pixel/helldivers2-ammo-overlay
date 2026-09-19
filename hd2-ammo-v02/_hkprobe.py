# -*- coding: utf-8 -*-
"""跨进程热键探针：本进程注册并等待按键，把每次触发写进日志文件。"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hotkeys import HotkeyThread   # noqa: E402

LOG = os.path.join(HERE, "_out", "hkprobe.log")
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
fired: list[str] = []
with open(LOG, "w", encoding="utf-8") as f:
    f.write("start\n")

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
while time.time() - t0 < secs:
    time.sleep(0.05)
with open(LOG, "a", encoding="utf-8") as f:
    f.write(f"end, total {len(fired)}\n")
hk.stop()
