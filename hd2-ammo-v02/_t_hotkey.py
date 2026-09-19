"""实测：live_ammo 真实跑起来后，模拟按键 F1/F2/F3/F4，看回调到底进不进。"""
import ctypes
import os
import subprocess
import sys
import time

user32 = ctypes.WinDLL("user32")
KEYEVENTF_KEYUP = 0x0002
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def tap(vk, hold=0.06):
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(hold)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
    time.sleep(0.3)


p = subprocess.Popen([PY, "live_ammo.py", "--duration", "16", "--no-overlay"],
                     cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, encoding="utf-8", errors="replace",
                     env=dict(os.environ, PYTHONIOENCODING="utf-8"))
time.sleep(4.0)
print("[test] live_ammo 已跑起来，开始发按键")
for name, vk in (("F1", 0x70), ("F2", 0x71), ("F3", 0x72), ("F4", 0x73)):
    tap(vk)
    print(f"[test] 已发送 {name}")

out = p.communicate(timeout=40)[0] or ""
lines = out.replace("\r", "\n").split("\n")
hit = [ln for ln in lines if "[热键]" in ln or "区域模式" in ln or "暂停" in ln]
print(f"---- 输出共 {len(lines)} 行，其中热键相关 {len(hit)} 行 ----")
for ln in hit:
    print("   ", ln)
print("---- 最后 3 行 ----")
for ln in [x for x in lines if x.strip()][-3:]:
    print("   ", ln)
