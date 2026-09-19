# -*- coding: utf-8 -*-
"""收掉占着 dist 里 exe 的残留进程，再用新打出来的覆盖它。"""
import ctypes
import os
import sys
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import k32, PROCESSENTRY32W, TH32CS_SNAPPROCESS  # noqa: E402

PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
k32.OpenProcess.restype = wintypes.HANDLE
k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]

D = Path(r"C:\Users\Administrator\WorkBuddy\2026-09-18-21-15-56\hd2-ammo-v02\dist")
WANT = {"hd2弹药叠加.exe", "hd2ammooverlay.exe"}

snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
e = PROCESSENTRY32W()
e.dwSize = ctypes.sizeof(e)
pids = []
ok = k32.Process32FirstW(snap, ctypes.byref(e))
while ok:
    if e.szExeFile.lower() in WANT:
        pids.append(e.th32ProcessID)
    ok = k32.Process32NextW(snap, ctypes.byref(e))
k32.CloseHandle(snap)
print("占用进程:", pids)

for pid in pids:
    h = k32.OpenProcess(PROCESS_TERMINATE, False, pid)
    if h:
        k32.TerminateProcess(h, 1)
        k32.CloseHandle(h)

import time
time.sleep(1.0)

new = D / "HD2AmmoOverlay.exe"
old = D / "HD2弹药叠加.exe"
if new.is_file():
    try:
        os.replace(str(new), str(old))
        print("已覆盖:", old.name, old.stat().st_size)
    except PermissionError as exc:
        print("覆盖仍失败:", exc)
else:
    print("没有新 exe 可覆盖:", new)
