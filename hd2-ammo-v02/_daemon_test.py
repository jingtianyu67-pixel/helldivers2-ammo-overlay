# -*- coding: utf-8 -*-
"""守护模式验收：单实例互斥 + 无游戏时不挂叠加层 + 日志落盘。"""
import os
import subprocess
import sys
import time

PY = r"C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
D = r"C:\Users\Administrator\WorkBuddy\2026-09-18-21-15-56\hd2-ammo-v02"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
ARGS = [PY, "launch.py", "--daemon", "--poll", "1"]


def start():
    return subprocess.Popen(ARGS, cwd=D, env=ENV, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")


print("=== 第一份 ===")
a = start()
time.sleep(2.5)
print("A alive:", a.poll() is None)

print("=== 第二份（应被互斥挡下） ===")
b = start()
try:
    ob, _ = b.communicate(timeout=20)
except subprocess.TimeoutExpired:
    b.kill()
    ob = "<超时未退出：互斥失效>"
print("B rc:", b.returncode)
print("B out:", ob)

print("=== 收掉第一份 ===")
a.terminate()
try:
    oa, _ = a.communicate(timeout=10)
except subprocess.TimeoutExpired:
    a.kill()
    oa = "<超时>"
print("A out:", oa)
