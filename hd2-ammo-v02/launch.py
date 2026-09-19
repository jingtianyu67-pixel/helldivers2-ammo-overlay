# -*- coding: utf-8 -*-
"""HD2 弹药叠加层启动器：游戏起来它起来，游戏关了它自己关门。

    python launch.py                      # 拉游戏 + 挂叠加层，游戏退出后自动收摊
    python launch.py --daemon             # 守护模式：不碰游戏，谁把游戏拉起来就跟谁走
    python launch.py --no-launch          # 不主动拉游戏，只等它自己起来
    python launch.py --attach             # 游戏已经在跑了，直接挂上去
    python launch.py -- --hz 30 --no-hotkey    # -- 之后的参数透传给 live_ammo.py
    python launch.py --log                # 叠加层输出写到 logs/ 下（无控制台时可查）

默认进程 helldivers2.exe、Steam AppID 553850，可用 --exe / --appid 改。

一次性模式（默认）
  1. 游戏不在跑 → 用 steam://rungameid/<appid> 让 Steam 拉起它；
  2. 检测到游戏进程 → 启动 live_ammo.py；
  3. 每秒盯一次游戏进程，掉线超过 --exit-grace 秒（默认 3s，防误判）就关掉叠加层并退出；
  4. 若 --boot-wait 秒内游戏一直没出现，说明拉不起来，收摊退出，不留孤儿进程。

守护模式（--daemon）
  不管游戏是谁拉起来的（Steam 库、桌面图标、远程串流），检测到 helldivers2.exe 就挂
  叠加层，进程消失就撤掉。适合「想照常点 Steam 里的 HD2」的用法。

做成 Steam 快捷方式：Steam → 库 → 添加游戏 → 添加非 Steam 游戏 → 选 启动_HD2叠加.bat。
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

from capture import app_dir

HERE = str(app_dir())
LIVE = os.path.join(HERE, "live_ammo.py")

TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.CloseHandle.argtypes = [wintypes.HANDLE]

# --------------------------------------------------------------------------- #
# Job Object：本进程一死（哪怕是任务管理器硬杀），里面挂着的叠加层跟着一起死，
# 不会留下抢不了手又关不掉的孤儿窗口。
# --------------------------------------------------------------------------- #
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong)]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


k32.CreateJobObjectW.restype = wintypes.HANDLE
k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
k32.SetInformationJobObject.restype = wintypes.BOOL
k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                        ctypes.c_void_p, wintypes.DWORD]
k32.AssignProcessToJobObject.restype = wintypes.BOOL
k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]


def make_killjob():
    h = k32.CreateJobObjectW(None, None)
    if not h:
        return None
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(h, JobObjectExtendedLimitInformation,
                                       ctypes.byref(info), ctypes.sizeof(info)):
        k32.CloseHandle(h)
        return None
    return h


JOB = make_killjob()
_job_warned = False


def process_running(exe: str) -> bool | None:
    """进程列表里有没有这个 exe。取进程快照失败时返回 None。"""
    want = exe.lower()
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == INVALID_HANDLE_VALUE:
        return None
    try:
        e = PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(e)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            if e.szExeFile.lower() == want:
                return True
            ok = k32.Process32NextW(snap, ctypes.byref(e))
        return False
    finally:
        k32.CloseHandle(snap)


def _say(msg: str, logf=None) -> None:
    """无控制台（--windowed exe）时 print 会被入口兜底到 devnull，日志只靠 logf。"""
    print(msg)
    if logf is not None:
        try:
            logf.write(msg + "\n")
        except Exception:
            pass


# 守护模式同一时间只留一份：两份会各起一个叠加层去抢 F1 热键。
ERROR_ALREADY_EXISTS = 183
k32.CreateMutexW.restype = wintypes.HANDLE
k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]


def single_instance():
    """拿到就返回句柄（进程活着就一直持有），已有实例返回 None，判断不了返回 True。"""
    h = k32.CreateMutexW(None, False, "Local\\HD2AmmoDaemon")
    if not h:
        return True
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        k32.CloseHandle(h)
        return None
    return h


def stop_child(p: subprocess.Popen | None) -> None:
    if p is None or p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()


CREATE_NO_WINDOW = 0x08000000


def overlay_exe() -> str:
    """派生叠加层用的解释器，一律用 console 版 python.exe + CREATE_NO_WINDOW。

    实测：pythonw 链路上派生出的叠加层出现过「窗口创建成功、API 报可见、屏幕上却
    没有」的情况；换成 python.exe 后同一份代码稳定上屏。CREATE_NO_WINDOW 保证
    不闪黑框。
    """
    exe = sys.executable
    if os.path.basename(exe).lower().startswith("pythonw"):
        cand = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.isfile(cand):
            return cand
    return exe


def overlay_cmd(passthru) -> list[str]:
    """叠加层的启动命令。

    源码运行：python(pythonw 时换回 python.exe) live_ammo.py ...
    打包成 exe：自己就是解释器 —— 用 sys.executable + `--overlay` 让同一个 exe
    以「纯叠加层」身份再起一份。不能再拼 live_ammo.py，那在 exe 里不存在。
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--overlay"] + list(passthru)
    return [overlay_exe(), LIVE] + list(passthru)


def spawn_overlay(args, logf) -> subprocess.Popen:
    cmd = overlay_cmd(args.passthru)
    if "--log" not in cmd:
        cmd.append("--log")          # 叠加层自己落盘到 logs/live_*.log
    print(f"[启动] 叠加层 -> {' '.join(cmd)}")
    if logf:
        logf.write(f"[启动] 叠加层 -> {' '.join(cmd)}\n")
    # 固定子进程输出编码：Windows 控制台默认 GBK，写日志/管道时会被解错
    cenv = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, cwd=HERE, env=cenv,
                         stdout=(logf if logf else subprocess.DEVNULL),
                         stderr=(subprocess.STDOUT if logf else subprocess.DEVNULL),
                         creationflags=CREATE_NO_WINDOW)
    global _job_warned
    if JOB is not None:
        if not k32.AssignProcessToJobObject(JOB, wintypes.HANDLE(int(p._handle))):
            if not _job_warned:
                _job_warned = True
                print("[启动] 提示：叠加层未纳入 job 保护，"
                      "本进程被强制结束时会留下孤儿（不影响正常使用）")
    print(f"[启动] 叠加层 PID {p.pid}，游戏退出后会自动关掉它")
    if sys.stdout is not None:           # exe 无控制台运行时 stdout 为 None
        sys.stdout.flush()
    return p


def run_daemon(args, logf) -> int:
    handle = single_instance()
    if handle is None:
        _say("[守护] 已经有一份在跑了，这份退出（同一时间只留一份，免得抢 F1）", logf)
        return 0
    _say(f"[守护] 每 {args.poll:g}s 看一眼 {args.exe}：游戏起来就挂叠加层，"
         f"关掉就撤。这个窗口/进程别关，关了就不自动挂了。", logf)
    child: subprocess.Popen | None = None
    miss = 0.0
    try:
        while True:
            time.sleep(args.poll)
            if process_running(args.exe):
                miss = 0.0
                if child is None or child.poll() is not None:
                    child = spawn_overlay(args, logf)
            elif child is not None and child.poll() is None:
                miss += args.poll
                if miss >= args.exit_grace:
                    _say("[守护] 游戏已关闭，撤掉叠加层", logf)
                    stop_child(child)
                    child = None
                    miss = 0.0
    except KeyboardInterrupt:
        _say("\n[守护] 中断退出", logf)
    finally:
        stop_child(child)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="HD2 弹药叠加层启动器",
                                 epilog="-- 之后的参数原样转给 live_ammo.py")
    ap.add_argument("--exe", default="helldivers2.exe", help="游戏进程名")
    ap.add_argument("--appid", default="553850", help="Steam AppID（用于拉起游戏）")
    ap.add_argument("--daemon", action="store_true",
                    help="守护模式：不拉游戏，只跟着游戏进程的生死挂载/撤销叠加层")
    ap.add_argument("--no-launch", action="store_true", help="不主动拉游戏，只等它自己起来")
    ap.add_argument("--attach", action="store_true", help="假定游戏已在运行，不等就挂")
    ap.add_argument("--boot-wait", type=float, default=300.0,
                    help="等游戏出现的最长秒数，默认 300")
    ap.add_argument("--exit-grace", type=float, default=3.0,
                    help="游戏进程消失多少秒后关闭叠加层，默认 3（防误判）")
    ap.add_argument("--poll", type=float, default=1.0, help="进程轮询间隔秒，默认 1")
    ap.add_argument("--log", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-log", action="store_true",
                    help="不写日志（默认写 logs/，无控制台时靠它排查）")
    ap.add_argument("passthru", nargs="*", help="透传给 live_ammo.py 的参数")
    args = ap.parse_args()

    if not getattr(sys, "frozen", False) and not os.path.isfile(LIVE):
        # 打包成 exe 后没有 live_ammo.py —— 叠加层由这个 exe 自己带 --overlay 起
        print(f"[错误] 找不到 {LIVE}")
        return 2

    logf = None
    if not args.no_log:
        d = os.path.join(HERE, "logs")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, time.strftime("ammo_%Y%m%d_%H%M%S.log"))
        logf = open(path, "w", encoding="utf-8", buffering=1)
        print(f"[启动] 日志 -> {path}")

    try:
        if args.daemon:
            return run_daemon(args, logf)

        running = process_running(args.exe)
        if running is None:
            print("[错误] 拿不到进程列表（快照失败），无法判断游戏状态")
            return 2
        print(f"[启动] {args.exe}：{'运行中' if running else '未运行'}")

        if not running:
            if args.attach:
                print(f"[错误] 用了 --attach 但 {args.exe} 没在跑")
                return 1
            if args.no_launch:
                print(f"[启动] --no-launch：等 {args.exe} 自己起来"
                      f"（最多 {args.boot_wait:g}s）")
            else:
                print(f"[启动] 让 Steam 拉起 AppID {args.appid} ...")
                os.startfile(f"steam://rungameid/{args.appid}")

        child: subprocess.Popen | None = None
        t0 = time.time()
        if not running:
            while True:
                time.sleep(args.poll)
                if process_running(args.exe):
                    print(f"[启动] 游戏进程出现（等了 {time.time() - t0:.1f}s）")
                    break
                if time.time() - t0 > args.boot_wait:
                    print(f"[启动] 等了 {args.boot_wait:g}s 游戏还没起来，放弃")
                    return 1

        child = spawn_overlay(args, logf)

        miss = 0.0
        while True:
            time.sleep(args.poll)
            if child.poll() is not None:
                print(f"[退出] 叠加层自己退出了（code {child.returncode}），收摊")
                break
            if process_running(args.exe):
                miss = 0.0
                continue
            miss += args.poll
            if miss >= args.exit_grace:
                print("[退出] 游戏已关闭，收摊（叠加层已停止）")
                break
    except KeyboardInterrupt:
        print("\n[退出] 手动中断")
    finally:
        try:
            stop_child(child)
        except NameError:
            pass
        if logf:
            logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
