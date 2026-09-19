# -*- coding: utf-8 -*-
"""Win32 全局热键（不需要窗口）。

RegisterHotKey 传 hwnd=NULL 时，WM_HOTKEY 会被投递到**调用线程**的消息队列，
所以注册和 GetMessage 循环必须在同一个线程里跑——本模块把两者封在一个线程内。

用法：
    hk = HotkeyThread([("F1", cb1), ("F2", cb2)])
    hk.start()
    hk.wait_ready()          # 拿到注册结果（hk.failed 列出注册失败的键）
    ...
    hk.stop()

注意：热键是**系统级独占**的。注册成功后，按键不会再传给游戏/其它程序。
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_NOREPEAT = 0x4000

VK: dict[str, int] = {f"F{i}": 0x6F + i for i in range(1, 13)}   # F1=0x70 .. F12=0x7B
VK.update({chr(c): c for c in range(0x30, 0x3A)})                # 0-9


class _MSG(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND),
                ("message", wintypes.UINT),
                ("wParam", wintypes.WPARAM),
                ("lParam", wintypes.LPARAM),
                ("time", wintypes.DWORD),
                ("pt", wintypes.POINT)]


user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int,
                                  wintypes.UINT, wintypes.UINT]
user32.RegisterHotKey.restype = wintypes.BOOL
user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
user32.UnregisterHotKey.restype = wintypes.BOOL
user32.GetMessageW.argtypes = [ctypes.POINTER(_MSG), wintypes.HWND,
                               wintypes.UINT, wintypes.UINT]
user32.GetMessageW.restype = ctypes.c_int
user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT,
                                      wintypes.WPARAM, wintypes.LPARAM]
user32.PostThreadMessageW.restype = wintypes.BOOL


class HotkeyThread(threading.Thread):
    """在后台线程里注册若干全局热键并派发回调。"""

    def __init__(self, bindings: list[tuple[str, object]], daemon: bool = True):
        super().__init__(name="hotkeys", daemon=daemon)
        self.bindings = list(bindings)
        self.failed: list[str] = []          # 注册失败的键名（多为已被占用）
        self.ok: list[str] = []
        self._map: dict[int, tuple[str, object]] = {}
        self._ready = threading.Event()

    # -- 生命周期 ------------------------------------------------------------ #
    def run(self) -> None:
        hid = 0
        for name, cb in self.bindings:
            vk = VK.get(name.upper())
            if vk is None:
                self.failed.append(name)
                continue
            hid += 1
            if user32.RegisterHotKey(None, hid, MOD_NOREPEAT, vk):
                self._map[hid] = (name, cb)
                self.ok.append(name)
            else:
                self.failed.append(name)
        self._ready.set()

        msg = _MSG()
        while True:
            got = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if got <= 0:                     # 0 = WM_QUIT，-1 = 出错
                break
            if msg.message == WM_HOTKEY:
                ent = self._map.get(int(msg.wParam))
                if ent is not None:
                    try:
                        ent[1]()
                    except Exception:        # 回调异常不能打断消息循环
                        pass
        for hid in self._map:
            user32.UnregisterHotKey(None, hid)

    def wait_ready(self, timeout: float = 2.0) -> bool:
        return self._ready.wait(timeout)

    def stop(self) -> None:
        if self.ident:
            user32.PostThreadMessageW(self.ident, WM_QUIT, 0, 0)
