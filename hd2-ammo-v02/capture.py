# -*- coding: utf-8 -*-
"""HD2 弹药识别 · 截图与配置公共模块（mss 版）。"""

from __future__ import annotations

import ctypes
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def app_dir() -> Path:
    """可写的工作目录：打包成 exe 后是 exe 所在目录，源码运行时是本文件所在目录。

    打包后不能再用 `__file__` —— onefile 模式下它指向临时解包目录（随进程消失，
    写进去的日志/改过的 config 全丢）。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def res_dir() -> Path:
    """只读资源目录：打包后是解包目录（PyInstaller 的 _MEIPASS），源码下同 app_dir()。

    assets/*.npy 这类只读资源从这里读；若 exe 旁边另放了一份同名目录，优先用旁边的
    （方便手工替换掩膜/模板而不用重新打包）。
    """
    if getattr(sys, "frozen", False):
        out = app_dir()
        if (out / "assets").is_dir():
            return out
        return Path(getattr(sys, "_MEIPASS", out))
    return Path(__file__).resolve().parent


BASE_DIR = app_dir()
CONFIG_PATH = BASE_DIR / "config.json"


def _bootstrap_config() -> None:
    """打包后首次运行：把内置的 config.json 释放到 exe 旁边。

    config 必须放在 exe 旁边而不是打进包里 —— 调参器/手工编辑都要能改它，
    写回解包目录等于每次运行都被覆盖。
    """
    if CONFIG_PATH.exists():
        return
    src = res_dir() / "config.json"
    try:
        if src.is_file() and src.resolve() != CONFIG_PATH.resolve():
            shutil.copyfile(src, CONFIG_PATH)
    except OSError:
        pass


_bootstrap_config()


def set_dpi_aware() -> None:
    """让屏幕坐标 == 物理像素。必须在创建任何窗口 / 截图前调用。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def screen_size() -> tuple[int, int]:
    """主显示器物理分辨率。"""
    u32 = ctypes.windll.user32
    return u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)


def _scale_region(region, ref, cur):
    if not ref or tuple(cur) == tuple(ref):
        return tuple(int(round(v)) for v in region)
    sx, sy = cur[0] / ref[0], cur[1] / ref[1]
    return tuple(int(round(v)) for v in
                 (region[0] * sx, region[1] * sy, region[2] * sx, region[3] * sy))


def resolve_regions(cfg: dict | None = None, verbose: bool = True
                    ) -> list[tuple[str, tuple[int, int, int, int]]]:
    """按优先级返回 [(name, (l, t, r, b)), ...]，已换算到当前屏幕物理像素。

    分辨率与 calibrated_for 不一致时按比例缩放，换显示器/换分辨率仍可用。
    """
    cfg = cfg or load_config()
    ref = cfg.get("calibrated_for")
    raw = cfg.get("regions")
    if raw is None:                     # 兼容旧版单区域配置
        raw = [{"name": "main", "region": cfg["region"]}]
    cur = screen_size() if ref else None
    out = []
    for item in raw:
        box = _scale_region(list(item["region"]), ref, cur)
        out.append((item.get("name", "main"), box))
    if verbose and ref and cur and tuple(cur) != tuple(ref):
        print(f"[配置] 分辨率 {ref[0]}x{ref[1]} → {cur[0]}x{cur[1]}，区域已按比例缩放")
    return out


def resolve_layout(cfg: dict | None = None, verbose: bool = True):
    """返回 (probe_rect, (dx, dy))，都已按当前屏幕物理像素换算；无 layout 段则 None。

    probe_rect 是屏幕固定位置的判据矩形，区域平移与分辨率缩放都要跟着走。
    """
    cfg = cfg or load_config()
    lay = cfg.get("layout")
    if not lay or not lay.get("enabled", True):
        return None
    ref = cfg.get("calibrated_for")
    cur = screen_size() if ref else None
    rect = _scale_region(list(lay["probe_rect"]), ref, cur)
    sh = list(lay.get("shift", (-106, 0)))
    if ref and cur and tuple(cur) != tuple(ref):
        sx, sy = cur[0] / ref[0], cur[1] / ref[1]
        sh = [sh[0] * sx, sh[1] * sy]
        if verbose:
            print(f"[配置] 布局判据与位移已随分辨率缩放：shift -> "
                  f"({sh[0]:.0f},{sh[1]:.0f})")
    return rect, (int(round(sh[0])), int(round(sh[1])))


def resolve_region(cfg: dict | None = None, verbose: bool = True) -> tuple[int, int, int, int]:
    """取优先级最高的那个区域（向后兼容）。"""
    return resolve_regions(cfg, verbose)[0][1]


def as_monitor(region: tuple[int, int, int, int]) -> dict:
    """(l, t, r, b) -> mss 需要的 dict。"""
    left, top, right, bottom = region
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def grab_rgb(sct, region: tuple[int, int, int, int]) -> np.ndarray:
    """截取区域，返回 (h, w, 3) RGB uint8 数组。"""
    shot = np.array(sct.grab(as_monitor(region)))
    return np.ascontiguousarray(shot[:, :, 2::-1])  # BGRA -> RGB，连续内存便于 PIL/numpy


def desktop_dir() -> Path:
    """当前用户桌面（兼容 OneDrive 重定向）。"""
    FOLDERID_Desktop = "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}"

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort),
                    ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]

    ole32 = ctypes.windll.ole32
    shell32 = ctypes.windll.shell32
    guid = GUID()
    ole32.CLSIDFromString(ctypes.c_wchar_p(FOLDERID_Desktop), ctypes.byref(guid))
    ptr = ctypes.c_wchar_p()
    if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(ptr)) != 0:
        raise OSError("无法解析桌面路径")
    try:
        return Path(ptr.value)
    finally:
        ole32.CoTaskMemFree(ctypes.cast(ptr, ctypes.c_void_p))
