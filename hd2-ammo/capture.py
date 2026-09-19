# -*- coding: utf-8 -*-
"""HD2 弹药识别 · 截图与配置公共模块（mss 版）。"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"


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
