# -*- coding: utf-8 -*-
"""弹药余量覆盖层：透明、置顶、点击穿透的分层窗口。

只在准星旁画一条弧线——**白色实心部分 = 剩余弹药量**，灰色是空槽。
低于阈值时填充转红。

实现要点：
  - WS_EX_LAYERED + UpdateLayeredWindow(ULW_ALPHA)，逐像素 alpha，
    不用 SetLayeredWindowAttributes 那种整窗透明。
  - WS_EX_TRANSPARENT 让鼠标事件穿透到游戏，WS_EX_NOACTIVATE 不抢焦点。
  - 窗口尺寸只覆盖弧线包围盒，不是整屏，减少合成开销。
  - 只有余量变化超过 1% 才重绘，静止时零开销。

依赖 config.json 的 "overlay" 段，见 README 或 config 里的注释。
"""

from __future__ import annotations

import ctypes
import math
import time
from ctypes import wintypes

import numpy as np
from PIL import Image, ImageDraw

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
SW_SHOWNOACTIVATE = 4
ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01
BI_RGB = 0
DIB_RGB_COLORS = 0
SM_CXSCREEN, SM_CYSCREEN = 0, 1
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

SS = 3          # 绘制超采样倍数，缩小时得到抗锯齿


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_byte), ("BlendFlags", ctypes.c_byte),
                ("SourceConstantAlpha", ctypes.c_byte), ("AlphaFormat", ctypes.c_byte)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p]
user32.GetDC.restype = wintypes.HDC
user32.GetDC.argtypes = [wintypes.HWND]
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateDIBSection.restype = ctypes.c_void_p
gdi32.CreateDIBSection.argtypes = [
    wintypes.HDC, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, wintypes.DWORD]
gdi32.SelectObject.restype = ctypes.c_void_p
gdi32.SelectObject.argtypes = [wintypes.HDC, ctypes.c_void_p]
gdi32.DeleteObject.restype = wintypes.BOOL
gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
gdi32.DeleteDC.restype = wintypes.BOOL
gdi32.DeleteDC.argtypes = [wintypes.HDC]
user32.UpdateLayeredWindow.argtypes = [
    wintypes.HWND, wintypes.HDC, ctypes.POINTER(wintypes.POINT),
    ctypes.POINTER(wintypes.SIZE), wintypes.HDC, ctypes.POINTER(wintypes.POINT),
    wintypes.DWORD, ctypes.POINTER(BLENDFUNCTION), wintypes.DWORD]
user32.SetWindowPos.restype = wintypes.BOOL
user32.SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, wintypes.UINT]
user32.GetWindowRect.restype = wintypes.BOOL
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]


# --------------------------------------------------------------------------- #
# 弧线几何
# --------------------------------------------------------------------------- #
class ArcGeometry:
    """一条圆弧。角度单位「度」，0° = 正右，顺时针为正（屏幕坐标 y 向下）。

    ang_start 是上端、ang_end 是下端，填充从下端往上游走。
    """

    def __init__(self, screen: tuple[int, int], cfg: dict):
        self.sw, self.sh = screen
        anchor = cfg.get("anchor", "center")
        if anchor == "center":
            ax, ay = self.sw / 2.0, self.sh / 2.0
        else:
            ax, ay = float(cfg.get("anchor_xy", [self.sw / 2, self.sh / 2])[0]), \
                     float(cfg.get("anchor_xy", [self.sw / 2, self.sh / 2])[1])
        ox, oy = cfg.get("offset", [-300, 0])
        self.cx, self.cy = ax + float(ox), ay + float(oy)
        self.r = float(cfg.get("radius", 356))
        self.w = float(cfg.get("width", 22))
        self.a0 = float(cfg.get("ang_start", -59))
        self.a1 = float(cfg.get("ang_end", 48))

    # -- 包围盒（含笔画宽度） -------------------------------------------------- #
    def bbox(self) -> tuple[int, int, int, int]:
        aa = [math.radians(self.a0 + (self.a1 - self.a0) * t / 90.0) for t in range(91)]
        xs = [self.cx + self.r * math.cos(t) for t in aa]
        ys = [self.cy + self.r * math.sin(t) for t in aa]
        p = self.w / 2.0 + 4
        return (int(math.floor(min(xs) - p)), int(math.floor(min(ys) - p)),
                int(math.ceil(max(xs) + p)), int(math.ceil(max(ys) + p)))

    def render(self, fraction: float, fill_rgb, track_rgb, track_alpha: float,
               alpha: float = 1.0) -> Image.Image:
        """画一张 RGBA 图：整条灰轨道 + 从下端起算的白色填充。

        alpha 是整条弧线的总体不透明度，乘在轨道与填充各自的 alpha 之上。
        """
        x0, y0, x1, y1 = self.bbox()
        w, h = x1 - x0, y1 - y0
        img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        box = [(self.cx - x0 - self.r) * SS, (self.cy - y0 - self.r) * SS,
               (self.cx - x0 + self.r) * SS, (self.cy - y0 + self.r) * SS]
        wid = max(1, int(round(self.w * SS)))

        a = max(0.0, min(1.0, float(alpha)))
        ta = int(round(max(0.0, min(1.0, track_alpha)) * a * 255))
        fa = int(round(a * 255))

        d.arc(box, self.a0, self.a1, fill=tuple(track_rgb) + (ta,), width=wid)

        span = self.a1 - self.a0
        f = max(0.0, min(1.0, fraction))
        if f > 0.0:
            if f >= 1.0:
                d.arc(box, self.a0, self.a1, fill=tuple(fill_rgb) + (fa,), width=wid)
            else:
                start = self.a1 - span * f
                d.arc(box, start, self.a1, fill=tuple(fill_rgb) + (fa,), width=wid)
        return img.resize((w, h), Image.LANCZOS)

    def anchor_point(self) -> tuple[int, int]:
        x0, y0, _, _ = self.bbox()
        return x0, y0


# --------------------------------------------------------------------------- #
# 分层窗口
# --------------------------------------------------------------------------- #
class AmmoOverlay:
    """把弹药余量画到屏幕上的透明窗口。无效读数时隐藏。"""

    def __init__(self, cfg: dict, screen: tuple[int, int]):
        o = cfg.get("overlay", {}) or {}
        self.cfg = o
        self.geo = ArcGeometry(screen, o)
        self.track_rgb = o.get("track_rgb", [129, 128, 129])
        self.track_alpha = float(o.get("track_alpha", 0.62))
        self.alpha = float(o.get("alpha", 1.0))
        self.fill_rgb = o.get("fill_rgb", [237, 240, 241])
        self.low_rgb = o.get("low_rgb", [214, 44, 40])
        self.low_threshold = float(o.get("low_threshold", 0.15))
        self.low_full_bar = bool(o.get("low_full_bar", True))
        # 红线也可以直接用「白像素个数」定：>0 时，数到的白像素 <= 这个数就转红。
        # 0 = 关闭，只按 low_threshold 的比例判。
        self.low_white_px = int(o.get("low_white_px", 0))
        self.min_step = float(o.get("redraw_step", 0.01))
        self.hide_on_invalid = bool(o.get("hide_on_invalid", True))
        self.margin = int(o.get("margin", 4))
        # 置顶重夺间隔（毫秒）。WS_EX_TOPMOST 只在创建那一刻生效，游戏窗口
        # 激活后会挤到置顶带之上，必须周期性抢回来。
        self.topmost_ms = int(o.get("topmost_reassert_ms", 500))
        self.on_show = None          # 首次上屏回调，给调用方打日志用

        self._last: float | None = None
        self._last_red: bool | None = None
        self._visible = False
        self._top_at = 0.0
        self.redraws = 0
        self.hwnd = None
        self._memdc = None
        self._dib = None

    # -- 窗口 ---------------------------------------------------------------- #
    def _create(self) -> None:
        x0, y0, x1, y1 = self.geo.bbox()
        ex = (WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
              | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
        self.hwnd = user32.CreateWindowExW(
            ex, "STATIC", "hd2ammo", WS_POPUP, x0, y0, x1 - x0, y1 - y0,
            None, None, None, None)
        if not self.hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self._x0, self._y0, self._w, self._h = x0, y0, x1 - x0, y1 - y0

    def _blit(self, img: Image.Image) -> None:
        """img 必须与窗口同尺寸的 RGBA。"""
        bgra = np.array(img, dtype=np.uint8)               # H,W,4 RGBA
        # 预乘 alpha，Windows 要的是 premultiplied BGRA
        a = bgra[:, :, 3:4].astype(np.uint16)
        rgb = (bgra[:, :, :3].astype(np.uint16) * a // 255).astype(np.uint8)
        buf = np.empty((self._h, self._w, 4), dtype=np.uint8)
        buf[:, :, 0] = rgb[:, :, 2]      # B
        buf[:, :, 1] = rgb[:, :, 1]      # G
        buf[:, :, 2] = rgb[:, :, 0]      # R
        buf[:, :, 3] = bgra[:, :, 3]     # A
        buf = np.ascontiguousarray(buf)

        hdc_screen = user32.GetDC(None)
        if self._memdc is None:
            self._memdc = gdi32.CreateCompatibleDC(hdc_screen)
            bmi = BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = self._w
            bmi.bmiHeader.biHeight = -self._h        # 负 = 自上而下
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = BI_RGB
            bits = ctypes.c_void_p()
            self._dib = gdi32.CreateDIBSection(
                hdc_screen, ctypes.byref(bmi), DIB_RGB_COLORS, ctypes.byref(bits),
                None, 0)
            self._bits = bits
            ctypes.memmove(bits, buf.tobytes(), buf.nbytes)
            gdi32.SelectObject(self._memdc, self._dib)
            self._bits_addr = bits.value

        ctypes.memmove(self._bits_addr, buf.tobytes(), buf.nbytes)

        pt_dst = wintypes.POINT(self._x0, self._y0)
        size = wintypes.SIZE(self._w, self._h)
        pt_src = wintypes.POINT(0, 0)
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        ok = user32.UpdateLayeredWindow(
            self.hwnd, hdc_screen, ctypes.byref(pt_dst), ctypes.byref(size),
            self._memdc, ctypes.byref(pt_src), 0, ctypes.byref(blend), ULW_ALPHA)
        user32.ReleaseDC(None, hdc_screen)
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    # -- 对外 ---------------------------------------------------------------- #
    def show(self, fraction: float, red: bool = False) -> None:
        if self.hwnd is None:
            self._create()
        if not self._visible:
            user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
            self._visible = True
            self._last = None
            self._top_at = 0.0
            if self.on_show is not None:
                self.on_show(self.rect())
        self._reassert_top()
        if (self._last is not None and not red == self._last_red
                and abs(fraction - self._last) < self.min_step):
            return
        if (self._last is not None and self._last_red == red
                and abs(fraction - self._last) < self.min_step):
            return
        fill = self.low_rgb if red else self.fill_rgb
        self._blit(self.geo.render(fraction, fill, self.track_rgb,
                                   self.track_alpha, self.alpha))
        self._last, self._last_red = fraction, red
        self.redraws += 1

    def _reassert_top(self) -> None:
        """按间隔重夺置顶，防止被无边框全屏的游戏窗口压到下面。"""
        if self.hwnd is None or not self._visible:
            return
        now = time.perf_counter()
        if now - self._top_at < self.topmost_ms / 1000.0:
            return
        self._top_at = now
        user32.SetWindowPos(self.hwnd, wintypes.HWND(HWND_TOPMOST), 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)

    def rect(self) -> tuple[int, int, int, int]:
        r = wintypes.RECT()
        if self.hwnd is None or not user32.GetWindowRect(self.hwnd, ctypes.byref(r)):
            return (0, 0, 0, 0)
        return (r.left, r.top, r.right, r.bottom)

    def status(self) -> str:
        if self.hwnd is None:
            return "窗口尚未创建（从未有有效读数）"
        l, t, r, b = self.rect()
        return (f"hwnd=0x{int(self.hwnd):X} 可见={'是' if user32.IsWindowVisible(self.hwnd) else '否'}"
                f" 矩形=({l},{t})-({r},{b}) 重绘={self.redraws}")

    def reconfigure(self, o: dict) -> None:
        """热更新 overlay 配置（供调参器使用）。

        包围盒尺寸没变就只挪窗口位置，避免拖动时反复重建窗口。
        """
        sw, sh = self.geo.sw, self.geo.sh
        old = (self._w, self._h) if self.hwnd is not None else None
        self.cfg = o
        self.geo = ArcGeometry((sw, sh), o)
        self.track_rgb = o.get("track_rgb", [129, 128, 129])
        self.track_alpha = float(o.get("track_alpha", 0.62))
        self.alpha = float(o.get("alpha", 1.0))
        self.fill_rgb = o.get("fill_rgb", [237, 240, 241])
        self.low_rgb = o.get("low_rgb", [214, 44, 40])
        self.low_threshold = float(o.get("low_threshold", 0.15))
        self.low_full_bar = bool(o.get("low_full_bar", True))
        # 红线也可以直接用「白像素个数」定：>0 时，数到的白像素 <= 这个数就转红。
        # 0 = 关闭，只按 low_threshold 的比例判。
        self.low_white_px = int(o.get("low_white_px", 0))
        self.min_step = float(o.get("redraw_step", 0.01))
        self.hide_on_invalid = bool(o.get("hide_on_invalid", True))
        self.topmost_ms = int(o.get("topmost_reassert_ms", 500))
        x0, y0, x1, y1 = self.geo.bbox()
        if self.hwnd is not None:
            if (x1 - x0, y1 - y0) != old:
                self.close()
            else:
                self._x0, self._y0 = x0, y0
        self._last = None

    def hide(self) -> None:
        if self.hwnd is not None and self._visible:
            user32.ShowWindow(self.hwnd, 0)
            self._visible = False

    def close(self) -> None:
        if self.hwnd is not None:
            user32.DestroyWindow(self.hwnd)
            self.hwnd = None
        self._memdc = self._dib = None
        self._visible = False

    # -- 便利入口 ------------------------------------------------------------ #
    def show_value(self, fraction: float, fill: int = 0) -> None:
        """按余量决定画什么：

        - 空弹匣（余量归零）→ 不画实心条，只剩灰色轨迹。
        - 低弹（低于 low_threshold，或白像素数 <= low_white_px）且 low_full_bar
          → 整条弧线泛红（不看具体余量多少）。
        - 其余 → 白色填充条，长度 = 余量。
        """
        if fraction <= 0.0:
            self.show(0.0, red=False)
            return
        red = (fraction < self.low_threshold
               or (self.low_white_px > 0 and fill <= self.low_white_px))
        self.show(1.0 if (red and self.low_full_bar) else fraction, red=red)

    def update(self, reading) -> None:
        """reading 为 None 或无效则隐藏。"""
        if reading is None or not reading.valid:
            if self.hide_on_invalid:
                self.hide()
            return
        self.show_value(reading.fraction, reading.fill)


def screen_size() -> tuple[int, int]:
    return (user32.GetSystemMetrics(SM_CXSCREEN),
            user32.GetSystemMetrics(SM_CYSCREEN))


if __name__ == "__main__":   # 自检：画四种状态存成 PNG
    import json
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "config.json"), encoding="utf-8"))
    g = ArcGeometry((2560, 1440), cfg.get("overlay", {}))
    print("包围盒", g.bbox(), " 圆心", (g.cx, g.cy), " 半径", g.r)
    rows = []
    for f, red in ((1.0, False), (0.5, False), (0.15, True), (0.05, True)):
        im = g.render(f, cfg["overlay"]["low_rgb"] if red else
                      cfg["overlay"]["fill_rgb"],
                      cfg["overlay"]["track_rgb"], cfg["overlay"]["track_alpha"])
        rows.append((f, red, im))
    W = sum(im.size[0] + 12 for _, _, im in rows)
    H = max(im.size[1] for _, _, im in rows)
    M = Image.new("RGBA", (W, H), (28, 28, 32, 255))
    x = 6
    for f, red, im in rows:
        M.alpha_composite(im, (x, (H - im.size[1]) // 2))
        x += im.size[0] + 12
    M.convert("RGB").save("_out/overlay_preview.png")
    print("-> _out/overlay_preview.png", M.size)
