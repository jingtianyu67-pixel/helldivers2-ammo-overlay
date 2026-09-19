# -*- coding: utf-8 -*-
"""准星圆环 → 绿色大十字覆盖层（独立脚本）。

屏幕正中截 300x300，识别 HD2 准星**下方那个白色空心圆环**（着弹点指示），
在环心覆盖一个绿色大十字。左右横线和中心小方块一概不参与定位 —— 它们是
固定在屏幕中心的另一套 UI，环才是跟着弹着点走的那个。

用法::

    python crosshair.py                    # 常驻运行，50 Hz，F2 开关
    python crosshair.py --hz 30            # 改循环频率
    python crosshair.py --debug            # 控制台打印分数/坐标
    python crosshair.py --replay _ch/s300  # 离线跑帧目录，出对照图（不开窗）
    python crosshair.py --no-hotkey        # 不注册热键

识别思路（纯 numpy，不依赖 opencv）:
  1. 亮度图减去大核均值 = 局部提亮量 W（准星是半透明白，比周围背景亮 30~45）。
  2. W > tau 记为「白点」，得到命中图 B。
  3. 在环候选半径上数 B 的命中率，减去盘内命中率与远邻域命中率：
     真环 = 一圈命中率高、内部空、外围安静。
  4. 取分数峰值的圆心，映射回屏幕坐标。
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
DEFAULTS = dict(
    region=300,                    # 截取（也是覆盖窗口）的边长，屏幕正中
    ref_h=1440,                    # 半径等尺寸的参考屏幕高度，其它分辨率按比例缩放
    blur=11,                       # 局部背景核，必须远小于环直径才能保住环的对比
    gap=3,                         # 环带与内外基线的间距（像素）
    search=14,                     # 圆环心相对屏幕正中的允许漂移半径（实测 ≤10）
    radii="8,9,10,11,12",          # 环中径候选（参考分辨率下像素，实测 9.5~10）
    block_th=10.0,                 # 中心方块门控：低于此值认为准星没显示（桌面实测 1.1，游戏帧 27~63）
    show_th=0.6,                   # 显示阈值（已按局部起伏归一化；准星显不显示由 block_th 门控负责）
    hide_th=0.4,                   # 隐藏阈值（滞回，低于此值才收起）
    ema_pos=0.55,                  # 位置平滑系数
    ema_score=0.45,                # 分数平滑系数
    cross_arm=34,                  # 十字半臂长
    cross_width=3,                 # 十字线宽
    cross_rgb="0,255,0",           # 十字颜色
    hz=50,
    hotkey="f2",
)

FLOAT_KEYS = ("block_th", "show_th", "hide_th", "ema_pos", "ema_score")
INT_KEYS = ("region", "blur", "gap", "search", "cross_arm", "cross_width", "hz")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def set_dpi_aware() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def boxblur(x: np.ndarray, k: int) -> np.ndarray:
    """k x k 均值滤波（积分图，边界按 edge 截断）。"""
    h, w = x.shape
    p = np.pad(x, k // 2, mode="edge")
    c = p.cumsum(0, dtype=np.float32).cumsum(1, dtype=np.float32)
    c = np.pad(c, ((1, 0), (1, 0)))
    out = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) * (1.0 / (k * k))
    return out[:h, :w]


def erase_cross(lum: np.ndarray, cx: float, cy: float, arm: int, w: int) -> np.ndarray:
    """把上一帧自己画出去的十字从亮度图里抹掉。

    覆盖层是置顶分层窗口，mss 抓屏会连它一起抓到（实测 300x300 里有 397 个纯绿像素）。
    绿色 (0,255,0) 的亮度是 150，会把局部背景均值拉低，于是在十字自己周围造出一圈
    假亮斑，检测器就锁死在十字当前的位置上不动 —— 这是「离线看着好、上机全废」的根因。
    做法：横条用正下方 d 像素处的行顶替，竖条用正右方 d 像素处的列顶替（源取已修好的图，
    避免把竖条的像素搬进横条）。d 取 2w+3，保证源行/列不落在十字自己的线宽+抗锯齿边上。
    """
    h, wd = lum.shape
    cx, cy = int(round(cx)), int(round(cy))
    d = 2 * w + 3
    out = lum.copy()

    y0, y1 = max(0, cy - w), min(h, cy + w + 1)
    x0, x1 = max(0, cx - arm), min(wd, cx + arm + 1)
    if y0 < y1 and x0 < x1 and y1 + d <= h:
        out[y0:y1, x0:x1] = lum[y0 + d:y1 + d, x0:x1]

    vy0, vy1 = max(0, cy - arm), min(h, cy + arm + 1)
    vx0, vx1 = max(0, cx - w), min(wd, cx + w + 1)
    if vy0 < vy1 and vx0 < vx1 and vx1 + d <= wd:
        out[vy0:vy1, vx0:vx1] = out[vy0:vy1, vx0 + d:vx1 + d]
    return out


_OFFS: dict = {}


def ring_offsets(R: float, width: float, ds: int) -> np.ndarray:
    """环带上的整数偏移集合（已按 ds 缩放）。"""
    key = (round(R, 2), round(width, 2), ds)
    if key in _OFFS:
        return _OFFS[key]
    r = R / ds
    wd = max(1.0, width / ds)
    pts = set()
    # 角向采样密度：每 1/r 弧度取一点（≈每像素一点）。1.6 太密，实测 1.0 结果无差别但快 40%。
    n = max(16, int(2 * np.pi * r * 1.0))
    steps = np.arange(-wd / 2, wd / 2 + 1e-6, 0.5 if wd > 1.0 else 0.34)
    for i in range(n):
        th = 2 * np.pi * i / n
        for dr in steps:
            pts.add((int(round((r + dr) * np.sin(th))),
                     int(round((r + dr) * np.cos(th)))))
    arr = np.array(sorted(pts))
    _OFFS[key] = arr
    return arr


def shift_add(acc: np.ndarray, src: np.ndarray, dy: int, dx: int) -> None:
    """acc += src 平移 (dy, dx)，越界部分丢弃（等价零填充）。"""
    h, w = src.shape
    acc[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)] += \
        src[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]


def ds_mean(x: np.ndarray, ds: int) -> np.ndarray:
    """块平均降采样（ds 倍）。"""
    h, w = x.shape
    h2, w2 = h // ds * ds, w // ds * ds
    return x[:h2, :w2].reshape(h2 // ds, ds, w2 // ds, ds).mean((1, 3))


# --------------------------------------------------------------------------- #
# 检测
# --------------------------------------------------------------------------- #
class RingFinder:
    """在 300x300 的中心区域里找准星圆环。

    判据 = 环带均值 − 内外基线（匹配滤波），不用绝对阈值：

        W = 亮度 − 局部均值（小核 11px）
        分数 = 环带(r=R)均值 − 0.5×环带(R−gap) − 0.5×环带(R+gap)
        归一化：再除以 W 的局部起伏尺度，使阈值不随场景明暗漂移

    为什么必须换成这个：
      · 旧判据是「提亮量 > tau(20) 的命中率」。环在亮雾场景里的提亮只有 12~18，
        被阈值整条滤掉，检测器转而去数中心方块的光晕 → 上机全废。
      · 旧背景核 31px 把半径 10 的环算进了背景里，环自我抵消，提亮再掉一截。
        换成小核 11px 后环的局部对比能到 15~30。
      · 匹配滤波对「亮团」不敏感：实心亮块的环带与内外基线同时高，相减抵消；
        只有「一圈亮、内外都低」的细环才得分。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.scale = float(cfg.get("scale", 1.0))
        self.region = int(cfg["region"])
        self.blur = max(5, int(round(int(cfg["blur"]) * self.scale)) | 1)
        self.block_th = float(cfg["block_th"])
        self.gap = max(2, int(round(float(cfg["gap"]) * self.scale)))
        self.search = int(cfg["search"])
        self.radii = [float(R) for R in cfg["radii"]]
        self.ring_width = 1.0                      # 环带取样厚度（像素），固定 1px 细环
        self.c = self.region // 2
        self.s = self.search
        self.arm = int(cfg["cross_arm"])
        self.cross_w = max(1, int(cfg["cross_width"]) // 2 + 1)
        # 要算的环带半径 = 目标半径 ∪ 各自的内外基线
        self.need = sorted({int(round(R)) for R in self.radii}
                           | {int(round(R)) - self.gap for R in self.radii}
                           | {int(round(R)) + self.gap for R in self.radii})
        self.need = [R for R in self.need if R >= 2]
        self.pad = max(self.need) + 3
        self._offs = {R: ring_offsets(R, self.ring_width, 1) for R in self.need}
        self.last_ms = 0.0
        self.last_raw = 0.0
        self.last_block = 0.0

    # -- 主入口 -------------------------------------------------------------- #
    def detect(self, bgra: np.ndarray, erase=None):
        """bgra: (region, region, 4) uint8（mss 原生 BGRA）。

        erase: 上一帧画十字的中心 (cx, cy)（窗口坐标）；给了就先把那处抹掉再检测。
        返回 (score, cx, cy, radius)；score 已按局部起伏归一化，cx/cy 是窗口内坐标。
        """
        t0 = time.perf_counter()
        f = bgra[:, :, :3].astype(np.float32)
        lum = 0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]
        if erase is not None:
            lum = erase_cross(lum, erase[0], erase[1], self.arm, self.cross_w)
        W = lum - boxblur(lum, self.blur)

        c, pad = self.c, self.pad
        lo = max(0, c - self.s - pad)
        hi = min(self.region, c + self.s + pad + 1)
        sub = W[lo:hi, lo:hi]
        v0 = c - self.s - lo
        v1 = c + self.s - lo + 1

        bands = {}
        for R in self.need:
            offs = self._offs[R]
            acc = np.zeros(sub.shape, np.float32)
            for dy, dx in offs:
                shift_add(acc, sub, dy, dx)
            bands[R] = acc * (1.0 / len(offs))

        # 本底起伏尺度（W 的局部标准差），用于归一化阈值
        w2 = boxblur(W * W, 2 * self.blur + 1)
        std = np.sqrt(np.maximum(w2, 1.0))[lo:hi, lo:hi]
        # 中心方块信号（准星是否显示），只做参考输出
        blk = boxblur(W, 7)
        self.last_block = float(blk[c - 4:c + 5, c - 4:c + 5].max())

        mask = np.full(sub.shape, -1e9, np.float32)
        mask[v0:v1, v0:v1] = 0.0
        best = (-1e9, 0, 0, 0.0, 0.0)
        g = self.gap
        for R in self.radii:
            r = int(round(R))
            sc = bands[r] - 0.5 * bands[r - g] - 0.5 * bands[r + g] + mask
            k = int(np.argmax(sc))
            iy, ix = divmod(k, sc.shape[1])
            if sc[iy, ix] > best[0]:
                best = (float(sc[iy, ix]), ix, iy, float(R), float(std[iy, ix]))

        raw, ix, iy, R, s_ = best
        self.last_raw = raw
        cx = lo + ix + 0.5
        cy = lo + iy + 0.5
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        # 门控：中心方块不在 = 准星没显示（收枪/冲刺），环的响应再高也是背景
        if self.last_block < self.block_th:
            return -1.0, cx, cy, R
        return raw / s_, cx, cy, R


# --------------------------------------------------------------------------- #
# 覆盖窗口
# --------------------------------------------------------------------------- #
user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
SW_SHOWNOACTIVATE = 4
SW_HIDE = 0
ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01
BI_RGB = 0
DIB_RGB_COLORS = 0
HWND_TOPMOST = -1
SWP_NOMOVE = 0x0001
SWP_NOSIZE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SM_CXSCREEN, SM_CYSCREEN = 0, 1
SS = 3


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


def build_cross(arm: int, width: int, rgb) -> np.ndarray:
    """预生成十字的预乘 BGRA 小块。"""
    size = 2 * arm + 1
    img = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = arm * SS
    w2 = max(1, width * SS // 2)
    d.rectangle((c - arm * SS, c - w2, c + arm * SS, c + w2), fill=(rgb[0], rgb[1], rgb[2], 255))
    d.rectangle((c - w2, c - arm * SS, c + w2, c + arm * SS), fill=(rgb[0], rgb[1], rgb[2], 255))
    img = img.resize((size, size), Image.LANCZOS)
    a = np.array(img, dtype=np.uint8)
    al = a[:, :, 3:4].astype(np.uint16)
    pm = (a[:, :, :3].astype(np.uint16) * al // 255).astype(np.uint8)
    out = np.empty((size, size, 4), np.uint8)
    out[:, :, 0] = pm[:, :, 2]
    out[:, :, 1] = pm[:, :, 1]
    out[:, :, 2] = pm[:, :, 0]
    out[:, :, 3] = a[:, :, 3]
    return np.ascontiguousarray(out)


class CrossOverlay:
    """固定 300x300、点击穿透、置顶的分层窗口，只在环心画十字。"""

    def __init__(self, cfg: dict, screen: tuple[int, int]):
        self.size = int(cfg["region"])
        self.arm = int(cfg["cross_arm"])
        self.cross = build_cross(self.arm, int(cfg["cross_width"]), cfg["cross_rgb"])
        sx, sy = screen
        self.x0 = (sx - self.size) // 2
        self.y0 = (sy - self.size) // 2
        self.buf = np.zeros((self.size, self.size, 4), np.uint8)
        self.topmost_ms = 500
        self.hwnd = None
        self._visible = False
        self._top_at = 0.0
        self._memdc = None
        self._bits_addr = None
        self.blits = 0

    def _create(self) -> None:
        ex = (WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
              | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
        self.hwnd = user32.CreateWindowExW(
            ex, "STATIC", "hd2cross", WS_POPUP, self.x0, self.y0,
            self.size, self.size, None, None, None, None)
        if not self.hwnd:
            raise ctypes.WinError(ctypes.get_last_error())

    def _blit(self) -> None:
        hdc_screen = user32.GetDC(None)
        if self._memdc is None:
            self._memdc = gdi32.CreateCompatibleDC(hdc_screen)
            bmi = BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = self.size
            bmi.bmiHeader.biHeight = -self.size
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = BI_RGB
            bits = ctypes.c_void_p()
            self._dib = gdi32.CreateDIBSection(
                hdc_screen, ctypes.byref(bmi), DIB_RGB_COLORS,
                ctypes.byref(bits), None, 0)
            gdi32.SelectObject(self._memdc, self._dib)
            self._bits_addr = bits.value
        ctypes.memmove(self._bits_addr, self.buf.tobytes(), self.buf.nbytes)
        pt_dst = wintypes.POINT(self.x0, self.y0)
        size = wintypes.SIZE(self.size, self.size)
        pt_src = wintypes.POINT(0, 0)
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        ok = user32.UpdateLayeredWindow(
            self.hwnd, hdc_screen, ctypes.byref(pt_dst), ctypes.byref(size),
            self._memdc, ctypes.byref(pt_src), 0, ctypes.byref(blend), ULW_ALPHA)
        user32.ReleaseDC(None, hdc_screen)
        self.blits += 1
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    def draw(self, cx: float, cy: float) -> None:
        if self.hwnd is None:
            self._create()
        self.buf[:] = 0
        arm = self.arm
        x = int(round(cx)) - arm
        y = int(round(cy)) - arm
        n = 2 * arm + 1
        self.buf[y:y + n, x:x + n] = self.cross
        if not self._visible:
            user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
            self._visible = True
            self._top_at = 0.0
        self._reassert_top()
        self._blit()

    def hide(self) -> None:
        if self.hwnd is not None and self._visible:
            user32.ShowWindow(self.hwnd, SW_HIDE)
            self._visible = False

    def _reassert_top(self) -> None:
        now = time.perf_counter()
        if now - self._top_at < self.topmost_ms / 1000.0:
            return
        self._top_at = now
        user32.SetWindowPos(self.hwnd, wintypes.HWND(HWND_TOPMOST), 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)

    def close(self) -> None:
        if self._memdc:
            gdi32.DeleteDC(self._memdc)
            self._memdc = None
        if getattr(self, "_dib", None):
            gdi32.DeleteObject(self._dib)
            self._dib = None
        if self.hwnd:
            user32.DestroyWindow(self.hwnd)
            self.hwnd = None


# --------------------------------------------------------------------------- #
# 热键（单键开关，独立线程）
# --------------------------------------------------------------------------- #
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000
VK = {"f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74,
      "f6": 0x75, "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79,
      "f11": 0x7A, "f12": 0x7B}


class _MSG(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND), ("message", wintypes.UINT),
                ("wParam", wintypes.WPARAM), ("lParam", wintypes.LPARAM),
                ("time", wintypes.DWORD), ("pt_x", wintypes.LONG),
                ("pt_y", wintypes.LONG)]


class HotkeyThread(threading.Thread):
    """注册一个功能键做开关。首选键被占用时自动退到备用键。"""

    def __init__(self, name: str, on_fire, fallbacks=("f3", "f4", "f6", "f8"),
                 daemon=True):
        super().__init__(daemon=daemon, name="hotkey")
        self.candidates = [name.lower()] + [k for k in fallbacks if k != name.lower()]
        self.on_fire = on_fire
        self.ready = threading.Event()
        self.ok = False
        self.ok_key = None

    def run(self) -> None:
        user32.RegisterHotKey.restype = wintypes.BOOL
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int,
                                          wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.argtypes = [ctypes.POINTER(_MSG), wintypes.HWND,
                                       wintypes.UINT, wintypes.UINT]
        for key in self.candidates:
            if key in VK and user32.RegisterHotKey(None, 1, MOD_NOREPEAT, VK[key]):
                self.ok_key = key
                break
        self.ok = self.ok_key is not None
        self.ready.set()
        if not self.ok:
            return
        msg = _MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                try:
                    self.on_fire()
                except Exception:
                    pass
        user32.UnregisterHotKey(None, 1)


# --------------------------------------------------------------------------- #
# 运行时
# --------------------------------------------------------------------------- #
class Session:
    """检测 + 平滑 + 覆盖层的运行时状态机。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.finder = RingFinder(cfg)
        self.ema_x = None
        self.ema_y = None
        self.ema_s = 0.0
        self.armed = False          # 是否已进入「显示」状态（滞回）
        self.frames = 0
        self.hits = 0
        self.drawn = None           # 上一帧实际画在屏幕上的十字中心（窗口坐标）
        self.erase = True

    def step(self, bgra: np.ndarray):
        """返回 (show, cx, cy, score_raw, score_ema)。"""
        score, cx, cy, R = self.finder.detect(
            bgra, erase=self.drawn if self.erase else None)
        self.frames += 1
        a_s = float(self.cfg["ema_score"])
        self.ema_s = a_s * score + (1 - a_s) * self.ema_s
        if self.ema_x is None or score >= float(self.cfg["show_th"]):
            a_p = float(self.cfg["ema_pos"])
            if self.ema_x is None:
                self.ema_x, self.ema_y = cx, cy
            else:
                self.ema_x = a_p * cx + (1 - a_p) * self.ema_x
                self.ema_y = a_p * cy + (1 - a_p) * self.ema_y
        # 滞回
        if not self.armed:
            if self.ema_s >= float(self.cfg["show_th"]) and self.ema_x is not None:
                self.armed = True
        else:
            if self.ema_s < float(self.cfg["hide_th"]):
                self.armed = False
        if self.armed and self.ema_s >= float(self.cfg["show_th"]):
            self.hits += 1
            self.drawn = (self.ema_x, self.ema_y)
            return True, self.ema_x, self.ema_y, score, self.ema_s
        self.drawn = None
        return False, self.ema_x or 0.0, self.ema_y or 0.0, score, self.ema_s


def screen_size() -> tuple[int, int]:
    return (user32.GetSystemMetrics(SM_CXSCREEN), user32.GetSystemMetrics(SM_CYSCREEN))


def run_live(cfg: dict, args) -> int:
    import mss
    set_dpi_aware()
    sw, sh = screen_size()
    # 环尺寸随分辨率缩放：参考高度 ref_h 下的标定值
    scale = sh / float(cfg["ref_h"])
    cfg = dict(cfg)
    cfg["scale"] = scale
    if abs(scale - 1.0) > 0.02:
        cfg["radii"] = tuple(round(R * scale, 2) for R in cfg["radii"])
        cfg["blur"] = max(5, int(round(cfg["blur"] * scale)) | 1)
        cfg["gap"] = max(2, int(round(cfg["gap"] * scale)))
        cfg["search"] = int(round(cfg["search"] * scale))
        cfg["cross_arm"] = int(round(cfg["cross_arm"] * scale))
        cfg["cross_width"] = max(1, int(round(cfg["cross_width"] * scale)))
    c = cfg["region"] // 2
    region = {"left": sw // 2 - c, "top": sh // 2 - c,
              "width": cfg["region"], "height": cfg["region"]}
    print(f"[准星十字] 屏幕 {sw}x{sh}  截取 {cfg['region']}x{cfg['region']} "
          f"@ ({region['left']},{region['top']})")
    if abs(scale - 1.0) > 0.02:
        print(f"[准星十字] 分辨率 ≠ {cfg['ref_h']}p，尺寸参数 ×{scale:.3f} → "
              f"半径 {cfg['radii']}  漂移 {cfg['search']}")
    print(f"[准星十字] 显示阈值 {cfg['show_th']} / 隐藏 {cfg['hide_th']}  "
          f"{cfg['hz']} Hz  热键 {cfg['hotkey'].upper()}")

    session = Session(cfg)
    overlay = CrossOverlay(cfg, (sw, sh))
    enabled = True

    def toggle() -> None:
        nonlocal enabled
        enabled = not enabled
        if not enabled:
            overlay.hide()
        print(f"[热键] {'开启' if enabled else '关闭'}", flush=True)

    if not args.no_hotkey:
        hk = HotkeyThread(cfg["hotkey"], toggle)
        hk.start()
        hk.ready.wait(2.0)
        if hk.ok:
            if hk.ok_key != cfg["hotkey"].lower():
                print(f"[热键] {cfg['hotkey'].upper()} 被占用，改用 {hk.ok_key.upper()}")
        else:
            print(f"[热键] 候选键全部注册失败，只能靠关进程退出")

    period = 1.0 / float(cfg["hz"])
    nxt = time.perf_counter()
    t0 = time.perf_counter()
    last_report = t0
    stop_at = t0 + float(args.duration) if args.duration else None
    try:
        with mss.mss() as sct:
            while True:
                nxt += period
                if enabled:
                    shot = sct.grab(region)
                    bgra = np.asarray(shot, dtype=np.uint8)
                    show, cx, cy, sc, sce = session.step(bgra)
                    if show:
                        overlay.draw(cx, cy)
                    else:
                        overlay.hide()
                now = time.perf_counter()
                if now - last_report >= 5.0:
                    fps = session.frames / (now - t0)
                    print(f"[5s] {fps:5.1f} fps  检出率 {session.hits}/{session.frames}  "
                          f"检测 {session.finder.last_ms:.2f} ms  blit {overlay.blits}",
                          flush=True)
                    last_report = now
                if stop_at is not None and now >= stop_at:
                    break
                d = nxt - time.perf_counter()
                if d > 0:
                    time.sleep(d)
                elif d < -period * 3:
                    nxt = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        overlay.close()
    if session.frames:
        fps = session.frames / (time.perf_counter() - t0)
        print(f"[结束] {session.frames} 帧  {fps:.1f} fps  显示 {session.hits} 帧  "
              f"检测 {session.finder.last_ms:.2f} ms  上屏 {overlay.blits} 次")
    return 0


def paint_cross(a: np.ndarray, cx: float, cy: float, cfg: dict) -> np.ndarray:
    """把十字按覆盖层的实际画法合成进 RGB 帧 —— 模拟 mss 抓到自己画的十字。"""
    arm = int(cfg["cross_arm"])
    n = 2 * arm + 1
    buf = build_cross(arm, int(cfg["cross_width"]), cfg["cross_rgb"])
    x, y = int(round(cx)) - arm, int(round(cy)) - arm
    h, w = a.shape[:2]
    if x < 0 or y < 0 or x + n > w or y + n > h:
        return a
    out = a.copy()
    sub = out[y:y + n, x:x + n].astype(np.float32)
    al = buf[:, :, 3:4].astype(np.float32) / 255.0
    src = buf[:, :, [2, 1, 0]].astype(np.float32)      # 预乘 BGRA → 预乘 RGB
    out[y:y + n, x:x + n] = np.clip(src + sub * (1.0 - al), 0, 255).astype(np.uint8)
    return out


def run_replay(cfg: dict, args) -> int:
    """离线跑一个帧目录，输出统计与对照图。

    加 --sim-cross 会按实机方式闭环：把上一帧画出去的十字合成进当前帧再检测。
    不开这个开关的离线结果与实机不等价（实测离线好看、上机全废就是这个原因）。
    """
    src = Path(args.replay)
    files = sorted([p for p in src.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".bmp")])
    if not files:
        print(f"[回放] {src} 里没有图片")
        return 1
    session = Session(cfg)
    session.erase = bool(getattr(args, "sim_cross", False)) and not args.no_erase
    rows = []
    painted = None
    for p in files:
        a = np.asarray(Image.open(p).convert("RGB"))
        h, w = a.shape[:2]
        if (h, w) != (cfg["region"], cfg["region"]):
            print(f"[回放] {p.name} 尺寸 {w}x{h} != {cfg['region']}，跳过")
            continue
        if args.sim_cross and painted is not None:
            a = paint_cross(a, painted[0], painted[1], cfg)
        bgra = np.empty((h, w, 4), np.uint8)
        bgra[:, :, 0] = a[:, :, 2]
        bgra[:, :, 1] = a[:, :, 1]
        bgra[:, :, 2] = a[:, :, 0]
        bgra[:, :, 3] = 255
        show, cx, cy, sc, sce = session.step(bgra)
        painted = session.drawn
        rows.append((p.name, show, cx, cy, sc, sce))

    sc = np.array([r[4] for r in rows])
    shown = sum(1 for r in rows if r[1])
    print(f"[回放] {len(rows)} 帧  显示 {shown} 帧  分数 中位 {np.median(sc):.3f} "
          f"p10 {np.percentile(sc, 10):.3f} max {sc.max():.3f}")
    print(f"[回放] 检测耗时中位 {session.finder.last_ms:.2f} ms/帧")

    # 对照图：每若干帧取一帧，画检测位置
    step = max(1, len(rows) // 24)
    tiles = []
    for i in range(0, len(rows), step):
        name, show, cx, cy, s_, se = rows[i]
        im = Image.open(src / name).convert("RGB")
        d = ImageDraw.Draw(im)
        arm = cfg["cross_arm"]
        col = (0, 255, 0) if show else (150, 150, 150)
        d.line((cx - arm, cy, cx + arm, cy), fill=col, width=1)
        d.line((cx, cy - arm, cx, cy + arm), fill=col, width=1)
        d.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), outline=(0, 200, 255), width=1)
        d.text((4, 4), f"{name[:3]} {s_:.2f}", fill=col)
        tiles.append(im)
    cols = 5
    rn = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cfg["region"] * cols, cfg["region"] * rn), (10, 10, 12))
    for k, t in enumerate(tiles):
        sheet.paste(t, (cfg["region"] * (k % cols), cfg["region"] * (k // cols)))
    out = src.parent / f"crosshair_{src.name}.png"
    sheet.save(out)
    print(f"[回放] 对照图 -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="HD2 准星圆环 → 绿色十字覆盖层")
    for k, v in DEFAULTS.items():
        flag = "--" + k.replace("_", "-")
        ap.add_argument(flag, type=type(v), default=v)
    ap.add_argument("--replay", metavar="DIR", help="离线跑帧目录")
    ap.add_argument("--sim-cross", action="store_true",
                    help="回放时闭环：把上一帧画出的十字合成进当前帧（模拟抓到自己）")
    ap.add_argument("--no-erase", action="store_true",
                    help="关掉「先抹掉自己画的十字」（用于 A/B 对比）")
    ap.add_argument("--duration", type=float, default=0, help="跑 N 秒后自动退出（0=常驻）")
    ap.add_argument("--no-hotkey", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    cfg = {k: getattr(args, k) for k in DEFAULTS}
    cfg["radii"] = tuple(float(x) for x in str(cfg["radii"]).replace("(", "").replace(")", "").split(","))
    cfg["cross_rgb"] = tuple(int(x) for x in str(cfg["cross_rgb"]).replace("(", "").replace(")", "").split(","))
    for k in FLOAT_KEYS:
        cfg[k] = float(cfg[k])
    for k in INT_KEYS:
        cfg[k] = int(cfg[k])
    if args.replay:
        return run_replay(cfg, args)
    return run_live(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
