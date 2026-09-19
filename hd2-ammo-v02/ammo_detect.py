# -*- coding: utf-8 -*-
"""弹性弹药图标检测器（HD2 弹匣余量）。

口径
----
**余量 = 弹匣体内的白像素数 ÷ 弹匣体面积**；**红色只回答「现在是不是低弹红」，不参与百分比。**

1. **什么叫白**：`lum > 本行背景 + margin`（背景取区域左右边缘列的中位亮度，逐行自适应）
   且 `chroma < 25`（接近灰）。雪地/夜战/黄沙背景都不会误触发。

2. **弹匣体**：白像素的最大连通域（外框恒白，低弹时轮廓仍在）→ 凸包 → 内缩
   `fill_inset` 像素。内缩量 = 外框线自身粗细，于是那条恒定可见的外框线整条被排除，
   分子里剩下的就是底部实心填充的白像素。

3. **形状校验（排除其它图标）**：凸包 → 等比归一化 32x32 → 与模板算 IoU。
   目标「倾斜长方体」IoU 中位 0.92；方形凹槽等其它图标中位 0.30。不达标即
   `valid=False`，输出「未检测到」，绝不猜。

4. **红色通道（低弹态，2026-09-19 补）**：HD2 在弹匣余量低于阈值时会把整个计量器
   转红（官方 HUD 行为："flashing red ammo gauge"）。此时白色像素整条消失 ——
   只认白的口径会直接判「未检测到」，覆盖层随之隐藏，低弹区间因此**整档缺失**。
   红色通道与白色通道完全对称、各自独立定位，只回答一件事：**这个弹匣现在是不是红的**。
   - 白色还在、但 body 内红像素占多数 → 红（填充被游戏染红、外框仍白）
   - 白色整条消失、红像素自成弹匣形状 → 红（整个计量器转红）
   两条路径都要求过**形状 IoU + 窗口邻域**两道校验，避免红天/血污/火山背景误触发。
   **百分比永远只由白像素算**，红色分量不进入分子。

5. **空壳通道（打空态，2026-09-19 补）**：弹匣打空时游戏把填充清空、只留一圈轮廓线，
   白像素从 ~544px 掉到 ~142px，低于 `mask_floor`（= 30% 窗口面积 = 145px）→ 旧逻辑
   判「无弹匣」，覆盖层整个隐藏，表现是「打空了啥都没有」。但轮廓的形状信息完好
   （实测凸包 IoU 0.59~0.88），所以这一档**改看形状不看面积**：面积不足时，只要
   形状 IoU >= `shell_min_iou`（比白色通道的 0.46 更严，因为这一档没有面积背书）、
   落在窗口邻域内、体内既无白填充也无红填充 → 判 `state="empty"`、余量 0。
   上层据此画一条 0% 的空轨（只有灰轨、没有填充），而不是把窗口藏起来。

6. **多区域粘性 fallback**：一旦某区域稳定出弹匣就锁定它，只有它连续失效才换另一组。

7. **时间去抖 + 满弹冻结**：百分比 EMA 平滑；`valid` 连续 hold_frames 帧才翻转；
   弹匣顶部有一块永远填不满的空腔（占窗口约 11%），故占比 >= snap_full 直接报 100%。

三种 `state`：`white`（按占比画白条）/ `low`（低弹红，整条泛红）/ `empty`（空弹匣，画空轨）。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
DEFAULT_TEMPLATE = ASSETS / "shape_template.npy"

# 满弹冻结阈值：弹匣图形顶部有一块空腔（占窗口约 11%）永远不参与填充，
# 所以真实满弹的面积占比上限只有 ~90%，不冻结就会长期读成「88%~96%」，
# 看着像弹药少了。>= 该比例直接报 100%。设 >1 即关闭。
SNAP_FULL = 0.88
NORM = 32


# --------------------------------------------------------------------------- #
# 基础形态学 / 几何工具（区域很小，纯 numpy 足够快）
# --------------------------------------------------------------------------- #
def _shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    o = np.zeros_like(m)
    h, w = m.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    o[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return o


def erode(m: np.ndarray, k: int = 1) -> np.ndarray:
    out = m.copy()
    for _ in range(k):
        e = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                e &= _shift(out, dy, dx)
        out = e
    return out


def dilate(m: np.ndarray, k: int = 1) -> np.ndarray:
    out = m.copy()
    for _ in range(k):
        d = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                d |= _shift(out, dy, dx)
        out = d
    return out


def close(m: np.ndarray, k: int = 1) -> np.ndarray:
    return erode(dilate(m, k), k)


def largest_cc(m: np.ndarray, min_px: int = 1) -> np.ndarray:
    """取最大 8 连通域。"""
    h, w = m.shape
    seen = np.zeros((h, w), bool)
    best: list[tuple[int, int]] = []
    for y0 in range(h):
        row = m[y0]
        for x0 in range(w):
            if not row[x0] or seen[y0, x0]:
                continue
            dq = deque([(y0, x0)])
            seen[y0, x0] = True
            pts = []
            while dq:
                y, x = dq.popleft()
                pts.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            dq.append((ny, nx))
            if len(pts) > len(best):
                best = pts
    out = np.zeros((h, w), bool)
    if len(best) >= min_px:
        for y, x in best:
            out[y, x] = True
    return out


def opening(m: np.ndarray, k: int = 1) -> np.ndarray:
    """开运算：先腐蚀再膨胀（保留为通用形态学工具，余量口径里已不用）。"""
    return dilate(erode(m, k), k)


def convex_hull_mask(m: np.ndarray) -> np.ndarray:
    """掩膜凸包（填充成实心）。断续轮廓用它比填洞稳健得多。"""
    ys, xs = np.where(m)
    if len(ys) == 0:
        return np.zeros_like(m)
    pts = sorted(set(zip(xs.tolist(), ys.tolist())))
    if len(pts) <= 2:
        out = np.zeros_like(m)
        for x, y in pts:
            out[y, x] = True
        return out

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]

    from PIL import Image, ImageDraw
    img = Image.new("1", (m.shape[1], m.shape[0]), 0)
    ImageDraw.Draw(img).polygon([(float(x), float(y)) for x, y in hull], fill=1)
    return np.array(img, bool)


def normalize_shape(m: np.ndarray, n: int = NORM) -> np.ndarray:
    """剪影 → 等比缩放居中到 n×n（尺度无关）。"""
    from PIL import Image
    ys, xs = np.where(m)
    if len(ys) == 0:
        return np.zeros((n, n), bool)
    c = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sc = n / max(c.shape)
    nw = max(1, int(round(c.shape[1] * sc)))
    nh = max(1, int(round(c.shape[0] * sc)))
    arr = np.array(Image.fromarray((c * 255).astype(np.uint8))
                   .resize((nw, nh), Image.BILINEAR)) > 110
    out = np.zeros((n, n), bool)
    oy, ox = (n - nh) // 2, (n - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = arr
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 0.0


def patch_count(m: np.ndarray, x: int, y: int, r: int = 1) -> int:
    """统计以 (x, y) 为中心、(2r+1) 见方邻域内的命中像素数。"""
    y0, y1 = max(0, y - r), min(m.shape[0], y + r + 1)
    x0, x1 = max(0, x - r), min(m.shape[1], x + r + 1)
    return int(m[y0:y1, x0:x1].sum())


# --------------------------------------------------------------------------- #
# 检测结果
# --------------------------------------------------------------------------- #
@dataclass
class AmmoReading:
    valid: bool
    fraction: float = 0.0
    fill: int = 0
    area: int = 0
    state: str = "none"          # white | low | empty | none
    shape_iou: float = 0.0
    white_px: int = 0
    mask_px: int = 0
    region: str = ""
    reason: str = ""
    # 两级判定的过程量：level1 = 存在性，level2 = 余量
    hint: str = ""               # 存在性判定的结论：probe:white / shape:white / red
    probe_white: int = 0         # 白色探针命中像素
    red_px: int = 0              # 弹匣体内的红像素数（低弹红判定用，不进百分比分子）

    def __str__(self) -> str:
        if not self.valid:
            return "未检测到"
        if self.state == "low":
            return "低弹(红)"
        return f"{self.fraction * 100:5.1f}%"

    @property
    def bar(self) -> str:
        n = int(round(self.fraction * 20))
        return "[" + "#" * n + "." * (20 - n) + "]"


# --------------------------------------------------------------------------- #
# 单区域检测器
# --------------------------------------------------------------------------- #
class RegionDetector:
    """针对一个屏幕区域的检测器。

    name        区域名（仅用于打印）
    window      固定窗口掩膜（区域内坐标，True=参与计分）。None → auto 模式
    template    归一化形状模板 (32x32)
    """

    def __init__(self, name: str, window: np.ndarray | None = None,
                 template: np.ndarray | None = None, *,
                 bg_margin: int = 18, lum_floor: int = 55, edge: int = 5,
                 shape_min_iou: float = 0.50,
                 min_mask_px: int = 30,
                 near_dilate: int = 3,
                 snap_full: float = SNAP_FULL,
                 probe_white: list | None = None,
                 probe_radius: int = 1, white_need: int = 20,
                 max_mask_ratio: float = 0.55, min_mask_ratio: float = 0.30,
                 fill_inset: int = 2, empty_frac: float = 0.06,
                 red_chroma: int = 55, red_min: int = 120,
                 red_min_px: int = 25, red_body_frac: float = 0.20,
                 red_shape_min_iou: float = 0.40,
                 shell_min_iou: float = 0.52, shell_min_px: int = 40):
        self.name = name
        self.window = None if window is None else np.asarray(window).astype(bool)
        self.template = (np.load(DEFAULT_TEMPLATE) if template is None
                         else np.asarray(template)).astype(bool)
        self.bg_margin = bg_margin
        self.lum_floor = lum_floor
        self.edge = edge
        self.shape_min_iou = shape_min_iou
        self.min_mask_px = min_mask_px
        self.near_dilate = near_dilate
        self.snap_full = float(snap_full)
        self.probe_white = [(int(x), int(y)) for x, y in (probe_white or [])]
        self.max_mask_ratio = float(max_mask_ratio)
        # 掩膜下限：不仅要有 min_mask_px 个像素，还得占窗口掩膜的一定比例。
        # 几十像素的小色块归一化后也可能撞上模板形状（实测 58px 的红色碎片 IoU 0.50），
        # 一旦放行就会被当成「低弹全红」，于是整条弧线无故变红。
        self.mask_floor = max(int(min_mask_px),
                              int(float(min_mask_ratio) * self.window.sum())) \
            if self.window is not None else int(min_mask_px)
        # 分母口径：以「弹匣自身」的内缩凸包为满弹面积，而不是标定时写死的窗口掩膜。
        # 图标会随分辨率/持枪姿态漂移，实测掩膜与弹匣 IoU 只有 0.37，分子被砍掉一半
        # （掩膜内 287px vs 弹匣内 570px）→ 必须用当下这一帧的形状当分母。
        self.fill_inset = max(1, int(fill_inset))   # 内缩像素，用来剥掉恒定可见的外框线
        self.empty_frac = float(empty_frac)
        # 红色通道参数：只用于「低弹红」状态判定，绝不参与百分比。
        self.red_chroma = int(red_chroma)           # R - max(G,B) 下限（色度）
        self.red_min = int(red_min)                 # R 绝对下限，压掉暗红背景
        self.red_min_px = int(red_min_px)           # 红像素绝对数下限（压噪点）
        self.red_body_frac = float(red_body_frac)   # 红像素占弹匣体面积的比例下限
        self.red_shape_min_iou = float(red_shape_min_iou)
        # 空壳通道参数：白像素掉到 mask_floor 以下（只剩余轮廓线）时，改看形状。
        self.shell_min_iou = float(shell_min_iou)
        self.shell_min_px = int(shell_min_px)
        self.probe_radius = max(1, int(probe_radius))
        self.white_need = int(white_need)

    # -- 掩膜 --------------------------------------------------------------- #
    def extract(self, rgb: np.ndarray) -> np.ndarray:
        """取出「白像素」掩膜。逐行自适应：比该行两侧中位亮度高出 bg_margin 才算白。

        整条链路只认白像素，不碰任何颜色判定（红/黄/棕一律不参与）。
        chroma 只用来排除彩色像素（沙地/天空的暖色也会很亮），阈值是「接近灰」。
        """
        a = rgb.astype(np.int16)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        chroma = r - np.maximum(g, b)
        e = max(1, min(self.edge, rgb.shape[1] // 3))
        side = np.concatenate([lum[:, :e], lum[:, -e:]], axis=1)
        bg = np.median(side, axis=1)
        return ((lum > bg[:, None] + self.bg_margin) & (lum > self.lum_floor)
                & (chroma < 25))

    def _collect(self, m: np.ndarray) -> np.ndarray:
        """取最大连通域 —— 弹匣图标自身是一整块，区域里其它 HUD 元素天然被排除。

        注意：这里**不能**做窗口邻域裁剪。裁剪会把其它图标也裁成窗口形状，
        使形状校验完全失效（实测正负样本 IoU 从 0.92/0.30 退化为 0.61/0.53）。
        """
        return largest_cc(m, self.min_mask_px)

    def extract_red(self, rgb: np.ndarray) -> np.ndarray:
        """红色掩膜 —— 只服务于「低弹红」状态判定，绝不参与余量百分比。

        判据是纯色度 + 亮度下限：`R - max(G,B) >= red_chroma` 且 `R >= red_min`。
        两者都要，是因为单靠色度会把暗红（火山地表、血污、红天空的阴影侧）
        一起收进来；加上亮度下限后，只有游戏 HUD 那种高饱和亮红才过。

        即便如此仍不足以定案，红色必须在**形状校验 + 窗口邻域**都通过之后才生效
        （见 `_red_fallback` / `analyze` 第二级），所以背景红不会误触发。
        """
        a = rgb.astype(np.int16)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        return ((r - np.maximum(g, b)) >= self.red_chroma) & (r >= self.red_min)

    def _red_reading(self, rgb: np.ndarray, body: np.ndarray | None = None):
        """红色通道定位：红像素自成弹匣形状 → 判定低弹红。

        这是「白色整条消失」时的兜底路径（整个计量器被游戏染红）。若白色还在，
        走的是 `analyze` 第二级里 body 内的红像素判定，不经过这里。

        返回 AmmoReading 或 None（不成立）。
        """
        red = self.extract_red(rgb)
        hot = largest_cc(red, self.min_mask_px)
        n = int(hot.sum())
        if n < max(self.red_min_px, self.mask_floor):
            return None
        # 位置校验：红色弹匣必须落在标定窗口的邻域里（挡掉画面别处的红色物体）
        if self.window is not None and \
                not bool((hot & dilate(self.window, self.near_dilate)).any()):
            return None
        hull = convex_hull_mask(hot)
        # 形状校验：和白色通道同一把尺子。红色碎片/红色场景过不了这一关。
        v = iou(normalize_shape(hull), self.template)
        if v < self.red_shape_min_iou:
            return None
        b = erode(hull, self.fill_inset) if body is None else body
        area = int(b.sum())
        if area < 20:
            return None
        red_px = int((red & b).sum())
        if red_px < self.red_min_px:
            return None
        return AmmoReading(True, red_px / area, red_px, area, "low", v,
                           n, n, self.name, "", hint="red", red_px=red_px)

    def _shell_reading(self, rgb: np.ndarray, white: np.ndarray | None = None):
        """空弹壳：白像素只剩外框轮廓线、体内无填充 → 判定「空弹匣」。

        空弹时游戏把填充清空、只留一圈轮廓，白像素从 ~544px 掉到 ~142px，
        低于 `mask_floor`（= 30% 窗口面积 = 145px）→ 旧逻辑判「无弹匣」，
        覆盖层整个隐藏，表现就是「打空了啥都没有」。但形状信息完好
        （实测凸包 IoU 0.59~0.88，门槛仅 0.46），所以这一档改看形状不看面积。

        返回 AmmoReading(valid=True, fraction=0, state="empty") 或 None。
        """
        if self.window is None:
            return None                     # 无标定窗口时 mask_floor 退化，这一档不启用
        w = self.extract(rgb) if white is None else white
        hot = largest_cc(w, self.min_mask_px)
        n = int(hot.sum())
        if n < self.shell_min_px:
            return None
        if n >= self.mask_floor:
            return None                     # 面积够 → 属于正常路径，空壳只兜「面积不足」这一档
        # 位置校验：轮廓必须压在标定窗口上（空弹壳贴着外框，会跨过窗口边界）
        if self.window is not None and \
                not bool((hot & dilate(self.window, self.near_dilate)).any()):
            return None
        hull = convex_hull_mask(hot)
        # 形状门槛比白色通道更严（0.52 vs 0.46）：这一档没有面积背书，全靠形状。
        v = iou(normalize_shape(hull), self.template)
        if v < self.shell_min_iou:
            return None
        body = erode(hull, self.fill_inset)
        area = int(body.sum())
        if area < 20:
            return None
        fill = int((w & body).sum())
        if fill > self.empty_frac * area:
            return None                     # 体内还有白填充 → 不该走空壳，交给正常路径
        red = self.extract_red(rgb)
        red_px = int((red & body).sum())
        if red_px >= max(self.red_min_px, self.red_body_frac * area):
            return None                     # 体内是红填充 → 那是低弹红，不是空弹
        return AmmoReading(True, 0.0, fill, area, "empty", v, n, n, self.name,
                           "", hint="shell", red_px=red_px)

    def _fail(self, rgb: np.ndarray, reason: str, white=None, **kw) -> AmmoReading:
        """白色通道判否 —— 依次再问红色通道、空壳通道。

        低弹时游戏把计量器整个转红，白色会一条不剩；打空时游戏把填充清空、
        只留轮廓线，白像素掉到面积门槛以下。两种情况下纯白口径都会输出
        「未检测到」并让覆盖层隐藏，所以每个失败出口都必须先过这两条通道。
        """
        r = self._red_reading(rgb)
        if r is not None:
            return r
        s = self._shell_reading(rgb, white)
        if s is not None:
            return s
        return AmmoReading(False, region=self.name, reason=reason, **kw)

    def probe(self, white: np.ndarray):
        """第一级：弹匣**存在性**判定，只用白色外框端点，与填充量无关。

        外框在弹匣存在时恒定可见，所以探端点比比对整块形状稳。两组区域
        尺寸不同 → 探针坐标必须各自标定（写在 config.json 的 regions[].probe）。

        **红色不参与任何判定**：红色探针实测极易误触发（红天/沙漠/血污场景整条
        弧线无故变红），而低弹时白色像素占比自然就掉下来了，红不红由占比决定。

        返回 (hint, 白探针命中)；hint ∈ {"white", None}。
        """
        if not self.probe_white:
            return None, 0
        r = self.probe_radius
        wp = sum(patch_count(white, x, y, r) for x, y in self.probe_white)
        if wp >= self.white_need:
            return "white", wp
        return None, wp

    # -- 主流程 ------------------------------------------------------------- #
    def analyze(self, rgb: np.ndarray) -> AmmoReading:
        if self.window is not None and rgb.shape[:2] != self.window.shape:
            raise ValueError(f"[{self.name}] 区域尺寸 {rgb.shape[:2]} 与窗口 "
                             f"{self.window.shape} 不符")
        white = self.extract(rgb)

        # ---- 第一级：有没有弹匣 -------------------------------------------- #
        hint, p_white = self.probe(white)

        # 弹匣 = 白像素的最大连通域（外框恒白，所以低弹时轮廓仍在，检测不会丢）。
        hot = self._collect(white)
        mask_px = int(hot.sum())
        white_px = mask_px
        hull = convex_hull_mask(hot)
        v = iou(normalize_shape(hull), self.template)

        if hint is None:
            # 探针沉默（该区域未标定探针，或视频等非本机尺度样本）→ 退回形状判存在
            if mask_px < self.mask_floor:
                return self._fail(rgb, f"无弹匣(掩膜{mask_px}px<{self.mask_floor})",
                                  shape_iou=v, mask_px=mask_px, probe_white=p_white)
            if self.window is not None and \
                    not bool((hot & dilate(self.window, self.near_dilate)).any()):
                return self._fail(rgb, "不在窗口内", shape_iou=v, mask_px=mask_px,
                                  probe_white=p_white)
            if v < self.shape_min_iou:
                return self._fail(rgb,
                                  f"形状不符(IoU {v:.2f}<{self.shape_min_iou})",
                                  shape_iou=v, mask_px=mask_px, white_px=white_px,
                                  probe_white=p_white)
            hint = "shape:white"
        else:
            # 白探针命中只是「候选」——持枪姿态下枪身剪影横跨该区域时，
            # 外框端点会全部命中（实测 54/54 满分），必须再过两道结构校验：
            #   1) 掩膜不能过大：弹匣图标占区域约 20%，枪身剪影能占 ~80%
            #   2) 凸包形状 IoU：弹匣 >= 0.50，枪身/纯色背景恒为 0.43
            area_all = int(rgb.shape[0]) * int(rgb.shape[1])
            if mask_px > self.max_mask_ratio * area_all:
                return self._fail(rgb,
                                  f"亮块过大({mask_px}px>{area_all * self.max_mask_ratio:.0f})",
                                  shape_iou=v, mask_px=mask_px, white_px=white_px,
                                  probe_white=p_white)
            if mask_px < self.mask_floor:
                return self._fail(rgb, f"掩膜过少({mask_px}<{self.mask_floor})",
                                  shape_iou=v, mask_px=mask_px, probe_white=p_white)
            if v < self.shape_min_iou:
                return self._fail(rgb,
                                  f"探针命中但形状不符(IoU {v:.2f}<{self.shape_min_iou})",
                                  shape_iou=v, mask_px=mask_px, white_px=white_px,
                                  probe_white=p_white)
            hint = "probe:white"

        # ---- 第二级：白像素占多少 ------------------------------------------ #
        # 口径就一句话：**弹匣体内有多少白像素 ÷ 弹匣体面积**。
        #   body = 弹匣凸包内缩 fill_inset 像素。fill_inset 取外框线自身的粗细（2px），
        #   于是那条恒定可见的外框线整条落在 body 之外，既不用做开运算去猜线粗，
        #   也不会把底部那块实心填充自己腐蚀小一圈（旧口径做实心块被削两圈，读数偏低）。
        #   实测（2560x1440）：满弹 93% → snap_full 冻成 100%；用户参考帧 22%。
        body = erode(hull, self.fill_inset)
        area = int(body.sum())
        if area < 20:
            return self._fail(rgb, "窗口过小", shape_iou=v, mask_px=mask_px,
                              probe_white=p_white)
        fill = int((white & body).sum())

        # ---- 红色通道 A 路径：白还在，但 body 内其实是红填充 ------------------ #
        # 低弹时游戏把计量器染红、外框仍是白的 → 白色通道照常定位到弹匣，但 body 内
        # 白像素近乎为零。必须在这一步就问一句「里面是不是红的」，否则会被当成空弹匣。
        red = self.extract_red(rgb)
        red_px = int((red & body).sum())
        if red_px >= self.red_min_px and red_px >= self.red_body_frac * area:
            return AmmoReading(True, red_px / area, red_px, area, "low", v,
                               white_px, mask_px, self.name, "",
                               hint=hint + "+red", probe_white=p_white, red_px=red_px)

        frac = fill / area
        if frac >= self.snap_full:
            frac = 1.0            # 满弹冻结：顶部空腔永远填不满，接近满就报满
        elif frac <= self.empty_frac:
            frac = 0.0            # 空弹匣的残余（外框内缘）不算数
        state = "empty" if frac <= 0.0 else "white"
        return AmmoReading(True, frac, fill, area, state, v,
                           white_px, mask_px, self.name, "",
                           hint=hint, probe_white=p_white, red_px=red_px)


# --------------------------------------------------------------------------- #
# HUD 布局探针（背包 / 无背包）
# --------------------------------------------------------------------------- #
class LayoutProbe:
    """判「这一帧的 HUD 是背着背包还是没背」。

    游戏里背不背补给背包，左下角弹药 HUD 会整体平移：实测（2560x1440）弹匣图标
    从 x 372 移到 x 266，**左移 106px，y 不变**。判据是用户给的固定矩形
    `rect`（标定分辨率坐标，默认 (330,1250)-(337,1295)）里有没有一条**白色竖线** ——
    那是背包图标与弹药区之间的分隔竖线，只有背着背包时才出现。

    为什么用「最长竖直连续亮段」而不是像素总数：竖线在 2560x1440 下约 8px 宽、
    36~45px 高且竖直连续；同一位置在无背包时只剩弹药数字的笔画（竖直连续 <= 20px）。
    总数对 JPEG 压缩/抗锯齿太敏感（实测总数 32 vs 1 反而好区分，但不稳），
    连续长度实测 28 vs 6，区分度足够且机理明确。
    """

    def __init__(self, rect, *, shift: tuple[int, int] = (-106, 0),
                 min_run: int = 24, lum: float = 130.0, interval: float = 1.0,
                 name: str = "背包"):
        self.rect = tuple(int(v) for v in rect)
        self.shift = (int(shift[0]), int(shift[1]))
        self.min_run = int(min_run)
        self.lum = float(lum)
        # 每 interval 秒才判一次：背包不会秒秒变化，判据本身要花 ~0.7ms。
        # 按时间而不是帧数节流 —— 采样率变了（50Hz / 25Hz / 卡顿掉帧）判据频率不变。
        self.interval = max(0.05, float(interval))
        self.name = name

    @property
    def size(self) -> tuple[int, int]:
        l, t, r, b = self.rect
        return r - l, b - t

    def longest_run(self, rgb: np.ndarray) -> int:
        """判据矩形内，任一列的最长竖直连续亮段长度（像素）。"""
        a = rgb.astype(np.int16)
        lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        m = lum > self.lum
        if not m.any():
            return 0
        runs = np.zeros(m.shape, np.int32)
        runs[0] = m[0]
        for i in range(1, m.shape[0]):                       # 只有 ~45 行，够快
            runs[i] = np.where(m[i], runs[i - 1] + 1, 0)
        return int(runs.max())

    def has_backpack(self, rgb: np.ndarray) -> bool:
        return self.longest_run(rgb) >= self.min_run


# --------------------------------------------------------------------------- #
# 多区域粘性 fallback
# --------------------------------------------------------------------------- #
@dataclass
class RegionSpec:
    name: str
    box: tuple[int, int, int, int]        # 屏幕物理像素 (l, t, r, b)
    detector: RegionDetector = field(repr=False)


class MultiRegionDetector:
    """粘性多区域检测 + 布局自适应（背包 / 无背包）。

    语义（按需求）：某一组区域出现弹匣就一直用它；这组失效时才换另一组；
    两组都识别不到则输出「未检测到」。

    截图策略（mss 每次 grab 有 ~8ms 固定开销，能省则省）：
      **一次截取 canvas**，再切片分发给各检测器 —— 1 次 grab 搞定所有区域、
      布局判据矩形、以及「区域 + shift」的备用位置。canvas 是这些矩形的并集，
      尺寸只比单区域大一点（166x50 量级），而 mss 的开销主要在固定成本上，
      抓大一点几乎不额外花钱。

    布局自适应：`layout`（LayoutProbe）每帧从 canvas 里切出判据矩形判断
    「有没有背背包」，据此决定区域坐标要不要加 `shift`（无背包时 HUD 左移）。
    布局切换会清空时间去抖状态，避免把上一种布局的读数带过来。
    """

    def __init__(self, specs: list[RegionSpec], *, hold_frames: int = 3,
                 ema_alpha: float = 0.5, switch_after: int = 2,
                 union_capture: bool = True, snap_full: float = SNAP_FULL,
                 layout: LayoutProbe | None = None):
        assert specs, "至少要有一个区域"
        self.specs = specs
        self.hold_frames = max(1, hold_frames)
        self.ema_alpha = ema_alpha
        self.switch_after = max(1, switch_after)
        self.snap_full = float(snap_full)
        self.active: int | None = None
        self.forced: int | None = None       # 手动锁定：只认这一组，不做 auto fallback
        self._miss = 0
        self._ema: float | None = None
        self._valid = False
        self._pend: deque[bool] = deque(maxlen=self.hold_frames)
        self.last_raw: AmmoReading | None = None
        self.last_scan: list[str] = []
        self.last_run = 0                    # 最近一帧判据矩形里的最长竖直亮段

        # -- 布局 -------------------------------------------------------------- #
        self.layout = layout
        self.layout_forced: bool | None = None    # None=自动判，True/False=强制
        self.has_backpack = True
        self.shift = (0, 0)
        self._shift_set = False
        self._layout_next_t = 0.0             # 下一次判布局的单调时刻（秒）
        self.update_canvas()

    # -- 布局 ---------------------------------------------------------------- #
    def update_canvas(self) -> None:
        """重算抓取范围：所有区域（含平移后的位置）+ 布局判据矩形的并集。"""
        xs0, ys0, xs1, ys1 = [], [], [], []
        shifts = {(0, 0)}
        if self.layout is not None:
            shifts.add(self.layout.shift)
        for s in self.specs:
            l, t, r, b = s.box
            for dx, dy in shifts:                      # 两种布局下的区域位置
                xs0.append(l + dx)
                ys0.append(t + dy)
                xs1.append(r + dx)
                ys1.append(b + dy)
        if self.layout is not None:                    # 判据矩形是屏幕固定位置
            pl, pt, pr, pb = self.layout.rect
            xs0.append(pl)
            ys0.append(pt)
            xs1.append(pr)
            ys1.append(pb)
        self.canvas = (min(xs0), min(ys0), max(xs1), max(ys1))
        # 向后兼容旧字段名
        self.union = self.canvas

    def _reset_temporal(self) -> None:
        """布局变了：清掉平滑与去抖，避免两种布局的读数混在一起。"""
        self._ema = None
        self._valid = False
        self._pend.clear()
        self._miss = 0

    def apply_layout(self, bag: bool) -> bool:
        """设置当前布局（受 layout_forced 覆盖），返回是否发生了切换。"""
        want = bag if self.layout_forced is None else self.layout_forced
        if self._shift_set and want == self.has_backpack:
            return False
        self.has_backpack = want
        self.shift = (0, 0) if want else tuple(self.layout.shift)
        self._shift_set = True
        self._reset_temporal()
        return True

    def force_layout(self, v: bool | None) -> None:
        """None = 自动按判据；True = 强制有背包；False = 强制无背包。"""
        if v == self.layout_forced:
            return
        self.layout_forced = v
        bare = self.has_backpack if v is None else bool(v)
        self.has_backpack = bare
        self.shift = (0, 0) if bare else tuple(self.layout.shift)
        self._shift_set = True
        self._reset_temporal()
        self._layout_next_t = 0.0             # 解除强制后立刻重新判一次

    # -- 区域尝试顺序 -------------------------------------------------------- #
    def force(self, i: int | None) -> None:
        """手动锁定第 i 组区域；传 None 回到自动（粘性 fallback）。

        切换时清掉时间去抖状态，否则会把上一组的读数带过来。
        """
        if i == self.forced:
            return
        self.forced = i
        self.active = i
        self._miss = 0
        self._ema = None
        self._valid = False
        self._pend.clear()

    @property
    def mode(self) -> str:
        if self.layout is not None:
            tag = "有背包" if self.has_backpack else "无背包"
            pre = "自动" if self.layout_forced is None else "强制"
            extra = ""
            if self.forced is not None:
                extra = f"·锁{self.specs[self.forced].name}"
            return f"{pre}{tag}{extra}"
        if self.forced is not None:
            return f"锁定{self.specs[self.forced].name}"
        return "自动" if self.active is None else f"粘住{self.specs[self.active].name}"

    def _order(self) -> list[int]:
        if self.forced is not None:
            return [self.forced]
        if self.active is None:
            return list(range(len(self.specs)))
        rest = [i for i in range(len(self.specs)) if i != self.active]
        return [self.active] + rest

    def _slice(self, cache: np.ndarray, spec: RegionSpec,
               shift: tuple[int, int] = (0, 0)) -> np.ndarray:
        ul, ut = self.canvas[0], self.canvas[1]
        dx, dy = shift
        l, t, r, b = spec.box
        return cache[t + dy - ut:b + dy - ut, l + dx - ul:r + dx - ul]

    def probe_layout(self, cache: np.ndarray) -> None:
        """从 canvas 里切出判据矩形，决定这一帧用哪套区域坐标。

        按 `layout.interval`（秒）节流：判一次要 ~0.7ms（竖直连续段是逐行扫描），
        而背包状态一秒内不会变。非检查帧沿用上一次结论。
        """
        if self.layout is None or self.layout_forced is not None:
            return
        now = time.monotonic()
        if now < self._layout_next_t:
            return
        self._layout_next_t = now + self.layout.interval
        pl, pt, pr, pb = self.layout.rect
        ul, ut = self.canvas[0], self.canvas[1]
        sub = cache[pt - ut:pb - ut, pl - ul:pr - ul]
        self.last_run = self.layout.longest_run(sub)
        self.apply_layout(self.last_run >= self.layout.min_run)   # 只扫一遍，不重复算

    def analyze(self, grab_box) -> AmmoReading:
        """grab_box((l, t, r, b)) -> 该矩形的 (h, w, 3) RGB 数组。

        每帧只截一次 canvas，布局判据与所有区域都从这张图里切。
        """
        order = self._order()
        raw: AmmoReading | None = None
        scan: list[str] = []

        cache = grab_box(self.canvas)
        self.probe_layout(cache)

        for i in order:
            spec = self.specs[i]
            sub = self._slice(cache, spec, self.shift)
            r = spec.detector.analyze(sub)
            scan.append(f"{spec.name}:{'-' if r.valid else r.reason or 'x'}")
            if r.valid:
                self.active = i
                self._miss = 0
                raw = r
                break
            if raw is None:
                raw = r
        self.last_scan = scan
        assert raw is not None

        if not raw.valid and self.forced is None:
            self._miss += 1
            if self._miss >= self.switch_after:
                self.active = None        # 放弃锁定，下轮重新按优先级找
        self.last_raw = raw

        # --- 时间去抖 ---
        if raw.valid:
            self._ema = raw.fraction if self._ema is None else (
                self.ema_alpha * raw.fraction + (1 - self.ema_alpha) * self._ema)
        self._pend.append(raw.valid)
        if (len(self._pend) == self._pend.maxlen
                and raw.valid != self._valid
                and all(v == raw.valid for v in self._pend)):
            self._valid = raw.valid
            if not raw.valid:
                self._ema = None

        # 平滑后再冻一次：EMA 会把刚满弹的那几帧从 1.0 拉下来。
        # 低弹红不参与冻结 —— 那一档的 fraction 是红像素占比，不是余量。
        frac = self._ema or 0.0
        if raw.state != "low" and frac >= self.snap_full:
            frac = 1.0
        return AmmoReading(self._valid, frac, raw.fill, raw.area,
                           raw.state if self._valid else "none", raw.shape_iou,
                           raw.white_px, raw.mask_px, raw.region,
                           raw.reason if not self._valid else "",
                           hint=raw.hint, probe_white=raw.probe_white,
                           red_px=raw.red_px)


# --------------------------------------------------------------------------- #
# 从配置构建
# --------------------------------------------------------------------------- #
def _load_npy(p, base: Path):
    p = Path(p)
    return np.load(p if p.is_absolute() else base / p)


def build_detector(cfg: dict, base: Path | None = None) -> MultiRegionDetector:
    if base is None:                     # 打包成 exe 后资源在解包目录，不是 __file__ 旁边
        from capture import res_dir
        base = res_dir()
    d = cfg.get("detect", {})
    tmpl = _load_npy(d["template"], base) if d.get("template") else None
    specs = []
    for item in cfg["regions"]:
        win = _load_npy(item["window"], base) if item.get("window") else None
        pr = item.get("probe") or {}
        det = RegionDetector(
            item["name"], win, tmpl,
            bg_margin=d.get("bg_margin", 18), lum_floor=d.get("lum_floor", 55),
            edge=d.get("edge", 5), shape_min_iou=d.get("shape_min_iou", 0.50),
            min_mask_px=d.get("min_mask_px", 30),
            near_dilate=d.get("near_dilate", 3),
            snap_full=d.get("snap_full", SNAP_FULL),
            probe_white=pr.get("white"),
            probe_radius=pr.get("radius", 1),
            white_need=pr.get("white_need", d.get("white_need", 20)),
            max_mask_ratio=d.get("max_mask_ratio", 0.55),
            min_mask_ratio=d.get("min_mask_ratio", 0.30),
            fill_inset=d.get("fill_inset", 2),
            empty_frac=d.get("empty_frac", 0.06),
            red_chroma=d.get("red_chroma", 55),
            red_min=d.get("red_min", 120),
            red_min_px=d.get("red_min_px", 25),
            red_body_frac=d.get("red_body_frac", 0.20),
            red_shape_min_iou=d.get("red_shape_min_iou", 0.40),
            shell_min_iou=d.get("shell_min_iou", 0.52),
            shell_min_px=d.get("shell_min_px", 40))
        l, t, r_, b_ = item["region"]
        if win is not None and win.shape != (b_ - t, r_ - l):
            raise ValueError(
                f"[{item['name']}] 区域 {r_ - l}x{b_ - t} 与窗口掩膜 "
                f"{win.shape[1]}x{win.shape[0]} 尺寸不符——掩膜必须按同一区域重新生成")
        specs.append(RegionSpec(item["name"], tuple(item["region"]), det))
    lay = cfg.get("layout")
    probe = None
    if lay and lay.get("enabled", True):
        probe = LayoutProbe(lay["probe_rect"],
                            shift=tuple(lay.get("shift", (-106, 0))),
                            min_run=lay.get("min_run", 24),
                            lum=lay.get("lum", 130),
                            interval=lay.get("interval", 1.0))
    return MultiRegionDetector(specs, hold_frames=d.get("hold_frames", 3),
                               ema_alpha=d.get("ema_alpha", 0.5),
                               switch_after=d.get("switch_after", 2),
                               snap_full=d.get("snap_full", SNAP_FULL),
                               layout=probe)


# 兼容旧接口
class AmmoDetector(RegionDetector):
    def __init__(self, mask_path=ASSETS / "interior_mask.npy", **kw):
        super().__init__(kw.pop("name", "main"), np.load(mask_path), **kw)
