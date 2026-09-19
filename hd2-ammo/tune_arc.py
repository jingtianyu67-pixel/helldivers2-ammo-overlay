# -*- coding: utf-8 -*-
"""弧线可视化调参器：坐标 / 大小 / 透明度 / 角度，所见即所得。

    python tune_arc.py

屏幕上会出现一层半透明遮罩 + 一条真实渲染的弧线（用和游戏里同一个
layered window，所以看到的就是最终效果）。旁边是一块控制面板。

操作
   左键拖动        移动弧线（圆心跟着走）
   滚轮            半径 ±6
   Shift + 滚轮    线宽 ±1
   Ctrl  + 滚轮    透明度 ±0.05
   方向键          微调位置 1px（按住 Shift 为 10px）
   Tab             开关背景遮罩（关掉只剩弧线，方便对着桌面/游戏看）
   S               保存到 config.json
   R               复位成 config.json 里的值
   Esc             退出（有未保存改动会问一句）

面板上的滑块与上面完全等价，随时同步。
"""

from __future__ import annotations

import json
import os
import sys
import tkinter as tk
from tkinter import messagebox

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from capture import set_dpi_aware                     # noqa: E402
from overlay import AmmoOverlay, ArcGeometry, screen_size  # noqa: E402

CFG_PATH = os.path.join(HERE, "config.json")

# 面板滑块：(键, 标题, 最小, 最大, 步进, 显示格式)
SLIDERS = [
    ("offset_x",  "位置 X（相对屏幕中心）", -2400, 2400, 1,    "{:.0f}"),
    ("offset_y",  "位置 Y（相对屏幕中心）", -1400, 1400, 1,    "{:.0f}"),
    ("radius",    "半径",                    20,   1200, 1,    "{:.0f}"),
    ("width",     "线宽",                    2,    140,  1,    "{:.0f}"),
    ("alpha",     "透明度",                  0.05, 1.0,  0.01, "{:.2f}"),
    ("ang_start", "起始角（上端，0°=正右，顺时针）", -180, 180, 1, "{:.0f}"),
    ("ang_end",   "结束角（下端，填充从这头往里长）", -180, 180, 1, "{:.0f}"),
]


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class Tuner:
    def __init__(self):
        set_dpi_aware()
        self.raw = json.load(open(CFG_PATH, encoding="utf-8"))
        src = json.loads(json.dumps(self.raw.get("overlay", {}) or {}))
        src.pop("_comment", None)
        self.o = src
        self.o.setdefault("anchor", "center")
        self.o.setdefault("offset", [-300, 0])
        self.o.setdefault("radius", 356)
        self.o.setdefault("width", 22)
        self.o.setdefault("alpha", 1.0)
        self.o.setdefault("ang_start", -59)
        self.o.setdefault("ang_end", 48)
        self.initial = json.loads(json.dumps(self.o))
        self.baseline = json.dumps(self.o, sort_keys=True)

        self.screen = screen_size()
        self.sw, self.sh = self.screen
        self.fraction = 0.62
        self._loading = True
        self._shot = None          # 拖动时的鼠标-圆心偏移

        self.root = tk.Tk()
        self.root.withdraw()

        # ---- 半透明背景遮罩（同时负责接鼠标） ------------------------------ #
        self.mask = tk.Toplevel(self.root)
        self.mask.overrideredirect(True)
        self.mask.geometry(f"{self.sw}x{self.sh}+0+0")
        self.mask.attributes("-topmost", True)
        self.mask.attributes("-alpha", 0.30)
        self.mask.configure(bg="#101014")
        self.cv = tk.Canvas(self.mask, width=self.sw, height=self.sh,
                            bg="#101014", highlightthickness=0, cursor="crosshair")
        self.cv.pack()

        # ---- 真实弧线覆盖层 ------------------------------------------------ #
        self.ov = AmmoOverlay({"overlay": self.o}, self.screen)

        self._build_panel()
        self._bind()

        self.mask.focus_force()
        self.sync_all()
        self._draw_guides()

    # ------------------------------------------------------------------ 面板 #
    def _build_panel(self) -> None:
        p = tk.Toplevel(self.root)
        self.panel = p
        p.title("弧线调参")
        p.attributes("-topmost", True)
        p.resizable(False, False)
        p.configure(bg="#f4f4f6")
        x = max(0, self.sw - 380)
        p.geometry(f"370x560+{x}+70")

        tk.Label(p, text="弧线调参  ·  改完按 S 保存", bg="#f4f4f6",
                 font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 4))

        self.scales = {}
        self.valuetext = {}
        body = tk.Frame(p, bg="#f4f4f6")
        body.pack(fill="x", padx=12)

        for key, title, lo, hi, res, fmt in SLIDERS:
            row = tk.Frame(body, bg="#f4f4f6")
            row.pack(fill="x", pady=(4, 0))
            head = tk.Frame(row, bg="#f4f4f6")
            head.pack(fill="x")
            tk.Label(head, text=title, bg="#f4f4f6", anchor="w",
                     font=("Microsoft YaHei UI", 9)).pack(side="left")
            vlab = tk.Label(head, text="", bg="#f4f4f6", fg="#0b5cad",
                            font=("Consolas", 9, "bold"))
            vlab.pack(side="right")
            self.valuetext[key] = (vlab, fmt)

            sc = tk.Scale(row, from_=lo, to=hi, resolution=res,
                          orient="horizontal", showvalue=False, length=330,
                          bg="#f4f4f6", highlightthickness=0, troughcolor="#dcdce4",
                          command=lambda v, k=key: self._on_slide(k, float(v)))
            sc.set(self._get(key))
            sc.pack(fill="x")
            self.scales[key] = sc

        # 演示余量
        row = tk.Frame(body, bg="#f4f4f6")
        row.pack(fill="x", pady=(8, 0))
        head = tk.Frame(row, bg="#f4f4f6")
        head.pack(fill="x")
        tk.Label(head, text="演示余量（只影响预览）", bg="#f4f4f6", anchor="w",
                 font=("Microsoft YaHei UI", 9)).pack(side="left")
        self.frac_text = tk.Label(head, text="", bg="#f4f4f6", fg="#0b5cad",
                                  font=("Consolas", 9, "bold"))
        self.frac_text.pack(side="right")
        self.frac_scale = tk.Scale(row, from_=0, to=1, resolution=0.01,
                                   orient="horizontal", showvalue=False, length=330,
                                   bg="#f4f4f6", highlightthickness=0,
                                   troughcolor="#dcdce4", command=self._on_frac)
        self.frac_scale.set(self.fraction)
        self.frac_scale.pack(fill="x")

        # 按钮
        btns = tk.Frame(p, bg="#f4f4f6")
        btns.pack(fill="x", padx=12, pady=(14, 6))
        for text, cmd in (("保存 (S)", self._save), ("复位 (R)", self._reset),
                          ("遮罩 (Tab)", self._toggle_mask), ("退出 (Esc)", self._quit)):
            tk.Button(btns, text=text, command=cmd, width=9,
                      font=("Microsoft YaHei UI", 9)).pack(side="left", padx=2)

        self.info = tk.Label(p, bg="#f4f4f6", justify="left", anchor="w",
                             font=("Consolas", 9), fg="#333")
        self.info.pack(fill="x", padx=12, pady=(6, 10))

        # 键盘事件必须绑到控件本身：Tk 的 Scale/Button 会先吃掉方向键。
        self._bind_keys(p)
        p.protocol("WM_DELETE_WINDOW", self._quit)
        self._loading = False

    def _bind_keys(self, widget):
        widget.bind("<Key>", self._key)
        for ch in widget.winfo_children():
            self._bind_keys(ch)

    def _bind(self) -> None:
        self.cv.bind("<Button-1>", self._drag_start)
        self.cv.bind("<B1-Motion>", self._drag_move)
        self.cv.bind("<ButtonRelease-1>", self._drag_end)
        self.cv.bind("<MouseWheel>", self._wheel)
        for w in (self.mask, self.panel):
            w.bind("<MouseWheel>", self._wheel)
            w.bind("<Key>", self._key)
        self.panel.focus_force()

    # ------------------------------------------------------------- 键值存取 #
    def _get(self, key):
        if key == "offset_x":
            return self.o["offset"][0]
        if key == "offset_y":
            return self.o["offset"][1]
        return self.o[key]

    def _set(self, key, val):
        if key == "offset_x":
            self.o["offset"][0] = val
        elif key == "offset_y":
            self.o["offset"][1] = val
        else:
            self.o[key] = val

    # ------------------------------------------------------------------ 交互 #
    def _on_slide(self, key, val):
        if self._loading:
            return
        self._set(key, val)
        self.sync_all()

    def _on_frac(self, val):
        if self._loading:
            return
        self.fraction = float(val)
        self.sync_all(guides=False)

    def _mouse_pos(self):
        return self.panel.winfo_pointerx(), self.panel.winfo_pointery()

    def _drag_start(self, _e):
        mx, my = self._mouse_pos()
        self._shot = (mx - self._center()[0], my - self._center()[1])
        self.mask.focus_force()

    def _drag_move(self, _e):
        if self._shot is None:
            return
        mx, my = self._mouse_pos()
        cx, cy = mx - self._shot[0], my - self._shot[1]
        ax, ay = self._anchor()
        self.o["offset"] = [round(cx - ax), round(cy - ay)]
        self.sync_all()

    def _drag_end(self, _e):
        self._shot = None

    def _wheel(self, e):
        d = 1 if e.delta > 0 else -1
        if e.state & 0x4:                       # Ctrl：透明度
            self.o["alpha"] = round(_clamp(self.o["alpha"] + d * 0.05, 0.05, 1.0), 2)
        elif e.state & 0x1:                     # Shift：线宽
            self.o["width"] = _clamp(self.o["width"] + d, 2, 140)
        else:                                   # 滚轮：半径
            self.o["radius"] = _clamp(self.o["radius"] + d * 6, 20, 1200)
        self.sync_all()

    def _key(self, e):
        k, st = e.keysym, e.state
        step = 10 if (st & 0x1) else 1
        ox, oy = self.o["offset"]
        if k == "Left":
            self.o["offset"] = [ox - step, oy]
        elif k == "Right":
            self.o["offset"] = [ox + step, oy]
        elif k == "Up":
            self.o["offset"] = [ox, oy - step]
        elif k == "Down":
            self.o["offset"] = [ox, oy + step]
        elif k == "Escape":
            self._quit()
            return "break"
        elif k in ("s", "S"):
            self._save()
            return "break"
        elif k in ("r", "R"):
            self._reset()
            return "break"
        elif k == "Tab":
            self._toggle_mask()
            return "break"
        else:
            return
        self.sync_all()
        return "break"

    def _toggle_mask(self):
        cur = self.mask.attributes("-alpha")
        self.mask.attributes("-alpha", 0.01 if cur > 0.05 else 0.30)
        self._draw_guides()

    # ------------------------------------------------------------ 几何/渲染 #
    def _anchor(self):
        if self.o.get("anchor", "center") == "center":
            return self.sw / 2.0, self.sh / 2.0
        a = self.o.get("anchor_xy", [self.sw / 2, self.sh / 2])
        return float(a[0]), float(a[1])

    def _center(self):
        ax, ay = self._anchor()
        ox, oy = self.o["offset"]
        return ax + ox, ay + oy

    def sync_all(self, guides: bool = True):
        self.ov.reconfigure(self.o)
        # 与运行时同一套判定：0 = 空弹匣不画实心条；低于阈值 = 整条泛红；其余 = 白条。
        self.ov.show_value(self.fraction, 0)
        self._refresh_widgets()
        self.panel.lift()
        if guides:
            self._draw_guides()

    def _refresh_widgets(self):
        self._loading = True
        for key, sc in self.scales.items():
            sc.set(self._get(key))
            lab, fmt = self.valuetext[key]
            lab.config(text=fmt.format(self._get(key)))
        self.frac_scale.set(self.fraction)
        self.frac_text.config(text=f"{self.fraction * 100:.0f}%")
        self._loading = False
        cx, cy = self._center()
        self.info.config(
            text=f"圆心 ({cx:.0f}, {cy:.0f})  偏移 {self.o['offset']}\n"
                 f"半径 {self.o['radius']:.0f}  线宽 {self.o['width']:.0f}  "
                 f"角度 {self.o['ang_start']:.0f}° → {self.o['ang_end']:.0f}°\n"
                 f"屏幕 {self.sw}×{self.sh}   锚点 {self.o.get('anchor', 'center')}")

    def _draw_guides(self):
        c = self.cv
        c.delete("all")
        mx, my = self.sw / 2, self.sh / 2
        c.create_line(mx, my - 26, mx, my + 26, fill="#7fb2ff", width=1)
        c.create_line(mx - 26, my, mx + 26, my, fill="#7fb2ff", width=1)
        c.create_text(mx + 34, my - 14, text="屏幕中心", fill="#7fb2ff", anchor="w",
                      font=("Microsoft YaHei UI", 9))
        cx, cy = self._center()
        c.create_line(cx - 18, cy, cx + 18, cy, fill="#ff9d3c", width=1)
        c.create_line(cx, cy - 18, cx, cy + 18, fill="#ff9d3c", width=1)
        x0, y0, x1, y1 = ArcGeometry(self.screen, self.o).bbox()
        c.create_rectangle(x0, y0, x1, y1, outline="#5a5a68", dash=(4, 4))
        c.create_text(cx, y1 + 14, text=f"r={self.o['radius']:.0f} w={self.o['width']:.0f} "
                                        f"a={self.o['alpha']:.2f}",
                      fill="#ff9d3c", font=("Consolas", 9))

    # ------------------------------------------------------------------ 命令 #
    def _save(self):
        self.raw.setdefault("overlay", {}).update(self.o)
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.raw, f, ensure_ascii=False, indent=2)
        self.baseline = json.dumps(self.o, sort_keys=True)
        self._toast("已保存到 config.json")

    def _reset(self):
        self.o = json.loads(json.dumps(self.initial))
        self.sync_all()
        self._toast("已复位成启动时的参数")

    def _toast(self, msg):
        self.info.config(text=msg + "\n（滑块数值见上方）")
        self.panel.after(1400, self._refresh_widgets)

    def _quit(self):
        if json.dumps(self.o, sort_keys=True) != self.baseline:
            self.mask.attributes("-topmost", False)
            self.panel.attributes("-topmost", False)
            ok = messagebox.askyesnocancel("退出", "有未保存的改动，先保存吗？")
            self.mask.attributes("-topmost", True)
            self.panel.attributes("-topmost", True)
            if ok is None:
                return
            if ok:
                self._save()
        self.ov.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def _selftest():
    """不开 GUI，只验几何与透明度是否生效。"""
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    o = json.loads(json.dumps(cfg["overlay"]))
    g = ArcGeometry((2560, 1440), o)
    print("包围盒", g.bbox(), "圆心", (g.cx, g.cy), "半径", g.r, "线宽", g.w)
    for a in (1.0, 0.5, 0.2):
        im = g.render(0.6, o["fill_rgb"], o["track_rgb"], o["track_alpha"], a)
        lo, hi = im.getchannel("A").getextrema()
        print(f"alpha={a}  尺寸{im.size}  alpha范围 {lo}~{hi}")
    need = ("offset", "radius", "width", "alpha", "ang_start", "ang_end")
    print("配置字段", {k: o.get(k) for k in need})


if __name__ == "__main__":
    if "--check" in sys.argv:
        _selftest()
    else:
        Tuner().run()
