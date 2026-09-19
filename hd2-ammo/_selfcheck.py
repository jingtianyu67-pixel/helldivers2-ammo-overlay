# -*- coding: utf-8 -*-
"""自检：用 AmmoDetector 跑一遍已存的样张，确认模块输出与标定结果一致。"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ammo_detect import AmmoDetector

SHOTS = Path(r"C:\Users\Administrator\Desktop\hd2_ammo_shots")
cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
det = AmmoDetector(margin=cfg["white_margin"], full_ref=cfg.get("full_ref"),
                   empty_ref=cfg.get("empty_ref", 0))
print(f"窗口 {det.area} px | 归一化 {det.empty_ref}~{det.full_ref} | 掩膜 {det.mask.shape}")
print(f"{'样张':<10}{'百分比':>9}{'填充':>7}{'白':>6}{'红':>6}  状态")
print("-" * 46)
for f in sorted(SHOTS.glob("*_1x.png")):
    tag = f.stem.split("_")[1]
    rgb = np.asarray(Image.open(f).convert("RGB"))
    rd = det.analyze(rgb)
    print(f"{tag:<10}{rd.fraction * 100:>8.1f}%{rd.fill:>7}{rd.white_px:>6}"
          f"{rd.red_px:>6}  {rd.state}{'' if rd.valid else '  <无效>'}")
