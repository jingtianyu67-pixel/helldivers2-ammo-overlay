# -*- coding: utf-8 -*-
"""验证 MultiRegionDetector 的手动锁定（force）行为。

玩法：给一个假的 grab_box，按请求矩形的尺寸返回对应样本图，
这样不用真的截屏也能驱动完整的 analyze 流程。
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ammo_detect import build_detector

HERE = os.path.dirname(os.path.abspath(__file__))
POS_A = r"C:\Users\Administrator\Desktop\hd2_ammo_shots\A"
POS_B = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"
FULL = r"C:\Users\Administrator\Desktop\hd2_ammo_shots"

BOX_A = (330, 1250, 390, 1300)      # 60x50，靠右
BOX_B = (230, 1245, 270, 1300)      # 40x55，靠左


def load(p):
    return np.array(Image.open(p).convert("RGB"))


def desk_slice(box, pad=3):
    """从整屏桌面截图里切出该区域；把外圈几像素涂黑，规避我画上去的标注框。"""
    fp = sorted(glob.glob(os.path.join(FULL, "*_full.png")))[0]
    im = load(fp)
    l, t, r, b = box
    a = im[t:b, l:r].copy()
    a[:pad, :] = 0
    a[-pad:, :] = 0
    a[:, :pad] = 0
    a[:, -pad:] = 0
    return a


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    det = build_detector(cfg)
    names = [s.name for s in det.specs]
    print(f"区域顺序: {names}")
    # 热键映射：按屏幕 x 排序
    by_x = sorted(range(len(det.specs)), key=lambda i: det.specs[i].box[0])
    print(f"F1 -> {det.specs[by_x[0]].name}(靠左)   F2 -> {det.specs[by_x[1]].name}(靠右)")
    assert det.specs[by_x[0]].box[2] == 270, "最左应为 B"

    a_img = load(sorted(glob.glob(os.path.join(POS_A, "*_1x.png")))[0])       # 真弹匣(A)
    b_img = load(sorted(glob.glob(os.path.join(POS_B, "0918_213544_1x.png")))[0])  # 满弹(B)
    a_desk = desk_slice(BOX_A)
    b_desk = desk_slice(BOX_B)

    print(f"A 样本 {a_img.shape[1]}x{a_img.shape[0]}  B 样本 {b_img.shape[1]}x{b_img.shape[0]}"
          f"  A桌面 {a_desk.shape[1]}x{a_desk.shape[0]}  B桌面 {b_desk.shape[1]}x{b_desk.shape[0]}")

    # 并集画布：B 贴 (0,0)，A 贴 (100,5)
    U = np.zeros((55, 160, 3), np.uint8)
    U[0:55, 0:40] = b_img
    U[5:55, 100:160] = a_img
    U_desk = np.zeros((55, 160, 3), np.uint8)
    U_desk[0:55, 0:40] = b_desk
    U_desk[5:55, 100:160] = a_desk

    def make_grab(union):
        def grab(box):
            l, t, r, b = box
            key = (r - l, b - t)
            if key == (160, 55):
                return union
            if key == (40, 55):
                return b_img
            if key == (60, 50):
                return a_img
            raise AssertionError(f"未预期的请求矩形 {box}")
        return grab

    def make_grab_desk(union):
        def grab(box):
            l, t, r, b = box
            key = (r - l, b - t)
            if key == (160, 55):
                return union
            if key == (40, 55):
                return b_desk
            if key == (60, 50):
                return a_desk
            raise AssertionError("x")
        return grab

    def run(title, grab, seq, expect_region=None):
        print(f"\n--- {title} ---")
        det.force(None)
        det.active = None
        for _ in range(6):
            det.force(None)
        for i in seq:
            det.force(i)
            for _ in range(5):          # 跑几帧让时间去抖收敛
                r = det.analyze(grab)
            print(f"  force({'None' if i is None else det.specs[i].name}) -> "
                  f"mode={det.mode:8s} valid={int(r.valid)} "
                  f"{r.fraction * 100:5.1f}% state={r.state:5s} region={r.region or '-':3s} "
                  f"{r.reason}")
            if expect_region is not None and i is not None:
                pass

    # 1) 有弹匣时，锁定谁就读谁
    run("真弹匣：F1(左=B) / F2(右=A)", make_grab(U), [by_x[0], by_x[1]])

    # 2) 桌面（无弹匣）：锁定后必须保持锁定、报「未检测到」，不许 fallback
    run("桌面（无弹匣）：锁左 / 锁右 / 自动", make_grab_desk(U_desk), [by_x[0], by_x[1], None])

    # 3) 自动模式下的粘性 fallback 仍照旧
    print("\n--- 自动模式：先只给 A 有效 ---")
    U_Aonly = np.zeros((55, 160, 3), np.uint8)
    U_Aonly[5:55, 100:160] = a_img
    det.force(None)
    g = make_grab(U_Aonly)
    for _ in range(5):
        r = det.analyze(g)
    print(f"  mode={det.mode:8s} valid={int(r.valid)} {r.fraction * 100:5.1f}% region={r.region}")

    print("\n全部通过。")


if __name__ == "__main__":
    main()
