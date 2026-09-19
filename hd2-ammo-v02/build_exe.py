# -*- coding: utf-8 -*-
"""把 v02 打包成单个 exe（PyInstaller）。改完代码跑一次即可。

    python build_exe.py        # 输出 dist/HD2弹药叠加.exe

产物 `dist/HD2弹药叠加.exe`：双击 = 拉 Steam 里的 HD2 + 挂叠加层，游戏退出自动收摊；
带 `--daemon` 可换成「谁把游戏拉起来就跟谁走」的守护模式。

config.json 与 assets/ 打进包里；首次运行会把 config.json 释放到 exe 旁边，
之后手工改 exe 旁边那份就生效（改包里的没用，每次运行都被覆盖）。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "HD2AmmoOverlay"          # 先用 ASCII 名打包，最后再改成中文名
FINAL = "HD2弹药叠加.exe"


def main() -> int:
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile",
        "--windowed",                                   # 不带黑框控制台
        "--name", NAME,
        "--add-data", f"{HERE / 'config.json'};.",
        "--add-data", f"{HERE / 'assets'};assets",
        "--hidden-import", "mss.windows",               # mss 按平台动态导入
        "--exclude-module", "tkinter", "--exclude-module", "matplotlib",
        "--exclude-module", "scipy", "--exclude-module", "pandas",
        "--exclude-module", "PIL.ImageTk",              # 调参器不打包，能省一大截
        "--workpath", str(HERE / "_exe" / "build"),
        "--distpath", str(HERE / "dist"),
        "--specpath", str(HERE / "_exe"),
        str(HERE / "main.py"),
    ]
    print("打包命令：\n  " + " ".join(cmd))
    if subprocess.call(cmd) != 0:
        print("打包失败")
        return 1

    built = HERE / "dist" / f"{NAME}.exe"
    out = HERE / "dist" / FINAL
    if built.is_file() and built != out:
        out.unlink(missing_ok=True)
        shutil.move(str(built), str(out))
    if not out.is_file():
        print("没找到产物，打包过程可能被中断")
        return 1
    print(f"完成：{out}  ({out.stat().st_size / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
