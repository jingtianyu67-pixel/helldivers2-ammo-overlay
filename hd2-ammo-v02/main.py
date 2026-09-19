# -*- coding: utf-8 -*-
"""打包入口（PyInstaller）：一个 exe 同时当「启动器」和「叠加层」。

    HD2弹药叠加.exe                # 默认：拉 Steam 里的 HD2、挂上叠加层，游戏退出自动收摊
    HD2弹药叠加.exe --daemon       # 守护模式：谁把游戏拉起来就跟谁走（挂机常驻用这个）
    HD2弹药叠加.exe --no-launch    # 不主动拉游戏，等它自己起来
    HD2弹药叠加.exe -- --hz 30     # `--` 之后的参数原样转给叠加层
    HD2弹药叠加.exe --overlay ...  # 只跑叠加层本身（启动器派生自己时用的身份，别手敲）

为什么要这个入口：打包成 exe 后 `sys.executable` 就是 exe 自己，launch.py 没法再拼
「解释器 + live_ammo.py」去派生子进程。于是让同一个 exe 换个参数再起一份，自己派生
自己 —— Job Object 保护、随游戏退出连带关闭这些都照旧。

源码运行不需要它，直接用 launch.py / live_ammo.py 即可。
"""

from __future__ import annotations

import os
import sys

OVERLAY_FLAG = "--overlay"
DAEMON_FLAG = "--daemon"


def _fix_streams() -> None:
    """`--windowed` 打包后从资源管理器双击启动时 sys.stdout/stderr 是 None。

    Python 对 `sys.stdout is None` 的 print 会抛 AttributeError —— 守护模式这种要长跑
    的身份一进来就崩。兜底到 devnull，真正的日志走 logs/。
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            try:
                setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
            except OSError:
                pass


def main() -> int:
    _fix_streams()
    argv = sys.argv[1:]
    if OVERLAY_FLAG in argv:
        sys.argv = sys.argv[:1] + [a for a in argv if a != OVERLAY_FLAG]
        import live_ammo
        return live_ammo.main()
    import launch
    return launch.main()


if __name__ == "__main__":
    sys.exit(main())
