@echo off
rem 双击即用：拉起 HD2 + 挂弹药叠加层，游戏一关叠加层跟着关。
rem 想加参数就写在后面，例如：启动_HD2叠加.bat --log
cd /d "%~dp0"
start "" "C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\pythonw.exe" "%~dp0launch.py" %*
