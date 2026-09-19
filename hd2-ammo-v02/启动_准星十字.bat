@echo off
rem 双击即用：屏幕正中 300x300 识别准星圆环，在环上盖绿色大十字。F2 开关，关窗口即退出。
rem 想调参数就写在后面，例如：启动_准星十字.bat --hz 30 --show-th 0.5
cd /d "%~dp0"
start "" "C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe" "%~dp0crosshair.py" %*
