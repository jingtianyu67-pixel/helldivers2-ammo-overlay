@echo off
rem 弧线调参器：双击启动。需要带 tkinter 的 Python（本机 E:\ai\python）。
cd /d "%~dp0"
set "Pyw=E:\ai\python\pythonw.exe"
if not exist "%Pyw%" set "Pyw=pythonw"
start "" "%Pyw%" "%~dp0tune_arc.py"
