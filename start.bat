@echo off
cd /d "%~dp0"
echo Starting AI KB Assistant...
echo (OCR warm-up takes a few seconds, please wait for "Uvicorn running")
.\venv\Scripts\python main.py
echo.
echo Service stopped. You can close this window.
pause
