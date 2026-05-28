@echo off
title AFP-Predictor V4
echo ==========================================
echo    AFP-Predictor V4
echo ==========================================
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    pause
    exit /b 1
)
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo [INFO] Installing dependencies...
pip install -q -r requirements.txt
if not exist "model\afp_model_augmented.pth" (
    echo [WARNING] Model file not found.
    pause
    exit /b 1
)
echo [SUCCESS] Starting server...
python app.py
pause
