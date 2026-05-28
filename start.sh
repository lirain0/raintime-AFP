#!/bin/bash
echo "=========================================="
echo "   AFP-Predictor V4"
echo "=========================================="
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found."
    exit 1
fi
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "[INFO] Installing dependencies..."
pip install -q -r requirements.txt
if [ ! -f "model/afp_model_augmented.pth" ]; then
    echo "[WARNING] Model file not found."
    exit 1
fi
echo "[SUCCESS] Starting server..."
python app.py
