# AFP-Predictor V4 - Deployment Package

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser.

## File Structure

```
AFP-Predictor-Deploy/
├── app.py                          # Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── model/                          # Model directory
│   └── afp_model_augmented.pth    # V4 model weights (place here)
├── Codes/                          # Core modules
│   ├── model.py                   # GAT+Transformer model
│   ├── features.py                # Feature extraction
│   └── bio_explanation_system.py  # Biological explanation
└── templates/
    └── index.html                 # Web interface
```

## Model File

Please place the trained model file at:
```
model/afp_model_augmented.pth
```

## API Endpoints

- `GET /` - Web interface
- `POST /predict` - Single sequence prediction
- `POST /predict_batch` - Batch prediction
- `GET /api/status` - System status

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Flask 2.0+
- torch-geometric 2.3+
