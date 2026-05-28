# AFP-Predictor V4

Antifungal Peptide Prediction Platform based on Deep Learning

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## Features

- Single sequence prediction
- Batch prediction
- Biological explanation
- Confidence assessment

## Model

- Architecture: GAT + Transformer
- Accuracy: 95.52%
- Input: Amino acid sequence
- Output: AFP probability

## API Endpoints

- `POST /predict` - Single sequence prediction
- `POST /predict_batch` - Batch prediction
- `GET /health` - Health check
