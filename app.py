#!/usr/bin/env python3
"""
AFP-Predictor V4 - Deployment Version
Single-file deployment with all features
"""

import os
import sys
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, session
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Codes'))
from model import AFPTransformerGNN
from features import sequence_to_data
from bio_explanation_system import BioExplanationSystem

app = Flask(__name__)
app.config['SECRET_KEY'] = 'afp-predictor-v4-deploy-key'

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'afp_model_augmented.pth')
THRESHOLD = 0.57
DEVICE = torch.device('cpu')

model = None
explainer = BioExplanationSystem()


def load_model():
    """Load model"""
    global model
    model = AFPTransformerGNN(
        input_dim=26, hidden_dim=128, num_heads=8,
        num_gat_layers=3, num_classes=2, dropout=0.3
    ).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("Model loaded successfully")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Please place the model file at: model/afp_model_augmented.pth")


def predict_single(sequence):
    """Predict single sequence"""
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    sequence = sequence.strip().upper()

    if not sequence or not all(aa in valid_aa for aa in sequence):
        return None

    data = sequence_to_data(sequence)
    data = data.to(DEVICE)

    with torch.no_grad():
        output = model(data)
        probs = F.softmax(output, dim=1)
        afp_prob = probs[0][1].item()

    is_afp = afp_prob >= THRESHOLD
    confidence = max(afp_prob, 1 - afp_prob)

    if confidence >= 0.95:
        confidence_level = "A (High)"
    elif confidence >= 0.85:
        confidence_level = "B (Medium)"
    elif confidence >= 0.70:
        confidence_level = "C (Low)"
    else:
        confidence_level = "D (Very Low)"

    bio_analysis = explainer.analyze(sequence, afp_prob)

    return {
        'sequence': sequence,
        'afp_probability': round(afp_prob * 100, 2),
        'is_afp': is_afp,
        'confidence_level': confidence_level,
        'confidence_score': round(confidence * 100, 2),
        'length': len(sequence),
        'bio_explanation': bio_analysis['explanation'],
        'reliability_score': bio_analysis['reliability_score'],
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip()

        result = predict_single(sequence)
        if result is None:
            return jsonify({'error': 'Invalid sequence. Only 20 standard amino acids allowed.'}), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        sequences = data.get('sequences', [])

        results = []
        for seq in sequences:
            result = predict_single(seq)
            if result:
                results.append(result)

        return jsonify({'results': results, 'count': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def status():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'threshold': THRESHOLD,
        'device': str(DEVICE)
    })


if __name__ == '__main__':
    print("=" * 60)
    print("AFP-Predictor V4 - Deployment Version")
    print("=" * 60)

    try:
        load_model()
        print("System ready")
        print("=" * 60)
    except Exception as e:
        print(f"Warning: {e}")
        print("System will start without model loaded")

    print("\nStarting server...")
    print("Access: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
