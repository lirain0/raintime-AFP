#!/usr/bin/env python3
"""
Biological Explanation System - V2
"""

import numpy as np
from collections import Counter
import math


class BioExplanationSystem:
    """Biological Explanation Scoring System"""

    def __init__(self):
        self.aa_properties = {
            'A': {'hydrophobicity': 1.8, 'charge': 0, 'size': 89, 'pI': 6.00},
            'C': {'hydrophobicity': 2.5, 'charge': 0, 'size': 121, 'pI': 5.07},
            'D': {'hydrophobicity': -3.5, 'charge': -1, 'size': 133, 'pI': 2.77},
            'E': {'hydrophobicity': -3.5, 'charge': -1, 'size': 147, 'pI': 3.22},
            'F': {'hydrophobicity': 2.8, 'charge': 0, 'size': 165, 'pI': 5.48},
            'G': {'hydrophobicity': -0.4, 'charge': 0, 'size': 75, 'pI': 5.97},
            'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'size': 155, 'pI': 7.59},
            'I': {'hydrophobicity': 4.5, 'charge': 0, 'size': 131, 'pI': 6.02},
            'K': {'hydrophobicity': -3.9, 'charge': 1, 'size': 146, 'pI': 9.74},
            'L': {'hydrophobicity': 3.8, 'charge': 0, 'size': 131, 'pI': 5.98},
            'M': {'hydrophobicity': 1.9, 'charge': 0, 'size': 149, 'pI': 5.74},
            'N': {'hydrophobicity': -3.5, 'charge': 0, 'size': 132, 'pI': 5.41},
            'P': {'hydrophobicity': -1.6, 'charge': 0, 'size': 115, 'pI': 6.30},
            'Q': {'hydrophobicity': -3.5, 'charge': 0, 'size': 146, 'pI': 5.65},
            'R': {'hydrophobicity': -4.5, 'charge': 1, 'size': 174, 'pI': 10.76},
            'S': {'hydrophobicity': -0.8, 'charge': 0, 'size': 105, 'pI': 5.68},
            'T': {'hydrophobicity': -0.7, 'charge': 0, 'size': 119, 'pI': 5.60},
            'V': {'hydrophobicity': 4.2, 'charge': 0, 'size': 117, 'pI': 5.96},
            'W': {'hydrophobicity': -0.9, 'charge': 0, 'size': 204, 'pI': 5.89},
            'Y': {'hydrophobicity': -1.3, 'charge': 0, 'size': 181, 'pI': 5.66},
        }

    def analyze(self, sequence, model_probability):
        """Analyze sequence"""
        seq = sequence.upper()
        features = {
            'basic': self._analyze_basic(seq),
            'composition': self._analyze_composition(seq),
            'physicochemical': self._analyze_physicochemical(seq),
        }
        confidence = self._evaluate_confidence(seq, model_probability, features)
        explanation = self._generate_explanation(seq, model_probability, features, confidence)
        return {
            'sequence': seq,
            'model_probability': model_probability,
            'confidence_level': confidence['level'],
            'reliability_score': confidence['score'],
            'explanation': explanation,
        }

    def _analyze_basic(self, seq):
        """Basic features"""
        return {
            'length': len(seq),
            'molecular_weight': sum(self.aa_properties[aa]['size'] for aa in seq if aa in self.aa_properties),
            'unique_aa_count': len(set(seq)),
        }

    def _analyze_composition(self, seq):
        """Amino acid composition"""
        aa_count = Counter(seq)
        total = len(seq)
        cationic = sum(aa_count.get(aa, 0) for aa in ['K', 'R', 'H'])
        hydrophobic = sum(aa_count.get(aa, 0) for aa in ['L', 'I', 'V', 'F', 'W', 'M', 'A'])
        return {
            'cationic_ratio': cationic / total if total > 0 else 0,
            'hydrophobic_ratio': hydrophobic / total if total > 0 else 0,
        }

    def _analyze_physicochemical(self, seq):
        """Physicochemical properties"""
        props = [self.aa_properties.get(aa, {}) for aa in seq]
        hydrophobicities = [p.get('hydrophobicity', 0) for p in props]
        charges = [p.get('charge', 0) for p in props]
        return {
            'hydrophobicity_mean': np.mean(hydrophobicities) if hydrophobicities else 0,
            'net_charge': sum(charges),
        }

    def _evaluate_confidence(self, seq, model_prob, features):
        """Evaluate confidence"""
        score = 0
        reasons = []
        model_confidence = max(model_prob, 1 - model_prob)
        if model_confidence >= 0.95:
            score += 40
            reasons.append("High model confidence")
        elif model_confidence >= 0.85:
            score += 30
            reasons.append("Medium-high model confidence")
        elif model_confidence >= 0.70:
            score += 20
        else:
            score += 10
            reasons.append("Low model confidence")

        comp = features['composition']
        if comp['cationic_ratio'] >= 0.15:
            score += 20
            reasons.append("High cationic content")

        if score >= 80:
            level = "A (High)"
        elif score >= 60:
            level = "B (Medium)"
        elif score >= 40:
            level = "C (Low)"
        else:
            level = "D (Very Low)"

        return {'score': score, 'level': level, 'reasons': reasons}

    def _generate_explanation(self, seq, model_prob, features, confidence):
        """Generate explanation"""
        pred_class = "AFP" if model_prob >= 0.57 else "Non-AFP"
        lines = [
            f"Prediction: {pred_class}",
            f"Model Confidence: {model_prob:.1%}",
            f"Reliability: {confidence['level']} (Score: {confidence['score']}/100)",
            "",
            "Key Features:",
            f"- Length: {len(seq)} amino acids",
            f"- Net Charge: +{features['physicochemical']['net_charge']:.1f}",
            f"- Cationic Ratio: {features['composition']['cationic_ratio']:.1%}",
        ]
        return "\n".join(lines)
