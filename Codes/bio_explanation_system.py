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
            'features': features,
            'explanation': explanation,
        }

    def _analyze_basic(self, seq):
        return {
            'length': len(seq),
            'molecular_weight': sum(self.aa_properties[aa]['size'] for aa in seq if aa in self.aa_properties),
            'unique_aa_count': len(set(seq)),
            'aa_diversity': len(set(seq)) / len(seq) if seq else 0,
        }

    def _analyze_composition(self, seq):
        aa_count = Counter(seq)
        total = len(seq)
        return {
            aa: {'count': count, 'percentage': round(count / total * 100, 2)}
            for aa, count in aa_count.items()
        }

    def _analyze_physicochemical(self, seq):
        hydrophobicity = [self.aa_properties[aa]['hydrophobicity'] for aa in seq if aa in self.aa_properties]
        charge = [self.aa_properties[aa]['charge'] for aa in seq if aa in self.aa_properties]
        return {
            'avg_hydrophobicity': round(np.mean(hydrophobicity), 3) if hydrophobicity else 0,
            'net_charge': round(sum(charge), 2),
            'positive_residues': sum(1 for c in charge if c > 0),
            'negative_residues': sum(1 for c in charge if c < 0),
        }

    def _evaluate_confidence(self, seq, prob, features):
        score = 50
        basic = features['basic']
        if 10 <= basic['length'] <= 50:
            score += 15
        if basic['aa_diversity'] >= 0.3:
            score += 10
        phy = features['physicochemical']
        if phy['net_charge'] > 0:
            score += 10
        if abs(prob - 0.5) > 0.3:
            score += 15
        level = 'A' if score >= 80 else 'B' if score >= 60 else 'C' if score >= 40 else 'D'
        return {'score': min(score, 100), 'level': level}

    def _generate_explanation(self, seq, prob, features, confidence):
        parts = []
        basic = features['basic']
        phy = features['physicochemical']
        parts.append(f"Sequence length: {basic['length']} amino acids")
        parts.append(f"Net charge: {phy['net_charge']:+} (at pH 7.0)")
        parts.append(f"Hydrophobicity: {phy['avg_hydrophobicity']}")
        if confidence['level'] in ['A', 'B']:
            parts.append("High confidence prediction - reliable for screening")
        else:
            parts.append("Moderate confidence - experimental validation recommended")
        return "; ".join(parts)
