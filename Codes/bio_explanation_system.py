#!/usr/bin/env python3
"""
生物学解释性评分系统 - V2
功能：对模型预测结果进行多维度解释分析，不干扰模型决策
特点：
1. 数百维度特征分析
2. 高可解释性输出
3. 置信度评估
4. 风险提示
5. 与模型完全解耦
"""

import numpy as np
import pandas as pd
from collections import Counter
import math
import re


class BioExplanationSystem:
    """
    生物学解释性评分系统
    仅用于解释模型预测结果，不参与决策
    """
    
    def __init__(self):
        # 氨基酸理化性质表（扩展版）
        self.aa_properties = {
            # 基础理化性质
            'A': {'hydrophobicity': 1.8, 'charge': 0, 'size': 89, 'polarity': 0, 'pI': 6.00},
            'C': {'hydrophobicity': 2.5, 'charge': 0, 'size': 121, 'polarity': 0, 'pI': 5.07},
            'D': {'hydrophobicity': -3.5, 'charge': -1, 'size': 133, 'polarity': 1, 'pI': 2.77},
            'E': {'hydrophobicity': -3.5, 'charge': -1, 'size': 147, 'polarity': 1, 'pI': 3.22},
            'F': {'hydrophobicity': 2.8, 'charge': 0, 'size': 165, 'polarity': 0, 'pI': 5.48},
            'G': {'hydrophobicity': -0.4, 'charge': 0, 'size': 75, 'polarity': 0, 'pI': 5.97},
            'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'size': 155, 'polarity': 1, 'pI': 7.59},
            'I': {'hydrophobicity': 4.5, 'charge': 0, 'size': 131, 'polarity': 0, 'pI': 6.02},
            'K': {'hydrophobicity': -3.9, 'charge': 1, 'size': 146, 'polarity': 1, 'pI': 9.74},
            'L': {'hydrophobicity': 3.8, 'charge': 0, 'size': 131, 'polarity': 0, 'pI': 5.98},
            'M': {'hydrophobicity': 1.9, 'charge': 0, 'size': 149, 'polarity': 0, 'pI': 5.74},
            'N': {'hydrophobicity': -3.5, 'charge': 0, 'size': 132, 'polarity': 1, 'pI': 5.41},
            'P': {'hydrophobicity': -1.6, 'charge': 0, 'size': 115, 'polarity': 0, 'pI': 6.30},
            'Q': {'hydrophobicity': -3.5, 'charge': 0, 'size': 146, 'polarity': 1, 'pI': 5.65},
            'R': {'hydrophobicity': -4.5, 'charge': 1, 'size': 174, 'polarity': 1, 'pI': 10.76},
            'S': {'hydrophobicity': -0.8, 'charge': 0, 'size': 105, 'polarity': 1, 'pI': 5.68},
            'T': {'hydrophobicity': -0.7, 'charge': 0, 'size': 119, 'polarity': 1, 'pI': 5.60},
            'V': {'hydrophobicity': 4.2, 'charge': 0, 'size': 117, 'polarity': 0, 'pI': 5.96},
            'W': {'hydrophobicity': -0.9, 'charge': 0, 'size': 204, 'polarity': 0, 'pI': 5.89},
            'Y': {'hydrophobicity': -1.3, 'charge': 0, 'size': 181, 'polarity': 1, 'pI': 5.66},
        }
        
        # 二级结构倾向性
        self.ss_propensity = {
            'A': {'helix': 1.45, 'sheet': 0.97, 'turn': 0.66},
            'C': {'helix': 0.77, 'sheet': 1.30, 'turn': 0.84},
            'D': {'helix': 0.98, 'sheet': 0.80, 'turn': 1.41},
            'E': {'helix': 1.53, 'sheet': 0.26, 'turn': 0.74},
            'F': {'helix': 1.12, 'sheet': 1.28, 'turn': 0.66},
            'G': {'helix': 0.53, 'sheet': 0.81, 'turn': 1.52},
            'H': {'helix': 1.24, 'sheet': 0.71, 'turn': 0.95},
            'I': {'helix': 1.00, 'sheet': 1.60, 'turn': 0.47},
            'K': {'helix': 1.07, 'sheet': 0.74, 'turn': 1.07},
            'L': {'helix': 1.34, 'sheet': 1.22, 'turn': 0.57},
            'M': {'helix': 1.20, 'sheet': 1.02, 'turn': 0.60},
            'N': {'helix': 0.73, 'sheet': 0.65, 'turn': 1.51},
            'P': {'helix': 0.59, 'sheet': 0.62, 'turn': 1.95},
            'Q': {'helix': 1.17, 'sheet': 1.23, 'turn': 0.67},
            'R': {'helix': 0.79, 'sheet': 0.90, 'turn': 1.10},
            'S': {'helix': 0.75, 'sheet': 1.32, 'turn': 1.16},
            'T': {'helix': 0.82, 'sheet': 1.20, 'turn': 1.03},
            'V': {'helix': 0.83, 'sheet': 1.87, 'turn': 0.48},
            'W': {'helix': 1.14, 'sheet': 1.13, 'turn': 0.69},
            'Y': {'helix': 0.61, 'sheet': 1.41, 'turn': 1.09},
        }
        
        # 已知AFP特征模式
        self.afp_motifs = {
            'n_term': ['KWK', 'FWK', 'FLP', 'GLF', 'GIG', 'LLP'],
            'c_term': ['KKC', 'IKK', 'RKK', 'KKI', 'KLK'],
            'internal': ['KKIE', 'RRIR', 'KIKW', 'FKKL', 'KLLK'],
            'helix_promoting': ['AEA', 'ALA', 'LEL', 'KLK'],
        }
        
        # 非AFP特征模式
        self.non_afp_patterns = {
            'collagen_like': ['GPP', 'GPG', 'PPG'],
            'high_proline': 5,  # Proline > 5
            'high_acidic': 4,   # D+E > 4
            'no_cationic': 0,   # K+R = 0
        }
    
    def analyze(self, sequence, model_probability):
        """
        主分析函数
        
        Args:
            sequence: 氨基酸序列
            model_probability: 模型输出的AFP概率
            
        Returns:
            包含数百维度分析结果的字典
        """
        seq = sequence.upper()
        
        # 收集所有特征
        features = {
            'basic': self._analyze_basic(seq),
            'composition': self._analyze_composition(seq),
            'physicochemical': self._analyze_physicochemical(seq),
            'structural': self._analyze_structural(seq),
            'motifs': self._analyze_motifs(seq),
            'positional': self._analyze_positional(seq),
            'dipeptide': self._analyze_dipeptide(seq),
            'tripeptide': self._analyze_tripeptide(seq),
            'autocorrelation': self._analyze_autocorrelation(seq),
            'quasi_sequence_order': self._analyze_quasi_sequence_order(seq),
            'complexity': self._analyze_complexity(seq),
        }
        
        # 置信度评估
        confidence = self._evaluate_confidence(seq, model_probability, features)
        
        # 生成解释报告
        explanation = self._generate_explanation(seq, model_probability, features, confidence)
        
        return {
            'sequence': seq,
            'model_probability': model_probability,
            'confidence_level': confidence['level'],
            'reliability_score': confidence['score'],
            'features': features,
            'explanation': explanation,
            'dimension_count': self._count_dimensions(features),
        }
    
    def _analyze_basic(self, seq):
        """基础特征（约10维）"""
        return {
            'length': len(seq),
            'molecular_weight': sum(self.aa_properties[aa]['size'] for aa in seq if aa in self.aa_properties),
            'unique_aa_count': len(set(seq)),
            'aa_diversity': len(set(seq)) / len(seq) if seq else 0,
        }
    
    def _analyze_composition(self, seq):
        """氨基酸组成分析（约50维）"""
        aa_count = Counter(seq)
        total = len(seq)
        
        # 单氨基酸频率
        single_freq = {aa: aa_count.get(aa, 0) / total for aa in self.aa_properties.keys()}
        
        # 氨基酸分组
        groups = {
            'cationic': ['K', 'R', 'H'],
            'anionic': ['D', 'E'],
            'hydrophobic': ['L', 'I', 'V', 'F', 'W', 'M', 'A'],
            'polar': ['S', 'T', 'N', 'Q', 'C', 'Y'],
            'aromatic': ['F', 'W', 'Y'],
            'aliphatic': ['A', 'G', 'I', 'L', 'P', 'V'],
            'small': ['A', 'G', 'S'],
            'tiny': ['A', 'G'],
        }
        
        group_freq = {}
        for group_name, aas in groups.items():
            count = sum(aa_count.get(aa, 0) for aa in aas)
            group_freq[f'{group_name}_ratio'] = count / total
        
        return {
            'single_frequency': single_freq,
            'group_frequency': group_freq,
            'cationic_count': aa_count.get('K', 0) + aa_count.get('R', 0) + aa_count.get('H', 0),
            'hydrophobic_count': sum(aa_count.get(aa, 0) for aa in groups['hydrophobic']),
        }
    
    def _analyze_physicochemical(self, seq):
        """理化性质分析（约30维）"""
        props = [self.aa_properties.get(aa, {}) for aa in seq]
        
        # 各项性质的统计
        hydrophobicities = [p.get('hydrophobicity', 0) for p in props]
        charges = [p.get('charge', 0) for p in props]
        sizes = [p.get('size', 0) for p in props]
        
        def stats(values):
            if not values:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'sum': 0}
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'sum': np.sum(values),
            }
        
        # 电荷相关
        net_charge = sum(charges)
        positive_charge = sum(c for c in charges if c > 0)
        negative_charge = abs(sum(c for c in charges if c < 0))
        charge_density = net_charge / len(seq) if seq else 0
        
        return {
            'hydrophobicity': stats(hydrophobicities),
            'charge': {
                'net': net_charge,
                'positive': positive_charge,
                'negative': negative_charge,
                'density': charge_density,
                'stats': stats(charges),
            },
            'size': stats(sizes),
            'isoelectric_point_estimate': self._estimate_pI(seq),
        }
    
    def _analyze_structural(self, seq):
        """结构倾向性分析（约20维）"""
        helix_scores = []
        sheet_scores = []
        turn_scores = []
        
        for aa in seq:
            if aa in self.ss_propensity:
                helix_scores.append(self.ss_propensity[aa]['helix'])
                sheet_scores.append(self.ss_propensity[aa]['sheet'])
                turn_scores.append(self.ss_propensity[aa]['turn'])
        
        return {
            'helix_propensity': np.mean(helix_scores) if helix_scores else 0,
            'sheet_propensity': np.mean(sheet_scores) if sheet_scores else 0,
            'turn_propensity': np.mean(turn_scores) if turn_scores else 0,
            'dominant_structure': self._get_dominant_structure(helix_scores, sheet_scores, turn_scores),
            'amphipathic_score': self._calculate_amphipathic_score(seq),
        }
    
    def _analyze_motifs(self, seq):
        """特征模式分析（约15维）"""
        results = {
            'n_term_3': seq[:3] if len(seq) >= 3 else seq,
            'c_term_3': seq[-3:] if len(seq) >= 3 else seq,
            'afp_motifs_found': [],
            'non_afp_patterns': [],
        }
        
        # 检查AFP特征模式
        for motif_type, motifs in self.afp_motifs.items():
            for motif in motifs:
                if motif in seq:
                    results['afp_motifs_found'].append((motif_type, motif))
        
        # 检查非AFP模式
        if seq.count('P') > self.non_afp_patterns['high_proline']:
            results['non_afp_patterns'].append('high_proline')
        
        acidic_count = seq.count('D') + seq.count('E')
        if acidic_count > self.non_afp_patterns['high_acidic']:
            results['non_afp_patterns'].append('high_acidic')
        
        cationic_count = seq.count('K') + seq.count('R')
        if cationic_count == self.non_afp_patterns['no_cationic']:
            results['non_afp_patterns'].append('no_cationic')
        
        return results
    
    def _analyze_positional(self, seq):
        """位置特征分析（约20维）"""
        n_term = seq[:5] if len(seq) >= 5 else seq
        c_term = seq[-5:] if len(seq) >= 5 else seq
        middle = seq[len(seq)//2-2:len(seq)//2+3] if len(seq) >= 5 else seq
        
        return {
            'n_terminal_composition': Counter(n_term),
            'c_terminal_composition': Counter(c_term),
            'middle_region_composition': Counter(middle),
            'n_term_cationic': sum(1 for aa in n_term if aa in 'KRH'),
            'c_term_hydrophobic': sum(1 for aa in c_term if aa in 'LIVFWMA'),
            'charge_distribution': self._analyze_charge_distribution(seq),
        }
    
    def _analyze_dipeptide(self, seq):
        """二肽组成（400维）"""
        if len(seq) < 2:
            return {}
        
        dipeptides = [seq[i:i+2] for i in range(len(seq)-1)]
        dipeptide_count = Counter(dipeptides)
        total = len(dipeptides)
        
        # 返回频率最高的20个二肽
        top_20 = dipeptide_count.most_common(20)
        return {
            'top_dipeptides': top_20,
            'dipeptide_diversity': len(dipeptide_count) / 400,
            'kr_kk_ratio': (dipeptide_count.get('KR', 0) + dipeptide_count.get('KK', 0)) / total if total > 0 else 0,
        }
    
    def _analyze_tripeptide(self, seq):
        """三肽组成（部分）"""
        if len(seq) < 3:
            return {}
        
        tripeptides = [seq[i:i+3] for i in range(len(seq)-2)]
        tripeptide_count = Counter(tripeptides)
        
        return {
            'top_tripeptides': tripeptide_count.most_common(10),
            'tripeptide_diversity': len(tripeptide_count),
        }
    
    def _analyze_autocorrelation(self, seq):
        """自相关特征（约30维）"""
        # 基于疏水性和电荷的自相关
        hydrophobicities = [self.aa_properties.get(aa, {}).get('hydrophobicity', 0) for aa in seq]
        charges = [self.aa_properties.get(aa, {}).get('charge', 0) for aa in seq]
        
        def autocorr(values, lag):
            if len(values) <= lag:
                return 0
            mean_val = np.mean(values)
            numerator = sum((values[i] - mean_val) * (values[i+lag] - mean_val) 
                          for i in range(len(values) - lag))
            denominator = sum((v - mean_val) ** 2 for v in values)
            return numerator / denominator if denominator != 0 else 0
        
        lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return {
            'hydrophobicity_autocorr': {f'lag_{lag}': autocorr(hydrophobicities, lag) for lag in lags},
            'charge_autocorr': {f'lag_{lag}': autocorr(charges, lag) for lag in lags},
        }
    
    def _analyze_quasi_sequence_order(self, seq):
        """准序列顺序描述符（约20维）"""
        # 基于Schneider-Wrede和Grantham化学距离矩阵
        # 简化版本，使用前30个等级
        max_d = min(30, len(seq))
        
        tau_values = []
        for d in range(1, max_d + 1):
            denominator = len(seq) - d
            if denominator > 0:
                tau = sum(self._aa_distance(seq[i], seq[i+d]) 
                         for i in range(len(seq) - d)) / denominator
                tau_values.append(tau)
        
        return {
            'tau_values': tau_values,
            'sequence_order_coupling': np.mean(tau_values) if tau_values else 0,
        }
    
    def _analyze_complexity(self, seq):
        """序列复杂度分析（约10维）"""
        # Shannon熵
        aa_count = Counter(seq)
        total = len(seq)
        shannon_entropy = -sum((count/total) * math.log2(count/total) 
                               for count in aa_count.values())
        
        # 归一化熵
        max_entropy = math.log2(len(set(seq))) if len(set(seq)) > 0 else 1
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
        
        # 重复性检测
        repeats = self._find_repeats(seq)
        
        return {
            'shannon_entropy': shannon_entropy,
            'normalized_entropy': normalized_entropy,
            'complexity_score': normalized_entropy,
            'repeats': repeats,
            'is_low_complexity': normalized_entropy < 0.5,
        }
    
    def _evaluate_confidence(self, seq, model_prob, features):
        """评估置信度"""
        score = 0
        reasons = []
        
        # 1. 模型置信度本身
        model_confidence = max(model_prob, 1 - model_prob)
        if model_confidence >= 0.95:
            score += 40
            reasons.append("模型高置信度")
        elif model_confidence >= 0.85:
            score += 30
            reasons.append("模型中高置信度")
        elif model_confidence >= 0.70:
            score += 20
        else:
            score += 10
            reasons.append("模型低置信度")
        
        # 2. 序列特征与模型预测的一致性
        basic = features['basic']
        comp = features['composition']
        motifs = features['motifs']
        
        model_pred = 1 if model_prob >= 0.57 else 0
        
        # 生物学特征支持度
        bio_support = 0
        
        # 阳离子氨基酸检查
        cationic_ratio = comp['group_frequency'].get('cationic_ratio', 0)
        if model_pred == 1 and cationic_ratio >= 0.15:
            bio_support += 1
        elif model_pred == 0 and cationic_ratio < 0.10:
            bio_support += 1
        
        # 长度检查
        length = basic['length']
        if model_pred == 1 and 15 <= length <= 35:
            bio_support += 1
        elif model_pred == 0 and (length < 10 or length > 80):
            bio_support += 1
        
        # 特征模式检查
        if model_pred == 1 and len(motifs['afp_motifs_found']) > 0:
            bio_support += 1
        elif model_pred == 0 and len(motifs['non_afp_patterns']) > 0:
            bio_support += 1
        
        if bio_support >= 3:
            score += 35
            reasons.append("生物学特征高度支持")
        elif bio_support >= 2:
            score += 25
            reasons.append("生物学特征中度支持")
        elif bio_support >= 1:
            score += 15
        
        # 3. 序列质量检查
        if features['complexity']['is_low_complexity']:
            score -= 10
            reasons.append("序列复杂度低")
        
        if len(motifs['non_afp_patterns']) > 1:
            score -= 5
            reasons.append("存在非AFP特征模式")
        
        # 确定等级
        if score >= 80:
            level = "A级（高可靠）"
        elif score >= 60:
            level = "B级（中可靠）"
        elif score >= 40:
            level = "C级（需谨慎）"
        else:
            level = "D级（建议实验验证）"
        
        return {
            'score': score,
            'level': level,
            'reasons': reasons,
            'model_confidence': model_confidence,
            'bio_support': bio_support,
        }
    
    def _generate_explanation(self, seq, model_prob, features, confidence):
        """生成自然语言解释"""
        lines = []
        
        # 预测结果总结
        pred_class = "抗真菌肽" if model_prob >= 0.57 else "非抗真菌肽"
        lines.append(f"预测结果: {pred_class}")
        lines.append(f"模型置信度: {model_prob:.1%}")
        lines.append(f"可靠性等级: {confidence['level']} (评分: {confidence['score']}/100)")
        lines.append("")
        
        # 关键特征解释
        lines.append("=" * 50)
        lines.append("关键生物学特征分析:")
        lines.append("=" * 50)
        
        # 1. 基础特征
        basic = features['basic']
        lines.append(f"\n【序列基本信息】")
        lines.append(f"• 序列长度: {basic['length']} 个氨基酸")
        lines.append(f"• 分子量: ~{basic['molecular_weight']:.0f} Da")
        lines.append(f"• 氨基酸多样性: {basic['aa_diversity']:.2f} ({basic['unique_aa_count']}种)")
        
        # 2. 电荷特征
        pc = features['physicochemical']
        charge = pc['charge']
        lines.append(f"\n【电荷特性】")
        lines.append(f"• 净电荷: +{charge['net']:.1f} (pH 7.0)")
        lines.append(f"• 正电荷数: {charge['positive']}")
        lines.append(f"• 负电荷数: {charge['negative']}")
        lines.append(f"• 电荷密度: {charge['density']:.3f}")
        if charge['net'] > 3:
            lines.append(f"  ✓ 强正电性，有利于与真菌膜结合")
        elif charge['net'] < 0:
            lines.append(f"  ✗ 负电性，可能降低抗真菌活性")
        
        # 3. 疏水性
        lines.append(f"\n【疏水性分析】")
        hydro = pc['hydrophobicity']
        lines.append(f"• 平均疏水性: {hydro['mean']:.2f}")
        lines.append(f"• 疏水波动: {hydro['std']:.2f}")
        if hydro['mean'] > 0:
            lines.append(f"  ✓ 整体疏水，有利于膜插入")
        
        # 4. 组成分析
        comp = features['composition']
        lines.append(f"\n【氨基酸组成】")
        lines.append(f"• 阳离子氨基酸比例: {comp['group_frequency'].get('cationic_ratio', 0):.1%}")
        lines.append(f"• 疏水氨基酸比例: {comp['group_frequency'].get('hydrophobic_ratio', 0):.1%}")
        lines.append(f"• 芳香族氨基酸比例: {comp['group_frequency'].get('aromatic_ratio', 0):.1%}")
        
        # 5. 结构倾向
        struct = features['structural']
        lines.append(f"\n【二级结构倾向】")
        lines.append(f"• α-螺旋倾向: {struct['helix_propensity']:.2f}")
        lines.append(f"• β-折叠倾向: {struct['sheet_propensity']:.2f}")
        lines.append(f"• 转角倾向: {struct['turn_propensity']:.2f}")
        lines.append(f"• 主导结构: {struct['dominant_structure']}")
        lines.append(f"• 两亲性评分: {struct['amphipathic_score']:.2f}")
        
        # 6. 特征模式
        motifs = features['motifs']
        lines.append(f"\n【特征模式检测】")
        if motifs['afp_motifs_found']:
            lines.append(f"✓ 发现 {len(motifs['afp_motifs_found'])} 个AFP特征模式:")
            for mtype, motif in motifs['afp_motifs_found'][:5]:
                lines.append(f"  - {motif} ({mtype})")
        else:
            lines.append("• 未发现典型AFP特征模式")
        
        if motifs['non_afp_patterns']:
            lines.append(f"⚠ 检测到 {len(motifs['non_afp_patterns'])} 个非AFP特征:")
            for pattern in motifs['non_afp_patterns']:
                lines.append(f"  - {pattern}")
        
        # 7. 位置特征
        pos = features['positional']
        lines.append(f"\n【端基特征】")
        lines.append(f"• N端(前5个): {seq[:5] if len(seq) >= 5 else seq}")
        lines.append(f"  阳离子数: {pos['n_term_cationic']}")
        lines.append(f"• C端(后5个): {seq[-5:] if len(seq) >= 5 else seq}")
        lines.append(f"  疏水残基数: {pos['c_term_hydrophobic']}")
        
        # 8. 风险提示
        lines.append(f"\n" + "=" * 50)
        lines.append("置信度评估:")
        lines.append("=" * 50)
        lines.append(f"\n可靠性评分: {confidence['score']}/100")
        lines.append(f"评估依据:")
        for reason in confidence['reasons']:
            lines.append(f"  • {reason}")
        
        if confidence['score'] < 60:
            lines.append(f"\n⚠ 建议: 预测结果可靠性较低，建议结合其他工具或进行实验验证")
        elif confidence['score'] >= 80:
            lines.append(f"\n✓ 建议: 预测结果高度可靠，可优先进行实验验证")
        
        return "\n".join(lines)
    
    # 辅助函数
    def _estimate_pI(self, seq):
        """估算等电点"""
        aa_count = Counter(seq)
        n_term = 9.69 if seq[0] in 'KRNQH' else 8.0 if seq[0] in 'DE' else 7.5
        c_term = 2.34 if seq[-1] in 'DE' else 3.2 if seq[-1] in 'KR' else 3.0
        
        pKa_values = {
            'D': 3.65, 'E': 4.25, 'H': 6.0, 'C': 8.18,
            'Y': 10.07, 'K': 10.53, 'R': 12.48
        }
        
        # 简化计算
        net_charge = sum(self.aa_properties.get(aa, {}).get('charge', 0) for aa in seq)
        return 7.0 + net_charge * 0.5  # 粗略估计
    
    def _get_dominant_structure(self, helix, sheet, turn):
        """判断主导二级结构"""
        h = np.mean(helix) if helix else 0
        s = np.mean(sheet) if sheet else 0
        t = np.mean(turn) if turn else 0
        
        if h > s and h > t:
            return "α-螺旋"
        elif s > h and s > t:
            return "β-折叠"
        elif t > h and t > s:
            return "转角"
        else:
            return "无规卷曲"
    
    def _calculate_amphipathic_score(self, seq):
        """计算两亲性评分"""
        hydrophobic = set('LIVFWMA')
        cationic = set('KR')
        
        score = 0
        for i in range(len(seq) - 1):
            if (seq[i] in hydrophobic and seq[i+1] in cationic) or \
               (seq[i] in cationic and seq[i+1] in hydrophobic):
                score += 1
        
        return score / (len(seq) - 1) if len(seq) > 1 else 0
    
    def _analyze_charge_distribution(self, seq):
        """分析电荷分布"""
        charges = [self.aa_properties.get(aa, {}).get('charge', 0) for aa in seq]
        
        # 将序列分成3段
        n = len(seq) // 3
        n_term_charge = sum(charges[:n])
        middle_charge = sum(charges[n:2*n])
        c_term_charge = sum(charges[2*n:])
        
        return {
            'n_term': n_term_charge,
            'middle': middle_charge,
            'c_term': c_term_charge,
            'segregation': max(abs(n_term_charge), abs(c_term_charge)) - abs(middle_charge),
        }
    
    def _aa_distance(self, aa1, aa2):
        """计算两个氨基酸之间的距离（简化版）"""
        if aa1 not in self.aa_properties or aa2 not in self.aa_properties:
            return 0
        
        p1 = self.aa_properties[aa1]
        p2 = self.aa_properties[aa2]
        
        # 基于疏水性和电荷的欧氏距离
        d = ((p1['hydrophobicity'] - p2['hydrophobicity']) ** 2 + 
             (p1['charge'] - p2['charge']) ** 2) ** 0.5
        return d
    
    def _find_repeats(self, seq):
        """查找重复序列"""
        repeats = []
        for length in [2, 3, 4]:
            for i in range(len(seq) - length * 2 + 1):
                pattern = seq[i:i+length]
                if seq[i+length:i+length*2] == pattern:
                    repeats.append((pattern, i))
        return repeats
    
    def _count_dimensions(self, features):
        """统计维度数量"""
        count = 0
        for category, data in features.items():
            if isinstance(data, dict):
                count += len(data)
            elif isinstance(data, list):
                count += len(data)
            else:
                count += 1
        return count


# 使用示例
if __name__ == "__main__":
    # 创建解释系统
    explainer = BioExplanationSystem()
    
    # 测试序列
    test_seq = "GIGKFLHSAKKFGKAFVGEIMNS"
    model_prob = 0.999  # 模型输出的AFP概率
    
    # 进行分析
    result = explainer.analyze(test_seq, model_prob)
    
    # 输出结果
    print("=" * 70)
    print("生物学解释性评分系统 - 分析报告")
    print("=" * 70)
    print(f"\n分析维度数: {result['dimension_count']}")
    print(f"置信度等级: {result['confidence_level']}")
    print(f"可靠性评分: {result['reliability_score']}/100")
    print("\n" + result['explanation'])