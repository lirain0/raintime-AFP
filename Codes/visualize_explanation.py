#!/usr/bin/env python3
"""
解释性分析可视化系统
生成置信度仪表盘、特征雷达图、氨基酸组成图等
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
import numpy as np
import io
import base64
from bio_explanation_system import BioExplanationSystem
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def create_confidence_gauge(ax, score, level):
    """创建置信度仪表盘"""
    # 背景圆环
    theta = np.linspace(0, np.pi, 100)
    for i, (start, end, color) in enumerate([
        (0, 0.4, '#ff4444'),      # D级 红色
        (0.4, 0.6, '#ff8800'),    # C级 橙色
        (0.6, 0.8, '#ffcc00'),    # B级 黄色
        (0.8, 1.0, '#44ff44'),    # A级 绿色
    ]):
        mask = (theta >= start * np.pi) & (theta <= end * np.pi)
        ax.fill_between(np.cos(theta[mask]), np.sin(theta[mask]), 
                        0.6, color=color, alpha=0.3)
    
    # 指针
    angle = score / 100 * np.pi
    ax.arrow(0, 0, 0.5*np.cos(angle), 0.5*np.sin(angle), 
             head_width=0.05, head_length=0.05, fc='black', ec='black', linewidth=3)
    
    # 中心文字 - 分数（大字体）
    ax.text(0, 0.1, f'{score}', fontsize=32, ha='center', va='center', 
            fontweight='bold', color='#333333')
    # 等级文字（小字体，在分数下方）
    ax.text(0, -0.25, f'{level}', fontsize=12, ha='center', va='center',
            color='#666666', wrap=True)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Reliability Score', fontsize=14, fontweight='bold', pad=20)


def create_radar_chart(ax, features):
    """创建特征雷达图"""
    categories = ['Charge', 'Hydrophobicity', 'Helix', 'Amphipathic', 
                  'Cationic', 'Diversity']
    
    # 计算各维度得分（归一化到0-1）
    pc = features['physicochemical']
    comp = features['composition']
    struct = features['structural']
    basic = features['basic']
    
    values = [
        min(pc['charge']['net'] / 10, 1.0),  # 电荷
        (pc['hydrophobicity']['mean'] + 5) / 10,  # 疏水性
        struct['helix_propensity'] / 2,  # 螺旋倾向
        struct['amphipathic_score'],  # 两亲性
        comp['group_frequency'].get('cationic_ratio', 0),  # 阳离子比例
        basic['aa_diversity'],  # 多样性
    ]
    
    # 闭合雷达图
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # 绘制
    ax.plot(angles, values, 'o-', linewidth=2, color='#2196F3')
    ax.fill(angles, values, alpha=0.25, color='#2196F3')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Feature Profile', fontsize=14, fontweight='bold', pad=20)


def create_aa_composition(ax, seq):
    """创建氨基酸组成饼图"""
    aa_count = {}
    for aa in seq:
        aa_count[aa] = aa_count.get(aa, 0) + 1
    
    # 分组显示
    groups = {
        'Cationic (K,R,H)': sum(aa_count.get(aa, 0) for aa in 'KRH'),
        'Hydrophobic (L,I,V,F,W,M,A)': sum(aa_count.get(aa, 0) for aa in 'LIVFWMA'),
        'Polar (S,T,N,Q,C,Y)': sum(aa_count.get(aa, 0) for aa in 'STNQCY'),
        'Anionic (D,E)': sum(aa_count.get(aa, 0) for aa in 'DE'),
        'Special (G,P)': sum(aa_count.get(aa, 0) for aa in 'GP'),
    }
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7']
    
    # 过滤掉为0的组
    groups = {k: v for k, v in groups.items() if v > 0}
    
    if groups:
        ax.pie(groups.values(), labels=groups.keys(), colors=colors[:len(groups)],
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
    ax.set_title('AA Composition', fontsize=14, fontweight='bold')


def create_sequence_annotation(ax, seq, motifs):
    """创建序列特征标注图"""
    # 设置颜色
    aa_colors = {
        'K': '#ff6b6b', 'R': '#ff6b6b', 'H': '#ff9999',  # 阳离子
        'L': '#4ecdc4', 'I': '#4ecdc4', 'V': '#4ecdc4', 'F': '#4ecdc4', 
        'W': '#4ecdc4', 'M': '#4ecdc4', 'A': '#4ecdc4',  # 疏水
        'D': '#f9ca24', 'E': '#f9ca24',  # 阴离子
        'S': '#a29bfe', 'T': '#a29bfe', 'N': '#a29bfe', 'Q': '#a29bfe',  # 极性
    }
    
    # 绘制序列
    box_size = 0.8
    spacing = 1.0
    
    for i, aa in enumerate(seq):
        color = aa_colors.get(aa, '#dfe6e9')
        rect = FancyBboxPatch((i*spacing, 0), box_size, box_size,
                              boxstyle="round,pad=0.05", 
                              facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(i*spacing + box_size/2, box_size/2, aa, 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 标注特征模式
    if motifs.get('afp_motifs_found'):
        for mtype, motif in motifs['afp_motifs_found'][:3]:
            pos = seq.find(motif)
            if pos >= 0:
                ax.plot([pos*spacing, (pos+len(motif))*spacing + box_size - spacing], 
                       [1.2, 1.2], 'r-', linewidth=3, alpha=0.7)
                ax.text((pos + len(motif)/2)*spacing, 1.5, motif, 
                       ha='center', fontsize=8, color='red')
    
    ax.set_xlim(-0.5, len(seq)*spacing + 0.5)
    ax.set_ylim(-0.5, 2)
    ax.axis('off')
    ax.set_title('Sequence Annotation', fontsize=14, fontweight='bold')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color='#ff6b6b', label='Cationic'),
        mpatches.Patch(color='#4ecdc4', label='Hydrophobic'),
        mpatches.Patch(color='#f9ca24', label='Anionic'),
        mpatches.Patch(color='#a29bfe', label='Polar'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def create_property_bars(ax, features):
    """创建属性条形图"""
    pc = features['physicochemical']
    comp = features['composition']
    
    properties = [
        ('Net Charge', pc['charge']['net'], 0, 10, '+'),
        ('Hydrophobicity', pc['hydrophobicity']['mean'], -5, 5, ''),
        ('Helix Prop.', features['structural']['helix_propensity'], 0, 2, ''),
        ('Cationic %', comp['group_frequency'].get('cationic_ratio', 0)*100, 0, 50, '%'),
    ]
    
    y_pos = np.arange(len(properties))
    values = [p[1] for p in properties]
    colors = ['#4CAF50' if p[1] > (p[2]+p[3])/2 else '#FFC107' for p in properties]
    
    bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p[0] for p in properties])
    ax.set_xlabel('Value')
    ax.set_title('Key Properties', fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for i, (bar, val, _, _, unit) in enumerate(zip(bars, values, 
                                                     [p[2] for p in properties],
                                                     [p[3] for p in properties],
                                                     [p[4] for p in properties])):
        ax.text(val + 0.5, i, f'{val:.1f}{unit}', va='center', fontsize=9)


def generate_visualization_report(sequence, model_probability, output_dir='../visualization_output'):
    """生成完整的可视化报告（保存到文件）"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行解释系统
    explainer = BioExplanationSystem()
    result = explainer.analyze(sequence, model_probability)
    
    features = result['features']
    confidence = result['confidence_level']
    score = result['reliability_score']
    
    # 简化等级显示（用于可视化）
    level_simple = confidence[0] if confidence else 'N'  # 只取A/B/C/D
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Bio-Explanation Analysis Report\nSequence: {sequence}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 布局：2x3网格
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. 置信度仪表盘
    ax1 = fig.add_subplot(gs[0, 0])
    create_confidence_gauge(ax1, score, level_simple)
    
    # 2. 特征雷达图
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    create_radar_chart(ax2, features)
    
    # 3. 氨基酸组成
    ax3 = fig.add_subplot(gs[0, 2])
    create_aa_composition(ax3, sequence)
    
    # 4. 序列标注
    ax4 = fig.add_subplot(gs[1, :2])
    create_sequence_annotation(ax4, sequence, features['motifs'])
    
    # 5. 关键属性
    ax5 = fig.add_subplot(gs[1, 2])
    create_property_bars(ax5, features)
    
    # 添加预测信息框
    pred_text = f"""
    Model Prediction: {'AFP' if model_probability >= 0.57 else 'non-AFP'}
    AFP Probability: {model_probability:.1%}
    Threshold: 0.57
    Reliability: Level {level_simple} ({score}/100)
    """
    fig.text(0.02, 0.02, pred_text, fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 保存
    safe_seq = sequence[:10] if len(sequence) > 10 else sequence
    output_path = os.path.join(output_dir, f'explanation_{safe_seq}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ 可视化报告已保存: {output_path}")
    
    plt.close()
    return output_path


def generate_visualization_base64(sequence, model_probability):
    """生成可视化报告并返回base64编码（用于Web端）"""
    
    # 运行解释系统
    explainer = BioExplanationSystem()
    result = explainer.analyze(sequence, model_probability)
    
    features = result['features']
    confidence = result['confidence_level']
    score = result['reliability_score']
    
    # 简化等级显示（用于可视化）
    level_simple = confidence[0] if confidence else 'N'  # 只取A/B/C/D
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Bio-Explanation Analysis Report\nSequence: {sequence}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 布局：2x3网格
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. 置信度仪表盘
    ax1 = fig.add_subplot(gs[0, 0])
    create_confidence_gauge(ax1, score, level_simple)
    
    # 2. 特征雷达图
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    create_radar_chart(ax2, features)
    
    # 3. 氨基酸组成
    ax3 = fig.add_subplot(gs[0, 2])
    create_aa_composition(ax3, sequence)
    
    # 4. 序列标注
    ax4 = fig.add_subplot(gs[1, :2])
    create_sequence_annotation(ax4, sequence, features['motifs'])
    
    # 5. 关键属性
    ax5 = fig.add_subplot(gs[1, 2])
    create_property_bars(ax5, features)
    
    # 添加预测信息框
    pred_text = f"""
    Model Prediction: {'AFP' if model_probability >= 0.57 else 'non-AFP'}
    AFP Probability: {model_probability:.1%}
    Threshold: 0.57
    Reliability: Level {level_simple} ({score}/100)
    """
    fig.text(0.02, 0.02, pred_text, fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 保存到内存
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    
    # 转换为base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    plt.close()
    
    return img_base64, result


def main():
    """主函数：生成正例和负例的可视化"""
    
    print("=" * 70)
    print("解释性分析可视化系统")
    print("=" * 70)
    
    # 测试案例
    test_cases = [
        {
            'name': 'Cecropin A (Positive)',
            'seq': 'GIGKFLHSAKKFGKAFVGEIMNS',
            'prob': 0.999,
        },
        {
            'name': 'Brevinin-1E (Positive)',
            'seq': 'FLPLLAGLAANFLPKIFCKITKKC',
            'prob': 0.883,
        },
        {
            'name': 'Collagen (Negative)',
            'seq': 'GPPGPPGPPGPP',
            'prob': 0.152,
        },
    ]
    
    for case in test_cases:
        print(f"\n处理: {case['name']}")
        print(f"序列: {case['seq']}")
        print(f"模型概率: {case['prob']}")
        
        generate_visualization_report(case['seq'], case['prob'])
    
    print("\n" + "=" * 70)
    print("所有可视化报告已生成完成！")
    print("输出目录: ../visualization_output/")
    print("=" * 70)


if __name__ == "__main__":
    main()