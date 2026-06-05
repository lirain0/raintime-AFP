#!/usr/bin/env python3
"""测试可视化生成"""

import sys
sys.path.append('/root/AFP-Predictor-Deploy/Codes')

from visualize_explanation import generate_visualization_base64, generate_visualization_report
from bio_explanation_system import BioExplanationSystem

# 测试案例
test_seq = "GIGKFLHSAKKFGKAFVGEIMNS"
model_prob = 0.999

print("=" * 70)
print("测试可视化生成")
print("=" * 70)

try:
    # 测试解释系统
    print("\n1. 测试解释系统...")
    explainer = BioExplanationSystem()
    result = explainer.analyze(test_seq, model_prob)
    print(f"   ✓ 解释系统运行成功")
    print(f"   - 置信度等级: {result['confidence_level']}")
    print(f"   - 可靠性评分: {result['reliability_score']}")
    print(f"   - 分析维度数: {result['dimension_count']}")
    
    # 测试可视化生成（保存到文件）
    print("\n2. 测试可视化报告生成（保存到文件）...")
    output_path = generate_visualization_report(test_seq, model_prob)
    print(f"   ✓ 可视化报告已保存: {output_path}")
    
    # 测试可视化生成（base64）
    print("\n3. 测试可视化生成（base64）...")
    img_base64, result = generate_visualization_base64(test_seq, model_prob)
    print(f"   ✓ Base64图片生成成功")
    print(f"   - 图片数据长度: {len(img_base64)} 字节")
    
    print("\n" + "=" * 70)
    print("所有测试通过！")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
