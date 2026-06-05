"""
数据增强策略
将数据集扩充2-3倍，提升模型泛化能力
"""

import pandas as pd
import numpy as np


def augment_sequence(seq, features, label):
    """
    序列增强策略
    1. 随机mask氨基酸（模拟突变）
    2. 添加高斯噪声
    3. 片段打乱
    """
    augmented = []

    # 策略1: 随机mask 10%氨基酸
    mask_indices = np.random.choice(len(seq), size=int(len(seq)*0.1), replace=False)
    seq_masked = list(seq)
    for idx in mask_indices:
        seq_masked[idx] = 'X'  # 用X表示mask
    augmented.append(('mask', ''.join(seq_masked), features, label))

    # 策略2: 添加特征噪声
    noise = np.random.normal(0, 0.01, features.shape)
    features_noisy = features + noise
    augmented.append(('noise', seq, features_noisy, label))

    return augmented


def create_augmented_dataset():
    """创建增强数据集"""
    # 读取原始数据
    df = pd.read_csv('../dataset_deepafp/Nodes1.csv')

    print(f"原始样本数: {len(df['row_id'].unique())}")

    # 对每个样本进行增强
    all_data = []
    for row_id in df['row_id'].unique():
        nodes = df[df['row_id'] == row_id]
        seq = ''.join(nodes['amino_acid'].tolist())
        label = nodes['label'].iloc[0]

        # 提取特征
        feature_cols = ['z1', 'z2', 'z3', 'z4', 'z5'] + [f'onehot_{i}' for i in range(1, 21)] + ['position']
        features = nodes[feature_cols].values

        # 添加原始样本
        all_data.append(('original', row_id, seq, features, label))

        # 添加增强样本
        aug_samples = augment_sequence(seq, features, label)
        for aug_type, aug_seq, aug_features, aug_label in aug_samples:
            all_data.append((aug_type, f"{row_id}_{aug_type}", aug_seq, aug_features, aug_label))

    print(f"增强后样本数: {len(all_data)}")
    print(f"增强倍数: {len(all_data)/len(df['row_id'].unique()):.1f}x")

    return all_data


if __name__ == '__main__':
    create_augmented_dataset()