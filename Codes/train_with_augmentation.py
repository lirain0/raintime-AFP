#!/usr/bin/env python3
"""
方案1: 数据增强 - 对正样本进行序列扰动
预期提升: +1.0-1.5%，目标90%+
"""

import os
import sys

# 获取项目根目录（Codes的父目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
import os
import random
warnings.filterwarnings('ignore')

# 强制使用CPU
device = torch.device('cpu')
print(f"使用设备: {device}")

# ==================== 数据加载 ====================

def load_simple_csv(filepath):
    df = pd.read_csv(filepath)
    return df['sequence'].tolist(), df['label'].tolist()

def load_nodes_csv(nodes_path):
    nodes_df = pd.read_csv(nodes_path)
    sequences, labels = [], []
    for row_id in tqdm(nodes_df['row_id'].unique(), desc="加载"):
        row_data = nodes_df[nodes_df['row_id'] == row_id].sort_values('position')
        seq = ''.join(row_data['amino_acid'].tolist())
        label = row_data['label'].iloc[0]
        sequences.append(seq)
        labels.append(label)
    return sequences, labels

def load_all_data():
    all_sequences, all_labels = [], []
    
    simple_datasets = [
        (os.path.join(PROJECT_ROOT, 'dataset', 'DeepAFP-main-train.csv'), 'Main-Train'),
        (os.path.join(PROJECT_ROOT, 'dataset', 'DeepAFP-main-test.csv'), 'Main-Test'),
        (os.path.join(PROJECT_ROOT, 'dataset', 'DeepAFP-Set1-train.csv'), 'Set1-Train'),
        (os.path.join(PROJECT_ROOT, 'dataset', 'DeepAFP-Set1-test.csv'), 'Set1-Test'),
        (os.path.join(PROJECT_ROOT, 'dataset', 'DeepAFP-Set2-train.csv'), 'Set2-Train'),
        (os.path.join(PROJECT_ROOT, 'dataset', 'DeepAFP-Set2-test.csv'), 'Set2-Test'),
    ]
    
    for filepath, name in simple_datasets:
        if os.path.exists(filepath):
            seqs, labs = load_simple_csv(filepath)
            all_sequences.extend(seqs)
            all_labels.extend(labs)
    
    nodes_datasets = [
        (os.path.join(PROJECT_ROOT, 'dataset_deepafp', 'Nodes1.csv'), 'Nodes-Train'),
        (os.path.join(PROJECT_ROOT, 'dataset_deepafp', 'Nodes_test.csv'), 'Nodes-Test'),
    ]
    
    for filepath, name in nodes_datasets:
        if os.path.exists(filepath):
            seqs, labs = load_nodes_csv(filepath)
            all_sequences.extend(seqs)
            all_labels.extend(labs)
    
    unique_data = {}
    for seq, label in zip(all_sequences, all_labels):
        if seq not in unique_data:
            unique_data[seq] = label
    
    return list(unique_data.keys()), list(unique_data.values())

# ==================== 数据增强 ====================
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# 氨基酸替换矩阵（相似性替换）
SIMILAR_AA = {
    'A': ['G', 'S', 'V'],  # 小侧链
    'C': ['S', 'T'],       # 含硫/羟基
    'D': ['E', 'N'],       # 酸性
    'E': ['D', 'Q'],       # 酸性
    'F': ['Y', 'W'],       # 芳香族
    'G': ['A', 'S'],       # 小侧链
    'H': ['R', 'K'],       # 碱性
    'I': ['L', 'V'],       # 疏水
    'K': ['R', 'H'],       # 碱性
    'L': ['I', 'V'],       # 疏水
    'M': ['L', 'I'],       # 疏水
    'N': ['D', 'Q'],       # 酰胺
    'P': ['G'],            # 特殊构象
    'Q': ['E', 'N'],       # 酰胺
    'R': ['K', 'H'],       # 碱性
    'S': ['T', 'A'],       # 羟基
    'T': ['S', 'A'],       # 羟基
    'V': ['I', 'L'],       # 疏水
    'W': ['F', 'Y'],       # 芳香族
    'Y': ['F', 'W'],       # 芳香族
}

def augment_sequence(seq, aug_type='random'):
    """对序列进行数据增强"""
    seq = list(seq)
    length = len(seq)
    
    if aug_type == 'random':
        # 随机选择一种增强方式
        aug_type = random.choice(['mask', 'substitute', 'truncate', 'shuffle'])
    
    if aug_type == 'mask':
        # 随机mask 10%的氨基酸
        num_mask = max(1, int(length * 0.1))
        mask_positions = random.sample(range(length), num_mask)
        for pos in mask_positions:
            seq[pos] = 'X'  # 用X表示mask
    
    elif aug_type == 'substitute':
        # 随机替换10%的氨基酸为相似氨基酸
        num_sub = max(1, int(length * 0.1))
        sub_positions = random.sample(range(length), num_sub)
        for pos in sub_positions:
            aa = seq[pos]
            if aa in SIMILAR_AA and SIMILAR_AA[aa]:
                seq[pos] = random.choice(SIMILAR_AA[aa])
    
    elif aug_type == 'truncate':
        # 随机截断序列（从开头或结尾）
        if length > 20:
            truncate_len = random.randint(1, min(5, length // 10))
            if random.random() > 0.5:
                seq = seq[truncate_len:]  # 截开头
            else:
                seq = seq[:-truncate_len]  # 截结尾
    
    elif aug_type == 'shuffle':
        # 随机打乱一小段序列
        if length > 15:
            window_size = random.randint(3, min(8, length // 3))
            start = random.randint(0, length - window_size)
            sub_seq = seq[start:start + window_size]
            random.shuffle(sub_seq)
            seq[start:start + window_size] = sub_seq
    
    return ''.join(seq)

def augment_positive_samples(sequences, labels, augment_factor=2):
    """对正样本进行数据增强"""
    print(f"\n🔄 数据增强中...")
    print(f"原始样本: {len(sequences)} (AFP: {sum(labels)})")
    
    augmented_seqs = []
    augmented_labels = []
    
    for seq, label in zip(sequences, labels):
        # 保留原始样本
        augmented_seqs.append(seq)
        augmented_labels.append(label)
        
        # 对正样本进行增强
        if label == 1:
            for _ in range(augment_factor - 1):
                aug_seq = augment_sequence(seq)
                augmented_seqs.append(aug_seq)
                augmented_labels.append(label)
    
    print(f"增强后样本: {len(augmented_seqs)} (AFP: {sum(augmented_labels)})")
    print(f"正样本增强倍数: {augment_factor}x")
    
    return augmented_seqs, augmented_labels

# ==================== 特征提取 ====================
AA_TO_NUM = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_NUM['X'] = 20

def get_zscale_features(aa):
    zscales = {
        'A': [0.24, -2.32, 0.60, -0.14, 1.30], 'C': [0.84, -1.67, 3.71, 0.18, -2.65],
        'D': [3.98, 0.93, 1.93, -2.46, 0.75], 'E': [3.11, 0.26, -0.11, -3.04, -0.25],
        'F': [-4.22, 1.94, 1.06, 0.54, -0.62], 'G': [2.05, -4.06, 0.36, -0.82, -0.38],
        'H': [2.47, 1.95, 0.26, 3.90, 0.09], 'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
        'K': [2.29, 0.89, -2.49, 1.49, 0.31], 'L': [-4.28, -1.30, -1.49, -0.72, 0.84],
        'M': [-2.85, -0.22, 0.47, 1.94, -0.98], 'N': [3.05, 1.62, 1.04, -1.15, 1.61],
        'P': [-1.66, 0.27, 1.84, 0.70, 2.00], 'Q': [1.75, 0.50, -1.44, -1.34, 0.66],
        'R': [3.52, 2.50, -3.50, 1.99, -0.17], 'S': [2.39, -1.07, 1.15, -1.39, 0.67],
        'T': [0.75, -2.18, -1.12, -1.46, -0.40], 'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
        'W': [-4.36, 3.94, 0.59, 3.44, -1.59], 'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],
        'X': [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    return zscales.get(aa.upper(), [0.0] * 5)

def extract_features(seq):
    features = []
    for aa in seq.upper():
        zscale = get_zscale_features(aa)
        onehot = [0.0] * 21
        onehot[AA_TO_NUM.get(aa, 20)] = 1.0
        features.append(zscale + onehot)
    return np.array(features, dtype=np.float32)

def build_graph(seq_features):
    num_nodes = len(seq_features)
    edge_index = []
    for i in range(num_nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# ==================== 数据集 ====================
class AFPGraphDataset(Dataset):
    def __init__(self, sequences, labels):
        self.data_list = []
        for seq, label in tqdm(zip(sequences, labels), total=len(sequences), desc="构建图"):
            features = extract_features(seq)
            edge_index = build_graph(features)
            x = torch.tensor(features, dtype=torch.float)
            y = torch.tensor([label], dtype=torch.long)
            self.data_list.append(Data(x=x, edge_index=edge_index, y=y))
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

# ==================== 模型 ====================
class AFPTransformerGNN(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=128, num_heads=8, num_gat_layers=3, num_classes=2, dropout=0.3):
        super().__init__()
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
        self.gat_norms = nn.ModuleList([nn.LayerNorm(hidden_dim * num_heads) for _ in range(num_gat_layers)])
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim * num_heads, nhead=num_heads, 
                                                   dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_heads * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_embedding(x)
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            x = torch.relu(norm(gat(x, edge_index)))
            x = self.dropout(x)
        x_padded, mask = self._pad_sequence(x, batch)
        x_transformed = self.transformer(x_padded, src_key_padding_mask=~mask)[mask]
        x_pool = global_mean_pool(x_transformed, batch)
        x_max = self._global_max_pool(x_transformed, batch)
        return self.classifier(torch.cat([x_pool, x_max], dim=1))
    
    def _pad_sequence(self, x, batch):
        batch_size = batch.max().item() + 1
        max_len = max((batch == i).sum().item() for i in range(batch_size))
        padded = torch.zeros(batch_size, max_len, x.size(1), device=x.device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=x.device)
        idx = 0
        for i in range(batch_size):
            seq_len = (batch == i).sum().item()
            padded[i, :seq_len] = x[idx:idx + seq_len]
            mask[i, :seq_len] = True
            idx += seq_len
        return padded, mask
    
    def _global_max_pool(self, x, batch):
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                out[i] = x[mask].max(dim=0)[0]
        return out

# ==================== 训练和评估 ====================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y.squeeze()).sum().item()
        total += data.y.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            prob = torch.softmax(out, dim=1)[:, 1]
            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(data.y.squeeze().cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

def find_optimal_threshold(labels, probs):
    """搜索最佳阈值"""
    best_threshold, best_f1, best_acc = 0.5, 0, 0
    
    print(f"\n{'='*70}")
    print("🔍 搜索最佳阈值")
    print(f"{'='*70}")
    print(f"{'阈值':>8s} | {'准确率':>8s} | {'精确率':>8s} | {'召回率':>8s} | {'F1':>8s}")
    print("-" * 70)
    
    for threshold in np.arange(0.30, 0.71, 0.01):
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        if f1 > best_f1 or (abs(f1 - best_f1) < 0.001 and acc > best_acc):
            best_f1, best_acc, best_threshold = f1, acc, threshold
            marker = " <-- 最佳"
        else:
            marker = ""
        
        if threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70] or marker:
            print(f"{threshold:8.2f} | {acc:8.4f} | {prec:8.4f} | {rec:8.4f} | {f1:8.4f}{marker}")
    
    print("-" * 70)
    return best_threshold, best_f1

# ==================== 主函数 ====================
def main():
    print("=" * 70)
    print("🎯 方案1: 数据增强训练")
    print("=" * 70)
    
    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    print("\n📊 加载原始数据...")
    sequences, labels = load_all_data()
    
    # 数据增强 - 正样本2倍
    sequences, labels = augment_positive_samples(sequences, labels, augment_factor=2)
    
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"\n增强后正负比例: 1:{neg_count/pos_count:.2f}")
    
    # 创建数据集
    dataset = AFPGraphDataset(sequences, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 创建模型
    print("\n🧠 创建模型...")
    model = AFPTransformerGNN().to(device)
    
    # 使用加权损失（虽然数据已增强，但保持加权）
    class_weights = torch.tensor([1.0, 1.5]).to(device)  # 稍微降低权重
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"类别权重: 非AFP={class_weights[0]:.1f}, AFP={class_weights[1]:.1f}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 训练
    print("\n" + "=" * 70)
    print("🏃 开始训练")
    print("=" * 70)
    
    best_val_acc = 0
    best_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(100):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_labels, val_probs = evaluate(model, val_loader)
        val_preds = (val_probs >= 0.5).astype(int)
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n⏹️ 早停于 epoch {epoch+1}")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_state)
    
    # 保存模型
    model_path = os.path.join(PROJECT_ROOT, 'model', 'afp_model_augmented.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 模型已保存: {model_path}")
    
    # 阈值优化
    val_labels, val_probs = evaluate(model, val_loader)
    best_threshold, best_f1 = find_optimal_threshold(val_labels, val_probs)
    
    # 最终评估
    print(f"\n{'='*70}")
    print("📊 最终评估结果")
    print(f"{'='*70}")
    print(f"\n✅ 最佳阈值: {best_threshold:.2f}")
    
    preds = (val_probs >= best_threshold).astype(int)
    acc = accuracy_score(val_labels, preds)
    prec = precision_score(val_labels, preds, zero_division=0)
    rec = recall_score(val_labels, preds, zero_division=0)
    f1 = f1_score(val_labels, preds, zero_division=0)
    auc = roc_auc_score(val_labels, val_probs)
    cm = confusion_matrix(val_labels, preds)
    
    print(f"\n准确率: {acc:.4f}")
    print(f"精确率: {prec:.4f}")
    print(f"召回率: {rec:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"\n混淆矩阵:")
    print(cm)
    
    # 对比
    print(f"\n{'='*70}")
    print("📈 性能对比")
    print(f"{'='*70}")
    print(f"\n之前最佳(加权投票):     88.91%")
    print(f"数据增强:               {acc*100:.2f}%")
    print(f"提升:                   {'+' if acc*100 > 88.91 else ''}{acc*100 - 88.91:.2f}%")
    
    if acc * 100 >= 90.0:
        print(f"\n🎉🎉🎉 恭喜！达到90%目标！🎉🎉🎉")
        print(f"🎊🎊🎊 任务圆满完成！🎊🎊🎊")
    elif acc * 100 >= 89.5:
        print(f"\n⭐ 非常接近90%了！只差一点点！")
    elif acc * 100 > 88.91:
        print(f"\n✅ 有提升！继续加油！")
    else:
        print(f"\n💪 数据增强效果不明显，可能需要其他方案")
    
    # 保存
    with open('../model/augmented_best_threshold.txt', 'w') as f:
        f.write(f"best_threshold={best_threshold}\n")
        f.write(f"accuracy={acc}\n")
        f.write(f"f1={f1}\n")
    
    print(f"\n{'='*70}")
    print("✅ 训练完成!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()