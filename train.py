#!/usr/bin/env python3
"""
Training Script - V4 with Data Augmentation
"""

import os
import sys

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import random

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Codes'))
from features import extract_features, build_graph
from model import AFPTransformerGNN

device = torch.device('cpu')

# ==================== Data Loading ====================

def load_simple_csv(filepath):
    df = pd.read_csv(filepath)
    return df['sequence'].tolist(), df['label'].tolist()

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
    
    unique_data = {}
    for seq, label in zip(all_sequences, all_labels):
        if seq not in unique_data:
            unique_data[seq] = label
    
    return list(unique_data.keys()), list(unique_data.values())

# ==================== Data Augmentation ====================
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

SIMILAR_AA = {
    'A': ['G', 'S', 'V'], 'C': ['S', 'T'], 'D': ['E', 'N'], 'E': ['D', 'Q'],
    'F': ['Y', 'W'], 'G': ['A', 'S'], 'H': ['R', 'K'], 'I': ['L', 'V'],
    'K': ['R', 'H'], 'L': ['I', 'V'], 'M': ['L', 'I'], 'N': ['D', 'Q'],
    'P': ['G'], 'Q': ['E', 'N'], 'R': ['K', 'H'], 'S': ['T', 'A'],
    'T': ['S', 'A'], 'V': ['I', 'L'], 'W': ['F', 'Y'], 'Y': ['F', 'W'],
}

def augment_sequence(seq, aug_type='random'):
    seq = list(seq)
    length = len(seq)
    
    if aug_type == 'random':
        aug_type = random.choice(['mask', 'substitute', 'truncate', 'shuffle'])
    
    if aug_type == 'mask':
        num_mask = max(1, int(length * 0.1))
        mask_positions = random.sample(range(length), num_mask)
        for pos in mask_positions:
            seq[pos] = 'X'
    
    elif aug_type == 'substitute':
        num_sub = max(1, int(length * 0.1))
        sub_positions = random.sample(range(length), num_sub)
        for pos in sub_positions:
            aa = seq[pos]
            if aa in SIMILAR_AA and SIMILAR_AA[aa]:
                seq[pos] = random.choice(SIMILAR_AA[aa])
    
    elif aug_type == 'truncate':
        if length > 20:
            truncate_len = random.randint(1, min(5, length // 10))
            if random.random() > 0.5:
                seq = seq[truncate_len:]
            else:
                seq = seq[:-truncate_len]
    
    elif aug_type == 'shuffle':
        if length > 15:
            window_size = random.randint(3, min(8, length // 3))
            start = random.randint(0, length - window_size)
            sub_seq = seq[start:start + window_size]
            random.shuffle(sub_seq)
            seq[start:start + window_size] = sub_seq
    
    return ''.join(seq)

def augment_positive_samples(sequences, labels, augment_factor=2):
    augmented_seqs = []
    augmented_labels = []
    
    for seq, label in zip(sequences, labels):
        augmented_seqs.append(seq)
        augmented_labels.append(label)
        
        if label == 1:
            for _ in range(augment_factor - 1):
                aug_seq = augment_sequence(seq)
                augmented_seqs.append(aug_seq)
                augmented_labels.append(label)
    
    return augmented_seqs, augmented_labels

# ==================== Dataset ====================
class AFPGraphDataset(Dataset):
    def __init__(self, sequences, labels):
        self.data_list = []
        for seq, label in zip(sequences, labels):
            features = extract_features(seq)
            edge_index = build_graph(features)
            x = torch.tensor(features, dtype=torch.float)
            y = torch.tensor([label], dtype=torch.long)
            self.data_list.append(Data(x=x, edge_index=edge_index, y=y))
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

# ==================== Training ====================
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
    best_threshold, best_f1, best_acc = 0.5, 0, 0
    
    for threshold in np.arange(0.30, 0.71, 0.01):
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        
        if f1 > best_f1 or (abs(f1 - best_f1) < 0.001 and acc > best_acc):
            best_f1, best_acc, best_threshold = f1, acc, threshold
    
    return best_threshold

# ==================== Main ====================
def main():
    print("=" * 70)
    print("AFP-Predictor V4 - Training with Data Augmentation")
    print("=" * 70)
    
    # Load data
    sequences, labels = load_all_data()
    print(f"Total samples: {len(sequences)} (AFP: {sum(labels)})")
    
    # Augment data
    sequences, labels = augment_positive_samples(sequences, labels, augment_factor=2)
    print(f"After augmentation: {len(sequences)} (AFP: {sum(labels)})")
    
    # Create dataset
    dataset = AFPGraphDataset(sequences, labels)
    
    # Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = AFPTransformerGNN(input_dim=26, hidden_dim=128, num_heads=8,
                              num_gat_layers=3, num_classes=2, dropout=0.3).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    # Training loop
    best_acc = 0
    for epoch in range(100):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        
        if epoch % 10 == 0:
            labels, probs = evaluate(model, test_loader)
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, zero_division=0)
            
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={acc:.4f}, F1={f1:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                model_save_path = os.path.join(PROJECT_ROOT, 'model', 'afp_model_augmented.pth')
                torch.save(model.state_dict(), model_save_path)
                print(f"  -> Saved best model (acc={acc:.4f})")
            
            scheduler.step(acc)
    
    # Final evaluation
    model_load_path = os.path.join(PROJECT_ROOT, 'model', 'afp_model_augmented.pth')
    model.load_state_dict(torch.load(model_load_path))
    labels, probs = evaluate(model, test_loader)
    threshold = find_optimal_threshold(labels, probs)
    preds = (probs >= threshold).astype(int)
    
    print("\n" + "=" * 70)
    print("Final Results:")
    print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
    print(f"Precision: {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"Recall: {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"F1: {f1_score(labels, preds, zero_division=0):.4f}")
    print(f"AUC: {roc_auc_score(labels, probs):.4f}")
    print(f"Best Threshold: {threshold:.2f}")
    print("=" * 70)

if __name__ == '__main__':
    main()
