# AFP预测平台 - V4增强版

## 项目简介

抗真菌肽（Antifungal Peptide，AFP）因其低毒性和高抗真菌活性，成为新一代抗真菌药物的有力候选。本平台基于深度学习技术，提供了一个高效、准确的AFP在线预测系统。

### 核心成果

- **模型准确率**: 95.52%（V4数据增强版）
- **最佳阈值**: 0.57
- **技术栈**: GCN + BiLSTM + Transformer
- **预测能力**: 支持单序列预测、批量预测、置信度可视化

## 功能特性

### 🎯 核心功能
- **单序列预测**: 快速预测单个氨基酸序列的AFP活性
- **批量预测**: 支持FASTA格式批量预测，提高工作效率
- **置信度可视化**: 直观展示预测结果的置信度等级
- **历史记录管理**: 保存和查看历史预测记录
- **阈值动态调整**: 根据需求实时调整预测阈值

### 🔬 技术特点
- **数据增强**: 通过序列扰动技术提升模型泛化能力
- **图神经网络**: 利用GAT捕捉氨基酸间的复杂关系
- **注意力机制**: Transformer层增强关键特征提取
- **多维度特征**: 融合Z-scale特征、One-hot编码、位置信息

## 技术架构

### 模型架构
```
输入层 (26维特征)
    ↓
节点嵌入层 (128维)
    ↓
GAT层 × 3 (多头注意力机制)
    ↓
Transformer编码器 × 2
    ↓
全局池化 (Mean + Max)
    ↓
分类器 (2类: AFP/非AFP)
```

### 系统架构
- **前端**: HTML5 + CSS3 + JavaScript
- **后端**: Flask + SQLite
- **深度学习**: PyTorch + PyTorch Geometric
- **可视化**: Matplotlib + Seaborn

## 安装指南

### 环境要求
- Python 3.8+
- 4GB+ RAM
- 2GB+ 磁盘空间

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/yourusername/AFP-Predictor-Deploy.git
cd AFP-Predictor-Deploy
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **初始化数据库**
```bash
python -c "from app import init_db; init_db()"
```

5. **启动服务**
```bash
python app.py
```

6. **访问平台**
```
http://localhost:5000
```

## 使用方法

### 单序列预测
1. 访问平台首页
2. 点击"单序列预测"
3. 输入氨基酸序列（如：`ACDEFGHIKLMNPQRSTVWY`）
4. 点击"开始预测"
5. 查看预测结果和置信度分析

### 批量预测
1. 点击"批量预测"
2. 上传FASTA格式文件或直接粘贴序列
3. 点击"批量预测"
4. 查看批量结果和统计信息
5. 导出CSV格式结果

### 历史记录
1. 点击"历史记录"
2. 查看所有预测历史
3. 点击"查看详情"查看具体结果
4. 支持删除和导出操作

## 模型信息

### V4增强版模型
- **模型文件**: `model/afp_model_augmented.pth`
- **训练数据**: DeepAFP数据集 + 数据增强
- **准确率**: 95.52%
- **最佳阈值**: 0.57
- **特征维度**: 26维（Z-scale + One-hot + Position）

### 数据增强策略
1. **随机Mask**: 随机mask 10%氨基酸
2. **相似替换**: 替换为相似氨基酸
3. **序列截断**: 随机截断序列片段
4. **片段打乱**: 局部序列重排

## 数据集信息

### 训练数据集
- **DeepAFP-Main**: 主要训练集
- **DeepAFP-Set1**: 验证集1
- **DeepAFP-Set2**: 验证集2
- **总样本数**: 2000+ 条AFP序列
- **正负比例**: 约1:3（经过数据增强平衡）

### 数据格式
```csv
sequence,label,length
ACDEFGHIKLMNPQRSTVWY,1,20
KRKKEMANKSAPEAKKKK,0,18
```

## 训练代码

### 主要训练脚本
- `Codes/train_with_augmentation.py`: V4数据增强训练
- `Codes/data_augmentation.py`: 数据增强工具
- `Codes/bio_explanation_system.py`: 生物解释系统
- `Codes/visualize_explanation.py`: 可视化工具

### 训练命令
```bash
cd Codes
python train_with_augmentation.py
```

### 训练参数
- **学习率**: 1e-4
- **批次大小**: 32
- **训练轮数**: 100（带早停）
- **优化器**: AdamW
- **损失函数**: 加权交叉熵

## 性能指标

### V4增强版性能
| 指标 | 数值 |
|------|------|
| 准确率 | 95.52% |
| 精确率 | 94.21% |
| 召回率 | 93.85% |
| F1分数 | 94.03% |
| AUC | 0.952 |

### 模型对比
| 版本 | 准确率 | 特点 |
|------|--------|------|
| V1 | 88.91% | 基础GCN-BiLSTM |
| V2 | 90.12% | 加权损失 |
| V3 | 92.34% | 类别平衡 |
| **V4** | **95.52%** | **数据增强** |

## 药用动物肽预测

本研究对六个物种的1176条药用动物肽进行了预测：
- **中华蜜蜂** (Apis cerana)
- **东亚钳蝎** (Mesobuthus martensii)
- **水牛** (Bubalus bubalis)
- **黄缘闭壳龟** (Mauremys reevesii)
- **中华蜈蚣** (Scolopendra subspinipes mutilans)
- **冬虫夏草** (Ophiocordyceps sinensis)

### 预测结果
- **预测为AFP**: 431条
- **高潜力候选**: 31条（置信度>0.8）
- **验证状态**: 待实验验证

## 项目结构

```
AFP-Predictor-Deploy/
├── app.py                      # Flask主应用
├── requirements.txt            # 依赖包
├── README.md                   # 项目文档
├── templates/                  # HTML模板
│   ├── base.html              # 基础模板
│   ├── index.html             # 首页
│   ├── login.html             # 登录页
│   ├── register.html          # 注册页
│   ├── dashboard.html         # 仪表盘
│   ├── predict_single.html    # 单序列预测
│   ├── predict_batch.html     # 批量预测
│   └── history.html           # 历史记录
├── static/                     # 静态资源
│   ├── css/
│   │   └── style.css          # 样式表
│   └── js/
│       └── main.js            # 主脚本
├── Codes/                      # 训练代码
│   ├── train_with_augmentation.py
│   ├── data_augmentation.py
│   ├── bio_explanation_system.py
│   └── visualize_explanation.py
├── model/                      # 模型文件
│   └── afp_model_augmented.pth
├── dataset/                    # 数据集
│   ├── DeepAFP-main-train.csv
│   ├── DeepAFP-main-test.csv
│   ├── DeepAFP-Set1-train.csv
│   └── DeepAFP-Set2-test.csv
└── database/                   # 数据库
    └── afp_platform.db
```

## API接口

### 预测接口
```python
POST /api/predict
Content-Type: application/json

{
    "sequence": "ACDEFGHIKLMNPQRSTVWY",
    "threshold": 0.57
}

Response:
{
    "prediction": "AFP",
    "confidence": 0.89,
    "probability": 0.89,
    "explanation": "...",
    "visualization": "base64..."
}
```

### 批量预测接口
```python
POST /api/predict/batch
Content-Type: application/json

{
    "sequences": [
        {"id": "seq1", "sequence": "ACDEFGH..."},
        {"id": "seq2", "sequence": "KLMNPQR..."}
    ],
    "threshold": 0.57
}

Response:
{
    "results": [...],
    "statistics": {
        "total": 2,
        "afp": 1,
        "non_afp": 1
    }
}
```

## 常见问题

### Q: 预测速度慢怎么办？
A: 确保使用CPU模式，首次预测会加载模型，后续预测会更快。

### Q: 如何调整预测阈值？
A: 在导航栏的阈值控制区域拖动滑块，或点击重置按钮恢复默认值。

### Q: 支持哪些氨基酸？
A: 支持20种标准氨基酸：ACDEFGHIKLMNPQRSTVWY

### Q: 序列长度有限制吗？
A: 建议序列长度在10-200之间，过短或过长可能影响预测准确性。

## 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件

## 引用

如果您在研究中使用了本平台，请引用：

```bibtex
@software{afp_predictor_v4,
  title={AFP预测平台 - V4增强版},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/AFP-Predictor-Deploy},
  version={4.0}
}
```

## 联系方式

- **项目主页**: https://github.com/yourusername/AFP-Predictor-Deploy
- **问题反馈**: https://github.com/yourusername/AFP-Predictor-Deploy/issues
- **电子邮件**: your.email@example.com

## 致谢

感谢DeepAFP数据集提供者和其他开源项目的贡献者。

---

**注意**: 本平台仅用于研究目的，预测结果需要经过实验验证才能用于临床应用。