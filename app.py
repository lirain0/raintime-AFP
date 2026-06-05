#!/usr/bin/env python3
"""
AFP预测平台 - V4版本
基于Windows系统的在线预测平台
功能：单序列预测、批量预测、置信度可视化、历史记录
技术栈：Flask + SQLite + HTML/CSS/JS
"""

import os
import sys
import hashlib
import datetime
import json
import base64
import io
from functools import wraps

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加Codes目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'Codes'))
from bio_explanation_system import BioExplanationSystem
from visualize_explanation import generate_visualization_report, generate_visualization_base64

# ==================== 配置 ====================
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'afp_prediction_platform_v4_secret_key_2024'
CORS(app)

# 数据库配置
DATABASE = os.path.join(os.path.dirname(__file__), 'database', 'afp_platform.db')

# V4数据增强模型配置 (95.52%准确率)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'afp_model_augmented.pth')
DEFAULT_THRESHOLD = 0.57  # 95.52%模型的默认最佳阈值

# 全局阈值配置（可在运行时调整）
class ThresholdConfig:
    current_threshold = DEFAULT_THRESHOLD
    
    @classmethod
    def get(cls):
        return cls.current_threshold
    
    @classmethod
    def set(cls, value):
        cls.current_threshold = float(value)
        return cls.current_threshold

# 确保目录存在
os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

# ==================== 数据库操作 ====================
def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库"""
    with get_db() as conn:
        # 用户表
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction_name TEXT,
                sequence_count INTEGER DEFAULT 1,
                result_summary TEXT,
                prediction_type TEXT DEFAULT 'single',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
            
            CREATE TABLE IF NOT EXISTS prediction_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                sequence_id TEXT,
                sequence TEXT NOT NULL,
                is_afp INTEGER NOT NULL,
                model_probability REAL NOT NULL,
                threshold REAL DEFAULT 0.57,
                confidence_level TEXT,
                reliability_score INTEGER,
                confidence_reasons TEXT,
                explanation_summary TEXT,
                recommendation TEXT,
                sequence_length INTEGER,
                net_charge REAL,
                cationic_ratio REAL,
                hydrophobicity_mean REAL,
                helix_propensity REAL,
                afp_motifs_count INTEGER,
                non_afp_patterns TEXT,
                features_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions (id)
            );
        ''')
        conn.commit()
        
        # 迁移：添加 prediction_type 字段（如果表已存在但没有该字段）
        try:
            conn.execute("SELECT prediction_type FROM predictions LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE predictions ADD COLUMN prediction_type TEXT DEFAULT 'single'")
            conn.commit()
            print("✓ 数据库迁移完成: 添加 prediction_type 字段")

# ==================== 模型管理器（单例模式） ====================
class ModelManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ModelManager._initialized:
            return
        self.model = None
        self.device = torch.device('cpu')
        self.explainer = BioExplanationSystem()
        self.model_loaded = False
        ModelManager._initialized = True
    
    def initialize(self):
        """初始化V4数据增强模型（95.52%准确率）"""
        if self.model_loaded:
            return
        
        try:
            # 导入模型类
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Codes'))
            from train_with_augmentation import AFPTransformerGNN
            
            # 创建模型（使用与训练时相同的配置）
            self.model = AFPTransformerGNN(
                input_dim=26, 
                hidden_dim=128, 
                num_heads=8, 
                num_gat_layers=3, 
                num_classes=2, 
                dropout=0.3
            ).to(self.device)
            
            # 加载模型权重
            if os.path.exists(MODEL_PATH):
                checkpoint = torch.load(MODEL_PATH, map_location=self.device)
                
                # 兼容两种保存格式
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # 处理 GAT 层参数名称兼容问题（旧版本 vs 新版本）
                # 旧版本: gat_layers.0.lin.weight → 新版本: gat_layers.0.lin_src.weight + gat_layers.0.lin_dst.weight
                new_state_dict = {}
                for key, value in state_dict.items():
                    if 'gat_layers' in key and '.lin.weight' in key:
                        # 将旧格式转换为新格式
                        src_key = key.replace('.lin.weight', '.lin_src.weight')
                        dst_key = key.replace('.lin.weight', '.lin_dst.weight')
                        new_state_dict[src_key] = value
                        new_state_dict[dst_key] = value
                    elif 'gat_layers' in key and '.lin.bias' in key:
                        # 偏置也需要转换
                        src_key = key.replace('.lin.bias', '.lin_src.bias')
                        dst_key = key.replace('.lin.bias', '.lin_dst.bias')
                        new_state_dict[src_key] = value
                        new_state_dict[dst_key] = value
                    else:
                        new_state_dict[key] = value
                
                # 使用 strict=False 跳过可能不匹配的参数
                self.model.load_state_dict(new_state_dict, strict=False)
                
                self.model.eval()
                self.model_loaded = True
                print(f"✓ V4数据增强模型加载成功: {MODEL_PATH}")
                print(f"✓ 模型配置: hidden_dim=128, num_heads=8, num_gat_layers=3")
                print(f"✓ 独立测试集准确率: 95.52%")
            else:
                print(f"⚠ 模型文件不存在: {MODEL_PATH}")
                print(f"  将使用模拟预测")
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    def predict(self, sequence):
        """预测单个序列"""
        if not self.model_loaded:
            self.initialize()
        
        if self.model is None:
            # 模拟预测（用于测试）
            return {
                'probability': 0.85,
                'is_afp': True,
                'error': 'Model not loaded, using mock data'
            }
        
        try:
            # 序列转图数据
            data = self._seq_to_graph(sequence)
            if data is None:
                return {'error': 'Invalid sequence'}
            
            with torch.no_grad():
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                prob = probs[0][1].item()
            
            return {
                'probability': prob,
                'is_afp': prob >= ThresholdConfig.get(),
                'threshold': ThresholdConfig.get()
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _seq_to_graph(self, seq):
        """序列转图数据（与train_with_augmentation.py完全一致）"""
        AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
        AA_TO_NUM = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        AA_TO_NUM['X'] = 20
        
        # Z-scale值（与训练脚本完全一致）
        Z_SCALE = {
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
        
        seq = seq.upper()
        seq_len = len(seq)
        valid_aa = set(AMINO_ACIDS)
        
        # 节点特征（与训练脚本完全一致：zscale 5维 + onehot 21维 = 26维）
        nodes = []
        for aa in seq:
            if aa not in valid_aa and aa != 'X':
                aa = 'X'  # 非标准氨基酸映射为X
            z = Z_SCALE.get(aa, [0.0]*5)
            oh = [0.0] * 21
            oh[AA_TO_NUM.get(aa, 20)] = 1.0
            nodes.append(z + oh)  # 5 + 21 = 26维，无position
        
        # 边（序列连接）
        edges = []
        for i in range(seq_len - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        
        if len(edges) == 0:
            edges = [[0, 0]]
        
        # 创建PyG Data
        from torch_geometric.data import Data
        import torch
        
        x = torch.tensor(nodes, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # 添加batch属性（单样本）
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(x.size(0), dtype=torch.long)
        
        return data
    
    def explain(self, sequence, model_probability):
        """生成解释"""
        return self.explainer.analyze(sequence, model_probability)
    
    def generate_visualization(self, sequence, model_probability):
        """生成可视化（返回base64图片）"""
        try:
            # 使用完整的可视化报告生成
            img_base64, result = generate_visualization_base64(sequence, model_probability)
            # 添加data URL前缀，使其可以直接作为img标签的src属性
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            print(f"可视化生成失败：{e}")
            import traceback
            traceback.print_exc()
            return None

# 全局模型管理器
model_manager = ModelManager()

# ==================== 登录验证装饰器 ====================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== 全局错误处理 ====================
@app.errorhandler(500)
def internal_error(error):
    """处理500错误，返回JSON格式"""
    import traceback
    traceback_str = traceback.format_exc()
    print(f"500错误: {error}\n{traceback_str}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(error),
        'traceback': traceback_str if app.debug else None
    }), 500

@app.errorhandler(Exception)
def handle_exception(error):
    """处理所有未捕获的异常"""
    import traceback
    traceback_str = traceback.format_exc()
    print(f"未捕获异常: {error}\n{traceback_str}")
    
    # 如果请求是API请求，返回JSON
    if request.is_json or request.path.startswith('/api/'):
        return jsonify({
            'error': 'Server error',
            'message': str(error),
            'traceback': traceback_str if app.debug else None
        }), 500
    
    # 否则返回HTML错误页面
    return render_template('base.html', error=str(error)), 500

# ==================== 路由 ====================

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        with get_db() as conn:
            user = conn.execute(
                'SELECT * FROM users WHERE username = ?', (username,)
            ).fetchone()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            
            with get_db() as conn:
                conn.execute(
                    'UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?',
                    (user['id'],)
                )
                conn.commit()
            
            return redirect(url_for('dashboard'))
        else:
            flash('用户名或密码错误', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """注册"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        if not username or not password:
            flash('用户名和密码不能为空', 'error')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        
        try:
            with get_db() as conn:
                conn.execute(
                    'INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                    (username, hashed_password, email)
                )
                conn.commit()
            flash('注册成功，请登录', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('用户名或邮箱已存在', 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """登出"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """用户仪表盘"""
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/predict/single')
@login_required
def predict_single():
    """单序列预测页面"""
    return render_template('predict_single.html')

@app.route('/predict/batch')
@login_required
def predict_batch():
    """批量预测页面"""
    return render_template('predict_batch.html')

@app.route('/history')
@login_required
def history():
    """历史记录页面"""
    with get_db() as conn:
        records = conn.execute(
            '''SELECT p.*, COUNT(pd.id) as detail_count 
               FROM predictions p 
               LEFT JOIN prediction_details pd ON p.id = pd.prediction_id 
               WHERE p.user_id = ? 
               GROUP BY p.id 
               ORDER BY p.created_at DESC''',
            (session['user_id'],)
        ).fetchall()
    
    return render_template('history.html', records=records)

# ==================== JSON序列化辅助函数 ====================
def convert_to_native(obj):
    """将numpy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

# ==================== API路由 ====================

@app.route('/api/predict/single', methods=['POST'])
@login_required
def api_predict_single():
    """单序列预测API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        sequence = data.get('sequence', '').strip().upper()
        
        # 验证序列
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        if not sequence:
            return jsonify({'error': 'Sequence is empty'}), 400
        if not all(aa in valid_aa for aa in sequence):
            return jsonify({'error': 'Invalid sequence: contains non-standard amino acids'}), 400
        
        # 模型预测
        pred_result = model_manager.predict(sequence)
        if 'error' in pred_result:
            print(f"预测错误: {pred_result['error']}")
            return jsonify({'error': pred_result['error']}), 500
        
        # 生成解释
        try:
            explanation = model_manager.explain(sequence, pred_result['probability'])
        except Exception as e:
            print(f"解释生成错误: {e}")
            explanation = {'error': str(e)}
        
        # 生成可视化
        try:
            visualization = model_manager.generate_visualization(
                sequence, pred_result['probability']
            )
        except Exception as e:
            print(f"可视化生成错误: {e}")
            visualization = None
        
        # 转换numpy类型为Python原生类型
        response_data = convert_to_native({
            'sequence': sequence,
            'prediction': pred_result,
            'explanation': explanation,
            'visualization': visualization
        })
        
        return jsonify(response_data)
    except Exception as e:
        import traceback
        print(f"API错误: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/predict/batch', methods=['POST'])
@login_required
def api_predict_batch():
    """批量预测API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        sequences = data.get('sequences', [])
        prediction_name = data.get('name', 'Batch Prediction')
        
        if not sequences:
            return jsonify({'error': 'No sequences provided'}), 400
        
        results = []
        for seq_data in sequences:
            try:
                seq_id = seq_data.get('id', '')
                sequence = seq_data.get('sequence', '').strip().upper()
                
                valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
                if not sequence or not all(aa in valid_aa for aa in sequence):
                    results.append({
                        'sequence_id': seq_id,
                        'sequence': sequence,
                        'error': 'Invalid sequence'
                    })
                    continue
                
                # 预测
                pred_result = model_manager.predict(sequence)
                if 'error' in pred_result:
                    results.append({
                        'sequence_id': seq_id,
                        'sequence': sequence,
                        'error': pred_result['error']
                    })
                    continue
                
                # 解释（不生成可视化，只存储数据）
                try:
                    explanation = model_manager.explain(sequence, pred_result['probability'])
                except Exception as e:
                    print(f"解释生成错误: {e}")
                    explanation = {'error': str(e)}
                
                results.append({
                    'sequence_id': seq_id,
                    'sequence': sequence,
                    'prediction': pred_result,
                    'explanation': explanation
                })
            except Exception as e:
                print(f"处理序列 {seq_id} 时出错: {e}")
                results.append({
                    'sequence_id': seq_id,
                    'sequence': sequence,
                    'error': str(e)
                })
        
        # 转换numpy类型为Python原生类型（不自动保存）
        response_data = convert_to_native({
            'results': results,
            'unsaved': True  # 标记为未保存状态
        })
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        print(f"批量预测API错误: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/visualize', methods=['POST'])
@login_required
def api_visualize():
    """生成可视化API（现场生成）"""
    data = request.get_json()
    sequence = data.get('sequence', '')
    model_probability = data.get('probability', 0.5)
    
    visualization = model_manager.generate_visualization(sequence, model_probability)
    
    if visualization:
        return jsonify({'visualization': visualization})
    else:
        return jsonify({'error': 'Failed to generate visualization'}), 500

@app.route('/api/save_prediction', methods=['POST'])
@login_required
def api_save_prediction():
    """手动保存预测结果"""
    try:
        data = request.get_json()
        print(f"收到保存请求: {data.keys() if data else 'No data'}")
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prediction_name = data.get('name', '未命名预测')
        results = data.get('results', [])
        prediction_type = data.get('type', 'single')  # 'single' 或 'batch'
        
        print(f"预测名称: {prediction_name}, 类型: {prediction_type}, 结果数量: {len(results)}")
        
        if not results:
            return jsonify({'error': 'No results to save'}), 400
        
        # 保存到数据库
        with get_db() as conn:
            # 检查是否有重名，如果有则自动添加序号
            original_name = prediction_name
            counter = 1
            while True:
                existing = conn.execute(
                    'SELECT id FROM predictions WHERE user_id = ? AND prediction_name = ?',
                    (session['user_id'], prediction_name)
                ).fetchone()
                if not existing:
                    break
                counter += 1
                prediction_name = f"{original_name} ({counter})"
            
            if prediction_name != original_name:
                print(f"名称重复，已自动重命名为: {prediction_name}")
            
            # 创建主记录
            cursor = conn.execute(
                '''INSERT INTO predictions (user_id, prediction_name, sequence_count, result_summary, prediction_type)
                   VALUES (?, ?, ?, ?, ?)''',
                (session['user_id'], prediction_name, len(results),
                 json.dumps({'total': len(results), 'afp': sum(1 for r in results if r.get('prediction', {}).get('is_afp', False))}),
                 prediction_type)
            )
            prediction_id = cursor.lastrowid
            print(f"创建主记录成功, ID: {prediction_id}")
            
            # 保存详情
            for i, result in enumerate(results):
                if 'error' in result:
                    print(f"跳过错误结果 {i}: {result.get('error')}")
                    continue
                
                exp = result.get('explanation', {})
                features = exp.get('features', {}) if isinstance(exp, dict) else {}
                
                print(f"保存详情 {i}: seq_id={result.get('sequence_id')}, seq={result.get('sequence', '')[:20]}...")
                
                # 转换numpy类型为Python原生类型
                features_native = convert_to_native(features)
                
                conn.execute(
                    '''INSERT INTO prediction_details 
                       (prediction_id, sequence_id, sequence, is_afp, model_probability,
                        threshold, confidence_level, reliability_score, confidence_reasons,
                        explanation_summary, recommendation, sequence_length, net_charge,
                        cationic_ratio, hydrophobicity_mean, helix_propensity,
                        afp_motifs_count, non_afp_patterns, features_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (prediction_id, result.get('sequence_id', ''), result.get('sequence', ''),
                     1 if result.get('prediction', {}).get('is_afp', False) else 0,
                     result.get('prediction', {}).get('probability', 0), ThresholdConfig.get(),
                     exp.get('confidence_level', 'Unknown'), exp.get('reliability_score', 0),
                     json.dumps(exp.get('confidence', {}).get('reasons', [])),
                     exp.get('explanation', '')[:500] if len(exp.get('explanation', '')) > 500 else exp.get('explanation', ''),
                     'See full explanation',
                     features_native.get('basic', {}).get('length', 0),
                     features_native.get('physicochemical', {}).get('charge', {}).get('net', 0),
                     features_native.get('composition', {}).get('group_frequency', {}).get('cationic_ratio', 0),
                     features_native.get('physicochemical', {}).get('hydrophobicity', {}).get('mean', 0),
                     features_native.get('structural', {}).get('helix_propensity', 0),
                     len(features_native.get('motifs', {}).get('afp_motifs_found', [])),
                     json.dumps(features_native.get('motifs', {}).get('non_afp_patterns', [])),
                     json.dumps(features_native))
                )
            
            conn.commit()
        
        return jsonify({
            'message': '预测结果已保存',
            'prediction_id': prediction_id,
            'prediction_name': prediction_name
        })
    
    except Exception as e:
        import traceback
        print(f"保存预测错误: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Save error: {str(e)}'}), 500

@app.route('/api/history', methods=['GET'])
@login_required
def api_history():
    """获取历史记录API"""
    with get_db() as conn:
        records = conn.execute(
            '''SELECT p.*, COUNT(pd.id) as detail_count 
               FROM predictions p 
               LEFT JOIN prediction_details pd ON p.id = pd.prediction_id 
               WHERE p.user_id = ? 
               GROUP BY p.id 
               ORDER BY p.created_at DESC''',
            (session['user_id'],)
        ).fetchall()
    
    return jsonify([{
        'id': r['id'],
        'name': r['prediction_name'],
        'sequence_count': r['sequence_count'],
        'prediction_type': r.get('prediction_type', 'single'),
        'result_summary': json.loads(r['result_summary']) if r['result_summary'] else {},
        'created_at': r['created_at']
    } for r in records])

@app.route('/api/history/<int:prediction_id>', methods=['GET'])
@login_required
def api_history_detail(prediction_id):
    """获取历史记录详情"""
    with get_db() as conn:
        # 验证权限
        record = conn.execute(
            'SELECT * FROM predictions WHERE id = ? AND user_id = ?',
            (prediction_id, session['user_id'])
        ).fetchone()
        
        if not record:
            return jsonify({'error': 'Not found'}), 404
        
        details = conn.execute(
            '''SELECT id, sequence_id, sequence, is_afp, model_probability,
                      confidence_level, reliability_score, sequence_length,
                      net_charge, cationic_ratio
               FROM prediction_details WHERE prediction_id = ?''',
            (prediction_id,)
        ).fetchall()
    
    return jsonify({
        'prediction': {
            'id': record['id'],
            'name': record['prediction_name'],
            'created_at': record['created_at']
        },
        'details': [{
            'id': d['id'],
            'sequence_id': d['sequence_id'],
            'sequence': d['sequence'],
            'is_afp': bool(d['is_afp']),
            'probability': d['model_probability'],
            'confidence_level': d['confidence_level'],
            'reliability_score': d['reliability_score'],
            'length': d['sequence_length'],
            'net_charge': d['net_charge'],
            'cationic_ratio': d['cationic_ratio']
        } for d in details]
    })

@app.route('/api/history/<int:prediction_id>/visualize', methods=['GET'])
@login_required
def api_history_visualize_redirect(prediction_id):
    """跳转到可视化详情页面（根据序列数量决定跳转目标）"""
    with get_db() as conn:
        record = conn.execute(
            'SELECT * FROM predictions WHERE id = ? AND user_id = ?',
            (prediction_id, session['user_id'])
        ).fetchone()
        
        if not record:
            return jsonify({'error': 'Not found'}), 404
        
        details = conn.execute(
            'SELECT sequence FROM prediction_details WHERE prediction_id = ?',
            (prediction_id,)
        ).fetchall()
        
        sequences = [d['sequence'] for d in details]
        # sqlite3.Row 使用字典式访问
        try:
            prediction_type = record['prediction_type']
        except (KeyError, IndexError):
            prediction_type = 'single'
    
    # 根据 prediction_type 决定跳转目标
    from urllib.parse import quote
    if prediction_type == 'batch' or len(sequences) > 1:
        # 批量预测：跳转到批量预测页面并自动填入（使用逗号分隔并URL编码）
        seqs = ','.join(sequences)
        return redirect(f'/predict/batch?sequences={quote(seqs)}&from_history={prediction_id}')
    else:
        # 单序列预测：跳转到单预测页面并自动填入
        seq = sequences[0] if sequences else ''
        return redirect(f'/predict/single?sequence={seq}&from_history={prediction_id}')

@app.route('/api/history/<int:prediction_id>/export', methods=['GET'])
@login_required
def api_export(prediction_id):
    """导出CSV"""
    with get_db() as conn:
        record = conn.execute(
            'SELECT * FROM predictions WHERE id = ? AND user_id = ?',
            (prediction_id, session['user_id'])
        ).fetchone()
        
        if not record:
            return jsonify({'error': 'Not found'}), 404
        
        details = conn.execute(
            'SELECT * FROM prediction_details WHERE prediction_id = ?',
            (prediction_id,)
        ).fetchall()
    
    # 生成CSV
    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['Sequence ID', 'Sequence', 'Prediction', 'Probability',
                     'Confidence Level', 'Reliability Score', 'Length', 'Net Charge'])
    
    for d in details:
        writer.writerow([
            d['sequence_id'], d['sequence'],
            'AFP' if d['is_afp'] else 'non-AFP',
            f"{d['model_probability']:.4f}",
            d['confidence_level'], d['reliability_score'],
            d['sequence_length'], d['net_charge']
        ])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'prediction_{prediction_id}.csv'
    )

@app.route('/api/history/<int:prediction_id>', methods=['DELETE'])
@login_required
def api_delete_history(prediction_id):
    """删除历史记录"""
    try:
        with get_db() as conn:
            # 验证权限
            record = conn.execute(
                'SELECT * FROM predictions WHERE id = ? AND user_id = ?',
                (prediction_id, session['user_id'])
            ).fetchone()
            
            if not record:
                return jsonify({'error': 'Not found or no permission'}), 404
            
            # 先删除详情记录
            conn.execute('DELETE FROM prediction_details WHERE prediction_id = ?', (prediction_id,))
            
            # 再删除主记录
            conn.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
            conn.commit()
            
            print(f"✓ 历史记录已删除: ID {prediction_id}")
            
            return jsonify({
                'message': 'Prediction deleted successfully',
                'prediction_id': prediction_id
            })
    except Exception as e:
        import traceback
        print(f"删除历史记录错误: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== 阈值管理 API ====================
@app.route('/api/threshold', methods=['GET'])
@login_required
def get_threshold():
    """获取当前阈值"""
    return jsonify({
        'threshold': ThresholdConfig.get(),
        'default': DEFAULT_THRESHOLD
    })

@app.route('/api/threshold', methods=['POST'])
@login_required
def set_threshold():
    """设置新阈值"""
    try:
        data = request.get_json()
        new_threshold = data.get('threshold')
        
        if new_threshold is None:
            return jsonify({'error': 'Missing threshold value'}), 400
        
        new_threshold = float(new_threshold)
        if not 0.0 <= new_threshold <= 1.0:
            return jsonify({'error': 'Threshold must be between 0 and 1'}), 400
        
        ThresholdConfig.set(new_threshold)
        print(f"✓ 阈值已更新: {new_threshold}")
        
        return jsonify({
            'message': 'Threshold updated successfully',
            'threshold': new_threshold,
            'default': DEFAULT_THRESHOLD
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/threshold/reset', methods=['POST'])
@login_required
def reset_threshold():
    """重置为默认阈值"""
    ThresholdConfig.set(DEFAULT_THRESHOLD)
    print(f"✓ 阈值已重置为默认值: {DEFAULT_THRESHOLD}")
    return jsonify({
        'message': 'Threshold reset to default',
        'threshold': DEFAULT_THRESHOLD,
        'default': DEFAULT_THRESHOLD
    })

# ==================== 启动 ====================
if __name__ == '__main__':
    init_db()
    model_manager.initialize()
    
    print("=" * 60)
    print("AFP预测平台 V4")
    print("=" * 60)
    print(f"访问地址: http://localhost:5001")
    print(f"数据库: {DATABASE}")
    if model_manager.model_loaded:
        print(f"V4数据增强模型: 已加载 (95.52%准确率)")
    else:
        print(f"V4数据增强模型: 未加载")
    print(f"分类阈值: {ThresholdConfig.get()} (可调整)")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)