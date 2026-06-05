#!/usr/bin/env python3
"""
项目完整性验证脚本
检查所有必需文件和路径是否正确
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def check_file(filepath, description):
    """检查文件是否存在"""
    full_path = os.path.join(PROJECT_ROOT, filepath)
    exists = os.path.exists(full_path)
    status = "✓" if exists else "✗"
    print(f"{status} {filepath:50s} - {description}")
    return exists

def check_directory(dirpath, description):
    """检查目录是否存在"""
    full_path = os.path.join(PROJECT_ROOT, dirpath)
    exists = os.path.isdir(full_path)
    status = "✓" if exists else "✗"
    print(f"{status} {dirpath:50s} - {description}")
    return exists

print("=" * 70)
print("AFP预测平台 - 项目完整性检查")
print("=" * 70)
print()

print("📁 核心目录检查:")
print("-" * 70)
dirs_ok = all([
    check_directory("templates", "HTML模板目录"),
    check_directory("static", "静态资源目录"),
    check_directory("static/css", "CSS样式目录"),
    check_directory("static/js", "JavaScript脚本目录"),
    check_directory("Codes", "训练代码目录"),
    check_directory("model", "模型文件目录"),
    check_directory("dataset", "数据集目录"),
])
print()

print("📄 核心文件检查:")
print("-" * 70)
files_ok = all([
    check_file("app.py", "Flask主应用"),
    check_file("requirements.txt", "依赖配置文件"),
    check_file("README.md", "项目文档"),
    check_file("Codes/bio_explanation_system.py", "生物解释系统"),
    check_file("Codes/visualize_explanation.py", "可视化系统"),
    check_file("Codes/features.py", "特征提取模块"),
    check_file("Codes/model.py", "模型定义"),
    check_file("Codes/train_with_augmentation.py", "数据增强训练"),
    check_file("Codes/data_augmentation.py", "数据增强工具"),
    check_file("train.py", "主训练脚本"),
    check_file("model/afp_model_augmented.pth", "V4增强模型"),
])
print()

print("🎨 HTML模板检查:")
print("-" * 70)
templates_ok = all([
    check_file("templates/base.html", "基础模板"),
    check_file("templates/index.html", "首页模板"),
    check_file("templates/login.html", "登录模板"),
    check_file("templates/register.html", "注册模板"),
    check_file("templates/dashboard.html", "仪表盘模板"),
    check_file("templates/predict_single.html", "单序列预测模板"),
    check_file("templates/predict_batch.html", "批量预测模板"),
    check_file("templates/history.html", "历史记录模板"),
])
print()

print("📊 数据集文件检查:")
print("-" * 70)
datasets_ok = all([
    check_file("dataset/DeepAFP-main-train.csv", "主训练集"),
    check_file("dataset/DeepAFP-main-test.csv", "主测试集"),
    check_file("dataset/DeepAFP-Set1-train.csv", "Set1训练集"),
    check_file("dataset/DeepAFP-Set1-test.csv", "Set1测试集"),
    check_file("dataset/DeepAFP-Set2-train.csv", "Set2训练集"),
    check_file("dataset/DeepAFP-Set2-test.csv", "Set2测试集"),
])
print()

print("🎨 静态资源检查:")
print("-" * 70)
static_ok = all([
    check_file("static/css/style.css", "主样式表"),
    check_file("static/js/main.js", "主脚本"),
])
print()

print("🚀 启动脚本检查:")
print("-" * 70)
startup_ok = all([
    check_file("start.sh", "Linux启动脚本"),
    check_file("start.bat", "Windows启动脚本"),
])
print()

print("=" * 70)
print("检查结果汇总:")
print("=" * 70)
all_ok = dirs_ok and files_ok and templates_ok and datasets_ok and static_ok and startup_ok

if all_ok:
    print("✓✓✓ 所有文件检查通过！项目结构完整！")
    print()
    print("下一步操作:")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 初始化数据库: python app.py (会自动创建)")
    print("3. 启动服务: python app.py")
    print("4. 访问平台: http://localhost:5001")
    print()
    sys.exit(0)
else:
    print("✗✗✗ 发现缺失文件！请检查上述失败项目。")
    print()
    sys.exit(1)
