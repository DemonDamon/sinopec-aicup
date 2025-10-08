"""
中国石化岩性识别AI竞赛 - 工具模块
包含数据加载、模型保存、可视化等通用功能
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import logging
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

from config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, SAMPLE_SUBMISSION_PATH,
    MODEL_SAVE_PATH, LOG_PATH, OUTPUT_PATH,
    FEATURE_COLUMNS, TARGET_COLUMN, SUBMISSION_TARGET_COLUMN, GROUP_COLUMN, DEPTH_COLUMN, ID_COLUMN,
    LITHOLOGY_MAPPING, EVALUATION_METRICS, LOGGING_CONFIG
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_logging():
    """设置日志配置"""
    log_file = LOG_PATH / f"lithology_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ] if LOGGING_CONFIG['console_handler'] else [
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__)

def load_data(data_type='train'):
    """
    加载数据
    
    Args:
        data_type: 数据类型 ('train', 'test', 'sample_submission')
        
    Returns:
        加载的数据
    """
    if data_type == 'train':
        data_path = TRAIN_DATA_PATH
    elif data_type == 'test':
        data_path = TEST_DATA_PATH
    elif data_type == 'sample_submission':
        data_path = SAMPLE_SUBMISSION_PATH
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    print(f"加载数据: {data_path}")
    data = pd.read_csv(data_path)
    print(f"数据形状: {data.shape}")
    
    return data

def save_model(model, model_path, model_type='sklearn'):
    """
    保存模型
    
    Args:
        model: 模型对象
        model_path: 模型文件路径（完整路径）
        model_type: 模型类型 ('sklearn', 'torch', 'custom')
    """
    # 确保路径是 Path 对象
    model_path = Path(model_path)
    
    # 确保父目录存在
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'sklearn':
        joblib.dump(model, model_path)
    elif model_type == 'torch':
        import torch
        torch.save(model.state_dict(), model_path)
    elif model_type == 'custom':
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"模型已保存: {model_path}")
    return model_path

def load_model(model_name, model_type='sklearn', model_class=None):
    """
    加载模型
    
    Args:
        model_name: 模型名称
        model_type: 模型类型 ('sklearn', 'torch', 'custom')
        model_class: 模型类（用于torch模型）
        
    Returns:
        加载的模型
    """
    model_dir = MODEL_SAVE_PATH / model_name
    
    if model_type == 'sklearn':
        model_path = model_dir / f"{model_name}.pkl"
        model = joblib.load(model_path)
    elif model_type == 'torch':
        model_path = model_dir / f"{model_name}.pth"
        import torch
        if model_class is None:
            raise ValueError("加载torch模型需要提供model_class")
        model = model_class()
        model.load_state_dict(torch.load(model_path))
    elif model_type == 'custom':
        model_path = model_dir / f"{model_name}.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    print(f"模型已加载: {model_path}")
    return model

def save_predictions(predictions, prediction_name, include_timestamp=True):
    """
    保存预测结果
    
    Args:
        predictions: 预测结果
        prediction_name: 预测文件名
        include_timestamp: 是否包含时间戳
        
    Returns:
        保存路径
    """
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prediction_name}_{timestamp}.csv"
    else:
        filename = f"{prediction_name}.csv"
    
    output_path = OUTPUT_PATH / filename
    predictions.to_csv(output_path, index=False)
    
    print(f"预测结果已保存: {output_path}")
    return output_path

def create_submission_file(test_ids, predictions, filename='submission.csv'):
    """
    创建提交文件
    
    Args:
        test_ids: 测试集ID
        predictions: 预测结果
        filename: 文件名
        
    Returns:
        提交文件路径
    """
    submission = pd.DataFrame({
        ID_COLUMN: test_ids,
        SUBMISSION_TARGET_COLUMN: predictions
    })
    
    submission_path = OUTPUT_PATH / filename
    submission.to_csv(submission_path, index=False)
    
    print(f"提交文件已创建: {submission_path}")
    return submission_path

def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    评估模型性能
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率
        model_name: 模型名称
        
    Returns:
        评估结果字典
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, log_loss
    )
    
    results = {}
    
    # 基本指标
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    results['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    results['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    
    # 如果有预测概率，计算AUC和log loss
    if y_pred_proba is not None:
        try:
            results['auc_macro'] = roc_auc_score(y_true, y_pred_proba, average='macro', multi_class='ovr')
            results['log_loss'] = log_loss(y_true, y_pred_proba)
        except:
            pass
    
    # 打印结果
    print(f"\n{model_name} 评估结果:")
    print("-" * 40)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    return results

def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        model_name: 模型名称
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[LITHOLOGY_MAPPING[i] for i in sorted(LITHOLOGY_MAPPING.keys())],
                yticklabels=[LITHOLOGY_MAPPING[i] for i in sorted(LITHOLOGY_MAPPING.keys())])
    plt.title(f'{model_name} - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
    
    plt.show()

def plot_feature_importance(feature_names, importance_values, model_name="Model", 
                          top_k=20, save_path=None):
    """
    绘制特征重要性
    
    Args:
        feature_names: 特征名称列表
        importance_values: 特征重要性值列表
        model_name: 模型名称
        top_k: 显示前k个重要特征
        save_path: 保存路径
    """
    # 创建特征重要性DataFrame并排序
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False)
    
    # 取前k个特征
    top_features = feature_importance.head(top_k)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'{model_name} - 特征重要性 (Top {top_k})')
    plt.xlabel('重要性')
    plt.ylabel('特征')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存: {save_path}")
    
    plt.show()
    
    return feature_importance

def plot_data_distribution(data, save_path=None):
    """
    绘制数据分布图
    
    Args:
        data: 数据
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 特征分布
    for i, col in enumerate(FEATURE_COLUMNS):
        ax = axes[i//2, i%2]
        data[col].hist(bins=50, ax=ax, alpha=0.7)
        ax.set_title(f'{col} 分布')
        ax.set_xlabel(col)
        ax.set_ylabel('频次')
    
    # 如果有目标变量，绘制类别分布
    if TARGET_COLUMN in data.columns:
        axes[1, 1].clear()
        target_counts = data[TARGET_COLUMN].value_counts()
        target_labels = [LITHOLOGY_MAPPING[i] for i in target_counts.index]
        axes[1, 1].bar(target_labels, target_counts.values)
        axes[1, 1].set_title('岩性类别分布')
        axes[1, 1].set_xlabel('岩性类别')
        axes[1, 1].set_ylabel('样本数量')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"数据分布图已保存: {save_path}")
    
    plt.show()

def plot_well_distribution(data, save_path=None):
    """
    绘制井分布图
    
    Args:
        data: 数据
        save_path: 保存路径
    """
    well_counts = data[GROUP_COLUMN].value_counts()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    well_counts.plot(kind='bar')
    plt.title('各井样本数量分布')
    plt.xlabel('井号')
    plt.ylabel('样本数量')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    well_counts.hist(bins=20)
    plt.title('井样本数量分布直方图')
    plt.xlabel('样本数量')
    plt.ylabel('井数量')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"井分布图已保存: {save_path}")
    
    plt.show()

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_experiment_results(results, experiment_name):
    """
    保存实验结果
    
    Args:
        results: 实验结果字典
        experiment_name: 实验名称
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{experiment_name}_{timestamp}.json"
    results_path = OUTPUT_PATH / filename
    
    # 添加时间戳和实验信息
    results['timestamp'] = timestamp
    results['experiment_name'] = experiment_name
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print(f"实验结果已保存: {results_path}")
    return results_path

def load_experiment_results(results_path):
    """
    加载实验结果
    
    Args:
        results_path: 结果文件路径
        
    Returns:
        实验结果字典
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

def set_random_seed(seed=42):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    print(f"随机种子已设置: {seed}")

def check_data_leakage(train_data, test_data):
    """
    检查数据泄漏
    
    Args:
        train_data: 训练数据
        test_data: 测试数据
        
    Returns:
        检查结果
    """
    results = {}
    
    # 检查井是否重叠
    train_wells = set(train_data[GROUP_COLUMN].unique())
    test_wells = set(test_data[GROUP_COLUMN].unique())
    overlapping_wells = train_wells.intersection(test_wells)
    
    results['overlapping_wells'] = list(overlapping_wells)
    results['n_overlapping_wells'] = len(overlapping_wells)
    results['train_wells'] = len(train_wells)
    results['test_wells'] = len(test_wells)
    
    # 检查深度范围是否重叠
    if DEPTH_COLUMN in train_data.columns and DEPTH_COLUMN in test_data.columns:
        train_depth_range = (train_data[DEPTH_COLUMN].min(), train_data[DEPTH_COLUMN].max())
        test_depth_range = (test_data[DEPTH_COLUMN].min(), test_data[DEPTH_COLUMN].max())
        
        results['train_depth_range'] = train_depth_range
        results['test_depth_range'] = test_depth_range
        
        # 检查深度重叠
        depth_overlap = not (train_depth_range[1] < test_depth_range[0] or 
                           test_depth_range[1] < train_depth_range[0])
        results['depth_overlap'] = depth_overlap
    
    print("数据泄漏检查结果:")
    print(f"重叠井数量: {results['n_overlapping_wells']}")
    if results['n_overlapping_wells'] > 0:
        print(f"重叠的井: {results['overlapping_wells']}")
    
    return results

def memory_usage_check():
    """检查内存使用情况"""
    import psutil
    
    memory = psutil.virtual_memory()
    print(f"内存使用情况:")
    print(f"总内存: {memory.total / 1024**3:.2f} GB")
    print(f"已用内存: {memory.used / 1024**3:.2f} GB")
    print(f"可用内存: {memory.available / 1024**3:.2f} GB")
    print(f"内存使用率: {memory.percent:.1f}%")
    
    return memory

def create_directory_structure():
    """创建项目目录结构"""
    directories = [MODEL_SAVE_PATH, LOG_PATH, OUTPUT_PATH]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"目录已创建: {directory}")

if __name__ == "__main__":
    # 测试代码
    print("测试工具模块...")
    
    # 设置随机种子
    set_random_seed(42)
    
    # 创建目录结构
    create_directory_structure()
    
    # 设置日志
    logger = setup_logging()
    logger.info("工具模块测试开始")
    
    # 检查内存使用
    memory_usage_check()
    
    # 加载数据测试
    try:
        train_data = load_data('train')
        print(f"训练数据加载成功: {train_data.shape}")
        
        # 绘制数据分布
        plot_data_distribution(train_data)
        
        # 绘制井分布
        plot_well_distribution(train_data)
        
    except Exception as e:
        print(f"数据加载失败: {e}")
    
    print("工具模块测试完成!")