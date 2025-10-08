"""
中国石化岩性识别AI竞赛 - 配置文件
包含所有模型参数、路径配置和超参数设置
"""

import os
from pathlib import Path

# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / "dataset" / "LithoClass_WellLog"

# 数据文件路径
TRAIN_DATA_PATH = DATA_ROOT / "train.csv"
TEST_DATA_PATH = DATA_ROOT / "validation_without_label.csv"
SAMPLE_SUBMISSION_PATH = DATA_ROOT / "sample_submission.csv"

# 输出路径
MODEL_SAVE_PATH = PROJECT_ROOT / "models"
LOG_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "output"

# 创建必要的目录
for path in [MODEL_SAVE_PATH, LOG_PATH, OUTPUT_PATH]:
    path.mkdir(exist_ok=True)

# ==================== 数据配置 ====================
# 数据列配置
FEATURE_COLUMNS = ['SP', 'GR', 'AC']  # 测井曲线特征
TARGET_COLUMN = 'label'  # 目标列（训练数据中的标签列）
SUBMISSION_TARGET_COLUMN = 'predict'  # 提交文件中的预测列名
GROUP_COLUMN = 'WELL'  # 分组列（用于GroupKFold）
DEPTH_COLUMN = 'DEPTH'  # 深度列
ID_COLUMN = 'id'  # ID列

# 岩性标签映射
LITHOLOGY_MAPPING = {
    0: '粉砂岩',
    1: '砂岩', 
    2: '泥岩'
}

# 类别权重（用于处理不平衡数据）
CLASS_WEIGHTS = {
    0: 1.0,
    1: 1.0,
    2: 1.0
}

# ==================== 数据预处理配置 ====================
# 异常值检测
OUTLIER_DETECTION = {
    'method': 'isolation_forest',
    'contamination': 0.1,
    'random_state': 42
}

# 缺失值填充
MISSING_VALUE_CONFIG = {
    'method': 'missforest',  # 'mean', 'median', 'missforest', 'lstm'
    'max_iter': 10,
    'random_state': 42
}

# 数据标准化
SCALING_CONFIG = {
    'method': 'standard',  # 'standard', 'minmax', 'robust'
    'feature_range': (0, 1)  # for minmax scaling
}

# ==================== 特征工程配置 ====================
# 滑动窗口特征
ROLLING_WINDOW_CONFIG = {
    'windows': [21, 51, 101],
    'features': ['mean', 'std', 'skew', 'kurt', 'min', 'max'],
    'center': True
}

# 梯度特征
GRADIENT_CONFIG = {
    'orders': [1, 2],  # 一阶和二阶梯度
    'method': 'gradient'  # 'diff', 'gradient'
}

# 测井相特征
LOG_FACIES_CONFIG = {
    'n_clusters': 8,
    'method': 'kmeans',  # 'kmeans', 'gmm'
    'random_state': 42,
    'include_distance': True
}

# 频域特征
FREQUENCY_CONFIG = {
    'methods': ['fft', 'wavelet'],
    'wavelet_name': 'db4',
    'wavelet_levels': 4,
    'fft_components': 10
}

# ==================== 模型配置 ====================
# 交叉验证配置
CV_CONFIG = {
    'method': 'group_kfold',
    'n_splits': 3,  # 修改为3，因为只有4个井
    'shuffle': True,
    'random_state': 42
}

# LightGBM配置
LIGHTGBM_CONFIG = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    'n_estimators': 1000,
    'early_stopping_rounds': 100,
    'class_weight': 'balanced'  # 处理不均衡数据
}

# CatBoost配置
CATBOOST_CONFIG = {
    'objective': 'MultiClass',
    'eval_metric': 'MultiClass',
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 100,
    'class_weights': [1.0, 1.0, 1.0]  # 可根据类别分布调整
}

# XGBoost配置
XGBOOST_CONFIG = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_estimators': 1000,
    'early_stopping_rounds': 100
}

# 深度学习模型配置
DEEP_LEARNING_CONFIG = {
    'model_type': '1d_cnn',  # '1d_cnn', 'gru', 'lstm'
    'sequence_length': 101,  # 序列长度
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.001,
    'dropout': 0.3,
    'early_stopping_patience': 10,
    'device': 'cpu'  # 'cuda' if available
}

# 1D-CNN特定配置
CNN_CONFIG = {
    'conv_layers': [
        {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 128, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 64, 'kernel_size': 3, 'activation': 'relu'}
    ],
    'pool_size': 2,
    'dense_layers': [128, 64],
    'output_classes': 3
}

# GRU/LSTM特定配置
RNN_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'bidirectional': True,
    'dense_layers': [128, 64],
    'output_classes': 3
}

# ==================== 集成学习配置 ====================
ENSEMBLE_CONFIG = {
    'method': 'stacking',  # 'voting', 'stacking'
    'meta_learner': 'logistic_regression',  # 'logistic_regression', 'lightgbm'
    'use_probabilities': True,
    'cv_folds': 5
}

# Stacking特定配置
STACKING_CONFIG = {
    'base_models': ['lightgbm', 'catboost', 'xgboost', 'deep_learning'],
    'meta_model_params': {
        'C': 1.0,
        'random_state': 42,
        'max_iter': 1000
    }
}

# ==================== 其他配置 ====================
# 随机种子
RANDOM_STATE = 42
RANDOM_SEED = 42

# 数据路径配置（兼容性）
DATA_PATHS = {
    'train': str(TRAIN_DATA_PATH),
    'test': str(TEST_DATA_PATH),
    'sample_submission': str(SAMPLE_SUBMISSION_PATH)
}

# 模型路径配置
MODEL_PATHS = {
    'save_dir': str(MODEL_SAVE_PATH),
    'lightgbm': str(MODEL_SAVE_PATH / 'lightgbm.pkl'),
    'catboost': str(MODEL_SAVE_PATH / 'catboost.pkl'),
    'xgboost': str(MODEL_SAVE_PATH / 'xgboost.pkl')
}

# 输出路径配置
OUTPUT_PATHS = {
    'predictions': str(OUTPUT_PATH / 'predictions'),
    'submissions': str(OUTPUT_PATH / 'submissions'),
    'experiments': str(OUTPUT_PATH / 'experiments'),
    'visualizations': str(OUTPUT_PATH / 'visualizations')
}

# 创建输出子目录
for path_key, path_value in OUTPUT_PATHS.items():
    Path(path_value).mkdir(parents=True, exist_ok=True)

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'console_handler': True
}

# 并行处理配置
PARALLEL_CONFIG = {
    'n_jobs': -1,  # 使用所有可用CPU核心
    'backend': 'threading'  # 'threading', 'multiprocessing'
}

# 模型评估指标
EVALUATION_METRICS = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']

# 特征重要性分析
FEATURE_IMPORTANCE_CONFIG = {
    'methods': ['permutation', 'shap'],
    'top_k': 20
}