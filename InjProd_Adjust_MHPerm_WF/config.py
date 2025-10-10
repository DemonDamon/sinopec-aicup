"""
配置文件 - 中国石化AI竞赛油井含水率预测
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / "dataset" / "InjProd_Adjust_MHPerm_WF"

# 数据文件路径
DATA_PATHS = {
    # 静态数据
    'well_info': DATA_ROOT / "坐标_well_info.csv",
    'well_deviation': DATA_ROOT / "井斜_well_deviation.csv", 
    'perforation': DATA_ROOT / "射孔数据表_well_perforation.csv",
    
    # 动态数据
    'daily_oil_train': DATA_ROOT / "单井油井日度数据train_daily_oil.csv",
    'daily_water': DATA_ROOT / "单井水井日度数据_daily_water.csv",
    'monthly_oil': DATA_ROOT / "单井油井月度数据__monthly_oil.csv",
    'monthly_water': DATA_ROOT / "单井水井月度数据_monthly_water.csv",
    
    # 预测数据
    'validation': DATA_ROOT / "单井油井日度待预测数据_validation_without_truth.csv",
    'sample_submission': DATA_ROOT / "sample_submission.csv"
}

# 输出路径
OUTPUT_PATHS = {
    'models': PROJECT_ROOT / "outputs" / "models",
    'features': PROJECT_ROOT / "outputs" / "features", 
    'submissions': PROJECT_ROOT / "outputs" / "submissions",
    'logs': PROJECT_ROOT / "outputs" / "logs"
}

# 创建输出目录
for path in OUTPUT_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# 模型参数
MODEL_CONFIG = {
    'lightgbm': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    },
    'xgboost': {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'random_state': 42
    }
}

# 特征工程参数
FEATURE_CONFIG = {
    'lag_days': [1, 3, 7, 14, 30],  # 滞后天数
    'rolling_windows': [3, 7, 14, 30],  # 滑动窗口大小
    'spatial_neighbors': 5,  # 空间邻居数量
    'min_samples_for_stats': 10  # 计算统计量的最小样本数
}

# 验证参数
VALIDATION_CONFIG = {
    'n_splits': 5,  # 时序交叉验证折数
    'test_size': 0.2,  # 测试集比例
    'gap': 0,  # 训练集和验证集之间的间隔天数
    'random_state': 42
}

# 目标变量
TARGET_COL = '含水'

# 时间列
TIME_COL = '日期'

# 井号列
WELL_COL = '井号'

# 预测ID列
ID_COL = 'id'

# 日期格式
DATE_FORMAT = '%Y/%m/%d'

# 随机种子
RANDOM_STATE = 42

# 并行处理参数
N_JOBS = -1

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': OUTPUT_PATHS['logs'] / 'training.log'
}