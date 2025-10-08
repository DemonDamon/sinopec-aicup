"""
快速高效的多算法集成模型
使用LightGBM、CatBoost、XGBoost三种算法进行集成
基于空间连续性特征，优化训练时间，保持高性能
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import RANDOM_STATE, OUTPUT_PATHS, MODEL_PATHS
from utils import load_data, set_random_seed, save_model
from base_models import BaseModelTrainer

set_random_seed(RANDOM_STATE)

def create_spatial_features(data):
    """创建精简但高效的空间连续性特征"""
    print("🔬 创建空间连续性特征...")
    
    # 按井和深度排序
    data_sorted = data.sort_values(['WELL', 'DEPTH']).copy()
    
    # 基础特征
    features = data_sorted[['DEPTH', 'SP', 'GR', 'AC']].copy()
    
    # 为每口井创建空间特征
    for well in data_sorted['WELL'].unique():
        well_mask = data_sorted['WELL'] == well
        well_data = data_sorted[well_mask].copy()
        
        if len(well_data) < 3:
            continue
            
        # 精选的滑动窗口特征 (只用最有效的)
        for window in [5, 11]:  # 减少窗口数量
            for col in ['SP', 'GR', 'AC']:
                # 滑动平均
                features.loc[well_mask, f'{col}_ma_{window}'] = (
                    well_data[col].rolling(window, center=True, min_periods=1).mean()
                )
                # 滑动标准差
                features.loc[well_mask, f'{col}_std_{window}'] = (
                    well_data[col].rolling(window, center=True, min_periods=1).std().fillna(0)
                )
        
        # 梯度特征
        for col in ['SP', 'GR', 'AC']:
            gradient = np.gradient(well_data[col].values)
            features.loc[well_mask, f'{col}_gradient'] = gradient
    
    # 深度标准化
    for well in data_sorted['WELL'].unique():
        well_mask = data_sorted['WELL'] == well
        well_depths = features.loc[well_mask, 'DEPTH']
        if len(well_depths) > 1:
            features.loc[well_mask, 'DEPTH_normalized'] = (
                (well_depths - well_depths.min()) / (well_depths.max() - well_depths.min())
            )
        else:
            features.loc[well_mask, 'DEPTH_normalized'] = 0.5
    
    # 岩性识别特征组合
    features['GR_SP_ratio'] = features['GR'] / (features['SP'] + 1e-6)
    features['sandstone_indicator'] = (
        (features['GR'] < features['GR'].quantile(0.3)) & 
        (features['SP'] > features['SP'].quantile(0.7))
    ).astype(int)
    
    # 恢复原始顺序
    features = features.reindex(data.index)
    
    print(f"✅ 创建了 {features.shape[1]} 个特征")
    return features

def train_fast_ensemble():
    """快速训练集成模型"""
    print("🎯 训练快速空间连续性集成模型...")
    
    # 1. 加载数据
    print("📊 加载数据...")
    train_data = load_data('train')
    test_data = load_data('test')
    
    # 2. 创建空间特征
    train_features = create_spatial_features(train_data)
    test_features = create_spatial_features(test_data)
    
    # 3. 准备训练数据
    from config import TARGET_COLUMN, ID_COLUMN
    
    X = train_features
    y = train_data[TARGET_COLUMN]
    
    # 确保测试集有相同的特征
    common_features = list(set(X.columns) & set(test_features.columns))
    X = X[common_features]
    X_test = test_features[common_features]
    
    print(f"📈 最终特征数量: {len(common_features)}")
    
    # 4. 特征标准化
    print("🔧 特征标准化...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    # 5. 定义3种不同算法的模型配置
    model_configs = [
        {
            'name': 'lightgbm_fast',
            'model_type': 'lightgbm',
            'params': {
                'objective': 'multiclass',
                'num_class': 3,
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'n_estimators': 800,
                'random_state': 42,
                'verbose': -1,
                'force_col_wise': True,
                'early_stopping_rounds': None  # 明确禁用early stopping
            }
        },
        {
            'name': 'catboost_fast',
            'model_type': 'catboost',
            'params': {
                'iterations': 800,
                'depth': 8,
                'learning_rate': 0.08,
                'l2_leaf_reg': 5,
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.8,
                'colsample_bylevel': 0.9,
                'random_seed': 123,
                'verbose': False
            }
        },
        {
            'name': 'xgboost_fast',
            'model_type': 'xgboost',
            'params': {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'min_child_weight': 1,
                'n_estimators': 800,
                'random_state': 456,
                'verbosity': 0,
                'early_stopping_rounds': None  # 明确禁用early stopping
            }
        }
    ]
    
    # 6. 快速交叉验证评估 (只用3折)
    print("\\n🔄 快速评估各个基础模型...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  # 减少折数
    
    model_scores = {}
    trained_models = {}
    
    for config in model_configs:
        print(f"\\n📊 评估 {config['name']}...")
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_train_fold = X_scaled.iloc[train_idx]
            X_val_fold = X_scaled.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # 训练模型
            trainer = BaseModelTrainer(model_type=config['model_type'], model_params=config['params'])
            trainer.train(X_train_fold, y_train_fold, early_stopping_rounds=None, verbose=False)
            
            # 预测和评估
            y_pred = trainer.predict(X_val_fold)
            f1 = f1_score(y_val_fold, y_pred, average='macro')
            cv_scores.append(f1)
        
        mean_f1 = np.mean(cv_scores)
        std_f1 = np.std(cv_scores)
        model_scores[config['name']] = {'mean': mean_f1, 'std': std_f1}
        
        print(f"  {config['name']}: {mean_f1:.4f} ± {std_f1:.4f}")
        
        # 训练完整模型
        trainer = BaseModelTrainer(model_type=config['model_type'], model_params=config['params'])
        trainer.train(X_scaled, y, early_stopping_rounds=None, verbose=False)
        trained_models[config['name']] = trainer
    
    # 7. 集成预测
    print("\\n🔄 集成预测...")
    
    # 软投票集成
    ensemble_predictions = []
    individual_predictions = {}
    
    for name, model in trained_models.items():
        pred_proba = model.model.predict_proba(X_test_scaled)
        individual_predictions[name] = pred_proba
        ensemble_predictions.append(pred_proba)
    
    # 简单平均
    avg_proba = np.mean(ensemble_predictions, axis=0)
    final_predictions = np.argmax(avg_proba, axis=1)
    
    # 8. 加权集成
    print("\\n🔄 加权集成...")
    weights = []
    for config in model_configs:
        weight = model_scores[config['name']]['mean']
        weights.append(weight)
    
    # 归一化权重
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    print("模型权重:")
    for i, config in enumerate(model_configs):
        print(f"  {config['name']}: {weights[i]:.3f}")
    
    # 加权平均
    weighted_proba = np.zeros_like(avg_proba)
    for i, (name, pred_proba) in enumerate(individual_predictions.items()):
        weighted_proba += weights[i] * pred_proba
    
    weighted_predictions = np.argmax(weighted_proba, axis=1)
    
    # 9. 快速评估集成效果 (只用一次验证)
    print("\\n🔄 评估集成效果...")
    
    # 使用一次随机分割快速评估
    from sklearn.model_selection import train_test_split
    X_train_quick, X_val_quick, y_train_quick, y_val_quick = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # 训练所有模型
    quick_predictions = []
    quick_weights = []
    
    for config in model_configs:
        trainer = BaseModelTrainer(model_type=config['model_type'], model_params=config['params'])
        trainer.train(X_train_quick, y_train_quick, early_stopping_rounds=None, verbose=False)
        
        pred_proba = trainer.model.predict_proba(X_val_quick)
        quick_predictions.append(pred_proba)
        quick_weights.append(model_scores[config['name']]['mean'])
    
    # 加权集成
    quick_weights = np.array(quick_weights)
    quick_weights = quick_weights / quick_weights.sum()
    
    ensemble_proba = np.zeros_like(quick_predictions[0])
    for i, pred_proba in enumerate(quick_predictions):
        ensemble_proba += quick_weights[i] * pred_proba
    
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    ensemble_f1 = f1_score(y_val_quick, ensemble_pred, average='macro')
    
    print(f"\\n📊 集成模型快速评估F1: {ensemble_f1:.4f}")
    
    # 10. 分析预测分布
    print("\\n📊 预测分布:")
    weighted_dist = pd.Series(weighted_predictions).value_counts().sort_index()
    for k, v in weighted_dist.items():
        print(f"  类别{k}: {v} ({v/len(weighted_predictions)*100:.1f}%)")
    
    # 11. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    submission_file = f"submission_fast_ensemble_{timestamp}.csv"
    submission_path = os.path.join(OUTPUT_PATHS['predictions'], submission_file)
    
    if ID_COLUMN in test_data.columns:
        test_ids = test_data[ID_COLUMN].values
    else:
        test_ids = list(range(len(X_test_scaled)))
    
    submission_df = pd.DataFrame({
        'id': test_ids,
        'predict': weighted_predictions
    })
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\\n💾 提交文件: {submission_path}")
    
    return {
        'ensemble_f1': ensemble_f1,
        'model_scores': model_scores,
        'weights': weights,
        'submission_path': submission_path,
        'distribution': weighted_dist.to_dict()
    }

if __name__ == "__main__":
    result = train_fast_ensemble()
    
    print(f"\\n🎉 多算法集成模型训练完成!")
    print(f"📊 集成模型F1: {result['ensemble_f1']:.4f}")
    print(f"📄 推荐提交: {result['submission_path']}")
    print(f"⚡ 使用LightGBM+CatBoost+XGBoost三种算法集成!")