"""
模型训练脚本 - 中国石化AI竞赛油井含水率预测
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import (
    OUTPUT_PATHS, TARGET_COL, TIME_COL, WELL_COL, 
    VALIDATION_CONFIG, RANDOM_STATE
)
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import ModelTrainer, LightGBMModel, XGBoostModel, EnsembleModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_PATHS['logs'] / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """按时间分割数据"""
    logger.info("按时间分割训练和验证数据...")
    
    # 按时间排序
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    
    # 按时间分割
    split_idx = int((1 - test_size) * len(df))
    train_df = df[:split_idx].copy()
    val_df = df[split_idx:].copy()
    
    logger.info(f"训练集大小: {len(train_df)}")
    logger.info(f"验证集大小: {len(val_df)}")
    logger.info(f"训练集日期范围: {train_df[TIME_COL].min()} 到 {train_df[TIME_COL].max()}")
    logger.info(f"验证集日期范围: {val_df[TIME_COL].min()} 到 {val_df[TIME_COL].max()}")
    
    return train_df, val_df

def prepare_features_and_target(df: pd.DataFrame):
    """准备特征和目标变量"""
    # 排除非特征列
    exclude_cols = [TARGET_COL, TIME_COL, WELL_COL, 'id', '生产层位', 'year_month']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    
    logger.info(f"特征数量: {len(feature_cols)}")
    logger.info(f"样本数量: {len(X)}")
    
    return X, y

def train_models(X_train: pd.DataFrame, y_train: pd.Series, 
                X_val: pd.DataFrame, y_val: pd.Series, 
                optimize: bool = False):
    """训练模型"""
    logger.info("开始训练模型...")
    
    trainer = ModelTrainer()
    results = {}

    # 定义模型参数
    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': RANDOM_STATE,
        'boosting_type': 'gbdt',
        # training params
        'early_stopping_rounds': 100,
        'verbose_eval': 100
    }

    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'lambda': 1,
        'alpha': 0,
        'seed': RANDOM_STATE,
        'n_jobs': -1,
        # training params
        'early_stopping_rounds': 100,
        'verbose_eval': 100
    }

    if optimize:
        logger.info("进行超参数优化...")
        # 优化LightGBM
        lgb_params = trainer.optimize_hyperparameters(
            pd.concat([X_train, X_val]), 
            pd.concat([y_train, y_val]), 
            'lightgbm', 
            n_trials=50
        )
        
        # 优化XGBoost
        xgb_params = trainer.optimize_hyperparameters(
            pd.concat([X_train, X_val]), 
            pd.concat([y_train, y_val]), 
            'xgboost', 
            n_trials=50
        )

    # 训练和评估模型
    models_to_train = {
        'lightgbm': LightGBMModel(lgb_params),
        'xgboost': XGBoostModel(xgb_params),
    }
    
    trained_models = {}
    for name, model in models_to_train.items():
        logger.info(f"训练{name}模型...")
        try:
            model.fit(X_train, y_train, X_val, y_val)
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            mae = mean_absolute_error(y_val, pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'predictions': pred
            }
            trained_models[name] = model
            logger.info(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        except Exception as e:
            logger.error(f"训练 {name} 模型失败: {e}")

    # 只有在两个模型都成功训练后才训练集成模型
    if len(trained_models) == 2:
        logger.info("训练集成模型...")
        ensemble_model = EnsembleModel(models=list(trained_models.values()))
        ensemble_pred = ensemble_model.predict(X_val)
        ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
        
        results['ensemble'] = {
            'model': ensemble_model,
            'rmse': ensemble_rmse,
            'mae': ensemble_mae,
            'predictions': ensemble_pred
        }
        logger.info(f"集成模型 - RMSE: {ensemble_rmse:.4f}, MAE: {ensemble_mae:.4f}")

    return results

def save_models_and_results(results: dict):
    """保存模型和结果"""
    logger.info("保存模型和结果...")
    
    # 保存模型
    for name, result in results.items():
        model_path = OUTPUT_PATHS['models'] / f'{name}_model.pkl'
        joblib.dump(result['model'], model_path)
        logger.info(f"{name} 模型已保存到: {model_path}")
    
    # 保存结果摘要
    summary = {}
    for name, result in results.items():
        summary[name] = {
            'rmse': result['rmse'],
            'mae': result['mae']
        }
    
    summary_df = pd.DataFrame(summary).T
    summary_path = OUTPUT_PATHS['models'] / 'model_summary.csv'
    summary_df.to_csv(summary_path)
    logger.info(f"模型结果摘要已保存到: {summary_path}")
    
    # 保存特征重要性
    for name, result in results.items():
        model = result['model']
        if hasattr(model, 'feature_importance') and model.feature_importance is not None:
            importance_path = OUTPUT_PATHS['features'] / f'{name}_feature_importance.csv'
            model.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"{name} 特征重要性已保存到: {importance_path}")
        elif hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            if not importance.empty:
                importance_path = OUTPUT_PATHS['features'] / f'{name}_feature_importance.csv'
                importance.to_csv(importance_path, index=False)
                logger.info(f"{name} 特征重要性已保存到: {importance_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练油井含水率预测模型')
    parser.add_argument('--optimize', action='store_true', help='是否进行超参数优化')
    parser.add_argument('--test_size', type=float, default=0.2, help='验证集比例')
    
    args = parser.parse_args()
    
    logger.info("开始训练流程...")
    logger.info(f"超参数优化: {args.optimize}")
    logger.info(f"验证集比例: {args.test_size}")
    
    try:
        # 1. 加载数据
        logger.info("加载数据...")
        loader = DataLoader()
        data = loader.load_all_data()
        train_df = data['daily_oil_train'].copy()
        train_df = train_df.dropna(subset=[TARGET_COL])

        # 2. 创建特征
        logger.info("创建特征...")
        feature_engineer = FeatureEngineer(loader)
        train_features_full = feature_engineer.create_all_features(train_df, is_training=True)

        # 3. 特征选择 (填充 selected_columns)
        logger.info("进行特征选择...")
        # fit_select_features 现在返回筛选后的df，但我们在这里只需要它填充 selected_columns
        _ = feature_engineer.fit_select_features(train_features_full.copy())
        logger.info(f"特征选择完成，选出 {len(feature_engineer.selected_columns)} 个特征")

        # 4. 保存特征工程器
        logger.info("保存特征工程器...")
        joblib.dump(feature_engineer, OUTPUT_PATHS['models'] / 'feature_engineer.pkl')

        # 5. 数据分割
        logger.info("按时间分割训练和验证数据...")
        train_split_df, val_split_df = split_data(train_features_full, args.test_size)

        # 6. 准备训练/验证数据
        logger.info("准备最终的训练和验证集...")
        X_train = train_split_df[feature_engineer.selected_columns]
        y_train = train_split_df[TARGET_COL]
        X_val = val_split_df[feature_engineer.selected_columns]
        y_val = val_split_df[TARGET_COL]
        
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        logger.info(f"X_train dtypes:\n{X_train.dtypes.to_string()}")

        # 7. 训练模型
        results = train_models(X_train, y_train, X_val, y_val, args.optimize)
        
        # 8. 保存模型和结果
        save_models_and_results(results)
        
        # 打印最终结果
        logger.info("\n" + "="*50)
        logger.info("训练完成！最终结果:")
        logger.info("="*50)
        
        for name, result in results.items():
            logger.info(f"{name:>10} - RMSE: {result['rmse']:>7.4f}, MAE: {result['mae']:>7.4f}")
        
        # 找出最佳模型
        best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
        best_rmse = results[best_model_name]['rmse']
        logger.info(f"\n最佳模型: {best_model_name} (RMSE: {best_rmse:.4f})")
        
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()