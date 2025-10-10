"""
模型定义模块 - 中国石化AI竞赛油井含水率预测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import joblib
import optuna
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, VALIDATION_CONFIG, OUTPUT_PATHS, TARGET_COL, TIME_COL, WELL_COL

logger = logging.getLogger(__name__)

class BaseModel:
    """基础模型类"""
    
    def __init__(self, model_name: str, params: Dict):
        self.model_name = model_name
        self.params = params
        self.model = None
        self.feature_importance = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """训练模型"""
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        raise NotImplementedError

    def get_feature_names(self) -> List[str]:
        """获取模型使用的特征名称"""
        raise NotImplementedError
        
    def save_model(self, path: Path):
        """保存模型"""
        joblib.dump(self.model, path)
        logger.info(f"模型已保存到: {path}")
        
    def load_model(self, path: Path):
        """加载模型"""
        self.model = joblib.load(path)
        logger.info(f"模型已从 {path} 加载")

class LightGBMModel(BaseModel):
    """LightGBM模型"""
    
    def __init__(self, params: Optional[Dict] = None):
        if params is None:
            params = MODEL_CONFIG['lightgbm'].copy()
        super().__init__('lightgbm', params)
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """训练LightGBM模型"""
        logger.info("开始训练LightGBM模型...")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, val_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
        
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # 获取特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info(f"特征重要性计算完成，前5个特征：\n{self.feature_importance.head()}")
        logger.info("LightGBM模型训练完成")
        
    def get_feature_names(self) -> List[str]:
        """获取LightGBM模型的特征名称"""
        if self.model:
            return self.model.feature_name()
        return []
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")
        # 允许在特征填充或截取后进行预测，避免形状不匹配报错
        return self.model.predict(
            X,
            num_iteration=self.model.best_iteration
        )

class XGBoostModel(BaseModel):
    """XGBoost模型"""
    
    def __init__(self, params: Optional[Dict] = None):
        if params is None:
            params = MODEL_CONFIG['xgboost'].copy()
        super().__init__('xgboost', params)
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """训练XGBoost模型"""
        logger.info("开始训练XGBoost模型...")
        
        self.model = xgb.XGBRegressor(**self.params)
        
        # 简化训练，避免版本兼容问题
        self.model.fit(X_train, y_train)
        
        # 获取特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("XGBoost模型训练完成")
        
    def get_feature_names(self) -> List[str]:
        """获取XGBoost模型的特征名称"""
        if self.model:
            return self.model.get_booster().feature_names
        return []
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")
        return self.model.predict(X)
        
    def save(self, path):
        """保存模型"""
        joblib.dump(self.model, path)
        logger.info(f"模型已保存到: {path}")
        
    def load_model(self, path: Path):
        """加载模型"""
        self.model = joblib.load(path)
        logger.info(f"模型已从 {path} 加载")

class EnsembleModel:
    """集成模型"""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """训练所有模型"""
        logger.info("开始训练集成模型...")
        
        for i, model in enumerate(self.models):
            logger.info(f"训练第 {i+1}/{len(self.models)} 个模型: {model.model_name}")
            model.fit(X_train, y_train, X_val, y_val)
        
        logger.info("集成模型训练完成")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """集成预测"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # 加权平均
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取集成特征重要性"""
        importance_dfs = []
        
        for i, model in enumerate(self.models):
            if model.feature_importance is not None:
                df = model.feature_importance.copy()
                df['model'] = model.model_name
                df['weight'] = self.weights[i]
                df['weighted_importance'] = df['importance'] * df['weight']
                importance_dfs.append(df)
        
        if importance_dfs:
            combined_df = pd.concat(importance_dfs, ignore_index=True)
            # 按特征聚合
            ensemble_importance = combined_df.groupby('feature')['weighted_importance'].sum().reset_index()
            ensemble_importance = ensemble_importance.sort_values('weighted_importance', ascending=False)
            return ensemble_importance
        else:
            return pd.DataFrame()

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self):
        self.models = {}
        self.cv_scores = {}
        
    def time_series_cv(self, X: pd.DataFrame, y: pd.Series, 
                      model: BaseModel, n_splits: int = 5) -> Dict:
        """时序交叉验证"""
        logger.info(f"开始时序交叉验证，折数: {n_splits}")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {'rmse': [], 'mae': []}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"训练第 {fold+1}/{n_splits} 折")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # 训练模型
            model_copy = self._copy_model(model)
            model_copy.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            
            # 预测和评估
            y_pred = model_copy.predict(X_val_fold)
            
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            mae = mean_absolute_error(y_val_fold, y_pred)
            
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            
            logger.info(f"第 {fold+1} 折 - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # 计算平均分数
        avg_rmse = np.mean(cv_scores['rmse'])
        std_rmse = np.std(cv_scores['rmse'])
        avg_mae = np.mean(cv_scores['mae'])
        std_mae = np.std(cv_scores['mae'])
        
        logger.info(f"交叉验证结果 - RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}, MAE: {avg_mae:.4f} ± {std_mae:.4f}")
        
        return {
            'rmse_mean': avg_rmse,
            'rmse_std': std_rmse,
            'mae_mean': avg_mae,
            'mae_std': std_mae,
            'rmse_scores': cv_scores['rmse'],
            'mae_scores': cv_scores['mae']
        }
    
    def _copy_model(self, model: BaseModel) -> BaseModel:
        """复制模型"""
        if isinstance(model, LightGBMModel):
            return LightGBMModel(model.params.copy())
        elif isinstance(model, XGBoostModel):
            return XGBoostModel(model.params.copy())
        else:
            raise ValueError(f"不支持的模型类型: {type(model)}")
    
    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          model_type: str = 'lightgbm') -> BaseModel:
        """训练单个模型"""
        logger.info(f"训练 {model_type} 模型...")
        
        if model_type == 'lightgbm':
            model = LightGBMModel()
        elif model_type == 'xgboost':
            model = XGBoostModel()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model.fit(X_train, y_train, X_val, y_val)
        
        # 评估模型
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        logger.info(f"{model_type} 验证集 - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        self.models[model_type] = model
        return model
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> EnsembleModel:
        """训练集成模型"""
        logger.info("训练集成模型...")
        
        # 训练基础模型
        lgb_model = LightGBMModel()
        xgb_model = XGBoostModel()
        
        models = [lgb_model, xgb_model]
        ensemble = EnsembleModel(models)
        ensemble.fit(X_train, y_train, X_val, y_val)
        
        # 评估集成模型
        y_pred = ensemble.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        logger.info(f"集成模型验证集 - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        self.models['ensemble'] = ensemble
        return ensemble
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                model_type: str = 'lightgbm', n_trials: int = 100) -> Dict:
        """超参数优化"""
        logger.info(f"开始 {model_type} 超参数优化，试验次数: {n_trials}")
        
        def objective(trial):
            if model_type == 'lightgbm':
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'verbose': -1,
                    'random_state': 42
                }
                model = LightGBMModel(params)
            elif model_type == 'xgboost':
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'random_state': 42
                }
                model = XGBoostModel(params)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 时序交叉验证
            cv_results = self.time_series_cv(X, y, model, n_splits=3)
            return cv_results['rmse_mean']
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"最佳参数: {study.best_params}")
        logger.info(f"最佳分数: {study.best_value:.4f}")
        
        return study.best_params

def main():
    """测试模型训练功能"""
    # 这里应该加载实际的特征数据
    # 为了演示，创建一些虚拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples) * 10 + 50)  # 模拟含水率
    
    # 分割数据
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 训练模型
    trainer = ModelTrainer()
    
    # 训练LightGBM
    lgb_model = trainer.train_single_model(X_train, y_train, X_val, y_val, 'lightgbm')
    
    # 训练XGBoost
    xgb_model = trainer.train_single_model(X_train, y_train, X_val, y_val, 'xgboost')
    
    # 训练集成模型
    ensemble_model = trainer.train_ensemble(X_train, y_train, X_val, y_val)
    
    print("模型训练测试完成")

if __name__ == "__main__":
    main()