"""
中国石化岩性识别AI竞赛 - 基础模型模块
包含LightGBM、CatBoost、XGBoost等基础模型
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from config import (
    LIGHTGBM_CONFIG, CATBOOST_CONFIG, XGBOOST_CONFIG,
    TARGET_COLUMN, FEATURE_COLUMNS, CLASS_WEIGHTS
)
from utils import save_model, load_model, evaluate_model

class BaseModelTrainer:
    """基础模型训练器"""
    
    def __init__(self, model_type='lightgbm', model_params=None, use_class_weights=True):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.use_class_weights = use_class_weights
        self.model = None
        self.feature_importance = None
        self.training_history = {}
        
    def _get_default_params(self):
        """获取默认参数"""
        if self.model_type == 'lightgbm':
            return LIGHTGBM_CONFIG.copy()
        elif self.model_type == 'catboost':
            return CATBOOST_CONFIG.copy()
        elif self.model_type == 'xgboost':
            return XGBOOST_CONFIG.copy()
        else:
            return {}
    
    def _prepare_class_weights(self, y):
        """准备类别权重"""
        if not self.use_class_weights:
            return None
            
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        class_weights = compute_class_weight(
            'balanced', classes=classes, y=y
        )
        
        return dict(zip(classes, class_weights))
    
    def create_model(self, X_train, y_train):
        """创建模型"""
        # 合并默认参数和用户参数
        params = self._get_default_params()
        params.update(self.model_params)
        
        # 处理类别权重
        class_weights = self._prepare_class_weights(y_train)
        
        if self.model_type == 'lightgbm':
            if class_weights and 'class_weight' not in params:
                params['class_weight'] = class_weights
            
            self.model = lgb.LGBMClassifier(**params)
            
        elif self.model_type == 'catboost':
            if class_weights and 'class_weights' not in params:
                # CatBoost使用不同的格式
                params['class_weights'] = list(class_weights.values())
            
            self.model = cb.CatBoostClassifier(**params)
            
        elif self.model_type == 'xgboost':
            if class_weights and 'scale_pos_weight' not in params:
                # XGBoost处理二分类权重
                if len(class_weights) == 2:
                    params['scale_pos_weight'] = class_weights[1] / class_weights[0]
            
            self.model = xgb.XGBClassifier(**params)
            
        elif self.model_type == 'random_forest':
            if class_weights and 'class_weight' not in params:
                params['class_weight'] = class_weights
            
            self.model = RandomForestClassifier(**params)
            
        elif self.model_type == 'logistic_regression':
            if class_weights and 'class_weight' not in params:
                params['class_weight'] = class_weights
            
            self.model = LogisticRegression(**params)
            
        elif self.model_type == 'svm':
            if class_weights and 'class_weight' not in params:
                params['class_weight'] = class_weights
            
            self.model = SVC(**params)
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              early_stopping_rounds=None, verbose=True):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            early_stopping_rounds: 早停轮数
            verbose: 是否显示训练过程
            
        Returns:
            训练好的模型
        """
        print(f"开始训练{self.model_type}模型...")
        print(f"训练集大小: {X_train.shape}")
        if X_val is not None:
            print(f"验证集大小: {X_val.shape}")
        
        # 创建模型
        if self.model is None:
            self.create_model(X_train, y_train)
        
        # 准备训练参数
        fit_params = {}
        
        if self.model_type in ['lightgbm', 'catboost', 'xgboost']:
            if X_val is not None and y_val is not None:
                if self.model_type == 'lightgbm':
                    fit_params['eval_set'] = [(X_val, y_val)]
                    fit_params['eval_metric'] = 'multi_logloss'
                    if early_stopping_rounds:
                        fit_params['early_stopping_rounds'] = early_stopping_rounds
                    if not verbose:
                        fit_params['callbacks'] = [lgb.log_evaluation(0)]
                        
                elif self.model_type == 'catboost':
                    fit_params['eval_set'] = [(X_val, y_val)]
                    if early_stopping_rounds:
                        fit_params['early_stopping_rounds'] = early_stopping_rounds
                    if not verbose:
                        fit_params['verbose'] = False
                        
                elif self.model_type == 'xgboost':
                    fit_params['eval_set'] = [(X_val, y_val)]
                    if early_stopping_rounds:
                        fit_params['early_stopping_rounds'] = early_stopping_rounds
                    if not verbose:
                        fit_params['verbose'] = False
        
        # 训练模型
        self.model.fit(X_train, y_train, **fit_params)
        
        # 保存特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # 保存训练历史
        if hasattr(self.model, 'evals_result_'):
            self.training_history = self.model.evals_result_
        
        print(f"{self.model_type}模型训练完成!")
        
        return self.model
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError(f"{self.model_type}模型不支持概率预测")
    
    def get_feature_importance(self, top_n=20):
        """获取特征重要性"""
        if self.feature_importance is None:
            return None
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'model_params': self.model_params
        }
        
        save_model(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        model_data = load_model(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data.get('feature_importance')
        self.training_history = model_data.get('training_history', {})
        self.model_params = model_data.get('model_params', {})
        
        print(f"模型已从{filepath}加载")
        
        return self.model

class ModelEnsemble:
    """模型集成器"""
    
    def __init__(self, models=None, weights=None, method='average'):
        self.models = models or []
        self.weights = weights
        self.method = method
        self.model_predictions = {}
        
    def add_model(self, model, name=None, weight=1.0):
        """添加模型"""
        if name is None:
            name = f"model_{len(self.models)}"
        
        self.models.append({
            'model': model,
            'name': name,
            'weight': weight
        })
    
    def predict(self, X, return_individual=False):
        """集成预测"""
        if not self.models:
            raise ValueError("没有可用的模型")
        
        predictions = []
        individual_preds = {}
        
        for model_info in self.models:
            model = model_info['model']
            name = model_info['name']
            weight = model_info['weight']
            
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            else:
                pred = model.model.predict(X)
            
            predictions.append(pred * weight)
            individual_preds[name] = pred
        
        # 集成预测
        if self.method == 'average':
            ensemble_pred = np.mean(predictions, axis=0)
        elif self.method == 'weighted_average':
            total_weight = sum(info['weight'] for info in self.models)
            ensemble_pred = np.sum(predictions, axis=0) / total_weight
        elif self.method == 'majority_vote':
            ensemble_pred = np.round(np.mean(predictions, axis=0))
        else:
            raise ValueError(f"不支持的集成方法: {self.method}")
        
        if return_individual:
            return ensemble_pred, individual_preds
        
        return ensemble_pred
    
    def predict_proba(self, X, return_individual=False):
        """集成概率预测"""
        if not self.models:
            raise ValueError("没有可用的模型")
        
        probabilities = []
        individual_probas = {}
        
        for model_info in self.models:
            model = model_info['model']
            name = model_info['name']
            weight = model_info['weight']
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            elif hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
                proba = model.model.predict_proba(X)
            else:
                continue
            
            probabilities.append(proba * weight)
            individual_probas[name] = proba
        
        if not probabilities:
            raise ValueError("没有支持概率预测的模型")
        
        # 集成概率
        if self.method in ['average', 'weighted_average']:
            if self.method == 'weighted_average':
                total_weight = sum(info['weight'] for info in self.models 
                                 if hasattr(info['model'], 'predict_proba') or 
                                    (hasattr(info['model'], 'model') and 
                                     hasattr(info['model'].model, 'predict_proba')))
                ensemble_proba = np.sum(probabilities, axis=0) / total_weight
            else:
                ensemble_proba = np.mean(probabilities, axis=0)
        else:
            ensemble_proba = np.mean(probabilities, axis=0)
        
        if return_individual:
            return ensemble_proba, individual_probas
        
        return ensemble_proba

def train_base_models(X_train, y_train, X_val=None, y_val=None, 
                     model_types=['lightgbm', 'catboost', 'xgboost'],
                     custom_params=None, use_early_stopping=True):
    """
    训练多个基础模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        model_types: 模型类型列表
        custom_params: 自定义参数字典
        use_early_stopping: 是否使用早停
        
    Returns:
        训练好的模型字典
    """
    print("开始训练基础模型...")
    
    if custom_params is None:
        custom_params = {}
    
    trained_models = {}
    
    for model_type in model_types:
        print(f"\n训练{model_type}模型...")
        
        # 获取模型参数
        params = custom_params.get(model_type, {})
        
        # 创建训练器
        trainer = BaseModelTrainer(
            model_type=model_type,
            model_params=params,
            use_class_weights=True
        )
        
        # 训练模型
        early_stopping_rounds = 100 if use_early_stopping else None
        trainer.train(
            X_train, y_train, X_val, y_val,
            early_stopping_rounds=early_stopping_rounds
        )
        
        # 评估模型
        if X_val is not None and y_val is not None:
            y_pred = trainer.predict(X_val)
            y_pred_proba = trainer.predict_proba(X_val) if hasattr(trainer.model, 'predict_proba') else None
            
            eval_result = evaluate_model(y_val, y_pred, y_pred_proba, f"{model_type}")
            trainer.eval_result = eval_result
        
        trained_models[model_type] = trainer
    
    print("\n所有基础模型训练完成!")
    
    return trained_models

def optimize_model_hyperparameters(X_train, y_train, groups, model_type='lightgbm',
                                 param_space=None, n_trials=100, cv_folds=5):
    """
    使用Optuna优化模型超参数
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        groups: 分组信息
        model_type: 模型类型
        param_space: 参数空间
        n_trials: 优化试验次数
        cv_folds: 交叉验证折数
        
    Returns:
        最优参数
    """
    try:
        import optuna
        from sklearn.model_selection import GroupKFold
    except ImportError:
        print("需要安装optuna库进行超参数优化")
        return None
    
    print(f"开始{model_type}模型超参数优化...")
    
    def objective(trial):
        # 定义参数空间
        if model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            }
        elif model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            }
        elif model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            }
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 如果提供了自定义参数空间，更新参数
        if param_space:
            for key, value in param_space.items():
                if isinstance(value, dict):
                    if value['type'] == 'int':
                        params[key] = trial.suggest_int(key, value['low'], value['high'])
                    elif value['type'] == 'float':
                        params[key] = trial.suggest_float(key, value['low'], value['high'])
                    elif value['type'] == 'categorical':
                        params[key] = trial.suggest_categorical(key, value['choices'])
        
        # 交叉验证
        cv = GroupKFold(n_splits=cv_folds)
        scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train, groups):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 训练模型
            trainer = BaseModelTrainer(model_type=model_type, model_params=params)
            trainer.train(X_tr, y_tr, X_val, y_val, early_stopping_rounds=50, verbose=False)
            
            # 预测和评估
            y_pred = trainer.predict(X_val)
            score = f1_score(y_val, y_pred, average='macro')
            scores.append(score)
        
        return np.mean(scores)
    
    # 创建优化器
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"最优参数: {study.best_params}")
    print(f"最优分数: {study.best_value:.4f}")
    
    return study.best_params

if __name__ == "__main__":
    # 测试代码
    from utils import load_data
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    print("测试基础模型模块...")
    
    # 加载和预处理数据
    train_data = load_data('train')
    
    # 数据预处理和特征工程
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(train_data, is_training=True)
    
    feature_engineer = FeatureEngineer()
    featured_data = feature_engineer.feature_engineering_pipeline(processed_data, is_training=True)
    
    # 准备训练数据
    feature_cols = [col for col in featured_data.columns 
                   if col not in [TARGET_COLUMN, 'WELL', 'DEPTH', 'id']]
    
    X = featured_data[feature_cols]
    y = featured_data[TARGET_COLUMN]
    
    # 简单分割数据
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    
    # 训练单个模型
    trainer = BaseModelTrainer(model_type='lightgbm')
    trainer.train(X_train, y_train, X_val, y_val)
    
    # 预测和评估
    y_pred = trainer.predict(X_val)
    y_pred_proba = trainer.predict_proba(X_val)
    
    eval_result = evaluate_model(y_val, y_pred, y_pred_proba, "LightGBM Test")
    
    # 显示特征重要性
    feature_importance = trainer.get_feature_importance(top_n=10)
    print("\n特征重要性 (Top 10):")
    print(feature_importance)
    
    print("基础模型模块测试完成!")