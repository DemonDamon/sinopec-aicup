"""
中国石化岩性识别AI竞赛 - 集成学习模块
包含Stacking、Voting、Blending等集成策略
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import warnings
warnings.filterwarnings('ignore')

from config import (
    ENSEMBLE_CONFIG, TARGET_COLUMN, GROUP_COLUMN,
    RANDOM_STATE, EVALUATION_METRICS
)
from utils import save_model, load_model, evaluate_model
from validation import CrossValidator

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Stacking集成学习器"""
    
    def __init__(self, base_models, meta_learner=None, cv=5, use_probas=True, 
                 use_original_features=False, random_state=42):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.cv = cv
        self.use_probas = use_probas
        self.use_original_features = use_original_features
        self.random_state = random_state
        self.trained_base_models = []
        self.meta_features_names = []
        
        # 默认元学习器
        if self.meta_learner is None:
            self.meta_learner = LogisticRegression(
                random_state=random_state,
                max_iter=1000
            )
    
    def _get_meta_features(self, X, y=None, groups=None, is_training=True):
        """生成元特征"""
        meta_features = []
        
        if is_training:
            print("生成训练集元特征...")
            
            # 使用交叉验证生成元特征
            if groups is not None:
                # 使用GroupKFold
                from sklearn.model_selection import GroupKFold
                cv_splitter = GroupKFold(n_splits=self.cv)
                cv_splits = list(cv_splitter.split(X, y, groups))
            else:
                # 使用StratifiedKFold
                cv_splitter = StratifiedKFold(
                    n_splits=self.cv, 
                    shuffle=True, 
                    random_state=self.random_state
                )
                cv_splits = list(cv_splitter.split(X, y))
            
            # 为每个基模型生成元特征
            for i, (model_name, model) in enumerate(self.base_models.items()):
                print(f"处理基模型 {i+1}/{len(self.base_models)}: {model_name}")
                
                if self.use_probas and hasattr(model, 'predict_proba'):
                    # 使用概率作为元特征
                    meta_pred = np.zeros((len(X), len(np.unique(y))))
                    
                    for train_idx, val_idx in cv_splits:
                        X_train_fold = X.iloc[train_idx]
                        y_train_fold = y.iloc[train_idx]
                        X_val_fold = X.iloc[val_idx]
                        
                        # 训练模型
                        model_clone = clone(model)
                        if hasattr(model_clone, 'fit'):
                            model_clone.fit(X_train_fold, y_train_fold)
                        else:
                            # 处理自定义训练器
                            model_clone.train(X_train_fold, y_train_fold)
                        
                        # 预测概率
                        if hasattr(model_clone, 'predict_proba'):
                            fold_pred = model_clone.predict_proba(X_val_fold)
                        else:
                            fold_pred = model_clone.model.predict_proba(X_val_fold)
                        
                        meta_pred[val_idx] = fold_pred
                    
                    meta_features.append(meta_pred)
                    
                    # 记录特征名称
                    for class_idx in range(meta_pred.shape[1]):
                        self.meta_features_names.append(f"{model_name}_proba_class_{class_idx}")
                
                else:
                    # 使用预测结果作为元特征
                    meta_pred = np.zeros(len(X))
                    
                    for train_idx, val_idx in cv_splits:
                        X_train_fold = X.iloc[train_idx]
                        y_train_fold = y.iloc[train_idx]
                        X_val_fold = X.iloc[val_idx]
                        
                        # 训练模型
                        model_clone = clone(model)
                        if hasattr(model_clone, 'fit'):
                            model_clone.fit(X_train_fold, y_train_fold)
                        else:
                            model_clone.train(X_train_fold, y_train_fold)
                        
                        # 预测
                        if hasattr(model_clone, 'predict'):
                            fold_pred = model_clone.predict(X_val_fold)
                        else:
                            fold_pred = model_clone.model.predict(X_val_fold)
                        
                        meta_pred[val_idx] = fold_pred
                    
                    meta_features.append(meta_pred.reshape(-1, 1))
                    self.meta_features_names.append(f"{model_name}_pred")
            
            # 训练所有基模型用于预测阶段
            print("训练最终基模型...")
            self.trained_base_models = []
            for model_name, model in self.base_models.items():
                model_clone = clone(model)
                if hasattr(model_clone, 'fit'):
                    model_clone.fit(X, y)
                else:
                    model_clone.train(X, y)
                self.trained_base_models.append((model_name, model_clone))
        
        else:
            print("生成测试集元特征...")
            
            # 使用训练好的基模型生成元特征
            for i, (model_name, model) in enumerate(self.trained_base_models):
                if self.use_probas and hasattr(model, 'predict_proba'):
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X)
                    else:
                        pred_proba = model.model.predict_proba(X)
                    meta_features.append(pred_proba)
                else:
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                    else:
                        pred = model.model.predict(X)
                    meta_features.append(pred.reshape(-1, 1))
        
        # 合并元特征
        meta_X = np.concatenate(meta_features, axis=1)
        
        # 是否包含原始特征
        if self.use_original_features:
            meta_X = np.concatenate([X.values, meta_X], axis=1)
        
        print(f"元特征维度: {meta_X.shape}")
        
        return meta_X
    
    def fit(self, X, y, groups=None):
        """训练Stacking模型"""
        print("开始训练Stacking集成模型...")
        print(f"基模型数量: {len(self.base_models)}")
        print(f"基模型: {list(self.base_models.keys())}")
        
        # 生成元特征
        meta_X = self._get_meta_features(X, y, groups, is_training=True)
        
        # 训练元学习器
        print("训练元学习器...")
        self.meta_learner.fit(meta_X, y)
        
        print("Stacking模型训练完成!")
        
        return self
    
    def predict(self, X):
        """预测"""
        meta_X = self._get_meta_features(X, is_training=False)
        return self.meta_learner.predict(meta_X)
    
    def predict_proba(self, X):
        """预测概率"""
        meta_X = self._get_meta_features(X, is_training=False)
        return self.meta_learner.predict_proba(meta_X)

class VotingEnsemble:
    """投票集成学习器"""
    
    def __init__(self, models, voting='soft', weights=None):
        self.models = models
        self.voting = voting  # 'hard' or 'soft'
        self.weights = weights
        self.trained_models = []
    
    def fit(self, X, y):
        """训练投票模型"""
        print("开始训练Voting集成模型...")
        print(f"投票方式: {self.voting}")
        print(f"模型数量: {len(self.models)}")
        
        self.trained_models = []
        
        for model_name, model in self.models.items():
            print(f"训练模型: {model_name}")
            
            model_clone = clone(model)
            if hasattr(model_clone, 'fit'):
                model_clone.fit(X, y)
            else:
                model_clone.train(X, y)
            
            self.trained_models.append((model_name, model_clone))
        
        print("Voting模型训练完成!")
        
        return self
    
    def predict(self, X):
        """预测"""
        if self.voting == 'hard':
            # 硬投票
            predictions = []
            
            for model_name, model in self.trained_models:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                else:
                    pred = model.model.predict(X)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # 加权投票
            if self.weights is not None:
                weighted_predictions = []
                for i, pred in enumerate(predictions):
                    weighted_predictions.append(pred * self.weights[i])
                predictions = np.array(weighted_predictions)
            
            # 多数投票
            from scipy import stats
            final_predictions = stats.mode(predictions, axis=0)[0].flatten()
            
            return final_predictions
        
        else:
            # 软投票
            probabilities = []
            
            for model_name, model in self.trained_models:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                elif hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
                    proba = model.model.predict_proba(X)
                else:
                    # 如果模型不支持概率预测，转换为one-hot
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                    else:
                        pred = model.model.predict(X)
                    
                    from sklearn.preprocessing import LabelBinarizer
                    lb = LabelBinarizer()
                    proba = lb.fit_transform(pred)
                
                probabilities.append(proba)
            
            probabilities = np.array(probabilities)
            
            # 加权平均
            if self.weights is not None:
                weighted_probabilities = []
                for i, proba in enumerate(probabilities):
                    weighted_probabilities.append(proba * self.weights[i])
                avg_probabilities = np.mean(weighted_probabilities, axis=0)
            else:
                avg_probabilities = np.mean(probabilities, axis=0)
            
            # 返回概率最大的类别
            return np.argmax(avg_probabilities, axis=1)
    
    def predict_proba(self, X):
        """预测概率"""
        if self.voting != 'soft':
            raise ValueError("只有软投票支持概率预测")
        
        probabilities = []
        
        for model_name, model in self.trained_models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            elif hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
                proba = model.model.predict_proba(X)
            else:
                continue
            
            probabilities.append(proba)
        
        probabilities = np.array(probabilities)
        
        # 加权平均
        if self.weights is not None:
            weighted_probabilities = []
            for i, proba in enumerate(probabilities):
                weighted_probabilities.append(proba * self.weights[i])
            avg_probabilities = np.mean(weighted_probabilities, axis=0)
        else:
            avg_probabilities = np.mean(probabilities, axis=0)
        
        return avg_probabilities

class BlendingEnsemble:
    """Blending集成学习器"""
    
    def __init__(self, base_models, meta_learner=None, holdout_ratio=0.2, 
                 use_probas=True, random_state=42):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.holdout_ratio = holdout_ratio
        self.use_probas = use_probas
        self.random_state = random_state
        self.trained_base_models = []
        
        # 默认元学习器
        if self.meta_learner is None:
            self.meta_learner = LogisticRegression(
                random_state=random_state,
                max_iter=1000
            )
    
    def fit(self, X, y, groups=None):
        """训练Blending模型"""
        print("开始训练Blending集成模型...")
        
        # 分割数据
        from sklearn.model_selection import train_test_split
        
        if groups is not None:
            # 按组分割
            unique_groups = groups.unique()
            train_groups, holdout_groups = train_test_split(
                unique_groups, 
                test_size=self.holdout_ratio,
                random_state=self.random_state
            )
            
            train_mask = groups.isin(train_groups)
            holdout_mask = groups.isin(holdout_groups)
            
            X_train, X_holdout = X[train_mask], X[holdout_mask]
            y_train, y_holdout = y[train_mask], y[holdout_mask]
        else:
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X, y, 
                test_size=self.holdout_ratio,
                random_state=self.random_state,
                stratify=y
            )
        
        print(f"训练集大小: {len(X_train)}")
        print(f"留出集大小: {len(X_holdout)}")
        
        # 训练基模型
        print("训练基模型...")
        self.trained_base_models = []
        meta_features = []
        
        for model_name, model in self.base_models.items():
            print(f"训练模型: {model_name}")
            
            model_clone = clone(model)
            if hasattr(model_clone, 'fit'):
                model_clone.fit(X_train, y_train)
            else:
                model_clone.train(X_train, y_train)
            
            self.trained_base_models.append((model_name, model_clone))
            
            # 在留出集上预测
            if self.use_probas and hasattr(model_clone, 'predict_proba'):
                if hasattr(model_clone, 'predict_proba'):
                    pred = model_clone.predict_proba(X_holdout)
                else:
                    pred = model_clone.model.predict_proba(X_holdout)
            else:
                if hasattr(model_clone, 'predict'):
                    pred = model_clone.predict(X_holdout)
                else:
                    pred = model_clone.model.predict(X_holdout)
                pred = pred.reshape(-1, 1)
            
            meta_features.append(pred)
        
        # 合并元特征
        meta_X = np.concatenate(meta_features, axis=1)
        
        # 训练元学习器
        print("训练元学习器...")
        self.meta_learner.fit(meta_X, y_holdout)
        
        print("Blending模型训练完成!")
        
        return self
    
    def predict(self, X):
        """预测"""
        meta_features = []
        
        for model_name, model in self.trained_base_models:
            if self.use_probas and hasattr(model, 'predict_proba'):
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                else:
                    pred = model.model.predict_proba(X)
            else:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                else:
                    pred = model.model.predict(X)
                pred = pred.reshape(-1, 1)
            
            meta_features.append(pred)
        
        meta_X = np.concatenate(meta_features, axis=1)
        return self.meta_learner.predict(meta_X)
    
    def predict_proba(self, X):
        """预测概率"""
        meta_features = []
        
        for model_name, model in self.trained_base_models:
            if self.use_probas and hasattr(model, 'predict_proba'):
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                else:
                    pred = model.model.predict_proba(X)
            else:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                else:
                    pred = model.model.predict(X)
                pred = pred.reshape(-1, 1)
            
            meta_features.append(pred)
        
        meta_X = np.concatenate(meta_features, axis=1)
        return self.meta_learner.predict_proba(meta_X)

class EnsembleManager:
    """集成学习管理器"""
    
    def __init__(self, ensemble_config=None):
        self.ensemble_config = ensemble_config or ENSEMBLE_CONFIG
        self.ensembles = {}
        self.ensemble_results = {}
    
    def create_stacking_ensemble(self, base_models, meta_learner=None, **kwargs):
        """创建Stacking集成"""
        config = {**self.ensemble_config['stacking'], **kwargs}
        
        ensemble = StackingEnsemble(
            base_models=base_models,
            meta_learner=meta_learner,
            **config
        )
        
        self.ensembles['stacking'] = ensemble
        return ensemble
    
    def create_voting_ensemble(self, base_models, **kwargs):
        """创建Voting集成"""
        config = {**self.ensemble_config['voting'], **kwargs}
        
        ensemble = VotingEnsemble(
            models=base_models,
            **config
        )
        
        self.ensembles['voting'] = ensemble
        return ensemble
    
    def create_blending_ensemble(self, base_models, meta_learner=None, **kwargs):
        """创建Blending集成"""
        config = {**self.ensemble_config['blending'], **kwargs}
        
        ensemble = BlendingEnsemble(
            base_models=base_models,
            meta_learner=meta_learner,
            **config
        )
        
        self.ensembles['blending'] = ensemble
        return ensemble
    
    def train_all_ensembles(self, X, y, groups=None):
        """训练所有集成模型"""
        print("开始训练所有集成模型...")
        
        for ensemble_name, ensemble in self.ensembles.items():
            print(f"\n训练{ensemble_name}集成...")
            ensemble.fit(X, y, groups)
        
        print("所有集成模型训练完成!")
    
    def evaluate_ensembles(self, X_val, y_val):
        """评估所有集成模型"""
        print("评估集成模型...")
        
        for ensemble_name, ensemble in self.ensembles.items():
            print(f"\n评估{ensemble_name}集成...")
            
            y_pred = ensemble.predict(X_val)
            y_pred_proba = ensemble.predict_proba(X_val) if hasattr(ensemble, 'predict_proba') else None
            
            eval_result = evaluate_model(y_val, y_pred, y_pred_proba, f"{ensemble_name} Ensemble")
            self.ensemble_results[ensemble_name] = eval_result
        
        return self.ensemble_results
    
    def get_best_ensemble(self, metric='f1_macro'):
        """获取最佳集成模型"""
        if not self.ensemble_results:
            return None
        
        best_ensemble_name = max(
            self.ensemble_results.keys(),
            key=lambda x: self.ensemble_results[x].get(metric, 0)
        )
        
        return best_ensemble_name, self.ensembles[best_ensemble_name]
    
    def save_ensembles(self, filepath_prefix):
        """保存所有集成模型"""
        for ensemble_name, ensemble in self.ensembles.items():
            filepath = f"{filepath_prefix}_{ensemble_name}.pkl"
            save_model(ensemble, filepath)
            print(f"{ensemble_name}集成模型已保存到: {filepath}")
    
    def load_ensembles(self, filepath_prefix, ensemble_names):
        """加载集成模型"""
        for ensemble_name in ensemble_names:
            filepath = f"{filepath_prefix}_{ensemble_name}.pkl"
            ensemble = load_model(filepath)
            self.ensembles[ensemble_name] = ensemble
            print(f"{ensemble_name}集成模型已从{filepath}加载")

def create_ensemble_pipeline(base_models, X_train, y_train, groups_train=None,
                           X_val=None, y_val=None, ensemble_types=['stacking']):
    """
    创建完整的集成学习流水线
    
    Args:
        base_models: 基模型字典
        X_train: 训练特征
        y_train: 训练标签
        groups_train: 训练分组
        X_val: 验证特征
        y_val: 验证标签
        ensemble_types: 集成类型列表
        
    Returns:
        训练好的集成模型管理器
    """
    print("创建集成学习流水线...")
    
    manager = EnsembleManager()
    
    # 创建不同类型的集成
    for ensemble_type in ensemble_types:
        if ensemble_type == 'stacking':
            manager.create_stacking_ensemble(base_models)
        elif ensemble_type == 'voting':
            manager.create_voting_ensemble(base_models)
        elif ensemble_type == 'blending':
            manager.create_blending_ensemble(base_models)
    
    # 训练集成模型
    manager.train_all_ensembles(X_train, y_train, groups_train)
    
    # 评估集成模型
    if X_val is not None and y_val is not None:
        manager.evaluate_ensembles(X_val, y_val)
        
        # 显示最佳集成
        best_name, best_ensemble = manager.get_best_ensemble()
        print(f"\n最佳集成模型: {best_name}")
    
    return manager

if __name__ == "__main__":
    # 测试代码
    from utils import load_data
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    from base_models import BaseModelTrainer
    
    print("测试集成学习模块...")
    
    # 加载和预处理数据
    train_data = load_data('train')
    
    # 数据预处理和特征工程
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(train_data, is_training=True)
    
    feature_engineer = FeatureEngineer()
    featured_data = feature_engineer.feature_engineering_pipeline(processed_data, is_training=True)
    
    # 准备训练数据
    feature_cols = [col for col in featured_data.columns 
                   if col not in [TARGET_COLUMN, GROUP_COLUMN, 'DEPTH', 'id']]
    
    X = featured_data[feature_cols]
    y = featured_data[TARGET_COLUMN]
    groups = featured_data[GROUP_COLUMN]
    
    # 简单分割数据
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建基模型
    base_models = {}
    
    # LightGBM
    lgb_trainer = BaseModelTrainer(model_type='lightgbm')
    lgb_trainer.train(X_train, y_train, X_val, y_val, verbose=False)
    base_models['lightgbm'] = lgb_trainer.model
    
    # 创建简单的集成
    voting_ensemble = VotingEnsemble(base_models, voting='soft')
    voting_ensemble.fit(X_train, y_train)
    
    # 预测和评估
    y_pred = voting_ensemble.predict(X_val)
    eval_result = evaluate_model(y_val, y_pred, None, "Voting Ensemble Test")
    
    print("集成学习模块测试完成!")