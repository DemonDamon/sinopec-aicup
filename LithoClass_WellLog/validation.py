"""
中国石化岩性识别AI竞赛 - 交叉验证模块
包含GroupKFold按井分组验证等功能
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from config import CV_CONFIG, GROUP_COLUMN, TARGET_COLUMN, EVALUATION_METRICS
from utils import evaluate_model, save_experiment_results

class CrossValidator:
    """交叉验证器"""
    
    def __init__(self, cv_method='group_kfold', n_splits=5, shuffle=True, random_state=42):
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.cv_results = []
        
    def create_cv_splits(self, X, y, groups=None):
        """
        创建交叉验证分割
        
        Args:
            X: 特征数据
            y: 目标变量
            groups: 分组变量（用于GroupKFold）
            
        Returns:
            交叉验证分割迭代器
        """
        if self.cv_method == 'group_kfold':
            if groups is None:
                raise ValueError("GroupKFold需要提供groups参数")
            cv = GroupKFold(n_splits=self.n_splits)
            return cv.split(X, y, groups)
            
        elif self.cv_method == 'stratified_kfold':
            cv = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            return cv.split(X, y)
            
        elif self.cv_method == 'kfold':
            cv = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            return cv.split(X, y)
        
        else:
            raise ValueError(f"不支持的交叉验证方法: {self.cv_method}")
    
    def validate_model(self, model, X, y, groups=None, model_name="Model", 
                      fit_params=None, predict_proba=True):
        """
        交叉验证模型
        
        Args:
            model: 模型对象
            X: 特征数据
            y: 目标变量
            groups: 分组变量
            model_name: 模型名称
            fit_params: 模型训练参数
            predict_proba: 是否预测概率
            
        Returns:
            交叉验证结果
        """
        print(f"开始{model_name}的{self.cv_method}交叉验证...")
        print(f"分割数: {self.n_splits}")
        
        if fit_params is None:
            fit_params = {}
        
        cv_splits = self.create_cv_splits(X, y, groups)
        fold_results = []
        oof_predictions = np.zeros(len(X))
        oof_probabilities = np.zeros((len(X), len(np.unique(y))))
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n处理第 {fold + 1}/{self.n_splits} 折...")
            
            # 分割数据
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
            
            # 检查类别分布
            train_dist = y_train.value_counts().sort_index()
            val_dist = y_val.value_counts().sort_index()
            print(f"训练集类别分布: {train_dist.to_dict()}")
            print(f"验证集类别分布: {val_dist.to_dict()}")
            
            # 训练模型
            if hasattr(model, 'fit'):
                # 处理早停参数
                if 'eval_set' in fit_params:
                    fit_params['eval_set'] = [(X_val, y_val)]
                
                model.fit(X_train, y_train, **fit_params)
            else:
                raise ValueError("模型必须有fit方法")
            
            # 预测
            y_pred = model.predict(X_val)
            oof_predictions[val_idx] = y_pred
            
            if predict_proba and hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_val)
                oof_probabilities[val_idx] = y_pred_proba
            
            # 评估当前折
            fold_result = evaluate_model(
                y_val, y_pred, 
                y_pred_proba if predict_proba and hasattr(model, 'predict_proba') else None,
                f"{model_name} - Fold {fold + 1}"
            )
            fold_result['fold'] = fold + 1
            fold_results.append(fold_result)
        
        # 计算总体OOF结果
        print(f"\n{model_name} - 总体OOF结果:")
        print("=" * 50)
        
        oof_result = evaluate_model(
            y, oof_predictions,
            oof_probabilities if predict_proba and hasattr(model, 'predict_proba') else None,
            f"{model_name} - OOF"
        )
        
        # 计算各折平均值和标准差
        metrics_summary = {}
        for metric in EVALUATION_METRICS:
            if metric in fold_results[0]:
                values = [fold[metric] for fold in fold_results]
                metrics_summary[f'{metric}_mean'] = np.mean(values)
                metrics_summary[f'{metric}_std'] = np.std(values)
                print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        # 保存结果
        cv_result = {
            'model_name': model_name,
            'cv_method': self.cv_method,
            'n_splits': self.n_splits,
            'fold_results': fold_results,
            'oof_result': oof_result,
            'metrics_summary': metrics_summary,
            'oof_predictions': oof_predictions.tolist(),
            'oof_probabilities': oof_probabilities.tolist() if predict_proba else None
        }
        
        self.cv_results.append(cv_result)
        
        return cv_result
    
    def compare_models(self, models_results, metric='f1_macro'):
        """
        比较多个模型的交叉验证结果
        
        Args:
            models_results: 模型结果列表
            metric: 比较指标
            
        Returns:
            比较结果
        """
        comparison = []
        
        for result in models_results:
            model_name = result['model_name']
            mean_score = result['metrics_summary'].get(f'{metric}_mean', 0)
            std_score = result['metrics_summary'].get(f'{metric}_std', 0)
            
            comparison.append({
                'model': model_name,
                'mean_score': mean_score,
                'std_score': std_score,
                'score_range': f"{mean_score:.4f} ± {std_score:.4f}"
            })
        
        # 按平均分数排序
        comparison.sort(key=lambda x: x['mean_score'], reverse=True)
        
        print(f"\n模型比较结果 (按{metric}排序):")
        print("=" * 60)
        print(f"{'排名':<4} {'模型':<20} {metric:<15} {'标准差':<10}")
        print("-" * 60)
        
        for i, result in enumerate(comparison):
            print(f"{i+1:<4} {result['model']:<20} {result['score_range']:<25}")
        
        return comparison
    
    def get_best_model_result(self, metric='f1_macro'):
        """
        获取最佳模型结果
        
        Args:
            metric: 评估指标
            
        Returns:
            最佳模型结果
        """
        if not self.cv_results:
            return None
        
        best_result = max(
            self.cv_results,
            key=lambda x: x['metrics_summary'].get(f'{metric}_mean', 0)
        )
        
        return best_result
    
    def save_cv_results(self, filename='cv_results'):
        """
        保存交叉验证结果
        
        Args:
            filename: 文件名
        """
        results_summary = {
            'cv_config': {
                'method': self.cv_method,
                'n_splits': self.n_splits,
                'shuffle': self.shuffle,
                'random_state': self.random_state
            },
            'results': self.cv_results
        }
        
        save_experiment_results(results_summary, filename)
        
        return results_summary

def validate_data_split(data, groups_col=GROUP_COLUMN, target_col=TARGET_COLUMN):
    """
    验证数据分割的合理性
    
    Args:
        data: 数据
        groups_col: 分组列
        target_col: 目标列
        
    Returns:
        验证结果
    """
    print("验证数据分割合理性...")
    
    # 检查分组分布
    group_counts = data[groups_col].value_counts()
    print(f"总分组数: {len(group_counts)}")
    print(f"每组样本数 - 最小: {group_counts.min()}, 最大: {group_counts.max()}")
    print(f"每组样本数 - 平均: {group_counts.mean():.2f}, 中位数: {group_counts.median():.2f}")
    
    # 检查每组的类别分布
    group_class_dist = data.groupby(groups_col)[target_col].value_counts().unstack(fill_value=0)
    
    # 检查是否有组只包含单一类别
    single_class_groups = []
    for group in group_class_dist.index:
        non_zero_classes = (group_class_dist.loc[group] > 0).sum()
        if non_zero_classes == 1:
            single_class_groups.append(group)
    
    print(f"只包含单一类别的组数: {len(single_class_groups)}")
    if single_class_groups:
        print(f"单一类别组: {single_class_groups[:10]}...")  # 只显示前10个
    
    # 检查类别分布
    overall_class_dist = data[target_col].value_counts().sort_index()
    print(f"总体类别分布: {overall_class_dist.to_dict()}")
    
    # 模拟GroupKFold分割
    cv = GroupKFold(n_splits=CV_CONFIG['n_splits'])
    X = data.drop(columns=[target_col])
    y = data[target_col]
    groups = data[groups_col]
    
    fold_stats = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])
        
        # 检查组是否重叠
        overlap = train_groups.intersection(val_groups)
        
        train_class_dist = y.iloc[train_idx].value_counts().sort_index()
        val_class_dist = y.iloc[val_idx].value_counts().sort_index()
        
        fold_stat = {
            'fold': fold + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_groups': len(train_groups),
            'val_groups': len(val_groups),
            'group_overlap': len(overlap),
            'train_class_dist': train_class_dist.to_dict(),
            'val_class_dist': val_class_dist.to_dict()
        }
        fold_stats.append(fold_stat)
        
        print(f"\n第{fold + 1}折:")
        print(f"  训练集: {len(train_idx)} 样本, {len(train_groups)} 组")
        print(f"  验证集: {len(val_idx)} 样本, {len(val_groups)} 组")
        print(f"  组重叠: {len(overlap)}")
        print(f"  训练集类别分布: {train_class_dist.to_dict()}")
        print(f"  验证集类别分布: {val_class_dist.to_dict()}")
    
    validation_result = {
        'total_groups': len(group_counts),
        'group_stats': {
            'min_samples': group_counts.min(),
            'max_samples': group_counts.max(),
            'mean_samples': group_counts.mean(),
            'median_samples': group_counts.median()
        },
        'single_class_groups': len(single_class_groups),
        'overall_class_distribution': overall_class_dist.to_dict(),
        'fold_stats': fold_stats
    }
    
    return validation_result

def create_holdout_validation(data, test_size=0.2, groups_col=GROUP_COLUMN, 
                            target_col=TARGET_COLUMN, random_state=42):
    """
    创建留出验证集
    
    Args:
        data: 数据
        test_size: 测试集比例
        groups_col: 分组列
        target_col: 目标列
        random_state: 随机种子
        
    Returns:
        训练集和验证集索引
    """
    from sklearn.model_selection import GroupShuffleSplit
    
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    groups = data[groups_col]
    
    train_idx, val_idx = next(gss.split(X, y, groups))
    
    print(f"留出验证集创建完成:")
    print(f"训练集大小: {len(train_idx)} ({len(train_idx)/len(data)*100:.1f}%)")
    print(f"验证集大小: {len(val_idx)} ({len(val_idx)/len(data)*100:.1f}%)")
    
    # 检查组分布
    train_groups = set(groups.iloc[train_idx])
    val_groups = set(groups.iloc[val_idx])
    overlap = train_groups.intersection(val_groups)
    
    print(f"训练集组数: {len(train_groups)}")
    print(f"验证集组数: {len(val_groups)}")
    print(f"组重叠数: {len(overlap)}")
    
    return train_idx, val_idx

if __name__ == "__main__":
    # 测试代码
    from utils import load_data
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    print("测试交叉验证模块...")
    
    # 加载和预处理数据
    train_data = load_data('train')
    
    # 验证数据分割
    validation_result = validate_data_split(train_data)
    
    # 数据预处理和特征工程
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(train_data, is_training=True)
    
    feature_engineer = FeatureEngineer()
    featured_data = feature_engineer.feature_engineering_pipeline(processed_data, is_training=True)
    
    # 准备交叉验证数据
    feature_cols = [col for col in featured_data.columns 
                   if col not in [TARGET_COLUMN, GROUP_COLUMN, DEPTH_COLUMN, 'id']]
    
    X = featured_data[feature_cols]
    y = featured_data[TARGET_COLUMN]
    groups = featured_data[GROUP_COLUMN]
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    
    # 创建交叉验证器
    cv = CrossValidator(
        cv_method=CV_CONFIG['method'],
        n_splits=CV_CONFIG['n_splits'],
        shuffle=CV_CONFIG['shuffle'],
        random_state=CV_CONFIG['random_state']
    )
    
    print("交叉验证模块测试完成!")