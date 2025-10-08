"""
中国石化岩性识别AI竞赛 - 数据预处理模块
包含异常值检测、缺失值处理、数据标准化等功能
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import (
    FEATURE_COLUMNS, TARGET_COLUMN, GROUP_COLUMN, DEPTH_COLUMN,
    OUTLIER_DETECTION, MISSING_VALUE_CONFIG, SCALING_CONFIG
)

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.outlier_detector = None
        self.imputer = None
        self.scaler = None
        self.feature_columns = FEATURE_COLUMNS.copy()
        
    def detect_outliers(self, data, method='isolation_forest', contamination=0.1):
        """
        异常值检测
        
        Args:
            data: 输入数据
            method: 检测方法 ('isolation_forest', 'zscore', 'iqr')
            contamination: 异常值比例
            
        Returns:
            outlier_mask: 异常值掩码 (True表示异常值)
        """
        if method == 'isolation_forest':
            if self.outlier_detector is None:
                self.outlier_detector = IsolationForest(
                    contamination=contamination,
                    random_state=OUTLIER_DETECTION['random_state']
                )
                outlier_labels = self.outlier_detector.fit_predict(data[self.feature_columns])
            else:
                outlier_labels = self.outlier_detector.predict(data[self.feature_columns])
            
            # -1表示异常值，1表示正常值
            outlier_mask = outlier_labels == -1
            
        elif method == 'zscore':
            # Z-score方法
            z_scores = np.abs(stats.zscore(data[self.feature_columns]))
            outlier_mask = (z_scores > 3).any(axis=1)
            
        elif method == 'iqr':
            # IQR方法
            outlier_mask = np.zeros(len(data), dtype=bool)
            for col in self.feature_columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask |= (data[col] < lower_bound) | (data[col] > upper_bound)
        
        return outlier_mask
    
    def handle_missing_values(self, data, method='mean'):
        """
        缺失值处理
        
        Args:
            data: 输入数据
            method: 填充方法 ('mean', 'median', 'mode', 'iterative', 'forward_fill', 'backward_fill')
            
        Returns:
            处理后的数据
        """
        data_processed = data.copy()
        
        if method == 'mean':
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy='mean')
                data_processed[self.feature_columns] = self.imputer.fit_transform(data_processed[self.feature_columns])
            else:
                data_processed[self.feature_columns] = self.imputer.transform(data_processed[self.feature_columns])
                
        elif method == 'median':
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy='median')
                data_processed[self.feature_columns] = self.imputer.fit_transform(data_processed[self.feature_columns])
            else:
                data_processed[self.feature_columns] = self.imputer.transform(data_processed[self.feature_columns])
                
        elif method == 'mode':
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy='most_frequent')
                data_processed[self.feature_columns] = self.imputer.fit_transform(data_processed[self.feature_columns])
            else:
                data_processed[self.feature_columns] = self.imputer.transform(data_processed[self.feature_columns])
                
        elif method == 'iterative':
            # 迭代填充（类似missForest）
            if self.imputer is None:
                self.imputer = IterativeImputer(
                    max_iter=MISSING_VALUE_CONFIG['max_iter'],
                    random_state=MISSING_VALUE_CONFIG['random_state']
                )
                data_processed[self.feature_columns] = self.imputer.fit_transform(data_processed[self.feature_columns])
            else:
                data_processed[self.feature_columns] = self.imputer.transform(data_processed[self.feature_columns])
                
        elif method == 'forward_fill':
            # 按井分组进行前向填充
            data_processed = data_processed.groupby(GROUP_COLUMN)[self.feature_columns].fillna(method='ffill')
            data_processed = data_processed.fillna(data_processed.mean())  # 处理剩余缺失值
            
        elif method == 'backward_fill':
            # 按井分组进行后向填充
            data_processed = data_processed.groupby(GROUP_COLUMN)[self.feature_columns].fillna(method='bfill')
            data_processed = data_processed.fillna(data_processed.mean())  # 处理剩余缺失值
            
        elif method == 'group_mean':
            # 按井分组填充均值
            group_means = data_processed.groupby(GROUP_COLUMN)[self.feature_columns].transform('mean')
            data_processed[self.feature_columns] = data_processed[self.feature_columns].fillna(group_means)
            # 处理整个井都缺失的情况
            data_processed[self.feature_columns] = data_processed[self.feature_columns].fillna(
                data_processed[self.feature_columns].mean()
            )
        
        return data_processed
    
    def scale_features(self, data, method='standard'):
        """
        特征标准化
        
        Args:
            data: 输入数据
            method: 标准化方法 ('standard', 'minmax', 'robust')
            
        Returns:
            标准化后的数据
        """
        data_processed = data.copy()
        
        if method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                data_processed[self.feature_columns] = self.scaler.fit_transform(data_processed[self.feature_columns])
            else:
                data_processed[self.feature_columns] = self.scaler.transform(data_processed[self.feature_columns])
                
        elif method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=SCALING_CONFIG['feature_range'])
                data_processed[self.feature_columns] = self.scaler.fit_transform(data_processed[self.feature_columns])
            else:
                data_processed[self.feature_columns] = self.scaler.transform(data_processed[self.feature_columns])
                
        elif method == 'robust':
            if self.scaler is None:
                self.scaler = RobustScaler()
                data_processed[self.feature_columns] = self.scaler.fit_transform(data_processed[self.feature_columns])
            else:
                data_processed[self.feature_columns] = self.scaler.transform(data_processed[self.feature_columns])
        
        return data_processed
    
    def remove_outliers(self, data, outlier_mask):
        """
        移除异常值
        
        Args:
            data: 输入数据
            outlier_mask: 异常值掩码
            
        Returns:
            移除异常值后的数据
        """
        return data[~outlier_mask].reset_index(drop=True)
    
    def clip_outliers(self, data, method='iqr', factor=1.5):
        """
        截断异常值而不是移除
        
        Args:
            data: 输入数据
            method: 截断方法 ('iqr', 'zscore')
            factor: 截断因子
            
        Returns:
            截断后的数据
        """
        data_processed = data.copy()
        
        for col in self.feature_columns:
            if method == 'iqr':
                Q1 = data_processed[col].quantile(0.25)
                Q3 = data_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                data_processed[col] = data_processed[col].clip(lower_bound, upper_bound)
                
            elif method == 'zscore':
                mean_val = data_processed[col].mean()
                std_val = data_processed[col].std()
                lower_bound = mean_val - factor * std_val
                upper_bound = mean_val + factor * std_val
                
                data_processed[col] = data_processed[col].clip(lower_bound, upper_bound)
        
        return data_processed
    
    def preprocess_pipeline(self, data, is_training=True, remove_outliers=False):
        """
        完整的数据预处理流水线
        
        Args:
            data: 输入数据
            is_training: 是否为训练阶段
            remove_outliers: 是否移除异常值
            
        Returns:
            预处理后的数据
        """
        print("开始数据预处理...")
        
        # 1. 检查数据基本信息
        print(f"原始数据形状: {data.shape}")
        print(f"缺失值统计:\n{data[self.feature_columns].isnull().sum()}")
        
        # 2. 处理缺失值
        print("处理缺失值...")
        data_processed = self.handle_missing_values(
            data, 
            method=MISSING_VALUE_CONFIG['method']
        )
        
        # 3. 异常值检测和处理
        if is_training:
            print("检测异常值...")
            outlier_mask = self.detect_outliers(
                data_processed,
                method=OUTLIER_DETECTION['method'],
                contamination=OUTLIER_DETECTION['contamination']
            )
            print(f"检测到异常值数量: {outlier_mask.sum()}")
            
            if remove_outliers:
                print("移除异常值...")
                data_processed = self.remove_outliers(data_processed, outlier_mask)
                print(f"移除异常值后数据形状: {data_processed.shape}")
            else:
                print("截断异常值...")
                data_processed = self.clip_outliers(data_processed)
        
        # 4. 特征标准化
        print("特征标准化...")
        data_processed = self.scale_features(
            data_processed,
            method=SCALING_CONFIG['method']
        )
        
        print("数据预处理完成!")
        print(f"最终数据形状: {data_processed.shape}")
        
        return data_processed
    
    def get_preprocessing_info(self):
        """获取预处理器信息"""
        info = {
            'outlier_detector': self.outlier_detector is not None,
            'imputer': self.imputer is not None,
            'scaler': self.scaler is not None,
            'feature_columns': self.feature_columns
        }
        return info

def analyze_data_quality(data):
    """
    数据质量分析
    
    Args:
        data: 输入数据
        
    Returns:
        数据质量报告
    """
    report = {}
    
    # 基本统计信息
    report['basic_info'] = {
        'shape': data.shape,
        'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
        'dtypes': data.dtypes.to_dict()
    }
    
    # 缺失值分析
    missing_info = data.isnull().sum()
    report['missing_values'] = {
        'count': missing_info.to_dict(),
        'percentage': (missing_info / len(data) * 100).to_dict()
    }
    
    # 数值特征统计
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    report['numeric_stats'] = data[numeric_cols].describe().to_dict()
    
    # 类别分布（如果有目标变量）
    if TARGET_COLUMN in data.columns:
        report['target_distribution'] = data[TARGET_COLUMN].value_counts().to_dict()
    
    # 井的分布
    if GROUP_COLUMN in data.columns:
        well_stats = data.groupby(GROUP_COLUMN).size()
        report['well_distribution'] = {
            'count': len(well_stats),
            'min_samples': well_stats.min(),
            'max_samples': well_stats.max(),
            'mean_samples': well_stats.mean(),
            'std_samples': well_stats.std()
        }
    
    return report

def print_data_quality_report(report):
    """打印数据质量报告"""
    print("=" * 50)
    print("数据质量分析报告")
    print("=" * 50)
    
    # 基本信息
    print(f"数据形状: {report['basic_info']['shape']}")
    print(f"内存使用: {report['basic_info']['memory_usage']:.2f} MB")
    
    # 缺失值信息
    print("\n缺失值统计:")
    for col, count in report['missing_values']['count'].items():
        if count > 0:
            pct = report['missing_values']['percentage'][col]
            print(f"  {col}: {count} ({pct:.2f}%)")
    
    # 目标变量分布
    if 'target_distribution' in report:
        print(f"\n目标变量分布:")
        for label, count in report['target_distribution'].items():
            print(f"  类别 {label}: {count}")
    
    # 井分布信息
    if 'well_distribution' in report:
        print(f"\n井分布统计:")
        print(f"  井数量: {report['well_distribution']['count']}")
        print(f"  每井样本数 - 最小: {report['well_distribution']['min_samples']}")
        print(f"  每井样本数 - 最大: {report['well_distribution']['max_samples']}")
        print(f"  每井样本数 - 平均: {report['well_distribution']['mean_samples']:.2f}")
    
    print("=" * 50)

if __name__ == "__main__":
    # 测试代码
    from utils import load_data
    
    # 加载数据
    train_data = load_data('train')
    
    # 数据质量分析
    quality_report = analyze_data_quality(train_data)
    print_data_quality_report(quality_report)
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(train_data, is_training=True)
    
    print(f"\n预处理后数据形状: {processed_data.shape}")
    print(f"预处理器信息: {preprocessor.get_preprocessing_info()}")