"""
中国石化岩性识别AI竞赛 - 特征工程模块
包含滑动窗口特征、梯度特征、测井相特征、频域特征等
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import savgol_filter
import pywt
import warnings
warnings.filterwarnings('ignore')

from config import (
    FEATURE_COLUMNS, GROUP_COLUMN, DEPTH_COLUMN,
    ROLLING_WINDOW_CONFIG, GRADIENT_CONFIG, LOG_FACIES_CONFIG, FREQUENCY_CONFIG
)

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        self.log_facies_model = None
        self.feature_names = []
        
    def create_rolling_features(self, data, windows=None, features=None, center=True):
        """
        创建滑动窗口特征
        
        Args:
            data: 输入数据
            windows: 窗口大小列表
            features: 统计特征列表
            center: 是否居中
            
        Returns:
            包含滑动窗口特征的数据
        """
        if windows is None:
            windows = ROLLING_WINDOW_CONFIG['windows']
        if features is None:
            features = ROLLING_WINDOW_CONFIG['features']
            
        data_with_features = data.copy()
        
        print(f"创建滑动窗口特征，窗口大小: {windows}")
        
        for window in windows:
            for col in FEATURE_COLUMNS:
                # 按井分组计算滑动窗口特征
                grouped = data_with_features.groupby(GROUP_COLUMN)[col]
                
                for feature in features:
                    feature_name = f"{col}_rolling_{window}_{feature}"
                    
                    if feature == 'mean':
                        data_with_features[feature_name] = grouped.transform(
                            lambda x: x.rolling(window=window, center=center, min_periods=1).mean()
                        )
                    elif feature == 'std':
                        data_with_features[feature_name] = grouped.transform(
                            lambda x: x.rolling(window=window, center=center, min_periods=1).std()
                        )
                    elif feature == 'skew':
                        data_with_features[feature_name] = grouped.transform(
                            lambda x: x.rolling(window=window, center=center, min_periods=3).skew()
                        )
                    elif feature == 'kurt':
                        data_with_features[feature_name] = grouped.transform(
                            lambda x: x.rolling(window=window, center=center, min_periods=4).kurt()
                        )
                    elif feature == 'min':
                        data_with_features[feature_name] = grouped.transform(
                            lambda x: x.rolling(window=window, center=center, min_periods=1).min()
                        )
                    elif feature == 'max':
                        data_with_features[feature_name] = grouped.transform(
                            lambda x: x.rolling(window=window, center=center, min_periods=1).max()
                        )
                    elif feature == 'median':
                        data_with_features[feature_name] = grouped.transform(
                            lambda x: x.rolling(window=window, center=center, min_periods=1).median()
                        )
                    elif feature == 'quantile_25':
                        data_with_features[feature_name] = grouped.transform(
                            lambda x: x.rolling(window=window, center=center, min_periods=1).quantile(0.25)
                        )
                    elif feature == 'quantile_75':
                        data_with_features[feature_name] = grouped.transform(
                            lambda x: x.rolling(window=window, center=center, min_periods=1).quantile(0.75)
                        )
                    
                    # 填充缺失值
                    data_with_features[feature_name] = data_with_features[feature_name].fillna(
                        data_with_features[col]
                    )
        
        return data_with_features
    
    def create_gradient_features(self, data, orders=None, method='gradient'):
        """
        创建梯度特征
        
        Args:
            data: 输入数据
            orders: 梯度阶数列表
            method: 计算方法 ('gradient', 'diff')
            
        Returns:
            包含梯度特征的数据
        """
        if orders is None:
            orders = GRADIENT_CONFIG['orders']
            
        data_with_features = data.copy()
        
        print(f"创建梯度特征，阶数: {orders}")
        
        for order in orders:
            for col in FEATURE_COLUMNS:
                feature_name = f"{col}_gradient_{order}"
                
                if method == 'gradient':
                    # 使用numpy.gradient计算梯度
                    grouped = data_with_features.groupby(GROUP_COLUMN)[col]
                    gradients = []
                    
                    for name, group in grouped:
                        if len(group) > order:
                            grad = np.gradient(group.values)
                            for _ in range(order - 1):
                                grad = np.gradient(grad)
                            gradients.extend(grad)
                        else:
                            gradients.extend([0] * len(group))
                    
                    data_with_features[feature_name] = gradients
                    
                elif method == 'diff':
                    # 使用差分计算梯度
                    grouped = data_with_features.groupby(GROUP_COLUMN)[col]
                    data_with_features[feature_name] = grouped.transform(
                        lambda x: x.diff(periods=order).fillna(0)
                    )
        
        return data_with_features
    
    def create_log_facies_features(self, data, n_clusters=None, method='kmeans', include_distance=True):
        """
        创建测井相特征
        
        Args:
            data: 输入数据
            n_clusters: 聚类数量
            method: 聚类方法 ('kmeans', 'gmm')
            include_distance: 是否包含到聚类中心的距离
            
        Returns:
            包含测井相特征的数据
        """
        if n_clusters is None:
            n_clusters = LOG_FACIES_CONFIG['n_clusters']
            
        data_with_features = data.copy()
        
        print(f"创建测井相特征，聚类数量: {n_clusters}")
        
        # 准备聚类数据
        cluster_data = data[FEATURE_COLUMNS].copy()
        
        # 标准化
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # 聚类
        if method == 'kmeans':
            if self.log_facies_model is None:
                self.log_facies_model = KMeans(
                    n_clusters=n_clusters,
                    random_state=LOG_FACIES_CONFIG['random_state'],
                    n_init=10
                )
                cluster_labels = self.log_facies_model.fit_predict(cluster_data_scaled)
            else:
                cluster_labels = self.log_facies_model.predict(cluster_data_scaled)
                
        elif method == 'gmm':
            if self.log_facies_model is None:
                self.log_facies_model = GaussianMixture(
                    n_components=n_clusters,
                    random_state=LOG_FACIES_CONFIG['random_state']
                )
                cluster_labels = self.log_facies_model.fit_predict(cluster_data_scaled)
            else:
                cluster_labels = self.log_facies_model.predict(cluster_data_scaled)
        
        # 添加聚类标签
        data_with_features['log_facies'] = cluster_labels
        
        # 添加到聚类中心的距离
        if include_distance:
            if hasattr(self.log_facies_model, 'cluster_centers_'):
                centers = self.log_facies_model.cluster_centers_
            elif hasattr(self.log_facies_model, 'means_'):
                centers = self.log_facies_model.means_
            else:
                centers = None
                
            if centers is not None:
                distances = []
                for i, point in enumerate(cluster_data_scaled):
                    cluster_id = cluster_labels[i]
                    distance = np.linalg.norm(point - centers[cluster_id])
                    distances.append(distance)
                
                data_with_features['log_facies_distance'] = distances
        
        # 创建聚类的one-hot编码
        for i in range(n_clusters):
            data_with_features[f'log_facies_{i}'] = (cluster_labels == i).astype(int)
        
        return data_with_features
    
    def create_frequency_features(self, data, methods=None):
        """
        创建频域特征
        
        Args:
            data: 输入数据
            methods: 频域分析方法列表
            
        Returns:
            包含频域特征的数据
        """
        if methods is None:
            methods = FREQUENCY_CONFIG['methods']
            
        data_with_features = data.copy()
        
        print(f"创建频域特征，方法: {methods}")
        
        for col in FEATURE_COLUMNS:
            # 按井分组处理
            grouped = data_with_features.groupby(GROUP_COLUMN)[col]
            
            for method in methods:
                if method == 'fft':
                    # FFT特征
                    fft_features = []
                    
                    for name, group in grouped:
                        if len(group) >= 8:  # 最小长度要求
                            # 计算FFT
                            fft_values = np.fft.fft(group.values)
                            fft_magnitude = np.abs(fft_values)
                            
                            # 取前几个主要频率分量
                            n_components = min(FREQUENCY_CONFIG['fft_components'], len(fft_magnitude) // 2)
                            main_components = fft_magnitude[1:n_components+1]  # 排除直流分量
                            
                            # 填充到固定长度
                            if len(main_components) < FREQUENCY_CONFIG['fft_components']:
                                main_components = np.pad(
                                    main_components, 
                                    (0, FREQUENCY_CONFIG['fft_components'] - len(main_components)),
                                    'constant'
                                )
                            
                            fft_features.extend([main_components] * len(group))
                        else:
                            # 对于太短的序列，用零填充
                            zero_components = np.zeros(FREQUENCY_CONFIG['fft_components'])
                            fft_features.extend([zero_components] * len(group))
                    
                    # 添加FFT特征
                    fft_array = np.array(fft_features)
                    for i in range(FREQUENCY_CONFIG['fft_components']):
                        data_with_features[f'{col}_fft_{i}'] = fft_array[:, i]
                
                elif method == 'wavelet':
                    # 小波变换特征
                    wavelet_features = []
                    
                    for name, group in grouped:
                        if len(group) >= 8:
                            try:
                                # 小波分解
                                coeffs = pywt.wavedec(
                                    group.values, 
                                    FREQUENCY_CONFIG['wavelet_name'],
                                    level=FREQUENCY_CONFIG['wavelet_levels']
                                )
                                
                                # 提取统计特征
                                wavelet_stats = []
                                for coeff in coeffs:
                                    if len(coeff) > 0:
                                        wavelet_stats.extend([
                                            np.mean(coeff),
                                            np.std(coeff),
                                            np.max(coeff),
                                            np.min(coeff)
                                        ])
                                
                                wavelet_features.extend([wavelet_stats] * len(group))
                            except:
                                # 如果小波变换失败，用零填充
                                zero_stats = [0] * (FREQUENCY_CONFIG['wavelet_levels'] + 1) * 4
                                wavelet_features.extend([zero_stats] * len(group))
                        else:
                            zero_stats = [0] * (FREQUENCY_CONFIG['wavelet_levels'] + 1) * 4
                            wavelet_features.extend([zero_stats] * len(group))
                    
                    # 添加小波特征
                    if wavelet_features:
                        wavelet_array = np.array(wavelet_features)
                        n_wavelet_features = wavelet_array.shape[1]
                        for i in range(n_wavelet_features):
                            data_with_features[f'{col}_wavelet_{i}'] = wavelet_array[:, i]
        
        return data_with_features
    
    def create_depth_features(self, data):
        """
        创建深度相关特征
        
        Args:
            data: 输入数据
            
        Returns:
            包含深度特征的数据
        """
        data_with_features = data.copy()
        
        print("创建深度相关特征")
        
        # 深度分位数特征
        depth_quantiles = data[DEPTH_COLUMN].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        for q, value in depth_quantiles.items():
            data_with_features[f'depth_quantile_{int(q*100)}'] = (data[DEPTH_COLUMN] <= value).astype(int)
        
        # 相对深度特征
        grouped = data_with_features.groupby(GROUP_COLUMN)[DEPTH_COLUMN]
        data_with_features['depth_relative'] = grouped.transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
        
        # 深度变化率
        data_with_features['depth_diff'] = grouped.transform(lambda x: x.diff().fillna(0))
        
        return data_with_features
    
    def create_well_features(self, data):
        """
        创建井相关特征
        
        Args:
            data: 输入数据
            
        Returns:
            包含井特征的数据
        """
        data_with_features = data.copy()
        
        print("创建井相关特征")
        
        # 井的统计特征
        well_stats = data.groupby(GROUP_COLUMN)[FEATURE_COLUMNS].agg(['mean', 'std', 'min', 'max'])
        well_stats.columns = ['_'.join(col).strip() for col in well_stats.columns.values]
        
        # 将井统计特征合并到原数据
        data_with_features = data_with_features.merge(
            well_stats, 
            left_on=GROUP_COLUMN, 
            right_index=True, 
            how='left'
        )
        
        # 相对于井均值的偏差
        for col in FEATURE_COLUMNS:
            data_with_features[f'{col}_deviation_from_well_mean'] = (
                data_with_features[col] - data_with_features[f'{col}_mean']
            )
        
        return data_with_features
    
    def create_interaction_features(self, data):
        """
        创建交互特征
        
        Args:
            data: 输入数据
            
        Returns:
            包含交互特征的数据
        """
        data_with_features = data.copy()
        
        print("创建交互特征")
        
        # 特征比值
        feature_pairs = [
            ('SP', 'GR'),
            ('SP', 'AC'),
            ('GR', 'AC')
        ]
        
        for feat1, feat2 in feature_pairs:
            # 比值特征
            data_with_features[f'{feat1}_{feat2}_ratio'] = (
                data_with_features[feat1] / (data_with_features[feat2] + 1e-8)
            )
            
            # 乘积特征
            data_with_features[f'{feat1}_{feat2}_product'] = (
                data_with_features[feat1] * data_with_features[feat2]
            )
            
            # 差值特征
            data_with_features[f'{feat1}_{feat2}_diff'] = (
                data_with_features[feat1] - data_with_features[feat2]
            )
        
        # 三特征组合
        data_with_features['SP_GR_AC_sum'] = (
            data_with_features['SP'] + data_with_features['GR'] + data_with_features['AC']
        )
        
        data_with_features['SP_GR_AC_product'] = (
            data_with_features['SP'] * data_with_features['GR'] * data_with_features['AC']
        )
        
        return data_with_features
    
    def feature_engineering_pipeline(self, data, is_training=True):
        """
        完整的特征工程流水线
        
        Args:
            data: 输入数据
            is_training: 是否为训练阶段
            
        Returns:
            特征工程后的数据
        """
        print("开始特征工程...")
        print(f"原始特征数量: {len(FEATURE_COLUMNS)}")
        
        data_processed = data.copy()
        
        # 1. 滑动窗口特征
        data_processed = self.create_rolling_features(data_processed)
        
        # 2. 梯度特征
        data_processed = self.create_gradient_features(data_processed)
        
        # 3. 测井相特征
        data_processed = self.create_log_facies_features(data_processed)
        
        # 4. 频域特征
        data_processed = self.create_frequency_features(data_processed)
        
        # 5. 深度特征
        data_processed = self.create_depth_features(data_processed)
        
        # 6. 井特征
        data_processed = self.create_well_features(data_processed)
        
        # 7. 交互特征
        data_processed = self.create_interaction_features(data_processed)
        
        # 获取新特征列名
        original_columns = set(data.columns)
        new_columns = set(data_processed.columns)
        self.feature_names = list(new_columns - original_columns)
        
        print(f"新增特征数量: {len(self.feature_names)}")
        print(f"总特征数量: {len(data_processed.columns)}")
        print("特征工程完成!")
        
        return data_processed
    
    def get_feature_names(self):
        """获取新增的特征名称"""
        return self.feature_names
    
    def get_feature_importance_by_type(self, feature_names, importance_values):
        """
        按特征类型分组显示特征重要性
        
        Args:
            feature_names: 特征名称列表
            importance_values: 特征重要性值列表
            
        Returns:
            按类型分组的特征重要性
        """
        feature_types = {
            'original': [],
            'rolling': [],
            'gradient': [],
            'log_facies': [],
            'frequency': [],
            'depth': [],
            'well': [],
            'interaction': []
        }
        
        for name, importance in zip(feature_names, importance_values):
            if name in FEATURE_COLUMNS:
                feature_types['original'].append((name, importance))
            elif 'rolling' in name:
                feature_types['rolling'].append((name, importance))
            elif 'gradient' in name:
                feature_types['gradient'].append((name, importance))
            elif 'log_facies' in name:
                feature_types['log_facies'].append((name, importance))
            elif 'fft' in name or 'wavelet' in name:
                feature_types['frequency'].append((name, importance))
            elif 'depth' in name:
                feature_types['depth'].append((name, importance))
            elif any(x in name for x in ['mean', 'std', 'min', 'max', 'deviation']):
                feature_types['well'].append((name, importance))
            else:
                feature_types['interaction'].append((name, importance))
        
        # 排序每个类型的特征
        for feature_type in feature_types:
            feature_types[feature_type].sort(key=lambda x: x[1], reverse=True)
        
        return feature_types

if __name__ == "__main__":
    # 测试代码
    from utils import load_data
    from data_preprocessing import DataPreprocessor
    
    # 加载和预处理数据
    train_data = load_data('train')
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(train_data, is_training=True)
    
    # 特征工程
    feature_engineer = FeatureEngineer()
    featured_data = feature_engineer.feature_engineering_pipeline(processed_data, is_training=True)
    
    print(f"\n特征工程后数据形状: {featured_data.shape}")
    print(f"新增特征: {len(feature_engineer.get_feature_names())}")
    print(f"前10个新特征: {feature_engineer.get_feature_names()[:10]}")