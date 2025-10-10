"""
特征工程模块 - 中国石化AI竞赛油井含水率预测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

from config import FEATURE_CONFIG, TIME_COL, WELL_COL, TARGET_COL
from data_loader import DataLoader

logger = logging.getLogger(__name__)

import re

class FeatureEngineer:
    """特征工程主类"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.data = data_loader.data
        self.label_encoders = {}
        # 用于确保训练/预测阶段的特征选择一致
        self.selected_columns: List[str] = []
        self.numeric_medians: Dict[str, float] = {}
        self.target_col = ['含水']
        self.time_cols = ['日期']
        self.well_id_col = ['井号']
        self.id_col = ['id']
        self.layer_col = ['生产层位']
        self.date_col = ['生产时间']

    def create_all_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """创建所有特征"""
        logger.info("开始特征工程...")
        
        # 基础特征
        df = self._create_basic_features(df)
        
        # 时序特征
        df = self._create_temporal_features(df, is_training)
        
        # 空间特征
        df = self._create_spatial_features(df)
        
        # 事件特征
        df = self._create_event_features(df)
        
        # 水驱机制特征
        df = self._create_water_drive_features(df)
        
        # 统计特征
        df = self._create_statistical_features(df, is_training)
        
        logger.info(f"特征工程完成，最终特征数: {df.shape[1]}")
        return df
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建基础特征"""
        logger.info("创建基础特征...")
        
        # 时间特征
        df['year'] = df[TIME_COL].dt.year
        df['month'] = df[TIME_COL].dt.month
        df['day'] = df[TIME_COL].dt.day
        df['dayofweek'] = df[TIME_COL].dt.dayofweek
        df['dayofyear'] = df[TIME_COL].dt.dayofyear
        df['quarter'] = df[TIME_COL].dt.quarter
        df['is_weekend'] = (df[TIME_COL].dt.dayofweek >= 5).astype(int)
        df['is_month_start'] = df[TIME_COL].dt.is_month_start.astype(int)
        df['is_month_end'] = df[TIME_COL].dt.is_month_end.astype(int)
        
        # 井号编码
        if WELL_COL not in self.label_encoders:
            self.label_encoders[WELL_COL] = LabelEncoder()
            df['well_encoded'] = self.label_encoders[WELL_COL].fit_transform(df[WELL_COL])
        else:
            df['well_encoded'] = self.label_encoders[WELL_COL].transform(df[WELL_COL])
        
        # 生产层位编码
        if '生产层位' in df.columns:
            if '生产层位' not in self.label_encoders:
                self.label_encoders['生产层位'] = LabelEncoder()
                df['layer_encoded'] = self.label_encoders['生产层位'].fit_transform(df['生产层位'].fillna('unknown'))
            else:
                df['layer_encoded'] = self.label_encoders['生产层位'].transform(df['生产层位'].fillna('unknown'))
        
        # 生产效率特征
        if '日产液(t)' in df.columns and '生产时间' in df.columns:
            df['production_efficiency'] = df['日产液(t)'] / (df['生产时间'] + 1e-6)
        
        if '日产油(t)' in df.columns and '日产液(t)' in df.columns:
            df['oil_liquid_ratio'] = df['日产油(t)'] / (df['日产液(t)'] + 1e-6)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """创建时序特征"""
        logger.info("创建时序特征...")
        
        # 按井号排序
        df = df.sort_values([WELL_COL, TIME_COL]).reset_index(drop=True)
        
        # 滞后特征
        lag_cols = ['日产液(t)', '生产时间']
        # if is_training and TARGET_COL in df.columns:
        #     lag_cols.append(TARGET_COL)
        
        for col in lag_cols:
            if col in df.columns:
                for lag in FEATURE_CONFIG['lag_days']:
                    df[f'{col}_lag_{lag}'] = df.groupby(WELL_COL)[col].shift(lag)
        
        # 滑动窗口统计特征
        for col in lag_cols:
            if col in df.columns:
                for window in FEATURE_CONFIG['rolling_windows']:
                    df[f'{col}_rolling_mean_{window}'] = df.groupby(WELL_COL)[col].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
                    df[f'{col}_rolling_std_{window}'] = df.groupby(WELL_COL)[col].rolling(window, min_periods=1).std().reset_index(0, drop=True)
                    df[f'{col}_rolling_max_{window}'] = df.groupby(WELL_COL)[col].rolling(window, min_periods=1).max().reset_index(0, drop=True)
                    df[f'{col}_rolling_min_{window}'] = df.groupby(WELL_COL)[col].rolling(window, min_periods=1).min().reset_index(0, drop=True)
        
        # 指数移动平均
        for col in lag_cols:
            if col in df.columns:
                df[f'{col}_ema_7'] = df.groupby(WELL_COL)[col].ewm(span=7).mean().reset_index(0, drop=True)
                df[f'{col}_ema_30'] = df.groupby(WELL_COL)[col].ewm(span=30).mean().reset_index(0, drop=True)
        
        # 趋势特征
        for col in lag_cols:
            if col in df.columns:
                df[f'{col}_diff_1'] = df.groupby(WELL_COL)[col].diff(1)
                df[f'{col}_diff_7'] = df.groupby(WELL_COL)[col].diff(7)
                df[f'{col}_pct_change_1'] = df.groupby(WELL_COL)[col].pct_change(1)
                df[f'{col}_pct_change_7'] = df.groupby(WELL_COL)[col].pct_change(7)
        
        return df
    
    def _create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建空间特征"""
        logger.info("创建空间特征...")
        
        if 'well_info' not in self.data:
            logger.warning("井位坐标数据不可用，跳过空间特征")
            return df
        
        well_info = self.data['well_info']
        
        # 合并井位坐标
        df = df.merge(well_info, on=WELL_COL, how='left')
        
        # 计算井间距离特征
        oil_wells, water_wells = self.data_loader.get_wells_info()
        
        # 为每口油井计算到最近注水井的距离
        distance_features = []
        
        for _, row in df.iterrows():
            if pd.isna(row['横坐标(m)']) or pd.isna(row['纵坐标(m)']):
                distance_features.append({
                    'min_dist_to_water_well': np.nan,
                    'avg_dist_to_nearest_3_water_wells': np.nan,
                    'nearest_water_well_count_1km': 0
                })
                continue
            
            current_pos = (row['横坐标(m)'], row['纵坐标(m)'])
            distances = []
            
            # 计算到所有水井的距离
            water_well_info = well_info[well_info[WELL_COL].isin(water_wells)]
            for _, water_well in water_well_info.iterrows():
                if pd.notna(water_well['横坐标(m)']) and pd.notna(water_well['纵坐标(m)']):
                    water_pos = (water_well['横坐标(m)'], water_well['纵坐标(m)'])
                    dist = np.sqrt((current_pos[0] - water_pos[0])**2 + (current_pos[1] - water_pos[1])**2)
                    distances.append(dist)
            
            if distances:
                distances = sorted(distances)
                min_dist = distances[0]
                avg_dist_3 = np.mean(distances[:min(3, len(distances))])
                count_1km = sum(1 for d in distances if d <= 1000)
            else:
                min_dist = np.nan
                avg_dist_3 = np.nan
                count_1km = 0
            
            distance_features.append({
                'min_dist_to_water_well': min_dist,
                'avg_dist_to_nearest_3_water_wells': avg_dist_3,
                'nearest_water_well_count_1km': count_1km
            })
        
        # 添加距离特征到数据框
        distance_df = pd.DataFrame(distance_features)
        df = pd.concat([df, distance_df], axis=1)
        
        return df
    
    def _create_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建事件特征（射孔）"""
        logger.info("创建事件特征...")
        
        if 'perforation' not in self.data:
            logger.warning("射孔数据不可用，跳过事件特征")
            return df
        
        perforation = self.data['perforation']
        
        # 为每个井-日期组合计算射孔特征
        event_features = []
        
        for _, row in df.iterrows():
            well = row[WELL_COL]
            date = row[TIME_COL]
            
            well_perforations = perforation[perforation[WELL_COL] == well]
            
            if well_perforations.empty:
                event_features.append({
                    'has_perforation': 0,
                    'days_since_last_perforation': np.nan,
                    'total_perforations': 0,
                    'perforation_thickness_total': 0
                })
                continue
            
            # 找到在当前日期之前的射孔
            past_perforations = well_perforations[well_perforations['起始日期'] <= date]
            
            if past_perforations.empty:
                has_perf = 0
                days_since = np.nan
                total_perfs = 0
                total_thickness = 0
            else:
                has_perf = 1
                last_perf_date = past_perforations['起始日期'].max()
                days_since = (date - last_perf_date).days
                total_perfs = len(past_perforations)
                total_thickness = past_perforations['射孔厚度(m)'].sum() if '射孔厚度(m)' in past_perforations.columns else 0
            
            event_features.append({
                'has_perforation': has_perf,
                'days_since_last_perforation': days_since,
                'total_perforations': total_perfs,
                'perforation_thickness_total': total_thickness
            })
        
        # 添加事件特征到数据框
        event_df = pd.DataFrame(event_features)
        df = pd.concat([df, event_df], axis=1)
        
        return df
    
    def _create_water_drive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建水驱机制特征"""
        logger.info("创建水驱机制特征...")
        
        if 'daily_water' not in self.data:
            logger.warning("水井日度数据不可用，跳过水驱特征")
            return df
        
        daily_water = self.data['daily_water']
        
        # 按日期聚合注水数据
        water_daily_agg = daily_water.groupby(TIME_COL).agg({
            '日注水(m3)': ['sum', 'mean', 'count'],
            '油 压(Mpa)': ['mean', 'max'],
            '干 压(Mpa)': ['mean', 'max']
        }).reset_index()
        
        # 扁平化列名
        water_daily_agg.columns = [TIME_COL, 'total_water_injection', 'avg_water_injection', 
                                  'active_water_wells', 'avg_oil_pressure', 'max_oil_pressure',
                                  'avg_dry_pressure', 'max_dry_pressure']
        
        # 合并到主数据框
        df = df.merge(water_daily_agg, on=TIME_COL, how='left')
        
        # 计算累积注水量
        df = df.sort_values(TIME_COL)
        df['cumulative_water_injection'] = df['total_water_injection'].cumsum()
        
        # 注水强度特征
        df['water_injection_intensity'] = df['total_water_injection'] / (df['active_water_wells'] + 1e-6)
        
        # 压力比特征
        df['pressure_ratio'] = df['avg_oil_pressure'] / (df['avg_dry_pressure'] + 1e-6)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """创建统计特征"""
        logger.info("创建统计特征...")
        
        # 按井统计特征
        well_stats = df.groupby(WELL_COL).agg({
            '日产液(t)': ['mean', 'std', 'max', 'min'],
            '生产时间': ['mean', 'std'],
        }).reset_index()
        
        # 扁平化列名
        well_stats.columns = [WELL_COL] + [f'well_{col[0]}_{col[1]}' for col in well_stats.columns[1:]]
        
        # 合并到主数据框
        df = df.merge(well_stats, on=WELL_COL, how='left')
        
        # 按月统计特征
        df['year_month'] = df[TIME_COL].dt.to_period('M')
        monthly_stats = df.groupby('year_month').agg({
            '日产液(t)': ['mean', 'std'],
            'total_water_injection': ['mean', 'std']
        }).reset_index()
        
        # 扁平化列名
        monthly_stats.columns = ['year_month'] + [f'monthly_{col[0]}_{col[1]}' for col in monthly_stats.columns[1:]]
        
        # 合并到主数据框
        df = df.merge(monthly_stats, on='year_month', how='left')
        
        return df
    
    def _clean_col_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理列名，替换特殊字符并确保唯一性"""
        cols = df.columns
        new_cols = []
        seen_cols = {}
        for col in cols:
            # 替换特殊字符，保留中文
            cleaned_col = re.sub(r'[\s\[\]{}()<>,]+', '_', col)
            
            # 确保唯一性
            if cleaned_col in seen_cols:
                seen_cols[cleaned_col] += 1
                new_col = f"{cleaned_col}_{seen_cols[cleaned_col]}"
            else:
                seen_cols[cleaned_col] = 0
                new_col = cleaned_col
            new_cols.append(new_col)
        df.columns = new_cols
        return df

    def create_all_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """创建所有特征"""
        logger.info("开始特征工程...")
        
        # 基础特征
        df = self._create_basic_features(df)
        
        # 时序特征
        df = self._create_temporal_features(df, is_training)
        
        # 空间特征
        df = self._create_spatial_features(df)
        
        # 事件特征
        df = self._create_event_features(df)
        
        # 水驱机制特征
        df = self._create_water_drive_features(df)
        
        # 统计特征
        df = self._create_statistical_features(df, is_training)
        
        # 清理列名
        df = self._clean_col_names(df)

        logger.info(f"特征工程完成，最终特征数: {len(df.columns)}")
        return df

    def fit_select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """在训练阶段进行特征选择，并保存所选列和填充值"""
        logger.info("开始进行特征选择...")
        
        # 排除目标列、ID列及其他非特征列
        exclude_from_selection = self.target_col + self.id_col + self.well_id_col + self.layer_col + self.date_col + ['year_month'] + self.time_cols
        
        # 确保所有排除列都存在于df中，避免KeyError
        existing_exclude_cols = [col for col in exclude_from_selection if col in df.columns]
        
        # 移除高缺失率的特征
        missing_threshold = 0.8
        missing_rates = df.isnull().mean()
        high_missing_cols = missing_rates[missing_rates > missing_threshold].index.tolist()
        if high_missing_cols:
            logger.info(f"移除高缺失率特征: {high_missing_cols}")
            df = df.drop(columns=high_missing_cols)
        
        # 移除常数特征
        constant_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        if constant_cols:
            logger.info(f"移除常数特征: {constant_cols}")
            df = df.drop(columns=constant_cols)
        
        # 处理无穷大值和缺失
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)
        self.numeric_medians = medians.to_dict()
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        df[categorical_cols] = df[categorical_cols].fillna('unknown')
        
        # 保存所选列，确保预测阶段一致
        self.selected_columns = list(df.columns)
        logger.info(f"特征选择前，特征数量: {len(self.selected_columns)}")
    
        # 移除目标列和ID列
        cols_to_exclude = self.target_col + self.time_cols + self.well_id_col + self.id_col + self.layer_col + self.date_col + ['year_month']
        self.selected_columns = [col for col in self.selected_columns if col not in cols_to_exclude]
    
        logger.info(f"移除目标列和ID列后，特征数量: {len(self.selected_columns)}")
    
        # 保存数值特征的中位数
        numeric_features = df[self.selected_columns].select_dtypes(include=np.number).columns.tolist()
        
        logger.info(f"特征选择完成，保留列数: {len(self.selected_columns)}")
        return df[self.selected_columns]

    def transform_select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """在预测阶段按训练阶段保存的列进行对齐和填充"""
        logger.info("进行特征选择(预测阶段，对齐训练列)...")
        if not self.selected_columns:
            logger.warning("训练阶段未保存selected_columns，退回到默认选择逻辑")
            return self.fit_select_features(df)
        
        # 先进行基本清洗
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # 对齐列：缺失列用训练阶段中位数/unknown填充
        for col in self.selected_columns:
            if col not in df.columns:
                if col in self.numeric_medians:
                    df[col] = self.numeric_medians[col]
                else:
                    df[col] = 'unknown'
        
        # 只保留训练阶段的列顺序
        df_aligned = df[self.selected_columns].copy()
        
        # 再次填充缺失值以防存在NaN
        num_cols = [c for c in df_aligned.columns if c in self.numeric_medians]
        df_aligned[num_cols] = df_aligned[num_cols].fillna(pd.Series(self.numeric_medians))
        obj_cols = df_aligned.select_dtypes(include=['object']).columns
        df_aligned[obj_cols] = df_aligned[obj_cols].fillna('unknown')
        
        logger.info(f"预测阶段特征对齐完成，列数: {df_aligned.shape[1]}")
        return df_aligned

def main():
    """测试特征工程功能"""
    # 加载数据
    loader = DataLoader()
    data = loader.load_all_data()
    
    # 创建特征工程器
    feature_engineer = FeatureEngineer(loader)
    
    # 在训练数据上创建特征
    train_df = data['daily_oil_train'].copy()
    train_features = feature_engineer.create_all_features(train_df, is_training=True)
    train_features = feature_engineer.select_features(train_features)
    
    print(f"训练特征形状: {train_features.shape}")
    print(f"特征列: {list(train_features.columns)}")
    
    # 在验证数据上创建特征
    val_df = data['validation'].copy()
    val_features = feature_engineer.create_all_features(val_df, is_training=False)
    val_features = feature_engineer.select_features(val_features)
    
    print(f"验证特征形状: {val_features.shape}")

if __name__ == "__main__":
    main()