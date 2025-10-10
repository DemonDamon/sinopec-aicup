"""
数据加载模块 - 中国石化AI竞赛油井含水率预测
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import DATA_PATHS, DATE_FORMAT, TIME_COL, WELL_COL, TARGET_COL

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """数据加载器类"""
    
    def __init__(self):
        self.data = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有数据文件"""
        logger.info("开始加载数据...")
        
        # 加载静态数据
        self.data['well_info'] = self._load_well_info()
        self.data['well_deviation'] = self._load_well_deviation()
        self.data['perforation'] = self._load_perforation()
        
        # 加载动态数据
        self.data['daily_oil_train'] = self._load_daily_oil_train()
        self.data['daily_water'] = self._load_daily_water()
        self.data['monthly_oil'] = self._load_monthly_oil()
        self.data['monthly_water'] = self._load_monthly_water()
        
        # 加载预测数据
        self.data['validation'] = self._load_validation()
        self.data['sample_submission'] = self._load_sample_submission()
        
        logger.info("数据加载完成")
        return self.data
    
    def _load_well_info(self) -> pd.DataFrame:
        """加载井位坐标数据"""
        logger.info("加载井位坐标数据...")
        try:
            df = pd.read_csv(DATA_PATHS['well_info'], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATHS['well_info'], encoding='gbk')
        
        # 统一列名格式
        df.columns = df.columns.str.strip()  # 去除空格
        if '井  号' in df.columns:
            df = df.rename(columns={'井  号': WELL_COL})
        
        logger.info(f"井位坐标数据: {df.shape}")
        return df
    
    def _load_well_deviation(self) -> pd.DataFrame:
        """加载井斜数据"""
        logger.info("加载井斜数据...")
        try:
            df = pd.read_csv(DATA_PATHS['well_deviation'], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATHS['well_deviation'], encoding='gbk')
        logger.info(f"井斜数据: {df.shape}")
        return df
    
    def _load_perforation(self) -> pd.DataFrame:
        """加载射孔数据"""
        logger.info("加载射孔数据...")
        try:
            df = pd.read_csv(DATA_PATHS['perforation'], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATHS['perforation'], encoding='gbk')
        
        # 处理日期列
        if '起始日期' in df.columns:
            df['起始日期'] = pd.to_datetime(df['起始日期'], format=DATE_FORMAT, errors='coerce')
        if '终止日期' in df.columns:
            df['终止日期'] = pd.to_datetime(df['终止日期'], format=DATE_FORMAT, errors='coerce')
            
        logger.info(f"射孔数据: {df.shape}")
        return df
    
    def _load_daily_oil_train(self) -> pd.DataFrame:
        """加载油井日度训练数据"""
        logger.info("加载油井日度训练数据...")
        try:
            df = pd.read_csv(DATA_PATHS['daily_oil_train'], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATHS['daily_oil_train'], encoding='gbk')
        
        # 处理日期列
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], format=DATE_FORMAT, errors='coerce')
        
        # 处理数值列
        numeric_cols = ['生产时间', '日产液(t)', '日产油(t)', TARGET_COL]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"油井日度训练数据: {df.shape}")
        logger.info(f"日期范围: {df[TIME_COL].min()} 到 {df[TIME_COL].max()}")
        logger.info(f"井数量: {df[WELL_COL].nunique()}")
        return df
    
    def _load_daily_water(self) -> pd.DataFrame:
        """加载水井日度数据"""
        logger.info("加载水井日度数据...")
        try:
            df = pd.read_csv(DATA_PATHS['daily_water'], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATHS['daily_water'], encoding='gbk')
        
        # 处理日期列
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], format=DATE_FORMAT, errors='coerce')
        
        # 处理数值列
        numeric_cols = ['生产时间', '日注水(m3)', '干 压(Mpa)', '油 压(Mpa)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"水井日度数据: {df.shape}")
        logger.info(f"日期范围: {df[TIME_COL].min()} 到 {df[TIME_COL].max()}")
        logger.info(f"井数量: {df[WELL_COL].nunique()}")
        return df
    
    def _load_monthly_oil(self) -> pd.DataFrame:
        """加载油井月度数据"""
        logger.info("加载油井月度数据...")
        try:
            df = pd.read_csv(DATA_PATHS['monthly_oil'], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATHS['monthly_oil'], encoding='gbk')
        
        # 处理年月列
        if '年' in df.columns and '月' in df.columns:
            df['年月'] = df['年'].astype(str) + df['月'].astype(str).str.zfill(2)
            df['年月'] = pd.to_datetime(df['年月'], format='%Y%m', errors='coerce')
        
        # 处理数值列
        numeric_cols = ['有效厚度(m)', '生产天数', TARGET_COL, '动液面(m)', 
                       '月产液(t)', '月产油(t)', '月产水(m^3)', '累产油(10^4t)', '累产水(10^4m^3)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"油井月度数据: {df.shape}")
        return df
    
    def _load_monthly_water(self) -> pd.DataFrame:
        """加载水井月度数据"""
        logger.info("加载水井月度数据...")
        try:
            df = pd.read_csv(DATA_PATHS['monthly_water'], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATHS['monthly_water'], encoding='gbk')
        
        # 处理年月列
        if '年' in df.columns and '月' in df.columns:
            df['年月'] = df['年'].astype(str) + df['月'].astype(str).str.zfill(2)
            df['年月'] = pd.to_datetime(df['年月'], format='%Y%m', errors='coerce')
        
        # 处理数值列
        numeric_cols = ['砂厚(m)', '生产天数', '日注能力(m3)', '干压(Mpa)', 
                       '油压(Mpa)', '套压(Mpa)', '月注水(m3)', '年注水(10^4m3)', '累注水(10^4m3)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"水井月度数据: {df.shape}")
        return df
    
    def _load_validation(self) -> pd.DataFrame:
        """加载验证数据"""
        logger.info("加载验证数据...")
        try:
            df = pd.read_csv(DATA_PATHS['validation'], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATHS['validation'], encoding='gbk')
        
        # 处理日期列
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], format=DATE_FORMAT, errors='coerce')
        
        # 处理数值列
        numeric_cols = ['生产时间', '日产液(t)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"验证数据: {df.shape}")
        logger.info(f"日期范围: {df[TIME_COL].min()} 到 {df[TIME_COL].max()}")
        logger.info(f"井数量: {df[WELL_COL].nunique()}")
        return df
    
    def _load_sample_submission(self) -> pd.DataFrame:
        """加载样例提交文件"""
        logger.info("加载样例提交文件...")
        try:
            df = pd.read_csv(DATA_PATHS['sample_submission'], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATHS['sample_submission'], encoding='gbk')
        logger.info(f"样例提交文件: {df.shape}")
        return df
    
    def get_data_summary(self) -> Dict:
        """获取数据摘要信息"""
        if not self.data:
            self.load_all_data()
        
        summary = {}
        for name, df in self.data.items():
            summary[name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict()
            }
        
        return summary
    
    def get_wells_info(self) -> Tuple[list, list]:
        """获取油井和水井列表"""
        if not self.data:
            self.load_all_data()
        
        oil_wells = self.data['daily_oil_train'][WELL_COL].unique().tolist()
        water_wells = self.data['daily_water'][WELL_COL].unique().tolist()
        
        logger.info(f"油井数量: {len(oil_wells)}")
        logger.info(f"水井数量: {len(water_wells)}")
        
        return oil_wells, water_wells
    
    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """获取数据的日期范围"""
        if not self.data:
            self.load_all_data()
        
        # 从训练数据获取日期范围
        train_dates = self.data['daily_oil_train'][TIME_COL].dropna()
        val_dates = self.data['validation'][TIME_COL].dropna()
        
        min_date = min(train_dates.min(), val_dates.min())
        max_date = max(train_dates.max(), val_dates.max())
        
        logger.info(f"数据日期范围: {min_date} 到 {max_date}")
        return min_date, max_date

def main():
    """测试数据加载功能"""
    loader = DataLoader()
    data = loader.load_all_data()
    
    # 打印数据摘要
    summary = loader.get_data_summary()
    for name, info in summary.items():
        print(f"\n{name}:")
        print(f"  形状: {info['shape']}")
        print(f"  列名: {info['columns']}")
        
    # 获取井信息
    oil_wells, water_wells = loader.get_wells_info()
    
    # 获取日期范围
    min_date, max_date = loader.get_date_range()

if __name__ == "__main__":
    main()