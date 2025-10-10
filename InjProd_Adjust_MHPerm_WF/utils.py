"""
工具函数模块 - 中国石化AI竞赛油井含水率预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_PATHS, TARGET_COL, TIME_COL, WELL_COL

logger = logging.getLogger(__name__)

def setup_plotting():
    """设置绘图样式"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (12, 8)

def plot_target_distribution(df: pd.DataFrame, save_path: Optional[Path] = None):
    """绘制目标变量分布"""
    setup_plotting()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 直方图
    axes[0, 0].hist(df[TARGET_COL].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('含水率分布')
    axes[0, 0].set_xlabel('含水率 (%)')
    axes[0, 0].set_ylabel('频次')
    
    # 箱线图
    axes[0, 1].boxplot(df[TARGET_COL].dropna())
    axes[0, 1].set_title('含水率箱线图')
    axes[0, 1].set_ylabel('含水率 (%)')
    
    # 时间序列图
    daily_avg = df.groupby(TIME_COL)[TARGET_COL].mean()
    axes[1, 0].plot(daily_avg.index, daily_avg.values)
    axes[1, 0].set_title('含水率时间趋势')
    axes[1, 0].set_xlabel('日期')
    axes[1, 0].set_ylabel('平均含水率 (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 按井统计
    well_avg = df.groupby(WELL_COL)[TARGET_COL].mean().sort_values()
    axes[1, 1].bar(range(len(well_avg)), well_avg.values)
    axes[1, 1].set_title('各井平均含水率')
    axes[1, 1].set_xlabel('井号（排序）')
    axes[1, 1].set_ylabel('平均含水率 (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"目标变量分布图已保存到: {save_path}")
    
    plt.show()

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, 
                          save_path: Optional[Path] = None):
    """绘制特征重要性"""
    setup_plotting()
    
    # 取前N个重要特征
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, y='feature', x='importance', orient='h')
    plt.title(f'Top {top_n} 特征重要性')
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"特征重要性图已保存到: {save_path}")
    
    plt.show()

def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = 'Model', 
                             save_path: Optional[Path] = None):
    """绘制预测值vs实际值散点图"""
    setup_plotting()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 散点图
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('实际值')
    axes[0].set_ylabel('预测值')
    axes[0].set_title(f'{model_name} - 预测值 vs 实际值')
    
    # 残差图
    residuals = y_pred - y_true
    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('预测值')
    axes[1].set_ylabel('残差')
    axes[1].set_title(f'{model_name} - 残差图')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"预测结果图已保存到: {save_path}")
    
    plt.show()

def plot_time_series_prediction(df: pd.DataFrame, y_true: np.ndarray, 
                               y_pred: np.ndarray, well_name: str = None,
                               save_path: Optional[Path] = None):
    """绘制时间序列预测结果"""
    setup_plotting()
    
    plt.figure(figsize=(15, 6))
    
    # 如果指定了井名，只显示该井的数据
    if well_name:
        well_mask = df[WELL_COL] == well_name
        dates = df[well_mask][TIME_COL]
        true_vals = y_true[well_mask]
        pred_vals = y_pred[well_mask]
        title = f'{well_name} - 含水率预测'
    else:
        # 显示所有数据的日均值
        df_plot = df.copy()
        df_plot['y_true'] = y_true
        df_plot['y_pred'] = y_pred
        daily_true = df_plot.groupby(TIME_COL)['y_true'].mean()
        daily_pred = df_plot.groupby(TIME_COL)['y_pred'].mean()
        dates = daily_true.index
        true_vals = daily_true.values
        pred_vals = daily_pred.values
        title = '全部井日均含水率预测'
    
    plt.plot(dates, true_vals, label='实际值', linewidth=2)
    plt.plot(dates, pred_vals, label='预测值', linewidth=2)
    plt.xlabel('日期')
    plt.ylabel('含水率 (%)')
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"时间序列预测图已保存到: {save_path}")
    
    plt.show()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }
    
    return metrics

def print_metrics(metrics: Dict[str, float], model_name: str = 'Model'):
    """打印评估指标"""
    logger.info(f"\n{model_name} 评估指标:")
    logger.info("-" * 30)
    for metric, value in metrics.items():
        logger.info(f"{metric:>6}: {value:>8.4f}")
    logger.info("-" * 30)

def analyze_feature_correlation(df: pd.DataFrame, target_col: str = TARGET_COL, 
                               top_n: int = 20, save_path: Optional[Path] = None):
    """分析特征与目标变量的相关性"""
    setup_plotting()
    
    # 计算相关性
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
    
    # 排除目标变量本身
    correlations = correlations.drop(target_col, errors='ignore')
    
    # 取前N个相关特征
    top_correlations = correlations.head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_correlations.values, y=top_correlations.index, orient='h')
    plt.title(f'Top {top_n} 特征与{target_col}的相关性')
    plt.xlabel('绝对相关系数')
    plt.ylabel('特征')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"特征相关性图已保存到: {save_path}")
    
    plt.show()
    
    return top_correlations

def create_data_summary_report(df: pd.DataFrame, save_path: Optional[Path] = None):
    """创建数据摘要报告"""
    logger.info("生成数据摘要报告...")
    
    report = []
    report.append("# 数据摘要报告\n")
    
    # 基本信息
    report.append("## 基本信息")
    report.append(f"- 数据形状: {df.shape}")
    report.append(f"- 时间范围: {df[TIME_COL].min()} 到 {df[TIME_COL].max()}")
    report.append(f"- 井数量: {df[WELL_COL].nunique()}")
    report.append(f"- 总记录数: {len(df)}\n")
    
    # 缺失值统计
    report.append("## 缺失值统计")
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
    
    if len(missing_stats) > 0:
        for col, count in missing_stats.items():
            pct = count / len(df) * 100
            report.append(f"- {col}: {count} ({pct:.2f}%)")
    else:
        report.append("- 无缺失值")
    report.append("")
    
    # 数值特征统计
    report.append("## 数值特征统计")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    desc_stats = df[numeric_cols].describe()
    
    for col in numeric_cols[:10]:  # 只显示前10个数值特征
        report.append(f"### {col}")
        stats = desc_stats[col]
        report.append(f"- 均值: {stats['mean']:.4f}")
        report.append(f"- 标准差: {stats['std']:.4f}")
        report.append(f"- 最小值: {stats['min']:.4f}")
        report.append(f"- 最大值: {stats['max']:.4f}")
        report.append("")
    
    # 目标变量分析
    if TARGET_COL in df.columns:
        report.append("## 目标变量分析")
        target_stats = df[TARGET_COL].describe()
        report.append(f"- 均值: {target_stats['mean']:.4f}")
        report.append(f"- 标准差: {target_stats['std']:.4f}")
        report.append(f"- 最小值: {target_stats['min']:.4f}")
        report.append(f"- 最大值: {target_stats['max']:.4f}")
        report.append(f"- 25%分位数: {target_stats['25%']:.4f}")
        report.append(f"- 50%分位数: {target_stats['50%']:.4f}")
        report.append(f"- 75%分位数: {target_stats['75%']:.4f}")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"数据摘要报告已保存到: {save_path}")
    
    return report_text

def validate_submission_format(submission_df: pd.DataFrame, 
                              sample_submission_path: Path) -> bool:
    """验证提交文件格式"""
    logger.info("验证提交文件格式...")
    
    try:
        # 加载样例提交文件
        sample_df = pd.read_csv(sample_submission_path)
        
        # 检查列名
        if list(submission_df.columns) != list(sample_df.columns):
            logger.error(f"列名不匹配。期望: {list(sample_df.columns)}, 实际: {list(submission_df.columns)}")
            return False
        
        # 检查行数
        if len(submission_df) != len(sample_df):
            logger.error(f"行数不匹配。期望: {len(sample_df)}, 实际: {len(submission_df)}")
            return False
        
        # 检查ID列
        if not submission_df['id'].equals(sample_df['id']):
            logger.error("ID列不匹配")
            return False
        
        # 检查预测值范围
        if submission_df['predict'].min() < 0 or submission_df['predict'].max() > 100:
            logger.warning(f"预测值超出合理范围 [0, 100]: {submission_df['predict'].min()} - {submission_df['predict'].max()}")
        
        logger.info("提交文件格式验证通过")
        return True
        
    except Exception as e:
        logger.error(f"验证提交文件格式时出错: {str(e)}")
        return False

def main():
    """测试工具函数"""
    # 创建一些测试数据
    np.random.seed(42)
    n_samples = 1000
    
    test_df = pd.DataFrame({
        TIME_COL: pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        WELL_COL: np.random.choice(['井1', '井2', '井3'], n_samples),
        TARGET_COL: np.random.normal(50, 20, n_samples),
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples)
    })
    
    # 测试绘图函数
    plot_target_distribution(test_df)
    
    # 测试指标计算
    y_true = test_df[TARGET_COL].values
    y_pred = y_true + np.random.normal(0, 5, len(y_true))
    
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, 'Test Model')
    
    print("工具函数测试完成")

if __name__ == "__main__":
    main()