"""
统一预测脚本 - 中国石化AI竞赛油井含水率预测
整合了标准预测和智能基线预测功能
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_PATHS, TARGET_COL, TIME_COL, WELL_COL, ID_COL
from data_loader import DataLoader

import joblib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_models_and_feature_engineer():
    """加载训练好的模型和特征工程器"""
    logger.info("加载模型和特征工程器...")
    
    # 加载特征工程器
    feature_engineer_path = OUTPUT_PATHS['models'] / 'feature_engineer.pkl'
    feature_engineer = None
    if feature_engineer_path.exists():
        try:
            feature_engineer = joblib.load(feature_engineer_path)
            logger.info("特征工程器加载完成")
        except Exception as e:
            logger.warning(f"特征工程器加载失败: {e}")
    else:
        logger.warning(f"特征工程器文件不存在: {feature_engineer_path}")
    
    # 加载模型
    models = {}
    model_files = {
        'lightgbm': 'lightgbm_model.pkl',
        'xgboost': 'xgboost_model.pkl',
        'ensemble': 'ensemble_model.pkl'
    }
    
    for name, filename in model_files.items():
        model_path = OUTPUT_PATHS['models'] / filename
        if model_path.exists():
            try:
                models[name] = joblib.load(model_path)
                logger.info(f"{name} 模型加载完成")
            except Exception as e:
                logger.warning(f"{name} 模型加载失败: {e}")
        else:
            logger.warning(f"{name} 模型文件不存在: {model_path}")
    
    return models, feature_engineer

def prepare_prediction_data_standard(data_loader, feature_engineer):
    """使用标准特征工程准备预测数据"""
    logger.info("使用标准特征工程准备预测数据...")
    val_df = data_loader.data['validation'].copy()
    val_features = feature_engineer.create_all_features(val_df, is_training=False)
    val_features = feature_engineer.transform_select_features(val_features)
    return val_features, val_df

def load_models(model_names: list, models_dir: Path) -> dict:
    """加载指定名称的模型"""
    models = {}
    for name in model_names:
        model_path = models_dir / f'{name}_model.pkl'
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    models[name] = joblib.load(f)
                logger.info(f"✅ 模型 '{name}' 加载成功")
            except Exception as e:
                logger.error(f"❌ 模型 '{name}' 加载失败: {e}")
        else:
            logger.warning(f"⚠️ 模型 '{name}' 文件未找到: {model_path}")
    return models

def make_predictions(models: dict, val_df: pd.DataFrame, val_features: pd.DataFrame, model_type: str):
    """使用指定模型进行预测"""
    logger.info(f"使用模型 '{model_type}' 进行预测...")
    
    model = models.get(model_type)
    if model is None:
        logger.error(f"模型 '{model_type}' 不可用")
        return None

    # 准备特征
    feature_cols = model.get_feature_names()
    X_val = val_features[feature_cols]
    
    predictions = model.predict(X_val)
    predictions = np.clip(predictions, 0, 100) # 预测值裁剪
    
    logger.info("预测完成")
    return predictions

def create_submission_file(val_df: pd.DataFrame, predictions: np.ndarray, output_path: Path):
    """创建提交文件"""
    logger.info(f"创建提交文件到: {output_path}")
    submission_df = val_df[[ID_COL]].copy()
    submission_df[TARGET_COL] = predictions
    submission_df.to_csv(output_path, index=False)
    logger.info("提交文件创建成功")
    return submission_df

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成油井含水率预测结果')
    parser.add_argument('--model', type=str, default='lightgbm', 
                       choices=['lightgbm', 'xgboost', 'ensemble'],
                       help='使用的模型类型')
    parser.add_argument('--output', type=str, 
                       default=str(OUTPUT_PATHS['submissions'] / 'submission.csv'),
                       help='输出文件路径')
    parser.add_argument('--mode', type=str, default='standard', 
                       choices=['standard'],
                       help='预测模式')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("开始统一预测流程...")
    logger.info(f"预测模式: {args.mode}")
    logger.info(f"使用模型: {args.model}")
    logger.info(f"输出路径: {args.output}")
    logger.info("="*60)

    try:
        # 加载数据
        data_loader = DataLoader()
        data_loader.load_all_data()
        logger.info(f"验证数据形状: {data_loader.data['validation'].shape}")

        # 加载特征工程器
        feature_engineer_path = OUTPUT_PATHS['models'] / 'feature_engineer.pkl'
        if not feature_engineer_path.exists():
            logger.error(f"未找到特征工程器: {feature_engineer_path}")
            return
        feature_engineer = joblib.load(feature_engineer_path)
        logger.info("✅ 特征工程器加载成功")

        # 准备预测数据
        val_features, val_df = prepare_prediction_data_standard(data_loader, feature_engineer)

        # 加载模型
        model_names = ['lightgbm', 'xgboost', 'ensemble']
        models = load_models(model_names, OUTPUT_PATHS['models'])
        if not models:
            logger.error("没有加载任何模型，无法继续预测。")
            return

        # 进行预测
        predictions = make_predictions(models, val_df, val_features, args.model)

        # 创建提交文件
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission_df = create_submission_file(val_df, predictions, output_path)

        logger.info("="*60)
        logger.info("🎯 预测流程完成！")
        logger.info(f"📁 最终提交文件: {output_path.absolute()}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"预测过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()