"""
中国石化岩性识别AI竞赛 - 预测模块
包含模型加载、预测和提交文件生成功能
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_PATHS, MODEL_PATHS, OUTPUT_PATHS,
    TARGET_COLUMN, SUBMISSION_TARGET_COLUMN, ID_COLUMN, FEATURE_COLUMNS,
    LITHOLOGY_MAPPING, RANDOM_STATE
)
from utils import (
    load_data, load_model, save_predictions, 
    create_submission_file, set_random_seed
)
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from base_models import BaseModelTrainer
from deep_models import DeepModelTrainer
from ensemble import EnsembleManager

# 设置随机种子
set_random_seed(RANDOM_STATE)

class PredictionPipeline:
    """预测流水线"""
    
    def __init__(self, model_dir=None, output_dir=None):
        self.model_dir = model_dir or MODEL_PATHS['save_dir']
        self.output_dir = output_dir or OUTPUT_PATHS['predictions']
        self.preprocessor = None
        self.feature_engineer = None
        self.models = {}
        self.ensemble_manager = None
        self.prediction_results = {}
        
    def load_preprocessors(self, preprocessor_path=None, feature_engineer_path=None):
        """加载预处理器和特征工程器"""
        print("加载预处理器和特征工程器...")
        
        if preprocessor_path is None:
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
        if feature_engineer_path is None:
            feature_engineer_path = os.path.join(self.model_dir, 'feature_engineer.pkl')
        
        try:
            import joblib
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"预处理器已从 {preprocessor_path} 加载")
        except Exception as e:
            print(f"加载预处理器失败: {e}")
            print("将创建新的预处理器")
            self.preprocessor = DataPreprocessor()
        
        try:
            import joblib
            self.feature_engineer = joblib.load(feature_engineer_path)
            print(f"特征工程器已从 {feature_engineer_path} 加载")
        except Exception as e:
            print(f"加载特征工程器失败: {e}")
            print("将创建新的特征工程器")
            self.feature_engineer = FeatureEngineer()
    
    def load_models(self, model_names=None):
        """加载训练好的模型"""
        print("加载训练好的模型...")
        
        if model_names is None:
            # 自动检测可用模型
            model_names = []
            for file in os.listdir(self.model_dir):
                if file.endswith('.pkl') and not file.startswith('preprocessor') and not file.startswith('feature_engineer'):
                    model_name = file.replace('.pkl', '')
                    model_names.append(model_name)
        
        for model_name in model_names:
            model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
            
            try:
                if 'ensemble' in model_name:
                    # 加载集成模型
                    if self.ensemble_manager is None:
                        self.ensemble_manager = EnsembleManager()
                    
                    ensemble_type = model_name.replace('ensemble_', '')
                    import joblib
                    ensemble = joblib.load(model_path)
                    self.ensemble_manager.ensembles[ensemble_type] = ensemble
                    print(f"集成模型 {ensemble_type} 已加载")
                
                else:
                    # 加载单个模型
                    import joblib
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    print(f"模型 {model_name} 已加载")
            
            except Exception as e:
                print(f"加载模型 {model_name} 失败: {e}")
        
        print(f"成功加载 {len(self.models)} 个单模型和 {len(self.ensemble_manager.ensembles) if self.ensemble_manager else 0} 个集成模型")
    
    def preprocess_test_data(self, test_data):
        """预处理测试数据"""
        print("预处理测试数据...")
        
        # 数据预处理
        if self.preprocessor is None:
            raise ValueError("预处理器未加载")
        
        processed_data = self.preprocessor.preprocess_pipeline(test_data, is_training=False)
        
        # 特征工程
        if self.feature_engineer is None:
            raise ValueError("特征工程器未加载")
        
        featured_data = self.feature_engineer.feature_engineering_pipeline(processed_data, is_training=False)
        
        print(f"预处理后数据形状: {featured_data.shape}")
        
        return featured_data
    
    def predict_with_single_models(self, X):
        """使用单个模型进行预测"""
        print("使用单个模型进行预测...")
        
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            print(f"使用 {model_name} 进行预测...")
            
            try:
                # 预测
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                elif hasattr(model, 'model') and hasattr(model.model, 'predict'):
                    pred = model.model.predict(X)
                else:
                    print(f"模型 {model_name} 没有predict方法")
                    continue
                
                predictions[model_name] = pred
                
                # 预测概率
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                    elif hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
                        proba = model.model.predict_proba(X)
                    else:
                        proba = None
                    
                    if proba is not None:
                        probabilities[model_name] = proba
                
                except Exception as e:
                    print(f"模型 {model_name} 概率预测失败: {e}")
            
            except Exception as e:
                print(f"模型 {model_name} 预测失败: {e}")
        
        return predictions, probabilities
    
    def predict_with_ensembles(self, X):
        """使用集成模型进行预测"""
        print("使用集成模型进行预测...")
        
        if self.ensemble_manager is None or not self.ensemble_manager.ensembles:
            print("没有可用的集成模型")
            return {}, {}
        
        predictions = {}
        probabilities = {}
        
        for ensemble_name, ensemble in self.ensemble_manager.ensembles.items():
            print(f"使用 {ensemble_name} 集成进行预测...")
            
            try:
                # 预测
                pred = ensemble.predict(X)
                predictions[f'ensemble_{ensemble_name}'] = pred
                
                # 预测概率
                try:
                    if hasattr(ensemble, 'predict_proba'):
                        proba = ensemble.predict_proba(X)
                        probabilities[f'ensemble_{ensemble_name}'] = proba
                except Exception as e:
                    print(f"集成 {ensemble_name} 概率预测失败: {e}")
            
            except Exception as e:
                print(f"集成 {ensemble_name} 预测失败: {e}")
        
        return predictions, probabilities
    
    def create_ensemble_prediction(self, predictions, method='voting'):
        """创建集成预测"""
        print(f"创建集成预测 (方法: {method})...")
        
        if not predictions:
            return None
        
        # 确保所有预测都是相同形状的1D数组
        pred_list = []
        for model_name, pred in predictions.items():
            pred = np.array(pred).flatten()  # 确保是1D数组
            pred_list.append(pred)
            print(f"模型 {model_name} 预测形状: {pred.shape}")
        
        # 检查所有预测是否有相同的长度
        lengths = [len(pred) for pred in pred_list]
        if len(set(lengths)) > 1:
            print(f"警告: 预测长度不一致: {lengths}")
            min_length = min(lengths)
            pred_list = [pred[:min_length] for pred in pred_list]
            print(f"截断到最小长度: {min_length}")
        
        pred_array = np.array(pred_list)
        print(f"集成预测数组形状: {pred_array.shape}")
        
        if method == 'voting':
            # 多数投票
            from scipy import stats
            ensemble_pred = stats.mode(pred_array, axis=0)[0].flatten()
        elif method == 'average':
            # 平均（适用于数值预测）
            ensemble_pred = np.mean(pred_array, axis=0)
        else:
            raise ValueError(f"不支持的集成方法: {method}")
        
        return ensemble_pred
    
    def predict(self, test_data, use_ensembles=True, create_ensemble=True):
        """完整预测流程"""
        print("开始预测流程...")
        
        # 预处理测试数据
        featured_data = self.preprocess_test_data(test_data)
        
        # 准备特征
        feature_cols = [col for col in featured_data.columns 
                       if col not in [TARGET_COLUMN, 'WELL', 'DEPTH', ID_COLUMN]]
        
        X = featured_data[feature_cols]
        print(f"特征数量: {len(feature_cols)}")
        
        # 单模型预测
        single_predictions, single_probabilities = self.predict_with_single_models(X)
        
        # 集成模型预测
        ensemble_predictions, ensemble_probabilities = {}, {}
        if use_ensembles:
            ensemble_predictions, ensemble_probabilities = self.predict_with_ensembles(X)
        
        # 合并所有预测
        all_predictions = {**single_predictions, **ensemble_predictions}
        all_probabilities = {**single_probabilities, **ensemble_probabilities}
        
        # 创建最终集成预测
        final_prediction = None
        if create_ensemble and len(all_predictions) > 1:
            final_prediction = self.create_ensemble_prediction(all_predictions)
        
        # 保存结果
        self.prediction_results = {
            'individual_predictions': all_predictions,
            'individual_probabilities': all_probabilities,
            'final_prediction': final_prediction,
            'test_ids': featured_data[ID_COLUMN] if ID_COLUMN in featured_data.columns else None
        }
        
        print(f"预测完成! 共使用 {len(all_predictions)} 个模型")
        
        return self.prediction_results
    

    
    def create_submission_files(self, filename_prefix='submission'):
        """创建提交文件（可直接用于竞赛提交）"""
        print("保存预测结果为提交文件...")
        
        if not self.prediction_results:
            print("没有预测结果可创建提交文件")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_files = []
        
        # 为每个模型创建提交文件
        for model_name, predictions in self.prediction_results['individual_predictions'].items():
            filename = f"{filename_prefix}_{model_name}_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # 确保预测是1D数组
            predictions = np.array(predictions).flatten()
            
            # 创建提交格式
            if self.prediction_results['test_ids'] is not None:
                test_ids = np.array(self.prediction_results['test_ids']).flatten()
                submission_df = pd.DataFrame({
                    ID_COLUMN: test_ids,
                    SUBMISSION_TARGET_COLUMN: predictions
                })
            else:
                # 如果没有test_ids，创建序号
                submission_df = pd.DataFrame({
                    ID_COLUMN: range(len(predictions)),
                    SUBMISSION_TARGET_COLUMN: predictions
                })
            
            # 保持数字标签格式，不进行映射（符合sample_submission.csv格式）
            # if LITHOLOGY_MAPPING:
            #     # 直接使用LITHOLOGY_MAPPING进行映射，将数字标签映射为文字标签
            #     submission_df[SUBMISSION_TARGET_COLUMN] = submission_df[SUBMISSION_TARGET_COLUMN].map(LITHOLOGY_MAPPING)
            
            submission_df.to_csv(filepath, index=False)
            submission_files.append(filepath)
            print(f"提交文件已创建: {filepath}")
        
        # 创建最终集成提交文件
        if self.prediction_results['final_prediction'] is not None:
            filename = f"{filename_prefix}_final_ensemble_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # 确保预测是1D数组
            final_predictions = np.array(self.prediction_results['final_prediction']).flatten()
            
            if self.prediction_results['test_ids'] is not None:
                test_ids = np.array(self.prediction_results['test_ids']).flatten()
                submission_df = pd.DataFrame({
                    ID_COLUMN: test_ids,
                    SUBMISSION_TARGET_COLUMN: final_predictions
                })
            else:
                # 如果没有test_ids，创建序号
                submission_df = pd.DataFrame({
                    ID_COLUMN: range(len(final_predictions)),
                    SUBMISSION_TARGET_COLUMN: final_predictions
                })
            
            # 保持数字标签格式，不进行映射（符合sample_submission.csv格式）
            # if LITHOLOGY_MAPPING:
            #     # 直接使用LITHOLOGY_MAPPING进行映射，将数字标签映射为文字标签
            #     submission_df[SUBMISSION_TARGET_COLUMN] = submission_df[SUBMISSION_TARGET_COLUMN].map(LITHOLOGY_MAPPING)
            
            submission_df.to_csv(filepath, index=False)
            submission_files.append(filepath)
            print(f"最终提交文件已创建: {filepath}")
        
        return submission_files
    
    def generate_prediction_report(self):
        """生成预测报告"""
        print("生成预测报告...")
        
        if not self.prediction_results:
            print("没有预测结果可生成报告")
            return
        
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'models_used': list(self.prediction_results['individual_predictions'].keys()),
            'num_predictions': len(self.prediction_results['test_ids']) if self.prediction_results['test_ids'] is not None else 0,
            'prediction_statistics': {}
        }
        
        # 统计每个模型的预测分布
        for model_name, predictions in self.prediction_results['individual_predictions'].items():
            unique, counts = np.unique(predictions, return_counts=True)
            # 转换为Python原生类型以便JSON序列化
            distribution = {int(k): int(v) for k, v in zip(unique, counts)}
            
            report['prediction_statistics'][model_name] = {
                'unique_predictions': int(len(unique)),
                'distribution': distribution,
                'most_common': int(unique[np.argmax(counts)])
            }
        
        # 最终预测统计
        if self.prediction_results['final_prediction'] is not None:
            final_pred = self.prediction_results['final_prediction']
            unique, counts = np.unique(final_pred, return_counts=True)
            # 转换为Python原生类型以便JSON序列化
            distribution = {int(k): int(v) for k, v in zip(unique, counts)}
            
            report['final_prediction_statistics'] = {
                'unique_predictions': int(len(unique)),
                'distribution': distribution,
                'most_common': int(unique[np.argmax(counts)])
            }
        
        # 保存报告
        report_filename = f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_filepath = os.path.join(self.output_dir, report_filename)
        
        import json
        with open(report_filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"预测报告已保存到: {report_filepath}")
        
        # 打印简要报告
        print("\n=== 预测报告摘要 ===")
        print(f"使用模型数量: {len(report['models_used'])}")
        print(f"预测样本数量: {report['num_predictions']}")
        print(f"使用的模型: {', '.join(report['models_used'])}")
        
        if 'final_prediction_statistics' in report:
            print(f"最终预测类别分布: {report['final_prediction_statistics']['distribution']}")
        
        return report

def predict_test_data(model_dir=None, test_data_path=None, output_dir=None,
                     model_names=None, use_ensembles=True):
    """
    预测测试数据的便捷函数
    
    Args:
        model_dir: 模型目录
        test_data_path: 测试数据路径
        output_dir: 输出目录
        model_names: 要使用的模型名称列表
        use_ensembles: 是否使用集成模型
        
    Returns:
        预测结果
    """
    print("开始测试数据预测...")
    
    # 创建预测流水线
    pipeline = PredictionPipeline(model_dir, output_dir)
    
    # 加载预处理器和模型
    pipeline.load_preprocessors()
    pipeline.load_models(model_names)
    
    # 加载测试数据
    if test_data_path is None:
        test_data = load_data('test')
    else:
        test_data = pd.read_csv(test_data_path)
    
    print(f"测试数据形状: {test_data.shape}")
    
    # 进行预测
    results = pipeline.predict(test_data, use_ensembles=use_ensembles)
    
    # 保存结果
    submission_files = pipeline.create_submission_files()
    report = pipeline.generate_prediction_report()
    
    print("预测完成!")
    
    return {
        'predictions': results,
        'submission_files': submission_files,
        'report': report
    }

if __name__ == "__main__":
    # 测试代码
    print("测试预测模块...")
    
    # 创建预测流水线
    pipeline = PredictionPipeline()
    
    # 加载测试数据
    try:
        test_data = load_data('test')
        print(f"测试数据形状: {test_data.shape}")
        
        # 如果没有训练好的模型，创建简单的测试
        if not os.path.exists(pipeline.model_dir):
            print("没有找到训练好的模型，创建测试预测...")
            
            # 创建简单的预测结果用于测试
            test_ids = test_data[ID_COLUMN] if ID_COLUMN in test_data.columns else range(len(test_data))
            dummy_predictions = np.random.choice([0, 1, 2], size=len(test_data))
            
            pipeline.prediction_results = {
                'individual_predictions': {'dummy_model': dummy_predictions},
                'individual_probabilities': {},
                'final_prediction': dummy_predictions,
                'test_ids': test_ids
            }
            
            # 保存测试结果
            pipeline.create_submission_files()
            pipeline.generate_prediction_report()
        
        else:
            # 使用真实模型进行预测
            pipeline.load_preprocessors()
            pipeline.load_models()
            results = pipeline.predict(test_data)
            pipeline.create_submission_files()
            pipeline.generate_prediction_report()
    
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("这是正常的，因为可能没有测试数据或训练好的模型")
    
    print("预测模块测试完成!")