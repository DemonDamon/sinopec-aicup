"""
中国石化岩性识别AI竞赛 - 主程序入口
完整的训练和预测流程
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_PATHS, MODEL_PATHS, OUTPUT_PATHS,
    TARGET_COLUMN, ID_COLUMN, FEATURE_COLUMNS,
    LITHOLOGY_MAPPING, RANDOM_STATE, CV_CONFIG,
    MISSING_VALUE_CONFIG, SCALING_CONFIG
)
from utils import (
    setup_logging, load_data, set_random_seed,
    create_directory_structure, memory_usage_check,
    save_experiment_results
)
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from validation import CrossValidator
from base_models import BaseModelTrainer, train_base_models
from deep_models import DeepModelTrainer, train_deep_models
from ensemble import EnsembleManager, create_ensemble_pipeline
from predict import PredictionPipeline, predict_test_data

# 设置随机种子
set_random_seed(RANDOM_STATE)

class LithoClassificationPipeline:
    """岩性识别完整流水线"""
    
    def __init__(self, config_override=None):
        """
        初始化流水线
        
        Args:
            config_override: 配置覆盖字典
        """
        self.config = config_override or {}
        self.logger = setup_logging()
        
        # 创建目录结构
        create_directory_structure()
        
        # 初始化组件
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.cv = CrossValidator(
            cv_method=CV_CONFIG['method'],
            n_splits=CV_CONFIG['n_splits'],
            random_state=RANDOM_STATE
        )
        self.base_trainer = BaseModelTrainer()
        self.deep_trainer = DeepModelTrainer()
        self.ensemble_manager = EnsembleManager()
        
        # 存储结果
        self.results = {
            'preprocessing': {},
            'feature_engineering': {},
            'cross_validation': {},
            'base_models': {},
            'deep_models': {},
            'ensemble_models': {},
            'final_predictions': {}
        }
        
        self.logger.info("岩性识别流水线初始化完成")
    
    def load_and_explore_data(self):
        """加载和探索数据"""
        self.logger.info("开始加载和探索数据...")
        
        # 加载数据
        train_data = load_data('train')
        test_data = load_data('test')
        
        self.logger.info(f"训练数据形状: {train_data.shape}")
        self.logger.info(f"测试数据形状: {test_data.shape}")
        
        # 数据探索
        self.logger.info("训练数据基本信息:")
        self.logger.info(f"缺失值数量: {train_data.isnull().sum().sum()}")
        self.logger.info(f"目标变量分布: {train_data[TARGET_COLUMN].value_counts().to_dict()}")
        self.logger.info(f"井的数量: {train_data['WELL'].nunique()}")
        
        # 保存数据探索结果
        self.results['data_exploration'] = {
            'train_shape': train_data.shape,
            'test_shape': test_data.shape,
            'train_missing': train_data.isnull().sum().to_dict(),
            'target_distribution': train_data[TARGET_COLUMN].value_counts().to_dict(),
            'num_wells': train_data['WELL'].nunique(),
            'feature_columns': FEATURE_COLUMNS
        }
        
        return train_data, test_data
    
    def run_preprocessing(self, train_data, test_data):
        """运行数据预处理"""
        self.logger.info("开始数据预处理...")
        
        # 训练预处理器
        processed_train = self.preprocessor.preprocess_pipeline(train_data, is_training=True)
        processed_test = self.preprocessor.preprocess_pipeline(test_data, is_training=False)
        
        self.logger.info(f"预处理后训练数据形状: {processed_train.shape}")
        self.logger.info(f"预处理后测试数据形状: {processed_test.shape}")
        
        # 保存预处理器
        preprocessor_path = os.path.join(MODEL_PATHS['save_dir'], 'preprocessor.pkl')
        from utils import save_model
        save_model(self.preprocessor, preprocessor_path)
        self.logger.info(f"预处理器已保存到: {preprocessor_path}")
        
        # 保存预处理结果
        self.results['preprocessing'] = {
            'processed_train_shape': processed_train.shape,
            'processed_test_shape': processed_test.shape,
            'missing_strategy': MISSING_VALUE_CONFIG['method'],
            'scaling_method': SCALING_CONFIG['method']
        }
        
        return processed_train, processed_test
    
    def run_feature_engineering(self, processed_train, processed_test):
        """运行特征工程"""
        self.logger.info("开始特征工程...")
        
        # 训练特征工程器
        featured_train = self.feature_engineer.feature_engineering_pipeline(
            processed_train, is_training=True
        )
        featured_test = self.feature_engineer.feature_engineering_pipeline(
            processed_test, is_training=False
        )
        
        self.logger.info(f"特征工程后训练数据形状: {featured_train.shape}")
        self.logger.info(f"特征工程后测试数据形状: {featured_test.shape}")
        
        # 获取新特征名称
        new_features = self.feature_engineer.get_feature_names()
        self.logger.info(f"新增特征数量: {len(new_features)}")
        
        # 保存特征工程器
        feature_engineer_path = os.path.join(MODEL_PATHS['save_dir'], 'feature_engineer.pkl')
        from utils import save_model
        save_model(self.feature_engineer, feature_engineer_path)
        self.logger.info(f"特征工程器已保存到: {feature_engineer_path}")
        
        # 保存特征工程结果
        self.results['feature_engineering'] = {
            'featured_train_shape': featured_train.shape,
            'featured_test_shape': featured_test.shape,
            'new_features_count': len(new_features),
            'new_features': new_features[:50]  # 只保存前50个特征名
        }
        
        return featured_train, featured_test
    
    def run_cross_validation(self, featured_train):
        """运行交叉验证"""
        self.logger.info("开始交叉验证...")
        
        # 准备数据
        feature_cols = [col for col in featured_train.columns 
                       if col not in [TARGET_COLUMN, 'WELL', 'DEPTH', ID_COLUMN]]
        
        X = featured_train[feature_cols]
        y = featured_train[TARGET_COLUMN]
        groups = featured_train['WELL']
        
        self.logger.info(f"特征数量: {len(feature_cols)}")
        self.logger.info(f"样本数量: {len(X)}")
        
        # 创建交叉验证分割
        cv_splits_gen = self.cv.create_cv_splits(X, y, groups)
        cv_splits = list(cv_splits_gen)  # 转换为列表
        self.logger.info(f"交叉验证折数: {len(cv_splits)}")
        
        # 验证数据分割
        from validation import validate_data_split
        # 重建数据框用于验证
        validation_data = featured_train[[TARGET_COLUMN, 'WELL']].copy()
        split_info = validate_data_split(validation_data)
        self.logger.info("交叉验证分割验证完成")
        
        # 保存交叉验证结果
        self.results['cross_validation'] = {
            'cv_method': CV_CONFIG['method'],
            'n_splits': len(cv_splits),
            'feature_count': len(feature_cols),
            'sample_count': len(X),
            'split_validation': split_info
        }
        
        return X, y, groups, cv_splits, feature_cols
    
    def run_base_models(self, X, y, groups, cv_splits):
        """运行基础模型训练"""
        self.logger.info("开始基础模型训练...")
        
        # 定义要训练的模型
        model_configs = {
            'lightgbm': {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 8,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': RANDOM_STATE,
                'verbose': -1
            },
            'catboost': {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 8,
                'random_seed': RANDOM_STATE,
                'verbose': False
            },
            'xgboost': {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': RANDOM_STATE,
                'verbosity': 0
            }
        }
        
        # 训练基础模型 - 使用交叉验证
        base_results = {}
        for model_name, config in model_configs.items():
            self.logger.info(f"训练 {model_name} 模型...")
            
            # 创建模型训练器
            trainer = BaseModelTrainer(
                model_type=model_name,
                model_params=config,
                use_class_weights=True
            )
            
            # 交叉验证训练
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
                
                # 训练模型
                trainer.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                
                # 预测和评估
                y_pred = trainer.predict(X_val_fold)
                from sklearn.metrics import f1_score
                score = f1_score(y_val_fold, y_pred, average='macro')
                cv_scores.append(score)
                
                self.logger.info(f"  第{fold+1}折 F1分数: {score:.4f}")
            
            mean_score = np.mean(cv_scores)
            self.logger.info(f"{model_name} 平均F1分数: {mean_score:.4f}")
            
            # 保存模型
            model_path = os.path.join(MODEL_PATHS['save_dir'], f'{model_name}.pkl')
            from utils import save_model
            save_model(trainer, model_path)
            
            base_results[model_name] = {
                'model': trainer,
                'cv_scores': cv_scores,
                'mean_score': mean_score
            }
        
        self.logger.info(f"基础模型训练完成，共训练 {len(base_results)} 个模型")
        
        # 保存基础模型结果
        self.results['base_models'] = {
            'models_trained': list(base_results.keys()),
            'cv_scores': {name: result['cv_scores'] for name, result in base_results.items()},
            'mean_scores': {name: np.mean(result['cv_scores']) for name, result in base_results.items()}
        }
        
        return base_results
    
    def run_deep_models(self, X, y, groups, cv_splits):
        """运行深度学习模型训练"""
        self.logger.info("开始深度学习模型训练...")
        
        # 定义深度学习模型配置
        deep_configs = {
            'cnn1d': {
                'sequence_length': 50,
                'num_features': X.shape[1],
                'num_classes': len(np.unique(y)),
                'filters': [64, 128, 64],
                'kernel_sizes': [3, 3, 3],
                'dropout_rate': 0.3,
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'gru': {
                'sequence_length': 50,
                'num_features': X.shape[1],
                'num_classes': len(np.unique(y)),
                'hidden_size': 128,
                'num_layers': 2,
                'dropout_rate': 0.3,
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        # 训练深度学习模型
        try:
            deep_results = train_deep_models(
                X, y, groups, cv_splits,
                model_configs=deep_configs,
                save_models=True,
                model_save_dir=MODEL_PATHS['save_dir']
            )
            
            self.logger.info(f"深度学习模型训练完成，共训练 {len(deep_results)} 个模型")
            
            # 保存深度学习模型结果
            self.results['deep_models'] = {
                'models_trained': list(deep_results.keys()),
                'cv_scores': {name: result['cv_scores'] for name, result in deep_results.items()},
                'mean_scores': {name: np.mean(result['cv_scores']) for name, result in deep_results.items()}
            }
            
        except Exception as e:
            self.logger.warning(f"深度学习模型训练失败: {e}")
            deep_results = {}
            self.results['deep_models'] = {'error': str(e)}
        
        return deep_results
    
    def run_ensemble_models(self, X, y, groups, cv_splits, base_results):
        """运行集成模型训练"""
        self.logger.info("开始集成模型训练...")
        
        # 准备基础模型
        base_models = {}
        for model_name, result in base_results.items():
            if 'model' in result:
                base_models[model_name] = result['model']
        
        if len(base_models) < 2:
            self.logger.warning("基础模型数量不足，跳过集成学习")
            return {}
        
        # 集成配置
        ensemble_configs = {
            'stacking': {
                'base_models': base_models,
                'meta_learner': 'logistic_regression',
                'use_probabilities': True,
                'include_original_features': False
            },
            'voting': {
                'base_models': base_models,
                'voting_type': 'soft',
                'weights': None
            },
            'blending': {
                'base_models': base_models,
                'meta_learner': 'logistic_regression',
                'holdout_size': 0.2
            }
        }
        
        # 训练集成模型
        try:
            ensemble_results = create_ensemble_pipeline(
                X, y, groups, cv_splits,
                ensemble_configs=ensemble_configs,
                save_models=True,
                model_save_dir=MODEL_PATHS['save_dir']
            )
            
            self.logger.info(f"集成模型训练完成，共训练 {len(ensemble_results)} 个集成")
            
            # 保存集成模型结果
            self.results['ensemble_models'] = {
                'ensembles_trained': list(ensemble_results.keys()),
                'cv_scores': {name: result['cv_scores'] for name, result in ensemble_results.items()},
                'mean_scores': {name: np.mean(result['cv_scores']) for name, result in ensemble_results.items()}
            }
            
        except Exception as e:
            self.logger.warning(f"集成模型训练失败: {e}")
            ensemble_results = {}
            self.results['ensemble_models'] = {'error': str(e)}
        
        return ensemble_results
    
    def run_prediction(self, test_data):
        """运行预测"""
        self.logger.info("开始预测...")
        
        # 使用预测流水线
        prediction_results = predict_test_data(
            model_dir=MODEL_PATHS['save_dir'],
            test_data_path=None,  # 直接传入数据
            output_dir=OUTPUT_PATHS['predictions'],
            use_ensembles=True
        )
        
        self.logger.info("预测完成")
        
        # 保存预测结果
        self.results['final_predictions'] = {
            'submission_files': prediction_results['submission_files'],
            'report_summary': prediction_results['report']
        }
        
        return prediction_results
    
    def save_experiment_results(self):
        """保存实验结果"""
        self.logger.info("保存实验结果...")
        
        # 添加实验元信息
        self.results['experiment_info'] = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'random_state': RANDOM_STATE,
            'cv_config': CV_CONFIG,
            'feature_columns': FEATURE_COLUMNS,
            'target_column': TARGET_COLUMN
        }
        
        # 保存结果
        experiment_file = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        experiment_path = os.path.join(OUTPUT_PATHS['experiments'], experiment_file)
        
        save_experiment_results(self.results, experiment_path)
        self.logger.info(f"实验结果已保存到: {experiment_path}")
        
        return experiment_path
    
    def run_full_pipeline(self, skip_deep_learning=False, skip_ensemble=False):
        """运行完整流水线"""
        self.logger.info("开始运行完整的岩性识别流水线...")
        
        start_time = datetime.now()
        
        try:
            # 1. 加载和探索数据
            train_data, test_data = self.load_and_explore_data()
            
            # 2. 数据预处理
            processed_train, processed_test = self.run_preprocessing(train_data, test_data)
            
            # 3. 特征工程
            featured_train, featured_test = self.run_feature_engineering(processed_train, processed_test)
            
            # 4. 交叉验证
            X, y, groups, cv_splits, feature_cols = self.run_cross_validation(featured_train)
            
            # 5. 基础模型训练
            base_results = self.run_base_models(X, y, groups, cv_splits)
            
            # 6. 深度学习模型训练（可选）
            deep_results = {}
            if not skip_deep_learning:
                deep_results = self.run_deep_models(X, y, groups, cv_splits)
            
            # 7. 集成模型训练（可选）
            ensemble_results = {}
            if not skip_ensemble and base_results:
                ensemble_results = self.run_ensemble_models(X, y, groups, cv_splits, base_results)
            
            # 8. 预测
            prediction_results = self.run_prediction(test_data)
            
            # 9. 保存实验结果
            experiment_path = self.save_experiment_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info(f"完整流水线运行完成! 总耗时: {duration}")
            
            # 打印结果摘要
            self.print_results_summary()
            
            return {
                'success': True,
                'duration': duration,
                'experiment_path': experiment_path,
                'results': self.results
            }
        
        except Exception as e:
            self.logger.error(f"流水线运行失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': self.results
            }
    
    def print_results_summary(self):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("岩性识别流水线结果摘要")
        print("="*60)
        
        # 数据信息
        if 'data_exploration' in self.results:
            data_info = self.results['data_exploration']
            print(f"训练数据: {data_info['train_shape']}")
            print(f"测试数据: {data_info['test_shape']}")
            print(f"井数量: {data_info['num_wells']}")
        
        # 特征工程
        if 'feature_engineering' in self.results:
            fe_info = self.results['feature_engineering']
            print(f"最终特征数量: {fe_info['featured_train_shape'][1]}")
            print(f"新增特征数量: {fe_info['new_features_count']}")
        
        # 模型性能
        print("\n模型性能 (交叉验证平均分数):")
        
        if 'base_models' in self.results and 'mean_scores' in self.results['base_models']:
            print("基础模型:")
            for model, score in self.results['base_models']['mean_scores'].items():
                print(f"  {model}: {score:.4f}")
        
        if 'deep_models' in self.results and 'mean_scores' in self.results['deep_models']:
            print("深度学习模型:")
            for model, score in self.results['deep_models']['mean_scores'].items():
                print(f"  {model}: {score:.4f}")
        
        if 'ensemble_models' in self.results and 'mean_scores' in self.results['ensemble_models']:
            print("集成模型:")
            for model, score in self.results['ensemble_models']['mean_scores'].items():
                print(f"  {model}: {score:.4f}")
        
        # 预测文件
        if 'final_predictions' in self.results:
            pred_info = self.results['final_predictions']
            print(f"\n生成的提交文件数量: {len(pred_info['submission_files'])}")
        
        print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='中国石化岩性识别AI竞赛')
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], default='full',
                       help='运行模式: train(仅训练), predict(仅预测), full(完整流程)')
    parser.add_argument('--skip-deep', action='store_true',
                       help='跳过深度学习模型训练')
    parser.add_argument('--skip-ensemble', action='store_true',
                       help='跳过集成模型训练')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='模型保存/加载目录')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config_override = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config_override = json.load(f)
    
    # 创建流水线
    pipeline = LithoClassificationPipeline(config_override)
    
    # 检查内存使用
    memory_usage_check()
    
    if args.mode == 'full':
        # 运行完整流水线
        result = pipeline.run_full_pipeline(
            skip_deep_learning=args.skip_deep,
            skip_ensemble=args.skip_ensemble
        )
        
        if result['success']:
            print(f"\n流水线运行成功! 耗时: {result['duration']}")
            print(f"实验结果保存在: {result['experiment_path']}")
        else:
            print(f"\n流水线运行失败: {result['error']}")
            sys.exit(1)
    
    elif args.mode == 'train':
        # 仅训练模式
        print("仅训练模式暂未实现，请使用完整模式")
        sys.exit(1)
    
    elif args.mode == 'predict':
        # 仅预测模式
        print("开始预测模式...")
        
        prediction_results = predict_test_data(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            use_ensembles=True
        )
        
        print("预测完成!")
        print(f"提交文件: {prediction_results['submission_files']}")

if __name__ == "__main__":
    main()