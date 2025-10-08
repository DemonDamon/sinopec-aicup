"""
中国石化岩性识别AI竞赛 - 深度学习模型模块
包含1D-CNN、GRU、LSTM等深度学习模型
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from config import (
    CNN_CONFIG, RNN_CONFIG, DEEP_LEARNING_CONFIG,
    TARGET_COLUMN, FEATURE_COLUMNS, RANDOM_STATE
)
from utils import save_model, load_model, evaluate_model, set_random_seed

# 设置随机种子
set_random_seed(RANDOM_STATE)

class WellLogDataset(Dataset):
    """测井数据集"""
    
    def __init__(self, X, y=None, sequence_length=50, stride=1):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequences = self._create_sequences()
        
    def _create_sequences(self):
        """创建序列数据"""
        sequences = []
        
        if isinstance(self.X, pd.DataFrame):
            X_values = self.X.values
        else:
            X_values = self.X
            
        if self.y is not None:
            if isinstance(self.y, pd.Series):
                y_values = self.y.values
            else:
                y_values = self.y
        
        # 创建滑动窗口序列
        for i in range(0, len(X_values) - self.sequence_length + 1, self.stride):
            seq_x = X_values[i:i + self.sequence_length]
            
            if self.y is not None:
                # 使用序列中间位置的标签
                seq_y = y_values[i + self.sequence_length // 2]
                sequences.append((seq_x, seq_y))
            else:
                sequences.append(seq_x)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.y is not None:
            seq_x, seq_y = self.sequences[idx]
            return torch.FloatTensor(seq_x), torch.LongTensor([seq_y])
        else:
            seq_x = self.sequences[idx]
            return torch.FloatTensor(seq_x)

class CNN1D(nn.Module):
    """1D卷积神经网络"""
    
    def __init__(self, input_dim, num_classes, config=None):
        super(CNN1D, self).__init__()
        
        if config is None:
            config = CNN_CONFIG
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 卷积层
        self.conv_layers = nn.ModuleList()
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(
            config['conv_filters'], config['kernel_sizes']
        )):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(config['dropout_rate'])
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc_layers = nn.ModuleList()
        in_features = config['conv_filters'][-1]
        
        for hidden_dim in config['hidden_dims']:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config['dropout_rate'])
            ))
            in_features = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        # 转换为 (batch_size, input_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # 卷积层
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # 全局池化
        x = self.global_pool(x)
        x = x.squeeze(-1)  # (batch_size, features)
        
        # 全连接层
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # 输出层
        x = self.output_layer(x)
        
        return x

class RNNModel(nn.Module):
    """RNN模型 (GRU/LSTM)"""
    
    def __init__(self, input_dim, num_classes, rnn_type='GRU', config=None):
        super(RNNModel, self).__init__()
        
        if config is None:
            config = RNN_CONFIG
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.rnn_type = rnn_type
        
        # RNN层
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                batch_first=True,
                dropout=config['dropout_rate'] if config['num_layers'] > 1 else 0,
                bidirectional=config['bidirectional']
            )
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                batch_first=True,
                dropout=config['dropout_rate'] if config['num_layers'] > 1 else 0,
                bidirectional=config['bidirectional']
            )
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")
        
        # 计算RNN输出维度
        rnn_output_dim = config['hidden_size']
        if config['bidirectional']:
            rnn_output_dim *= 2
        
        # 注意力机制
        if config.get('use_attention', False):
            self.attention = nn.MultiheadAttention(
                embed_dim=rnn_output_dim,
                num_heads=config.get('attention_heads', 8),
                dropout=config['dropout_rate'],
                batch_first=True
            )
        else:
            self.attention = None
        
        # 全连接层
        self.fc_layers = nn.ModuleList()
        in_features = rnn_output_dim
        
        for hidden_dim in config['hidden_dims']:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config['dropout_rate'])
            ))
            in_features = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # RNN层
        rnn_output, _ = self.rnn(x)
        # rnn_output shape: (batch_size, sequence_length, hidden_size * directions)
        
        # 注意力机制
        if self.attention is not None:
            attn_output, _ = self.attention(rnn_output, rnn_output, rnn_output)
            # 使用平均池化
            x = torch.mean(attn_output, dim=1)
        else:
            # 使用最后一个时间步的输出
            x = rnn_output[:, -1, :]
        
        # 全连接层
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # 输出层
        x = self.output_layer(x)
        
        return x

class DeepModelTrainer:
    """深度学习模型训练器"""
    
    def __init__(self, model_type='CNN1D', model_params=None, training_params=None):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.training_params = training_params or DEEP_LEARNING_CONFIG
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        
        print(f"使用设备: {self.device}")
    
    def _prepare_data(self, X, y=None, sequence_length=None, stride=1):
        """准备数据"""
        if sequence_length is None:
            sequence_length = self.training_params['sequence_length']
        
        dataset = WellLogDataset(X, y, sequence_length, stride)
        return dataset
    
    def _create_model(self, input_dim, num_classes):
        """创建模型"""
        if self.model_type == 'CNN1D':
            config = {**CNN_CONFIG, **self.model_params}
            model = CNN1D(input_dim, num_classes, config)
        elif self.model_type in ['GRU', 'LSTM']:
            config = {**RNN_CONFIG, **self.model_params}
            model = RNNModel(input_dim, num_classes, self.model_type, config)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        return model.to(self.device)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            verbose: 是否显示训练过程
            
        Returns:
            训练好的模型
        """
        print(f"开始训练{self.model_type}模型...")
        
        # 编码标签
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        num_classes = len(self.label_encoder.classes_)
        input_dim = X_train.shape[1]
        
        print(f"输入维度: {input_dim}")
        print(f"类别数量: {num_classes}")
        print(f"类别: {self.label_encoder.classes_}")
        
        # 创建模型
        self.model = self._create_model(input_dim, num_classes)
        
        # 准备数据集
        train_dataset = self._prepare_data(X_train, y_train_encoded)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_params['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            val_dataset = self._prepare_data(X_val, y_val_encoded)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_params['batch_size'],
                shuffle=False,
                num_workers=0
            )
        else:
            val_loader = None
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_params['learning_rate'],
            weight_decay=self.training_params['weight_decay']
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=verbose
        )
        
        # 训练循环
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(self.training_params['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).squeeze()
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # 验证阶段
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device).squeeze()
                        
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        
                        val_predictions.extend(predicted.cpu().numpy())
                        val_targets.extend(batch_y.cpu().numpy())
                
                val_loss /= len(val_loader)
                val_acc = accuracy_score(val_targets, val_predictions)
                val_f1 = f1_score(val_targets, val_predictions, average='macro')
                
                # 更新学习率
                scheduler.step(val_loss)
                
                # 保存训练历史
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_acc'].append(val_acc)
                self.training_history['val_f1'].append(val_f1)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.training_params['epochs']}: "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                
                # 早停
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'best_model_temp.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= self.training_params['early_stopping_patience']:
                    print(f"早停在第 {epoch+1} 轮")
                    break
            else:
                self.training_history['train_loss'].append(train_loss)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.training_params['epochs']}: "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 加载最佳模型
        if val_loader is not None:
            self.model.load_state_dict(torch.load('best_model_temp.pth'))
            import os
            os.remove('best_model_temp.pth')
        
        print(f"{self.model_type}模型训练完成!")
        
        return self.model
    
    def predict(self, X, batch_size=None):
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if batch_size is None:
            batch_size = self.training_params['batch_size']
        
        self.model.eval()
        dataset = self._prepare_data(X, stride=self.training_params['sequence_length'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        predictions = []
        
        with torch.no_grad():
            for batch_x in dataloader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        # 解码标签
        predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X, batch_size=None):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if batch_size is None:
            batch_size = self.training_params['batch_size']
        
        self.model.eval()
        dataset = self._prepare_data(X, stride=self.training_params['sequence_length'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        probabilities = []
        
        with torch.no_grad():
            for batch_x in dataloader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def save_model(self, filepath):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'model_params': self.model_params,
            'training_params': self.training_params,
            'label_encoder': self.label_encoder,
            'training_history': self.training_history,
            'model_architecture': str(self.model)
        }
        
        save_model(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath, input_dim, num_classes):
        """加载模型"""
        model_data = load_model(filepath)
        
        self.model_type = model_data['model_type']
        self.model_params = model_data['model_params']
        self.training_params = model_data['training_params']
        self.label_encoder = model_data['label_encoder']
        self.training_history = model_data['training_history']
        
        # 重新创建模型
        self.model = self._create_model(input_dim, num_classes)
        self.model.load_state_dict(model_data['model_state_dict'])
        
        print(f"模型已从{filepath}加载")
        
        return self.model

def train_deep_models(X_train, y_train, X_val=None, y_val=None,
                     model_types=['CNN1D', 'GRU', 'LSTM'],
                     custom_params=None):
    """
    训练多个深度学习模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        model_types: 模型类型列表
        custom_params: 自定义参数字典
        
    Returns:
        训练好的模型字典
    """
    print("开始训练深度学习模型...")
    
    if custom_params is None:
        custom_params = {}
    
    trained_models = {}
    
    for model_type in model_types:
        print(f"\n训练{model_type}模型...")
        
        # 获取模型参数
        model_params = custom_params.get(f'{model_type}_model', {})
        training_params = custom_params.get(f'{model_type}_training', {})
        
        # 创建训练器
        trainer = DeepModelTrainer(
            model_type=model_type,
            model_params=model_params,
            training_params=training_params
        )
        
        # 训练模型
        trainer.train(X_train, y_train, X_val, y_val)
        
        # 评估模型
        if X_val is not None and y_val is not None:
            y_pred = trainer.predict(X_val)
            y_pred_proba = trainer.predict_proba(X_val)
            
            eval_result = evaluate_model(y_val, y_pred, y_pred_proba, f"{model_type}")
            trainer.eval_result = eval_result
        
        trained_models[model_type] = trainer
    
    print("\n所有深度学习模型训练完成!")
    
    return trained_models

if __name__ == "__main__":
    # 测试代码
    from utils import load_data
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    print("测试深度学习模型模块...")
    
    # 加载和预处理数据
    train_data = load_data('train')
    
    # 数据预处理和特征工程
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(train_data, is_training=True)
    
    feature_engineer = FeatureEngineer()
    featured_data = feature_engineer.feature_engineering_pipeline(processed_data, is_training=True)
    
    # 准备训练数据
    feature_cols = [col for col in featured_data.columns 
                   if col not in [TARGET_COLUMN, 'WELL', 'DEPTH', 'id']]
    
    X = featured_data[feature_cols]
    y = featured_data[TARGET_COLUMN]
    
    # 简单分割数据
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    
    # 训练CNN模型
    trainer = DeepModelTrainer(model_type='CNN1D')
    
    # 使用较小的参数进行快速测试
    test_params = {
        'epochs': 5,
        'batch_size': 32,
        'sequence_length': 20
    }
    trainer.training_params.update(test_params)
    
    trainer.train(X_train, y_train, X_val, y_val)
    
    # 预测和评估
    y_pred = trainer.predict(X_val)
    y_pred_proba = trainer.predict_proba(X_val)
    
    eval_result = evaluate_model(y_val, y_pred, y_pred_proba, "CNN1D Test")
    
    print("深度学习模型模块测试完成!")