# 中国石化岩性识别AI竞赛

## 项目简介

本项目是中国石化岩性识别AI竞赛的完整解决方案，基于测井数据进行岩性分类预测。项目采用机器学习和深度学习相结合的方法，实现了从数据预处理到模型集成的完整流水线。

## 项目结构

```
sinopec-aicup/
├── LithoClass_WellLog/          # 主要代码目录
│   ├── config.py                # 配置文件
│   ├── main.py                  # 主程序入口
│   ├── data_preprocessing.py    # 数据预处理模块
│   ├── feature_engineering.py   # 特征工程模块
│   ├── validation.py            # 交叉验证模块
│   ├── base_models.py           # 基础模型模块
│   ├── deep_models.py           # 深度学习模型模块
│   ├── ensemble.py              # 集成学习模块
│   ├── predict.py               # 预测模块
│   ├── utils.py                 # 工具函数
│   ├── models/                  # 模型保存目录
│   ├── output/                  # 输出结果目录
│   └── logs/                    # 日志目录
├── dataset/                     # 数据集目录
│   └── LithoClass_WellLog/      # 岩性识别数据
│       ├── train.csv            # 训练数据
│       ├── validation_without_label.csv  # 测试数据
│       └── sample_submission.csv # 提交样例
├── requirements.txt             # 依赖包列表
└── README.md                    # 项目说明
```

## 技术特点

### 数据预处理
- 缺失值处理（多种策略：均值、中位数、前向填充等）
- 异常值检测和处理（IQR方法、Z-score方法）
- 数据标准化（StandardScaler、MinMaxScaler、RobustScaler）
- 数据平衡处理（SMOTE、随机欠采样等）

### 特征工程
- **滑动窗口特征**：统计特征（均值、标准差、偏度、峰度等）
- **梯度特征**：一阶、二阶梯度特征
- **测井相特征**：基于聚类的测井相识别
- **频域特征**：FFT变换和小波变换特征
- **深度特征**：深度相关的统计特征
- **井特征**：井级别的统计特征
- **交互特征**：特征间的比值、乘积、差值等

### 模型算法
- **基础模型**：LightGBM、CatBoost、XGBoost、RandomForest、LogisticRegression、SVM
- **深度学习模型**：1D-CNN、GRU、LSTM
- **集成学习**：Stacking、Voting、Blending

### 验证策略
- **GroupKFold**：按井分组的交叉验证，避免数据泄露
- **StratifiedKFold**：分层交叉验证，保持类别分布
- **时间序列验证**：考虑深度顺序的验证方法

### 不平衡数据处理
- 类别权重调整
- SMOTE过采样
- 随机欠采样
- 集成方法中的平衡策略

## 快速开始

### 环境要求
- Python 3.8+
- 主要依赖包见 `requirements.txt`

### 安装依赖
```bash
pip install -r requirements.txt
```

### 数据准备
将竞赛数据放置在 `dataset/LithoClass_WellLog/` 目录下：
- `train.csv`：训练数据
- `validation_without_label.csv`：测试数据
- `sample_submission.csv`：提交样例

### 运行完整流水线
```bash
cd LithoClass_WellLog
python main.py --mode full
```

### 仅运行预测
```bash
cd LithoClass_WellLog
python main.py --mode predict --model-dir ./models
```

### 命令行参数
- `--mode`：运行模式（full/train/predict）
- `--skip-deep`：跳过深度学习模型训练
- `--skip-ensemble`：跳过集成模型训练
- `--model-dir`：模型保存/加载目录
- `--output-dir`：输出目录
- `--config`：配置文件路径

## 配置说明

主要配置在 `config.py` 中：

```python
# 数据路径配置
DATA_PATHS = {
    'train': '../dataset/LithoClass_WellLog/train.csv',
    'test': '../dataset/LithoClass_WellLog/validation_without_label.csv',
    'sample_submission': '../dataset/LithoClass_WellLog/sample_submission.csv'
}

# 交叉验证配置
CV_CONFIG = {
    'method': 'GroupKFold',  # GroupKFold按井分组
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}

# 特征列配置
FEATURE_COLUMNS = [
    'GR', 'SP', 'CALI', 'RDEP', 'RMED', 'RXO', 'RHOB', 
    'NPHI', 'PEF', 'DTC', 'DTS'
]
```

## 模型性能

项目实现了多种模型的训练和集成：

### 基础模型
- **LightGBM**：梯度提升决策树，速度快，效果好
- **CatBoost**：处理类别特征优秀，鲁棒性强
- **XGBoost**：经典梯度提升算法，性能稳定

### 深度学习模型
- **1D-CNN**：一维卷积神经网络，适合序列数据
- **GRU/LSTM**：循环神经网络，捕获时序依赖

### 集成策略
- **Stacking**：两层模型结构，元学习器学习基模型预测
- **Voting**：硬投票和软投票结合
- **Blending**：基于holdout验证的集成

## 输出结果

运行完成后，会在 `output/` 目录下生成：

- `predictions/`：各模型的预测结果
- `submissions/`：符合竞赛格式的提交文件
- `experiments/`：实验结果和日志
- `visualizations/`：可视化图表

## 项目亮点

1. **完整的机器学习流水线**：从数据预处理到模型部署的完整解决方案
2. **丰富的特征工程**：多种特征提取方法，提升模型性能
3. **多样化的模型算法**：传统机器学习与深度学习相结合
4. **严格的验证策略**：GroupKFold避免数据泄露，确保模型泛化能力
5. **高效的集成学习**：多种集成策略，提升预测精度
6. **完善的代码结构**：模块化设计，易于维护和扩展
7. **详细的日志记录**：完整的实验记录和结果追踪

## 注意事项

1. **数据泄露防护**：使用GroupKFold按井分组，确保同一口井的数据不会同时出现在训练集和验证集中
2. **内存管理**：大数据集训练时注意内存使用，可适当调整批次大小
3. **模型保存**：训练完成的模型会自动保存，支持断点续训
4. **参数调优**：可根据具体数据特点调整模型超参数

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目仅用于学习和竞赛目的。