# 中国石化AI竞赛解决方案

## 项目简介

本项目包含中国石化AI竞赛的两个赛题解决方案：

### 🏆 岩性识别AI竞赛 (LithoClass_WellLog)
基于测井曲线数据（SP、GR、AC）进行岩性分类预测，识别粉砂岩、砂岩、泥岩三种岩性类别。

**核心成果**: 集成模型F1分数达到0.9554 (95.54%)

### 🛢️ 注采调整多目标优化竞赛 (InjProd_Adjust_MHPerm_WF)
基于油田生产数据进行注采调整的多目标优化，提升油田开发效果。

## 项目结构

```
sinopec-aicup/
├── LithoClass_WellLog/          # 岩性识别竞赛
│   ├── train_fast_ensemble.py   # 🚀 主训练脚本
│   ├── base_models.py           # 模型训练器
│   ├── config.py                # 配置文件
│   ├── utils.py                 # 工具函数
│   └── README.md                # 详细说明文档
├── dataset/                     # 数据集目录
│   ├── LithoClass_WellLog/      # 岩性识别数据
│   └── InjProd_Adjust_MHPerm_WF/ # 注采调整数据
├── requirements.txt             # 依赖包列表
└── README.md                    # 本文档
```

### 🏆 岩性识别竞赛使用方法
```bash
cd LithoClass_WellLog
python train_fast_ensemble.py
```

### 🛢️ 注采调整竞赛使用方法
```bash
# 待开发
```

## 详细说明

- **岩性识别竞赛**: 详细技术文档请查看 [LithoClass_WellLog/README.md](LithoClass_WellLog/README.md)
- **注采调整竞赛**: 开发中...

## 许可证

本项目仅用于学习和竞赛目的。