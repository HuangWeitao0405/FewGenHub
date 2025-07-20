# FewGenHub: 多模态少样本学习实验平台

## 项目概述
FewGenHub是一个专注于少样本学习和视觉语言模型研究的实验平台，包含三个主要模式，分别针对不同的任务场景和研究方向。

## 模式说明

### Mode 1: 文本到图像生成
- **功能**：基于文本描述生成图像
- **核心文件**：`text2image.ipynb`
- **主要技术**：文本编码器、图像生成模型
- **使用方法**：打开Jupyter Notebook并运行所有单元格

### Mode 2: 基于ViT的少样本图像分类
- **功能**：使用视觉Transformer(ViT)进行少样本图像分类实验
- **核心文件**：`run.ipynb`
- **实验结果**：包含多种数据集(OxfordPets, Flowers102等)的混淆矩阵和训练历史可视化
- **使用方法**：
  1. 安装依赖：`pip install -r requirements.txt`
  2. 运行Jupyter Notebook：`jupyter notebook run.ipynb`
  3. 查看实验结果图表和模型性能指标

### Mode 3: 视觉语言模型的提示学习
- **功能**：实现CoOp和CoCoOp等提示学习方法，用于视觉语言模型适应下游任务
- **核心内容**：
  - CoOp: 学习视觉语言模型的提示
  - CoCoOp: 条件提示学习
- **主要技术**：CLIP模型、提示工程、少样本学习
- **使用方法**：参考`mode3/CoOp-main/CoOp-main/COOP.md`和`COCOOP.md`的详细说明

## 安装指南
1. 克隆仓库
2. 安装依赖：`pip install -r requirements.txt`
3. 按照各模式的具体说明运行实验

## 目录结构
```
├── LICENSE
├── mode1/            # 文本到图像生成
├── mode2/            # ViT少样本图像分类
├── mode3/            # 提示学习(CoOp/CoCoOp)
└── requirements.txt  # 项目依赖
```

## 许可证
本项目采用MIT许可证 - 详见LICENSE文件