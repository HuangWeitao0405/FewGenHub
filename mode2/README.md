# FewGenHub: 基于ViT的少样本图像分类实验

## 项目概述
本项目使用Vision Transformer (ViT)模型在OxfordIIITPet和Flowers102数据集上进行图像分类实验，探索迁移学习在不同数据集上的表现。实验结果表明，ViT模型在宠物分类任务上达到91.82%的准确率，在花卉分类任务上达到82.09%的准确率。

## 代码实现思路

### 1. 环境配置与依赖
- 使用PyTorch深度学习框架
- 多GPU训练支持(nn.DataParallel)
- 主要依赖库: torch, torchvision, tqdm, matplotlib, seaborn, scikit-learn

### 2. 数据集准备
- **OxfordIIITPet**: 37类宠物图像，训练集2944张，测试集准确率91.82%
- **Flowers102**: 102类花卉图像，测试集准确率82.09%
- 数据增强: 随机裁剪、水平翻转、标准化等预处理

### 3. 模型架构
采用预训练的ViT-B/16模型，替换分类头以适应不同数据集：
```python
def create_vit_model(num_classes=37):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    # 修改分类头
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model
```

### 4. 训练策略
- **超参数**: batch_size=32, epochs=10
- **优化器**: Adam
- **损失函数**: CrossEntropyLoss
- **学习率调度**: ReduceLROnPlateau (factor=0.5, patience=3)

### 5. 迁移学习实验
- OxfordPets → Flowers102: 81.92%准确率
- Flowers102 → OxfordPets: 92.37%准确率

## 实验结果

### 分类报告
OxfordPets数据集上的分类报告示例：
```
              precision    recall  f1-score   support

    Abyssinian       0.88      0.87      0.87        98
American Bulldog       0.81      0.87      0.84       100
...
   Yorkshire Terrier       0.99      1.00      1.00       100

    accuracy                           0.92      3669
   macro avg       0.92      0.92      0.92      3669
weighted avg       0.92      0.92      0.92      3669
```

### 可视化结果
- 混淆矩阵: confusion_matrix_OxfordPets.png, confusion_matrix_Flowers102.png
- 训练历史: training_history_OxfordPets.png, training_history_Flowers102.png

## 使用方法
1. 安装依赖: `pip install -r requirements.txt`
2. 运行训练脚本: `jupyter notebook run.ipynb`
3. 查看实验结果图表和模型性能指标

## 结论
ViT模型在图像分类任务上表现优异，通过迁移学习可以有效提升模型在不同数据集上的泛化能力。特别是从复杂数据集迁移到简单数据集时，准确率提升更为明显。