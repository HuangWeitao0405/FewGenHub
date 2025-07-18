# OxfordPets ViT 微调模型项目介绍

## 项目概述
本项目使用 Vision Transformer (ViT) 模型对 Oxford-IIIT Pet 数据集进行分类任务的微调。通过迁移学习技术，利用预训练的 ViT-B/16 模型权重，针对宠物分类任务进行适应性调整，最终实现对37种宠物类别的准确识别。

## 环境配置
### 依赖库
- Python 3.10+
- PyTorch 1.10+
- torchvision
- numpy
- matplotlib
- scikit-learn
- seaborn
- tqdm

## 数据集介绍
### Oxford-IIIT Pet 数据集
- 包含37种宠物类别，共计约7000张图像
- 每个类别包含大约200张图像
- 已划分训练集、验证集和测试集

### 数据预处理
```python
# 训练集转换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试集转换
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 模型架构
### ViT-B/16 模型
- 基于预训练的 ViT-B/16 模型
- 修改分类头以适应37个类别的分类任务
- 使用迁移学习，仅训练分类头部分参数

```python
def create_vit_model(num_classes=37):
    # 加载预训练的ViT模型
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    
    # 冻结大部分预训练层，只训练分类头
    for param in model.parameters():
        param.requires_grad = False
    
    # 修改分类头以适应OxfordPets数据集
    model.heads.head = nn.Linear(model.hidden_dim, num_classes)
    
    # 解冻分类头的参数
    for param in model.heads.head.parameters():
        param.requires_grad = True
    
    return model
```

## 训练流程
### 训练参数
- 批大小(batch size): 32
- 训练轮数(epochs): 10
- 优化器: Adam
- 初始学习率: 1e-4
- 学习率调度: ReduceLROnPlateau
- 损失函数: CrossEntropyLoss

### 训练过程
1. 数据加载与划分
   - 训练集: 80% 的 trainval 数据
   - 验证集: 20% 的 trainval 数据
   - 测试集: 官方提供的 test 数据

2. 模型训练
   - 使用早停策略保存验证集准确率最高的模型
   - 支持多GPU并行训练

3. 训练监控
   - 实时显示训练损失和准确率
   - 每个epoch结束后在验证集上评估

## 评估结果
### 性能指标
- 测试集准确率: ~90% (具体数值需根据实际训练结果)
- 分类报告: 包含每个类别的精确率、召回率和F1分数

### 可视化结果
- 混淆矩阵: confusion_matrix.png
- 训练历史: training_history.png (包含损失和准确率曲线)

## 文件说明
- `VIT_OxfordPets.py`: 模型训练和评估的主程序
- `VIT_OxfordPets.pth`: 训练好的模型权重
- `confusion_matrix.png`: 混淆矩阵可视化结果
- `training_history.png`: 训练过程中的损失和准确率曲线

## 使用方法
### 训练模型
```bash
python VIT_OxfordPets.py
```

### 评估模型
程序会在训练完成后自动在测试集上进行评估

## 总结
本项目通过微调ViT模型实现了对宠物类别的高精度分类。迁移学习策略有效提高了模型性能并减少了训练时间。项目代码结构清晰，包含了完整的数据预处理、模型构建、训练和评估流程，可作为基于Transformer的图像分类任务的参考实现。