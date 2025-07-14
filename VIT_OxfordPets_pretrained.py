import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import vit_b_16
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 检测GPU数量
device_count = torch.cuda.device_count()
print(f"发现{device_count}个GPU")

# 数据预处理和增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 获取OxfordPets数据集
print("正在下载和准备数据集...")
dataset = datasets.OxfordIIITPet(
    root='./data',
    split='trainval',
    download=True,
    transform=train_transform
)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建测试集
test_dataset = datasets.OxfordIIITPet(
    root='./data',
    split='test',
    download=True,
    transform=test_transform
)

# 创建数据加载器
batch_size = 32

# 单GPU和多GPU都使用相同的数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 构建ViT模型
def create_vit_model(num_classes=37):
    # 加载预训练的ViT模型
    model = vit_b_16(pretrained=False)  # 从头开始训练，不使用预训练权重
    
    # 修改分类头以适应OxfordPets数据集
    model.heads.head = nn.Linear(model.hidden_dim, num_classes)
    
    return model

# 实例化模型
model = create_vit_model()

# 使用DataParallel在多个GPU上并行运行
if device_count > 1:
    print(f"使用{device_count}个GPU进行训练")
    model = nn.DataParallel(model)

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_progress.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
        train_loss = running_loss / len(train_dataset)
        train_acc = 100. * correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for inputs, labels in val_progress:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_progress.set_postfix({'loss': loss.item(), 'acc': 100.*val_correct/val_total})
        
        val_loss = val_loss / len(val_dataset)
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史数据
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 对于DataParallel模型，需要保存model.module
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), 'best_vit_model.pth')
            print(f'保存最佳模型 (验证准确率: {best_val_acc:.2f}%)')
    
    return history

# 训练模型
print("开始训练模型...")
epochs = 30  # 可以根据需要调整
history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device)

# 评估模型
def evaluate_model(model, data_loader, device):
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='评估模型'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    return all_labels, all_predictions

# 加载最佳模型进行评估
model = create_vit_model()
model.load_state_dict(torch.load('best_vit_model.pth'))
model = model.to(device)

# 如果有多个GPU，在评估时也使用DataParallel
if device_count > 1:
    model = nn.DataParallel(model)

model.eval()

# 在测试集上评估
print("在测试集上评估模型...")
true_labels, predictions = evaluate_model(model, test_loader, device)

# 生成分类报告
print("\n分类报告:")
print(classification_report(true_labels, predictions, target_names=dataset.classes))

# 计算准确率
accuracy = np.mean(np.array(true_labels) == np.array(predictions))
print(f"\n测试集准确率: {accuracy * 100:.2f}%")

# 绘制混淆矩阵
plt.figure(figsize=(15, 12))
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# 绘制训练历史
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='训练损失')
plt.plot(history['val_loss'], label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.title('训练和验证损失')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='训练准确率')
plt.plot(history['val_acc'], label='验证准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率 (%)')
plt.title('训练和验证准确率')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

print("模型训练和评估完成!")
print("混淆矩阵和训练历史已保存为图片文件.")