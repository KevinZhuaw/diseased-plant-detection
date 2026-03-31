import os
import sys
import warnings

warnings.filterwarnings('ignore')

# 先检查环境
print("Checking environment...")

try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
except Exception as e:
    print(f"Error importing PyTorch: {e}")
    sys.exit(1)

# 检查其他必要库
required_libs = ['PIL', 'numpy', 'sklearn', 'pandas']
for lib in required_libs:
    try:
        __import__(lib.lower() if lib == 'PIL' else lib)
        print(f"✓ {lib} is available")
    except ImportError:
        print(f"✗ {lib} not found, installing...")
        os.system(f"pip install {lib}")

# 重新导入
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

if device.type == 'cuda':
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # 设置GPU内存优化
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

# 数据集路径
data_path = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"


# 简化的数据集类
class SimpleLeafDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = []
        self.samples = []

        # 获取所有类别
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)

        self.classes.sort()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        # 找到健康类别
        self.healthy_idx = None
        for idx, cls in enumerate(self.classes):
            if 'healthy' in cls.lower():
                self.healthy_idx = idx
                break

        # 收集所有样本
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.root_dir, class_name)

            # 限制每个类别的样本数量，避免内存问题（可调整）
            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # 如果样本太多，可以采样（可选）
            if len(image_files) > 2000:  # 限制每个类别最大样本数
                import random
                image_files = random.sample(image_files, 2000)

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                self.samples.append((img_path, class_idx))

        print(f"{split}: {len(self.samples)} samples, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 如果图片损坏，使用一个空白图片
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            image = self.transform(image)

        # 二分类标签：0=健康，1=有病
        binary_label = 0 if label == self.healthy_idx else 1

        return image, label, binary_label


# 数据变换
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 简化模型（使用更小的模型节省内存）
class SimpleLeafClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleLeafClassifier, self).__init__()
        # 使用ResNet18而不是ResNet50节省内存
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features

        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # 多分类头
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # 二分类头
        self.binary_classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        multi_out = self.classifier(features)
        binary_out = self.binary_classifier(features)

        return multi_out, binary_out


def train_simple():
    print("\n" + "=" * 60)
    print("SIMPLE LEAF DISEASE CLASSIFICATION TRAINING")
    print("=" * 60)

    # 创建数据集
    print("\nLoading datasets...")
    train_dataset = SimpleLeafDataset(data_path, 'train', train_transform)
    val_dataset = SimpleLeafDataset(data_path, 'val', test_transform)

    # 数据加载器（使用更小的batch size）
    batch_size = 8 if torch.cuda.is_available() else 4
    print(f"Batch size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)  # Windows上num_workers=0更稳定
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # 创建模型
    model = SimpleLeafClassifier(len(train_dataset.classes)).to(device)

    # 损失函数和优化器
    criterion_multi = nn.CrossEntropyLoss()
    criterion_binary = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练循环
    num_epochs = 20
    best_acc = 0.0

    print("\nStarting training...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_multi = 0
        correct_binary = 0
        total = 0

        for batch_idx, (images, labels_multi, labels_binary) in enumerate(train_loader):
            images = images.to(device)
            labels_multi = labels_multi.to(device)
            labels_binary = labels_binary.to(device)

            optimizer.zero_grad()

            outputs_multi, outputs_binary = model(images)

            loss_multi = criterion_multi(outputs_multi, labels_multi)
            loss_binary = criterion_binary(outputs_binary, labels_binary)
            loss = loss_multi + 0.3 * loss_binary  # 调整权重

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 计算准确率
            _, preds_multi = torch.max(outputs_multi, 1)
            _, preds_binary = torch.max(outputs_binary, 1)

            correct_multi += (preds_multi == labels_multi).sum().item()
            correct_binary += (preds_binary == labels_binary).sum().item()
            total += labels_multi.size(0)

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        # 验证阶段
        model.eval()
        val_correct_multi = 0
        val_correct_binary = 0
        val_total = 0

        with torch.no_grad():
            for images, labels_multi, labels_binary in val_loader:
                images = images.to(device)
                labels_multi = labels_multi.to(device)
                labels_binary = labels_binary.to(device)

                outputs_multi, outputs_binary = model(images)

                _, preds_multi = torch.max(outputs_multi, 1)
                _, preds_binary = torch.max(outputs_binary, 1)

                val_correct_multi += (preds_multi == labels_multi).sum().item()
                val_correct_binary += (preds_binary == labels_binary).sum().item()
                val_total += labels_multi.size(0)

        # 计算准确率
        train_acc_multi = 100 * correct_multi / total
        train_acc_binary = 100 * correct_binary / total
        val_acc_multi = 100 * val_correct_multi / val_total
        val_acc_binary = 100 * val_correct_binary / val_total

        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}')
        print(f'Train Acc - Multi: {train_acc_multi:.2f}%, Binary: {train_acc_binary:.2f}%')
        print(f'Val Acc - Multi: {val_acc_multi:.2f}%, Binary: {val_acc_binary:.2f}%')

        # 学习率调整
        scheduler.step()

        # 保存最佳模型
        if val_acc_multi > best_acc:
            best_acc = val_acc_multi
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_multi': val_acc_multi,
                'classes': train_dataset.classes,
                'healthy_idx': train_dataset.healthy_idx
            }, 'best_simple_model.pth')
            print(f'Best model saved! Accuracy: {val_acc_multi:.2f}%')

    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")

    return model, train_dataset.classes, train_dataset.healthy_idx


def evaluate_model(model, class_names, healthy_idx):
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    # 加载测试集
    test_dataset = SimpleLeafDataset(data_path, 'test', test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    model.eval()

    all_labels_multi = []
    all_preds_multi = []
    all_labels_binary = []
    all_preds_binary = []

    with torch.no_grad():
        for images, labels_multi, labels_binary in test_loader:
            images = images.to(device)

            outputs_multi, outputs_binary = model(images)

            _, preds_multi = torch.max(outputs_multi, 1)
            _, preds_binary = torch.max(outputs_binary, 1)

            all_labels_multi.extend(labels_multi.cpu().numpy())
            all_preds_multi.extend(preds_multi.cpu().numpy())
            all_labels_binary.extend(labels_binary.cpu().numpy())
            all_preds_binary.extend(preds_binary.cpu().numpy())

    # 计算指标
    print("\nMULTI-CLASS RESULTS:")
    print("-" * 40)
    accuracy_multi = accuracy_score(all_labels_multi, all_preds_multi)
    precision_multi = precision_score(all_labels_multi, all_preds_multi, average='weighted', zero_division=0)
    recall_multi = recall_score(all_labels_multi, all_preds_multi, average='weighted', zero_division=0)
    f1_multi = f1_score(all_labels_multi, all_preds_multi, average='weighted', zero_division=0)

    print(f"Accuracy:  {accuracy_multi:.4f}")
    print(f"Precision: {precision_multi:.4f}")
    print(f"Recall:    {recall_multi:.4f}")
    print(f"F1 Score:  {f1_multi:.4f}")

    print("\nBINARY RESULTS (Healthy vs Diseased):")
    print("-" * 40)
    accuracy_binary = accuracy_score(all_labels_binary, all_preds_binary)
    precision_binary = precision_score(all_labels_binary, all_preds_binary, average='binary', zero_division=0)
    recall_binary = recall_score(all_labels_binary, all_preds_binary, average='binary', zero_division=0)
    f1_binary = f1_score(all_labels_binary, all_preds_binary, average='binary', zero_division=0)

    print(f"Accuracy:  {accuracy_binary:.4f}")
    print(f"Precision: {precision_binary:.4f}")
    print(f"Recall:    {recall_binary:.4f}")
    print(f"F1 Score:  {f1_binary:.4f}")

    # 类别统计
    print("\nCLASS-WISE ACCURACY:")
    print("-" * 40)
    for i, class_name in enumerate(class_names):
        indices = [j for j, label in enumerate(all_labels_multi) if label == i]
        if indices:
            correct = sum([1 for j in indices if all_preds_multi[j] == i])
            total = len(indices)
            accuracy = correct / total if total > 0 else 0
            health_status = "(Healthy)" if i == healthy_idx else "(Diseased)"
            print(f"{class_name:30} {health_status:15} {accuracy:.4f} ({correct}/{total})")

    return {
        'multi': {'accuracy': accuracy_multi, 'precision': precision_multi,
                  'recall': recall_multi, 'f1': f1_multi},
        'binary': {'accuracy': accuracy_binary, 'precision': precision_binary,
                   'recall': recall_binary, 'f1': f1_binary},
        'class_names': class_names,
        'healthy_idx': healthy_idx
    }


# 主函数
def main():
    try:
        # 训练模型
        model, class_names, healthy_idx = train_simple()

        # 评估模型
        results = evaluate_model(model, class_names, healthy_idx)

        # 保存结果
        print("\nSaving results...")
        import json
        with open('training_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        print("\nResults saved to 'training_results.json'")
        print("Model saved as 'best_simple_model.pth'")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 检查CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will be slow on CPU.")
        print("建议检查NVIDIA驱动和CUDA安装。")

    main()