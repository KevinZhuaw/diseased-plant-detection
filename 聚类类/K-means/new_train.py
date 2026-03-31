# deep_kmeans_centers.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import pandas as pd
import time
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

# 清理内存
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据集路径
data_dir = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")


# 获取类别信息
def get_class_info(data_dir):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    # 健康vs病害映射
    binary_mapping = {}
    for cls in classes:
        binary_mapping[cls] = 0 if 'healthy' in cls.lower() else 1

    return classes, class_to_idx, idx_to_class, binary_mapping


classes, class_to_idx, idx_to_class, binary_mapping = get_class_info(train_dir)
num_classes = len(classes)
print(f"Total classes: {num_classes}")


# 数据集类
class LeafDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_samples_per_class=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        for class_name in classes:
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if max_samples_per_class and len(images) > max_samples_per_class:
                images = images[:max_samples_per_class]

            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                label = class_to_idx[class_name]
                binary_label = binary_mapping[class_name]
                self.samples.append((img_path, label, binary_label))

        print(f"Loaded {len(self.samples)} images from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, binary_label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            image = self.transform(image)

        return image, label, binary_label


# 数据变换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 创建数据集
print("Loading datasets...")
train_dataset = LeafDataset(train_dir, transform=train_transform, max_samples_per_class=500)
val_dataset = LeafDataset(val_dir, transform=val_transform, max_samples_per_class=100)
test_dataset = LeafDataset(test_dir, transform=val_transform)

# 数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {batch_size}")


# 深度K-means中心模型
class DeepKMeansCenters(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # 特征提取器
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        # 特征投影层
        self.projector = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim)
        )

        # 可学习的类别中心 (类似K-means中心点)
        self.class_centers = nn.Parameter(torch.randn(num_classes, feature_dim))

        # 二分类头
        self.binary_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        projected = self.projector(features)

        # 计算到每个类别中心的距离 (负距离，越大越近)
        distances = torch.cdist(projected.unsqueeze(0), self.class_centers.unsqueeze(0)).squeeze(0)
        similarities = -distances  # 使用负距离作为相似度

        # 二分类输出
        binary_output = self.binary_head(projected)

        return binary_output, similarities, projected


# 深度K-means损失函数
class DeepKMeansLoss(nn.Module):
    def __init__(self, lambda_center=0.1, lambda_binary=0.5):
        super().__init__()
        self.lambda_center = lambda_center
        self.lambda_binary = lambda_binary
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, binary_output, similarities, features, labels, binary_labels, class_centers):
        # 二分类损失
        binary_loss = self.ce_loss(binary_output, binary_labels)

        # 中心吸引损失：样本应该靠近自己的类别中心
        batch_size = features.size(0)
        center_loss = 0

        for i in range(batch_size):
            target_center = class_centers[labels[i]]
            center_loss += torch.norm(features[i] - target_center, p=2)

        center_loss = center_loss / batch_size

        # 相似度损失：使用交叉熵，相似度应该匹配真实标签
        similarity_loss = self.ce_loss(similarities, labels)

        # 总损失
        total_loss = (self.lambda_binary * binary_loss +
                      self.lambda_center * center_loss +
                      similarity_loss)

        return total_loss, binary_loss, center_loss, similarity_loss


# 训练函数
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    binary_correct = 0
    multiclass_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, labels, binary_labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        binary_labels = binary_labels.to(device)

        optimizer.zero_grad()

        binary_output, similarities, features = model(images)

        # 预测
        _, binary_preds = torch.max(binary_output, 1)
        _, multiclass_preds = torch.max(similarities, 1)

        # 计算损失
        loss, binary_loss, center_loss, similarity_loss = criterion(
            binary_output, similarities, features, labels, binary_labels, model.class_centers
        )

        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        binary_correct += (binary_preds == binary_labels).sum().item()
        multiclass_correct += (multiclass_preds == labels).sum().item()
        total_samples += images.size(0)

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * multiclass_correct / total_samples:.1f}%'
        })

        # 定期清理缓存
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    binary_acc = 100. * binary_correct / total_samples
    multiclass_acc = 100. * multiclass_correct / total_samples

    return avg_loss, binary_acc, multiclass_acc


# 验证函数
@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    binary_correct = 0
    multiclass_correct = 0
    total_samples = 0

    all_binary_preds = []
    all_binary_labels = []
    all_multiclass_preds = []
    all_multiclass_labels = []

    pbar = tqdm(loader, desc="Validation")
    for images, labels, binary_labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        binary_labels = binary_labels.to(device)

        binary_output, similarities, features = model(images)

        # 预测
        _, binary_preds = torch.max(binary_output, 1)
        _, multiclass_preds = torch.max(similarities, 1)

        # 计算损失
        loss, _, _, _ = criterion(
            binary_output, similarities, features, labels, binary_labels, model.class_centers
        )

        total_loss += loss.item()
        binary_correct += (binary_preds == binary_labels).sum().item()
        multiclass_correct += (multiclass_preds == labels).sum().item()
        total_samples += images.size(0)

        # 收集预测结果
        all_binary_preds.extend(binary_preds.cpu().numpy())
        all_binary_labels.extend(binary_labels.cpu().numpy())
        all_multiclass_preds.extend(multiclass_preds.cpu().numpy())
        all_multiclass_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            'acc': f'{100. * multiclass_correct / total_samples:.1f}%'
        })

    avg_loss = total_loss / len(loader)
    binary_acc = 100. * binary_correct / total_samples
    multiclass_acc = 100. * multiclass_correct / total_samples

    # 计算详细指标
    binary_metrics = compute_metrics(all_binary_labels, all_binary_preds, binary=True)
    multiclass_metrics = compute_metrics(all_multiclass_labels, all_multiclass_preds, binary=False)

    return avg_loss, binary_acc, multiclass_acc, binary_metrics, multiclass_metrics


# 计算评估指标
def compute_metrics(true_labels, pred_labels, binary=False):
    if binary:
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='binary', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='binary', zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average='binary', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(true_labels, pred_labels)
        }
    else:
        accuracy = accuracy_score(true_labels, pred_labels)
        precision_macro = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

        precision_weighted = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        recall_weighted = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

        class_report = classification_report(true_labels, pred_labels,
                                             target_names=classes,
                                             zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': confusion_matrix(true_labels, pred_labels),
            'classification_report': class_report
        }

    return metrics


# 主训练循环
def main():
    print("=" * 70)
    print("DEEP K-MEANS CENTERS MODEL FOR LEAF DISEASE CLASSIFICATION")
    print("=" * 70)

    # 创建模型
    model = DeepKMeansCenters(num_classes=num_classes, feature_dim=256).to(device)

    # 损失函数和优化器
    criterion = DeepKMeansLoss(lambda_center=0.1, lambda_binary=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    num_epochs = 30
    best_val_acc = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # 训练
        train_loss, train_binary_acc, train_multiclass_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # 验证
        val_loss, val_binary_acc, val_multiclass_acc, val_binary_metrics, val_multiclass_metrics = validate_epoch(
            model, val_loader, criterion, device
        )

        # 更新学习率
        scheduler.step()

        # 保存指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_multiclass_acc)
        val_accs.append(val_multiclass_acc)

        # 打印结果
        print(
            f"Train Loss: {train_loss:.4f}, Binary Acc: {train_binary_acc:.2f}%, Multiclass Acc: {train_multiclass_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Binary Acc: {val_binary_acc:.2f}%, Multiclass Acc: {val_multiclass_acc:.2f}%")

        # 保存最佳模型
        if val_multiclass_acc > best_val_acc:
            best_val_acc = val_multiclass_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_multiclass_acc,
                'classes': classes,
                'class_to_idx': class_to_idx,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, "best_deep_kmeans_model.pth")
            print(f"✓ Saved best model with val acc: {val_multiclass_acc:.2f}%")

        # 清理缓存
        torch.cuda.empty_cache()

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # 测试最佳模型
    print("\n" + "=" * 70)
    print("TESTING BEST MODEL")
    print("=" * 70)

    # 加载最佳模型
    checkpoint = torch.load("best_deep_kmeans_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 测试
    test_loss, test_binary_acc, test_multiclass_acc, test_binary_metrics, test_multiclass_metrics = validate_epoch(
        model, test_loader, criterion, device
    )

    print(f"\nTest Results:")
    print(f"Binary Classification Accuracy: {test_binary_acc:.2f}%")
    print(f"Multiclass Classification Accuracy: {test_multiclass_acc:.2f}%")

    print(f"\nDetailed Binary Classification Metrics (Test):")
    print(f"Accuracy: {test_binary_metrics['accuracy']:.4f}")
    print(f"Precision: {test_binary_metrics['precision']:.4f}")
    print(f"Recall: {test_binary_metrics['recall']:.4f}")
    print(f"F1-Score: {test_binary_metrics['f1_score']:.4f}")

    print(f"\nDetailed Multiclass Classification Metrics (Test):")
    print(f"Macro Average Precision: {test_multiclass_metrics['precision_macro']:.4f}")
    print(f"Macro Average Recall: {test_multiclass_metrics['recall_macro']:.4f}")
    print(f"Macro Average F1-Score: {test_multiclass_metrics['f1_macro']:.4f}")
    print(f"Weighted Average Precision: {test_multiclass_metrics['precision_weighted']:.4f}")
    print(f"Weighted Average Recall: {test_multiclass_metrics['recall_weighted']:.4f}")
    print(f"Weighted Average F1-Score: {test_multiclass_metrics['f1_weighted']:.4f}")

    print(f"\nClassification Report (Test):")
    print(test_multiclass_metrics['classification_report'])

    # 保存测试结果
    save_test_results(test_binary_metrics, test_multiclass_metrics)

    # 预测示例
    print("\n" + "=" * 70)
    print("PREDICTION DEMONSTRATION")
    print("=" * 70)

    # 找一个测试图像
    test_classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    if test_classes:
        class_path = os.path.join(test_dir, test_classes[0])
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            test_image = os.path.join(class_path, images[0])
            print(f"\nPredicting sample image: {os.path.basename(test_image)}")

            # 预测单张图像
            result = predict_single_image(model, test_image)
            if result:
                print(f"  Disease Status: {result['disease_status']}")
                print(f"  Disease Type: {result['disease_type']}")
                print(f"  Binary Confidence: {result['binary_confidence']:.4f}")
                print(f"  Multiclass Confidence: {result['multiclass_confidence']:.4f}")

                print(f"\n  Top 3 predictions:")
                for i, (disease, prob) in enumerate(result['top3_predictions']):
                    print(f"    {i + 1}. {disease}: {prob:.4f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(train_losses) + 1)

        # 损失曲线
        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('deep_kmeans_training_curves.png', dpi=300, bbox_inches='tight')
        print("\nTraining curves saved as 'deep_kmeans_training_curves.png'")
        plt.close()
    except Exception as e:
        print(f"Error plotting curves: {e}")


def save_test_results(binary_metrics, multiclass_metrics):
    """保存测试结果"""
    results_dir = "deep_kmeans_results"
    os.makedirs(results_dir, exist_ok=True)

    # 保存每个类别的性能
    class_names = classes
    cm = multiclass_metrics['confusion_matrix']

    # 计算每个类别的指标
    class_metrics = []
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics.append({
            'class': class_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'samples': cm[i, :].sum()
        })

    df_class = pd.DataFrame(class_metrics)
    df_class.to_csv(os.path.join(results_dir, 'class_metrics.csv'), index=False)

    # 保存汇总结果
    summary_data = [
        {'metric': 'binary_accuracy', 'value': binary_metrics['accuracy']},
        {'metric': 'binary_precision', 'value': binary_metrics['precision']},
        {'metric': 'binary_recall', 'value': binary_metrics['recall']},
        {'metric': 'binary_f1', 'value': binary_metrics['f1_score']},
        {'metric': 'multiclass_accuracy', 'value': multiclass_metrics['accuracy']},
        {'metric': 'macro_precision', 'value': multiclass_metrics['precision_macro']},
        {'metric': 'macro_recall', 'value': multiclass_metrics['recall_macro']},
        {'metric': 'macro_f1', 'value': multiclass_metrics['f1_macro']},
        {'metric': 'weighted_precision', 'value': multiclass_metrics['precision_weighted']},
        {'metric': 'weighted_recall', 'value': multiclass_metrics['recall_weighted']},
        {'metric': 'weighted_f1', 'value': multiclass_metrics['f1_weighted']}
    ]

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(results_dir, 'summary_metrics.csv'), index=False)

    # 保存混淆矩阵
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(results_dir, 'confusion_matrix.csv'))

    print(f"\nResults saved to '{results_dir}' folder")


def predict_single_image(model, image_path):
    """预测单张图像"""
    model.eval()

    # 加载和预处理图像
    try:
        image = Image.open(image_path).convert('RGB')
    except:
        print(f"Error loading image: {image_path}")
        return None

    transform = val_transform
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        binary_output, similarities, _ = model(image_tensor)

        # 获取二分类结果
        binary_probs = torch.softmax(binary_output, dim=1)
        binary_pred = torch.argmax(binary_probs, dim=1)
        binary_confidence = torch.max(binary_probs, dim=1)[0].item()

        # 获取多分类结果
        multiclass_probs = torch.softmax(similarities, dim=1)
        multiclass_pred = torch.argmax(multiclass_probs, dim=1)
        multiclass_confidence = torch.max(multiclass_probs, dim=1)[0].item()

        # 获取类别名称
        disease_status = "Healthy" if binary_pred.item() == 0 else "Diseased"
        disease_type = idx_to_class[multiclass_pred.item()]

        # 获取前3个最可能的类别
        top3_probs, top3_indices = torch.topk(multiclass_probs, 3)
        top3_diseases = [(idx_to_class[idx.item()], prob.item())
                         for idx, prob in zip(top3_indices[0], top3_probs[0])]

    return {
        'disease_status': disease_status,
        'disease_type': disease_type,
        'binary_confidence': binary_confidence,
        'multiclass_confidence': multiclass_confidence,
        'top3_predictions': top3_diseases
    }


if __name__ == "__main__":
    main()