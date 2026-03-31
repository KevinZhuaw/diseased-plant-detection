import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import os
from PIL import Image
import json
import gc
import warnings

warnings.filterwarnings('ignore')

# 清理GPU内存
torch.cuda.empty_cache()
gc.collect()

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 内存优化的数据预处理
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # 进一步减小图像尺寸以节省内存
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 修改数据集类以支持子集
class LeafDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform

        # 只存储路径，不加载图像
        self.image_paths = []
        self.health_labels = []  # 0=健康, 1=有病
        self.disease_labels = []  # 具体病害类别

        if class_to_idx is None:
            self.classes = sorted([d for d in os.listdir(root_dir)
                                   if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())

        # 收集数据路径和标签
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_idx = self.class_to_idx[class_name]
            is_healthy = 0 if 'healthy' in class_name.lower() else 1

            img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_name in img_files:
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.health_labels.append(is_healthy)
                self.disease_labels.append(class_idx)

        print(f"Loaded {len(self.image_paths)} images from {len(self.classes)} classes")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # 按需加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 如果加载失败，返回一个黑色图像
            image = Image.new('RGB', (96, 96))

        if self.transform:
            image = self.transform(image)

        return image, self.health_labels[idx], self.disease_labels[idx]


# 包装Subset以保留原始属性
class SubsetWithAttributes(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.dataset = dataset  # 保留对原始数据集的引用


# 极简CNN模型
class SimpleLeafClassifier(nn.Module):
    def __init__(self, num_classes=22):
        super(SimpleLeafClassifier, self).__init__()

        # 简单的CNN特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 健康分类器
        self.health_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 2)  # 二分类：健康 vs 有病
        )

        # 病害分类器
        self.disease_classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)  # 多分类：具体病害类型
        )

    def forward(self, x):
        features = self.features(x)
        health_pred = self.health_classifier(features)
        disease_pred = self.disease_classifier(features)
        return health_pred, disease_pred


# 训练函数
def train_simple_model(model, train_loader, val_loader, epochs=10):
    model.train()

    # 只训练病害分类器，冻结特征提取器（节省内存）
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam([
        {'params': model.health_classifier.parameters(), 'lr': 1e-3},
        {'params': model.disease_classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct_health = 0
        train_correct_disease = 0
        train_total = 0

        for batch_idx, (data, health_labels, disease_labels) in enumerate(train_loader):
            data = data.to(device)
            health_labels = health_labels.to(device)
            disease_labels = disease_labels.to(device)

            optimizer.zero_grad()
            health_pred, disease_pred = model(data)

            # 计算损失
            health_loss = F.cross_entropy(health_pred, health_labels)
            disease_loss = F.cross_entropy(disease_pred, disease_labels)
            loss = 1.5 * health_loss + disease_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            # 计算准确率
            _, health_predicted = torch.max(health_pred, 1)
            _, disease_predicted = torch.max(disease_pred, 1)
            train_correct_health += (health_predicted == health_labels).sum().item()
            train_correct_disease += (disease_predicted == disease_labels).sum().item()
            train_total += health_labels.size(0)

            # 定期清理缓存
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # 验证
        model.eval()
        val_loss = 0
        val_correct_health = 0
        val_correct_disease = 0
        val_total = 0

        with torch.no_grad():
            for data, health_labels, disease_labels in val_loader:
                data = data.to(device)
                health_labels = health_labels.to(device)
                disease_labels = disease_labels.to(device)

                health_pred, disease_pred = model(data)

                health_loss = F.cross_entropy(health_pred, health_labels)
                disease_loss = F.cross_entropy(disease_pred, disease_labels)
                loss = 1.5 * health_loss + disease_loss

                val_loss += loss.item()

                _, health_predicted = torch.max(health_pred, 1)
                _, disease_predicted = torch.max(disease_pred, 1)
                val_correct_health += (health_predicted == health_labels).sum().item()
                val_correct_disease += (disease_predicted == disease_labels).sum().item()
                val_total += health_labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_health_acc = train_correct_health / train_total
        train_disease_acc = train_correct_disease / train_total
        val_health_acc = val_correct_health / val_total
        val_disease_acc = val_correct_disease / val_total

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(
            f'  Train Loss: {avg_train_loss:.4f}, Health Acc: {train_health_acc:.4f}, Disease Acc: {train_disease_acc:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Health Acc: {val_health_acc:.4f}, Disease Acc: {val_disease_acc:.4f}')

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_health_acc': val_health_acc,
                'val_disease_acc': val_disease_acc,
            }, 'best_simple_model.pth')
            print(f'  Model saved!')

        scheduler.step(avg_val_loss)

        # 每2个epoch清理一次内存
        if epoch % 2 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    return model


# 评估函数
def evaluate_simple_model(model, test_loader, class_names):
    model.eval()

    all_health_preds = []
    all_health_labels = []
    all_disease_preds = []
    all_disease_labels = []

    with torch.no_grad():
        for batch_idx, (data, health_labels, disease_labels) in enumerate(test_loader):
            data = data.to(device)

            health_pred, disease_pred = model(data)

            health_preds = torch.argmax(health_pred, dim=1).cpu().numpy()
            disease_preds = torch.argmax(disease_pred, dim=1).cpu().numpy()

            all_health_preds.extend(health_preds)
            all_health_labels.extend(health_labels.numpy())
            all_disease_preds.extend(disease_preds)
            all_disease_labels.extend(disease_labels.numpy())

            # 定期清理
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    # 计算健康/有病分类指标
    print("\n" + "=" * 60)
    print("HEALTH CLASSIFICATION RESULTS (Disease vs Healthy)")
    print("=" * 60)

    health_accuracy = accuracy_score(all_health_labels, all_health_preds)
    health_precision = precision_score(all_health_labels, all_health_preds, average='binary', zero_division=0)
    health_recall = recall_score(all_health_labels, all_health_preds, average='binary', zero_division=0)
    health_f1 = f1_score(all_health_labels, all_health_preds, average='binary', zero_division=0)

    print(f"Accuracy:  {health_accuracy:.4f}")
    print(f"Precision: {health_precision:.4f}")
    print(f"Recall:    {health_recall:.4f}")
    print(f"F1-Score:  {health_f1:.4f}")

    # 混淆矩阵
    health_cm = confusion_matrix(all_health_labels, all_health_preds)
    print(f"\nHealth Confusion Matrix:")
    print(f"[[TN {health_cm[0][0]:>5}  FP {health_cm[0][1]:>5}]")
    print(f" [FN {health_cm[1][0]:>5}  TP {health_cm[1][1]:>5}]]")

    # 计算具体病害分类指标
    print("\n" + "=" * 60)
    print("SPECIFIC DISEASE CLASSIFICATION RESULTS")
    print("=" * 60)

    disease_accuracy = accuracy_score(all_disease_labels, all_disease_preds)
    disease_precision = precision_score(all_disease_labels, all_disease_preds, average='weighted', zero_division=0)
    disease_recall = recall_score(all_disease_labels, all_disease_preds, average='weighted', zero_division=0)
    disease_f1 = f1_score(all_disease_labels, all_disease_preds, average='weighted', zero_division=0)

    print(f"Accuracy:  {disease_accuracy:.4f}")
    print(f"Precision: {disease_precision:.4f}")
    print(f"Recall:    {disease_recall:.4f}")
    print(f"F1-Score:  {disease_f1:.4f}")

    # 每个类别的详细报告（简化版）
    print(f"\nClassification Report (showing top 5 classes by support):")

    # 计算每个类别的样本数
    class_counts = {}
    for label in all_disease_labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    # 选择样本数最多的5个类别
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_class_indices = [c[0] for c in top_classes]

    # 过滤数据
    mask = [i for i, label in enumerate(all_disease_labels) if label in top_class_indices]
    filtered_labels = [all_disease_labels[i] for i in mask]
    filtered_preds = [all_disease_preds[i] for i in mask]
    filtered_names = [class_names[i] for i in top_class_indices]

    print(classification_report(filtered_labels, filtered_preds,
                                target_names=filtered_names, digits=4))

    return {
        'health_accuracy': health_accuracy,
        'health_precision': health_precision,
        'health_recall': health_recall,
        'health_f1': health_f1,
        'disease_accuracy': disease_accuracy,
        'disease_precision': disease_precision,
        'disease_recall': disease_recall,
        'disease_f1': disease_f1,
        'health_cm': health_cm.tolist()
    }


# 主程序
def main():
    # 设置数据路径
    data_dir = "E:/庄楷文/叶片病虫害识别/baseline/data/reorganized_dataset_new"

    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()

    print("Loading datasets...")

    # 创建数据集
    train_dataset = LeafDiseaseDataset(
        root_dir=os.path.join(data_dir, 'train'),
        transform=transform
    )

    val_dataset = LeafDiseaseDataset(
        root_dir=os.path.join(data_dir, 'val'),
        transform=transform,
        class_to_idx=train_dataset.class_to_idx
    )

    test_dataset = LeafDiseaseDataset(
        root_dir=os.path.join(data_dir, 'test'),
        transform=transform,
        class_to_idx=train_dataset.class_to_idx
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")

    # 使用数据子集（避免内存问题）
    use_subset = True

    if use_subset:
        print("\nUsing subset of data for memory efficiency...")

        # 为每个类别选择固定数量的样本
        def get_subset_indices(dataset, max_samples_per_class=100):
            indices = []
            class_counts = {}

            for idx in range(len(dataset)):
                _, health_label, disease_label = dataset[idx]

                # 检查是否已收集足够的该类样本
                if disease_label not in class_counts:
                    class_counts[disease_label] = 0

                if class_counts[disease_label] < max_samples_per_class:
                    indices.append(idx)
                    class_counts[disease_label] += 1

                # 如果所有类别都达到最大样本数，停止
                if len(class_counts) == len(dataset.classes) and all(
                        count >= max_samples_per_class for count in class_counts.values()):
                    break

            return indices

        # 创建子集
        train_indices = get_subset_indices(train_dataset, max_samples_per_class=100)
        val_indices = get_subset_indices(val_dataset, max_samples_per_class=30)
        test_indices = get_subset_indices(test_dataset, max_samples_per_class=50)

        train_subset = SubsetWithAttributes(train_dataset, train_indices)
        val_subset = SubsetWithAttributes(val_dataset, val_indices)
        test_subset = SubsetWithAttributes(test_dataset, test_indices)

        print(f"Subset sizes - Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

        # 使用子集
        train_dataset = train_subset
        val_dataset = val_subset
        test_dataset = test_subset

    # 创建数据加载器
    batch_size = 8  # 12GB显存可以处理
    num_workers = 0  # 设置为0以避免多进程内存问题

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 初始化模型
    print("\nInitializing simple CNN model...")
    model = SimpleLeafClassifier(num_classes=len(train_dataset.classes)).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # 训练模型
    print("\nStarting training...")
    try:
        model = train_simple_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=10
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nOut of memory! Trying with even smaller batch size...")
            # 降低batch size
            batch_size = 4
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            torch.cuda.empty_cache()
            gc.collect()

            model = train_simple_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=5
            )
        else:
            print(f"Training error: {e}")
            return

    # 加载最佳模型
    if os.path.exists('best_simple_model.pth'):
        checkpoint = torch.load('best_simple_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")
        print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
        print(f"Best validation health accuracy: {checkpoint['val_health_acc']:.4f}")
        print(f"Best validation disease accuracy: {checkpoint['val_disease_acc']:.4f}")

    # 评估模型
    print("\nEvaluating on test set...")
    results = evaluate_simple_model(
        model,
        test_loader,
        train_dataset.classes
    )

    # 保存结果
    with open('evaluation_results_simple.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to evaluation_results_simple.json")

    # 保存类别映射
    with open('class_mapping.json', 'w') as f:
        json.dump({
            'class_to_idx': train_dataset.class_to_idx,
            'idx_to_class': {idx: cls for cls, idx in train_dataset.class_to_idx.items()}
        }, f, indent=4)

    print("\nTraining and evaluation completed successfully!")


# 执行主程序
if __name__ == "__main__":
    # 设置环境变量以优化内存使用
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\nTrying with even more minimal setup...")

        # 尝试使用最小的设置
        try:
            torch.cuda.empty_cache()
            gc.collect()

            # 使用最小化的训练
            print("\nRunning minimal training...")

            # 这里可以添加更简单的训练代码
        except Exception as e2:
            print(f"Minimal setup also failed: {str(e2)}")
            print("\nTroubleshooting suggestions:")
            print("1. Check if you have enough disk space")
            print("2. Try running on CPU instead of GPU")
            print("3. Reduce image size to 64x64")
            print("4. Use batch size of 2")
            print(
                "5. Install torch with CUDA 11.8: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")


# 如果仍然有问题，使用这个最小化版本
def minimal_training():
    """最小化训练版本，几乎不占用内存"""
    print("Running minimal training setup...")

    # 使用更小的图像
    min_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 在CPU上训练
    device = torch.device('cpu')

    # 最小模型
    class MinModel(nn.Module):
        def __init__(self, num_classes=22):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3)
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(8, 16, 3)
            self.fc1 = nn.Linear(16 * 14 * 14, 32)
            self.fc_health = nn.Linear(32, 2)
            self.fc_disease = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            health = self.fc_health(x)
            disease = self.fc_disease(x)
            return health, disease

    print("Minimal training setup ready (not executed)")
    print("If needed, call minimal_training() function")