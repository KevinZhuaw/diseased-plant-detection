# improved_leaf_classifier_fixed.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ==================== 配置 ====================
class Config:
    IMG_SIZE = 128
    BATCH_SIZE = 8
    SAMPLE_LIMIT_TRAIN = 200
    SAMPLE_LIMIT_VAL = 100
    SAMPLE_LIMIT_TEST = 200
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-5

    # 类别权重（解决不平衡问题）
    HEALTHY_CLASS_WEIGHT = 0.7
    DISEASE_CLASS_WEIGHT = 1.3


# ==================== 数据集 ====================
class EnhancedDataset(Dataset):
    def __init__(self, data_path, transform=None, sample_limit=None, mode='train'):
        self.data_path = data_path
        self.transform = transform
        self.sample_limit = sample_limit
        self.mode = mode
        self.samples = []

        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(data_path)
                               if os.path.isdir(os.path.join(data_path, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        print(f"Loading {mode} data from {len(self.classes)} classes...")

        for cls_name in self.classes:
            cls_path = os.path.join(data_path, cls_name)
            if not os.path.exists(cls_path):
                continue

            images = glob.glob(os.path.join(cls_path, "*.jpg"))

            if self.sample_limit and len(images) > self.sample_limit:
                if mode == 'train':
                    indices = np.random.choice(len(images), self.sample_limit, replace=False)
                else:
                    indices = np.linspace(0, len(images) - 1, self.sample_limit, dtype=int)
                images = [images[i] for i in indices]

            for img_path in images:
                self.samples.append({
                    'path': img_path,
                    'disease_label': self.class_to_idx[cls_name],
                    'healthy_label': 1 if 'healthy' in cls_name.lower() else 0,
                    'class_weight': Config.HEALTHY_CLASS_WEIGHT if 'healthy' in cls_name.lower()
                    else Config.DISEASE_CLASS_WEIGHT
                })

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            img = Image.open(sample['path']).convert('RGB')
        except:
            img = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), color='white')

        if self.transform:
            img = self.transform(img)

        return (img,
                sample['disease_label'],
                sample['healthy_label'])


# ==================== 数据增强 ====================
def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


# ==================== 改进的模型 ====================
class ImprovedClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedClassifier, self).__init__()

        # 使用预训练的ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 冻结前几层，只训练最后几层
        for param in list(resnet.parameters())[:-10]:
            param.requires_grad = False

        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # 获取特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            features = self.features(dummy_input)
            feature_dim = features.view(1, -1).size(1)

        # 健康/病害二分类
        self.healthy_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        # 具体病害多分类
        self.disease_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        print(f"Model feature dimension: {feature_dim}")

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)

        healthy_out = self.healthy_head(features)
        disease_out = self.disease_head(features)

        return healthy_out, disease_out


# ==================== 修复的训练器 ====================
class FixedTrainer:
    def __init__(self, num_classes, device):
        self.device = device
        self.num_classes = num_classes

        # 初始化模型
        self.model = ImprovedClassifier(num_classes).to(device)

        # 优化器 - 只训练需要梯度的参数
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.AdamW(trainable_params,
                                     lr=Config.LEARNING_RATE,
                                     weight_decay=Config.WEIGHT_DECAY)

        # 学习率调度器 - 移除verbose参数
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )

        # 损失函数 - 添加类别权重
        self.healthy_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))
        self.disease_criterion = nn.CrossEntropyLoss()

        self.best_val_acc = 0
        self.train_history = {
            'loss': [], 'healthy_acc': [], 'disease_acc': [],
            'val_healthy_acc': [], 'val_disease_acc': []
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct_healthy = 0
        correct_disease = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for images, disease_labels, healthy_labels in progress_bar:
            images = images.to(self.device)
            disease_labels = disease_labels.to(self.device)
            healthy_labels = healthy_labels.to(self.device)

            self.optimizer.zero_grad()

            healthy_out, disease_out = self.model(images)

            loss_healthy = self.healthy_criterion(healthy_out, healthy_labels)
            loss_disease = self.disease_criterion(disease_out, disease_labels)
            loss = loss_healthy * 1.5 + loss_disease * 0.5

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            _, healthy_pred = torch.max(healthy_out, 1)
            _, disease_pred = torch.max(disease_out, 1)

            correct_healthy += (healthy_pred == healthy_labels).sum().item()
            correct_disease += (disease_pred == disease_labels).sum().item()
            total += healthy_labels.size(0)

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'h_acc': f'{100 * correct_healthy / total:.1f}%',
                'd_acc': f'{100 * correct_disease / total:.1f}%'
            })

            if total % 200 == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        acc_healthy = 100 * correct_healthy / total
        acc_disease = 100 * correct_disease / total

        return avg_loss, acc_healthy, acc_disease

    def validate(self, val_loader):
        self.model.eval()
        correct_healthy = 0
        correct_disease = 0
        total = 0

        all_healthy_true = []
        all_healthy_pred = []

        with torch.no_grad():
            for images, disease_labels, healthy_labels in val_loader:
                images = images.to(self.device)
                disease_labels = disease_labels.to(self.device)
                healthy_labels = healthy_labels.to(self.device)

                healthy_out, disease_out = self.model(images)

                _, healthy_pred = torch.max(healthy_out, 1)
                _, disease_pred = torch.max(disease_out, 1)

                correct_healthy += (healthy_pred == healthy_labels).sum().item()
                correct_disease += (disease_pred == disease_labels).sum().item()
                total += healthy_labels.size(0)

                all_healthy_true.extend(healthy_labels.cpu().numpy())
                all_healthy_pred.extend(healthy_pred.cpu().numpy())

        acc_healthy = 100 * correct_healthy / total
        acc_disease = 100 * correct_disease / total

        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(all_healthy_true, all_healthy_pred):
            cm[t][p] += 1

        return acc_healthy, acc_disease, cm

    def train(self, train_loader, val_loader, epochs):
        print(f"\nStarting training for {epochs} epochs...")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss, train_acc_healthy, train_acc_disease = self.train_epoch(train_loader)
            val_acc_healthy, val_acc_disease, val_cm = self.validate(val_loader)

            # 更新学习率
            self.scheduler.step(val_acc_healthy + val_acc_disease)

            self.train_history['loss'].append(train_loss)
            self.train_history['healthy_acc'].append(train_acc_healthy)
            self.train_history['disease_acc'].append(train_acc_disease)
            self.train_history['val_healthy_acc'].append(val_acc_healthy)
            self.train_history['val_disease_acc'].append(val_acc_disease)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train - Healthy Acc: {train_acc_healthy:.2f}%, Disease Acc: {train_acc_disease:.2f}%")
            print(f"Val - Healthy Acc: {val_acc_healthy:.2f}%, Disease Acc: {val_acc_disease:.2f}%")
            print(f"Confusion Matrix:")
            print(f"[[TN={val_cm[0][0]} FP={val_cm[0][1]}]")
            print(f" [FN={val_cm[1][0]} TP={val_cm[1][1]}]]")

            val_total_acc = val_acc_healthy + val_acc_disease
            if val_total_acc > self.best_val_acc:
                self.best_val_acc = val_total_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_total_acc,
                    'history': self.train_history
                }, 'best_improved_model.pth')
                print(f"Saved best model with val accuracy: {val_total_acc:.2f}")

            torch.cuda.empty_cache()

    def test(self, test_loader, class_names):
        print("\n" + "=" * 50)
        print("Testing on test set")
        print("=" * 50)

        self.model.eval()
        all_healthy_true = []
        all_healthy_pred = []
        all_disease_true = []
        all_disease_pred = []

        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Testing")
            for images, disease_labels, healthy_labels in progress_bar:
                images = images.to(self.device)
                disease_labels = disease_labels.to(self.device)
                healthy_labels = healthy_labels.to(self.device)

                healthy_out, disease_out = self.model(images)

                _, healthy_pred = torch.max(healthy_out, 1)
                _, disease_pred = torch.max(disease_out, 1)

                all_healthy_true.extend(healthy_labels.cpu().numpy())
                all_healthy_pred.extend(healthy_pred.cpu().numpy())
                all_disease_true.extend(disease_labels.cpu().numpy())
                all_disease_pred.extend(disease_pred.cpu().numpy())

        results = self.calculate_metrics(all_healthy_true, all_healthy_pred,
                                         all_disease_true, all_disease_pred,
                                         class_names)

        return results

    def calculate_metrics(self, healthy_true, healthy_pred, disease_true, disease_pred, class_names):
        def binary_metrics(true, pred):
            if len(true) == 0:
                return 0, 0, 0, 0, (0, 0, 0, 0)

            true = np.array(true)
            pred = np.array(pred)

            accuracy = 100 * np.mean(true == pred)

            tn = np.sum((true == 0) & (pred == 0))
            fp = np.sum((true == 0) & (pred == 1))
            fn = np.sum((true == 1) & (pred == 0))
            tp = np.sum((true == 1) & (pred == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            return accuracy, precision, recall, f1, (tn, fp, fn, tp)

        def multiclass_accuracy(true, pred, num_classes, class_names):
            if len(true) == 0:
                return 0, []

            class_correct = [0] * num_classes
            class_total = [0] * num_classes

            for i in range(len(true)):
                class_label = true[i]
                class_total[class_label] += 1
                if true[i] == pred[i]:
                    class_correct[class_label] += 1

            total_correct = sum(class_correct)
            total_samples = sum(class_total)
            accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0

            class_accuracies = []
            for i in range(num_classes):
                if class_total[i] > 0:
                    class_accuracies.append({
                        'class_name': class_names[i],
                        'accuracy': 100 * class_correct[i] / class_total[i],
                        'samples': class_total[i],
                        'correct': class_correct[i]
                    })
                else:
                    class_accuracies.append({
                        'class_name': class_names[i],
                        'accuracy': 0,
                        'samples': 0,
                        'correct': 0
                    })

            return accuracy, class_accuracies

        healthy_acc, healthy_precision, healthy_recall, healthy_f1, cm_stats = binary_metrics(
            healthy_true, healthy_pred)
        tn, fp, fn, tp = cm_stats

        disease_acc, class_accuracies = multiclass_accuracy(disease_true, disease_pred,
                                                            len(class_names), class_names)

        return {
            'healthy': {
                'accuracy': healthy_acc,
                'precision': healthy_precision,
                'recall': healthy_recall,
                'f1': healthy_f1,
                'confusion_matrix': [[tn, fp], [fn, tp]]
            },
            'disease': {
                'accuracy': disease_acc,
                'class_accuracies': class_accuracies
            }
        }


# ==================== 主函数 ====================
BASE_PATH = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"


def main():
    print("=" * 60)
    print("IMPROVED LEAF DISEASE CLASSIFICATION SYSTEM (FIXED)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        torch.cuda.empty_cache()

    print("\nCreating datasets...")

    train_transform = get_transforms('train')
    test_transform = get_transforms('test')

    train_dataset = EnhancedDataset(
        os.path.join(BASE_PATH, "train"),
        transform=train_transform,
        sample_limit=Config.SAMPLE_LIMIT_TRAIN,
        mode='train'
    )

    val_dataset = EnhancedDataset(
        os.path.join(BASE_PATH, "val"),
        transform=test_transform,
        sample_limit=Config.SAMPLE_LIMIT_VAL,
        mode='val'
    )

    test_dataset = EnhancedDataset(
        os.path.join(BASE_PATH, "test"),
        transform=test_transform,
        sample_limit=Config.SAMPLE_LIMIT_TEST,
        mode='test'
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)

    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    print(f"Number of classes: {len(train_dataset.classes)}")

    trainer = FixedTrainer(len(train_dataset.classes), device)

    try:
        trainer.train(train_loader, val_loader, Config.EPOCHS)

        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        checkpoint = torch.load('best_improved_model.pth', map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])

        results = trainer.test(test_loader, train_dataset.classes)

        print_results(results, train_dataset.classes)

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nRunning simplified evaluation instead...")
        evaluate_simplified_model()


def print_results(results, class_names):
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"\n1. HEALTHY vs DISEASE CLASSIFICATION:")
    print(f"   Accuracy: {results['healthy']['accuracy']:.2f}%")
    print(f"   Precision: {results['healthy']['precision']:.4f}")
    print(f"   Recall: {results['healthy']['recall']:.4f}")
    print(f"   F1-Score: {results['healthy']['f1']:.4f}")

    cm = results['healthy']['confusion_matrix']
    print(f"\n   Confusion Matrix:")
    print(f"   [[TN={cm[0][0]} FP={cm[0][1]}]")
    print(f"    [FN={cm[1][0]} TP={cm[1][1]}]]")

    sensitivity = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
    specificity = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0

    print(f"   Sensitivity (Recall): {sensitivity:.4f}")
    print(f"   Specificity: {specificity:.4f}")

    print(f"\n2. SPECIFIC DISEASE CLASSIFICATION:")
    print(f"   Overall Accuracy: {results['disease']['accuracy']:.2f}%")

    print(f"\n   Class-wise Accuracies (sorted by accuracy):")
    class_accuracies = sorted(results['disease']['class_accuracies'],
                              key=lambda x: x['accuracy'], reverse=True)

    for i, cls_info in enumerate(class_accuracies[:15]):  # 显示前15个
        print(f"   {i + 1:2d}. {cls_info['class_name'][:35]:35s}: {cls_info['accuracy']:6.1f}% "
              f"({cls_info['correct']}/{cls_info['samples']})")

    save_detailed_results(results, class_names)


def save_detailed_results(results, class_names):
    with open('detailed_results.txt', 'w') as f:
        f.write("DETAILED CLASSIFICATION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("HEALTHY vs DISEASE CLASSIFICATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {results['healthy']['accuracy']:.2f}%\n")
        f.write(f"Precision: {results['healthy']['precision']:.4f}\n")
        f.write(f"Recall: {results['healthy']['recall']:.4f}\n")
        f.write(f"F1-Score: {results['healthy']['f1']:.4f}\n")

        cm = results['healthy']['confusion_matrix']
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"True Negatives (健康判为健康): {cm[0][0]}\n")
        f.write(f"False Positives (健康判为病害): {cm[0][1]}\n")
        f.write(f"False Negatives (病害判为健康): {cm[1][0]}\n")
        f.write(f"True Positives (病害判为病害): {cm[1][1]}\n")

        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
        specificity = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        f.write(f"Sensitivity (Recall): {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")

        f.write("\n\nSPECIFIC DISEASE CLASSIFICATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Accuracy: {results['disease']['accuracy']:.2f}%\n")

        f.write("\nClass-wise Accuracies:\n")
        class_accuracies = sorted(results['disease']['class_accuracies'],
                                  key=lambda x: x['accuracy'], reverse=True)

        for i, cls_info in enumerate(class_accuracies):
            f.write(f"{i + 1:3d}. {cls_info['class_name']:45s}: {cls_info['accuracy']:6.1f}% "
                    f"({cls_info['correct']:3d}/{cls_info['samples']:3d})\n")

    print("\nDetailed results saved to 'detailed_results.txt'")


def evaluate_simplified_model():
    """评估简化版本模型"""
    print("\n" + "=" * 60)
    print("EVALUATING SIMPLIFIED MODEL")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载简化模型
    class SimpleModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2)

            self.fc1 = nn.Linear(128 * 8 * 8, 256)  # 64->32->16->8
            self.fc_healthy = nn.Linear(256, 2)
            self.fc_disease = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 8 * 8)
            x = F.relu(self.fc1(x))
            return self.fc_healthy(x), self.fc_disease(x)

    # 获取类别信息
    def get_classes(data_path):
        return sorted([d for d in os.listdir(data_path)
                       if os.path.isdir(os.path.join(data_path, d))])

    classes = get_classes(os.path.join(BASE_PATH, "train"))
    num_classes = len(classes)

    model = SimpleModel(num_classes).to(device)
    model.load_state_dict(torch.load('simplified_model.pth', map_location=device))
    model.eval()

    # 创建测试集
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = EnhancedDataset(
        os.path.join(BASE_PATH, "test"),
        transform=transform,
        sample_limit=200,
        mode='test'
    )

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"Testing on {len(test_dataset)} samples...")

    all_healthy_true = []
    all_healthy_pred = []
    all_disease_true = []
    all_disease_pred = []

    with torch.no_grad():
        for images, disease_labels, healthy_labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            disease_labels = disease_labels.to(device)
            healthy_labels = healthy_labels.to(device)

            healthy_out, disease_out = model(images)

            _, healthy_pred = torch.max(healthy_out, 1)
            _, disease_pred = torch.max(disease_out, 1)

            all_healthy_true.extend(healthy_labels.cpu().numpy())
            all_healthy_pred.extend(healthy_pred.cpu().numpy())
            all_disease_true.extend(disease_labels.cpu().numpy())
            all_disease_pred.extend(disease_pred.cpu().numpy())

    # 计算指标
    def calculate_metrics(healthy_true, healthy_pred, disease_true, disease_pred):
        healthy_true = np.array(healthy_true)
        healthy_pred = np.array(healthy_pred)
        disease_true = np.array(disease_true)
        disease_pred = np.array(disease_pred)

        # 健康/病害指标
        healthy_acc = 100 * np.mean(healthy_true == healthy_pred)

        tn = np.sum((healthy_true == 0) & (healthy_pred == 0))
        fp = np.sum((healthy_true == 0) & (healthy_pred == 1))
        fn = np.sum((healthy_true == 1) & (healthy_pred == 0))
        tp = np.sum((healthy_true == 1) & (healthy_pred == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 病害分类指标
        disease_acc = 100 * np.mean(disease_true == disease_pred)

        return {
            'healthy_acc': healthy_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': [[tn, fp], [fn, tp]],
            'disease_acc': disease_acc
        }

    results = calculate_metrics(all_healthy_true, all_healthy_pred,
                                all_disease_true, all_disease_pred)

    print("\n" + "=" * 60)
    print("SIMPLIFIED MODEL RESULTS")
    print("=" * 60)

    print(f"\n1. HEALTHY vs DISEASE CLASSIFICATION:")
    print(f"   Accuracy: {results['healthy_acc']:.2f}%")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall: {results['recall']:.4f}")
    print(f"   F1-Score: {results['f1']:.4f}")

    cm = results['confusion_matrix']
    print(f"\n   Confusion Matrix:")
    print(f"   [[TN={cm[0][0]} FP={cm[0][1]}]")
    print(f"    [FN={cm[1][0]} TP={cm[1][1]}]]")

    print(f"\n2. SPECIFIC DISEASE CLASSIFICATION:")
    print(f"   Overall Accuracy: {results['disease_acc']:.2f}%")

    # 保存结果
    with open('simplified_model_results.txt', 'w') as f:
        f.write("SIMPLIFIED MODEL RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Healthy vs Disease Accuracy: {results['healthy_acc']:.2f}%\n")
        f.write(f"Healthy vs Disease Precision: {results['precision']:.4f}\n")
        f.write(f"Healthy vs Disease Recall: {results['recall']:.4f}\n")
        f.write(f"Healthy vs Disease F1-Score: {results['f1']:.4f}\n")
        f.write(f"Specific Disease Accuracy: {results['disease_acc']:.2f}%\n")

    print("\nResults saved to 'simplified_model_results.txt'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        evaluate_simplified_model()