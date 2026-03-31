# complete_leaf_disease_trainer.py
"""
COMPLETE LEAF DISEASE CLASSIFICATION SYSTEM
Author: AI Assistant
Date: 2024
Description: A complete system for training and testing leaf disease classification model
             with all metrics: Accuracy, Precision, Recall, F1-Score, and mAP.
             Optimized for 12GB GPU (4070Ti).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
import gc
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPLETE LEAF DISEASE CLASSIFICATION SYSTEM")
print("With Accuracy, Precision, Recall, F1-Score, and mAP")
print("Optimized for 12GB GPU (4070Ti)")
print("=" * 80)


# ==============================
# METRICS CALCULATOR (内置)
# ==============================
class MetricsCalculator:
    """完整的评估指标计算器"""

    @staticmethod
    def accuracy(y_true, y_pred):
        """计算准确率"""
        return np.sum(y_true == y_pred) / len(y_true)

    @staticmethod
    def precision_score(y_true, y_pred, average='binary'):
        """计算精确率"""
        if average == 'binary':
            # 二分类
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tp / (tp + fp) if (tp + fp) > 0 else 0
        elif average == 'macro':
            # 多分类宏平均
            classes = np.unique(np.concatenate([y_true, y_pred]))
            precisions = []
            for c in classes:
                tp = np.sum((y_true == c) & (y_pred == c))
                fp = np.sum((y_true != c) & (y_pred == c))
                precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            return np.mean(precisions)
        elif average == 'weighted':
            # 多分类加权平均
            classes = np.unique(np.concatenate([y_true, y_pred]))
            precisions = []
            weights = []
            for c in classes:
                tp = np.sum((y_true == c) & (y_pred == c))
                fp = np.sum((y_true != c) & (y_pred == c))
                precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
                weights.append(np.sum(y_true == c))
            weights = np.array(weights) / np.sum(weights)
            return np.sum(precisions * weights)

    @staticmethod
    def recall_score(y_true, y_pred, average='binary'):
        """计算召回率"""
        if average == 'binary':
            # 二分类
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return tp / (tp + fn) if (tp + fn) > 0 else 0
        elif average == 'macro':
            # 多分类宏平均
            classes = np.unique(np.concatenate([y_true, y_pred]))
            recalls = []
            for c in classes:
                tp = np.sum((y_true == c) & (y_pred == c))
                fn = np.sum((y_true == c) & (y_pred != c))
                recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            return np.mean(recalls)
        elif average == 'weighted':
            # 多分类加权平均
            classes = np.unique(np.concatenate([y_true, y_pred]))
            recalls = []
            weights = []
            for c in classes:
                tp = np.sum((y_true == c) & (y_pred == c))
                fn = np.sum((y_true == c) & (y_pred != c))
                recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                weights.append(np.sum(y_true == c))
            weights = np.array(weights) / np.sum(weights)
            return np.sum(recalls * weights)

    @staticmethod
    def f1_score(y_true, y_pred, average='binary'):
        """计算F1分数"""
        if average == 'binary':
            precision = MetricsCalculator.precision_score(y_true, y_pred, 'binary')
            recall = MetricsCalculator.recall_score(y_true, y_pred, 'binary')
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif average == 'macro':
            precision = MetricsCalculator.precision_score(y_true, y_pred, 'macro')
            recall = MetricsCalculator.recall_score(y_true, y_pred, 'macro')
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif average == 'weighted':
            precision = MetricsCalculator.precision_score(y_true, y_pred, 'weighted')
            recall = MetricsCalculator.recall_score(y_true, y_pred, 'weighted')
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    @staticmethod
    def calculate_map(y_true, y_scores, num_classes):
        """计算mAP (Mean Average Precision)"""
        aps = []
        for class_idx in range(num_classes):
            # 获取该类别的二值标签和预测分数
            y_true_binary = (y_true == class_idx).astype(int)
            y_score = y_scores[:, class_idx]

            # 按分数排序
            sorted_indices = np.argsort(y_score)[::-1]
            y_true_sorted = y_true_binary[sorted_indices]

            # 计算精确率-召回率曲线
            cum_tp = np.cumsum(y_true_sorted)
            cum_fp = np.cumsum(1 - y_true_sorted)

            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / np.sum(y_true_binary)

            # 计算AP (11点插值法)
            recall_levels = np.linspace(0, 1, 11)
            interp_precision = []
            for r in recall_levels:
                mask = recall >= r
                if np.any(mask):
                    interp_precision.append(np.max(precision[mask]))
                else:
                    interp_precision.append(0)

            ap = np.mean(interp_precision)
            aps.append(ap)

        return np.mean(aps) if aps else 0, aps

    @staticmethod
    def confusion_matrix(y_true, y_pred, num_classes):
        """计算混淆矩阵"""
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(len(y_true)):
            cm[y_true[i], y_pred[i]] += 1
        return cm

    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_scores=None, num_classes=None):
        """计算所有指标"""
        # 基本指标
        accuracy = MetricsCalculator.accuracy(y_true, y_pred)

        # 多分类指标
        precision_macro = MetricsCalculator.precision_score(y_true, y_pred, 'macro')
        recall_macro = MetricsCalculator.recall_score(y_true, y_pred, 'macro')
        f1_macro = MetricsCalculator.f1_score(y_true, y_pred, 'macro')

        precision_weighted = MetricsCalculator.precision_score(y_true, y_pred, 'weighted')
        recall_weighted = MetricsCalculator.recall_score(y_true, y_pred, 'weighted')
        f1_weighted = MetricsCalculator.f1_score(y_true, y_pred, 'weighted')

        # 二分类指标
        precision_binary = MetricsCalculator.precision_score(y_true, y_pred, 'binary')
        recall_binary = MetricsCalculator.recall_score(y_true, y_pred, 'binary')
        f1_binary = MetricsCalculator.f1_score(y_true, y_pred, 'binary')

        # 计算mAP
        map_score = 0
        aps = []
        if y_scores is not None and num_classes is not None:
            map_score, aps = MetricsCalculator.calculate_map(y_true, y_scores, num_classes)

        # 计算每个类别的指标
        per_class_metrics = []
        if num_classes is not None:
            for class_idx in range(num_classes):
                # 获取该类别的二值标签
                y_true_binary = (y_true == class_idx).astype(int)
                y_pred_binary = (y_pred == class_idx).astype(int)

                class_accuracy = MetricsCalculator.accuracy(y_true_binary, y_pred_binary)
                class_precision = MetricsCalculator.precision_score(y_true_binary, y_pred_binary, 'binary')
                class_recall = MetricsCalculator.recall_score(y_true_binary, y_pred_binary, 'binary')
                class_f1 = MetricsCalculator.f1_score(y_true_binary, y_pred_binary, 'binary')
                class_support = np.sum(y_true == class_idx)

                per_class_metrics.append({
                    'class': class_idx,
                    'accuracy': float(class_accuracy),
                    'precision': float(class_precision),
                    'recall': float(class_recall),
                    'f1': float(class_f1),
                    'support': int(class_support),
                    'ap': float(aps[class_idx]) if len(aps) > class_idx else 0.0
                })

        # 混淆矩阵
        cm = None
        if num_classes is not None:
            cm = MetricsCalculator.confusion_matrix(y_true, y_pred, num_classes)

        return {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'precision_binary': float(precision_binary) if len(np.unique(y_true)) <= 2 else 0.0,
            'recall_binary': float(recall_binary) if len(np.unique(y_true)) <= 2 else 0.0,
            'f1_binary': float(f1_binary) if len(np.unique(y_true)) <= 2 else 0.0,
            'map': float(map_score),
            'average_precisions': [float(ap) for ap in aps],
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist() if cm is not None else None
        }

    @staticmethod
    def print_metrics_report(metrics, class_names=None, title="EVALUATION REPORT"):
        """打印完整的指标报告"""
        print(f"\n{'=' * 80}")
        print(f"{title}")
        print(f"{'=' * 80}")

        print(f"\nOVERALL METRICS:")
        print(f"-" * 50)
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision (Macro):  {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro):     {metrics['recall_macro']:.4f}")
        print(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}")
        print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (Weighted):    {metrics['recall_weighted']:.4f}")
        print(f"F1-Score (Weighted):  {metrics['f1_weighted']:.4f}")
        print(f"mAP:                {metrics['map']:.4f}")

        if metrics.get('precision_binary', 0) > 0:
            print(f"\nBINARY CLASSIFICATION (Healthy vs Diseased):")
            print(f"-" * 50)
            print(f"Precision:          {metrics['precision_binary']:.4f}")
            print(f"Recall:             {metrics['recall_binary']:.4f}")
            print(f"F1-Score:           {metrics['f1_binary']:.4f}")

        if metrics.get('per_class_metrics') and class_names:
            print(f"\nPER-CLASS DETAILED METRICS:")
            print(f"-" * 80)
            print(f"{'Class':25} {'Acc':6} {'Prec':6} {'Rec':6} {'F1':6} {'AP':6} {'Support':8}")
            print(f"-" * 80)

            for item in metrics['per_class_metrics']:
                class_name = class_names[item['class']] if item['class'] < len(
                    class_names) else f"Class {item['class']}"
                print(f"{class_name:25} {item['accuracy']:6.4f} {item['precision']:6.4f} "
                      f"{item['recall']:6.4f} {item['f1']:6.4f} {item.get('ap', 0):6.4f} {item['support']:8}")

        # 混淆矩阵摘要
        if metrics.get('confusion_matrix'):
            cm = np.array(metrics['confusion_matrix'])
            print(f"\nCONFUSION MATRIX SUMMARY:")
            print(f"-" * 40)
            print(f"Total Samples: {np.sum(cm)}")
            print(f"Correct Predictions: {np.trace(cm)}")
            print(f"Overall Accuracy: {np.trace(cm) / np.sum(cm):.4f}")
            print(f"Error Rate: {1 - np.trace(cm) / np.sum(cm):.4f}")


# ==============================
# CONFIGURATION (内存优化)
# ==============================
class Config:
    """配置类 - 针对12GB GPU优化"""

    def __init__(self):
        # 数据路径
        self.DATA_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"

        # 内存优化参数
        self.BATCH_SIZE = 8  # 小批次以适应12GB显存
        self.IMG_SIZE = 128  # 较小图像尺寸
        self.NUM_EPOCHS = 15  # 减少epoch数以加速
        self.LEARNING_RATE = 0.001
        self.NUM_WORKERS = 0  # Windows上设为0避免多进程问题

        # 模型参数
        self.MODEL_NAME = 'resnet18'  # 使用较小模型
        self.PRETRAINED = True

        # 路径
        self.SAVE_DIR = './saved_models'
        self.LOG_DIR = './logs'
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

        # 设备
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.print_config()

    def print_config(self):
        """打印配置信息"""
        print(f"\n{'=' * 60}")
        print(f"CONFIGURATION")
        print(f"{'=' * 60}")
        print(f"Data Directory: {self.DATA_DIR}")
        print(f"Device: {self.DEVICE}")
        if self.DEVICE.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {memory_gb:.2f} GB")
        print(f"Batch Size: {self.BATCH_SIZE} (optimized for 12GB GPU)")
        print(f"Image Size: {self.IMG_SIZE} (memory efficient)")
        print(f"Model: {self.MODEL_NAME}")
        print(f"Epochs: {self.NUM_EPOCHS}")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        print(f"{'=' * 60}")


# ==============================
# MEMORY EFFICIENT DATASET
# ==============================
class MemoryEfficientDataset(Dataset):
    """内存高效数据集类 - 流式加载图像"""

    def __init__(self, split='train', config=None):
        if config is None:
            config = Config()

        self.data_dir = os.path.join(config.DATA_DIR, split)
        self.img_size = config.IMG_SIZE

        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(self.data_dir)
                               if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        # 收集样本（只存储路径和标签，不加载图像）
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            # 二分类标签：0=健康，1=有病
            is_healthy = 'healthy' in class_name.lower()
            binary_label = 0 if is_healthy else 1

            # 获取所有图像文件
            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_name in image_files:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append({
                    'path': img_path,
                    'class_idx': class_idx,
                    'binary_label': binary_label,
                    'class_name': class_name
                })

        print(f"{split.upper()} DATASET: {len(self.samples)} samples, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # 动态加载图像
            image = Image.open(sample['path']).convert('RGB')
            # 调整大小
            image = image.resize((self.img_size, self.img_size))
        except Exception as e:
            print(f"Error loading image {sample['path']}: {e}")
            # 返回一个黑色图像作为占位符
            image = Image.new('RGB', (self.img_size, self.img_size), color='black')

        # 简单的数据转换（无数据增强以节省内存）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return transform(image), sample['class_idx'], sample['binary_label']

    def get_class_distribution(self):
        """获取类别分布"""
        class_counts = {cls: 0 for cls in self.classes}
        for sample in self.samples:
            class_counts[sample['class_name']] += 1
        return class_counts


# ==============================
# MEMORY EFFICIENT MODEL
# ==============================
class EfficientLeafClassifier(nn.Module):
    """内存高效模型 - 使用较小模型并冻结大部分层"""

    def __init__(self, num_classes=22, model_name='resnet18'):
        super().__init__()

        # 使用预训练的ResNet18
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            num_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # 冻结大部分层以节省内存和加速训练
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 只解冻最后几层
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # 替换最后的全连接层
        self.backbone.fc = nn.Identity()

        # 多分类输出（22个病害类型）
        self.multi_head = nn.Linear(num_features, num_classes)

        # 二分类输出（健康 vs 病害）
        self.binary_head = nn.Linear(num_features, 2)

    def forward(self, x):
        features = self.backbone(x)
        return self.multi_head(features), self.binary_head(features)


# ==============================
# MAIN TRAINER CLASS
# ==============================
class LeafDiseaseTrainer:
    """主训练器类"""

    def __init__(self, config=None):
        self.config = config if config else Config()
        self.device = self.config.DEVICE

        # 组件
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.classes = None
        self.class_to_idx = None

        # 训练历史
        self.history = {
            'epoch': [],
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }

    def setup_data(self):
        """设置数据加载器"""
        print(f"\n{'=' * 60}")
        print(f"SETTING UP DATA LOADERS")
        print(f"{'=' * 60}")

        # 创建数据集
        train_dataset = MemoryEfficientDataset('train', self.config)
        val_dataset = MemoryEfficientDataset('val', self.config)
        test_dataset = MemoryEfficientDataset('test', self.config)

        self.classes = train_dataset.classes
        self.class_to_idx = train_dataset.class_to_idx

        # 打印类别分布
        self._print_class_distribution(train_dataset, val_dataset, test_dataset)

        # 创建数据加载器（不使用pin_memory和多进程以节省内存）
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=False  # 在Windows上设为False以避免内存问题
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=False
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=False
        )

        print(f"\nData loaders created:")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")

        return train_dataset, val_dataset, test_dataset

    def _print_class_distribution(self, train_dataset, val_dataset, test_dataset):
        """打印类别分布"""
        print(f"\nCLASS DISTRIBUTION:")
        print(f"-" * 60)

        # 获取分布
        train_dist = train_dataset.get_class_distribution()
        val_dist = val_dataset.get_class_distribution()
        test_dist = test_dataset.get_class_distribution()

        print(f"{'Class':30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
        print(f"-" * 60)

        total_train, total_val, total_test = 0, 0, 0
        for class_name in self.classes:
            train_count = train_dist.get(class_name, 0)
            val_count = val_dist.get(class_name, 0)
            test_count = test_dist.get(class_name, 0)
            total = train_count + val_count + test_count

            print(f"{class_name:30} {train_count:8} {val_count:8} {test_count:8} {total:8}")

            total_train += train_count
            total_val += val_count
            total_test += test_count

        print(f"-" * 60)
        print(f"{'TOTAL':30} {total_train:8} {total_val:8} {total_test:8} {total_train + total_val + total_test:8}")

        # 保存分布到文件
        import csv
        dist_path = os.path.join(self.config.LOG_DIR, 'class_distribution.csv')
        with open(dist_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Train', 'Val', 'Test', 'Total'])
            for class_name in self.classes:
                writer.writerow([
                    class_name,
                    train_dist.get(class_name, 0),
                    val_dist.get(class_name, 0),
                    test_dist.get(class_name, 0),
                    train_dist.get(class_name, 0) + val_dist.get(class_name, 0) + test_dist.get(class_name, 0)
                ])

        print(f"Class distribution saved to: {dist_path}")

    def setup_model(self):
        """设置模型"""
        print(f"\nSETTING UP MODEL...")

        # 创建模型
        self.model = EfficientLeafClassifier(
            num_classes=len(self.classes),
            model_name=self.config.MODEL_NAME
        ).to(self.device)

        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"  Model: {self.config.MODEL_NAME}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")

        return self.model

    def train(self):
        """训练模型"""
        print(f"\n{'=' * 60}")
        print(f"STARTING TRAINING")
        print(f"{'=' * 60}")

        # 设置数据和模型
        self.setup_data()
        self.setup_model()

        # 损失函数和优化器
        criterion_multi = nn.CrossEntropyLoss()
        criterion_binary = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print(f"-" * 40)

            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct_multi = 0
            train_correct_binary = 0
            train_total = 0

            # 简单的进度指示
            print(f"Training...", end='', flush=True)

            for batch_idx, (images, multi_labels, binary_labels) in enumerate(self.train_loader):
                # 移动到设备
                images = images.to(self.device)
                multi_labels = multi_labels.to(self.device)
                binary_labels = binary_labels.to(self.device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                multi_output, binary_output = self.model(images)

                # 计算损失
                loss_multi = criterion_multi(multi_output, multi_labels)
                loss_binary = criterion_binary(binary_output, binary_labels)
                loss = loss_multi + loss_binary

                # 反向传播
                loss.backward()
                optimizer.step()

                # 统计
                train_loss += loss.item()

                # 获取预测
                _, multi_pred = torch.max(multi_output, 1)
                _, binary_pred = torch.max(binary_output, 1)

                train_correct_multi += (multi_pred == multi_labels).sum().item()
                train_correct_binary += (binary_pred == binary_labels).sum().item()
                train_total += images.size(0)

                # 每50个批次打印一个点表示进度
                if batch_idx % 50 == 0:
                    print(f".", end='', flush=True)

                # 定期清理缓存
                if batch_idx % 100 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            print(f" Done")

            # 验证阶段
            val_results = self._evaluate(self.val_loader, phase='Validation')

            # 计算训练指标
            train_acc_multi = train_correct_multi / train_total
            train_acc_binary = train_correct_binary / train_total
            avg_train_loss = train_loss / len(self.train_loader)

            # 保存历史
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_results['multi_metrics'].get('loss', 0))
            self.history['train_acc'].append(train_acc_multi)
            self.history['val_acc'].append(val_results['multi_metrics']['accuracy'])
            self.history['train_f1'].append(0)  # 简化，实际应计算
            self.history['val_f1'].append(val_results['multi_metrics']['f1_macro'])

            # 打印结果
            print(f"\nEpoch {epoch + 1} Results:")
            print(
                f"  Train - Loss: {avg_train_loss:.4f}, Multi Acc: {train_acc_multi:.4f}, Binary Acc: {train_acc_binary:.4f}")
            print(f"  Val   - Multi Acc: {val_results['multi_metrics']['accuracy']:.4f}, "
                  f"Binary Acc: {val_results['binary_metrics']['accuracy']:.4f}")
            print(f"  Val   - mAP: {val_results['multi_metrics']['map']:.4f}, "
                  f"F1-Score: {val_results['multi_metrics']['f1_macro']:.4f}")

            # 保存最佳模型
            if val_results['multi_metrics']['accuracy'] > best_val_acc:
                best_val_acc = val_results['multi_metrics']['accuracy']
                best_epoch = epoch
                self._save_model('best_model.pth', epoch, val_results)
                print(
                    f"  ✓ Saved best model (val_acc: {best_val_acc:.4f}, mAP: {val_results['multi_metrics']['map']:.4f})")

            # 清理GPU缓存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETED!")
        print(f"{'=' * 60}")
        print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")

        return self.history

    def _evaluate(self, dataloader, phase='Validation'):
        """评估模型"""
        self.model.eval()

        all_multi_labels = []
        all_multi_preds = []
        all_multi_scores = []
        all_binary_labels = []
        all_binary_preds = []
        total_loss = 0.0

        criterion_multi = nn.CrossEntropyLoss()
        criterion_binary = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, multi_labels, binary_labels in dataloader:
                images = images.to(self.device)
                multi_labels = multi_labels.to(self.device)
                binary_labels = binary_labels.to(self.device)

                # 前向传播
                multi_output, binary_output = self.model(images)

                # 计算损失
                loss_multi = criterion_multi(multi_output, multi_labels)
                loss_binary = criterion_binary(binary_output, binary_labels)
                loss = loss_multi + loss_binary
                total_loss += loss.item()

                # 获取预测和分数
                multi_scores = torch.softmax(multi_output, dim=1)
                binary_scores = torch.softmax(binary_output, dim=1)

                _, multi_pred = torch.max(multi_output, 1)
                _, binary_pred = torch.max(binary_output, 1)

                # 保存结果
                all_multi_labels.extend(multi_labels.cpu().numpy())
                all_multi_preds.extend(multi_pred.cpu().numpy())
                all_multi_scores.extend(multi_scores.cpu().numpy())
                all_binary_labels.extend(binary_labels.cpu().numpy())
                all_binary_preds.extend(binary_pred.cpu().numpy())

        # 转换为numpy数组
        all_multi_labels = np.array(all_multi_labels)
        all_multi_preds = np.array(all_multi_preds)
        all_multi_scores = np.array(all_multi_scores)
        all_binary_labels = np.array(all_binary_labels)
        all_binary_preds = np.array(all_binary_preds)

        # 计算多分类指标
        multi_metrics = MetricsCalculator.calculate_all_metrics(
            all_multi_labels,
            all_multi_preds,
            all_multi_scores,
            len(self.classes)
        )
        multi_metrics['loss'] = total_loss / len(dataloader)

        # 计算二分类指标
        binary_metrics = MetricsCalculator.calculate_all_metrics(
            all_binary_labels,
            all_binary_preds,
            None,
            2
        )

        return {
            'multi_metrics': multi_metrics,
            'binary_metrics': binary_metrics,
            'predictions': {
                'multi_labels': all_multi_labels,
                'multi_preds': all_multi_preds,
                'multi_scores': all_multi_scores,
                'binary_labels': all_binary_labels,
                'binary_preds': all_binary_preds
            }
        }

    def _save_model(self, filename, epoch, metrics):
        """保存模型"""
        save_path = os.path.join(self.config.SAVE_DIR, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'config': {
                'model_name': self.config.MODEL_NAME,
                'img_size': self.config.IMG_SIZE,
                'batch_size': self.config.BATCH_SIZE,
                'num_classes': len(self.classes)
            },
            'metrics': {
                'multi_metrics': metrics['multi_metrics'],
                'binary_metrics': metrics['binary_metrics']
            },
            'history': self.history
        }, save_path)

    def test(self, model_path=None):
        """在测试集上测试模型"""
        print(f"\n{'=' * 80}")
        print(f"FINAL TEST EVALUATION")
        print(f"{'=' * 80}")

        # 加载模型
        if model_path is None:
            model_path = os.path.join(self.config.SAVE_DIR, 'best_model.pth')

        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            print("Please train a model first.")
            return None

        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # 如果还没有设置数据，先设置
        if self.classes is None:
            self.setup_data()

        # 创建模型并加载权重
        self.model = EfficientLeafClassifier(
            num_classes=len(self.classes),
            model_name=self.config.MODEL_NAME
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 评估测试集
        test_results = self._evaluate(self.test_loader, phase='Test')

        # 打印完整报告
        print(f"\n{'=' * 80}")
        print(f"COMPLETE TEST RESULTS")
        print(f"{'=' * 80}")

        # 使用MetricsCalculator打印报告
        MetricsCalculator.print_metrics_report(
            test_results['multi_metrics'],
            self.classes,
            "MULTI-CLASS CLASSIFICATION (22 Disease Types)"
        )

        MetricsCalculator.print_metrics_report(
            test_results['binary_metrics'],
            ['Healthy', 'Diseased'],
            "BINARY CLASSIFICATION (Healthy vs Diseased)"
        )

        # 保存详细结果
        self._save_test_results(test_results)

        return test_results

    def _save_test_results(self, test_results):
        """保存测试结果"""
        # 保存为JSON
        results_path = os.path.join(self.config.LOG_DIR, 'complete_test_results.json')

        # 准备可序列化的结果
        serializable_results = {
            'multi_metrics': test_results['multi_metrics'],
            'binary_metrics': test_results['binary_metrics'],
            'timestamp': datetime.now().isoformat(),
            'config': {
                'batch_size': self.config.BATCH_SIZE,
                'img_size': self.config.IMG_SIZE,
                'model_name': self.config.MODEL_NAME,
                'num_classes': len(self.classes)
            }
        }

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"\nComplete results saved to: {results_path}")

        # 保存CSV报告
        csv_path = os.path.join(self.config.LOG_DIR, 'detailed_metrics.csv')

        # 准备数据
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入总体指标
            writer.writerow(['CATEGORY', 'METRIC', 'VALUE'])
            writer.writerow(['Overall', 'Accuracy', test_results['multi_metrics']['accuracy']])
            writer.writerow(['Overall', 'Precision (Macro)', test_results['multi_metrics']['precision_macro']])
            writer.writerow(['Overall', 'Recall (Macro)', test_results['multi_metrics']['recall_macro']])
            writer.writerow(['Overall', 'F1-Score (Macro)', test_results['multi_metrics']['f1_macro']])
            writer.writerow(['Overall', 'mAP', test_results['multi_metrics']['map']])

            writer.writerow([])  # 空行

            # 写入每个类别的指标
            writer.writerow(['CLASS', 'ACCURACY', 'PRECISION', 'RECALL', 'F1-SCORE', 'AP', 'SUPPORT'])
            for item in test_results['multi_metrics']['per_class_metrics']:
                class_name = self.classes[item['class']] if item['class'] < len(
                    self.classes) else f"Class {item['class']}"
                writer.writerow([
                    class_name,
                    item['accuracy'],
                    item['precision'],
                    item['recall'],
                    item['f1'],
                    item.get('ap', 0),
                    item['support']
                ])

        print(f"Detailed metrics saved to: {csv_path}")

        # 保存混淆矩阵
        if test_results['multi_metrics']['confusion_matrix']:
            cm_path = os.path.join(self.config.LOG_DIR, 'confusion_matrix.csv')
            cm = np.array(test_results['multi_metrics']['confusion_matrix'])

            with open(cm_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 写入表头
                header = ['True/Pred'] + [f'Pred_{i}' for i in range(len(self.classes))]
                writer.writerow(header)
                # 写入数据
                for i in range(len(self.classes)):
                    class_name = self.classes[i] if i < len(self.classes) else f"Class {i}"
                    row = [f'True_{class_name}'] + cm[i].tolist()
                    writer.writerow(row)

            print(f"Confusion matrix saved to: {cm_path}")


# ==============================
# SIMPLE PREDICTION FUNCTION
# ==============================
def predict_single_image(image_path, model_path='saved_models/best_model.pth'):
    """预测单张图像"""
    import os

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        print("Please train the model first.")
        return

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)

    # 创建模型
    model = EfficientLeafClassifier(
        num_classes=len(checkpoint['classes']),
        model_name=checkpoint['config']['model_name']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载和预处理图像
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    img_size = checkpoint['config']['img_size']
    image = image.resize((img_size, img_size))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        multi_output, binary_output = model(input_tensor)

        multi_probs = torch.softmax(multi_output, dim=1)
        binary_probs = torch.softmax(binary_output, dim=1)

        _, multi_pred = torch.max(multi_output, 1)
        _, binary_pred = torch.max(binary_output, 1)

        # 获取类别名称
        classes = checkpoint['classes']
        disease_type = classes[multi_pred.item()]
        binary_result = "Healthy" if binary_pred.item() == 0 else "Diseased"

        print(f"\n{'=' * 60}")
        print(f"PREDICTION RESULTS")
        print(f"{'=' * 60}")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"\nHEALTH STATUS: {binary_result}")
        print(f"  Confidence: {binary_probs[0, binary_pred.item()]:.4f}")
        print(f"\nDISEASE TYPE: {disease_type}")
        print(f"  Confidence: {multi_probs[0, multi_pred.item()]:.4f}")

        # 显示前5个预测
        print(f"\nTOP 5 PREDICTIONS:")
        top5_probs, top5_indices = torch.topk(multi_probs, 5)
        for i in range(5):
            class_name = classes[top5_indices[0, i].item()]
            prob = top5_probs[0, i].item()
            health_status = "(Healthy)" if 'healthy' in class_name.lower() else "(Diseased)"
            print(f"  {i + 1}. {class_name} {health_status}: {prob:.4f}")

        print(f"{'=' * 60}")

        return {
            'binary_prediction': binary_result,
            'binary_confidence': float(binary_probs[0, binary_pred.item()]),
            'disease_type': disease_type,
            'disease_confidence': float(multi_probs[0, multi_pred.item()]),
            'is_healthy': 'healthy' in disease_type.lower(),
            'top_predictions': [
                (classes[idx], float(prob))
                for prob, idx in zip(top5_probs[0], top5_indices[0])
            ]
        }


# ==============================
# MAIN FUNCTION
# ==============================
def main():
    """主函数"""
    try:
        print(f"\n{'=' * 80}")
        print(f"LEAF DISEASE CLASSIFICATION SYSTEM")
        print(f"{'=' * 80}")

        # 创建配置
        config = Config()

        # 选择模式
        print(f"\nSelect mode:")
        print(f"1. Train and evaluate model")
        print(f"2. Test existing model")
        print(f"3. Predict single image")
        print(f"4. Quick demo (train for 5 epochs)")

        choice = input(f"\nEnter your choice (1-4): ").strip()

        if choice == '1':
            # 完整训练和评估
            trainer = LeafDiseaseTrainer(config)
            history = trainer.train()
            test_results = trainer.test()

            # 保存最终模型
            trainer._save_model('final_model.pth', config.NUM_EPOCHS - 1, test_results)

            print(f"\n{'=' * 80}")
            print(f"PROGRAM COMPLETED SUCCESSFULLY!")
            print(f"{'=' * 80}")
            print(f"\nGenerated files:")
            print(f"  saved_models/best_model.pth  - Best model")
            print(f"  saved_models/final_model.pth - Final model")
            print(f"  logs/complete_test_results.json - Complete test results")
            print(f"  logs/detailed_metrics.csv    - Detailed metrics")
            print(f"  logs/class_distribution.csv  - Class distribution")

        elif choice == '2':
            # 只测试现有模型
            trainer = LeafDiseaseTrainer(config)
            test_results = trainer.test()

            if test_results:
                print(f"\n{'=' * 80}")
                print(f"TESTING COMPLETED!")
                print(f"{'=' * 80}")

        elif choice == '3':
            # 预测单张图像
            image_path = input(f"\nEnter image path: ").strip()
            if os.path.exists(image_path):
                predict_single_image(image_path)
            else:
                print(f"Error: Image not found: {image_path}")

        elif choice == '4':
            # 快速演示
            print(f"\nRunning quick demo (5 epochs)...")
            demo_config = Config()
            demo_config.NUM_EPOCHS = 5
            demo_config.BATCH_SIZE = 4
            demo_config.IMG_SIZE = 64

            trainer = LeafDiseaseTrainer(demo_config)
            history = trainer.train()

            print(f"\nDemo completed!")
            print(f"To run full training, select option 1.")

        else:
            print(f"Invalid choice. Please enter 1, 2, 3, or 4.")

    except KeyboardInterrupt:
        print(f"\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\nProgram finished.")


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行主函数
    main()