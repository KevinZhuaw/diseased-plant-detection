# main_fixed.py
"""
SimCLR叶片病害检测系统 - 修复AP计算问题
"""

import os
import sys
import random
import time
import datetime
import warnings
from PIL import Image
from pathlib import Path
import json

# 设置环境变量避免CUDA内存问题
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

warnings.filterwarnings('ignore')

print("=" * 70)
print("SimCLR叶片病害检测系统 - 完整评估版")
print("=" * 70)

# ========== 导入库 ==========
try:
    import numpy as np

    print(f"✅ NumPy 版本: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy 导入失败: {e}")
    exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from torch.cuda.amp import GradScaler, autocast

    # 设置PyTorch选项
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  显存总量: {total_memory:.1f} GB")

except ImportError as e:
    print(f"❌ PyTorch 导入失败: {e}")
    exit(1)

try:
    from tqdm import tqdm
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
        average_precision_score, precision_recall_curve, roc_curve, auc
    )
    from sklearn.preprocessing import StandardScaler, label_binarize
    import pandas as pd

    print("✅ 其他依赖导入成功")
except ImportError as e:
    print(f"❌ 依赖导入失败: {e}")
    exit(1)


# ========== 配置参数 ==========
class Config:
    # 数据集路径
    DATASET_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\balanced-crop-dataset-new"

    # 图像参数
    IMAGE_SIZE = 224
    SAMPLE_RATIO = 0.1  # 使用10%的数据进行测试，避免内存问题

    # SimCLR训练参数
    EPOCHS = 5  # 先训练5个epoch进行测试
    BATCH_SIZE = 32  # 减小批次大小
    LEARNING_RATE = 0.03
    WEIGHT_DECAY = 1e-4
    TEMPERATURE = 0.07

    # 分类器训练参数
    CLASSIFIER_EPOCHS = 10  # 减少分类器训练轮次
    CLASSIFIER_LR = 0.01

    # 模型参数
    PROJECTION_DIM = 128
    HIDDEN_DIM = 512

    # 硬件参数
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0  # 虚拟环境中避免多进程问题
    PIN_MEMORY = False

    # 评估参数
    THRESHOLD = 0.5  # 分类阈值

    def __init__(self):
        # 创建时间戳的唯一保存目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.SAVE_DIR = f"./simclr_results_{timestamp}"
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.SAVE_DIR, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.SAVE_DIR, "models"), exist_ok=True)

        # 结果文件路径
        self.RESULTS_TXT = os.path.join(self.SAVE_DIR, "evaluation_results.txt")

        print(f"📁 创建新的保存目录: {self.SAVE_DIR}")

        # 检查数据集目录
        print(f"\n📂 检查数据集目录: {self.DATASET_DIR}")
        if not os.path.exists(self.DATASET_DIR):
            print(f"❌ 数据集目录不存在!")
            return

    def print_config(self):
        print("=" * 70)
        print("📊 配置参数:")
        print(f"  图像尺寸: {self.IMAGE_SIZE}x{self.IMAGE_SIZE}")
        print(f"  采样比例: {self.SAMPLE_RATIO}")
        print(f"  SimCLR预训练轮次: {self.EPOCHS}")
        print(f"  批次大小: {self.BATCH_SIZE}")
        print(f"  学习率: {self.LEARNING_RATE}")
        print(f"  温度参数: {self.TEMPERATURE}")
        print(f"  保存目录: {self.SAVE_DIR}")
        print(f"  结果文件: {self.RESULTS_TXT}")
        print("=" * 70)


# ========== 数据增强 ==========
class SimCLRDataAugmentation:
    """SimCLR的数据增强策略"""

    def __init__(self, image_size=224):
        self.image_size = image_size
        self.color_jitter = transforms.ColorJitter(
            0.8, 0.8, 0.8, 0.2
        )

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=image_size // 20 * 2 + 1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class LeafDiseaseDataset(Dataset):
    """叶片病害数据集"""

    def __init__(self, data_dir, mode='train', transform=None, sample_ratio=1.0):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.samples = []
        self.labels = []
        self.label_names = ['healthy', 'diseased']

        if mode == 'train':
            base_dir = self.data_dir / 'train'
        else:
            base_dir = self.data_dir / 'test'

        print(f"\n📥 加载{mode}数据...")
        for label_idx, label_name in enumerate(self.label_names):
            dir_path = base_dir / label_name

            if not dir_path.exists():
                print(f"⚠️  目录不存在: {dir_path}")
                continue

            img_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_files.extend(list(dir_path.glob(f'*{ext}')))
                img_files.extend(list(dir_path.glob(f'*{ext.upper()}')))

            img_files = sorted(img_files)
            print(f"  📁 {label_name} - 找到 {len(img_files)} 张图片")

            if sample_ratio < 1.0 and len(img_files) > 0:
                sample_size = max(1, int(len(img_files) * sample_ratio))
                if len(img_files) > sample_size:
                    img_files = random.sample(img_files, sample_size)

            for img_file in img_files:
                self.samples.append(str(img_file))
                self.labels.append(label_idx)

        print(f"✅ {mode}数据加载完成: {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')

            if self.transform is not None:
                if isinstance(self.transform, SimCLRDataAugmentation):
                    image_i, image_j = self.transform(image)
                    return image_i, image_j, label, img_path
                else:
                    image = self.transform(image)
                    return image, label, img_path
            return image, label, img_path
        except Exception as e:
            print(f"⚠️ 加载图像失败 {img_path}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            if isinstance(self.transform, SimCLRDataAugmentation):
                return dummy_image, dummy_image, label, img_path
            else:
                return dummy_image, label, img_path


# ========== SimCLR模型 ==========
class ProjectionHead(nn.Module):
    """SimCLR投影头"""

    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)


class SimCLR(nn.Module):
    """SimCLR模型"""

    def __init__(self, base_model='resnet50', projection_dim=128,
                 hidden_dim=512, temperature=0.07):
        super().__init__()
        self.temperature = temperature

        # 基础编码器
        backbone = models.__dict__[base_model](pretrained=False)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.num_features = backbone.fc.in_features

        # 投影头
        self.projection = ProjectionHead(
            input_dim=self.num_features,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )

    def forward(self, x, return_features=False):
        """前向传播"""
        h = self.encoder(x)
        h = torch.flatten(h, 1)

        if return_features:
            return h
        else:
            z = self.projection(h)
            return z

    def contrastive_loss(self, z_i, z_j):
        """NT-Xent损失函数"""
        batch_size = z_i.size(0)

        # 拼接特征
        z = torch.cat([z_i, z_j], dim=0)

        # 计算相似度矩阵
        similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        # 创建正样本和负样本掩码
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        positives = similarity_matrix[mask]

        # 排除对角线元素
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

        # 计算损失
        logits = positives / self.temperature
        neg_logits = similarity_matrix / self.temperature

        # 计算交叉熵损失
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        loss = F.cross_entropy(torch.cat([logits.unsqueeze(1), neg_logits], dim=1), labels)

        return loss


# ========== 训练器类 ==========
class SimCLRTrainer:
    """SimCLR训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # 清空GPU缓存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # 创建模型
        self.model = SimCLR(
            projection_dim=128,
            hidden_dim=config.HIDDEN_DIM,
            temperature=config.TEMPERATURE
        ).to(self.device)

        # 使用DataParallel如果有多GPU
        if torch.cuda.device_count() > 1:
            print(f"🚀 使用 {torch.cuda.device_count()} 个GPU")
            self.model = nn.DataParallel(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🔧 创建SimCLR模型: 总参数: {total_params:,}")

        # 优化器和损失函数
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.EPOCHS,
            eta_min=0
        )

        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        self.losses = []

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.EPOCHS}')
        for batch_idx, (images_i, images_j, _, _) in enumerate(pbar):
            images_i = images_i.to(self.device, non_blocking=False)
            images_j = images_j.to(self.device, non_blocking=False)

            if self.scaler is not None:
                with autocast():
                    z_i = self.model(images_i)
                    z_j = self.model(images_j)

                    # 修复：使用正确的模型实例调用contrastive_loss
                    if isinstance(self.model, nn.DataParallel):
                        loss = self.model.module.contrastive_loss(z_i, z_j)
                    else:
                        loss = self.model.contrastive_loss(z_i, z_j)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                z_i = self.model(images_i)
                z_j = self.model(images_j)

                # 修复：使用正确的模型实例调用contrastive_loss
                if isinstance(self.model, nn.DataParallel):
                    loss = self.model.module.contrastive_loss(z_i, z_j)
                else:
                    loss = self.model.contrastive_loss(z_i, z_j)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # 显示进度
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        self.scheduler.step()

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss

    def train(self, train_loader):
        """训练循环"""
        print("🚀 开始SimCLR预训练...")
        best_loss = float('inf')

        for epoch in range(self.config.EPOCHS):
            start_time = time.time()
            loss = self.train_epoch(train_loader, epoch)
            self.losses.append(loss)
            epoch_time = time.time() - start_time

            print(f"  Epoch {epoch + 1}/{self.config.EPOCHS}: "
                  f"loss={loss:.4f}, time={epoch_time:.1f}s")

            if loss < best_loss:
                best_loss = loss
                self.save_checkpoint("simclr_best.pth")

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"simclr_epoch_{epoch + 1}.pth")

        print(f"✅ SimCLR预训练完成! 最佳损失: {best_loss:.4f}")
        self.save_checkpoint("simclr_final.pth")
        return self.losses

    def save_checkpoint(self, filename):
        """保存检查点"""
        model_state_dict = self.model.state_dict()
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()

        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': self.losses,
            'config': self.config.__dict__
        }

        save_path = os.path.join(self.config.SAVE_DIR, "models", filename)
        torch.save(checkpoint, save_path)
        print(f"💾 模型已保存: {save_path}")

    def extract_features(self, data_loader):
        """提取特征"""
        self.model.eval()
        features_list = []
        labels_list = []
        image_paths = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="提取特征"):
                if len(batch) == 4:
                    images, _, labels, paths = batch
                else:
                    images, labels, paths = batch

                images = images.to(self.device, non_blocking=False)

                # 获取模型（如果是DataParallel）
                model_to_use = self.model
                if isinstance(self.model, nn.DataParallel):
                    model_to_use = self.model.module

                features = model_to_use(images, return_features=True)

                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())
                image_paths.extend(paths)

        features = np.vstack(features_list) if features_list else np.array([])
        labels = np.concatenate(labels_list) if labels_list else np.array([])

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return features, labels, image_paths


# ========== 分类器 ==========
class SimpleClassifier(nn.Module):
    """简单分类器"""

    def __init__(self, input_dim, num_classes=2, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ClassifierTrainer:
    """分类器训练器"""

    def __init__(self, input_dim, num_classes=2, device='cuda', save_dir='./'):
        self.device = torch.device(device)
        self.save_dir = save_dir
        self.model = SimpleClassifier(input_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.best_acc = 0

    def train(self, train_features, train_labels, val_features=None, val_labels=None,
              epochs=50, lr=0.01, batch_size=128):
        """训练分类器"""
        print("🎯 训练分类器...")

        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(train_features),
            torch.LongTensor(train_labels)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device, non_blocking=False)
                batch_labels = batch_labels.to(self.device, non_blocking=False)

                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()
            train_acc = self.evaluate(train_features, train_labels)

            if val_features is not None and val_labels is not None:
                val_acc = self.evaluate(val_features, val_labels)
                val_accuracies.append(val_acc)

                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.save_best_model()

                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs}: "
                          f"loss={epoch_loss / len(train_loader):.4f}, "
                          f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs}: "
                          f"loss={epoch_loss / len(train_loader):.4f}, "
                          f"train_acc={train_acc:.4f}")

            train_losses.append(epoch_loss / len(train_loader))

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"✅ 分类器训练完成")
        if val_features is not None:
            print(f"   最佳验证准确率: {self.best_acc:.4f}")

        return train_losses, val_accuracies

    def evaluate(self, features, labels):
        """评估分类器"""
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device, non_blocking=False)
            outputs = self.model(features_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()
        accuracy = accuracy_score(labels, predictions)
        return accuracy

    def predict(self, features):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device, non_blocking=False)
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            predictions = outputs.argmax(dim=1).cpu().numpy()
        return predictions, probabilities

    def save_best_model(self):
        """保存最佳模型"""
        model_path = os.path.join(self.save_dir, "models", "classifier_best.pth")
        torch.save(self.model.state_dict(), model_path)

    def load_best_model(self):
        """加载最佳模型"""
        model_path = os.path.join(self.save_dir, "models", "classifier_best.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ 分类器最佳模型已加载: {model_path}")
            return True
        else:
            print(f"⚠️  分类器最佳模型不存在: {model_path}")
            return False


# ========== 评估模块 ==========
class ModelEvaluator:
    """模型评估器"""

    def __init__(self, config):
        self.config = config
        self.label_names = ['健康叶片', '病叶']

    def calculate_all_metrics(self, y_true, y_pred, y_prob=None):
        """计算所有评估指标"""
        print("\n📊 计算评估指标...")

        metrics = {}

        # 1. 四大基本指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)

        # 2. 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # 3. 分类报告
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.label_names, output_dict=True, zero_division=0
        )

        # 4. mAP (平均精度均值) - 修复：处理二分类情况
        if y_prob is not None:
            try:
                # 对于二分类问题，average_precision_score默认返回一个标量
                metrics['ap'] = average_precision_score(y_true, y_prob[:, 1])
                metrics['map'] = metrics['ap']  # 对于二分类，AP就是mAP

                # ROC曲线和AUC
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                metrics['roc_auc'] = auc(fpr, tpr)
                metrics['roc_curve'] = (fpr, tpr)

                # 精确率-召回率曲线
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob[:, 1])
                metrics['pr_curve'] = (precision_curve, recall_curve)
                metrics['pr_auc'] = auc(recall_curve, precision_curve)
            except Exception as e:
                print(f"⚠️  计算mAP和ROC时出错: {e}")
                metrics['map'] = 0.0
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0

        return metrics

    def create_confusion_matrix_plot(self, cm, save_path):
        """创建混淆矩阵可视化"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_names,
                    yticklabels=self.label_names)
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ 混淆矩阵图已保存: {save_path}")

    def create_roc_curve_plot(self, fpr, tpr, roc_auc, save_path):
        """创建ROC曲线图"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ ROC曲线图已保存: {save_path}")

    def create_pr_curve_plot(self, precision, recall, ap, save_path):
        """创建精确率-召回率曲线图"""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2,
                 label=f'PR曲线 (AP = {ap:.3f})')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ PR曲线图已保存: {save_path}")

    def save_results_to_txt(self, metrics, additional_info=None):
        """将结果保存到txt文件"""
        print(f"\n💾 保存结果到文件: {self.config.RESULTS_TXT}")

        with open(self.config.RESULTS_TXT, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SimCLR叶片病害检测系统 - 评估结果\n")
            f.write("=" * 70 + "\n\n")

            # 1. 基本信息和时间戳
            f.write(f"评估时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"保存目录: {self.config.SAVE_DIR}\n")
            f.write(f"数据集: {self.config.DATASET_DIR}\n\n")

            # 2. 配置信息
            f.write("-" * 50 + "\n")
            f.write("📊 配置参数:\n")
            f.write("-" * 50 + "\n")
            f.write(f"图像尺寸: {self.config.IMAGE_SIZE}x{self.config.IMAGE_SIZE}\n")
            f.write(f"SimCLR轮次: {self.config.EPOCHS}\n")
            f.write(f"批次大小: {self.config.BATCH_SIZE}\n")
            f.write(f"学习率: {self.config.LEARNING_RATE}\n")
            f.write(f"温度参数: {self.config.TEMPERATURE}\n\n")

            # 3. 四大基本指标
            f.write("-" * 50 + "\n")
            f.write("📈 四大基本指标:\n")
            f.write("-" * 50 + "\n")
            f.write(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}\n")
            f.write(f"精确率 (Precision): {metrics['precision']:.4f}\n")
            f.write(f"召回率 (Recall):    {metrics['recall']:.4f}\n")
            f.write(f"F1分数:            {metrics['f1_score']:.4f}\n\n")

            # 4. mAP和AUC
            f.write("-" * 50 + "\n")
            f.write("📊 mAP和AUC指标:\n")
            f.write("-" * 50 + "\n")
            if 'map' in metrics and metrics['map'] > 0:
                f.write(f"平均精度 (AP): {metrics['map']:.4f}\n")
            if 'pr_auc' in metrics and metrics['pr_auc'] > 0:
                f.write(f"PR曲线下面积: {metrics['pr_auc']:.4f}\n")
            if 'roc_auc' in metrics and metrics['roc_auc'] > 0:
                f.write(f"ROC曲线下面积 (AUC): {metrics['roc_auc']:.4f}\n")
            f.write("\n")

            # 5. 混淆矩阵
            f.write("-" * 50 + "\n")
            f.write("📋 混淆矩阵:\n")
            f.write("-" * 50 + "\n")
            cm = metrics['confusion_matrix']
            f.write(f"        预测健康 | 预测病害\n")
            f.write(f"真实健康: {cm[0, 0]:6d} | {cm[0, 1]:6d}\n")
            f.write(f"真实病害: {cm[1, 0]:6d} | {cm[1, 1]:6d}\n")

            # 计算额外统计量
            tn, fp, fn, tp = cm.ravel()
            f.write(f"\n详细统计:\n")
            f.write(f"  真阴性 (TN): {tn}\n")
            f.write(f"  假阳性 (FP): {fp}\n")
            f.write(f"  假阴性 (FN): {fn}\n")
            f.write(f"  真阳性 (TP): {tp}\n\n")

            # 6. 分类报告
            f.write("-" * 50 + "\n")
            f.write("📋 详细分类报告:\n")
            f.write("-" * 50 + "\n")
            report = metrics['classification_report']

            # 写入每个类别的报告
            for label_name in self.label_names:
                if label_name in report:
                    f.write(f"{label_name}:\n")
                    f.write(f"  精确率: {report[label_name]['precision']:.4f}\n")
                    f.write(f"  召回率: {report[label_name]['recall']:.4f}\n")
                    f.write(f"  F1分数: {report[label_name]['f1-score']:.4f}\n")
                    f.write(f"  支持数: {report[label_name]['support']}\n")
                    f.write("\n")

            # 写入总体统计
            if 'accuracy' in report:
                f.write(f"总体准确率: {report['accuracy']:.4f}\n")
            if 'macro avg' in report:
                f.write(f"宏平均: 精确率={report['macro avg']['precision']:.4f}, "
                        f"召回率={report['macro avg']['recall']:.4f}, "
                        f"F1={report['macro avg']['f1-score']:.4f}\n")
            if 'weighted avg' in report:
                f.write(f"加权平均: 精确率={report['weighted avg']['precision']:.4f}, "
                        f"召回率={report['weighted avg']['recall']:.4f}, "
                        f"F1={report['weighted avg']['f1-score']:.4f}\n")
            f.write("\n")

            # 7. 额外信息
            if additional_info:
                f.write("-" * 50 + "\n")
                f.write("📝 额外信息:\n")
                f.write("-" * 50 + "\n")
                for key, value in additional_info.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

            # 8. 文件位置信息
            f.write("-" * 50 + "\n")
            f.write("📁 生成的文件:\n")
            f.write("-" * 50 + "\n")
            f.write(f"1. 结果文件: {self.config.RESULTS_TXT}\n")
            f.write(f"2. 混淆矩阵图: {os.path.join(self.config.SAVE_DIR, 'images', 'confusion_matrix.png')}\n")
            if 'roc_curve' in metrics and metrics['roc_auc'] > 0:
                f.write(f"3. ROC曲线图: {os.path.join(self.config.SAVE_DIR, 'images', 'roc_curve.png')}\n")
            if 'pr_curve' in metrics:
                f.write(f"4. PR曲线图: {os.path.join(self.config.SAVE_DIR, 'images', 'pr_curve.png')}\n")
            f.write(f"5. 模型文件: {os.path.join(self.config.SAVE_DIR, 'models')}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("评估完成\n")
            f.write("=" * 70 + "\n")

        print(f"✅ 评估结果已保存到: {self.config.RESULTS_TXT}")

        # 同时在屏幕上显示关键结果
        print("\n" + "=" * 70)
        print("关键评估结果:")
        print("=" * 70)
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1_score']:.4f}")
        if 'map' in metrics and metrics['map'] > 0:
            print(f"AP:    {metrics['map']:.4f}")
        if 'pr_auc' in metrics and metrics['pr_auc'] > 0:
            print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        if 'roc_auc' in metrics and metrics['roc_auc'] > 0:
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    def create_visualizations(self, metrics):
        """创建所有可视化图表"""
        # 1. 混淆矩阵图
        cm_path = os.path.join(self.config.SAVE_DIR, "images", "confusion_matrix.png")
        self.create_confusion_matrix_plot(metrics['confusion_matrix'], cm_path)

        # 2. ROC曲线图
        if 'roc_curve' in metrics and metrics.get('roc_auc', 0) > 0:
            roc_path = os.path.join(self.config.SAVE_DIR, "images", "roc_curve.png")
            fpr, tpr = metrics['roc_curve']
            self.create_roc_curve_plot(fpr, tpr, metrics['roc_auc'], roc_path)

        # 3. PR曲线图
        if 'pr_curve' in metrics:
            pr_path = os.path.join(self.config.SAVE_DIR, "images", "pr_curve.png")
            precision, recall = metrics['pr_curve']
            # 使用AP或PR-AUC
            ap = metrics.get('ap', metrics.get('map', 0.0))
            self.create_pr_curve_plot(precision, recall, ap, pr_path)


# ========== 主训练流程 ==========
def main():
    config = Config()
    config.print_config()

    if not os.path.exists(config.DATASET_DIR):
        print(f"\n❌ 错误: 数据集目录不存在: {config.DATASET_DIR}")
        return 1

    # 数据增强
    augmentation = SimCLRDataAugmentation(image_size=config.IMAGE_SIZE)

    print("\n📊 加载数据集...")
    try:
        # 使用较小的采样比例避免内存问题
        train_dataset = LeafDiseaseDataset(
            config.DATASET_DIR,
            mode='train',
            transform=augmentation,
            sample_ratio=min(0.1, config.SAMPLE_RATIO)  # 最多使用10%的数据
        )

        test_dataset = LeafDiseaseDataset(
            config.DATASET_DIR,
            mode='test',
            transform=transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            sample_ratio=min(0.05, config.SAMPLE_RATIO)  # 最多使用5%的测试数据
        )
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("❌ 错误: 数据集为空")
        return 1

    print(f"\n✅ 数据集加载完成")
    print(f"  训练集: {len(train_dataset)} 张图片")
    print(f"  测试集: {len(test_dataset)} 张图片")

    # 创建DataLoader
    train_batch_size = min(config.BATCH_SIZE, len(train_dataset))
    test_batch_size = min(config.BATCH_SIZE, len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    try:
        # ========== SimCLR预训练 ==========
        print("\n" + "=" * 70)
        print("🚀 开始SimCLR自监督预训练")
        print("=" * 70)

        simclr_trainer = SimCLRTrainer(config)
        simclr_losses = simclr_trainer.train(train_loader)

        # ========== 提取特征 ==========
        print("\n" + "=" * 70)
        print("🔍 提取特征")
        print("=" * 70)

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("提取训练特征...")
        train_features_dataset = LeafDiseaseDataset(
            config.DATASET_DIR,
            mode='train',
            transform=transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            sample_ratio=min(0.05, config.SAMPLE_RATIO)  # 使用更少的数据提取特征
        )

        train_features_loader = DataLoader(
            train_features_dataset,
            batch_size=min(32, len(train_features_dataset)),
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        train_features, train_labels, _ = simclr_trainer.extract_features(train_features_loader)

        print("提取测试特征...")
        test_features, test_labels, _ = simclr_trainer.extract_features(test_loader)

        if len(train_features) == 0 or len(test_features) == 0:
            print("❌ 错误: 特征提取失败")
            return 1

        print(f"\n✅ 特征提取完成:")
        print(f"  训练特征: {train_features.shape}")
        print(f"  测试特征: {test_features.shape}")

        # ========== 特征标准化 ==========
        print("\n🔧 特征标准化...")
        scaler = StandardScaler()
        train_features_normalized = scaler.fit_transform(train_features)
        test_features_normalized = scaler.transform(test_features)
        print("✅ 特征标准化完成")

        # ========== 训练分类器 ==========
        print("\n" + "=" * 70)
        print("🎯 训练分类器")
        print("=" * 70)

        classifier_trainer = ClassifierTrainer(
            input_dim=train_features_normalized.shape[1],
            num_classes=2,
            device=config.DEVICE,
            save_dir=config.SAVE_DIR
        )

        classifier_losses, classifier_accuracies = classifier_trainer.train(
            train_features_normalized,
            train_labels,
            test_features_normalized,
            test_labels,
            epochs=config.CLASSIFIER_EPOCHS,
            lr=config.CLASSIFIER_LR,
            batch_size=64
        )

        classifier_trainer.load_best_model()

        # ========== 评估模型 ==========
        print("\n" + "=" * 70)
        print("📊 评估模型性能")
        print("=" * 70)

        # 获取预测结果
        test_predictions, test_probabilities = classifier_trainer.predict(test_features_normalized)

        # 创建评估器并计算所有指标
        evaluator = ModelEvaluator(config)
        metrics = evaluator.calculate_all_metrics(test_labels, test_predictions, test_probabilities)

        # 创建可视化图表
        evaluator.create_visualizations(metrics)

        # 准备额外信息
        additional_info = {
            "训练集大小": len(train_dataset),
            "测试集大小": len(test_dataset),
            "特征维度": train_features_normalized.shape[1],
            "最佳验证准确率": f"{classifier_trainer.best_acc:.4f}",
            "SimCLR训练轮次": config.EPOCHS,
            "分类器训练轮次": config.CLASSIFIER_EPOCHS
        }

        # 保存所有结果到txt文件
        evaluator.save_results_to_txt(metrics, additional_info)

        # ========== 保存详细结果 ==========
        print("\n💾 保存详细结果...")

        # 保存原始数据
        results = {
            'test_labels': test_labels,
            'test_predictions': test_predictions,
            'test_probabilities': test_probabilities,
            'metrics': metrics,
            'train_features_shape': train_features.shape,
            'test_features_shape': test_features.shape,
            'simclr_losses': simclr_losses,
            'classifier_losses': classifier_losses,
            'classifier_accuracies': classifier_accuracies,
            'config': config.__dict__
        }

        results_path = os.path.join(config.SAVE_DIR, "detailed_results.npz")
        np.savez(results_path, **results)

        print(f"✅ 详细结果已保存: {results_path}")
        print(f"\n🎉 SimCLR叶片病害检测完成!")

        # 显示结果文件路径
        print(f"\n📁 所有结果文件保存在: {config.SAVE_DIR}")
        print(f"📄 评估报告: {config.RESULTS_TXT}")

        return 0

    except torch.cuda.OutOfMemoryError:
        print(f"\n❌ GPU内存不足!")
        print("建议:")
        print(f"1. 减小 BATCH_SIZE (当前: {config.BATCH_SIZE})")
        print(f"2. 减小 SAMPLE_RATIO (当前: {config.SAMPLE_RATIO})")
        print(f"3. 减小 IMAGE_SIZE (当前: {config.IMAGE_SIZE})")
        return 1

    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ========== 命令行接口 ==========
if __name__ == "__main__":
    # 检查虚拟环境
    print("检查Python环境...")
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")

    # 运行主程序
    exit_code = main()

    if exit_code == 0:
        print("\n" + "=" * 70)
        print("✅ 程序成功完成!")
        print("=" * 70)
    else:
        print(f"\n❌ 程序失败，退出码: {exit_code}")

    sys.exit(exit_code)