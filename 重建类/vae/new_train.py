import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import time
from datetime import datetime
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns
from collections import Counter
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


# ==================== 配置设置 ====================
class Config:
    """训练配置类"""
    # 路径配置
    DATASET_PATH = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"
    RESULTS_PATH = r"E:\庄楷文\叶片病虫害识别\baseline\results\ae_fixed"

    # 添加时间戳，避免冲突
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_PATH = f"{RESULTS_PATH}_{timestamp}"

    # 模型配置
    IMG_SIZE = 224  # 固定为224×224
    LATENT_DIM = 256  # 潜在空间维度
    CHANNELS = 3  # RGB图像

    # 训练配置
    BATCH_SIZE = 16  # 减少批处理大小，节省显存
    NUM_EPOCHS = 30  # 训练轮数
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2  # 减少数据加载工作线程
    USE_AMP = True  # 混合精度训练

    # 早停设置
    PATIENCE = 10


# 打印设备信息
print("=" * 60)
print("🚀 AutoEncoder Training with Classification Evaluation")
print("=" * 60)
print(f"💻 Device: {Config.DEVICE}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"🖼️ Image Size: {Config.IMG_SIZE}")
print(f"📦 Batch Size: {Config.BATCH_SIZE}")
print(f"🎯 Latent Dim: {Config.LATENT_DIM}")
print(f"📁 Results Path: {Config.RESULTS_PATH}")
print("=" * 60)


# ==================== 数据集类 ====================
class AutoencoderDataset(Dataset):
    """增强的数据集类，包含真实标签"""

    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.mode = mode

        # 收集所有图像和标签
        if os.path.exists(self.root_dir):
            classes = sorted([d for d in os.listdir(self.root_dir)
                              if os.path.isdir(os.path.join(self.root_dir, d))])

            # 创建标签映射
            self.label_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
            self.idx_to_label = {idx: class_name for idx, class_name in enumerate(classes)}

            for class_name in classes:
                class_path = os.path.join(self.root_dir, class_name)
                class_idx = self.label_to_idx[class_name]

                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.image_paths.append(os.path.join(class_path, img_file))
                        self.labels.append(class_idx)

        print(f"{mode.upper()}: Loaded {len(self.image_paths)} images from {len(self.label_to_idx)} classes")

        # 打印类别分布
        if self.labels:
            label_counts = Counter(self.labels)
            print(f"  Class distribution:")
            for label_idx, count in label_counts.items():
                label_name = self.idx_to_label[label_idx]
                print(f"    {label_name}: {count} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, img_path  # 返回图像、标签和路径
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回黑色图像
            if self.transform:
                image = self.transform(Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), color='black'))
            return image, label, img_path

    def get_class_names(self):
        """获取类别名称列表"""
        return [self.idx_to_label[i] for i in range(len(self.idx_to_label))]


# ==================== 完整的自编码器分类模型 ====================
class CompleteAutoencoder(nn.Module):
    """完整的自编码器+分类器模型"""

    def __init__(self, input_channels=3, latent_dim=512, num_classes=22):
        super(CompleteAutoencoder, self).__init__()

        # 编码器 - 使用与之前训练相同的架构
        self.encoder = nn.Sequential(
            # 输入: (3, 224, 224)
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 56, 56)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (256, 28, 28)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # (1024, 7, 7)
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # 解码器
        self.decoder = nn.Sequential(
            # 输入: (1024, 7, 7)
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (256, 28, 28)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 56, 56)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),  # (3, 224, 224)
            nn.Sigmoid()
        )

    def forward(self, x, mode='all'):
        """前向传播"""
        encoded = self.encoder(x)

        if mode == 'encode':
            return encoded

        # 分类分支
        pooled = self.global_avg_pool(encoded)
        flattened = pooled.view(pooled.size(0), -1)
        classification = self.classifier(flattened)

        if mode == 'classify':
            return classification

        # 解码分支
        decoded = self.decoder(encoded)

        if mode == 'decode':
            return decoded

        return decoded, classification

    def encode(self, x):
        """仅编码"""
        return self.encoder(x)

    def classify(self, x):
        """仅分类"""
        encoded = self.encoder(x)
        pooled = self.global_avg_pool(encoded)
        flattened = pooled.view(pooled.size(0), -1)
        return self.classifier(flattened)


# ==================== 简化的训练器 ====================
class SimpleTrainer:
    def __init__(self, model, config, train_dataset):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.train_dataset = train_dataset
        self.class_names = train_dataset.get_class_names() if hasattr(train_dataset, 'get_class_names') else None

        # 创建结果目录
        self.results_dir = config.RESULTS_PATH
        os.makedirs(self.results_dir, exist_ok=True)

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None

        # 损失函数和优化器
        self.recon_criterion = nn.MSELoss()  # 重建损失
        self.class_criterion = nn.CrossEntropyLoss()  # 分类损失
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                    weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # 训练记录
        self.train_recon_losses = []
        self.train_class_losses = []
        self.train_accuracies = []
        self.val_recon_losses = []
        self.val_class_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')

        # 保存配置
        self._save_config()

    def _save_config(self):
        config_dict = {
            "device": str(self.device),
            "img_size": self.config.IMG_SIZE,
            "batch_size": self.config.BATCH_SIZE,
            "latent_dim": self.config.LATENT_DIM,
            "num_epochs": self.config.NUM_EPOCHS,
            "learning_rate": self.config.LEARNING_RATE,
            "use_amp": self.config.USE_AMP,
            "class_names": self.class_names,
            "num_classes": len(self.class_names) if self.class_names else 0,
            "model_architecture": "CompleteAutoencoder",
            "model_channels": [64, 128, 256, 512, 1024],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(self.results_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_recon_loss = 0
        total_class_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 混合精度训练
            if self.config.USE_AMP and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    reconstructed, classification = self.model(images)
                    recon_loss = self.recon_criterion(reconstructed, images)
                    class_loss = self.class_criterion(classification, labels)
                    loss = recon_loss + class_loss * 2.0  # 分类损失权重更高

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                reconstructed, classification = self.model(images)
                recon_loss = self.recon_criterion(reconstructed, images)
                class_loss = self.class_criterion(classification, labels)
                loss = recon_loss + class_loss * 2.0

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()

            # 计算准确率
            _, predicted = torch.max(classification, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # 释放内存
            del reconstructed, classification, loss
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # 打印进度
            if batch_idx % 50 == 0:
                accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
                print(f"  Batch {batch_idx}/{len(train_loader)}, "
                      f"Recon Loss: {recon_loss.item():.6f}, "
                      f"Class Loss: {class_loss.item():.6f}, "
                      f"Acc: {accuracy:.2f}%")
                if self.device.type == 'cuda':
                    memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                    memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                    print(f"    GPU Memory: {memory_allocated:.2f}/{memory_reserved:.2f} GB")

        avg_recon_loss = total_recon_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_class_loss = total_class_loss / len(train_loader) if len(train_loader) > 0 else 0
        accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0

        return avg_recon_loss, avg_class_loss, accuracy

    def validate(self, val_loader):
        self.model.eval()
        total_recon_loss = 0
        total_class_loss = 0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                reconstructed, classification = self.model(images)
                recon_loss = self.recon_criterion(reconstructed, images)
                class_loss = self.class_criterion(classification, labels)

                total_recon_loss += recon_loss.item()
                total_class_loss += class_loss.item()

                # 计算准确率
                _, predicted = torch.max(classification, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                # 收集预测结果
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                # 释放内存
                del reconstructed, classification
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        avg_recon_loss = total_recon_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_class_loss = total_class_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0

        return avg_recon_loss, avg_class_loss, accuracy, all_labels, all_predictions

    def train(self, train_loader, val_loader):
        print(f"\n开始训练，设备: {self.device}")
        if self.class_names:
            print(f"类别数量: {len(self.class_names)}")
            print(f"类别: {self.class_names}")

        start_time = time.time()
        log_file = os.path.join(self.results_dir, "training_log.txt")

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"设备: {self.device}\n")
            f.write(f"类别数量: {len(self.class_names) if self.class_names else 'N/A'}\n")
            f.write(f"训练集大小: {len(train_loader.dataset)}\n")
            f.write(f"验证集大小: {len(val_loader.dataset)}\n")
            f.write(f"模型架构: CompleteAutoencoder\n")
            f.write("=" * 60 + "\n\n")

        # 清除CUDA缓存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"\n📅 Epoch [{epoch}/{self.config.NUM_EPOCHS}]")

            # 训练
            train_recon_loss, train_class_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.train_recon_losses.append(train_recon_loss)
            self.train_class_losses.append(train_class_loss)
            self.train_accuracies.append(train_acc)

            # 验证
            val_recon_loss, val_class_loss, val_acc, val_labels, val_preds = self.validate(val_loader)
            self.val_recon_losses.append(val_recon_loss)
            self.val_class_losses.append(val_class_loss)
            self.val_accuracies.append(val_acc)

            # 计算验证集上的评估指标
            if epoch % 5 == 0 or epoch == 1:
                self._evaluate_epoch(val_labels, val_preds, epoch)

            # 学习率调整
            self.scheduler.step(val_class_loss)

            # 保存最佳模型（基于验证准确率）
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_val_loss = val_class_loss
                model_path = os.path.join(self.results_dir, f"best_model_epoch{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_recon_loss': train_recon_loss,
                    'train_class_loss': train_class_loss,
                    'train_accuracy': train_acc,
                    'val_recon_loss': val_recon_loss,
                    'val_class_loss': val_class_loss,
                    'val_accuracy': val_acc,
                    'class_names': self.class_names,
                    'model_architecture': 'CompleteAutoencoder',
                }, model_path)
                print(f"✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")

            # 记录日志
            current_lr = self.optimizer.param_groups[0]['lr']
            log_msg = (f"Epoch {epoch:03d}/{self.config.NUM_EPOCHS} | "
                       f"Train: Recon Loss: {train_recon_loss:.6f}, "
                       f"Class Loss: {train_class_loss:.6f}, "
                       f"Acc: {train_acc:.2f}% | "
                       f"Val: Recon Loss: {val_recon_loss:.6f}, "
                       f"Class Loss: {val_class_loss:.6f}, "
                       f"Acc: {val_acc:.2f}% | "
                       f"LR: {current_lr:.6f} | "
                       f"Best Val Acc: {self.best_val_accuracy:.2f}%")

            print(log_msg)

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + "\n")

            # 定期保存检查点
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(self.results_dir, f"checkpoint_epoch{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_recon_losses': self.train_recon_losses,
                    'train_class_losses': self.train_class_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_recon_losses': self.val_recon_losses,
                    'val_class_losses': self.val_class_losses,
                    'val_accuracies': self.val_accuracies,
                    'best_val_accuracy': self.best_val_accuracy,
                    'class_names': self.class_names,
                    'model_architecture': 'CompleteAutoencoder',
                }, checkpoint_path)

            # 清除CUDA缓存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # 保存最终模型
        final_path = os.path.join(self.results_dir, "final_model.pth")
        torch.save({
            'epoch': self.config.NUM_EPOCHS,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_recon_losses': self.train_recon_losses,
            'train_class_losses': self.train_class_losses,
            'train_accuracies': self.train_accuracies,
            'val_recon_losses': self.val_recon_losses,
            'val_class_losses': self.val_class_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy,
            'class_names': self.class_names,
            'model_architecture': 'CompleteAutoencoder',
        }, final_path)

        # 绘制损失曲线
        self.plot_loss_curves()

        # 绘制准确率曲线
        self.plot_accuracy_curves()

        # 保存训练摘要
        self.save_summary(time.time() - start_time)

        return self.train_recon_losses, self.train_class_losses, self.val_recon_losses, self.val_class_losses

    def _evaluate_epoch(self, labels, predictions, epoch):
        """评估单轮性能"""
        if not self.class_names or len(labels) == 0:
            return

        # 计算分类报告
        report = classification_report(labels, predictions,
                                       target_names=self.class_names,
                                       output_dict=True)

        # 保存分类报告
        report_file = os.path.join(self.results_dir, f"classification_report_epoch{epoch}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印主要指标
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

        print(f"  📊 Epoch {epoch} 评估指标:")
        print(f"    准确率: {accuracy:.4f}")
        print(f"    精确率: {precision:.4f}")
        print(f"    召回率: {recall:.4f}")
        print(f"    F1分数: {f1:.4f}")

    def plot_loss_curves(self):
        """绘制损失曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 重建损失
        axes[0].plot(self.train_recon_losses, label='训练重建损失', linewidth=2)
        axes[0].plot(self.val_recon_losses, label='验证重建损失', linewidth=2)
        axes[0].set_xlabel('轮次')
        axes[0].set_ylabel('重建损失 (MSE)')
        axes[0].set_title('重建损失曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 分类损失
        axes[1].plot(self.train_class_losses, label='训练分类损失', linewidth=2)
        axes[1].plot(self.val_class_losses, label='验证分类损失', linewidth=2)
        axes[1].set_xlabel('轮次')
        axes[1].set_ylabel('分类损失 (CrossEntropy)')
        axes[1].set_title('分类损失曲线')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "loss_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 损失曲线已保存")

    def plot_accuracy_curves(self):
        """绘制准确率曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_accuracies, label='训练准确率', linewidth=2)
        plt.plot(self.val_accuracies, label='验证准确率', linewidth=2)
        plt.xlabel('轮次')
        plt.ylabel('准确率 (%)')
        plt.title('训练准确率曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.results_dir, "accuracy_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 准确率曲线已保存")

    def save_summary(self, total_time):
        """保存训练摘要"""
        summary_file = os.path.join(self.results_dir, "training_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("AutoEncoder训练摘要\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"设备: {self.device}\n")
            f.write(f"总训练时间: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)\n")
            f.write(f"总轮次: {len(self.train_accuracies)}\n")
            f.write(f"最佳验证准确率: {self.best_val_accuracy:.2f}%\n")
            f.write(f"最终训练准确率: {self.train_accuracies[-1]:.2f}%\n")
            f.write(f"最终验证准确率: {self.val_accuracies[-1]:.2f}%\n\n")

            if self.class_names:
                f.write(f"类别数量: {len(self.class_names)}\n")
                f.write(f"类别: {', '.join(self.class_names)}\n\n")

            f.write("模型参数统计:\n")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            f.write(f"  总参数: {total_params:,}\n")
            f.write(f"  可训练参数: {trainable_params:,}\n")
            f.write(f"  非训练参数: {total_params - trainable_params:,}\n")


# ==================== 测试评估函数 ====================
def evaluate_model(model, test_loader, results_dir, device, class_names):
    """评估模型性能"""
    print("\n🧪 评估模型中...")

    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    error_samples = []

    recon_criterion = nn.MSELoss()
    total_recon_loss = 0

    with torch.no_grad():
        for batch_idx, (images, labels, img_paths) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            reconstructed, classification = model(images)

            # 计算重建损失
            recon_loss = recon_criterion(reconstructed, images)
            total_recon_loss += recon_loss.item()

            # 获取预测结果
            probs = torch.softmax(classification, dim=1)
            _, predicted = torch.max(classification, 1)

            # 收集数据
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # 收集错误样本
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    error_samples.append({
                        'image_path': img_paths[i],
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item(),
                        'confidence': probs[i][predicted[i]].item(),
                        'original_image': images[i].cpu(),
                        'reconstructed_image': reconstructed[i].cpu()
                    })

            # 打印进度
            if batch_idx % 20 == 0:
                print(f"  处理批次 {batch_idx}/{len(test_loader)}")

            # 释放内存
            del reconstructed, classification, probs
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # 计算平均重建损失
    avg_recon_loss = total_recon_loss / len(test_loader)

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

    # 生成分类报告
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    # 保存结果
    save_evaluation_results(all_labels, all_predictions, all_probs, error_samples,
                            accuracy, precision, recall, f1, avg_recon_loss,
                            cm, report, class_names, results_dir)

    return accuracy, precision, recall, f1, avg_recon_loss


def save_evaluation_results(all_labels, all_predictions, all_probs, error_samples,
                            accuracy, precision, recall, f1, recon_loss,
                            cm, report, class_names, results_dir):
    """保存评估结果"""

    print(f"\n📊 测试结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  平均重建损失: {recon_loss:.6f}")
    print(f"  错误样本数: {len(error_samples)}/{len(all_labels)}")

    # 保存测试结果
    test_file = os.path.join(results_dir, "test_results.txt")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("AutoEncoder测试结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试样本数: {len(all_labels)}\n")
        f.write(f"准确率: {accuracy:.4f}\n")
        f.write(f"精确率: {precision:.4f}\n")
        f.write(f"召回率: {recall:.4f}\n")
        f.write(f"F1分数: {f1:.4f}\n")
        f.write(f"平均重建损失: {recon_loss:.6f}\n")
        f.write(f"错误样本数: {len(error_samples)}/{len(all_labels)}\n")
        f.write(f"错误率: {len(error_samples) / len(all_labels):.2%}\n\n")

        f.write(f"\n类别数量: {len(class_names)}\n")
        f.write(f"类别: {', '.join(class_names)}\n")

    # 保存分类报告
    report_file = os.path.join(results_dir, "classification_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 保存文本分类报告
    report_text = classification_report(all_labels, all_predictions, target_names=class_names)
    with open(os.path.join(results_dir, "classification_report.txt"), 'w', encoding='utf-8') as f:
        f.write("分类报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)

    # 绘制混淆矩阵
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('归一化混淆矩阵')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存错误样本
    if error_samples:
        save_error_samples_visualization(error_samples, class_names, results_dir)
        save_error_samples_csv(error_samples, class_names, results_dir)

    print(f"✅ 评估结果已保存到 {results_dir}")


def save_error_samples_visualization(error_samples, class_names, results_dir, max_samples=20):
    """保存错误样本可视化"""
    num_samples = min(max_samples, len(error_samples))

    if num_samples == 0:
        return

    # 创建子图
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 6, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        sample = error_samples[idx]
        row = idx // n_cols
        col = (idx % n_cols) * 2

        # 原始图像
        original_img = sample['original_image'].permute(1, 2, 0).numpy()
        original_img = np.clip(original_img, 0, 1)

        axes[row, col].imshow(original_img)
        true_label = class_names[sample['true_label']]
        pred_label = class_names[sample['predicted_label']]
        axes[row, col].set_title(f"真实: {true_label}\n预测: {pred_label}\n置信度: {sample['confidence']:.3f}",
                                 fontsize=8)
        axes[row, col].axis('off')

        # 重建图像
        recon_img = sample['reconstructed_image'].permute(1, 2, 0).numpy()
        recon_img = np.clip(recon_img, 0, 1)

        axes[row, col + 1].imshow(recon_img)
        axes[row, col + 1].set_title(f"重建图像", fontsize=8)
        axes[row, col + 1].axis('off')

    # 隐藏多余的子图
    for idx in range(num_samples * 2, n_rows * n_cols * 2):
        row = idx // (n_cols * 2)
        col = idx % (n_cols * 2)
        axes[row, col].axis('off')

    plt.suptitle(f"错误分类样本示例 (共{len(error_samples)}个错误样本)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "error_samples.png"), dpi=200, bbox_inches='tight')
    plt.close()


def save_error_samples_csv(error_samples, class_names, results_dir):
    """保存错误样本到CSV"""
    error_data = []
    for sample in error_samples:
        error_data.append({
            'image_path': sample['image_path'],
            'true_label': class_names[sample['true_label']],
            'predicted_label': class_names[sample['predicted_label']],
            'confidence': sample['confidence'],
            'true_label_id': sample['true_label'],
            'predicted_label_id': sample['predicted_label']
        })

    error_df = pd.DataFrame(error_data)
    error_df.to_csv(os.path.join(results_dir, "error_samples.csv"), index=False, encoding='utf-8-sig')


# ==================== 安全加载模型的函数 ====================
def safe_load_model(model, model_path, device):
    """安全加载模型，处理架构不匹配的情况"""
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # 检查模型架构是否匹配
        if 'model_architecture' in checkpoint:
            expected_architecture = 'CompleteAutoencoder'
            if checkpoint['model_architecture'] != expected_architecture:
                print(f"⚠️  警告: 检查点模型架构 ({checkpoint['model_architecture']}) "
                      f"与当前模型架构 ({expected_architecture}) 不匹配")

        # 获取当前模型的状态字典
        model_dict = model.state_dict()

        # 获取检查点中的状态字典
        checkpoint_dict = checkpoint['model_state_dict']

        # 过滤检查点中与当前模型匹配的参数
        filtered_checkpoint = {}
        for k, v in checkpoint_dict.items():
            if k in model_dict and v.size() == model_dict[k].size():
                filtered_checkpoint[k] = v
            else:
                print(f"⚠️  跳过参数 {k}: 尺寸不匹配或不在当前模型中")

        # 加载匹配的参数
        model_dict.update(filtered_checkpoint)
        model.load_state_dict(model_dict)

        print(f"✅ 成功加载模型 (Epoch {checkpoint.get('epoch', 'N/A')})")
        print(f"   加载了 {len(filtered_checkpoint)}/{len(checkpoint_dict)} 个参数")

        return model, checkpoint

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("⚠️  使用随机初始化的模型")
        return model, None


# ==================== 验证模型尺寸 ====================
def validate_model_dimensions():
    """验证模型输入输出尺寸"""
    print("\n🔍 验证模型尺寸...")

    # 创建测试模型
    test_model = CompleteAutoencoder(input_channels=3, latent_dim=512, num_classes=22)
    test_model.to(Config.DEVICE)

    # 创建测试输入
    test_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)

    # 测试前向传播
    test_model.eval()
    with torch.no_grad():
        recon_output, class_output = test_model(test_input)

    print(f"输入尺寸: {test_input.shape}")
    print(f"重建输出尺寸: {recon_output.shape}")
    print(f"分类输出尺寸: {class_output.shape}")

    # 检查尺寸是否匹配
    if test_input.shape == recon_output.shape:
        print("✅ 模型尺寸正确: 输入输出尺寸匹配")

        # 打印模型参数
        total_params = sum(p.numel() for p in test_model.parameters())
        print(f"📊 模型总参数: {total_params:,}")

        return True
    else:
        print(f"❌ 模型尺寸错误: 输入 {test_input.shape} vs 重建输出 {recon_output.shape}")
        return False


# ==================== 主函数 ====================
def main():
    print("🚀 开始AutoEncoder训练与评估")

    # 清除CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 首先验证模型尺寸
    if not validate_model_dimensions():
        print("❌ 模型尺寸验证失败，请检查模型结构")
        return

    # 检查数据集
    if not os.path.exists(Config.DATASET_PATH):
        print(f"❌ 数据集不存在: {Config.DATASET_PATH}")
        return

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
    ])

    # 加载数据集
    print("\n📂 加载数据集中...")
    try:
        train_dataset = AutoencoderDataset(Config.DATASET_PATH, transform, 'train')
        val_dataset = AutoencoderDataset(Config.DATASET_PATH, transform, 'val')
        test_dataset = AutoencoderDataset(Config.DATASET_PATH, transform, 'test')

        class_names = train_dataset.get_class_names()

        print(f"✅ 训练集: {len(train_dataset)} 张图片")
        print(f"✅ 验证集: {len(val_dataset)} 张图片")
        print(f"✅ 测试集: {len(test_dataset)} 张图片")
        print(f"✅ 类别数量: {len(class_names)}")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE.type == 'cuda'),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE.type == 'cuda')
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )

    # 创建模型
    print("\n🤖 创建模型中...")
    model = CompleteAutoencoder(
        input_channels=Config.CHANNELS,
        latent_dim=512,  # 使用与架构匹配的潜在维度
        num_classes=len(class_names)
    ).to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 总参数: {total_params:,}")
    print(f"📊 可训练参数: {trainable_params:,}")
    print(f"📊 非训练参数: {total_params - trainable_params:,}")

    # 训练
    trainer = SimpleTrainer(model, Config, train_dataset)
    trainer.train(train_loader, val_loader)

    # 测试最佳模型
    print("\n" + "=" * 60)
    print("🧪 测试最佳模型...")

    # 查找最佳模型
    best_models = [f for f in os.listdir(Config.RESULTS_PATH) if f.startswith('best_model_')]

    if best_models:
        # 按epoch编号排序，选择最新的最佳模型
        try:
            best_models_sorted = sorted(best_models, key=lambda x: int(x.split('epoch')[1].split('.')[0]))
            best_model_path = os.path.join(Config.RESULTS_PATH, best_models_sorted[-1])
            print(f"📂 加载最佳模型: {best_model_path}")

            # 安全加载模型
            model, checkpoint = safe_load_model(model, best_model_path, Config.DEVICE)

            if checkpoint and 'class_names' in checkpoint:
                class_names = checkpoint['class_names']
                print(f"✅ 加载类别名称")
        except Exception as e:
            print(f"❌ 加载最佳模型失败: {e}")
            print("⚠️  使用刚训练完的模型进行评估")
    else:
        print("⚠️  未找到最佳模型，使用刚训练完的模型进行评估")

    # 测试重建和分类性能
    accuracy, precision, recall, f1, recon_loss = evaluate_model(
        model, test_loader, Config.RESULTS_PATH, Config.DEVICE, class_names
    )

    # 最终输出
    print("\n" + "=" * 60)
    print("🎉 训练与评估完成!")
    print("=" * 60)
    print(f"📁 结果目录: {Config.RESULTS_PATH}")
    print(f"🏆 最佳验证准确率: {trainer.best_val_accuracy:.2f}%")
    print(f"🧪 测试准确率: {accuracy:.4f}")
    print(f"📊 测试精确率: {precision:.4f}")
    print(f"📊 测试召回率: {recall:.4f}")
    print(f"📊 测试F1分数: {f1:.4f}")
    print(f"📊 类别数量: {len(class_names)}")

    print("\n📁 生成的文件:")
    print(f"  📄 config.json - 训练配置")
    print(f"  📄 training_log.txt - 训练日志")
    print(f"  📊 training_summary.txt - 训练摘要")
    print(f"  📈 loss_curves.png - 损失曲线")
    print(f"  📈 accuracy_curve.png - 准确率曲线")
    print(f"  🤖 best_model_*.pth - 最佳模型权重")
    print(f"  🧪 test_results.txt - 测试结果")
    print(f"  📄 classification_report.json - 分类报告(JSON)")
    print(f"  📄 classification_report.txt - 分类报告(文本)")
    print(f"  📊 confusion_matrix.png - 混淆矩阵")
    print(f"  📊 confusion_matrix_normalized.png - 归一化混淆矩阵")
    print(f"  🖼️ error_samples.png - 错误样本可视化")
    print(f"  📄 error_samples.csv - 错误样本列表")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练错误: {e}")
        import traceback

        traceback.print_exc()