"""
CAE框架 - 兼容旧版本PyTorch的修复版
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ==================== CAE配置设置 ====================
class CAEConfig:
    """CAE训练配置类"""
    # 路径配置
    DATASET_PATH = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"
    RESULTS_PATH = r"E:\庄楷文\叶片病虫害识别\baseline\results\cae_fixed"

    # 添加时间戳，避免冲突
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_PATH = f"{RESULTS_PATH}_{timestamp}"

    # 模型配置
    IMG_SIZE = 128  # 降低图像尺寸以节省内存
    CHANNELS = 3

    # CAE特定配置
    USE_DROPOUT = True
    DROPOUT_RATE = 0.3
    USE_BATCHNORM = True

    # 训练配置
    BATCH_SIZE = 32  # 增加批处理大小，因为数据量很大
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # 学习率调度
    USE_SCHEDULER = True
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5

    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4  # 增加工作线程，因为数据量大
    USE_AMP = True

    # 早停设置
    PATIENCE = 15
    MIN_DELTA = 0.001


print("=" * 60)
print("🚀 Convolutional AutoEncoder (CAE) Training")
print("=" * 60)
print(f"💻 Device: {CAEConfig.DEVICE}")
print(f"🖼️ Image Size: {CAEConfig.IMG_SIZE}")
print(f"📦 Batch Size: {CAEConfig.BATCH_SIZE}")
print(f"📁 Results Path: {CAEConfig.RESULTS_PATH}")
print("=" * 60)


# ==================== 简化的CAE模型 ====================
class SimpleCAE(nn.Module):
    """简化版CAE，减少内存使用"""

    def __init__(self, input_channels=3, num_classes=22):
        super(SimpleCAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            # 输入: (3, 128, 128)
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # 解码器
        self.decoder = nn.Sequential(
            # 输入: (256, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (3, 128, 128)
            nn.Sigmoid()
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x, mode='all'):
        """前向传播"""
        encoded = self.encoder(x)

        if mode == 'encode':
            return encoded

        # 分类分支
        classification = self.classifier(encoded)

        if mode == 'classify':
            return classification

        # 解码分支
        decoded = self.decoder(encoded)

        if mode == 'decode':
            return decoded

        return decoded, classification


# ==================== 数据集类 ====================
class CAEDataset(Dataset):
    """CAE数据集类"""

    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.mode = mode

        if os.path.exists(self.root_dir):
            classes = sorted([d for d in os.listdir(self.root_dir)
                              if os.path.isdir(os.path.join(self.root_dir, d))])

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            if self.transform:
                image = self.transform(Image.new('RGB', (CAEConfig.IMG_SIZE, CAEConfig.IMG_SIZE), color='black'))
            return image, label, img_path

    def get_class_names(self):
        return [self.idx_to_label[i] for i in range(len(self.idx_to_label))]


# ==================== 修复的CAE训练器 ====================
class FixedCAETrainer:
    def __init__(self, model, config, train_dataset):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.train_dataset = train_dataset
        self.class_names = train_dataset.get_class_names()

        # 创建结果目录
        self.results_dir = config.RESULTS_PATH
        os.makedirs(self.results_dir, exist_ok=True)

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.USE_AMP else None

        # 损失函数和优化器
        self.recon_criterion = nn.MSELoss()
        self.class_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                    weight_decay=config.WEIGHT_DECAY)

        # 修复：移除verbose参数
        if config.USE_SCHEDULER:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE
            )
        else:
            self.scheduler = None

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0

        # 保存配置
        self._save_config()

    def _save_config(self):
        config_dict = {
            "device": str(self.device),
            "img_size": self.config.IMG_SIZE,
            "batch_size": self.config.BATCH_SIZE,
            "num_epochs": self.config.NUM_EPOCHS,
            "learning_rate": self.config.LEARNING_RATE,
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
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
                    total_loss = recon_loss * 0.5 + class_loss

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                reconstructed, classification = self.model(images)

                recon_loss = self.recon_criterion(reconstructed, images)
                class_loss = self.class_criterion(classification, labels)
                total_loss = recon_loss * 0.5 + class_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()

            # 计算准确率
            _, predicted = torch.max(classification, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # 打印进度
            if batch_idx % 100 == 0:
                accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
                print(f"  Batch {batch_idx}/{len(train_loader)}, "
                      f"Recon: {recon_loss.item():.4f}, "
                      f"Class: {class_loss.item():.4f}, "
                      f"Acc: {accuracy:.2f}%")

            # 清理内存
            if batch_idx % 200 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0

        return avg_recon_loss, avg_class_loss, accuracy

    def validate(self, val_loader):
        self.model.eval()
        total_recon_loss = 0
        total_class_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                reconstructed, classification = self.model(images)

                recon_loss = self.recon_criterion(reconstructed, images)
                class_loss = self.class_criterion(classification, labels)

                total_recon_loss += recon_loss.item()
                total_class_loss += class_loss.item()

                _, predicted = torch.max(classification, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_recon_loss = total_recon_loss / len(val_loader)
        avg_class_loss = total_class_loss / len(val_loader)
        accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0

        return avg_recon_loss, avg_class_loss, accuracy

    def train(self, train_loader, val_loader):
        print(f"\n开始CAE训练，设备: {self.device}")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        print(f"类别数量: {len(self.class_names)}")

        start_time = time.time()
        log_file = os.path.join(self.results_dir, "training_log.txt")

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"CAE训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"设备: {self.device}\n")
            f.write("=" * 60 + "\n\n")

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"\n📅 Epoch [{epoch}/{self.config.NUM_EPOCHS}]")

            # 训练
            train_recon_loss, train_class_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_recon_loss + train_class_loss)
            self.train_accuracies.append(train_acc)

            # 验证
            val_recon_loss, val_class_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_recon_loss + val_class_loss)
            self.val_accuracies.append(val_acc)

            # 学习率调整
            if self.scheduler:
                self.scheduler.step(val_recon_loss + val_class_loss)

            # 保存最佳模型
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                model_path = os.path.join(self.results_dir, f"best_model_epoch{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'class_names': self.class_names,
                }, model_path)
                print(f"✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")

            # 记录日志
            current_lr = self.optimizer.param_groups[0]['lr']
            log_msg = (f"Epoch {epoch:03d}/{self.config.NUM_EPOCHS} | "
                       f"Train: Recon: {train_recon_loss:.4f}, "
                       f"Class: {train_class_loss:.4f}, "
                       f"Acc: {train_acc:.2f}% | "
                       f"Val: Recon: {val_recon_loss:.4f}, "
                       f"Class: {val_class_loss:.4f}, "
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
                    'train_accuracies': self.train_accuracies,
                    'val_accuracies': self.val_accuracies,
                    'best_val_accuracy': self.best_val_accuracy,
                    'class_names': self.class_names,
                }, checkpoint_path)

        # 保存最终模型
        final_path = os.path.join(self.results_dir, "final_model.pth")
        torch.save({
            'epoch': self.config.NUM_EPOCHS,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy,
            'class_names': self.class_names,
        }, final_path)

        # 绘制曲线
        self.plot_curves()

        # 保存摘要
        self.save_summary(time.time() - start_time)

        return self.train_accuracies, self.val_accuracies

    def plot_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        axes[0].plot(self.train_losses, label='训练损失', linewidth=2)
        axes[0].plot(self.val_losses, label='验证损失', linewidth=2)
        axes[0].set_xlabel('轮次')
        axes[0].set_ylabel('损失')
        axes[0].set_title('损失曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[1].plot(self.train_accuracies, label='训练准确率', linewidth=2)
        axes[1].plot(self.val_accuracies, label='验证准确率', linewidth=2)
        axes[1].set_xlabel('轮次')
        axes[1].set_ylabel('准确率 (%)')
        axes[1].set_title('准确率曲线')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 训练曲线已保存")

    def save_summary(self, total_time):
        """保存训练摘要"""
        summary_file = os.path.join(self.results_dir, "training_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("CAE训练摘要\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"设备: {self.device}\n")
            f.write(f"总训练时间: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)\n")
            f.write(f"总轮次: {len(self.train_accuracies)}\n")
            f.write(f"最佳验证准确率: {self.best_val_accuracy:.2f}%\n")
            f.write(f"最终训练准确率: {self.train_accuracies[-1]:.2f}%\n")
            f.write(f"最终验证准确率: {self.val_accuracies[-1]:.2f}%\n\n")

            f.write(f"类别数量: {len(self.class_names)}\n")
            f.write(f"类别: {', '.join(self.class_names)}\n")


# ==================== 主函数 ====================
def main():
    print("🚀 开始CAE训练与评估")

    # 清除CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 检查数据集
    if not os.path.exists(CAEConfig.DATASET_PATH):
        print(f"❌ 数据集不存在: {CAEConfig.DATASET_PATH}")
        return

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((CAEConfig.IMG_SIZE, CAEConfig.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CAEConfig.IMG_SIZE, CAEConfig.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    print("\n📂 加载数据集中...")
    try:
        train_dataset = CAEDataset(CAEConfig.DATASET_PATH, transform, 'train')
        val_dataset = CAEDataset(CAEConfig.DATASET_PATH, val_transform, 'val')
        test_dataset = CAEDataset(CAEConfig.DATASET_PATH, val_transform, 'test')

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
        batch_size=CAEConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=CAEConfig.NUM_WORKERS,
        pin_memory=(CAEConfig.DEVICE.type == 'cuda'),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CAEConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=CAEConfig.NUM_WORKERS,
        pin_memory=(CAEConfig.DEVICE.type == 'cuda')
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CAEConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=CAEConfig.NUM_WORKERS
    )

    # 创建模型
    print("\n🤖 创建CAE模型中...")
    model = SimpleCAE(
        input_channels=CAEConfig.CHANNELS,
        num_classes=len(class_names)
    ).to(CAEConfig.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 总参数: {total_params:,}")

    # 训练
    trainer = FixedCAETrainer(model, CAEConfig, train_dataset)
    trainer.train(train_loader, val_loader)

    # 测试
    print("\n🧪 测试模型中...")
    evaluate_model(model, test_loader, CAEConfig.RESULTS_PATH, CAEConfig.DEVICE, class_names)

    # 最终输出
    print("\n" + "=" * 60)
    print("🎉 CAE训练与评估完成!")
    print("=" * 60)
    print(f"📁 结果目录: {CAEConfig.RESULTS_PATH}")
    print(f"🏆 最佳验证准确率: {trainer.best_val_accuracy:.2f}%")
    print(f"📊 类别数量: {len(class_names)}")


def evaluate_model(model, test_loader, results_dir, device, class_names):
    """评估模型性能"""
    print("\n🧪 评估模型中...")

    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            classification = model(images, mode='classify')

            # 获取预测结果
            _, predicted = torch.max(classification, 1)

            # 收集数据
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # 打印进度
            if batch_idx % 50 == 0:
                print(f"  处理批次 {batch_idx}/{len(test_loader)}")

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

    # 保存结果
    print(f"\n📊 测试结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  测试样本数: {len(all_labels)}")

    # 保存到文件
    test_file = os.path.join(results_dir, "test_results.txt")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("CAE测试结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"测试样本数: {len(all_labels)}\n")
        f.write(f"准确率: {accuracy:.4f}\n")
        f.write(f"精确率: {precision:.4f}\n")
        f.write(f"召回率: {recall:.4f}\n")
        f.write(f"F1分数: {f1:.4f}\n")

    return accuracy


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  CAE训练被用户中断")
    except Exception as e:
        print(f"\n❌ CAE训练错误: {e}")
        import traceback

        traceback.print_exc()