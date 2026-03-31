"""
VAE框架 - 内存优化版本
解决CPU内存不足的问题
"""

import os
import gc
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


# ==================== VAE配置设置 ====================
class VAEConfig:
    """VAE训练配置类"""
    # 路径配置
    DATASET_PATH = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"
    RESULTS_PATH = r"E:\庄楷文\叶片病虫害识别\baseline\results\vae_memory_fixed"

    # 添加时间戳，避免冲突
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_PATH = f"{RESULTS_PATH}_{timestamp}"

    # 模型配置
    IMG_SIZE = 128  # 降低图像尺寸以节省内存
    LATENT_DIM = 128  # 降低潜在维度
    CHANNELS = 3

    # VAE特定配置
    BETA = 1.0
    USE_ANNEALING = True
    ANNEALING_STEPS = 1000

    # 训练配置
    BATCH_SIZE = 8  # 减少批处理大小
    NUM_EPOCHS = 30  # 减少轮次
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 1  # 减少工作线程
    USE_AMP = True

    # 早停设置
    PATIENCE = 10


# ==================== 简化的VAE模型 ====================
class SimpleVAE(nn.Module):
    """简化版的VAE，减少内存使用"""

    def __init__(self, input_channels=3, latent_dim=128, num_classes=22):
        super(SimpleVAE, self).__init__()
        self.latent_dim = latent_dim

        # 简化编码器
        self.encoder = nn.Sequential(
            # 输入: (3, 128, 128)
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # 计算编码器输出的特征图大小
        self.encoded_size = 256 * 8 * 8

        # 均值和对数方差的全连接层
        self.fc_mu = nn.Linear(self.encoded_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_size, latent_dim)

        # 解码器的输入层
        self.fc_decode = nn.Linear(latent_dim, self.encoded_size)

        # 简化解码器
        self.decoder = nn.Sequential(
            # 输入: (256, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # (3, 128, 128)
            nn.Sigmoid()
        )

        # 简化分类器
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def encode(self, x):
        """编码输入并返回均值和对数方差"""
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """解码潜在变量"""
        decoded = self.fc_decode(z)
        decoded = decoded.view(decoded.size(0), 256, 8, 8)
        return self.decoder(decoded)

    def forward(self, x, mode='all'):
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if mode == 'encode':
            return mu, logvar, z

        # 分类分支
        classification = self.classifier(z)

        if mode == 'classify':
            return classification

        # 解码分支
        reconstructed = self.decode(z)

        if mode == 'decode':
            return reconstructed

        return reconstructed, classification, mu, logvar

    def generate(self, num_samples, device):
        """从先验分布生成样本"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)


# ==================== 内存友好的数据集类 ====================
class MemoryFriendlyDataset(Dataset):
    """节省内存的数据集类"""

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

            self.label_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
            self.idx_to_label = {idx: class_name for idx, class_name in enumerate(classes)}

            for class_name in classes:
                class_path = os.path.join(self.root_dir, class_name)
                class_idx = self.label_to_idx[class_name]

                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.image_paths.append(os.path.join(class_path, img_file))
                        self.labels.append(class_idx)

        print(f"{mode.upper()}: 加载 {len(self.image_paths)} 张图片，{len(self.label_to_idx)} 个类别")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # 延迟加载，不缓存图像
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            print(f"加载图像错误 {img_path}: {e}")
            # 返回空图像
            if self.transform:
                image = self.transform(Image.new('RGB', (VAEConfig.IMG_SIZE, VAEConfig.IMG_SIZE), color='black'))
            return image, label, img_path

    def get_class_names(self):
        return [self.idx_to_label[i] for i in range(len(self.idx_to_label))]


# ==================== 内存友好的训练器 ====================
class MemoryFriendlyVAETrainer:
    def __init__(self, model, config, train_dataset):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.train_dataset = train_dataset
        self.class_names = train_dataset.get_class_names()

        # KL散度退火参数
        self.global_step = 0
        self.annealing_rate = 1.0 / config.ANNEALING_STEPS if config.USE_ANNEALING else 0
        self.kl_weight = 0.0 if config.USE_ANNEALING else config.BETA

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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

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
            "latent_dim": self.config.LATENT_DIM,
            "num_epochs": self.config.NUM_EPOCHS,
            "class_names": self.class_names,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(self.results_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def kl_divergence(self, mu, logvar):
        """计算KL散度"""
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss / mu.size(0)

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_recon_loss = 0
        total_kl_loss = 0
        total_class_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 更新KL权重
            if self.config.USE_ANNEALING:
                self.kl_weight = min(1.0, self.global_step * self.annealing_rate)
                self.kl_weight *= self.config.BETA
                self.global_step += 1

            # 混合精度训练
            if self.config.USE_AMP and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    reconstructed, classification, mu, logvar = self.model(images)

                    recon_loss = self.recon_criterion(reconstructed, images)
                    kl_loss = self.kl_divergence(mu, logvar)
                    class_loss = self.class_criterion(classification, labels)

                    total_loss = recon_loss + self.kl_weight * kl_loss + class_loss

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                reconstructed, classification, mu, logvar = self.model(images)

                recon_loss = self.recon_criterion(reconstructed, images)
                kl_loss = self.kl_divergence(mu, logvar)
                class_loss = self.class_criterion(classification, labels)

                total_loss = recon_loss + self.kl_weight * kl_loss + class_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_class_loss += class_loss.item()

            # 计算准确率
            _, predicted = torch.max(classification, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # 清理内存
            del reconstructed, classification, mu, logvar, total_loss
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

            # 打印进度
            if batch_idx % 100 == 0:
                accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
                print(f"  批次 {batch_idx}/{len(train_loader)}, "
                      f"重建损失: {recon_loss.item():.4f}, "
                      f"准确率: {accuracy:.2f}%")

        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0

        return avg_recon_loss, avg_kl_loss, avg_class_loss, accuracy

    def validate(self, val_loader):
        self.model.eval()
        total_recon_loss = 0
        total_kl_loss = 0
        total_class_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                reconstructed, classification, mu, logvar = self.model(images)

                recon_loss = self.recon_criterion(reconstructed, images)
                kl_loss = self.kl_divergence(mu, logvar)
                class_loss = self.class_criterion(classification, labels)

                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_class_loss += class_loss.item()

                _, predicted = torch.max(classification, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_recon_loss = total_recon_loss / len(val_loader)
        avg_kl_loss = total_kl_loss / len(val_loader)
        avg_class_loss = total_class_loss / len(val_loader)
        accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0

        return avg_recon_loss, avg_kl_loss, avg_class_loss, accuracy

    def train(self, train_loader, val_loader):
        print(f"\n开始训练，设备: {self.device}")
        print(f"图像尺寸: {self.config.IMG_SIZE}")
        print(f"批处理大小: {self.config.BATCH_SIZE}")
        print(f"类别数量: {len(self.class_names)}")

        start_time = time.time()

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"\n📅 轮次 [{epoch}/{self.config.NUM_EPOCHS}]")

            # 训练
            train_recon_loss, train_kl_loss, train_class_loss, train_acc = \
                self.train_epoch(train_loader, epoch)

            # 验证
            val_recon_loss, val_kl_loss, val_class_loss, val_acc = \
                self.validate(val_loader)

            self.train_losses.append(train_recon_loss + train_kl_loss + train_class_loss)
            self.val_losses.append(val_recon_loss + val_kl_loss + val_class_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # 保存最佳模型
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                model_path = os.path.join(self.results_dir, f"best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'class_names': self.class_names,
                }, model_path)
                print(f"✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")

            # 打印日志
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"训练 - 重建: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f}, "
                  f"分类: {train_class_loss:.4f}, 准确率: {train_acc:.2f}%")
            print(f"验证 - 重建: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f}, "
                  f"分类: {val_class_loss:.4f}, 准确率: {val_acc:.2f}%")
            print(f"学习率: {current_lr:.6f}, KL权重: {self.kl_weight:.4f}")

            # 清理内存
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

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
            f.write("VAE训练摘要 (内存优化版)\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"设备: {self.device}\n")
            f.write(f"总训练时间: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)\n")
            f.write(f"总轮次: {len(self.train_accuracies)}\n")
            f.write(f"最佳验证准确率: {self.best_val_accuracy:.2f}%\n")
            f.write(f"最终训练准确率: {self.train_accuracies[-1]:.2f}%\n")
            f.write(f"最终验证准确率: {self.val_accuracies[-1]:.2f}%\n\n")

            f.write(f"类别数量: {len(self.class_names)}\n")
            f.write(f"类别: {', '.join(self.class_names)}\n")


# ==================== 简化评估函数 ====================
def simple_evaluate_vae(model, test_loader, results_dir, device, class_names):
    """简化评估函数，避免内存问题"""
    print("\n🧪 评估模型中...")

    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 只进行前向传播获取分类结果
            classification = model(images, mode='classify')

            # 获取预测结果
            _, predicted = torch.max(classification, 1)

            # 收集数据
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # 打印进度
            if batch_idx % 50 == 0:
                print(f"  处理批次 {batch_idx}/{len(test_loader)}")

            # 清理内存
            del classification, predicted
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

    # 生成分类报告
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    # 保存结果
    print(f"\n📊 测试结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")

    # 保存测试结果
    test_file = os.path.join(results_dir, "test_results.txt")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("VAE测试结果 (简化版)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"测试样本数: {len(all_labels)}\n")
        f.write(f"准确率: {accuracy:.4f}\n")
        f.write(f"精确率: {precision:.4f}\n")
        f.write(f"召回率: {recall:.4f}\n")
        f.write(f"F1分数: {f1:.4f}\n")
        f.write(f"\n类别数量: {len(class_names)}\n")
        f.write(f"类别: {', '.join(class_names)}\n")

    # 绘制简化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return accuracy, precision, recall, f1


# ==================== 主函数 ====================
def main():
    print("🚀 开始VAE训练与评估 (内存优化版)")

    # 检查数据集
    if not os.path.exists(VAEConfig.DATASET_PATH):
        print(f"❌ 数据集不存在: {VAEConfig.DATASET_PATH}")
        return

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((VAEConfig.IMG_SIZE, VAEConfig.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
    ])

    # 加载数据集
    print("\n📂 加载数据集中...")
    try:
        train_dataset = MemoryFriendlyDataset(VAEConfig.DATASET_PATH, transform, 'train')
        val_dataset = MemoryFriendlyDataset(VAEConfig.DATASET_PATH, transform, 'val')
        test_dataset = MemoryFriendlyDataset(VAEConfig.DATASET_PATH, transform, 'test')

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
        batch_size=VAEConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=VAEConfig.NUM_WORKERS,
        pin_memory=False,  # 禁用pinned memory，减少内存压力
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAEConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=VAEConfig.NUM_WORKERS,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=VAEConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=VAEConfig.NUM_WORKERS,
        pin_memory=False
    )

    # 创建模型
    print("\n🤖 创建模型中...")
    model = SimpleVAE(
        input_channels=VAEConfig.CHANNELS,
        latent_dim=VAEConfig.LATENT_DIM,
        num_classes=len(class_names)
    ).to(VAEConfig.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 总参数: {total_params:,}")

    # 训练
    trainer = MemoryFriendlyVAETrainer(model, VAEConfig, train_dataset)
    trainer.train(train_loader, val_loader)

    # 测试
    print("\n🧪 测试模型中...")
    model_path = os.path.join(VAEConfig.RESULTS_PATH, "best_model.pth")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=VAEConfig.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载最佳模型 (Epoch {checkpoint.get('epoch', 'N/A')})")

    accuracy, precision, recall, f1 = simple_evaluate_vae(
        model, test_loader, VAEConfig.RESULTS_PATH, VAEConfig.DEVICE, class_names
    )

    # 最终输出
    print("\n" + "=" * 60)
    print("🎉 VAE训练与评估完成!")
    print("=" * 60)
    print(f"📁 结果目录: {VAEConfig.RESULTS_PATH}")
    print(f"🏆 最佳验证准确率: {trainer.best_val_accuracy:.2f}%")
    print(f"🧪 测试准确率: {accuracy:.4f}")
    print(f"📊 类别数量: {len(class_names)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练错误: {e}")
        import traceback

        traceback.print_exc()