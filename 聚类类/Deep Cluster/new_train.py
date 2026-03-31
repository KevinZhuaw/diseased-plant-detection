"""
内存优化的叶片病虫害识别训练系统 - 针对RTX 4070 Ti 12GB优化
单个文件版本，无需OpenCV
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ==================== 内存优化配置 ====================
# 在导入其他库之前设置环境变量，优化内存使用
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'  # 减少内存碎片
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步CUDA操作

# ==================== 简化的依赖检查 ====================
def check_dependencies():
    """检查基本依赖"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA可用 - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("✗ CUDA不可用，将使用CPU（训练会很慢）")
    except ImportError:
        print("✗ PyTorch未安装")
        print("安装命令: pip install torch torchvision")
        sys.exit(1)

# 检查依赖
print("检查依赖...")
check_dependencies()

# ==================== 导入库 ====================
print("导入库...")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，减少内存
import matplotlib.pyplot as plt

# ==================== 内存监控器 ====================
class MemoryMonitor:
    """GPU内存监控器"""
    @staticmethod
    def get_gpu_memory():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return allocated, reserved
        return 0, 0

    @staticmethod
    def print_memory_status(label=""):
        if torch.cuda.is_available():
            allocated, reserved = MemoryMonitor.get_gpu_memory()
            print(f"{label} GPU内存: {allocated:.2f}GB已分配, {reserved:.2f}GB已保留")

    @staticmethod
    def clear_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ==================== 优化的配置类 ====================
class OptimizedConfig:
    """针对RTX 4070 Ti 12GB优化的配置"""

    def __init__(self):
        # 数据路径
        self.data_dir = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"

        # 内存优化配置
        self.image_size = 224  # 保持224，因为预训练模型需要这个尺寸
        self.batch_size = 8    # 减小批大小以节省内存
        self.num_workers = 2   # 减少工作线程数

        # 模型配置 - 使用更小的模型
        self.backbone = "resnet34"  # 比resnet50小，但效果接近
        self.pretrained = True

        # 训练配置
        self.num_epochs = 20         # 减少epoch数
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 梯度累积（模拟更大的batch size）
        self.gradient_accumulation_steps = 2  # 有效batch size = 8 * 2 = 16

        # 混合精度训练（减少内存，加快训练）
        self.use_amp = True

        # 输出目录
        self.checkpoint_dir = "optimized_checkpoints"
        self.results_dir = "optimized_results"

        # 创建目录
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        Path(self.results_dir).mkdir(exist_ok=True)

        # 打印配置
        self._print_config()

    def _print_config(self):
        """打印配置信息"""
        print("\n" + "="*60)
        print("优化配置 (针对RTX 4070 Ti 12GB)")
        print("="*60)
        print(f"设备: {self.device}")
        print(f"批大小: {self.batch_size} (累积步数: {self.gradient_accumulation_steps})")
        print(f"有效批大小: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"模型: {self.backbone}")
        print(f"图像尺寸: {self.image_size}")
        print(f"训练轮数: {self.num_epochs}")
        print(f"混合精度训练: {self.use_amp}")
        print("="*60)

# ==================== 内存优化的数据集类 ====================
class MemoryEfficientDataset(Dataset):
    """内存优化的数据集"""

    def __init__(self, data_dir, split='train', transform=None, max_samples=2000):
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform
        self.max_samples = max_samples

        # 验证路径
        if not self.data_dir.exists():
            raise FileNotFoundError(f"目录不存在: {self.data_dir}")

        # 获取类别（惰性加载）
        self.classes = []
        self.samples = []
        self._load_minimal_metadata()

        print(f"{split}集: {len(self.samples)}个样本, {len(self.classes)}个类别")

    def _load_minimal_metadata(self):
        """只加载元数据，不加载图像"""
        # 获取类别
        self.classes = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(self.data_dir / d)
        ])

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 找到健康类别
        self.healthy_idx = None
        for idx, cls in enumerate(self.classes):
            if 'healthy' in cls.lower():
                self.healthy_idx = idx
                break

        # 收集样本路径
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = self.data_dir / class_name

            # 获取图像文件
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend([str(p) for p in class_dir.glob(ext)])

            # 限制样本数量（重要！避免内存溢出）
            if self.max_samples and len(image_files) > self.max_samples:
                image_files = random.sample(image_files, self.max_samples)

            for img_path in image_files:
                self.samples.append((img_path, class_idx))

        # 打乱顺序
        if self.split == 'train':
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, multi_label = self.samples[idx]

        # 按需加载图像（减少内存占用）
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 创建空白图像作为后备
            image = Image.new('RGB', (224, 224), (255, 255, 255))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 二分类标签
        binary_label = 0 if multi_label == self.healthy_idx else 1

        return image, multi_label, binary_label, img_path

# ==================== 轻量级模型 ====================
class LightweightModel(nn.Module):
    """轻量级双任务模型"""

    def __init__(self, backbone_name='resnet34', num_classes=22, pretrained=True):
        super().__init__()

        # 加载预训练骨干网络
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"不支持的骨干网络: {backbone_name}")

        # 冻结部分层以节省内存
        for param in backbone.parameters():
            param.requires_grad = False

        # 解冻最后几层
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        # 特征提取器
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # 简单的分类头
        self.multi_classifier = nn.Linear(feature_dim, num_classes)
        self.binary_classifier = nn.Linear(feature_dim, 2)

        # 统计参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    def forward(self, x):
        # 提取特征
        features = self.features(x)
        features = features.view(features.size(0), -1)

        # 分类
        multi_output = self.multi_classifier(features)
        binary_output = self.binary_classifier(features)

        return multi_output, binary_output

# ==================== 优化的训练器 ====================
class OptimizedTrainer:
    """内存优化的训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        # 初始化模型
        self.model = LightweightModel(
            backbone_name=config.backbone,
            num_classes=config.num_classes,
            pretrained=config.pretrained
        ).to(self.device)

        # 损失函数
        self.criterion_multi = nn.CrossEntropyLoss()
        self.criterion_binary = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

        # 训练历史
        self.history = {
            'train_loss': [], 'train_acc_multi': [], 'train_acc_binary': [],
            'val_loss': [], 'val_acc_multi': [], 'val_acc_binary': []
        }

        # 最佳模型
        self.best_acc = 0.0

        print(f"训练器初始化完成，使用设备: {self.device}")
        MemoryMonitor.print_memory_status("初始化后")

    def calculate_accuracy(self, predictions, targets):
        """计算准确率"""
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        return 100.0 * correct / total

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct_multi = 0
        correct_binary = 0
        total = 0

        # 梯度累积计数器
        accumulation_steps = 0

        pbar = tqdm(train_loader, desc=f"训练轮次 {epoch+1}/{self.config.num_epochs}")

        for batch_idx, (images, multi_labels, binary_labels, _) in enumerate(pbar):
            images = images.to(self.device)
            multi_labels = multi_labels.to(self.device)
            binary_labels = binary_labels.to(self.device)

            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # 前向传播
                multi_output, binary_output = self.model(images)

                # 计算损失
                loss_multi = self.criterion_multi(multi_output, multi_labels)
                loss_binary = self.criterion_binary(binary_output, binary_labels)
                loss = loss_multi + 0.3 * loss_binary

            # 梯度累积
            loss = loss / self.config.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_steps += 1

            # 累积足够步数后更新参数
            if accumulation_steps % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                accumulation_steps = 0

            # 统计
            total_loss += loss.item() * self.config.gradient_accumulation_steps

            # 计算准确率
            _, preds_multi = torch.max(multi_output, 1)
            _, preds_binary = torch.max(binary_output, 1)

            correct_multi += (preds_multi == multi_labels).sum().item()
            correct_binary += (preds_binary == binary_labels).sum().item()
            total += images.size(0)

            # 更新进度条
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc_multi': f"{correct_multi/total:.3f}",
                    'mem': f"{torch.cuda.memory_allocated()/1024**3:.2f}GB"
                })

        # 计算epoch指标
        epoch_loss = total_loss / len(train_loader)
        epoch_acc_multi = 100.0 * correct_multi / total
        epoch_acc_binary = 100.0 * correct_binary / total

        return epoch_loss, epoch_acc_multi, epoch_acc_binary

    def validate(self, val_loader, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct_multi = 0
        correct_binary = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"验证轮次 {epoch+1}/{self.config.num_epochs}")

            for images, multi_labels, binary_labels, _ in pbar:
                images = images.to(self.device)
                multi_labels = multi_labels.to(self.device)
                binary_labels = binary_labels.to(self.device)

                # 前向传播
                multi_output, binary_output = self.model(images)

                # 计算损失
                loss_multi = self.criterion_multi(multi_output, multi_labels)
                loss_binary = self.criterion_binary(binary_output, binary_labels)
                loss = loss_multi + 0.3 * loss_binary

                total_loss += loss.item()

                # 计算准确率
                _, preds_multi = torch.max(multi_output, 1)
                _, preds_binary = torch.max(binary_output, 1)

                correct_multi += (preds_multi == multi_labels).sum().item()
                correct_binary += (preds_binary == binary_labels).sum().item()
                total += images.size(0)

        # 计算指标
        val_loss = total_loss / len(val_loader)
        val_acc_multi = 100.0 * correct_multi / total
        val_acc_binary = 100.0 * correct_binary / total

        return val_loss, val_acc_multi, val_acc_binary

    def train(self, train_loader, val_loader):
        """训练主循环"""
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)

        for epoch in range(self.config.num_epochs):
            print(f"\n轮次 {epoch+1}/{self.config.num_epochs}")

            # 清空GPU缓存
            MemoryMonitor.clear_cache()

            # 训练
            train_loss, train_acc_multi, train_acc_binary = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss, val_acc_multi, val_acc_binary = self.validate(val_loader, epoch)

            # 保存历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc_multi'].append(train_acc_multi)
            self.history['train_acc_binary'].append(train_acc_binary)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc_multi'].append(val_acc_multi)
            self.history['val_acc_binary'].append(val_acc_binary)

            # 打印结果
            print(f"训练 - 损失: {train_loss:.4f}, 多分类准确率: {train_acc_multi:.2f}%, 二分类准确率: {train_acc_binary:.2f}%")
            print(f"验证 - 损失: {val_loss:.4f}, 多分类准确率: {val_acc_multi:.2f}%, 二分类准确率: {val_acc_binary:.2f}%")

            # 保存最佳模型
            if val_acc_multi > self.best_acc:
                self.best_acc = val_acc_multi
                self.save_model(epoch, val_acc_multi)
                print(f"✓ 最佳模型已保存! 准确率: {val_acc_multi:.2f}%")

            # 每5轮清理一次内存
            if (epoch + 1) % 5 == 0:
                MemoryMonitor.clear_cache()
                MemoryMonitor.print_memory_status(f"轮次{epoch+1}后")

        print(f"\n训练完成! 最佳验证准确率: {self.best_acc:.2f}%")
        return self.history

    def save_model(self, epoch, accuracy):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config.__dict__
        }

        torch.save(checkpoint, f"{self.config.checkpoint_dir}/best_model.pth")

    def evaluate(self, test_loader, class_names, healthy_idx):
        """评估模型"""
        print("\n" + "="*60)
        print("在测试集上评估")
        print("="*60)

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_binary_predictions = []
        all_binary_labels = []
        all_confidences = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="评估中")
            for images, multi_labels, binary_labels, _ in pbar:
                images = images.to(self.device)

                # 前向传播
                multi_output, binary_output = self.model(images)

                # 获取预测
                multi_probs = torch.softmax(multi_output, dim=1)
                binary_probs = torch.softmax(binary_output, dim=1)

                multi_confidences, multi_preds = torch.max(multi_probs, 1)
                binary_confidences, binary_preds = torch.max(binary_probs, 1)

                # 收集结果
                all_predictions.extend(multi_preds.cpu().numpy())
                all_labels.extend(multi_labels.numpy())
                all_binary_predictions.extend(binary_preds.cpu().numpy())
                all_binary_labels.extend(binary_labels.numpy())
                all_confidences.extend(multi_confidences.cpu().numpy())

        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        multi_accuracy = accuracy_score(all_labels, all_predictions)
        multi_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        multi_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        multi_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        binary_accuracy = accuracy_score(all_binary_labels, all_binary_predictions)
        binary_precision = precision_score(all_binary_labels, all_binary_predictions, average='binary', zero_division=0)
        binary_recall = recall_score(all_binary_labels, all_binary_predictions, average='binary', zero_division=0)
        binary_f1 = f1_score(all_binary_labels, all_binary_predictions, average='binary', zero_division=0)

        # 打印结果
        print(f"\n多分类结果:")
        print(f"  准确率: {multi_accuracy:.4f}")
        print(f"  精确率: {multi_precision:.4f}")
        print(f"  召回率: {multi_recall:.4f}")
        print(f"  F1分数: {multi_f1:.4f}")

        print(f"\n二分类结果:")
        print(f"  准确率: {binary_accuracy:.4f}")
        print(f"  精确率: {binary_precision:.4f}")
        print(f"  召回率: {binary_recall:.4f}")
        print(f"  F1分数: {binary_f1:.4f}")

        # 保存结果
        results = {
            'multi': {
                'accuracy': float(multi_accuracy),
                'precision': float(multi_precision),
                'recall': float(multi_recall),
                'f1': float(multi_f1)
            },
            'binary': {
                'accuracy': float(binary_accuracy),
                'precision': float(binary_precision),
                'recall': float(binary_recall),
                'f1': float(binary_f1)
            },
            'class_names': class_names,
            'healthy_idx': int(healthy_idx)
        }

        with open(f"{self.config.results_dir}/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\n结果已保存到: {self.config.results_dir}/evaluation_results.json")

        return results

# ==================== 简单可视化 ====================
def plot_training_history(history, config):
    """绘制简单的训练历史图"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='训练')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='验证')
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 多分类准确率
    axes[0, 1].plot(epochs, history['train_acc_multi'], 'b-', label='训练')
    axes[0, 1].plot(epochs, history['val_acc_multi'], 'r-', label='验证')
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].set_title('多分类准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 二分类准确率
    axes[1, 0].plot(epochs, history['train_acc_binary'], 'b-', label='训练')
    axes[1, 0].plot(epochs, history['val_acc_binary'], 'r-', label='验证')
    axes[1, 0].set_xlabel('轮次')
    axes[1, 0].set_ylabel('准确率 (%)')
    axes[1, 0].set_title('二分类准确率')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 最佳模型标注
    best_epoch = np.argmax(history['val_acc_multi'])
    axes[0, 1].axvline(x=best_epoch+1, color='g', linestyle='--', alpha=0.7)
    axes[0, 1].text(best_epoch+1, history['val_acc_multi'][best_epoch]*0.9,
                   f'最佳: {history["val_acc_multi"][best_epoch]:.1f}%',
                   color='g')

    plt.tight_layout()
    plt.savefig(f"{config.results_dir}/training_history.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"训练历史图已保存: {config.results_dir}/training_history.png")

# ==================== 主函数 ====================
def main():
    """主函数"""
    print("\n" + "="*60)
    print("叶片病虫害识别 - 内存优化版本")
    print("="*60)

    try:
        # 1. 初始化配置
        print("\n1. 初始化配置...")
        config = OptimizedConfig()

        # 2. 准备数据
        print("\n2. 准备数据...")

        # 简单的数据增强
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 创建数据集（限制样本数量）
        print("加载训练集...")
        train_dataset = MemoryEfficientDataset(
            config.data_dir, 'train', train_transform, max_samples=1000
        )

        print("加载验证集...")
        val_dataset = MemoryEfficientDataset(
            config.data_dir, 'val', val_transform, max_samples=500
        )

        print("加载测试集...")
        test_dataset = MemoryEfficientDataset(
            config.data_dir, 'test', val_transform
        )

        # 更新类别数
        config.num_classes = len(train_dataset.classes)
        print(f"数据集: {config.num_classes}个类别")
        print(f"  训练集: {len(train_dataset)}个样本")
        print(f"  验证集: {len(val_dataset)}个样本")
        print(f"  测试集: {len(test_dataset)}个样本")

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True  # 丢弃不完整的批次
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        # 3. 创建和训练模型
        print("\n3. 创建模型...")
        trainer = OptimizedTrainer(config)

        print("\n4. 开始训练...")
        history = trainer.train(train_loader, val_loader)

        # 4. 可视化训练历史
        print("\n5. 可视化结果...")
        plot_training_history(history, config)

        # 5. 评估模型
        print("\n6. 评估模型...")
        results = trainer.evaluate(test_loader, train_dataset.classes, train_dataset.healthy_idx)

        # 6. 保存配置和类别信息
        config_info = {
            'data_dir': config.data_dir,
            'image_size': config.image_size,
            'batch_size': config.batch_size,
            'backbone': config.backbone,
            'num_classes': config.num_classes,
            'num_epochs': config.num_epochs,
            'learning_rate': config.learning_rate,
            'classes': train_dataset.classes,
            'healthy_idx': int(train_dataset.healthy_idx)
        }

        with open(f"{config.results_dir}/config_info.json", 'w') as f:
            json.dump(config_info, f, indent=4)

        # 7. 创建简单的预测脚本
        create_simple_predictor(config, train_dataset.classes, train_dataset.healthy_idx)

        print("\n" + "="*60)
        print("训练完成!")
        print(f"结果保存在: {config.results_dir}/")
        print(f"模型保存在: {config.checkpoint_dir}/")
        print("="*60)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

        # 尝试保存错误信息
        try:
            with open("error_log.txt", "w") as f:
                f.write(f"错误: {str(e)}\n")
                f.write(traceback.format_exc())
            print("错误日志已保存到: error_log.txt")
        except:
            pass

def create_simple_predictor(config, class_names, healthy_idx):
    """创建简单的预测脚本"""
    predictor_code = '''import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

class SimpleLeafPredictor:
    def __init__(self, model_path, class_names, healthy_idx, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.healthy_idx = healthy_idx
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型（需要根据实际架构调整）
        # 这里使用简化版本
        from torchvision import models
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.backbone = models.resnet34(pretrained=False)
                self.backbone.fc = nn.Linear(512, num_classes)
                
            def forward(self, x):
                return self.backbone(x)
        
        model = SimpleModel(len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)
        
        class_name = self.class_names[pred.item()]
        is_healthy = pred.item() == self.healthy_idx
        
        return {
            'class': class_name,
            'confidence': confidence.item(),
            'is_healthy': is_healthy
        }

# 使用示例
if __name__ == "__main__":
    import json
    with open('optimized_results/config_info.json', 'r') as f:
        config = json.load(f)
    
    predictor = SimpleLeafPredictor(
        'optimized_checkpoints/best_model.pth',
        config['classes'],
        config['healthy_idx']
    )
    
    result = predictor.predict('test_image.jpg')
    print(f"预测结果: {result}")
'''

    with open(f"{config.results_dir}/simple_predictor.py", 'w') as f:
        f.write(predictor_code)

    print(f"预测脚本已保存: {config.results_dir}/simple_predictor.py")

# ==================== 运行 ====================
if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # 运行主函数
    main()