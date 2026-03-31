# moco_fixed.py
"""
MoCo叶片病害检测系统 - 修复CUDA内存映射冲突版本
"""

import os
import sys
import random
import time
import datetime
import warnings
from PIL import Image
from pathlib import Path

# 设置环境变量避免CUDA内存问题
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 更好的错误信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第一个GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 可扩展内存段

warnings.filterwarnings('ignore')

print("=" * 60)
print("MoCo叶片病害检测系统 - 修复版")
print("=" * 60)

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
        # 检查GPU内存
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        cached = torch.cuda.memory_reserved(0) / 1024 ** 3
        print(f"  显存总量: {total_memory:.1f} GB")
        print(f"  已分配: {allocated:.2f} GB, 已缓存: {cached:.2f} GB")

except ImportError as e:
    print(f"❌ PyTorch 导入失败: {e}")
    exit(1)

try:
    from tqdm import tqdm
    import matplotlib

    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
        classification_report
    from sklearn.preprocessing import StandardScaler

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
    SAMPLE_RATIO = 1.0

    # MoCo训练参数
    EPOCHS = 30
    BATCH_SIZE = 64  # 根据你的12GB显存调整
    LEARNING_RATE = 0.03
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    TEMPERATURE = 0.07
    MOMENTUM_COEFF = 0.999

    # 分类器训练参数
    CLASSIFIER_EPOCHS = 30
    CLASSIFIER_LR = 0.01

    # 模型参数
    FEATURE_DIM = 128
    PROJECTION_DIM = 256
    QUEUE_SIZE = 65536

    # 硬件参数 - 修复关键设置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0  # ❗ 关键修复：设为0，避免多进程内存冲突
    PIN_MEMORY = False  # ❗ 关键修复：设为False

    def __init__(self):
        # 创建时间戳的唯一保存目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.SAVE_DIR = f"./moco_fixed_{timestamp}"
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        print(f"📁 创建新的保存目录: {self.SAVE_DIR}")

        # 检查数据集目录
        print(f"\n📂 检查数据集目录: {self.DATASET_DIR}")
        if not os.path.exists(self.DATASET_DIR):
            print(f"❌ 数据集目录不存在!")
            return

    def print_config(self):
        print("=" * 60)
        print("📊 配置参数:")
        print(f"  图像尺寸: {self.IMAGE_SIZE}x{self.IMAGE_SIZE}")
        print(f"  采样比例: {self.SAMPLE_RATIO}")
        print(f"  MoCo预训练轮次: {self.EPOCHS}")
        print(f"  批次大小: {self.BATCH_SIZE}")
        print(f"  学习率: {self.LEARNING_RATE}")
        print(f"  保存目录: {self.SAVE_DIR}")
        print(f"  工作进程数: {self.NUM_WORKERS}")
        print(f"  内存固定: {self.PIN_MEMORY}")
        print("=" * 60)


# ========== 数据增强和数据集类 ==========
class MoCoDataAugmentation:
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class LeafDiseaseDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None, sample_ratio=1.0):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.samples = []
        self.labels = []

        if mode == 'train':
            base_dir = self.data_dir / 'train'
        else:
            base_dir = self.data_dir / 'test'

        subdirs = ['healthy', 'diseased']

        print(f"\n📥 加载{mode}数据...")
        for label_idx, subdir in enumerate(subdirs):
            dir_path = base_dir / subdir

            if not dir_path.exists():
                print(f"⚠️  目录不存在: {dir_path}")
                continue

            img_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_files.extend(list(dir_path.glob(f'*{ext}')))
                img_files.extend(list(dir_path.glob(f'*{ext.upper()}')))

            img_files = sorted(img_files)
            print(f"  📁 {subdir} - 找到 {len(img_files)} 张图片")

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
            image = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                if isinstance(self.transform, MoCoDataAugmentation):
                    image_q, image_k = self.transform(image)
                    return image_q, image_k, label
                else:
                    image = self.transform(image)
                    return image, label
            return image, label
        except Exception as e:
            print(f"⚠️ 加载图像失败 {img_path}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            if isinstance(self.transform, MoCoDataAugmentation):
                return dummy_image, dummy_image, label
            else:
                return dummy_image, label


# ========== 模型组件 ==========
class MLPHead(nn.Module):
    def __init__(self, in_dim, projection_dim=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        return self.net(x)


class EncoderBase(nn.Module):
    def __init__(self, arch='resnet50', feature_dim=128, projection_dim=256):
        super().__init__()
        backbone = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = MLPHead(2048, projection_dim)
        self.prediction = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, feature_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projection(h)
        p = self.prediction(z)
        return F.normalize(p, dim=1)


class MoCo(nn.Module):
    def __init__(self, feature_dim=128, projection_dim=256,
                 queue_size=65536, temperature=0.07, momentum=0.999):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.momentum = momentum
        self.queue_size = queue_size

        self.encoder_q = EncoderBase('resnet50', feature_dim, projection_dim)
        self.encoder_k = EncoderBase('resnet50', feature_dim, projection_dim)

        # 初始化键编码器
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 创建队列
        self.register_buffer("queue", torch.randn(feature_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.queue_size:
            batch_size = self.queue_size - ptr

        self.queue[:, ptr:ptr + batch_size] = keys.T[:, :batch_size]
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = k.detach()

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        self._dequeue_and_enqueue(k)
        return logits, labels


# ========== 训练器类 ==========
class MoCoTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # 清空GPU缓存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.model = MoCo(
            feature_dim=config.FEATURE_DIM,
            projection_dim=config.PROJECTION_DIM,
            queue_size=config.QUEUE_SIZE,
            temperature=config.TEMPERATURE,
            momentum=config.MOMENTUM_COEFF
        ).to(self.device)

        # 使用DataParallel如果有多GPU
        if torch.cuda.device_count() > 1:
            print(f"🚀 使用 {torch.cuda.device_count()} 个GPU")
            self.model = nn.DataParallel(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🔧 创建MoCo模型: 总参数: {total_params:,}")

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.EPOCHS,
            eta_min=0
        )
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        self.losses = []
        self.accuracies = []

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.EPOCHS}')
        for batch_idx, (im_q, im_k, _) in enumerate(pbar):
            im_q = im_q.to(self.device, non_blocking=False)  # ❗ non_blocking=False
            im_k = im_k.to(self.device, non_blocking=False)  # ❗ non_blocking=False

            if self.scaler is not None:
                with autocast():
                    logits, labels = self.model(im_q, im_k)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, labels = self.model(im_q, im_k)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            acc = (logits.argmax(dim=1) == labels).float().mean().item()
            epoch_loss += loss.item()
            epoch_acc += acc
            num_batches += 1

            # 显示GPU内存使用情况
            if torch.cuda.is_available() and (batch_idx % 50 == 0):
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                cached = torch.cuda.memory_reserved() / 1024 ** 3
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc:.4f}',
                    'GPU_mem': f'{allocated:.2f}/{cached:.2f} GB'
                })

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_acc = epoch_acc / num_batches if num_batches > 0 else 0
        self.scheduler.step()

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, avg_acc

    def train(self, train_loader):
        print("🚀 开始MoCo预训练...")
        best_loss = float('inf')

        for epoch in range(self.config.EPOCHS):
            start_time = time.time()
            loss, acc = self.train_epoch(train_loader, epoch)
            self.losses.append(loss)
            self.accuracies.append(acc)
            epoch_time = time.time() - start_time

            print(f"  Epoch {epoch + 1}/{self.config.EPOCHS}: "
                  f"loss={loss:.4f}, acc={acc:.4f}, time={epoch_time:.1f}s")

            if loss < best_loss:
                best_loss = loss
                self.save_checkpoint(f"moco_best.pth")

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"moco_epoch_{epoch + 1}.pth")

        print(f"✅ MoCo预训练完成! 最佳损失: {best_loss:.4f}")
        self.save_checkpoint("moco_final.pth")
        return self.losses, self.accuracies

    def save_checkpoint(self, filename):
        # 获取模型状态（如果是DataParallel）
        model_state_dict = self.model.state_dict()
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()

        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': self.losses,
            'accuracies': self.accuracies,
            'config': self.config.__dict__
        }
        save_path = os.path.join(self.config.SAVE_DIR, filename)
        torch.save(checkpoint, save_path)
        print(f"💾 模型已保存: {save_path}")

    def extract_features(self, data_loader):
        self.model.eval()
        features_list = []
        labels_list = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="提取特征"):
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images, _, labels = batch

                images = images.to(self.device, non_blocking=False)  # ❗ non_blocking=False

                # 获取模型（如果是DataParallel）
                model_to_use = self.model
                if isinstance(self.model, nn.DataParallel):
                    model_to_use = self.model.module

                encoder = model_to_use.encoder_q
                h = encoder.encoder(images)
                h = torch.flatten(h, 1)

                features_list.append(h.cpu().numpy())
                labels_list.append(labels.numpy())

                # 每处理一些批次就清理缓存
                if len(features_list) % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        features = np.vstack(features_list) if features_list else np.array([])
        labels = np.concatenate(labels_list) if labels_list else np.array([])

        # 最终清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return features, labels


# ========== 分类器 ==========
class Classifier(nn.Module):
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
    def __init__(self, input_dim, num_classes=2, device='cuda', save_dir='./'):
        self.device = torch.device(device)
        self.save_dir = save_dir
        self.model = Classifier(input_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_features, train_labels, val_features=None, val_labels=None,
              epochs=50, lr=0.01, batch_size=128):
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
            pin_memory=False  # ❗ 设为False
        )

        if val_features is not None and val_labels is not None:
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(val_features),
                torch.LongTensor(val_labels)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False  # ❗ 设为False
            )
        else:
            val_loader = None

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device, non_blocking=False)  # ❗
                batch_labels = batch_labels.to(self.device, non_blocking=False)  # ❗

                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()
            train_acc = self.evaluate(train_features, train_labels)

            if val_loader is not None:
                val_acc = self.evaluate(val_features, val_labels)
                val_accuracies.append(val_acc)

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(self.model.state_dict(),
                               os.path.join(self.save_dir, "classifier_best.pth"))

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs}: loss={epoch_loss / len(train_loader):.4f}, "
                          f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs}: loss={epoch_loss / len(train_loader):.4f}, "
                          f"train_acc={train_acc:.4f}")

            train_losses.append(epoch_loss / len(train_loader))

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"✅ 分类器训练完成")
        if val_loader is not None:
            print(f"   最佳验证准确率: {best_acc:.4f}")

        return train_losses, val_accuracies if val_loader else []

    def evaluate(self, features, labels):
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device, non_blocking=False)
            outputs = self.model(features_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()
        accuracy = accuracy_score(labels, predictions)
        return accuracy

    def predict(self, features):
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device, non_blocking=False)
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            predictions = outputs.argmax(dim=1).cpu().numpy()
        return predictions, probabilities

    def load_best_model(self):
        model_path = os.path.join(self.save_dir, "classifier_best.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ 分类器最佳模型已加载: {model_path}")
            return True
        else:
            print(f"⚠️  分类器最佳模型不存在: {model_path}")
            return False


# ========== 评估函数 ==========
def evaluate_performance(y_true, y_pred):
    print("\n📊 模型性能评估:")
    print("-" * 50)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数:            {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n混淆矩阵:")
    print(f"     TN: {cm[0, 0]}   FP: {cm[0, 1]}")
    print(f"     FN: {cm[1, 0]}   TP: {cm[1, 1]}")

    print(f"\n详细分类报告:")
    print(classification_report(y_true, y_pred, target_names=['健康叶片', '病叶'], zero_division=0))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


# ========== 主训练流程 ==========
def main():
    config = Config()
    config.print_config()

    if not os.path.exists(config.DATASET_DIR):
        print(f"\n❌ 错误: 数据集目录不存在: {config.DATASET_DIR}")
        return 1

    augmentation = MoCoDataAugmentation(image_size=config.IMAGE_SIZE)

    print("\n📊 加载数据集...")
    try:
        train_dataset = LeafDiseaseDataset(
            config.DATASET_DIR,
            mode='train',
            transform=augmentation,
            sample_ratio=config.SAMPLE_RATIO
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
            sample_ratio=config.SAMPLE_RATIO
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

    # 动态调整batch_size
    def calculate_batch_size(dataset_size, base_batch_size=64):
        if dataset_size < 100:
            return min(base_batch_size, dataset_size)
        elif dataset_size < 1000:
            return min(32, base_batch_size)
        else:
            return base_batch_size

    train_batch_size = calculate_batch_size(len(train_dataset), config.BATCH_SIZE)
    test_batch_size = calculate_batch_size(len(test_dataset), config.BATCH_SIZE)

    print(f"  训练batch_size: {train_batch_size}")
    print(f"  测试batch_size: {test_batch_size}")

    # 创建DataLoader - ❗关键修复：pin_memory=False
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,  # 设为0
        pin_memory=config.PIN_MEMORY,  # 设为False
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,  # 设为0
        pin_memory=config.PIN_MEMORY  # 设为False
    )

    try:
        # ========== MoCo预训练 ==========
        print("\n" + "=" * 60)
        print("🚀 开始MoCo自监督预训练")
        print("=" * 60)

        moco_trainer = MoCoTrainer(config)
        moco_losses, moco_accuracies = moco_trainer.train(train_loader)

        # ========== 提取特征 ==========
        print("\n" + "=" * 60)
        print("🔍 提取特征")
        print("=" * 60)

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
            sample_ratio=config.SAMPLE_RATIO
        )

        # ❗关键修复：pin_memory=False
        train_features_loader = DataLoader(
            train_features_dataset,
            batch_size=min(64, len(train_features_dataset)),  # 减小batch_size
            shuffle=False,
            num_workers=config.NUM_WORKERS,  # 设为0
            pin_memory=config.PIN_MEMORY  # 设为False
        )

        train_features, train_labels = moco_trainer.extract_features(train_features_loader)

        print("提取测试特征...")
        test_features, test_labels = moco_trainer.extract_features(test_loader)

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
        print("\n" + "=" * 60)
        print("🎯 训练分类器")
        print("=" * 60)

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
        print("\n" + "=" * 60)
        print("📊 评估模型性能")
        print("=" * 60)

        test_predictions, test_probabilities = classifier_trainer.predict(test_features_normalized)
        metrics = evaluate_performance(test_labels, test_predictions)

        # ========== 保存结果 ==========
        print("\n💾 保存结果...")
        results = {
            'test_labels': test_labels,
            'test_predictions': test_predictions,
            'test_probabilities': test_probabilities,
            'metrics': metrics,
            'train_features_shape': train_features.shape,
            'test_features_shape': test_features.shape,
            'moco_losses': moco_losses,
            'moco_accuracies': moco_accuracies,
            'classifier_losses': classifier_losses,
            'classifier_accuracies': classifier_accuracies
        }

        results_path = os.path.join(config.SAVE_DIR, "results.npz")
        np.savez(results_path, **results)

        print(f"✅ 结果已保存: {results_path}")
        print(f"\n🎉 MoCo叶片病害检测完成!")

        return 0

    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ========== 命令行接口 ==========
if __name__ == "__main__":
    exit_code = main()

    if exit_code == 0:
        print("\n✅ 程序成功完成!")
    else:
        print(f"\n❌ 程序失败，退出码: {exit_code}")

    sys.exit(exit_code)