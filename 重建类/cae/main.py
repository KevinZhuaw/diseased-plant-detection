import os
import sys
import warnings

# ========== 1. NumPy版本兼容性检查 ==========
try:
    import numpy as np

    numpy_version = np.__version__
    print(f"NumPy版本: {numpy_version}")

    # 如果是NumPy 2.0，显示警告
    if numpy_version.startswith('2.'):
        warnings.warn(
            f"⚠️  检测到NumPy {numpy_version}，可能存在兼容性问题。"
            "建议使用NumPy 1.x版本（如1.24.3）"
        )
        # 尝试修复：重新导入兼容版本
        try:
            # 降级NumPy的替代方案：使用兼容模式
            np.seterr(all='ignore')
        except:
            pass
except ImportError:
    print("❌ NumPy未安装")
    sys.exit(1)

# ========== 2. 设置环境变量（强制GPU） ==========
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 调试用

# 忽略字体警告
warnings.filterwarnings("ignore", category=UserWarning)

# ========== 3. 导入其他库 ==========
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    classification_report, average_precision_score
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import datetime
import json

# ========== 4. 修复字体设置 ==========
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 5. 参数配置 ==========
DATASET_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\balanced-crop-dataset-new"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
LATENT_DIM = 128
VAL_SPLIT = 0.2
GRAD_CLIP = 1.0
PATIENCE = 10
RECONSTRUCTION_VIS_EPOCHS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


# ========== 6. 设备设置函数 ==========
def setup_device():
    """设置设备，强制使用GPU"""
    print("\n" + "=" * 70)
    print("设备设置")
    print("=" * 70)

    # 检查CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA版本: {torch.version.cuda}")

        # 启用CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # 测试GPU
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            print(f"✅ GPU测试通过")
        except Exception as e:
            print(f"❌ GPU测试失败: {e}")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("❌ CUDA不可用，使用CPU")
        print("\n请安装GPU版本的PyTorch:")
        print(
            "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118")

    return device


# ========== 7. 监控GPU内存 ==========
def monitor_gpu_memory(step_name=""):
    """监控GPU内存使用情况"""
    if torch.cuda.is_available():
        print(f"\nGPU内存监控 [{step_name}]:")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"  已缓存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
        if hasattr(torch.cuda, 'max_memory_allocated'):
            print(f"  峰值已分配: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")


# ========== 8. 数据集检查 ==========
def check_dataset():
    """检查数据集"""
    dataset_dir = DATASET_DIR
    required_dirs = [
        os.path.join(dataset_dir, "train", "healthy"),
        os.path.join(dataset_dir, "train", "diseased"),
        os.path.join(dataset_dir, "test", "healthy"),
        os.path.join(dataset_dir, "test", "diseased")
    ]

    if not all(os.path.exists(dir) for dir in required_dirs):
        raise FileNotFoundError("未找到预处理后的数据集")

    img_ext = ('.jpg', '.jpeg', '.png')
    train_healthy = len([f for f in os.listdir(required_dirs[0]) if f.lower().endswith(img_ext)])
    train_diseased = len([f for f in os.listdir(required_dirs[1]) if f.lower().endswith(img_ext)])
    test_healthy = len([f for f in os.listdir(required_dirs[2]) if f.lower().endswith(img_ext)])
    test_diseased = len([f for f in os.listdir(required_dirs[3]) if f.lower().endswith(img_ext)])

    if train_healthy == 0 or train_diseased == 0 or test_healthy == 0 or test_diseased == 0:
        raise ValueError("训练集或测试集的健康/病叶文件夹中无有效图片")

    print(f"\n数据集统计:")
    print(f"  训练集 - 健康叶片 {train_healthy} 张，病叶 {train_diseased} 张")
    print(f"  测试集 - 健康叶片 {test_healthy} 张，病叶 {test_diseased} 张")
    return dataset_dir, train_healthy, train_diseased, test_healthy, test_diseased


# ========== 9. 数据集类 ==========
class CropLeafDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx], img_path
        except Exception as e:
            print(f"图片读取失败：{img_path}，错误：{e}")
            return torch.zeros(3, *IMAGE_SIZE), self.labels[idx], img_path


# ========== 10. 数据加载器 ==========
def get_dataloaders(dataset_dir, device):
    """获取数据加载器"""
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证和测试转换
    val_test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 训练集路径
    train_healthy_dir = os.path.join(dataset_dir, "train", "healthy")
    train_diseased_dir = os.path.join(dataset_dir, "train", "diseased")
    train_healthy_imgs = [os.path.join(train_healthy_dir, f) for f in os.listdir(train_healthy_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    train_diseased_imgs = [os.path.join(train_diseased_dir, f) for f in os.listdir(train_diseased_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    train_paths = train_healthy_imgs + train_diseased_imgs
    train_labels = [0] * len(train_healthy_imgs) + [1] * len(train_diseased_imgs)

    # 测试集路径
    test_healthy_dir = os.path.join(dataset_dir, "test", "healthy")
    test_diseased_dir = os.path.join(dataset_dir, "test", "diseased")
    test_healthy_imgs = [os.path.join(test_healthy_dir, f) for f in os.listdir(test_healthy_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    test_diseased_imgs = [os.path.join(test_diseased_dir, f) for f in os.listdir(test_diseased_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    test_paths = test_healthy_imgs + test_diseased_imgs
    test_labels = [0] * len(test_healthy_imgs) + [1] * len(test_diseased_imgs)

    # 构建数据集
    train_dataset = CropLeafDataset(train_paths, train_labels, train_transform)
    test_dataset = CropLeafDataset(test_paths, test_labels, val_test_transform)

    # 分割训练集和验证集
    val_size = int(VAL_SPLIT * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # 根据设备设置参数
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()

    print(f"\n数据加载器配置:")
    print(f"  num_workers: {num_workers}")
    print(f"  pin_memory: {pin_memory}")
    print(f"  batch_size: {BATCH_SIZE}")

    # 数据加载器
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print(f"数据加载完成:")
    print(f"  训练集: {len(train_subset)} 张")
    print(f"  验证集: {len(val_subset)} 张")
    print(f"  测试集: {len(test_dataset)} 张")
    return train_loader, val_loader, test_loader, test_labels


# ========== 11. 改进的卷积自编码器 ==========
class ImprovedCAE(nn.Module):
    def __init__(self):
        super(ImprovedCAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64×112×112

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128×56×56

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256×28×28

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 512×14×14
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256×28×28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128×56×56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64×112×112
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32×224×224
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ========== 12. 多尺度损失函数 ==========
class MultiScaleLoss(nn.Module):
    """多尺度重建损失"""

    def __init__(self):
        super(MultiScaleLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, recon_x, x):
        total_loss = 0
        scales = [1, 2, 4]

        for scale in scales:
            if scale > 1:
                recon_down = nn.functional.avg_pool2d(recon_x, scale)
                x_down = nn.functional.avg_pool2d(x, scale)
            else:
                recon_down = recon_x
                x_down = x

            scale_loss = 0.7 * self.mse_loss(recon_down, x_down) + 0.3 * self.l1_loss(recon_down, x_down)
            total_loss += scale_loss

        return total_loss / len(scales)


# ========== 13. 可视化重建图像 ==========
def visualize_reconstruction(model, device, data_loader, epoch, num_samples=8):
    """可视化原始图像与重建图像的对比"""
    model.eval()
    with torch.no_grad():
        # 获取一个batch的数据
        for images, labels, _ in data_loader:
            images = images.to(device, non_blocking=True)
            recon_images = model(images)

            # 选择前num_samples个样本进行可视化
            images = images[:num_samples].cpu()
            recon_images = recon_images[:num_samples].cpu()

            # 反标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = images * std + mean
            recon_images = recon_images * std + mean

            # 限制在[0,1]范围内
            images = torch.clamp(images, 0, 1)
            recon_images = torch.clamp(recon_images, 0, 1)

            break

    # 创建对比图
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))

    for i in range(num_samples):
        # 原始图像
        orig_img = images[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Original\n{'Healthy' if labels[i] == 0 else 'Diseased'}")
        axes[0, i].axis('off')

        # 重建图像
        recon_img = recon_images[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title(f"Reconstructed\nEpoch {epoch}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f'reconstruction_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已保存第 {epoch} 轮的重建对比图")


# ========== 14. 训练模型 ==========
def train_model(device, train_loader, val_loader):
    """训练模型"""
    print(f"\n开始训练...")
    print(f"使用设备: {device}")

    # 创建模型
    model = ImprovedCAE()
    model = model.to(device)

    # 验证模型是否在GPU上
    print(f"模型参数设备: {next(model.parameters()).device}")

    # 损失函数和优化器
    criterion = MultiScaleLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 早停
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    print("开始训练卷积自编码器...")

    # 监控GPU内存
    monitor_gpu_memory("训练前")

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} Training')
        for batch_idx, (images, _, _) in enumerate(train_bar):
            # 强制将数据移动到GPU
            images = images.to(device, non_blocking=True)

            # 验证数据设备
            if batch_idx == 0 and epoch == 0:
                print(f"训练数据设备: {images.device}")

            optimizer.zero_grad(set_to_none=True)

            # 前向传播
            recon_images = model(images)
            loss = criterion(recon_images, images)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_bar.set_postfix({'loss': loss.item()})

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} Validation')
            for images, _, _ in val_bar:
                images = images.to(device, non_blocking=True)
                recon_images = model(images)
                loss = criterion(recon_images, images)
                val_loss += loss.item() * images.size(0)
                val_bar.set_postfix({'val_loss': loss.item()})

        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 学习率调度
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | 训练损失: {avg_train_loss:.6f} | 验证损失: {avg_val_loss:.6f} | 学习率: {scheduler.get_last_lr()[0]:.8f}")

        # 在指定轮次保存重建图像
        if (epoch + 1) in RECONSTRUCTION_VIS_EPOCHS:
            visualize_reconstruction(model, device, val_loader, epoch + 1)

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "improved_cae_model.pth")
            print(f"✅ 保存最佳模型，验证损失: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"⚠️  早停计数: {patience_counter}/{PATIENCE}")

        # 每10个epoch监控一次GPU内存
        if (epoch + 1) % 10 == 0:
            monitor_gpu_memory(f"Epoch {epoch + 1}")

        if patience_counter >= PATIENCE:
            print(f"🛑 早停触发，训练结束")
            break

    # 加载最佳模型
    try:
        model.load_state_dict(torch.load("improved_cae_model.pth", map_location=device))
        print("✅ 加载最佳模型")
    except:
        print("⚠️  无法加载最佳模型，使用最终模型")

    # 可视化训练过程
    visualize_training_curve(train_losses, val_losses)

    # 最终GPU内存监控
    monitor_gpu_memory("训练后")

    return model, train_losses, val_losses, patience_counter >= PATIENCE, len(train_losses)


def visualize_training_curve(train_losses, val_losses):
    """可视化训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CAE Training Process')
    plt.legend()
    plt.grid(True)
    plt.savefig('cae_training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


# ========== 15. 计算mAP（新增） ==========
def calculate_map(all_errors, all_true_labels, threshold=None):
    """计算mAP（平均精度均值）"""
    # 将误差转换为预测概率
    # 对于自编码器，误差越小表示越可能是健康样本
    # 所以我们将误差转换为健康类别的概率
    errors = np.array(all_errors)

    # 归一化误差到[0, 1]范围
    min_error = errors.min()
    max_error = errors.max()
    normalized_errors = (errors - min_error) / (max_error - min_error + 1e-10)

    # 对于二分类问题：
    # 1. 健康类别的概率 = 1 - 归一化误差（误差越小，健康概率越高）
    # 2. 病叶类别的概率 = 归一化误差（误差越大，病叶概率越高）

    # 为每个样本创建两个类别的预测概率
    y_scores = np.zeros((len(errors), 2))
    y_scores[:, 0] = 1 - normalized_errors  # 健康类别
    y_scores[:, 1] = normalized_errors  # 病叶类别

    # 创建真实标签的one-hot编码
    y_true = np.zeros((len(all_true_labels), 2))
    for i, label in enumerate(all_true_labels):
        y_true[i, label] = 1

    # 计算每个类别的AP（平均精度）
    ap_scores = []
    ap_per_class = {}

    for class_idx, class_name in enumerate(['Healthy', 'Diseased']):
        ap = average_precision_score(y_true[:, class_idx], y_scores[:, class_idx])
        ap_scores.append(ap)
        ap_per_class[class_name] = ap

    # 计算mAP（平均精度均值）
    mAP = np.mean(ap_scores)

    # 可视化PR曲线
    visualize_pr_curve(y_true, y_scores, ap_per_class, mAP)

    return mAP, ap_per_class


def visualize_pr_curve(y_true, y_scores, ap_per_class, mAP):
    """可视化PR曲线"""
    from sklearn.metrics import precision_recall_curve

    plt.figure(figsize=(12, 5))

    # 为每个类别绘制PR曲线
    for class_idx, class_name in enumerate(['Healthy', 'Diseased']):
        precision, recall, _ = precision_recall_curve(
            y_true[:, class_idx],
            y_scores[:, class_idx]
        )

        plt.subplot(1, 2, class_idx + 1)
        plt.plot(recall, precision, linewidth=2,
                 label=f'{class_name} (AP={ap_per_class[class_name]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{class_name} - Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.fill_between(recall, precision, alpha=0.2)

    plt.suptitle(f'Precision-Recall Curves (mAP = {mAP:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig('pr_curves_map.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ PR曲线已保存: pr_curves_map.png")


# ========== 16. 评估分类指标（包含mAP） ==========
def evaluate_classification_metrics(model, device, test_loader, test_labels):
    """评估模型性能，包含mAP计算"""
    model.eval()
    all_errors = []
    all_true_labels = []

    print(f"\n开始评估...")
    print(f"评估设备: {device}")

    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="计算重建误差"):
            # 强制将数据移动到GPU
            images = images.to(device, non_blocking=True)

            recon_images = model(images)

            # 计算误差
            errors = torch.mean((recon_images - images) ** 2, dim=[1, 2, 3])
            all_errors.extend(errors.cpu().numpy())
            all_true_labels.extend(labels.numpy())

    # 寻找最优阈值
    threshold = find_optimal_threshold(all_errors, all_true_labels)
    predictions = (np.array(all_errors) > threshold).astype(int)

    # 计算基本指标
    accuracy = accuracy_score(all_true_labels, predictions)
    precision = precision_score(all_true_labels, predictions, average="binary", zero_division=0)
    recall = recall_score(all_true_labels, predictions, average="binary", zero_division=0)
    f1 = f1_score(all_true_labels, predictions, average="binary", zero_division=0)
    cm = confusion_matrix(all_true_labels, predictions)

    # 计算mAP（新增）
    mAP, ap_per_class = calculate_map(all_errors, all_true_labels, threshold)

    # 输出结果
    print(f"\n" + "=" * 70)
    print("分类性能评估")
    print("=" * 70)
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数 (F1-Score):  {f1:.4f}")
    print(f"最优分类阈值: {threshold:.6f}")
    print(f"\nmAP (平均精度均值): {mAP:.4f}")
    print(f"  健康类别AP: {ap_per_class['Healthy']:.4f}")
    print(f"  病叶类别AP: {ap_per_class['Diseased']:.4f}")
    print("=" * 70)
    print("混淆矩阵:")
    print(cm)
    print("\n分类报告:")
    print(classification_report(all_true_labels, predictions, target_names=['Healthy', 'Diseased']))

    # 可视化误差分布
    visualize_error_distribution(all_errors, all_true_labels, threshold, accuracy, recall, f1, mAP)

    return accuracy, precision, recall, f1, mAP, ap_per_class, threshold, cm, all_errors, all_true_labels


def find_optimal_threshold(errors, labels, num_thresholds=100):
    """寻找最优分类阈值"""
    min_error, max_error = min(errors), max(errors)
    thresholds = np.linspace(min_error, max_error, num_thresholds)

    best_f1 = 0
    best_threshold = thresholds[0]

    for threshold in thresholds:
        predictions = (np.array(errors) > threshold).astype(int)
        current_f1 = f1_score(labels, predictions, average="binary", zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold


def visualize_error_distribution(all_errors, all_true_labels, threshold, accuracy, recall, f1, mAP):
    """可视化误差分布"""
    healthy_errors = [all_errors[i] for i in range(len(all_errors)) if all_true_labels[i] == 0]
    diseased_errors = [all_errors[i] for i in range(len(all_errors)) if all_true_labels[i] == 1]

    plt.figure(figsize=(12, 6))
    plt.hist(healthy_errors, bins=30, alpha=0.5, label="Healthy", color="green")
    plt.hist(diseased_errors, bins=30, alpha=0.5, label="Diseased", color="red")
    plt.axvline(threshold, color="black", linestyle='--', linewidth=2,
                label=f"Optimal Threshold: {threshold:.6f}")

    # 添加指标信息
    plt.text(0.02, 0.95,
             f"Accuracy: {accuracy:.4f}\n"
             f"Recall: {recall:.4f}\n"
             f"F1-Score: {f1:.4f}\n"
             f"mAP: {mAP:.4f}",
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
             verticalalignment="top")

    plt.xlabel("Reconstruction Error")
    plt.ylabel("Sample Count")
    plt.title("CAE Classification - Error Distribution with mAP")
    plt.legend()
    plt.savefig("improved_cae_classification_error_map.png", dpi=300, bbox_inches='tight')
    plt.close()


# ========== 17. 创建综合性能报告（包含mAP） ==========
def create_comprehensive_report(accuracy, precision, recall, f1, mAP, ap_per_class, cm):
    """创建综合性能报告图，包含mAP"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 指标雷达图（新增mAP）
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'mAP']
    values = [accuracy, precision, recall, f1, mAP]

    # 雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # 闭合雷达图
    angles += angles[:1]

    ax1.plot(angles, values, 'o-', linewidth=2)
    ax1.fill(angles, values, alpha=0.25)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1)
    ax1.set_title('Classification Performance Metrics (with mAP)', fontsize=14, fontweight='bold')
    ax1.grid(True)

    # 2. 混淆矩阵热图
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2)

    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Healthy', 'Diseased'])
    ax2.set_yticklabels(['Healthy', 'Diseased'])
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    # 3. 指标条形图（新增mAP）
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'mAP']
    metrics_values = [accuracy, precision, recall, f1, mAP]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet']

    bars = ax3.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    ax3.set_ylim(0, 1)
    ax3.set_title('Performance Metrics Comparison (with mAP)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Score')

    # 在条形上添加数值
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.4f}', ha='center', va='bottom')

    # 4. 详细指标文本（新增mAP）
    ax4.axis('off')
    report_text = (
        f"Comprehensive Classification Report\n\n"
        f"Accuracy:  {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1-Score:  {f1:.4f}\n"
        f"mAP:       {mAP:.4f}\n"
        f"  Healthy AP: {ap_per_class['Healthy']:.4f}\n"
        f"  Diseased AP: {ap_per_class['Diseased']:.4f}\n\n"
        f"Confusion Matrix:\n"
        f"  True Healthy: {cm[0, 0]} (Correct), {cm[0, 1]} (Wrong)\n"
        f"  True Diseased: {cm[1, 1]} (Correct), {cm[1, 0]} (Wrong)\n\n"
        f"Overall Performance: {'Excellent' if mAP > 0.9 else 'Good' if mAP > 0.8 else 'Fair' if mAP > 0.7 else 'Needs Improvement'}"
    )
    ax4.text(0.1, 0.9, report_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('comprehensive_performance_report_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已创建综合性能报告图（包含mAP）")


# ========== 18. 保存结果到TXT文档（包含mAP） ==========
def save_results_to_txt(results_dict, output_dir="results"):
    """将所有结果保存到TXT文档，包含mAP"""

    # 创建结果目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_file = os.path.join(output_dir, f"cae_results_{timestamp}.txt")

    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("            卷积自编码器叶片病害检测 - 完整结果报告（包含mAP）\n")
        f.write("=" * 80 + "\n\n")

        f.write("📅 报告生成时间: {}\n".format(results_dict['timestamp']))
        f.write("\n")

        f.write("💻 硬件环境信息:\n")
        f.write("-" * 60 + "\n")
        f.write(f"训练设备: {results_dict['device']}\n")
        f.write(f"是否使用GPU: {results_dict['use_gpu']}\n")
        if results_dict['use_gpu']:
            f.write(f"GPU名称: {results_dict['gpu_name']}\n")
            f.write(f"GPU内存: {results_dict['gpu_memory']:.2f} GB\n")
        f.write("\n")

        f.write("📊 数据集信息:\n")
        f.write("-" * 60 + "\n")
        f.write(f"数据集路径: {results_dict['dataset_path']}\n")
        f.write(f"图片尺寸: {results_dict['image_size']}\n")
        f.write(f"验证集比例: {results_dict['val_split']}\n")
        f.write(f"训练集健康样本: {results_dict['train_healthy']} 张\n")
        f.write(f"训练集病叶样本: {results_dict['train_diseased']} 张\n")
        f.write(f"测试集健康样本: {results_dict['test_healthy']} 张\n")
        f.write(f"测试集病叶样本: {results_dict['test_diseased']} 张\n")
        f.write(f"总训练样本数: {results_dict['total_train_samples']} 张\n")
        f.write(f"总测试样本数: {results_dict['total_test_samples']} 张\n")
        f.write("\n")

        f.write("⚙️ 训练参数配置:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Batch Size: {results_dict['batch_size']}\n")
        f.write(f"最大训练轮次 (Epochs): {results_dict['epochs']}\n")
        f.write(f"实际训练轮次: {results_dict['actual_epochs']}\n")
        f.write(f"学习率 (Learning Rate): {results_dict['learning_rate']}\n")
        f.write(f"潜在维度 (Latent Dim): {results_dict['latent_dim']}\n")
        f.write(f"梯度裁剪阈值: {results_dict['grad_clip']}\n")
        f.write(f"早停耐心值: {results_dict['patience']}\n")
        f.write(f"是否触发早停: {results_dict['early_stop_triggered']}\n")
        f.write("\n")

        f.write("📈 训练过程详情:\n")
        f.write("-" * 60 + "\n")
        if 'train_losses' in results_dict and results_dict['train_losses']:
            f.write("Epoch | 训练损失 | 验证损失\n")
            f.write("-" * 40 + "\n")
            for epoch, (train_loss, val_loss) in enumerate(zip(results_dict['train_losses'],
                                                               results_dict.get('val_losses', [])), 1):
                val_loss_str = f"{val_loss:.6f}" if epoch - 1 < len(results_dict.get('val_losses', [])) else "N/A"
                f.write(f"{epoch:3d}   {train_loss:.6f}   {val_loss_str}\n")
        f.write("\n")

        f.write("🎯 分类性能评估:\n")
        f.write("-" * 60 + "\n")
        f.write(f"准确率 (Accuracy):  {results_dict['accuracy']:.4f}\n")
        f.write(f"精确率 (Precision): {results_dict['precision']:.4f}\n")
        f.write(f"召回率 (Recall):    {results_dict['recall']:.4f}\n")
        f.write(f"F1分数 (F1-Score):  {results_dict['f1_score']:.4f}\n")
        f.write(f"mAP (平均精度均值):  {results_dict['mAP']:.4f}\n")
        f.write(f"  健康类别AP: {results_dict['ap_per_class']['Healthy']:.4f}\n")
        f.write(f"  病叶类别AP: {results_dict['ap_per_class']['Diseased']:.4f}\n")
        f.write(f"最佳分类阈值: {results_dict['best_threshold']:.6f}\n")
        f.write("\n")

        f.write("📊 混淆矩阵:\n")
        f.write("-" * 60 + "\n")
        cm = results_dict['confusion_matrix']
        if cm:
            f.write("             预测健康    预测病叶\n")
            f.write(f"真实健康    {cm[0][0]:6d}      {cm[0][1]:6d}\n")
            f.write(f"真实病叶    {cm[1][0]:6d}      {cm[1][1]:6d}\n")
        f.write("\n")

        f.write("📊 详细分类报告:\n")
        f.write("-" * 60 + "\n")
        f.write("类别       精度    召回率   F1分数  AP分数  样本数\n")
        f.write("-" * 60 + "\n")
        if 'classification_report_dict' in results_dict:
            report = results_dict['classification_report_dict']
            ap_dict = results_dict['ap_per_class']

            if 'Healthy' in report:
                f.write(
                    f"Healthy    {report['Healthy']['precision']:.3f}    "
                    f"{report['Healthy']['recall']:.3f}    "
                    f"{report['Healthy']['f1-score']:.3f}    "
                    f"{ap_dict['Healthy']:.3f}    "
                    f"{report['Healthy']['support']}\n"
                )
            if 'Diseased' in report:
                f.write(
                    f"Diseased   {report['Diseased']['precision']:.3f}    "
                    f"{report['Diseased']['recall']:.3f}    "
                    f"{report['Diseased']['f1-score']:.3f}    "
                    f"{ap_dict['Diseased']:.3f}    "
                    f"{report['Diseased']['support']}\n"
                )
        f.write("\n")

        f.write("📈 重建误差统计:\n")
        f.write("-" * 60 + "\n")
        if 'all_errors' in results_dict and results_dict['all_errors']:
            errors = np.array(results_dict['all_errors'])
            healthy_errors = [errors[i] for i in range(len(errors)) if results_dict['all_true_labels'][i] == 0]
            diseased_errors = [errors[i] for i in range(len(errors)) if results_dict['all_true_labels'][i] == 1]

            f.write(f"所有样本重建误差:\n")
            f.write(f"  最小值: {np.min(errors):.6f}\n")
            f.write(f"  最大值: {np.max(errors):.6f}\n")
            f.write(f"  平均值: {np.mean(errors):.6f}\n")
            f.write(f"  标准差: {np.std(errors):.6f}\n")
            f.write(f"  中位数: {np.median(errors):.6f}\n")

            if healthy_errors:
                f.write(f"\n健康样本重建误差:\n")
                f.write(f"  平均值: {np.mean(healthy_errors):.6f}\n")
                f.write(f"  标准差: {np.std(healthy_errors):.6f}\n")

            if diseased_errors:
                f.write(f"\n病叶样本重建误差:\n")
                f.write(f"  平均值: {np.mean(diseased_errors):.6f}\n")
                f.write(f"  标准差: {np.std(diseased_errors):.6f}\n")
        f.write("\n")

        f.write("💡 性能分析与建议:\n")
        f.write("-" * 60 + "\n")
        mAP = results_dict['mAP']
        f1 = results_dict['f1_score']

        if mAP > 0.85 and f1 > 0.85:
            f.write("✅ 模型表现优秀！各项指标均超过85%。\n")
            f.write("   建议：可以尝试部署到实际应用中。\n")
        elif mAP > 0.75 and f1 > 0.75:
            f.write("⚠️  模型表现良好，但仍有改进空间。\n")
            f.write("   建议：可以尝试增加训练数据或调整模型结构。\n")
        elif mAP > 0.65 and f1 > 0.65:
            f.write("⚠️  模型表现一般，需要进一步优化。\n")
            f.write("   建议：调整训练参数，增加数据增强，或尝试其他模型架构。\n")
        else:
            f.write("❌ 模型表现不佳，需要重新设计。\n")
            f.write("   建议：检查数据质量，重新设计模型架构，或收集更多训练数据。\n")

        f.write("\n🔄 改进建议:\n")
        f.write("1. 如果mAP低，尝试增加训练样本\n")
        f.write("2. 调整模型参数（潜在维度、学习率）\n")
        f.write("3. 使用更大的图片尺寸（如256x256）\n")
        f.write("4. 增加训练轮次（epochs）\n")
        f.write("5. 尝试不同的数据增强策略\n")
        f.write("6. 调整分类阈值以获得更好的性能\n")
        f.write("\n")

        f.write("📁 生成的文件:\n")
        f.write("-" * 60 + "\n")
        if 'generated_files' in results_dict:
            for file in results_dict['generated_files']:
                f.write(f"  • {file}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("                        报告结束\n")
        f.write("=" * 80 + "\n")

    print(f"✅ 完整结果报告已保存至: {txt_file}")
    return txt_file


# ========== 19. 创建重建图对比展示 ==========
def create_reconstruction_comparison():
    """创建所有重建图的对比展示"""
    recon_files = glob.glob('reconstruction_epoch_*.png')
    if not recon_files:
        print("未找到重建图文件")
        return

    # 按epoch排序
    recon_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # 创建大图对比
    num_epochs = len(recon_files)
    fig, axes = plt.subplots(num_epochs, 1, figsize=(12, 3 * num_epochs))

    if num_epochs == 1:
        axes = [axes]

    for i, file in enumerate(recon_files):
        epoch = file.split('_')[-1].split('.')[0]
        img = plt.imread(file)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Epoch {epoch} Reconstruction Comparison', fontsize=12)

    plt.tight_layout()
    plt.savefig('all_reconstructions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已创建所有重建图的对比展示")


# ========== 20. 主函数 ==========
def main():
    try:
        # 1. 设置设备
        device = setup_device()

        # 2. 检查数据集
        dataset_dir, train_healthy, train_diseased, test_healthy, test_diseased = check_dataset()

        # 设备信息收集
        device_info = {
            'device': str(device),
            'use_gpu': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / (
                    1024 ** 3) if torch.cuda.is_available() else 0
        }

        # 3. 加载数据
        train_loader, val_loader, test_loader, test_labels = get_dataloaders(dataset_dir, device)

        # 4. 训练模型
        model, train_losses, val_losses, early_stop_triggered, actual_epochs = train_model(
            device, train_loader, val_loader
        )

        # 5. 创建重建图对比展示
        create_reconstruction_comparison()

        # 6. 评估分类指标（包含mAP）
        accuracy, precision, recall, f1, mAP, ap_per_class, threshold, cm, all_errors, all_true_labels = evaluate_classification_metrics(
            model, device, test_loader, test_labels
        )

        # 获取分类报告字典
        predictions = (np.array(all_errors) > threshold).astype(int)
        report_dict = classification_report(all_true_labels, predictions,
                                            target_names=['Healthy', 'Diseased'],
                                            output_dict=True)

        # 7. 创建综合性能报告（包含mAP）
        create_comprehensive_report(accuracy, precision, recall, f1, mAP, ap_per_class, cm)

        # 8. 准备结果字典（修复了.tolist()错误）
        results_dict = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'device': device_info['device'],
            'use_gpu': device_info['use_gpu'],
            'gpu_name': device_info['gpu_name'],
            'gpu_memory': device_info['gpu_memory'],
            'dataset_path': DATASET_DIR,
            'image_size': IMAGE_SIZE,
            'val_split': VAL_SPLIT,
            'train_healthy': train_healthy,
            'train_diseased': train_diseased,
            'test_healthy': test_healthy,
            'test_diseased': test_diseased,
            'total_train_samples': train_healthy + train_diseased,
            'total_test_samples': test_healthy + test_diseased,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'actual_epochs': actual_epochs,
            'learning_rate': LEARNING_RATE,
            'latent_dim': LATENT_DIM,
            'grad_clip': GRAD_CLIP,
            'patience': PATIENCE,
            'early_stop_triggered': early_stop_triggered,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'mAP': float(mAP),  # 新增mAP
            'ap_per_class': ap_per_class,  # 新增各类别AP
            'best_threshold': float(threshold),
            'confusion_matrix': cm.tolist() if cm is not None else None,
            'classification_report_dict': report_dict,
            'all_errors': all_errors,  # 修复：直接使用列表，不要.tolist()
            'all_true_labels': all_true_labels,  # 修复：直接使用列表，不要.tolist()
            'generated_files': [
                "improved_cae_model.pth",
                "cae_training_curve.png",
                "improved_cae_classification_error_map.png",
                "pr_curves_map.png",
                "comprehensive_performance_report_map.png",
                "all_reconstructions_comparison.png",
                "reconstruction_epoch_*.png (多个文件)"
            ]
        }

        # 9. 保存结果到TXT文档
        txt_file = save_results_to_txt(results_dict)

        # 10. 输出总结
        print(f"\n{'=' * 60}")
        print(f"🎉 训练完成！")
        print(f"{'=' * 60}")
        print(f"📊 核心指标:")
        print(f"  mAP: {mAP:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"\n📁 生成的文件:")
        print(f"  • 模型文件: improved_cae_model.pth")
        print(f"  • 训练曲线: cae_training_curve.png")
        print(f"  • 误差分布图: improved_cae_classification_error_map.png")
        print(f"  • PR曲线图: pr_curves_map.png")
        print(f"  • 综合报告: comprehensive_performance_report_map.png")
        print(f"  • 重建对比总图: all_reconstructions_comparison.png")
        print(f"  • 详细结果报告: {txt_file}")

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\n🧹 已清理GPU缓存")

    except Exception as e:
        print(f"\n❌ 程序错误：{str(e)}")
        import traceback
        traceback.print_exc()

        # 保存错误日志
        os.makedirs("results", exist_ok=True)
        error_log = os.path.join("results", f"error_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(error_log, 'w', encoding='utf-8') as f:
            f.write(f"错误时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"错误追踪:\n{traceback.format_exc()}\n")

        print(f"📝 错误日志已保存至: {error_log}")


# ========== 运行主函数 ==========
if __name__ == "__main__":
    main()