import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.manifold import TSNE
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# 忽略字体警告
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------- 简化字体设置 --------------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 只保留一个中文字体
plt.rcParams["axes.unicode_minus"] = False

# -------------------------- 核心参数调整（优化重构质量） --------------------------
DATASET_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\processed-crop-dataset-split"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 3e-4
LATENT_DIM = 512
VAL_SPLIT = 0.2
VIS_SAMPLE_NUM = 8
img_ext = ('.jpg', '.jpeg', '.png')
GRAD_CLIP = 1.0
PATIENCE = 20


# -------------------------- 1. 环境检查 --------------------------
def check_environment():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"使用GPU训练：{torch.cuda.get_device_name(0)}")
        print(f"GPU显存：{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    else:
        raise RuntimeError("未检测到可用GPU！请确保已安装CUDA并正确配置PyTorch-GPU版本")
    return device


# -------------------------- 2. 数据集检查 --------------------------
def check_dataset():
    required_dirs = [
        os.path.join(DATASET_DIR, "train", "healthy"),
        os.path.join(DATASET_DIR, "train", "diseased"),
        os.path.join(DATASET_DIR, "test", "healthy"),
        os.path.join(DATASET_DIR, "test", "diseased")
    ]
    if not all(os.path.exists(d) for d in required_dirs):
        raise FileNotFoundError("未找到预处理后的数据集")

    train_healthy = len([f for f in os.listdir(required_dirs[0]) if f.lower().endswith(img_ext)])
    train_diseased = len([f for f in os.listdir(required_dirs[1]) if f.lower().endswith(img_ext)])
    test_healthy = len([f for f in os.listdir(required_dirs[2]) if f.lower().endswith(img_ext)])
    test_diseased = len([f for f in os.listdir(required_dirs[3]) if f.lower().endswith(img_ext)])

    if train_healthy == 0 or train_diseased == 0 or test_healthy == 0 or test_diseased == 0:
        raise ValueError("训练集或测试集的健康/病叶文件夹中无有效图片")

    print(f"预处理后数据集总统计：")
    print(f"  训练集 - 健康叶片 {train_healthy} 张，病叶 {train_diseased} 张")
    print(f"  测试集 - 健康叶片 {test_healthy} 张，病叶 {test_diseased} 张")
    return DATASET_DIR


# -------------------------- 3. 数据集与数据加载 --------------------------
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


def get_dataloaders(dataset_dir):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载训练集
    train_healthy_dir = os.path.join(dataset_dir, "train", "healthy")
    train_diseased_dir = os.path.join(dataset_dir, "train", "diseased")
    train_healthy_imgs = [os.path.join(train_healthy_dir, f) for f in os.listdir(train_healthy_dir) if
                          f.lower().endswith(img_ext)]
    train_diseased_imgs = [os.path.join(train_diseased_dir, f) for f in os.listdir(train_diseased_dir) if
                           f.lower().endswith(img_ext)]
    train_paths = train_healthy_imgs + train_diseased_imgs
    train_labels = [0] * len(train_healthy_imgs) + [1] * len(train_diseased_imgs)
    full_train_dataset = CropLeafDataset(train_paths, train_labels, train_transform)
    val_size = int(VAL_SPLIT * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # 加载测试集
    test_healthy_imgs = [os.path.join(dataset_dir, "test", "healthy", f) for f in
                         os.listdir(os.path.join(dataset_dir, "test", "healthy")) if f.lower().endswith(img_ext)]
    test_diseased_imgs = [os.path.join(dataset_dir, "test", "diseased", f) for f in
                          os.listdir(os.path.join(dataset_dir, "test", "diseased")) if f.lower().endswith(img_ext)]
    test_paths = test_healthy_imgs + test_diseased_imgs
    test_labels = [0] * len(test_healthy_imgs) + [1] * len(test_diseased_imgs)
    test_dataset = CropLeafDataset(test_paths, test_labels, val_test_transform)

    # 数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"数据加载完成：")
    print(f"  训练集 {len(train_dataset)} 张，验证集 {len(val_dataset)} 张，测试集 {len(test_dataset)} 张")
    return train_loader, val_loader, test_loader, test_labels, val_dataset, test_dataset


# -------------------------- 4. 改进的AE模型 --------------------------
class ImprovedAE(nn.Module):
    def __init__(self):
        super(ImprovedAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),  # 64×112×112
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 128×56×56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 256×28×28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),  # 512×14×14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 512×7×7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(512 * 7 * 7, LATENT_DIM),
            nn.BatchNorm1d(LATENT_DIM),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 512 * 7 * 7),
            nn.BatchNorm1d(512 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (512, 7, 7)),

            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# -------------------------- 5. 简化的损失函数（避免设备不匹配） --------------------------
class MultiScaleLoss(nn.Module):
    """多尺度损失：在不同分辨率下计算重建误差"""

    def __init__(self):
        super(MultiScaleLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, recon_x, x):
        total_loss = 0
        scales = [1, 2, 4]

        for scale in scales:
            if scale > 1:
                # 下采样
                recon_down = nn.functional.avg_pool2d(recon_x, scale)
                x_down = nn.functional.avg_pool2d(x, scale)
            else:
                recon_down = recon_x
                x_down = x

            # 组合MSE和L1损失
            scale_loss = 0.7 * self.mse_loss(recon_down, x_down) + 0.3 * self.l1_loss(recon_down, x_down)
            total_loss += scale_loss

        return total_loss / len(scales)


class EdgeAwareLoss(nn.Module):
    """修复的设备兼容边缘感知损失"""

    def __init__(self, device):
        super(EdgeAwareLoss, self).__init__()
        self.device = device
        self.mse_loss = nn.MSELoss()

        # 创建Sobel核并注册为buffer（这样会自动移动到设备）
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # 注册为buffer，这样它们会随模型一起移动
        self.register_buffer('sobel_x', sobel_x_kernel)
        self.register_buffer('sobel_y', sobel_y_kernel)

    def forward(self, recon_x, x):
        # 转换为灰度图计算边缘
        recon_gray = 0.2989 * recon_x[:, 0, :, :] + 0.5870 * recon_x[:, 1, :, :] + 0.1140 * recon_x[:, 2, :, :]
        x_gray = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]

        recon_gray = recon_gray.unsqueeze(1)
        x_gray = x_gray.unsqueeze(1)

        # 计算边缘 - 使用卷积操作
        recon_edge_x = nn.functional.conv2d(recon_gray, self.sobel_x, padding=1)
        recon_edge_y = nn.functional.conv2d(recon_gray, self.sobel_y, padding=1)
        recon_edge = torch.sqrt(recon_edge_x ** 2 + recon_edge_y ** 2 + 1e-8)

        x_edge_x = nn.functional.conv2d(x_gray, self.sobel_x, padding=1)
        x_edge_y = nn.functional.conv2d(x_gray, self.sobel_y, padding=1)
        x_edge = torch.sqrt(x_edge_x ** 2 + x_edge_y ** 2 + 1e-8)

        # 边缘损失
        edge_loss = self.mse_loss(recon_edge, x_edge)

        return edge_loss


def improved_ae_loss(recon_x, x, multi_scale_loss_fn, edge_loss_fn):
    """改进的AE损失函数组合"""
    # 1. 多尺度重建损失（主要损失）
    multi_scale_loss = multi_scale_loss_fn(recon_x, x)

    # 2. 边缘感知损失（细节损失）
    edge_loss = edge_loss_fn(recon_x, x)

    # 3. 像素级L1损失（保证整体结构）
    pixel_l1_loss = nn.L1Loss()(recon_x, x)

    # 组合损失
    total_loss = multi_scale_loss * 0.6 + edge_loss * 0.3 + pixel_l1_loss * 0.1

    return total_loss


# -------------------------- 6. 训练函数 --------------------------
def train_improved_ae(model, train_loader, val_loader, device):
    model.to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 初始化损失函数 - 确保边缘损失在正确设备上
    multi_scale_loss_fn = MultiScaleLoss()
    edge_loss_fn = EdgeAwareLoss(device).to(device)

    # 早停设置
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{EPOCHS}] Training')

        for batch_idx, (data, _, _) in enumerate(train_bar):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, _ = model(data)
            loss = improved_ae_loss(recon_batch, data, multi_scale_loss_fn, edge_loss_fn)

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()
            train_loss += loss.item()

            train_bar.set_postfix(loss=loss.item())

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _, _ in val_loader:
                data = data.to(device)
                recon_batch, _ = model(data)
                loss = improved_ae_loss(recon_batch, data, multi_scale_loss_fn, edge_loss_fn)
                val_loss += loss.item()

        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step()

        print(f'Epoch {epoch + 1}/{EPOCHS}:')
        print(f'  训练损失: {train_loss:.6f}')
        print(f'  验证损失: {val_loss:.6f}')
        print(f'  学习率: {scheduler.get_last_lr()[0]:.8f}')

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_ae_model.pth')
            print(f'  保存最佳模型，验证损失: {val_loss:.6f}')
        else:
            patience_counter += 1
            print(f'  早停计数: {patience_counter}/{PATIENCE}')

        if patience_counter >= PATIENCE:
            print(f'  早停触发，训练结束')
            break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_ae_model.pth'))
    return model, train_losses, val_losses


# -------------------------- 7. 分类评估函数 --------------------------
def evaluate_classification(model, test_loader, device, test_labels):
    """使用重构误差进行分类并评估性能"""
    model.eval()
    all_recon_errors = []
    all_true_labels = []

    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = data.to(device)
            recon_batch, _ = model(data)

            # 计算每个样本的重构误差（MSE）
            mse_loss = nn.MSELoss(reduction='none')
            batch_errors = mse_loss(recon_batch, data).mean(dim=[1, 2, 3])

            all_recon_errors.extend(batch_errors.cpu().numpy())
            all_true_labels.extend(labels.numpy())

    # 将重构误差转换为分类结果
    # 假设：健康叶片的重构误差较低，病叶的重构误差较高
    # 使用中位数作为阈值
    threshold = np.median(all_recon_errors)
    predicted_labels = [1 if error > threshold else 0 for error in all_recon_errors]

    # 计算分类指标
    accuracy = accuracy_score(all_true_labels, predicted_labels)
    precision = precision_score(all_true_labels, predicted_labels, average='binary')
    recall = recall_score(all_true_labels, predicted_labels, average='binary')
    f1 = f1_score(all_true_labels, predicted_labels, average='binary')

    # 计算混淆矩阵
    cm = confusion_matrix(all_true_labels, predicted_labels)

    # 打印结果
    print("\n" + "=" * 50)
    print("分类性能评估")
    print("=" * 50)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"分类阈值: {threshold:.6f}")
    print("\n混淆矩阵:")
    print(cm)
    print("\n分类报告:")
    print(classification_report(all_true_labels, predicted_labels, target_names=['健康', '病叶']))

    # 可视化重构误差分布
    plt.figure(figsize=(10, 6))
    healthy_errors = [all_recon_errors[i] for i in range(len(all_recon_errors)) if all_true_labels[i] == 0]
    diseased_errors = [all_recon_errors[i] for i in range(len(all_recon_errors)) if all_true_labels[i] == 1]

    plt.hist(healthy_errors, bins=50, alpha=0.7, label='健康叶片', color='green')
    plt.hist(diseased_errors, bins=50, alpha=0.7, label='病叶', color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f'分类阈值: {threshold:.4f}')
    plt.xlabel('重构误差')
    plt.ylabel('频数')
    plt.title('健康叶片与病叶的重构误差分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('reconstruction_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    return accuracy, precision, recall, f1, cm, all_recon_errors, all_true_labels


# -------------------------- 8. 可视化重构结果 --------------------------
def visualize_improved_reconstruction(model, test_loader, device, num_samples=8):
    model.eval()
    with torch.no_grad():
        # 获取一批测试数据
        data_iter = iter(test_loader)
        images, labels, paths = next(data_iter)

        images = images[:num_samples].to(device)
        recon_images, _ = model(images)

        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        images_denorm = images * std + mean
        recon_denorm = recon_images * std + mean

        # 计算PSNR（近似）
        mse_values = []
        for i in range(num_samples):
            mse = torch.mean((images_denorm[i] - recon_denorm[i]) ** 2)
            mse_values.append(mse.item())

        # 可视化
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))

        for i in range(num_samples):
            # 原始图像
            orig_img = images_denorm[i].cpu().permute(1, 2, 0).numpy()
            orig_img = np.clip(orig_img, 0, 1)

            # 重构图像
            recon_img = recon_denorm[i].cpu().permute(1, 2, 0).numpy()
            recon_img = np.clip(recon_img, 0, 1)

            # 计算PSNR（近似）
            mse = mse_values[i]
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100

            # 显示图像
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original\n{paths[i].split("/")[-1][:15]}...')
            axes[0, i].axis('off')

            axes[1, i].imshow(recon_img)
            axes[1, i].set_title(f'Reconstructed\nPSNR: {psnr:.2f}dB')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig('improved_ae_reconstruction.png', dpi=300, bbox_inches='tight')
        plt.show()


# -------------------------- 主函数 --------------------------
def main():
    # 检查环境
    device = check_environment()

    # 检查数据集
    dataset_dir = check_dataset()

    # 获取数据加载器
    train_loader, val_loader, test_loader, test_labels, val_dataset, test_dataset = get_dataloaders(dataset_dir)

    # 初始化改进的AE模型
    model = ImprovedAE()
    print(f"改进的AE模型初始化完成")
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("开始训练改进的AE模型...")
    trained_model, train_losses, val_losses = train_improved_ae(model, train_loader, val_loader, device)

    # 可视化训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Improved AE Model Training Process')
    plt.legend()
    plt.grid(True)
    plt.savefig('improved_ae_training.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 可视化重构结果
    print("可视化重构结果...")
    visualize_improved_reconstruction(trained_model, test_loader, device)

    # 评估分类性能
    print("评估分类性能...")
    accuracy, precision, recall, f1, cm, errors, true_labels = evaluate_classification(
        trained_model, test_loader, device, test_labels
    )

    # 保存评估结果
    with open('classification_results.txt', 'w') as f:
        f.write("AE模型分类性能评估结果\n")
        f.write("=" * 40 + "\n")
        f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
        f.write(f"精确率 (Precision): {precision:.4f}\n")
        f.write(f"召回率 (Recall): {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"混淆矩阵:\n{cm}\n")


if __name__ == "__main__":
    main()