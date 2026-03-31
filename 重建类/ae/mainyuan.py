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


# 设置随机种子确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ==================== 自定义数据集类 ====================
class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 收集所有图像路径
        categories = sorted([d for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))])

        for category_idx, category in enumerate(categories):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                image_files = [f for f in os.listdir(category_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

                if max_samples:
                    image_files = image_files[:min(len(image_files), max_samples)]

                for img_file in image_files:
                    img_path = os.path.join(category_path, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(category_idx)

        print(f"Loaded {len(self.image_paths)} images from {len(categories)} categories")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 返回一个黑色图像作为占位符
            image = Image.new('RGB', (128, 128), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label


# ==================== 自编码器模型 ====================
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 16 x 64 x 64
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32 x 32 x 32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64 x 16 x 16
            nn.ReLU(True)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ==================== 训练函数 ====================
def train_autoencoder_simple(model, train_loader, val_loader, num_epochs=20, result_dir='./ae_simple_results'):
    """简单的训练函数"""

    os.makedirs(result_dir, exist_ok=True)

    # 使用MSE损失和Adam优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 结果记录
    results = {
        'train_losses': [],
        'val_losses': [],
        'best_val_loss': float('inf'),
        'training_time': 0
    }

    start_time = time.time()

    # 创建结果文件
    result_file = os.path.join(result_dir, 'training_results.txt')
    with open(result_file, 'w') as f:
        f.write("AutoEncoder Training Results\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Simple AutoEncoder\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Train samples: {len(train_loader.dataset)}\n")
        f.write(f"Validation samples: {len(val_loader.dataset)}\n")
        f.write("=" * 60 + "\n\n")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            # 前向传播
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        results['train_losses'].append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        results['val_losses'].append(avg_val_loss)

        # 更新最佳损失
        if avg_val_loss < results['best_val_loss']:
            results['best_val_loss'] = avg_val_loss
            torch.save(model.state_dict(), os.path.join(result_dir, 'best_model.pth'))

        # 记录到文件
        with open(result_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs}\n")
            f.write(f"  Train Loss: {avg_train_loss:.6f}\n")
            f.write(f"  Val Loss: {avg_val_loss:.6f}\n")
            f.write(f"  Best Val Loss: {results['best_val_loss']:.6f}\n\n")

        # 打印进度
        print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

    results['training_time'] = time.time() - start_time

    # 记录最终结果
    with open(result_file, 'a') as f:
        f.write("=" * 60 + "\n")
        f.write("Training Summary\n")
        f.write(f"Total training time: {results['training_time']:.2f} seconds\n")
        f.write(f"Best validation loss: {results['best_val_loss']:.6f}\n")
        f.write(f"Final training loss: {results['train_losses'][-1]:.6f}\n")
        f.write(f"Final validation loss: {results['val_losses'][-1]:.6f}\n")

    return results


# ==================== 测试函数 ====================
def test_autoencoder(model, test_loader, result_dir):
    """在测试集上评估模型"""

    model.eval()
    criterion = nn.MSELoss()

    test_loss = 0.0
    test_results = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            test_loss += loss.item()

            # 保存一些样本的重建损失
            if i < 5:  # 只保存前5个batch的样本
                for j in range(min(3, images.size(0))):  # 每个batch保存3个样本
                    sample_loss = criterion(reconstructed[j:j + 1], images[j:j + 1]).item()
                    test_results.append({
                        'batch': i,
                        'sample': j,
                        'label': labels[j].item(),
                        'loss': sample_loss
                    })

    avg_test_loss = test_loss / len(test_loader)

    # 保存测试结果
    test_file = os.path.join(result_dir, 'test_results.txt')
    with open(test_file, 'w') as f:
        f.write("AutoEncoder Test Results\n")
        f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test samples: {len(test_loader.dataset)}\n")
        f.write(f"Average test loss: {avg_test_loss:.6f}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Sample Results (first few samples):\n")
        f.write("Batch | Sample | Label | Loss\n")
        f.write("-" * 40 + "\n")

        for result in test_results:
            f.write(f"{result['batch']:5d} | {result['sample']:6d} | {result['label']:5d} | {result['loss']:.6f}\n")

    return avg_test_loss


# ==================== 主函数 ====================
def main():
    # 数据集路径
    dataset_path = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"
    result_dir = r"E:\庄楷文\叶片病虫害识别\baseline\results\ae"

    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # 创建数据集
    print("Loading datasets...")

    # 为了快速测试，可以限制样本数量
    max_samples = 1000  # 每个数据集最多1000个样本，如果需要完整数据，设为None

    train_dataset = PlantDiseaseDataset(
        os.path.join(dataset_path, 'train'),
        transform=transform,
        max_samples=max_samples
    )

    val_dataset = PlantDiseaseDataset(
        os.path.join(dataset_path, 'val'),
        transform=transform,
        max_samples=max_samples // 5  # 验证集样本更少
    )

    test_dataset = PlantDiseaseDataset(
        os.path.join(dataset_path, 'test'),
        transform=transform,
        max_samples=max_samples // 2  # 测试集样本适中
    )

    # 创建数据加载器
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    # 创建模型
    print("\nCreating AutoEncoder model...")
    model = SimpleAutoencoder().to(device)

    # 训练模型
    print("\nStarting training...")
    training_results = train_autoencoder_simple(
        model,
        train_loader,
        val_loader,
        num_epochs=20,
        result_dir=result_dir
    )

    # 加载最佳模型
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(os.path.join(result_dir, 'best_model.pth')))

    # 在测试集上评估
    print("Testing on test set...")
    test_loss = test_autoencoder(model, test_loader, result_dir)

    # 创建汇总文件
    summary_file = os.path.join(result_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("AutoEncoder Experiment Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. Dataset Information\n")
        f.write(f"   Train samples: {len(train_dataset)}\n")
        f.write(f"   Validation samples: {len(val_dataset)}\n")
        f.write(f"   Test samples: {len(test_dataset)}\n")
        f.write(f"   Batch size: {batch_size}\n")
        f.write(f"   Max samples per set: {max_samples}\n\n")

        f.write("2. Model Architecture\n")
        f.write("   Encoder: 3 -> 16 -> 32 -> 64 channels\n")
        f.write("   Decoder: 64 -> 32 -> 16 -> 3 channels\n")
        f.write("   Activation: ReLU (encoder), Sigmoid (final decoder layer)\n\n")

        f.write("3. Training Parameters\n")
        f.write(f"   Epochs: 20\n")
        f.write(f"   Loss function: MSE\n")
        f.write(f"   Optimizer: Adam (lr=0.001)\n")
        f.write(f"   Device: {device}\n\n")

        f.write("4. Results\n")
        f.write(f"   Best validation loss: {training_results['best_val_loss']:.6f}\n")
        f.write(f"   Test loss: {test_loss:.6f}\n")
        f.write(f"   Final training loss: {training_results['train_losses'][-1]:.6f}\n")
        f.write(f"   Final validation loss: {training_results['val_losses'][-1]:.6f}\n")
        f.write(f"   Total training time: {training_results['training_time']:.2f} seconds\n\n")

        f.write("5. Files Generated\n")
        f.write(f"   {result_dir}/training_results.txt - Detailed training log\n")
        f.write(f"   {result_dir}/test_results.txt - Test results\n")
        f.write(f"   {result_dir}/best_model.pth - Best model weights\n")
        f.write(f"   {result_dir}/summary.txt - This summary file\n")

    print("\n" + "=" * 60)
    print("Experiment completed!")
    print(f"Results saved to: {result_dir}")
    print(f"Best validation loss: {training_results['best_val_loss']:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()