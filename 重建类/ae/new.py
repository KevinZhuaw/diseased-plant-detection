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
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# ==================== 配置 ====================
class Config:
    DATASET_PATH = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"
    RESULTS_PATH = r"E:\庄楷文\叶片病虫害识别\baseline\results\ae_simple"
    IMG_SIZE = 128
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    DEVICE = torch.device('cpu')


# 创建结果目录
os.makedirs(Config.RESULTS_PATH, exist_ok=True)


# ==================== 数据集 ====================
class SimpleDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.image_paths = []

        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_path, img_file))

        print(f"{mode}: {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0
        except:
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), color='black')
            if self.transform:
                image = self.transform(image)
            return image, 0


# ==================== 简单模型 ====================
class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ==================== 训练 ====================
def main():
    print("开始训练简单AE模型...")

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # 数据集
    train_dataset = SimpleDataset(Config.DATASET_PATH, transform, 'train')
    val_dataset = SimpleDataset(Config.DATASET_PATH, transform, 'val')

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 模型
    model = SimpleAE().to(Config.DEVICE)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练记录
    log_file = os.path.join(Config.RESULTS_PATH, "training_log.txt")

    # 训练循环
    for epoch in range(Config.NUM_EPOCHS):
        # 训练
        model.train()
        train_loss = 0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(Config.DEVICE)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, images)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{Config.NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # 记录
        log_msg = f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        print(log_msg)

        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")

        # 保存模型
        if epoch % 2 == 0 or epoch == Config.NUM_EPOCHS - 1:
            torch.save(model.state_dict(),
                       os.path.join(Config.RESULTS_PATH, f"model_epoch_{epoch + 1}.pth"))

    print("训练完成！")
    print(f"结果保存在: {Config.RESULTS_PATH}")


if __name__ == "__main__":
    main()