# ============================================================================
# DINOv2 自监督学习 - 腰果叶片病虫害识别
# 版本：A - 原生518x518训练
# GPU: RTX 4070Ti 12GB (显存占用10.5GB)
# ============================================================================

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

# -------------------- Windows多进程修复 --------------------
import torch.multiprocessing as mp

if os.name == 'nt':
    try:
        mp.set_start_method('spawn', force=True)
        print("✅ Windows: 多进程模式已设置为 'spawn'")
    except RuntimeError:
        pass


# ============================================================================
# 第一部分：配置参数（关键修复！）
# ============================================================================

class Config:
    """统一配置参数 - 518x518原生训练版"""
    # ---------- 数据路径 ----------
    data_root = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"
    train_images_dir = os.path.join(data_root, "train")
    train_labeled_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    # ---------- 自监督训练参数（518原生尺寸）----------
    image_size = 518  # DINOv2原生训练尺寸，必须518！
    embed_dim = 768

    # 显存优化配置（4070Ti 12GB 安全值）
    ssl_batch_size = 4  # 518尺寸下，4是安全值
    ssl_epochs = 50
    ssl_lr = 1e-4
    ssl_warmup_epochs = 10
    ssl_num_workers = 0  # Windows必须为0
    n_local_crops = 6  # 从8降到6，减少显存压力
    out_dim = 65536

    # ---------- 下游任务参数（224推理，保持高效）----------
    downstream_batch_size = 64
    downstream_epochs = 50
    downstream_lr = 0.01
    downstream_num_workers = 0
    downstream_image_size = 224  # 推理用224，速度快10倍

    # ---------- 输出路径 ----------
    output_dir = "./ssl_leaf_disease_output"
    os.makedirs(output_dir, exist_ok=True)
    ssl_weights_path = os.path.join(output_dir, "dinov2_ssl_leaf_final.pth")
    linear_weights_path = os.path.join(output_dir, "linear_probe_best.pth")
    results_path = os.path.join(output_dir, "test_results.txt")


config = Config()


# ============================================================================
# 第二部分：DINOv2模型加载（518x518版本）
# ============================================================================

def build_dinov2_backbone():
    """加载DINOv2 ViT-B/14 - 518x518原生版本"""
    print("🔄 正在加载DINOv2模型（518x518）...")

    try:
        from timm import create_model
        # 创建518x518输入的DINOv2模型
        backbone = create_model(
            'vit_base_patch14_dinov2',
            pretrained=False,
            num_classes=0,
            img_size=518  # 原生518尺寸！
        )
        print("✅ 使用timm DINOv2架构，输入尺寸518x518")
        return backbone, 768
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        raise


# ============================================================================
# 第三部分：MultiCropWrapper（修复维度问题）
# ============================================================================

class MultiCropWrapper(nn.Module):
    """修复版：正确处理518x518输入的batch维度"""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        batch_features = []
        for img in x:
            if img.dim() == 3:
                img = img.unsqueeze(0)  # [1, C, H, W]
            features = self.backbone(img)  # [1, 768]
            batch_features.append(features)

        return self.head(torch.cat(batch_features, dim=0))


# ============================================================================
# 第四部分：DINOv2核心组件
# ============================================================================

class DINOLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp=0.04, teacher_temp=0.07,
                 student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp, config.ssl_warmup_epochs),
            np.ones(config.ssl_epochs - config.ssl_warmup_epochs) * teacher_temp
        ])

    def forward(self, student_output, teacher_output, epoch):
        teacher_temp = self.teacher_temp_schedule[epoch]

        student_out = student_output / self.student_temp
        student_out = student_out.chunk(len(student_output) // len(teacher_output))

        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()

        total_loss = 0
        n_loss_terms = 0
        for i, q in enumerate(student_out):
            for v in range(len(teacher_out)):
                if i == v:
                    continue
                loss = torch.sum(-q * F.log_softmax(teacher_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


# ============================================================================
# 第五部分：自监督数据集（518x518版本）
# ============================================================================

class DINOV2Transform:
    """DINOv2数据增强 - 518x518版本，6个局部裁剪"""

    def __init__(self, global_crops_scale=(0.32, 1.0), local_crops_scale=(0.05, 0.32),
                 local_crops_number=6, image_size=518):
        self.local_crops_number = local_crops_number

        # 全局裁剪 - 518x518
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(23)], p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 局部裁剪 - 也是518x518
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=local_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(13)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform(image))
        crops.append(self.global_transform(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


class SSLImageDataset(Dataset):
    """自监督数据集 - 518x518版本"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))

        print(f"📸 自监督数据集：{len(self.image_paths)} 张图像（518x518训练）")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                crops = self.transform(image)
                return crops
            return image
        except Exception as e:
            print(f"⚠️ 读取失败: {os.path.basename(img_path)}")
            dummy = torch.randn(3, config.image_size, config.image_size)
            return [dummy, dummy] + [dummy.clone() for _ in range(config.n_local_crops)]


# ============================================================================
# 第六部分：下游任务数据集（224x224推理）
# ============================================================================

class LeafDiseaseDataset(Dataset):
    """下游任务数据集 - 224x224推理"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        health_keywords = ['healthy', 'Health', 'HEALTH']

        for cls_name in sorted(classes):
            cls_path = os.path.join(root_dir, cls_name)
            if cls_name not in self.class_to_idx:
                self.class_to_idx[cls_name] = len(self.class_to_idx)

            is_healthy = 1 if any(kw in cls_name for kw in health_keywords) else 0

            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(cls_path, img_name),
                        is_healthy,
                        self.class_to_idx[cls_name]
                    ))

        print(f"📊 {os.path.basename(root_dir)}: {len(self.samples)}张, {len(self.class_to_idx)}类（224推理）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, binary_label, fine_label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, binary_label, fine_label


class DINOv2LinearProbe(nn.Module):
    """冻结DINOv2 + 双分支线性分类头（224推理）"""

    def __init__(self, backbone, num_classes_2, num_classes_22):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.binary_head = nn.Linear(config.embed_dim, num_classes_2)
        self.fine_head = nn.Linear(config.embed_dim, num_classes_22)

        nn.init.normal_(self.binary_head.weight, std=0.01)
        nn.init.normal_(self.fine_head.weight, std=0.01)
        nn.init.constant_(self.binary_head.bias, 0)
        nn.init.constant_(self.fine_head.bias, 0)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.binary_head(features), self.fine_head(features)


# ============================================================================
# 第七部分：自监督训练（518x518）
# ============================================================================

def train_ssl():
    print("\n" + "=" * 60)
    print("🔬 阶段一：DINOv2自监督预训练（518x518原生尺寸）")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 数据增强 - 518版本
    transform_ssl = DINOV2Transform(
        local_crops_number=config.n_local_crops,
        image_size=config.image_size
    )

    # 数据集
    dataset = SSLImageDataset(config.train_images_dir, transform=transform_ssl)

    def collate_crops(batch):
        all_crops = []
        for sample in batch:
            all_crops.extend(sample)
        return all_crops

    loader = DataLoader(
        dataset,
        batch_size=config.ssl_batch_size,
        shuffle=True,
        num_workers=config.ssl_num_workers,
        collate_fn=collate_crops,
        drop_last=True
    )

    # 构建模型
    backbone, embed_dim = build_dinov2_backbone()
    student = MultiCropWrapper(
        backbone,
        DINOHead(embed_dim, config.out_dim)
    ).to(device)

    teacher_backbone, _ = build_dinov2_backbone()
    teacher = MultiCropWrapper(
        teacher_backbone,
        DINOHead(embed_dim, config.out_dim)
    ).to(device)

    # 初始化教师
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.load_state_dict(student.state_dict())

    # 损失函数
    dino_loss = DINOLoss(config.out_dim).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.ssl_lr,
        weight_decay=0.04,
        betas=(0.9, 0.999)
    )

    scaler = GradScaler()

    print(f"\n🚀 开始518x518自监督训练...")
    print(f"   数据集: {len(dataset)} 张图像")
    print(f"   Batch size: {config.ssl_batch_size}")
    print(f"   局部裁剪数: {config.n_local_crops}")
    print(f"   每epoch: {len(loader)} 步")
    print(f"   总epochs: {config.ssl_epochs}")
    print(f"   显存预估: 10.5GB\n")

    global_step = 0
    for epoch in range(config.ssl_epochs):
        student.train()
        teacher.train()

        pbar = tqdm(loader, desc=f"SSL Epoch {epoch + 1}/{config.ssl_epochs}")
        epoch_loss = 0

        for crops in pbar:
            crops = [c.to(device) for c in crops]

            with autocast():
                teacher_output = teacher(crops[:2])
                student_output = student(crops)
                loss = dino_loss(student_output, teacher_output, epoch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                m = 0.996
                for p, p_t in zip(student.parameters(), teacher.parameters()):
                    p_t.data.mul_(m).add_((1 - m) * p.data)

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mem': f'{torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB'
            })

        avg_loss = epoch_loss / len(loader)
        print(f"   Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(teacher.state_dict(),
                       os.path.join(config.output_dir, f'dinov2_ssl_epoch{epoch + 1}.pth'))

    torch.save(teacher.state_dict(), config.ssl_weights_path)
    print(f"\n✅ 自监督训练完成！")
    return teacher


# ============================================================================
# 第八部分：下游任务评估（224x224推理）
# ============================================================================

@torch.no_grad()
def evaluate(model, dataloader, device, phase="val"):
    model.eval()
    all_binary_pred, all_binary_label = [], []
    all_fine_pred, all_fine_label = [], []

    for images, labels_binary, labels_fine in tqdm(dataloader, desc=f"Evaluating {phase}"):
        images = images.to(device)
        labels_binary = labels_binary.to(device)

        out_binary, out_fine = model(images)

        pred_binary = torch.argmax(out_binary, dim=1)
        pred_fine = torch.argmax(out_fine, dim=1)

        all_binary_pred.extend(pred_binary.cpu().numpy())
        all_binary_label.extend(labels_binary.cpu().numpy())
        all_fine_pred.extend(pred_fine.cpu().numpy())
        all_fine_label.extend(labels_fine.cpu().numpy())

    acc_b = accuracy_score(all_binary_label, all_binary_pred)
    prec_b = precision_score(all_binary_label, all_binary_pred, average='binary', zero_division=0)
    rec_b = recall_score(all_binary_label, all_binary_pred, average='binary', zero_division=0)
    f1_b = f1_score(all_binary_label, all_binary_pred, average='binary', zero_division=0)

    acc_f = accuracy_score(all_fine_label, all_fine_pred)
    prec_f = precision_score(all_fine_label, all_fine_pred, average='macro', zero_division=0)
    rec_f = recall_score(all_fine_label, all_fine_pred, average='macro', zero_division=0)
    f1_f = f1_score(all_fine_label, all_fine_pred, average='macro', zero_division=0)

    return {
        'binary': {'acc': acc_b, 'prec': prec_b, 'rec': rec_b, 'f1': f1_b},
        'fine': {'acc': acc_f, 'prec': prec_f, 'rec': rec_f, 'f1': f1_f}
    }


def train_linear_probe(ssl_backbone):
    print("\n" + "=" * 60)
    print("🎯 阶段二：下游任务线性探测（224x224快速推理）")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 下游任务用224x224
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = LeafDiseaseDataset(config.train_labeled_dir, transform=transform_train)
    val_dataset = LeafDiseaseDataset(config.val_dir, transform=transform_eval)
    test_dataset = LeafDiseaseDataset(config.test_dir, transform=transform_eval)

    train_loader = DataLoader(train_dataset, batch_size=config.downstream_batch_size,
                              shuffle=True, num_workers=config.downstream_num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.downstream_batch_size,
                            shuffle=False, num_workers=config.downstream_num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.downstream_batch_size,
                             shuffle=False, num_workers=config.downstream_num_workers)

    model = DINOv2LinearProbe(
        backbone=ssl_backbone,
        num_classes_2=2,
        num_classes_22=len(train_dataset.class_to_idx)
    ).to(device)

    criterion_binary = nn.CrossEntropyLoss()
    criterion_fine = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        list(model.binary_head.parameters()) + list(model.fine_head.parameters()),
        lr=config.downstream_lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.downstream_epochs)

    best_f1 = 0
    for epoch in range(config.downstream_epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Linear Epoch {epoch + 1}/{config.downstream_epochs}")
        for images, labels_binary, labels_fine in pbar:
            images = images.to(device)
            labels_binary = labels_binary.to(device)
            labels_fine = labels_fine.to(device)

            optimizer.zero_grad()

            out_binary, out_fine = model(images)
            loss = criterion_binary(out_binary, labels_binary) + criterion_fine(out_fine, labels_fine)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, "val")

        if val_metrics['fine']['f1'] > best_f1:
            best_f1 = val_metrics['fine']['f1']
            torch.save(model.state_dict(), config.linear_weights_path)
            print(f"   ✅ 新最佳模型! F1: {best_f1:.4f}")

    model.load_state_dict(torch.load(config.linear_weights_path))
    test_metrics = evaluate(model, test_loader, device, "TEST")

    return test_metrics, train_dataset.class_to_idx


# ============================================================================
# 第九部分：主函数
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("🌿 DINOv2 自监督学习 - 腰果叶片病虫害识别")
    print("   版本: 518x518原生训练 + 224x224快速推理")
    print("=" * 60)
    print(f"数据目录: {config.data_root}")
    print(f"训练模式: 518x518, batch_size={config.ssl_batch_size}, local_crops={config.n_local_crops}")
    print(f"显存目标: 10.5GB/12GB")
    print("=" * 60 + "\n")

    # 阶段一：自监督训练
    ssl_model = train_ssl()

    # 提取backbone
    backbone = ssl_model.backbone if hasattr(ssl_model, 'backbone') else ssl_model

    # 阶段二：下游任务
    test_metrics, class_to_idx = train_linear_probe(backbone)

    # 输出结果
    print("\n" + "=" * 60)
    print("✅ FINAL TEST RESULTS")
    print("=" * 60)
    print("\n=== Binary Classification (Healthy vs Diseased) ===")
    print(f"Accuracy:  {test_metrics['binary']['acc']:.4f}")
    print(f"Precision: {test_metrics['binary']['prec']:.4f}")
    print(f"Recall:    {test_metrics['binary']['rec']:.4f}")
    print(f"F1-Score:  {test_metrics['binary']['f1']:.4f}")

    print("\n=== Fine-grained Classification (22 Classes) ===")
    print(f"Accuracy:  {test_metrics['fine']['acc']:.4f}")
    print(f"Precision (Macro): {test_metrics['fine']['prec']:.4f}")
    print(f"Recall (Macro):    {test_metrics['fine']['rec']:.4f}")
    print(f"F1-Score (Macro):  {test_metrics['fine']['f1']:.4f}")

    # 保存结果
    with open(config.results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DINOv2 SELF-SUPERVISED LEARNING - CASHEW LEAF DISEASE\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training: 518x518, batch_size={config.ssl_batch_size}, epochs={config.ssl_epochs}\n")
        f.write(f"Inference: 224x224\n\n")
        f.write("=== Binary Classification ===\n")
        f.write(f"Accuracy:  {test_metrics['binary']['acc']:.4f}\n")
        f.write(f"Precision: {test_metrics['binary']['prec']:.4f}\n")
        f.write(f"Recall:    {test_metrics['binary']['rec']:.4f}\n")
        f.write(f"F1-Score:  {test_metrics['binary']['f1']:.4f}\n\n")
        f.write("=== Fine-grained Classification (22 Classes) ===\n")
        f.write(f"Accuracy:  {test_metrics['fine']['acc']:.4f}\n")
        f.write(f"Precision (Macro): {test_metrics['fine']['prec']:.4f}\n")
        f.write(f"Recall (Macro):    {test_metrics['fine']['rec']:.4f}\n")
        f.write(f"F1-Score (Macro):  {test_metrics['fine']['f1']:.4f}\n")

    print(f"\n📄 结果已保存: {config.results_path}")
    print("\n" + "=" * 60)
    print("🎉 全流程完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()