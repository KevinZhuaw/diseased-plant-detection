# ============================================================================
# DINOv2 自监督训练 - 12GB显存最终版（不用梯度检查点）
# ============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import gc

# -------------------- 环境设置 --------------------
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# ============================================================================
# 配置参数（为12GB显存精确优化）
# ============================================================================

class Config:
    # 数据路径
    train_dir = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new\train"
    weights_path = r"E:\庄楷文\叶片病虫害识别\baseline\自监督类\DINO\dinov2_vitb14_pretrain.pth"

    # ---------- 12GB显存最终配置 ----------
    batch_size = 1  # 必须为1
    n_local_crops = 1  # 从2降到1！大幅减少显存
    grad_accum = 8  # 增加梯度累积步数
    epochs = 50
    lr = 1e-5  # 更低的学习率

    # 禁用梯度检查点
    use_checkpointing = False

    # 输出路径
    output_dir = "./dinov2_12gb_final"
    os.makedirs(output_dir, exist_ok=True)


config = Config()


# ============================================================================
# DINOv2模型（简化版）
# ============================================================================

class DINOV2Model(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        if isinstance(x, list):
            features = []
            for img in x:
                if img.dim() == 3:
                    img = img.unsqueeze(0)

                with torch.cuda.amp.autocast():
                    feat = self.backbone(img)

                # 立即移回CPU，只保留head在GPU
                features.append(feat.cpu())

            # 只在head计算时移回GPU
            x = torch.cat([f.cuda() for f in features], dim=0)

        return self.head(x)


def create_dinov2_backbone():
    """创建DINOv2 backbone并加载权重"""
    from timm import create_model

    backbone = create_model(
        'vit_base_patch14_dinov2',
        pretrained=False,
        num_classes=0,
        img_size=224
    )

    if os.path.exists(config.weights_path):
        state_dict = torch.load(config.weights_path, map_location='cpu')

        if 'student' in state_dict:
            state_dict = state_dict['student']
        elif 'teacher' in state_dict:
            state_dict = state_dict['teacher']

        model_dict = backbone.state_dict()
        state_dict = {k: v for k, v in state_dict.items()
                      if k in model_dict and v.shape == model_dict[k].shape}

        backbone.load_state_dict(state_dict, strict=False)
        print(f"✅ 加载 {len(state_dict)} 个权重")

    return backbone


class DINOHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=65536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 384),
        )
        self.last_layer = nn.Linear(384, out_dim, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


# ============================================================================
# DINO损失函数（内存优化版）
# ============================================================================

class DINOLoss(nn.Module):
    def __init__(self, out_dim=65536):
        super().__init__()
        self.student_temp = 0.1
        self.teacher_temp = 0.07
        self.center_momentum = 0.9
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        B = teacher_output.shape[0] // 2

        student_output = student_output / self.student_temp
        teacher_output = (teacher_output - self.center) / self.teacher_temp
        teacher_output = F.softmax(teacher_output, dim=-1).detach()

        total_loss = 0
        n_pairs = 0

        for i in range(B):
            s_start = i * (2 + config.n_local_crops)
            s_end = (i + 1) * (2 + config.n_local_crops)
            t_start = i * 2
            t_end = (i + 1) * 2

            student_i = student_output[s_start:s_end]
            teacher_i = teacher_output[t_start:t_end]

            for s in student_i:
                s_log = F.log_softmax(s.unsqueeze(0), dim=-1)
                for t in teacher_i:
                    total_loss += torch.sum(-t * s_log)
                    n_pairs += 1

        loss = total_loss / n_pairs
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# ============================================================================
# 数据增强（进一步简化）
# ============================================================================

class DINOV2Transform:
    def __init__(self):
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.32, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.05, 0.32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform(image))
        crops.append(self.global_transform(image))
        for _ in range(config.n_local_crops):
            crops.append(self.local_transform(image))
        return crops


# ============================================================================
# 数据集
# ============================================================================

class SSLImageDataset(Dataset):
    def __init__(self):
        self.image_paths = []
        self.transform = DINOV2Transform()

        print(f"📂 扫描目录: {config.train_dir}")
        for root, _, files in os.walk(config.train_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))

        print(f"✅ 找到 {len(self.image_paths)} 张图片")
        print(f"   每张图生成 {2 + config.n_local_crops} 个crops")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            return self.transform(image)
        except:
            dummy = torch.randn(3, 224, 224)
            return [dummy, dummy] + [dummy.clone() for _ in range(config.n_local_crops)]


# ============================================================================
# 训练函数（终极简化版）
# ============================================================================

def train():
    print("\n" + "=" * 60)
    print("DINOv2 自监督训练 - 12GB显存最终版")
    print("=" * 60)
    print(f"Batch size: {config.batch_size}")
    print(f"局部裁剪: {config.n_local_crops}")
    print(f"梯度累积: {config.grad_accum}")
    print(f"总crops数: {2 + config.n_local_crops}")
    print("=" * 60 + "\n")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    gc.collect()

    # 数据集
    dataset = SSLImageDataset()

    def collate_crops(batch):
        crops = []
        for sample in batch:
            crops.extend(sample)
        return crops

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_crops,
        drop_last=True,
        pin_memory=False
    )

    # 创建模型
    print("\n🔄 创建学生模型...")
    student_backbone = create_dinov2_backbone()
    student_head = DINOHead()
    student = DINOV2Model(student_backbone, student_head).to(device)

    print("🔄 创建教师模型...")
    teacher_backbone = create_dinov2_backbone()
    teacher_head = DINOHead()
    teacher = DINOV2Model(teacher_backbone, teacher_head).to(device)

    # 冻结教师
    for p in teacher.parameters():
        p.requires_grad = False

    # 初始化教师
    with torch.no_grad():
        for p, p_t in zip(student.parameters(), teacher.parameters()):
            p_t.data.copy_(p.data)

    # 损失函数
    criterion = DINOLoss().to(device)

    # 优化器
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr)
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n🚀 开始训练...")
    print(f"   每epoch: {len(loader)} 步")
    print(f"   总epochs: {config.epochs}\n")

    for epoch in range(config.epochs):
        student.train()
        teacher.train()

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        optimizer.zero_grad()

        for idx, crops in enumerate(pbar):
            # 每10步清理显存
            if idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # 分离全局裁剪和所有裁剪
            global_crops = crops[:2 * config.batch_size]
            all_crops = crops

            # 移动到GPU
            global_crops = [c.to(device, non_blocking=True) for c in global_crops]
            all_crops = [c.to(device, non_blocking=True) for c in all_crops]

            with torch.cuda.amp.autocast():
                # 前向传播
                student_output = student(all_crops)
                with torch.no_grad():
                    teacher_output = teacher(global_crops)

                # 计算损失
                loss = criterion(student_output, teacher_output)
                scaled_loss = loss / config.grad_accum

            # 反向传播
            scaler.scale(scaled_loss).backward()

            # 梯度累积更新
            if (idx + 1) % config.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # EMA更新教师
                with torch.no_grad():
                    m = 0.996
                    for p, p_t in zip(student.parameters(), teacher.parameters()):
                        p_t.data.mul_(m).add_((1 - m) * p.data)

                # 更新后清理
                torch.cuda.empty_cache()

            current_loss = loss.item()

            # 清理
            del global_crops, all_crops, student_output, teacher_output, loss, scaled_loss

            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'mem': f'{torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB'
            })

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(config.output_dir, f'dinov2_epoch{epoch + 1}.pth')
            torch.save({
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'epoch': epoch
            }, save_path)
            print(f"\n💾 保存: {save_path}")
            print(f"   显存峰值: {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f}GB")
            torch.cuda.reset_peak_memory_stats()

    # 保存最终模型
    torch.save(teacher.state_dict(), os.path.join(config.output_dir, 'dinov2_final.pth'))
    print(f"\n✅ 训练完成！")


if __name__ == "__main__":
    train()