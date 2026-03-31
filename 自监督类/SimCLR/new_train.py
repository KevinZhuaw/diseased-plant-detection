"""
测试和评估脚本
修复PyTorch 2.6的weights_only加载问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, average_precision_score,
    roc_auc_score
)
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========================== 配置参数 ==========================
class Config:
    data_dir = "E:/庄楷文/叶片病虫害识别/baseline/data/reorganized_dataset_new"
    output_dir = "simclr_output"
    batch_size = 8
    image_size = 128
    num_workers = 0

config = Config()

# ========================== 数据集类 ==========================
class LeafDiseaseDataset(Dataset):
    def __init__(self, root_dir: str, mode: str = 'test'):
        self.root_dir = root_dir
        self.mode = mode

        # 测试集变换
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # 收集样本
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    class_idx = self.class_to_idx[class_name]
                    binary_label = 0 if 'healthy' in class_name.lower() else 1
                    self.samples.append({
                        'path': img_path,
                        'class_idx': class_idx,
                        'binary_label': binary_label
                    })

        print(f"{mode} dataset: {len(self.samples)} samples, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            img = Image.open(sample['path']).convert('RGB')
            img = self.transform(img)
            return img, sample['class_idx'], sample['binary_label']
        except:
            dummy = torch.zeros(3, config.image_size, config.image_size)
            return dummy, sample['class_idx'], sample['binary_label']

# ========================== SimCLR模型 ==========================
class SimCLRModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # 编码器
        self.encoder = models.resnet18(weights=None)  # 不加载预训练权重
        self.encoder.fc = nn.Identity()

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # 分类头
        self.multi_classifier = nn.Linear(512, num_classes)
        self.binary_classifier = nn.Linear(512, 2)

    def forward(self, x):
        features = self.encoder(x)
        multi_logits = self.multi_classifier(features)
        binary_logits = self.binary_classifier(features)
        return multi_logits, binary_logits

# ========================== 修复加载函数 ==========================
def load_model_fixed(checkpoint_path, num_classes, device):
    """修复模型加载问题"""
    print(f"Loading model from {checkpoint_path}")

    # 创建模型
    model = SimCLRModel(num_classes).to(device)

    try:
        # 方法1: 尝试使用weights_only=False
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded model with model_state_dict")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Loaded model directly from state_dict")

        return model

    except Exception as e1:
        print(f"Method 1 failed: {e1}")

        try:
            # 方法2: 如果保存的是完整模型
            model = torch.load(checkpoint_path, map_location=device)
            print("✓ Loaded full model directly")
            return model
        except Exception as e2:
            print(f"Method 2 failed: {e2}")

            # 方法3: 手动处理检查点
            print("Attempting manual checkpoint handling...")
            from collections import OrderedDict

            checkpoint = torch.load(checkpoint_path, map_location=device)

            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # 处理可能的键名前缀
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # 移除可能的"module."前缀
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
            print("✓ Loaded model with manual state_dict processing")

            return model

# ========================== 评估函数 ==========================
def calculate_metrics(multi_preds, binary_preds, multi_labels, binary_labels, multi_probs, binary_probs):
    """计算所有评估指标"""

    # 二分类指标
    binary_accuracy = accuracy_score(binary_labels, binary_preds)
    binary_precision = precision_score(binary_labels, binary_preds, average='weighted', zero_division=0)
    binary_recall = recall_score(binary_labels, binary_preds, average='weighted', zero_division=0)
    binary_f1 = f1_score(binary_labels, binary_preds, average='weighted', zero_division=0)

    # 多分类指标
    multi_accuracy = accuracy_score(multi_labels, multi_preds)
    multi_precision = precision_score(multi_labels, multi_preds, average='weighted', zero_division=0)
    multi_recall = recall_score(multi_labels, multi_preds, average='weighted', zero_division=0)
    multi_f1 = f1_score(multi_labels, multi_preds, average='weighted', zero_division=0)

    # mAP (多类别平均精度)
    try:
        n_classes = np.max(multi_labels) + 1
        y_true_onehot = np.eye(n_classes)[multi_labels]
        multi_ap = average_precision_score(y_true_onehot, multi_probs, average='weighted')
    except:
        multi_ap = 0.0

    # 二分类AUC
    try:
        binary_auc = roc_auc_score(binary_labels, binary_probs[:, 1])
    except:
        binary_auc = 0.0

    # 混淆矩阵
    binary_cm = confusion_matrix(binary_labels, binary_preds)
    multi_cm = confusion_matrix(multi_labels, multi_preds)

    # 每类指标
    multi_report = classification_report(multi_labels, multi_preds, output_dict=True, zero_division=0)

    return {
        # 二分类结果
        'binary_accuracy': binary_accuracy,
        'binary_precision': binary_precision,
        'binary_recall': binary_recall,
        'binary_f1': binary_f1,
        'binary_auc': binary_auc,
        'binary_confusion_matrix': binary_cm,

        # 多分类结果
        'multi_accuracy': multi_accuracy,
        'multi_precision': multi_precision,
        'multi_recall': multi_recall,
        'multi_f1': multi_f1,
        'multi_mAP': multi_ap,
        'multi_confusion_matrix': multi_cm,
        'multi_report': multi_report
    }

def evaluate_model(model, data_loader, device):
    """评估模型"""
    model.eval()

    all_multi_preds = []
    all_binary_preds = []
    all_multi_labels = []
    all_binary_labels = []
    all_multi_probs = []
    all_binary_probs = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Evaluating')
        for images, multi_labels, binary_labels in progress_bar:
            images = images.to(device)

            # 前向传播
            multi_logits, binary_logits = model(images)

            # 获取预测
            multi_probs = F.softmax(multi_logits, dim=1)
            binary_probs = F.softmax(binary_logits, dim=1)

            multi_preds = torch.argmax(multi_logits, dim=1)
            binary_preds = torch.argmax(binary_logits, dim=1)

            # 收集结果
            all_multi_preds.extend(multi_preds.cpu().numpy())
            all_binary_preds.extend(binary_preds.cpu().numpy())
            all_multi_labels.extend(multi_labels.numpy())
            all_binary_labels.extend(binary_labels.numpy())
            all_multi_probs.extend(multi_probs.cpu().numpy())
            all_binary_probs.extend(binary_probs.cpu().numpy())

    # 计算指标
    metrics = calculate_metrics(
        np.array(all_multi_preds),
        np.array(all_binary_preds),
        np.array(all_multi_labels),
        np.array(all_binary_labels),
        np.array(all_multi_probs),
        np.array(all_binary_probs)
    )

    return metrics, {
        'multi_preds': all_multi_preds,
        'binary_preds': all_binary_preds,
        'multi_labels': all_multi_labels,
        'binary_labels': all_binary_labels
    }

# ========================== 可视化函数 ==========================
def plot_confusion_matrices(binary_cm, multi_cm, class_names):
    """绘制混淆矩阵"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 二分类混淆矩阵
    axes[0].imshow(binary_cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('Binary Classification: Healthy vs Disease', fontsize=12)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Healthy', 'Disease'])
    axes[0].set_yticklabels(['Healthy', 'Disease'])

    # 添加数值
    thresh = binary_cm.max() / 2.
    for i in range(binary_cm.shape[0]):
        for j in range(binary_cm.shape[1]):
            axes[0].text(j, i, format(binary_cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if binary_cm[i, j] > thresh else "black")

    # 多分类混淆矩阵（显示前10类）
    max_classes = min(10, multi_cm.shape[0])
    axes[1].imshow(multi_cm[:max_classes, :max_classes], interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].set_title('Multi-class Classification (Top 10 Classes)', fontsize=12)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].set_xticks(range(max_classes))
    axes[1].set_yticks(range(max_classes))
    axes[1].set_xticklabels([class_names[i][:15] for i in range(max_classes)], rotation=45, ha='right')
    axes[1].set_yticklabels([class_names[i][:15] for i in range(max_classes)])

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_distribution(class_names, train_counts, val_counts, test_counts):
    """绘制类别分布图"""
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 8))
    rects1 = ax.bar(x - width, train_counts, width, label='Train', color='skyblue')
    rects2 = ax.bar(x, val_counts, width, label='Validation', color='lightgreen')
    rects3 = ax.bar(x + width, test_counts, width, label='Test', color='salmon')

    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels([name[:20] for name in class_names], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ========================== 主函数 ==========================
def main():
    print("="*80)
    print("MODEL EVALUATION AND TESTING")
    print("="*80)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'results'), exist_ok=True)

    try:
        # 1. 加载数据集
        print("\nLoading datasets...")

        train_dataset = LeafDiseaseDataset(
            os.path.join(config.data_dir, 'train'),
            mode='train'
        )

        val_dataset = LeafDiseaseDataset(
            os.path.join(config.data_dir, 'val'),
            mode='val'
        )

        test_dataset = LeafDiseaseDataset(
            os.path.join(config.data_dir, 'test'),
            mode='test'
        )

        # 获取类别信息
        class_names = train_dataset.classes
        num_classes = len(class_names)

        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")

        # 2. 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=False
        )

        # 3. 加载模型
        print("\nLoading trained model...")

        # 查找最佳模型文件
        checkpoint_dir = os.path.join(config.output_dir, 'checkpoints')
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

        if not checkpoint_files:
            print("No checkpoint files found!")
            return

        # 优先使用best_model.pth
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
        else:
            # 使用最新的检查点
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])

        # 加载模型
        model = load_model_fixed(checkpoint_path, num_classes, device)

        # 4. 评估模型
        print("\n" + "="*60)
        print("Evaluating on Validation Set")
        print("="*60)

        val_metrics, val_predictions = evaluate_model(model, val_loader, device)

        print("\n" + "="*60)
        print("Evaluating on Test Set")
        print("="*60)

        test_metrics, test_predictions = evaluate_model(model, test_loader, device)

        # 5. 打印结果
        print("\n" + "="*80)
        print("VALIDATION SET RESULTS")
        print("="*80)

        print("\nBinary Classification (Healthy vs Disease):")
        print(f"  Accuracy:  {val_metrics['binary_accuracy']:.4f}")
        print(f"  Precision: {val_metrics['binary_precision']:.4f}")
        print(f"  Recall:    {val_metrics['binary_recall']:.4f}")
        print(f"  F1-Score:  {val_metrics['binary_f1']:.4f}")
        print(f"  AUC:       {val_metrics['binary_auc']:.4f}")

        print("\nMulti-class Classification (Specific Diseases):")
        print(f"  Accuracy:  {val_metrics['multi_accuracy']:.4f}")
        print(f"  Precision: {val_metrics['multi_precision']:.4f}")
        print(f"  Recall:    {val_metrics['multi_recall']:.4f}")
        print(f"  F1-Score:  {val_metrics['multi_f1']:.4f}")
        print(f"  mAP:       {val_metrics['multi_mAP']:.4f}")

        print("\n" + "="*80)
        print("TEST SET RESULTS")
        print("="*80)

        print("\nBinary Classification (Healthy vs Disease):")
        print(f"  Accuracy:  {test_metrics['binary_accuracy']:.4f}")
        print(f"  Precision: {test_metrics['binary_precision']:.4f}")
        print(f"  Recall:    {test_metrics['binary_recall']:.4f}")
        print(f"  F1-Score:  {test_metrics['binary_f1']:.4f}")
        print(f"  AUC:       {test_metrics['binary_auc']:.4f}")

        print("\nMulti-class Classification (Specific Diseases):")
        print(f"  Accuracy:  {test_metrics['multi_accuracy']:.4f}")
        print(f"  Precision: {test_metrics['multi_precision']:.4f}")
        print(f"  Recall:    {test_metrics['multi_recall']:.4f}")
        print(f"  F1-Score:  {test_metrics['multi_f1']:.4f}")
        print(f"  mAP:       {test_metrics['multi_mAP']:.4f}")

        # 6. 保存结果
        print("\nSaving results...")

        # 保存指标到CSV
        metrics_data = {
            'Dataset': ['Validation', 'Test'],
            'Binary_Accuracy': [val_metrics['binary_accuracy'], test_metrics['binary_accuracy']],
            'Binary_Precision': [val_metrics['binary_precision'], test_metrics['binary_precision']],
            'Binary_Recall': [val_metrics['binary_recall'], test_metrics['binary_recall']],
            'Binary_F1': [val_metrics['binary_f1'], test_metrics['binary_f1']],
            'Binary_AUC': [val_metrics['binary_auc'], test_metrics['binary_auc']],
            'Multi_Accuracy': [val_metrics['multi_accuracy'], test_metrics['multi_accuracy']],
            'Multi_Precision': [val_metrics['multi_precision'], test_metrics['multi_precision']],
            'Multi_Recall': [val_metrics['multi_recall'], test_metrics['multi_recall']],
            'Multi_F1': [val_metrics['multi_f1'], test_metrics['multi_f1']],
            'Multi_mAP': [val_metrics['multi_mAP'], test_metrics['multi_mAP']]
        }

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(config.output_dir, 'results', 'evaluation_metrics.csv'), index=False)

        # 保存详细报告
        with open(os.path.join(config.output_dir, 'results', 'detailed_report.txt'), 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write("Dataset Statistics:\n")
            f.write(f"  Training samples: {len(train_dataset)}\n")
            f.write(f"  Validation samples: {len(val_dataset)}\n")
            f.write(f"  Test samples: {len(test_dataset)}\n")
            f.write(f"  Number of classes: {num_classes}\n\n")

            f.write("="*80 + "\n")
            f.write("VALIDATION SET RESULTS\n")
            f.write("="*80 + "\n\n")

            f.write("Binary Classification (Healthy vs Disease):\n")
            f.write(f"  Accuracy:  {val_metrics['binary_accuracy']:.4f}\n")
            f.write(f"  Precision: {val_metrics['binary_precision']:.4f}\n")
            f.write(f"  Recall:    {val_metrics['binary_recall']:.4f}\n")
            f.write(f"  F1-Score:  {val_metrics['binary_f1']:.4f}\n")
            f.write(f"  AUC:       {val_metrics['binary_auc']:.4f}\n\n")

            f.write("Multi-class Classification (Specific Diseases):\n")
            f.write(f"  Accuracy:  {val_metrics['multi_accuracy']:.4f}\n")
            f.write(f"  Precision: {val_metrics['multi_precision']:.4f}\n")
            f.write(f"  Recall:    {val_metrics['multi_recall']:.4f}\n")
            f.write(f"  F1-Score:  {val_metrics['multi_f1']:.4f}\n")
            f.write(f"  mAP:       {val_metrics['multi_mAP']:.4f}\n\n")

            f.write("="*80 + "\n")
            f.write("TEST SET RESULTS\n")
            f.write("="*80 + "\n\n")

            f.write("Binary Classification (Healthy vs Disease):\n")
            f.write(f"  Accuracy:  {test_metrics['binary_accuracy']:.4f}\n")
            f.write(f"  Precision: {test_metrics['binary_precision']:.4f}\n")
            f.write(f"  Recall:    {test_metrics['binary_recall']:.4f}\n")
            f.write(f"  F1-Score:  {test_metrics['binary_f1']:.4f}\n")
            f.write(f"  AUC:       {test_metrics['binary_auc']:.4f}\n\n")

            f.write("Multi-class Classification (Specific Diseases):\n")
            f.write(f"  Accuracy:  {test_metrics['multi_accuracy']:.4f}\n")
            f.write(f"  Precision: {test_metrics['multi_precision']:.4f}\n")
            f.write(f"  Recall:    {test_metrics['multi_recall']:.4f}\n")
            f.write(f"  F1-Score:  {test_metrics['multi_f1']:.4f}\n")
            f.write(f"  mAP:       {test_metrics['multi_mAP']:.4f}\n")

        # 7. 可视化
        print("Creating visualizations...")

        # 绘制混淆矩阵
        plot_confusion_matrices(
            test_metrics['binary_confusion_matrix'],
            test_metrics['multi_confusion_matrix'],
            class_names
        )

        # 绘制类别分布
        train_counts = [len([s for s in train_dataset.samples if s['class_idx'] == i])
                       for i in range(num_classes)]
        val_counts = [len([s for s in val_dataset.samples if s['class_idx'] == i])
                     for i in range(num_classes)]
        test_counts = [len([s for s in test_dataset.samples if s['class_idx'] == i])
                      for i in range(num_classes)]

        plot_class_distribution(class_names, train_counts, val_counts, test_counts)

        # 8. 生成每类详细报告
        print("\nGenerating per-class metrics...")

        # 获取多分类报告
        multi_report = test_metrics['multi_report']

        # 创建每类指标表格
        class_metrics = []
        for i, class_name in enumerate(class_names):
            if str(i) in multi_report:
                metrics = multi_report[str(i)]
                class_metrics.append({
                    'Class': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': metrics['support']
                })

        class_df = pd.DataFrame(class_metrics)
        class_df.to_csv(os.path.join(config.output_dir, 'results', 'per_class_metrics.csv'), index=False)

        print("\n" + "="*80)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved to: {config.output_dir}/results/")
        print("\nFiles created:")
        print("  - evaluation_metrics.csv")
        print("  - per_class_metrics.csv")
        print("  - detailed_report.txt")
        print("  - confusion_matrices.png")
        print("  - class_distribution.png")

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\nGPU memory cleared. Current usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()