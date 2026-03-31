"""
Deep SVDD Plant Disease Recognition with Complete Metrics
Fixed multiprocessing issues and includes all required evaluation metrics
"""

import os
import sys
import time
import glob
import json
import numpy as np
from datetime import datetime
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torchvision.transforms as transforms

# Fix for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==================== GPU Setup ====================
def setup_device():
    """Setup GPU with comprehensive diagnostics"""
    print("\n" + "="*60)
    print("HARDWARE DIAGNOSTICS")
    print("="*60)

    # PyTorch info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"✅ GPU Detected: {gpu_name}")
        print(f"   Memory: {gpu_memory:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")

        # Set CUDA optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Clear GPU cache
        torch.cuda.empty_cache()

    else:
        device = torch.device("cpu")
        print("⚠️  WARNING: CUDA is not available!")
        print("   Using CPU - training will be SLOW")

    print("="*60)
    return device

# ==================== Configuration ====================
class Config:
    """Configuration class optimized for RTX 4070Ti"""
    # Paths
    DATASET_PATH = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"
    RESULTS_PATH = r"E:\zhuangkaiwen\results\deep_svdd_complete"

    # Device
    DEVICE = setup_device()
    USE_GPU = torch.cuda.is_available()

    # Model parameters
    IMG_SIZE = 128
    BATCH_SIZE = 32 if USE_GPU else 8
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    LATENT_DIM = 256

    # Dataset
    HEALTHY_CLASS = "healthy"

    # Training
    VAL_SPLIT = 0.15
    PATIENCE = 15

    # No multiprocessing on Windows to avoid MemoryError
    NUM_WORKERS = 0  # Windows multiprocessing issue fix

# ==================== Dataset ====================
class PlantDataset(Dataset):
    """Plant disease dataset"""

    def __init__(self, data_dir, mode="train", binary=True):
        self.data_dir = os.path.join(data_dir, mode)
        self.mode = mode
        self.binary = binary

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        print(f"\n📁 Loading {mode} dataset...")

        # Get all classes
        all_classes = sorted([d for d in os.listdir(self.data_dir)
                            if os.path.isdir(os.path.join(self.data_dir, d))])

        if not all_classes:
            raise ValueError(f"No classes found in {self.data_dir}")

        self.samples = []
        self.class_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        if binary:
            # Binary classification
            self._setup_binary_classes(all_classes)
        else:
            # Multi-class classification
            self._setup_multiclass_classes(all_classes)

        print(f"✅ Loaded {len(self.samples)} images, {len(self.class_names)} classes")

        # Print class distribution
        self._print_class_distribution()

    def _setup_binary_classes(self, all_classes):
        """Setup binary classification classes"""
        self.class_names = ["healthy", "diseased"]
        self.class_to_idx = {"healthy": 0, "diseased": 1}
        self.idx_to_class = {0: "healthy", 1: "diseased"}

        # Find healthy class
        healthy_class = None
        for cls in all_classes:
            if Config.HEALTHY_CLASS.lower() in cls.lower():
                healthy_class = cls
                break

        if healthy_class is None and all_classes:
            # Try alternative names
            healthy_keywords = ['healthy', 'normal', 'good', 'ok', 'control']
            for cls in all_classes:
                if any(keyword in cls.lower() for keyword in healthy_keywords):
                    healthy_class = cls
                    break

        if healthy_class is None and all_classes:
            healthy_class = all_classes[0]
            print(f"⚠️  Using {healthy_class} as healthy class")

        # Load images
        for cls in all_classes:
            cls_path = os.path.join(self.data_dir, cls)
            images = self._load_images_from_folder(cls_path)
            label = 0 if cls == healthy_class else 1

            for img_path in images:
                self.samples.append({
                    'path': img_path,
                    'label': label,
                    'class_name': cls,
                    'binary_label': label
                })

    def _setup_multiclass_classes(self, all_classes):
        """Setup multi-class classification classes"""
        # Filter out healthy class
        disease_classes = []
        for cls in all_classes:
            if Config.HEALTHY_CLASS.lower() not in cls.lower():
                disease_classes.append(cls)

        if not disease_classes:
            raise ValueError("No disease classes found for multi-class classification")

        self.class_names = sorted(disease_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.class_names)}

        # Load images
        for cls in disease_classes:
            cls_path = os.path.join(self.data_dir, cls)
            images = self._load_images_from_folder(cls_path)
            label = self.class_to_idx[cls]

            for img_path in images:
                self.samples.append({
                    'path': img_path,
                    'label': label,
                    'class_name': cls,
                    'binary_label': 1  # All are diseased for multi-class
                })

    def _load_images_from_folder(self, folder_path):
        """Load all images from a folder"""
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP']
        images = []

        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(folder_path, f"*.{ext}")))

        return sorted(images)

    def _print_class_distribution(self):
        """Print class distribution"""
        if self.binary:
            healthy_count = sum(1 for s in self.samples if s['label'] == 0)
            diseased_count = sum(1 for s in self.samples if s['label'] == 1)
            print(f"📊 Class Distribution:")
            print(f"   Healthy: {healthy_count} ({healthy_count/len(self.samples)*100:.1f}%)")
            print(f"   Diseased: {diseased_count} ({diseased_count/len(self.samples)*100:.1f}%)")
        else:
            from collections import Counter
            class_counts = Counter([s['class_name'] for s in self.samples])
            print(f"📊 Class Distribution (Top 10):")
            for class_name, count in list(class_counts.items())[:10]:
                percentage = count / len(self.samples) * 100
                print(f"   {class_name}: {count} ({percentage:.1f}%)")

            if len(class_counts) > 10:
                print(f"   ... and {len(class_counts) - 10} more classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['path']
        label = sample['label']

        try:
            # Load and process image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((Config.IMG_SIZE, Config.IMG_SIZE))

            # Convert to tensor and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img_array = (img_array - mean) / std

            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()

            return img_tensor, label, os.path.basename(img_path)

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE), label, "error.jpg"

# ==================== Model ====================
class DeepSVDDModel(nn.Module):
    """Deep SVDD model with improved architecture"""

    def __init__(self, latent_dim=256):
        super(DeepSVDDModel, self).__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

# ==================== Metrics Calculator ====================
class MetricsCalculator:
    """Calculate all required metrics"""

    @staticmethod
    def calculate_metrics(y_true, y_pred, y_scores=None, average='binary'):
        """Calculate all classification metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )

        results = {}

        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1 with different averaging
        try:
            results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            results['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            results['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            # Per-class metrics
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

            results['precision_per_class'] = precision_per_class.tolist()
            results['recall_per_class'] = recall_per_class.tolist()
            results['f1_per_class'] = f1_per_class.tolist()

        except Exception as e:
            print(f"⚠️  Error calculating some metrics: {e}")

        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            results['confusion_matrix'] = cm.tolist()
        except:
            results['confusion_matrix'] = None

        # Classification report
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            results['classification_report'] = report
        except:
            results['classification_report'] = None

        # mAP calculation for multi-class
        if y_scores is not None and len(np.unique(y_true)) > 2:
            try:
                results['map'] = MetricsCalculator.calculate_map(y_true, y_pred, y_scores)
            except:
                results['map'] = None

        return results

    @staticmethod
    def calculate_map(y_true, y_pred, y_scores, num_classes=None):
        """Calculate mean Average Precision (mAP)"""
        try:
            if num_classes is None:
                num_classes = len(np.unique(y_true))

            # For simplicity, we'll calculate AP for each class and average
            aps = []
            for class_idx in range(num_classes):
                try:
                    # Create binary labels for this class
                    y_true_binary = (y_true == class_idx).astype(int)
                    y_pred_binary = (y_pred == class_idx).astype(int)

                    # Calculate precision-recall curve
                    from sklearn.metrics import precision_recall_curve, average_precision_score

                    # Get scores for this class (use predictions as proxy for scores)
                    if y_scores.ndim > 1 and y_scores.shape[1] > 1:
                        # Multi-class scores
                        scores = y_scores[:, class_idx]
                    else:
                        # Binary scores
                        scores = y_scores if len(y_scores.shape) == 1 else y_scores.flatten()

                    # Calculate AP
                    ap = average_precision_score(y_true_binary, scores)
                    aps.append(ap)
                except:
                    aps.append(0.0)

            # Filter out invalid APs
            valid_aps = [ap for ap in aps if not np.isnan(ap)]
            if valid_aps:
                return np.mean(valid_aps)
            else:
                return 0.0
        except Exception as e:
            print(f"⚠️  Error calculating mAP: {e}")
            return 0.0

    @staticmethod
    def print_metrics_summary(metrics, stage_name, class_names=None):
        """Print metrics summary"""
        print(f"\n📊 {stage_name} METRICS SUMMARY")
        print("-"*40)

        if 'accuracy' in metrics:
            print(f"Accuracy: {metrics['accuracy']:.4f}")

        if 'precision_macro' in metrics:
            print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
            print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")

        if 'recall_macro' in metrics:
            print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
            print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")

        if 'f1_macro' in metrics:
            print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
            print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")

        if 'map' in metrics and metrics['map'] is not None:
            print(f"mAP: {metrics['map']:.4f}")

        # Per-class metrics
        if class_names and 'precision_per_class' in metrics:
            print(f"\n📈 Per-class metrics (Top 5):")
            # Combine metrics for sorting
            class_metrics = []
            for i, class_name in enumerate(class_names):
                if i < len(metrics['precision_per_class']):
                    class_metrics.append({
                        'class': class_name,
                        'precision': metrics['precision_per_class'][i],
                        'recall': metrics['recall_per_class'][i],
                        'f1': metrics['f1_per_class'][i]
                    })

            # Sort by F1 score
            class_metrics.sort(key=lambda x: x['f1'], reverse=True)

            for i, cm in enumerate(class_metrics[:5]):
                print(f"  {i+1}. {cm['class']}: "
                      f"P={cm['precision']:.3f}, R={cm['recall']:.3f}, F1={cm['f1']:.3f}")

# ==================== Trainer ====================
class Trainer:
    """Trainer for Deep SVDD with metrics tracking"""

    def __init__(self, is_binary=True, num_classes=2, class_names=None):
        self.device = Config.DEVICE
        self.is_binary = is_binary
        self.num_classes = num_classes
        self.class_names = class_names if class_names else []

        # Initialize model(s)
        if is_binary:
            self.model = DeepSVDDModel(Config.LATENT_DIM).to(self.device)
            self.center = None
        else:
            self.models = nn.ModuleList([
                DeepSVDDModel(Config.LATENT_DIM).to(self.device)
                for _ in range(num_classes)
            ])
            self.centers = [None] * num_classes

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }

        # Metrics
        self.metrics_calculator = MetricsCalculator()

        print(f"✅ Trainer initialized for {'binary' if is_binary else 'multi-class'} classification")
        print(f"   Device: {self.device}")
        print(f"   Number of classes: {num_classes}")

    def compute_center(self, dataloader, class_idx=0):
        """Compute hypersphere center"""
        if self.is_binary:
            model = self.model
        else:
            model = self.models[class_idx]

        model.eval()
        center = torch.zeros(Config.LATENT_DIM, device=self.device)
        n_samples = 0

        print(f"\n🔍 Computing center for class {class_idx}...")

        with torch.no_grad():
            for data, labels, _ in tqdm(dataloader, desc="Processing"):
                data = data.to(self.device)
                labels = labels.to(self.device)

                if self.is_binary:
                    mask = (labels == 0)  # Healthy samples
                else:
                    mask = (labels == class_idx)

                if mask.sum() > 0:
                    features = model(data[mask])
                    center += features.sum(dim=0)
                    n_samples += features.size(0)

        if n_samples > 0:
            center = center / n_samples
            if self.is_binary:
                self.center = center
                print(f"✅ Binary center computed from {n_samples} samples")
            else:
                self.centers[class_idx] = center
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class {class_idx}"
                print(f"✅ Center for {class_name} computed from {n_samples} samples")
        else:
            center = torch.randn(Config.LATENT_DIM, device=self.device) * 0.1
            if self.is_binary:
                self.center = center
                print(f"⚠️  No samples found, using random center for binary")
            else:
                self.centers[class_idx] = center
                print(f"⚠️  No samples found for class {class_idx}, using random center")

        return center

    def compute_threshold(self, dataloader):
        """Compute anomaly threshold"""
        if not self.is_binary:
            return None

        print("\n📊 Computing anomaly threshold...")
        self.model.eval()
        distances = []

        with torch.no_grad():
            for data, labels, _ in tqdm(dataloader, desc="Calculating distances"):
                data = data.to(self.device)
                mask = (labels == 0)  # Healthy samples

                if mask.sum() > 0:
                    features = self.model(data[mask])
                    dist = torch.sum((features - self.center) ** 2, dim=1)
                    distances.extend(dist.cpu().numpy())

        if distances:
            distances = np.array(distances)
            # Use 95th percentile as threshold
            threshold = np.percentile(distances, 95)
            self.threshold = threshold
            print(f"✅ Threshold computed: {threshold:.6f}")
            return threshold
        else:
            print("⚠️  No healthy distances computed for threshold")
            return None

    def train_epoch(self, dataloader, optimizer):
        """Train one epoch"""
        if self.is_binary:
            self.model.train()
        else:
            for model in self.models:
                model.train()

        total_loss = 0.0

        pbar = tqdm(dataloader, desc="Training")
        for data, labels, _ in pbar:
            data, labels = data.to(self.device), labels.to(self.device)

            if self.is_binary:
                features = self.model(data)
                distances = torch.sum((features - self.center) ** 2, dim=1)
                loss = distances.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
            else:
                total_class_loss = 0.0
                optimizer.zero_grad()

                for class_idx in range(self.num_classes):
                    mask = (labels == class_idx)
                    if mask.sum() > 0:
                        class_data = data[mask]
                        features = self.models[class_idx](class_data)
                        distances = torch.sum((features - self.centers[class_idx]) ** 2, dim=1)
                        class_loss = distances.mean()
                        class_loss.backward()
                        total_class_loss += class_loss.item()

                optimizer.step()
                batch_loss = total_class_loss

            total_loss += batch_loss
            pbar.set_postfix({'loss': f'{batch_loss:.6f}'})

        return total_loss / len(dataloader)

    def validate(self, dataloader, return_predictions=False):
        """Validate model"""
        if self.is_binary:
            self.model.eval()
        else:
            for model in self.models:
                model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for data, labels, _ in tqdm(dataloader, desc="Validating"):
                data, labels = data.to(self.device), labels.to(self.device)

                if self.is_binary:
                    features = self.model(data)
                    distances = torch.sum((features - self.center) ** 2, dim=1)
                    loss = distances.mean()

                    if self.threshold is not None:
                        predictions = (distances > self.threshold).long()
                    else:
                        predictions = (distances > torch.median(distances)).long()

                    all_scores.extend(distances.cpu().numpy())
                else:
                    batch_size = data.size(0)
                    distances = torch.zeros(batch_size, self.num_classes, device=self.device)

                    for class_idx in range(self.num_classes):
                        features = self.models[class_idx](data)
                        dist = torch.sum((features - self.centers[class_idx]) ** 2, dim=1)
                        distances[:, class_idx] = dist

                    loss = 0.0
                    for i in range(batch_size):
                        loss += distances[i, labels[i]]
                    loss = loss / batch_size

                    # Get predictions (closest center)
                    _, predictions = torch.min(distances, dim=1)

                    # Use negative distances as scores (lower distance = higher score)
                    all_scores.extend((-distances).cpu().numpy())

                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(dataloader)

        if return_predictions:
            metrics = self.metrics_calculator.calculate_metrics(
                all_labels, all_predictions, all_scores,
                average='binary' if self.is_binary else 'macro'
            )
            return avg_loss, metrics, all_predictions, all_labels, all_scores
        else:
            metrics = self.metrics_calculator.calculate_metrics(
                all_labels, all_predictions, all_scores,
                average='binary' if self.is_binary else 'macro'
            )
            return avg_loss, metrics

    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"\n🚀 Starting training")
        print(f"   Epochs: {Config.EPOCHS}")
        print(f"   Batch size: {Config.BATCH_SIZE}")
        print(f"   Learning rate: {Config.LEARNING_RATE}")

        # Compute centers
        print(f"\n🔍 Computing centers...")
        if self.is_binary:
            self.compute_center(train_loader)
            self.compute_threshold(train_loader)
        else:
            for i in range(self.num_classes):
                self.compute_center(train_loader, i)

        # Optimizer
        if self.is_binary:
            optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        else:
            params = []
            for model in self.models:
                params.extend(list(model.parameters()))
            optimizer = optim.Adam(params, lr=Config.LEARNING_RATE)

        # Training loop
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{Config.EPOCHS}")
            print(f"{'='*50}")

            # Training
            train_loss = self.train_epoch(train_loader, optimizer)
            self.history['train_loss'].append(train_loss)

            # Validation
            val_loss, val_metrics = self.validate(val_loader)

            # Update history
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_metrics.get('accuracy', 0))
            self.history['val_precision'].append(val_metrics.get('precision_macro', 0))
            self.history['val_recall'].append(val_metrics.get('recall_macro', 0))
            self.history['val_f1'].append(val_metrics.get('f1_macro', 0))

            # Print epoch results
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            if 'accuracy' in val_metrics:
                print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            if 'precision_macro' in val_metrics:
                print(f"Val Precision: {val_metrics['precision_macro']:.4f}")
            if 'recall_macro' in val_metrics:
                print(f"Val Recall: {val_metrics['recall_macro']:.4f}")
            if 'f1_macro' in val_metrics:
                print(f"Val F1-Score: {val_metrics['f1_macro']:.4f}")

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.save_model(f"best_model_epoch_{epoch}")
                print(f"✅ New best model saved")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= Config.PATIENCE:
                print(f"⚠️  Early stopping at epoch {epoch}")
                break

        print(f"\n✅ Training completed!")
        print(f"   Best validation loss: {best_loss:.6f}")

        return self.history

    def save_model(self, name):
        """Save model"""
        save_dir = os.path.join(Config.RESULTS_PATH, "models")
        os.makedirs(save_dir, exist_ok=True)

        if self.is_binary:
            save_path = os.path.join(save_dir, f"{name}_binary.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'center': self.center,
                'threshold': self.threshold,
                'is_binary': True,
                'latent_dim': Config.LATENT_DIM
            }, save_path)
        else:
            save_path = os.path.join(save_dir, f"{name}_multiclass.pth")
            model_states = [model.state_dict() for model in self.models]
            torch.save({
                'model_states': model_states,
                'centers': self.centers,
                'is_binary': False,
                'latent_dim': Config.LATENT_DIM,
                'class_names': self.class_names
            }, save_path)

        print(f"💾 Model saved: {save_path}")

# ==================== Main System ====================
class PlantDiseaseSystem:
    """Main system for plant disease recognition with complete metrics"""

    def __init__(self):
        self.device = Config.DEVICE
        self.binary_trainer = None
        self.multiclass_trainer = None
        self.binary_class_names = ["healthy", "diseased"]
        self.multiclass_class_names = []

        # Create directories
        os.makedirs(Config.RESULTS_PATH, exist_ok=True)
        os.makedirs(os.path.join(Config.RESULTS_PATH, "models"), exist_ok=True)
        os.makedirs(os.path.join(Config.RESULTS_PATH, "plots"), exist_ok=True)
        os.makedirs(os.path.join(Config.RESULTS_PATH, "results"), exist_ok=True)
        os.makedirs(os.path.join(Config.RESULTS_PATH, "evaluation"), exist_ok=True)

        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()

        print(f"\n✅ Plant Disease Recognition System Initialized")
        print(f"   Device: {self.device}")
        print(f"   Using GPU: {Config.USE_GPU}")
        print(f"   Results will be saved to: {Config.RESULTS_PATH}")

    def create_dataloaders(self, binary=True, mode="train"):
        """Create dataloaders"""
        # Create dataset
        dataset = PlantDataset(
            data_dir=Config.DATASET_PATH,
            mode=mode,
            binary=binary
        )

        if binary:
            class_names = self.binary_class_names
        else:
            self.multiclass_class_names = dataset.class_names
            class_names = self.multiclass_class_names

        print(f"\n📊 {mode.capitalize()} Dataset:")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Number of classes: {len(class_names)}")

        # For training, split into train/val
        if mode == "train":
            train_size = int((1 - Config.VAL_SPLIT) * len(dataset))
            val_size = len(dataset) - train_size

            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size]
            )

            print(f"   Training samples: {len(train_dataset)}")
            print(f"   Validation samples: {len(val_dataset)}")

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.USE_GPU
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.USE_GPU
            )

            return train_loader, val_loader, class_names

        else:
            # For test mode
            test_loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.USE_GPU
            )

            return test_loader, class_names

    def train_binary(self):
        """Train binary classification"""
        print("\n" + "="*60)
        print("STAGE 1: BINARY CLASSIFICATION (Healthy vs Diseased)")
        print("="*60)

        # Create dataloaders
        train_loader, val_loader, class_names = self.create_dataloaders(
            binary=True, mode="train"
        )

        # Create trainer
        self.binary_trainer = Trainer(
            is_binary=True,
            num_classes=2,
            class_names=class_names
        )

        # Train
        history = self.binary_trainer.train(train_loader, val_loader)

        # Save history
        self.save_history(history, "binary")

        print("\n✅ Binary classification training completed!")
        return history

    def train_multiclass(self):
        """Train multi-class classification"""
        print("\n" + "="*60)
        print("STAGE 2: MULTI-CLASS CLASSIFICATION (Specific Diseases)")
        print("="*60)

        # Create dataloaders
        train_loader, val_loader, class_names = self.create_dataloaders(
            binary=False, mode="train"
        )

        # Create trainer
        self.multiclass_trainer = Trainer(
            is_binary=False,
            num_classes=len(class_names),
            class_names=class_names
        )

        # Train
        history = self.multiclass_trainer.train(train_loader, val_loader)

        # Save history
        self.save_history(history, "multiclass")

        print("\n✅ Multi-class classification training completed!")
        return history

    def evaluate(self):
        """Evaluate models with complete metrics"""
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION WITH ALL METRICS")
        print("="*60)

        results = {}

        # Binary evaluation
        if self.binary_trainer:
            print("\n" + "="*40)
            print("1. BINARY CLASSIFICATION EVALUATION")
            print("="*40)

            test_loader, class_names = self.create_dataloaders(binary=True, mode="test")

            # Get predictions and metrics
            test_loss, test_metrics, test_preds, test_labels, test_scores = self.binary_trainer.validate(
                test_loader, return_predictions=True
            )

            # Add loss to metrics
            test_metrics['loss'] = test_loss

            # Print metrics summary
            self.metrics_calculator.print_metrics_summary(test_metrics, "Binary Test", class_names)

            results['binary'] = test_metrics
            results['binary']['stage'] = "Binary Classification (Healthy vs Diseased)"
            results['binary']['num_classes'] = 2

            # Save detailed results
            self._save_detailed_results(test_metrics, "binary", class_names)

        # Multi-class evaluation
        if self.multiclass_trainer:
            print("\n" + "="*40)
            print("2. MULTI-CLASS CLASSIFICATION EVALUATION")
            print("="*40)

            test_loader, class_names = self.create_dataloaders(binary=False, mode="test")

            # Get predictions and metrics
            test_loss, test_metrics, test_preds, test_labels, test_scores = self.multiclass_trainer.validate(
                test_loader, return_predictions=True
            )

            # Add loss to metrics
            test_metrics['loss'] = test_loss

            # Print metrics summary
            self.metrics_calculator.print_metrics_summary(test_metrics, "Multi-class Test", class_names)

            results['multiclass'] = test_metrics
            results['multiclass']['stage'] = "Multi-class Classification (Specific Diseases)"
            results['multiclass']['num_classes'] = len(class_names)
            results['multiclass']['class_names'] = class_names

            # Save detailed results
            self._save_detailed_results(test_metrics, "multiclass", class_names)

        # Save overall evaluation results
        self.save_comprehensive_report(results)

        return results

    def _save_detailed_results(self, metrics, stage, class_names):
        """Save detailed metrics to file"""
        results_dir = os.path.join(Config.RESULTS_PATH, "evaluation", stage)
        os.makedirs(results_dir, exist_ok=True)

        # Save metrics as JSON
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)

        # Save metrics as text
        txt_path = os.path.join(results_dir, "metrics_summary.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{stage.upper()} CLASSIFICATION METRICS\n")
            f.write("="*60 + "\n\n")

            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Classes: {len(class_names)}\n\n")

            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write("-"*40 + "\n")

            if 'loss' in metrics:
                f.write(f"Loss: {metrics['loss']:.6f}\n")

            if 'accuracy' in metrics:
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")

            if 'precision_macro' in metrics:
                f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
                f.write(f"Precision (Weighted): {metrics['precision_weighted']:.4f}\n")

            if 'recall_macro' in metrics:
                f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
                f.write(f"Recall (Weighted): {metrics['recall_weighted']:.4f}\n")

            if 'f1_macro' in metrics:
                f.write(f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n")
                f.write(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}\n")

            if 'map' in metrics and metrics['map'] is not None:
                f.write(f"mAP: {metrics['map']:.4f}\n")

            f.write("\n")

            # Per-class metrics
            if 'precision_per_class' in metrics and class_names:
                f.write("PER-CLASS METRICS:\n")
                f.write("-"*40 + "\n")

                for i, class_name in enumerate(class_names):
                    if i < len(metrics['precision_per_class']):
                        f.write(f"{class_name}:\n")
                        f.write(f"  Precision: {metrics['precision_per_class'][i]:.4f}\n")
                        f.write(f"  Recall: {metrics['recall_per_class'][i]:.4f}\n")
                        f.write(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}\n\n")

        print(f"💾 Detailed {stage} metrics saved to: {txt_path}")

    def save_comprehensive_report(self, results):
        """Save comprehensive evaluation report"""
        report_path = os.path.join(Config.RESULTS_PATH, "evaluation", "comprehensive_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PLANT DISEASE RECOGNITION - COMPREHENSIVE EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Using GPU: {Config.USE_GPU}\n")
            f.write(f"Model: Deep SVDD\n")
            f.write(f"Image Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}\n\n")

            # Binary results
            if 'binary' in results:
                binary_metrics = results['binary']
                f.write("="*80 + "\n")
                f.write("BINARY CLASSIFICATION RESULTS (Healthy vs Diseased)\n")
                f.write("="*80 + "\n\n")

                f.write("Overall Performance:\n")
                f.write("-"*40 + "\n")
                f.write(f"Loss: {binary_metrics.get('loss', 0):.6f}\n")
                f.write(f"Accuracy: {binary_metrics.get('accuracy', 0):.4f}\n")
                f.write(f"Precision (Macro): {binary_metrics.get('precision_macro', 0):.4f}\n")
                f.write(f"Recall (Macro): {binary_metrics.get('recall_macro', 0):.4f}\n")
                f.write(f"F1-Score (Macro): {binary_metrics.get('f1_macro', 0):.4f}\n")

                if 'map' in binary_metrics and binary_metrics['map'] is not None:
                    f.write(f"mAP: {binary_metrics['map']:.4f}\n")

                f.write("\n")

                # Confusion matrix
                if 'confusion_matrix' in binary_metrics and binary_metrics['confusion_matrix'] is not None:
                    f.write("Confusion Matrix:\n")
                    f.write("-"*40 + "\n")
                    cm = binary_metrics['confusion_matrix']
                    f.write("      Predicted\n")
                    f.write("      Healthy Diseased\n")
                    f.write(f"True Healthy   {cm[0][0]:4d}   {cm[0][1]:4d}\n")
                    f.write(f"     Diseased  {cm[1][0]:4d}   {cm[1][1]:4d}\n")

                f.write("\n\n")

            # Multi-class results
            if 'multiclass' in results:
                multiclass_metrics = results['multiclass']
                f.write("="*80 + "\n")
                f.write("MULTI-CLASS CLASSIFICATION RESULTS (Specific Diseases)\n")
                f.write("="*80 + "\n\n")

                f.write("Overall Performance:\n")
                f.write("-"*40 + "\n")
                f.write(f"Loss: {multiclass_metrics.get('loss', 0):.6f}\n")
                f.write(f"Accuracy: {multiclass_metrics.get('accuracy', 0):.4f}\n")
                f.write(f"Precision (Macro): {multiclass_metrics.get('precision_macro', 0):.4f}\n")
                f.write(f"Precision (Weighted): {multiclass_metrics.get('precision_weighted', 0):.4f}\n")
                f.write(f"Recall (Macro): {multiclass_metrics.get('recall_macro', 0):.4f}\n")
                f.write(f"Recall (Weighted): {multiclass_metrics.get('recall_weighted', 0):.4f}\n")
                f.write(f"F1-Score (Macro): {multiclass_metrics.get('f1_macro', 0):.4f}\n")
                f.write(f"F1-Score (Weighted): {multiclass_metrics.get('f1_weighted', 0):.4f}\n")

                if 'map' in multiclass_metrics and multiclass_metrics['map'] is not None:
                    f.write(f"mAP: {multiclass_metrics['map']:.4f}\n")

                f.write(f"Number of Classes: {multiclass_metrics.get('num_classes', 0)}\n\n")

                # Top and bottom performing classes
                if 'precision_per_class' in multiclass_metrics and 'class_names' in multiclass_metrics:
                    class_names = multiclass_metrics['class_names']
                    precision_list = multiclass_metrics['precision_per_class']
                    recall_list = multiclass_metrics['recall_per_class']
                    f1_list = multiclass_metrics['f1_per_class']

                    # Combine metrics
                    class_metrics = []
                    for i, class_name in enumerate(class_names):
                        if i < len(precision_list):
                            class_metrics.append({
                                'class': class_name,
                                'precision': precision_list[i],
                                'recall': recall_list[i],
                                'f1': f1_list[i]
                            })

                    # Sort by F1 score
                    class_metrics.sort(key=lambda x: x['f1'], reverse=True)

                    f.write("Top 10 Performing Classes (by F1-Score):\n")
                    f.write("-"*40 + "\n")
                    for i, cm in enumerate(class_metrics[:10]):
                        f.write(f"{i+1:2d}. {cm['class']:<30} "
                               f"F1: {cm['f1']:.4f} (P: {cm['precision']:.4f}, R: {cm['recall']:.4f})\n")

                    if len(class_metrics) > 10:
                        f.write(f"\nBottom {min(5, len(class_metrics)-10)} Performing Classes:\n")
                        f.write("-"*40 + "\n")
                        for i, cm in enumerate(class_metrics[-min(5, len(class_metrics)-10):], 1):
                            idx = len(class_metrics) - min(5, len(class_metrics)-10) + i
                            f.write(f"{idx:2d}. {cm['class']:<30} "
                                   f"F1: {cm['f1']:.4f} (P: {cm['precision']:.4f}, R: {cm['recall']:.4f})\n")

                f.write("\n")

            # Summary
            f.write("="*80 + "\n")
            f.write("SUMMARY AND RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")

            # Binary summary
            if 'binary' in results:
                binary_acc = results['binary'].get('accuracy', 0)
                binary_f1 = results['binary'].get('f1_macro', 0)

                f.write("Binary Classification (Healthy vs Diseased):\n")
                f.write("-"*40 + "\n")
                f.write(f"Status: {'✅ EXCELLENT' if binary_acc > 0.9 else '✅ GOOD' if binary_acc > 0.8 else '⚠️  NEEDS IMPROVEMENT'}\n")
                f.write(f"Accuracy: {binary_acc:.4f}\n")
                f.write(f"F1-Score: {binary_f1:.4f}\n")
                f.write(f"Recommendation: {'No action needed' if binary_acc > 0.85 else 'Consider collecting more balanced training data'}\n\n")

            # Multi-class summary
            if 'multiclass' in results:
                multiclass_acc = results['multiclass'].get('accuracy', 0)
                multiclass_f1 = results['multiclass'].get('f1_macro', 0)

                f.write("Multi-class Classification (Specific Diseases):\n")
                f.write("-"*40 + "\n")
                f.write(f"Status: {'✅ EXCELLENT' if multiclass_acc > 0.8 else '✅ GOOD' if multiclass_acc > 0.7 else '⚠️  NEEDS IMPROVEMENT'}\n")
                f.write(f"Accuracy: {multiclass_acc:.4f}\n")
                f.write(f"F1-Score: {multiclass_f1:.4f}\n")
                f.write(f"Recommendation: {'No action needed' if multiclass_acc > 0.75 else 'Consider data augmentation or model architecture improvements'}\n\n")

            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"\n✅ Comprehensive report saved to: {report_path}")

    def save_history(self, history, stage):
        """Save training history"""
        # Save to JSON
        history_path = os.path.join(Config.RESULTS_PATH, "results", f"{stage}_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)

        # Create plot
        self.plot_training_history(history, stage)

        print(f"📊 Training history saved for {stage} stage")

    def plot_training_history(self, history, stage):
        """Plot training history"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        epochs = range(1, len(history['train_loss']) + 1)

        # Training loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title(f'{stage} - Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Validation loss
        axes[0, 1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title(f'{stage} - Validation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Validation accuracy
        axes[0, 2].plot(epochs, history['val_accuracy'], 'g-', label='Validation Accuracy', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title(f'{stage} - Validation Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim([0, 1])

        # Validation precision
        axes[1, 0].plot(epochs, history['val_precision'], 'orange', label='Validation Precision', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title(f'{stage} - Validation Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])

        # Validation recall
        axes[1, 1].plot(epochs, history['val_recall'], 'purple', label='Validation Recall', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title(f'{stage} - Validation Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])

        # Validation F1-score
        axes[1, 2].plot(epochs, history['val_f1'], 'brown', label='Validation F1-Score', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('F1-Score')
        axes[1, 2].set_title(f'{stage} - Validation F1-Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim([0, 1])

        plt.tight_layout()
        plot_path = os.path.join(Config.RESULTS_PATH, "plots", f"{stage}_training_history.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📈 Training plots saved: {plot_path}")

# ==================== Main Function ====================
def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("PLANT DISEASE RECOGNITION SYSTEM WITH COMPLETE METRICS")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create system
    system = PlantDiseaseSystem()

    # Interactive menu
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Full Pipeline (Train Both + Evaluate with All Metrics)")
        print("2. Train Binary Classification Only")
        print("3. Train Multi-class Classification Only")
        print("4. Evaluate Only (Requires Trained Models)")
        print("5. Quick Test (10 Epochs Each)")
        print("6. Check Dataset Statistics")
        print("7. Exit")
        print("="*60)

        choice = input("\nSelect option (1-7): ").strip()

        try:
            if choice == "1":
                # Full pipeline
                start_time = time.time()

                print("\n🚀 Starting Full Pipeline...")
                print("This will train both binary and multi-class models, then evaluate with all metrics.")

                # Train binary
                binary_history = system.train_binary()

                # Train multi-class
                multiclass_history = system.train_multiclass()

                # Evaluate with all metrics
                results = system.evaluate()

                total_time = time.time() - start_time
                hours, remainder = divmod(total_time, 3600)
                minutes, seconds = divmod(remainder, 60)

                print(f"\n✅ Full pipeline completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

                # Print final summary
                if 'binary' in results:
                    binary_acc = results['binary'].get('accuracy', 0)
                    binary_f1 = results['binary'].get('f1_macro', 0)
                    print(f"📊 Binary Results: Accuracy={binary_acc:.4f}, F1-Score={binary_f1:.4f}")

                if 'multiclass' in results:
                    multiclass_acc = results['multiclass'].get('accuracy', 0)
                    multiclass_f1 = results['multiclass'].get('f1_macro', 0)
                    print(f"📊 Multi-class Results: Accuracy={multiclass_acc:.4f}, F1-Score={multiclass_f1:.4f}")

                print(f"\n📋 Comprehensive report saved in: {os.path.join(Config.RESULTS_PATH, 'evaluation')}")

            elif choice == "2":
                # Train binary only
                start_time = time.time()
                system.train_binary()
                total_time = time.time() - start_time
                print(f"\n✅ Binary training completed in {total_time:.2f} seconds")

            elif choice == "3":
                # Train multi-class only
                start_time = time.time()
                system.train_multiclass()
                total_time = time.time() - start_time
                print(f"\n✅ Multi-class training completed in {total_time:.2f} seconds")

            elif choice == "4":
                # Evaluate only
                if system.binary_trainer is None and system.multiclass_trainer is None:
                    print("\n⚠️  No trained models found!")
                    print("Please train models first or load saved models.")
                else:
                    start_time = time.time()
                    results = system.evaluate()
                    total_time = time.time() - start_time
                    print(f"\n✅ Evaluation completed in {total_time:.2f} seconds")

            elif choice == "5":
                # Quick test
                print("\n🔧 Quick Test Mode (10 epochs each)")
                original_epochs = Config.EPOCHS
                Config.EPOCHS = 10

                start_time = time.time()

                print("\n" + "="*40)
                print("Quick Binary Training")
                print("="*40)
                system.train_binary()

                print("\n" + "="*40)
                print("Quick Multi-class Training")
                print("="*40)
                system.train_multiclass()

                print("\n" + "="*40)
                print("Quick Evaluation")
                print("="*40)
                system.evaluate()

                Config.EPOCHS = original_epochs
                total_time = time.time() - start_time
                print(f"\n✅ Quick test completed in {total_time:.2f} seconds")

            elif choice == "6":
                # Check dataset statistics
                print("\n📊 Dataset Statistics:")
                print("-"*40)

                modes = ["train", "test"]
                for mode in modes:
                    print(f"\n{mode.upper()} mode:")

                    # Binary dataset
                    try:
                        binary_dataset = PlantDataset(Config.DATASET_PATH, mode=mode, binary=True)
                        print(f"  Binary: {len(binary_dataset)} samples")
                    except Exception as e:
                        print(f"  Binary dataset error: {e}")

                    # Multi-class dataset
                    try:
                        multiclass_dataset = PlantDataset(Config.DATASET_PATH, mode=mode, binary=False)
                        print(f"  Multi-class: {len(multiclass_dataset)} samples, {len(multiclass_dataset.class_names)} classes")
                    except Exception as e:
                        print(f"  Multi-class dataset error: {e}")

            elif choice == "7":
                print("\n👋 Exiting...")
                break

            else:
                print("\n❌ Invalid choice. Please select 1-7.")

        except KeyboardInterrupt:
            print("\n\n⚠️  Operation cancelled by user")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

        # Ask if user wants to continue
        if choice != "7":
            continue_choice = input("\nContinue? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\n👋 Exiting...")
                break

    print(f"\n{'='*80}")
    print("PROGRAM COMPLETED")
    print(f"{'='*80}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {Config.RESULTS_PATH}")

# ==================== Entry Point ====================
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run main function
    main()