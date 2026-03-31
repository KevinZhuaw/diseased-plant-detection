# fixed_training.py
"""
Fixed Leaf Disease Classification Training
Fixed BatchNorm issue and optimized for better performance
"""

import os
import sys
import json
import time
import math
from datetime import datetime
from pathlib import Path

# Force CPU usage to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("=" * 80)
print("FIXED LEAF DISEASE CLASSIFICATION SYSTEM")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check and import packages
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image
    import numpy as np

    print("✓ All required packages imported successfully")
except ImportError as e:
    print(f"✗ Missing package: {e}")
    sys.exit(1)


class FixedLeafDataset(Dataset):
    """Fixed dataset with proper error handling"""

    def __init__(self, root_dir, mode='train', img_size=224, max_samples_per_class=500):
        self.root_dir = Path(root_dir) / mode
        self.img_size = img_size
        self.mode = mode

        if not self.root_dir.exists():
            print(f"Error: Directory {self.root_dir} does not exist!")
            self.classes = []
            self.samples = []
            return

        # Get classes
        try:
            self.classes = sorted([d.name for d in self.root_dir.iterdir()
                                   if d.is_dir()])
        except Exception as e:
            print(f"Error reading directory: {e}")
            self.classes = []
            self.samples = []
            return

        if not self.classes:
            print(f"Warning: No classes found in {self.root_dir}")
            return

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}

        # Collect samples
        self.samples = []
        print(f"\nLoading {mode} dataset...")

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            # Get image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
                image_files.extend(list(class_dir.glob(f'*{ext}')))

            # Limit samples per class for faster training
            if max_samples_per_class and len(image_files) > max_samples_per_class:
                import random
                image_files = random.sample(image_files, max_samples_per_class)

            for img_path in image_files:
                try:
                    # Check if file exists and is readable
                    if img_path.exists() and img_path.stat().st_size > 0:
                        label = self.class_to_idx[class_name]
                        self.samples.append((str(img_path), label))
                except Exception as e:
                    print(f"Warning: Could not add {img_path}: {e}")

        if not self.samples:
            print(f"Error: No valid images found in {self.root_dir}")
            return

        print(f"✓ Loaded {len(self.samples)} images from {mode} set")
        print(f"✓ Number of classes: {len(self.classes)}")

        # Create binary labels
        self.binary_labels = []
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            is_healthy = 'healthy' in class_name.lower()
            self.binary_labels.append(0 if is_healthy else 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, disease_label = self.samples[idx]
        binary_label = self.binary_labels[idx]

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a valid placeholder image
            img = Image.new('RGB', (self.img_size, self.img_size), color=(0, 0, 0))

        # Define transformations
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        try:
            img_tensor = transform(img)
        except Exception as e:
            print(f"Error transforming image {img_path}: {e}")
            img_tensor = torch.zeros((3, self.img_size, self.img_size))

        return img_tensor, disease_label, binary_label


class FixedDualHeadModel(nn.Module):
    """Fixed model with GroupNorm instead of BatchNorm to avoid batch size issues"""

    def __init__(self, num_classes, backbone='resnet18'):
        super().__init__()

        # Load pre-trained backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Disease classifier with GroupNorm instead of BatchNorm
        self.disease_classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GroupNorm(32, 512),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.GroupNorm(32, 256),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Binary classifier with GroupNorm instead of BatchNorm
        self.binary_classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GroupNorm(32, 256),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GroupNorm(32, 128),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        # Store num_classes for reference
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        disease_out = self.disease_classifier(features)
        binary_out = self.binary_classifier(features)
        return disease_out, binary_out


class FixedTrainer:
    """Fixed trainer with batch size handling"""

    def __init__(self, data_path, output_dir='fixed_results'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.model = None
        self.class_names = None

        # Results storage
        self.results = {
            'training_info': {},
            'training_history': [],
            'validation_results': {},
            'test_results': {},
            'per_class_results': {},
            'model_info': {},
            'dataset_stats': {}
        }

        # Training parameters
        self.batch_size = 16
        self.learning_rate = 0.001
        self.num_epochs = 15

    def load_datasets(self):
        """Load all datasets with error handling"""
        print("\n" + "=" * 80)
        print("LOADING DATASETS")
        print("=" * 80)

        try:
            # Load with limited samples for faster training
            self.train_dataset = FixedLeafDataset(
                self.data_path, 'train', img_size=128, max_samples_per_class=300
            )
            self.val_dataset = FixedLeafDataset(
                self.data_path, 'val', img_size=128, max_samples_per_class=100
            )
            self.test_dataset = FixedLeafDataset(
                self.data_path, 'test', img_size=128, max_samples_per_class=200
            )

            if len(self.train_dataset.samples) == 0:
                raise ValueError("No training data found!")

            self.class_names = self.train_dataset.classes

            print(f"\n✓ Successfully loaded datasets:")
            print(f"  Training samples: {len(self.train_dataset)}")
            print(f"  Validation samples: {len(self.val_dataset)}")
            print(f"  Test samples: {len(self.test_dataset)}")
            print(f"  Number of classes: {len(self.class_names)}")

            # Show first 10 classes
            print(f"\nFirst 10 classes:")
            for i, cls in enumerate(self.class_names[:10]):
                print(f"  {i + 1:2d}. {cls}")
            if len(self.class_names) > 10:
                print(f"  ... and {len(self.class_names) - 10} more classes")

        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise

    def create_model(self):
        """Create and initialize model"""
        print(f"\nCreating model...")

        try:
            self.model = FixedDualHeadModel(
                num_classes=len(self.class_names),
                backbone='resnet18'
            ).to(self.device)

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            self.results['model_info'] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'device': str(self.device)
            }

            print(f"✓ Model created successfully")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")

        except Exception as e:
            print(f"Error creating model: {e}")
            raise

    def create_data_loaders(self):
        """Create data loaders with proper batch handling"""
        print(f"\nCreating data loaders with batch size: {self.batch_size}")

        # Use drop_last=True for training to avoid single-sample batches
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,  # Important: drop last incomplete batch
            pin_memory=False
        )

        # For validation and test, we can keep all samples
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False
        )

        print(f"✓ Data loaders created:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader, epoch, criterion_disease, criterion_binary, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct_disease = 0
        total_correct_binary = 0
        total_samples = 0

        for batch_idx, (images, disease_labels, binary_labels) in enumerate(train_loader):
            # Skip if batch size is too small (shouldn't happen with drop_last=True)
            if images.size(0) < 2:
                continue

            images = images.to(self.device)
            disease_labels = disease_labels.to(self.device)
            binary_labels = binary_labels.to(self.device)

            # Forward pass
            disease_output, binary_output = self.model(images)

            # Calculate losses
            loss_disease = criterion_disease(disease_output, disease_labels)
            loss_binary = criterion_binary(binary_output, binary_labels)
            loss = loss_disease + 0.5 * loss_binary

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()

            # Calculate accuracy
            _, disease_preds = torch.max(disease_output, 1)
            _, binary_preds = torch.max(binary_output, 1)

            total_correct_disease += (disease_preds == disease_labels).sum().item()
            total_correct_binary += (binary_preds == binary_labels).sum().item()
            total_samples += disease_labels.size(0)

            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                batch_acc = (disease_preds == disease_labels).float().mean().item()
                print(f"Epoch {epoch + 1}/{self.num_epochs}, "
                      f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {batch_acc * 100:.2f}%")

        # Calculate epoch statistics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc_disease = total_correct_disease / total_samples
        epoch_acc_binary = total_correct_binary / total_samples

        return epoch_loss, epoch_acc_disease, epoch_acc_binary

    def validate(self, val_loader, criterion_disease, criterion_binary):
        """Validate the model"""
        self.model.eval()
        total_loss_disease = 0.0
        total_loss_binary = 0.0
        total_correct_disease = 0
        total_correct_binary = 0
        total_samples = 0

        all_disease_preds = []
        all_disease_labels = []
        all_binary_preds = []
        all_binary_labels = []

        with torch.no_grad():
            for images, disease_labels, binary_labels in val_loader:
                images = images.to(self.device)
                disease_labels = disease_labels.to(self.device)
                binary_labels = binary_labels.to(self.device)

                # Forward pass
                disease_output, binary_output = self.model(images)

                # Calculate losses
                loss_disease = criterion_disease(disease_output, disease_labels)
                loss_binary = criterion_binary(binary_output, binary_labels)

                total_loss_disease += loss_disease.item()
                total_loss_binary += loss_binary.item()

                # Get predictions
                _, disease_preds = torch.max(disease_output, 1)
                _, binary_preds = torch.max(binary_output, 1)

                total_correct_disease += (disease_preds == disease_labels).sum().item()
                total_correct_binary += (binary_preds == binary_labels).sum().item()
                total_samples += disease_labels.size(0)

                # Store for metrics calculation
                all_disease_preds.extend(disease_preds.cpu().numpy())
                all_disease_labels.extend(disease_labels.cpu().numpy())
                all_binary_preds.extend(binary_preds.cpu().numpy())
                all_binary_labels.extend(binary_labels.cpu().numpy())

        # Calculate metrics
        val_loss_disease = total_loss_disease / len(val_loader)
        val_loss_binary = total_loss_binary / len(val_loader)
        val_acc_disease = total_correct_disease / total_samples
        val_acc_binary = total_correct_binary / total_samples

        # Convert to numpy arrays for metric calculation
        all_disease_preds = np.array(all_disease_preds)
        all_disease_labels = np.array(all_disease_labels)
        all_binary_preds = np.array(all_binary_preds)
        all_binary_labels = np.array(all_binary_labels)

        # Calculate precision, recall, F1
        def calculate_binary_metrics(preds, labels):
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            return precision, recall, f1

        binary_precision, binary_recall, binary_f1 = calculate_binary_metrics(
            all_binary_preds, all_binary_labels
        )

        # Calculate per-class accuracy for disease classification
        per_class_acc = {}
        for class_idx in range(len(self.class_names)):
            mask = all_disease_labels == class_idx
            if mask.sum() > 0:
                class_acc = (all_disease_preds[mask] == class_idx).mean()
                per_class_acc[self.class_names[class_idx]] = float(class_acc)

        metrics = {
            'loss_disease': val_loss_disease,
            'loss_binary': val_loss_binary,
            'acc_disease': val_acc_disease,
            'acc_binary': val_acc_binary,
            'binary_precision': binary_precision,
            'binary_recall': binary_recall,
            'binary_f1': binary_f1,
            'per_class_accuracy': per_class_acc
        }

        return metrics

    def train(self):
        """Main training loop"""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)

        # Create data loaders
        train_loader, val_loader, _ = self.create_data_loaders()

        # Loss functions
        criterion_disease = nn.CrossEntropyLoss()
        criterion_binary = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6
        )

        # Training loop
        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            # Train for one epoch
            train_loss, train_acc_disease, train_acc_binary = self.train_epoch(
                train_loader, epoch, criterion_disease, criterion_binary, optimizer
            )

            # Validate
            val_metrics = self.validate(val_loader, criterion_disease, criterion_binary)

            # Update learning rate
            scheduler.step()

            # Record epoch results
            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc_disease': train_acc_disease,
                'train_acc_binary': train_acc_binary,
                'val_acc_disease': val_metrics['acc_disease'],
                'val_acc_binary': val_metrics['acc_binary'],
                'val_binary_precision': val_metrics['binary_precision'],
                'val_binary_recall': val_metrics['binary_recall'],
                'val_binary_f1': val_metrics['binary_f1'],
                'epoch_time': time.time() - epoch_start
            }

            self.results['training_history'].append(epoch_result)

            print(f"\nEpoch {epoch + 1}/{self.num_epochs} Summary:")
            print(f"  Training - Loss: {train_loss:.4f}, "
                  f"Disease Acc: {train_acc_disease:.4f}, "
                  f"Binary Acc: {train_acc_binary:.4f}")
            print(f"  Validation - Disease Acc: {val_metrics['acc_disease']:.4f}, "
                  f"Binary Acc: {val_metrics['acc_binary']:.4f}, "
                  f"Binary F1: {val_metrics['binary_f1']:.4f}")

            # Save best model
            if val_metrics['acc_disease'] > best_val_acc:
                best_val_acc = val_metrics['acc_disease']
                self.save_model(f"best_model_epoch{epoch + 1}_acc{best_val_acc:.4f}.pth")
                print(f"  ✓ Saved best model (Val Acc: {best_val_acc:.4f})")

        total_time = time.time() - start_time

        # Store training info
        self.results['training_info'] = {
            'total_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'total_training_time': total_time,
            'best_val_accuracy': best_val_acc
        }

        print(f"\n✓ Training completed in {total_time:.2f} seconds")
        print(f"✓ Best validation accuracy: {best_val_acc:.4f}")

        return True

    def test(self):
        """Test the model on test set"""
        print("\n" + "=" * 80)
        print("TESTING MODEL")
        print("=" * 80)

        # Create test loader
        _, _, test_loader = self.create_data_loaders()

        # Load best model if available
        best_model_path = self.output_dir / "best_model.pth"
        if best_model_path.exists():
            print("Loading best model for testing...")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate on test set
        test_metrics = self.validate(test_loader,
                                     nn.CrossEntropyLoss(),
                                     nn.CrossEntropyLoss())

        # Store test results
        self.results['test_results'] = test_metrics

        print(f"\nTest Results:")
        print(f"  Disease Classification Accuracy: {test_metrics['acc_disease']:.4f}")
        print(f"  Binary Classification Accuracy: {test_metrics['acc_binary']:.4f}")
        print(f"  Binary Precision: {test_metrics['binary_precision']:.4f}")
        print(f"  Binary Recall: {test_metrics['binary_recall']:.4f}")
        print(f"  Binary F1-Score: {test_metrics['binary_f1']:.4f}")

        # Show top 10 and bottom 10 classes
        sorted_classes = sorted(test_metrics['per_class_accuracy'].items(),
                                key=lambda x: x[1], reverse=True)

        print(f"\nTop 10 Performing Classes:")
        for i, (class_name, acc) in enumerate(sorted_classes[:10], 1):
            print(f"  {i:2d}. {class_name:<30} {acc:.4f}")

        print(f"\nBottom 10 Performing Classes:")
        for i, (class_name, acc) in enumerate(sorted_classes[-10:], 1):
            print(f"  {i:2d}. {class_name:<30} {acc:.4f}")

        return test_metrics

    def save_model(self, filename):
        """Save model to file"""
        model_path = self.output_dir / filename
        torch.save({
            'epoch': len(self.results['training_history']),
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'val_accuracy': self.results['training_history'][-1]['val_acc_disease']
        }, model_path)

        # Also save as best_model.pth
        torch.save({
            'epoch': len(self.results['training_history']),
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'val_accuracy': self.results['training_history'][-1]['val_acc_disease']
        }, self.output_dir / "best_model.pth")

        print(f"Model saved to: {model_path}")

    def save_results(self):
        """Save all results to JSON file"""
        results_file = self.output_dir / "training_results.json"

        # Convert any non-serializable objects
        def default_serializer(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return str(obj)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, default=default_serializer, indent=2)

        print(f"Results saved to: {results_file}")

    def generate_summary_txt(self):
        """Generate comprehensive summary in TXT format"""
        summary_file = self.output_dir / "complete_summary.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LEAF DISEASE CLASSIFICATION - COMPLETE SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Path: {self.data_path}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")

            # Dataset Information
            f.write("=" * 80 + "\n")
            f.write("DATASET INFORMATION\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Number of Classes: {len(self.class_names)}\n")
            f.write(f"Training Samples: {len(self.train_dataset)}\n")
            f.write(f"Validation Samples: {len(self.val_dataset)}\n")
            f.write(f"Test Samples: {len(self.test_dataset)}\n\n")

            f.write("Class List:\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"  {i + 1:3d}. {class_name}\n")

            # Model Information
            f.write("\n" + "=" * 80 + "\n")
            f.write("MODEL INFORMATION\n")
            f.write("=" * 80 + "\n\n")

            model_info = self.results['model_info']
            f.write(f"Total Parameters: {model_info['total_parameters']:,}\n")
            f.write(f"Trainable Parameters: {model_info['trainable_parameters']:,}\n")
            f.write(f"Training Device: {model_info['device']}\n\n")

            # Training Information
            f.write("=" * 80 + "\n")
            f.write("TRAINING INFORMATION\n")
            f.write("=" * 80 + "\n\n")

            train_info = self.results['training_info']
            f.write(f"Total Epochs: {train_info['total_epochs']}\n")
            f.write(f"Batch Size: {train_info['batch_size']}\n")
            f.write(f"Learning Rate: {train_info['learning_rate']}\n")
            f.write(f"Total Training Time: {train_info['total_training_time']:.2f} seconds\n")
            f.write(f"Best Validation Accuracy: {train_info['best_val_accuracy']:.4f}\n\n")

            # Training History
            f.write("=" * 80 + "\n")
            f.write("TRAINING HISTORY\n")
            f.write("=" * 80 + "\n\n")

            f.write("Epoch | Train Loss | Train Acc | Val Acc | Binary Acc | Time\n")
            f.write("-" * 80 + "\n")

            for epoch_data in self.results['training_history']:
                f.write(f"{epoch_data['epoch']:5d} | "
                        f"{epoch_data['train_loss']:10.4f} | "
                        f"{epoch_data['train_acc_disease']:9.4f} | "
                        f"{epoch_data['val_acc_disease']:7.4f} | "
                        f"{epoch_data['val_acc_binary']:10.4f} | "
                        f"{epoch_data['epoch_time']:.1f}s\n")

            # Test Results
            if 'test_results' in self.results and self.results['test_results']:
                f.write("\n" + "=" * 80 + "\n")
                f.write("TEST RESULTS\n")
                f.write("=" * 80 + "\n\n")

                test_results = self.results['test_results']

                # Binary Classification Results
                f.write("BINARY CLASSIFICATION (Healthy vs Diseased):\n")
                f.write("-" * 50 + "\n")
                f.write(f"Accuracy:  {test_results['acc_binary']:.4f}\n")
                f.write(f"Precision: {test_results['binary_precision']:.4f}\n")
                f.write(f"Recall:    {test_results['binary_recall']:.4f}\n")
                f.write(f"F1-Score:  {test_results['binary_f1']:.4f}\n\n")

                # Multi-class Classification Results
                f.write("MULTI-CLASS CLASSIFICATION (Specific Diseases):\n")
                f.write("-" * 50 + "\n")
                f.write(f"Accuracy:  {test_results['acc_disease']:.4f}\n\n")

                # Per-class Accuracy
                f.write("PER-CLASS ACCURACY:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Class Name':<35} {'Accuracy':<10}\n")
                f.write("-" * 60 + "\n")

                sorted_classes = sorted(test_results['per_class_accuracy'].items(),
                                        key=lambda x: x[1], reverse=True)

                for class_name, accuracy in sorted_classes:
                    f.write(f"{class_name:<35} {accuracy:.4f}\n")

                # Performance Summary
                f.write("\n" + "=" * 80 + "\n")
                f.write("PERFORMANCE SUMMARY\n")
                f.write("=" * 80 + "\n\n")

                accuracies = list(test_results['per_class_accuracy'].values())
                avg_accuracy = np.mean(accuracies) if accuracies else 0.0
                max_accuracy = max(accuracies) if accuracies else 0.0
                min_accuracy = min(accuracies) if accuracies else 0.0

                f.write(f"Overall Binary Classification Accuracy:   {test_results['acc_binary']:.4f}\n")
                f.write(f"Overall Multi-class Classification Accuracy: {test_results['acc_disease']:.4f}\n")
                f.write(f"Average Class Accuracy:                  {avg_accuracy:.4f}\n")
                f.write(f"Best Class Accuracy:                     {max_accuracy:.4f}\n")
                f.write(f"Worst Class Accuracy:                    {min_accuracy:.4f}\n")
                f.write(
                    f"Standard Deviation of Class Accuracies:  {np.std(accuracies) if len(accuracies) > 1 else 0.0:.4f}\n\n")

                # Recommendations
                f.write("=" * 80 + "\n")
                f.write("RECOMMENDATIONS\n")
                f.write("=" * 80 + "\n\n")

                if avg_accuracy < 0.7:
                    f.write("Performance needs improvement. Suggestions:\n")
                    f.write("1. Increase training data for low-performing classes\n")
                    f.write("2. Apply more data augmentation\n")
                    f.write("3. Try different model architectures\n")
                    f.write("4. Adjust hyperparameters (learning rate, batch size)\n")
                    f.write("5. Use class weighting for imbalanced classes\n")
                elif avg_accuracy < 0.85:
                    f.write("Good performance achieved. For further improvement:\n")
                    f.write("1. Train for more epochs\n")
                    f.write("2. Try larger model architectures\n")
                    f.write("3. Use ensemble methods\n")
                    f.write("4. Fine-tune on specific problem areas\n")
                else:
                    f.write("Excellent performance achieved!\n")
                    f.write("Consider deploying the model for real-world use.\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("=" * 80 + "\n")

        print(f"Summary saved to: {summary_file}")
        return summary_file

    def run(self):
        """Run complete training pipeline"""
        try:
            # Load datasets
            self.load_datasets()

            # Create model
            self.create_model()

            # Train model
            success = self.train()
            if not success:
                return False

            # Test model
            self.test()

            # Save results
            self.save_results()

            # Generate summary
            self.generate_summary_txt()

            print("\n" + "=" * 80)
            print("PROCESS COMPLETED SUCCESSFULLY!")
            print("=" * 80)

            return True

        except Exception as e:
            print(f"\nError during execution: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function"""
    # Configuration
    data_path = "E:/庄楷文/叶片病虫害识别/baseline/data/reorganized_dataset_new"
    output_dir = "leaf_disease_final_results"

    print("Fixed Leaf Disease Classification System")
    print(f"Data Path: {data_path}")
    print(f"Output Directory: {output_dir}")
    print()

    # Check data path
    if not os.path.exists(data_path):
        print(f"Error: Data path does not exist: {data_path}")
        return

    # Create trainer
    trainer = FixedTrainer(data_path, output_dir)

    # Run training
    success = trainer.run()

    if success:
        print(f"\n✓ All processes completed successfully!")
        print(f"✓ Results saved to: {output_dir}/")
        print(f"✓ Summary file: {output_dir}/complete_summary.txt")
    else:
        print("\n✗ Process failed. Please check the error messages above.")


if __name__ == "__main__":
    main()