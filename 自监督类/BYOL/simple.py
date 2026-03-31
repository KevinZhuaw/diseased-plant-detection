# train_compatible.py
"""
Leaf Disease Classification - Compatible Version
Uses NumPy 1.x to avoid compatibility issues
"""

import os
import sys

# Force CPU usage to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("=" * 70)
print("LEAF DISEASE CLASSIFICATION - COMPATIBLE VERSION (NumPy 1.x)")
print("=" * 70)

# Check Python version
print(f"Python version: {sys.version}")
print()

# Check basic imports
try:
    import numpy as np

    print(f"✓ NumPy version: {np.__version__}")

    if np.__version__.startswith('2'):
        print("⚠ WARNING: NumPy 2.x detected. Some packages may not be compatible.")
        print("   Consider downgrading: pip install numpy==1.24.3")
except Exception as e:
    print(f"✗ Error importing NumPy: {e}")
    sys.exit(1)

# Try to import other critical packages
print("\nChecking package compatibility...")

import torch

print(f"✓ PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

print("✓ PyTorch modules imported successfully")

from PIL import Image

print("✓ PIL imported successfully")

print("\nAll required packages imported successfully!")
print("=" * 70)


# Now define the rest of the code
class CompatibleLeafDataset(Dataset):
    """Dataset compatible with NumPy 1.x"""

    def __init__(self, root_dir, mode='train', img_size=64, max_per_class=100):
        self.root_dir = os.path.join(root_dir, mode)
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Check if directory exists
        if not os.path.exists(self.root_dir):
            print(f"Error: Directory {self.root_dir} does not exist!")
            self.classes = []
            self.samples = []
            return

        # Get classes
        try:
            self.classes = sorted([d for d in os.listdir(self.root_dir)
                                   if os.path.isdir(os.path.join(self.root_dir, d))])
        except Exception as e:
            print(f"Error listing directories: {e}")
            self.classes = []
            self.samples = []
            return

        if not self.classes:
            print(f"No classes found in {self.root_dir}")
            return

        # Create class mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect samples
        self.samples = []
        self.binary_labels = []  # 0 = healthy, 1 = diseased

        print(f"\nCollecting images from {mode} set...")
        total_collected = 0

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"  Warning: Class directory not found: {class_dir}")
                continue

            # Get image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                if os.path.exists(class_dir):
                    try:
                        files = [f for f in os.listdir(class_dir) if f.lower().endswith(ext)]
                        image_files.extend(files)
                    except Exception as e:
                        print(f"  Error reading {class_dir}: {e}")

            # Limit number of samples per class
            image_files = image_files[:max_per_class]

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                self.samples.append((img_path, class_idx))

                # Determine binary label
                is_healthy = 'healthy' in class_name.lower()
                self.binary_labels.append(0 if is_healthy else 1)
                total_collected += 1

            if image_files:
                print(f"  {class_name}: collected {len(image_files)} images")

        print(f"\n✓ Total images collected for {mode}: {len(self.samples)}")
        print(f"✓ Number of classes: {len(self.classes)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, disease_label = self.samples[idx]
        binary_label = self.binary_labels[idx]

        try:
            # Load image
            img = Image.open(img_path).convert('RGB')

            # Apply transformations
            if self.transform:
                img = self.transform(img)

            return img, disease_label, binary_label

        except Exception as e:
            # Return a placeholder image if loading fails
            print(f"Error loading {img_path}: {e}")
            placeholder = torch.zeros((3, self.img_size, self.img_size))
            return placeholder, disease_label, binary_label


class SimpleLeafModel(nn.Module):
    """Simple model for leaf disease classification"""

    def __init__(self, num_classes, dropout=0.3):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Disease classifier (multi-class)
        self.disease_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # Healthy/diseased classifier (binary)
        self.binary_classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)

        disease_out = self.disease_classifier(features)
        binary_out = self.binary_classifier(features)

        return disease_out, binary_out


def train_and_validate():
    """Main training and validation function"""
    print("\n" + "=" * 70)
    print("TRAINING PHASE")
    print("=" * 70)

    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Data path
    data_path = "E:/庄楷文/叶片病虫害识别/baseline/data/reorganized_dataset_new"

    # Check path
    if not os.path.exists(data_path):
        print(f"\nError: Data path does not exist!")
        print(f"Please check: {data_path}")
        return None, None

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CompatibleLeafDataset(data_path, 'train', img_size=64, max_per_class=200)
    val_dataset = CompatibleLeafDataset(data_path, 'val', img_size=64, max_per_class=50)

    if len(train_dataset.samples) == 0 or len(train_dataset.classes) == 0:
        print("\nNo training data found!")
        return None, None

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    num_classes = len(train_dataset.classes)
    print(f"\nCreating model with {num_classes} classes...")
    model = SimpleLeafModel(num_classes=num_classes).to(device)

    # Loss functions and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    epochs = 10
    best_val_acc = 0.0

    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 70)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct_disease = 0
        train_correct_binary = 0
        train_total = 0

        for batch_idx, (images, disease_labels, binary_labels) in enumerate(train_loader):
            images = images.to(device)
            disease_labels = disease_labels.to(device)
            binary_labels = binary_labels.to(device)

            # Forward pass
            disease_out, binary_out = model(images)

            # Calculate losses
            loss_disease = criterion(disease_out, disease_labels)
            loss_binary = criterion(binary_out, binary_labels)
            loss = loss_disease + 0.5 * loss_binary

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()

            # Calculate accuracy
            _, disease_preds = torch.max(disease_out, 1)
            _, binary_preds = torch.max(binary_out, 1)

            train_correct_disease += (disease_preds == disease_labels).sum().item()
            train_correct_binary += (binary_preds == binary_labels).sum().item()
            train_total += disease_labels.size(0)

            # Print progress
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_correct_disease = 0
        val_correct_binary = 0
        val_total = 0

        with torch.no_grad():
            for images, disease_labels, binary_labels in val_loader:
                images = images.to(device)

                disease_out, binary_out = model(images)

                _, disease_preds = torch.max(disease_out, 1)
                _, binary_preds = torch.max(binary_out, 1)

                val_correct_disease += (disease_preds == disease_labels).sum().item()
                val_correct_binary += (binary_preds == binary_labels).sum().item()
                val_total += disease_labels.size(0)

        # Calculate metrics
        train_acc_disease = train_correct_disease / train_total
        train_acc_binary = train_correct_binary / train_total
        val_acc_disease = val_correct_disease / val_total
        val_acc_binary = val_correct_binary / val_total

        # Update learning rate
        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{epochs} Summary:")
        print(f"  Training - Disease Acc: {train_acc_disease:.4f}, Binary Acc: {train_acc_binary:.4f}")
        print(f"  Validation - Disease Acc: {val_acc_disease:.4f}, Binary Acc: {val_acc_binary:.4f}")

        # Save best model
        if val_acc_disease > best_val_acc:
            best_val_acc = val_acc_disease
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'class_names': train_dataset.classes
            }, 'compatible_best_model.pth')
            print(f"  ✓ Saved best model (Val Acc: {best_val_acc:.4f})")

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
    return model, train_dataset.classes


def evaluate_on_test(model, class_names):
    """Evaluate the model on test set"""
    print("\n" + "=" * 70)
    print("TESTING PHASE")
    print("=" * 70)

    # Set device
    device = torch.device('cpu')

    # Data path
    data_path = "E:/庄楷文/叶片病虫害识别/baseline/data/reorganized_dataset_new"

    # Create test dataset
    print("\nLoading test dataset...")
    test_dataset = CompatibleLeafDataset(data_path, 'test', img_size=64, max_per_class=100)

    if len(test_dataset.samples) == 0:
        print("No test data found!")
        return None

    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate
    model.eval()
    all_disease_preds = []
    all_disease_labels = []
    all_binary_preds = []
    all_binary_labels = []

    print("\nRunning predictions on test set...")
    with torch.no_grad():
        for batch_idx, (images, disease_labels, binary_labels) in enumerate(test_loader):
            images = images.to(device)

            disease_out, binary_out = model(images)

            _, disease_preds = torch.max(disease_out, 1)
            _, binary_preds = torch.max(binary_out, 1)

            all_disease_preds.extend(disease_preds.cpu().numpy())
            all_disease_labels.extend(disease_labels.numpy())
            all_binary_preds.extend(binary_preds.cpu().numpy())
            all_binary_labels.extend(binary_labels.numpy())

            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx}")

    # Convert to numpy arrays
    all_disease_preds = np.array(all_disease_preds)
    all_disease_labels = np.array(all_disease_labels)
    all_binary_preds = np.array(all_binary_preds)
    all_binary_labels = np.array(all_binary_labels)

    # Calculate metrics using numpy
    print("\nCalculating metrics...")

    # Binary classification metrics
    binary_correct = (all_binary_preds == all_binary_labels).sum()
    binary_total = len(all_binary_labels)
    binary_accuracy = binary_correct / binary_total

    # For precision/recall/f1, we'll implement simple versions
    def simple_precision_recall_f1(preds, labels):
        # For binary classification
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    # Disease classification metrics
    disease_correct = (all_disease_preds == all_disease_labels).sum()
    disease_total = len(all_disease_labels)
    disease_accuracy = disease_correct / disease_total

    # Calculate per-class accuracy
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_disease_labels == i
        if class_mask.sum() > 0:
            class_correct = (all_disease_preds[class_mask] == i).sum()
            class_total = class_mask.sum()
            class_accuracies[class_name] = class_correct / class_total

    # Display results
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(f"\n1. BINARY CLASSIFICATION (Healthy vs Diseased):")
    print(f"   Accuracy:  {binary_accuracy:.4f}")

    # Calculate binary precision/recall/f1
    binary_precision, binary_recall, binary_f1 = simple_precision_recall_f1(all_binary_preds, all_binary_labels)
    print(f"   Precision: {binary_precision:.4f}")
    print(f"   Recall:    {binary_recall:.4f}")
    print(f"   F1-Score:  {binary_f1:.4f}")

    print(f"\n2. MULTI-CLASS CLASSIFICATION (Specific Diseases):")
    print(f"   Accuracy:  {disease_accuracy:.4f}")
    print(f"   Total test samples: {disease_total}")

    # Show top 10 class accuracies
    print(f"\n3. TOP 10 CLASS ACCURACIES:")
    print("-" * 50)
    print(f"{'Class Name':<30} {'Accuracy':<10} {'Samples':<10}")
    print("-" * 50)

    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    for class_name, accuracy in sorted_classes[:10]:
        sample_count = (all_disease_labels == class_names.index(class_name)).sum()
        print(f"{class_name:<30} {accuracy:<10.4f} {sample_count:<10}")

    # Save results
    results = {
        'binary_accuracy': float(binary_accuracy),
        'binary_precision': float(binary_precision),
        'binary_recall': float(binary_recall),
        'binary_f1': float(binary_f1),
        'disease_accuracy': float(disease_accuracy),
        'class_accuracies': {k: float(v) for k, v in class_accuracies.items()},
        'class_names': class_names,
        'predictions': {
            'disease': all_disease_preds.tolist(),
            'binary': all_binary_preds.tolist()
        },
        'labels': {
            'disease': all_disease_labels.tolist(),
            'binary': all_binary_labels.tolist()
        }
    }

    # Save to file
    import json
    with open('compatible_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    np.save('compatible_test_results.npy', results)

    print(f"\nResults saved to: compatible_test_results.json and compatible_test_results.npy")

    return results


def main():
    """Main function"""
    try:
        # Train and validate
        model, class_names = train_and_validate()

        if model is None:
            print("\nTraining failed. Cannot proceed with testing.")
            return

        # Load best model if available
        if os.path.exists('compatible_best_model.pth'):
            print("\nLoading best model for testing...")
            checkpoint = torch.load('compatible_best_model.pth', map_location='cpu')

            # Recreate model
            model = SimpleLeafModel(num_classes=len(class_names))
            model.load_state_dict(checkpoint['model_state_dict'])
            class_names = checkpoint['class_names']
            print(f"Loaded model with validation accuracy: {checkpoint['val_acc']:.4f}")

        # Test
        results = evaluate_on_test(model, class_names)

        if results:
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"Binary Classification Accuracy:   {results['binary_accuracy']:.4f}")
            print(f"Multi-class Classification Accuracy: {results['disease_accuracy']:.4f}")
            print(f"Number of classes: {len(class_names)}")

        print("\n" + "=" * 70)
        print("PROGRAM COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()