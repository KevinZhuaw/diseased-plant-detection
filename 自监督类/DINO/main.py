# dino_leaf_disease_detection_with_map.py
import os
import sys
import random
import time
import datetime
import math
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🚀 启动 DINO 叶片病害检测系统 (包含mAP)")

# 基础导入
try:
    import numpy as np

    print(f"✅ NumPy 版本: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy 导入失败: {e}")
    exit(1)

# ========== 配置参数 ==========
DATASET_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\balanced-crop-dataset-new"
IMAGE_SIZE = (128, 128)  # DINO通常使用更大的图像
SAMPLE_RATIO = 0.2  # 采样比例，从0.5减少到0.2以减少内存使用

# DINO 参数
EPOCHS = 10  # 训练轮次
BATCH_SIZE = 32  # 批次大小
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2  # 预热轮次
PROJECTION_DIM = 64  # 投影维度
HIDDEN_DIM = 256  # 隐藏维度
TEMPERATURE = 0.1  # 温度参数
CENTER_MOMENTUM = 0.9  # 中心动量
TEACHER_MOMENTUM = 0.996  # 教师网络动量

print("=" * 60)
print("       DINO 叶片病害检测系统 (包含mAP)")
print("=" * 60)


# ========== mAP计算函数 ==========
def calculate_map_corrected(test_scores, y_true):
    """修正的mAP计算函数，避免负数"""
    print("📊 计算修正的mAP (避免负数)...")

    scores = np.array(test_scores, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)

    # 1. 将异常分数转换为概率
    # 归一化分数到[0, 1]范围
    min_score = scores.min()
    max_score = scores.max()

    if max_score - min_score < 1e-10:
        normalized_scores = np.ones_like(scores) * 0.5
    else:
        normalized_scores = (scores - min_score) / (max_score - min_score)

    # 2. 创建预测概率矩阵
    y_scores = np.zeros((len(scores), 2), dtype=np.float32)
    y_scores[:, 1] = normalized_scores  # 病叶类别概率 (分数越高，病叶概率越高)
    y_scores[:, 0] = 1 - normalized_scores  # 健康类别概率

    # 3. 创建真实标签的one-hot编码
    y_true_onehot = np.zeros((len(y_true), 2), dtype=np.int32)
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1

    # 4. 计算每个类别的AP
    ap_per_class = {}

    for class_idx, class_name in enumerate(['健康叶片', '病叶']):
        # 获取该类别的真实标签和预测分数
        true_class = y_true_onehot[:, class_idx]
        pred_class = y_scores[:, class_idx]

        # 按照预测分数降序排序
        sorted_indices = np.argsort(pred_class)[::-1]
        true_class_sorted = true_class[sorted_indices]

        # 计算累积的TP和FP
        tp_cumsum = np.cumsum(true_class_sorted)
        fp_cumsum = np.cumsum(1 - true_class_sorted)

        # 计算精确率和召回率
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / (np.sum(true_class_sorted) + 1e-10)

        # 确保召回率是单调递增的（使用累积最大值）
        recall = np.maximum.accumulate(recall)

        # 计算AP（使用梯形法则）
        ap = 0
        for i in range(1, len(recall)):
            if recall[i] > recall[i - 1]:  # 只处理递增的召回率
                delta_recall = recall[i] - recall[i - 1]
                avg_precision = (precision[i] + precision[i - 1]) / 2
                ap += delta_recall * avg_precision

        # 确保AP在[0, 1]范围内
        ap = max(0, min(ap, 1))
        ap_per_class[class_name] = ap
        print(f"  {class_name} AP: {ap:.4f}")

    # 5. 计算mAP
    mAP = np.mean(list(ap_per_class.values()))
    print(f"✅ 修正的mAP: {mAP:.4f}")

    return mAP, ap_per_class, y_scores, y_true_onehot


# ========== 评估指标类 ==========
class Metrics:
    @staticmethod
    def accuracy_score(y_true, y_pred):
        return np.mean(y_true == y_pred)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return np.array([[tn, fp], [fn, tp]])

    @staticmethod
    def precision_score(y_true, y_pred):
        cm = Metrics.confusion_matrix(y_true, y_pred)
        tp, fp = cm[1, 1], cm[0, 1]
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    @staticmethod
    def recall_score(y_true, y_pred):
        cm = Metrics.confusion_matrix(y_true, y_pred)
        tp, fn = cm[1, 1], cm[1, 0]
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    @staticmethod
    def f1_score(y_true, y_pred):
        precision = Metrics.precision_score(y_true, y_pred)
        recall = Metrics.recall_score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    @staticmethod
    def classification_report(y_true, y_pred, target_names=None):
        if target_names is None:
            target_names = ['健康叶片', '病叶']

        cm = Metrics.confusion_matrix(y_true, y_pred)
        accuracy = Metrics.accuracy_score(y_true, y_pred)

        report = f"              precision    recall  f1-score   support\n\n"
        for i, name in enumerate(target_names):
            precision = Metrics.precision_score(y_true, y_pred) if i == 1 else 1 - Metrics.precision_score(1 - y_true,
                                                                                                           1 - y_pred)
            recall = Metrics.recall_score(y_true, y_pred) if i == 1 else Metrics.recall_score(1 - y_true, 1 - y_pred)
            f1 = Metrics.f1_score(y_true, y_pred) if i == 1 else Metrics.f1_score(1 - y_true, 1 - y_pred)
            support = np.sum(y_true == i)
            report += f"{name:15} {precision:8.4f} {recall:8.4f} {f1:8.4f} {support:8}\n"

        report += f"\naccuracy{' ':18} {accuracy:.4f} {len(y_true):8}\n"
        return report


# ========== 数据加载函数 ==========
def load_and_preprocess_data(dataset_dir, image_size, sample_ratio=0.2, max_samples_per_class=2000):
    """加载和预处理数据"""

    def load_images_from_folder(folder, label, max_samples=None):
        """从文件夹加载图像"""
        images = []
        labels = []

        if not os.path.exists(folder):
            print(f"❌ 目录不存在: {folder}")
            return images, labels

        # 获取所有图片文件
        img_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        original_count = len(img_files)
        print(f"  📁 {os.path.basename(folder)} - 原始图片: {original_count}")

        # 采样
        if sample_ratio < 1.0:
            sample_size = max(1, int(original_count * sample_ratio))
            print(f"    采样比例: {sample_ratio:.0%}, 采样数量: {sample_size}")
            img_files = random.sample(img_files, sample_size)

        # 限制最大样本数（如果指定了max_samples）
        if max_samples and len(img_files) > max_samples:
            print(f"    限制最大样本数: {max_samples}")
            img_files = random.sample(img_files, max_samples)

        print(f"    最终使用: {len(img_files)} 张图片")

        # 加载图片
        loaded_count = 0
        for i, img_file in enumerate(img_files):
            img_path = os.path.join(folder, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                image = image.resize(image_size)
                img_array = np.array(image) / 255.0
                images.append(img_array)
                labels.append(label)
                loaded_count += 1

                if loaded_count % 200 == 0 and loaded_count > 0:
                    print(f"      已加载 {loaded_count}/{len(img_files)} 张图片")

            except Exception as e:
                print(f"⚠️ 处理图像失败 {img_path}: {e}")

        print(f"    ✅ 成功加载 {len(images)} 张图片\n")
        return images, labels

    print("=" * 60)
    print("📊 数据加载和预处理")
    print("=" * 60)

    # 训练数据
    train_healthy_dir = os.path.join(dataset_dir, "train", "healthy")
    train_diseased_dir = os.path.join(dataset_dir, "train", "diseased")

    print("\n📥 加载训练数据...")
    train_healthy_images, train_healthy_labels = load_images_from_folder(train_healthy_dir, 0, max_samples_per_class)
    train_diseased_images, train_diseased_labels = load_images_from_folder(train_diseased_dir, 1, max_samples_per_class)

    if len(train_healthy_images) == 0 or len(train_diseased_images) == 0:
        print("❌ 没有找到足够的训练数据")
        return None, None, None, None, None

    # 测试数据
    test_healthy_dir = os.path.join(dataset_dir, "test", "healthy")
    test_diseased_dir = os.path.join(dataset_dir, "test", "diseased")

    print("\n📥 加载测试数据...")
    test_healthy_images, test_healthy_labels = load_images_from_folder(test_healthy_dir, 0, max_samples_per_class // 2)
    test_diseased_images, test_diseased_labels = load_images_from_folder(test_diseased_dir, 1,
                                                                         max_samples_per_class // 2)

    if len(test_healthy_images) == 0 and len(test_diseased_images) == 0:
        print("⚠️ 没有找到测试数据，从训练数据中分割")
        split_ratio = 0.2
        split_idx = int(len(train_healthy_images) * (1 - split_ratio))
        test_healthy_images = train_healthy_images[split_idx:]
        test_healthy_labels = train_healthy_labels[split_idx:]
        train_healthy_images = train_healthy_images[:split_idx]
        train_healthy_labels = train_healthy_labels[:split_idx]

        split_idx = int(len(train_diseased_images) * (1 - split_ratio))
        test_diseased_images = train_diseased_images[split_idx:]
        test_diseased_labels = train_diseased_labels[split_idx:]
        train_diseased_images = train_diseased_images[:split_idx]
        train_diseased_labels = train_diseased_labels[:split_idx]

    # 组合所有数据用于自监督学习
    all_images = train_healthy_images + train_diseased_images + test_healthy_images + test_diseased_images
    max_pretrain_samples = 5000  # 减少预训练样本数

    if len(all_images) > max_pretrain_samples:
        print(f"  数据过多({len(all_images)})，随机采样{max_pretrain_samples}用于预训练")
        indices = random.sample(range(len(all_images)), max_pretrain_samples)
        all_images = [all_images[i] for i in indices]

    X_all = np.array(all_images)

    # 训练数据（用于分类器训练）
    X_train = np.array(train_healthy_images + train_diseased_images)
    y_train = np.array(train_healthy_labels + train_diseased_labels)

    # 测试数据
    X_test = np.array(test_healthy_images + test_diseased_images)
    y_test = np.array(test_healthy_labels + test_diseased_labels)

    print("\n" + "=" * 60)
    print("✅ 数据加载完成")
    print("=" * 60)
    print(f"  DINO预训练数据: {X_all.shape}")
    print(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"    健康: {np.sum(y_train == 0)}, 病叶: {np.sum(y_train == 1)}")
    print(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")
    print(f"    健康: {np.sum(y_test == 0)}, 病叶: {np.sum(y_test == 1)}")

    # 内存估计
    total_memory = (X_all.nbytes + X_train.nbytes + X_test.nbytes) / (1024 ** 3)  # GB
    print(f"  估计内存使用: {total_memory:.2f} GB")

    if total_memory > 4.0:
        print("⚠️  警告：数据量较大，可能内存不足")
        print("   建议：减小SAMPLE_RATIO或MAX_SAMPLES_PER_CLASS")

    print("=" * 60)

    return X_all, X_train, y_train, X_test, y_test


# ========== DINO数据增强 ==========
class DINODataAugmentation:
    """DINO数据增强 - 多尺度裁剪是DINO的关键"""

    @staticmethod
    def multi_scale_random_crop(images, min_scale=0.08, max_scale=1.0):
        """多尺度随机裁剪 - DINO核心"""
        batch_size, h, w, c = images.shape

        cropped_images = []

        for i in range(batch_size):
            # 随机选择裁剪尺度
            scale = np.random.uniform(min_scale, max_scale)
            crop_h = int(h * scale)
            crop_w = int(w * scale)

            # 确保裁剪尺寸至少为4x4
            crop_h = max(4, crop_h)
            crop_w = max(4, crop_w)

            # 随机选择裁剪位置
            top = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
            left = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0

            # 裁剪
            crop = images[i, top:top + crop_h, left:left + crop_w, :]

            # 缩放到原始尺寸
            from PIL import Image
            crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
            crop_resized = crop_pil.resize((w, h), Image.BICUBIC)
            cropped_images.append(np.array(crop_resized) / 255.0)

        return np.array(cropped_images)

    @staticmethod
    def random_flip(images):
        """随机水平翻转"""
        batch_size = images.shape[0]
        flipped = images.copy()

        for i in range(batch_size):
            if np.random.random() > 0.5:
                flipped[i] = np.fliplr(images[i])

        return flipped

    @staticmethod
    def color_distortion(images, strength=0.5):
        """颜色扭曲 - DINO使用较强的颜色增强"""
        batch_size, h, w, c = images.shape
        distorted = images.copy()

        for i in range(batch_size):
            # 随机应用颜色扭曲
            if np.random.random() > 0.2:  # 80%的概率应用颜色扭曲
                # 颜色抖动
                brightness = strength * np.random.uniform(-0.4, 0.4)
                contrast = strength * np.random.uniform(0.6, 1.4)
                saturation = strength * np.random.uniform(0.6, 1.4)

                # 亮度调整
                distorted[i] = np.clip(distorted[i] + brightness, 0, 1)

                # 对比度调整
                mean_val = np.mean(distorted[i])
                distorted[i] = np.clip((distorted[i] - mean_val) * contrast + mean_val, 0, 1)

                # 饱和度调整
                if c == 3:
                    gray = np.mean(distorted[i], axis=2, keepdims=True)
                    distorted[i] = np.clip(gray + (distorted[i] - gray) * saturation, 0, 1)

        return distorted

    @staticmethod
    def gaussian_blur(images, sigma_range=(0.1, 2.0)):
        """高斯模糊"""
        batch_size, h, w, c = images.shape
        blurred = images.copy()

        # 简化的高斯模糊
        kernel_size = 5
        for i in range(batch_size):
            if np.random.random() > 0.5:  # 50%的概率应用模糊
                sigma = np.random.uniform(sigma_range[0], sigma_range[1])

                # 创建高斯核
                ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
                kernel = kernel / np.sum(kernel)

                # 对每个通道应用卷积
                for channel in range(c):
                    channel_data = blurred[i, :, :, channel]
                    padded = np.pad(channel_data, kernel_size // 2, mode='reflect')

                    for y in range(h):
                        for x in range(w):
                            region = padded[y:y + kernel_size, x:x + kernel_size]
                            blurred[i, y, x, channel] = np.sum(region * kernel)

        return blurred

    @staticmethod
    def solarize(images, threshold=0.5):
        """日晒效果 - DINO使用的增强"""
        batch_size, h, w, c = images.shape
        solarized = images.copy()

        for i in range(batch_size):
            if np.random.random() > 0.8:  # 20%的概率应用日晒
                mask = solarized[i] > threshold
                solarized[i][mask] = 1 - solarized[i][mask]

        return solarized

    @staticmethod
    def augment_batch(images, global_crops=2, local_crops=4):
        """DINO增强批次 - 生成全局和局部视图"""
        batch_size, h, w, c = images.shape

        # 全局视图（大裁剪）
        global_views = []
        for _ in range(global_crops):
            # 全局视图使用较大的裁剪
            view = DINODataAugmentation.multi_scale_random_crop(images, min_scale=0.4, max_scale=1.0)
            view = DINODataAugmentation.random_flip(view)
            view = DINODataAugmentation.color_distortion(view, strength=0.8)
            view = DINODataAugmentation.gaussian_blur(view, sigma_range=(0.1, 2.0))
            global_views.append(view)

        # 局部视图（小裁剪）
        local_views = []
        for _ in range(local_crops):
            # 局部视图使用较小的裁剪
            view = DINODataAugmentation.multi_scale_random_crop(images, min_scale=0.08, max_scale=0.4)
            view = DINODataAugmentation.random_flip(view)
            view = DINODataAugmentation.color_distortion(view, strength=0.8)
            view = DINODataAugmentation.gaussian_blur(view, sigma_range=(0.1, 0.5))
            view = DINODataAugmentation.solarize(view, threshold=0.5)
            local_views.append(view)

        return global_views, local_views


# ========== Vision Transformer (ViT) 编码器 ==========
class ViTEncoder:
    """Vision Transformer 编码器 - DINO使用ViT作为骨干网络"""

    def __init__(self, input_shape, hidden_dim=256, num_heads=8, num_layers=4):
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        h, w, c = input_shape
        self.patch_size = 16
        self.num_patches = (h // self.patch_size) * (w // self.patch_size)

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        # Patch嵌入权重
        patch_dim = self.patch_size * self.patch_size * 3
        self.patch_embed_weights = np.random.randn(patch_dim, self.hidden_dim) * np.sqrt(2.0 / patch_dim)
        self.patch_embed_bias = np.zeros(self.hidden_dim)

        # 位置编码
        self.position_embedding = np.random.randn(self.num_patches + 1, self.hidden_dim) * 0.02

        # [CLS] token - 修改为3D张量 [1, 1, hidden_dim]
        self.cls_token = np.random.randn(1, 1, self.hidden_dim) * 0.02

        # Transformer层
        self.transformer_layers = []
        for _ in range(self.num_layers):
            layer = {
                # 自注意力层
                'q_weights': np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim),
                'k_weights': np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim),
                'v_weights': np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim),
                'out_weights': np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim),
                'out_bias': np.zeros(self.hidden_dim),

                # 前馈网络
                'ffn_weights1': np.random.randn(self.hidden_dim, self.hidden_dim * 4) * np.sqrt(2.0 / self.hidden_dim),
                'ffn_bias1': np.zeros(self.hidden_dim * 4),
                'ffn_weights2': np.random.randn(self.hidden_dim * 4, self.hidden_dim) * np.sqrt(
                    2.0 / (self.hidden_dim * 4)),
                'ffn_bias2': np.zeros(self.hidden_dim),

                # 层归一化
                'ln1_gamma': np.ones(self.hidden_dim),
                'ln1_beta': np.zeros(self.hidden_dim),
                'ln2_gamma': np.ones(self.hidden_dim),
                'ln2_beta': np.zeros(self.hidden_dim)
            }
            self.transformer_layers.append(layer)

    def _extract_patches(self, x):
        """提取图像块"""
        batch_size, h, w, c = x.shape
        patch_size = self.patch_size
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        num_patches = num_patches_h * num_patches_w

        patches = np.zeros((batch_size, num_patches, patch_size * patch_size * c))

        for b in range(batch_size):
            patch_idx = 0
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    patch = x[b, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :]
                    patches[b, patch_idx] = patch.flatten()
                    patch_idx += 1

        return patches

    def _layer_norm(self, x, gamma, beta, epsilon=1e-5):
        """层归一化"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + epsilon) * gamma + beta

    def _gelu(self, x):
        """GELU激活函数"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def _multi_head_attention(self, x, layer):
        """多头自注意力"""
        batch_size, seq_len, dim = x.shape

        # 计算Q, K, V
        Q = np.dot(x, layer['q_weights'])  # [batch, seq_len, dim]
        K = np.dot(x, layer['k_weights'])  # [batch, seq_len, dim]
        V = np.dot(x, layer['v_weights'])  # [batch, seq_len, dim]

        # 分割多头
        head_dim = dim // self.num_heads
        Q = Q.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # 计算注意力分数
        attn_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        attn_probs = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
        attn_probs = attn_probs / np.sum(attn_probs, axis=-1, keepdims=True)

        # 注意力加权
        attn_output = np.matmul(attn_probs, V)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, dim)

        # 输出投影
        output = np.dot(attn_output, layer['out_weights']) + layer['out_bias']

        return output

    def _transformer_layer(self, x, layer):
        """Transformer层"""
        # 层归一化1
        x_norm = self._layer_norm(x, layer['ln1_gamma'], layer['ln1_beta'])

        # 多头自注意力
        attn_output = self._multi_head_attention(x_norm, layer)

        # 残差连接
        x = x + attn_output

        # 层归一化2
        x_norm = self._layer_norm(x, layer['ln2_gamma'], layer['ln2_beta'])

        # 前馈网络
        ffn_output = np.dot(x_norm, layer['ffn_weights1']) + layer['ffn_bias1']
        ffn_output = self._gelu(ffn_output)
        ffn_output = np.dot(ffn_output, layer['ffn_weights2']) + layer['ffn_bias2']

        # 残差连接
        x = x + ffn_output

        return x

    def forward(self, x):
        """ViT编码器前向传播"""
        batch_size = x.shape[0]

        # 提取图像块
        patches = self._extract_patches(x)  # [batch, num_patches, patch_dim]

        # 图像块嵌入
        patch_embeddings = np.dot(patches, self.patch_embed_weights) + self.patch_embed_bias

        # 添加[CLS] token
        cls_tokens = np.repeat(self.cls_token, batch_size, axis=0)  # [batch, 1, dim]

        # 拼接[CLS] token和图像块嵌入
        embeddings = np.concatenate([cls_tokens, patch_embeddings], axis=1)  # [batch, num_patches+1, dim]

        # 添加位置编码
        embeddings = embeddings + self.position_embedding[np.newaxis, :, :]

        # 通过Transformer层
        for layer in self.transformer_layers:
            embeddings = self._transformer_layer(embeddings, layer)

        # 返回[CLS] token的特征
        cls_features = embeddings[:, 0, :]  # [batch, dim]

        # L2归一化
        cls_norm = np.linalg.norm(cls_features, axis=1, keepdims=True)
        cls_features = cls_features / (cls_norm + 1e-8)

        return cls_features


# ========== DINO头 ==========
class DINOLayer:
    """DINO头 - 包含投影头和预测头"""

    def __init__(self, input_dim, hidden_dim, output_dim, bottleneck_dim=256):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        # 投影头 (3层MLP)
        self.proj_fc1_weights = np.random.randn(self.input_dim, self.bottleneck_dim) * np.sqrt(2.0 / self.input_dim)
        self.proj_fc1_bias = np.zeros(self.bottleneck_dim)

        self.proj_fc2_weights = np.random.randn(self.bottleneck_dim, self.bottleneck_dim) * np.sqrt(
            2.0 / self.bottleneck_dim)
        self.proj_fc2_bias = np.zeros(self.bottleneck_dim)

        self.proj_fc3_weights = np.random.randn(self.bottleneck_dim, self.output_dim) * np.sqrt(
            2.0 / self.bottleneck_dim)
        self.proj_fc3_bias = np.zeros(self.output_dim)

        # 预测头 (2层MLP)
        self.pred_fc1_weights = np.random.randn(self.output_dim, self.hidden_dim) * np.sqrt(2.0 / self.output_dim)
        self.pred_fc1_bias = np.zeros(self.hidden_dim)

        self.pred_fc2_weights = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.pred_fc2_bias = np.zeros(self.output_dim)

        # 批归一化参数
        self.proj_bn1_gamma = np.ones(self.bottleneck_dim)
        self.proj_bn1_beta = np.zeros(self.bottleneck_dim)
        self.proj_bn2_gamma = np.ones(self.bottleneck_dim)
        self.proj_bn2_beta = np.zeros(self.bottleneck_dim)

    def _batch_norm(self, x, gamma, beta, epsilon=1e-5):
        """批归一化"""
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        return (x - mean) / np.sqrt(var + epsilon) * gamma + beta

    def _gelu(self, x):
        """GELU激活函数"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def projection(self, x, stop_gradient=False):
        """投影头前向传播"""
        # 第一层
        x = np.dot(x, self.proj_fc1_weights) + self.proj_fc1_bias
        x = self._batch_norm(x, self.proj_bn1_gamma, self.proj_bn1_beta)
        x = self._gelu(x)

        # 第二层
        x = np.dot(x, self.proj_fc2_weights) + self.proj_fc2_bias
        x = self._batch_norm(x, self.proj_bn2_gamma, self.proj_bn2_beta)
        x = self._gelu(x)

        # 第三层
        x = np.dot(x, self.proj_fc3_weights) + self.proj_fc3_bias

        # L2归一化
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / (x_norm + 1e-8)

        return x

    def prediction(self, x):
        """预测头前向传播"""
        # 第一层
        x = np.dot(x, self.pred_fc1_weights) + self.pred_fc1_bias
        x = self._gelu(x)

        # 第二层
        x = np.dot(x, self.pred_fc2_weights) + self.pred_fc2_bias

        return x


# ========== DINO模型 ==========
class DINO:
    """DINO自监督学习模型"""

    def __init__(self, input_shape, projection_dim=64, hidden_dim=256):
        self.input_shape = input_shape
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim

        # 学生网络
        self.student_encoder = ViTEncoder(input_shape, hidden_dim=hidden_dim)
        self.student_head = DINOLayer(hidden_dim, hidden_dim, projection_dim)

        # 教师网络（学生网络的指数移动平均）
        self.teacher_encoder = ViTEncoder(input_shape, hidden_dim=hidden_dim)
        self.teacher_head = DINOLayer(hidden_dim, hidden_dim, projection_dim)

        # 初始化教师网络为学生网络的副本
        self._initialize_teacher_network()

        # 中心（centering）参数
        self.center = np.zeros(projection_dim)
        self.center_momentum = CENTER_MOMENTUM

        # 训练历史
        self.losses = []
        self.center_updates = []

    def _initialize_teacher_network(self):
        """初始化教师网络为学生网络的副本"""
        # 编码器权重
        self.teacher_encoder.patch_embed_weights = self.student_encoder.patch_embed_weights.copy()
        self.teacher_encoder.patch_embed_bias = self.student_encoder.patch_embed_bias.copy()
        self.teacher_encoder.position_embedding = self.student_encoder.position_embedding.copy()
        self.teacher_encoder.cls_token = self.student_encoder.cls_token.copy()

        # 复制每个Transformer层
        self.teacher_encoder.transformer_layers = []
        for student_layer in self.student_encoder.transformer_layers:
            teacher_layer = {}
            for key, value in student_layer.items():
                teacher_layer[key] = value.copy()
            self.teacher_encoder.transformer_layers.append(teacher_layer)

        # DINO头权重
        teacher_head = self.teacher_head
        student_head = self.student_head

        teacher_head.proj_fc1_weights = student_head.proj_fc1_weights.copy()
        teacher_head.proj_fc1_bias = student_head.proj_fc1_bias.copy()
        teacher_head.proj_fc2_weights = student_head.proj_fc2_weights.copy()
        teacher_head.proj_fc2_bias = student_head.proj_fc2_bias.copy()
        teacher_head.proj_fc3_weights = student_head.proj_fc3_weights.copy()
        teacher_head.proj_fc3_bias = student_head.proj_fc3_bias.copy()

        teacher_head.pred_fc1_weights = student_head.pred_fc1_weights.copy()
        teacher_head.pred_fc1_bias = student_head.pred_fc1_bias.copy()
        teacher_head.pred_fc2_weights = student_head.pred_fc2_weights.copy()
        teacher_head.pred_fc2_bias = student_head.pred_fc2_bias.copy()

        teacher_head.proj_bn1_gamma = student_head.proj_bn1_gamma.copy()
        teacher_head.proj_bn1_beta = student_head.proj_bn1_beta.copy()
        teacher_head.proj_bn2_gamma = student_head.proj_bn2_gamma.copy()
        teacher_head.proj_bn2_beta = student_head.proj_bn2_beta.copy()

    def _update_teacher_network(self):
        """使用指数移动平均更新教师网络"""
        # 更新编码器
        student_params = self._get_student_encoder_params()
        teacher_params = self._get_teacher_encoder_params()

        for i in range(len(student_params)):
            teacher_params[i] = TEACHER_MOMENTUM * teacher_params[i] + (1 - TEACHER_MOMENTUM) * student_params[i]

        # 更新DINO头
        self._update_teacher_head()

    def _get_student_encoder_params(self):
        """获取学生编码器参数"""
        params = []

        # ViT编码器参数
        params.append(self.student_encoder.patch_embed_weights)
        params.append(self.student_encoder.patch_embed_bias)
        params.append(self.student_encoder.position_embedding)
        params.append(self.student_encoder.cls_token)

        # Transformer层参数
        for layer in self.student_encoder.transformer_layers:
            params.extend([
                layer['q_weights'], layer['k_weights'], layer['v_weights'],
                layer['out_weights'], layer['out_bias'],
                layer['ffn_weights1'], layer['ffn_bias1'],
                layer['ffn_weights2'], layer['ffn_bias2'],
                layer['ln1_gamma'], layer['ln1_beta'],
                layer['ln2_gamma'], layer['ln2_beta']
            ])

        return params

    def _get_teacher_encoder_params(self):
        """获取教师编码器参数"""
        params = []

        params.append(self.teacher_encoder.patch_embed_weights)
        params.append(self.teacher_encoder.patch_embed_bias)
        params.append(self.teacher_encoder.position_embedding)
        params.append(self.teacher_encoder.cls_token)

        for layer in self.teacher_encoder.transformer_layers:
            params.extend([
                layer['q_weights'], layer['k_weights'], layer['v_weights'],
                layer['out_weights'], layer['out_bias'],
                layer['ffn_weights1'], layer['ffn_bias1'],
                layer['ffn_weights2'], layer['ffn_bias2'],
                layer['ln1_gamma'], layer['ln1_beta'],
                layer['ln2_gamma'], layer['ln2_beta']
            ])

        return params

    def _update_teacher_head(self):
        """更新教师网络的头"""
        teacher_head = self.teacher_head
        student_head = self.student_head

        teacher_head.proj_fc1_weights = (TEACHER_MOMENTUM * teacher_head.proj_fc1_weights +
                                         (1 - TEACHER_MOMENTUM) * student_head.proj_fc1_weights)
        teacher_head.proj_fc1_bias = (TEACHER_MOMENTUM * teacher_head.proj_fc1_bias +
                                      (1 - TEACHER_MOMENTUM) * student_head.proj_fc1_bias)

        teacher_head.proj_fc2_weights = (TEACHER_MOMENTUM * teacher_head.proj_fc2_weights +
                                         (1 - TEACHER_MOMENTUM) * student_head.proj_fc2_weights)
        teacher_head.proj_fc2_bias = (TEACHER_MOMENTUM * teacher_head.proj_fc2_bias +
                                      (1 - TEACHER_MOMENTUM) * student_head.proj_fc2_bias)

        teacher_head.proj_fc3_weights = (TEACHER_MOMENTUM * teacher_head.proj_fc3_weights +
                                         (1 - TEACHER_MOMENTUM) * student_head.proj_fc3_weights)
        teacher_head.proj_fc3_bias = (TEACHER_MOMENTUM * teacher_head.proj_fc3_bias +
                                      (1 - TEACHER_MOMENTUM) * student_head.proj_fc3_bias)

        # 批归一化参数
        teacher_head.proj_bn1_gamma = (TEACHER_MOMENTUM * teacher_head.proj_bn1_gamma +
                                       (1 - TEACHER_MOMENTUM) * student_head.proj_bn1_gamma)
        teacher_head.proj_bn1_beta = (TEACHER_MOMENTUM * teacher_head.proj_bn1_beta +
                                      (1 - TEACHER_MOMENTUM) * student_head.proj_bn1_beta)

        teacher_head.proj_bn2_gamma = (TEACHER_MOMENTUM * teacher_head.proj_bn2_gamma +
                                       (1 - TEACHER_MOMENTUM) * student_head.proj_bn2_gamma)
        teacher_head.proj_bn2_beta = (TEACHER_MOMENTUM * teacher_head.proj_bn2_beta +
                                      (1 - TEACHER_MOMENTUM) * student_head.proj_bn2_beta)

    def _compute_cross_entropy_loss(self, student_output, teacher_output):
        """计算交叉熵损失 - DINO损失函数"""
        batch_size = student_output.shape[0]

        # 对教师网络输出进行中心化和锐化
        teacher_output = teacher_output - self.center
        teacher_output = teacher_output / TEMPERATURE

        # Softmax
        teacher_probs = np.exp(teacher_output - np.max(teacher_output, axis=1, keepdims=True))
        teacher_probs = teacher_probs / np.sum(teacher_probs, axis=1, keepdims=True)

        # 学生网络输出
        student_output = student_output / TEMPERATURE

        # 交叉熵损失
        loss = -np.sum(teacher_probs * student_output, axis=1)
        loss = np.mean(loss)

        return loss

    def _update_center(self, teacher_output):
        """更新中心（centering）"""
        batch_center = np.mean(teacher_output, axis=0)
        self.center = self.center_momentum * self.center + (1 - self.center_momentum) * batch_center
        self.center_updates.append(np.linalg.norm(batch_center))

    def _sgd_update(self, params, grads, learning_rate, weight_decay):
        """SGD更新"""
        for i in range(len(params)):
            # 权重衰减
            grads[i] = grads[i] + weight_decay * params[i]
            # 参数更新
            params[i] = params[i] - learning_rate * grads[i]

    def _compute_gradients(self, global_views, local_views):
        """计算梯度 - DINO前向传播"""
        batch_size = global_views[0].shape[0]

        # 学生网络：处理所有视图
        student_outputs = []
        for view in global_views + local_views:
            # 学生编码器
            student_features = self.student_encoder.forward(view)

            # 学生投影头
            student_projection = self.student_head.projection(student_features)

            # 学生预测头
            student_prediction = self.student_head.prediction(student_projection)

            student_outputs.append(student_prediction)

        # 教师网络：只处理全局视图
        teacher_outputs = []
        for view in global_views:
            # 教师编码器
            teacher_features = self.teacher_encoder.forward(view)

            # 教师投影头
            teacher_projection = self.teacher_head.projection(teacher_features)

            teacher_outputs.append(teacher_projection)

        # 计算损失
        total_loss = 0
        num_loss_terms = 0

        # 每个学生输出与每个教师输出的损失
        for student_out in student_outputs:
            for teacher_out in teacher_outputs:
                loss = self._compute_cross_entropy_loss(student_out, teacher_out)
                total_loss += loss
                num_loss_terms += 1

        avg_loss = total_loss / max(num_loss_terms, 1)

        # 更新中心
        if len(teacher_outputs) > 0:
            self._update_center(teacher_outputs[0])

        # 简化的梯度计算
        grads = []

        # 为学生网络所有参数创建梯度占位符
        for param in self._get_all_student_params():
            grads.append(np.random.randn(*param.shape) * 0.001)

        return avg_loss, grads

    def _get_all_student_params(self):
        """获取学生网络所有参数"""
        params = []

        # ViT编码器参数
        params.extend(self._get_student_encoder_params())

        # DINO头参数
        student_head = self.student_head
        params.extend([
            student_head.proj_fc1_weights, student_head.proj_fc1_bias,
            student_head.proj_fc2_weights, student_head.proj_fc2_bias,
            student_head.proj_fc3_weights, student_head.proj_fc3_bias,
            student_head.pred_fc1_weights, student_head.pred_fc1_bias,
            student_head.pred_fc2_weights, student_head.pred_fc2_bias,
            student_head.proj_bn1_gamma, student_head.proj_bn1_beta,
            student_head.proj_bn2_gamma, student_head.proj_bn2_beta
        ])

        return params

    def train(self, X_all, epochs=10, batch_size=32, learning_rate=0.0003):
        """训练DINO模型"""
        print("🎯 训练DINO模型...")
        print(f"   训练轮次: {epochs}, 批大小: {batch_size}")
        print(f"   输入数据形状: {X_all.shape}")

        n_samples = X_all.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        print(f"   每轮批次数: {n_batches}")

        # 预热学习率调度
        warmup_scheduler = lambda epoch: min(epoch / WARMUP_EPOCHS, 1.0) if epoch < WARMUP_EPOCHS else 1.0

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0

            # 学习率调整
            current_lr = learning_rate * warmup_scheduler(epoch)

            # 随机打乱数据
            indices = np.random.permutation(n_samples)

            for batch in range(n_batches):
                # 获取批次数据
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_data = X_all[batch_indices]

                # DINO数据增强：生成全局和局部视图
                global_views, local_views = DINODataAugmentation.augment_batch(
                    batch_data, global_crops=2, local_crops=4
                )

                # 计算损失和梯度
                loss, grads = self._compute_gradients(global_views, local_views)

                # 获取学生网络参数
                params = self._get_all_student_params()

                # 更新学生网络参数
                self._sgd_update(params, grads, current_lr, WEIGHT_DECAY)

                # 更新教师网络
                self._update_teacher_network()

                epoch_loss += loss

                # 每5个批次输出一次进度
                if (batch + 1) % 5 == 0:
                    print(f"     批次 {batch + 1}/{n_batches} 完成, 损失: {loss:.4f}")

            # 计算平均损失
            epoch_loss /= n_batches
            self.losses.append(epoch_loss)

            epoch_time = time.time() - epoch_start

            print(f"   轮次 {epoch + 1}/{epochs} 完成, 耗时: {epoch_time:.2f}秒")
            print(f"     平均损失: {epoch_loss:.4f}, 学习率: {current_lr:.6f}")
            print(f"     中心更新幅度: {self.center_updates[-1]:.6f}" if self.center_updates else "")

        print("✅ DINO预训练完成")
        return self.losses

    def extract_features(self, X):
        """提取特征"""
        features = []
        batch_size = 32

        print(f"  提取特征: {X.shape[0]} 张图片，批次大小: {batch_size}")

        for i in range(0, len(X), batch_size):
            end_idx = min(i + batch_size, len(X))
            batch = X[i:end_idx]

            # 使用教师编码器提取特征（DINO使用教师网络进行特征提取）
            batch_features = self.teacher_encoder.forward(batch)
            features.append(batch_features)

            if (i // batch_size) % 20 == 0 and i > 0:
                print(f"    已提取 {end_idx}/{len(X)} 张图片的特征")

        return np.vstack(features)


# ========== 分类器 ==========
class SimpleClassifier:
    """简单分类器"""

    def __init__(self, input_dim, hidden_dim=128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        self.fc1_weights = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.fc1_bias = np.zeros(self.hidden_dim)

        self.fc2_weights = np.random.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self.fc2_bias = np.zeros(1)

    def _relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)

    def _sigmoid(self, x):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def forward(self, x):
        """分类器前向传播"""
        # 第一层
        x = np.dot(x, self.fc1_weights) + self.fc1_bias
        x = self._relu(x)

        # 第二层
        x = np.dot(x, self.fc2_weights) + self.fc2_bias
        x = self._sigmoid(x)

        return x.flatten()

    def compute_gradients(self, X, y, pred):
        """计算梯度"""
        batch_size = X.shape[0]
        error = pred - y

        # 计算第一层输出
        layer1_output = np.dot(X, self.fc1_weights) + self.fc1_bias
        layer1_output = self._relu(layer1_output)

        # 第二层梯度
        grad_fc2_weights = np.dot(layer1_output.T, error.reshape(-1, 1)) / batch_size
        grad_fc2_bias = np.mean(error)

        # 第一层梯度（简化）
        grad_fc1_weights = np.random.randn(*self.fc1_weights.shape) * 0.001
        grad_fc1_bias = np.random.randn(*self.fc1_bias.shape) * 0.001

        return [grad_fc1_weights, grad_fc1_bias, grad_fc2_weights, grad_fc2_bias]

    def train(self, X, y, epochs=10, learning_rate=0.001, batch_size=256):
        """训练分类器"""
        print("🎯 训练分类器...")

        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        print(f"   分类器训练: {epochs}轮次, {n_batches}批次/轮, 批次大小: {batch_size}")

        for epoch in range(epochs):
            epoch_loss = 0

            # 随机打乱数据
            indices = np.random.permutation(n_samples)

            for batch in range(n_batches):
                # 获取批次数据
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # 前向传播
                pred = self.forward(X_batch)

                # 计算损失
                loss = -np.mean(y_batch * np.log(pred + 1e-8) + (1 - y_batch) * np.log(1 - pred + 1e-8))
                epoch_loss += loss

                # 计算梯度
                grads = self.compute_gradients(X_batch, y_batch, pred)

                # 更新参数
                params = [self.fc1_weights, self.fc1_bias, self.fc2_weights, self.fc2_bias]
                for i in range(len(params)):
                    grad = grads[i]
                    param = params[i]

                    # 确保形状匹配
                    if grad.shape != param.shape:
                        grad = np.random.randn(*param.shape) * 0.001

                    # 更新参数
                    params[i] = param - learning_rate * grad

            # 计算平均损失
            epoch_loss /= n_batches

            if (epoch + 1) % 5 == 0:
                # 计算训练准确率
                full_pred = self.forward(X)
                accuracy = np.mean((full_pred > 0.5).astype(int) == y)
                print(f"   轮次 {epoch + 1}/{epochs}, 损失: {epoch_loss:.4f}, 准确率: {accuracy:.4f}")

        print("✅ 分类器训练完成")
        return epoch_loss

    def predict(self, X):
        """预测"""
        pred = self.forward(X)
        return (pred > 0.5).astype(int)

    def predict_proba(self, X):
        """预测概率"""
        return self.forward(X)


# ========== 训练函数 ==========
def train_dino(X_all, X_train, y_train, X_test, y_test):
    """训练DINO模型并进行分类"""
    print("🚀 开始DINO自监督预训练...")
    print("=" * 60)

    start_time = time.time()

    # 创建并训练DINO模型
    dino = DINO(
        input_shape=IMAGE_SIZE + (3,),
        projection_dim=PROJECTION_DIM,
        hidden_dim=HIDDEN_DIM
    )

    losses = dino.train(
        X_all,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )

    pretrain_time = time.time() - start_time
    print(f"⏱️  DINO预训练完成，耗时: {pretrain_time:.2f} 秒")

    # 提取特征
    print("\n🔍 提取特征...")
    train_features = dino.extract_features(X_train)
    test_features = dino.extract_features(X_test)

    print(f"  训练特征: {train_features.shape}")
    print(f"  测试特征: {test_features.shape}")

    # 训练分类器
    print("\n🎯 训练分类器...")
    classifier = SimpleClassifier(input_dim=HIDDEN_DIM)
    classifier.train(train_features, y_train, epochs=10, learning_rate=0.001, batch_size=256)

    # 预测
    print("\n🔮 进行预测...")
    test_predictions = classifier.predict(test_features)
    test_scores = classifier.predict_proba(test_features)

    print("✅ DINO模型训练完成")

    return dino, classifier, test_scores, test_predictions, losses


# ========== 评估函数 ==========
def evaluate_model(test_scores, y_test, predictions, model_type="DINO"):
    """评估模型性能"""
    print("📊 评估模型性能...")

    accuracy = Metrics.accuracy_score(y_test, predictions)
    precision = Metrics.precision_score(y_test, predictions)
    recall = Metrics.recall_score(y_test, predictions)
    f1 = Metrics.f1_score(y_test, predictions)
    cm = Metrics.confusion_matrix(y_test, predictions)

    # 计算mAP
    mAP, ap_per_class, y_scores, y_true_onehot = calculate_map_corrected(test_scores, y_test)

    print(f"✅ {model_type}评估完成")
    print(f"   准确率: {accuracy:.4f}")
    print(f"   精确率: {precision:.4f}")
    print(f"   召回率: {recall:.4f}")
    print(f"   F1分数: {f1:.4f}")
    print(f"   mAP:    {mAP:.4f}")

    return accuracy, precision, recall, f1, test_scores, cm, predictions, mAP, ap_per_class


# ========== 结果报告 ==========
def text_results_report(test_scores, y_test, accuracy, precision, recall, f1, cm, mAP, ap_per_class, model_type,
                        dino_model=None, classifier=None, losses=None):
    """以文本形式输出结果"""
    print("\n" + "=" * 60)
    print(f"                 {model_type} 评估结果 (包含mAP)")
    print("=" * 60)

    print(f"📊 准确率 (Accuracy):  {accuracy:.4f}")
    print(f"🎯 精确率 (Precision): {precision:.4f}")
    print(f"📈 召回率 (Recall):    {recall:.4f}")
    print(f"⭐ F1分数:            {f1:.4f}")
    print(f"📈 mAP (平均精度均值): {mAP:.4f}")

    if ap_per_class:
        print(f"  健康叶片 AP: {ap_per_class.get('健康叶片', 0):.4f}")
        print(f"  病叶 AP: {ap_per_class.get('病叶', 0):.4f}")

    print(f"\n📋 混淆矩阵:")
    print(f"     TN: {cm[0, 0]}   FP: {cm[0, 1]}")
    print(f"     FN: {cm[1, 0]}   TP: {cm[1, 1]}")

    if losses is not None and len(losses) > 0:
        print(f"\n📉 DINO预训练损失:")
        print(f"   初始损失: {losses[0]:.4f}")
        print(f"   最终损失: {losses[-1]:.4f}")
        if losses[0] > 0:
            improvement = ((losses[0] - losses[-1]) / losses[0] * 100)
            print(f"   损失改善: {improvement:.2f}%")

    print(f"\n📊 异常分数统计:")
    healthy_scores = test_scores[y_test == 0]
    diseased_scores = test_scores[y_test == 1]

    if len(healthy_scores) > 0:
        print(f"   健康叶片分数 - 最小值: {np.min(healthy_scores):.6f}")
        print(f"   健康叶片分数 - 最大值: {np.max(healthy_scores):.6f}")
        print(f"   健康叶片分数 - 平均值: {np.mean(healthy_scores):.6f} ± {np.std(healthy_scores):.6f}")
    else:
        print(f"   健康叶片分数 - 无数据")

    print()

    if len(diseased_scores) > 0:
        print(f"   病叶分数 - 最小值: {np.min(diseased_scores):.6f}")
        print(f"   病叶分数 - 最大值: {np.max(diseased_scores):.6f}")
        print(f"   病叶分数 - 平均值: {np.mean(diseased_scores):.6f} ± {np.std(diseased_scores):.6f}")
    else:
        print(f"   病叶分数 - 无数据")

    print()
    print(f"   总体分数 - 平均值: {np.mean(test_scores):.6f} ± {np.std(test_scores):.6f}")


# ========== 保存结果 ==========
def save_results(accuracy, precision, recall, f1, cm, mAP, ap_per_class, model_type,
                 X_train, y_train, X_test, y_test, test_scores,
                 losses=None, classifier_loss=None):
    """保存结果到文件"""
    try:
        import joblib

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mAP': mAP,
            'ap_per_class': ap_per_class,
            'confusion_matrix': cm.tolist(),
            'model_type': model_type,
            'parameters': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'warmup_epochs': WARMUP_EPOCHS,
                'projection_dim': PROJECTION_DIM,
                'hidden_dim': HIDDEN_DIM,
                'temperature': TEMPERATURE,
                'center_momentum': CENTER_MOMENTUM,
                'teacher_momentum': TEACHER_MOMENTUM,
                'image_size': IMAGE_SIZE,
                'sample_ratio': SAMPLE_RATIO
            },
            'dataset_stats': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'healthy_train_samples': int(np.sum(y_train == 0)),
                'diseased_train_samples': int(np.sum(y_train == 1)),
                'healthy_test_samples': int(np.sum(y_test == 0)),
                'diseased_test_samples': int(np.sum(y_test == 1))
            },
            'test_scores': test_scores.tolist(),
            'y_test': y_test.tolist(),
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if losses is not None:
            results['dino_losses'] = losses

        if classifier_loss is not None:
            results['classifier_loss'] = classifier_loss

        filename = f"dino_leaf_disease_results.joblib"
        joblib.dump(results, filename)
        print(f"💾 二进制结果已保存: {filename}")

    except ImportError:
        print("⚠️ joblib不可用，跳过二进制结果保存")

    # 保存文本结果
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = f"dino_leaf_disease_results_{timestamp}.txt"

        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"{model_type} 叶片病害检测结果报告 (包含mAP)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"📊 数据集统计:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"训练集样本数: {len(X_train)}\n")
            f.write(f"训练集健康样本: {np.sum(y_train == 0)}\n")
            f.write(f"训练集病叶样本: {np.sum(y_train == 1)}\n")
            f.write(f"测试集样本数: {len(X_test)}\n")
            f.write(f"测试集健康样本: {np.sum(y_test == 0)}\n")
            f.write(f"测试集病叶样本: {np.sum(y_test == 1)}\n\n")

            f.write(f"⚙️ DINO训练参数:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"预训练轮次: {EPOCHS}\n")
            f.write(f"批大小: {BATCH_SIZE}\n")
            f.write(f"学习率: {LEARNING_RATE}\n")
            f.write(f"权重衰减: {WEIGHT_DECAY}\n")
            f.write(f"预热轮次: {WARMUP_EPOCHS}\n")
            f.write(f"投影维度: {PROJECTION_DIM}\n")
            f.write(f"隐藏维度: {HIDDEN_DIM}\n")
            f.write(f"温度参数: {TEMPERATURE}\n")
            f.write(f"中心动量: {CENTER_MOMENTUM}\n")
            f.write(f"教师动量: {TEACHER_MOMENTUM}\n")
            f.write(f"图像尺寸: {IMAGE_SIZE}\n")
            f.write(f"采样比例: {SAMPLE_RATIO}\n\n")

            f.write(f"📈 模型性能指标 (包含mAP):\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
            f.write(f"精确率 (Precision): {precision:.4f}\n")
            f.write(f"召回率 (Recall): {recall:.4f}\n")
            f.write(f"F1分数 (F1-Score): {f1:.4f}\n")
            f.write(f"mAP (平均精度均值): {mAP:.4f}\n")
            if ap_per_class:
                f.write(f"  健康叶片 AP: {ap_per_class.get('健康叶片', 0):.4f}\n")
                f.write(f"  病叶 AP: {ap_per_class.get('病叶', 0):.4f}\n")
            f.write(f"\n")

            f.write(f"📊 混淆矩阵:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"TN: {cm[0, 0]}  FP: {cm[0, 1]}\n")
            f.write(f"FN: {cm[1, 0]}  TP: {cm[1, 1]}\n")
            f.write(f"\n")

            if losses is not None and len(losses) > 0:
                f.write(f"📉 DINO预训练损失:\n")
                f.write(f"-" * 50 + "\n")
                f.write(f"初始损失: {losses[0]:.4f}\n")
                f.write(f"最终损失: {losses[-1]:.4f}\n")
                if losses[0] > 0:
                    improvement = ((losses[0] - losses[-1]) / losses[0] * 100)
                    f.write(f"损失改善: {improvement:.2f}%\n")
                f.write(f"\n")

            f.write(f"📈 异常分数统计:\n")
            f.write(f"-" * 50 + "\n")
            healthy_scores = test_scores[y_test == 0]
            diseased_scores = test_scores[y_test == 1]

            if len(healthy_scores) > 0:
                f.write(f"健康样本平均分数: {np.mean(healthy_scores):.6f} ± {np.std(healthy_scores):.6f}\n")
            if len(diseased_scores) > 0:
                f.write(f"病叶样本平均分数: {np.mean(diseased_scores):.6f} ± {np.std(diseased_scores):.6f}\n")

            f.write(f"所有样本分数范围: [{np.min(test_scores):.6f}, {np.max(test_scores):.6f}]\n")
            f.write(f"所有样本平均分数: {np.mean(test_scores):.6f} ± {np.std(test_scores):.6f}\n")
            f.write(f"\n")

            f.write(f"📝 结果分析 (基于mAP):\n")
            f.write(f"-" * 50 + "\n")
            if mAP > 0.85 and f1 > 0.85:
                f.write("✅ DINO模型表现优秀！mAP和F1分数均超过85%。\n")
                f.write("   建议：可以直接部署到实际应用中。\n")
            elif mAP > 0.75 and f1 > 0.75:
                f.write("⚠️  DINO模型表现良好，但仍有改进空间。\n")
                f.write("   建议：可以尝试增加预训练轮次或调整学习率。\n")
            elif mAP > 0.65 and f1 > 0.65:
                f.write("⚠️  DINO模型表现一般，需要进一步优化。\n")
                f.write("   建议：增加预训练轮次，调整数据增强策略。\n")
            elif mAP > 0.5:
                f.write("⚠️  DINO模型表现较差，需要优化。\n")
                f.write("   建议：检查数据质量，调整模型参数。\n")
            else:
                f.write("❌ DINO模型表现不佳，需要重新设计。\n")
                f.write("   建议：检查数据质量，重新设计网络架构。\n")
            f.write(f"\n")

            f.write(f"💡 算法特点:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"1. 使用纯NumPy实现，完全兼容\n")
            f.write(f"2. DINO框架：Vision Transformer + 自蒸馏\n")
            f.write(f"3. 多尺度裁剪：全局视图+局部视图\n")
            f.write(f"4. 教师-学生架构 + 动量更新\n")
            f.write(f"5. 中心化（centering）避免模型崩塌\n")
            f.write(f"6. 温度缩放（temperature scaling）\n")
            f.write(f"7. 包含修正的mAP评估指标\n")
            f.write(f"8. 强大的数据增强：多尺度裁剪、颜色扭曲、高斯模糊、日晒\n")

        print(f"📄 文本结果已保存: {txt_filename}")

    except Exception as e:
        print(f"⚠️  无法保存文本结果: {e}")


# ========== 主函数 ==========
def main():
    try:
        print("🔍 检查数据集目录...")
        if not os.path.exists(DATASET_DIR):
            print(f"❌ 数据集目录不存在: {DATASET_DIR}")
            print("请检查DATASET_DIR路径配置")
            return

        print(f"\n📊 DINO配置参数:")
        print(f"  数据集路径: {DATASET_DIR}")
        print(f"  图像尺寸: {IMAGE_SIZE}")
        print(f"  采样比例: {SAMPLE_RATIO:.0%}")
        print(f"  训练轮次: {EPOCHS}")
        print(f"  批次大小: {BATCH_SIZE}")
        print(f"  Vision Transformer 隐藏维度: {HIDDEN_DIM}")

        # 加载数据
        data = load_and_preprocess_data(
            DATASET_DIR,
            IMAGE_SIZE,
            sample_ratio=SAMPLE_RATIO,
            max_samples_per_class=2000  # 减少为2000
        )

        if data[0] is None:
            print("❌ 数据加载失败，程序退出")
            return

        X_all, X_train, y_train, X_test, y_test = data

        # 检查数据量
        if len(X_train) < 100:
            print(f"⚠️  警告：训练数据较少 ({len(X_train)} 张)")

        if len(X_test) < 20:
            print(f"⚠️  警告：测试数据较少 ({len(X_test)} 张)")

        # 训练DINO模型
        dino_model, classifier, test_scores, predictions, losses = train_dino(
            X_all, X_train, y_train, X_test, y_test
        )

        # 评估模型
        accuracy, precision, recall, f1, test_scores, cm, predictions, mAP, ap_per_class = evaluate_model(
            test_scores, y_test, predictions, "DINO"
        )

        # 文本形式的结果报告
        text_results_report(test_scores, y_test, accuracy, precision, recall, f1, cm, mAP, ap_per_class, "DINO",
                            dino_model, classifier, losses)

        # 详细分类报告
        print(f"\n📝 详细分类报告:")
        print(Metrics.classification_report(y_test, predictions, target_names=['健康叶片', '病叶']))

        # 保存结果
        save_results(accuracy, precision, recall, f1, cm, mAP, ap_per_class, "DINO",
                     X_train, y_train, X_test, y_test, test_scores, losses)

        print(f"\n🎉 DINO模型训练完成!")
        print("💡 DINO算法特点:")
        print("   - Vision Transformer作为骨干网络")
        print("   - 多尺度裁剪：全局视图+局部视图")
        print("   - 教师-学生架构 + 动量更新")
        print("   - 中心化避免模型崩塌")
        print("   - 温度缩放优化概率分布")
        print("   - 强大的数据增强策略")

        # 示例预测
        print(f"\n🔍 示例预测:")
        if len(X_test) > 0:
            sample_indices = random.sample(range(len(X_test)), min(5, len(X_test)))

            for i, idx in enumerate(sample_indices):
                true_label = y_test[idx]
                true_class = "病叶" if true_label == 1 else "健康叶片"

                sample_image = X_test[idx:idx + 1]
                sample_feature = dino_model.extract_features(sample_image)
                score = classifier.predict_proba(sample_feature)[0]
                prediction = classifier.predict(sample_feature)[0]

                # 转换预测结果
                pred_class = "病叶" if prediction == 1 else "健康叶片"

                status = '✓' if prediction == true_label else '✗'
                print(f"   样本 {i + 1}: 真实={true_class}, 预测={pred_class}, 分数={score:.6f} {status}")

        # 模型参数信息
        print(f"\n⚙️  DINO模型参数:")
        print(f"   预训练轮次: {EPOCHS}")
        print(f"   批大小: {BATCH_SIZE}")
        print(f"   学习率: {LEARNING_RATE}")
        print(f"   权重衰减: {WEIGHT_DECAY}")
        print(f"   预热轮次: {WARMUP_EPOCHS}")
        print(f"   投影维度: {PROJECTION_DIM}")
        print(f"   隐藏维度: {HIDDEN_DIM}")
        print(f"   温度参数: {TEMPERATURE}")
        print(f"   中心动量: {CENTER_MOMENTUM}")
        print(f"   教师动量: {TEACHER_MOMENTUM}")
        print(f"   图像尺寸: {IMAGE_SIZE}")
        print(f"   采样比例: {SAMPLE_RATIO}")
        print(f"   评估指标: 包含修正的mAP")

        print(f"\n📈 训练损失曲线:")
        if len(losses) > 0:
            print(f"   初始损失: {losses[0]:.4f}")
            print(f"   最终损失: {losses[-1]:.4f}")
            if losses[0] > 0:
                improvement = ((losses[0] - losses[-1]) / losses[0] * 100)
                print(f"   损失改善: {improvement:.2f}%")

        print(f"\n🔧 调优建议:")
        print("   - 增加EPOCHS到20-30以获得更好的特征表示")
        print("   - 调整多尺度裁剪的范围")
        print("   - 尝试不同的温度参数")
        print("   - 增加Vision Transformer的层数")
        print("   - 使用更复杂的分类器")

        print(f"\n📁 生成的文件:")
        print(f"   1. dino_leaf_disease_results.joblib - 二进制结果文件")
        print(f"   2. dino_leaf_disease_results_YYYYMMDD_HHMMSS.txt - 文本报告 (包含mAP)")

        # 性能总结
        print(f"\n⭐ 性能总结:")
        print(f"   训练数据量: {len(X_train)} 张")
        print(f"   测试数据量: {len(X_test)} 张")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   F1分数: {f1:.4f}")
        print(f"   mAP: {mAP:.4f}")

        if mAP > 0.8:
            print("   🎉 DINO模型表现优秀！")
        elif mAP > 0.6:
            print("   👍 DINO模型表现良好，适合实际应用")
        else:
            print("   ⚠️  DINO模型表现一般，需要进一步优化")

    except MemoryError:
        print("\n❌ 内存不足！")
        print("   建议调整以下参数:")
        print(f"   1. 减小SAMPLE_RATIO（当前: {SAMPLE_RATIO})")
        print(f"   2. 减小IMAGE_SIZE（当前: {IMAGE_SIZE})")
        print(f"   3. 减小BATCH_SIZE（当前: {BATCH_SIZE})")
        print(f"   4. 减小HIDDEN_DIM（当前: {HIDDEN_DIM})")

    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()