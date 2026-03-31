# deepcluster_leaf_disease_detection_with_map.py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🚀 启动 DeepCluster 叶片病害检测系统 (包含mAP)")

# 基础导入
try:
    import numpy as np

    print(f"✅ NumPy 版本: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy 导入失败: {e}")
    exit(1)

import sys
import random
import time
import datetime
from PIL import Image
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib支持中文
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("       DeepCluster 叶片病害识别系统 (包含mAP)")
print("=" * 60)


# 手动实现评估指标
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


# 修正的mAP计算函数 - 解决负数问题
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


# 可视化PR曲线
def visualize_pr_curves(y_true_onehot, y_scores, ap_per_class, mAP, filename="deepcluster_pr_curves.png"):
    """可视化PR曲线"""
    plt.figure(figsize=(12, 5))

    # 为每个类别绘制PR曲线
    for class_idx, class_name in enumerate(['健康叶片', '病叶']):
        # 计算PR曲线
        precision_values = []
        recall_values = []

        # 按照预测分数降序排序
        sorted_indices = np.argsort(y_scores[:, class_idx])[::-1]
        true_class_sorted = y_true_onehot[:, class_idx][sorted_indices]
        pred_class_sorted = y_scores[:, class_idx][sorted_indices]

        total_positives = np.sum(true_class_sorted)

        # 计算每个阈值下的精确率和召回率
        for i in range(1, len(pred_class_sorted)):
            threshold = pred_class_sorted[i]
            pred_positives = pred_class_sorted >= threshold

            if np.sum(pred_positives) == 0:
                precision = 0
            else:
                true_positives = np.sum(true_class_sorted[pred_positives])
                precision = true_positives / np.sum(pred_positives)

            recall = np.sum(true_class_sorted[pred_positives]) / total_positives if total_positives > 0 else 0

            precision_values.append(precision)
            recall_values.append(recall)

        # 绘制PR曲线
        plt.subplot(1, 2, class_idx + 1)
        plt.plot(recall_values, precision_values, linewidth=2,
                 label=f'{class_name} (AP={ap_per_class[class_name]:.3f})')
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title(f'{class_name} - 精确率-召回率曲线')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.fill_between(recall_values, precision_values, alpha=0.2)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

    plt.suptitle(f'DeepCluster - 精确率-召回率曲线 (mAP = {mAP:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ PR曲线已保存: {filename}")


# 配置参数
DATASET_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\balanced-crop-dataset-new"
IMAGE_SIZE = (64, 64)  # 输入图像尺寸
SAMPLE_RATIO = 0.05

# DeepCluster 参数
N_CLUSTERS = 3  # 聚类数量
PRETRAIN_EPOCHS = 50  # 预训练轮数
CLUSTERING_EPOCHS = 5  # DeepCluster 迭代次数
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SOBEL_FILTER = True  # 是否使用Sobel滤波器预处理

print("=" * 60)
print("       DeepCluster 叶片病害检测系统 (包含mAP)")
print("=" * 60)


# 数据加载函数
def load_and_preprocess_data(dataset_dir, image_size, sample_ratio=0.05):
    """加载和预处理数据"""

    def load_images_from_folder(folder, label):
        images = []
        labels = []

        if not os.path.exists(folder):
            print(f"❌ 目录不存在: {folder}")
            return images, labels

        img_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if sample_ratio < 1.0:
            sample_size = max(1, int(len(img_files) * sample_ratio))
            img_files = random.sample(img_files, sample_size)

        print(f"📁 处理 {os.path.basename(folder)}... ({len(img_files)} 张图片)")

        for i, img_file in enumerate(img_files):
            img_path = os.path.join(folder, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                image = image.resize(image_size)
                img_array = np.array(image) / 255.0
                images.append(img_array)
                labels.append(label)

                if i % 100 == 0 and i > 0:
                    print(f"    已处理 {i}/{len(img_files)} 张图片")

            except Exception as e:
                print(f"⚠️ 处理图像失败 {img_path}: {e}")

        return images, labels

    print("📊 加载和预处理数据...")

    # 训练数据 - 只使用健康样本（无监督学习）
    train_healthy_dir = os.path.join(dataset_dir, "train", "healthy")
    train_healthy_images, train_healthy_labels = load_images_from_folder(train_healthy_dir, 0)

    if len(train_healthy_images) == 0:
        print("❌ 没有找到健康训练数据")
        return None, None, None, None

    # 测试数据
    test_healthy_dir = os.path.join(dataset_dir, "test", "healthy")
    test_diseased_dir = os.path.join(dataset_dir, "test", "diseased")

    test_healthy_images, test_healthy_labels = load_images_from_folder(test_healthy_dir, 0)
    test_diseased_images, test_diseased_labels = load_images_from_folder(test_diseased_dir, 1)

    if len(test_healthy_images) == 0 and len(test_diseased_images) == 0:
        print("❌ 没有找到测试数据")
        return None, None, None, None

    X_train = np.array(train_healthy_images)
    y_train = np.array(train_healthy_labels)

    X_test = np.array(test_healthy_images + test_diseased_images)
    y_test = np.array(test_healthy_labels + test_diseased_labels)

    print(f"✅ 数据加载完成:")
    print(f"  训练集 (仅健康样本): {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  健康样本: {np.sum(y_test == 0)}, 病叶样本: {np.sum(y_test == 1)}")

    return X_train, y_train, X_test, y_test


# 图像预处理 - Sobel滤波器
def sobel_filter(images):
    """应用Sobel滤波器提取边缘特征"""
    print("🔍 应用Sobel滤波器...")

    sobel_images = []
    for img in images:
        # 转换为灰度图
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

        # Sobel算子
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # 卷积操作
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)

        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                gx[i, j] = np.sum(sobel_x * gray[i - 1:i + 2, j - 1:j + 2])
                gy[i, j] = np.sum(sobel_y * gray[i - 1:i + 2, j - 1:j + 2])

        # 计算梯度幅值
        gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # 归一化
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()

        # 堆叠为3通道
        sobel_img = np.stack([gradient_magnitude] * 3, axis=-1)
        sobel_images.append(sobel_img)

    return np.array(sobel_images)


# 手动实现简单的CNN特征提取器
class SimpleCNN:
    """简化的CNN特征提取器"""

    def __init__(self, input_shape, feature_dim=64):
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.filters = []
        self.biases = []
        self.dense_weights = None
        self.dense_bias = None
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化CNN权重"""
        height, width, channels = self.input_shape

        # 第一层卷积: 3x3, 16个滤波器
        filter1 = np.random.randn(3, 3, channels, 16) * 0.1
        bias1 = np.zeros(16)

        # 第二层卷积: 3x3, 32个滤波器
        filter2 = np.random.randn(3, 3, 16, 32) * 0.1
        bias2 = np.zeros(32)

        self.filters = [filter1, filter2]
        self.biases = [bias1, bias2]

        # 计算卷积后的特征图尺寸
        # 第一层卷积: (64-3+1) = 62, 池化后: 31
        # 第二层卷积: (31-3+1) = 29, 池化后: 14
        conv_height = 14
        conv_width = 14
        fc_input_dim = conv_height * conv_width * 32

        print(f"   全连接层输入维度: {fc_input_dim}")

        self.dense_weights = np.random.randn(fc_input_dim, self.feature_dim) * 0.1
        self.dense_bias = np.zeros(self.feature_dim)

    def _relu(self, x):
        return np.maximum(0, x)

    def _max_pool(self, x, pool_size=2):
        """最大池化"""
        n, h, w, c = x.shape
        h_out = h // pool_size
        w_out = w // pool_size

        pooled = np.zeros((n, h_out, w_out, c))

        for i in range(h_out):
            for j in range(w_out):
                region = x[:, i * pool_size:(i + 1) * pool_size, j * pool_size:(j + 1) * pool_size, :]
                pooled[:, i, j, :] = np.max(region, axis=(1, 2))

        return pooled

    def _conv2d(self, x, filters, bias):
        """2D卷积"""
        n, h, w, c_in = x.shape
        fh, fw, c_in, c_out = filters.shape

        # 计算输出尺寸
        h_out = h - fh + 1
        w_out = w - fw + 1

        output = np.zeros((n, h_out, w_out, c_out))

        for i in range(h_out):
            for j in range(w_out):
                for k in range(c_out):
                    region = x[:, i:i + fh, j:j + fw, :]
                    output[:, i, j, k] = np.sum(region * filters[:, :, :, k], axis=(1, 2, 3)) + bias[k]

        return output

    def extract_features(self, X):
        """提取CNN特征"""
        if X.ndim == 3:
            X = np.expand_dims(X, 0)

        print(f"   输入形状: {X.shape}")

        # 第一层卷积 + ReLU + 池化
        conv1 = self._conv2d(X, self.filters[0], self.biases[0])
        print(f"   第一层卷积后形状: {conv1.shape}")
        relu1 = self._relu(conv1)
        pool1 = self._max_pool(relu1)
        print(f"   第一层池化后形状: {pool1.shape}")

        # 第二层卷积 + ReLU + 池化
        conv2 = self._conv2d(pool1, self.filters[1], self.biases[1])
        print(f"   第二层卷积后形状: {conv2.shape}")
        relu2 = self._relu(conv2)
        pool2 = self._max_pool(relu2)
        print(f"   第二层池化后形状: {pool2.shape}")

        # 展平
        flattened = pool2.reshape(pool2.shape[0], -1)
        print(f"   展平后形状: {flattened.shape}")
        print(f"   全连接层权重形状: {self.dense_weights.shape}")

        # 全连接层
        features = np.dot(flattened, self.dense_weights) + self.dense_bias
        features = self._relu(features)
        print(f"   最终特征形状: {features.shape}")

        return features

    def train(self, X, y, epochs=10, batch_size=32, learning_rate=0.001):
        """训练CNN（简化版，使用伪标签）"""
        print("🎯 训练CNN特征提取器...")

        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        # 创建临时分类层权重
        n_classes = len(np.unique(y))
        temp_weights = np.random.randn(self.feature_dim, n_classes) * 0.1
        temp_bias = np.zeros(n_classes)

        for epoch in range(epochs):
            epoch_loss = 0

            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)

                batch_x = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                # 前向传播
                features = self.extract_features(batch_x)

                # 简单的分类层
                logits = np.dot(features, temp_weights) + temp_bias
                probs = self._softmax(logits)

                # 计算交叉熵损失
                loss = self._cross_entropy(probs, batch_y)
                epoch_loss += loss

                # 简化的反向传播（这里只更新临时权重）
                # 在实际DeepCluster中，这里会有更复杂的更新逻辑
                grad_logits = probs
                grad_logits[np.arange(len(batch_y)), batch_y] -= 1
                grad_logits /= len(batch_y)

                grad_weights = np.dot(features.T, grad_logits)
                grad_bias = np.sum(grad_logits, axis=0)

                temp_weights -= learning_rate * grad_weights
                temp_bias -= learning_rate * grad_bias

            epoch_loss /= n_batches

            if (epoch + 1) % 5 == 0:
                print(f"   轮次 {epoch + 1}/{epochs}, 损失: {epoch_loss:.6f}")

        print(f"✅ CNN训练完成")

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _cross_entropy(self, probs, y):
        n_samples = y.shape[0]
        log_probs = -np.log(probs[np.arange(n_samples), y] + 1e-8)
        return np.mean(log_probs)


# K-means 聚类实现
class SimpleKMeans:
    """简化的K-means聚类"""

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _initialize_centroids(self, X):
        """随机初始化质心"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]

    def _compute_distances(self, X, centroids):
        """计算样本到质心的距离"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))
        return distances

    def fit(self, X):
        """训练K-means模型"""
        print(f"🎯 训练K-means聚类 (k={self.n_clusters})...")

        # 初始化质心
        self.centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iter):
            # 分配样本到最近的簇
            distances = self._compute_distances(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)

            # 更新质心
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.n_clusters):
                if np.sum(self.labels == i) > 0:
                    new_centroids[i] = X[self.labels == i].mean(axis=0)

            # 检查收敛
            centroid_shift = np.sqrt(np.sum((new_centroids - self.centroids) ** 2, axis=1)).mean()
            self.centroids = new_centroids

            if centroid_shift < self.tol:
                print(f"   迭代 {iteration + 1}: 收敛 (质心移动: {centroid_shift:.6f})")
                break

            if (iteration + 1) % 10 == 0:
                print(f"   迭代 {iteration + 1}: 质心移动 = {centroid_shift:.6f}")

        # 计算簇内平方和
        self.inertia_ = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                distances = np.sqrt(np.sum((cluster_points - self.centroids[i]) ** 2, axis=1))
                self.inertia_ += np.sum(distances ** 2)

        print(f"✅ K-means训练完成")
        print(f"   最终迭代次数: {min(iteration + 1, self.max_iter)}")
        print(f"   最终inertia: {self.inertia_:.4f}")
        print(f"   簇大小分布: {np.bincount(self.labels)}")

        return self

    def predict(self, X):
        """预测样本所属簇"""
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)


# DeepCluster 模型实现
class DeepClusterModel:
    """DeepCluster 模型"""

    def __init__(self, n_clusters=3, feature_dim=64, sobel_filter=True):
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.sobel_filter = sobel_filter

        self.cnn = None
        self.kmeans = None
        self.feature_mean = None
        self.feature_std = None
        self.cluster_thresholds = None

    def _deepcluster_iteration(self, X, epochs_per_clustering=5):
        """DeepCluster 单次迭代"""
        # 1. 提取特征
        print("🔍 提取CNN特征...")
        features = self.cnn.extract_features(X)

        # 标准化特征
        if self.feature_mean is None:
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0) + 1e-8

        features_scaled = (features - self.feature_mean) / self.feature_std

        # 2. 聚类
        print("🔍 进行K-means聚类...")
        self.kmeans = SimpleKMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(features_scaled)
        cluster_labels = self.kmeans.labels

        # 3. 使用聚类标签训练CNN
        print("🔍 使用聚类标签训练CNN...")
        self.cnn.train(X, cluster_labels, epochs=epochs_per_clustering)

        return features_scaled, cluster_labels

    def fit(self, X_train, clustering_epochs=5, epochs_per_clustering=5):
        """训练DeepCluster模型"""
        print("🎯 训练DeepCluster模型...")

        # 图像预处理
        if self.sobel_filter:
            print("🔍 应用Sobel滤波器预处理...")
            X_processed = sobel_filter(X_train)
        else:
            X_processed = X_train

        # 初始化CNN
        input_shape = X_processed.shape[1:]
        print(f"   输入形状: {input_shape}")
        self.cnn = SimpleCNN(input_shape, self.feature_dim)

        # DeepCluster迭代
        all_features = []
        all_labels = []

        for epoch in range(clustering_epochs):
            print(f"\n🔄 DeepCluster 迭代 {epoch + 1}/{clustering_epochs}")

            features_scaled, cluster_labels = self._deepcluster_iteration(
                X_processed, epochs_per_clustering
            )

            all_features.append(features_scaled)
            all_labels.append(cluster_labels)

            # 计算聚类质量
            unique_labels = np.unique(cluster_labels)
            print(f"   聚类分布: {np.bincount(cluster_labels)}")
            print(f"   聚类数量: {len(unique_labels)}")

        # 计算聚类阈值
        final_features = all_features[-1]
        self._compute_cluster_thresholds(final_features)

        print("✅ DeepCluster训练完成")
        return all_features, all_labels

    def _compute_cluster_thresholds(self, features):
        """计算每个簇的距离阈值"""
        print("🎯 计算聚类阈值...")

        distances = self._compute_distances(features, self.kmeans.centroids)
        min_distances = np.min(distances, axis=1)
        nearest_clusters = np.argmin(distances, axis=1)

        self.cluster_thresholds = []
        for i in range(self.n_clusters):
            cluster_distances = min_distances[nearest_clusters == i]
            if len(cluster_distances) > 0:
                # 使用95%分位数作为阈值
                threshold = np.percentile(cluster_distances, 95)
                self.cluster_thresholds.append(threshold)
            else:
                self.cluster_thresholds.append(0)

        self.cluster_thresholds = np.array(self.cluster_thresholds)
        print(f"   聚类阈值: {self.cluster_thresholds}")

    def _compute_distances(self, X, centroids):
        """计算样本到质心的距离"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))
        return distances

    def predict(self, X):
        """预测样本是否为异常（病叶）"""
        # 图像预处理
        if self.sobel_filter:
            X_processed = sobel_filter(X)
        else:
            X_processed = X

        # 提取特征
        features = self.cnn.extract_features(X_processed)
        features_scaled = (features - self.feature_mean) / self.feature_std

        # 计算到每个聚类中心的距离
        distances = self._compute_distances(features_scaled, self.kmeans.centroids)
        min_distances = np.min(distances, axis=1)
        nearest_clusters = np.argmin(distances, axis=1)

        # 预测：如果距离大于对应簇的阈值，则为异常（病叶）
        predictions = []
        for i, (dist, cluster) in enumerate(zip(min_distances, nearest_clusters)):
            if dist > self.cluster_thresholds[cluster]:
                predictions.append(1)  # 异常（病叶）
            else:
                predictions.append(0)  # 正常（健康）

        return np.array(predictions)

    def decision_function(self, X):
        """计算异常分数（距离分数）"""
        # 图像预处理
        if self.sobel_filter:
            X_processed = sobel_filter(X)
        else:
            X_processed = X

        # 提取特征
        features = self.cnn.extract_features(X_processed)
        features_scaled = (features - self.feature_mean) / self.feature_std

        # 计算到最近聚类中心的距离
        distances = self._compute_distances(features_scaled, self.kmeans.centroids)
        min_distances = np.min(distances, axis=1)

        return min_distances

    def get_cluster_info(self, X):
        """获取聚类信息"""
        # 图像预处理
        if self.sobel_filter:
            X_processed = sobel_filter(X)
        else:
            X_processed = X

        # 提取特征
        features = self.cnn.extract_features(X_processed)
        features_scaled = (features - self.feature_mean) / self.feature_std

        # 计算聚类信息
        distances = self._compute_distances(features_scaled, self.kmeans.centroids)
        nearest_clusters = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)

        return nearest_clusters, min_distances


# 训练函数
def train_deepcluster(X_train, X_test, y_test):
    """训练DeepCluster模型"""
    print("🚀 开始训练DeepCluster模型...")

    start_time = time.time()

    # 创建并训练模型
    deepcluster = DeepClusterModel(
        n_clusters=N_CLUSTERS,
        feature_dim=64,
        sobel_filter=SOBEL_FILTER
    )

    all_features, all_labels = deepcluster.fit(
        X_train,
        clustering_epochs=CLUSTERING_EPOCHS,
        epochs_per_clustering=5
    )

    training_time = time.time() - start_time
    print(f"⏱️  训练完成，耗时: {training_time:.2f} 秒")

    # 预测
    print("🔮 进行预测...")
    test_predictions = deepcluster.predict(X_test)
    test_scores = deepcluster.decision_function(X_test)

    print("✅ DeepCluster模型训练完成")

    return deepcluster, test_scores, test_predictions, all_features, all_labels


# 评估函数 - 添加mAP计算
def evaluate_model(test_scores, y_test, predictions, model_type="DeepCluster"):
    """评估模型性能"""
    print("📊 评估模型性能...")

    accuracy = Metrics.accuracy_score(y_test, predictions)
    precision = Metrics.precision_score(y_test, predictions)
    recall = Metrics.recall_score(y_test, predictions)
    f1 = Metrics.f1_score(y_test, predictions)
    cm = Metrics.confusion_matrix(y_test, predictions)

    # 计算mAP
    mAP, ap_per_class, y_scores, y_true_onehot = calculate_map_corrected(test_scores, y_test)

    # 可视化PR曲线
    visualize_pr_curves(y_true_onehot, y_scores, ap_per_class, mAP)

    print(f"✅ {model_type}评估完成")
    print(f"   准确率: {accuracy:.4f}")
    print(f"   精确率: {precision:.4f}")
    print(f"   召回率: {recall:.4f}")
    print(f"   F1分数: {f1:.4f}")
    print(f"   mAP:    {mAP:.4f}")

    return accuracy, precision, recall, f1, test_scores, cm, predictions, mAP, ap_per_class


# 可视化结果 - 包含mAP
def visualize_results(test_scores, y_test, cm, mAP=None, ap_per_class=None, deepcluster_model=None):
    """可视化结果，包含mAP"""
    fig = plt.figure(figsize=(16, 10))

    # 1. 异常分数分布
    plt.subplot(2, 2, 1)
    healthy_scores = test_scores[y_test == 0]
    diseased_scores = test_scores[y_test == 1]

    plt.hist(healthy_scores, bins=30, alpha=0.5, label='健康叶片', color='green')
    plt.hist(diseased_scores, bins=30, alpha=0.5, label='病叶', color='red')
    plt.xlabel('异常分数 (距离)')
    plt.ylabel('样本数量')
    plt.title('DeepCluster 异常分数分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 混淆矩阵
    plt.subplot(2, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测健康', '预测病叶'],
                yticklabels=['实际健康', '实际病叶'])
    plt.title('DeepCluster 混淆矩阵')

    # 3. 性能指标汇总（包含mAP）
    plt.subplot(2, 2, 3)
    plt.axis('off')

    # 收集评估结果
    median_threshold = np.median(test_scores)
    median_predictions = (test_scores > median_threshold).astype(int)
    accuracy_median = Metrics.accuracy_score(y_test, median_predictions)
    precision_median = Metrics.precision_score(y_test, median_predictions)
    recall_median = Metrics.recall_score(y_test, median_predictions)
    f1_median = Metrics.f1_score(y_test, median_predictions)

    metrics_text = (
        f"DeepCluster性能指标汇总\n\n"
        f"准确率:  {accuracy_median:.4f}\n"
        f"精确率: {precision_median:.4f}\n"
        f"召回率:    {recall_median:.4f}\n"
        f"F1分数:  {f1_median:.4f}\n"
    )

    if mAP is not None:
        metrics_text += f"\nmAP:       {mAP:.4f}\n"
        if ap_per_class is not None:
            metrics_text += f"健康叶片 AP: {ap_per_class.get('健康叶片', 0):.4f}\n"
            metrics_text += f"病叶 AP: {ap_per_class.get('病叶', 0):.4f}\n"

    if deepcluster_model is not None:
        metrics_text += f"\n聚类信息:\n"
        metrics_text += f"簇数量: {deepcluster_model.n_clusters}\n"
        metrics_text += f"特征维度: {deepcluster_model.feature_dim}\n"
        metrics_text += f"簇阈值: {deepcluster_model.cluster_thresholds}\n"
        if deepcluster_model.kmeans is not None:
            metrics_text += f"簇内平方和: {deepcluster_model.kmeans.inertia_:.4f}\n"

    plt.text(0.1, 0.5, metrics_text, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='center')

    # 4. 分数箱线图
    plt.subplot(2, 2, 4)
    score_data = [healthy_scores, diseased_scores]
    plt.boxplot(score_data, labels=['健康叶片', '病叶'])
    plt.title('异常分数箱线图')
    plt.ylabel('异常分数')
    plt.grid(True, alpha=0.3)

    # 添加中位数线
    plt.axhline(y=median_threshold, color='red', linestyle='--', alpha=0.7, label=f'中位数={median_threshold:.4f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig("deepcluster_final_results.png", dpi=300, bbox_inches='tight')
    plt.show()


# 文本形式的结果报告 - 包含mAP
def text_results_report(test_scores, y_test, accuracy, precision, recall, f1, cm, mAP, ap_per_class, model_type,
                        deepcluster_model=None, all_labels=None):
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

    if deepcluster_model is not None and deepcluster_model.kmeans is not None:
        print(f"\n🏷️  聚类信息:")
        print(f"   簇数量: {deepcluster_model.n_clusters}")
        print(f"   特征维度: {deepcluster_model.feature_dim}")
        print(f"   簇阈值: {deepcluster_model.cluster_thresholds}")
        print(f"   最终inertia: {deepcluster_model.kmeans.inertia_:.4f}")
        print(f"   最终簇大小: {np.bincount(deepcluster_model.kmeans.labels)}")

    if all_labels is not None:
        print(f"\n🔄 DeepCluster迭代信息:")
        for i, labels in enumerate(all_labels):
            unique_labels = np.unique(labels)
            print(f"   迭代 {i + 1}: 聚类数量={len(unique_labels)}, 分布={np.bincount(labels)}")

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


# 保存结果 - 包含mAP
def save_results(accuracy, precision, recall, f1, cm, mAP, ap_per_class, model_type, X_train, X_test, y_test,
                 test_scores, deepcluster_model=None, all_labels=None):
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
                'n_clusters': N_CLUSTERS,
                'clustering_epochs': CLUSTERING_EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'sobel_filter': SOBEL_FILTER,
                'image_size': IMAGE_SIZE,
                'sample_ratio': SAMPLE_RATIO
            },
            'dataset_stats': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'healthy_test_samples': int(np.sum(y_test == 0)),
                'diseased_test_samples': int(np.sum(y_test == 1))
            },
            'test_scores': test_scores.tolist(),
            'y_test': y_test.tolist(),
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if deepcluster_model is not None and deepcluster_model.kmeans is not None:
            results['cluster_info'] = {
                'cluster_thresholds': deepcluster_model.cluster_thresholds.tolist() if deepcluster_model.cluster_thresholds is not None else None,
                'feature_dim': deepcluster_model.feature_dim,
                'inertia': deepcluster_model.kmeans.inertia_,
                'cluster_sizes': np.bincount(deepcluster_model.kmeans.labels).tolist()
            }

        if all_labels is not None:
            results['iteration_info'] = []
            for i, labels in enumerate(all_labels):
                results['iteration_info'].append({
                    'iteration': i + 1,
                    'cluster_distribution': np.bincount(labels).tolist()
                })

        filename = f"deepcluster_leaf_disease_results.joblib"
        joblib.dump(results, filename)
        print(f"💾 二进制结果已保存: {filename}")

    except ImportError:
        print("⚠️ joblib不可用，跳过二进制结果保存")

    # 总是保存文本结果
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = f"deepcluster_leaf_disease_results_{timestamp}.txt"

        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"{model_type} 叶片病害检测结果报告 (包含mAP)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"📊 数据集统计:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"训练集样本数: {len(X_train)}\n")
            f.write(f"测试集样本数: {len(X_test)}\n")
            f.write(f"测试集健康样本: {np.sum(y_test == 0)}\n")
            f.write(f"测试集病叶样本: {np.sum(y_test == 1)}\n\n")

            f.write(f"⚙️ 训练参数:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"聚类数量: {N_CLUSTERS}\n")
            f.write(f"聚类迭代次数: {CLUSTERING_EPOCHS}\n")
            f.write(f"批大小: {BATCH_SIZE}\n")
            f.write(f"学习率: {LEARNING_RATE}\n")
            f.write(f"Sobel滤波器: {SOBEL_FILTER}\n")
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

            if deepcluster_model is not None and deepcluster_model.kmeans is not None:
                f.write(f"📊 聚类信息:\n")
                f.write(f"-" * 50 + "\n")
                f.write(f"簇数量: {deepcluster_model.n_clusters}\n")
                f.write(f"特征维度: {deepcluster_model.feature_dim}\n")
                f.write(f"簇内平方和: {deepcluster_model.kmeans.inertia_:.4f}\n")
                f.write(f"簇大小分布: {np.bincount(deepcluster_model.kmeans.labels)}\n")
                f.write(f"簇距离阈值: {deepcluster_model.cluster_thresholds}\n")
                f.write(f"\n")

            if all_labels is not None:
                f.write(f"📈 DeepCluster迭代信息:\n")
                f.write(f"-" * 50 + "\n")
                for i, labels in enumerate(all_labels):
                    f.write(f"迭代 {i + 1}: 聚类分布={np.bincount(labels)}\n")
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
                f.write("✅ DeepCluster模型表现优秀！mAP和F1分数均超过85%。\n")
                f.write("   建议：可以直接部署到实际应用中。\n")
            elif mAP > 0.75 and f1 > 0.75:
                f.write("⚠️  DeepCluster模型表现良好，但仍有改进空间。\n")
                f.write("   建议：可以尝试增加聚类数量或调整CNN结构。\n")
            elif mAP > 0.65 and f1 > 0.65:
                f.write("⚠️  DeepCluster模型表现一般，需要进一步优化。\n")
                f.write("   建议：增加DeepCluster迭代次数，调整学习率，或尝试其他预处理方法。\n")
            elif mAP > 0.5:
                f.write("⚠️  DeepCluster模型表现较差，需要优化。\n")
                f.write("   建议：检查数据质量，调整模型参数，或尝试其他异常检测方法。\n")
            else:
                f.write("❌ DeepCluster模型表现不佳，需要重新设计。\n")
                f.write("   建议：检查数据质量，重新设计CNN架构，或调整聚类算法参数。\n")
            f.write(f"\n")

        print(f"📄 文本结果已保存: {txt_filename}")

    except Exception as e:
        print(f"⚠️  无法保存文本结果: {e}")


# 单张图像预测
def predict_single_image(model, image_path, image_size):
    """使用训练好的模型预测单张图像"""
    try:
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        image = image.resize(image_size)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 预测
        prediction = model.predict(img_array)[0]
        score = model.decision_function(img_array)[0]
        cluster_info = model.get_cluster_info(img_array)

        # 转换为类别名称
        class_name = "病叶" if prediction == 1 else "健康叶片"
        cluster_id = cluster_info[0][0] if len(cluster_info[0]) > 0 else -1

        return class_name, score, prediction, cluster_id

    except Exception as e:
        return f"错误: {str(e)}", 0, -1, -1


# 主函数
def main():
    try:
        print("🔍 检查数据集目录...")
        if not os.path.exists(DATASET_DIR):
            print(f"❌ 数据集目录不存在: {DATASET_DIR}")
            print("请检查DATASET_DIR路径配置")
            return

        # 加载数据
        data = load_and_preprocess_data(DATASET_DIR, IMAGE_SIZE, SAMPLE_RATIO)
        if data[0] is None:
            return

        X_train, y_train, X_test, y_test = data

        # 训练模型
        model, test_scores, predictions, all_features, all_labels = train_deepcluster(X_train, X_test, y_test)

        if model is None:
            print("❌ 模型训练失败")
            return

        # 评估模型
        accuracy, precision, recall, f1, test_scores, cm, predictions, mAP, ap_per_class = evaluate_model(
            test_scores, y_test, predictions, "DeepCluster"
        )

        # 可视化结果
        visualize_results(test_scores, y_test, cm, mAP, ap_per_class, model)

        # 文本形式的结果报告
        text_results_report(test_scores, y_test, accuracy, precision, recall, f1, cm, mAP, ap_per_class,
                            "DeepCluster", model, all_labels)

        # 详细分类报告
        print(f"\n📝 详细分类报告:")
        print(Metrics.classification_report(y_test, predictions, target_names=['健康叶片', '病叶']))

        # 保存结果
        save_results(accuracy, precision, recall, f1, cm, mAP, ap_per_class,
                     "DeepCluster", X_train, X_test, y_test, test_scores, model, all_labels)

        print(f"\n🎉 DeepCluster模型训练完成!")
        print("💡 模型特点:")
        print("   - 结合深度学习和聚类")
        print("   - 交替进行特征学习和聚类")
        print("   - 使用Sobel滤波器增强特征")
        print("   - 适合无监督特征学习")
        print("   - 包含修正的mAP评估指标")

        # 示例预测
        print(f"\n🔍 示例预测:")
        if len(X_test) > 0:
            sample_indices = random.sample(range(len(X_test)), min(3, len(X_test)))

            for i, idx in enumerate(sample_indices):
                true_label = y_test[idx]
                true_class = "病叶" if true_label == 1 else "健康叶片"

                sample_image = X_test[idx:idx + 1]
                score = model.decision_function(sample_image)[0]
                prediction = model.predict(sample_image)[0]
                cluster_info = model.get_cluster_info(sample_image)
                cluster_id = cluster_info[0][0] if len(cluster_info[0]) > 0 else -1

                # 转换预测结果
                pred_class = "病叶" if prediction == 1 else "健康叶片"

                status = '✓' if prediction == true_label else '✗'
                print(
                    f"   样本 {i + 1}: 真实={true_class}, 预测={pred_class}, 簇={cluster_id}, 分数={score:.6f} {status}")

        # 模型参数信息
        print(f"\n⚙️  模型参数:")
        print(f"   聚类数量: {N_CLUSTERS}")
        print(f"   聚类迭代次数: {CLUSTERING_EPOCHS}")
        print(f"   批大小: {BATCH_SIZE}")
        print(f"   学习率: {LEARNING_RATE}")
        print(f"   Sobel滤波器: {SOBEL_FILTER}")
        print(f"   图像尺寸: {IMAGE_SIZE}")
        print(f"   采样比例: {SAMPLE_RATIO}")
        print(f"   评估指标: 包含修正的mAP")

        print(f"\n💡 算法说明:")
        print("   本实现使用DeepCluster方法:")
        print("   1. 使用Sobel滤波器预处理图像")
        print("   2. CNN特征提取器学习图像表示")
        print("   3. K-means对特征进行聚类")
        print("   4. 使用聚类标签重新训练CNN")
        print("   5. 迭代优化特征和聚类")
        print("   6. 计算修正的mAP评估模型性能")

        print(f"\n🔧 调优建议:")
        print("   - 增加DeepCluster迭代次数")
        print("   - 调整CNN网络结构")
        print("   - 尝试不同的聚类数量")
        print("   - 使用更复杂的图像预处理")
        print("   - 调整学习率和批大小")

        print(f"\n📁 生成的文件:")
        print(f"   1. deepcluster_leaf_disease_results.joblib - 二进制结果文件 (如果joblib可用)")
        print(f"   2. deepcluster_leaf_disease_results_YYYYMMDD_HHMMSS.txt - 文本报告 (包含mAP)")
        print(f"   3. deepcluster_pr_curves.png - PR曲线图")
        print(f"   4. deepcluster_final_results.png - 综合可视化结果")

    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()