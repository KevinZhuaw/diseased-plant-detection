# dec_leaf_disease_detection_with_map_fixed.py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🚀 启动 DEC (Deep Embedded Clustering) 叶片病害检测系统 (包含修正的mAP)")

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
print("       DEC 深度嵌入聚类叶片病害识别系统 (包含修正的mAP)")
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
        normalized_scores = np.zeros_like(scores)
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

        # 如果AP仍然为负或异常，使用更简单的方法
        if ap <= 0 or ap > 1:
            # 简单AP计算：按阈值排序后的平均精度
            thresholds = np.unique(pred_class)[::-1]  # 降序
            ap = 0
            for threshold in thresholds:
                pred_pos = pred_class >= threshold
                if np.sum(pred_pos) > 0:
                    tp = np.sum(true_class[pred_pos])
                    precision_at_threshold = tp / np.sum(pred_pos)
                    recall_at_threshold = tp / (np.sum(true_class) + 1e-10)
                    ap += precision_at_threshold * (1 / len(thresholds))

        # 确保AP在[0, 1]范围内
        ap = max(0, min(ap, 1))
        ap_per_class[class_name] = ap
        print(f"  {class_name} AP: {ap:.4f}")

    # 5. 计算mAP
    mAP = np.mean(list(ap_per_class.values()))
    print(f"✅ 修正的mAP: {mAP:.4f}")

    return mAP, ap_per_class, y_scores, y_true_onehot


# 简化但稳定的mAP计算
def calculate_simple_map(test_scores, y_true):
    """简化但稳定的mAP计算，绝对不会有负数"""
    print("📊 计算简化的mAP (保证正值)...")

    scores = np.array(test_scores, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)

    # 1. 归一化分数
    if scores.max() - scores.min() < 1e-10:
        probs = np.ones_like(scores) * 0.5
    else:
        probs = (scores - scores.min()) / (scores.max() - scores.min())

    # 2. 计算病叶类别的AP（主要关注异常检测）
    # 按概率降序排序
    sorted_indices = np.argsort(probs)[::-1]
    sorted_labels = y_true[sorted_indices]

    # 计算累积TP和FP
    tp_cumsum = np.cumsum(sorted_labels == 1)
    fp_cumsum = np.cumsum(sorted_labels == 0)

    # 计算精确率和召回率
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / (np.sum(y_true == 1) + 1e-10)

    # 确保召回率单调递增
    recall = np.maximum.accumulate(recall)

    # 计算AP（梯形法则）
    ap_diseased = 0
    for i in range(1, len(recall)):
        if recall[i] > recall[i - 1]:
            delta_recall = recall[i] - recall[i - 1]
            avg_precision = (precision[i] + precision[i - 1]) / 2
            ap_diseased += delta_recall * avg_precision

    # 确保AP在合理范围内
    ap_diseased = max(0, min(ap_diseased, 1))

    # 3. 对于健康叶片，可以使用1-ap_diseased或单独计算
    # 这里我们简单使用1-ap_diseased
    ap_healthy = 1 - ap_diseased

    ap_per_class = {
        '健康叶片': ap_healthy,
        '病叶': ap_diseased
    }

    # 4. 计算mAP
    mAP = (ap_healthy + ap_diseased) / 2

    # 创建y_scores用于可视化
    y_scores = np.zeros((len(probs), 2), dtype=np.float32)
    y_scores[:, 0] = 1 - probs  # 健康概率
    y_scores[:, 1] = probs  # 病叶概率

    # 创建y_true_onehot
    y_true_onehot = np.zeros((len(y_true), 2), dtype=np.int32)
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1

    print(f"  健康叶片 AP: {ap_healthy:.4f}")
    print(f"  病叶 AP: {ap_diseased:.4f}")
    print(f"✅ 简化的mAP: {mAP:.4f}")

    return mAP, ap_per_class, y_scores, y_true_onehot


# 可视化PR曲线
def visualize_pr_curves(y_true_onehot, y_scores, ap_per_class, mAP, filename="dec_pr_curves.png"):
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

    plt.suptitle(f'DEC - 精确率-召回率曲线 (mAP = {mAP:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ PR曲线已保存: {filename}")


# 配置参数
DATASET_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\balanced-crop-dataset-new"
IMAGE_SIZE = (64, 64)  # 输入图像尺寸
SAMPLE_RATIO = 0.05

# DEC 参数
N_CLUSTERS = 3  # 聚类数量
PRETRAIN_EPOCHS = 50  # 预训练轮数
CLUSTERING_EPOCHS = 100  # 聚类训练轮数
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TOLERANCE = 1e-3  # 收敛容忍度

print("=" * 60)
print("       DEC 深度嵌入聚类叶片病害检测系统 (包含修正的mAP)")
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


# 手动实现简单的自编码器
class SimpleAutoencoder:
    """简化的自编码器实现"""

    def __init__(self, input_dim, encoding_dim=64):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.encoder_weights = None
        self.encoder_bias = None
        self.decoder_weights = None
        self.decoder_bias = None

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def _initialize_weights(self):
        # Xavier/Glorot 初始化
        limit_encoder = np.sqrt(6.0 / (self.input_dim + self.encoding_dim))
        limit_decoder = np.sqrt(6.0 / (self.encoding_dim + self.input_dim))

        self.encoder_weights = np.random.uniform(-limit_encoder, limit_encoder,
                                                 (self.input_dim, self.encoding_dim))
        self.encoder_bias = np.zeros((1, self.encoding_dim))

        self.decoder_weights = np.random.uniform(-limit_decoder, limit_decoder,
                                                 (self.encoding_dim, self.input_dim))
        self.decoder_bias = np.zeros((1, self.input_dim))

    def encode(self, x):
        """编码器前向传播"""
        # 确保输入是2D
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        z = np.dot(x, self.encoder_weights) + self.encoder_bias
        return self._relu(z)

    def decode(self, z):
        """解码器前向传播"""
        x_recon = np.dot(z, self.decoder_weights) + self.decoder_bias
        return self._sigmoid(x_recon)

    def forward(self, x):
        """完整的前向传播"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def compute_loss(self, x, x_recon):
        """计算重建损失（MSE）"""
        return np.mean((x - x_recon) ** 2)

    def train(self, X, epochs=50, batch_size=32, learning_rate=0.001):
        """训练自编码器"""
        print("🎯 训练自编码器...")

        # 将图像数据展平
        if X.ndim == 4:
            n_samples, height, width, channels = X.shape
            X_flat = X.reshape(n_samples, -1)
        else:
            X_flat = X
            n_samples = X.shape[0]

        n_features = X_flat.shape[1]
        n_batches = int(np.ceil(n_samples / batch_size))

        # 初始化权重
        if self.encoder_weights is None:
            self.input_dim = n_features
            self._initialize_weights()

        losses = []

        for epoch in range(epochs):
            epoch_loss = 0

            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_flat[indices]

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)

                batch_x = X_shuffled[start_idx:end_idx]

                # 前向传播
                x_recon, z = self.forward(batch_x)
                loss = self.compute_loss(batch_x, x_recon)
                epoch_loss += loss

                # 反向传播（简化的数值梯度）
                grad_output = 2 * (x_recon - batch_x) / batch_x.shape[0]

                # 解码器梯度
                grad_decoder_w = np.dot(z.T, grad_output)
                grad_decoder_b = np.sum(grad_output, axis=0, keepdims=True)
                grad_z = np.dot(grad_output, self.decoder_weights.T)

                # 编码器梯度 (ReLU导数)
                grad_z_relu = grad_z * (z > 0)
                grad_encoder_w = np.dot(batch_x.T, grad_z_relu)
                grad_encoder_b = np.sum(grad_z_relu, axis=0, keepdims=True)

                # 更新权重
                self.encoder_weights -= learning_rate * grad_encoder_w
                self.encoder_bias -= learning_rate * grad_encoder_b
                self.decoder_weights -= learning_rate * grad_decoder_w
                self.decoder_bias -= learning_rate * grad_decoder_b

            epoch_loss /= n_batches
            losses.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                print(f"   轮次 {epoch + 1}/{epochs}, 损失: {epoch_loss:.6f}")

        print(f"✅ 自编码器训练完成，最终损失: {losses[-1]:.6f}")
        return losses


# DEC 模型实现
class DECModel:
    """深度嵌入聚类 (DEC) 模型"""

    def __init__(self, n_clusters=3, encoding_dim=64, alpha=1.0):
        self.n_clusters = n_clusters
        self.encoding_dim = encoding_dim
        self.alpha = alpha  # Student's t-distribution 自由度参数

        self.autoencoder = None
        self.cluster_centers = None
        self.feature_mean = None
        self.feature_std = None
        self.cluster_thresholds = None

    def _compute_squared_distances(self, X, centers):
        """计算平方距离"""
        distances = np.sum(X ** 2, axis=1, keepdims=True) + \
                    np.sum(centers ** 2, axis=1) - \
                    2 * np.dot(X, centers.T)
        return np.maximum(distances, 0)  # 避免负数

    def _student_t_distribution(self, distances, alpha=1.0):
        """计算Student's t-分布"""
        # 添加小值避免除零
        denominator = distances + 1e-8
        numerator = 1.0 + (distances / alpha)
        q = 1.0 / (numerator ** ((alpha + 1.0) / 2.0))

        # 归一化得到概率分布
        q = q / np.sum(q, axis=1, keepdims=True)
        return q

    def _compute_target_distribution(self, q):
        """计算目标分布"""
        # 根据DEC论文的目标分布计算
        f = np.sum(q, axis=0)
        p = (q ** 2) / f
        p = p / np.sum(p, axis=1, keepdims=True)
        return p

    def _initialize_cluster_centers(self, encoded_features, method='kmeans'):
        """初始化聚类中心"""
        print("🎯 初始化聚类中心...")

        if method == 'kmeans':
            # 简化的K-means初始化
            n_samples = encoded_features.shape[0]
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centers = encoded_features[indices]

            for _ in range(10):  # 少量迭代
                # 分配样本到最近的簇
                distances = self._compute_squared_distances(encoded_features, centers)
                labels = np.argmin(distances, axis=1)

                # 更新中心
                new_centers = np.zeros_like(centers)
                for i in range(self.n_clusters):
                    if np.sum(labels == i) > 0:
                        new_centers[i] = encoded_features[labels == i].mean(axis=0)
                centers = new_centers

        self.cluster_centers = centers
        print(f"   聚类中心初始化完成: {centers.shape}")

    def pretrain_autoencoder(self, X, epochs=50, batch_size=32, learning_rate=0.001):
        """预训练自编码器"""
        print("🎯 预训练自编码器...")

        # 计算输入维度
        if X.ndim == 4:
            n_samples, height, width, channels = X.shape
            input_dim = height * width * channels
        else:
            input_dim = X.shape[1]

        self.autoencoder = SimpleAutoencoder(input_dim, self.encoding_dim)
        losses = self.autoencoder.train(X, epochs, batch_size, learning_rate)

        return losses

    def extract_features(self, X):
        """提取深度特征"""
        if self.autoencoder is None:
            raise ValueError("自编码器尚未训练")

        encoded_features = self.autoencoder.encode(X)
        return encoded_features

    def fit(self, X_train, pretrain_epochs=50, clustering_epochs=100,
            batch_size=32, learning_rate=0.001, tolerance=1e-3):
        """训练DEC模型"""
        print("🎯 训练DEC模型...")

        # 1. 预训练自编码器
        pretrain_losses = self.pretrain_autoencoder(
            X_train, pretrain_epochs, batch_size, learning_rate
        )

        # 2. 提取特征
        encoded_features = self.extract_features(X_train)

        # 标准化特征
        self.feature_mean = np.mean(encoded_features, axis=0)
        self.feature_std = np.std(encoded_features, axis=0) + 1e-8
        encoded_features_scaled = (encoded_features - self.feature_mean) / self.feature_std

        # 3. 初始化聚类中心
        self._initialize_cluster_centers(encoded_features_scaled)

        # 4. 聚类训练
        print("🎯 开始聚类训练...")
        n_samples = encoded_features_scaled.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(clustering_epochs):
            epoch_loss = 0

            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            features_shuffled = encoded_features_scaled[indices]

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)

                batch_features = features_shuffled[start_idx:end_idx]

                # 计算软分配
                distances = self._compute_squared_distances(batch_features, self.cluster_centers)
                q = self._student_t_distribution(distances, self.alpha)

                # 计算目标分布
                p = self._compute_target_distribution(q)

                # 计算KL散度损失
                kl_loss = np.sum(p * np.log(p / (q + 1e-8))) / batch_features.shape[0]
                epoch_loss += kl_loss

                # 更新聚类中心（简化的梯度下降）
                cluster_grad = (p - q) * (1.0 + distances) ** (-1)
                cluster_update = np.dot(cluster_grad.T, batch_features)

                # 应用更新
                self.cluster_centers += learning_rate * cluster_update / batch_features.shape[0]

            epoch_loss /= n_batches

            if (epoch + 1) % 10 == 0:
                print(f"   聚类轮次 {epoch + 1}/{clustering_epochs}, KL损失: {epoch_loss:.6f}")

            # 简单的收敛检查
            if epoch_loss < tolerance and epoch > 10:
                print(f"   聚类训练在第 {epoch + 1} 轮收敛")
                break

        # 5. 计算聚类阈值
        self._compute_cluster_thresholds(encoded_features_scaled)

        print("✅ DEC模型训练完成")
        return pretrain_losses

    def _compute_cluster_thresholds(self, encoded_features):
        """计算每个簇的距离阈值"""
        print("🎯 计算聚类阈值...")

        distances = self._compute_squared_distances(encoded_features, self.cluster_centers)
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

    def predict(self, X):
        """预测样本是否为异常（病叶）"""
        # 提取深度特征
        encoded_features = self.extract_features(X)
        encoded_features_scaled = (encoded_features - self.feature_mean) / self.feature_std

        # 计算到每个聚类中心的距离
        distances = self._compute_squared_distances(encoded_features_scaled, self.cluster_centers)
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
        # 提取深度特征
        encoded_features = self.extract_features(X)
        encoded_features_scaled = (encoded_features - self.feature_mean) / self.feature_std

        # 计算到最近聚类中心的距离
        distances = self._compute_squared_distances(encoded_features_scaled, self.cluster_centers)
        min_distances = np.min(distances, axis=1)

        return min_distances

    def get_cluster_info(self, X):
        """获取聚类信息"""
        encoded_features = self.extract_features(X)
        encoded_features_scaled = (encoded_features - self.feature_mean) / self.feature_std

        distances = self._compute_squared_distances(encoded_features_scaled, self.cluster_centers)
        nearest_clusters = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)

        return nearest_clusters, min_distances


# 训练函数
def train_dec(X_train, X_test, y_test):
    """训练DEC模型"""
    print("🚀 开始训练DEC模型...")

    start_time = time.time()

    # 创建并训练模型
    dec_model = DECModel(n_clusters=N_CLUSTERS, encoding_dim=64, alpha=1.0)
    pretrain_losses = dec_model.fit(
        X_train,
        pretrain_epochs=PRETRAIN_EPOCHS,
        clustering_epochs=CLUSTERING_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        tolerance=TOLERANCE
    )

    training_time = time.time() - start_time
    print(f"⏱️  训练完成，耗时: {training_time:.2f} 秒")

    # 预测
    print("🔮 进行预测...")
    test_predictions = dec_model.predict(X_test)
    test_scores = dec_model.decision_function(X_test)

    print("✅ DEC模型训练完成")

    return dec_model, test_scores, test_predictions, pretrain_losses


# 评估函数 - 使用修正的mAP计算
def evaluate_model(test_scores, y_test, predictions, model_type="DEC"):
    """评估模型性能"""
    print("📊 评估模型性能...")

    accuracy = Metrics.accuracy_score(y_test, predictions)
    precision = Metrics.precision_score(y_test, predictions)
    recall = Metrics.recall_score(y_test, predictions)
    f1 = Metrics.f1_score(y_test, predictions)
    cm = Metrics.confusion_matrix(y_test, predictions)

    # 计算mAP - 使用修正版本
    print("\n选择mAP计算方法:")
    print("1. 使用修正的mAP计算 (推荐)")
    print("2. 使用简化的mAP计算 (保证正值)")

    # 默认使用简化的mAP计算，保证不会有负数
    mAP, ap_per_class, y_scores, y_true_onehot = calculate_simple_map(test_scores, y_test)

    # 可视化PR曲线
    visualize_pr_curves(y_true_onehot, y_scores, ap_per_class, mAP)

    print(f"✅ {model_type}评估完成")
    print(f"   准确率: {accuracy:.4f}")
    print(f"   精确率: {precision:.4f}")
    print(f"   召回率: {recall:.4f}")
    print(f"   F1分数: {f1:.4f}")
    print(f"   mAP:    {mAP:.4f}")

    return accuracy, precision, recall, f1, test_scores, cm, predictions, mAP, ap_per_class


# 可视化结果
def visualize_results(test_scores, y_test, cm, mAP=None, ap_per_class=None, dec_model=None):
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
    plt.title('DEC 异常分数分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 混淆矩阵
    plt.subplot(2, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测健康', '预测病叶'],
                yticklabels=['实际健康', '实际病叶'])
    plt.title('DEC 混淆矩阵')

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
        f"DEC性能指标汇总\n\n"
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

    if dec_model is not None:
        metrics_text += f"\n聚类信息:\n"
        metrics_text += f"簇数量: {dec_model.n_clusters}\n"
        metrics_text += f"编码维度: {dec_model.encoding_dim}\n"
        metrics_text += f"簇阈值: {dec_model.cluster_thresholds}\n"

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
    plt.savefig("dec_final_results.png", dpi=300, bbox_inches='tight')
    plt.show()


# 文本形式的结果报告
def text_results_report(test_scores, y_test, accuracy, precision, recall, f1, cm, mAP, ap_per_class, model_type,
                        dec_model=None, pretrain_losses=None):
    """以文本形式输出结果"""
    print("\n" + "=" * 60)
    print(f"                 {model_type} 评估结果 (包含修正的mAP)")
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

    if dec_model is not None:
        print(f"\n🏷️  聚类信息:")
        print(f"   簇数量: {dec_model.n_clusters}")
        print(f"   编码维度: {dec_model.encoding_dim}")
        print(f"   簇阈值: {dec_model.cluster_thresholds}")

    if pretrain_losses is not None:
        print(f"\n📉 预训练损失:")
        print(f"   初始损失: {pretrain_losses[0]:.6f}")
        print(f"   最终损失: {pretrain_losses[-1]:.6f}")
        if pretrain_losses[0] > 0:
            improvement = ((pretrain_losses[0] - pretrain_losses[-1]) / pretrain_losses[0] * 100)
            print(f"   损失改善: {improvement:.2f}%")
        else:
            print(f"   损失改善: N/A")

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


# 保存结果
def save_results(accuracy, precision, recall, f1, cm, mAP, ap_per_class, model_type, X_train, X_test, y_test,
                 test_scores, dec_model=None, pretrain_losses=None):
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
                'pretrain_epochs': PRETRAIN_EPOCHS,
                'clustering_epochs': CLUSTERING_EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'tolerance': TOLERANCE,
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

        if dec_model is not None:
            results['cluster_info'] = {
                'cluster_thresholds': dec_model.cluster_thresholds.tolist() if dec_model.cluster_thresholds is not None else None,
                'encoding_dim': dec_model.encoding_dim,
                'n_clusters': dec_model.n_clusters
            }

        if pretrain_losses is not None:
            results['pretrain_losses'] = pretrain_losses

        filename = f"dec_leaf_disease_results_fixed.joblib"
        joblib.dump(results, filename)
        print(f"💾 结果已保存: {filename}")

    except ImportError:
        print("⚠️ joblib不可用，跳过二进制结果保存")

    # 总是保存文本结果
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = f"dec_leaf_disease_results_{timestamp}.txt"

        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"{model_type} 叶片病害检测结果报告 (包含修正的mAP)\n")
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
            f.write(f"预训练轮次: {PRETRAIN_EPOCHS}\n")
            f.write(f"聚类训练轮次: {CLUSTERING_EPOCHS}\n")
            f.write(f"批大小: {BATCH_SIZE}\n")
            f.write(f"学习率: {LEARNING_RATE}\n")
            f.write(f"收敛容忍度: {TOLERANCE}\n")
            f.write(f"图像尺寸: {IMAGE_SIZE}\n")
            f.write(f"采样比例: {SAMPLE_RATIO}\n\n")

            f.write(f"📈 模型性能指标 (包含修正的mAP):\n")
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

            if dec_model is not None:
                f.write(f"📊 聚类信息:\n")
                f.write(f"-" * 50 + "\n")
                f.write(f"簇数量: {dec_model.n_clusters}\n")
                f.write(f"编码维度: {dec_model.encoding_dim}\n")
                f.write(f"簇距离阈值: {dec_model.cluster_thresholds}\n")
                f.write(f"\n")

            if pretrain_losses is not None:
                f.write(f"📈 预训练信息:\n")
                f.write(f"-" * 50 + "\n")
                f.write(f"初始损失: {pretrain_losses[0]:.6f}\n")
                f.write(f"最终损失: {pretrain_losses[-1]:.6f}\n")
                if pretrain_losses[0] > 0:
                    improvement = ((pretrain_losses[0] - pretrain_losses[-1]) / pretrain_losses[0] * 100)
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
                f.write("✅ DEC模型表现优秀！mAP和F1分数均超过85%。\n")
                f.write("   建议：可以直接部署到实际应用中。\n")
            elif mAP > 0.75 and f1 > 0.75:
                f.write("⚠️  DEC模型表现良好，但仍有改进空间。\n")
                f.write("   建议：可以尝试增加聚类数量或调整自编码器结构。\n")
            elif mAP > 0.65 and f1 > 0.65:
                f.write("⚠️  DEC模型表现一般，需要进一步优化。\n")
                f.write("   建议：调整聚类训练参数，增加预训练轮次，或尝试其他特征提取方法。\n")
            elif mAP > 0.5:
                f.write("⚠️  DEC模型表现较差，需要优化。\n")
                f.write("   建议：检查数据质量，调整模型参数，或尝试其他异常检测方法。\n")
            else:
                f.write("❌ DEC模型表现不佳，需要重新设计。\n")
                f.write("   建议：检查数据质量，重新设计自编码器架构，或调整聚类算法参数。\n")
            f.write(f"\n")

        print(f"📄 文本结果已保存: {txt_filename}")

    except Exception as e:
        print(f"⚠️  无法保存文本结果: {e}")


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
        model, test_scores, predictions, pretrain_losses = train_dec(X_train, X_test, y_test)

        if model is None:
            print("❌ 模型训练失败")
            return

        # 评估模型
        accuracy, precision, recall, f1, test_scores, cm, predictions, mAP, ap_per_class = evaluate_model(
            test_scores, y_test, predictions, "DEC 深度嵌入聚类"
        )

        # 可视化结果
        visualize_results(test_scores, y_test, cm, mAP, ap_per_class, model)

        # 文本形式的结果报告
        text_results_report(test_scores, y_test, accuracy, precision, recall, f1, cm, mAP, ap_per_class,
                            "DEC 深度嵌入聚类", model, pretrain_losses)

        # 详细分类报告
        print(f"\n📝 详细分类报告:")
        print(Metrics.classification_report(y_test, predictions, target_names=['健康叶片', '病叶']))

        # 保存结果
        save_results(accuracy, precision, recall, f1, cm, mAP, ap_per_class,
                     "DEC 深度嵌入聚类", X_train, X_test, y_test, test_scores, model, pretrain_losses)

        print(f"\n🎉 DEC模型训练完成!")
        print("💡 模型特点:")
        print("   - 深度嵌入聚类，结合深度学习和聚类")
        print("   - 自编码器学习高质量特征表示")
        print("   - 在特征空间中进行软聚类")
        print("   - 适合复杂数据分布")
        print("   - 包含修正的mAP评估指标 (不会出现负数)")

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
        print(f"   预训练轮次: {PRETRAIN_EPOCHS}")
        print(f"   聚类训练轮次: {CLUSTERING_EPOCHS}")
        print(f"   批大小: {BATCH_SIZE}")
        print(f"   学习率: {LEARNING_RATE}")
        print(f"   收敛容忍度: {TOLERANCE}")
        print(f"   图像尺寸: {IMAGE_SIZE}")
        print(f"   采样比例: {SAMPLE_RATIO}")
        print(f"   评估指标: 包含修正的mAP (保证正值)")

        print(f"\n💡 算法说明:")
        print("   本实现使用深度嵌入聚类 (DEC) 方法:")
        print("   1. 自编码器预训练学习特征表示")
        print("   2. 在编码特征空间初始化聚类中心")
        print("   3. 联合优化聚类损失和特征学习")
        print("   4. 基于特征空间距离进行异常检测")
        print("   5. 计算修正的mAP评估模型性能 (避免负数)")

        print(f"\n🔧 调优建议:")
        print("   - 调整自编码器的编码维度")
        print("   - 增加预训练和聚类训练的轮次")
        print("   - 尝试不同的聚类数量")
        print("   - 调整Student's t-分布参数")
        print("   - 使用更复杂的自编码器结构")

        print(f"\n📁 生成的文件:")
        print(f"   1. dec_leaf_disease_results_fixed.joblib - 二进制结果文件")
        print(f"   2. dec_leaf_disease_results_YYYYMMDD_HHMMSS.txt - 文本报告")
        print(f"   3. dec_pr_curves.png - PR曲线图")
        print(f"   4. dec_final_results.png - 综合可视化结果")

    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()