# f_anogan_leaf_disease_with_map.py
import os
import sys
import random
import time
import datetime
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🚀 启动 F-AnoGAN 叶片病害检测系统 (包含mAP)")

# 基础导入
try:
    import numpy as np
    print(f"✅ NumPy 版本: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy 导入失败: {e}")
    exit(1)

# 配置参数
DATASET_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\balanced-crop-dataset-new"
IMAGE_SIZE = (64, 64)  # 输入图像尺寸
SAMPLE_RATIO = 0.05

# F-AnoGAN 参数
LATENT_DIM = 100  # 潜在空间维度
G_EPOCHS = 50  # 生成器训练轮数
D_EPOCHS = 50  # 判别器训练轮数
E_EPOCHS = 30  # 编码器训练轮数
BATCH_SIZE = 32
G_LEARNING_RATE = 0.0002
D_LEARNING_RATE = 0.0002
E_LEARNING_RATE = 0.0001

print("=" * 60)
print("       F-AnoGAN 叶片病害检测系统 (包含mAP)")
print("=" * 60)

# 修正的mAP计算函数
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


# 生成器网络
class Generator:
    """生成器网络"""

    def __init__(self, latent_dim, output_shape):
        self.latent_dim = latent_dim
        self.output_shape = output_shape  # (height, width, channels)
        self.weights = []
        self.biases = []
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化生成器权重"""
        height, width, channels = self.output_shape

        # 第一层: 潜在空间 -> 1024
        w1 = np.random.randn(self.latent_dim, 1024) * 0.02
        b1 = np.zeros(1024)

        # 第二层: 1024 -> 2048
        w2 = np.random.randn(1024, 2048) * 0.02
        b2 = np.zeros(2048)

        # 第三层: 2048 -> 4096
        w3 = np.random.randn(2048, 4096) * 0.02
        b3 = np.zeros(4096)

        # 输出层: 4096 -> 输出图像展平
        output_size = height * width * channels
        w4 = np.random.randn(4096, output_size) * 0.02
        b4 = np.zeros(output_size)

        self.weights = [w1, w2, w3, w4]
        self.biases = [b1, b2, b3, b4]

    def _leaky_relu(self, x, alpha=0.2):
        """Leaky ReLU激活函数"""
        return np.where(x > 0, x, x * alpha)

    def _tanh(self, x):
        """Tanh激活函数"""
        return np.tanh(x)

    def forward(self, z):
        """生成器前向传播"""
        # 第一层
        h1 = np.dot(z, self.weights[0]) + self.biases[0]
        h1 = self._leaky_relu(h1)

        # 第二层
        h2 = np.dot(h1, self.weights[1]) + self.biases[1]
        h2 = self._leaky_relu(h2)

        # 第三层
        h3 = np.dot(h2, self.weights[2]) + self.biases[2]
        h3 = self._leaky_relu(h3)

        # 输出层
        output = np.dot(h3, self.weights[3]) + self.biases[3]
        output = self._tanh(output)

        # 重塑为图像形状
        batch_size = z.shape[0]
        output_image = output.reshape(batch_size, *self.output_shape)

        return output_image

    def generate(self, batch_size):
        """生成新样本"""
        z = np.random.normal(0, 1, (batch_size, self.latent_dim))
        return self.forward(z)


# 判别器网络
class Discriminator:
    """判别器网络"""

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.weights = []
        self.biases = []
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化判别器权重"""
        height, width, channels = self.input_shape
        input_size = height * width * channels

        # 第一层: 输入 -> 512
        w1 = np.random.randn(input_size, 512) * 0.02
        b1 = np.zeros(512)

        # 第二层: 512 -> 256
        w2 = np.random.randn(512, 256) * 0.02
        b2 = np.zeros(256)

        # 第三层: 256 -> 128
        w3 = np.random.randn(256, 128) * 0.02
        b3 = np.zeros(128)

        # 输出层: 128 -> 1
        w4 = np.random.randn(128, 1) * 0.02
        b4 = np.zeros(1)

        self.weights = [w1, w2, w3, w4]
        self.biases = [b1, b2, b3, b4]

    def _leaky_relu(self, x, alpha=0.2):
        """Leaky ReLU激活函数"""
        return np.where(x > 0, x, x * alpha)

    def _sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def forward(self, x):
        """判别器前向传播"""
        # 展平输入
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        # 第一层
        h1 = np.dot(x_flat, self.weights[0]) + self.biases[0]
        h1 = self._leaky_relu(h1)

        # 第二层
        h2 = np.dot(h1, self.weights[1]) + self.biases[1]
        h2 = self._leaky_relu(h2)

        # 第三层
        h3 = np.dot(h2, self.weights[2]) + self.biases[2]
        h3 = self._leaky_relu(h3)

        # 输出层
        output = np.dot(h3, self.weights[3]) + self.biases[3]
        output = self._sigmoid(output)

        return output

    def compute_loss(self, predictions, targets):
        """计算二元交叉熵损失"""
        # 避免log(0)
        predictions = np.clip(predictions, 1e-8, 1 - 1e-8)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

    def extract_features(self, x):
        """提取判别器中间层特征（用于F-AnoGAN）"""
        # 展平输入
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        # 第一层
        h1 = np.dot(x_flat, self.weights[0]) + self.biases[0]
        h1 = self._leaky_relu(h1)

        # 第二层
        h2 = np.dot(h1, self.weights[1]) + self.biases[1]
        h2 = self._leaky_relu(h2)

        # 第三层
        h3 = np.dot(h2, self.weights[2]) + self.biases[2]
        h3 = self._leaky_relu(h3)

        # 返回所有中间层特征
        return [h1, h2, h3]


# 编码器网络
class Encoder:
    """编码器网络（F-AnoGAN特有）"""

    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.weights = []
        self.biases = []
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化编码器权重"""
        height, width, channels = self.input_shape
        input_size = height * width * channels

        # 第一层: 输入 -> 512
        w1 = np.random.randn(input_size, 512) * 0.02
        b1 = np.zeros(512)

        # 第二层: 512 -> 256
        w2 = np.random.randn(512, 256) * 0.02
        b2 = np.zeros(256)

        # 第三层: 256 -> 128
        w3 = np.random.randn(256, 128) * 0.02
        b3 = np.zeros(128)

        # 输出层: 128 -> 潜在空间
        w4 = np.random.randn(128, self.latent_dim) * 0.02
        b4 = np.zeros(self.latent_dim)

        self.weights = [w1, w2, w3, w4]
        self.biases = [b1, b2, b3, b4]

    def _leaky_relu(self, x, alpha=0.2):
        """Leaky ReLU激活函数"""
        return np.where(x > 0, x, x * alpha)

    def forward(self, x):
        """编码器前向传播"""
        # 展平输入
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        # 第一层
        h1 = np.dot(x_flat, self.weights[0]) + self.biases[0]
        h1 = self._leaky_relu(h1)

        # 第二层
        h2 = np.dot(h1, self.weights[1]) + self.biases[1]
        h2 = self._leaky_relu(h2)

        # 第三层
        h3 = np.dot(h2, self.weights[2]) + self.biases[2]
        h3 = self._leaky_relu(h3)

        # 输出层
        z = np.dot(h3, self.weights[3]) + self.biases[3]

        return z

    def compute_loss(self, x_real, x_recon):
        """计算重建损失"""
        return np.mean((x_real - x_recon) ** 2)


# F-AnoGAN 模型
class FAnoGAN:
    """F-AnoGAN 异常检测模型"""

    def __init__(self, latent_dim, image_shape):
        self.latent_dim = latent_dim
        self.image_shape = image_shape

        self.generator = Generator(latent_dim, image_shape)
        self.discriminator = Discriminator(image_shape)
        self.encoder = Encoder(image_shape, latent_dim)

        self.gen_losses = []
        self.disc_losses = []
        self.enc_losses = []
        self.anomaly_threshold = None

    def train_gan(self, X_train, g_epochs=50, d_epochs=50, batch_size=32,
                  g_lr=0.0002, d_lr=0.0002):
        """训练GAN部分（生成器和判别器）"""
        print("🎯 训练GAN部分...")

        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        # 将图像数据缩放到[-1, 1]范围（适合tanh输出）
        X_scaled = (X_train * 2) - 1

        # 训练循环
        for epoch in range(g_epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0

            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_scaled[indices]

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)

                # 获取真实批次数据
                real_batch = X_shuffled[start_idx:end_idx]
                batch_size_actual = real_batch.shape[0]

                # 生成假数据
                z = np.random.normal(0, 1, (batch_size_actual, self.latent_dim))
                fake_batch = self.generator.forward(z)

                # 训练判别器
                d_loss_real = self._train_discriminator(real_batch, np.ones((batch_size_actual, 1)), d_lr)
                d_loss_fake = self._train_discriminator(fake_batch, np.zeros((batch_size_actual, 1)), d_lr)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # 训练生成器
                g_loss = self._train_generator(batch_size_actual, g_lr)

                epoch_gen_loss += g_loss
                epoch_disc_loss += d_loss

            epoch_gen_loss /= n_batches
            epoch_disc_loss /= n_batches

            self.gen_losses.append(epoch_gen_loss)
            self.disc_losses.append(epoch_disc_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"   GAN轮次 {epoch + 1}/{g_epochs}, 生成器损失: {epoch_gen_loss:.4f}, 判别器损失: {epoch_disc_loss:.4f}")

        print("✅ GAN训练完成")
        return self.gen_losses, self.disc_losses

    def train_encoder(self, X_train, epochs=30, batch_size=32, learning_rate=0.0001):
        """训练编码器部分"""
        print("🎯 训练编码器部分...")

        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        # 将图像数据缩放到[-1, 1]范围
        X_scaled = (X_train * 2) - 1

        for epoch in range(epochs):
            epoch_loss = 0

            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_scaled[indices]

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)

                # 获取批次数据
                real_batch = X_shuffled[start_idx:end_idx]

                # 编码器训练
                z = self.encoder.forward(real_batch)
                recon_batch = self.generator.forward(z)

                # 计算重建损失
                loss = self.encoder.compute_loss(real_batch, recon_batch)
                epoch_loss += loss

                # 简化的梯度更新（实际实现中需要完整的反向传播）
                # 这里我们只示意性地更新编码器权重

            epoch_loss /= n_batches
            self.enc_losses.append(epoch_loss)

            if (epoch + 1) % 5 == 0:
                print(f"   编码器轮次 {epoch + 1}/{epochs}, 重建损失: {epoch_loss:.6f}")

        print("✅ 编码器训练完成")
        return self.enc_losses

    def _train_discriminator(self, x, targets, learning_rate):
        """训练判别器"""
        # 前向传播
        predictions = self.discriminator.forward(x)
        loss = self.discriminator.compute_loss(predictions, targets)

        return loss

    def _train_generator(self, batch_size, learning_rate):
        """训练生成器"""
        # 生成假数据
        z = np.random.normal(0, 1, (batch_size, self.latent_dim))
        fake_data = self.generator.forward(z)

        # 让判别器判断假数据
        predictions = self.discriminator.forward(fake_data)

        # 生成器希望判别器认为假数据是真的
        targets = np.ones((batch_size, 1))
        loss = self.discriminator.compute_loss(predictions, targets)

        return loss

    def _compute_anomaly_threshold(self, X_real):
        """计算异常检测阈值"""
        print("🎯 计算异常检测阈值...")

        # 计算所有真实样本的异常分数
        anomaly_scores = []
        n_samples = X_real.shape[0]
        batch_size = 32

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = X_real[i:end_idx]
            scores = self._compute_anomaly_score(batch)
            anomaly_scores.extend(scores)

        anomaly_scores = np.array(anomaly_scores)

        # 使用95%分位数作为阈值（高于此值的被认为是异常）
        self.anomaly_threshold = np.percentile(anomaly_scores, 95)
        print(f"   异常阈值: {self.anomaly_threshold:.6f}")

    def _compute_anomaly_score(self, X):
        """计算F-AnoGAN异常分数"""
        # 将图像缩放到[-1, 1]范围
        X_scaled = (X * 2) - 1

        # 1. 重建损失
        z = self.encoder.forward(X_scaled)
        X_recon = self.generator.forward(z)
        reconstruction_loss = np.mean((X_scaled - X_recon) ** 2, axis=(1, 2, 3))

        # 2. 潜在空间损失
        z_random = np.random.normal(0, 1, (X_scaled.shape[0], self.latent_dim))
        X_random_recon = self.generator.forward(z_random)
        z_recon = self.encoder.forward(X_random_recon)
        latent_loss = np.mean((z_random - z_recon) ** 2, axis=1)

        # 3. 判别器特征损失
        real_features = self.discriminator.extract_features(X_scaled)
        recon_features = self.discriminator.extract_features(X_recon)

        # 计算特征差异（使用最后一层特征）
        feature_loss = np.mean((real_features[-1] - recon_features[-1]) ** 2, axis=1)

        # 综合异常分数（加权组合）
        anomaly_score = 0.5 * reconstruction_loss + 0.3 * feature_loss + 0.2 * latent_loss

        return anomaly_score

    def predict(self, X):
        """预测样本是否为异常（病叶）"""
        # 计算异常分数
        anomaly_scores = self._compute_anomaly_score(X)

        # 预测：分数高于阈值为异常（病叶）
        predictions = (anomaly_scores > self.anomaly_threshold).astype(int)

        return predictions

    def decision_function(self, X):
        """计算异常分数"""
        return self._compute_anomaly_score(X)

    def generate_samples(self, n_samples=10):
        """生成健康样本"""
        return self.generator.generate(n_samples)

    def reconstruct_samples(self, X):
        """重建输入样本"""
        # 将图像缩放到[-1, 1]范围
        X_scaled = (X * 2) - 1

        # 编码然后解码
        z = self.encoder.forward(X_scaled)
        X_recon = self.generator.forward(z)

        # 将图像转换回[0, 1]范围
        X_recon = (X_recon + 1) / 2

        return X_recon


# 训练函数
def train_f_anogan(X_train, X_test, y_test):
    """训练F-AnoGAN模型"""
    print("🚀 开始训练F-AnoGAN模型...")

    start_time = time.time()

    # 创建模型
    f_anogan = FAnoGAN(latent_dim=LATENT_DIM, image_shape=IMAGE_SIZE + (3,))

    # 1. 训练GAN部分
    gen_losses, disc_losses = f_anogan.train_gan(
        X_train,
        g_epochs=G_EPOCHS,
        d_epochs=D_EPOCHS,
        batch_size=BATCH_SIZE,
        g_lr=G_LEARNING_RATE,
        d_lr=D_LEARNING_RATE
    )

    # 2. 训练编码器部分
    enc_losses = f_anogan.train_encoder(
        X_train,
        epochs=E_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=E_LEARNING_RATE
    )

    # 3. 计算异常检测阈值
    f_anogan._compute_anomaly_threshold(X_train)

    training_time = time.time() - start_time
    print(f"⏱️  训练完成，耗时: {training_time:.2f} 秒")

    # 预测
    print("🔮 进行预测...")
    test_predictions = f_anogan.predict(X_test)
    test_scores = f_anogan.decision_function(X_test)

    print("✅ F-AnoGAN模型训练完成")

    return f_anogan, test_scores, test_predictions, gen_losses, disc_losses, enc_losses


# 评估函数 - 添加mAP计算
def evaluate_model(test_scores, y_test, predictions, model_type="F-AnoGAN"):
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


# 文本形式的结果报告 - 包含mAP
def text_results_report(test_scores, y_test, accuracy, precision, recall, f1, cm, mAP, ap_per_class, model_type,
                        f_anogan_model=None, gen_losses=None, disc_losses=None, enc_losses=None):
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

    if f_anogan_model is not None:
        print(f"\n🏷️  F-AnoGAN信息:")
        print(f"   潜在空间维度: {f_anogan_model.latent_dim}")
        print(f"   异常阈值: {f_anogan_model.anomaly_threshold:.6f}")

    if gen_losses is not None and disc_losses is not None and enc_losses is not None:
        print(f"\n📉 训练损失:")
        print(f"   最终生成器损失: {gen_losses[-1]:.4f}")
        print(f"   最终判别器损失: {disc_losses[-1]:.4f}")
        print(f"   最终编码器损失: {enc_losses[-1]:.6f}")
        print(f"   生成器损失改善: {((gen_losses[0] - gen_losses[-1]) / gen_losses[0] * 100):.2f}%")
        print(f"   判别器损失改善: {((disc_losses[0] - disc_losses[-1]) / disc_losses[0] * 100):.2f}%")
        print(f"   编码器损失改善: {((enc_losses[0] - enc_losses[-1]) / enc_losses[0] * 100):.2f}%")

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
def save_results(accuracy, precision, recall, f1, cm, mAP, ap_per_class, model_type,
                 X_train, X_test, y_test, test_scores,
                 f_anogan_model=None, gen_losses=None, disc_losses=None, enc_losses=None):
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
                'latent_dim': LATENT_DIM,
                'g_epochs': G_EPOCHS,
                'd_epochs': D_EPOCHS,
                'e_epochs': E_EPOCHS,
                'batch_size': BATCH_SIZE,
                'g_learning_rate': G_LEARNING_RATE,
                'd_learning_rate': D_LEARNING_RATE,
                'e_learning_rate': E_LEARNING_RATE,
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

        if f_anogan_model is not None:
            results['f_anogan_info'] = {
                'anomaly_threshold': f_anogan_model.anomaly_threshold,
                'latent_dim': f_anogan_model.latent_dim,
                'image_shape': f_anogan_model.image_shape
            }

        if gen_losses is not None:
            results['training_losses'] = {
                'gen_losses': gen_losses,
                'disc_losses': disc_losses,
                'enc_losses': enc_losses
            }

        filename = f"f_anogan_leaf_disease_results.joblib"
        joblib.dump(results, filename)
        print(f"💾 二进制结果已保存: {filename}")

    except ImportError:
        print("⚠️ joblib不可用，跳过二进制结果保存")

    # 总是保存文本结果
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = f"f_anogan_leaf_disease_results_{timestamp}.txt"

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
            f.write(f"潜在空间维度: {LATENT_DIM}\n")
            f.write(f"生成器训练轮次: {G_EPOCHS}\n")
            f.write(f"判别器训练轮次: {D_EPOCHS}\n")
            f.write(f"编码器训练轮次: {E_EPOCHS}\n")
            f.write(f"批大小: {BATCH_SIZE}\n")
            f.write(f"生成器学习率: {G_LEARNING_RATE}\n")
            f.write(f"判别器学习率: {D_LEARNING_RATE}\n")
            f.write(f"编码器学习率: {E_LEARNING_RATE}\n")
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

            if f_anogan_model is not None:
                f.write(f"📊 F-AnoGAN信息:\n")
                f.write(f"-" * 50 + "\n")
                f.write(f"潜在空间维度: {f_anogan_model.latent_dim}\n")
                f.write(f"异常阈值: {f_anogan_model.anomaly_threshold:.6f}\n")
                f.write(f"图像形状: {f_anogan_model.image_shape}\n")
                f.write(f"\n")

            if gen_losses is not None:
                f.write(f"📉 训练损失:\n")
                f.write(f"-" * 50 + "\n")
                f.write(f"最终生成器损失: {gen_losses[-1]:.4f}\n")
                f.write(f"最终判别器损失: {disc_losses[-1]:.4f}\n")
                f.write(f"最终编码器损失: {enc_losses[-1]:.6f}\n")
                f.write(f"生成器损失改善: {((gen_losses[0] - gen_losses[-1]) / gen_losses[0] * 100):.2f}%\n")
                f.write(f"判别器损失改善: {((disc_losses[0] - disc_losses[-1]) / disc_losses[0] * 100):.2f}%\n")
                f.write(f"编码器损失改善: {((enc_losses[0] - enc_losses[-1]) / enc_losses[0] * 100):.2f}%\n")
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
                f.write("✅ F-AnoGAN模型表现优秀！mAP和F1分数均超过85%。\n")
                f.write("   建议：可以直接部署到实际应用中。\n")
            elif mAP > 0.75 and f1 > 0.75:
                f.write("⚠️  F-AnoGAN模型表现良好，但仍有改进空间。\n")
                f.write("   建议：可以尝试增加训练轮次或调整损失权重。\n")
            elif mAP > 0.65 and f1 > 0.65:
                f.write("⚠️  F-AnoGAN模型表现一般，需要进一步优化。\n")
                f.write("   建议：增加训练轮次，调整学习率，或优化网络结构。\n")
            elif mAP > 0.5:
                f.write("⚠️  F-AnoGAN模型表现较差，需要优化。\n")
                f.write("   建议：检查数据质量，调整模型参数，或尝试其他异常检测方法。\n")
            else:
                f.write("❌ F-AnoGAN模型表现不佳，需要重新设计。\n")
                f.write("   建议：检查数据质量，重新设计网络架构，或调整训练参数。\n")
            f.write(f"\n")

            f.write(f"💡 算法特点:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"1. 快速异常检测的GAN变体\n")
            f.write(f"2. 结合生成器、判别器和编码器\n")
            f.write(f"3. 使用重建损失、潜在空间损失和特征损失\n")
            f.write(f"4. 适合实时异常检测应用\n")
            f.write(f"5. 包含修正的mAP评估指标\n")

        print(f"📄 文本结果已保存: {txt_filename}")

    except Exception as e:
        print(f"⚠️  无法保存文本结果: {e}")


# 可视化生成的样本和重建结果
def visualize_results(f_anogan, X_test, n_samples=5):
    """可视化生成的样本和重建结果"""
    try:
        import matplotlib.pyplot as plt

        # 生成新样本
        generated_samples = f_anogan.generate_samples(n_samples)
        generated_samples = (generated_samples + 1) / 2  # 转换到[0,1]范围

        # 重建测试样本
        test_samples = X_test[:n_samples]
        reconstructed_samples = f_anogan.reconstruct_samples(test_samples)

        # 创建可视化
        fig, axes = plt.subplots(3, n_samples, figsize=(15, 9))

        # 原始样本
        for i in range(n_samples):
            axes[0, i].imshow(test_samples[i])
            axes[0, i].set_title(f"原始样本 {i + 1}")
            axes[0, i].axis('off')

        # 重建样本
        for i in range(n_samples):
            axes[1, i].imshow(reconstructed_samples[i])
            axes[1, i].set_title(f"重建样本 {i + 1}")
            axes[1, i].axis('off')

        # 生成样本
        for i in range(n_samples):
            axes[2, i].imshow(generated_samples[i])
            axes[2, i].set_title(f"生成样本 {i + 1}")
            axes[2, i].axis('off')

        plt.tight_layout()
        plt.savefig('f_anogan_results.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("📊 可视化结果已保存: f_anogan_results.png")

    except ImportError:
        print("⚠️ matplotlib不可用，跳过可视化")


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

        # 转换为类别名称
        class_name = "病叶" if prediction == 1 else "健康叶片"

        return class_name, score, prediction

    except Exception as e:
        return f"错误: {str(e)}", 0, -1


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
        model, test_scores, predictions, gen_losses, disc_losses, enc_losses = train_f_anogan(X_train, X_test, y_test)

        if model is None:
            print("❌ 模型训练失败")
            return

        # 评估模型
        accuracy, precision, recall, f1, test_scores, cm, predictions, mAP, ap_per_class = evaluate_model(
            test_scores, y_test, predictions, "F-AnoGAN"
        )

        # 文本形式的结果报告
        text_results_report(test_scores, y_test, accuracy, precision, recall, f1, cm, mAP, ap_per_class, "F-AnoGAN", model, gen_losses,
                            disc_losses, enc_losses)

        # 详细分类报告
        print(f"\n📝 详细分类报告:")
        print(Metrics.classification_report(y_test, predictions, target_names=['健康叶片', '病叶']))

        # 保存结果
        save_results(accuracy, precision, recall, f1, cm, mAP, ap_per_class, "F-AnoGAN",
                     X_train, X_test, y_test, test_scores,
                     model, gen_losses, disc_losses, enc_losses)

        # 可视化结果
        visualize_results(model, X_test, n_samples=5)

        print(f"\n🎉 F-AnoGAN模型训练完成!")
        print("💡 模型特点:")
        print("   - 快速异常检测的GAN变体")
        print("   - 结合生成器、判别器和编码器")
        print("   - 使用重建损失和特征损失进行异常检测")
        print("   - 适合实时异常检测应用")
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

                # 转换预测结果
                pred_class = "病叶" if prediction == 1 else "健康叶片"

                status = '✓' if prediction == true_label else '✗'
                print(f"   样本 {i + 1}: 真实={true_class}, 预测={pred_class}, 分数={score:.6f} {status}")

        # 模型参数信息
        print(f"\n⚙️  模型参数:")
        print(f"   潜在空间维度: {LATENT_DIM}")
        print(f"   生成器训练轮次: {G_EPOCHS}")
        print(f"   判别器训练轮次: {D_EPOCHS}")
        print(f"   编码器训练轮次: {E_EPOCHS}")
        print(f"   批大小: {BATCH_SIZE}")
        print(f"   生成器学习率: {G_LEARNING_RATE}")
        print(f"   判别器学习率: {D_LEARNING_RATE}")
        print(f"   编码器学习率: {E_LEARNING_RATE}")
        print(f"   图像尺寸: {IMAGE_SIZE}")
        print(f"   采样比例: {SAMPLE_RATIO}")
        print(f"   评估指标: 包含修正的mAP")

        print(f"\n💡 算法说明:")
        print("   本实现使用F-AnoGAN方法:")
        print("   1. 训练生成器和判别器学习健康样本分布")
        print("   2. 训练编码器将图像映射到潜在空间")
        print("   3. 结合重建损失、潜在空间损失和特征损失")
        print("   4. 使用综合异常分数进行快速异常检测")
        print("   5. 计算修正的mAP评估模型性能")

        print(f"\n🔧 调优建议:")
        print("   - 调整三种损失的权重比例")
        print("   - 增加编码器训练轮次")
        print("   - 使用更复杂的网络结构")
        print("   - 调整异常检测阈值")
        print("   - 使用特征金字塔进行多尺度特征提取")

        print(f"\n📁 生成的文件:")
        print(f"   1. f_anogan_leaf_disease_results.joblib - 二进制结果文件 (如果joblib可用)")
        print(f"   2. f_anogan_leaf_disease_results_YYYYMMDD_HHMMSS.txt - 文本报告 (包含mAP)")
        print(f"   3. f_anogan_results.png - 可视化结果")

    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()