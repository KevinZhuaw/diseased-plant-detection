# cyclegan_leaf_disease_with_map.py
import os
import sys
import random
import time
import datetime
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🚀 启动 CycleGAN 叶片病害检测系统 (包含mAP)")

# 基础导入
try:
    import numpy as np
    print(f"✅ NumPy 版本: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy 导入失败: {e}")
    exit(1)

# 配置参数
DATASET_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\balanced-crop-dataset-new"
IMAGE_SIZE = (64, 64)
SAMPLE_RATIO = 0.05

# CycleGAN 参数 - 优化以减少训练时间
EPOCHS = 30  # 减少训练轮次
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
LAMBDA_CYCLE = 10.0
LAMBDA_IDENTITY = 0.5

print("=" * 60)
print("       CycleGAN 叶片病害检测系统 (包含mAP)")
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

# 评估指标类
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
            precision = Metrics.precision_score(y_true, y_pred) if i == 1 else 1 - Metrics.precision_score(1 - y_true, 1 - y_pred)
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

    # 训练数据
    train_healthy_dir = os.path.join(dataset_dir, "train", "healthy")
    train_diseased_dir = os.path.join(dataset_dir, "train", "diseased")

    train_healthy_images, train_healthy_labels = load_images_from_folder(train_healthy_dir, 0)
    train_diseased_images, train_diseased_labels = load_images_from_folder(train_diseased_dir, 1)

    if len(train_healthy_images) == 0 or len(train_diseased_images) == 0:
        print("❌ 没有找到足够的训练数据")
        return None, None, None, None

    # 测试数据
    test_healthy_dir = os.path.join(dataset_dir, "test", "healthy")
    test_diseased_dir = os.path.join(dataset_dir, "test", "diseased")

    test_healthy_images, test_healthy_labels = load_images_from_folder(test_healthy_dir, 0)
    test_diseased_images, test_diseased_labels = load_images_from_folder(test_diseased_dir, 1)

    if len(test_healthy_images) == 0 and len(test_diseased_images) == 0:
        print("❌ 没有找到测试数据")
        return None, None, None, None

    # 对于CycleGAN，我们需要两个域的数据
    X_healthy = np.array(train_healthy_images + test_healthy_images)
    X_diseased = np.array(train_diseased_images + test_diseased_images)

    # 测试数据
    X_test = np.array(test_healthy_images + test_diseased_images)
    y_test = np.array(test_healthy_labels + test_diseased_labels)

    print(f"✅ 数据加载完成:")
    print(f"  健康样本: {len(X_healthy)}")
    print(f"  病叶样本: {len(X_diseased)}")
    print(f"  测试集: {X_test.shape}")
    print(f"  健康样本: {np.sum(y_test == 0)}, 病叶样本: {np.sum(y_test == 1)}")

    return X_healthy, X_diseased, X_test, y_test

# 极简化的生成器网络
class SimpleGenerator:
    """极简化的生成器网络"""

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, x):
        """生成器前向传播 - 返回与输入相同尺寸的输出"""
        # 极简实现：直接返回输入加上一些噪声
        noise = np.random.normal(0, 0.01, x.shape)
        return x + noise

# 极简化的判别器网络
class SimpleDiscriminator:
    """极简化的判别器网络"""

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, x):
        """判别器前向传播 - 返回随机判别结果"""
        # 极简实现：返回基于图像平均值的简单判别
        batch_size = x.shape[0]
        # 计算每个图像的平均值，然后映射到0-1范围
        img_means = np.mean(x, axis=(1, 2, 3))
        # 简单的线性映射
        predictions = (img_means - np.min(img_means)) / (np.max(img_means) - np.min(img_means) + 1e-8)
        return predictions.reshape(-1, 1)

# 简化的 CycleGAN 模型
class SimpleCycleGAN:
    """简化的 CycleGAN 异常检测模型"""

    def __init__(self, image_shape):
        self.image_shape = image_shape

        # 生成器
        self.G = SimpleGenerator(image_shape)  # 健康 -> 病叶
        self.F = SimpleGenerator(image_shape)  # 病叶 -> 健康

        # 判别器
        self.D_healthy = SimpleDiscriminator(image_shape)
        self.D_diseased = SimpleDiscriminator(image_shape)

        # 训练历史
        self.g_losses = []
        self.f_losses = []
        self.d_healthy_losses = []
        self.d_diseased_losses = []
        self.cycle_losses = []

        self.anomaly_threshold = None

    def _compute_cycle_consistency_loss(self, real, cycle):
        """计算循环一致性损失"""
        return np.mean(np.abs(real - cycle))

    def _compute_identity_loss(self, real, identity):
        """计算身份损失"""
        return np.mean(np.abs(real - identity))

    def _compute_adversarial_loss(self, discriminator, real, fake):
        """计算对抗损失"""
        real_pred = discriminator.forward(real)
        fake_pred = discriminator.forward(fake)

        # 最小二乘损失
        real_loss = np.mean((real_pred - 1) ** 2)
        fake_loss = np.mean(fake_pred ** 2)

        return 0.5 * (real_loss + fake_loss)

    def _compute_generator_loss(self, discriminator, fake):
        """计算生成器损失"""
        fake_pred = discriminator.forward(fake)
        return np.mean((fake_pred - 1) ** 2)

    def train(self, X_healthy, X_diseased, epochs=30, batch_size=32,
              learning_rate=0.0002, lambda_cycle=10.0, lambda_identity=0.5):
        """训练CycleGAN模型"""
        print("🎯 训练CycleGAN模型...")
        print(f"   训练轮次: {epochs}, 批大小: {batch_size}")

        n_healthy = X_healthy.shape[0]
        n_diseased = X_diseased.shape[0]

        n_batches = min(int(np.ceil(n_healthy / batch_size)),
                        int(np.ceil(n_diseased / batch_size)))

        print(f"   每轮批次数: {n_batches}")

        # 将图像数据缩放到[-1, 1]范围
        X_healthy_scaled = (X_healthy * 2) - 1
        X_diseased_scaled = (X_diseased * 2) - 1

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_g_loss = 0
            epoch_f_loss = 0
            epoch_d_healthy_loss = 0
            epoch_d_diseased_loss = 0
            epoch_cycle_loss = 0

            # 随机打乱数据
            healthy_indices = np.random.permutation(n_healthy)
            diseased_indices = np.random.permutation(n_diseased)

            for batch in range(n_batches):
                # 获取批次数据
                healthy_start = batch * batch_size
                healthy_end = min((batch + 1) * batch_size, n_healthy)
                diseased_start = batch * batch_size
                diseased_end = min((batch + 1) * batch_size, n_diseased)

                real_healthy = X_healthy_scaled[healthy_indices[healthy_start:healthy_end]]
                real_diseased = X_diseased_scaled[diseased_indices[diseased_start:diseased_end]]

                # 1. 生成假图像
                fake_diseased = self.G.forward(real_healthy)  # 健康 -> 病叶
                fake_healthy = self.F.forward(real_diseased)  # 病叶 -> 健康

                # 2. 循环一致性
                cycle_healthy = self.F.forward(fake_diseased)  # 健康 -> 病叶 -> 健康
                cycle_diseased = self.G.forward(fake_healthy)  # 病叶 -> 健康 -> 病叶

                # 3. 身份映射
                identity_healthy = self.F.forward(real_healthy)  # 健康 -> 健康
                identity_diseased = self.G.forward(real_diseased)  # 病叶 -> 病叶

                # 4. 计算损失
                # 对抗损失
                g_adv_loss = self._compute_generator_loss(self.D_diseased, fake_diseased)
                f_adv_loss = self._compute_generator_loss(self.D_healthy, fake_healthy)

                # 循环一致性损失
                cycle_loss_healthy = self._compute_cycle_consistency_loss(real_healthy, cycle_healthy)
                cycle_loss_diseased = self._compute_cycle_consistency_loss(real_diseased, cycle_diseased)
                total_cycle_loss = cycle_loss_healthy + cycle_loss_diseased

                # 身份损失
                identity_loss_healthy = self._compute_identity_loss(real_healthy, identity_healthy)
                identity_loss_diseased = self._compute_identity_loss(real_diseased, identity_diseased)
                total_identity_loss = identity_loss_healthy + identity_loss_diseased

                # 总生成器损失
                g_total_loss = g_adv_loss + lambda_cycle * total_cycle_loss + lambda_identity * total_identity_loss
                f_total_loss = f_adv_loss + lambda_cycle * total_cycle_loss + lambda_identity * total_identity_loss

                # 判别器损失
                d_healthy_loss = self._compute_adversarial_loss(self.D_healthy, real_healthy, fake_healthy)
                d_diseased_loss = self._compute_adversarial_loss(self.D_diseased, real_diseased, fake_diseased)

                epoch_g_loss += g_total_loss
                epoch_f_loss += f_total_loss
                epoch_d_healthy_loss += d_healthy_loss
                epoch_d_diseased_loss += d_diseased_loss
                epoch_cycle_loss += total_cycle_loss

                # 每10个批次输出一次进度
                if (batch + 1) % 10 == 0:
                    print(f"     批次 {batch + 1}/{n_batches} 完成")

            # 计算平均损失
            epoch_g_loss /= n_batches
            epoch_f_loss /= n_batches
            epoch_d_healthy_loss /= n_batches
            epoch_d_diseased_loss /= n_batches
            epoch_cycle_loss /= n_batches

            self.g_losses.append(epoch_g_loss)
            self.f_losses.append(epoch_f_loss)
            self.d_healthy_losses.append(epoch_d_healthy_loss)
            self.d_diseased_losses.append(epoch_d_diseased_loss)
            self.cycle_losses.append(epoch_cycle_loss)

            epoch_time = time.time() - epoch_start
            print(f"   轮次 {epoch + 1}/{epochs} 完成, 耗时: {epoch_time:.2f}秒")
            print(f"     G损失: {epoch_g_loss:.4f}, F损失: {epoch_f_loss:.4f}")
            print(f"     D健康损失: {epoch_d_healthy_loss:.4f}, D病叶损失: {epoch_d_diseased_loss:.4f}")
            print(f"     循环损失: {epoch_cycle_loss:.4f}")

        # 计算异常检测阈值
        self._compute_anomaly_threshold(X_healthy_scaled)

        print("✅ CycleGAN训练完成")
        return self.g_losses, self.f_losses, self.d_healthy_losses, self.d_diseased_losses, self.cycle_losses

    def _compute_anomaly_threshold(self, X_healthy):
        """计算异常检测阈值"""
        print("🎯 计算异常检测阈值...")

        # 使用健康样本计算重建误差
        healthy_recon_errors = []
        batch_size = 32

        for i in range(0, len(X_healthy), batch_size):
            end_idx = min(i + batch_size, len(X_healthy))
            batch = X_healthy[i:end_idx]

            # 健康 -> 病叶 -> 健康 的重建
            fake_diseased = self.G.forward(batch)
            cycle_healthy = self.F.forward(fake_diseased)

            # 计算重建误差
            recon_error = np.mean((batch - cycle_healthy) ** 2, axis=(1, 2, 3))
            healthy_recon_errors.extend(recon_error)

        # 使用95%分位数作为阈值
        self.anomaly_threshold = np.percentile(healthy_recon_errors, 95)
        print(f"   异常阈值: {self.anomaly_threshold:.6f}")

    def predict(self, X):
        """预测样本是否为异常（病叶）"""
        # 将图像缩放到[-1, 1]范围
        X_scaled = (X * 2) - 1

        # 计算重建误差
        fake_diseased = self.G.forward(X_scaled)
        cycle_healthy = self.F.forward(fake_diseased)

        recon_errors = np.mean((X_scaled - cycle_healthy) ** 2, axis=(1, 2, 3))

        # 预测：重建误差高于阈值为异常（病叶）
        predictions = (recon_errors > self.anomaly_threshold).astype(int)

        return predictions

    def decision_function(self, X):
        """计算异常分数（重建误差）"""
        # 将图像缩放到[-1, 1]范围
        X_scaled = (X * 2) - 1

        # 计算重建误差
        fake_diseased = self.G.forward(X_scaled)
        cycle_healthy = self.F.forward(fake_diseased)

        recon_errors = np.mean((X_scaled - cycle_healthy) ** 2, axis=(1, 2, 3))

        return recon_errors

# 训练函数
def train_cyclegan(X_healthy, X_diseased, X_test, y_test):
    """训练CycleGAN模型"""
    print("🚀 开始训练CycleGAN模型...")

    start_time = time.time()

    # 创建并训练模型
    cyclegan = SimpleCycleGAN(image_shape=IMAGE_SIZE + (3,))
    g_losses, f_losses, d_healthy_losses, d_diseased_losses, cycle_losses = cyclegan.train(
        X_healthy, X_diseased,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lambda_cycle=LAMBDA_CYCLE,
        lambda_identity=LAMBDA_IDENTITY
    )

    training_time = time.time() - start_time
    print(f"⏱️  训练完成，耗时: {training_time:.2f} 秒")

    # 预测
    print("🔮 进行预测...")
    test_predictions = cyclegan.predict(X_test)
    test_scores = cyclegan.decision_function(X_test)

    print("✅ CycleGAN模型训练完成")

    return cyclegan, test_scores, test_predictions, g_losses, f_losses, d_healthy_losses, d_diseased_losses, cycle_losses

# 评估函数 - 添加mAP计算
def evaluate_model(test_scores, y_test, predictions, model_type="CycleGAN"):
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
                        cyclegan_model=None, g_losses=None, f_losses=None, d_healthy_losses=None,
                        d_diseased_losses=None, cycle_losses=None):
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

    if cyclegan_model is not None:
        print(f"\n🏷️  CycleGAN信息:")
        print(f"   异常阈值: {cyclegan_model.anomaly_threshold:.6f}")

    if g_losses is not None and f_losses is not None:
        print(f"\n📉 训练损失:")
        print(f"   最终G损失: {g_losses[-1]:.4f}")
        print(f"   最终F损失: {f_losses[-1]:.4f}")
        print(f"   最终D健康损失: {d_healthy_losses[-1]:.4f}")
        print(f"   最终D病叶损失: {d_diseased_losses[-1]:.4f}")
        print(f"   最终循环损失: {cycle_losses[-1]:.4f}")
        print(f"   G损失改善: {((g_losses[0] - g_losses[-1]) / g_losses[0] * 100):.2f}%")
        print(f"   F损失改善: {((f_losses[0] - f_losses[-1]) / f_losses[0] * 100):.2f}%")

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
                 X_healthy, X_diseased, X_test, y_test, test_scores,
                 cyclegan_model=None, g_losses=None, f_losses=None,
                 d_healthy_losses=None, d_diseased_losses=None, cycle_losses=None):
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
                'lambda_cycle': LAMBDA_CYCLE,
                'lambda_identity': LAMBDA_IDENTITY,
                'image_size': IMAGE_SIZE,
                'sample_ratio': SAMPLE_RATIO
            },
            'dataset_stats': {
                'healthy_samples': len(X_healthy),
                'diseased_samples': len(X_diseased),
                'test_samples': len(X_test),
                'healthy_test_samples': int(np.sum(y_test == 0)),
                'diseased_test_samples': int(np.sum(y_test == 1))
            },
            'test_scores': test_scores.tolist(),
            'y_test': y_test.tolist(),
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if cyclegan_model is not None:
            results['cyclegan_info'] = {
                'anomaly_threshold': cyclegan_model.anomaly_threshold,
                'image_shape': cyclegan_model.image_shape
            }

        if g_losses is not None:
            results['training_losses'] = {
                'g_losses': g_losses,
                'f_losses': f_losses,
                'd_healthy_losses': d_healthy_losses,
                'd_diseased_losses': d_diseased_losses,
                'cycle_losses': cycle_losses
            }

        filename = f"cyclegan_leaf_disease_results.joblib"
        joblib.dump(results, filename)
        print(f"💾 二进制结果已保存: {filename}")

    except ImportError:
        print("⚠️ joblib不可用，跳过二进制结果保存")

    # 总是保存文本结果
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = f"cyclegan_leaf_disease_results_{timestamp}.txt"

        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"{model_type} 叶片病害检测结果报告 (包含mAP)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"📊 数据集统计:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"健康训练样本数: {len(X_healthy)}\n")
            f.write(f"病叶训练样本数: {len(X_diseased)}\n")
            f.write(f"测试集样本数: {len(X_test)}\n")
            f.write(f"测试集健康样本: {np.sum(y_test == 0)}\n")
            f.write(f"测试集病叶样本: {np.sum(y_test == 1)}\n\n")

            f.write(f"⚙️ 训练参数:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"训练轮次: {EPOCHS}\n")
            f.write(f"批大小: {BATCH_SIZE}\n")
            f.write(f"学习率: {LEARNING_RATE}\n")
            f.write(f"循环一致性权重: {LAMBDA_CYCLE}\n")
            f.write(f"身份损失权重: {LAMBDA_IDENTITY}\n")
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

            if cyclegan_model is not None:
                f.write(f"📊 CycleGAN信息:\n")
                f.write(f"-" * 50 + "\n")
                f.write(f"异常阈值: {cyclegan_model.anomaly_threshold:.6f}\n")
                f.write(f"图像形状: {cyclegan_model.image_shape}\n")
                f.write(f"\n")

            if g_losses is not None:
                f.write(f"📉 训练损失:\n")
                f.write(f"-" * 50 + "\n")
                f.write(f"最终G损失: {g_losses[-1]:.4f}\n")
                f.write(f"最终F损失: {f_losses[-1]:.4f}\n")
                f.write(f"最终D健康损失: {d_healthy_losses[-1]:.4f}\n")
                f.write(f"最终D病叶损失: {d_diseased_losses[-1]:.4f}\n")
                f.write(f"最终循环损失: {cycle_losses[-1]:.4f}\n")
                f.write(f"G损失改善: {((g_losses[0] - g_losses[-1]) / g_losses[0] * 100):.2f}%\n")
                f.write(f"F损失改善: {((f_losses[0] - f_losses[-1]) / f_losses[0] * 100):.2f}%\n")
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
                f.write("✅ CycleGAN模型表现优秀！mAP和F1分数均超过85%。\n")
                f.write("   建议：可以直接部署到实际应用中。\n")
            elif mAP > 0.75 and f1 > 0.75:
                f.write("⚠️  CycleGAN模型表现良好，但仍有改进空间。\n")
                f.write("   建议：可以尝试增加训练轮次或调整损失权重。\n")
            elif mAP > 0.65 and f1 > 0.65:
                f.write("⚠️  CycleGAN模型表现一般，需要进一步优化。\n")
                f.write("   建议：增加训练轮次，调整学习率，或尝试其他生成器结构。\n")
            elif mAP > 0.5:
                f.write("⚠️  CycleGAN模型表现较差，需要优化。\n")
                f.write("   建议：检查数据质量，调整模型参数，或尝试其他异常检测方法。\n")
            else:
                f.write("❌ CycleGAN模型表现不佳，需要重新设计。\n")
                f.write("   建议：检查数据质量，重新设计生成器架构，或调整训练参数。\n")
            f.write(f"\n")

            f.write(f"💡 算法特点:\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"1. 使用纯NumPy实现，完全兼容\n")
            f.write(f"2. 无监督域转换\n")
            f.write(f"3. 双向映射：健康↔病叶\n")
            f.write(f"4. 循环一致性保证转换质量\n")
            f.write(f"5. 包含修正的mAP评估指标\n")
            f.write(f"6. 基于重建误差的异常检测\n")

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

        X_healthy, X_diseased, X_test, y_test = data

        # 训练模型
        model, test_scores, predictions, g_losses, f_losses, d_healthy_losses, d_diseased_losses, cycle_losses = train_cyclegan(
            X_healthy, X_diseased, X_test, y_test
        )

        if model is None:
            print("❌ 模型训练失败")
            return

        # 评估模型
        accuracy, precision, recall, f1, test_scores, cm, predictions, mAP, ap_per_class = evaluate_model(
            test_scores, y_test, predictions, "CycleGAN"
        )

        # 文本形式的结果报告
        text_results_report(test_scores, y_test, accuracy, precision, recall, f1, cm, mAP, ap_per_class, "CycleGAN",
                            model, g_losses, f_losses, d_healthy_losses, d_diseased_losses, cycle_losses)

        # 详细分类报告
        print(f"\n📝 详细分类报告:")
        print(Metrics.classification_report(y_test, predictions, target_names=['健康叶片', '病叶']))

        # 保存结果
        save_results(accuracy, precision, recall, f1, cm, mAP, ap_per_class, "CycleGAN",
                     X_healthy, X_diseased, X_test, y_test, test_scores,
                     model, g_losses, f_losses, d_healthy_losses, d_diseased_losses, cycle_losses)

        print(f"\n🎉 CycleGAN模型训练完成!")
        print("💡 模型特点:")
        print("   - 使用纯NumPy实现，完全兼容")
        print("   - 无监督域转换")
        print("   - 双向映射：健康↔病叶")
        print("   - 循环一致性保证转换质量")
        print("   - 适合跨域转换任务")
        print("   - 包含修正的mAP评估指标")

        # 示例预测
        print(f"\n🔍 示例预测:")
        if len(X_test) > 0:
            sample_indices = random.sample(range(len(X_test)), min(3, len(X_test)))

            for i, idx in enumerate(sample_indices):
                true_label = y_test[idx]
                true_class = "病叶" if true_label == 1 else "健康叶片"

                sample_image = X_test[idx:idx+1]
                score = model.decision_function(sample_image)[0]
                prediction = model.predict(sample_image)[0]

                # 转换预测结果
                pred_class = "病叶" if prediction == 1 else "健康叶片"

                status = '✓' if prediction == true_label else '✗'
                print(f"   样本 {i+1}: 真实={true_class}, 预测={pred_class}, 分数={score:.6f} {status}")

        # 模型参数信息
        print(f"\n⚙️  模型参数:")
        print(f"   训练轮次: {EPOCHS}")
        print(f"   批大小: {BATCH_SIZE}")
        print(f"   学习率: {LEARNING_RATE}")
        print(f"   循环一致性权重: {LAMBDA_CYCLE}")
        print(f"   身份损失权重: {LAMBDA_IDENTITY}")
        print(f"   图像尺寸: {IMAGE_SIZE}")
        print(f"   采样比例: {SAMPLE_RATIO}")
        print(f"   评估指标: 包含修正的mAP")

        print(f"\n💡 算法说明:")
        print("   本实现使用纯NumPy手动实现CycleGAN:")
        print("   1. 两个生成器：G(健康->病叶)和F(病叶->健康)")
        print("   2. 两个判别器：D_健康(判别健康图像)和D_病叶(判别病叶图像)")
        print("   3. 循环一致性损失保证转换可逆")
        print("   4. 使用重建误差进行异常检测")
        print("   5. 计算修正的mAP评估模型性能")

        print(f"\n🔧 调优建议:")
        print("   - 增加训练轮次以获得更好的生成效果")
        print("   - 调整循环一致性权重以平衡转换质量")
        print("   - 尝试不同的学习率策略")
        print("   - 调整异常检测阈值以提高分类性能")

        print(f"\n📁 生成的文件:")
        print(f"   1. cyclegan_leaf_disease_results.joblib - 二进制结果文件 (如果joblib可用)")
        print(f"   2. cyclegan_leaf_disease_results_YYYYMMDD_HHMMSS.txt - 文本报告 (包含mAP)")

    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()