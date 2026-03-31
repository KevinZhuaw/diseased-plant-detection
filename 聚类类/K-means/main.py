# kmeans_leaf_disease_detection_with_map.py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🚀 启动 K-means 叶片病害检测系统 (包含mAP)")

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
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib支持中文
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("          K-means 聚类叶片病害识别系统 (包含mAP)")
print("=" * 60)


# 手动实现评估指标（与之前相同）
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
def visualize_pr_curves(y_true_onehot, y_scores, ap_per_class, mAP, filename="kmeans_pr_curves.png"):
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

    plt.suptitle(f'K-means - 精确率-召回率曲线 (mAP = {mAP:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ PR曲线已保存: {filename}")


# 配置参数
DATASET_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\balanced-crop-dataset-new"
IMAGE_SIZE = (64, 64)  # 减小尺寸以加快处理
SAMPLE_RATIO = 0.05

# K-means 参数
N_CLUSTERS = 3  # 聚类数量
MAX_ITERATIONS = 100  # 最大迭代次数
TOLERANCE = 1e-4  # 收敛容忍度

print("=" * 60)
print("          K-means 聚类叶片病害检测系统 (包含mAP)")
print("=" * 60)


# 数据加载函数（与之前相同）
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

    # 训练数据 - 只使用健康样本（无监督学习特点）
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


# 特征提取函数（与之前相同）
def extract_features(images, method='simple'):
    """从图像中提取特征"""
    print(f"🔍 使用 '{method}' 方法提取特征...")

    if method == 'simple':
        # 简单展平
        features = images.reshape(images.shape[0], -1)

    elif method == 'histogram':
        # 颜色直方图
        features = []
        for img in images:
            # 计算RGB通道的直方图
            hist_r = np.histogram(img[:, :, 0], bins=16, range=(0, 1))[0]
            hist_g = np.histogram(img[:, :, 1], bins=16, range=(0, 1))[0]
            hist_b = np.histogram(img[:, :, 2], bins=16, range=(0, 1))[0]
            # 合并直方图
            hist = np.concatenate([hist_r, hist_g, hist_b])
            features.append(hist)
        features = np.array(features)

    elif method == 'texture':
        # 简单的纹理特征 (梯度幅值)
        features = []
        for img in images:
            # 计算梯度
            gx = np.gradient(img, axis=0)
            gy = np.gradient(img, axis=1)
            # 计算梯度幅值
            gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
            # 取平均值作为特征
            texture_feature = np.mean(gradient_magnitude, axis=(0, 1))
            features.append(texture_feature)
        features = np.array(features)

    else:
        # 默认使用展平
        features = images.reshape(images.shape[0], -1)

    print(f"   特征维度: {features.shape}")
    return features


# 手动实现 K-means 聚类
class SimpleKMeans:
    """简化的 K-means 聚类实现"""

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.feature_mean = None
        self.feature_std = None
        self.cluster_thresholds = None

    def _initialize_centroids(self, X):
        """随机初始化质心"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X, centroids):
        """分配样本到最近的质心"""
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        """更新质心位置"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            if np.sum(labels == i) > 0:
                new_centroids[i] = X[labels == i].mean(axis=0)
        return new_centroids

    def fit(self, X_train):
        """训练 K-means 模型"""
        print(f"🎯 训练 K-means 聚类 (k={self.n_clusters})...")

        # 提取特征
        X_features = extract_features(X_train, method='histogram')

        # 手动标准化特征
        self.feature_mean = np.mean(X_features, axis=0)
        self.feature_std = np.std(X_features, axis=0) + 1e-8
        X_scaled = (X_features - self.feature_mean) / self.feature_std

        # 初始化质心
        self.centroids = self._initialize_centroids(X_scaled)

        # K-means 迭代
        for iteration in range(self.max_iter):
            # 分配样本到簇
            labels = self._assign_clusters(X_scaled, self.centroids)

            # 更新质心
            new_centroids = self._update_centroids(X_scaled, labels)

            # 检查收敛
            centroid_shift = np.sqrt(((new_centroids - self.centroids) ** 2).sum(axis=1)).mean()

            self.centroids = new_centroids
            self.labels = labels

            if centroid_shift < self.tol:
                print(f"   迭代 {iteration + 1}: 收敛 (质心移动: {centroid_shift:.6f})")
                break

            if (iteration + 1) % 10 == 0:
                print(f"   迭代 {iteration + 1}: 质心移动 = {centroid_shift:.6f}")

        # 计算每个簇的距离阈值（用于异常检测）
        self.cluster_thresholds = self._compute_cluster_thresholds(X_scaled)

        # 计算 inertia（簇内平方和）
        self.inertia_ = self._compute_inertia(X_scaled)

        print(f"✅ K-means 训练完成")
        print(f"   最终迭代次数: {min(iteration + 1, self.max_iter)}")
        print(f"   最终 inertia: {self.inertia_:.4f}")
        print(f"   簇大小: {np.bincount(labels)}")

        return self

    def _compute_cluster_thresholds(self, X):
        """计算每个簇的距离阈值"""
        thresholds = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                distances = np.sqrt(((cluster_points - self.centroids[i]) ** 2).sum(axis=1))
                # 使用95%分位数作为阈值
                threshold = np.percentile(distances, 95)
                thresholds.append(threshold)
            else:
                thresholds.append(0)
        return np.array(thresholds)

    def _compute_inertia(self, X):
        """计算簇内平方和"""
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                distances = np.sqrt(((cluster_points - self.centroids[i]) ** 2).sum(axis=1))
                inertia += np.sum(distances ** 2)
        return inertia

    def predict(self, X):
        """预测样本是否为异常（病叶）"""
        # 提取特征
        X_features = extract_features(X, method='histogram')

        # 标准化
        X_scaled = (X_features - self.feature_mean) / self.feature_std

        # 计算到每个质心的距离
        distances = np.sqrt(((X_scaled - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))

        # 找到最近的簇和距离
        min_distances = np.min(distances, axis=0)
        nearest_clusters = np.argmin(distances, axis=0)

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
        # 提取特征
        X_features = extract_features(X, method='histogram')

        # 标准化
        X_scaled = (X_features - self.feature_mean) / self.feature_std

        # 计算到最近质心的距离
        distances = np.sqrt(((X_scaled - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        min_distances = np.min(distances, axis=0)

        return min_distances

    def get_cluster_info(self, X):
        """获取聚类信息"""
        X_features = extract_features(X, method='histogram')
        X_scaled = (X_features - self.feature_mean) / self.feature_std

        distances = np.sqrt(((X_scaled - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        nearest_clusters = np.argmin(distances, axis=0)
        min_distances = np.min(distances, axis=0)

        return nearest_clusters, min_distances


# 训练函数
def train_kmeans(X_train, X_test, y_test):
    """训练 K-means 模型"""
    print("🚀 开始训练 K-means...")

    start_time = time.time()

    # 创建并训练模型
    kmeans = SimpleKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITERATIONS, tol=TOLERANCE)
    kmeans.fit(X_train)

    training_time = time.time() - start_time
    print(f"⏱️  训练完成，耗时: {training_time:.2f} 秒")

    # 预测
    print("🔮 进行预测...")
    test_predictions = kmeans.predict(X_test)
    test_scores = kmeans.decision_function(X_test)

    print("✅ K-means 训练完成")

    return kmeans, test_scores, test_predictions


# 评估函数 - 添加mAP计算
def evaluate_model(test_scores, y_test, predictions, model_type="K-means"):
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
def visualize_results(test_scores, y_test, cm, mAP=None, ap_per_class=None, kmeans_model=None):
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
    plt.title('K-means 异常分数分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 混淆矩阵
    plt.subplot(2, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测健康', '预测病叶'],
                yticklabels=['实际健康', '实际病叶'])
    plt.title('K-means 混淆矩阵')

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
        f"K-means性能指标汇总\n\n"
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

    if kmeans_model is not None:
        metrics_text += f"\n聚类信息:\n"
        metrics_text += f"簇数量: {kmeans_model.n_clusters}\n"
        metrics_text += f"簇内平方和: {kmeans_model.inertia_:.4f}\n"
        metrics_text += f"簇大小分布: {np.bincount(kmeans_model.labels)}\n"
        metrics_text += f"簇距离阈值: {kmeans_model.cluster_thresholds}\n"

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
    plt.savefig("kmeans_final_results.png", dpi=300, bbox_inches='tight')
    plt.show()


# 文本形式的结果报告 - 包含mAP
def text_results_report(test_scores, y_test, accuracy, precision, recall, f1, cm, mAP, ap_per_class, model_type,
                        kmeans_model=None):
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

    if kmeans_model is not None:
        print(f"\n🏷️  聚类信息:")
        print(f"   簇数量: {kmeans_model.n_clusters}")
        print(f"   簇内平方和 (inertia): {kmeans_model.inertia_:.4f}")
        print(f"   簇大小分布: {np.bincount(kmeans_model.labels)}")
        print(f"   簇距离阈值: {kmeans_model.cluster_thresholds}")

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
                 test_scores, kmeans_model=None):
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
                'max_iterations': MAX_ITERATIONS,
                'tolerance': TOLERANCE,
                'image_size': IMAGE_SIZE,
                'sample_ratio': SAMPLE_RATIO,
                'feature_method': 'histogram'
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

        if kmeans_model is not None:
            results['cluster_info'] = {
                'inertia': kmeans_model.inertia_,
                'cluster_sizes': np.bincount(kmeans_model.labels).tolist(),
                'cluster_thresholds': kmeans_model.cluster_thresholds.tolist() if kmeans_model.cluster_thresholds is not None else None
            }

        filename = f"kmeans_leaf_disease_results.joblib"
        joblib.dump(results, filename)
        print(f"💾 二进制结果已保存: {filename}")

    except ImportError:
        print("⚠️ joblib不可用，跳过二进制结果保存")

    # 总是保存文本结果
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = f"kmeans_leaf_disease_results_{timestamp}.txt"

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
            f.write(f"最大迭代次数: {MAX_ITERATIONS}\n")
            f.write(f"收敛容忍度: {TOLERANCE}\n")
            f.write(f"图像尺寸: {IMAGE_SIZE}\n")
            f.write(f"采样比例: {SAMPLE_RATIO}\n")
            f.write(f"特征提取方法: 颜色直方图\n\n")

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

            if kmeans_model is not None:
                f.write(f"📊 聚类信息:\n")
                f.write(f"-" * 50 + "\n")
                f.write(f"簇数量: {kmeans_model.n_clusters}\n")
                f.write(f"簇内平方和: {kmeans_model.inertia_:.4f}\n")
                f.write(f"簇大小分布: {np.bincount(kmeans_model.labels)}\n")
                f.write(f"簇距离阈值: {kmeans_model.cluster_thresholds}\n")
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
                f.write("✅ K-means模型表现优秀！mAP和F1分数均超过85%。\n")
                f.write("   建议：可以直接部署到实际应用中。\n")
            elif mAP > 0.75 and f1 > 0.75:
                f.write("⚠️  K-means模型表现良好，但仍有改进空间。\n")
                f.write("   建议：可以尝试增加聚类数量或调整特征提取方法。\n")
            elif mAP > 0.65 and f1 > 0.65:
                f.write("⚠️  K-means模型表现一般，需要进一步优化。\n")
                f.write("   建议：增加聚类数量，调整距离阈值，或尝试其他特征提取方法。\n")
            elif mAP > 0.5:
                f.write("⚠️  K-means模型表现较差，需要优化。\n")
                f.write("   建议：检查数据质量，调整模型参数，或尝试其他异常检测方法。\n")
            else:
                f.write("❌ K-means模型表现不佳，需要重新设计。\n")
                f.write("   建议：检查数据质量，重新设计特征提取方法，或调整聚类算法参数。\n")
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
        model, test_scores, predictions = train_kmeans(X_train, X_test, y_test)

        if model is None:
            print("❌ 模型训练失败")
            return

        # 评估模型
        accuracy, precision, recall, f1, test_scores, cm, predictions, mAP, ap_per_class = evaluate_model(
            test_scores, y_test, predictions, "K-means 聚类"
        )

        # 可视化结果
        visualize_results(test_scores, y_test, cm, mAP, ap_per_class, model)

        # 文本形式的结果报告
        text_results_report(test_scores, y_test, accuracy, precision, recall, f1, cm, mAP, ap_per_class,
                            "K-means", model)

        # 详细分类报告
        print(f"\n📝 详细分类报告:")
        print(Metrics.classification_report(y_test, predictions, target_names=['健康叶片', '病叶']))

        # 保存结果
        save_results(accuracy, precision, recall, f1, cm, mAP, ap_per_class,
                     "K-means 聚类", X_train, X_test, y_test, test_scores, model)

        print(f"\n🎉 K-means 训练完成!")
        print("💡 模型特点:")
        print("   - 无监督学习，仅使用健康样本训练")
        print("   - 基于聚类的异常检测")
        print("   - 自动发现数据中的自然分组")
        print("   - 适合发现数据中的潜在模式")
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
                cluster_id = model.get_cluster_info(sample_image)[0][0]

                # 转换预测结果
                pred_class = "病叶" if prediction == 1 else "健康叶片"

                status = '✓' if prediction == true_label else '✗'
                print(
                    f"   样本 {i + 1}: 真实={true_class}, 预测={pred_class}, 簇={cluster_id}, 分数={score:.6f} {status}")

        # 模型参数信息
        print(f"\n⚙️  模型参数:")
        print(f"   聚类数量: {N_CLUSTERS}")
        print(f"   最大迭代次数: {MAX_ITERATIONS}")
        print(f"   收敛容忍度: {TOLERANCE}")
        print(f"   图像尺寸: {IMAGE_SIZE}")
        print(f"   采样比例: {SAMPLE_RATIO}")
        print(f"   特征提取: 颜色直方图")
        print(f"   评估指标: 包含修正的mAP")

        print(f"\n💡 算法说明:")
        print("   本实现使用基于 K-means 的异常检测方法:")
        print("   1. 提取图像的颜色直方图特征")
        print("   2. 对健康样本进行 K-means 聚类")
        print("   3. 为每个簇计算距离阈值")
        print("   4. 测试样本分配到最近簇，距离大于阈值则为异常")
        print("   5. 计算修正的mAP评估模型性能")

        print(f"\n🔧 调优建议:")
        print("   - 可以尝试不同的聚类数量 (n_clusters)")
        print("   - 调整距离阈值的百分位数")
        print("   - 尝试不同的特征提取方法")
        print("   - 使用肘部法则确定最佳聚类数量")

        print(f"\n📁 生成的文件:")
        print(f"   1. kmeans_leaf_disease_results.joblib - 二进制结果文件 (如果joblib可用)")
        print(f"   2. kmeans_leaf_disease_results_YYYYMMDD_HHMMSS.txt - 文本报告 (包含mAP)")
        print(f"   3. kmeans_pr_curves.png - PR曲线图")
        print(f"   4. kmeans_final_results.png - 综合可视化结果")

    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()