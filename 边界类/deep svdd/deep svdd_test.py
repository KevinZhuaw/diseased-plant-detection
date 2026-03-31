# deep_svdd_with_map.py
import os

# 在导入其他库之前设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
import sys
import random
import datetime
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 检查TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models

    print(f"✅ TensorFlow 版本: {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow 导入失败: {e}")
    sys.exit(1)

# 检查scikit-learn
try:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn 可用")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn 不可用，将跳过mAP计算")

# -------------------------- 配置参数 --------------------------
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
LATENT_DIM = 32
SAMPLE_RATIO = 0.05

# -------------------------- 设置matplotlib --------------------------
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 自定义评估函数 --------------------------
def calculate_metrics(y_true, y_pred):
    """手动计算评估指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算混淆矩阵
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 混淆矩阵
    cm = np.array([[tn, fp], [fn, tp]])

    return accuracy, precision, recall, f1, cm


def find_best_threshold(scores, y_true):
    """寻找最佳阈值"""
    min_score = np.min(scores)
    max_score = np.max(scores)
    thresholds = np.linspace(min_score, max_score, 50)

    best_f1 = 0
    best_threshold = thresholds[0]
    best_predictions = None

    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        _, _, _, f1, _ = calculate_metrics(y_true, predictions)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_predictions = predictions

    return best_threshold, best_f1, best_predictions


# -------------------------- 计算mAP --------------------------
def calculate_map(test_scores, y_true):
    """计算平均精度均值（mAP）"""
    if not SKLEARN_AVAILABLE:
        print("⚠️  scikit-learn不可用，跳过mAP计算")
        return 0.0, {}

    print("\n📊 计算mAP (平均精度均值)...")

    # 将异常分数转换为概率
    scores = np.array(test_scores, dtype=np.float32)

    # 归一化分数到[0, 1]范围
    min_score = scores.min()
    max_score = scores.max()

    if max_score - min_score < 1e-10:
        normalized_scores = np.zeros_like(scores)
    else:
        normalized_scores = (scores - min_score) / (max_score - min_score)

    # 对于二分类问题：
    # 健康类别的概率 = 1 - 归一化分数（分数越小，健康概率越高）
    # 病叶类别的概率 = 归一化分数（分数越大，病叶概率越高）

    # 为每个样本创建两个类别的预测概率
    y_scores = np.zeros((len(scores), 2), dtype=np.float32)
    y_scores[:, 0] = 1 - normalized_scores  # 健康类别概率
    y_scores[:, 1] = normalized_scores  # 病叶类别概率

    # 创建真实标签的one-hot编码
    y_true_onehot = np.zeros((len(y_true), 2), dtype=np.int32)
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1

    # 计算每个类别的AP（平均精度）
    ap_scores = []
    ap_per_class = {}

    for class_idx, class_name in enumerate(['健康叶片', '病叶']):
        ap = average_precision_score(y_true_onehot[:, class_idx], y_scores[:, class_idx])
        ap_scores.append(ap)
        ap_per_class[class_name] = ap
        print(f"  {class_name} AP: {ap:.4f}")

    # 计算mAP（平均精度均值）
    mAP = np.mean(ap_scores)
    print(f"✅ mAP: {mAP:.4f}")

    # 可视化PR曲线
    visualize_pr_curves(y_true_onehot, y_scores, ap_per_class, mAP)

    return mAP, ap_per_class


def visualize_pr_curves(y_true, y_scores, ap_per_class, mAP):
    """可视化PR曲线"""
    plt.figure(figsize=(12, 5))

    # 为每个类别绘制PR曲线
    for class_idx, class_name in enumerate(['健康叶片', '病叶']):
        precision, recall, _ = precision_recall_curve(
            y_true[:, class_idx],
            y_scores[:, class_idx]
        )

        plt.subplot(1, 2, class_idx + 1)
        plt.plot(recall, precision, linewidth=2,
                 label=f'{class_name} (AP={ap_per_class[class_name]:.3f})')
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title(f'{class_name} - 精确率-召回率曲线')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.fill_between(recall, precision, alpha=0.2)

    plt.suptitle(f'Deep SVDD - 精确率-召回率曲线 (mAP = {mAP:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig('deep_svdd_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ PR曲线已保存: deep_svdd_pr_curves.png")


# -------------------------- 数据加载函数 --------------------------
def load_and_preprocess_data(dataset_dir, image_size, sample_ratio=1.0):
    """加载和预处理数据"""

    def load_images_from_folder(folder, label, sample_ratio=1.0):
        images = []
        labels = []

        # 检查文件夹是否存在
        if not os.path.exists(folder):
            print(f"❌ 文件夹不存在: {folder}")
            return images, labels

        # 获取所有图片文件
        try:
            img_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        except Exception as e:
            print(f"❌ 无法读取文件夹 {folder}: {e}")
            return images, labels

        if len(img_files) == 0:
            print(f"⚠️  文件夹中没有图片文件: {folder}")
            return images, labels

        # 采样
        if sample_ratio < 1.0:
            sample_size = max(1, int(len(img_files) * sample_ratio))
            img_files = random.sample(img_files, sample_size)

        print(f"处理 {os.path.basename(folder)} (共 {len(img_files)} 张图片)...")

        for img_file in tqdm(img_files, desc=f"   {os.path.basename(folder)}"):
            img_path = os.path.join(folder, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                image = image.resize(image_size)
                img_array = np.array(image) / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"⚠️  处理图像失败 {img_path}: {e}")

        return images, labels

    print("📊 加载和预处理数据...")

    # 训练数据 - Deep SVDD只使用健康样本
    train_healthy_path = os.path.join(dataset_dir, "train", "healthy")

    train_healthy_images, train_healthy_labels = load_images_from_folder(
        train_healthy_path, 0, sample_ratio
    )

    if len(train_healthy_images) == 0:
        print(f"❌ 训练集健康样本为空，请检查路径: {train_healthy_path}")
        return None, None, None, None

    # 测试数据
    test_healthy_path = os.path.join(dataset_dir, "test", "healthy")
    test_diseased_path = os.path.join(dataset_dir, "test", "diseased")

    test_healthy_images, test_healthy_labels = load_images_from_folder(
        test_healthy_path, 0, sample_ratio
    )

    test_diseased_images, test_diseased_labels = load_images_from_folder(
        test_diseased_path, 1, sample_ratio
    )

    if len(test_healthy_images) == 0 and len(test_diseased_images) == 0:
        print("❌ 测试集样本为空")
        return None, None, None, None

    # 合并数据
    X_train = np.array(train_healthy_images)
    y_train = np.array(train_healthy_labels)

    X_test = np.array(test_healthy_images + test_diseased_images)
    y_test = np.array(test_healthy_labels + test_diseased_labels)

    print(f"✅ 数据加载完成:")
    print(f"  训练集 (仅健康): {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  测试集 - 健康: {len(test_healthy_images)}, 病叶: {len(test_diseased_images)}")

    return X_train, y_train, X_test, y_test


# -------------------------- Deep SVDD模型 --------------------------
def build_simple_encoder(input_shape, latent_dim):
    """构建简化的编码器"""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    encoded = layers.Dense(latent_dim, name='encoded')(x)

    encoder = models.Model(inputs, encoded, name="encoder")
    return encoder


class SimpleDeepSVDD:
    """简化的Deep SVDD实现"""

    def __init__(self, encoder, latent_dim):
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.center = None

    def initialize_center(self, X_train):
        """初始化中心点"""
        print("初始化中心点...")
        Z_train = self.encoder.predict(X_train, verbose=0, batch_size=32)
        self.center = np.mean(Z_train, axis=0)
        print(f"中心点维度: {self.center.shape}")
        return self.center

    def compute_scores(self, X):
        """计算异常分数"""
        Z = self.encoder.predict(X, verbose=0, batch_size=32)
        scores = np.sum((Z - self.center) ** 2, axis=1)
        return scores


# -------------------------- 训练函数 --------------------------
def train_simple_svdd(X_train, input_shape, latent_dim, epochs=30, batch_size=32):
    """训练简化的Deep SVDD"""
    print("🚀 开始训练Deep SVDD...")

    # 构建编码器
    encoder = build_simple_encoder(input_shape, latent_dim)

    # 创建Deep SVDD模型
    deep_svdd = SimpleDeepSVDD(encoder, latent_dim)

    # 初始化中心
    deep_svdd.initialize_center(X_train[:min(1000, len(X_train))])

    # 编译编码器（作为自编码器训练）
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # 构建简单的自编码器
    inputs = layers.Input(shape=input_shape)
    encoded = encoder(inputs)

    # 简单的解码器
    x = layers.Dense(128, activation='relu')(encoded)
    x = layers.Dense(64 * 16 * 16, activation='relu')(x)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(inputs, decoded)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    print("模型架构摘要:")
    print(f"输入形状: {input_shape}")
    print(f"编码器参数量: {encoder.count_params():,}")
    print(f"自编码器参数量: {autoencoder.count_params():,}")

    # 训练
    print(f"开始训练，共 {epochs} 个epochs...")

    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ]
    )

    # 保存模型
    encoder.save("deep_svdd_encoder.h5")
    print("✅ Deep SVDD模型保存完成")

    return deep_svdd, encoder, history


# -------------------------- 评估函数 --------------------------
def evaluate_model(deep_svdd, X_test, y_test):
    """评估模型"""
    print("📊 评估模型性能...")

    # 计算异常分数
    test_scores = deep_svdd.compute_scores(X_test)

    # 计算mAP
    mAP = 0.0
    ap_per_class = {}
    if SKLEARN_AVAILABLE:
        mAP, ap_per_class = calculate_map(test_scores, y_test)

    # 寻找最佳阈值
    best_threshold, best_f1, best_predictions = find_best_threshold(test_scores, y_test)

    # 计算所有指标
    accuracy, precision, recall, f1, cm = calculate_metrics(y_test, best_predictions)

    # 计算分数统计
    healthy_scores = test_scores[y_test == 0]
    diseased_scores = test_scores[y_test == 1]

    score_stats = {
        'healthy_mean': float(np.mean(healthy_scores)) if len(healthy_scores) > 0 else 0,
        'healthy_std': float(np.std(healthy_scores)) if len(healthy_scores) > 0 else 0,
        'diseased_mean': float(np.mean(diseased_scores)) if len(diseased_scores) > 0 else 0,
        'diseased_std': float(np.std(diseased_scores)) if len(diseased_scores) > 0 else 0,
        'score_min': float(np.min(test_scores)),
        'score_max': float(np.max(test_scores)),
        'score_mean': float(np.mean(test_scores)),
        'score_std': float(np.std(test_scores))
    }

    return accuracy, precision, recall, f1, best_threshold, test_scores, cm, best_predictions, score_stats, mAP, ap_per_class


# -------------------------- 可视化函数 --------------------------
def visualize_results(history, test_scores, y_test, threshold, cm, mAP=None, ap_per_class=None):
    """可视化结果，包含mAP"""
    fig = plt.figure(figsize=(20, 12))

    # 1. 训练历史
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='训练损失')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失 (MSE)')
    plt.title('Deep SVDD训练历史')
    plt.legend()
    plt.grid(True)

    # 2. 异常分数分布
    plt.subplot(2, 3, 2)
    healthy_scores = test_scores[y_test == 0]
    diseased_scores = test_scores[y_test == 1]

    plt.hist(healthy_scores, bins=30, alpha=0.5, label='健康叶片', color='green')
    plt.hist(diseased_scores, bins=30, alpha=0.5, label='病叶', color='red')
    plt.axvline(threshold, color='blue', linestyle='--', linewidth=2,
                label=f'最佳阈值: {threshold:.6f}')
    plt.xlabel('异常分数')
    plt.ylabel('样本数量')
    plt.title('Deep SVDD异常分数分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 混淆矩阵
    plt.subplot(2, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测健康', '预测病叶'],
                yticklabels=['实际健康', '实际病叶'])
    plt.title('Deep SVDD混淆矩阵')

    # 4. 性能指标汇总（包含mAP）
    plt.subplot(2, 3, 4)
    plt.axis('off')

    # 收集评估结果
    accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm) if np.sum(cm) > 0 else 0
    precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    recall = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics_text = (
        f"Deep SVDD性能指标汇总\n\n"
        f"准确率:  {accuracy:.4f}\n"
        f"精确率: {precision:.4f}\n"
        f"召回率:    {recall:.4f}\n"
        f"F1分数:  {f1:.4f}\n"
    )

    if mAP is not None:
        metrics_text += f"\nmAP:       {mAP:.4f}\n"
        if ap_per_class is not None:
            metrics_text += f"健康叶片 AP: {ap_per_class.get('健康叶片', 0):.4f}\n"
            metrics_text += f"病叶 AP: {ap_per_class.get('病叶', 0):.4f}\n"

    metrics_text += f"\n最佳阈值: {threshold:.6f}"

    plt.text(0.1, 0.5, metrics_text, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='center')

    # 5. 分数箱线图
    plt.subplot(2, 3, 5)
    score_data = [healthy_scores, diseased_scores]
    plt.boxplot(score_data, labels=['健康叶片', '病叶'])
    plt.title('异常分数箱线图')
    plt.ylabel('异常分数')
    plt.grid(True, alpha=0.3)

    # 添加阈值线
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'阈值={threshold:.4f}')
    plt.legend()

    # 6. ROC曲线样式图
    plt.subplot(2, 3, 6)

    # 计算不同阈值下的TPR和FPR
    thresholds = np.linspace(np.min(test_scores), np.max(test_scores), 50)
    tprs = []
    fprs = []

    for t in thresholds:
        preds = (test_scores > t).astype(int)
        tp = np.sum((y_test == 1) & (preds == 1))
        fn = np.sum((y_test == 1) & (preds == 0))
        fp = np.sum((y_test == 0) & (preds == 1))
        tn = np.sum((y_test == 0) & (preds == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)

    plt.plot(fprs, tprs, linewidth=2, color='blue')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # 随机分类线
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('ROC样式曲线')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.tight_layout()
    plt.savefig("deep_svdd_final_results.png", dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------- 结果保存函数 --------------------------
def save_results_to_doc(results_dict, output_dir="results"):
    """将结果保存到文档"""

    # 创建结果目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存为文本文件
    txt_file = os.path.join(output_dir, f"deep_svdd_results_{timestamp}.txt")

    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("                Deep SVDD 叶片病害检测结果报告 (包含mAP)\n")
        f.write("=" * 70 + "\n\n")

        f.write("📅 报告生成时间: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("\n")

        f.write("📊 数据集信息:\n")
        f.write("-" * 50 + "\n")
        f.write(f"数据集路径: {results_dict['dataset_path']}\n")
        f.write(f"图片尺寸: {results_dict['image_size']}\n")
        f.write(f"采样比例: {results_dict['sample_ratio']}\n")
        f.write(f"训练集样本数: {results_dict.get('train_samples', 'N/A')}\n")
        f.write(f"测试集样本数: {results_dict.get('test_samples', 'N/A')}\n")
        f.write(f"测试集健康样本: {results_dict.get('healthy_samples', 'N/A')}\n")
        f.write(f"测试集病叶样本: {results_dict.get('diseased_samples', 'N/A')}\n")
        f.write("\n")

        f.write("⚙️ 训练参数:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Batch Size: {results_dict['batch_size']}\n")
        f.write(f"Epochs: {results_dict['epochs']}\n")
        f.write(f"学习率: {results_dict['learning_rate']}\n")
        f.write(f"潜在维度: {results_dict['latent_dim']}\n")
        f.write("\n")

        f.write("📈 模型性能指标 (包含mAP):\n")
        f.write("-" * 50 + "\n")
        f.write(f"准确率 (Accuracy): {results_dict['accuracy']:.4f}\n")
        f.write(f"精确率 (Precision): {results_dict['precision']:.4f}\n")
        f.write(f"召回率 (Recall): {results_dict['recall']:.4f}\n")
        f.write(f"F1分数 (F1-Score): {results_dict['f1_score']:.4f}\n")
        f.write(f"mAP (平均精度均值): {results_dict['mAP']:.4f}\n")
        if 'ap_per_class' in results_dict:
            ap_dict = results_dict['ap_per_class']
            f.write(f"  健康叶片 AP: {ap_dict.get('健康叶片', 0):.4f}\n")
            f.write(f"  病叶 AP: {ap_dict.get('病叶', 0):.4f}\n")
        f.write(f"最佳阈值: {results_dict['best_threshold']:.6f}\n")
        f.write("\n")

        f.write("📊 混淆矩阵:\n")
        f.write("-" * 50 + "\n")
        cm = results_dict['confusion_matrix']
        if cm:
            f.write("           预测健康    预测病叶\n")
            f.write(f"真实健康   {cm[0][0]:6d}      {cm[0][1]:6d}\n")
            f.write(f"真实病叶   {cm[1][0]:6d}      {cm[1][1]:6d}\n")
        f.write("\n")

        f.write("📊 详细统计:\n")
        f.write("-" * 50 + "\n")
        if cm:
            f.write(f"真阳性 (TP): {cm[1][1]}\n")
            f.write(f"真阴性 (TN): {cm[0][0]}\n")
            f.write(f"假阳性 (FP): {cm[0][1]}\n")
            f.write(f"假阴性 (FN): {cm[1][0]}\n")
            f.write(f"总样本数: {cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]}\n")
        f.write("\n")

        f.write("📈 异常分数统计:\n")
        f.write("-" * 50 + "\n")
        score_stats = results_dict.get('score_stats', {})
        if score_stats:
            f.write(
                f"健康样本平均分数: {score_stats.get('healthy_mean', 0):.4f} ± {score_stats.get('healthy_std', 0):.4f}\n")
            f.write(
                f"病叶样本平均分数: {score_stats.get('diseased_mean', 0):.4f} ± {score_stats.get('diseased_std', 0):.4f}\n")
            f.write(
                f"所有样本分数范围: [{score_stats.get('score_min', 0):.4f}, {score_stats.get('score_max', 0):.4f}]\n")
            f.write(
                f"所有样本平均分数: {score_stats.get('score_mean', 0):.4f} ± {score_stats.get('score_std', 0):.4f}\n")
        f.write("\n")

        # 训练历史
        if 'history' in results_dict and results_dict['history']:
            history = results_dict['history']
            f.write("📈 训练历史 (最后10个epochs):\n")
            f.write("-" * 50 + "\n")
            if hasattr(history, 'history'):
                losses = history.history.get('loss', [])
                val_losses = history.history.get('val_loss', [])

                # 只显示最后10个epochs
                start_idx = max(0, len(losses) - 10)
                for i in range(start_idx, len(losses)):
                    epoch_num = i + 1
                    loss = losses[i] if i < len(losses) else 'N/A'
                    val_loss = val_losses[i] if i < len(val_losses) else 'N/A'
                    f.write(f"Epoch {epoch_num:3d}: 训练损失={loss:.4f}, 验证损失={val_loss:.4f}\n")
            f.write("\n")

        f.write("📝 结果分析 (基于mAP):\n")
        f.write("-" * 50 + "\n")
        mAP_val = results_dict.get('mAP', 0)
        f1_val = results_dict.get('f1_score', 0)

        if mAP_val > 0.85 and f1_val > 0.85:
            f.write("✅ 模型表现优秀！mAP和F1分数均超过85%。\n")
            f.write("   建议：可以直接部署到实际应用中。\n")
        elif mAP_val > 0.75 and f1_val > 0.75:
            f.write("⚠️  模型表现良好，但仍有改进空间。\n")
            f.write("   建议：可以尝试增加训练数据或调整模型结构。\n")
        elif mAP_val > 0.65 and f1_val > 0.65:
            f.write("⚠️  模型表现一般，需要进一步优化。\n")
            f.write("   建议：调整训练参数，增加数据增强，或尝试其他模型架构。\n")
        else:
            f.write("❌ 模型表现不佳，需要重新设计。\n")
            f.write("   建议：检查数据质量，重新设计模型架构，或收集更多训练数据。\n")

        f.write("\n")
        f.write("💡 改进建议:\n")
        f.write("1. 如果mAP低，尝试增加训练样本\n")
        f.write("2. 调整模型参数（如潜在维度、学习率）\n")
        f.write("3. 使用更大的图片尺寸（如224x224）\n")
        f.write("4. 增加训练轮次（epochs）\n")
        f.write("5. 调整中心点初始化策略\n")
        f.write("6. 尝试不同的编码器架构\n")
        f.write("7. 调整异常分数计算方法\n")
        f.write("\n")

        f.write("📁 生成的文件:\n")
        f.write("-" * 50 + "\n")
        if 'generated_files' in results_dict:
            for file in results_dict['generated_files']:
                f.write(f"  • {file}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("                     报告结束\n")
        f.write("=" * 70 + "\n")

    print(f"✅ 文本报告已保存至: {txt_file}")

    # 2. 保存为JSON文件
    json_file = os.path.join(output_dir, f"deep_svdd_results_{timestamp}.json")

    # 转换历史对象为可序列化的格式
    serializable_results = results_dict.copy()
    if 'history' in serializable_results and serializable_results['history']:
        if hasattr(serializable_results['history'], 'history'):
            serializable_results['history'] = {
                'loss': serializable_results['history'].history.get('loss', []),
                'val_loss': serializable_results['history'].history.get('val_loss', [])
            }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON报告已保存至: {json_file}")

    return txt_file, json_file


# -------------------------- ASCII可视化函数 --------------------------
def print_ascii_results(history, test_scores, y_test, threshold, cm, mAP=None):
    """使用ASCII字符打印简单的可视化结果"""

    print("\n" + "=" * 70)
    print("                    ASCII 可视化结果")
    print("=" * 70)

    # 1. 训练历史
    if history and hasattr(history, 'history'):
        losses = history.history.get('loss', [])
        if len(losses) > 0:
            print("\n📈 训练历史 (损失曲线):")
            print("-" * 50)

            # 找到最大值和最小值
            max_loss = max(losses)
            min_loss = min(losses)
            loss_range = max_loss - min_loss

            if loss_range > 0:
                # 简单的ASCII图表
                for i, loss in enumerate(losses):
                    if i % max(1, len(losses) // 20) == 0:  # 显示部分点
                        bar_length = int(50 * (loss - min_loss) / loss_range)
                        bar = "█" * bar_length
                        print(f"Epoch {i + 1:3d}: {bar} {loss:.4f}")

    # 2. 分数分布
    if len(test_scores) > 0 and len(y_test) > 0:
        healthy_scores = test_scores[y_test == 0]
        diseased_scores = test_scores[y_test == 1]

        if len(healthy_scores) > 0 and len(diseased_scores) > 0:
            print("\n📊 异常分数分布:")
            print("-" * 50)

            # 创建简单的直方图
            min_score = min(np.min(healthy_scores), np.min(diseased_scores))
            max_score = max(np.max(healthy_scores), np.max(diseased_scores))
            score_range = max_score - min_score

            if score_range > 0:
                bins = 20
                bin_width = score_range / bins

                print(f"阈值线: {'━' * 20}│ 阈值={threshold:.4f}")
                print(f"{'健康':<10} {'病叶':<10} {'分数范围':<20}")
                print("-" * 50)

                for i in range(bins):
                    bin_start = min_score + i * bin_width
                    bin_end = min_score + (i + 1) * bin_width

                    healthy_count = np.sum((healthy_scores >= bin_start) & (healthy_scores < bin_end))
                    diseased_count = np.sum((diseased_scores >= bin_start) & (diseased_scores < bin_end))

                    # 创建简单的条形图
                    healthy_bar = "█" * int(20 * healthy_count / max(1, max(len(healthy_scores), len(diseased_scores))))
                    diseased_bar = "█" * int(
                        20 * diseased_count / max(1, max(len(healthy_scores), len(diseased_scores))))

                    # 标记阈值位置
                    threshold_marker = "│" if bin_start <= threshold < bin_end else " "

                    print(f"{healthy_bar:<20} {diseased_bar:<20} {threshold_marker} [{bin_start:.3f}, {bin_end:.3f})")

    # 3. 混淆矩阵
    if cm is not None:
        print("\n📊 混淆矩阵:")
        print("-" * 30)
        print(f"        预测健康  预测病叶")
        print(f"真实健康   {cm[0][0]:4d}       {cm[0][1]:4d}")
        print(f"真实病叶   {cm[1][0]:4d}       {cm[1][1]:4d}")

    # 4. mAP信息
    if mAP is not None:
        print(f"\n📊 mAP (平均精度均值): {mAP:.4f}")

    print("\n" + "=" * 70)


# -------------------------- 主函数 --------------------------
def main():
    try:
        print("=" * 70)
        print("           Deep SVDD 叶片病害检测系统 (包含mAP)")
        print("=" * 70)

        # 设置数据集路径
        DATASET_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\balanced-crop-dataset-new"

        # 检查数据集目录
        if not os.path.exists(DATASET_DIR):
            print(f"❌ 数据集目录不存在: {DATASET_DIR}")
            # 尝试查找其他可能的位置
            possible_paths = [
                r"E:\庄楷文\叶片病虫害识别\baseline\data\balanced-crop-dataset",
                r"E:\叶片病虫害识别\baseline\data\balanced-crop-dataset",
                os.path.join(os.path.dirname(DATASET_DIR), "balanced-crop-dataset")
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    print(f"✅ 找到数据集: {path}")
                    DATASET_DIR = path
                    break
            else:
                print("❌ 未找到数据集，请检查路径")
                return

        print(f"✅ 使用数据集路径: {DATASET_DIR}")

        # 加载数据
        X_train, y_train, X_test, y_test = load_and_preprocess_data(
            DATASET_DIR, IMAGE_SIZE, SAMPLE_RATIO
        )

        if X_train is None or len(X_train) == 0:
            print("❌ 没有找到训练数据")
            return

        print(f"\n📊 数据集统计:")
        print(f"  训练集样本数: {len(X_train)}")
        print(f"  测试集样本数: {len(X_test)}")
        print(f"  测试集健康样本: {np.sum(y_test == 0)}")
        print(f"  测试集病叶样本: {np.sum(y_test == 1)}")

        # 训练模型
        input_shape = X_train.shape[1:]
        deep_svdd, encoder, history = train_simple_svdd(
            X_train, input_shape, LATENT_DIM, EPOCHS, BATCH_SIZE
        )

        # 评估模型
        accuracy, precision, recall, f1, threshold, test_scores, cm, predictions, score_stats, mAP, ap_per_class = evaluate_model(
            deep_svdd, X_test, y_test
        )

        # 可视化结果
        visualize_results(history, test_scores, y_test, threshold, cm, mAP, ap_per_class)

        # 准备结果字典
        results_dict = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_path': DATASET_DIR,
            'image_size': IMAGE_SIZE,
            'sample_ratio': SAMPLE_RATIO,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'latent_dim': LATENT_DIM,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'healthy_samples': int(np.sum(y_test == 0)),
            'diseased_samples': int(np.sum(y_test == 1)),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'mAP': float(mAP),
            'ap_per_class': ap_per_class,
            'best_threshold': float(threshold),
            'confusion_matrix': cm.tolist() if cm is not None else None,
            'test_scores': test_scores.tolist() if test_scores is not None else None,
            'y_test': y_test.tolist() if y_test is not None else None,
            'predictions': predictions.tolist() if predictions is not None else None,
            'score_stats': score_stats,
            'history': history,
            'generated_files': [
                "deep_svdd_encoder.h5",
                "deep_svdd_final_results.png",
                "deep_svdd_pr_curves.png"
            ]
        }

        # 保存结果到文档
        txt_file, json_file = save_results_to_doc(results_dict)

        # 打印ASCII可视化结果
        print_ascii_results(history, test_scores, y_test, threshold, cm, mAP)

        # 输出总结
        print("\n" + "=" * 70)
        print("                    训练完成!")
        print("=" * 70)
        print(f"📊 最终性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  mAP:    {mAP:.4f}")
        if ap_per_class:
            print(f"  健康叶片 AP: {ap_per_class.get('健康叶片', 0):.4f}")
            print(f"  病叶 AP: {ap_per_class.get('病叶', 0):.4f}")
        print(f"  最佳阈值: {threshold:.6f}")
        print("\n💾 保存的文件:")
        print(f"  文本报告: {txt_file}")
        print(f"  JSON报告: {json_file}")
        print(f"  编码器模型: deep_svdd_encoder.h5")
        print(f"  可视化结果: deep_svdd_final_results.png")
        print(f"  PR曲线: deep_svdd_pr_curves.png")
        print("\n🎉 所有任务完成!")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

        # 保存错误日志
        os.makedirs("results", exist_ok=True)
        error_log = os.path.join("results", f"error_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(error_log, 'w', encoding='utf-8') as f:
            f.write(f"错误时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"错误追踪:\n{traceback.format_exc()}\n")

        print(f"📝 错误日志已保存至: {error_log}")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    main()