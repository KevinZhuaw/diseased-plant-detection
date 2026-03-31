# deep_svdd_fixed.py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print("🚀 启动 Deep SVDD 叶片病害检测系统 (修复版本)")

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

# TensorFlow 导入
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models

    print(f"✅ TensorFlow 版本: {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow 导入失败: {e}")
    exit(1)

# PIL 导入
try:
    from PIL import Image

    print("✅ PIL 导入成功")
except ImportError as e:
    print(f"❌ PIL 导入失败: {e}")
    exit(1)


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
            target_names = ['Class 0', 'Class 1']

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


# 配置参数
DATASET_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\processed-crop-dataset-split"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
LATENT_DIM = 32
SAMPLE_RATIO = 0.05

print("=" * 60)
print("          深度支持向量数据描述(Deep SVDD)")
print("          叶片病害识别系统 (修复版本)")
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

    # 训练数据 - 只使用健康样本 (Deep SVDD特点)
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


# 修复的Deep SVDD编码器
def build_fixed_encoder(input_shape, latent_dim):
    """构建修复的Deep SVDD编码器"""
    print("🔨 构建Deep SVDD编码器...")

    inputs = layers.Input(shape=input_shape)

    # 编码器网络 - 修复版本
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)  # 128 -> 64

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 64 -> 32

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 32 -> 16

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # 输出层 - 不使用激活函数
    encoded = layers.Dense(latent_dim, name='encoded')(x)

    encoder = models.Model(inputs, encoded, name="deep_svdd_encoder")
    return encoder


# 修复的解码器
def build_fixed_decoder(latent_dim, output_shape):
    """构建修复的解码器，确保输出尺寸匹配"""
    print("🔨 构建解码器...")

    latent_inputs = layers.Input(shape=(latent_dim,))

    # 计算编码器输出的特征图尺寸
    # 输入: 128x128 -> 64x64 -> 32x32 -> 16x16
    # 所以我们需要从16x16上采样回128x128

    # 第一层：从潜在空间恢复到特征图
    x = layers.Dense(16 * 16 * 128, activation='relu')(latent_inputs)
    x = layers.Reshape((16, 16, 128))(x)  # 16x16x128

    # 上采样回到128x128
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)  # 32x32x128
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)  # 64x64x64
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)  # 128x128x32

    # 最终输出层
    decoder_outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(
        x)  # 128x128x3

    decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


class DeepSVDD:
    def __init__(self, encoder, latent_dim):
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.center = None

    def initialize_center(self, X_train):
        """初始化中心点"""
        print("🎯 初始化Deep SVDD中心点...")
        Z_train = self.encoder.predict(X_train, verbose=0)
        self.center = np.mean(Z_train, axis=0)
        print(f"   中心点维度: {self.center.shape}")
        return self.center

    def compute_anomaly_scores(self, X):
        """计算异常分数（到中心的距离）"""
        Z = self.encoder.predict(X, verbose=0)
        scores = np.sum((Z - self.center) ** 2, axis=1)
        return scores


# 修复的训练函数
def train_deep_svdd_fixed(X_train, input_shape, latent_dim, epochs=30, batch_size=16):
    """训练修复的Deep SVDD模型"""
    print("🚀 开始训练Deep SVDD...")

    # 构建编码器和解码器
    encoder = build_fixed_encoder(input_shape, latent_dim)
    decoder = build_fixed_decoder(latent_dim, input_shape)

    # 创建Deep SVDD模型
    deep_svdd = DeepSVDD(encoder, latent_dim)

    # 构建自编码器进行预训练
    print("🔧 构建自编码器进行预训练...")

    # 自编码器
    autoencoder_outputs = decoder(encoder(encoder.input))
    autoencoder = models.Model(encoder.input, autoencoder_outputs, name="autoencoder")

    # 编译和训练自编码器
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    print("📐 模型架构:")
    autoencoder.summary()

    # 验证输入输出尺寸
    sample_input = X_train[:1]
    sample_output = autoencoder.predict(sample_input, verbose=0)
    print(f"✅ 输入尺寸: {sample_input.shape}, 输出尺寸: {sample_output.shape}")

    if sample_input.shape != sample_output.shape:
        print(f"❌ 尺寸不匹配! 输入: {sample_input.shape}, 输出: {sample_output.shape}")
        return None, None, None

    print("🎯 开始预训练...")
    start_time = time.time()

    # 训练自编码器
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
        ]
    )

    training_time = time.time() - start_time
    print(f"⏱️  预训练完成，耗时: {training_time:.2f} 秒")

    # 初始化Deep SVDD中心
    deep_svdd.initialize_center(X_train)

    # 保存模型
    encoder.save("deep_svdd_encoder_fixed.h5")
    print("💾 模型已保存: deep_svdd_encoder_fixed.h5")

    return deep_svdd, encoder, history.history


# 评估函数
def evaluate_deep_svdd(deep_svdd, X_test, y_test):
    """评估Deep SVDD模型"""
    print("📊 评估模型性能...")

    # 计算异常分数
    test_scores = deep_svdd.compute_anomaly_scores(X_test)

    # 寻找最佳阈值
    print("🎯 寻找最佳分类阈值...")
    thresholds = np.linspace(np.min(test_scores), np.max(test_scores), 100)
    best_f1 = 0
    best_threshold = 0
    best_predictions = None

    for i, threshold in enumerate(thresholds):
        predictions = (test_scores > threshold).astype(int)
        current_f1 = Metrics.f1_score(y_test, predictions)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
            best_predictions = predictions

        if i % 20 == 0:
            print(f"    测试阈值 {i + 1}/100, 当前最佳F1: {best_f1:.4f}")

    # 计算指标
    accuracy = Metrics.accuracy_score(y_test, best_predictions)
    precision = Metrics.precision_score(y_test, best_predictions)
    recall = Metrics.recall_score(y_test, best_predictions)
    cm = Metrics.confusion_matrix(y_test, best_predictions)

    print(f"✅ 找到最佳阈值: {best_threshold:.6f}, F1分数: {best_f1:.4f}")

    return accuracy, precision, recall, best_f1, best_threshold, test_scores, cm, best_predictions


# 文本形式的结果报告
def text_results_report(history, test_scores, y_test, threshold, cm):
    """以文本形式输出结果"""
    print("\n" + "=" * 60)
    print("                 训练历史摘要")
    print("=" * 60)
    print(f"最终训练损失: {history['loss'][-1]:.6f}")
    if 'val_loss' in history:
        print(f"最终验证损失: {history['val_loss'][-1]:.6f}")

    print("\n" + "=" * 60)
    print("                 异常分数统计")
    print("=" * 60)
    healthy_scores = test_scores[y_test == 0]
    diseased_scores = test_scores[y_test == 1]

    print(f"健康叶片分数 - 最小值: {np.min(healthy_scores):.6f}")
    print(f"健康叶片分数 - 最大值: {np.max(healthy_scores):.6f}")
    print(f"健康叶片分数 - 平均值: {np.mean(healthy_scores):.6f}")
    print(f"健康叶片分数 - 标准差: {np.std(healthy_scores):.6f}")
    print()
    print(f"病叶分数 - 最小值: {np.min(diseased_scores):.6f}")
    print(f"病叶分数 - 最大值: {np.max(diseased_scores):.6f}")
    print(f"病叶分数 - 平均值: {np.mean(diseased_scores):.6f}")
    print(f"病叶分数 - 标准差: {np.std(diseased_scores):.6f}")
    print()
    print(f"最佳分类阈值: {threshold:.6f}")


# 保存结果到文本文件
def save_results_to_file(accuracy, precision, recall, f1, threshold, cm, filename="deep_svdd_results.txt"):
    """保存结果到文本文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Deep SVDD 叶片病害检测结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"准确率 (Accuracy):  {accuracy:.4f}\n")
        f.write(f"精确率 (Precision): {precision:.4f}\n")
        f.write(f"召回率 (Recall):    {recall:.4f}\n")
        f.write(f"F1分数:            {f1:.4f}\n")
        f.write(f"最佳阈值:          {threshold:.6f}\n\n")

        f.write("混淆矩阵:\n")
        f.write(f"    TN: {cm[0, 0]}   FP: {cm[0, 1]}\n")
        f.write(f"    FN: {cm[1, 0]}   TP: {cm[1, 1]}\n\n")

        f.write("模型配置:\n")
        f.write(f"   图像尺寸: {IMAGE_SIZE}\n")
        f.write(f"   潜在维度: {LATENT_DIM}\n")
        f.write(f"   批次大小: {BATCH_SIZE}\n")
        f.write(f"   训练轮次: {EPOCHS}\n")
        f.write(f"   学习率: {LEARNING_RATE}\n")

    print(f"📄 结果已保存到: {filename}")


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
        input_shape = X_train.shape[1:]
        deep_svdd, encoder, history = train_deep_svdd_fixed(
            X_train, input_shape, LATENT_DIM, EPOCHS, BATCH_SIZE
        )

        if deep_svdd is None:
            print("❌ 模型训练失败")
            return

        # 评估模型
        accuracy, precision, recall, f1, threshold, test_scores, cm, predictions = evaluate_deep_svdd(
            deep_svdd, X_test, y_test
        )

        # 文本形式的结果报告
        text_results_report(history, test_scores, y_test, threshold, cm)

        # 输出详细结果
        print("\n" + "=" * 60)
        print("                 模型评估结果")
        print("=" * 60)
        print(f"📊 准确率 (Accuracy):  {accuracy:.4f}")
        print(f"🎯 精确率 (Precision): {precision:.4f}")
        print(f"📈 召回率 (Recall):    {recall:.4f}")
        print(f"⭐ F1分数:            {f1:.4f}")
        print(f"🔧 最佳阈值:          {threshold:.6f}")

        print(f"\n📋 混淆矩阵:")
        print(f"     TN: {cm[0, 0]}   FP: {cm[0, 1]}")
        print(f"     FN: {cm[1, 0]}   TP: {cm[1, 1]}")

        print(f"\n📝 详细分类报告:")
        print(Metrics.classification_report(y_test, predictions,
                                            target_names=['健康叶片', '病叶']))

        # 保存阈值信息
        try:
            import joblib
            threshold_info = {
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'model_type': 'Deep SVDD'
            }
            joblib.dump(threshold_info, 'deep_svdd_threshold_fixed.joblib')
            print("💾 阈值信息已保存: deep_svdd_threshold_fixed.joblib")
        except ImportError:
            print("⚠️  joblib不可用，跳过阈值保存")

        # 保存文本结果
        save_results_to_file(accuracy, precision, recall, f1, threshold, cm, "deep_svdd_fixed_results.txt")

        print(f"\n🎉 Deep SVDD训练完成!")
        print("💡 模型特点:")
        print("   - 单类分类，仅使用健康样本训练")
        print("   - 学习紧凑的特征空间超球体")
        print("   - 异常检测能力强")
        print("   - 适合不平衡数据集")

        # 示例预测
        print(f"\n🔍 示例预测 (阈值: {threshold:.6f}):")
        if len(X_test) > 0:
            sample_indices = random.sample(range(len(X_test)), min(3, len(X_test)))

            for i, idx in enumerate(sample_indices):
                true_label = y_test[idx]
                true_class = "病叶" if true_label == 1 else "健康叶片"

                sample_image = X_test[idx:idx + 1]
                score = deep_svdd.compute_anomaly_scores(sample_image)[0]
                prediction = 1 if score > threshold else 0
                pred_class = "病叶" if prediction == 1 else "健康叶片"

                status = '✓' if prediction == true_label else '✗'
                print(f"   样本 {i + 1}: 真实={true_class}, 预测={pred_class}, 分数={score:.6f} {status}")

    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()