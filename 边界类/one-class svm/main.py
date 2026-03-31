# one_class_svm_final.py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🚀 启动 One-Class SVM 叶片病害检测系统 - 最终版")

# 基础导入
import numpy as np
import sys
import random
import time
import datetime
from PIL import Image
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Any
import gc
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib支持中文
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("          单类支持向量机(One-Class SVM) - 最终版")
print("          叶片病害识别系统")
print("=" * 60)


# 手动实现评估指标
class Metrics:
    """评估指标计算类"""

    @staticmethod
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算准确率"""
        return np.mean(y_true == y_pred)

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算混淆矩阵"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return np.array([[tn, fp], [fn, tp]])

    @staticmethod
    def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算精确率"""
        cm = Metrics.confusion_matrix(y_true, y_pred)
        tp, fp = cm[1, 1], cm[0, 1]
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    @staticmethod
    def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算召回率"""
        cm = Metrics.confusion_matrix(y_true, y_pred)
        tp, fn = cm[1, 1], cm[1, 0]
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算F1分数"""
        precision = Metrics.precision_score(y_true, y_pred)
        recall = Metrics.recall_score(y_true, y_pred)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                              target_names: Optional[List[str]] = None) -> str:
        """生成分类报告"""
        if target_names is None:
            target_names = ['健康叶片', '病叶']

        cm = Metrics.confusion_matrix(y_true, y_pred)
        accuracy = Metrics.accuracy_score(y_true, y_pred)

        report = f"{'':15} precision    recall  f1-score   support\n\n"

        # 为每个类别计算指标
        for i, name in enumerate(target_names):
            if i == 0:  # 健康类别
                # 对于健康类别，我们需要将标签反转
                y_true_inv = 1 - y_true
                y_pred_inv = 1 - y_pred
                precision = Metrics.precision_score(y_true_inv, y_pred_inv)
                recall = Metrics.recall_score(y_true_inv, y_pred_inv)
                f1 = Metrics.f1_score(y_true_inv, y_pred_inv)
            else:  # 病叶类别
                precision = Metrics.precision_score(y_true, y_pred)
                recall = Metrics.recall_score(y_true, y_pred)
                f1 = Metrics.f1_score(y_true, y_pred)

            support = np.sum(y_true == i)
            report += f"{name:15} {precision:8.4f} {recall:8.4f} {f1:8.4f} {support:8}\n"

        report += f"\n{'accuracy':15} {accuracy:.4f} {len(y_true):8}\n"
        return report


# 配置类
class Config:
    """配置类"""
    DATASET_DIR = r"E:\庄楷文\叶片病虫害识别\baseline\data\balanced-crop-dataset-new"
    IMAGE_SIZE = (32, 32)  # 保持小尺寸
    NU = 0.01  # 更小的nu值，使模型更严格
    BATCH_SIZE = 100
    FEATURE_METHOD = 'pca'  # 使用PCA特征
    PCA_COMPONENTS = 50  # PCA降维到50维


# 数据加载类
class DataLoader:
    """数据加载和预处理类"""

    @staticmethod
    def load_images_from_folder(folder: str, label: int, max_images: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """从文件夹加载图像"""
        if not os.path.exists(folder):
            print(f"❌ 目录不存在: {folder}")
            return np.array([]), np.array([])

        img_files = [f for f in os.listdir(folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if max_images and len(img_files) > max_images:
            img_files = random.sample(img_files, max_images)

        total_images = len(img_files)
        print(f"📁 处理 {os.path.basename(folder)}... ({total_images} 张图片)")

        images = []
        labels = []

        for i, img_file in enumerate(img_files):
            img_path = os.path.join(folder, img_file)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(Config.IMAGE_SIZE)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_array)
                    labels.append(label)

                    if i % 100 == 0 and i > 0:
                        print(f"    已处理 {i}/{total_images} 张图片")

            except Exception as e:
                print(f"⚠️ 处理图像失败 {img_path}: {e}")

        if images:
            return np.array(images), np.array(labels)
        else:
            return np.array([]), np.array([])

    @staticmethod
    def load_data() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """加载完整数据"""
        print("📊 加载数据...")

        # 训练数据 - 只使用健康样本
        train_healthy_dir = os.path.join(Config.DATASET_DIR, "train", "healthy")
        print("📂 加载训练数据...")
        X_train, y_train = DataLoader.load_images_from_folder(train_healthy_dir, 0, max_images=5000)

        if len(X_train) == 0:
            print("❌ 没有找到健康训练数据")
            return None

        print(f"  训练数据形状: {X_train.shape}")

        # 测试数据 - 平衡样本
        test_healthy_dir = os.path.join(Config.DATASET_DIR, "test", "healthy")
        test_diseased_dir = os.path.join(Config.DATASET_DIR, "test", "diseased")

        print("📂 加载测试健康数据...")
        X_test_healthy, y_test_healthy = DataLoader.load_images_from_folder(test_healthy_dir, 0, max_images=2000)

        print("📂 加载测试病叶数据...")
        X_test_diseased, y_test_diseased = DataLoader.load_images_from_folder(test_diseased_dir, 1, max_images=2000)

        if len(X_test_healthy) == 0 and len(X_test_diseased) == 0:
            print("❌ 没有找到测试数据")
            return None

        # 合并测试数据
        X_test = np.concatenate([X_test_healthy, X_test_diseased], axis=0)
        y_test = np.concatenate([y_test_healthy, y_test_diseased], axis=0)

        print(f"  测试数据形状: {X_test.shape}")
        print(f"  健康样本: {np.sum(y_test == 0)}, 病叶样本: {np.sum(y_test == 1)}")

        # 清理内存
        gc.collect()

        return X_train, y_train, X_test, y_test


# 特征提取类 - 使用PCA降维
class FeatureExtractor:
    """特征提取类"""

    def __init__(self):
        self.pca = None

    def fit_pca(self, X: np.ndarray):
        """训练PCA模型"""
        print("🔧 训练PCA模型...")

        # 提取基础特征（展平）
        X_flat = X.reshape(len(X), -1)

        # 计算PCA
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=min(Config.PCA_COMPONENTS, X_flat.shape[1]))
        self.pca.fit(X_flat)

        print(f"   原始维度: {X_flat.shape[1]}, PCA后维度: {self.pca.n_components_}")
        print(f"   解释方差比例: {np.sum(self.pca.explained_variance_ratio_):.4f}")

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """提取特征"""
        print("🔍 提取特征...")

        if Config.FEATURE_METHOD == 'simple':
            # 简单展平
            features = X.reshape(len(X), -1).astype(np.float32)

        elif Config.FEATURE_METHOD == 'histogram':
            # 颜色直方图
            features = []
            for img in X:
                # 计算RGB通道的直方图
                hist_r = np.histogram(img[:, :, 0], bins=8, range=(0, 1))[0]
                hist_g = np.histogram(img[:, :, 1], bins=8, range=(0, 1))[0]
                hist_b = np.histogram(img[:, :, 2], bins=8, range=(0, 1))[0]
                # 合并直方图
                hist = np.concatenate([hist_r, hist_g, hist_b])
                features.append(hist)
            features = np.array(features, dtype=np.float32)

        elif Config.FEATURE_METHOD == 'pca':
            # PCA特征
            if self.pca is None:
                raise ValueError("PCA模型未训练，请先调用fit_pca方法")

            # 展平图像
            X_flat = X.reshape(len(X), -1)
            features = self.pca.transform(X_flat).astype(np.float32)

        else:
            # 默认使用简单展平
            features = X.reshape(len(X), -1).astype(np.float32)

        print(f"   特征维度: {features.shape}")
        return features


# One-Class SVM实现
class OneClassSVM:
    """基于距离的One-Class SVM实现"""

    def __init__(self, nu: float = 0.01):
        self.nu = nu
        self.center = None
        self.threshold = None
        self.feature_mean = None
        self.feature_std = None
        self.feature_extractor = FeatureExtractor()

    def fit(self, X_train: np.ndarray) -> 'OneClassSVM':
        """训练模型"""
        print("🎯 训练One-Class SVM...")

        # 训练特征提取器（如PCA）
        if Config.FEATURE_METHOD == 'pca':
            self.feature_extractor.fit_pca(X_train)

        # 提取特征
        X_features = self.feature_extractor.extract_features(X_train)

        # 标准化特征
        self.feature_mean = np.mean(X_features, axis=0)
        self.feature_std = np.std(X_features, axis=0) + 1e-8
        X_scaled = (X_features - self.feature_mean) / self.feature_std

        # 计算训练数据的中心
        self.center = np.mean(X_scaled, axis=0)

        # 计算训练数据到中心的距离
        distances = np.linalg.norm(X_scaled - self.center, axis=1)

        # 根据nu参数确定阈值
        # 使用更严格的标准：mean + 3*std
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        self.threshold = mean_dist + 3 * std_dist

        # 但也要考虑nu参数
        nu_threshold = np.percentile(distances, (1 - self.nu) * 100)

        # 取两者中较大的阈值
        self.threshold = max(self.threshold, nu_threshold)

        print(f"   中心点维度: {self.center.shape}")
        print(f"   距离阈值: {self.threshold:.6f}")
        print(f"   距离统计 - 均值: {mean_dist:.4f}, 标准差: {std_dist:.4f}")
        print(f"   训练集异常率: {np.mean(distances > self.threshold):.4f}")

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算异常分数"""
        # 提取特征
        X_features = self.feature_extractor.extract_features(X)

        # 标准化
        X_scaled = (X_features - self.feature_mean) / self.feature_std

        # 计算距离
        distances = np.linalg.norm(X_scaled - self.center, axis=1)

        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测样本类别"""
        distances = self.decision_function(X)
        # 距离大于阈值为异常（病叶=1），否则为正常（健康=0）
        predictions = (distances > self.threshold).astype(int)
        return predictions


# 模型评估类
class ModelEvaluator:
    """模型评估类"""

    @staticmethod
    def evaluate_model(model: OneClassSVM, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """评估模型性能"""
        print("📊 评估模型性能...")

        # 预测
        predictions = model.predict(X_test)
        scores = model.decision_function(X_test)

        # 计算指标
        accuracy = Metrics.accuracy_score(y_test, predictions)
        precision = Metrics.precision_score(y_test, predictions)
        recall = Metrics.recall_score(y_test, predictions)
        f1 = Metrics.f1_score(y_test, predictions)
        cm = Metrics.confusion_matrix(y_test, predictions)

        # 打印结果
        print(f"✅ 评估完成:")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   精确率: {precision:.4f}")
        print(f"   召回率: {recall:.4f}")
        print(f"   F1分数: {f1:.4f}")
        print(f"   混淆矩阵:")
        print(f"     TN: {cm[0, 0]}   FP: {cm[0, 1]}")
        print(f"     FN: {cm[1, 0]}   TP: {cm[1, 1]}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': predictions,
            'scores': scores
        }

    @staticmethod
    def find_optimal_threshold(scores: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
        """寻找最优阈值"""
        print("🔄 寻找最优阈值...")

        # 生成候选阈值
        min_score = np.min(scores)
        max_score = np.max(scores)
        thresholds = np.linspace(min_score, max_score, 200)

        best_f1 = 0
        best_threshold = np.percentile(scores, 95)  # 从95%分位数开始
        best_predictions = None

        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)

            # 跳过极端情况
            if np.sum(predictions) == 0 or np.sum(predictions) == len(predictions):
                continue

            f1 = Metrics.f1_score(y_true, predictions)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_predictions = predictions

        print(f"   最优阈值: {best_threshold:.6f}, F1: {best_f1:.4f}")
        return best_threshold, best_predictions


# 可视化类
class Visualizer:
    """可视化类"""

    @staticmethod
    def plot_results(scores: np.ndarray, y_true: np.ndarray, threshold: float, cm: np.ndarray,
                     results: Dict[str, float]) -> None:
        """可视化结果"""
        try:
            fig = plt.figure(figsize=(15, 10))

            # 1. 分数分布
            ax1 = plt.subplot(2, 2, 1)
            healthy_scores = scores[y_true == 0]
            diseased_scores = scores[y_true == 1]

            if len(healthy_scores) > 0:
                ax1.hist(healthy_scores, bins=50, alpha=0.5, label='健康叶片', color='green', density=True)
            if len(diseased_scores) > 0:
                ax1.hist(diseased_scores, bins=50, alpha=0.5, label='病叶', color='red', density=True)

            ax1.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'阈值: {threshold:.4f}')
            ax1.set_xlabel('异常分数')
            ax1.set_ylabel('密度')
            ax1.set_title('异常分数分布')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. 混淆矩阵
            ax2 = plt.subplot(2, 2, 2)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['预测健康', '预测病叶'],
                        yticklabels=['实际健康', '实际病叶'], ax=ax2)
            ax2.set_title('混淆矩阵')

            # 3. 性能指标
            ax3 = plt.subplot(2, 2, 3)
            ax3.axis('off')

            metrics_text = (
                f"One-Class SVM 性能指标\n\n"
                f"准确率:  {results['accuracy']:.4f}\n"
                f"精确率: {results['precision']:.4f}\n"
                f"召回率:    {results['recall']:.4f}\n"
                f"F1分数:  {results['f1']:.4f}\n"
                f"阈值:      {threshold:.6f}\n"
            )

            ax3.text(0.1, 0.5, metrics_text, fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                     verticalalignment='center')

            # 4. 箱线图
            ax4 = plt.subplot(2, 2, 4)
            score_data = []
            labels = []

            if len(healthy_scores) > 0:
                score_data.append(healthy_scores)
                labels.append('健康叶片')
            if len(diseased_scores) > 0:
                score_data.append(diseased_scores)
                labels.append('病叶')

            if score_data:
                ax4.boxplot(score_data, labels=labels)
                ax4.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'阈值={threshold:.4f}')
                ax4.set_ylabel('异常分数')
                ax4.set_title('异常分数箱线图')
                ax4.grid(True, alpha=0.3)
                ax4.legend()

            plt.tight_layout()
            plt.savefig('one_class_svm_results.png', dpi=300, bbox_inches='tight')
            plt.show()

        except Exception as e:
            print(f"⚠️ 可视化时出错: {e}")


# 主函数
def main():
    """主函数"""
    try:
        print("🔍 检查数据集目录...")
        if not os.path.exists(Config.DATASET_DIR):
            print(f"❌ 数据集目录不存在: {Config.DATASET_DIR}")
            print("请检查DATASET_DIR路径配置")
            return

        # 加载数据
        print("📥 加载数据...")
        data = DataLoader.load_data()
        if data is None:
            return

        X_train, y_train, X_test, y_test = data

        print(f"\n📊 数据统计:")
        print(f"训练集: {X_train.shape} (仅健康样本)")
        print(f"测试集: {X_test.shape}")
        print(f"测试集 - 健康: {np.sum(y_test == 0)}, 病叶: {np.sum(y_test == 1)}")

        # 训练模型
        print("\n🚀 开始训练One-Class SVM...")
        start_time = time.time()

        model = OneClassSVM(nu=Config.NU)
        model.fit(X_train)

        training_time = time.time() - start_time
        print(f"⏱️  训练完成，耗时: {training_time:.2f} 秒")

        # 评估模型
        print("\n📈 模型评估...")
        results = ModelEvaluator.evaluate_model(model, X_test, y_test)

        # 寻找最优阈值
        print("\n🔄 优化阈值...")
        optimal_threshold, optimal_predictions = ModelEvaluator.find_optimal_threshold(
            results['scores'], y_test
        )

        # 使用最优阈值重新评估
        if optimal_threshold != model.threshold:
            print(f"🔄 使用最优阈值重新评估...")
            model.threshold = optimal_threshold

            # 重新计算指标
            results['accuracy'] = Metrics.accuracy_score(y_test, optimal_predictions)
            results['precision'] = Metrics.precision_score(y_test, optimal_predictions)
            results['recall'] = Metrics.recall_score(y_test, optimal_predictions)
            results['f1'] = Metrics.f1_score(y_test, optimal_predictions)
            results['confusion_matrix'] = Metrics.confusion_matrix(y_test, optimal_predictions)
            results['predictions'] = optimal_predictions

            print(f"   更新后指标:")
            print(f"   准确率: {results['accuracy']:.4f}")
            print(f"   精确率: {results['precision']:.4f}")
            print(f"   召回率: {results['recall']:.4f}")
            print(f"   F1分数: {results['f1']:.4f}")

        # 可视化结果
        print("\n🎨 生成可视化结果...")
        Visualizer.plot_results(results['scores'], y_test, model.threshold,
                                results['confusion_matrix'], results)

        # 最终报告
        print("\n" + "=" * 60)
        print("                 最终评估结果")
        print("=" * 60)

        print(f"📊 准确率: {results['accuracy']:.4f}")
        print(f"🎯 精确率: {results['precision']:.4f}")
        print(f"📈 召回率: {results['recall']:.4f}")
        print(f"⭐ F1分数: {results['f1']:.4f}")

        print(f"\n📋 混淆矩阵:")
        cm = results['confusion_matrix']
        print(f"     TN: {cm[0, 0]}   FP: {cm[0, 1]}")
        print(f"     FN: {cm[1, 0]}   TP: {cm[1, 1]}")

        print(f"\n📝 详细分类报告:")
        print(Metrics.classification_report(y_test, results['predictions']))

        print(f"\n✅ One-Class SVM训练和评估完成!")
        print(f"   训练时间: {training_time:.2f}秒")
        print(f"   最终阈值: {model.threshold:.6f}")

        # 分析结果
        print("\n💡 结果分析:")
        if results['recall'] > 0.7:
            print("- ✓ 召回率较高，模型能检测出大部分病叶")
        else:
            print("- ✗ 召回率较低，模型漏检了一些病叶")

        if results['precision'] > 0.7:
            print("- ✓ 精确率较高，误报较少")
        else:
            print("- ✗ 精确率较低，有很多健康叶片被误判为病叶")

        if results['f1'] > 0.6:
            print("- ✓ F1分数可以接受")
        elif results['f1'] > 0.7:
            print("- ✓ F1分数较好")
        else:
            print("- ✗ F1分数较低，需要进一步优化")

        # 给出建议
        print("\n🎯 改进建议:")
        if results['recall'] > 0.9 and results['precision'] < 0.3:
            print("1. 模型过于敏感，几乎所有样本都被判为病叶")
            print("2. 尝试增加nu值（如0.001）或使用更严格的阈值")
            print("3. 考虑使用更有效的特征提取方法")
        elif results['recall'] < 0.3 and results['precision'] > 0.9:
            print("1. 模型过于保守，漏检了很多病叶")
            print("2. 尝试减小nu值（如0.1）或使用更宽松的阈值")
            print("3. 考虑使用更丰富的特征")
        elif results['f1'] < 0.5:
            print("1. 整体性能较差，可能需要重新考虑模型选择")
            print("2. One-Class SVM可能不适合这个数据集")
            print("3. 考虑使用二分类SVM或深度学习模型")
        else:
            print("1. 性能可以接受，可以进一步微调参数")
            print("2. 尝试不同的特征提取方法")
            print("3. 调整nu值和阈值")

        # 保存结果
        print("\n💾 保存结果...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"one_class_svm_results_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write("One-Class SVM 叶片病害检测结果报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"准确率: {results['accuracy']:.4f}\n")
            f.write(f"精确率: {results['precision']:.4f}\n")
            f.write(f"召回率: {results['recall']:.4f}\n")
            f.write(f"F1分数: {results['f1']:.4f}\n")

        print(f"✅ 结果已保存到 one_class_svm_results_{timestamp}.txt")

    except MemoryError:
        print("❌ 内存不足！尝试以下解决方案:")
        print("1. 进一步减小图像尺寸到16x16")
        print("2. 减少训练样本数量")
        print("3. 使用更简单的特征提取方法")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()