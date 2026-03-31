"""
One-Class SVM - 简化版训练与图像识别系统
只包含训练和识别功能，输出相似格式的结果
"""

import os
import sys
import time
import glob
from datetime import datetime
import json
import warnings
import numpy as np
from PIL import Image
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

warnings.filterwarnings('ignore')

# ==================== 基本检查 ====================
print("=" * 60)
print("🔍 One-Class SVM 环境检查...")
print("=" * 60)

print(f"Python版本: {sys.version}")
print(f"✅ NumPy版本: {np.__version__}")

try:
    from PIL import Image
    print("✅ PIL导入成功")
except Exception as e:
    print(f"❌ PIL导入失败: {e}")
    sys.exit(1)

# ==================== 配置类 ====================
class Config:
    """配置类 - 与Deep SVDD保持相似配置"""
    # 路径配置
    DATASET_PATH = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"
    RESULTS_PATH = r"E:\zhuangkaiwen\results\oneclass_svm"

    # 识别结果保存路径
    RECOGNITION_RESULTS_PATH = r"E:\zhuangkaiwen\results\oneclass_svm_recognition"

    # 要识别的图像路径
    RECOGNITION_IMAGE_PATH = r"E:\zhuangkaiwen\test_images"

    # 图像处理参数
    IMG_SIZE = 96  # 与Deep SVDD相同
    BATCH_SIZE = 32  # 与Deep SVDD相同

    # One-Class SVM参数
    NU = 0.1  # 异常值比例估计
    KERNEL = 'rbf'  # 核函数类型
    GAMMA = 'scale'  # RBF核参数
    PCA_COMPONENTS = 50  # PCA降维维度
    FEATURE_EXTRACTOR = 'hog'  # 特征提取方法: 'hog', 'lbp', 'color_hist', 'combined'

    # 训练配置
    TRAIN_CLASS_COUNT = 22  # 与Deep SVDD相同

    # 识别配置
    CONFIDENCE_THRESHOLD = 0.7  # 置信度阈值
    TOP_K_RESULTS = 3  # 显示前K个结果

    # 性能优化
    PARALLEL_JOBS = -1  # 并行工作数(-1使用所有核心)

# ==================== 特征提取器 ====================
class FeatureExtractor:
    """多种特征提取方法"""

    def __init__(self, method='hog'):
        self.method = method
        self.img_size = Config.IMG_SIZE

    def extract_hog_features(self, image):
        """提取HOG特征"""
        try:
            from skimage.feature import hog
            from skimage import exposure

            # 转换为灰度图
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2).astype(np.uint8)
            else:
                gray = image

            # 计算HOG特征
            features = hog(gray,
                          orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          visualize=False,
                          feature_vector=True,
                          block_norm='L2-Hys')

            return features
        except Exception as e:
            print(f"⚠️  HOG特征提取失败: {e}")
            return np.zeros(1764)  # 默认维度

    def extract_lbp_features(self, image):
        """提取LBP特征"""
        try:
            from skimage.feature import local_binary_pattern
            from skimage import color

            # 转换为灰度图
            if len(image.shape) == 3:
                gray = color.rgb2gray(image)
            else:
                gray = image

            # 计算LBP特征
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

            # 计算直方图
            hist, _ = np.histogram(lbp.ravel(),
                                  bins=np.arange(0, n_points + 3),
                                  range=(0, n_points + 2))

            # 归一化
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)

            return hist
        except Exception as e:
            print(f"⚠️  LBP特征提取失败: {e}")
            return np.zeros(59)  # 默认维度

    def extract_color_histogram(self, image):
        """提取颜色直方图特征"""
        try:
            # 将图像转换为HSV色彩空间
            if len(image.shape) == 2:
                # 如果是灰度图，直接计算直方图
                hist = np.histogram(image, bins=32, range=(0, 256))[0]
            else:
                # RGB图像，对每个通道计算直方图
                hist_r = np.histogram(image[:, :, 0], bins=32, range=(0, 256))[0]
                hist_g = np.histogram(image[:, :, 1], bins=32, range=(0, 256))[0]
                hist_b = np.histogram(image[:, :, 2], bins=32, range=(0, 256))[0]
                hist = np.concatenate([hist_r, hist_g, hist_b])

            # 归一化
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)

            return hist
        except Exception as e:
            print(f"⚠️  颜色直方图提取失败: {e}")
            if len(image.shape) == 2:
                return np.zeros(32)
            else:
                return np.zeros(96)

    def extract_combined_features(self, image):
        """提取组合特征"""
        hog_features = self.extract_hog_features(image)
        lbp_features = self.extract_lbp_features(image)
        color_features = self.extract_color_histogram(image)

        # 组合所有特征
        combined = np.concatenate([hog_features, lbp_features, color_features])
        return combined

    def extract_features(self, image):
        """根据配置提取特征"""
        if self.method == 'hog':
            return self.extract_hog_features(image)
        elif self.method == 'lbp':
            return self.extract_lbp_features(image)
        elif self.method == 'color_hist':
            return self.extract_color_histogram(image)
        elif self.method == 'combined':
            return self.extract_combined_features(image)
        else:
            return self.extract_hog_features(image)  # 默认使用HOG

# ==================== 图像加载器 ====================
class ImageLoader:
    """图像加载器"""

    def __init__(self, img_size=96):
        self.img_size = img_size
        self.feature_extractor = FeatureExtractor(Config.FEATURE_EXTRACTOR)

    def load_image(self, image_path):
        """加载图像并提取特征"""
        try:
            # 加载图像
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))

            # 转换为numpy数组
            img_array = np.array(img, dtype=np.uint8)

            # 提取特征
            features = self.feature_extractor.extract_features(img_array)

            return features
        except Exception as e:
            print(f"⚠️  无法加载图像 {os.path.basename(image_path)}: {e}")
            # 返回零特征
            test_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            return self.feature_extractor.extract_features(test_img)

    def load_image_for_recognition(self, image_path):
        """为识别加载图像"""
        try:
            # 加载图像
            img = Image.open(image_path).convert('RGB')
            original_size = img.size
            img_resized = img.resize((self.img_size, self.img_size))

            # 转换为numpy数组
            img_array = np.array(img_resized, dtype=np.uint8)

            # 提取特征
            features = self.feature_extractor.extract_features(img_array)

            return features, original_size, os.path.basename(image_path)
        except Exception as e:
            print(f"⚠️  无法加载图像 {os.path.basename(image_path)}: {e}")
            return None, None, None

# ==================== 数据集 ====================
class StreamingDataset:
    """数据集"""

    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # 创建图像加载器
        self.loader = ImageLoader(Config.IMG_SIZE)

        self._scan_data()

    def _scan_data(self):
        """扫描数据"""
        data_dir = os.path.join(self.root_dir, self.mode)

        if not os.path.exists(data_dir):
            print(f"❌ 目录不存在: {data_dir}")
            return

        # 获取所有类别
        all_classes = sorted([d for d in os.listdir(data_dir)
                            if os.path.isdir(os.path.join(data_dir, d))])

        # 限制类别数量
        if Config.TRAIN_CLASS_COUNT > 0:
            all_classes = all_classes[:Config.TRAIN_CLASS_COUNT]

        print(f"\n📂 扫描{self.mode}数据...")

        total_files = 0

        for class_idx, class_name in enumerate(all_classes):
            class_path = os.path.join(data_dir, class_name)

            # 添加到映射
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name
            self.class_names.append(class_name)

            # 获取图像文件
            try:
                image_files = [f for f in os.listdir(class_path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

                # 添加路径和标签
                for img_file in image_files:
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(class_idx)

                print(f"  {class_name}: {len(image_files)} 张图片")
                total_files += len(image_files)

            except Exception as e:
                print(f"⚠️  扫描类别 {class_name} 时出错: {e}")
                continue

        print(f"✅ {self.mode}集总计: {total_files} 张图片, {len(self.class_names)} 个类别")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像和提取特征
        features = self.loader.load_image(self.image_paths[idx])
        return features, self.labels[idx], self.image_paths[idx]

    def get_class_samples(self, class_idx):
        """获取指定类别的样本索引"""
        class_indices = []
        for i in range(len(self.labels)):
            if self.labels[i] == class_idx:
                class_indices.append(i)
        return class_indices

    def get_class_features(self, class_idx, max_samples=0):
        """获取指定类别的所有特征"""
        class_indices = self.get_class_samples(class_idx)

        if max_samples > 0 and len(class_indices) > max_samples:
            import random
            random.shuffle(class_indices)
            class_indices = class_indices[:max_samples]

        features_list = []
        for idx in class_indices:
            feature, _, _ = self[idx]
            features_list.append(feature)

        if features_list:
            return np.vstack(features_list)
        else:
            return np.array([])

# ==================== 目录创建 ====================
def create_dirs():
    """创建所有需要的目录"""
    # 创建主结果目录
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    print(f"📁 结果目录: {Config.RESULTS_PATH}")

    # 创建模型目录
    models_dir = os.path.join(Config.RESULTS_PATH, "models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"📁 模型目录: {models_dir}")

    # 创建日志目录
    logs_dir = os.path.join(Config.RESULTS_PATH, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"📁 日志目录: {logs_dir}")

    # 创建识别结果目录
    recognition_dir = os.path.join(Config.RECOGNITION_RESULTS_PATH, "txt_results")
    os.makedirs(recognition_dir, exist_ok=True)
    print(f"📁 识别结果目录: {recognition_dir}")

    return models_dir, logs_dir, recognition_dir

# 创建所有目录
models_dir, logs_dir, recognition_dir = create_dirs()

# ==================== One-Class SVM 训练器 ====================
class OneClassSVMTrainer:
    """One-Class SVM 训练器"""

    def __init__(self, pca_components=None):
        self.pca_components = pca_components or Config.PCA_COMPONENTS
        self.pca = None
        self.scaler = StandardScaler()
        self.models = {}
        self.class_names = []
        self.feature_dim = None

    def train_class(self, class_name, features):
        """训练单个类别的One-Class SVM"""
        print(f"  训练 {class_name}，样本数: {features.shape[0]}")

        if features.shape[0] < 10:
            print(f"  ⚠️  样本太少 ({features.shape[0]})，跳过")
            return None

        # 标准化特征
        features_scaled = self.scaler.transform(features)

        # 应用PCA降维
        if self.pca:
            features_pca = self.pca.transform(features_scaled)
        else:
            features_pca = features_scaled

        # 训练One-Class SVM
        oc_svm = OneClassSVM(
            nu=Config.NU,
            kernel=Config.KERNEL,
            gamma=Config.GAMMA,
            verbose=False
        )

        try:
            oc_svm.fit(features_pca)
            print(f"  ✅ {class_name} 训练完成")
            return oc_svm
        except Exception as e:
            print(f"  ❌ {class_name} 训练失败: {e}")
            return None

    def train_all_classes(self, dataset):
        """训练所有类别"""
        print(f"\n🔄 开始训练所有类别，共 {len(dataset.class_names)} 个类别")

        # 收集所有特征用于PCA拟合
        all_features = []
        all_labels = []

        print("🔍 收集所有类别特征...")
        for class_idx, class_name in enumerate(dataset.class_names):
            features = dataset.get_class_features(class_idx)
            if features.size > 0:
                all_features.append(features)
                all_labels.extend([class_idx] * features.shape[0])

        if not all_features:
            print("❌ 没有收集到任何特征")
            return False

        all_features = np.vstack(all_features)
        self.feature_dim = all_features.shape[1]
        print(f"✅ 收集到 {all_features.shape[0]} 个样本，特征维度: {self.feature_dim}")

        # 拟合标准化器
        print("🔧 拟合标准化器...")
        self.scaler.fit(all_features)

        # 应用标准化
        all_features_scaled = self.scaler.transform(all_features)

        # 拟合PCA（如果需要）
        if self.pca_components and self.pca_components < self.feature_dim:
            print(f"🔧 拟合PCA，目标维度: {self.pca_components}")
            self.pca = PCA(n_components=self.pca_components)
            self.pca.fit(all_features_scaled)
            print(f"  PCA解释方差比: {self.pca.explained_variance_ratio_.sum():.3f}")

        # 训练每个类别
        print(f"\n🚀 开始训练每个类别的One-Class SVM...")

        for class_idx, class_name in enumerate(dataset.class_names):
            print(f"\n{'='*40}")
            print(f"训练类别 {class_idx+1}/{len(dataset.class_names)}: {class_name}")
            print(f"{'='*40}")

            # 获取该类别的特征
            features = dataset.get_class_features(class_idx)

            if features.size == 0:
                print(f"  ⚠️  没有特征，跳过")
                continue

            # 训练模型
            oc_svm = self.train_class(class_name, features)

            if oc_svm is not None:
                self.models[class_name] = oc_svm
                self.class_names.append(class_name)

        print(f"\n✅ 训练完成!")
        print(f"   成功训练类别数: {len(self.models)}/{len(dataset.class_names)}")

        return len(self.models) > 0

    def save_models(self, save_dir):
        """保存所有模型"""
        if not self.models:
            print("❌ 没有模型可保存")
            return False

        # 创建保存数据
        save_data = {
            'models': self.models,
            'scaler': self.scaler,
            'pca': self.pca,
            'class_names': self.class_names,
            'feature_dim': self.feature_dim,
            'config': {
                'img_size': Config.IMG_SIZE,
                'pca_components': self.pca_components,
                'nu': Config.NU,
                'kernel': Config.KERNEL,
                'gamma': Config.GAMMA,
                'feature_extractor': Config.FEATURE_EXTRACTOR
            },
            'saved_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 保存模型
        try:
            model_file = os.path.join(save_dir, "oneclass_svm_models.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(save_data, f)

            print(f"✅ 模型保存成功: {model_file}")
            print(f"   大小: {os.path.getsize(model_file)/1024:.1f} KB")
            return True
        except Exception as e:
            print(f"❌ 模型保存失败: {e}")
            return False

    def load_models(self, model_file):
        """加载模型"""
        try:
            with open(model_file, 'rb') as f:
                save_data = pickle.load(f)

            self.models = save_data['models']
            self.scaler = save_data['scaler']
            self.pca = save_data.get('pca')
            self.class_names = save_data['class_names']
            self.feature_dim = save_data['feature_dim']

            print(f"✅ 模型加载成功: {model_file}")
            print(f"   类别数: {len(self.models)}")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

# ==================== 图像识别器 ====================
class OneClassSVMRecognizer:
    """One-Class SVM 图像识别器"""

    def __init__(self, trainer=None, model_file=None):
        self.loader = ImageLoader(Config.IMG_SIZE)

        if trainer is not None:
            self.trainer = trainer
        elif model_file is not None and os.path.exists(model_file):
            self.trainer = OneClassSVMTrainer()
            if not self.trainer.load_models(model_file):
                raise ValueError("无法加载模型文件")
        else:
            raise ValueError("需要提供trainer或model_file")

    def extract_features(self, image_path):
        """从图像提取特征"""
        result = self.loader.load_image_for_recognition(image_path)
        if result[0] is None:
            return None, None, None

        features, original_size, filename = result
        return features, original_size, filename

    def predict_image(self, image_path):
        """预测单个图像"""
        # 提取特征
        features, original_size, filename = self.extract_features(image_path)
        if features is None:
            return None

        # 预处理特征
        features = features.reshape(1, -1)
        features_scaled = self.trainer.scaler.transform(features)

        if self.trainer.pca:
            features_processed = self.trainer.pca.transform(features_scaled)
        else:
            features_processed = features_scaled

        # 计算每个模型的决策函数值
        decision_scores = {}
        for class_name, model in self.trainer.models.items():
            try:
                # 决策函数值（越大表示越正常）
                score = model.decision_function(features_processed)[0]
                decision_scores[class_name] = score
            except Exception as e:
                print(f"⚠️  预测类别 {class_name} 失败: {e}")
                decision_scores[class_name] = -float('inf')

        # 转换为置信度（使用softmax）
        scores = np.array(list(decision_scores.values()))

        # 处理异常值
        if np.all(scores == -float('inf')):
            confidences = np.ones(len(scores)) / len(scores)
        else:
            # 将所有分数转换为正数
            scores_shifted = scores - np.min(scores) + 1e-10
            exp_scores = np.exp(scores_shifted)
            confidences = exp_scores / np.sum(exp_scores)

        # 排序结果
        class_names = list(decision_scores.keys())
        sorted_indices = np.argsort(-confidences)  # 降序排序

        # 准备结果
        results = []
        for idx in sorted_indices[:Config.TOP_K_RESULTS]:
            results.append({
                'class': class_names[idx],
                'decision_score': float(decision_scores[class_names[idx]]),
                'confidence': float(confidences[idx])
            })

        return {
            'filename': filename,
            'original_size': original_size,
            'top_results': results,
            'predicted_class': results[0]['class'] if results else 'Unknown',
            'confidence': results[0]['confidence'] if results else 0.0,
            'decision_scores': decision_scores
        }

    def predict_batch(self, image_paths, batch_size=32):
        """批量预测"""
        results = []
        total_images = len(image_paths)

        print(f"🔍 开始批量识别 {total_images} 张图像...")

        for i in range(0, total_images, batch_size):
            batch_paths = image_paths[i:i+batch_size]

            for img_path in batch_paths:
                result = self.predict_image(img_path)
                if result:
                    results.append(result)

            # 显示进度
            processed = min(i + batch_size, total_images)
            print(f"    已处理 {processed}/{total_images} 张图像")

        return results

    def predict_directory(self, directory_path):
        """预测目录中的所有图像"""
        # 支持的图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(directory_path, ext)))
            image_paths.extend(glob.glob(os.path.join(directory_path, ext.upper())))

        print(f"📂 在目录中找到 {len(image_paths)} 张图像: {directory_path}")

        if not image_paths:
            print("⚠️  目录中没有找到图像文件")
            return []

        return self.predict_batch(image_paths)

# ==================== 结果保存函数 ====================
def save_recognition_results(results, output_dir, method="One-Class SVM"):
    """保存识别结果到txt文件"""
    if not results:
        print("⚠️  没有识别结果可保存")
        return None, None

    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存详细结果
    detailed_file = os.path.join(output_dir, f"detailed_results_{timestamp}.txt")

    with open(detailed_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"📊 {method} 图像识别结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总图像数: {len(results)}\n")
        f.write(f"特征提取方法: {Config.FEATURE_EXTRACTOR}\n")
        f.write(f"One-Class SVM参数: nu={Config.NU}, kernel={Config.KERNEL}, gamma={Config.GAMMA}\n")
        f.write(f"PCA维度: {Config.PCA_COMPONENTS}\n\n")

        # 统计信息
        class_counts = {}
        high_confidence_count = 0
        avg_confidence = 0

        for i, result in enumerate(results, 1):
            predicted_class = result['predicted_class']
            confidence = result['confidence']

            # 更新类别计数
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            avg_confidence += confidence

            # 更新高置信度计数
            if confidence >= Config.CONFIDENCE_THRESHOLD:
                high_confidence_count += 1

            # 写入详细结果
            f.write(f"【图像 {i}】\n")
            f.write(f"  文件名: {result['filename']}\n")
            f.write(f"  原始尺寸: {result['original_size'][0]}x{result['original_size'][1]}\n")
            f.write(f"  预测类别: {predicted_class}\n")
            f.write(f"  置信度: {confidence:.4f}\n")

            if confidence >= Config.CONFIDENCE_THRESHOLD:
                f.write(f"  状态: ✅ 高置信度\n")
            else:
                f.write(f"  状态: ⚠️  低置信度\n")

            f.write(f"  前{Config.TOP_K_RESULTS}个结果:\n")
            for j, top_result in enumerate(result['top_results'], 1):
                f.write(f"    {j}. {top_result['class']} - 决策分数: {top_result['decision_score']:.4f}, 置信度: {top_result['confidence']:.4f}\n")

            f.write("-" * 60 + "\n\n")

        # 计算平均置信度
        if results:
            avg_confidence /= len(results)

        # 写入统计信息
        f.write("=" * 80 + "\n")
        f.write("📈 统计信息\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"高置信度图像数: {high_confidence_count}/{len(results)} ({(high_confidence_count/len(results)*100):.1f}%)\n")
        f.write(f"低置信度图像数: {len(results)-high_confidence_count}/{len(results)} ({((len(results)-high_confidence_count)/len(results)*100):.1f}%)\n")
        f.write(f"平均置信度: {avg_confidence:.4f}\n\n")

        f.write("类别分布:\n")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results)) * 100
            f.write(f"  {class_name}: {count} 张 ({percentage:.1f}%)\n")

    print(f"✅ 详细结果保存到: {detailed_file}")

    # 保存简要结果
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"📋 {method} 图像识别摘要\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("图像识别结果:\n")
        f.write("-" * 80 + "\n")
        f.write("序号\t文件名\t\t预测类别\t置信度\t状态\n")
        f.write("-" * 80 + "\n")

        for i, result in enumerate(results, 1):
            filename = result['filename']
            predicted_class = result['predicted_class']
            confidence = result['confidence']

            # 缩短文件名显示
            if len(filename) > 20:
                display_name = filename[:17] + "..."
            else:
                display_name = filename.ljust(20)

            # 状态标记
            if confidence >= Config.CONFIDENCE_THRESHOLD:
                status = "✅"
            else:
                status = "⚠️"

            f.write(f"{i:3d}\t{display_name}\t{predicted_class[:15]:15s}\t{confidence:.4f}\t{status}\n")

    print(f"✅ 简要结果保存到: {summary_file}")

    return detailed_file, summary_file

# ==================== 评估函数 ====================
def evaluate_model(trainer, test_dataset, max_eval=500):
    """评估One-Class SVM模型"""
    print(f"🧪 开始评估...")
    print(f"   将评估 {max_eval} 个样本")

    correct = 0
    total = 0

    # 限制评估数量
    max_eval = min(max_eval, len(test_dataset.image_paths))

    for i in range(max_eval):
        # 加载测试样本
        features, true_label, _ = test_dataset[i]
        true_class = test_dataset.class_names[true_label]

        # 预测
        features = features.reshape(1, -1)
        features_scaled = trainer.scaler.transform(features)

        if trainer.pca:
            features_processed = trainer.pca.transform(features_scaled)
        else:
            features_processed = features_scaled

        # 找到决策函数值最大的类别
        best_class = None
        best_score = -float('inf')

        for class_name, model in trainer.models.items():
            try:
                score = model.decision_function(features_processed)[0]
                if score > best_score:
                    best_score = score
                    best_class = class_name
            except:
                continue

        if best_class is not None:
            total += 1
            if best_class == true_class:
                correct += 1

        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"    已处理 {i+1}/{max_eval} 个样本")

    test_accuracy = correct / total if total > 0 else 0
    print(f"\n📊 评估结果:")
    print(f"   总样本数: {total}")
    print(f"   正确预测: {correct}")
    print(f"   测试准确率: {test_accuracy:.4f} ({correct}/{total})")

    return test_accuracy

# ==================== 训练函数 ====================
def train_one_class_svm():
    """训练One-Class SVM模型"""
    print("\n" + "=" * 60)
    print("🔧 One-Class SVM 训练模式")
    print("=" * 60)

    # 检查数据集
    if not os.path.exists(Config.DATASET_PATH):
        print(f"❌ 数据集不存在: {Config.DATASET_PATH}")
        return None

    # 扫描训练数据
    print("\n📊 扫描训练数据集...")
    train_dataset = StreamingDataset(Config.DATASET_PATH, 'train')

    if len(train_dataset.image_paths) == 0:
        print("❌ 训练数据为空")
        return None

    print(f"\n📊 训练集统计:")
    print(f"   类别数: {len(train_dataset.class_names)}")
    print(f"   总样本数: {len(train_dataset.image_paths)}")
    print(f"   特征提取方法: {Config.FEATURE_EXTRACTOR}")
    print(f"   One-Class SVM参数: nu={Config.NU}, kernel={Config.KERNEL}")

    # 创建训练器
    trainer = OneClassSVMTrainer()

    # 训练模型
    start_time = time.time()
    success = trainer.train_all_classes(train_dataset)
    training_time = time.time() - start_time

    if not success:
        print("❌ 训练失败")
        return None

    print(f"✅ 训练完成! 用时: {training_time:.1f}秒")

    # 保存模型
    print("\n💾 保存模型...")
    save_success = trainer.save_models(models_dir)

    if not save_success:
        print("⚠️  模型保存失败")

    # 扫描测试数据
    print("\n📂 扫描测试数据...")
    test_dataset = StreamingDataset(Config.DATASET_PATH, 'test')

    if len(test_dataset.image_paths) == 0:
        print("⚠️  测试数据为空，跳过评估")
        test_accuracy = 0
    else:
        # 评估模型
        test_accuracy = evaluate_model(trainer, test_dataset, max_eval=500)

    # 保存训练结果
    results = {
        'training_stats': {
            'total_classes': len(train_dataset.class_names),
            'trained_classes': len(trainer.models),
            'train_samples': len(train_dataset.image_paths),
            'test_samples': len(test_dataset.image_paths),
            'training_time': training_time,
            'test_accuracy': float(test_accuracy)
        },
        'config': {
            'img_size': Config.IMG_SIZE,
            'feature_extractor': Config.FEATURE_EXTRACTOR,
            'pca_components': Config.PCA_COMPONENTS,
            'nu': Config.NU,
            'kernel': Config.KERNEL,
            'gamma': Config.GAMMA,
            'train_class_count': Config.TRAIN_CLASS_COUNT
        },
        'training_completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    results_file = os.path.join(Config.RESULTS_PATH, "training_results.json")
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 训练结果保存到: {results_file}")
    except Exception as e:
        print(f"⚠️  保存训练结果失败: {e}")

    # 生成报告
    print("\n" + "=" * 60)
    print("📊 One-Class SVM 训练报告")
    print("=" * 60)
    print(f"训练完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集路径: {Config.DATASET_PATH}")
    print(f"结果目录: {Config.RESULTS_PATH}")
    print(f"特征提取方法: {Config.FEATURE_EXTRACTOR}")
    print(f"One-Class SVM参数: nu={Config.NU}, kernel={Config.KERNEL}")
    print(f"成功训练类别: {len(trainer.models)}/{len(train_dataset.class_names)}")
    print(f"训练样本总数: {len(train_dataset.image_paths)}")
    print(f"测试样本总数: {len(test_dataset.image_paths)}")
    print(f"测试准确率: {test_accuracy:.4f}")
    print(f"训练用时: {training_time:.1f}秒")

    print("\n📁 生成的文件:")
    print(f"  📄 training_results.json - 完整训练结果")
    print(f"  🤖 models/oneclass_svm_models.pkl - 训练好的模型")

    return trainer

# ==================== 识别函数 ====================
def recognize_images():
    """使用One-Class SVM识别图像"""
    print("\n" + "=" * 60)
    print("🔍 One-Class SVM 图像识别模式")
    print("=" * 60)

    # 查找模型文件
    model_file = os.path.join(models_dir, "oneclass_svm_models.pkl")
    if not os.path.exists(model_file):
        print(f"❌ 没有找到模型文件，请先训练模型")
        return

    print(f"📂 加载模型: {model_file}")

    try:
        # 创建识别器
        recognizer = OneClassSVMRecognizer(model_file=model_file)
        print(f"✅ 模型加载成功，共 {len(recognizer.trainer.models)} 个类别")
    except Exception as e:
        print(f"❌ 创建识别器失败: {e}")
        return

    # 检查识别图像路径
    if not os.path.exists(Config.RECOGNITION_IMAGE_PATH):
        print(f"❌ 识别图像路径不存在: {Config.RECOGNITION_IMAGE_PATH}")
        print("请修改 Config.RECOGNITION_IMAGE_PATH 为有效的图像路径")
        return

    # 识别图像
    if os.path.isfile(Config.RECOGNITION_IMAGE_PATH):
        # 单个图像
        print(f"🔍 识别单个图像: {Config.RECOGNITION_IMAGE_PATH}")
        result = recognizer.predict_image(Config.RECOGNITION_IMAGE_PATH)

        if result:
            # 显示结果
            print(f"\n📊 识别结果:")
            print(f"  文件名: {result['filename']}")
            print(f"  原始尺寸: {result['original_size'][0]}x{result['original_size'][1]}")
            print(f"  预测类别: {result['predicted_class']}")
            print(f"  置信度: {result['confidence']:.4f}")

            if result['confidence'] >= Config.CONFIDENCE_THRESHOLD:
                print(f"  状态: ✅ 高置信度")
            else:
                print(f"  状态: ⚠️  低置信度")

            print(f"\n  前{Config.TOP_K_RESULTS}个结果:")
            for i, top_result in enumerate(result['top_results'], 1):
                print(f"    {i}. {top_result['class']} - 决策分数: {top_result['decision_score']:.4f}, 置信度: {top_result['confidence']:.4f}")

            # 保存结果
            save_recognition_results([result], recognition_dir, "One-Class SVM")
    else:
        # 目录中的多个图像
        print(f"🔍 识别目录中的图像: {Config.RECOGNITION_IMAGE_PATH}")
        results = recognizer.predict_directory(Config.RECOGNITION_IMAGE_PATH)

        if results:
            print(f"\n✅ 完成识别 {len(results)} 张图像")

            # 保存结果
            save_recognition_results(results, recognition_dir, "One-Class SVM")

# ==================== 主函数 ====================
def main():
    print("\n" + "=" * 60)
    print("🚀 One-Class SVM - 图像识别系统")
    print("=" * 60)

    # 显示菜单
    print("\n请选择操作模式:")
    print("1. 训练One-Class SVM模型")
    print("2. 使用One-Class SVM进行图像识别")
    print("3. 训练并识别")

    choice = input("\n请输入选择 (1/2/3): ").strip()

    if choice == "1":
        train_one_class_svm()
    elif choice == "2":
        recognize_images()
    elif choice == "3":
        print("\n" + "=" * 60)
        print("🔧 第一步：训练One-Class SVM模型")
        print("=" * 60)
        trainer = train_one_class_svm()
        if trainer is not None:
            print("\n" + "=" * 60)
            print("🔍 第二步：图像识别")
            print("=" * 60)
            recognize_images()
    else:
        print("❌ 无效选择，使用默认模式：训练并识别")
        trainer = train_one_class_svm()
        if trainer is not None:
            recognize_images()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 操作被用户中断")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()