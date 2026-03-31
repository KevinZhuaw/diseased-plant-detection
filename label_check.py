"""
诊断脚本 - 找出问题所在
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def diagnose_results(y_true, y_pred, y_scores=None):
    """
    诊断模型结果
    """
    print("=" * 60)
    print("诊断分析")
    print("=" * 60)

    # 1. 基本统计
    print("\n📊 基本统计:")
    print(f"  样本总数: {len(y_true)}")
    print(f"  真实标签分布:")
    unique, counts = np.unique(y_true, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    类别 {label}: {count} ({count / len(y_true) * 100:.1f}%)")

    print(f"\n  预测标签分布:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    类别 {label}: {count} ({count / len(y_pred) * 100:.1f}%)")

    # 2. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📈 混淆矩阵:")
    print("    预测")
    print(f"    0   1")
    print(f"真 0 {cm[0, 0]:4d} {cm[0, 1]:4d}")
    print(f"实 1 {cm[1, 0]:4d} {cm[1, 1]:4d}")

    # 3. 问题分析
    print(f"\n🔍 问题分析:")

    # 检查是否所有预测都相同
    if len(np.unique(y_pred)) == 1:
        print(f"  ❌ 严重问题: 所有样本都被预测为同一类别 ({y_pred[0]})")
        print(f"     这解释了为什么准确率如此低")

        # 检查哪个类别被过度预测
        majority_class = y_pred[0]
        majority_count = np.sum(y_true == majority_class)
        minority_count = len(y_true) - majority_count

        print(f"     真实数据中:")
        print(f"       类别 {majority_class}: {majority_count} 个样本")
        print(f"       其他类别: {minority_count} 个样本")

        if majority_count == 0:
            print(f"     ⚠️ 模型预测的类别在真实数据中不存在!")
        else:
            expected_accuracy = majority_count / len(y_true)
            print(f"     即使全部猜{majority_class}，准确率也应为: {expected_accuracy:.4f}")

    # 4. 建议
    print(f"\n💡 建议:")
    print(f"  1. 检查数据集是否平衡")
    print(f"  2. 验证数据加载和预处理是否正确")
    print(f"  3. 检查Deep SVDD的中心计算是否正确")
    print(f"  4. 调整阈值设置")
    print(f"  5. 增加训练轮数或调整学习率")

    return cm


def check_dataset_balance(dataset_path, mode="train", binary=True):
    """
    检查数据集平衡性
    """
    print(f"\n📁 检查数据集: {dataset_path}/{mode}")

    import glob
    from collections import Counter

    data_dir = os.path.join(dataset_path, mode)
    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return

    # 获取所有类别
    all_classes = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]

    if binary:
        # 二分类：找到健康类别
        healthy_class = None
        for cls in all_classes:
            if "healthy" in cls.lower():
                healthy_class = cls
                break

        if healthy_class is None and all_classes:
            healthy_class = all_classes[0]

        # 统计数量
        class_counts = {}
        for cls in all_classes:
            cls_path = os.path.join(data_dir, cls)
            images = glob.glob(os.path.join(cls_path, "*.jpg")) + \
                     glob.glob(os.path.join(cls_path, "*.jpeg")) + \
                     glob.glob(os.path.join(cls_path, "*.png"))

            if cls == healthy_class:
                label = "healthy"
            else:
                label = "diseased"

            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += len(images)

        print(f"📊 二分类数据集统计:")
        total = sum(class_counts.values())
        for label, count in class_counts.items():
            percentage = count / total * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")

            if percentage < 20 or percentage > 80:
                print(f"    ⚠️ 类别不平衡!")

    else:
        # 多分类
        class_counts = {}
        for cls in all_classes:
            cls_path = os.path.join(data_dir, cls)
            images = glob.glob(os.path.join(cls_path, "*.jpg")) + \
                     glob.glob(os.path.join(cls_path, "*.jpeg")) + \
                     glob.glob(os.path.join(cls_path, "*.png"))

            class_counts[cls] = len(images)

        print(f"📊 多分类数据集统计 (共{len(class_counts)}个类别):")
        total = sum(class_counts.values())

        # 按数量排序
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"  前5个最多的类别:")
        for cls, count in sorted_counts[:5]:
            percentage = count / total * 100
            print(f"    {cls}: {count} ({percentage:.1f}%)")

        print(f"\n  前5个最少的类别:")
        for cls, count in sorted_counts[-5:]:
            percentage = count / total * 100
            print(f"    {cls}: {count} ({percentage:.1f}%)")

            if percentage < 1:
                print(f"      ⚠️ 样本过少!")


# 使用示例
if __name__ == "__main__":
    # 假设您有预测结果
    # y_true = [...]  # 真实标签
    # y_pred = [...]  # 预测标签

    # diagnose_results(y_true, y_pred)

    # 检查数据集
    dataset_path = r"E:\庄楷文\叶片病虫害识别\baseline\data\reorganized_dataset_new"

    print("=" * 60)
    print("数据集诊断")
    print("=" * 60)

    for mode in ["train", "test"]:
        print(f"\n检查{mode}集:")
        check_dataset_balance(dataset_path, mode=mode, binary=True)
        check_dataset_balance(dataset_path, mode=mode, binary=False)