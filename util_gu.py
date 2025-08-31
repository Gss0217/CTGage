import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter
from scipy.ndimage import zoom
from scipy.stats import gaussian_kde
import random
import os

class init_data(Dataset):
    def __init__(self, data):
        self.data=data
    def __getitem__(self, index):
        return (torch.tensor(self.data[index,1:]),torch.tensor(self.data[index,0])) # data label
    def __len__(self):
        return len(self.data)

def load_npy_data(file_path):
    """
    从指定的 .npy 文件中加载数据，并提取孕龄和胎心数据。
    
    参数:
        file_path (str): .npy 文件路径。
    
    返回:
        tuple: (gestational_ages, array(fhr_data))
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")

    # 加载 .npy 文件
    data = np.load(file_path, allow_pickle=True)

    # 假设 data 是一个包含字典的 NumPy 数组
    if isinstance(data, np.ndarray) and len(data) > 0 and isinstance(data[0], dict):
        # 提取所有样本的数据
        gestational_ages = []
        fhr_data = []
        for sample in data:
            gestational_ages.append(sample['gestational_age'])
            fhr_data.append(sample['fhr_data'])
        return np.array(gestational_ages, dtype=object), np.array(fhr_data, dtype=object)
    else:
        raise ValueError(f"文件 {file_path} 中的数据格式不正确")

def extract_labels_and_data(file_path, have_id=True):
    """
    从包含多个元组的列表中提取标签和数据。
    
    参数:
        data_list (list of tuple): 每个元组包含 (id, 标签, 数据)。
    
    返回:
        tuple: (labels, data)
            labels (numpy.ndarray): 包含所有标签的数组。
            data (numpy.ndarray): 包含所有数据的数组。
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")

    # 加载 .npy 文件
    data_list = np.load(file_path, allow_pickle=True)

    if have_id == True:
        # 初始化标签和数据列表
        ids = []
        labels = []
        data = []

        # 遍历数据列表，提取标签和数据
        for item in data_list:
            id_, label, fhr_data = item
            ids.append(id_)
            labels.append(label)
            data.append(fhr_data)

        # 将标签和数据转换为 NumPy 数组
        ids = np.array(ids,dtype=object)
        labels = np.array(labels, dtype=object)
        data = np.array(data, dtype=object)

        return ids, labels, data
    else:
        # 初始化标签和数据列表
        labels = []
        data = []

        # 遍历数据列表，提取标签和数据
        for item in data_list:
            label, fhr_data = item
            labels.append(label)
            data.append(fhr_data)

        # 将标签和数据转换为 NumPy 数组
        labels = np.array(labels, dtype=object)
        data = np.array(data, dtype=object)

        return labels, data

def sliding_window_data_augmentation(ids, gestational_ages, fhr_data, window_size, step_size):
    augmented_id = []
    augmented_ga = []
    augmented_fhr = []

    enlarge_factor = 2

    for id_, ga, fhr in zip(ids, gestational_ages, fhr_data):
        # 1. 原始滑动窗口
        for start in range(0, len(fhr) - window_size + 1, int(step_size)):
            augmented_id.append(id_)
            augmented_ga.append(ga)
            augmented_fhr.append(fhr[start:start + window_size])

        # 2. 只对小孕龄做「密集窗口 + 时间扭曲 + 噪声」
        if ga < 220:
            # 2-a 更细步长
            for start in range(0, len(fhr) - window_size + 1, max(1, step_size)):
                # 2-b 时间扭曲 3 次
                for _ in range(enlarge_factor):
                    warp = np.random.uniform(0.8, 1.2)
                    stretched = zoom(fhr, warp, order=1)
                    # 2-c 加高斯噪声
                    noise = np.random.normal(0, 0.01, stretched.shape)
                    noisy = stretched + noise
                    if len(noisy) >= window_size:
                        for s in range(0, len(noisy) - window_size + 1, max(1, step_size)):
                            augmented_id.append(id_)
                            augmented_ga.append(ga)
                            augmented_fhr.append(noisy[s:s + window_size])

    return np.array(augmented_id), np.array(augmented_ga), np.array(augmented_fhr)

def filter_data_by_label(ids, labels, data, min_label=0, max_label=294):
    """
    筛选标签在指定范围内的数据。
    
    参数:
        labels (numpy.ndarray): 包含标签的数组。
        data (numpy.ndarray): 包含数据的数组。
        min_label (int): 最小标签值，默认为 0。
        max_label (int): 最大标签值，默认为 294。
    
    返回:
        tuple: (filtered_labels, filtered_data)
            filtered_labels (numpy.ndarray): 筛选后的标签数组。
            filtered_data (numpy.ndarray): 筛选后的数据数组。
    """
    labels = np.array(labels, dtype=float)

    # 检查 labels 中是否存在无效值
    if not np.all(np.isfinite(labels)):
        print("警告：labels 中存在无效值，将无效值替换为 0")
        # 将无效值替换为 0
        labels = np.where(np.isfinite(labels), labels, 0)

    # 筛选条件：标签大于最小值且小于或等于最大值
    mask = (labels > min_label) & (labels <= max_label)
    
    # 应用筛选条件
    filtered_labels = labels[mask]
    filtered_data = data[mask]
    filtered_id = ids[mask]
    
    return filtered_id, filtered_labels, filtered_data

def filter_test_data_by_label(labels, data, min_label=0, max_label=294):
    """
    筛选标签在指定范围内的数据。
    
    参数:
        labels (numpy.ndarray): 包含标签的数组。
        data (numpy.ndarray): 包含数据的数组。
        min_label (int): 最小标签值，默认为 0。
        max_label (int): 最大标签值，默认为 294。
    
    返回:
        tuple: (filtered_labels, filtered_data)
            filtered_labels (numpy.ndarray): 筛选后的标签数组。
            filtered_data (numpy.ndarray): 筛选后的数据数组。
    """
    labels = np.array(labels, dtype=float)

    # 检查 labels 中是否存在无效值
    if not np.all(np.isfinite(labels)):
        print("警告：labels 中存在无效值，将无效值替换为 0")
        # 将无效值替换为 0
        labels = np.where(np.isfinite(labels), labels, 0)

    # 筛选条件：标签大于最小值且小于或等于最大值
    mask = (labels > min_label) & (labels <= max_label)
    
    # 应用筛选条件
    filtered_labels = labels[mask]
    filtered_data = data[mask]
    
    return filtered_labels, filtered_data

def filter_test_data_by_length(gestational_ages, fhr_data, min_length=4000, max_length=4000):
    """
    筛选数据，确保数据长度不超过指定的最大长度。
    
    参数:
    - gestational_ages: 孕周数据，形状为 (n_samples,)
    - fhr_data: FHR 数据，形状为 (n_samples, sequence_length)
    - max_length: 最大数据长度，默认为 4000
    
    返回:
    - filtered_gestational_ages: 筛选后的孕周数据
    - filtered_fhr_data: 筛选后的 FHR 数据
    """
    filtered_gestational_ages = []
    filtered_fhr_data = []
    
    for ga, fhr in zip(gestational_ages, fhr_data):
        # 去除长程
        if len(fhr) <= max_length:
            filtered_gestational_ages.append(ga)
            filtered_fhr_data.append(fhr)
        # 去除短程
        # if len(fhr) >= min_length:
        #     filtered_gestational_ages.append(ga)
        #     filtered_fhr_data.append(fhr)
    
    return np.array(filtered_gestational_ages), np.array(filtered_fhr_data)

def random_resample_all(ids, data_list, labels, target_length):
    """
    随机重采样多组数据及其对应的标签，使其达到指定的固定长度。
    如果目标长度大于原始长度，则抛弃该数据及其对应的标签。
    
    参数:
    data_list (list of lists or np.ndarray): 包含多组数据的列表，每组数据可以是列表或 NumPy 数组。
    labels (list): 对应的标签列表。
    target_length (int): 目标长度。
    
    返回:
    tuple: (resampled_data_list, resampled_labels, discarded_indices)
        - resampled_data_list: 重采样后的数据列表。
        - resampled_labels: 重采样后的标签列表。
        - discarded_indices: 被抛弃的数据索引列表。
    """
    resampled_data_list = []
    resampled_labels = []
    resampled_id = []
    discarded_indices = []

    for i, (id_, data, label) in enumerate(zip(ids, data_list, labels)):
        # 确保输入数据是 NumPy 数组
        data = np.array(data)
        
        # 获取原始数据的长度
        original_length = len(data)
        
        # 如果目标长度大于原始长度，抛弃该数据及其对应的标签
        if target_length > original_length:
            discarded_indices.append(i)
            continue
        
        # 如果目标长度小于或等于原始长度，进行随机裁剪
        start_index = np.random.randint(0, original_length - target_length + 1)
        resampled_data = data[start_index:start_index + target_length]
        
        resampled_data_list.append(resampled_data)
        resampled_labels.append(label)
        resampled_id.append(id_)
    
    return resampled_id, resampled_data_list, resampled_labels

def random_sample_by_label(data, labels, label_threshold):
    """
    根据标签随机筛选数据
    :param data: 数据列表
    :param labels: 标签列表，与数据一一对应
    :param label_threshold: 标签的阈值
    :return: 筛选后的数据列表和对应的标签列表
    """
    # 创建一个字典来存储每个标签的数据
    label_data = {}
    
    # 将数据按标签分组
    for item, label in zip(data, labels):
        if label not in label_data:
            label_data[label] = []
        label_data[label].append(item)
    
    # 创建两个列表来存储筛选后的数据和标签
    sampled_data = []
    sampled_labels = []
    
    # 设置阈值的浮动范围（例如 ±10%）
    threshold_range = int(label_threshold * 0.5)  # 50% 的浮动范围
    min_threshold = max(1, label_threshold - threshold_range)  # 阈值下限
    max_threshold = max(min_threshold, label_threshold - threshold_range * 0.5)  # 阈值上限
    sample_count = random.randint(min_threshold, max_threshold)
    
    # 遍历每个标签的数据
    for label, items in label_data.items():
        # 如果该标签的数据量超过阈值，则随机抽取阈值范围内的数据量
        if len(items) > sample_count:
            # 随机选择一个在阈值范围内的数量
            sampled_items = random.sample(items, sample_count)
            sampled_data.extend(sampled_items)
            sampled_labels.extend([label] * sample_count)
        else:
            # 如果数据量不超过阈值，则保留所有数据和对应的标签
            sampled_data.extend(items)
            sampled_labels.extend([label] * len(items))
    
    return sampled_data, sampled_labels

def print_labels_distribution(labels):
    # 计算基本统计信息
    min_label = np.min(labels)
    max_label = np.max(labels)
    mean_label = np.mean(labels)
    std_label = np.std(labels)
    
    # 打印基本统计信息
    print(f"最小值: {min_label}")
    print(f"最大值: {max_label}")
    print(f"均值: {mean_label}")
    print(f"标准差: {std_label}")
    
    # 计算不同值的分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # 打印不同值的分布
    print("不同值的分布：")
    for label, count in zip(unique_labels, counts):
        print(f"标签 {label}: {count} 次")

def sample_balanced_targets(batch_size, label_range):
    """
    从 label_range 中均匀采样 batch_size 个孕龄值
    label_range: 形如 np.arange(100, 295, 1.0)
    """
    p = np.ones_like(label_range, dtype=np.float32)
    p /= p.sum()
    sampled = np.random.choice(label_range, size=batch_size, p=p)
    return torch.tensor(sampled, dtype=torch.float32).reshape(-1, 1)

def fhr_to_hrv(fhr, fs=4):
    """
    fhr: shape (T,) 的连续 FHR 信号，单位 bpm
    fs : 采样率，默认 4 Hz
    返回: 这段信号的 6 个时域 HRV 特征
    """
    # 先把 FHR -> RR 间隔（毫秒）
    rr = 60 * 1000 / fhr               # 逐点 RR
    # 去掉异常值（<300 ms 或 >2000 ms）
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 10:                   # 太短的片段用 0 填充
        return np.zeros(6)
    # 时域特征
    mean_rr   = np.mean(rr)
    sdnn      = np.std(rr)
    rmssd     = np.sqrt(np.mean(np.diff(rr) ** 2))
    pnn50     = np.sum(np.abs(np.diff(rr)) > 50) / (len(rr) - 1) * 100
    median_rr = np.median(rr)
    mad_rr    = np.median(np.abs(rr - median_rr))
    return np.array([mean_rr, sdnn, rmssd, pnn50, median_rr, mad_rr], dtype=np.float32)