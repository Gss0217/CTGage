import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter
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

def extract_labels_and_data(file_path):
    """
    从包含多个元组的列表中提取标签和数据。
    
    参数:
        data_list (list of tuple): 每个元组包含 (标签, 数据)。
    
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

    # 初始化标签和数据列表
    names = []
    labels = []
    data = []

    # 遍历数据列表，提取标签和数据
    for item in data_list:
        name, label, fhr_data = item
        names.append(name)
        labels.append(label)
        data.append(fhr_data)

    # 将标签和数据转换为 NumPy 数组
    names = np.array(names, dtype=object)
    labels = np.array(labels, dtype=object)
    data = np.array(data, dtype=object)

    return names, labels, data

def sliding_window_data_augmentation(gestational_ages, fhr_data, window_size, step_size):
    augmented_ga = []
    augmented_fhr = []
    for ga, fhr in zip(gestational_ages, fhr_data):
        
        if ga < 240:
            stride = step_size / 2
        elif ga > 280:
            stride = step_size / 2
        else:
            stride = step_size

        # stride = step_size

        for start in range(0, len(fhr) - window_size + 1, int(stride)):
            augmented_ga.append(ga)
            augmented_fhr.append(fhr[start:start + window_size])
    return np.array(augmented_ga), np.array(augmented_fhr)

def filter_data_by_label_train(ids, labels, data, min_label=0, max_label=294):
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

def filter_test_data_by_label(names, labels, data, min_label=0, max_label=294):
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
    filtered_names = names[mask]
    filtered_labels = labels[mask]
    filtered_data = data[mask]
    
    return filtered_names, filtered_labels, filtered_data

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

def random_resample_all(names, data_list, labels, target_length):
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
    resampled_names = []
    resampled_data_list = []
    resampled_labels = []
    discarded_indices = []

    for i, (name, data, label) in enumerate(zip(names, data_list, labels)):
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
        
        resampled_names.append(name)
        resampled_data_list.append(resampled_data)
        resampled_labels.append(label)
    
    return resampled_names, resampled_data_list, resampled_labels

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


def extract_labels_and_data_train(file_path, have_id=True):
    """
    从包含多个元组的列表中提取标签和数据。
    
    参数:
        data_list (list of tuple): 每个元组包含 (标签, 数据)。
    
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