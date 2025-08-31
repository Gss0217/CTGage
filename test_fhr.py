import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from net1d import Net1D
from util_gu import *
import warnings
warnings.filterwarnings("ignore")
'''
# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = Net1D(
    in_channels=1,
    base_filters=256,
    ratio=1,
    filter_list=[128, 256, 256, 512, 512],
    m_blocks_list=[1, 2, 2, 1, 1],
    kernel_size=16,
    stride=2,
    groups_width=16,
    n_classes=1,
    use_bn=True,
    use_do=True,
    dropout_rate=0.2
).to(device)

model_dir = '/data/gjs/AIOG/FHRage/model/20250729_231225/'
model_path = f'{model_dir}model/global_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 创建自定义数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 定义文件路径
base_dir = '/data/gjs/AIOG/FHRage/model/20250729_231225/'
npy_file = f'{base_dir}all_disease_combined_fhr.npy'  # 替换为你的.npy文件路径
train_path = '/data/gjs/AIOG/FHRage/npy_normal/fhr_data_with_labels_and_ids-train.npy'


# 加载.npy文件
data = np.load(npy_file, allow_pickle=True)
train_id_raw, train_ga_raw, train_fhr_raw = extract_labels_and_data(train_path)
train_id_filter, train_ga_filter, train_fhr_filter = filter_data_by_label(train_id_raw, train_ga_raw, train_fhr_raw)

# 检查数据结构
print("原始数据类型:", type(data))
print("原始数据长度:", len(data))
print("第一个样本的形状:", data[0].shape)

# 提取每个样本的前1834个数据，如果数据不够就跳过这个样本
num_data_points = 1834
all_combined_data = np.array([sample[:num_data_points] for sample in data if len(sample) >= num_data_points])

# 检查重新组织后的数据形状
print("重新组织后的数据形状:", all_combined_data.shape)




# 假设前34列是疾病信息，其余是胎心数据
num_disease_columns = 34
disease_info = all_combined_data[:, :num_disease_columns]
fhr_data = all_combined_data[:, num_disease_columns:]

# 数据归一化
all_points = np.concatenate(train_fhr_filter)
mean = all_points.mean()
std = all_points.std()
train_fhr_norm = (train_fhr_filter - mean) / (std + 1e-6)
target_fhr_data = (fhr_data - mean) / (std + 1e-6)

print(disease_info)
print(target_fhr_data)
print(disease_info.shape)
print(target_fhr_data.shape)

# 根据孕龄差进行分组
gestational_age_diff = disease_info[:, 2]  # 假设第三列是孕龄差
group_1_indices = np.where(gestational_age_diff <= -21)[0]  # 小于等于 -21
# group_2_indices = np.where((gestational_age_diff > -21) & (gestational_age_diff <= -7))[0]  # 大于 -21 且小于等于 -7
group_3_indices = np.where((gestational_age_diff > -7) & (gestational_age_diff < 7))[0]  # 大于 -7 且小于 7
# group_4_indices = np.where((gestational_age_diff >= 7) & (gestational_age_diff < 21))[0]  # 大于等于 7 且小于 21
group_5_indices = np.where(gestational_age_diff >= 21)[0]  # 大于等于 21

print(group_1_indices)

# 提取各组的胎心数据
group_1_fhr_data = target_fhr_data[group_1_indices]
# group_2_fhr_data = fhr_data[group_2_indices]
group_3_fhr_data = target_fhr_data[group_3_indices]
# group_4_fhr_data = fhr_data[group_4_indices]
group_5_fhr_data = target_fhr_data[group_5_indices]

# 提取特征
def extract_features(model, data_loader):
    features = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            # 假设模型的特征提取部分是 model.feature_extractor
            feature = model.extract_last_conv_features(batch)
            features.append(feature.cpu().numpy())
    return np.concatenate(features, axis=0)

def get_features_from_layer(model, layer_path, data_loader):
    def hook(module, input, output):
        # print("Hook triggered:", output.shape)
        features.append(output.cpu().detach().numpy())

    # Split the layer path into parts
    parts = layer_path.split('.')
    target_module = model

    # Traverse the module hierarchy
    for part in parts:
        if part.isdigit():
            target_module = target_module[int(part)]
        else:
            target_module = getattr(target_module, part)

    # Register the hook
    handle = target_module.register_forward_hook(hook)
    features = []

    # Forward pass to trigger the hook
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            _ = model(batch)

    # Remove the hook
    handle.remove()
    return np.concatenate(features, axis=0)

# 创建 DataLoader
def create_dataloader(data):
    data = np.array(data, dtype=np.float32)
    data = np.expand_dims(data, axis=1)
    dataset = MyDataset(data)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, drop_last=False)

# 提取每组的特征
layer_names = [
    'stage_list.0.block_list.0.se_fc2',
    'stage_list.2.block_list.0.se_fc2',
    'stage_list.4.block_list.0.se_fc2'
]

for layer_name in layer_names:  # 选择你想要提取特征的层
    group_1_dl = create_dataloader(group_1_fhr_data)
    group_3_dl = create_dataloader(group_3_fhr_data)
    group_5_dl = create_dataloader(group_5_fhr_data)

    group_1_features = get_features_from_layer(model, layer_name, group_1_dl)
    group_3_features = get_features_from_layer(model, layer_name, group_3_dl)
    group_5_features = get_features_from_layer(model, layer_name, group_5_dl)

    group_1_features = group_1_features.reshape(group_1_features.shape[0],-1)
    group_3_features = group_3_features.reshape(group_3_features.shape[0],-1)
    group_5_features = group_5_features.reshape(group_5_features.shape[0],-1)

    print(layer_name)
    print(group_1_features.shape)
    # print(group_2_fhr_data)
    print(group_3_features.shape)
    # print(group_4_fhr_data)
    print(group_5_features.shape)

    if layer_name == 'stage_list.0.block_list.0.se_fc2':
        photo_name = 'stage 0'
    elif layer_name == 'stage_list.2.block_list.0.se_fc2':
        photo_name = 'stage 2'
    elif layer_name == 'stage_list.4.block_list.0.se_fc2':
        photo_name = 'stage 4'




    # 应用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42)

    # 对每组数据分别进行 t-SNE 降维
    group_1_tsne_result = tsne.fit_transform(group_1_features)
    # group_2_tsne_result = tsne.fit_transform(group_2_fhr_data)
    group_3_tsne_result = tsne.fit_transform(group_3_features)
    # group_4_tsne_result = tsne.fit_transform(group_4_fhr_data)
    group_5_tsne_result = tsne.fit_transform(group_5_features)

    plt.rcParams.update({
        'font.size': 20,           # 全局字号
        'axes.labelsize': 22,      # xy轴标签
        'axes.titlesize': 24,      # 标题
        'xtick.labelsize': 20,     # x轴刻度
        'ytick.labelsize': 20,     # y轴刻度
        'legend.fontsize': 20      # 图例
    })
    
    # 创建一个图形
    plt.figure(figsize=(10, 8))

    # 绘制各组的数据
    # colors = ['red', 'orange', 'green', 'blue', 'purple']
    # labels = ['Group 1 (≤-21)', 'Group 2 (-21,-7]', 'Group 3 (-7,7)', 'Group 4 [7,21)', 'Group 5 (≥21)']
    # for i, tsne_result in enumerate([group_1_tsne_result, group_2_tsne_result, group_3_tsne_result, group_4_tsne_result, group_5_tsne_result]):
    #     plt.scatter(tsne_result[:, 0], tsne_result[:, 1], label=labels[i], color=colors[i])

    # colors = [ 'green', 'blue','red']
    # colors = ['#90EE90', '#ADD8E6', '#FFB6C1']
    colors = [
        '#A8D8A8',  # 柔和薄荷绿（低饱和绿）
        '#A8C8E8',  # 柔和雾蓝（低饱和蓝）
        '#F0A8A8'   # 柔和玫瑰红（低饱和红）
    ]
    labels = ['Group 3 (-7,7)', 'Group 5 (>21)','Group 1 (<-21)']
    for i, tsne_result in enumerate([group_3_tsne_result, group_5_tsne_result,group_1_tsne_result]):
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], label=labels[i], color=colors[i])

    # 添加图例
    plt.legend()

    # 添加标题和坐标轴标签
    plt.title(f't-SNE for Different CTGage-gap Groups of {photo_name}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # 显示图形
    plt.savefig(f'/data/gjs/AIOG/FHRage/model/20250729_231225/photo/TSNE_feature/{layer_name}.png')


'''

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from net1d import Net1D
from util_gu import *
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
import random

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = Net1D(
    in_channels=1,
    base_filters=256,
    ratio=1,
    filter_list=[128, 256, 256, 512, 512],
    m_blocks_list=[1, 2, 2, 1, 1],
    kernel_size=16,
    stride=2,
    groups_width=16,
    n_classes=1,
    use_bn=True,
    use_do=True,
    dropout_rate=0.2
).to(device)

model_dir = '/data/gjs/AIOG/FHRage/model/20250729_231225/'
model_path = f'{model_dir}model/global_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 创建自定义数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_path = '/data/gjs/AIOG/FHRage/npy_normal/fhr_data_with_labels_and_ids-train.npy'
# test_path = f'/data/gjs/AIOG/FHRage/model/20250729_231225/target_fhr_data_{group_number}.npy'
test_path = f'{model_dir}all_test_name_gap_fhr.npy'

test_name_fhr = np.load(test_path, allow_pickle=True)
test_name = []
test_gap = []
test_fhr = []
for data in test_name_fhr:
    name = data[0]
    gap = data[1]
    fhr = data[2:1802]
    test_name.append(name)
    test_gap.append(gap)
    test_fhr.append(fhr)

test_name = np.array(test_name)
test_gap = np.array(test_gap, dtype=np.float32)
test_fhr = np.array(test_fhr, dtype=np.float32)
original_test_fhr = test_fhr.copy()

print(test_fhr.shape)


train_id_raw, train_ga_raw, train_fhr_raw = extract_labels_and_data(train_path)
train_id_filter, train_ga_filter, train_fhr_filter = filter_data_by_label(train_id_raw, train_ga_raw, train_fhr_raw)


# 数据归一化
all_points = np.concatenate(train_fhr_filter)
mean = all_points.mean()
std = all_points.std()
train_fhr_norm = (train_fhr_filter - mean) / (std + 1e-6)
target_fhr_data = (test_fhr - mean) / (std + 1e-6)

# 创建数据集
target_fhr_data = np.array(target_fhr_data, dtype=np.float32)
test_data = np.expand_dims(target_fhr_data, axis=1)
print(test_data.shape)
test_ds = MyDataset(test_data)

# 创建 DataLoader
batch_size = 1  # 每次处理一个样本
shuffle = False  # 不打乱数据以便于后续处理
num_workers = 16
drop_last = False
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

def group_samples(gaps, names, data, original_data):
    groups = {1: [], 2: [], 3: [], 4: [], 5: []}
    for i, gap in enumerate(gaps):
        if gap < -21:
            groups[1].append((names[i], data[i], original_data[i]))
        elif -21 <= gap < -7:
            groups[2].append((names[i], data[i], original_data[i]))
        elif -7 <= gap < 7:
            groups[3].append((names[i], data[i], original_data[i]))
        elif 7 <= gap < 21:
            groups[4].append((names[i], data[i], original_data[i]))
        else:  # gap >= 21
            groups[5].append((names[i], data[i], original_data[i]))
    return groups

# 分组
groups = group_samples(test_gap, test_name, test_data, original_test_fhr)

# 定义一个函数来计算输入数据的梯度
def get_input_gradients(model, input_data):
    input_data.requires_grad_(True)
    output = model(input_data)
    output.mean().backward()  # 取输出的均值作为标量进行反向传播
    gradients = input_data.grad.data.cpu().numpy()
    return gradients

color_max = 1 # 1e-10
# 自定义颜色映射从特定的蓝色到特定的红色
blue_color = (118/255, 149/255, 189/255)  # RGB: (118, 149, 189)
red_color = (255/255, 0/255, 0/255)   # RGB: (186, 106, 106)
cmap = LinearSegmentedColormap.from_list("custom", [blue_color, red_color])

# 设置随机种子以确保结果可复现
random.seed(42)

specific_names = [
    '78af5e6ce9c95259975a8c0a762dd177',
    '1d73644f5cd31b5964bf6a625f352e23',

    '84d20dc7d112467515ecea832a02038b',
    'f19bdeb5762dc8eba4c6a744dbf009b2',

    '6c9fbd31f547e00a32de6f7d0e0275fe',
    'f968c685fefbd20959bcf6b860441fc2',

    '53bedfb9ba652b2671c5cd6bb86fc1c2',
    '9e91f3e74b63c7270dffd5809631c939',

    '9b0606d91994817488e36724c3792cb0',
    'b26f0e369c0041749cb8ace5728fe2e2',
    'd31e1ad5103b948a49e0dd64daf36c10',
    'cc6fa49c2dc78a996709adcb8027e60e',
    '7a1f7ea724e40aefa746152aa96e982d'

    ]

plt.rcParams.update({
    'font.size': 20,           # 全局字号
    'axes.labelsize': 22,      # xy轴标签
    'axes.titlesize': 24,      # 标题
    'xtick.labelsize': 20,     # x轴刻度
    'ytick.labelsize': 20,     # y轴刻度
    'legend.fontsize': 20      # 图例
})
    
for group_id in [1, 2, 3, 4, 5]:
    samples = groups[group_id]
    # # 如果样本数超过100个，则随机抽取100个
    # if len(samples) > 50:
    #     samples = random.sample(samples, 50) 
    selected_samples = [sample for sample in samples if sample[0] in specific_names]
    if selected_samples == None:
        continue
    for idx, (name, sample, original_sample) in enumerate(selected_samples):
        # 计算梯度
        sample_norm = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
        gradients = get_input_gradients(model, sample_norm)

        # 可视化原始信号和注意力部分
        plt.figure(figsize=(30, 6))

        # 计算注意力权重
        attention_weights = np.abs(gradients.squeeze())
        attention_weights = np.log1p(attention_weights)
        attention_weights = attention_weights / np.max(attention_weights) * color_max

        # 使用高斯平滑处理注意力权重
        attention_weights_smoothed = gaussian_filter1d(attention_weights, sigma=5)

        # 使用颜色映射为信号分配颜色
        colors = cmap(attention_weights_smoothed / color_max)

        # 把 0 值替换为 nan
        original_sample_plot = original_sample.astype(float)
        original_sample_plot[original_sample_plot == 0] = np.nan

        # 绘制带有颜色的信号
        for i in range(len(original_sample_plot) - 1):
            if original_sample_plot[i] >= 0 and original_sample_plot[i + 1] >= 0:
                plt.plot([i, i + 1], 
                         [original_sample_plot[i], original_sample_plot[i + 1]], 
                         color=colors[i], linewidth=4)

        # 设置纵坐标范围
        plt.ylim(90, 220)

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, orientation='vertical')
        cbar.set_label('Attention Weights')

        # 添加图例
        plt.title(f'Original Signal with Attention-based Coloring (Sample {name})')
        plt.xlabel('Time')
        plt.ylabel('CTG')

        plt.tight_layout()
        # 保存图像
        plt.savefig(f'/data/gjs/AIOG/FHRage/model/20250729_231225/photo/fhr_paper/group_{group_id}_sample_{name}.png', bbox_inches='tight', dpi=300)

