import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import torch
import seaborn as sns
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import *
from util_d import *
from net1d import Net1D
from Dist_loss import DistLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

device_str = "cuda:2"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
######################################################################################
# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, names, data, labels):
        self.names = names
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.names[idx], self.data[idx], self.labels[idx]
############################################################################################
# Path and name setting
year = '2018'
test_data_mode = 'people'  # 'people','third'
test_data_normal = 0 # 1:包括正常样本 0:不包括正常样本

if test_data_mode == 'people':
    print('人民测试：')
elif test_data_mode == 'third':
    print('三院测试：')

if test_data_normal == 1:
    print('患病+正常')
else:
    print('只有患病')

# test_people_path = '/data/gjs/AIOG/FHRage/npy_disease/name_label_data_normal.npy'
test_people_d_path = f'/data/gjs/AIOG/FHRage/npy_disease/name_label_data_{year}.npy' # 某一年的患病数据
# test_people_d_path = f'/data/gjs/AIOG/FHRage/npy_disease/name_label_data_all_disease.npy' # 所有患病
# test_people_d_path = f'/data/gjs/AIOG/FHRage/npy_disease/name_label_data_normal.npy' # 正常数据

photo_path = '/home/gjs/Dist-Loss/photo/散点图/'
photo_test_name = f'test_{year}.png'
######################################################################################
batch_size = 256
if test_data_mode == 'people':

    # 加载测试数据
#     test_people_name, test_people_ga, test_people_fhr = extract_labels_and_data(test_people_path)
#     print(f'people_hospital_normal_data: {test_people_ga.shape}, {test_people_fhr.shape}')
    test_people_d_name, test_people_d_ga, test_people_d_fhr = extract_labels_and_data(test_people_d_path)
    print(f'people_hospital_disease_data: {test_people_d_ga.shape}, {test_people_d_fhr.shape}')

    # 筛选数据
#     test_people_name, test_people_ga, test_people_fhr = filter_test_data_by_label(test_people_name,test_people_ga, test_people_fhr)
    test_people_d_name, test_people_d_ga, test_people_d_fhr = filter_test_data_by_label(test_people_d_name, test_people_d_ga, test_people_d_fhr)
    
    if test_data_normal == 1:
        # 拼接数据
        test_people_name = np.concatenate((test_people_name, test_people_d_name), axis=0)
        test_people_ga = np.concatenate((test_people_ga, test_people_d_ga), axis=0)
        test_people_fhr = np.concatenate((test_people_fhr, test_people_d_fhr), axis=0)
    else:
        test_people_name = test_people_d_name
        test_people_fhr = test_people_d_fhr
        test_people_ga = test_people_d_ga

    test_name = test_people_name
    test_fhr = test_people_fhr
    test_ga = test_people_ga


# 统一数据长度
test_name, test_fhr, test_ga = random_resample_all(test_name, test_fhr, test_ga, 1800)

# 确保数据是数值类型
test_name = np.array(test_name, dtype=object)
test_fhr = np.array(test_fhr, dtype=np.float32)
test_fhr = np.expand_dims(test_fhr, axis=1)
test_ga = np.array(test_ga, dtype=np.float32)
print(f'test_data: {test_name.shape}, {test_fhr.shape}, {test_ga.shape}')

# 统计测试集中每个阶段的数据数量
# test_ga_counts = Counter(test_ga)
# print("测试集中每个阶段的数据数量：")
# for stage, count in test_ga_counts.items():
#     print(f"阶段 {stage}: {count} 条数据")
######################################################################################
# label distribution estimation
min_label, max_label = 0, 294
step = 1.0

# dataloader
batch_size = 256
shuffle = True
num_workers = 16
drop_last = True

# 使用默认带宽选择方法
kde = KernelDensity(kernel='gaussian')


kde.fit(test_ga.reshape(-1, 1))
test_density = np.exp(kde.score_samples(np.arange(min_label, max_label, step).reshape(-1, 1)))
test_batch_theoretical_labels = get_batch_theoretical_labels(test_density, batch_size, min_label, step=step)
test_batch_theoretical_labels = torch.tensor(test_batch_theoretical_labels, dtype=torch.float32).reshape(-1,1).to(device)

# print(test_batch_theoretical_labels)
######################################################################################
# 绘制标签分布和理论标签分布
plt.figure()

# 测试集
ax1 = plt.gca()  # 获取当前轴
ax2 = ax1.twinx()  # 创建一个共享 x 轴但有独立 y 轴的轴

# 绘制实际标签分布
ax1.hist(test_ga, bins=30, alpha=0.5, label='Actual Test Labels', color='blue')
ax1.set_xlabel('Label Value')
ax1.set_ylabel('Frequency (Actual)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 绘制理论标签分布
ax2.hist(test_batch_theoretical_labels.cpu().numpy(), bins=30, alpha=0.5, label='Theoretical Test Labels', color='orange')
ax2.set_ylabel('Frequency (Theoretical)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

ax1.set_title('Test Label Distribution vs Theoretical Label Distribution')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
plt.savefig(f'Test Theoretical Label Distribution.png')
######################################################################################
# 创建自定义数据集
test_ds = MyDataset(test_name, test_fhr, test_ga)

# 创建 DataLoader
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

print(f"Test data size: {len(test_ds)}")
######################################################################################
in_channels = 1
base_filters = 256
ratio = 1
filter_list = [256, 512, 512, 1024, 1024]
m_blocks_list = [2,3,3,2,2]
kernel_size = 16
stride = 2
groups_width = 16
verbose = False 
use_bn=True
use_do=False
n_classes = 1

net = Net1D(
    in_channels=in_channels,
    base_filters=base_filters,
    ratio=ratio,
    filter_list=filter_list,
    m_blocks_list=m_blocks_list,
    kernel_size=kernel_size,
    stride=stride,
    groups_width=groups_width,
    verbose=verbose,
    n_classes=n_classes,
    use_bn=use_bn,
    use_do=use_do,
    )

net = net.to(device)
######################################################################################
model_save_path = '/data/gjs/AIOG/FHRage/model/model_people_aug_special_DistLoss.pth'
net.load_state_dict(torch.load(model_save_path, map_location=device))
print(f"Loaded model from {model_save_path}")

# 初始化预测和真实值列表
pred, true, names = [], [], []

net.load_state_dict(torch.load(model_save_path, map_location=device))

# for names_batch, inp, tar in test_dl:
for names_batch, inp, tar in tqdm(test_dl, desc="Processing", unit="batch"):
    with torch.inference_mode():
        inp, tar = inp.to(device), tar.to(device)
        out = net(inp)
        out = list(out.detach().cpu().numpy().squeeze())
        pred += out
        true += list(tar.cpu().numpy().squeeze())
        names.extend(names_batch)
pred = np.round(np.array(pred), 1)
true = np.round(np.array(true), 1)

# 计算 MAE
mae = mean_absolute_error(true, pred)
print(f'MAE: {mae:.3f}')

# 计算 MSE
mse = mean_squared_error(true, pred)
print(f'MSE: {mse:.3f}')
######################################################################################
true_values = true
predictions = pred

# print("Length of test_name:", len(names))
# print("Length of true_values:", len(true_values))
# print("Length of predictions:", len(predictions))
# print("Length of difference:", len(predictions - true_values))
######################################################################################
error = pred - true
mean = error.mean()
std = error.std()
print(f'error mean: {mean:.3f}, error std: {std:.3f}')
coef = np.corrcoef(pred, true)[0][1]
print(f'coef:{coef}')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

count_pred = Counter(pred)
count_true = Counter(true)
sorted_pred_keys, sorted_pred_values = zip(*sorted(zip(count_pred.keys(), count_pred.values())))
sorted_true_keys, sorted_true_values = zip(*sorted(zip(count_true.keys(), count_true.values())))
axs[0].bar(sorted_true_keys, sorted_true_values, alpha=1, width=0.05, color='darkred', label='Actual distribution')
axs[0].bar(sorted_pred_keys, sorted_pred_values, alpha=1, width=0.05, label='Predicted distribution')
# axs[0].set_ylim(0, 10)
axs[0].set_xlabel('Values')
axs[0].set_ylabel('Counts')
axs[0].set_title('Bar Plot')
axs[0].legend()

pred_percentiles = np.percentile(pred, np.linspace(0, 100, 100))
true_percentiles = np.percentile(true, np.linspace(0, 100, 100))
axs[1].plot(true_percentiles,pred_percentiles, marker='o', linestyle='')
axs[1].plot([200, 290], [200, 290], color='red', linestyle='--')  
axs[1].set_xlim(200, 290)
axs[1].set_ylim(200, 290)
axs[1].set_xlabel('True Values Percentiles')
axs[1].set_ylabel('Predicted Values Percentiles')
axs[1].set_title('QQ Plot')

plt.tight_layout()
plt.savefig(f'{photo_path}QQ')
plt.clf()
# 绘制详细散点图
plt.figure()
plt.scatter(true, pred, s=0.5)

# 添加标题和标签
plt.xlabel('Actual Gestational Age (days)')
plt.ylabel('Predicted Gestational Age (days)')

plt.xlim(200, 290)
plt.ylim(200, 290)
plt.savefig(f'{photo_path}{photo_test_name}')
######################################################################################

plt.clf()
# 绘制详细散点图
actual_gestational_age = np.array(true_values)
predicted_gestational_age = np.array(predictions)
# 计算数据的最小值和最大值
# min_age = min(np.min(actual_gestational_age), np.min(predicted_gestational_age)) - 2
# max_age = max(np.max(actual_gestational_age), np.max(predicted_gestational_age)) + 2
min_age = 0
max_age = 290

# 创建散点图
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制散点图
ax.scatter(actual_gestational_age, predicted_gestational_age, s=0.5)
# 绘制45度参考线（理想的一致性线）
ax.plot([min_age, max_age], [min_age, max_age], color='red', linestyle='--', label='Ideal Consistency')

# 绘制相差21天的灰色虚线
ax.plot([min_age, max_age], [min_age + 21, max_age + 21], color='gray', linestyle='--', label='±21 days')
ax.plot([min_age, max_age], [min_age - 21, max_age - 21], color='gray', linestyle='--')

# 添加图例
ax.legend()

# 添加标题和标签
ax.set_xlabel('Actual Gestational Age (days)')
ax.set_ylabel('Predicted Gestational Age (days)')

if test_data_mode == 'people':
    plt.xlim(200, 290)
    plt.ylim(200, 290)
elif test_data_mode == 'third':
    plt.xlim(200, 290)
    plt.ylim(200, 290)

plt.savefig(f'{photo_path}21_{photo_test_name}')

# 创建一个 DataFrame 来保存结果
results_df = pd.DataFrame({
    'name': names,
    'true_value': true_values,
    'prediction': predictions,
    'difference': predictions - true_values
})

# 保存到 CSV 文件
csv_file_path = f'/home/gjs/Dist-Loss/test_results/result_{year}.csv'  # 指定保存路径
results_df.to_csv(csv_file_path, index=False)
print(f"结果已保存到 {csv_file_path}")
