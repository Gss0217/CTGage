import torch, numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from net1d import Net1D
from dataset import MyDataset
from utils import *
from util_gu import *
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, linregress
import os
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(87)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, fhr, label, ids=None):
        self.fhr = torch.tensor(fhr, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)
        self.ids = ids  # 新增

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if self.ids is not None:
            return self.fhr[idx], self.label[idx], self.ids[idx]
        else:
            return self.fhr[idx], self.label[idx]

device = torch.device('cuda:2')

is_disease = 2022 # 0:normal / 1:disease / 2018-2022:year / -1:外部验证

model_dir = '/data/gjs/AIOG/FHRage/model/20250729_231225/'

train_path = '/data/gjs/AIOG/FHRage/npy_normal/fhr_data_with_labels_and_ids-train.npy'


if is_disease == 1:
    test_path = '/data/gjs/AIOG/FHRage/npy_disease/name_label_data_all_disease.npy'
    photo_title = 'All Diseased'
elif is_disease == 0:
    test_path = '/data/gjs/AIOG/FHRage/npy_disease/name_label_data_normal.npy'
    photo_title = 'Normal'
elif is_disease == -1:
    test_path = '/data/gjs/AIOG/PK_Third_Hospital/data/processed_data_pre/npy_long/normal_data.npy'
    photo_title = 'Third'
else:
    year = is_disease
    test_path = f'/data/gjs/AIOG/FHRage/npy_disease/name_label_data_{year}.npy'
    photo_title = year

print(photo_title)

window_size = 1800
batch_size = 128

# 加载数据集
train_id_raw, train_ga_raw, train_fhr_raw = extract_labels_and_data(train_path)
train_id_filter, train_ga_filter, train_fhr_filter = filter_data_by_label(train_id_raw, train_ga_raw, train_fhr_raw)
test_id_raw, test_ga_raw, test_fhr_raw = extract_labels_and_data(test_path)
print(len(test_fhr_raw))
test_id_filter, test_ga_filter, test_fhr_filter = filter_data_by_label(test_id_raw, test_ga_raw, test_fhr_raw)



# 数据归一化
all_points = np.concatenate(train_fhr_filter)
mean = all_points.mean()
std = all_points.std()
train_fhr_norm = (train_fhr_filter - mean) / (std + 1e-6)
test_fhr_norm = (test_fhr_filter - mean) / (std + 1e-6)


test_id_rand, test_fhr_rand, test_ga_rand = random_resample_all(test_id_filter, test_fhr_norm, test_ga_filter, window_size)

test_id = test_id_rand
test_fhr = np.array(test_fhr_rand, dtype=np.float32)
test_ga = np.array(test_ga_rand, dtype=np.float32)

test_fhr   = np.expand_dims(test_fhr,   1)

print(test_fhr.shape)


test_ds = MyDataset(test_fhr, test_ga)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

# 构建模型并加载权重
net = Net1D(
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

def evaluate_and_plot(model, data_loader, title, save_path, names=None):
    true_values = []
    predicted_values = []

    # 收集所有数据
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            pred = model(x)
            true_values.extend(y.cpu().numpy().ravel())
            predicted_values.extend(pred.cpu().numpy().ravel())

    # 构造 DataFrame
    df = pd.DataFrame({
        'true': true_values,
        'pred': predicted_values
    })

    true_values = df['true'].values
    predicted_values = df['pred'].values

    # 计算指标
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r, _ = pearsonr(true_values, predicted_values)

    # 计算拟合直线
    slope, intercept, r_value, p_value, std_err = linregress(true_values, predicted_values)
    fit_line = slope * np.array(true_values) + intercept

    plt.rcParams.update({
        'font.size': 14,           # 全局字号
        'axes.labelsize': 16,      # xy轴标签
        'axes.titlesize': 18,      # 标题
        'xtick.labelsize': 14,     # x轴刻度
        'ytick.labelsize': 14,     # y轴刻度
        'legend.fontsize': 14      # 图例
    })

    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, predicted_values, s=10, alpha=0.5, color='tab:blue')
    plt.plot([180, 300], [180, 300], 'r--', linewidth=1)
    # plt.plot(true_values, fit_line, 'g-', linewidth=2, label=f'Fit y={slope:.2f}x+{intercept:.2f}')

    text_str = f'MAE = {mae:.2f}\nMSE = {mse:.2f}\nPearson = {r:.3f}'
    plt.text(260, 190, text_str, fontsize=10, bbox=dict(boxstyle='square', facecolor='white'))

    plt.xlabel('True CTGage (days)')
    plt.ylabel('Predicted CTGage (days)')
    plt.title(title)
    plt.xlim(180, 300)
    plt.ylim(180, 300)
    plt.tight_layout()
    plt.savefig(save_path+'.png')
    print(f"Saved plot to {save_path}")

    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, predicted_values, s=1, alpha=0.5, color='tab:blue')
    plt.plot([180, 300], [180, 300], 'r--', linewidth=1)
    # plt.plot(true_values, fit_line, 'g-', linewidth=2, label=f'Fit y={slope:.2f}x+{intercept:.2f}')
    
    plt.plot([180, 300], [180-21, 300-21], 'gray', linestyle='--', linewidth=1)
    plt.plot([180, 300], [180+21, 300+21], 'gray', linestyle='--', linewidth=1)

    plt.plot([], [], 'gray', linestyle='--', linewidth=1, label='±21 days')
    plt.legend(loc='upper left', fontsize=8)

    text_str = f'MAE = {mae:.2f}\nMSE = {mse:.2f}\nPearson = {r:.3f}'
    plt.text(260, 190, text_str, fontsize=10, bbox=dict(boxstyle='square', facecolor='white'))

    plt.xlabel('True CTGage (days)')
    plt.ylabel('Predicted CTGage (days)')
    plt.title(title)
    plt.xlim(180, 300)
    plt.ylim(180, 300)
    plt.tight_layout()
    plt.savefig(save_path+'_21.png')
    print(f"Saved plot to {save_path}")

    ga_min = 240
    ga_max = 280
    mae = mean_absolute_error(true_values, predicted_values)
    mae_low  = mean_absolute_error(true_values[true_values < ga_min],  predicted_values[true_values < ga_min])
    mae_mid  = mean_absolute_error(true_values[(true_values >= ga_min) & (true_values <= ga_max)], predicted_values[(true_values >= ga_min) & (true_values <= ga_max)])
    mae_high = mean_absolute_error(true_values[true_values > ga_max], predicted_values[true_values > ga_max])

    print(f"Overall MAE : {mae:.3f}")
    print(f"<{ga_min}  d MAE : {mae_low:.3f}")
    print(f"{ga_min}-{ga_max} MAE : {mae_mid:.3f}")
    print(f">{ga_max}  d MAE : {mae_high:.3f}")

    print(f"{title} | MAE: {mae:.4f} | MSE: {mse:.4f} | Pearson: {r:.4f}")

    # results_df = pd.DataFrame({
    #     'name': names,
    #     'true_value': true_values,
    #     'prediction': predicted_values,
    #     'difference': predicted_values - true_values
    # })

    # 保存到 CSV 文件
    csv_file_path = f'{model_dir}result_{photo_title}.csv'  # 指定保存路径
    # results_df.to_csv(csv_file_path, index=False)
    print(f"结果已保存到 {csv_file_path}")

# 加载最佳模型
model_path = f'{model_dir}model/global_model.pth'
net.load_state_dict(torch.load(model_path))
net.eval()

# 评估
evaluate_and_plot(
    net,
    test_dl,
    photo_title,
    os.path.join(model_dir, f'{photo_title}'),
    names=test_id
)

