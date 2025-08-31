"""
train.py
训练一个全局模型，解决数据不平衡问题
"""
from utils import *
from util_gu import *
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from net1d import Net1D
from dataset import MyDataset
from Dist_loss import DistLoss
from torch.nn import HuberLoss
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, linregress
from scipy.stats import truncnorm
import os, gc, torch, numpy as np
import pandas as pd
import matplotlib
import scipy.stats as st
import matplotlib.pyplot as plt
import collections, pprint
import sys

matplotlib.use('Agg')

# -------------------- 1. 基础配置 --------------------

# 设备配置
device_str = "cuda:2"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 数据路径
train_path = '/data/CTGage/npy_normal/fhr_data_with_labels_and_ids-train.npy'
val_path   = '/data/CTGage/npy_normal/fhr_data_with_labels_and_ids-val.npy'

# 超参数
epochs = 500
lr = 1e-3
weight_decay = 1e-3
batch_size = 128
window_size = 1800
step_size = 200
patience = 100
min_delta = 2e-4
min_label, max_label = 100, 294
step = 1.5
bw_method = 7

# 时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = f'/data/CTGage/model/{timestamp}'
os.makedirs(model_dir, exist_ok=True)

# -------------------- 2. 数据加载与预处理 --------------------

print("\nLoading & preprocessing data ...")

# 加载原始数据
train_id_raw, train_ga_raw, train_fhr_raw = extract_labels_and_data(train_path)
train_id_filter, train_ga_filter, train_fhr_filter = filter_data_by_label(train_id_raw, train_ga_raw, train_fhr_raw)
val_id_raw, val_ga_raw, val_fhr_raw = extract_labels_and_data(val_path)
val_id_filter, val_ga_filter, val_fhr_filter = filter_data_by_label(val_id_raw, val_ga_raw, val_fhr_raw)


# 数据归一化
all_points = np.concatenate(train_fhr_filter)
mean = all_points.mean()
std = all_points.std()
train_fhr_norm = (train_fhr_filter - mean) / (std + 1e-6)
val_fhr_norm = (val_fhr_filter - mean) / (std + 1e-6)

# 数据增强
train_id_aug, train_ga_aug, train_fhr_aug = sliding_window_data_augmentation(train_id_filter, train_ga_filter, train_fhr_norm, window_size, step_size)

# 验证集处理
val_id_rand, val_fhr_rand, val_ga_rand = random_resample_all(val_id_filter, val_fhr_norm, val_ga_filter, window_size)

# 数据重命名
train_id = train_id_aug
train_fhr = train_fhr_aug
train_ga = train_ga_aug
val_id = val_id_rand
val_fhr = np.array(val_fhr_rand, dtype=np.float32)
val_ga = np.array(val_ga_rand, dtype=np.float32)

# -------------------------------------------------
def get_y_theo(train_ga, min_label, max_label, step, name='train'):
    # 1. 计算高斯参数（基于训练集）
    mu    = np.mean(train_ga)
    sigma = np.std(train_ga) * 0.8

    # 2. 构造截断正态分布
    a, b = (min_label - mu) / sigma, (max_label - mu) / sigma
    trunc_dist = truncnorm(a, b, loc=mu, scale=sigma)

    # 3. 计算密度 + 强制保底
    label_range = np.arange(min_label, max_label + step, step)
    density = trunc_dist.pdf(label_range)

    # ✅ 强制每个 bin 都有占比（极小保底）
    eps = 1e-4
    density += eps
    density /= density.sum()

    # 4. 生成 batch 的理论标签
    y_theo = get_batch_theoretical_labels(density, batch_size, min_label, step=step)
    y_theo = torch.tensor(y_theo, dtype=torch.float32).to(device).unsqueeze(1)

    plt.figure(figsize=(8, 4))

    bins = np.arange(min_label, max_label + step, step)
    plt.hist(train_ga, bins=bins, alpha=0.5, label='True GA', color='tab:blue', density=True)
    plt.hist(y_theo.cpu().numpy(),   bins=bins, alpha=0.5, label='Theo GA',   color='tab:orange', density=True)

    x = np.linspace(min_label, max_label, 500)
    y = st.norm.pdf(x, loc=mu, scale=sigma)
    y /= y.sum() * step     # 与直方图密度对齐

    plt.plot(x, y, 'r-', lw=2, label=f'Theoretical (μ={mu:.1f}, σ={sigma:.1f})')
    plt.xlabel('Gestational Age (days)')
    plt.ylabel('Density')
    plt.title(f'{name} True Labels vs Theoretical Distribution')
    plt.legend()
    plt.tight_layout()
    photo_path = os.path.join(model_dir, f"{name}_distribution.png")
    plt.savefig(photo_path)

    return y_theo

y_theo_train = get_y_theo(train_ga, min_label, max_label, step, 'Train')
y_theo_val = get_y_theo(val_ga, min_label, max_label, step, 'Val')

# 增加维度
train_fhr = np.expand_dims(train_fhr, 1)
val_fhr   = np.expand_dims(val_fhr,   1)

# 计算每个样本的权重
count = collections.Counter(train_ga)
weights = np.array([1.0 / count[int(y)] for y in train_ga])
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

# 创建数据集和数据加载器
train_ds = MyDataset(train_fhr, train_ga, train_id)
train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)

val_ds = MyDataset(val_fhr, val_ga, val_id)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)


# -------------------- 3. 模型配置 --------------------

def build_model():
    return Net1D(
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
    )

# 损失函数
criterion_l1 = nn.L1Loss()
loss_fn = DistLoss(
    loss_fn='L1',
    loss_weight= 0.3, # 0.3,
    regularization_strength=0.1,
    require_loss_values=True
).to(device)

# -------------------- 4. 训练全局模型 --------------------

print("\nTraining global model ...")

net = build_model().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

best_val_loss = float('inf')
best_state = None
counter = 0
train_total, train_dist, train_plain, train_slope = 0.0, 0.0, 0.0, 0.0

for epoch in range(epochs):
    # 训练
    net.train()
    train_loss = 0
    for x, y, _ in train_dl:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        pred = net(x)
        
        total_train, dist_train, plain_train, slope_train = loss_fn(pred, y, y_theo_train)
        
        loss = total_train

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() # 批次平均
        train_total += total_train.item() * y.size(0) # 样本平均
        train_dist += dist_train.item() * y.size(0)
        train_plain += plain_train.item() * y.size(0)
        train_slope += slope_train.item() * y.size(0)
    train_loss /= len(train_dl)
    train_total /= len(train_ds)
    train_dist /= len(train_ds)
    train_plain /= len(train_ds)
    train_slope /= len(train_ds)

    # 验证
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y, _ in val_dl:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            pred = net(x)

            total_val, dist_val, plain_val, slope_val = loss_fn(pred, y, y_theo_val)
        
    scheduler.step(total_val)
    total_val /= len(val_ds)
    dist_val /= len(val_ds)
    plain_val /= len(val_ds)
    slope_val /= len(val_ds)
    val_loss = total_val 

    print(f"Epoch {epoch+1:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | lr {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Train: total={total_train:.4f}, dist={dist_train:.4f}, plain={plain_train:.4f}, slope={slope_train:.4f}")
    print(f"Val: total={total_val:.4f}, dist={dist_val:.4f}, plain={plain_val:.4f}, slope={slope_val:.4f}")

    # 早停
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_state = net.state_dict()
        counter = 0
        # 保存模型
        model_path = os.path.join(model_dir, "global_model.pth")
        torch.save(best_state, model_path)
        print(f"Saved {model_path}")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break

# -------------------- 5. 生成整体散点图 --------------------

def evaluate_train(model, data_loader, title, save_path):
    true_values = []
    predicted_values = []
    subject_ids = []
    with torch.no_grad():
        for x, y, ids in data_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            pred = model(x)
            true_values.extend(y.cpu().numpy().ravel())
            predicted_values.extend(pred.cpu().numpy().ravel())
            subject_ids.extend(ids.cpu().numpy().ravel())

    print("Sample subject IDs:", subject_ids[:5])

    # 计算拟合直线
    slope, intercept, r_value, p_value, std_err = linregress(true_values, predicted_values)
    fit_line = slope * np.array(true_values) + intercept

    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, predicted_values, s=10, alpha=0.5, color='tab:blue')
    plt.plot([200, 300], [200, 300], 'r--', linewidth=1)
    plt.plot(true_values, fit_line, 'g-', linewidth=2, label=f'Fit y={slope:.2f}x+{intercept:.2f}')
    plt.xlabel('True CTGage (days)')
    plt.ylabel('Predicted CTGage (days)')
    plt.title(title)
    plt.xlim(200, 300)
    plt.ylim(200, 300)
    plt.tight_layout()
    plt.savefig(save_path)
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r, _ = pearsonr(true_values, predicted_values)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f'Pearson: {r:.4f}')
    print(f"Saved plot to {save_path}")


# 加载最佳模型
net.load_state_dict(torch.load(model_path))
net.eval()

# 评估训练集和验证集
evaluate_train(net, train_dl, "Train Set: True vs Predicted", os.path.join(model_dir, 'train_scatter.png'))
