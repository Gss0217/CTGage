import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from sklearn.utils import resample
import pywt

# 指定CSV文件的路径
file_path = '/data/gjs/AIOG/FHRage/model/20250729_231225/all_combined.csv'  # 替换为你的CSV文件路径

target_dir = '/data/gjs/AIOG/FHRage/model/20250729_231225/photo/outcome_non/'

# 定义新的列名列表
new_column_names = ['name','prediction','true_value','difference','死胎','流产','减胎','胎停','胎死宫内','分娩方式',
                    '1min评分','5min评分','10min评分','HDP', 'GDM', 'thyroid disease', 'anemia', 'immune disease', 
                    'pregnancy with hysteromyoma', 'maternal congenital disease', 'oligohydramnios',' polyhydramnios',
                    'uterine malformations',' umbilical cord problems', 'placental pathology', 
                    'premature infants', 'low birth weight infants','neonatal asphyxia',
                    'fetal distress', 'giant infants','fetal malformations', 'chromosomal malformations',
                    'neonatal congenital heart disease',' neonatal congenital renal cysts', 'year'] 

outcome_m = ['HDP', 'GDM', 'thyroid disease', 'anemia', 'immune disease', 'pregnancy with hysteromyoma', 
                'maternal congenital disease', 'oligohydramnios',' polyhydramnios', 'uterine malformations',
                ' umbilical cord problems', 'placental pathology']

outcome_f = ['premature infants', 'low birth weight infants','neonatal asphyxia',
                'fetal distress','fetal malformations']

# 使用pandas的read_csv函数读取文件
df = pd.read_csv(file_path, header=0, names=new_column_names)

# 假设 difference 是孕龄差
df['gestational_age_diff'] = df['difference']

# 定义自定义分桶函数
def f2(var, df, bins, target):
    # 自定义分桶边界，让靠近 0 的地方分桶更密集
    positive_boundaries = np.logspace(0, np.log10(df[var].max()), bins // 2)
    negative_boundaries = -np.logspace(0, np.log10(-df[var].min()), bins // 2)[::-1]
    bin_boundaries = np.concatenate([negative_boundaries, [0], positive_boundaries])

    mean_age_difference = []
    ratio = []
    bin_up = []
    bin_low = []

    for i in range(len(bin_boundaries) - 1):
        bin_low_val = bin_boundaries[i]
        bin_up_val = bin_boundaries[i + 1]
        group = df[(df[var] > bin_low_val) & (df[var] <= bin_up_val)]

        bin_low.append(bin_low_val)
        bin_up.append(bin_up_val)

        if len(group) > 0:
            affected_count = group[target].sum()  # 患病数量
            unaffected_count = len(group) - affected_count  # 不患病数量
            ratio_val = affected_count / unaffected_count if unaffected_count != 0 else np.nan
            ratio.append(ratio_val)
            med_age_difference = group[var].mean()
            mean_age_difference.append(med_age_difference)
        else:
            ratio.append(np.nan)
            mean_age_difference.append(np.nan)

    result_df = pd.DataFrame({var: mean_age_difference, 'ratio': ratio, 'bin_up': bin_up, 'bin_low': bin_low})
    result_df = result_df.dropna()
    return result_df

# 定义移动平均函数
def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

# 定义不良妊娠结局的列表
outcomes = outcome_f

# 遍历每个不良妊娠结局
for outcome in outcomes:

    col1 = 'gestational_age_diff'
    target = outcome

    # 计算每个分桶中的患病比例
    result_age_difference = f2(col1, df, bins=50, target=target)

    result_age_difference = result_age_difference.sort_values(by=col1)
    x_data = result_age_difference[col1]
    y_data = result_age_difference['ratio']

    # 应用移动平均法进行平滑处理
    window_size = 3  # 可以根据需要调整窗口大小
    y_data_smoothed = moving_average(y_data, window_size)

    # 找到横坐标为 0 时的纵坐标值
    zero_index = (x_data >= 0).idxmax()
    zero_value = y_data_smoothed[zero_index]

    # 筛选出不小于 0 点纵坐标值的数据
    valid_indices = y_data_smoothed >= zero_value
    x_data = x_data[valid_indices]
    y_data_smoothed = y_data_smoothed[valid_indices]

    # 合并相邻纵坐标值相等的数据点
    new_x_data = []
    new_y_data = []
    i = 0
    while i < len(x_data):
        start_i = i
        while i + 1 < len(x_data) and y_data_smoothed.iloc[i] == y_data_smoothed.iloc[i + 1]:
            i = i + 1
        # 取该区间横坐标的平均值
        new_x = x_data.iloc[start_i:i + 1].mean()
        new_y = y_data_smoothed.iloc[start_i]
        new_x_data.append(new_x)
        new_y_data.append(new_y)
        i = i + 1
    x_data = pd.Series(new_x_data)
    y_data_smoothed = pd.Series(new_y_data)

    # 找出左侧最高点的索引
    left_max_index = 0
    for i in range(1, len(y_data_smoothed)):
        if y_data_smoothed.iloc[i] < y_data_smoothed.iloc[left_max_index]:
            break
        left_max_index = i

    # 找出右侧最高点的索引
    right_max_index = len(y_data_smoothed) - 1
    for i in range(len(y_data_smoothed) - 2, -1, -1):
        if y_data_smoothed.iloc[i] < y_data_smoothed.iloc[right_max_index]:
            break
        right_max_index = i

    # 计算增大和减小 5% 的数据
    y_data_increase = y_data_smoothed * 1.05
    y_data_decrease = y_data_smoothed * 0.95

    plt.rcParams.update({
        'font.size': 16,           # 全局字号
        'axes.labelsize': 18,      # xy轴标签
        'axes.titlesize': 20,      # 标题
        'xtick.labelsize': 16,     # x轴刻度
        'ytick.labelsize': 16,     # y轴刻度
        'legend.fontsize': 16      # 图例
    })

    # 使用 LOWESS 进行平滑
    lowess = sm.nonparametric.lowess(y_data_smoothed, x_data, frac=0.8)  # frac 控制平滑程度，可以根据需要调整
    x_lowess = lowess[:, 0]
    y_lowess = lowess[:, 1]

    # 计算增大和减小 5% 的数据
    y_lowess_increase = y_lowess * 1.05
    y_lowess_decrease = y_lowess * 0.95

    # 使用高斯平滑进一步平滑
    cs = CubicSpline(x_lowess, y_lowess)
    x_new = np.linspace(x_data.min(), x_data.max(), 1000)  # 增加点数
    y_new = cs(x_new)  # 计算对应的纵坐标值

    sigma = 3  # 增加高斯核的标准差
    y_gaussian = gaussian_filter1d(y_new, sigma=sigma)

    # 使用小波变换平滑
    wavelet = 'db4'  # 选择小波基
    level = 3  # 选择分解层数
    coeffs = pywt.wavedec(y_gaussian, wavelet, level=level)
    coeffs[-level] *= 0  # 将高频部分置零以平滑数据

    y_wavelet = pywt.waverec(coeffs, wavelet)

    # 截取有效部分（小波变换可能导致数据长度变化）
    y_wavelet = y_wavelet[:len(x_new)]

    # 计算增大和减小 5% 的数据
    y_wavelet_increase = y_wavelet * 1.05
    y_wavelet_decrease = y_wavelet * 0.95

    # 绘制图形
    fig = plt.figure(figsize=(12, 6))
    plt.plot(x_new, y_wavelet, color='blue', label='Smoothed Curve (Wavelet)')
    plt.plot(x_new, y_wavelet_increase, color='black', alpha=0.5, label='Upper Bound (+1 SD)', linestyle=':')
    plt.plot(x_new, y_wavelet_decrease, color='black', alpha=0.5, label='Lower Bound (-1 SD)', linestyle=':')
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.3)
    plt.xlabel('CTGage-gap')
    plt.ylabel('Ratio of Affected/Unaffected')
    plt.title(f'{outcome}')
    plt.savefig(f"{target_dir}main_{outcome}_smoothed.png")
    plt.show()
    plt.close(fig)
