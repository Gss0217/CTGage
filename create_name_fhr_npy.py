import os
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm

# 定义文件路径
base_dir = '/data/gjs/AIOG/FHRage/model/20250729_231225/'
csv_file = f'{base_dir}all_combined.csv'
target_directory = '/data/gjs/AIOG/PK_People_Hospital/taixin/'  # 替换为目标目录路径
output_directory = base_dir  # 替换为输出目录路径

# 读取.csv文件
df = pd.read_csv(csv_file)

# 定义读取文本文件并转换为Pandas DataFrame的函数
def read_txt(file_name):
    with open(file_name, 'r') as file:
        data = file.read()
    # 使用ast.literal_eval将字符串解析为Python对象
    data_list = ast.literal_eval(data)
    # 将解析后的数据转换为DataFrame
    return pd.DataFrame(data_list)

# 初始化一个空的列表，用于存储所有样本的整合数据
all_combined_data = []

# 使用 tqdm 显示进度条
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
    file_name = row['name']  # 获取文件名
    gap = row['difference'] # 获取孕龄差
    heart_rate_file = os.path.join(target_directory, f'{file_name}.txt')  # 假设心率数据文件名与文件名相同，扩展名为.txt

    # 检查心率数据文件是否存在
    if os.path.exists(heart_rate_file):
        # 读取心率数据
        try:
            df_heart_rate = read_txt(heart_rate_file)
            # 提取'y'列的值并转换为NumPy数组
            heart_rate_data = np.array(df_heart_rate['y'])
        except Exception as e:
            print(f"Error processing file {heart_rate_file}: {e}")
            continue

        # 将疾病信息和心率数据组合
        combined_data = np.concatenate((np.array([file_name]), np.array([gap]),heart_rate_data))

        # 将整合后的数据添加到列表中
        all_combined_data.append(combined_data)

#         print(f'Processed {file_name}')
    else:
        print(f'Heart rate file not found for {file_name}')

# 将所有样本的整合数据保存为.npy文件
output_file = os.path.join(output_directory, 'all_test_name_gap_fhr.npy')
np.save(output_file, all_combined_data)
print(len(all_combined_data))
print(f'All data has been processed and saved to {output_file}')