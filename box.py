import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 指定CSV文件的路径
file_path = '/data/gjs/AIOG/FHRage/model/20250729_231225/all_combined.csv'  # 替换为你的CSV文件路径

# 定义新的列名列表
new_column_names = ['name','prediction','true_value','difference','死胎','流产','减胎','胎停','胎死宫内','分娩方式',
                    '1min评分','5min评分','10min评分','HDP', 'GDM', 'thyroid disease', 'anemia', 'immune disease', 
                    'pregnancy with hysteromyoma', 'maternal congenital disease', 'oligohydramnios','polyhydramnios',
                    'uterine malformations','umbilical cord problems', 'placental lesions', 
                    'premature infants suitable for gestational age', 'low birth weight infants',' neonatal asphyxia ',
                    ' fetal distress', 'giant infants',' fetal malformations', 'chromosomal malformations',
                    ' neonatal congenital heart disease ',' neonatal congenital renal cysts', 'year'] 

outcome_m = ['GDM', 'anemia', 'maternal congenital disease', 'polyhydramnios', 'umbilical cord problems', 'placental lesions']

# 使用pandas的read_excel函数读取文件
df = pd.read_csv(file_path, header=0, names=new_column_names)
    

# 将患病值为 -1 的样本赋值为 0
for disease in outcome_m:
    df[disease] = df[disease].replace(-1, 0)

# 将数据转换为长格式
df_long = pd.melt(df, id_vars=['difference'], value_vars=outcome_m, var_name='disease', value_name='status')

# 修改 status 列的值
df_long['status'] = df_long['status'].map({0: 'Normal', 1: 'Diseased'})

plt.rcParams.update({
    'font.size': 14,           # 全局字号
    'axes.labelsize': 16,      # xy轴标签
    'axes.titlesize': 18,      # 标题
    'xtick.labelsize': 14,     # x轴刻度
    'ytick.labelsize': 14,     # y轴刻度
    'legend.fontsize': 14      # 图例
})

# 创建一个画布
plt.figure(figsize=(10, 8))

# 绘制箱型图
sns.boxplot(x='disease', y='difference', hue='status', data=df_long, palette='pastel')

# 设置标题和标签
plt.title('Boxplot of CTGage-gap by Maternal Diseases')
plt.xlabel('Maternal Diseases')
plt.ylabel('CTGage-gap')
plt.xticks(rotation=45)  # 旋转x轴标签，避免重叠

# 显示图例
plt.legend(title='Status', loc='upper right')

# 显示图形
plt.tight_layout()
plt.savefig('/data/gjs/AIOG/FHRage/model/20250729_231225/photo/box.png')

# 提取每个疾病状态下的difference的统计信息
summary_stats = df_long.groupby(['disease', 'status'])['difference'].describe()
print("Summary Statistics for Boxplot Data:")
print(summary_stats)