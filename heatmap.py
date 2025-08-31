import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 指定CSV文件的路径
file_path = '/data/gjs/AIOG/FHRage/model/20250729_231225/all_combined.csv'  # 替换为你的CSV文件路径

# 定义新的列名列表
new_column_names = ['name','prediction','true_value','difference','死胎','流产','减胎','胎停','胎死宫内','分娩方式',
                    '1min评分','5min评分','10min评分','HDP', 'GDM', 'thyroid disease', 'anemia', 'immune disease', 
                    'pregnancy with hysteromyoma', 'maternal congenital disease', 'oligohydramnios',' polyhydramnios',
                    'uterine malformations',' umbilical cord problems', 'placental pathology', 
                    'premature infants', 'low birth weight infants','neonatal asphyxia',
                    'fetal distress', 'giant infants',' fetal malformations', 'chromosomal malformations',
                    ' neonatal congenital heart disease ',' neonatal congenital renal cysts', 'year'] 

outcome_m = ['HDP', 'GDM', 'thyroid disease', 'anemia', 'immune disease', 'pregnancy with hysteromyoma', 'maternal congenital disease', 'oligohydramnios',' polyhydramnios', 'uterine malformations',' umbilical cord problems', 'placental pathology']

outcome_f = ['premature infants', 'neonatal asphyxia', 'fetal distress']

# 使用pandas的read_excel函数读取文件
df = pd.read_csv(file_path, header=0, names=new_column_names)
    

outcome = outcome_f

# 过滤数据范围
df = df[(df['true_value'] >= 35*7) & (df['true_value'] <= 42*7)]  # 35~42周

diff_bins = [-float('inf'), -21, -14, -7, 0, 7, 14, 21, float('inf')]
diff_labels = ['<-21', '-21~-14', '-14~-7', '-7~0', '0~7', '7~14', '14~21', '>21']
df['diff_week_group'] = pd.cut(df['difference'], bins=diff_bins, labels=diff_labels)

# 创建以1周为单位的真实孕龄分组（Y轴）
true_weeks = np.arange(35, 41)  # [35, 36, ..., 42]
df['true_week_group'] = pd.cut(df['true_value'] / 7, bins=true_weeks.tolist() + [43], labels=[f'{w}' for w in true_weeks])


plt.rcParams.update({
    'font.size': 14,           # 全局字号
    'axes.labelsize': 16,      # xy轴标签
    'axes.titlesize': 18,      # 标题
    'xtick.labelsize': 14,     # x轴刻度
    'ytick.labelsize': 14,     # y轴刻度
    'legend.fontsize': 14      # 图例
})

for outcome in outcome_f:
    # 计算占比：透视表用 mean，得到 0-1 的比例
    pivot_ratio = (df.pivot_table(
        index='diff_week_group',
        columns='true_week_group',
        values=outcome,
        aggfunc='mean',
        fill_value=0
    ) * 100)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        pivot_ratio,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        cbar_kws={'label': 'Proportion (%)'}  # 色条标题
    )
    plt.title(f'Proportion of {outcome}', fontsize=16)
    plt.xlabel('True CTGage (weeks)', fontsize=14)
    plt.ylabel('CTGage-gap (days)', fontsize=14)

    # 保存并关闭
    save_path = f'/data/gjs/AIOG/FHRage/model/20250729_231225/photo/heatmap/heatmap_{outcome}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()