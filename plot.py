import os
import numpy as np
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter1d

import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.pyplot import MultipleLocator
from pylab import *
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1,
                rc={"lines.linewidth": 2.5})

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
color_lib = sns.color_palette()

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16


def to_percent(temp, position):
    return '%2.f'%(temp) + '%'

way = 'a'  # way 是三种图
dataset = "CREMA-D"  # CREMA-D


# --------box plot--------#
if way == 'a':
    # 设置seaborn样式
    sns.set(style="whitegrid")

    # 数据, seed 42 47 1024; meta ratio 0.0, 0.5, 1.0, 1.5, 2.0
    data = [[69.62, 70.83, 71.77],
            [70.97, 70.43, 71.63],
            [71.24, 71.77, 72.04],
            [72.72, 71.24, 72.98],
            [72.31, 72.45, 71.91]]

    # 设置箱线图颜色，并调整亮度
    colors = ['gray', '#FFD47F', '#F7C1CF', 'salmon', 'cyan']

    # 创建箱线图
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=np.transpose(data), palette=colors, width=0.7)

    # 设置x轴和y轴标签
    ax.set_xticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'])
    ax.set_xlabel('Hyperparameter Setting', fontsize=14, labelpad=10)
    ax.set_ylabel('Model Performance', fontsize=14, labelpad=10)

    # 设置标题
    # ax.set_title('Model Performance Distribution for Different Hyperparameter Settings', fontsize=16, pad=20)

    # 显示网格
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

    # 增加背景线条
    for i in np.arange(69, 74, 2):
        ax.axhline(i, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

    # 设置刻度标签的大小
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 显示图表
    plt.tight_layout()
    # plt.show()
    plt.savefig("hyperparameter.pdf", dpi=300)

