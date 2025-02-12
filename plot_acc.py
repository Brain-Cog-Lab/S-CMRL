import os
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
import pandas as pd
from matplotlib.ticker import FuncFormatter
import seaborn as sns


def to_percent(temp, position):
    return '%.2f'%(temp) + '%'

# 假设你的结果目录为当前工作目录下的results文件夹
base_dir = "/mnt/home/hexiang/MCF/SNN/exp_results_snr/"

# 定义你要对比的四种方法及其名称标识
methods = {
    "WeightAttention": "method_WeightAttention",
    "SCA": "method_SCA",
    "CMCI": "method_CMCI",
    "SpatialTemporal": "method_SpatialTemporal"
}

method_labels = {
    "WeightAttention": "WeightAttention",
    "SCA": "SCA",
    "CMCI": "CMCI",
    "SpatialTemporal": "S-CMRL (Ours)",
}

# 定义需要对比的SNR列表
snr_list = [0, 5, 10, 15, 20, 25, 30]

# 为每种方法保存对应SNR的准确率
method_accuracies = {m: [] for m in methods}


# 解析log.txt文件的函数
def parse_accuracy_from_log(log_file_path):
    # 假设log.txt的最后一行包含准确率信息，比如："Final Accuracy: 87.5"
    # 可根据实际情况修改解析方式
    if not os.path.exists(log_file_path):
        return None
    with open(log_file_path, 'r') as f:
        lines = f.read().strip().split('\n')
    last_line = lines[-1]

    # 尝试从最后一行解析数字,假设最后一行包含"Accuracy: x.xx"
    # 如果是纯数字也可以使用 float(last_line)
    # 假设格式为 "Final Accuracy: 87.5"
    match = re.search(r'Best metric:\s+([\d\.]+)', last_line)
    if match:
        acc_str = match.group(1)
        accuracy = float(acc_str)
        return accuracy
    else:
        # 根据实际情况修改解析
        return None


# 遍历每种方法和SNR，提取准确率
for method_name, method_flag in methods.items():
    for snr in snr_list:
        # 寻找匹配该方法与snr的目录
        # 目录结构中包含 "snr_-10", "snr_0", "snr_20", "snr_30"
        # 且包含对应method_flag，比如 "method_CMCI"
        # 由于你的目录结构很复杂，这里提供一个可能的匹配方式
        # 假设结果目录中每个实验在单独文件夹中
        # 我们可以扫描base_dir下的所有文件夹，找到同时包含method_flag与"snr_SNR"的文件夹名
        exp_dir = None
        for d in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, d)):
                # 根据命名特征检查
                # 例如: d中包含 method_CMCI 并包含 snr_-10
                # 注意 -10 需要转为字符串并匹配"snr_-10"
                snr_str = f"snr_{snr}"
                if method_flag in d and snr_str in d:
                    if method_flag == "method_SpatialTemporal":
                        if "temperature_0.07" in d:
                            exp_dir = os.path.join(base_dir, d)
                            break
                    else:
                        if "temperature_0.1" in d:
                            exp_dir = os.path.join(base_dir, d)
                            break


        if exp_dir is None:
            # 如果找不到对应目录,则记录None或0
            method_accuracies[method_name].append(None)
            continue

        # 假设log.txt文件位于exp_dir中
        log_file = os.path.join(exp_dir, "log.txt")
        accuracy = parse_accuracy_from_log(log_file)
        method_accuracies[method_name].append(accuracy)

# 开始绘图
plt.figure(figsize=(8, 5))
for method_name in methods:
    plt.plot(snr_list, method_accuracies[method_name], marker='o', label=method_labels[method_name])

plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
# plt.title("Accuracy Comparison under Different SNR Levels")

# X轴如果需要只显示整数
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))

plt.grid(True)
plt.legend()

# 保存为PDF文件，指定dpi
output_pdf_path = "figs/accuracy_comparison_snr.pdf"
plt.savefig(output_pdf_path, format='pdf', dpi=300)
plt.show()


# # -------------------画res图的-------------------
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter
#
# def to_percent(temp, position):
#     """将数值格式化为百分比字符串"""
#     return '%.2f%%' % temp
#
# # 原始数据
# crmea_without = 66.53
# crmea_with = 71.64
# audio_acc = 65.86
# visual_acc = 43.15
#
#
# sns.set_style('whitegrid')
# sns.set_palette('muted')
#
# # 创建画布 + 两个纵坐标
# fig, ax1 = plt.subplots(figsize=(6,4))
#
# # 为了让每个数据集在 x 轴上各占一个位置：
# x_crmea = 0
# x_urban = 0.6
# width = 0.125  # 两个柱子在同一个 x 上左右偏移
# bar_width = 0.07
#
# # -------------------------
# # 1) 在左轴(ax1)绘制 CRMEA‐D
# # -------------------------
# ax1.bar(x_crmea, visual_acc,
#         width=bar_width, color='#7B92C7')
# ax1.bar(x_crmea + width, audio_acc,
#         width=bar_width, color='#ADD9EE')
# ax1.bar(x_crmea + width * 2, crmea_without,
#         width=bar_width, color='#F7C1CF')
# ax1.bar(x_crmea + width * 3, crmea_with,
#         width=bar_width, color='#FFD47F')
#
# # Adding value labels on top of each bar
# for i, value in enumerate([visual_acc, audio_acc, crmea_without, crmea_with]):
#     ax1.text(x_crmea + width * i, value + 0.5, f'{value}%', ha='center')
#
# ax1.set_ylim([30, 80])  # 仅供示例，实际可自行调节
# ax1.set_ylabel("Accuracy (%)")  # 左轴标签
# ax1.set_xlabel("Modality")  # 左轴标签
# # -------------------------
# # 设置 X 轴刻度与标签
# # -------------------------
# ax1.set_xticks([x_crmea, x_crmea + width * 1, x_crmea + width * 2, x_crmea + width * 3.1])
# ax1.set_xticklabels(['Visual', 'Audio', 'Audio-Visual', 'Audio-Visual w/ CMRL'])
#
# # # 如果希望只显示一份图例，可以手动合并，或只在 ax1 上显示
# # handles1, labels1 = ax1.get_legend_handles_labels()
# # # 上下两组其实是相同颜色/含义，这里只取一组即可
# # ax1.legend(handles1, labels1, loc='upper left')
# plt.subplots_adjust(bottom=0.2)
# plt.tight_layout()
# # plt.show()
# plt.savefig('withres.pdf', dpi=300)