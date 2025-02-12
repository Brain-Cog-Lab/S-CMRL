# -*- coding: utf-8 -*-            
# Time : 2022/11/12 17:40
# Author : Regulus
# FileName: dvs_acc_vis.py
# Explain:
# Software: PyCharm

import os
import sys

import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from tonic.datasets import NCALTECH101, CIFAR10DVS
from braincog.datasets.datasets import unpack_mix_param, DATA_DIR
from braincog.datasets.cut_mix import *
import tonic
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

import seaborn as sns
from matplotlib.pyplot import MultipleLocator
from pylab import *
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1,
                rc={"lines.linewidth": 2.5})

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
color_lib = sns.color_palette()

def extract_csv(file, type='main'):
    data = np.loadtxt(open(file, "rb"), delimiter=",", skiprows=1, usecols=[0, 3])
    data = data[50:100, :]
    return data[:, 0], data[:, 1]


def to_percent(temp, position):
    return '%2.f'%(temp) + '%'


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'NimbusRomNo9L'  # [NimbusRomNo9L, Times New Roman]
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # plt.rcParams['font.size'] = 16
    fig, ax = plt.subplots(figsize=(14, 7))  # (12, 6)
    dataset = 'CREMAD'  # [CEPDVS, NCALTECH101, omniglot]
    traindataratio = '1.0'
    # ax1 = ax.inset_axes([0.2, 0.3, 0.28, 0.22])
    # ax2 = ax.inset_axes([0.65, 0.3, 0.28, 0.22])
    ANN_VIS = False

    # # # # ------ plot additional ---------
    # epoch_list, acc_list = extract_csv("/home/hexiang/DomainAdaptation_DVS/Results/Baseline/VGG_SNN-NCALTECH101-10-seed_42-bs_120-DA_False-ls_0.0-lr_0.005-traindataratio_1.0-TET_loss_True-refined_True/"
    #                                    "summary.csv", type='main')
    # ax.plot(range(150, 299), acc_list, linewidth=2, label="Finetune training maximum accuracy: 79.43")
    # plt.xlabel('Training epochs', fontsize=44)
    # plt.ylabel('Accuracy (Test set)', fontsize=44)
    # plt.grid(which='major', axis='x', ls=':', lw=0.8, color='c', alpha=0.5)
    # plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.5)
    # # xticks([x for x in range(0, len(epoch_list))], ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))
    # plt.xticks(np.linspace(150, 300, 6))
    # # ax.xaxis.set_major_locator(MultipleLocator(1))
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    # # plt.axhline(79.66, color="r", linestyle="dashdot", label="Direct Training Best:79.66")
    # ax.plot(range(150, 299), [79.66] * 149, color='red', linestyle='dashdot', label='Direct training maximum accuracy: 79.66')
    # ax.legend(bbox_to_anchor=(1, 0), loc=4, fontsize=41)
    # ax.yaxis.set_major_locator(MultipleLocator(10))  # 设置y轴的主要刻度间隔
    # plt.tick_params(axis='both', which='major', labelsize=38)
    # # plt.text(350, 75, "Baseline:79.88", size=15, color="r",  weight="light", bbox=dict(facecolor="r", alpha=0.2))
    # plt.tight_layout()
    # plt.savefig('acc_finetune.svg', dpi=300)
    # sys.exit()
    #
    # # ------ plot block a ------------
    modality = "visual"
    major_locator = 8
    embed_dims = 256
    if modality == "visual":
        major_locator = 4
    legend_list = ['Baseline', "S-CMRL (ours)"]
    baseline_root = "/mnt/home/hexiang/S-CMRL/SNN/exp_results/"
    trainresults_root = "/mnt/home/hexiang/S-CMRL/SNN/exp_results_singleModality/"

    file_root = "spikformer-CREMAD-{}-interaction-Add-attn-method_Spatial-cross-attn_False-alpha_1.0-contrastive-False-temperature_0.1-snr_-100-embed_dims_{}-LIFNode-4/summary.csv".format(modality, embed_dims)

    load_root = None
    type_list = ["single", "multimodal"]

    show_epoch = 0

    for i in range(2):
        load_root = baseline_root if i == 0 else trainresults_root
        epoch_lists = []
        acc_lists = []
        file = os.path.join(load_root, file_root)

        epoch_list, acc_list = extract_csv(file, type=type_list[i])
        epoch_lists.append(epoch_list)
        acc_lists.append(acc_list)
        acc_mean = np.max(np.array(acc_lists), axis=1)
        acc_std = np.max(np.array(acc_lists), axis=1).std(0)
        print("for {}, acc max:{}, acc max mean:{} acc var:{}".format(legend_list[i], acc_mean, acc_mean.mean(), acc_std))
        # print("acc list:{}".format(np.max(np.array(acc_lists), axis=1)))

        acc_lists_mean_total = np.array(acc_lists).mean(0)[show_epoch:]
        acc_lists_std_total = np.array(acc_lists).std(0)[show_epoch:]

        ax.plot(range(1, len(acc_lists_mean_total) + 1), acc_lists_mean_total, linewidth=2, label=legend_list[i])

    ax.legend(bbox_to_anchor=(1, 0), loc=4, fontsize=26)
    plt.xlabel('Epochs', fontsize=32)
    plt.ylabel('Test Accuracy (Test set)', fontsize=32)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    # plt.show()
    plt.grid(which='major', axis='x', ls=':', lw=0.8, color='c', alpha=0.5)
    plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=32)

    plt.xticks(range(1, len(acc_lists_mean_total) + 2, 10), [str(i * 10 + 50) for i in range(6)])
    plt.tight_layout()
    ax.yaxis.set_major_locator(MultipleLocator(4))  # 设置y轴的主要刻度间隔
    plt.savefig('{}_{}.pdf'.format(dataset, modality), dpi=300)
    sys.exit()

# ------ plot block b ------------
way = "e"
if way == 'a':
    ncaltech101_acc_baseline = [44.48, 55.63, 64.37, 67.69, 70.46, 72.76, 75.86, 76.55, 78.39, 79.66]
    ncaltech101_acc_ours     = [58.50, 69.20, 75.51, 78.62, 81.83, 85.63, 87.36, 88.97, 91.95, 92.64]
    ax.plot(range(0, 10), ncaltech101_acc_baseline, '*', linewidth=2, label='baseline (TET)', linestyle="dashdot")
    ax.plot(range(0, 10), ncaltech101_acc_ours, 'o', linewidth=2, label='knowledge transfer strategy (Ours)', linestyle="dashdot")
    ax.legend(bbox_to_anchor=(1, 0), loc=4, fontsize=29)
    plt.xlabel('Ratio of training data', fontsize=29)
    plt.ylabel('Test Accuracy (Test set)', fontsize=29)
    plt.grid(which='major', axis='x', ls=':', lw=0.8, color='c', alpha=0.5)
    plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.5)
    xticks([x for x in range(0, 10)], ("10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"), fontsize=29)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(10))  # 设置y轴的主要刻度间隔
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.tick_params(axis='both', which='major', labelsize=29)
    plt.tight_layout()
    plt.savefig('ncaltech101_dataratio.svg', dpi=300)
elif way == 'b':
    ncaltech101_acc_baseline = [79.66, 79.66, 79.66, 79.66, 79.66, 79.66]
    ncaltech101_acc_ours     = [79.66, 80.00, 82.41, 86.20, 88.50, 92.64]  # base, 10%, 30%, 50%, 80%, 100%
    ax.plot(range(0, 6), ncaltech101_acc_baseline, '*', linewidth=2, label='Baseline (TET)', linestyle="dashdot")
    ax.plot(range(0, 6), ncaltech101_acc_ours, 'o', linewidth=2, label='Knowledge transfer strategy (Ours)', linestyle="dashdot")
    ax.set_ylim([78, 94])
    ax.legend(bbox_to_anchor=(0, 1), loc="upper left", fontsize=29)
    plt.xlabel('Rgb data amout ratio', fontsize=29)
    plt.ylabel('Test Accuracy (Test set)', fontsize=29)
    plt.grid(which='major', axis='x', ls=':', lw=0.8, color='c', alpha=0.5)
    plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.5)
    xticks([x for x in range(0, 6)], ("0%", "10%", "30%", "50%", "80%", "100%"))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.tick_params(axis='both', which='major', labelsize=29)
    plt.tight_layout()
    plt.savefig('ncaltech101_dvsdataratio.pdf', dpi=600)
elif way == 'c':
    x_data = ['CEP-DVS', 'N-Caltech101']
    ax_data = [28.9, 30.50]
    ax1_data = [90.57, 92.64]

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    ax.set_ylim([28, 31.5])
    ax.set_yticks = np.arange(28, 31.5)
    ax.set_yticklabels = np.arange(28, 31.5)

    bar_width = 0.25
    ax.set_ylabel('Test Accuracy (Test set)', fontsize=12)
    lns1 = ax.bar(x=np.array([0.0]), width=bar_width, height=ax_data[0], label='w/o v channel', color='steelblue', alpha=0.8)
    plt.text(0.0, ax_data[0] + 0.0005, '%s' % ax_data[0], ha='center', fontsize=12)
    lns2 = ax.bar(
        x=np.array([0.25]), width=bar_width, height=ax_data[1], label='w/ v channel', color='indianred', alpha=0.8)
    plt.text(0.25, ax_data[1] + 0.0005, '%s' % ax_data[1], ha='center', fontsize=12)



    ax1 = ax.twinx()  # this is the important function

    ax1.set_ylim([90, 93])
    ax1.set_yticks = np.arange(90, 93)
    ax1.set_yticklabels = np.arange(90, 93)
    ax1.set_ylabel('Test Accuracy (Test set)', fontsize=12);
    lns3 = ax1.bar(x=np.array([1.0]), width=bar_width, height=ax1_data[0],  fc='steelblue',
                   alpha=0.8)
    plt.text(1.0, ax1_data[0] + 0.0005, '%s' % ax1_data[0], ha='center', fontsize=12)
    lns3 = ax1.bar(x=np.array([1.25]), width=bar_width, height=ax1_data[1],  fc='indianred',
                   alpha=0.8)
    plt.text(1.25, ax1_data[1] + 0.0005, '%s' % ax1_data[1], ha='center', fontsize=12)

    plt.xticks(np.arange(len(x_data)) + bar_width / 2, x_data)
    plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.35)
    ax.set_xlabel('Datasets', fontsize=12)

    fig.legend(loc=1, bbox_to_anchor=(0.38, 1), bbox_transform=ax.transAxes, fontsize=10.5)
    plt.tight_layout()  # 超出边界加上的
    plt.savefig('v_channel.pdf', dpi=300)  # 图表输出
elif way == 'd':
    state_dict = torch.load("/home/hexiang/DomainAdaptation_DVS/Results2/train_DomainAdaptation/Transfer_VGG_SNN-NCALTECH101-10-bs_120-seed_42-DA_False-ls_0.0-domainLoss_True_coefficient0.5-traindataratio_1.0-rgbdataratio_1.0-TET_loss_True-hsv_True-sl_True-regularization_True/model_best.pth.tar",
               map_location='cpu')['state_dict']
    coeffi = []
    for key in state_dict:
        if key.startswith("coefficient."):
            coeffi.append(state_dict[key])
    coeffi_top     = [1.0] * 10
    coeffi_mid = [0.5] * 10
    coeffi_bottom = [0.0] * 10
    line1, = ax.plot(range(1, 11), torch.sigmoid(torch.tensor(coeffi)), linewidth=2, color='r', label='Spatio-Temporal Regularization', linestyle="dashdot")
    line2, = ax.plot(range(1, 11), coeffi_top, linewidth=2, color='c', label='Fixed coefficient at 1.0', linestyle="dashdot")
    line3, = ax.plot(range(1, 11), coeffi_mid, linewidth=2, color='c', linestyle="dashdot")
    line4, = ax.plot(range(1, 11), coeffi_bottom, linewidth=2, color='c', linestyle="dashdot")
    ax.set_ylim([-0.1, 1.1])
    # 设置图例
    custom_lines = [line1, line2]
    custom_labels = ['STR', "Fixed coefficient"]
    ax.legend(custom_lines, custom_labels, bbox_to_anchor=(1, 0.16), loc=4, fontsize=26)
    for line, label in [(line1, "92.64"), (line2, "89.31"), (line3, "89.43"), (line4, "84.13")]:
        xdata, ydata = line.get_data()
        if label == "92.64":
            plt.annotate(label, (xdata[-1], ydata[-1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=28, fontweight='bold')
        else:
            plt.annotate(label, (xdata[-1], ydata[-1]), textcoords="offset points", xytext=(0, 5), ha='center',
                         fontsize=28,)
    plt.xlabel('Time step', fontsize=32)
    plt.ylabel('Coefficient $\sigma(\eta_t)$ Value', fontsize=32)
    plt.grid(which='major', axis='x', ls=':', lw=0.8, color='c', alpha=0.5)
    plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.tick_params(axis='both', which='major', labelsize=32)
    plt.tight_layout()
    plt.savefig('coeffi.pdf', dpi=300)  # 图表输出