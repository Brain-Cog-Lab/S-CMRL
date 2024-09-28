import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_ssil import IcreLoader, exemplarLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from model.audio_visual_model_incremental import IncreAudioVisualNet
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np
from datetime import datetime
import random
from itertools import cycle
import h5py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


os.environ["OMP_NUM_THREADS"] = "20"  # 设置OpenMP计算库的线程数
os.environ["MKL_NUM_THREADS"] = "20"  # 设置MKL-DNN CPU加速库的线程数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AVE', choices=['AVE', 'ksounds', 'VGGSound_100'])
parser.add_argument('--modality', type=str, default='visual', choices=['visual', 'audio', 'audio-visual'])
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--infer_batch_size', type=int, default=32)
parser.add_argument('--exemplar_batch_size', type=int, default=128)

parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--max_epoches', type=int, default=500)
parser.add_argument('--num_classes', type=int, default=28)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=boolean_string, default=False)
parser.add_argument("--milestones", type=int, default=[500], nargs='+', help="")
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--class_num_per_step', type=int, default=7)

parser.add_argument('--memory_size', type=int, default=340)

args = parser.parse_args()


def CE_loss(num_classes, logits, label):
    targets = F.one_hot(label, num_classes=num_classes)
    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

    return loss


def top_1_acc(logits, task_num):
    top1_res = logits.argmax(dim=1)
    count = torch.bincount(top1_res, minlength=args.class_num_per_step*task_num)

    cnt_list = []
    for i in range(task_num):
        cnt_list.append(count[i*args.class_num_per_step: (i+1)*args.class_num_per_step].sum())

    return cnt_list




def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if dataset.args.dataset == 'AVE':
        dataset.all_visual_pretrained_features = np.load(dataset.visual_pretrained_feature_path,
                                                         allow_pickle=True).item()
    else:
        dataset.all_visual_pretrained_features = h5py.File(dataset.visual_pretrained_feature_path, 'r')
    dataset.all_audio_pretrained_features = np.load(dataset.audio_pretrained_feature_path, allow_pickle=True).item()


def detailed_test(args, step, test_data_set, task_best_acc_list):
    print("=====================================")
    print("Start testing...")
    print("=====================================")
    # model = IncreAudioVisualNet(args=args)

    model = torch.load(
        './save/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, step-1, args.modality))

    model.to(device)

    test_loader = DataLoader(test_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)

    all_test_out_logits = torch.Tensor([])
    all_test_labels = torch.Tensor([])
    model.eval()
    with torch.no_grad():
        for test_data, test_labels in tqdm(test_loader):
            if args.modality == 'visual':
                test_visual = test_data
                test_visual = test_visual.to(device)
                test_out_logits = model(visual=test_visual)
            elif args.modality == 'audio':
                test_audio = test_data
                test_audio = test_audio.to(device)
                test_out_logits = model(audio=test_audio)
            else:
                test_visual = test_data[0]
                test_audio = test_data[1]
                test_visual = test_visual.to(device)
                test_audio = test_audio.to(device)
                test_out_logits = model(visual=test_visual, audio=test_audio)
            test_out_logits = F.softmax(test_out_logits, dim=-1).detach().cpu()
            all_test_out_logits = torch.cat((all_test_out_logits, test_out_logits), dim=0)
            all_test_labels = torch.cat((all_test_labels, test_labels), dim=0)
    test_top1 = top_1_acc(all_test_out_logits, step)

    return test_top1


if __name__ == "__main__":
    print(args)

    total_incremental_steps = args.num_classes // args.class_num_per_step

    setup_seed(args.seed)

    print('Training start time: {}'.format(datetime.now()))

    test_set = IcreLoader(args=args, mode='test', modality=args.modality)

    exemplar_set = exemplarLoader(args=args, modality=args.modality)

    ckpts_root = './save/{}/{}/'.format(args.dataset, args.modality)
    figs_root = './save/fig/{}/{}/'.format(args.dataset, args.modality)

    if not os.path.exists(ckpts_root):
        os.makedirs(ckpts_root)
    if not os.path.exists(figs_root):
        os.makedirs(figs_root)

    task_best_acc_list = []

    step_forgetting_list = []

    step_accuracy_list = []

    exemplar_class_vids = None
    for step in range(total_incremental_steps):
        if step < 10:
            continue
        test_set.set_incremental_step(step)

        # exemplar_set._set_incremental_step_(step)

        print("***************New Step***************************")
        print('Incremental step: {}'.format(step))

        # train(args, step, train_set, val_set, exemplar_set)
        step_ratio = detailed_test(args, step, test_set, task_best_acc_list)



    import matplotlib.pyplot as plt
    import numpy as np

    modality = "na"
    # 数据
    groups = [['a', 'b'], ['a', 'b', 'c'], ['a', 'b', 'c', 'd']]
    if modality == "audio":
        values = [[565, 589], [566, 490, 489], [488, 440, 601, 429]]
    elif modality == "visual":
        values = [[687, 467], [515, 562, 468], [427, 429, 620, 482]]
    elif modality == "audio-visual":
        values = [[652, 502], [504, 526, 515], [441, 449, 497, 571]]
    else:
        values = [[687+565, 467+589], [515+566, 562+490, 468+489], [427+488, 429+440, 620+601, 482+429]]
    proportions = []
    for group in values:
        total = sum(group)  # 计算每个小组的总和
        proportion = [value / total for value in group]  # 计算每个值的比例
        proportions.append(proportion)  # 将结果添加到列表中

    values = proportions
    # 颜色
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

    fig, ax = plt.subplots(figsize=(10, 8))

    # 堆叠柱状图的位置
    x_positions = np.arange(len(groups))  # 每个组的 X 轴位置

    # 每个组数据的宽度
    width = 0.2

    # 用来跟踪哪些标签已经被加入到图例中
    legend_added = []

    # 绘制每个组的堆叠柱状图
    for idx, (group, group_values) in enumerate(zip(groups, values)):
        cumulative = np.zeros(len(group_values))
        for i in range(len(group_values)):
            label = group[i]
            if label not in legend_added:
                ax.bar(x_positions[idx], group_values[i], width, bottom=cumulative[i],
                       color=colors[i % len(colors)], label=label)
                legend_added.append(label)
            else:
                ax.bar(x_positions[idx], group_values[i], width, bottom=cumulative[i],
                       color=colors[i % len(colors)])
            # 更新累积和以添加到下一个柱子
            if i < len(group_values) - 1:
                cumulative[i + 1] = cumulative[i] + group_values[i]

    # 设置 X 轴标签
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Group 1', 'Group 2', 'Group 3'])

    # 添加图例
    ax.legend(title='Labels', bbox_to_anchor=(1, 1))

    # 设置轴标签
    ax.set_xlabel('Groups')
    ax.set_ylabel('Values')

    # 显示图表
    plt.show()
