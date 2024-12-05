import os
import csv
import shutil
from tqdm import tqdm
import numpy as np
import random

# 数据路径
data_root = "./data/"
output_root = "/home/hexiang/data/CREMA-D/processed_dataset"

# 创建输出目录结构
categories = ['NEU', 'HAP', 'SAD', 'FEA', 'DIS', 'ANG']
for split in ['train', 'test']:
    for modality in ['audio', 'visual']:
        for category in categories:
            os.makedirs(os.path.join(output_root, split, modality, category), exist_ok=True)


# 读取并处理数据集
def process_dataset(args, mode='train'):
    image = []
    audio = []
    label = []

    visual_feature_path = args.visual_path
    audio_feature_path = args.audio_path

    # 设置 CSV 文件路径
    train_csv = os.path.join(data_root, args.dataset + '/train.csv')
    test_csv = os.path.join(data_root, args.dataset + '/test.csv')
    csv_file = train_csv if mode == 'train' else test_csv

    # 读取 CSV 文件并获取样本路径和标签
    class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}
    with open(csv_file, encoding='UTF-8-sig') as f2:
        csv_reader = csv.reader(f2)
        for item in csv_reader:
            audio_path = os.path.join(audio_feature_path, item[0] + '.wav')
            visual_path = os.path.join(visual_feature_path, 'Image-{:02d}-FPS'.format(args.fps), item[0])
            label_name = item[1]

            if os.path.exists(audio_path) and os.path.exists(visual_path):
                image.append(visual_path)
                audio.append(audio_path)
                label.append(label_name)
            else:
                print("{}: exist_{}".format(audio_path, os.path.exists(audio_path)))
                print("{}: exist_{}".format(visual_path, os.path.exists(visual_path)))

    # 处理数据并复制到新的目录结构中
    for idx in tqdm(range(len(label))):
        category = label[idx]
        visual_file = image[idx]
        audio_file = audio[idx]

        # 创建类别目录
        visual_category_dir = os.path.join(output_root, mode, "visual", category)
        audio_category_dir = os.path.join(output_root, mode, "audio", category)

        # 获取目标文件路径
        visual_filename = os.path.basename(visual_file) + ".jpg"
        audio_filename = os.path.basename(audio_file)
        dst_visual = os.path.join(visual_category_dir, visual_filename)
        dst_audio = os.path.join(audio_category_dir, audio_filename)

        visual_file = os.path.join(visual_file, os.listdir(visual_file)[random.randint(0, len(os.listdir(visual_file)) - 1)])

        # 复制文件
        shutil.copy2(visual_file, dst_visual)
        shutil.copy2(audio_file, dst_audio)

# 执行处理
class Args:
    dataset = 'CREMAD'
    visual_path = "/mnt/home/hexiang/datasets/CREMA-D/"
    audio_path = "/mnt/home/hexiang/datasets/CREMA-D/AudioWAV/"
    fps = 1

args = Args()
process_dataset(args, mode='train')
process_dataset(args, mode='test')

print("\n数据处理完成！")
