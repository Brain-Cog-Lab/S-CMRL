import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_lwf import IcreLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from model.audio_visual_model_incremental import IncreAudioVisualNet
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np
from datetime import datetime
import random
import h5py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Prepare_logger(args):
    import logging
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(0)
    logger.addHandler(handler)

    # 指定要创建的文件夹路径
    folder_path = './save/{}/{}'.format(args.dataset, args.modality)

    # 如果文件夹不存在，则创建文件夹
    os.makedirs(folder_path, exist_ok=True)

    logfile = './save/{}/{}/modality_{}.log'.format(args.dataset, args.modality, args.modality)
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

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
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--max_epoches', type=int, default=500)
parser.add_argument('--num_classes', type=int, default=28)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=boolean_string, default=False)
parser.add_argument("--milestones", type=int, default=[500], nargs='+', help="")
parser.add_argument('--seed', type=int, default=2024)

parser.add_argument('--class_num_per_step', type=int, default=7)

args = parser.parse_args()

logger = Prepare_logger(args)

def CE_loss(num_classes, logits, label):
    targets = F.one_hot(label, num_classes=num_classes)

    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

    return loss

def top_1_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()

def adjust_learning_rate(args, optimizer, epoch):
    miles_list = np.array(args.milestones) - 1
    if epoch in miles_list:
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.1
        logger.info('Reduce lr from {} to {}'.format(current_lr, new_lr))
        for param_group in optimizer.param_groups: 
            param_group['lr'] = new_lr

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if dataset.args.dataset == 'AVE':
        dataset.all_visual_pretrained_features = np.load(dataset.visual_pretrained_feature_path, allow_pickle=True).item()
    else:
        dataset.all_visual_pretrained_features = h5py.File(dataset.visual_pretrained_feature_path, 'r')
    dataset.all_audio_pretrained_features = np.load(dataset.audio_pretrained_feature_path, allow_pickle=True).item()


def train(args, step, train_data_set, val_data_set):
    T = 2

    train_loader = DataLoader(train_data_set, batch_size=args.train_batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True, persistent_workers=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False, persistent_workers=True, worker_init_fn=worker_init_fn)
    
    step_out_class_num = (step + 1) * args.class_num_per_step
    if step == 0:
        model = IncreAudioVisualNet(args, step_out_class_num)
    else:
        model = torch.load('./save/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, step-1, args.modality))
        model.incremental_classifier(step_out_class_num)
        old_model = torch.load('./save/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, step-1, args.modality))
        last_step_out_class_num = step * args.class_num_per_step

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if step != 0:
            old_model = nn.DataParallel(old_model)
    
    model = model.to(device)
    if step != 0:
        old_model = old_model.to(device)
        old_model.eval()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0
    for epoch in range(args.max_epoches):
        train_loss = 0.0
        num_steps = 0
        model.train()

        iterator = tqdm(train_loader)
        
        for samples in iterator:
            data, labels = samples
            labels = labels.to(device)
            if args.modality == 'visual':
                visual = data
                visual = visual.to(device)
                out = model(visual=visual)
            elif args.modality == 'audio':
                audio = data
                audio = audio.to(device)
                out = model(audio=audio)
            else:
                visual = data[0]
                audio = data[1]
                visual = visual.to(device)
                audio = audio.to(device)
                out = model(visual=visual, audio=audio)

            if step == 0:
                loss = CE_loss(step_out_class_num, out, labels)
            else:
                with torch.no_grad():
                    if args.modality == 'visual':
                        old_out = old_model(visual=visual).detach()
                    elif args.modality == 'audio':
                        old_out = old_model(audio=audio).detach()
                    else:
                        old_out = old_model(visual=visual, audio=audio).detach()

                loss_KD = torch.zeros(step).to(device)
                
                for t in range(step):
                    start = t * args.class_num_per_step
                    end = (t + 1) * args.class_num_per_step

                    soft_target = F.softmax(old_out[:, start:end] / T, dim=1)
                    output_log = F.log_softmax(out[:, start:end] / T, dim=1)
                    loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                loss_KD = loss_KD.sum()
                
                out = out[:, last_step_out_class_num:]
                labels = labels % args.class_num_per_step
                # labels = labels.to(device)
                ce_loss_value = CE_loss(args.class_num_per_step, out, labels)
    
                loss = loss_KD + ce_loss_value
            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1
        train_loss /= num_steps
        train_loss_list.append(train_loss)
        logger.info('Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss))

        all_val_out_logits = torch.Tensor([])
        all_val_labels = torch.Tensor([])
        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader):
                # val_labels = val_labels.to(device)
                if args.modality == 'visual':
                    val_visual = val_data
                    val_visual = val_visual.to(device)
                    if torch.cuda.device_count() > 1:
                        val_out_logits = model.module.forward(visual=val_visual)
                    else:
                        val_out_logits = model(visual=val_visual)
                elif args.modality == 'audio':
                    val_audio = val_data
                    val_audio = val_audio.to(device)
                    if torch.cuda.device_count() > 1:
                        val_out_logits = model.module.forward(audio=val_audio)
                    else:
                        val_out_logits = model(audio=val_audio)
                else:
                    val_visual = val_data[0]
                    val_audio = val_data[1]
                    val_visual = val_visual.to(device)
                    val_audio = val_audio.to(device)
                    if torch.cuda.device_count() > 1:
                        val_out_logits = model.module.forward(visual=val_visual, audio=val_audio)
                    else:
                        val_out_logits = model(visual=val_visual, audio=val_audio)
                val_out_logits = F.softmax(val_out_logits, dim=-1).detach().cpu()
                all_val_out_logits = torch.cat((all_val_out_logits, val_out_logits), dim=0)
                all_val_labels = torch.cat((all_val_labels, val_labels), dim=0)
        val_top1 = top_1_acc(all_val_out_logits, all_val_labels)
        val_acc_list.append(val_top1)
        logger.info('Epoch:{} val_res:{:.6f} '.format(epoch, val_top1))

        if val_top1 > best_val_res:
            best_val_res = val_top1
            logger.info('Saving best model at Epoch {}'.format(epoch))
            if torch.cuda.device_count() > 1:
                torch.save(model.module, './save/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, step, args.modality))
            else:
                torch.save(model, './save/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, step, args.modality))

        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig('./save/fig/{}/{}/{}_train_loss_step_{}.png'.format(args.dataset, args.modality, args.modality, step))
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label='val_acc')
        plt.legend()
        plt.savefig('./save/fig/{}/{}/{}_val_acc_step_{}.png'.format(args.dataset, args.modality, args.modality, step))
        plt.close()


        if args.lr_decay:
            adjust_learning_rate(args, opt, epoch)


def detailed_test(args, step, test_data_set, task_best_acc_list):
    logger.info("=====================================")
    logger.info("Start testing...")
    logger.info("=====================================")

    model = torch.load('./save/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, step, args.modality))
    model.to(device)

    test_loader = DataLoader(test_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)
    
    all_test_out_logits = torch.Tensor([])
    all_test_labels = torch.Tensor([])
    model.eval()
    with torch.no_grad():
        for test_data, test_labels in tqdm(test_loader):
            # test_labels = test_labels.to(device)
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
    test_top1 = top_1_acc(all_test_out_logits, all_test_labels)
    logger.info("Incremental step {} Testing res: {:.6f}".format(step, test_top1))
    
    old_task_acc_list = []
    mean_old_task_acc = 0.

    for i in range(step+1):
        step_class_list = range(i*args.class_num_per_step, (i+1)*args.class_num_per_step)
        step_class_idxs = []
        for c in step_class_list:
            idxs = np.where(all_test_labels.numpy() == c)[0].tolist()
            step_class_idxs += idxs
        step_class_idxs = np.array(step_class_idxs)
        i_labels = torch.Tensor(all_test_labels.numpy()[step_class_idxs])
        i_logits = torch.Tensor(all_test_out_logits.numpy()[step_class_idxs])
        i_acc = top_1_acc(i_logits, i_labels)
        mean_old_task_acc += i_acc
        if i == step:
            curren_step_acc = i_acc
        else:
            old_task_acc_list.append(i_acc)
    if step > 0:
        forgetting = np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list))
        logger.info('forgetting: {:.6f}'.format(forgetting))
        for i in range(len(task_best_acc_list)):
            task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])
    else:
        forgetting = None
    task_best_acc_list.append(curren_step_acc)

    mean_old_task_acc = mean_old_task_acc / (step + 1)
    return mean_old_task_acc, forgetting



if __name__ == "__main__":
    logger.info(args)

    total_incremental_steps = args.num_classes // args.class_num_per_step

    setup_seed(args.seed)
    
    logger.info('Training start time: {}'.format(datetime.now()))

    train_set = IcreLoader(args=args, mode='train', modality=args.modality)
    val_set = IcreLoader(args=args, mode='val', modality=args.modality)
    test_set = IcreLoader(args=args, mode='test', modality=args.modality)


    task_best_acc_list = []

    step_forgetting_list = []

    step_accuracy_list = []

    ckpts_root = './save/{}/{}/'.format(args.dataset, args.modality)
    figs_root = './save/fig/{}/{}/'.format(args.dataset, args.modality)

    if not os.path.exists(ckpts_root):
        os.makedirs(ckpts_root)
    if not os.path.exists(figs_root):
        os.makedirs(figs_root)

    for step in range(total_incremental_steps):
        train_set.set_incremental_step(step)
        val_set.set_incremental_step(step)
        test_set.set_incremental_step(step)

        logger.info('Incremental step: {}'.format(step))
        train(args, step, train_set, val_set)
        step_accuracy, step_forgetting = detailed_test(args, step, test_set, task_best_acc_list)
        step_accuracy_list.append(step_accuracy)
        if step_forgetting is not None:
            step_forgetting_list.append(step_forgetting)
    Mean_accuracy = np.mean(step_accuracy_list)
    logger.info('Average Accuracy: {:.6f}'.format(Mean_accuracy))
    Mean_forgetting = np.mean(step_forgetting_list)
    logger.info('Average Forgetting: {:.6f}'.format(Mean_forgetting))
    
    if args.dataset != 'AVE' and args.modality != 'audio':
        train_set.close_visual_features_h5()
        val_set.close_visual_features_h5()
        test_set.close_visual_features_h5()


