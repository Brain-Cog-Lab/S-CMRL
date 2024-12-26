import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.gym.gym_env import EnvSpec
from src.models.models import (
    BatchNormMLP,
    Policy,
    VisionModel,
)
from src.run.arguments import get_args
from src.utils.constants import ENV_TO_ID
from src.utils.data import data_loading, FrozenEmbeddingDataset,Load_CASC
from src.utils.utils import fuse_embeddings_flare, set_seed
from src.utils.constants import ENV_TO_ID, ENV_TO_SUITE
from src.gym.gym_wrapper import env_constructor
from src.utils.utils import (
    compute_metrics_from_paths,
    fuse_embeddings_flare,
    generate_videos,
    sample_paths,
    set_seed,
)
import dmc2gym, gym, mj_envs, mjrl.envs
from torch.nn import functional as F
from pyvirtualdisplay import Display
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    https://github.com/ildoonet/pytorch-gradual-warmup-lr
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

# 创建虚拟显示
display = Display(visible=0, size=(1024, 768))
display.start()

if __name__ == "__main__":
    # Hyperparameters
    args = get_args()
    task_conditioning = (
        args.middle_adapter_type == "middle_adapter_cond"
        or args.top_adapter_type == "top_adapter_cond"
        or args.policy_type == "policy_cond"
    )
    args.use_cls = args.use_cls == 1

    # Setting random seed
    set_seed(args.seed)

    # Tensorboard
    tb_path = (
        f"logs/train/tb/{args.seed}_{args.middle_adapter_type}"
        f"_{args.top_adapter_type}_{args.policy_type}_{args.use_cls}"
        f"_{args.expe_name}"
    )
    writer = SummaryWriter(tb_path)
    writer.add_text("Args", str(args), 0)

    # ckpt saving
    ckpts_path = tb_path.replace("/tb/", "/ckpts/")
    if not os.path.exists(ckpts_path):
        os.makedirs(ckpts_path)
    tasknum=9
    pacsnum=4
    gymnum=5
    # Data loading
    (
        timesteps_all_envs,
        highest_action_dim,
    ) = data_loading()

    pacs=Load_CASC(args.batch_size)
    # Dataset and dataloader
    dataset={}
    dataloader={}
    testdataloader={}
    policy={}
    vision_model=[]
    observation_dim={}
    env_spec={}
    for t in range(tasknum):
        if t<pacsnum:
            dataloader[t]=pacs.train_datasets[t]
            testdataloader[t]=pacs.test_datasets[t]
            vision_model.append(VisionModel(t,7).to("cuda"))
            vision_model[-1].train()
        else:
            dataset[t] = FrozenEmbeddingDataset(
                timesteps_all_envs=timesteps_all_envs[t-pacsnum],
                history_window=args.history_window,
                highest_action_dim=highest_action_dim[t-pacsnum],
            )
            dataloader[t] = DataLoader(
                dataset[t],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=False,
            )
            testdataloader[t]=None

            horizon = None
            observation_dim[t] = args.history_window * args.img_emb_size + highest_action_dim[t-pacsnum]
            env_spec[t] = EnvSpec(observation_dim[t], highest_action_dim[t-pacsnum], horizon)

            policy[t] = Policy(
                env_spec=env_spec[t],
                hidden_sizes=(256,256,256),
                nonlinearity='relu',
                dropout=0,
            ).to('cuda')
            policy[t].train()
            vision_model.append(VisionModel(t,768).to("cuda"))
            vision_model[-1].train()

        print("Dataloader length: ", len(dataloader[t]))

        #Vision Model

    
    task_embedding_predictor = None
    acc=[]
    # Training
    for task in range(tasknum):
        for t in range(task):
            vision_model[t].eval()
        vision_model[task].train()
        task_embedding_predictor = None

        # Optimizer and loss
        optimized_weights = []
        name_optimized_weights = []
        if task>=pacsnum:
            optimized_weights += list(policy[task].parameters())
            for name, _ in policy[task].named_parameters():
                name_optimized_weights.append(name)

        vision_params_to_train = []
        for name, param in vision_model[task].named_parameters():
            vision_params_to_train.append(param)
            name_optimized_weights.append(name)
        optimized_weights += vision_params_to_train

        optimizer = torch.optim.Adam(optimized_weights, lr=0.01,weight_decay= 0.0005, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                   [100,120],
                                                                   gamma=0.1)
        warmup_scheduler = GradualWarmupScheduler(optimizer,
                                                multiplier=1,
                                                total_epoch=10,
                                                after_scheduler=scheduler)
        #loss_func = torch.nn.MSELoss(reduction="none")

        if task>0:
            num_edges=task*3
            h_e =torch.full((num_edges, 6), 0, dtype=torch.long)
            h_a = torch.full((num_edges, 6), 0.0)
            lossmin=1
            lossmax=0
            evoepoch=60
            for epoch in tqdm(range(evoepoch)):
                running_loss = 0.0
                per_task_running_loss = [[] for _ in range(args.ntasks)]
                warmup_scheduler.step()
                if epoch == 10:
                    vision_model[-1].fc_agr.reset_parameters()
                p_n = vision_model[task].probability()
                selected_ops=torch.multinomial(p_n, 1).view(-1)
                print("prob:",p_n)
                print(selected_ops)
                for mb_idx, batch in tqdm(enumerate(dataloader[task])):
                    # Zeroing gradients
                    optimizer.zero_grad()
                    # Data
                    if task>=pacsnum:
                        actions_mask = batch["actions_mask"].bool().to(args.device)
                        proprio_input = batch["proprio_input"].float().to(args.device)
                        images = batch["images"].float().to(args.device)
                        task_id = batch["task_id"].to(args.device)
                        tar = batch["actions"].float().to(args.device)
                    else:
                        images = batch[0].float().to(args.device)
                        tar = batch[1].float().to(args.device)

                    task_embedding = None

                    # Vision model forward pass
                    emb = vision_model[task](
                        task,
                        images,
                        task_embedding,
                        selected_ops,
                        vision_model[:task]
                    )
                    # if task>=pacsnum:
                    #     # Fusing embeddings for the last 3 frames
                    #     feat = fuse_embeddings_flare(emb)

                    #     # Concatenating the proprioception input
                    #     feat = torch.cat([feat, proprio_input], dim=-1)

                    #     # Policy forward pass
                    #     pred = policy[task](feat)
                    #     loss = loss_func(pred, tar.detach())
                    #     loss = loss.view(-1)
                    #     loss = loss.mean()
                    #     loss.backward()
                    #     optimizer.step()
                    # else:
                    pred=emb
                    loss =  F.cross_entropy(pred, tar)
                    loss.backward()
                    optimizer.step()


                    running_loss += loss.to("cpu").data.numpy().ravel()[0]

                    # Backward pass
                if epoch==0:
                    lossmax=lossmin=running_loss
                else:
                    if running_loss>lossmax:
                        lossmax=running_loss
                    if running_loss<lossmin:
                        lossmin=running_loss
                # Logging average loss for the epoch
                writer.add_scalar("Loss/train", running_loss / (mb_idx + 1), epoch + 1)
                if lossmin==lossmax:
                    evoloss=0.5
                else:
                    evoloss=(evoloss-lossmin)/(lossmax-lossmin)
                for i, idx in enumerate(selected_ops):
                    h_e[i][idx] += 1
                    h_a[i][idx] = 1-evoloss

                # 4 update the probability
                for k in range(num_edges):
                    dh_e_k = torch.reshape(h_e[k], (1, -1)) - torch.reshape(h_e[k], (-1, 1))

                    dh_a_k = torch.reshape(h_a[k], (1, -1)) - torch.reshape(h_a[k], (-1, 1))

                    vector1 = torch.sum((dh_e_k < 0) * (dh_a_k > 0), dim=0)
                    vector2 = torch.sum((dh_e_k > 0) * (dh_a_k < 0), dim=0)
                    vision_model[task].p[k] += (0.1 * (vector1-vector2).float())
                    vision_model[task].p[k] = F.softmax(vision_model[task].p[k], dim=0)

            geno=vision_model[task].genotype()
        else:
            geno=None


        if task==0:
            n_epochs=80
        else:
            n_epochs=30
        for epoch in tqdm(range(n_epochs)):
            vision_model[task].train()
            running_loss = 0.0
            per_task_running_loss = [[] for _ in range(args.ntasks)]
            warmup_scheduler.step()
            if epoch == 10:
                vision_model[-1].fc_agr.reset_parameters()
            preds, targets = [], []
            for mb_idx, batch in tqdm(enumerate(dataloader[task])):
                # Zeroing gradients
                optimizer.zero_grad()

                # Data
                if task>=pacsnum:
                    actions_mask = batch["actions_mask"].bool().to(args.device)
                    proprio_input = batch["proprio_input"].float().to(args.device)
                    images = batch["images"].float().to(args.device)
                    task_id = batch["task_id"].to(args.device)
                    tar = batch["actions"].float().to(args.device)
                else:
                    images = batch[0].float().to(args.device)
                    tar = batch[1].to(args.device)

                # # Task embedding prediction
                # if task_conditioning:
                #     task_embedding = task_embedding_predictor(task_id)
                # else:
                task_embedding = None

                # Vision model forward pass
                emb = vision_model[task](
                    task, 
                    images,
                    task_embedding,
                    geno,
                    vision_model[:task]
                )

                # Fusing embeddings for the last 3 frames
                # if task>=pacsnum:
                #         # Fusing embeddings for the last 3 frames
                #     feat = fuse_embeddings_flare(emb)

                #         # Concatenating the proprioception input
                #     feat = torch.cat([feat, proprio_input], dim=-1)

                #         # Policy forward pass
                #     pred = policy[task](feat)
                #     loss = loss_func(pred, tar.detach())
                #     loss = loss.view(-1)
                #     loss = loss.mean()
                #     loss.backward()
                #     optimizer.step()
                # else:
                pred=emb
                loss =  F.cross_entropy(pred, tar)
                loss.backward()
                optimizer.step()

                # Computing per-task loss
                # for i in range(args.ntasks):
                #     task_loss = loss[task_id == i]
                #     task_actions_mask = actions_mask[task_id == i]
                #     task_loss = task_loss.view(-1)
                #     task_actions_mask = task_actions_mask.view(-1)
                #     task_loss = task_loss[task_actions_mask]
                #     per_task_running_loss[i].append(task_loss)

                # Masking loss
                preds.append(emb.detach().cpu().numpy())
                targets.append(tar.long().cpu().numpy())
                running_loss += loss.to("cpu").data.numpy().ravel()[0]
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            print(np.max(preds.argmax(0)),np.mean(preds.argmax(0)))
            top1_acc = (preds.argmax(1) == targets).sum() /targets.shape[0] * 100
            print("---------------------traintttacc--------------", top1_acc)

            vision_model[task].eval()
            predst, targetst = [], []
            with torch.no_grad():            
                for mb_idx, batch in tqdm(enumerate(dataloader[task])):

                    # Data
                    if task>=pacsnum:
                        actions_mask = batch["actions_mask"].bool().to(args.device)
                        proprio_input = batch["proprio_input"].float().to(args.device)
                        images = batch["images"].float().to(args.device)
                        task_id = batch["task_id"].to(args.device)
                        tar = batch["actions"].float().to(args.device)
                    else:
                        images = batch[0].float().to(args.device)
                        tar = batch[1].to(args.device)

                    task_embedding = None

                    # Vision model forward pass
                    emb = vision_model[task](
                        task, 
                        images,
                        task_embedding,
                        geno,
                        vision_model[:task]
                    )

                    # Masking loss
                    predst.append(emb.detach().cpu().numpy())
                    targetst.append(tar.long().cpu().numpy())

            predst = np.concatenate(predst, axis=0)
            targetst = np.concatenate(targetst, axis=0)
            print(pred.shape)
            top1_acc = (predst.argmax(1) == targetst).sum() /targetst.shape[0] * 100
            print("---------------------test top1_acc--------------", top1_acc,running_loss)

            # Logging average loss for the epoch
            writer.add_scalar("Loss/train", running_loss / (mb_idx + 1), epoch + 1)

            # # Logging average per-task loss for the epoch
            # for env in tqdm(list(ENV_TO_ID.keys())[:args.ntasks]):
            #     env_id = ENV_TO_ID[env]
            #     writer.add_scalar(
            #         f"Loss/train_policy_{env}",
            #         torch.cat(per_task_running_loss[env_id]).mean().item(),
            #         epoch + 1,
            #     )

            # Saving ckpts

        ckpt_dict = {
                    "seed": args.seed,
                    "epoch": epoch,
                    "epoch_loss": running_loss / (mb_idx + 1),
                    "optimizer_state_dict": optimizer.state_dict(),
        }
        if task>=pacsnum:
            ckpt_dict["policy_state_dict"] = policy[task].state_dict()
        else:
            ckpt_dict["policy_state_dict"] = None
        ckpt_dict["vision_model_state_dict"] = vision_model[task].state_dict()

                # if task_conditioning:
                #     ckpt_dict["task_embedding_predictor_state_dict"] = (
                #         task_embedding_predictor.state_dict()
                #     )

        torch.save(ckpt_dict, os.path.join(ckpts_path, f"ckpt_{epoch}_task_{task}.pth"))

        if task>=pacsnum:
            env_name=list(ENV_TO_SUITE.keys())[task-pacsnum]
            policye = BatchNormMLP(
            env_spec=env_spec[task],
            hidden_sizes=(256,256,256),
            seed=args.seed,
            nonlinearity='relu',
            dropout=0,
            )
            policye.model=policy[task].policy
            policye.model.eval()

            suite = ENV_TO_SUITE[env_name]
            env_kwargs = {
            "env_name": env_name,
            "suite": suite,
            "device": 'cuda',
            "image_width": 256,
            "image_height": 256,
            "camera_name": 0,
            "embedding_name": "vc1_vitb",
            "pixel_based": True,
            "seed": args.seed,
            "history_window": 3,
            "task": task,
            "add_proprio": False,
            "proprio_key": None,
            }

            vision_model[task].eval()
            env_kwargs["vision_model"] = vision_model
            policy_cond = True
            env_kwargs["policy_cond"] = policy_cond
            env_kwargs["task_embedding"] = None
            env_kwargs["policy_observation_dim"] = observation_dim[task]
            env_kwargs["highest_action_dim"] = highest_action_dim[task-pacsnum]
            e = env_constructor(
                **env_kwargs,
                fuse_embeddings=fuse_embeddings_flare,
            )

            paths = sample_paths(
            num_trajs=50,
            env=e,
            policy=policye,
            eval_mode=True,
            horizon=e.horizon,
            base_seed=args.seed,
            )
            # appr.after_learn(task_i, vval_loader, cfg['batch_size'],device)
            mean_return, mean_score = compute_metrics_from_paths(env=e,suite=suite,paths=paths)

            # Generating sample videos

            eval_videos_path = os.path.join('/data1/hanbing/task_conditioned_adaptation-main/task_conditioned_adaptation-main/vlog/', str(task))
            generate_videos(paths, eval_videos_path)

            writer.add_scalar("end_task/mean", task, mean_return, mean_score)
        else:
            vision_model[task].eval()
            preds, targets = [], []
            with torch.no_grad():            
                for i, (inputs, lbls) in enumerate(testdataloader[task]):
                    inputs = inputs.to(args.device, non_blocking=True)
                    emb = vision_model[task](
                    task, 
                    images,
                    task_embedding,
                    geno,
                    vision_model[:task]
                    )
                    preds.append(emb.detach().cpu().numpy())
                    targets.append(lbls.long().cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            top1_acc = (preds.argmax(1) == targets).sum() /preds.shape[0] * 100
            acc.append(top1_acc)
            writer.add_scalar("end_task/top1_acc", task, top1_acc)
            print("---------------------test accc--------------",acc)


