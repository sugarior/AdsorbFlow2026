import torch
import numpy as np
from copy import deepcopy
import os
import logging
from time import time
import wandb
from torch.utils.data import DataLoader
from collections import OrderedDict
# 保持你原来的导入
from utils.parse_args import parse_args
from models.get_models import get_optim, get_goat
from configs.oc_dataset_config import get_dataset_info
from train_epoch import train_epoch, test_adsorb
from ocp_data.get_oc_datasets import get_dataloaders
import utils.utilis_func as uf
from utils.utilis_func import setup_experiment_dir, wandb_setup,load_checkpoint,create_logger
from ocp_data.set_dataset_path import set_path



@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag



def main(args):
    """
    单机单卡版 AdsorbFlow 训练脚本
    """
    # 1. 硬件环境初始化 (取消 DDP 相关 init)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(args.global_seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0) # 默认使用第一张卡
    
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float64':
        dtype = torch.float64

    # 2. 实验目录构建 (保留自动递增逻辑)
    experiment_dir, checkpoint_dir = setup_experiment_dir(args)
    logger = create_logger(experiment_dir)
    set_path(args)
    
    # 3. WandB 初始化 (保留)
    wandb_setup(args)
    
    logger.info(f"🚀 AdsorbFlow 单机版启动 | 实验目录: {experiment_dir}")

    # 4. 加载物理先验 (OCP 核心)
    dataset_info = get_dataset_info(args) 
    args.lattice_mean = dataset_info['lattice_stats']['mean'].to(device, dtype)
    args.lattice_std = dataset_info['lattice_stats']['std'].to(device, dtype)
    args.displacement_std = dataset_info['displacement_std']
    args.pos_mean = dataset_info['pos_mean'].to(device, dtype)
    args.pos_std = dataset_info['pos_std'].to(device, dtype)
    # 5. 维度自动计算

    args.in_node_nf = args.num_atom_types + 3
    lut = torch.zeros(args.num_atom_types, dtype=torch.long)

    # 2. 填充映射关系
    for atomic_nb, id_idx in dataset_info['atom_nb_to_id'].items():
        
        atomic_nb = int(atomic_nb)
        id_idx = int(id_idx)

        lut[atomic_nb] = id_idx

    # 3. 将其挂载到 args 并在训练前送到 GPU
    args.atom_lut = lut.to(device)

    # 6. 获取数据加载器 (注意：修改 get_dataloaders 调用)
    # 确保 get_dataloaders 内部在 world_size=1 时不使用 DistributedSampler
    dataloaders= get_dataloaders(args)

    # 7. 初始化模型与优化器
    model, _, _ = get_goat(args, device, dataset_info, dataloaders['train'])
    model = model.to(device)
    model = model.float() if dtype == torch.float32 else model.double()
    
    # 核心：保留 EMA 影子模型
    model_ema = deepcopy(model).to(device)
    model_ema = model_ema.float() if dtype == torch.float32 else model_ema.double()
    requires_grad(model_ema, False)
    ema_updater = uf.EMA(args.ema_decay)

    optim = get_optim(args, model)
    gradnorm_queue = uf.Queue()
    gradnorm_queue.add(3000)
    best_val_loss = 1e8

    # 8. 断点续训 (保持原样)
    begin_epoch = 0
    if args.resume is not None:
        model, model_ema,optim, begin_epoch, best_val_loss = load_checkpoint(model, model_ema,optim,args.resume,device)

        for param_group in optim.param_groups:
            param_group['lr'] = args.lr
            
        logger.info(f"🔄 Checkpoint 已加载，学习率已手动重置为: {args.lr}")

    
    

    # --- 核心训练循环 ---
    for epoch in range(begin_epoch, args.epochs):
        
        logger.info(f"Beginning epoch {epoch}...")
        start_epoch = time()
        
        
        # A. 训练一轮
        train_loss = train_epoch(
            args=args, loader=dataloaders['train'], epoch=epoch, model=model,
            model_ema=model_ema, ema=ema_updater, device=device, dtype=dtype,
            optim=optim, gradnorm_queue=gradnorm_queue
        )
        

        # B. 定期验证 (基于 Validation Loss)
        if epoch % args.test_epochs == 0:
            val_loss = test_adsorb(
                args=args, loader=dataloaders['val'], epoch=epoch, 
                eval_model=model_ema, device=device, dtype=dtype
            )
            
            # C. 模型保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(checkpoint_dir, "best_model.pt")
                checkpoint = {
                    "model": model.state_dict(), # 注意：去除了 .module
                    "model_ema": model_ema.state_dict(),
                    "opt": optim.state_dict(),
                    "args": args,
                    "epoch": epoch,
                    "val_loss": val_loss
                }
                torch.save(checkpoint, save_path)
                logger.info(f"★ New Best Val Loss: {val_loss:.4f}! Saved to {save_path}")

            current_global_step = (epoch + 1) * len(dataloaders['train'])
            # 日志打印
            logger.info(f"E:{epoch} | Train:{train_loss:.4f} | Val:{val_loss:.4f} | BestVal:{best_val_loss:.4f}")
            wandb.log({"epoch": epoch,
            "epoch_metrics/train_loss": train_loss,
            "epoch_metrics/val_loss": val_loss,
            "train/step": current_global_step
            })

    logger.info("Training Complete!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
