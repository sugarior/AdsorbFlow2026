import os
import logging
from time import time
import wandb
import torch
import numpy as np
from ase import Atoms
from ase.io import write



def setup_experiment_dir(args):
    """
    构建实验目录，包含自动递增的版本号逻辑
    """
    base_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(base_dir, exist_ok=True)
    
    # 自动递增版本号
    existing_versions = [d for d in os.listdir(base_dir) if d.startswith("version_")]
    version_numbers = [int(d.split("_")[1]) for d in existing_versions if d.split("_")[1].isdigit()]
    next_version = max(version_numbers) + 1 if version_numbers else 0
    
    experiment_dir = os.path.join(base_dir, f"version_{next_version}")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return experiment_dir, checkpoint_dir

def wandb_setup(args):
    """
    初始化 WandB 连接
    """
    wandb.init(
        project=args.wandb_project,
        name=args.exp_name,
        config=vars(args)
    )

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/*", step_metric="train/step") # 让验证也对齐训练进度
    wandb.define_metric("epoch_metrics/*", step_metric="epoch")

def load_checkpoint(model, model_ema,optimizer, checkpoint_path, device):
    """
    加载模型和优化器状态
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model_ema.load_state_dict(checkpoint['model_ema'])
    optimizer.load_state_dict(checkpoint['opt'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    return model, model_ema,optimizer, start_epoch, best_val_loss

def create_logger(log_dir):
    """
    创建日志记录器
    """
    logger = logging.getLogger("AdsorbFlow2026")
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setLevel(logging.INFO)
    
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器到 logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    
class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)
    

def get_pbc_diff(pos_1, pos_0, cell, batch_idx):
    """
    计算从 pos_0 到 pos_1 的最小镜像位移向量 (u = x1 - x0)
    pos_1: [Total_N, 3] (目标态)
    pos_0: [Total_N, 3] (初始态)
    cell: [Batch, 3, 3] (晶格矩阵)
    batch_idx: [Total_N] (节点所属 Batch 索引)
    """
    # 1. 计算原始笛卡尔差值
    diff = pos_1 - pos_0 # [Total_N, 3]

    # 2. 将差值转换到分数坐标空间 (Fractional Coordinates)
    # 分数坐标 = 笛卡尔坐标 @ 晶格矩阵的逆
    inv_cell = torch.inverse(cell) # [Batch, 3, 3]
    batch_inv_cell = inv_cell[batch_idx] # [Total_N, 3, 3]
    
    # 使用 einsum 进行高效的矩阵-向量乘法
    # [Total_N, 3] @ [Total_N, 3, 3] -> [Total_N, 3]
    frac_diff = torch.einsum('ni,nij->nj', diff, batch_inv_cell)

    # 3. 核心：最小镜像修正
    # 将位移限制在 [-0.5, 0.5] 之间。如果移动了 0.8，其实就是向反方向移动了 -0.2
    frac_diff = frac_diff - torch.round(frac_diff)

    # 4. 转回笛卡尔空间得到真实物理位移
    # 物理位移 = 分数位移 @ 晶格矩阵
    batch_cell = cell[batch_idx]
    true_diff = torch.einsum('ni,nij->nj', frac_diff, batch_cell)

    return true_diff

def normalize_pos(pos, mean, std, target_range=3.0):
    """
    将原始坐标归一化到约 [-target_range, target_range]
    pos: [N, 3] 原始坐标
    mean: [3] 或 [1] 坐标均值 (来自 dataset_info)
    std: [3] 或 [1] 坐标标准差 (来自 dataset_info)
    """
    # 确保均值和方差在正确的设备上
    mean = mean.to(pos.device)
    std = std.to(pos.device)
    
    # 1. 标准化到 N(0, 1)
    pos_norm = (pos - mean) / (std + 1e-8)
    
    # 2. 缩放到目标范围
    # 物理意义：让 1 个标准差的位移对应 target_range/scale 的数值
    # 通常取 target_range=3.0，因为正态分布下 99% 的数据在 3 sigma 内
    return pos_norm * (target_range / 3.0)

def denormalize_pos(pos_norm, mean, std, target_range=3.0):
    """
    将模型输出的归一化坐标还原回物理空间的“埃”单位
    """
    mean = mean.to(pos_norm.device)
    std = std.to(pos_norm.device)
    
    # 1. 逆缩放
    pos_rescaled = pos_norm / (target_range / 3.0)
    
    # 2. 逆标准化
    pos_orig = pos_rescaled * std + mean
    
    return pos_orig

def export_comparison_cif(pos_true, pos_rec, atomic_numbers, cell, save_path, sample_idx=0,adsorbate_mask=None):
    """
    将真值和重构值导出为 CIF 文件
    pos_true: [Total_N, 3] 物理空间真值
    pos_rec: [Total_N, 3] 物理空间重构值
    atomic_numbers: [Total_N] 原子序数
    cell: [3, 3] 晶格矩阵
    sample_idx: 样本编号，用于文件名
    """
    # 1. 确保数据在 CPU 上且为 numpy
    pos_true = pos_true.detach().cpu().numpy()
    pos_rec = pos_rec.detach().cpu().numpy()
    atomic_numbers = atomic_numbers.detach().cpu().numpy()
    cell = cell.detach().cpu().numpy()

    mask = adsorbate_mask.detach().cpu().numpy().flatten().astype(bool)

    pos_hybrid = pos_true.copy()
    # 将吸附质的部分替换为 VAE 重构出来的坐标
    pos_hybrid[mask] = pos_rec[mask]

    # 2. 创建真值结构 (Ground Truth)
    # 我们给真值原子打个特殊的标签，或者保持原样
    atoms_true = Atoms(numbers=atomic_numbers, 
                       positions=pos_true, 
                       cell=cell, 
                       pbc=True)
    
    # 3. 创建重构结构 (Reconstructed)
    # 【大师级技巧】：为了在 VESTA 中一眼分清，我们可以把重构结构的原子
    # 暂时“伪装”成另一种不相关的元素（例如全变成 Si 或 S），
    # 这样在 VESTA 里它们会有不同的颜色。
    atoms_rec = Atoms(numbers=atomic_numbers, 
                      positions=pos_hybrid, 
                      cell=cell, 
                      pbc=True)

    # 4. 导出文件
    true_file = f"{save_path}/sample_{sample_idx}_GT.cif"
    rec_file = f"{save_path}/sample_{sample_idx}_HYBRID.cif"
    
    write(true_file, atoms_true)
    write(rec_file, atoms_rec)
    
    print(f"✅ 已导出可视化文件到: {true_file} 和 {rec_file}")