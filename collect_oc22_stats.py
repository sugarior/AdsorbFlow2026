import torch
from tqdm import tqdm
import numpy as np
import json
import os
import sys
sys.path.append('/sda1/zzwang/decopm3/DEEPHER1030')
from ocpmodels.datasets import data_list_collater,LmdbDataset
from torch.utils.data import DataLoader
from ocp_data.get_oc_datasets import OCP2FlowDataset
DataPath = "/sda1/zzwang/decopm3/oc22data/is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train"
sys.path.append('/sda1/zzwang/decopm3/ICLR2025-GOAT')

def collect_oc22_stats(dataloader,device='cuda'):
    all_atom_types = set()
    all_n_nodes = []
    # all_lengths = []
    # all_angles = []
    all_lattice_params = []
    all_displacements = []
    running_sum = torch.zeros(3).to(device)
    running_sq_sum = torch.zeros(3).to(device)
    total_atoms = 0

    print("开始统计数据集信息...")
    for batch in tqdm(dataloader):
        pos = batch.pos.to(device)
        pos_relaxed = batch.pos_relaxed.to(device)
        cell = batch.cell.to(device)
        batch_idx = batch.batch.to(device)

        running_sum += pos_relaxed.sum(dim=0)
        running_sq_sum += (pos_relaxed**2).sum(dim=0)
        total_atoms += pos_relaxed.size(0)

        # --- 快速统计 ---
        all_atom_types.update(batch.atomic_numbers.tolist())
        all_n_nodes.extend(batch.natoms.tolist())
        
        l_params = torch.cat([batch.lengths, batch.angles], dim=-1)
        all_lattice_params.append(l_params.cpu()) # 统计量回传 CPU 节省显存

        # --- GPU 加速 MIC 位移计算 ---
        with torch.no_grad():
            inv_cell = torch.inverse(cell)
            batch_inv_cell = inv_cell[batch_idx]

            # 分数坐标转换
            frac_pos = torch.einsum('ni,nij->nj', pos, batch_inv_cell)
            frac_pos_relaxed = torch.einsum('ni,nij->nj', pos_relaxed, batch_inv_cell)
            frac_diff = frac_pos_relaxed - frac_pos
            
            # MIC 修正
            frac_diff = frac_diff - torch.round(frac_diff)

            # 转回笛卡尔
            true_diff = torch.einsum('ni,nij->nj', frac_diff, cell[batch_idx])
            
            # 性能优化：只记录每个原子的模长
            disp_norm = torch.norm(true_diff, dim=-1).cpu()
            all_displacements.append(disp_norm)
    # 计算均值和方差
    lengths_stack = torch.cat(all_lattice_params, dim=0)
    displace_stack = torch.cat(all_displacements, dim=0)
    # 计算全局均值和标准差
    pos_mean = running_sum / total_atoms
    # Var = E[X^2] - (E[X])^2
    pos_var = (running_sq_sum / total_atoms) - (pos_mean ** 2)
    pos_std = torch.sqrt(pos_var + 1e-8)
    
    stats = {
        'atom_decoder': sorted(list(all_atom_types)),
        'atom_nb_to_id': {nb: i for i, nb in enumerate(sorted(list(all_atom_types)))},
        'n_nodes': dict(zip(*np.unique(all_n_nodes, return_counts=True))),
        'max_n_nodes': max(all_n_nodes),
        'lattice_mean': lengths_stack.mean(dim=0),
        'lattice_std': lengths_stack.std(dim=0),
        'displacement_std': displace_stack.std(),
        'pos_mean': pos_mean.cpu(),
        'pos_std': pos_std.cpu()
    }
    return stats

def save_dataset_info(stats, save_dir='configs'):
    """
    stats: collect_oc22_stats 运行的结果字典
    save_dir: 存放路径，建议放在项目根目录的 config 文件夹下
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 路径定义
    pt_path = os.path.join(save_dir, 'oc22_stats.pt')
    json_path = os.path.join(save_dir, 'oc22_stats_readable.json')

    # --- 1. 使用 PyTorch 保存完整对象 (包含 Tensor) ---
    # 这是模型加载时真正读取的文件
    torch.save(stats, pt_path)
    print(f"✅ 完整统计信息已保存至: {pt_path}")

    # --- 2. 转换为 JSON 供人类查阅 (将 Tensor 转为 List) ---
    def make_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        
        # 2. 处理 Tensor
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        
        # 3. 处理字典（关键：同时清洗 K 和 V）
        if isinstance(obj, dict):
            # 对 k 进行强制转换，确保它是 Python 原生 int 或 str
            return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
        
        # 4. 处理序列
        if isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        
        return obj

    readable_stats = make_serializable(stats)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(readable_stats, f, indent=4, ensure_ascii=False)
    print(f"📝 人类可读预览已保存至: {json_path}")

if __name__ == "__main__":
    DataPath = "/sda1/zzwang/decopm3/oc22data/is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train"

    dataset = LmdbDataset({"src":DataPath})
    dataset = OCP2FlowDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=data_list_collater)
    # 1. 运行统计
    oc22_stats = collect_oc22_stats(dataloader)
    
    # 2. 调用存储函数
    save_dataset_info(oc22_stats, save_dir='./configs')