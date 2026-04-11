import torch
from torch.utils.data import Dataset
from ase import Atoms
from ase.neighborlist import neighbor_list
from ocpmodels.datasets import LmdbDataset,data_list_collater
from torch.utils.data import DataLoader
class OCP2FlowDataset(Dataset):


    def __init__(self, ocp_dataset, cutoff=6.0):
        """
        包装 OCP 的 LmdbDataset，提取晶胞信息，计算边
        """
        self.dataset = ocp_dataset
        self.cutoff = cutoff

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. 获取原始 OCP Data 对象
        data = self.dataset[idx]
        
            
        # 3. 晶格参数计算
        # OCP 的 cell 通常是 [1, 3, 3]
        if hasattr(data, 'cell'):
            lengths, angles = get_lattice_params_single(data.cell)
            data.lengths = lengths # [1, 3]
            data.angles = angles   # [1, 3]
            
            
        
        if not hasattr(data, 'edge_index') or data.edge_index is None:
           
            edge_index, cell_offsets = get_pbc_edges(data, self.cutoff)
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets # 这对 EGNN 修正距离极其重要
            #data.edge_index = torch.empty((2, 0), dtype=torch.long)
            

        return data

def get_pbc_edges(data,cutoff=6.0):
    # 1. 将 PyG Data 转回 ASE Atoms 对象
    # 注意：这里需要 data.pos, data.atomic_numbers, data.cell
    # OCP 的 cell 形状通常是 [1, 3, 3]，需要 squeeze 成 [3, 3]
    atoms = Atoms(
        numbers=data.atomic_numbers,
        positions=data.pos,
        cell=data.cell.squeeze(0),
        pbc=True
    )

    # 2. 调用 ASE 的高效邻居列表算法 (计算带 PBC 的边)
    # i: 源节点, j: 目标节点, S: 跨越的晶格偏移量 (offsets)
    i, j, S = neighbor_list('ijS', atoms, cutoff)

    # 3. 转换为 PyTorch 张量
    edge_index = torch.stack([torch.tensor(i), torch.tensor(j)], dim=0).long()
    cell_offsets = torch.tensor(S).float()
    
    return edge_index, cell_offsets


def get_lattice_params_single(matrix):
    """
    计算单个晶胞的参数。
    输入 matrix: [1, 3, 3] 或 [3, 3]
    输出 lengths: [1, 3]
    输出 angles: [1, 3]
    """
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0) # 统一变成 [1, 3, 3] 处理
        
    # 1. 提取三个晶格矢量 a, b, c
    v1 = matrix[:, 0, :]
    v2 = matrix[:, 1, :]
    v3 = matrix[:, 2, :]

    # 2. 计算边长
    len1 = torch.norm(v1, dim=1)
    len2 = torch.norm(v2, dim=1)
    len3 = torch.norm(v3, dim=1)
    lengths = torch.stack([len1, len2, len3], dim=1)

    # 3. 计算夹角
    def compute_angle(v_a, v_b, l_a, l_b):
        cosine = torch.sum(v_a * v_b, dim=1) / (l_a * l_b)
        cosine = torch.clamp(cosine, -1.0, 1.0)
        angle_rad = torch.acos(cosine)
        return torch.rad2deg(angle_rad)

    alpha = compute_angle(v2, v3, len2, len3)
    beta  = compute_angle(v1, v3, len1, len3)
    gamma = compute_angle(v1, v2, len1, len2)
    
    angles = torch.stack([alpha, beta, gamma], dim=1)
    
    return lengths, angles

def get_dataloaders(args):
    '''
    获取数据加载器
    可以通过 args.dataset 来选择不同的数据集配置
    可以通过下标来选择不同的数据集划分（train/val/test）
    返回一个包含 'train', 'val', 'test' 的字典。
    '''

    splits = {
        'train': {'src': args.train_src, 'shuffle': True},
        'val':   {'src': args.val_src,   'shuffle': False},
        'test':  {'src': args.test_src,  'shuffle': False}
    }

    dataloaders = {}

    
    for split_name, split_cfg in splits.items():
        if split_cfg['src'] is None:
            continue # 如果没有提供该划分的路径，则跳过

        # A. 实例化 OCP 原始 LmdbDataset
        # 注意：这里传入的是 OCP 期望的字典配置
        raw_dataset = LmdbDataset({"src": split_cfg['src']})

        # B. 嵌套你的自定义 AdsorbFlow 包装类
        # 负责计算 cell_params, edge_index 等物理特征
        flow_dataset = OCP2FlowDataset(raw_dataset, cutoff=args.cutoff)

        # C. 实例化 DataLoader
        loader = DataLoader(
            flow_dataset,
            batch_size=args.batch_size,
            shuffle=split_cfg['shuffle'], # 只有训练集打乱
            num_workers=args.num_workers,
            collate_fn=data_list_collater, # 必须使用 OCP 专用聚合函数
            pin_memory=True if torch.cuda.is_available() else False
        )

        # D. 存入字典
        dataloaders[split_name] = loader

    return dataloaders


def get_dataloader_for_src(args, src, shuffle=False, batch_size=None):
    """
    从单个 LMDB 路径构造与训练一致的 DataLoader（用于导出、评估等）。
    """
    if src is None:
        raise ValueError("get_dataloader_for_src: src 不能为 None")
    bs = batch_size if batch_size is not None else args.batch_size
    raw_dataset = LmdbDataset({"src": src})
    flow_dataset = OCP2FlowDataset(raw_dataset, cutoff=args.cutoff)
    return DataLoader(
        flow_dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=data_list_collater,
        pin_memory=True if torch.cuda.is_available() else False,
    )