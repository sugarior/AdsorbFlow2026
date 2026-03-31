import torch
import copy


oc22_is2rs = {
    'name': 'oc22_is2rs',
    'atom_nb_to_id': {}, # 待填充
    'atom_decoder': [], # 待填充
    'n_nodes': {},      # 待填充 (直方图)
    'max_n_nodes': '250',

    'lattice_stats': {
        'mean':[],
        'std':[]
    },
    'displacement_std': 0.0, # 待填充
    'with_pbc': True,
    'with_h': True
}

'''
{
    "name": "oc22_is2rs",
    "atom_nb_to_id": {1: 0, 6: 1, 8: 2, 29: 3, ...},
    "atom_decoder": [1, 6, 8, 29, ...],
    "n_nodes": {48: 1200, 56: 800, ...},
    "max_n_nodes": 120,
    "lattice_stats": {
        "mean": [10.52, 10.52, 30.15, 90.0, 90.0, 120.0],
        "std": [1.2, 1.2, 5.5, 0.1, 0.1, 0.1]
    },
    "pos_mean": [0.0, 0.0, 0.0],
    "pos_std": [0.0, 0.0, 0.0],
    "displacement_std": 0.156, # 假设这是统计出的 MIC 位移标准差
    "with_h": True,
    "with_pbc": True
}
'''
def get_dataset_info(args):
    if args.dataset == 'oc22':
        # 核心：加载我们之前存好的统计文件，并注入到配置字典中
        dataset_info = load_oc22_info(args.oc22_stats_path)
    else:
        raise Exception(f"Wrong dataset {args.dataset}")
    return dataset_info


def load_oc22_info(path):
    stats = torch.load(path)
    info = copy.deepcopy(oc22_is2rs)

    info['atom_decoder'] = stats['atom_decoder']

    info['atom_nb_to_id'] = stats['atom_nb_to_id']

    info['lattice_stats']['mean'] = stats['lattice_mean']

    info['lattice_stats']['std'] = stats['lattice_std']

    info['displacement_std'] = stats['displacement_std']

    info['n_nodes'] = stats['n_nodes']

    info['pos_mean'] = stats['pos_mean']

    info['pos_std'] = stats['pos_std']

    #print(f"✅ 加载 OC22 统计信息成功 | 原子类型数量: {len(info['atom_decoder'])} | 最大节点数: {info['max_n_nodes']} | 晶格长度均值: {info['lattice_stats']['mean']} | 晶格长度标准差: {info['lattice_stats']['std']} | 位移标准差: {info['displacement_std']}")
    return info