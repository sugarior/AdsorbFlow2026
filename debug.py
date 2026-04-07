from utils.parse_args import parse_args
from configs.oc_dataset_config import get_dataset_info
from ocp_data.get_oc_datasets import get_dataloaders
from ocp_data.set_dataset_path import set_path
from train_epoch import prepare_batch_data,test_adsorb,evaluate_reconstruction,load_model_weights,evaluate_flow_sample
import torch
from models.get_models import get_autoencoder,get_goat
import numpy as np


def debug_data_info(args):
    dataset_info = get_dataset_info(args)
    print("🔍 数据集统计信息:"
          f"\n- 原子类型 (atom_decoder): {dataset_info['atom_decoder']}"
            f"\n- 原子类型数量 (num_atom_types): {len(dataset_info['atom_decoder'])}"
            f"\n- 节点数量分布 (n_nodes): {dataset_info['n_nodes']}"
            f"\n- 最大节点数量 (max_n_nodes): {dataset_info['max_n_nodes']}"
            f"\n- 晶格长度均值 (lattice_mean): {dataset_info['lattice_stats']['mean']}"
            f"\n- 晶格长度标准差 (lattice_std): {dataset_info['lattice_stats']['std']}"
            f"\n- 位移标准差 (displacement_std): {dataset_info['displacement_std']}"
            f"\n- 原子位置均值 (pos_mean): {dataset_info['pos_mean']}"
            f"\n- 原子位置标准差 (pos_std): {dataset_info['pos_std']}"
    )


def debug_prepare_batch_data(args,device):
    loader = get_dataloaders(args)['train']
    dataset_info = get_dataset_info(args)
    batch = next(iter(loader))

    x, h, x_init, u_target, node_mask, edge_index, edge_attr, adsorbate_mask = prepare_batch_data(args, batch, device=device, dtype=torch.float32)

    print ("✅ prepare_batch_data 输出信息:"
          f"\n- x (relaxed positions): {x.shape}, dtype={x.dtype}"
          f"\n- h (node features): {h.shape}, dtype={h.dtype}"
          f"\n- x_init (initial positions): {x_init.shape}, dtype={x_init.dtype}"
          f"\n- u_target (normalized displacements): {u_target.shape}, dtype={u_target.dtype}"
          f"\n- node_mask: {node_mask.shape}, dtype={node_mask.dtype}"
          f"\n- edge_index: {edge_index.shape}, dtype={edge_index.dtype}"
          f"\n- edge_attr: {edge_attr.shape}, dtype={edge_attr.dtype}"
          f"\n- adsorbate_mask: {adsorbate_mask.shape}, dtype={adsorbate_mask.dtype}"
    )


def debug_vae(args, batch, device):
    model = get_autoencoder(args, device)
    print("✅ VAE 模型结构:")
    #print(model)
    x, h, x_init, u_target, node_mask, edge_index, edge_attr, adsorbate_mask,batch_idx = prepare_batch_data(args, batch, device=device, dtype=torch.float32)
    print(f"🕵️ 物理量纲自检:")
    print(f"-> x_norm max: {x.abs().max().item():.2f}")
    print(f"-> edge_attr_norm max: {edge_attr.abs().max().item():.2f}")
    print(f"-> h_norm max: {h.abs().max().item():.2f}")
    loss, loss_dict = model.compute_loss(x, h, node_mask, edge_index, edge_attr, context=None, batch_idx=batch_idx, adsorbate_mask=adsorbate_mask)


    loss.backward()  # 确保反向传播正常工作
    print("✅ VAE compute_loss 输出:")
    print(f"- loss: {loss.item()}")
    print(f"- loss_dict: {loss_dict}")


def debug_test(args, loader, epoch, eval_model, device, dtype):
    test_adsorb(args, loader, epoch, eval_model, device, dtype)

def debug_eval_recon(args,device,dtype):
    model = get_autoencoder(args,device)
    loader = get_dataloaders(args)['val']


    model = load_model_weights(model,args.test_checkpoint,device)

    results = evaluate_reconstruction(model,loader,args,device,dtype)
    analyze_results(results)

def debug_eval_flow_sample(args, device, dtype):
    assert args.probabilistic_model == "flow"
    flow_ckpt = args.test_checkpoint
    model, _, _ = get_goat(args, device)
    model = load_model_weights(model, flow_ckpt, device)
    loader = get_dataloaders(args)['val']
    results = evaluate_flow_sample(model, loader, args, device, dtype)
    analyze_results(results)

def analyze_results(results):
    """
    打印详细的误差统计数据
    """
    if not results:
        print("❌ 评估结果为空。")
        return

    res = np.array(results)
    
    print("\n" + "="*30)
    print("      VAE 重构能力评估报告")
    print("="*30)
    # 物理单位通常是埃 (A)
    print(f"平均 RMSD (Mean):   {np.mean(res):.4f} Å")
    print(f"中位数 (Median):    {np.median(res):.4f} Å")
    print(f"最小值 (Min):       {np.min(res):.4f} Å")
    print(f"最大值 (Max):       {np.max(res):.4f} Å")
    print(f"标准差 (Std Dev):   {np.std(res):.4f} Å")
    print("-" * 30)
    
    # 统计“及格率” (例如小于 0.1A 的占比)
    success_rate = (res < 0.1).mean() * 100
    print(f"精度优于 0.1Å 的占比: {success_rate:.2f}%")
    print("="*30)
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    args = parse_args()
    set_path(args)
    print(args)
    # 4. 加载物理先验 (OCP 核心)
    dataset_info = get_dataset_info(args) 
    args.lattice_mean = dataset_info['lattice_stats']['mean'].to(device, dtype)
    args.lattice_std = dataset_info['lattice_stats']['std'].to(device, dtype)
    args.displacement_std = dataset_info['displacement_std']
    args.pos_mean = dataset_info['pos_mean'].to(device, dtype)
    args.pos_std = dataset_info['pos_std'].to(device, dtype)

    # 5. 维度自动计算
    #args.num_atom_types = len(dataset_info['atom_decoder'])

    args.in_node_nf = args.num_atom_types + 3
    args.context_node_nf = 0
    lut = torch.zeros(args.num_atom_types, dtype=torch.long)

    # 2. 填充映射关系
    for atomic_nb, id_idx in dataset_info['atom_nb_to_id'].items():
        
        atomic_nb = int(atomic_nb)
        id_idx = int(id_idx)

        lut[atomic_nb] = id_idx

    # 3. 将其挂载到 args 并在训练前送到 GPU
    args.atom_lut = lut.to(device)

    debug_eval_recon(args,device,dtype)


    #debug_vae(args, next(iter(get_dataloaders(args)['train'])), device)
    #debug_data_info(args)

    # loder = get_dataloaders(args)['val']
    # model,_,_ = get_goat(args,device)
    # print(model.device)
    # debug_test(args= args,loader=loder,epoch=1,eval_model=model,device=device,dtype=dtype)

    
    
