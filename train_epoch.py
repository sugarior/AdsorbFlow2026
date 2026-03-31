import torch
from tqdm import tqdm
from torch.nn import functional as F
import utils.utilis_func as uf
import wandb


def prepare_batch_data(args, batch, device, dtype,partition = 'Train'):
    '''
    返回x(relaxed), h, x_init,u_target,node_mask, edge_index,edge_attr, adsorbate_mask
    '''
    x = batch.pos_relaxed.to(device, dtype)
    x_init = batch.pos.to(device, dtype)

    x = uf.normalize_pos(x, args.pos_mean, args.pos_std)
    x_init = uf.normalize_pos(x_init, args.pos_mean, args.pos_std)

    batch_idx = batch.batch.to(device)

    # 2. 节点特征 h 的三位一体拼接 [原子身份 + 角色Tags + 晶格上下文]
    
    atom_indices = args.atom_lut[batch.atomic_numbers.long()]  # [N]
    h_atom = F.one_hot(atom_indices, num_classes=args.num_atom_types).to(device,dtype) # [N, 100]
    h_tag = F.one_hot(batch.tags.long(), num_classes=3).to(device,dtype) # [N, 3]

    # 晶格上下文特征（以晶格为中心，统计不同类型原子在不同距离范围内的数量）
    lattice_raw = torch.cat([batch.lengths, batch.angles], dim=-1).to(device, dtype)

    l_mean = args.lattice_mean.to(device, dtype)
    l_std = args.lattice_std.to(device, dtype)

    # 3. 现在的运算就安全了
    lattice_norm = (lattice_raw - l_mean) / (l_std + 1e-8)

    h_lattice = lattice_norm[batch_idx] # [N, 6]

    h = torch.cat([h_atom, h_tag, h_lattice], dim=1) # [N, 109]

    # 3. 边特征 edge_attr 的拼接 [距离 + PBC 偏移 + 角度信息]
    edge_index = batch.edge_index.to(device)
    cell = batch.cell.to(device, dtype)
    cell_offsets = batch.cell_offsets.to(device, dtype)

    #不是很懂
    edge_batch_idx = batch_idx[edge_index[0]]
    edge_attr = torch.einsum('bi,bij->bj', cell_offsets, cell[edge_batch_idx])

    
    edge_attr = (edge_attr / (args.pos_std.to(device) + 1e-8)) 


    # node_mask: [N, 1] (PyG 稀疏模式下默认为 1)
    node_mask = torch.ones((x.size(0), 1), device=device, dtype=dtype)
    # adsorbate_mask: [N, 1] (只有吸附物原子为 1，用于锁定地基)
    adsorbate_mask = (batch.tags == 2).view(-1, 1).to(device, dtype)

    with torch.no_grad():
        # 计算 (x_relaxed - x_init) 的最小镜像位移
        u_raw = uf.get_pbc_diff(x, x_init, cell, batch_idx)
        # 利用统计出的位移标准差进行归一化，解决梯度消失
        u_target = u_raw / (args.displacement_std + 1e-8)

    return x, h, x_init, u_target, node_mask, edge_index, edge_attr, adsorbate_mask,batch_idx


def train_epoch(args, loader, epoch, model, model_ema, ema, device, dtype, optim, gradnorm_queue):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    loader = tqdm(loader, desc=f"Epoch {epoch} - Training", ncols = args.tqdm_ncols)

    for i, batch in enumerate(loader):

        x,h,x_init,u_target,node_mask,edge_index,edge_attr,adsorbate_mask,batch_idx= prepare_batch_data(args, batch, device, dtype, partition='Train')
        # A. 前向传播与损失计算
        loss ,loss_dict= model(x=x,x_init=x_init,h=h, node_mask =node_mask, edge_index=edge_index, edge_attr=edge_attr, context=None, batch_idx=batch_idx, adsorbate_mask=adsorbate_mask)
        
        # B. 反向传播与优化器更新
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optim.step()
        
        # C. EMA 更新
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)
        
        # D. 记录损失
        current_loss = loss.item()
        total_loss += current_loss
        loader.set_postfix({
            "loss": f"{current_loss:.4f}"
            # "rec_error": f"{loss_dict['rec_error']:.4f}",
            # "kl_h": f"{loss_dict['kl_h']:.4f}",
            # "kl_x": f"{loss_dict['kl_x']:.4f}"
        })
            
        current_global_step = epoch * n_batches + i
        # B. 每隔 N 个 Step 把数据发给 WandB
        # 建议不要每个 step 都发，防止网络拥堵，通常每 10-50 step 发一次
        if i % args.log_every == 0:
            wandb.log({
                "train/step": current_global_step,
                "train/step_loss": current_loss,
                # "train/rec_error": loss_dict['rec_error'].item(),
                # "train/kl_h": loss_dict['kl_h'].item(),
                # "train/kl_x": loss_dict['kl_x'].item(),
                })
    
    avg_loss = total_loss / n_batches
    return avg_loss

def test_adsorb(args, loader, epoch, eval_model, device, dtype):
    eval_model.eval()
    total_loss = 0.0

    loader = tqdm(loader, desc=f"Epoch {epoch} - Valiating", ncols = args.tqdm_ncols)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # A. 前向传播与损失计算
            x,h,x_init,u_target,node_mask,edge_index,edge_attr,adsorbate_mask,batch_idx= prepare_batch_data(args, batch, device, dtype, partition='Val')
            loss,loss_dict = eval_model(x=x,x_init=x_init,h=h, node_mask =node_mask, edge_index=edge_index, edge_attr=edge_attr, context=None, batch_idx=batch_idx, adsorbate_mask=adsorbate_mask)
            # 这里需要替换为实际的损失计算逻辑

            # B. 记录损失
            current_loss = loss.item()
            total_loss += current_loss
            loader.set_postfix({
                "loss": f"{current_loss:.4f}",
                # "rec_error": f"{loss_dict['rec_error']:.4f}",
                # "kl_h": f"{loss_dict['kl_h']:.4f}",
                # "kl_x": f"{loss_dict['kl_x']:.4f}"
            })

    
    avg_loss = total_loss / len(loader)
    return avg_loss

@torch.no_grad()
def evaluate_reconstruction(model, loader, args,device,dtype):
    model.eval()
    results = []

    loader = tqdm(loader, desc=f" evaluating", ncols = args.tqdm_ncols)

    i =0

    for batch in loader:
        batch.to(device)
        # A. 炼金与归一化
        x_true_phys = batch.pos_relaxed.to(device) 
        x_init_phys = batch.pos.to(device)
        x,h,x_init,u_target,node_mask,edge_index,edge_attr,adsorbate_mask,batch_idx= prepare_batch_data(args, batch, device, dtype, partition='Test')

        # B. 走一遍 VAE 闭环 (x -> z -> x_rec)
        # 注意：这里要用不带噪声的采样，即直接取 mu
        z_x_mu, _, z_h_mu, _ = model.encode(x, h, node_mask, edge_index, edge_attr, None, batch_idx)
        z_xh = torch.cat([z_x_mu, z_h_mu], dim=1)
        x_rec_norm, h_rec = model.decoder._forward(z_xh, node_mask, edge_index, edge_attr, None, batch_idx)

        # C. 反归一化还原回物理空间
        x_rec_phys = uf.denormalize_pos(x_rec_norm, args.pos_mean, args.pos_std)

        if i % 10 == 0:
        # 提取当前 Batch 的第一个分子（假设你想看 Batch 里的第 0 个）
        # 注意：需要根据 batch_idx 切片出属于该分子的原子
            mask = (batch_idx == 0)

            #print(f'DEBUG mask device {mask.device},batch_atomnum {batch.atomic_numbers.device}')
            
            uf.export_comparison_cif(
                pos_true = x_true_phys[mask], 
                pos_rec = x_rec_phys[mask], 
                atomic_numbers = batch.atomic_numbers[mask], 
                cell = batch.cell[0], # 取该分子的 cell
                save_path = args.cif_save_path,
                sample_idx = i,
                adsorbate_mask= adsorbate_mask[mask]

            )
            
        i+=1

        x_rec_xy = x_rec_phys[:, :2]
        x_true_xy = x_true_phys[:, :2]
        # D. 计算吸附物 RMSD
        # 只取 tag=2 的部分
        dist = torch.norm(x_rec_xy- x_true_xy, p=2, dim=-1)
# 这一行算出了当前 batch 的平均位移误差 (埃)
        current_mae = (dist * adsorbate_mask.squeeze()).sum() / (adsorbate_mask.sum() + 1e-8)
        
        results.append(current_mae.item())

    return results

def load_model_weights(model, checkpoint_path, device):
    """
    大师级权重加载器：
    1. 自动处理 DDP 前缀。
    2. 处理不同的字典键名（'model' 或 'state_dict'）。
    3. 建议优先加载 'model_ema' 以获得更平滑的评估效果。
    """
    print(f"📂 正在从 {checkpoint_path} 加载权重...")
    
    # map_location 确保在没有 GPU 的机器上也能 load
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 逻辑优先级：EMA 权重 > 普通权重
    if 'model_ema' in checkpoint:
        state_dict = checkpoint['model_ema']
        print("💡 检测到 EMA 权重，优先使用。")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint # 假设直接保存的是 state_dict

    

    # 加载到模型
    # strict=True 确保维度完全对齐，109维对109维
    model.load_state_dict(state_dict, strict=True)
    print("✅ 权重加载成功！")
    return model