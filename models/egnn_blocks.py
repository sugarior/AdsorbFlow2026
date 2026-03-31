import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from models.egnn import EGNN, GNN


class EGNN_dynamics(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super().__init__()
        self.mode = mode
        
        self.egnn = EGNN(
            in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
            hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method)
        self.in_node_nf = in_node_nf
        # elif mode == 'gnn_dynamics':
        #     self.gnn = GNN(
        #         in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
        #         hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
        #         act_fn=act_fn, n_layers=n_layers, attention=attention,
        #         normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.context_node_nf = context_node_nf
        # self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        # self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_index,edge_attr, context=None,batch_idx=None):
        """
        核心修改：将所有物理环境信息（边、晶格、掩码）通过闭包锁死
        """
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_index, edge_attr, context, batch_idx)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_index,edge_attr, context=None,batch_idx=None):
        
        xh = xh* node_mask
        x = xh[:, 0:self.n_dims].clone()
        h = xh[:, self.n_dims:].clone()

        original_h_dims = h.size(1)

        #这一段处理时间看不懂
        if self.condition_time:
            if t.numel() == 1: # 标量情况
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else: # 向量情况 [Batch_Size]
                # 利用 batch_idx 像发传单一样把 t 发给每个原子
                h_time = t[batch_idx]# [Total_N, 1]
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            h_context = context[batch_idx]
            h = torch.cat([h, h_context], dim=1)

        
        h_final, x_final = self.egnn(h, x, edge_index, edge_attr, node_mask)
        vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        

        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]
        if self.condition_time:
            h_final = h_final[:, :-1]


        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        h_final = torch.zeros_like(h_final) 

        return torch.cat([vel, h_final], dim=1)
    

    # def get_adj_matrix(self, n_nodes, batch_size, device):
    #     if n_nodes in self._edges_dict:
    #         edges_dic_b = self._edges_dict[n_nodes]
    #         if batch_size in edges_dic_b:
    #             return edges_dic_b[batch_size]
    #         else:
    #             # get edges for a single sample
    #             rows, cols = [], []
    #             for batch_idx in range(batch_size):
    #                 for i in range(n_nodes):
    #                     for j in range(n_nodes):
    #                         rows.append(i + batch_idx * n_nodes)
    #                         cols.append(j + batch_idx * n_nodes)
    #             edges = [torch.LongTensor(rows).to(device),
    #                      torch.LongTensor(cols).to(device)]
    #             edges_dic_b[batch_size] = edges
    #             return edges
    #     else:
    #         self._edges_dict[n_nodes] = {}
    #         return self.get_adj_matrix(n_nodes, batch_size, device)


class EGNN_encoder(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=False):
        '''
        :param in_node_nf: Number of invariant features for input nodes.'''
        super().__init__()

        include_charges = int(include_charges)
        num_classes = in_node_nf - include_charges
        #print(f"DEBUG EGNNENCODER device:{device}")
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, out_node_nf=hidden_nf, 
                in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, out_node_nf=hidden_nf + 3, 
                in_edge_nf=0, hidden_nf=hidden_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)
        
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, out_node_nf * 2 + 1)).to(device)

        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        #self._edges_dict = {}
        # self.condition_time = condition_time

        self.out_node_nf = out_node_nf

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_index, edge_attr, context, batch_idx):
        """修正点：必须包含 edge_index 等参数"""
        def fwd(time, state):
            return self._forward(
                state, 
                node_mask, 
                edge_index, 
                edge_attr, 
                context, 
                batch_idx
            )
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_index,edge_attr, context=None,batch_idx=None):
        '''
        xh: [Total_N, 3 + 109]
        node_mask: [Total_N, 1]
        '''
        x = xh[:, :self.n_dims].clone() # [Total_N, 3]
        h = xh[:, self.n_dims:].clone() # [Total_N, 109]

        x = x * node_mask
        h = h * node_mask

        if context is not None:
            if batch_idx is None:
                raise ValueError("在 2D 堆叠模式下，注入 context 必须提供 batch_idx")
            
            # 将 [Batch_Size, d_ctx] 映射到 [Total_N, d_ctx]
            # 这样每个原子都能拿到属于自己那个分子的全局上下文
            context_ext = context[batch_idx]
            h = torch.cat([h, context_ext], dim=1) # 特征列数增加
        h_final, x_final = self.egnn(
            h, x, edge_index, 
            edge_attr=edge_attr, 
            node_mask=node_mask
        )

        #print(f"DEBUG: h_final device: {h_final.device}")
        #print(f"DEBUG: final_mlp device: {next(self.final_mlp.parameters()).device}")

        params = self.final_mlp(h_final) # [Total_N, 1 + 2 * out_node_nf]
        params = params * node_mask # 再次 Mask 保证安全

        vel_mean = x_final

        vel_logvar = params[:, :1] # [Total_N, 1]
        # 强制限制在 [-10, 10] 之间，防止 exp 爆炸
        vel_logvar = torch.clamp(vel_logvar, min=-10, max=10)
        molecule_vel_logvar = scatter_mean(vel_logvar, batch_idx, dim=0) # [Batch_Size, 1]
        
        # 将分子级的方差广播回节点级，并转为标准差
        vel_std = torch.exp(0.5 * molecule_vel_logvar)[batch_idx] # [Total_N, 1]

        # h_mean 和 h_std (特征分布)
        h_mean = params[:, 1 : 1 + self.out_node_nf] # [Total_N, out_node_nf]
        h_logvar_h = params[:, 1 + self.out_node_nf :]
        h_std = torch.exp(0.5 * h_logvar_h) # [Total_N, out_node_nf]

        # 6. 数值稳定性补丁
        if torch.any(torch.isnan(vel_std)):
            vel_std = torch.ones_like(vel_std) * 0.01



        return vel_mean, vel_std, h_mean, h_std
    # def get_adj_matrix(self, n_nodes, batch_size, device):
    #     if n_nodes in self._edges_dict:
    #         edges_dic_b = self._edges_dict[n_nodes]
    #         if batch_size in edges_dic_b:
    #             return edges_dic_b[batch_size]
    #         else:
    #             # get edges for a single sample
    #             rows, cols = [], []
    #             for batch_idx in range(batch_size):
    #                 for i in range(n_nodes):
    #                     for j in range(n_nodes):
    #                         rows.append(i + batch_idx * n_nodes)
    #                         cols.append(j + batch_idx * n_nodes)
    #             edges = [torch.LongTensor(rows).to(device),
    #                      torch.LongTensor(cols).to(device)]
    #             edges_dic_b[batch_size] = edges
    #             return edges
    #     else:
    #         self._edges_dict[n_nodes] = {}
    #         return self.get_adj_matrix(n_nodes, batch_size, device)


class EGNN_decoder(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True):
        super().__init__()

        include_charges = int(include_charges)
        num_classes = out_node_nf - include_charges

        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, out_node_nf=out_node_nf, 
                in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, out_node_nf=out_node_nf + 3, 
                in_edge_nf=0, hidden_nf=hidden_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        #self._edges_dict = {}
        # self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_index, edge_attr, context, batch_idx):
        """修正点：必须包含 edge_index 等参数"""
        def fwd(time, state):
            return self._forward(
                state, 
                node_mask, 
                edge_index, 
                edge_attr, 
                context, 
                batch_idx
            )
        return fwd


    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_index,edge_attr, context=None,batch_idx=None):
       

        x = xh[:, :self.n_dims].clone() # 潜坐标 [Total_N, 3]
        h = xh[:, self.n_dims:].clone() # 潜特征 [Total_N, latent_nf]
    
        x = x * node_mask
        h = h * node_mask

        if context is not None:
            if batch_idx is None:
                raise ValueError("在 2D 模式下，Decoder 注入 context 必须提供 batch_idx")
            
            context_ext = context[batch_idx]
            h = torch.cat([h, context_ext], dim=1) 

        h_dec, x_dec = self.egnn(
            h, x, edge_index, 
            edge_attr=edge_attr, 
            node_mask=node_mask
        )

        h_final = h_dec

        vel = x_dec
        return vel, h_final
    
    # def get_adj_matrix(self, n_nodes, batch_size, device):
    #     if n_nodes in self._edges_dict:
    #         edges_dic_b = self._edges_dict[n_nodes]
    #         if batch_size in edges_dic_b:
    #             return edges_dic_b[batch_size]
    #         else:
    #             # get edges for a single sample
    #             rows, cols = [], []
    #             for batch_idx in range(batch_size):
    #                 for i in range(n_nodes):
    #                     for j in range(n_nodes):
    #                         rows.append(i + batch_idx * n_nodes)
    #                         cols.append(j + batch_idx * n_nodes)
    #             edges = [torch.LongTensor(rows).to(device),
    #                      torch.LongTensor(cols).to(device)]
    #             edges_dic_b[batch_size] = edges
    #             return edges
    #     else:
    #         self._edges_dict[n_nodes] = {}
    #         return self.get_adj_matrix(n_nodes, batch_size, device)
