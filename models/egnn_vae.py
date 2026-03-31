import torch
from torch.nn import functional as F
from utils import utilis_func as uf
from models.egnn_blocks import EGNN_encoder, EGNN_decoder
from models.egnn_diffusion import sum_except_batch, gaussian_KL, gaussian_KL_for_dimension
from torch_scatter import scatter_add

class EnHierarchicalVAE(torch.nn.Module):
    """
    The E(n) Hierarchical VAE Module.
    """

    def __init__(
            self,
            encoder: EGNN_encoder,
            decoder: EGNN_decoder,
            in_node_nf: int, n_dims: int, latent_node_nf: int,
            kl_weight: float,
            norm_values=(1., 1., 1.), norm_biases=(None, 0., 0.),
            ):
        super().__init__()


        self.encoder = encoder
        self.decoder = decoder

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.latent_node_nf = latent_node_nf
        self.num_classes = self.in_node_nf 
        self.kl_weight = kl_weight

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def compute_reconstruction_error(self, xh_rec, xh,batch_idx=None,adsorbate_mask = None):
        """Computes reconstruction error."""
        # Error on positions.
        x_rec = xh_rec[:, :self.n_dims]
        x = xh[:, :self.n_dims]
        error_x = torch.sum((x_rec - x) ** 2, dim=-1)


        weighted_error_x = error_x * adsorbate_mask.squeeze()

        #print(f"DEBUG0: x_rec max={x_rec.max().item():.2f}, x_true max={x.max().item():.2f}")

        h_cat_rec = xh_rec[:, self.n_dims : self.n_dims + self.num_classes]
        h_cat_true = xh[:, self.n_dims : self.n_dims + self.num_classes]
    
        target_ids = h_cat_true.argmax(dim=1)
        per_atom_error_h = F.cross_entropy(h_cat_rec, target_ids, reduction='none')

        #print(f"DEBUG1: per_atom_error_h sum={per_atom_error_h.sum().item():.2e}")

        num_nodes = scatter_add(torch.ones_like(batch_idx).float(), batch_idx, dim=0) + 1e-8
        num_ads = scatter_add(adsorbate_mask.squeeze(), batch_idx, dim=0) + 1e-8

        loss_x = scatter_add(weighted_error_x, batch_idx, dim=0) / num_ads
        loss_h = scatter_add(per_atom_error_h, batch_idx, dim=0) / num_nodes
       

        #print(f"models/egnn_vae.py compute_reconstruction_error debug: size of error : {error.shape}, size of batch_idx: {batch_idx.shape}")
        # if self.training:
        #     loss_x = scatter_add(weighted_error_x, batch_idx, dim=0) / num_ads
        #     loss_h = scatter_add(per_atom_error_h, batch_idx, dim=0) / num_nodes
        
        error = 50*loss_x + loss_h*0.1

        # print(f"DEBUG2: x_rec max={x_rec.max().item():.2f}, x_true max={x.max().item():.2f}")
        # print(f"DEBUG: error_x sum={weighted_error_x.sum().item():.2e}, error_h sum={per_atom_error_h.sum().item():.2e}, error ={error.sum().item():.2e}")

        return error

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        eps = torch.randn_like(mu) 
    
        # 应用 mask 确保噪声不会污染无效节点
        eps = eps * node_mask 
        
        # 执行 z = mu + sigma * eps
        return mu + sigma * eps

    def compute_loss(self, x, x_init,h, node_mask, edge_index, edge_attr, context, batch_idx,adsorbate_mask):
        """Computes an estimator for the variational lower bound."""
        """
            VAE 训练总入口：计算 ELBO (重构损失 + KL 散度)
            x: [Total_N, 3], h: [Total_N, 109]
        """

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h], dim=1)

        # Encoder output.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encode(x, h, node_mask, edge_index, edge_attr, context, batch_idx)

        kl_h_node = 0.5 * torch.sum(
            z_h_mu.pow(2) + z_h_sigma.pow(2) - 2 * torch.log(z_h_sigma + 1e-8) - 1, 
            dim=1
        )
        loss_kl_h = scatter_add(kl_h_node, batch_idx, dim=0)

        kl_x_node = 0.5 * torch.sum(
            z_x_mu.pow(2) + z_x_sigma.expand(-1, 3).pow(2) - 2 * torch.log(z_x_sigma.expand(-1, 3) + 1e-8) - 1,
            dim=1
        )
        loss_kl_x = scatter_add(kl_x_node, batch_idx, dim=0) # [Batch_Size]
        loss_kl = loss_kl_h + loss_kl_x

        num_nodes_per_mol = scatter_add(torch.ones_like(batch_idx).float(), batch_idx, dim=0) + 1e-8
        loss_kl = loss_kl / num_nodes_per_mol

        # Infer latent z.
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=1)
        
        z_xh_sigma = torch.cat([z_x_sigma.expand(-1, 3), z_h_sigma], dim=1)
        z_xh = self.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        

        # Decoder output (reconstruction).
        x_rec, h_rec = self.decoder._forward(
            z_xh, node_mask, edge_index, edge_attr, context, batch_idx
        )
        xh_rec = torch.cat([x_rec, h_rec], dim=1)
        loss_recon_scalar = self.compute_reconstruction_error(
            xh_rec, xh, batch_idx, adsorbate_mask
        )

        #print(f"models/egnn_vae.py compute_loss debug: size of loss_recon_scalar: {loss_recon_scalar.shape}, size of loss_kl: {loss_kl.shape}")
        # Combining the terms
        assert loss_recon_scalar.size() == loss_kl.size()
        total_loss = loss_recon_scalar + self.kl_weight * loss_kl

        assert len(total_loss.shape) == 1, f'{total_loss.shape} has more than only batch dim.'


        total_loss = total_loss.mean()  # 平均到 batch 级别
        
        return total_loss, {'loss_t': total_loss.squeeze(), 'rec_error': loss_recon_scalar.mean().squeeze(), 'kl_h': loss_kl_h.mean().squeeze(), 'kl_x': loss_kl_x.mean().squeeze()}

    def forward(self, x, x_init,h, node_mask=None, edge_index=None, edge_attr=None, context=None,batch_idx=None,adsorbate_mask=None):
        """
        Computes the ELBO if training. And if eval then always computes NLL.
        """

        #print('EGNN_VAE forward called with batch:', x,h)
        loss, loss_dict = self.compute_loss( x, x_init,h, node_mask, edge_index, edge_attr, context, batch_idx,adsorbate_mask)

        loss = loss

        return loss,loss_dict

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = torch.randn(n_nodes, self.n_dims)
    
        z_h = torch.randn(n_nodes, self.latent_node_nf)
        
        # 维度修正：使用 dim=1 拼接
        z = torch.cat([z_x, z_h], dim=1) 
        
        # 应用 node_mask 确保 Padding 为 0 (如果有的话)
        return z * node_mask

    def encode(self, x, h, node_mask=None, edge_index=None, edge_attr=None, context=None,batch_idx=None):
        """
            针对 OCP 2D 堆叠数据的等变编码逻辑
            x: [Total_N, 3]
            h: [Total_N, 109] (已经由 prepare_batch_data 拼好)
            node_mask: [Total_N, 1]
        """

        xh = torch.cat([x, h], dim=1) # [Total_N, 112]

        # Encoder output.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encoder._forward(xh, node_mask, edge_index, edge_attr, context,batch_idx)

        # bs, _, _ = z_x_mu.size()
        # sigma_0_x = torch.ones(bs, 1, 1).to(z_x_mu) * 0.0032
        # sigma_0_h = torch.ones(bs, 1, self.latent_node_nf).to(z_h_mu) * 0.0032

        return z_x_mu, z_x_sigma, z_h_mu, z_h_sigma

    def decode(self, z_xh, node_mask=None, edge_index=None, edge_attr=None, context=None):
        """Computes p(x|z)."""

        # Decoder output (reconstruction).
        x_recon, h_recon = self.decoder._forward(z_xh, node_mask, edge_index, edge_attr, context)


        xh = torch.cat([x_recon, h_recon], dim=1)

        x = xh[:, :self.n_dims] # 提取前 3 维
        if node_mask is not None:
            x = x * node_mask

        h_cat_logits = xh[:, self.n_dims : self.n_dims + self.num_classes]
        h_cat_idx = torch.argmax(h_cat_logits, dim=1)
        h_cat = F.one_hot(h_cat_idx, num_classes=self.num_classes).float()
        h_rest = xh[:, self.n_dims + self.num_classes:]

        h_final = torch.cat([h_cat, h_rest], dim=1)

        if node_mask is not None:
            h_final = h_final * node_mask
        return x, h_final

    @torch.no_grad()
    def reconstruct(self, x, h, node_mask=None, edge_mask=None, context=None):
        pass

    def log_info(self):
        """
        Some info logging of the model.
        """
        info = None
        print(info)

        return info