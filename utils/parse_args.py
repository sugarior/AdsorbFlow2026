import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='AdsorbFlow Training Script')
    parser.add_argument('--exp_name', type=str, default='debugflow_3_vae_648_All')

    #账户设置
    parser.add_argument('--wandb_project', type=str, default='AdsorbFlow2026-flow')
    parser.add_argument('--wandb_entity', type=str, default='zzwang')

    #模型设置
    parser.add_argument('--model', type=str, default='egnn_vae')
    parser.add_argument('--nf', type=int, default=128)
    parser.add_argument('--latent_nf', type=int, default=64)
    parser.add_argument('--context_node_nf', type=int, default=0)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--attention', action='store_true',default=False)
    parser.add_argument('--tanh', action='store_true',default=True)
    parser.add_argument('--norm_constant', type=float, default=1.0)
    parser.add_argument('--inv_sublayers', type=int, default=2, help='每个 Block 内部特征更新层数')
    parser.add_argument('--sin_embedding', action='store_true',default=False)
    parser.add_argument('--normalization_factor', type=float, default=1.0)
    parser.add_argument('--aggregation_method', type=str, default='mean', choices=['mean', 'sum', 'max'])
    parser.add_argument('--kl_weight', type=float, default=0.01)
    parser.add_argument('--tag_nf', type=int, default=3)
    #数据设置
    parser.add_argument('--dataset', type=str, default='oc22')
    parser.add_argument('--train_src', type=str, default='data/oc22/train')
    parser.add_argument('--val_src', type=str, default='data/oc22/val')
    parser.add_argument('--test_src', type=str, default='data/oc22/test')
    parser.add_argument('--cutoff', type=float, default=5.0)
    parser.add_argument('--oc22_stats_path', type=str, default='./configs/oc22_stats.pt')
    parser.add_argument('--num_atom_types', type=int, default=100)
    parser.add_argument('--cif_save_path',type = str ,default='/sda1/zzwang/decopm3/AdsorbFlow2026/cif/debug_flow_vae648')
    parser.add_argument(
        '--generate_lmdb_path',
        type=str,
        default='/sda1/zzwang/decopm3/AdsorbFlow2026/lmdb/debug_flow_648',
        help='若指定为目录：将 flow 推理得到的物理坐标写入该目录下 data.lmdb（键 length + 0..N-1，与 OCP LmdbDataset 多文件之一格式一致）',
    )
    parser.add_argument(
        '--generate_lmdb_src',
        type=str,
        default='/sda1/zzwang/decopm3/oc22data/is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/val_id',
        help='生成 LMDB 时读取的源数据路径（LMDB 目录或单文件）；不填则使用 val_src',
    )
    #文件夹设置
    parser.add_argument('--output_dir', type=str, default='./outputs')


    #训练设置
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--vae_path',type=str ,default='/sda1/zzwang/decopm3/AdsorbFlow2026/outputs/debug_vae_1_kl1-nlayer8/version_1/checkpoints/best_model.pt')
    parser.add_argument('--test_checkpoint',type = str,default='/sda1/zzwang/decopm3/AdsorbFlow2026/outputs/debugflow_3_vae_648/version_2/checkpoints/best_model.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--test_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--tqdm_ncols', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--probabilistic_model',default='flow', choices=['vae', 'flow'])
    parser.add_argument('--global_seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default='float32', choices=['float16', 'float32', 'float64'])
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--diffusion_steps',type=int ,default= 10000)
    parser.add_argument('--diffusion_loss_type',type=str,default="ot")
    parser.add_argument('--time_nf',type=int,default=12)
    
    args = parser.parse_args()
    return args