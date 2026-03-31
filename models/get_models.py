import torch
from models.egnn_blocks import EGNN_encoder, EGNN_decoder,EGNN_dynamics
from models.egnn_vae import EnHierarchicalVAE
from models.goat import GeometricOptimalTransportFlow
from os.path import join

def get_autoencoder(args, device):
    in_node_nf = args.num_atom_types + args.tag_nf + args.context_node_nf+6
    # print('Autoencoder models are _not_ conditioned on time.')
    print(f"debug device in get_autoencoder: {device}")
    hidden_nf = args.nf 
    latent_nf = args.latent_nf

    print(f"get_autoencoder: in_node_nf={in_node_nf}, hidden_nf={hidden_nf}, latent_nf={latent_nf}")

    encoder = EGNN_encoder(
        in_node_nf=in_node_nf, context_node_nf=args.context_node_nf, out_node_nf=args.latent_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh,  norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method
    )

    decoder = EGNN_decoder(
        in_node_nf=args.latent_nf, context_node_nf=args.context_node_nf, out_node_nf=in_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method
    )

    vae = EnHierarchicalVAE(
        encoder=encoder,
        decoder=decoder,
        in_node_nf=in_node_nf,
        n_dims=3,
        latent_node_nf=args.latent_nf,
        kl_weight=args.kl_weight
    )

    return vae

def get_optim(args, model):
    if args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    return optim


def get_goat(args, device, dataset_info=None, train_loader=None):
    model = None
    if args.probabilistic_model == 'vae':
        model = get_autoencoder(args, device)
    elif args.probabilistic_model =='flow':
        in_node_nf = args.latent_nf + args.time_nf
        feature_model = get_autoencoder(args, device)

        net_dynamics = EGNN_dynamics(
        in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)


        if args.vae_path is not None:
            feature_model_dict = torch.load(args.vae_path, map_location='cpu')
            #print(f'DEBUG dict ={feature_model_dict}')
            feature_model.load_state_dict(feature_model_dict['model'])
            print('load from ', args.vae_path)
        else:
            raise ValueError("训练 Flow 模式必须提供 --vae_path！")

        flow = GeometricOptimalTransportFlow(
        dynamics=net_dynamics,
        in_node_nf=in_node_nf,
        n_dims=3,
        vae=feature_model,
        timesteps=args.diffusion_steps,
        loss_type=args.diffusion_loss_type,
        time_nf=args.time_nf,
        device=device,
        #distill=args.distill,
        )
        
        model = flow

    return model, None, None