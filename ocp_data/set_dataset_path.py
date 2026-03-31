


def set_path(args):
    if args.dataset == 'oc20':
        return
    elif args.dataset == 'oc22':
        args.train_src = '/sda1/zzwang/decopm3/oc22data/is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train'
        args.val_src = '/sda1/zzwang/decopm3/oc22data/is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/val_id'
        args.test_src = '/sda1/zzwang/decopm3/oc22data/is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/test_id'
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")