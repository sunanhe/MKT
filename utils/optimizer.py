import torch

import utils.lr_decay as lrd

def build_optimizer(args, model):

    params_name = None

    params, param_group_names = lrd.param_groups_lrd(model, args.fix_layer, args.weight_decay,
        layer_decay=args.layer_decay
    )
    params_name = []
    for k, v in param_group_names.items():
        params_name += v["params"]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    for name, param in model.named_parameters():
        if name not in params_name:
            param.requires_grad = False

    return optimizer