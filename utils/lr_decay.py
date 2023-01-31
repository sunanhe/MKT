def param_groups_lrd(model, fix_layer, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}  

    if hasattr(model, "blocks"):
        num_layers = len(model.blocks) + 1
    elif hasattr(model, "transformer"):
        num_layers = model.transformer.layers + 1
    else:
        num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
        # import pdb; pdb.set_trace()
        if hasattr(model, "blocks"):
            layer_id = get_layer_id_for_vit(n, num_layers)
        else:
            layer_id = get_layer_id_for_clip(n, num_layers)

        if layer_id > fix_layer:

            group_name = "layer_%d_%s" % (layer_id, g_decay)

            if group_name not in param_group_names:
                this_scale = layer_scales[layer_id]

                param_group_names[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }
                param_groups[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }

            param_group_names[group_name]["params"].append(n)
            param_groups[group_name]["params"].append(p)

    return list(param_groups.values()), param_group_names


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers

def get_layer_id_for_clip(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """

    if name in ['cls_token', 'pos_embed', "class_embedding"]: 
        return 0
    elif name.startswith('patch_embed'):  
        return 0
    elif name.startswith('conv1'):  
        return 0
    elif name.startswith('ln_pre'):  
        return 0
    elif name.startswith('positional_embedding'):  
        return 0
    elif name.startswith('transformer.resblocks'):
        return int(name.split('.')[2]) + 1
    else:
        return num_layers


