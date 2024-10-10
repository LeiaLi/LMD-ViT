import traceback

import torch
import torch.nn as nn
import os
from collections import OrderedDict

# from model_hw_ry import UformerPrune
from model_hw import LMD


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True


def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)


def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir, "model_epoch_{}_{}.pth".format(epoch, session))
    torch.save(state, model_out_path)


def load_pretrained(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if 'module.' in k else k
        new_state_dict[name] = v
    checkpoint['state_dict'] = new_state_dict
    # del checkpoint["state_dict"]['output_proj.proj.0.weight']
    # del checkpoint["state_dict"]['output_proj.proj.0.bias']
    # del_ckpt = []
    # for k in checkpoint["state_dict"]:
    #     if 'score_predictor' in k:
    #         del_ckpt.append(k)
    # for del_k in del_ckpt:
    #     del checkpoint["state_dict"][del_k]
    # for k in checkpoint["state_dict"]:
    #     if 'mlp' in k:
    #         del_ckpt.append(k)
    # for del_k in del_ckpt:
    #     del checkpoint["state_dict"][del_k]
        
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model.module.load_state_dict(checkpoint["state_dict"], strict=True)
        # model.module.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        # model.load_state_dict(checkpoint["state_dict"], strict=False)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if 'module.' in k else k
        new_state_dict[name] = v
    checkpoint['state_dict'] = new_state_dict
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model.module.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])
    print(f"| Successfully load model.")


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint.get("global_steps", 0)
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from model_hw import LMD

    arch = opt.arch

    print('You choose ' + arch + '...')
    if arch == 'LMD-ViT':
        model_restoration = LMD(
            img_size=opt.train_ps, embed_dim=32, win_size=8, token_projection='linear',
            token_mlp='lefflocal',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True, dd_in=opt.dd_in,
            drop_path_rate=opt.drop_path_rate, prune_loc=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    else:
        raise Exception("Arch error!")

    return model_restoration


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print(f'| Trainable Parameters: %.3fM' % (num_params / 1e6))
    return num_params
