import os
import numpy as np
import torch
import yaml
from thop import profile
from ptflops import get_model_complexity_info
import torch.nn.functional as F

def test_flops(model, input):
    # sample = np.load(test_paths[0], allow_pickle=True).item()
    # input = sample['img_blur']
    # input = torch.from_numpy(input).permute(2,0,1).float().div(255).unsqueeze(dim=0)
    # input = F.interpolate(input, scale_factor=0.25, mode='nearest')
    _, _, H, W = input.shape
    l_pad1 = ((H//32+1)*32-H)%32
    l_pad2 = ((W//32+1)*32-W)%32
    input = F.pad(input, [0, l_pad2, 0, l_pad1], mode='reflect').cuda()
    macs, params = profile(model, inputs=(input, ))
    from thop import clever_format
    macs, params = clever_format([macs, params], "%.3f")
    print('thop result:', macs, params)
    inp_shape = (3, H+l_pad1, W+l_pad2)
    macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=False)
    print('ptflops result:', macs, params)