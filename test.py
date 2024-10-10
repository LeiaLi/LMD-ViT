import torch
from model_hw import LMD
from test_flops import test_flops
import glob
import numpy as np
import torch.nn.functional as F
import pyiqa
import argparse
import os
import cv2
from skimage import img_as_ubyte
from tqdm import tqdm
import utils
from utils.timer import Timer
from utils import image_utils
from utils.image_utils import batch_weighted_PSNR, SSIM, cal_prec_accu_reca   
from PIL import Image
import math
import torchvision.transforms.functional as TF

def expand2rect(timg, blur_mask, factor=16.0):
    _, _, h, w = timg.size()

    Xh = int(math.ceil(h / float(factor)) * factor)
    Xw = int(math.ceil(w / float(factor)) * factor)

    img = torch.zeros(1, 3, Xh, Xw).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, Xh, Xw).type_as(timg)

    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((Xh - h) // 2):((Xh - h) // 2 + h), ((Xw - w) // 2):((Xw - w) // 2 + w)] = timg
    mask[:, :, ((Xh - h) // 2):((Xh - h) // 2 + h), ((Xw - w) // 2):((Xw - w) // 2 + w)].fill_(1)
    # mask[:, :, ((Xh - h) // 2):((Xh - h) // 2 + h), ((Xw - w) // 2):((Xw - w) // 2 + w)] = torch.unsqueeze(blur_mask[:,0,:,:],dim=1)

    return img, mask

def main(args):
    
    test_paths = sorted(glob.glob(args.input_dir+ '/*.npy'))###
    print(len(test_paths))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    net = LMD(
            img_size=512, embed_dim=32, win_size=8, token_projection='linear',
            token_mlp='lefflocal',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True, dd_in=3,
            drop_path_rate=0.1, prune_loc=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    
    if args.ckpt_path is not None:
        
        checkpoint = torch.load(args.ckpt_path)
        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        checkpoint['state_dict'] = new_state_dict
        if isinstance(net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            net.module.load_state_dict(checkpoint["state_dict"], strict=True)
        else:
            net.load_state_dict(checkpoint["state_dict"], strict=True)
        net = net.to(device)
            
    os.makedirs(args.result_dir, exist_ok=True)

    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = SSIM(window_size=11, size_average=False)
    psnr_count, ssim_count, w_psnr_count, w_ssim_count, sample_count = 0, 0, 0, 0, 0
    
    with torch.no_grad():
        net.eval()
        for idx, test_path in enumerate(tqdm(test_paths)):
            sample_count += 1
            split_path = test_path.split('/') ####

            sample = np.load(test_path, allow_pickle=True).item()
            blur = sample['img_blur']
            gt = sample['img_gt']
            mask = np.tile(np.uint8(sample['blur_mask'] * 255)[:, :, None], [1, 1, 3])
            # print(mask, np.min(mask), np.max(mask))
                
            blur = torch.from_numpy(blur).permute(2,0,1).float().div(255).unsqueeze(dim=0).to(device)
            gt = torch.from_numpy(gt).permute(2,0,1).float().div(255).unsqueeze(dim=0).to(device)
            mask = torch.from_numpy(mask).permute(2,0,1).float().div(255).unsqueeze(dim=0).to(device)
            mask_shortcut = mask # 3 channels
            # breakpoint()
            blur, mask = expand2rect(blur, mask, factor=128)
            _, _, H, W = blur.shape
            # test_flops(net, blur)
            
            if sample_count == 1:
                deblur, _1, pred_score_list, decision_list = net(blur)
                mask_shortcut_ = torch.unsqueeze(mask_shortcut[:,0,:,:],dim=0)            

            with Timer(enable=True, name='test'):
                deblur, _1, pred_score_list, decision_list = net(blur)
                mask_shortcut_ = torch.unsqueeze(mask_shortcut[:,0,:,:], dim=0)
                    
            deblur = torch.masked_select(deblur, mask.bool()).reshape(1, 3, gt.shape[2], gt.shape[3])
            deblur = deblur[:, :, :H, :W]
            deblur_save = torch.clamp(deblur,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            # cv2.imwrite(f'{args.result_dir}/{sample_count}.png',cv2.cvtColor(img_as_ubyte(deblur_save), cv2.COLOR_RGB2BGR))
            
            deblur = torch.clamp(deblur, 0, 1)
            gt = torch.clamp(gt, 0, 1)
            psnr_count += psnr_metric(deblur, gt).item()
            ssim_count += -ssim_metric(deblur, gt).sum().item()
            w_psnr_count += batch_weighted_PSNR(deblur, gt, mask_shortcut).sum().item()
            w_ssim_count += -ssim_metric(deblur, gt, mask_shortcut).sum().item()
        psnr_ave = psnr_count / sample_count
        ssim_ave = ssim_count / sample_count
        w_psnr_ave = w_psnr_count / sample_count
        w_ssim_ave = w_ssim_count / sample_count
        print('average psnr:', psnr_ave, 'average ssim:', ssim_ave, 'weighted psnr:', w_psnr_ave, 'weighted ssim:s', w_ssim_ave)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single Image Local Motion Deblurring using LMD-ViT')
    parser.add_argument('--input_dir', default='./val_data', type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='/./results', type=str, help='Directory for results')
    parser.add_argument('--ckpt_path', default='./ckpt/model_LMDVIT.pth', type=str, help='Path to weights')
    parser.add_argument('--dataset', default='ReLoBlur', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
    args = parser.parse_args()
    main(args)
