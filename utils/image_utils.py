import torch
import numpy as np
import pickle
import cv2
from PIL import Image
import torchvision
import pdb
import math
import matplotlib as plt
import time
import torch.nn.functional as F

from criterions.ssim_loss import SSIM
from pyiqa import create_metric
from skimage import img_as_ubyte
import utils


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])


def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])


def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict


def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)


def load_npy(filepath):
    img = np.load(filepath)
    return img


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.
    return img


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def batch_PSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean([1, 2, 3]).sqrt()
    psnr = 20 * torch.log10(1 / rmse)
    return psnr


def batch_weighted_PSNR(tar_img, prd_img, mask):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    imdff2 = ((imdff ** 2) * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])
    rmse = imdff2.sqrt()
    psnr = 20 * torch.log10(1 / rmse)
    return psnr


class batch_iqa(torch.nn.Module):
    def __init__(self):
        super(batch_iqa, self).__init__()
        self.lpips = create_metric('lpips')
        self.niqe = create_metric('niqe')
        self.ssim_loss = SSIM(window_size=11, size_average=False)

    def forward(self, img1, img2, mask, average=True):
        mask = mask.repeat(1, 3, 1, 1)
        n = img1.shape[0]

        psnr_value = batch_PSNR(img1, img2).sum()
        weighted_psnr_value = batch_weighted_PSNR(img1, img2, mask).sum()
        ssim_value = -self.ssim_loss(img1, img2).sum()
        weighted_ssim_value = -self.ssim_loss(img1, img2, mask).sum()
        lpips_value = self.lpips(img1, img2).sum()
        # if not torch.isnan(img1).any() and  not torch.isnan(img2).any():
        #     niqe_value = self.niqe(img1, img2).sum()
        # else:
        #     niqe_value = torch.tensor(0, dtype=torch.float64).cuda()
        niqe_value = torch.tensor(0, dtype=torch.float64).cuda()
        iqa_list = [psnr_value, weighted_psnr_value, ssim_value, weighted_ssim_value, lpips_value, niqe_value]
        if average:
            iqa_list = [value / n for value in iqa_list]
        return iqa_list


def fft_loss(img_p, img_g, weight=None):
    pred_fft = torch.fft.rfft2(img_p)
    label_fft = torch.fft.rfft2(img_g)
    return F.l1_loss(pred_fft, label_fft, reduction='none').mean([1, 2, 3])


def l1_loss(img_p, img_g, weight=None):
    l1_loss = F.l1_loss(img_p, img_g, reduction='none')
    if weight is None:
        return l1_loss.mean([1, 2, 3])
    return (l1_loss * weight).sum([1, 2, 3]) / weight.sum([1, 2, 3]).clamp_min(1) / 3


def ssim_loss(img_p, img_g, weight=None):
    ssim = SSIM(window_size=11, size_average=False)
    ssim_loss_value = ssim(img_p, img_g, weight)
    return ssim_loss_value


def calc_losses(opt, img_p, img_g, blur_mask=None, suffix=''):
    lamda_dict = {'lambda_l1': 1.0, 'lambda_fft': 0.1, 'lambda_fft_amp': 0.0, 'lambda_ssim': 1.0, 'lambda_gate': 0}
    losses_ = {}
    total_loss = 0

    B, N_rpt = img_p.shape[0], 1
    if opt.shift_loss:
        shifted_dxys = opt.shifted_dxys
        N_rpt = len(shifted_dxys) ** 2
        img_shift_g = []
        for dy in shifted_dxys:
            for dx in shifted_dxys:
                img_shift_g.append(F.pad(img_g, [dx, -dx, dy, -dy], mode='reflect'))
        # [24, 3, 256, 256]
        img_g = torch.stack(img_shift_g, 1)
        img_g = img_g.flatten(0, 1)  # torch.Size([24, 9, 3, 256, 256])
        img_p = img_p[:, None, ...].repeat([1, N_rpt, 1, 1, 1])
        img_p = img_p.flatten(0, 1)
        if blur_mask is not None:
            blur_mask = blur_mask[:, None, ...].repeat([1, N_rpt, 1, 1, 1])
            blur_mask = blur_mask.flatten(0, 1)
    # metric_keys = [k for k in ['l1', 'ssim', 'fft'] if lamda_dict[f'lambda_{k}'] > 0]
    losses_['l1'] = l1_loss(img_p, img_g, blur_mask).reshape(B, N_rpt).amin(1)
    losses_['ssim'] = ssim_loss(img_p, img_g, blur_mask).reshape(B, N_rpt).amin(1)
    losses_['fft'] = fft_loss(img_p, img_g, blur_mask).reshape(B, N_rpt).amin(1)

    # losses_ = {k: getattr(f'{k}_loss')(img_p, img_g, blur_mask).reshape(B, N_rpt).amin(1)
    #             for k in metric_keys}
    losses_ = {k: v.mean() for k, v in losses_.items() if not torch.isnan(v).any()}
    # print(losses_)

    total_loss += sum([v * lamda_dict[f'lambda_{k}'] for k, v in losses_.items()])

    # for k, v in losses_.items():
    #     losses[f'{k}S'] = v.item()
    # losses = {f'{k}{suffix}': v for k, v in losses.items()}
    return {k: v.item() for k, v in losses_.items()}, total_loss

def cal_prec_accu_reca(blur_mask, pruned_map,index,prune_layer_num):
    precision = 0
    accuracy = 0
    recall = 0
    h = blur_mask.shape[2]
    w = blur_mask.shape[3]
    # print('blur_mask0:',blur_mask)
    blur_mask = blur_mask[:,:,(h-1436)//2:(h-1436)//2+1436, (w-2152)//2:(w-2152)//2+2152]
    pruned_map = pruned_map[:,:,(h-1436)//2:(h-1436)//2+1436, (w-2152)//2:(w-2152)//2+2152]
    # save_img(f'/home/tiger/nfs/xx/deblur/LMD-ViT-NIPS/gopro_sharp_results/visualizations_prune/blur_mask/{index}.png', img_as_ubyte(blur_mask[0].permute(1, 2, 0).cpu().numpy()))
    true_values = blur_mask > 0 # channel=3

    true_values_ = true_values[0].repeat(3,1,1)
    true_values_ = true_values_.permute(1, 2, 0).cpu().numpy()
    # cv2.imwrite(f'/home/tiger/nfs/xx/deblur/LMD-ViT-NIPS/relo_results/visualizations_prune/true_values/{index}_{prune_layer_num}.png',img_as_ubyte(true_values_))
    
    predictions = pruned_map > 0
    predictions_ = predictions[0].repeat(3,1,1)
    predictions_ = predictions_.permute(1, 2, 0).cpu().numpy()
    # cv2.imwrite(f'/home/tiger/nfs/xx/deblur/LMD-ViT-NIPS/gopro_sharp_results/visualizations_prune/predictions/{index}_{prune_layer_num}.png',img_as_ubyte(predictions_)) #.astype(int)*255
    predictions = predictions.reshape(-1).cpu().numpy()
    N = true_values.shape[0]

    TP = ((predictions == 1) & (true_values == 1)).sum()
    FP = ((predictions == 1) & (true_values == 0)).sum()
    FN = ((predictions == 0) & (true_values == 1)).sum()
    TN = ((predictions == 0) & (true_values == 0)).sum()
    # print('TP:',TP, 'FP:',FP, 'FN:',FN, 'TN:',TN)
    accuracy = (true_values == predictions).sum() / N
    if TP + FP > 0:
        precision = TP / (TP + FP)
    if TP + FN > 0:
        recall = TP / (TP + FN)
    print(f'{index}_{prune_layer_num}:', precision, accuracy, recall)
    return precision, accuracy, recall
        
        
def print_details(writer, prefix, global_step, train_pred_score_list, train_decision_list, 
                       gt, input_, restored, blur_mask, loss, index):
    precision_list = []
    for prune_layer_num in range(len(train_pred_score_list)):
        pred_score = train_pred_score_list[prune_layer_num]
        m = torch.nn.Softmax(dim=-1)
        pred_score_map = m(pred_score)
        pred_score_map0 = pred_score_map[..., :1]
        pred_score_map0 = print_batch_grid(pred_score_map0.cpu())
        pred_score_map1 = pred_score_map[..., 1:]
        pred_score_map1 = print_batch_grid(pred_score_map1.cpu())
        if prefix=="train":
            writer.add_image(f' {prefix} pred score map0 at layer_{prune_layer_num} / hw', pred_score_map0, global_step)
            writer.add_image(f' {prefix} pred score map1 at layer_{prune_layer_num} / hw', pred_score_map1, global_step)
        if prefix=="val":
            writer.add_image(f' {prefix}_{index} pred score map0 at layer_{prune_layer_num} / hw', pred_score_map0, global_step)
            writer.add_image(f' {prefix}_{index} pred score map1 at layer_{prune_layer_num} / hw', pred_score_map1, global_step)
        
        if len(train_decision_list)>0:#全不remove token
            train_decision_map = train_decision_list[prune_layer_num]
            
            resizd_d = F.interpolate(train_decision_map, scale_factor=blur_mask.shape[2]/train_decision_map.shape[2], mode='nearest')
            # pdb.set_trace()
            precision  = cal_precision(blur_mask, resizd_d)
            precision_list.append(precision)
            save_dir = f'/mnt/bn/ailabrenyi/projects/xx/deblur/LMD_ViT0511/logs/visulizationPrune2/{index}_{prune_layer_num}.png'
            resizd_d = resizd_d[0].permute(1, 2, 0).cpu().numpy()
            resizd_d = np.tile(resizd_d, [1,1,3])
            save_img(save_dir,img_as_ubyte(resizd_d))
            
            writer.add_image(f'{prefix}_{index} selected tokens at layer_{prune_layer_num} / global step', 
                         print_batch_grid(train_decision_map.cpu()), global_step)
    precision = sum(precision_list)/len(train_decision_list)     
    '''
    if prefix=="train":
        writer.add_image(f'{prefix} restored image / global step', print_batch_grid(restored.cpu()), global_step)
        writer.add_image(f'{prefix} mask image / global step', print_batch_grid(blur_mask.cpu()), global_step)
        writer.add_image(f'{prefix} input image / global step', print_batch_grid(input_.cpu()), global_step)
        writer.add_image(f'{prefix} gt image / global step', print_batch_grid(gt.cpu()), global_step)
           
    if prefix=="val":
        writer.add_image(f'{prefix}_{index} restored image / global step', print_batch_grid(restored.cpu()), global_step)
        writer.add_image(f'{prefix}_{index} mask image / global step', print_batch_grid(blur_mask.cpu()), global_step)
        writer.add_image(f'{prefix}_{index} input image / global step', print_batch_grid(input_.cpu()), global_step)
        writer.add_image(f'{prefix}_{index} gt image / global step', print_batch_grid(gt.cpu()), global_step)
    '''
    return precision
def print_batch_grid(batch_cpu_tensor):
    batch_numpy_tensor = batch_cpu_tensor.numpy() * 255
    if batch_numpy_tensor.shape[-1] == 1:
        batch_numpy_tensor.repeat(3, axis=3)
        batch_numpy_tensor = np.transpose(batch_numpy_tensor, (0, 3, 1, 2))
    if batch_numpy_tensor.shape[1] == 1:
        batch_numpy_tensor.repeat(3, axis=1)
    batch_numpy_tensor = np.clip(batch_numpy_tensor, 0, 255).astype(np.uint8)
    grid = torchvision.utils.make_grid(torch.from_numpy(batch_numpy_tensor))

    return grid

def cal_precision(blur_mask, pruned_map):
    # recall_list = []
    precision = 0
    true_values = blur_mask > 0
    # pdb.set_trace()
    true_values = true_values.reshape(-1).cpu().numpy()
    predictions = pruned_map > 0
    predictions = predictions.reshape(-1).cpu().numpy()
    N = true_values.shape[0]
    # accuracy = (true_values == predictions).sum() / N
    # self.results['mask_acc'].append(accuracy)
    TP = ((predictions == 1) & (true_values == 1)).sum()
    FP = ((predictions == 1) & (true_values == 0)).sum()
    FN = ((predictions == 0) & (true_values == 1)).sum()
    if TP + FP > 0:
        precision = TP / (TP + FP)
        # self.results['mask_precision'].append(precision)
    # if TP + FN > 0:
    #     recall = TP / (TP + FN)
        # recall_list.append(recall)
    return precision
        
def print_train_details(writer, global_step, train_pred_score_list, train_decision_list,
                        batch_size, input_, restored, blur_mask, loss):
    for prune_layer_num in range(len(train_pred_score_list)):
        pred_score = train_pred_score_list[prune_layer_num]
        m = torch.nn.Softmax(dim=-1)
        pred_score_map = m(pred_score)
        pred_score_map0 = pred_score_map[..., :1]
        pred_score_map0 = print_batch_grid(pred_score_map0.cpu())
        pred_score_map1 = pred_score_map[..., 1:]
        pred_score_map1 = print_batch_grid(pred_score_map1.cpu())

        # train_decision = train_decision_list[prune_layer_num]
        # train_decision_map = train_decision.reshape(
        #     train_decision.shape[0], int(math.sqrt(train_decision.shape[1])), int(math.sqrt(train_decision.shape[1])),train_decision.shape[2])

        start = time.time()  ###

        # for b in range(batch_size // 12):
        # for hw in range(pred_score.shape[1]):
        # writer.add_scalar(f'train pred score_diff 0_{b} at layer_{prune_layer_num} / hw',
        #                   pred_score[b, hw, 0] - pred_score[b, hw, 1], hw)
        # writer.add_scalar(f'train pred score_diff 1_{b} at layer_{prune_layer_num} / hw', pred_score_1[b,hw], hw)
        writer.add_image(f'train pred score map0 at layer_{prune_layer_num} / hw', pred_score_map0, global_step)
        writer.add_image(f'train pred score map1 at layer_{prune_layer_num} / hw', pred_score_map1, global_step)

        # writer.add_image(f'train selected tokens at layer_{prune_layer_num} / global step', 
        #                  print_batch_grid(train_decision_map.cpu()), global_step)

        end = time.time()
        print('time:', end - start)

    writer.add_image(f'train restored image / global step', print_batch_grid(restored.cpu()), global_step)
    writer.add_image(f'train mask image / global step', print_batch_grid(blur_mask.cpu()), global_step)
    writer.add_image(f'train input image / global step', print_batch_grid(input_.cpu()), global_step)
    writer.add_scalar(f'training loss / global step', loss, global_step)


def print_val_details(writer, global_step, val_pred_score_list, val_decision_list, batch_size, blur_mask, index):
    for prune_layer_num in range(len(val_pred_score_list)):
        pred_score = val_pred_score_list[prune_layer_num].cpu()
        val_decision = val_decision_list[prune_layer_num]
        val_decision_map = val_decision.reshape(
            val_decision.shape[0], int(blur_mask.shape[2] // 16), int(blur_mask.shape[-1] // 16))
        # mask_ratio = torch.sum(blur_mask, dim=(1,2,3))/(blur_mask.shape[2]*blur_mask.shape[3])
        # selected_token_ratio = torch.sum(val_decision,dim=1)/val_decision.shape[1]
        start = time.time()  ###
        for b in range(batch_size):
            for hw in range(pred_score.shape[1]):
                # writer.add_histogram(f'{index} validation pred score {b} at layer_{prune_layer_num} / hw', pred_score[b,hw].cpu(), hw)
                # writer.add_histogram(f'{index} validation decision of {b} at layer_{prune_layer_num} / hw', val_decision[b,hw].cpu(), hw)
                writer.add_scalar(f'{index} validation pred score {b} at layer_{prune_layer_num} / hw',
                                  pred_score[b, hw], hw)
                # writer.add_scalar(f'{index} validation decision of {b} at layer_{prune_layer_num} / hw', val_decision[b,hw], hw)
            # writer.add_scalar(f'{index} validation real blur ratio of {b} / global step', mask_ratio[b], global_step)
            # writer.add_scalar(f'{index} validation selected ratio of {b} at layer_{prune_layer_num}/ global step', selected_token_ratio[b], global_step)
        end = time.time()
        print('time:', end - start)
        writer.add_image(f'{index} validation selected tokens at layer_{prune_layer_num} / global step',
                         print_batch_grid(val_decision_map.cpu()), global_step)
