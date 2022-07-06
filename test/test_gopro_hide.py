import numpy as np
import os,sys,math
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))

import scipy.io as sio
from dataset.dataset_motiondeblur import *
import utils

from model import UNet,Uformer

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='Image motion deblurring evaluation on GoPro/HIDE')
parser.add_argument('--input_dir', default='/data1/wangzd/datasets/deblurring/GoPro/test/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/data1/wangzd/uformer_cvpr/results_release/deblurring/GoPro/Uformer_B/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='/data1/wangzd/uformer_cvpr/logs/motiondeblur/GoPro/Uformer_B_1129/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer_B', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--query_embed', action='store_true', default=False, help='query embedding for the decoder')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_deblur_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration= utils.get_arch(args)
# model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()


def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    
    return img, mask


with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))
        rgb_noisy, mask = expand2square(data_test[1].cuda(), factor=128) 
        filenames = data_test[2]

        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.masked_select(rgb_restored,mask.bool()).reshape(1,3,rgb_gt.shape[0],rgb_gt.shape[1])
        rgb_restored = torch.clamp(rgb_restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))

        psnr = psnr_loss(rgb_restored, rgb_gt)
        ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True)
        psnr_val_rgb.append(psnr)
        ssim_val_rgb.append(ssim)
        print("PSNR:",psnr,", SSIM:", ssim, filenames[0], rgb_restored.shape)
        utils.save_img(os.path.join(args.result_dir,filenames[0]+'.PNG'), img_as_ubyte(rgb_restored))
        with open(os.path.join(args.result_dir,'psnr_ssim.txt'),'a') as f:
            f.write(filenames[0]+'.PNG ---->'+"PSNR: %.4f, SSIM: %.4f] "% (psnr, ssim)+'\n')
psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
print("PSNR: %f, SSIM: %f " %(psnr_val_rgb,ssim_val_rgb))
with open(os.path.join(args.result_dir,'psnr_ssim.txt'),'a') as f:
    f.write("Arch:"+args.arch+", PSNR: %.4f, SSIM: %.4f] "% (psnr_val_rgb, ssim_val_rgb)+'\n')