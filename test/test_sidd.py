import numpy as np
import os,sys
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
from dataset.dataset_denoise import *
import utils
import math
from model import UNet,Uformer

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='Image denoising evaluation on SIDD')
parser.add_argument('--input_dir', default='/data1/wangzd/datasets/denoising/sidd_val/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/data1/wangzd/uformer_cvpr/results_release/denoising/SIDD/Uformer_B/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='/data1/wangzd/uformer_cvpr/logs/denoising/SIDD/Uformer_B_1129/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='3', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer_B', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
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

result_dir_mat = os.path.join(args.result_dir, 'mat')
utils.mkdir(result_dir_mat)

# if args.save_images:
result_dir_img = os.path.join(args.result_dir, 'png')
utils.mkdir(result_dir_img)

# test_dataset = get_validation_data(args.input_dir)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration= utils.get_arch(args)

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

# Process data
filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
Inoisy /=255.
print(Inoisy.shape)
restored = np.zeros_like(Inoisy)
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
            _, _, h, w = noisy_patch.shape
            noisy_patch, mask = expand2square(noisy_patch, factor=128) 
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.masked_select(restored_patch,mask.bool()).reshape(1,3,h,w)
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored[i,k,:,:,:] = restored_patch

            save_file = os.path.join(result_dir_img, '%04d_%02d.png'%(i+1,k+1))
            utils.save_img(save_file, img_as_ubyte(restored_patch))

# save denoised data
sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored,})