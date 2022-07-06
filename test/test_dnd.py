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
import math

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))

import scipy.io as sio
from dataset.dataset_denoise import *
import utils

from model import UNet,Uformer

from utils.bundle_submissions import bundle_submissions_srgb_v1
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='/data1/wangzd/datasets/denoising/dnd/input/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/data1/wangzd/uformer_cvpr/results_release/denoising/DND/Uformer_B/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='/data1/wangzd/uformer_cvpr/logs/denoising/SIDD/Uformer_B_1129/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
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

utils.mkdir(os.path.join(args.result_dir,'matfile'))
utils.mkdir(os.path.join(args.result_dir,'png'))

test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration= utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

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
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_noisy = data_test[0].cuda()
        _, _, h, w = rgb_noisy.shape
        filenames = data_test[1]
        rgb_noisy, mask = expand2square(rgb_noisy, factor=128) 
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.masked_select(rgb_restored,mask.bool()).reshape(1,3,h,w)
        rgb_restored = torch.clamp(rgb_restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))

        denoised_img = img_as_ubyte(rgb_restored)
        print(filenames[0])
        utils.save_img(os.path.join(args.result_dir,'png/',filenames[0]+ '.png'), denoised_img)
        save_file = os.path.join(args.result_dir,'matfile/', filenames[0] +'.mat')
        sio.savemat(save_file, {'Idenoised_crop': np.float32(rgb_restored)})

  

bundle_submissions_srgb_v1(os.path.join(args.result_dir,'matfile/'), 'srgb_results_for_server_submission/')
os.system("rm {}".format(os.path.join(args.result_dir,'matfile/*.mat')))