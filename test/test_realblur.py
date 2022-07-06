import numpy as np
import os,sys
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))

from skimage import img_as_ubyte,io
from pdb import set_trace as stx
from dataset.dataset_motiondeblur import *
import utils
import math

from glob import glob
from natsort import natsorted
import cv2
from skimage.metrics import structural_similarity
import concurrent.futures

from model import UNet,Uformer

parser = argparse.ArgumentParser(description='Image motion deblurring evaluation on RealBlur_J/RealBlur_R')

parser.add_argument('--input_dir', default='/data1/wangzd/datasets/deblurring', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/data1/wangzd/uformer_cvpr/results_release/deblurring/', type=str, help='Directory for results')
parser.add_argument('--dataset', default='RealBlur_J,RealBlur_R', type=str, help='Test Dataset') 
parser.add_argument('--weights', default='/data1/wangzd/uformer_cvpr/logs/motiondeblur/GoPro/Uformer_B_1129/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='3', type=str, help='CUDA_VISIBLE_DEVICES')
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


def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 0

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift

def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)


def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False, data_range = 1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad,pad:-pad,:]
    crop_cr1 = cr1[pad:-pad,pad:-pad,:]
    ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

def proc(filename):
    tar,prd = filename
    tar_img = io.imread(tar)
    prd_img = io.imread(prd)
    
    tar_img = tar_img.astype(np.float32)/255.0
    prd_img = prd_img.astype(np.float32)/255.0
    
    prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)

    PSNR = compute_psnr(tar_img, prd_img, cr1, data_range=1)
    SSIM = compute_ssim(tar_img, prd_img, cr1)
    return (PSNR,SSIM)


if __name__ == "__main__":
    model_restoration = utils.get_arch(args)

    utils.load_checkpoint(model_restoration,args.weights)
    print("===>Testing using weights: ",args.weights)
    model_restoration.cuda()
    model_restoration.eval()
    for dataset in args.dataset.split(','):
        rgb_dir_test = os.path.join(args.input_dir, dataset, 'test', 'input')
        test_dataset = get_test_data(rgb_dir_test, img_options={})
        test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=False)

        result_dir  = os.path.join(args.result_dir, dataset, args.arch)
        utils.mkdir(result_dir)


        with torch.no_grad():
            for ii, data_test in enumerate(tqdm(test_loader), 0):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                input_    = data_test[0].cuda()
                _, _, h, w = input_.shape
                filenames = data_test[1]

                input_, mask = expand2square(input_, factor=128) 

                restored = model_restoration(input_)
                restored = torch.masked_select(restored,mask.bool()).reshape(1,3,h,w)
                restored = torch.clamp(restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
                print(filenames, restored.shape)
                utils.save_img(os.path.join(result_dir,filenames[0]+'.png'), img_as_ubyte(restored))
    
    ## evaluate
    results = {}
    for dataset in args.dataset.split(','):

        file_path = os.path.join(args.result_dir, dataset, args.arch)
        gt_path = os.path.join(args.input_dir, dataset, 'test', 'target')
        print(file_path,gt_path)
        path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
        gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))

        assert len(path_list) != 0, "Predicted files not found"
        assert len(gt_list) != 0, "Target files not found"

        psnr, ssim = [], []
        img_files =[(i, j) for i,j in zip(gt_list,path_list)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
                print(filename[1],PSNR_SSIM)
                with open(os.path.join(file_path,'psnr_ssim.txt'),'a') as f:
                    f.write(filename[1]+'PSNR: {:f} SSIM: {:f}'.format(PSNR_SSIM[0], PSNR_SSIM[1])+'\n')
                psnr.append(PSNR_SSIM[0])
                ssim.append(PSNR_SSIM[1])

        avg_psnr = sum(psnr)/len(psnr)
        avg_ssim = sum(ssim)/len(ssim)
        results[dataset+' psnr']= avg_psnr
        results[dataset+' ssim']= avg_ssim
        print('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(dataset, avg_psnr, avg_ssim))
        with open(os.path.join(file_path,'psnr_ssim.txt'),'a') as f:
            f.write('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(dataset, avg_psnr, avg_ssim)+'\n')
    print(results)
