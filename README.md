# Uformer: A General U-Shaped Transformer for Image Restoration (CVPR 2022)
<b>Zhendong Wang, <a href='https://vinthony.github.io'>Xiaodong Cun</a>, <a href='https://jianminbao.github.io/'>Jianmin Bao</a>, <a href='http://staff.ustc.edu.cn/~zhwg/'>Wengang Zhou</a>, <a href='http://people.ucas.ac.cn/~jzliu?language=en'>Jianzhuang Liu</a>, <a href='http://staff.ustc.edu.cn/~lihq/en/'>Houqiang Li </a> </b>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/deblurring-on-realblur-j-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-j-trained-on-gopro?p=uformer-a-general-u-shaped-transformer-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/deblurring-on-realblur-r-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-r-trained-on-gopro?p=uformer-a-general-u-shaped-transformer-for)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/image-denoising-on-dnd)](https://paperswithcode.com/sota/image-denoising-on-dnd?p=uformer-a-general-u-shaped-transformer-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/image-denoising-on-sidd)](https://paperswithcode.com/sota/image-denoising-on-sidd?p=uformer-a-general-u-shaped-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/deblurring-on-gopro)](https://paperswithcode.com/sota/deblurring-on-gopro?p=uformer-a-general-u-shaped-transformer-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/deblurring-on-hide-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-hide-trained-on-gopro?p=uformer-a-general-u-shaped-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/image-defocus-deblurring-on-dpd)](https://paperswithcode.com/sota/image-defocus-deblurring-on-dpd?p=uformer-a-general-u-shaped-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/image-enhancement-on-tip-2018)](https://paperswithcode.com/sota/image-enhancement-on-tip-2018?p=uformer-a-general-u-shaped-transformer-for)

Paper link: [[Arxiv]](https://arxiv.org/abs/2106.03106) [[CVPR]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.pdf)


### Update:
* **2022.07.06** Upload new codes and models for our Uformer. 
* **2022.04.09** Upload results of Uformer on denoising (SIDD, DND), motion deblurring (GoPro, HIDE, RealBlur-J/-R), and defocus deblurring (DPDD). 
* **2022.03.02** Uformer has been accepted by CVPR 2022! :fire:
* **2021.11.30** Update Uformer in [Arxiv link](https://arxiv.org/abs/2106.03106). The new code, models and results will be uploaded.
* **2021.10.28** Release the results of Uformer32 on SIDD and DND.
* **2021.09.30** Release pre-trained Uformer16 for SIDD denoising.
* **2021.08.19** Release a pre-trained model(Uformer32)! Add a script for FLOP/GMAC calculation.
* **2021.07.29** Add a script for testing the pre-trained model on the arbitrary-resolution images.

<hr>
<i>In this paper, we present Uformer, an effective and efficient Transformer-based architecture, in which we build a hierarchical encoder-decoder network using the Transformer block for image restoration. Uformer has two core designs to make it suitable for this task. The first key element is a local-enhanced window Transformer block, where we use non-overlapping window-based self-attention to reduce the computational requirement and employ the depth-wise convolution in the feed-forward network to further improve its potential for capturing local context. The second key element is that we explore three skip-connection schemes to effectively deliver information from the encoder to the decoder. Powered by these two designs, Uformer enjoys a high capability for capturing useful dependencies for image restoration. Extensive experiments on several image restoration tasks demonstrate the superiority of Uformer, including image denoising, deraining, deblurring and demoireing. We expect that our work will encourage further research to explore Transformer-based architectures for low-level vision tasks.</i>

![Uformer](fig/Uformer.png)

## Package dependencies
The project is built with PyTorch 1.9.0, Python3.7, CUDA11.1. For package dependencies, you can install them by:
```bash
pip install -r requirements.txt
```

## Pretrained model
- Uformer_B: [SIDD](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhendongwang_mail_ustc_edu_cn/Ea7hMP82A0xFlOKPlQnBJy0B9gVP-1MJL75mR4QKBMGc2w?e=iOz0zz) |
[GoPro](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhendongwang_mail_ustc_edu_cn/EfCPoTSEKJRAshoE6EAC_3YB7oNkbLUX6AUgWSCwoJe0oA?e=jai90x)

## Results from the pretrained model
- Uformer_B: [SIDD](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/EtcRYRDGWhBIlQa3EYBp4FYBao7ZZT2dPc5k1Qe-CdPh3A?e=PjBMub) | [DND](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Ekv3A5ic_4RChFa9XXquF_MB8M8tFd7spyHGJi_8obycnA?e=W7xeHe) | [GoPro](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/ElFalK0Qb8NHnyvhkSe1APgB5D0urGRMLnu2nBXJhtzdIw?e=D2XBhS) | [HIDE](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Eh4p1_kZ95xIopXDNyhl-Q0B65xX6C3J_fL-TQDbgvofqQ?e=8766eT) | [RealBlur-J](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/EpHFC9FauEpHhJDsFruEmmQBJ4_ZZaMgjaO9SHmB_vocaA?e=3a4b8A) | [RealBlur-R](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Eo2EC8rmkapNu9CxcYLwFpYBD8tX8pvfX_60jJIP8TGgGQ?e=yGbkQ8) | [DPDD](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/EvVAI84ZvlNChWsZA6QY4IkBc201zdTAs_g2Ufd5l0FgIQ?e=2DTlah)


## Data preparation 
### Denoising
For training data of SIDD, you can download the SIDD-Medium dataset from the [official url](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php).
Then generate training patches for training by:
```python
python3 generate_patches_SIDD.py --src_dir ../SIDD_Medium_Srgb/Data --tar_dir ../datasets/denoising/sidd/train
```

For evaluation on SIDD and DND, you can download data from [here](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Ev832uKaw2JJhwROKqiXGfMBttyFko_zrDVzfSbFFDoi4Q?e=S3p5hQ).


### Deblurring
For training on GoPro, and evaluation on GoPro, HIDE, RealBlur-J and RealBlur-R, you can download data from [here](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Ev832uKaw2JJhwROKqiXGfMBttyFko_zrDVzfSbFFDoi4Q?e=S3p5hQ).


Then put all the denoising data into `../datasets/denoising`, and all the deblurring data into `../datasets/deblurring`.

## Training
### Denoising
To train Uformer on SIDD, you can begin the training by:

```sh
sh script/train_denoise.sh
```
### Deblurring
To train Uformer on GoPro, you can begin the training by:

```sh
sh script/train_motiondeblur.sh
```


## Evaluation
To evaluate Uformer, you can run:

```sh
sh script/test.sh
```
For evaluate on each dataset, you should uncomment corresponding line.

## Computational Cost

We provide a simple script to calculate the flops by ourselves, a simple script has been added in `model.py`. You can change the configuration and run:

```python
python3 model.py
```

> The manual calculation of GMacs in this repo differs slightly from the main paper, but they do not influence the conclusion. We will correct the paper later.


## Citation
If you find this project useful in your research, please consider citing:

```
@InProceedings{Wang_2022_CVPR,
    author    = {Wang, Zhendong and Cun, Xiaodong and Bao, Jianmin and Zhou, Wengang and Liu, Jianzhuang and Li, Houqiang},
    title     = {Uformer: A General U-Shaped Transformer for Image Restoration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {17683-17693}
}
```

## Acknowledgement

This code borrows heavily from [MIRNet](https://github.com/swz30/MIRNet) and [SwinTransformer](https://github.com/microsoft/Swin-Transformer).


## Contact
Please contact us if there is any question or suggestion(Zhendong Wang ZhendongWang6@outlook.com, Xiaodong Cun vinthony@gmail.com).
