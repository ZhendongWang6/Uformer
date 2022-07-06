python3 ./train/train_denoise.py --arch Uformer_B --batch_size 32 --gpu '0,1' \
    --train_ps 128 --train_dir ../datasets/denoising/sidd/train --env _0706 \
    --val_dir ../datasets/denoising/sidd/val --save_dir ./logs/ \
    --dataset sidd --warmup 