#!/bin/bash 

OPTS=""
OPTS+="--id Binaural "

OPTS+="--list_train /home/h-okano/DiffBinaural/FairPlay/splits_csv/split1/train.csv "
OPTS+="--list_val /home/h-okano/DiffBinaural/FairPlay/splits_csv/split1/val.csv "

# Models
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 64 "
OPTS+="--loss l1 "
OPTS+="--weighted_loss 0 " #weighted_lossはなしにしてる

# logscale in frequency
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--arch_frame clip " # [resnet18, clip]
OPTS+="--num_frames 4 " #特徴量としてどれくらい画像を使うか
OPTS+="--vidRate 8 " #動画のfps

# audio-related
OPTS+="--audLen 16384 " # 16384
OPTS+="--audRate 16000 " #16000
OPTS+="--num_mels 64 "

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--gpu_ids 5 "
OPTS+="--workers 20 "
OPTS+="--batch_size_per_gpu 32 "
OPTS+="--lr_frame 1e-4 " #1e-4
OPTS+="--lr_unet 1e-4 " #1e-4
OPTS+="--num_epoch 400 "
OPTS+="--lr_steps 100 200 300 "
OPTS+="--dup_trainset 4 "
OPTS+="--eval_epoch 1 "

# where to save the results
OPTS+="--ckpt /home/h-okano/DiffBinaural/checkpoints "

# display, viz
OPTS+="--disp_iter 200 "
OPTS+="--num_vis 2 "
OPTS+="--num_val 20 " #どれくらい検証データとして使うかを表す

OPTS+="--split val "
OPTS+="--mode train"

python -u /home/h-okano/DiffBinaural/main.py $OPTS
