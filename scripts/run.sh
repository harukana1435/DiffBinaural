#!/bin/bash 

OPTS=""
OPTS+="--id pos2d_jsaigo "

OPTS+="--list_train /home/h-okano/DiffBinaural/FairPlay/splits_csv/split1/train.csv "
OPTS+="--list_val /home/h-okano/DiffBinaural/FairPlay/splits_csv/split1/val.csv "

# Models

# frames-related
OPTS+="--arch_frame clip_pos2d " # [resnet18, clip, clip_pos]
OPTS+="--num_frames 5 " #特徴量としてどれくらい画像を使うか
OPTS+="--vidRate 8 " #動画のfps
OPTS+="--max_sources 4 "

# audio-related
OPTS+="--audLen 20480 " # 16384, 32768 20224
OPTS+="--audRate 22050 " #16000
OPTS+="--num_mels 80 "

# learning params
OPTS+="--num_gpus 2 "
OPTS+="--gpu_ids 0,1 "
OPTS+="--workers 20 "
OPTS+="--batch_size_per_gpu 4 "
OPTS+="--lr_frame 1e-4 " #1e-4
OPTS+="--lr_unet 1e-4 " #1e-4
OPTS+="--num_epoch 1000 "

OPTS+="--lr_steps 200 400 600 "
OPTS+="--dup_trainset 1 "
OPTS+="--eval_epoch 5 "

# where to save the results
OPTS+="--ckpt /home/h-okano/DiffBinaural/checkpoints "

# display, viz
OPTS+="--disp_iter 200 "
OPTS+="--num_val 40 " #どれくらい検証データとして使うかを表す

OPTS+="--split val "
OPTS+="--mode train"

CUDA_VISIBLE_DEVICES=3,4 python -u /home/h-okano/DiffBinaural/train_pos.py $OPTS
