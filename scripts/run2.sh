#!/bin/bash 

OPTS=""
OPTS+="--id Binaural_Lenear_pos_left_silent "

OPTS+="--list_train /home/h-okano/DiffBinaural/FairPlay/splits_csv/split1/train.csv "
OPTS+="--list_val /home/h-okano/DiffBinaural/FairPlay/splits_csv/split1/val.csv "

# Models
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 64 "
OPTS+="--loss l1 "
OPTS+="--weighted_loss 0 " #weighted_lossはなしにしてる

# frames-related
OPTS+="--arch_frame clip_pos " # [resnet18, clip]
OPTS+="--num_frames 5 " #特徴量としてどれくらい画像を使うか
OPTS+="--vidRate 8 " #動画のfps
OPTS+="--max_sources 4 "

# audio-related
OPTS+="--audLen 20480 " # 16384, 32768
OPTS+="--audRate 16000 " #16000
OPTS+="--num_mels 80 "

# learning params
OPTS+="--num_gpus 2 "
OPTS+="--gpu_ids 3,4 "
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
OPTS+="--num_vis 20 "
OPTS+="--num_val 40 " #どれくらい検証データとして使うかを表す

OPTS+="--split val "
OPTS+="--mode train"

python -u /home/h-okano/DiffBinaural/main_pos2.py $OPTS
