#!/bin/bash 

OPTS=""
OPTS+="--id Binaural "

OPTS+="--list_test /home/h-okano/DiffBinaural/FairPlay/splits_csv/split1/test.csv "

# Models
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 64 "
OPTS+="--loss l1 "

# frames-related
OPTS+="--arch_frame clip_pos " # [resnet18, clip]
OPTS+="--num_frames 5 " #特徴量としてどれくらい画像を使うか
OPTS+="--vidRate 8 " #動画のfps
OPTS+="--max_sources 4 "

# audio-related
OPTS+="--audLen 20480 " # 16384
OPTS+="--audRate 16000 " #16000
OPTS+="--num_mels 80 "

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--gpu_ids 6 "
OPTS+="--workers 20 "
OPTS+="--batch_size_per_gpu 8 "

# where to save the results
OPTS+="--ckpt /home/h-okano/DiffBinaural/checkpoints/pos_right-frames5-channels64-epoch1000-step200_400_600-lr_unet0.0001 "

# display, viz
OPTS+="--disp_iter 200 "
OPTS+="--num_vis 20 "
OPTS+="--num_val 40 " #どれくらい検証データとして使うかを表す

OPTS+="--split test "
OPTS+="--mode eval "

OPTS+="--output_dir /home/h-okano/DiffBinaural/processed_data/generated_mel_right_pos "

python -u /home/h-okano/DiffBinaural/test_pos.py $OPTS
