import os
import subprocess
from pathlib import Path

# ファイルが存在するか確認し、存在する場合は終了する関数
def check_and_exit_if_file_exists(filepath):
    if os.path.exists(filepath):
        #print(f"ファイル {filepath} が既に存在します。プログラムを終了します。")
        return True
    else :
        return False

def run_spatiotemporal_detection(input_dir, output_dir, script_path, config_path, checkpoint_url, det_config_path, det_checkpoint_url, label_map_path):
    """
    指定したディレクトリ内のすべての.mp4動画に対してspatiotemporal detectionを実行する。

    Args:
        input_dir (str): 入力動画のディレクトリパス
        output_dir (str): 出力動画の保存先ディレクトリパス
        script_path (str): 実行するスクリプトのパス
        config_path (str): 動作検出の設定ファイルパス
        checkpoint_url (str): 動作検出のチェックポイントURL
        det_config_path (str): 物体検出の設定ファイルパス
        det_checkpoint_url (str): 物体検出のチェックポイントURL
        label_map_path (str): ラベルマップのパス
    """
    # 入力動画ディレクトリのすべての.mp4ファイルを取得
    video_files = list(Path(input_dir).glob("*.mp4"))
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    for video_file in video_files:
        # 入力動画のファイル名と出力先を設定
        input_video_path = str(video_file)
        output_npy_path = os.path.join(output_dir, video_file.stem + ".npy")
        
        
        # コマンドを構築
        command = [
            "python", script_path, input_video_path, output_npy_path,
            "--config", config_path,
            "--checkpoint", checkpoint_url,
            "--det-config", det_config_path,
            "--det-checkpoint", det_checkpoint_url,
            "--det-score-thr", "0.9",
            "--action-score-thr", "0.5",
            "--label-map", label_map_path,
            "--predict-stepsize", "8",
            "--output-stepsize", "4",
            "--output-fps", "6"
        ]

        # 実行
        #print(f"Processing {input_video_path}...")
        
        if check_and_exit_if_file_exists(output_npy_path):
            #print("already processed")
            continue

        
        try:
            subprocess.run(command, check=True)
            #print(f"Successfully processed: {input_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {input_video_path}: {e}")


# 入力データとパラメータを設定
input_dir = "../dataset/videos/"  # 動画の入力ディレクトリ
output_dir = "../processed_data/det_npy/"  # 処理結果を保存する出力ディレクトリ
script_path = "/home/h-okano/mmaction2/demo/spatiotemporal_det_8fps.py"  # 実行スクリプト
config_path = "/home/h-okano/mmaction2/configs/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py"
checkpoint_url = "https://download.openmmlab.com/mmaction/v1.0/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-bf93c9ea.pth"
det_config_path = "/home/h-okano/mmaction2/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py"
det_checkpoint_url = "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
label_map_path = "/home/h-okano/mmaction2/tools/data/ava/label_map.txt"

# 関数を呼び出してすべての動画を処理
run_spatiotemporal_detection(input_dir, output_dir, script_path, config_path, checkpoint_url, det_config_path, det_checkpoint_url, label_map_path)
