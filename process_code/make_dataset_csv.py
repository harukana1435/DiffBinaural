import h5py
import os
import csv

# HDF5ファイルが格納されているディレクトリのパス
splits_dir = "/home/h-okano/DiffBinaural/FairPlay/splits"

# CSVファイルを保存する基準ディレクトリ
output_dir = "/home/h-okano/DiffBinaural/FairPlay/splits_csv"

# 音声ファイルと関連するディレクトリのベースパス
audio_base_path = "/home/h-okano/DiffBinaural/FairPlay/binaural_audios"
frames_base_path = "/home/h-okano/DiffBinaural/FairPlay/frames"
mapping_base_path = "/home/h-okano/DiffBinaural/processed_data/3d_mapping_npz"

# HDF5ファイルを処理してCSVに保存
for root, _, files in os.walk(splits_dir):
    for file in files:
        if file.endswith(".h5"):
            h5_file_path = os.path.join(root, file)
            print(f"Processing file: {h5_file_path}")
            
            # HDF5ファイルを開く
            with h5py.File(h5_file_path, "r") as h5_file:
                # "audio" データセットの内容を取得
                if "audio" in h5_file:
                    audio_paths = h5_file["audio"][:]
                    
                    # ディレクトリ構造を再現するためのパスを作成
                    relative_path = os.path.relpath(root, splits_dir)  # splits_dirからの相対パス
                    target_dir = os.path.join(output_dir, relative_path)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # CSVファイルのパスを生成
                    csv_file_path = os.path.join(target_dir, file.replace(".h5", ".csv"))
                    
                    # CSVに書き込むデータを準備
                    csv_data = []
                    for audio_path in audio_paths:
                        # 音声パスをバイト文字列からデコード
                        audio_path_str = audio_path.decode("utf-8")
                        
                        # 動画の番号を抽出（ファイル名の最後6桁部分）
                        video_id = os.path.splitext(os.path.basename(audio_path_str))[0]
                        
                        # 条件に基づくパスを生成
                        frames_dir = os.path.join(frames_base_path, f"{video_id}.mp4")
                        mapping_dir = os.path.join(mapping_base_path, f"{video_id}.npz")
                        audio_file = os.path.join(audio_base_path, f"{video_id}.wav")
                        
                        # 画像の枚数をカウント
                        if os.path.exists(frames_dir):
                            image_count = len([f for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))])
                        else:
                            image_count = 0  # ディレクトリが存在しない場合は0
                        
                        # データを追加
                        csv_data.append([audio_file, frames_dir, image_count, mapping_dir])
                    
                    # CSVファイルに書き込み
                    with open(csv_file_path, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        # ヘッダー行
                        writer.writerow(["Audio Path", "Frames Directory", "Image Count", "3D Mapping Path"])
                        # データ行
                        writer.writerows(csv_data)
                    
                    print(f"CSVファイルが生成されました: {csv_file_path}")
