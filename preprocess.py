import os
import ffmpeg
from pathlib import Path

# 入力ディレクトリ（すべてのwavとmp4が入っているディレクトリ）
audio_dir = "/home/h-okano/DETbinaural/dataset/binaural_audios"
video_dir = "/home/h-okano/DET2binaural/dataset/videos"

# 出力ディレクトリ
output_audio_dir = "data/binaural_audios"
output_frames_dir = "/home/h-okano/DET2binaural/dataset/videos"

# 必要なディレクトリを作成
os.makedirs(output_audio_dir, exist_ok=True)
os.makedirs(output_frames_dir, exist_ok=True)

# 1. オーディオの抽出（11025Hz）
def extract_audio(video_path, output_audio_path):
    (
        ffmpeg
        .input(video_path)
        .output(output_audio_path, ar=11025, ac=1)  # 11025Hz, モノラル
        .run(quiet=True, overwrite_output=True)
    )

# 2. フレームの抽出（8fps）
def extract_frames(video_path, output_frame_dir):
    os.makedirs(output_frame_dir, exist_ok=True)
    (
        ffmpeg
        .input(video_path)
        .output(f"{output_frame_dir}/%06d.jpg", r=4)  # 8フレーム/秒
        .run(quiet=True, overwrite_output=True)
    )

# 3. フォルダ名に.mp4を付加する関数
def rename_to_include_mp4(output_frames_dir):
    for folder in os.listdir(output_frames_dir):
        folder_path = os.path.join(output_frames_dir, folder)
        if os.path.isdir(folder_path) and not folder.endswith(".mp4"):
            new_folder_name = f"{folder}.mp4"
            new_folder_path = os.path.join(output_frames_dir, new_folder_name)
            os.rename(folder_path, new_folder_path)
            print(f"Renamed {folder} to {new_folder_name}")

# すべての動画を処理
video_files = Path(video_dir).glob("*.mp4")

for video_file in video_files:
    video_id = video_file.stem  # ファイル名から拡張子を除いた部分
    
    # 出力先のパス
    output_audio_path = os.path.join(output_audio_dir, f"{video_id}.wav")
    output_frame_dir = os.path.join(output_frames_dir, video_id)  # 拡張子なしのフォルダを作成
    
    # オーディオを抽出
    extract_audio(video_file, output_audio_path)
    
    # フレームを抽出
    extract_frames(video_file, output_frame_dir)

    print(f"Processed {video_file}")

# すべての処理が終わった後にフォルダ名を変更
rename_to_include_mp4(output_frames_dir)
