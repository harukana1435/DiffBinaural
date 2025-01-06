import os
import pandas as pd

# 入力ディレクトリの設定
audio_dir = '/home/h-okano/DiffBinaural/dataset/binaural_audios_16000Hz'
output_file = '/home/h-okano/DiffBinaural/melspec_process/data/output.tsv'

# ファイルの取得
def extract_audio_info(audio_dir):
    data = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                video_path = audio_path.replace('binaural_audios_16000Hz', 'videos').replace('.wav', '')
                duration = get_audio_duration(audio_path)
                mel_mean_path = audio_path.replace('dataset/binaural_audios_16000Hz', 'processed_data/melspec_mean').replace('.wav', '_mel.npy')
                mel_diff_path = audio_path.replace('dataset/binaural_audios_16000Hz', 'processed_data/melspec_diff').replace('.wav', '_mel.npy')
                
                # 結果を格納
                data.append({
                    'name': str(os.path.splitext(file)[0]),
                    'dataset': 'FAIR-Play',
                    'audio_path': audio_path,
                    'video_path': video_path,
                    'mel_mean_path': mel_mean_path,
                    'mel_diff_path':mel_diff_path,
                    'duration': duration
                })
    
    return data

# 音声ファイルの長さを取得する関数
def get_audio_duration(audio_path):
    from pydub.utils import mediainfo
    info = mediainfo(audio_path)
    return float(info['duration'])

# データを抽出してTSVに保存
def save_to_tsv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, sep='\t', index=False)

# メイン処理
audio_data = extract_audio_info(audio_dir)
save_to_tsv(audio_data, output_file)

print(f"データは {output_file} に保存されました。")
