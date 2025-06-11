import os
import librosa
import soundfile as sf

# 対象ディレクトリ
src_dir = '/home/h-okano/DiffBinaural/FairPlay/binaural_audios'
out_dir = '/home/h-okano/DiffBinaural/FairPlay/binaural_audios_22050Hz'
# リサンプリング後のサンプルレート
target_sr = 22050

# ディレクトリ内の全ての WAV ファイルを処理
for filename in os.listdir(src_dir):
    if filename.lower().endswith('.wav'):
        filepath = os.path.join(src_dir, filename)
        print(f'Processing: {filepath}')
        
        # 元のサンプルレートを保持して読み込み（sr=Noneで元のサンプルレートで読み込み）
        y, sr = librosa.load(filepath, sr=None, mono=False)
        
        # もし元のサンプルレートと異なる場合はリサンプリング
        if sr != target_sr:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        else:
            y_resampled = y
        
        # 新しいファイル名を作成（例: 元ファイル名_22050.wav）
        new_filename = os.path.splitext(filename)[0]+'.wav'
        new_filepath = os.path.join(out_dir, new_filename)
        
        # リサンプリング後のデータを保存
        sf.write(new_filepath, y_resampled.T, target_sr)
        print(f'Saved resampled file as: {new_filepath}')
