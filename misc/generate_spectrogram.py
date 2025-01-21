import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np

def save_mel_spectrogram(audio_path, save_prefix, n_mels=128, sample_rate=16000):
    # 音声ファイルを読み込む
    waveform, sr = torchaudio.load(audio_path)
    
    # ステレオ音声の場合、左右チャンネルに分割
    if waveform.shape[0] == 2:
        left_channel = waveform[0]
        right_channel = waveform[1]
    else:
        # モノラルの場合はそのまま使用
        left_channel = right_channel = waveform

    # 右と左のチャンネルを足して割った音声（加算）
    added_waveform = (left_channel + right_channel) / 2
    
    # 右と左のチャンネルを引いて割った音声（減算）
    subtracted_waveform = (left_channel - right_channel) / 2

    # メルスペクトログラムを計算するための変換
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels)
    
    # 加算されたメルスペクトログラムを計算
    mel_spectrogram_added = mel_transform(added_waveform.unsqueeze(0))  # チャンネル数を合わせるためにunsqueeze(0)を使用
    mel_spectrogram_added_db = mel_spectrogram_added.log2()

    # 減算されたメルスペクトログラムを計算
    mel_spectrogram_subtracted = mel_transform(subtracted_waveform.unsqueeze(0))  # チャンネル数を合わせるためにunsqueeze(0)を使用
    mel_spectrogram_subtracted_db = mel_spectrogram_subtracted.log2()

    # 加算されたスペクトログラムの画像を保存
    plt.figure(figsize=(10, 6))
    plt.imshow(mel_spectrogram_added_db[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")
    plt.title('Mel Spectrogram (Added Channels)')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (mel)')
    added_save_path = f"{save_prefix}_added.png"
    plt.tight_layout()
    plt.savefig(added_save_path)
    plt.close()

    # 減算されたスペクトログラムの画像を保存
    plt.figure(figsize=(10, 6))
    plt.imshow(mel_spectrogram_subtracted_db[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")
    plt.title('Mel Spectrogram (Subtracted Channels)')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (mel)')
    subtracted_save_path = f"{save_prefix}_subtracted.png"
    plt.tight_layout()
    plt.savefig(subtracted_save_path)
    plt.close()

# 使用例
audio_path = '/home/h-okano/DiffBinaural/FairPlay/binaural_audios_16000Hz/000035.wav'  # ここに音声ファイルのパスを指定
save_prefix = '/home/h-okano/DiffBinaural/misc/temp/mel_spectrogram'  # 保存する画像のプレフィックス
save_mel_spectrogram(audio_path, save_prefix)
