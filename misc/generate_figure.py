import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np

def add_strong_noise(spectrogram, noise_factor):
    """
    スペクトログラムに強いノイズを加える関数。
    
    Parameters:
        spectrogram (Tensor): メルスペクトログラム (チャンネル x 周波数軸 x 時間軸)
        noise_factor (float): ノイズの強度（0〜1の範囲で指定）
        
    Returns:
        Tensor: 強いノイズを加えたスペクトログラム
    """
    # 正規分布からのノイズを強くスケーリング
    noise = torch.randn_like(spectrogram) * noise_factor * spectrogram.std()
    noisy_spectrogram = spectrogram + noise
    # ノイズによって負の値が出ないように制限
    noisy_spectrogram = torch.clamp(noisy_spectrogram, min=0.0)
    return noisy_spectrogram

def save_noisy_spectrogram_images(audio_path, save_prefix, n_mels=128, sample_rate=16000, num_images=5, max_noise_factor=1.0):
    """
    スペクトログラムに強いノイズを加えながらいくつかの画像を保存する関数。
    
    Parameters:
        audio_path (str): 音声ファイルのパス
        save_prefix (str): 画像保存時のファイル名のプレフィックス
        n_mels (int): メルスペクトログラムのフィルタバンク数
        sample_rate (int): サンプリングレート
        num_images (int): 生成する画像の数
        max_noise_factor (float): 最大ノイズ強度
    """
    # 音声ファイルを読み込む（ステレオの場合、2チャンネル）
    waveform, sr = torchaudio.load(audio_path)
    
    # 左右チャンネルの差分を取る（左右のチャンネルを引き算し、2で割る）
    difference_waveform = (waveform[0] - waveform[1]) / 2.0

    # メルスペクトログラムの変換
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels)
    
    # 差分波形からメルスペクトログラムを計算
    mel_spectrogram = mel_transform(difference_waveform.unsqueeze(0))  # チャンネル数を合わせるために unsqueeze(0)

    # メルスペクトログラムの対数スケールに変換（デシベルに変換）
    mel_spectrogram_db = mel_spectrogram.log2()

    # 強いノイズを加えたスペクトログラムを複数保存
    for i in range(num_images):
        noise_factor = (i + 1) * (max_noise_factor / num_images)  # ノイズ強度の決定
        noisy_spectrogram_db = add_strong_noise(mel_spectrogram_db, noise_factor)
        
        # 画像をプロット
        plt.figure(figsize=(10, 6))
        plt.imshow(noisy_spectrogram_db[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format="%+2.0f dB")
        plt.title(f'Mel Spectrogram with Strong Noise (Factor {noise_factor:.2f})')
        plt.xlabel('Time (frames)')
        plt.ylabel('Frequency (mel)')
        
        # 画像を保存
        plt.tight_layout()
        plt.savefig(f'./misc/temp/{save_prefix}_noisy_{i + 1}.png')
        plt.close()

# 使用例
audio_path = '/home/h-okano/DiffBinaural/FairPlay/binaural_audios_16000Hz/000035.wav'  # ここに音声ファイルのパスを指定
save_prefix = 'noisy_spectrogram'  # 保存する画像のプレフィックス
save_noisy_spectrogram_images(audio_path, save_prefix)
