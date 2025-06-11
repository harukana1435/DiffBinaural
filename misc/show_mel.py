import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from librosa.filters import mel as librosa_mel_fn
import librosa
import torch.nn.functional as F

mel_basis = {}
hann_window = {}

def get_audio_filelist(self, file):
        # トレーニングデータのファイルを読み込む
        with open(file, 'r', encoding='utf-8') as fi:
            reader = csv.reader(fi)
            next(reader)  # 1行目（カラム名）をスキップ
            training_files = [row[0]  # Audio Pathの部分（1列目）
                              for row in reader if len(row) > 0]
        return training_files

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False, where=""):
    if torch.min(y) < -1.:
        print('mel min value is ', torch.min(y), where)
    if torch.max(y) > 1.:
        print('mel max value is ', torch.max(y), where)

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    return spec

# .npyファイルのパス
file_path = "/home/h-okano/DiffBinaural/processed_data/generated_mel_right/000059.npy"

# 出力画像の保存パス
output_diffbinaural = "/home/h-okano/DiffBinaural/misc/temp/diff-binaural_right.png"

# 出力画像の保存パス
output_gt = "/home/h-okano/DiffBinaural/misc/temp/gt_right.png"
output_sepstereo = "/home/h-okano/DiffBinaural/misc/temp/sepstereo_right.png"
output_mono2binaural = "/home/h-okano/DiffBinaural/misc/temp/mono2binaural_right.png"

audio_input, sr = librosa.load("/home/h-okano/BigVGAN/results2/result_000214/input_binaural.wav", mono=False, sr=None)
audio_sepstereo, sr = librosa.load("/home/h-okano/BigVGAN/results2/result_000214/predicted_binaural.wav", mono=False, sr=None)
audio_mono2binaural, sr = librosa.load("/home/h-okano/BigVGAN/results2/result_000214/predicted_binaural.wav", mono=False, sr=None)

input_right = audio_input[1]
sepstereo_right = audio_sepstereo[1]
mono2binaural_right = audio_mono2binaural[1]
mono = (audio_input[0]+audio_input[1])/2

# target_length = 861*256
# if len(input_right) < target_length:
#     pad_length = target_length - len(input_right)
#     input_process = F.pad(torch.FloatTensor(input_right), (0, pad_length))
# if len(sepstereo_right) < target_length:
#     pad_length = target_length - len(sepstereo_right)
#     sepstereo_process = F.pad(torch.FloatTensor(sepstereo_right), (0, pad_length))
# if len(mono2binaural_right) < target_length:
#     pad_length = target_length - len(mono2binaural_right)
#     mono2binaural_process = F.pad(torch.FloatTensor(mono2binaural_right), (0, pad_length))  
# if len(mono) < target_length:
#     pad_length = target_length - len(mono)
#     mono_process = F.pad(torch.FloatTensor(mono), (0, pad_length)) 
      
# input_mel = mel_spectrogram(torch.FloatTensor(input_process).unsqueeze(0), 1024, 80, 22050, 256, 1024, 0, 8000).squeeze(0).numpy()
# sepstereo_mel = mel_spectrogram(torch.FloatTensor(sepstereo_process).unsqueeze(0), 1024, 80, 22050, 256, 1024, 0, 8000).squeeze(0).numpy()
# mono2binaural_mel = mel_spectrogram(torch.FloatTensor(mono2binaural_process).unsqueeze(0), 1024, 80, 22050, 256, 1024, 0, 8000).squeeze(0).numpy()
# mono_mel = mel_spectrogram(torch.FloatTensor(mono_process).unsqueeze(0), 1024, 80, 22050, 256, 1024, 0, 8000).squeeze(0).numpy()


# データをロード
mel_spectrogram = torch.load(file_path)

mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)

# input_mel = torch.tensor(input_mel, dtype=torch.float32)

# sepstereo_mel = torch.tensor(sepstereo_mel, dtype=torch.float32)
# mono2binaural_mel = torch.tensor(mono2binaural_mel, dtype=torch.float32)
# mono_mel = torch.tensor(mono_mel, dtype=torch.float32)


def mse_distance(mel1: torch.Tensor, mel2: torch.Tensor) -> torch.Tensor:
    """
    メルスペクトログラム間のMSE（Mean Squared Error）を計算
    :param mel1: メルスペクトログラム1 (形状: [F, T])
    :param mel2: メルスペクトログラム2 (形状: [F, T])
    :return: MSEスカラー値
    """
    return F.mse_loss(mel1, mel2)

# print(f"diff-binaural {mse_distance(input_mel, mel_spectrogram)}")
# print(f"mono {mse_distance(mono_mel, mel_spectrogram)}")
# print(f"sepstereo {mse_distance(sepstereo_mel, mel_spectrogram)}")
# print(f"mono2binaural {mse_distance(mono2binaural_mel, mel_spectrogram)}")


# データ型を確認し、必要なら変換
# if not np.issubdtype(mel_spectrogram.dtype, np.number):
#     mel_spectrogram = mel_spectrogram.astype(np.float32)  # 数値型に変換
    
# 可視化
plt.figure(figsize=(10, 4))
plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label='Amplitude')
plt.xlabel('Time Frames')
plt.ylabel('Mel Filterbank Channels')
plt.title('Mel Spectrogram')

# 画像を保存
plt.savefig(output_diffbinaural)
plt.close()

# 各メルフィルタの中心周波数を取得
mel_frequencies = librosa.mel_frequencies(n_mels=80, fmin=0, fmax=11025)

plt.figure(figsize=(10, 4))
plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label='Amplitude')
plt.xlabel('Time Frames')

# 例えば、10個の目盛りにする場合
num_ticks = 10
tick_positions = np.linspace(0, 80 - 1, num_ticks).astype(int)
tick_labels = [f"{mel_frequencies[i]:.0f}" for i in tick_positions]
plt.yticks(tick_positions, tick_labels)

plt.ylabel('Frequency (Hz)')
plt.title('Mel Spectrogram')
plt.savefig(output_diffbinaural)
plt.close()


# 可視化
plt.figure(figsize=(10, 4))
plt.imshow(input_mel, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label='Amplitude')
plt.xlabel('Time Frames')
plt.ylabel('Mel Filterbank Channels')
plt.title('Mel Spectrogram')
plt.savefig(output_gt)
plt.close()

plt.figure(figsize=(10, 4))
plt.imshow(sepstereo_mel, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label='Amplitude')
plt.xlabel('Time Frames')
plt.ylabel('Mel Filterbank Channels')
plt.title('Mel Spectrogram')
plt.savefig(output_sepstereo)
plt.close()

plt.figure(figsize=(10, 4))
plt.imshow(mono2binaural_mel, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label='Amplitude')
plt.xlabel('Time Frames')
plt.ylabel('Mel Filterbank Channels')
plt.title('Mel Spectrogram')
plt.savefig(output_mono2binaural)
plt.close()