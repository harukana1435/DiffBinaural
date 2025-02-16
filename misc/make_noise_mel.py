import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False, where=""):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    return spec

# ファイルパス
file_path = "/home/h-okano/DiffBinaural/processed_data/generated_mel_pos/000016.npy"
output_path = "/home/h-okano/DiffBinaural/misc/temp/pred_mel_spectrogram_diff.png"
output_path2 = "/home/h-okano/DiffBinaural/misc/temp/noise2.png"
output_path_noise = "/home/h-okano/DiffBinaural/misc/temp/noise_only.png"

# オーディオをロード
audio, sr = librosa.load("/home/h-okano/DiffBinaural/results_pos/result_000016/input_binaural.wav", mono=False, sr=None)
right = audio[1]

# 長さを調整
target_length = 640 * 256
if len(right) < target_length:
    pad_length = target_length - len(right)
    right_process = F.pad(torch.FloatTensor(right), (0, pad_length))
    left_process = F.pad(torch.FloatTensor(audio[0]), (0, pad_length))

# メルスペクトログラムを計算
right_mel = mel_spectrogram(torch.FloatTensor((left_process-right_process)/2).unsqueeze(0), 
                             1024, 64, 16384, 256, 1024, 0, 8000).squeeze(0).numpy()

# 元のメルスペクトログラムをロード
mel_spectrogram = torch.load(file_path).numpy()

# ガウシアンノイズを付加
sigma = 0.2  # ノイズの強さ（変更可能）
noise = np.random.normal(0, sigma, right_mel.shape)
noisy_mel = right_mel + noise

# 完全なノイズ画像
pure_noise = np.random.normal(0, sigma, right_mel.shape)

# メルスペクトログラム（ノイズなし）を保存
plt.figure(figsize=(10, 4))
plt.imshow(right_mel, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label='Amplitude')
plt.xlabel('Time Frames')
plt.ylabel('Mel Filterbank Channels')
plt.title('Original Mel Spectrogram')
plt.savefig(output_path)
plt.close()

# メルスペクトログラム（ノイズ付き）を保存
plt.figure(figsize=(10, 4))
plt.imshow(noisy_mel, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label='Amplitude')
plt.xlabel('Time Frames')
plt.ylabel('Mel Filterbank Channels')
plt.title('Noisy Mel Spectrogram')
plt.savefig(output_path2)
plt.close()

# 完全なノイズ画像を保存
plt.figure(figsize=(10, 4))
plt.imshow(pure_noise, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label='Amplitude')
plt.xlabel('Time Frames')
plt.ylabel('Mel Filterbank Channels')
plt.title('Pure Noise')
plt.savefig(output_path_noise)
plt.close()
