import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from librosa.filters import mel as librosa_mel_fn
import librosa
import torch.nn.functional as F
import os
import statistics as stat

mel_basis = {}
hann_window = {}

def get_audio_filelist(file):
        # トレーニングデータのファイルを読み込む
        with open(file, 'r', encoding='utf-8') as fi:
            reader = csv.reader(fi)
            next(reader)  # 1行目（カラム名）をスキップ
            training_files = [row[0]  # Audio Pathの部分（1列目）
                              for row in reader if len(row) > 0]
        return training_files

def generate_mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False, where=""):
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



def mse_distance(mel1: torch.Tensor, mel2: torch.Tensor) -> torch.Tensor:
    """
    メルスペクトログラム間のMSE（Mean Squared Error）を計算
    :param mel1: メルスペクトログラム1 (形状: [F, T])
    :param mel2: メルスペクトログラム2 (形状: [F, T])
    :return: MSEスカラー値
    """
    return F.mse_loss(mel1, mel2)
def l2_distance(mel1: torch.Tensor, mel2: torch.Tensor) -> torch.Tensor:
    """
    メルスペクトログラム間のL2距離（ユークリッド距離）を計算
    :param mel1: メルスペクトログラム1 (形状: [F, T])
    :param mel2: メルスペクトログラム2 (形状: [F, T])
    :return: L2距離（ユークリッド距離）のスカラー値
    """
    return torch.norm(mel1 - mel2, p=2)  # L2ノルム（ユークリッド距離）

def process_evaluate(basename):
    file_path = f"/home/h-okano/DiffBinaural/processed_data/generated_mel_right/{basename}.npy"
    audio_input, sr = librosa.load(f"/home/h-okano/DiffBinaural/results_pos/result_{basename}/input_binaural.wav", mono=False, sr=None)
    audio_sepstereo, sr = librosa.load(f"/home/h-okano/SepStereo/results/{basename}/predicted_binaural.wav", mono=False, sr=None)
    audio_mono2binaural, sr = librosa.load(f"/home/h-okano/mono2binaural/results2/result_{basename}/predicted_binaural.wav", mono=False, sr=None)
    input_right = audio_input[1]
    sepstereo_right = audio_sepstereo[1]
    mono2binaural_right = audio_mono2binaural[1]
    mono = (audio_input[0]+audio_input[1])/2

    target_length = 640*256
    if len(input_right) < target_length:
        pad_length = target_length - len(input_right)
        input_process = F.pad(torch.FloatTensor(input_right), (0, pad_length))
    if len(sepstereo_right) < target_length:
        pad_length = target_length - len(sepstereo_right)
        sepstereo_process = F.pad(torch.FloatTensor(sepstereo_right), (0, pad_length))
    if len(mono2binaural_right) < target_length:
        pad_length = target_length - len(mono2binaural_right)
        mono2binaural_process = F.pad(torch.FloatTensor(mono2binaural_right), (0, pad_length))  
    if len(mono) < target_length:
        pad_length = target_length - len(mono)
        mono_process = F.pad(torch.FloatTensor(mono), (0, pad_length)) 

    input_mel = generate_mel_spectrogram(torch.FloatTensor(input_process).unsqueeze(0), 1024, 80, 16000, 256, 1024, 0, 8000).squeeze(0).numpy()
    sepstereo_mel = generate_mel_spectrogram(torch.FloatTensor(sepstereo_process).unsqueeze(0), 1024, 80, 16000, 256, 1024, 0, 8000).squeeze(0).numpy()
    mono2binaural_mel = generate_mel_spectrogram(torch.FloatTensor(mono2binaural_process).unsqueeze(0), 1024, 80, 16000, 256, 1024, 0, 8000).squeeze(0).numpy()
    mono_mel = generate_mel_spectrogram(torch.FloatTensor(mono_process).unsqueeze(0), 1024, 80, 16000, 256, 1024, 0, 8000).squeeze(0).numpy()
    diffbinaural_mel = torch.load(file_path)
    
    input_mel = torch.tensor(input_mel, dtype=torch.float32)
    diffbinaural_mel = torch.tensor(diffbinaural_mel, dtype=torch.float32)
    sepstereo_mel = torch.tensor(sepstereo_mel, dtype=torch.float32)
    mono2binaural_mel = torch.tensor(mono2binaural_mel, dtype=torch.float32)
    mono_mel = torch.tensor(mono_mel, dtype=torch.float32)
    
    return (
    l2_distance(diffbinaural_mel, input_mel).item(),
    l2_distance(sepstereo_mel, input_mel).item(),
    l2_distance(mono2binaural_mel, input_mel).item(),
    l2_distance(mono_mel, input_mel).item()
)


if __name__=='__main__':
    files = get_audio_filelist("/home/h-okano/DiffBinaural/FairPlay/splits_csv/split1/test.csv")
    diffbinaural_list = []
    mono2binaural_list = []
    sepstereo_list = []
    mono_list = []
    for i, file in enumerate(files):
        basename = os.path.splitext(os.path.split(file)[-1])[0]
        diffbinaural, sepstereo, mono2binaural, mono = process_evaluate(basename)
        diffbinaural_list.append(diffbinaural)
        sepstereo_list.append(sepstereo)
        mono2binaural_list.append(mono2binaural)
        mono_list.append(mono)
    
    print(f"diffbinaural mean{stat.mean(diffbinaural_list)} var{stat.stdev(diffbinaural_list)}")
    print(f"sepstereo mean{stat.mean(sepstereo_list)} var{stat.stdev(sepstereo_list)}")
    print(f"mono2binaural mean{stat.mean(mono2binaural_list)} var{stat.stdev(mono2binaural_list)}")
    print(f"mono mean{stat.mean(mono_list)} var{stat.stdev(mono_list)}")
        