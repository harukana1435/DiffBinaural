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
    file_path_right = f"/home/h-okano/DiffBinaural/processed_data/generated_mel_right/{basename}.npy"
    file_path_left = f"/home/h-okano/DiffBinaural/processed_data/generated_mel_left/{basename}.npy"

    audio_input, sr = librosa.load(f"/home/h-okano/SepStereo/results/{basename}/input_binaural.wav", mono=False, sr=None)
    audio_sepstereo, sr = librosa.load(f"/home/h-okano/SepStereo/results/{basename}/predicted_binaural.wav", mono=False, sr=None)
    audio_mono2binaural, sr = librosa.load(f"/home/h-okano/mono2binaural/results2/result_{basename}/predicted_binaural.wav", mono=False, sr=None)
    
    input_left = audio_input[0]
    input_right = audio_input[1]
    sepstereo_left = audio_sepstereo[0]
    sepstereo_right = audio_sepstereo[1]
    mono2binaural_left = audio_mono2binaural[0]
    mono2binaural_right = audio_mono2binaural[1]
    mono = (audio_input[0] + audio_input[1]) / 2

    target_length = 850 * 256

    def pad_to_target(x):
        return F.pad(torch.FloatTensor(x), (0, target_length - len(x))) if len(x) < target_length else torch.FloatTensor(x)

    input_left = pad_to_target(input_left)
    input_right = pad_to_target(input_right)
    sepstereo_left = pad_to_target(sepstereo_left)
    sepstereo_right = pad_to_target(sepstereo_right)
    mono2binaural_left = pad_to_target(mono2binaural_left)
    mono2binaural_right = pad_to_target(mono2binaural_right)
    mono = pad_to_target(mono)

    sr = 22050

    def to_mel(x): return generate_mel_spectrogram(x.unsqueeze(0), 1024, 80, sr, 256, 1024, 0, sr // 2).squeeze(0)
    
    def crop_or_pad_mel(mel: torch.Tensor, target_frames: int = 850) -> torch.Tensor:
        current_frames = mel.shape[1]
        if current_frames > target_frames:
            mel = mel[:, :target_frames]
        elif current_frames < target_frames:
            pad_amount = target_frames - current_frames
            mel = F.pad(mel, (0, pad_amount), mode="constant", value=0)
        return mel


    target_mel_frames = 850  # target_length / hop_size

    input_mel_left = crop_or_pad_mel(to_mel(input_left), target_mel_frames)
    input_mel_right = crop_or_pad_mel(to_mel(input_right), target_mel_frames)
    sepstereo_mel_left = crop_or_pad_mel(to_mel(sepstereo_left), target_mel_frames)
    sepstereo_mel_right = crop_or_pad_mel(to_mel(sepstereo_right), target_mel_frames)
    mono2binaural_mel_left = crop_or_pad_mel(to_mel(mono2binaural_left), target_mel_frames)
    mono2binaural_mel_right = crop_or_pad_mel(to_mel(mono2binaural_right), target_mel_frames)
    mono_mel = crop_or_pad_mel(to_mel(mono), target_mel_frames)
    diffbinaural_mel_left = crop_or_pad_mel(torch.load(file_path_left), target_mel_frames)
    diffbinaural_mel_right = crop_or_pad_mel(torch.load(file_path_right), target_mel_frames)


    diffbinaural_mel_left = torch.load(file_path_left)[:, :target_mel_frames]
    diffbinaural_mel_right = torch.load(file_path_right)[:, :target_mel_frames]

    # 距離を左右で平均する
    diffbinaural_l2 = (l2_distance(diffbinaural_mel_left, input_mel_left) + l2_distance(diffbinaural_mel_right, input_mel_right)) / 2
    sepstereo_l2 = (l2_distance(sepstereo_mel_left, input_mel_left) + l2_distance(sepstereo_mel_right, input_mel_right)) / 2
    mono2binaural_l2 = (l2_distance(mono2binaural_mel_left, input_mel_left) + l2_distance(mono2binaural_mel_right, input_mel_right)) / 2
    mono_l2 = (l2_distance(mono_mel, input_mel_left) + l2_distance(mono_mel, input_mel_right)) / 2

    return (
        diffbinaural_l2.item(),
        sepstereo_l2.item(),
        mono2binaural_l2.item(),
        mono_l2.item()
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
        