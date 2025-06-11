import torch
import torchaudio
import librosa
import os
import csv

class MelSpectrogramCalculator:
    def __init__(self):
        self.mel_basis_cache = {}
        self.hann_window_cache = {}

    def mel_spectrogram_origin(self, y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin=0, fmax=11025, center=False):
        # 入力の音声が-1〜1に収まっていない場合に警告
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))

        # mel_basis と hann_window をキャッシュから取得
        mel_key = str(fmax) + '_' + str(y.device)
        if mel_key not in self.mel_basis_cache:
            mel = librosa.filters.mel(sampling_rate, n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            self.mel_basis_cache[mel_key] = torch.from_numpy(mel).float().to(y.device)

        if mel_key not in self.hann_window_cache:
            self.hann_window_cache[mel_key] = torch.hann_window(win_size).to(y.device)

        # 音声データをパッドする
        y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
        y = y.squeeze(1)

        # STFTを計算する
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=self.hann_window_cache[mel_key],
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

        # 複素数の絶対値を計算する
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        # メルスペクトログラムを計算する
        spec = torch.matmul(self.mel_basis_cache[mel_key], spec)

        return spec

def calculate_average_rms_loudness():
    calculator = MelSpectrogramCalculator()
    n_fft = 1024
    hop_size = 256
    num_mels = 80
    sampling_rate = 22050
    win_size = 1024
    fmax = 11025

    total_left_rms = 0
    total_right_rms = 0
    file_count = 0

    with open("/home/h-okano/DiffBinaural/FairPlay/splits_csv/split1/test.csv", 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # ヘッダーをスキップ

        for row in csv_reader:
            audio_path = row[0]
            try:
                waveform, sr = torchaudio.load(audio_path)

                if waveform.ndim == 1:
                    print(f"Skipping mono file: {audio_path}")
                    continue

                left_channel = waveform[0, :]
                right_channel = waveform[1, :]

                left_mel = calculator.mel_spectrogram_origin(left_channel.unsqueeze(0), n_fft, num_mels, sampling_rate, hop_size, win_size, fmax=fmax)
                right_mel = calculator.mel_spectrogram_origin(right_channel.unsqueeze(0), n_fft, num_mels, sampling_rate, hop_size, win_size, fmax=fmax)

                loudness_left = torch.sqrt(torch.mean(torch.pow(left_mel[:, :-1], 2))).item()
                loudness_right = torch.sqrt(torch.mean(torch.pow(right_mel[:, :-1], 2))).item()

                total_left_rms += loudness_left
                total_right_rms += loudness_right
                file_count += 1
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

    if file_count > 0:
        average_left_rms = total_left_rms / file_count
        average_right_rms = total_right_rms / file_count
        print(f"Average left channel RMS loudness: {average_left_rms}")
        print(f"Average right channel RMS loudness: {average_right_rms}")
    else:
        print("No valid binaural audio files found in the CSV.")

if __name__ == "__main__":
    calculate_average_rms_loudness()
