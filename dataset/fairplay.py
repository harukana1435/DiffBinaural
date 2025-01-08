import os
import random
import numpy as np
import csv
from .base import BaseDataset
import torchaudio
import torch


class FairPlayDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(FairPlayDataset, self).__init__(
            list_sample, opt, **kwargs)


    def __getitem__(self, index):
        frames = None
        audios = None
        audio_path = None
        path_frames = []
        path_frames_ids = []
        #path_frames_det = ['' for n in range(N)] #今回はdetection結果はいらないとする
        path_audios = None

        if self.split == 'train':
            # the first video
            audio_path = self.list_sample[index]
        elif self.split == 'val':
            audio_path = self.list_sample[index]
            if not self.split == 'train':
                random.seed(1234)
        else: #テストのときだけどこれいる？
            test_path = "/home/h-okano/DAVIS/MUSIC21/test.csv"
            test_lis = self.get_audio_filelist(test_path)
            random.seed(index) # fixed
            samples = self.list_sample[index]
            audio_path = self.list_sample[index]
            
        basename = os.path.splitext(os.path.basename(audio_path))[0]
            
        try:
            # 音声の抽出
            audio, start_point = self._load_audio(audio_path)
        except Exception as e:
            print(f"Error loading audio for basename: {basename}")
            print(f"Details: {e}")
            audio = None  # エラー時は None を設定

        #左と右の音声の平均と差分を計算して、メルスペクトログラムに計算
        if audio is not None: 
            left_audio, right_audio = audio[0], audio[1]
            mix_audio = torch.FloatTensor(((left_audio + right_audio) / 2).unsqueeze(0))
            diff_audio = torch.FloatTensor(((left_audio - right_audio) / 2).unsqueeze(0))

            try:
                # メルスペクトログラムの計算
                mix_mel = self.mel_spectrogram(mix_audio, self.fft_size, self.num_mels,
                                                self.audLen, self.stft_hop, self.stft_frame, self.fmin, self.fmax,
                                                center=False)
            except Exception as e:
                print(f"Error calculating mel spectrogram for basename: {basename}")
                print(f"Details: {e}")
                mix_mel = None  # エラー時は None を設定

            try:
                diff_mel = self.mel_spectrogram(diff_audio, self.fft_size, self.num_mels,
                                                      self.audLen, self.stft_hop, self.stft_frame, self.fmin, self.fmax,
                                                      center=False)
            except Exception as e:
                print(f"Error calculating diff audio mel spectrogram for basename: {basename}")
                print(f"Details: {e}")
                diff_mel = None  # エラー時は None を設定

        #ビデオフレーム、3dマップの番号を抽出
        start_time = start_point/self.audRate
        end_time = (start_point+self.audLen)/self.audRate

        start_frame = int(start_time * self.vidRate)
        end_frame = int(end_time* self.vidRate)
        
        frame_indices = np.linspace(start_frame, end_frame, self.num_frames, dtype=int)
        even_frame_indices = []
        for idx in frame_indices:
            if idx % 2 == 0:
                even_frame_indices.append(idx)
            else:
                even_frame_indices.append(idx - 1 if idx > 1 else 2)  # 偶数に丸める

        #読み込み
        
        frame_paths = [
            os.path.join(self.dir_frames, f"{basename}.mp4", f"{i:06d}.jpg") for i in even_frame_indices
        ]
        pointcloud_paths = [
            os.path.join(self.dir_pointcloud, f"{basename}.mp4", f"{i:06d}.npz") for i in even_frame_indices
        ]
        
        try:
    # フレームパスを生成して読み込む
            frames = self._load_frames(frame_paths)
        except Exception as e:
            print(f"Error loading frames for basename: {basename}")
            print(f"Details: {e}")
            frames = None  # エラー時は None を返すなどの処理

        try:
            # ポイントクラウドパスを生成して読み込む
            pointclouds = self._load_pointclouds(pointcloud_paths)
        except Exception as e:
            print(f"Error loading pointclouds for basename: {basename}")
            print(f"Details: {e}")
            pointclouds = None  # エラー時は None を返すなどの処理

        ret_dict = {'mix_mel': mix_mel, 'diff_mel':diff_mel, 'frames': frames,
                    'pointclouds':pointclouds}
        return ret_dict
    
    
