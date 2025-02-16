import os
import random
import numpy as np
import csv
from .base import BaseDataset
import torchaudio
import torch


class FairPlayPosDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(FairPlayPosDataset, self).__init__(
            list_sample, opt, **kwargs)


    def __getitem__(self, index):
        frames = None
        audio_path = None

        audio_path = self.list_sample[index]    
        
        basename = os.path.splitext(os.path.basename(audio_path))[0]
            
        try:
            # 音声の抽出
            audio, start_point = self._load_audio(audio_path)
            audio = torch.FloatTensor(audio)
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
                                                self.audRate, self.stft_hop, self.stft_frame, self.fmin, self.fmax,
                                                center=False)
            except Exception as e:
                print(f"Error calculating mel spectrogram for basename: {basename}")
                print(f"Details: {e}")
                mix_mel = None  # エラー時は None を設定

            try:
                diff_mel = self.mel_spectrogram(diff_audio, self.fft_size, self.num_mels,
                                                      self.audRate, self.stft_hop, self.stft_frame, self.fmin, self.fmax,
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
            if idx == 0:
                even_frame_indices.append(idx+2)
            elif idx % 2 == 0:
                even_frame_indices.append(idx)
            else:
                even_frame_indices.append(idx - 1 if idx > 1 else 2)  # 偶数に丸める

        #読み込み
        
        frame_paths = [
            os.path.join(self.dir_frames, f"{basename}.mp4", f"{i:06d}.jpg") for i in even_frame_indices
        ]
        
        det_pos_data_path = os.path.join(self.dir_det_pos, basename+".npy")
        det_pos_data = np.load(det_pos_data_path, allow_pickle=True).item()
        
        for i, num in enumerate(even_frame_indices):
            if num >= det_pos_data['bounding_boxes'].shape[0]*2:
                even_frame_indices[i] = det_pos_data['bounding_boxes'].shape[0]*2
                
        det_data = [det_pos_data['bounding_boxes'][i//2-1] for i in even_frame_indices]
        
        frames, mask = self._load_frames_det(frame_paths, det_data)
        
        mask = np.array([mask for _ in range(self.num_frames)])
        
        pos_data = [det_pos_data['pos_3d'][i//2-1] for i in even_frame_indices]
        pos_data = np.array([np.pad(data, ((0, self.max_sources-data.shape[0]),(0,0)), constant_values=0) for data in pos_data])
            
            

        ret_dict = {'mix_mel': mix_mel, 'diff_mel':diff_mel, 'frames': frames,
                    'pos_data':pos_data, 'mask':mask}
        return ret_dict
    
    
