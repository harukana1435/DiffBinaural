import random
import os
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchaudio
import librosa
from PIL import Image
import soundfile as sf
import clip
from . import video_transforms as vtransforms
from librosa.filters import mel as librosa_mel_fn

_, preprocess = clip.load("ViT-B/32", device="cuda")

class BaseDataset(torchdata.Dataset):
    def __init__(self, list_sample, opt, max_sample=-1, split='train'):
        # params
        self.num_frames = opt.num_frames
        self.vidRate = opt.vidRate #8 動画のフレームレート
        self.imgSize = opt.imgSize
        self.audRate = opt.audRate
        self.audLen = opt.audLen #65536 16000Hzで読み込む
        self.audSec = 1. * self.audLen / self.audRate

        # STFT params
        self.log_freq = opt.log_freq
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.fft_size = opt.stft_frame
        self.num_mels = opt.num_mels
        self.fmin = 0
        self.fmax = opt.audRate//2
        self.HS = opt.stft_frame // 2 + 1
        self.WS = (self.audLen + 1) // self.stft_hop
        
        self.mel_basis_cache = {}  # mel_basis をキャッシュするための辞書
        self.hann_window_cache = {}  # hann_window をキャッシュするための辞書

        #ディレクトリ
        self.dir_frames= opt.dir_frames
        self.dir_pointcloud=opt.dir_pointcloud

        self.split = split
        self.seed = opt.seed
        random.seed(self.seed)
        

        # initialize video transform
        self._init_vtransform()

        # list_sample can be a python list or a csv file of list
        if isinstance(list_sample, str):
            self.list_sample = self.get_audio_filelist(list_sample)
            
        elif isinstance(list_sample, list):
            self.list_sample = list_sample
        else:
            raise('Error list_sample!')

        if self.split == 'train':
            self.list_sample *= opt.dup_trainset # デフォルトはdup_trainsetは5に設定してある
            random.shuffle(self.list_sample)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def __len__(self):
        return len(self.list_sample)

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            transform_list.append(vtransforms.Resize(int(self.imgSize * 1.1), InterpolationMode.BICUBIC))
            transform_list.append(vtransforms.RandomCrop(self.imgSize))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(vtransforms.Resize(self.imgSize, InterpolationMode.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(self.imgSize))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.vid_transform = transforms.Compose(transform_list)
        self.clip_transform = transforms.Compose([vtransforms.Stack()])

    # image transform funcs, deprecated
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize), InterpolationMode.BICUBIC),
                #transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize), InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])


    def get_audio_filelist(self, file):
        # トレーニングデータのファイルを読み込む
        with open(file, 'r', encoding='utf-8') as fi:
            reader = csv.reader(fi)
            next(reader)  # 1行目（カラム名）をスキップ
            training_files = [row[0]  # Audio Pathの部分（1列目）
                              for row in reader if len(row) > 0]
        return training_files

    def _load_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame(path))
        frames = self.vid_transform(frames) #(B, L, C, H, W)
        return frames
    
    
    def _load_pointclouds(self, paths):
        pointclouds = []
        for path in paths:
            pointclouds.append(self._load_pointcloud(path))
        return pointclouds

    def _load_frames_clip(self, paths):
        frames = []
        for path in paths:
            frames.append(preprocess(Image.open(path)))
        frames = self.clip_transform(frames)
        return frames

    def _load_frames_det(self, paths, path_frames_ids,  path_frames_det):
        det_res = np.load(path_frames_det)
        frames = []
        N = len(paths)
        for n in range(N):
            path = paths[n]
            id = path_frames_ids[n]
            frames.append(self._load_frame_det(path, id, det_res))
        frames = self.vid_transform(frames)
        return frames


    def _load_frame_det(self, path, id, det_res):

        # load image
        img = Image.open(path).convert('RGB')

        # get box
        idx = np.where(det_res[:, 0] == id)
        if len(idx[0])!=0:
            n = np.argmax(det_res[idx, 2], axis=1)
            bb = det_res[idx[0][n[0]], 3:]
            # crop image
            img = img.crop((bb[0], bb[1], bb[2], bb[3]))
        return img

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img
    
    def _load_pointcloud(self, path):
        pointcloud = np.load(path)
        pointcloud = pointcloud["depth_map_3d"]
        reshaped_pointcloud = pointcloud.reshape(-1, 3)
        return reshaped_pointcloud

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop) #strg_frameが1024でhopが256
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio_file(self, path):
        audio_raw, rate = librosa.load(path, sr=self.audRate, mono=False)
        return audio_raw, rate

    def _load_audio(self, path):
        # load audio
        audio, rate = self._load_audio_file(path)

        # resample
        if rate != self.audRate:
            print('resmaple {}->{}'.format(rate, self.audRate))
            audio = librosa.resample(audio, rate, self.audRate)
        
        # repeat if audio is too short
        if audio.shape[-1] < self.audLen:
            audio = torch.nn.functional.pad(audio, (0, self.audLen - audio.shape[-1]), 'constant')
            audio_start = 0
        else:
            max_audio_start = audio.shape[-1] - self.audLen
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start+self.audLen]

        return audio, audio_start

    def _mix_n_and_stft(self, audios):
        N = len(audios)
        mags = [None for n in range(N)]

        audio_mix = 0
        for n in range(N):
            audio_mix += audios[n]
        audio_mix /= N

        # STFT
        amp_mix, phase_mix = self._stft(audio_mix)
        for n in range(N):
            ampN, _ = self._stft(audios[n])
            mags[n] = ampN.unsqueeze(0)

        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])

        return amp_mix.unsqueeze(0), mags, phase_mix.unsqueeze(0), audio_mix

    def dummy_mix_data(self, N):
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        mags = [None for n in range(N)]

        amp_mix = torch.zeros(1, self.HS, self.WS)
        phase_mix = torch.zeros(1, self.HS, self.WS)

        for n in range(N):
            frames[n] = torch.zeros(
                3, self.num_frames, self.imgSize, self.imgSize)
            audios[n] = torch.zeros(self.audLen)
            mags[n] = torch.zeros(1, self.HS, self.WS)
        return amp_mix, mags, frames, audios, phase_mix

    def mel_spectrogram(self, y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
        # 入力の音声が-1〜1に収まっていない場合に警告
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))

        # mel_basis と hann_window をキャッシュから取得
        mel_key = str(fmax) + '_' + str(y.device)
        if mel_key not in self.mel_basis_cache:
            mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
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