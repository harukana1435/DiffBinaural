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

class GenAudioPosDataset(torchdata.Dataset):
    def __init__(self, audio_path, opt):
        # params
        self.num_frames = opt.num_frames
        self.vidRate = opt.vidRate #8 動画のフレームレート
        self.imgSize = opt.imgSize
        self.audRate = opt.audRate
        self.audLen = opt.audLen #65536 16000Hzで読み込む
        self.audSec = 1. * self.audLen / self.audRate

        # STFT params
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.fft_size = opt.stft_frame
        self.num_mels = opt.num_mels
        self.fmin = 0
        self.fmax = opt.audRate//2
        
        self.mel_basis_cache = {}  # mel_basis をキャッシュするための辞書
        self.hann_window_cache = {}  # hann_window をキャッシュするための辞書

        #ディレクトリ
        self.dir_frames= opt.dir_frames
        self.dir_det_pos=opt.dir_det_pos
        
        self.max_sources = opt.max_sources

        self.seed = opt.seed
        random.seed(self.seed)
        
        self.split = opt.split

        # initialize video transform
        self._init_vtransform()
        
        
        
        self.audio_path=audio_path
        self.basename = os.path.splitext(os.path.basename(audio_path))[0]
        
        self.audio, sr = self._load_audio_file(audio_path)
        # resample
        if sr != self.audRate:
            print('resmaple {}->{}'.format(sr, self.audRate))
            audio = librosa.resample(audio, sr, self.audRate)
        
        self.genstart_list=[]
        
        sliding_window_start = 0
        self.generate_stride = self.audLen//4
        while sliding_window_start+self.audLen <= self.audio.shape[-1]:
            self.genstart_list.append(sliding_window_start)
            sliding_window_start+=self.generate_stride
        if sliding_window_start + self.audLen > self.audio.shape[-1]:
            self.genstart_list.append(sliding_window_start)
            padding_length = (sliding_window_start + self.audLen) - self.audio.shape[-1]
            self.audio = np.pad(self.audio,((0, 0), (0, padding_length)), 'constant')
            
        self.audio = torch.FloatTensor(self.audio)
        self.mix_audio = torch.FloatTensor(((self.audio[0] + self.audio[1]) / 2).unsqueeze(0))
        self.left_audio, self.right_audio = self.audio[0].unsqueeze(0), self.audio[1].unsqueeze(0)
        

        num_sample = len(self.genstart_list)
        assert num_sample > 0
        print('# {} samples: {}'.format(self.basename, num_sample))

    def __len__(self):
        return len(self.genstart_list)
    
    
    def __getitem__(self, index):
        frames = None
        audio_path = None

        start_point = self.genstart_list[index]    
        
        mix_audio = self.mix_audio[:, start_point:start_point+self.audLen]
    
        # メルスペクトログラムの計算
        mix_mel = self.mel_spectrogram_origin(mix_audio, self.fft_size, self.num_mels,
                                        self.audRate, self.stft_hop, self.stft_frame, 0, 11025)

        left_audio = self.left_audio[:, start_point:start_point+self.audLen]
            
        left_mel = self.mel_spectrogram_origin(left_audio, self.fft_size, self.num_mels,
                                              self.audRate, self.stft_hop, self.stft_frame, 0, 11025)
        
        right_audio = self.right_audio[:, start_point:start_point+self.audLen]
        
        right_mel = self.mel_spectrogram_origin(right_audio, self.fft_size, self.num_mels,
                                              self.audRate, self.stft_hop, self.stft_frame, 0, 11025)

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
            os.path.join(self.dir_frames, f"{self.basename}.mp4", f"{i:06d}.jpg") for i in even_frame_indices
        ]
        
        det_pos_data_path = os.path.join(self.dir_det_pos, self.basename+".npy")
        det_pos_data = np.load(det_pos_data_path, allow_pickle=True).item()
        
        for i, num in enumerate(even_frame_indices):
            if num >= det_pos_data['bounding_boxes'].shape[0]*2:
                even_frame_indices[i] = det_pos_data['bounding_boxes'].shape[0]*2
                
        det_data = [det_pos_data['bounding_boxes'][i//2-1] for i in even_frame_indices]
        
        frames, mask = self._load_frames_det(frame_paths, det_data)
        
        mask = np.array([mask for _ in range(self.num_frames)])
        
        pos_data = [det_pos_data['pos_3d'][i//2-1] for i in even_frame_indices]
        pos_data = np.array([np.pad(data, ((0, self.max_sources-data.shape[0]),(0,0)), constant_values=0) for data in pos_data])

            
        binaural_mel = np.concatenate((left_mel, right_mel), axis=0)

        ret_dict = {'mix_mel': mix_mel, 'binaural_mel':binaural_mel, 'frames': frames,
                    'pos_data':pos_data, 'mask':mask, 'start_time_frame':start_point//self.stft_hop, 'total_time_frame':self.audio.shape[-1]//self.stft_hop}
        return ret_dict

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
        self.det_transform = transforms.Compose([vtransforms.Stack()])
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
    
    
    def _load_frames_clip(self, paths):
        frames = []
        for path in paths:
            frames.append(preprocess(Image.open(path)))
        frames = self.clip_transform(frames)
        return frames

    def _load_frames_det(self, paths, det_data):
        frames = []
        mask = np.array([False]*self.max_sources)
        N = len(paths)
        sources_num = len(det_data[0])
        for n in range(N):
            source_frames = []
            path = paths[n]
            for source in range(self.max_sources):
                if source <= sources_num-1:
                    bb = det_data[n][source]
                    if not np.array_equal(bb, [0, 0, 0, 0]):
                        source_frames.append(self._load_frame_det(path, bb))
                        mask[source] = False
                    else:
                        source_frames.append(Image.new('RGB', (self.imgSize, self.imgSize), (0, 0, 0)))
                        mask[source] = True
                else:
                    source_frames.append(Image.new('RGB', (self.imgSize, self.imgSize), (0, 0, 0)))
                    mask[source] = True
            source_frames = self.vid_transform(source_frames)
            frames.append(source_frames)
        frames = self.det_transform(frames)
        return frames, mask


    def _load_frame_det(self, path, bb):
        # load image
        img = Image.open(path).convert('RGB')
        # get box
        img = img.crop((bb[0], bb[1], bb[2], bb[3]))
        #print(bb)
        return img

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img
    
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

    def mel_spectrogram(self, y, n_fft, num_mels, sampling_rate, hop_size, win_size):
        # mel_basis と hann_window をキャッシュから取得
        mel_key = str(y.device)
        if mel_key not in self.mel_basis_cache:
            mel = librosa_mel_fn(sampling_rate, n_fft, num_mels)
            self.mel_basis_cache[mel_key] = torch.from_numpy(mel).float().to(y.device)
        
        if mel_key not in self.hann_window_cache:
            self.hann_window_cache[mel_key] = torch.hann_window(win_size).to(y.device)

        # STFTを計算する
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=self.hann_window_cache[mel_key],
                          center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

        # 複素数の絶対値を計算する
        spec = torch.abs(spec)

        # メルスペクトログラムを計算する
        mel_spec = torch.matmul(self.mel_basis_cache[mel_key], spec)
        
        #mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)

        return mel_spec

    def mel_spectrogram_origin(self, y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
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
