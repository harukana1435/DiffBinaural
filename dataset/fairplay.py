import os
import random
import numpy as np
import csv
from .base import BaseDataset


class FairPlayDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(FairPlayDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate #8
        self.audLen = opt.audLen #65535 11025だとしたら、5.94秒ぐらいになる

    def __getitem__(self, index):
        frames = None
        audios = None
        infos = []
        path_frames = []
        path_frames_ids = []
        #path_frames_det = ['' for n in range(N)] #今回はdetection結果はいらないとする
        path_audios = ''
        center_frames = 0

        if self.split == 'train':
            # the first video
            infos[0] = self.list_sample[index]
            cls = infos[0][0].split('/')[1]
            class_list.append(cls)
            for n in range(1, N):
                indexN = random.randint(0, len(self.list_sample)-1)
                sample = self.list_sample[indexN]
                while sample[0].split('/')[1] in class_list:
                    indexN = random.randint(0, len(self.list_sample) - 1)
                    sample = self.list_sample[indexN]
                infos[n] = sample
                class_list.append(sample[0].split('/')[1])
        elif self.split == 'val':
            infos[0] = self.list_sample[index]
            cls = infos[0][0].split('/')[1]
            class_list.append(cls)
            if not self.split == 'train':
                random.seed(1234)

            for n in range(1, N):
                indexN = random.randint(0, len(self.list_sample) - 1)
                sample = self.list_sample[indexN]
                while sample[0].split('/')[1] in class_list:
                    indexN = random.randint(0, len(self.list_sample) - 1)
                    print(indexN)
                    sample = self.list_sample[indexN]
                infos[n] = sample
                class_list.append(sample[0].split('/')[1])
        else:
            csv_lis_path = "/home/h-okano/DAVIS/MUSIC21/test.csv"
            csv_lis = []
            for row in csv.reader(open(csv_lis_path, 'r'), delimiter=','):
                if len(row) < 2:
                    continue
                csv_lis.append(row)
            random.seed(index) # fixed
            samples_4 = self.list_sample[index]
            samples = samples_4
            # samples = samples_4[::2]
            for n in range(N):
                sample = samples[n].replace(" ", "")
                for i in range(len(csv_lis)):
                    data = csv_lis[i]
                    if sample in data: 
                        infos[n] = data
                        break

        # select frames
        idx_margin = self.num_frames // 2 * self.stride_frames

        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            if self.split == 'train':
                # random, not to sample start and end n-frames
                if int(count_framesN)-idx_margin <= idx_margin+1:
                    center_frameN = int(count_framesN) // 2 + 1
                else:
                    center_frameN = random.randint(
                        idx_margin+1, int(count_framesN)-idx_margin)
            else:
                center_frameN = int(count_framesN) // 2 + 1
            center_frames[n] = center_frameN
            
            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                if center_frameN + idx_offset <= 1:
                    idx_offset = 1-center_frameN
                elif center_frameN + idx_offset >= int(count_framesN):
                    idx_offset = int(count_framesN) - center_frameN

                path_frames[n].append(
                    os.path.join("/home/h-okano/DAVIS/MUSIC21/frames",
                        path_frameN[1:],
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))
                path_frames_ids[n].append(center_frameN + idx_offset)
            
            # detectionは今回使わないはず
            # path_frames_det[n] = os.path.join("/YOUR_ROOT/MUSIC/detection_results",
            #             path_frameN[1:]+'.npy')

            path_audios[n] = os.path.join("/home/h-okano/DAVIS/MUSIC21/audios", path_audioN[1:])
        
        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames(path_frames[n])

                # jitter audio
                center_timeN = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix, audio_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)
            audio_mix = audios[0]

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags, 
        'audio_mix': audio_mix}
        ret_dict['audios'] = audios
        if self.split != 'train':
            # ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
