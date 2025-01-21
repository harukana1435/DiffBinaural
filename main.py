# System libs
import os
import random
import time
import json

# Numerical libs
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.io.wavfile as wavfile
# from scipy.misc import imsave
from imageio import imwrite as imsave
from mir_eval.separation import bss_eval_sources
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# Our libs
from utils.arguments import ArgParser
from dataset import FairPlayDataset
from modules import models
from diffusion_utils import diffusion_pytorch
from utils.helpers import AverageMeter, magnitude2heatmap, \
    istft_reconstruction, warpgrid, makedirs
import warnings
# UserWarningとFutureWarningを無視する
#warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper, self).__init__()
        self.net_frame, self.net = nets
        self.sampler = diffusion_pytorch.GaussianDiffusion(
            self.net,
            image_size = 64,
            timesteps = 400,   # number of steps
            sampling_timesteps = 15, # if ddim else None
            loss_type = 'l1',    # L1 or L2
            objective = 'pred_noise', # pred_noise or pred_x0
            beta_schedule = 'cosine', #linear or cosine or sigmoid 64×64の画像なので、コサインとした
            ddim_sampling_eta = 1.,
            auto_normalize = False,
            min_snr_loss_weight=False
        )
        self.scale_factor = 0.15

    def move_to_device(self, device):
            """
            モジュール全体と内包モジュールを指定デバイスに移動する。

            Args:
                device (torch.device): 移動先のデバイス。
            """
            self.to(device)
            if hasattr(self.net_frame, "to"):
                self.net_frame.to(device)
            if hasattr(self.net, "to"):
                self.net.to(device)
            if hasattr(self.sampler, "to"):
                self.sampler.to(device)  # diffusion_pytorch.GaussianDiffusion も移動可能なら移動

            print(f"NetWrapper and its components moved to {device}")


    def forward(self, batch_data, args, t):
        mix_mel = batch_data['mix_mel'] # (B, C, F, T) C=1, F=64, T=64
        diff_mel = batch_data['diff_mel'] # (B, C, F, T) C=1, F=64, T=64
        frames = batch_data['frames'] #(B, L, C, H, W) L=4, C=3, H=224, W=224 
        pointclouds = batch_data['pointclouds']
        
        print(mix_mel.shape, flush=True)

        B = mix_mel.size(0)
        T = mix_mel.size(2)

        if args.weighted_loss:
            weight = mix_mel
            # weight = torch.clamp(weight, 1e-3, 10)
            weight = weight > 1e-3 #mixe_melの値が1e-3以上のものにweightをつけている
        else:
            weight = torch.ones_like(mix_mel)

        # LOG magnitude
        log_mix_mel = torch.log1p(mix_mel) * self.scale_factor #正規化
        log_diff_mel = torch.log1p(diff_mel) * self.scale_factor #正規化
        #log_mix_mel.clamp_(0., 1.)
        #log_diff_mel.clamp_(0., 1.)
        # detach
        log_mix_mel = log_mix_mel.detach()
        log_diff_mel = log_diff_mel.detach()

        # Frame feature (conditions)
        feat_frames = self.net_frame.forward_multiframe(frames, pool=False) #(B, C, T)
        
        # Loss
        loss_mel = 1e3*self.sampler(log_diff_mel, [log_mix_mel, feat_frames], log=False, weight=weight) #weightは分離音声に対して、一定のスペクトログラムはオフにする

        return loss_mel


    def sample(self, batch_data, args): #サンプルのときは最後に、hifiganに入れられるように正規化しないといけないが、hifigan側でやったほうがいいかも
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['frames']
        mag_mix = mag_mix + 1e-10

        N = args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix.to(args.device), grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n].float().to(args.device), grid_warp, align_corners=True)

        # LOG magnitude
        log_mag_mix = torch.log1p(mag_mix) * self.scale_factor
        log_mag0 = torch.log1p(mags[0]) * self.scale_factor
        log_mag2 = torch.log1p(mags[1]) * self.scale_factor
        log_mag_mix.clamp_(0., 1.)
        log_mag0.clamp_(0., 1.)
        log_mag2.clamp_(0., 1.)
        # detach        
        log_mag_mix = log_mag_mix.detach()
        log_mag0 = log_mag0.detach()
        log_mag2 = log_mag2.detach()
        

        # Frame feature (conditions)
        feat_frames = [None for n in range(N)]
        for n in range(N):
            feat_frames[n] = self.net_frame.forward_multiframe(frames[n].to(args.device), pool=False)
        
        # ddim sampling
        preds0 = self.sampler.ddim_sample(condition=[log_mag_mix, feat_frames[0]], shape=log_mag_mix.shape, return_all_timesteps = True, silence_mask_sampling=True)    
        preds1 = self.sampler.ddim_sample(condition=[log_mag_mix, feat_frames[1]], shape=log_mag_mix.shape, return_all_timesteps = True,silence_mask_sampling=True)

        pred0 = preds0[:, -1, ...]
        pred1 = preds1[:, -1, ...]

        pred0 = pred0 / self.scale_factor
        pred1 = pred1 / self.scale_factor
        X0_pred = torch.exp(pred0.abs()) - 1
        X1_pred = torch.exp(pred1.abs()) - 1

        pred_mags = [None for n in range(2)]
        pred_mags[0] = X0_pred  
        pred_mags[1] = X1_pred 

        return {'pred_mags': pred_mags, 'mag_mix': mag_mix, 'mags': mags}



SDR_pred = []
SDR_mix = []
COUNT = 0

# Calculate metrics
def calc_metrics(batch_data, outputs, args):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']

    pred_mags = outputs['pred_mags'].copy()

    # unwarp log scale
    N = 2 #args.num_mix-1
    B = mag_mix.size(0)
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, mag_mix.size(3), warp=False)).to(args.device)
            pred_mags[n] = F.grid_sample(pred_mags[n], grid_unwarp, align_corners=True)

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_mags[n] = pred_mags[n].detach().cpu().numpy()

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = pred_mags[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())
            SDR_mix.append(sdr_mix)
            SDR_pred.append(sdr)
    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]


# Visualize predictions
def output_visuals(batch_data, outputs, args):
    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    infos = batch_data['infos']
    mags = batch_data['mags']

    pred_mags = outputs['pred_mags'].copy()

    # unwarp log scale
    N = args.num_mix #-1
    B = mag_mix.size(0)
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, mag_mix.shape[3], warp=False)).to(args.device)
            pred_mags[n] = F.grid_sample(pred_mags[n], grid_unwarp, align_corners=True)
            mags[n] = F.grid_sample(mags[n], grid_unwarp, align_corners=True)

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_mags[n] = pred_mags[n].detach().cpu().numpy()

    # loop over each sample
    for j in range(B):
        row_elements = []

        # video names
        prefix = []
        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(args.vis, prefix))

        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        mix_amp = magnitude2heatmap(mag_mix[j, 0])
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        imsave(os.path.join(args.vis, filename_mixmag), mix_amp[::-1, :, :])
        wavfile.write(os.path.join(args.vis, filename_mixwav), args.audRate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):

            # GT and predicted audio recovery
            gt_mag = mags[n][j, 0]
            gt_mag = gt_mag.cpu().numpy()
            gt_wav = istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=args.stft_hop)
            pred_mag = pred_mags[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

            # ouput spectrogram (log of magnitude, show colormap)
            filename_gtmag = os.path.join(prefix, 'gtamp{}.jpg'.format(n+1))
            filename_predmag = os.path.join(prefix, 'predamp{}.jpg'.format(n+1))
            gt_mag = magnitude2heatmap(gt_mag)
            pred_mag = magnitude2heatmap(pred_mag)
            imsave(os.path.join(args.vis, filename_gtmag), gt_mag[::-1, :, :])
            imsave(os.path.join(args.vis, filename_predmag), pred_mag[::-1, :, :])

            # output audio
            filename_gtwav = os.path.join(prefix, 'gt{}.wav'.format(n+1))
            filename_predwav = os.path.join(prefix, 'pred{}.wav'.format(n+1))
            wavfile.write(os.path.join(args.vis, filename_gtwav), args.audRate, gt_wav)
            wavfile.write(os.path.join(args.vis, filename_predwav), args.audRate, preds_wav[n])


def evaluate(netWrapper, loader, history, epoch, args, writer):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=False)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    for i, batch_data in enumerate(loader):
        # forward pass
        outputs = netWrapper.module.sample(batch_data, args)

        # calculate metrics
        sdr_mix, sdr, sir, sar = calc_metrics(batch_data, outputs, args)

        sdr_mix_meter.update(sdr_mix)
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)
    
    print('[Eval Summary] Epoch: {},'
          'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
          .format(epoch,
                  sdr_mix_meter.average(),
                  sdr_meter.average(),
                  sir_meter.average(),
                  sar_meter.average()))
    
    history['val']['epoch'].append(epoch)
    history['val']['sdr'].append(sdr_meter.average())
    history['val']['sir'].append(sir_meter.average())
    history['val']['sar'].append(sar_meter.average())
    
    if args.mode != "eval":
        writer.add_scalar('eval sdr',
                        sdr_meter.average(),
                        epoch)
        writer.add_scalar('eval sir',
                        sir_meter.average(),
                        epoch)
        writer.add_scalar('eval sar',
                        sar_meter.average(),
                        epoch)

# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, args, writer, running_loss):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()

    for i, batch_data in enumerate(tqdm(loader, desc="Training Progress", ncols=100)):
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        # forward pass
        optimizer.zero_grad()
        t = torch.randint(0, args.num_train_timesteps, (args.batch_size,), device='cuda').long() #diffusion modelの中で、定義されてるからいらないかも
        err = netWrapper.forward(batch_data, args, t)
        err = err.mean()

        # backward
        err.backward()
        nn.utils.clip_grad_norm_(netWrapper.parameters(), 5.0)
        optimizer.step()

        running_loss += err.item()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_unet: {}, lr_frame: {}, '
                  'loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_unet, args.lr_frame, err.item()))
            writer.add_scalar('training loss',
                            running_loss / args.disp_iter,
                            epoch * len(loader) + i)
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.mean().item())
            running_loss = 0.0


def checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_frame, net_unet) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'
    
    with open(os.path.join(args.ckpt,'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_frame.state_dict(),
               '{}/frame_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_unet.state_dict(),
               '{}/unet_{}'.format(args.ckpt, suffix_latest))

    cur_sdr = history['val']['sdr'][-1]
    if cur_sdr > args.best_sdr:
        print("saving best at {} epoch".format(epoch))
        args.best_sdr = cur_sdr
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(args.ckpt, suffix_best))
        torch.save(net_unet.state_dict(),
                   '{}/unet_{}'.format(args.ckpt, suffix_best))


def create_optimizer(nets, args):
    (net_frame, net_unet) = nets
    param_groups = [{'params': net_unet.parameters(), 'lr': args.lr_unet},
                    {'params': net_frame.parameters(), 'lr': args.lr_frame}]
    return torch.optim.AdamW(param_groups)


def adjust_learning_rate(optimizer, args):
    args.lr_unet *= 0.5
    args.lr_frame *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5

def main(args):
    # Network Builders
    builder = models.ModelBuilder()
    net_frame = builder.build_visual(
        pool_type=args.img_pool,
        weights=args.weights_frame,
        arch_frame=args.arch_frame)
    net_unet = builder.build_unet(weights=args.weights_unet)
    nets = (net_frame, net_unet)

    # Dataset and Loader
    dataset_train = FairPlayDataset(
        args.list_train, args, split='train')
    dataset_val = FairPlayDataset(
        args.list_val, args, max_sample=args.num_val, split=args.split)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers), #元々は2だった
        drop_last=False)

    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    writer = SummaryWriter(f'{args.ckpt}/runs')
    args.writer = writer

    # Wrap networks for multiple GPUs
    netWrapper = NetWrapper(nets)

    print(f"Using {len(args.gpu_ids)} GPUs: {args.gpu_ids}")
        
    netWrapper.move_to_device(args.device)  # モデルをデバイスに移動
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=args.gpu_ids)  # ラップ
    # モデルの最初のパラメータのデバイスを確認
    model_device = next(netWrapper.module.net.parameters()).device
    print(f"The model is on device: {model_device}")


    # Set up optimizer
    optimizer = create_optimizer(nets, args)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'mel': []}}

    # Eval mode
    if args.mode == 'eval':
        args.testing = True
        evaluate(netWrapper, loader_val, history, 0, args, writer)
        print('Evaluation Done!')
        return
        
    running_loss = 0.
    
    # Training loop
    for epoch in range(1, args.num_epoch + 1):
        train(netWrapper, loader_train, optimizer, history, epoch, args, writer, running_loss)
        writer.flush()

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            args.testing = True
            evaluate(netWrapper, loader_val, history, epoch, args, writer)
            writer.flush()
            args.testing = False
            # checkpointing
            checkpoint(nets, history, epoch, args)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))  # 設定
    torch.cuda.set_device(args.gpu_ids[0])  # 明示的にデバイスを設定
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # experiment name
    if args.mode == 'train' or args.mode == 'eval':
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-frames{}'.format(args.num_frames)
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])
        args.id += '-lr_unet{}'.format(args.lr_unet)

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    if args.mode == 'train':
        makedirs(args.ckpt, remove=False)

    elif args.mode == 'eval':
        args.weights_unet = os.path.join(args.ckpt, 'unet_best.pth')
        args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")
    args.best_sdr = -float("inf")
    args.testing = False


    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
