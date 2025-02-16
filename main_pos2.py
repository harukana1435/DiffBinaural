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
from dataset.fairplay import FairPlayDataset
from dataset.fairplay_pos_left import FairPlayPosLeftDataset
from modules import models
from diffusion_utils import diffusion_pytorch
from utils.helpers import AverageMeter, magnitude2heatmap, \
    istft_reconstruction, warpgrid, makedirs, save_mel_to_tensorboard, _nested_map, save_checkpoint,load_checkpoint,scan_checkpoint
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
            image_size = 80,
            timesteps = 1000,   # number of steps
            sampling_timesteps = 25, # if ddim else None
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


    def forward(self, batch_data, args):
        mix_mel = batch_data['mix_mel'] # (B, C, F, T) C=1, F=64, T=64
        diff_mel = batch_data['diff_mel'] # (B, C, F, T) C=1, F=64, T=64
        frames = batch_data['frames'] #(B, C, L, N, H, W) L=5, N=4, C=3, H=224, W=224 
        pos = batch_data['pos_data'] # (B, L, N, 3) 距離、仰角、方位角の順番
        mask = batch_data['mask'] #(B, L, N)
        

        B = mix_mel.size(0)
        T = mix_mel.size(2)

        if args.weighted_loss:
            weight = mix_mel
            # weight = torch.clamp(weight, 1e-3, 10)
            weight = weight > 1e-3 #mixe_melの値が1e-3以上のものにweightをつけている
        else: #こっち
            weight = torch.ones_like(mix_mel)

        # LOG magnitude
        log_mix_mel = torch.log1p(mix_mel) * self.scale_factor #正規化
        log_diff_mel = torch.log1p(diff_mel) * self.scale_factor #正規化

        # detach
        log_mix_mel = log_mix_mel.detach()
        log_diff_mel = log_diff_mel.detach()

        # Frame feature (conditions)
        feat_frames = self.net_frame.forward_multiframe(frames, pos, mask) #(B, C)
        
        # Loss
        loss_mel = 1e3*self.sampler(log_diff_mel, [log_mix_mel, feat_frames], log=False, weight=weight) #weightは分離音声に対して、一定のスペクトログラムはオフにする

        return loss_mel


    def sample(self, batch_data, args): #サンプルのときは最後に、hifiganに入れられるように正規化しないといけないが、hifigan側でやったほうがいいかも
        model_device = next(self.net_frame.parameters()).device
        batch_data = _nested_map(batch_data, lambda x: x.to(model_device) if isinstance(x, torch.Tensor) else x)
        mix_mel = batch_data['mix_mel'] # (B, C, F, T) C=1, F=64, T=64
        diff_mel = batch_data['diff_mel'] # (B, C, F, T) C=1, F=64, T=64
        frames = batch_data['frames'] #(B, L, C, H, W) L=4, C=3, H=224, W=224
        pos = batch_data['pos_data'] # (B, L, N, 3) 距離、仰角、方位角の順番
        mask = batch_data['mask'] #(B, L, N)

        B = mix_mel.size(0)
        T = mix_mel.size(2)

        # LOG magnitude
        log_mix_mel = torch.log1p(mix_mel) * self.scale_factor #正規化
        
        # detach
        log_mix_mel = log_mix_mel.detach()
        
        # Frame feature (conditions)
        feat_frames = self.net_frame.forward_multiframe(frames, pos, mask) #(B, C)
        
        # ddim sampling
        preds = self.sampler.ddim_sample(condition=[log_mix_mel, feat_frames], return_all_timesteps = True, silence_mask_sampling = True)

        pred = preds[:, -1, ...]

        pred = pred / self.scale_factor
        pred_mag = torch.exp(pred.abs()) - 1

        return {'pred_mag': pred_mag, 'gt_mag': diff_mel}


def calc_metrics(batch_data, outputs, args):
    # メートルの初期化
    l2_distance_meter = AverageMeter()

    # 真のメルスペクトログラムと予測を取得
    gt_mag = batch_data['diff_mel']
    pred_mag = outputs['pred_mag']
    
    gt_mag = gt_mag.to(pred_mag.device)

    # バッチサイズの取得
    B = gt_mag.shape[0]

    # 各サンプルごとに処理
    for j in range(B):
        # gt_mag と pred_mag の L2 ノルム（距離）の計算
        # 各要素ごとの差のL2ノルムを計算
        l2_distance = np.linalg.norm(gt_mag[j].cpu().numpy() - pred_mag[j].cpu().numpy())

        # メートルを更新
        l2_distance_meter.update(l2_distance)

    return l2_distance_meter.average()


def evaluate(netWrapper, loader, history, epoch, args, writer):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=False)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    mel_l2 = AverageMeter()


    for i, batch_data in enumerate(loader):
        # forward pass
        outputs = netWrapper.module.sample(batch_data, args)

        # calculate metrics
        data_l2 = calc_metrics(batch_data, outputs, args)

        mel_l2.update(data_l2)
    
    print('[Eval Summary] Epoch: {},'
          'mel_l2: {:.4f}'
          .format(epoch, mel_l2.average()))
    
    history['val']['epoch'].append(epoch)
    history['val']['mel_l2'].append(mel_l2.average())
    
    save_mel_to_tensorboard(batch_data, outputs, writer, epoch)
    
    if args.mode != "eval":
        writer.add_scalar('eval mel_l2',
                        mel_l2.average(),
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
        err = netWrapper.forward(batch_data, args)
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


def basic_checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_frame, net_unet) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(net_frame.state_dict(),
               '{}/frame_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_unet.state_dict(),
               '{}/unet_{}'.format(args.ckpt, suffix_latest))    

    cur_mel_l2 = history['val']['mel_l2'][-1]
    if cur_mel_l2 < args.best_mel_l2:
        print("saving best at {} epoch".format(epoch))
        args.best_mel_l2 = cur_mel_l2
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(args.ckpt, suffix_best))
        torch.save(net_unet.state_dict(),
                   '{}/unet_{}'.format(args.ckpt, suffix_best))


def advanced_checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_frame, net_unet) = nets
    
    with open(os.path.join(args.ckpt,'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    checkpoint_path_history = "{}/history_{:06d}".format(args.ckpt, epoch)
    save_checkpoint(checkpoint_path_history, {'history':history})
    checkpoint_path_model = "{}/frame_{:06d}".format(args.ckpt, epoch)
    save_checkpoint(checkpoint_path_model, net_frame.state_dict())
    checkpoint_path_model = "{}/unet_{:06d}".format(args.ckpt, epoch)
    save_checkpoint(checkpoint_path_model, net_unet.state_dict())



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
    dataset_train = FairPlayPosLeftDataset(
        args.list_train, args, split='train')
    dataset_val = FairPlayPosLeftDataset(
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
    if args.history_path and args.mode=='train':
        history = torch.load(args.history_path)['history']
        last_epoch = history['val']['epoch'][-1]+1
        for epoch in range(1, last_epoch): #learning rateを調整している
            if epoch in args.lr_steps:
                adjust_learning_rate(optimizer, args)
    else:
        history = {
            'train': {'epoch': [], 'err': []},
            'val': {'epoch': [], 'err': [], 'mel_l2': []}}
        last_epoch = 1

    # Eval mode
    if args.mode == 'eval':
        args.testing = True
        evaluate(netWrapper, loader_val, history, 0, args, writer)
        print('Evaluation Done!')
        return
        
    running_loss = 0.
    
    # Training loop
    for epoch in range(last_epoch, args.num_epoch + 1):
        #evaluate(netWrapper, loader_val, history, epoch, args, writer)
        train(netWrapper, loader_train, optimizer, history, epoch, args, writer, running_loss)
        writer.flush()

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            args.testing = True
            evaluate(netWrapper, loader_val, history, epoch, args, writer)
            writer.flush()
            args.testing = False
            # checkpointing
            basic_checkpoint(nets, history, epoch, args)
            
            if epoch % (args.eval_epoch*10)==0:
                advanced_checkpoint(nets, history, epoch, args)

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
        if os.path.isdir(args.ckpt):
            frame_path = scan_checkpoint(args.ckpt, 'frame_')
            unet_path = scan_checkpoint(args.ckpt, 'unet_')
            history_path = scan_checkpoint(args.ckpt, 'history_')
            args.history_path = history_path
            if args.history_path:
                args.weights_unet = unet_path
                args.weights_frame = frame_path
                args.history_path = history_path
        else:    
            makedirs(args.ckpt, remove=False)
            args.history_path = None

    elif args.mode == 'eval':
        args.weights_unet = os.path.join(args.ckpt, 'unet_best.pth')
        args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")
    args.best_mel_l2 = float("inf")
    args.testing = False


    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
