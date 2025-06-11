import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from modules.attention import MaskedAttention

# helpers functions

def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))

    if(Relu):
        model.append(nn.ReLU())

    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class Resnet(nn.Module):
    def __init__(self, original_resnet,pool_type='maxpool', use_transformer=False):
        super(Resnet, self).__init__()
        self.pool_type = pool_type
        self.features = nn.Sequential(
            *list(original_resnet.children())[:-1])

        for param in self.features.parameters():
            param.requires_grad = False

        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = nn.Transformer(d_model=512, num_encoder_layers=3, num_decoder_layers=1, dim_feedforward=512, batch_first=True)

    def forward(self, x, pool=True):
        x = self.features(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        x = self.features(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x[:, 0:1, ...]
        x = x.permute(0, 2, 1, 3, 4)

        x = torch.mean(x, dim=(3,4))

        # transformer
        if self.use_transformer:
            x = self.transformer(x.transpose(1,2), x.transpose(1,2)).transpose(1,2)

        if not pool:
            return x

        x = torch.mean(x, dim=2)

        x = x.view(B, C)
        return x

class Clip(nn.Module):
    def __init__(self, model, pool_type='maxpool', use_transformer=False):
        super(Clip, self).__init__()
        self.pool_type = pool_type
        self.model = model
        #print(*list(model.children()))
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.use_transformer = use_transformer
        if use_transformer:
            self.temporal_transformer = nn.Transformer(d_model=512, num_encoder_layers=3, num_decoder_layers=1, dim_feedforward=2048, batch_first=True)

            # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            # self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # for param in self.temporal_transformer.parameters():
        #     param.requires_grad = False

    def forward(self, x, pool=True):
        x = self.model.encode_image(x)

        return x

    def forward_text(self, x):
        x = self.model.encode_text(x)
        return x
        
    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        
        x = self.model.encode_image(x)

        (_, C) = x.size()
        x = x.view(B, T, C)

        # transformer
        # x = self.temporal_transformer(x.transpose(1,2)).transpose(1,2)
        x = self.temporal_transformer(x, x).transpose(1,2) #(B, C, T)
        
        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = torch.mean(x, 2)
        elif self.pool_type == 'maxpool':
            x = torch.max(x, 2)[0]

        return x


class Clip(nn.Module):
    def __init__(self, model, pool_type='maxpool', use_transformer=False):
        super(Clip, self).__init__()
        self.pool_type = pool_type
        self.model = model
        self.emb_dim = 512
        #print(*list(model.children()))
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.temporal_transformer_encoder = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,    # 埋め込み次元
            nhead=8,                  # ヘッド数
            dim_feedforward=2048,     # フィードフォワードネットワークのサイズ
            batch_first=True          # バッチサイズが最初に来る場合
        )
            # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            # self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # for param in self.temporal_transformer.parameters():
        #     param.requires_grad = False

    def forward(self, x, pool=True):
        x = self.model.encode_image(x)

        return x

    def forward_text(self, x):
        x = self.model.encode_text(x)
        return x
        
    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        
        x = self.model.encode_image(x)

        (_, C) = x.size()
        x = x.view(B, T, C)

        # transformer
        x = self.temporal_transformer_encoder(x) #(B, T, C)
        
        x = torch.mean(x, dim=1) #(B, 512)

        return x
    
    
    
    
class Clip_Pos(nn.Module):
    def __init__(self, model, pool_type='maxpool', dropout = 0.1):
        super(Clip_Pos, self).__init__()
        self.pool_type = pool_type
        self.model = model
        self.max_sources = 4
        
        self.emb_dim = 512
        
        #print(*list(model.children()))
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.pos_emb = SinusoidalPosEmb(64)
        self.pos_emb_mlp = nn.Linear(192, 1024)
        self.pos_emb_act = nn.GELU()
        
       
        
        self.pos_attention = MaskedAttention(query_dim=self.emb_dim, heads=8, dim_head=64) #query_dimは音源の最大値であるN=4がはいる。
        self.pos_layer1 = nn.LayerNorm(self.emb_dim)
        self.pos_ff = PositionwiseFeedForward(self.emb_dim, self.emb_dim*4)
        self.pos_layer2 = nn.LayerNorm(self.emb_dim)
        self.pos_dropout = nn.Dropout(dropout)
        
        self.temporal_transformer_encoder = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,    # 埋め込み次元
            nhead=8,                  # ヘッド数
            dim_feedforward=2048,     # フィードフォワードネットワークのサイズ
            batch_first=True          # バッチサイズが最初に来る場合
        )
            # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            # self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # for param in self.temporal_transformer.parameters():
        #     param.requires_grad = False

    def forward(self, x, pool=True):
        x = self.model.encode_image(x)

        return x

    def forward_text(self, x):
        x = self.model.encode_text(x)
        return x
        
    def forward_multiframe(self, x, pos, mask):
        (B, C, T, N, H, W) = x.size()
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B * T * N, C, H, W)
        
        x = self.model.encode_image(x)

        (_, C) = x.size()
        x = x.view(B*T, N, C)

        pos = pos.view(B*T*N*3)
        pos = self.pos_emb(pos)
        pos = pos.view(B*T*N, -1)
        pos = self.pos_emb_mlp(pos)
        pos = self.pos_emb_act(pos)
        pos = pos.view(B*T, N, 1024)
        scale, shift = pos.chunk(2, dim=2)
        x = x * (scale + 1) + shift
        
        mask = mask.view(B*T, N)
        pos_attn = self.pos_attention(x, mask) #(B*T, N, 512)
        x = x + self.pos_dropout(pos_attn)
        x = self.pos_layer1(x)
        ff_output = self.pos_ff(x)
        x = x + self.pos_dropout(ff_output)
        x = self.pos_layer2(x)
        
        x = torch.max(x, dim=1)[0] #(B*T, 512)
        
        x = x.view(B, T, 512)
        
        # transformer
        x = self.temporal_transformer_encoder(x)
        
        x = torch.mean(x, dim=1) #(B, 512)

        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): The dimension of the model (also the input and output dimension).
            d_ff (int): The dimension of the feed-forward hidden layer.
            dropout (float): Dropout probability.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor, shape [batch_size, seq_len, d_model]

        Returns:
            Tensor: Output tensor, shape [batch_size, seq_len, d_model]
        """
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class Clip_Pos2D(nn.Module):
    def __init__(self, model, pool_type='maxpool', dropout = 0.1):
        super(Clip_Pos2D, self).__init__() # Corrected super call
        self.pool_type = pool_type
        self.model = model
        self.max_sources = 4

        self.emb_dim = 512 # Dimension of CLIP image features

        #print(*list(model.children()))
        for param in self.model.parameters():
            param.requires_grad = False

        # Sinusoidal embeddings for elevation and azimuth
        self.pos_emb_dim = 64 # Dimension for each angle's sinusoidal embedding
        self.pos_emb_ele = SinusoidalPosEmb(self.pos_emb_dim) # Elevation embedding
        self.pos_emb_azi = SinusoidalPosEmb(self.pos_emb_dim) # Azimuth embedding

        # Separate MLPs for scale (from elevation) and shift (from azimuth)
        self.mlp_scale = nn.Sequential(
            nn.Linear(self.pos_emb_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.emb_dim) # Output 512 dims for scale
        )
        self.mlp_shift = nn.Sequential(
            nn.Linear(self.pos_emb_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.emb_dim) # Output 512 dims for shift
        )

        # Attention and feedforward layers for refining features after positional modulation
        self.pos_attention = MaskedAttention(query_dim=self.emb_dim, heads=8, dim_head=64)
        self.pos_layer1 = nn.LayerNorm(self.emb_dim)
        self.pos_ff = PositionwiseFeedForward(self.emb_dim, self.emb_dim*4)
        self.pos_layer2 = nn.LayerNorm(self.emb_dim)
        self.pos_dropout = nn.Dropout(dropout)
        
        self.temporal_transformer_encoder = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,    # 埋め込み次元
            nhead=8,                  # ヘッド数
            dim_feedforward=2048,     # フィードフォワードネットワークのサイズ
            batch_first=True          # バッチサイズが最初に来る場合
        )
            # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            # self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # for param in self.temporal_transformer.parameters():
        #     param.requires_grad = False

    def forward(self, x, pool=True):
        x = self.model.encode_image(x)

        return x

    def forward_text(self, x):
        x = self.model.encode_text(x)
        return x
        
    def forward_multiframe(self, x, pos, mask):
        (B, C, T, N, H, W) = x.size()
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B * T * N, C, H, W)
        
        x = self.model.encode_image(x)

        (_, C) = x.size()
        x = x.view(B*T, N, C) # x shape: (B*T, N, 512)

        # --- Positional Embedding and Modulation ---
        # Assuming pos shape is (B, T, N, 2) where pos[..., 0] is elevation, pos[..., 1] is azimuth
        pos_ele = pos[..., 0] # Elevation (B, T, N)
        pos_azi = pos[..., 1] # Azimuth (B, T, N)

        # Apply sinusoidal embeddings
        # Reshape angles to (B*T*N) before passing to embedding
        emb_ele = self.pos_emb_ele(pos_ele.reshape(-1)) # (B*T*N, 64)
        emb_azi = self.pos_emb_azi(pos_azi.reshape(-1)) # (B*T*N, 64)

        # Calculate scale from elevation embedding and shift from azimuth embedding
        scale_flat = self.mlp_scale(emb_ele) # (B*T*N, 512)
        shift_flat = self.mlp_shift(emb_azi) # (B*T*N, 512)

        # Reshape scale and shift to match x's dimensions for broadcasting
        scale = scale_flat.view(B*T, N, self.emb_dim) # (B*T, N, 512)
        shift = shift_flat.view(B*T, N, self.emb_dim) # (B*T, N, 512)

        # Apply scale and shift to image features x
        x = x * (scale + 1) + shift
        # --- End Positional Embedding and Modulation ---

        mask = mask.view(B*T, N)
        # Apply attention mechanism using the modulated features
        pos_attn = self.pos_attention(x, mask) #(B*T, N, 512)
        x = x + self.pos_dropout(pos_attn)
        x = self.pos_layer1(x)
        ff_output = self.pos_ff(x)
        x = x + self.pos_dropout(ff_output)
        x = self.pos_layer2(x)
        
        x = torch.max(x, dim=1)[0] #(B*T, 512)
        
        x = x.view(B, T, 512)
        
        # transformer
        x = self.temporal_transformer_encoder(x)
        
        x = torch.mean(x, dim=1) #(B, 512)

        return x
