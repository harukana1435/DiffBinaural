from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from modules.norms import *


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out * 2, 1)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim_out, 1)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, time_emb_dim = None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads #32 × 4
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 2)
        ) if exists(time_emb_dim) else None


    def forward(self, x, time_emb = None, f_attn=False, t_attn=False):
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        _, _, t, f = x.shape

        if f_attn:
            x = rearrange(x, 'b c t (x y) -> b (c x) t y', x = f//f_attn, y = f_attn)           
        elif t_attn:
            x = rearrange(x, 'b c (x y) f -> b (c x) y f', x = t//t_attn, y = t_attn)  
        else:
            pass

        b, c, h, w = x.shape

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)


        if f_attn:
            out = rearrange(out, 'b (c x) t y -> b c t (x y)', x = f//f_attn, y = f_attn)  
        elif t_attn:
            out = rearrange(out, 'b (c x) y f -> b c (x y) f', x = t//t_attn, y = t_attn)    
        else:
            pass

        return self.to_out(out)
    
    
class LinearAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, f_attn=None, t_attn=None):
        super().__init__()

        self.f_attn = f_attn
        self.t_attn = t_attn

        # 周波数方向のアテンション
        self.f_linear_attn = LinearAttention(dim, heads=n_heads, dim_head=d_head)

        # 時間軸方向のアテンション
        self.t_linear_attn = LinearAttention(dim, heads=n_heads, dim_head=d_head)

        # Conv2dで元の形状に戻すための層
        self.conv_out = nn.Conv2d(dim*2, dim, 1)

    def forward(self, x):
        # 周波数方向のアテンション
        f_attn_out = self.f_linear_attn(x, f_attn=self.f_attn)

        # 時間軸方向のアテンション
        t_attn_out = self.t_linear_attn(x, t_attn=self.t_attn)

        # 残差接続と連結
        combined = torch.cat([f_attn_out, t_attn_out], dim=1)

        # Conv2dで元の形状に戻す
        out = self.conv_out(combined)

        return out

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, time_emb_dim = None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 2)
        ) if exists(time_emb_dim) else None

    def forward(self, x, time_emb = None):
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)



class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):# 如果设置了context_dim就不是自注意力了
        super().__init__()
        inner_dim = dim_head * heads # inner_dim == SpatialTransformer.model_channels
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(query_dim, inner_dim, 1)
        self.to_k = nn.Conv2d(context_dim, inner_dim, 1)
        self.to_v = nn.Conv2d(context_dim, inner_dim, 1)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):# x:(b,c,h,w), context:(b,context_dim)
        b, c, h, w = x.shape
        q = self.to_q(x)# q:(b,inner_dim, h, w)

        context = context[:, :, None, None].expand(-1, -1, q.shape[-2], q.shape[-1]) if context is not None else x #context:(b, context_dim, h, w)

        
        k = self.to_k(context)# (b,,inner_dim, h, w)
        v = self.to_v(context)# (b,,inner_dim, h, w)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), (q, k, v))# n is seq_len for k and v

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        
        if exists(mask):  # マスクが存在する場合
            max_neg_value = -torch.finfo(sim.dtype).max  # 負の最大値（マスク対象に適用）
            
            # マスクの形状を (B, H, N_q, N_k) に変換
            mask = rearrange(mask, 'b j -> b 1 1 j')  # (B, 1, 1, N_k) に変換
            mask = repeat(mask, 'b 1 1 j -> b h i j', h=self.heads, i=q.shape[2])  # (B, H, N_q, N_k)
        
            # マスクを適用（False の位置に max_neg_value を埋める）
            sim.masked_fill_(~mask, max_neg_value)
                    
        
        attn = sim.softmax(dim = -1)
        
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class UnifiedAttention(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, resolutions = [1, 4, 8], time_emb_dim = None):
        super().__init__()
        self.self_attn = Attention(dim, heads=n_heads, dim_head=d_head, time_emb_dim=time_emb_dim)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # self.linear_attn1 = LinearAttentionBlock(dim, heads=n_heads, dim_head=d_head,f_attn=1, t_attn=1)
        # self.linear_attn4 = LinearAttentionBlock(dim, heads=n_heads, dim_head=d_head,f_attn=4, t_attn=4)
        self.linear_attn8 = LinearAttentionBlock(dim, heads=n_heads, dim_head=d_head,f_attn=8, t_attn=8)
        self.closs_attn = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is cross-attention
        self.norm1 = LayerNorm(dim)
        # self.norm2 = LayerNorm(dim)
        # self.norm3 = LayerNorm(dim)
        self.norm4 = LayerNorm(dim)
        self.norm5 = LayerNorm(dim)
        self.norm6 = LayerNorm(dim)

    def forward(self, x, context=None, time_emb=None):
        x = self.self_attn(self.norm1(x), time_emb) + x
        # x = self.linear_attn1(self.norm2(x)) + x
        # x = self.linear_attn4(self.norm3(x)) + x
        x = self.linear_attn8(self.norm4(x)) + x
        x = self.closs_attn(self.norm5(x), context=context) + x
        x = self.ff(self.norm6(x)) + x
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, n_heads=4, d_head=32, dropout=0, context_dim=None, groups=8, time_emb_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = in_channels
        self.gnorm = nn.GroupNorm(groups, in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 1)

        self.transformer_block = UnifiedAttention(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, time_emb_dim=time_emb_dim)
        self.proj_out = nn.Conv2d(inner_dim,
                                in_channels,
                                1)

    def forward(self, x, context=None, time_emb = None):
        b, c, h, w = x.shape
        x_in = x
        x = self.gnorm(x)# group norm
        x = self.proj_in(x)# no shape change
        
        x = self.transformer_block(x, context=context, time_emb = time_emb)# context shape [b,context_dim]
        
        x = self.proj_out(x)
        return x + x_in