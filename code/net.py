import os
import math
import torch
import numbers
import math
import torch.nn as nn
import scipy.io as sio
from skimage import io
import torch.optim as optim
from operator import truediv
from einops import rearrange
import torch.nn.functional as F

from torch_3D_wavelets import DWT_3D, IDWT_3D
from torch_wavelets import DWT_2D, IDWT_2D
import parameter

from model.CRNet import CRBlock

parameter._init()


# 残差单元
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # Resudual connect: fn(x) + x
        return self.fn(x, **kwargs) + x


# 层归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # using Layer Normalization before input to fn layer
        return self.fn(self.norm(x), **kwargs)


# 前馈网络
class FeedForward(nn.Module):
    # Feed Forward Neural Network
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # Two linear network with GELU and Dropout
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FusionAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        # get q,k,v from a single weight matrix
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # print("fusion Attention")
        b, _, n, _, h = *x.shape, self.heads
        feature = x[:, 0, :, :]
        proto = x[:, 1, :, :]
        q = self.to_q(feature)
        k = self.to_k(proto)
        v = self.to_v(proto)
        # split q,k,v from [batch, patch_num, head_num*head_dim] -> [batch, head_num, patch_num, head_dim]
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        # transpose(k) * q / sqrt(head_dim) -> [batch, head_num, patch_num, patch_num]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # concat all output -> [batch, patch_num, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        # Linear + Dropout
        out = self.to_out(out)
        out = out.unsqueeze(1)
        out = torch.cat([out, out], dim=1)
        # out: [batch, patch_num, embedding_dim]
        return out


class WaveAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, sr_ratio=1, dropout=0.):
        super().__init__()

        self.dim = dim  # (输入维度) 512
        self.num_heads = heads  # (heads数) 4
        self.dim_head = dim_head  # (heads维度) 128
        self.inner_dim = self.dim_head * self.num_heads  # =128*4=512

        self.scale = self.dim_head ** -0.5
        self.sr_ratio = sr_ratio

        self.dwt = DWT_3D(wave='haar')
        self.idwt = IDWT_3D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(self.inner_dim, self.inner_dim // 4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.inner_dim // 4),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(self.inner_dim, self.inner_dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(self.inner_dim),
            nn.ReLU(inplace=True),
        )
        self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.q = nn.Linear(dim, self.inner_dim, bias=False)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.inner_dim, bias=False)
        )
        self.proj = nn.Linear(self.inner_dim + self.inner_dim // 4, dim)

    def forward(self, x):
        # print("wave Attention")
        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        # print(B, H, W, C)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x_dwt = self.dwt(self.reduce(x))
        x_dwt = self.filter(x_dwt)
        x_idwt = self.idwt(x_dwt)
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-2) * x_idwt.size(-1)).transpose(1, 2)
        kv = self.kv_embed(x_dwt).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.kv(kv).reshape(B, -1, int(H / 2), self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(torch.cat([x, x_idwt], dim=-1))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, type):
        super().__init__()
        self.layers = nn.ModuleList([])
        if type == 0:
            for _ in range(depth):
                # using multi-self-attention and feed forward neural network repeatly
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, WaveAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))
        if type == 1:
            for _ in range(depth):
                # using multi-self-attention and feed forward neural network repeatly
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, FusionAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x



class fusionBlock(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim,
                 channels, dim_head, dropout=0., emb_dropout=0.):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2
        self.image_size = image_size
        self.patch_size = patch_size
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, dim), requires_grad=True)  ####

        self.to_embedding = nn.Linear(self.patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # BaseNet
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, 1)

        # 消融实验2.去掉fusionAttn
        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, 2)

        self.embedding_to = nn.Linear(dim, self.patch_dim)

    def forward(self, proto, x):
        p = self.patch_size
        b, c, h, w = x.shape
        hh = int(h / p)
        x_embed = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        proto_embed = rearrange(proto, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x_embed = self.to_embedding(x_embed)
        proto_embed = self.to_embedding(proto_embed)
        b, n, c = x_embed.shape
        x_embed += self.pos[:, :n]
        x_embed = self.dropout(x_embed)

        proto_embed += self.pos[:, :n]
        proto_embed = self.dropout(proto_embed)

        x_embed = x_embed.unsqueeze(1)
        proto_embed = proto_embed.unsqueeze(1)

        embed = torch.cat([x_embed, proto_embed], dim=1)
        embed = self.transformer(embed)
        embed = embed[:, 0, :, :]
        x = self.embedding_to(embed)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=hh, p1=p, p2=p)
        return x


class mixblock(nn.Module):
    def __init__(self, n_feats):
        super(mixblock, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.conv2=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.ones(1))
    def forward(self,x):
        return self.alpha*self.conv1(x)+self.beta*self.conv2(x)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


import torch.nn.functional as f
class selfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(selfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = 1.0 / (out_channels ** 0.5)

    def forward(self, feature, feature_map):
        query = self.query_conv(feature)
        key = self.key_conv(feature)
        value = self.value_conv(feature)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * self.scale

        attention_weights = f.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value)

        output_feature_map = (feature_map + attended_values)

        return output_feature_map


class FIM(nn.Module):
    def __init__(self, n_feats):
        super(FIM, self).__init__()
        self.encoder = mixblock(n_feats)
        self.decoder_high = mixblock(n_feats)  # nn.Sequential(one_module(n_feats),

        self.decoder_low = nn.Sequential(mixblock(n_feats), mixblock(n_feats), mixblock(n_feats))
        self.alise = nn.Conv2d(n_feats,n_feats,1,1,0,bias=False)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats*2,n_feats,3,1,1,bias=False)  # one_module(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)
        self.raw_alpha=nn.Parameter(torch.ones(1))

        self.raw_alpha.data.fill_(0)
        self.ega=selfAttention(n_feats, n_feats)
        self.n_feats = n_feats
    def forward(self, x, y):

        # 获取维度
        B, C1, C2, H, W = x.shape

        # Step 1: 合并通道维度 C1 和 C2 → [B, C1*C2, H, W]
        x = x.view(B, C1 * C2, H, W)

        # Step 2: 压缩通道维度
        compress = nn.Conv2d(in_channels=C1 * C2, out_channels=self.n_feats, kernel_size=1).to(x.device)
        x = compress(x)  # [B, 64, H, W]

        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)

        # high=high+self.ega(high,high)*self.raw_alpha
        x2=x2+self.ega(x2,x2)*self.raw_alpha
        x2=self.decoder_low(x2)

        y1 = self.encoder(y)
        y2 = self.down(y1)
        y_high = y1 - F.interpolate(y2, size=y.size()[-2:], mode='bilinear', align_corners=True)

        # high=high+self.ega(high,high)*self.raw_alpha
        y2=y2+self.ega(y2,y2)*self.raw_alpha
        y2=self.decoder_low(y2)

        x3 = x2+y2

        # x3 = self.decoder_low(x2)
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)

        x = self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x

        y3 = y2
        # x3 = self.decoder_low(x2)
        y_high1 = self.decoder_high(y_high) + high1
        y4 = F.interpolate(y3, size=y.size()[-2:], mode='bilinear', align_corners=True)

        y = self.alise(self.att(self.alise2(torch.cat([y4, y_high1], dim=1)))) + y

        # Step 3: 解压为原通道数
        decompress = nn.Conv2d(in_channels=self.n_feats, out_channels=C1 * C2, kernel_size=1).to(x.device)
        x_decompressed = decompress(x)  # [B, C1*C2, H, W]

        # Step 4: reshape 回原始形状
        x = x_decompressed.view(B, C1, C2, H, W)



        return x, y



class PICNet(nn.Module):
    def __init__(self, out_features, layer_num=3):
        super(PICNet, self).__init__()
        self.out_features = out_features
        self.layer_num = layer_num
        cuda = parameter.get_value('cuda')
        batch_size = parameter.get_value('batch_size')
        if cuda == 'cuda0':
            device = torch.device("cuda:0")
        if cuda == 'cuda1':
            device = torch.device("cuda:1")

        channels = parameter.get_value('channels')
        data_type = parameter.get_value('data_type')
        window_size = parameter.get_value('windowSize')

        if window_size == 8:
            last_kernel_size = 1
            last_image_size = 4
        else:

            last_kernel_size = 3
            last_image_size = window_size - 6
        last_channels = [1024, 1536, 2304, 4096, 6400][(window_size - 8) // 2]

        self.hsi_prototype = nn.Parameter(torch.ones(batch_size, 256, last_image_size, last_image_size),
                                          requires_grad=True)
        self.sar_prototype = nn.Parameter(torch.ones(batch_size, 256, last_image_size, last_image_size),
                                          requires_grad=True)
        self.hsi_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(9, 3, 3), padding=0),
            nn.BatchNorm3d(num_features=8),
            nn.ReLU(inplace=True)
        )
        self.hsi_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(7, 3, 3), padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(inplace=True)
        )
        self.hsi_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, last_kernel_size, last_kernel_size), padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.hsi_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32 * (channels - 18), out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        # SAR
        if data_type in {3, 4}:
            self.sar_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=0),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            )
            self.sar_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(inplace=True)
            )
            self.sar_conv3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=last_kernel_size, padding=0),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(inplace=True)
            )
        # LIDAR
        elif data_type in {0, 1, 2}:
            self.sar_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            )
            self.sar_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(inplace=True)
            )
            self.sar_conv3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=last_kernel_size, padding=0),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(inplace=True)
            )
        else:
            print("No dataTpye to sarconv")

        self.fusionBlock = fusionBlock(image_size=last_image_size, patch_size=1, dim=512, depth=0, heads=8,
                                       mlp_dim=1024,
                                       channels=256, dim_head=64, dropout=0., emb_dropout=0)

        self.drop_hsi = nn.Dropout(0.6)
        self.drop_sar = nn.Dropout(0.6)
        self.drop_fusion = nn.Dropout(0.6)

        self.fusionlinear_hsi = nn.Linear(in_features=last_channels, out_features=self.out_features)
        self.fusionlinear_sar = nn.Linear(in_features=last_channels, out_features=self.out_features)
        self.fusionlinear_fusion = nn.Linear(in_features=last_channels * 2, out_features=self.out_features)
        self.weight = nn.Parameter(torch.ones(2))
        self.fim1 = FIM(64)
        self.fim2 = FIM(128)
        self.fim3 = FIM(256)
    def forward(self, hsi, sar):
        
        hsi_feat1 = self.hsi_conv1(hsi)
        sar_feat1 = self.sar_conv1(sar)
        for i in range(self.layer_num):
            hsi_fim1, sar_fim1 = self.fim1(hsi_feat1, sar_feat1)
        hsi_feat1 = hsi_feat1 + hsi_fim1
        sar_feat1 = sar_feat1 + sar_fim1

        hsi_feat2 = self.hsi_conv2(hsi_feat1)
        sar_feat2 = self.sar_conv2(sar_feat1)
        for i in range(self.layer_num):
            hsi_fim2, sar_fim2 = self.fim2(hsi_feat2, sar_feat2)
        hsi_feat2 = hsi_feat2 + hsi_fim2
        sar_feat2 = sar_feat2 + sar_fim2

        
        hsi_feat3 = self.hsi_conv3(hsi_feat2)
        sar_feat3 = self.sar_conv3(sar_feat2)
        for i in range(self.layer_num):
            hsi_fim3, sar_fim3 = self.fim3(hsi_feat3, sar_feat3)
        hsi_feat3 = hsi_feat3 + hsi_fim3
        sar_feat3 = sar_feat3 + sar_fim3


        hsi_feat3 = hsi_feat3.reshape(-1, hsi_feat3.shape[1] * hsi_feat3.shape[2], hsi_feat3.shape[3],
                                      hsi_feat3.shape[4])
        hsi_feat3 = self.hsi_conv4(hsi_feat3)

        hsi_compensation = self.fusionBlock(self.sar_prototype, hsi_feat3)
        hsi_fusion = hsi_feat3 + hsi_compensation
        hsi_feat4 = hsi_fusion.reshape(-1, hsi_fusion.shape[1], hsi_fusion.shape[2] * hsi_fusion.shape[3])

        sar_compensation = self.fusionBlock(self.hsi_prototype, sar_feat3)
        sar_feat3 = sar_feat3 + sar_compensation
        sar_feat4 = sar_feat3.reshape(-1, sar_feat3.shape[1], sar_feat3.shape[2] * sar_feat3.shape[3])

        fusion_feat = torch.cat((hsi_feat4, sar_feat4), dim=1)

        hsi_feat = F.max_pool1d(hsi_feat4, kernel_size=4)
        hsi_feat = hsi_feat.reshape(-1, hsi_feat.shape[1] * hsi_feat.shape[2])

        sar_feat = F.max_pool1d(sar_feat4, kernel_size=4)
        sar_feat = sar_feat.reshape(-1, sar_feat.shape[1] * sar_feat.shape[2])
        fusion_feat = F.max_pool1d(fusion_feat, kernel_size=4)
        fusion_feat = fusion_feat.reshape(-1, fusion_feat.shape[1] * fusion_feat.shape[2])

        hsi_feat = self.drop_hsi(hsi_feat)
        sar_feat = self.drop_sar(sar_feat)
        fusion_feat = self.drop_fusion(fusion_feat)

        output_hsi = self.fusionlinear_hsi(hsi_feat)
        output_sar = self.fusionlinear_sar(sar_feat)
        output_fusion = self.fusionlinear_fusion(fusion_feat)

        weights = torch.sigmoid(self.weight)
        outputs = weights[0] * output_hsi + weights[1] * output_sar + output_fusion
        return outputs, hsi_feat3, hsi_compensation, sar_feat3, sar_compensation


class BaseNet(nn.Module):
    def __init__(self, out_features):
        super(BaseNet, self).__init__()
        self.out_features = out_features
        self.hsi_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(9, 3, 3), padding=0),
            nn.BatchNorm3d(num_features=8),
            nn.ReLU(inplace=True)
        )
        self.hsi_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(7, 3, 3), padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(inplace=True)
        )
        self.hsi_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, 3, 3), padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.hsi_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        self.sar_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.sar_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        self.sar_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.drop_hsi = nn.Dropout(0.6)
        self.drop_sar = nn.Dropout(0.6)
        self.drop_fusion = nn.Dropout(0.6)

        self.fusionlinear_hsi = nn.Linear(in_features=1024, out_features=self.out_features)
        self.fusionlinear_sar = nn.Linear(in_features=1024, out_features=self.out_features)
        self.fusionlinear_fusion = nn.Linear(in_features=2048, out_features=self.out_features)
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, hsi, sar):
        sar_feat1 = self.sar_conv1(sar)
        sar_feat2 = self.sar_conv2(sar_feat1)
        sar_feat3 = self.sar_conv3(sar_feat2)
        sar_feat4 = sar_feat3.reshape(-1, sar_feat3.shape[1],
                                      sar_feat3.shape[2] * sar_feat3.shape[3])

        hsi_feat1 = self.hsi_conv1(hsi)
        hsi_feat2 = self.hsi_conv2(hsi_feat1)
        hsi_feat3 = self.hsi_conv3(hsi_feat2)
        hsi_feat3 = hsi_feat3.reshape(-1, hsi_feat3.shape[1] * hsi_feat3.shape[2],
                                      hsi_feat3.shape[3], hsi_feat3.shape[4])
        hsi_feat3 = self.hsi_conv4(hsi_feat3)
        hsi_feat4 = hsi_feat3.reshape(-1, hsi_feat3.shape[1],
                                      hsi_feat3.shape[2] * hsi_feat3.shape[3])

        fusion_feat = torch.cat((hsi_feat4, sar_feat4), dim=1)

        hsi_feat = F.max_pool1d(hsi_feat4, kernel_size=4)
        hsi_feat = hsi_feat.reshape(-1, hsi_feat.shape[1] * hsi_feat.shape[2])
        sar_feat = F.max_pool1d(sar_feat4, kernel_size=4)
        sar_feat = sar_feat.reshape(-1, sar_feat.shape[1] * sar_feat.shape[2])
        fusion_feat = F.max_pool1d(fusion_feat, kernel_size=4)
        fusion_feat = fusion_feat.reshape(-1, fusion_feat.shape[1] * fusion_feat.shape[2])

        hsi_feat = self.drop_hsi(hsi_feat)
        sar_feat = self.drop_sar(sar_feat)
        fusion_feat = self.drop_fusion(fusion_feat)

        output_hsi = self.fusionlinear_hsi(hsi_feat)
        output_sar = self.fusionlinear_sar(sar_feat)
        output_fusion = self.fusionlinear_fusion(fusion_feat)

        weights = torch.sigmoid(self.weight)
        outputs = weights[0] * output_hsi + weights[1] * output_sar + output_fusion
        return outputs
