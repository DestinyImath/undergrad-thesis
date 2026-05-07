import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, List, Union
from einops import rearrange

LOCAL_WEIGHT_DIR = "./pretrained_weights"
os.makedirs(LOCAL_WEIGHT_DIR, exist_ok=True)
torch.hub.set_dir(LOCAL_WEIGHT_DIR)

# ================= 1. 基础参数化组件 =================

def get_conv_layer(spatial_dims: int, in_c: int, out_c: int, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
    """根据空间维度自动选择 2D 或 3D 卷积"""
    if spatial_dims == 2:
        return nn.Conv2d(in_c, out_c, kernel_size, stride, padding, dilation, bias=bias)
    return nn.Conv3d(in_c, out_c, kernel_size, stride, padding, dilation, bias=bias)

def get_norm_layer(spatial_dims: int, channels: int):
    """自动选择归一化层"""
    if spatial_dims == 2:
        return nn.BatchNorm2d(channels)
    return nn.BatchNorm3d(channels)

def get_pool_layer(spatial_dims: int, stride: int):
    """根据 stride 自动选择池化层"""
    kernel_size = stride
    if spatial_dims == 2:
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    return nn.MaxPool3d(kernel_size=kernel_size, stride=stride)

class FlexibleDoubleConv(nn.Module):
    """参数化的双卷积块"""
    def __init__(self, spatial_dims, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            get_conv_layer(spatial_dims, in_c, out_c),
            get_norm_layer(spatial_dims, out_c),
            nn.ReLU(inplace=True),
            get_conv_layer(spatial_dims, out_c, out_c),
            get_norm_layer(spatial_dims, out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# ================= 2. 特殊模块设计 =================

class ASPP(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super().__init__()
        self.stages = nn.ModuleList()
        # 1x1 卷积分支
        self.stages.append(nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1, padding=0),
            get_norm_layer(spatial_dims, out_channels),
            nn.ReLU(inplace=True)
        ))
        # 多尺度空洞卷积分支
        for rate in rates[1:]:
            self.stages.append(nn.Sequential(
                get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate),
                get_norm_layer(spatial_dims, out_channels),
                nn.ReLU(inplace=True)
            ))
        # 全局池化分支
        pool_layer = nn.AdaptiveAvgPool2d((1, 1)) if spatial_dims == 2 else nn.AdaptiveAvgPool3d((1, 1, 1))
        self.global_pool = nn.Sequential(
            pool_layer,
            get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1, padding=0),
            get_norm_layer(spatial_dims, out_channels),
            nn.ReLU(inplace=True)
        )
        self.bottleneck = nn.Sequential(
            get_conv_layer(spatial_dims, out_channels * (len(rates) + 1), out_channels, kernel_size=1, padding=0),
            get_norm_layer(spatial_dims, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = [stage(x) for stage in self.stages]
        pool = self.global_pool(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear' if x.dim()==4 else 'trilinear', align_corners=True)
        res.append(pool)
        return self.bottleneck(torch.cat(res, dim=1))

class SwinBlock(nn.Module):
    def __init__(self, dim, input_res, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_res = input_res
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(0.),
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.)
        )

        if self.shift_size > 0:
            H, W = self.input_res
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = rearrange(img_mask, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c', p1=window_size, p2=window_size)
            mask_windows = mask_windows.squeeze(-1)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_res
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # 循环移位
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # 窗口划分
        x_windows = rearrange(x, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        x = rearrange(attn_windows, '(b h w) (p1 p2) c -> b (h p1) (w p2) c', h=H//self.window_size, w=W//self.window_size, p1=self.window_size, p2=self.window_size)

        # 反向移位
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        x = x.view(B, L, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
    
class BasicLayer(nn.Module):
    """对应架构图中的 Swin Transformer Block x2"""
    def __init__(self, dim, input_res, num_heads, window_size=8):
        super().__init__()
        # 官方架构：第一个 Block 是 W-MSA (shift=0)，第二个是 SW-MSA (shift=window_size//2)
        self.block1 = SwinBlock(dim, input_res, num_heads, window_size, shift_size=0)
        self.block2 = SwinBlock(dim, input_res, num_heads, window_size, shift_size=window_size//2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_res, dim):
        super().__init__()
        self.input_res = input_res
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_res
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0, x1, x2, x3 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1) 
        x = self.norm(x.view(B, -1, 4 * C))
        return self.reduction(x)

class PatchExpand(nn.Module):
    def __init__(self, input_res, dim):
        super().__init__()
        self.input_res = input_res
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        H, W = self.input_res
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        return self.norm(x.view(B, -1, C//4))

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_res, dim):
        super().__init__()
        self.input_res = input_res
        self.expand = nn.Linear(dim, 16 * dim, bias=False)

    def forward(self, x):
        H, W = self.input_res
        x = self.expand(x)
        B, L, C = x.shape
        x = rearrange(x.view(B, H, W, C), 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=4, p2=4, c=C//16)
        return x

# ================= 3. 神经网络部分 =================
class FlexibleUNet(nn.Module):
    """
    全功能灵活 U-Net 架构 (消融实验终极版)
    - 支持任意深度与通道配置。
    - 支持 ASPP 瓶颈层开关 (use_aspp)。
    - 支持 Attention Gate 注意力机制开关 (use_attention)。
    """
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 3,
        out_channels: int = 1,
        channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        feature_size: int = None, 
        use_aspp: bool = False,
        use_attention: bool = False  # 🌟 [新增] 注意力机制开关，默认为 False
    ):
        super().__init__()
        
        # 参数处理
        if feature_size is not None:
            channels = tuple([feature_size * (2**i) for i in range(len(channels))])
            
        self.spatial_dims = spatial_dims
        self.use_aspp = use_aspp
        self.use_attention = use_attention  # 🌟 [新增] 保存状态
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        if self.use_attention:
            self.attentions = nn.ModuleList() # 🌟 [新增] 仅在开启时初始化注意力列表

        # --- 构建 Encoder (下采样路径) ---
        self.encoder.append(FlexibleDoubleConv(spatial_dims, in_channels, channels[0]))
        
        for i in range(len(strides)):
            self.pools.append(get_pool_layer(spatial_dims, strides[i]))
            self.encoder.append(FlexibleDoubleConv(spatial_dims, channels[i], channels[i+1]))

        # --- 瓶颈层可选 ASPP ---
        if self.use_aspp:
            self.aspp = ASPP(spatial_dims, channels[-1], channels[-1])

        # --- 构建 Decoder (上采样路径) ---
        for i in range(len(strides), 0, -1):
            # 1. 上采样 (插值)
            self.ups.append(nn.Upsample(scale_factor=strides[i-1], mode='bilinear' if spatial_dims==2 else 'trilinear', align_corners=True))
            
            # 2. 🌟 [新增] 如果开启 Attention，则实例化 AttentionGate
            if self.use_attention:
                self.attentions.append(
                    AttentionGate(
                        g_channels=channels[i], 
                        s_channels=channels[i-1], 
                        F_int=channels[i-1] // 2
                    )
                )
                
            # 3. 融合后的卷积 (输入通道 = 当前层 + 跳跃连接通道)
            self.decoder.append(FlexibleDoubleConv(spatial_dims, channels[i] + channels[i-1], channels[i-1]))

        # 最后的输出预测头 (严格指定 padding=0)
        self.final_conv = get_conv_layer(spatial_dims, channels[0], out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        enc_features = []
        
        # --- Encoder forward ---
        x = self.encoder[0](x)
        enc_features.append(x)
        for i in range(len(self.pools)):
            x = self.pools[i](x)
            x = self.encoder[i+1](x)
            if i < len(self.pools) - 1:
                enc_features.append(x)
        
        # --- Bottleneck ---
        if self.use_aspp:
            x = self.aspp(x)
            
        # --- Decoder forward ---
        for i in range(len(self.ups)):
            # 1. 执行上采样，得到深层指引信号 (g)
            x = self.ups[i](x)
            # 2. 提取对应的跳跃连接特征 (s)
            skip = enc_features[-(i+1)]
            
            # 3. 🌟 [核心修改] 根据开关决定是否执行 Attention 过滤
            if self.use_attention:
                # 开启注意力：用上采样的 x 去过滤 skip
                filtered_skip = self.attentions[i](g=x, s=skip)
                
                # 尺寸对齐 (防御奇数维度问题)
                if x.shape[2:] != filtered_skip.shape[2:]:
                    x = F.interpolate(x, size=filtered_skip.shape[2:], mode='bilinear' if self.spatial_dims==2 else 'trilinear', align_corners=True)
                
                # 拼接过滤后的特征
                x = torch.cat([x, filtered_skip], dim=1)
                
            else:
                # 关闭注意力：原汁原味的原始逻辑
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear' if self.spatial_dims==2 else 'trilinear', align_corners=True)
                
                # 直接拼接原始特征
                x = torch.cat([x, skip], dim=1)
            
            # 4. 执行解码器双卷积
            x = self.decoder[i](x)
            
        return self.final_conv(x)
    
class DeepLabV3Plus(nn.Module):
    """
    完全遵照 DeepLab V3+ 官方网络拓扑图实现的架构。
    特色：非对称结构、单次浅层跳跃连接 (Low-level features)、双重 4 倍上采样 (Upsample by 4)。
    """
    def __init__(self, spatial_dims=2, in_channels=3, out_channels=1, backbone_channels=(64, 128, 256, 512), aspp_rates=[1, 6, 12, 18]):
        super().__init__()
        self.spatial_dims = spatial_dims

        # --- 1. Encoder: DCNN Backbone ---
        # (使用基础卷积块模拟特征提取与分辨率下降)
        # Stage 1 -> 1/2 分辨率
        self.enc_stage1 = nn.Sequential(
            FlexibleDoubleConv(spatial_dims, in_channels, backbone_channels[0]),
            get_pool_layer(spatial_dims, stride=2)
        )
        # Stage 2 -> 1/4 分辨率 (🌟 提取给 Decoder 的 Low-level features)
        self.enc_stage2 = nn.Sequential(
            FlexibleDoubleConv(spatial_dims, backbone_channels[0], backbone_channels[1]),
            get_pool_layer(spatial_dims, stride=2)
        )
        # Stage 3 -> 1/8 分辨率
        self.enc_stage3 = nn.Sequential(
            FlexibleDoubleConv(spatial_dims, backbone_channels[1], backbone_channels[2]),
            get_pool_layer(spatial_dims, stride=2)
        )
        # Stage 4 -> 1/16 分辨率 (🌟 提供给 ASPP 的 Deep features)
        self.enc_stage4 = nn.Sequential(
            FlexibleDoubleConv(spatial_dims, backbone_channels[2], backbone_channels[3]),
            get_pool_layer(spatial_dims, stride=2)
        )

        # --- 2. Encoder: ASPP 模块 ---
        # 接收 1/16 分辨率的特征图，通过空洞卷积提取多尺度全局信息
        self.aspp = ASPP(spatial_dims, in_channels=backbone_channels[3], out_channels=256, rates=aspp_rates)

        # --- 3. Decoder ---
        # 图示：处理 Low-level feature 的 1x1 卷积 (降维至 48，防止浅层噪声淹没深层语义)
        self.low_level_conv = nn.Sequential(
            get_conv_layer(spatial_dims, backbone_channels[1], 48, kernel_size=1, padding=0), # 严格遵循 padding=0
            get_norm_layer(spatial_dims, 48),
            nn.ReLU(inplace=True)
        )

        # 图示：融合后的 3x3 卷积 (ASPP 输出的 256 + 降维后的 48 = 304 通道)
        self.decoder_conv = FlexibleDoubleConv(spatial_dims, 256 + 48, 256)

        # 图示：最终预测的 1x1 卷积
        self.final_conv = get_conv_layer(spatial_dims, 256, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        input_shape = x.shape[2:] # 记录初始尺寸，例如 (256, 256)

        # --- 数据流向：Encoder 阶段 ---
        feat1_2 = self.enc_stage1(x)
        low_level_feat = self.enc_stage2(feat1_2) # 到达 1/4 分辨率 (例如 64x64)
        feat1_8 = self.enc_stage3(low_level_feat)
        deep_feat = self.enc_stage4(feat1_8)      # 到达 1/16 分辨率 (例如 16x16)

        # 进入 ASPP 提取全局特征
        aspp_out = self.aspp(deep_feat)           # 尺寸保持 1/16 分辨率

        # --- 数据流向：Decoder 阶段 ---
        # 1. 图示：Upsample by 4 (将 ASPP 输出放大 4 倍，使其与 low_level_feat 齐平)
        aspp_up = F.interpolate(aspp_out, scale_factor=4, mode='bilinear' if self.spatial_dims==2 else 'trilinear', align_corners=False)

        # 2. 图示：Low-level features 进入 1x1 Conv
        low_level_processed = self.low_level_conv(low_level_feat)

        # (安全对齐机制：应对除不尽的奇数分辨率)
        if aspp_up.shape[2:] != low_level_processed.shape[2:]:
            aspp_up = F.interpolate(aspp_up, size=low_level_processed.shape[2:], mode='bilinear' if self.spatial_dims==2 else 'trilinear', align_corners=False)

        # 3. 图示：Concat
        concat_feat = torch.cat([aspp_up, low_level_processed], dim=1)

        # 4. 图示：3x3 Conv 细化特征
        dec_out = self.decoder_conv(concat_feat)

        # 5. 图示：Upsample by 4 (最后再一次放大 4 倍，强制恢复到原图大小)
        out = F.interpolate(dec_out, size=input_shape, mode='bilinear' if self.spatial_dims==2 else 'trilinear', align_corners=False)

        # 6. 图示：Prediction (1x1 Conv 生成掩膜)
        return self.final_conv(out)

class UniversalDeepLabV3Plus(nn.Module):
    def __init__(
        self, 
        backbone, 
        low_level_channels: int, 
        high_level_channels: int, 
        out_channels: int = 1,
        spatial_dims: int = 2,
        aspp_out_channels: int = 256,
        decoder_channels: int = 256,
        use_aspp: bool = True,      # 🌟 消融开关 1
        use_decoder: bool = True    # 🌟 消融开关 2
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.backbone = backbone
        self.use_aspp = use_aspp
        self.use_decoder = use_decoder

        # --- 1. ASPP 模块消融控制 ---
        if self.use_aspp:
            self.aspp = ASPP(spatial_dims, high_level_channels, aspp_out_channels)
        else:
            # 如果关闭 ASPP，用一个简单的 1x1 卷积代替，仅作通道降维，无多尺度感受野
            self.aspp = get_conv_layer(spatial_dims, high_level_channels, aspp_out_channels, kernel_size=1, padding=0)

        # --- 2. Decoder 模块消融控制 ---
        if self.use_decoder:
            self.low_level_conv = nn.Sequential(
                get_conv_layer(spatial_dims, low_level_channels, 48, kernel_size=1, padding=0),
                get_norm_layer(spatial_dims, 48),
                nn.ReLU(inplace=True)
            )
            self.decoder_conv = nn.Sequential(
                FlexibleDoubleConv(spatial_dims, aspp_out_channels + 48, decoder_channels),
                get_conv_layer(spatial_dims, decoder_channels, decoder_channels, kernel_size=3, padding=1)
            )
            self.final_conv = get_conv_layer(spatial_dims, decoder_channels, out_channels, kernel_size=1, padding=0)
        else:
            # 如果没有 Decoder，ASPP 的输出直接通过 1x1 卷积输出预测结果
            self.final_conv = get_conv_layer(spatial_dims, aspp_out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        input_shape = x.shape[2:] 
        low_level_feat, deep_feat = self.backbone(x)

        # 无论是否是真正的 ASPP，都经过 self.aspp 处理统一维度
        aspp_out = self.aspp(deep_feat) 

        # --- Decoder 消融分流 ---
        if self.use_decoder:
            aspp_up = F.interpolate(aspp_out, scale_factor=4, mode='bilinear', align_corners=False)
            low_level_processed = self.low_level_conv(low_level_feat)
            if aspp_up.shape[2:] != low_level_processed.shape[2:]:
                aspp_up = F.interpolate(aspp_up, size=low_level_processed.shape[2:], mode='bilinear', align_corners=False)
            
            x_combined = torch.cat([aspp_up, low_level_processed], dim=1)
            x_refined = self.decoder_conv(x_combined)
            out = F.interpolate(x_refined, size=input_shape, mode='bilinear', align_corners=False)
            return self.final_conv(out)
        else:
            # 【消融实验分支】：直接将 1/16 的特征图暴力放大 16 倍回原图
            out = self.final_conv(aspp_out)
            out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
            return out

class StandardSwinUnet(nn.Module):
    def __init__(self, img_size=256, in_channels=3, out_channels=1, feature_size=96):
        super().__init__()
        self.img_size = img_size
        self.feature_size = feature_size
        
        # [1] Patch Partition & Linear Embedding
        self.patch_embed = nn.Conv2d(in_channels, feature_size, kernel_size=4, stride=4)
        
        # --- Encoder ---
        # Stage 1: [H/4, W/4, C]  (64x64)
        self.enc1 = BasicLayer(feature_size, (img_size//4, img_size//4), num_heads=3)
        self.down1 = PatchMerging((img_size//4, img_size//4), feature_size)
        
        # Stage 2: [H/8, W/8, 2C] (32x32)
        self.enc2 = BasicLayer(2*feature_size, (img_size//8, img_size//8), num_heads=6)
        self.down2 = PatchMerging((img_size//8, img_size//8), 2*feature_size)
        
        # Stage 3: [H/16, W/16, 4C] (16x16)
        self.enc3 = BasicLayer(4*feature_size, (img_size//16, img_size//16), num_heads=12)
        self.down3 = PatchMerging((img_size//16, img_size//16), 4*feature_size)
        
        # --- Bottleneck ---
        # Stage 4: [H/32, W/32, 8C] (8x8)
        self.bottleneck = BasicLayer(8*feature_size, (img_size//32, img_size//32), num_heads=24)
        
        # --- Decoder ---
        # Up 1: to 16x16
        self.up1 = PatchExpand((img_size//32, img_size//32), 8*feature_size)
        self.concat_linear1 = nn.Linear(8*feature_size, 4*feature_size) 
        self.dec1 = BasicLayer(4*feature_size, (img_size//16, img_size//16), num_heads=12)
        
        # Up 2: to 32x32
        self.up2 = PatchExpand((img_size//16, img_size//16), 4*feature_size)
        self.concat_linear2 = nn.Linear(4*feature_size, 2*feature_size)
        self.dec2 = BasicLayer(2*feature_size, (img_size//8, img_size//8), num_heads=6)
        
        # Up 3: to 64x64
        self.up3 = PatchExpand((img_size//8, img_size//8), 2*feature_size)
        self.concat_linear3 = nn.Linear(2*feature_size, feature_size)
        self.dec3 = BasicLayer(feature_size, (img_size//4, img_size//4), num_heads=3)
        
        # --- Final Output ---
        self.final_up = FinalPatchExpand_X4((img_size//4, img_size//4), feature_size)
        self.final_conv = nn.Conv2d(feature_size, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. 嵌入层
        x = self.patch_embed(x) 
        x = rearrange(x, 'b c h w -> b (h w) c') 
        
        # 2. Encoder (带有 Skip Connections)
        enc1 = self.enc1(x)            # 🌟 跳跃连接 1 (64x64)
        x_d1 = self.down1(enc1)        
        
        enc2 = self.enc2(x_d1)         # 🌟 跳跃连接 2 (32x32)
        x_d2 = self.down2(enc2)        
        
        enc3 = self.enc3(x_d2)         # 🌟 跳跃连接 3 (16x16)
        x_d3 = self.down3(enc3)        
        
        # 3. Bottleneck
        x_b = self.bottleneck(x_d3)    # (8x8)
        
        # 4. Decoder (跳跃连接 -> 拼接 -> 降维 -> 细化)
        x_u1 = self.up1(x_b)           
        x_u1 = torch.cat([x_u1, enc3], dim=-1)  
        x_u1 = self.concat_linear1(x_u1)
        x_u1 = self.dec1(x_u1)
        
        x_u2 = self.up2(x_u1)
        x_u2 = torch.cat([x_u2, enc2], dim=-1)
        x_u2 = self.concat_linear2(x_u2)
        x_u2 = self.dec2(x_u2)
        
        x_u3 = self.up3(x_u2)
        x_u3 = torch.cat([x_u3, enc1], dim=-1)
        x_u3 = self.concat_linear3(x_u3)
        x_u3 = self.dec3(x_u3)
        
        # 5. Final Output
        out = self.final_up(x_u3)      
        out = rearrange(out, 'b h w c -> b c h w')
        return self.final_conv(out)

# ================= 4. 损失函数 (Loss Functions) =================
class CustomDiceLoss(nn.Module):
    """
    自研的 Dice Loss (支持自动 Sigmoid 激活)
    """
    def __init__(self, sigmoid=True, smooth=1e-5):
        super(CustomDiceLoss, self).__init__()
        self.sigmoid = sigmoid
        self.smooth = smooth # 平滑系数，防止分母为0，同时加速训练初期的收敛

    def forward(self, y_pred, y_true):
        # 如果模型输出的是未经激活的 logits，先做 sigmoid 映射到 0~1 的概率区间
        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)
        
        # 将预测图和真实标签展平为一维向量 (Batch_size * H * W)
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # 计算交集
        intersection = (y_pred * y_true).sum()
        
        # 计算 Dice 系数: 2 * |X ∩ Y| / (|X| + |Y|)
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        
        # 损失函数是要最小化的，所以用 1 减去 Dice 系数
        return 1.0 - dice

class CustomDiceCELoss(nn.Module):
    """
    完全替代 MONAI 的 DiceCELoss
    混合损失：Dice Loss + Binary Cross Entropy (BCE) Loss
    """
    def __init__(self, sigmoid=True, weight_dice=1.0, weight_ce=1.0):
        super(CustomDiceCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        
        # 初始化 Dice Loss 分支
        self.dice_loss = CustomDiceLoss(sigmoid=sigmoid)
        
        # 初始化 Cross Entropy 分支
        # 注意：PyTorch 官方强烈建议使用 BCEWithLogitsLoss 而不是先 Sigmoid 再 BCELoss
        # 因为 BCEWithLogitsLoss 内部应用了 Log-Sum-Exp 技巧，数值计算极其稳定，不会出现梯度爆炸或消失
        if sigmoid:
            self.ce_loss = nn.BCEWithLogitsLoss()
        else:
            self.ce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true):
        # 分别计算两种 Loss
        dice = self.dice_loss(y_pred, y_true)
        ce = self.ce_loss(y_pred, y_true)
        
        # 按权重相加返回
        return self.weight_dice * dice + self.weight_ce * ce

# =================== 5. 用于耦合的 DCNN ====================

class ResNetBackbone(nn.Module):
    """
    包装 torchvision 的 ResNet 作为 DeepLab 的 Encoder
    """
    def __init__(self, model_name="resnet50"):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        res = getattr(models, model_name)(weights=weights)
        
        # 提取各个阶段
        self.initial = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool) # 1/4 分辨率
        self.layer1 = res.layer1 # 1/4 分辨率
        self.layer2 = res.layer2 # 1/8 分辨率
        self.layer3 = res.layer3 # 1/16 分辨率

    def forward(self, x):
        x = self.initial(x)
        low_level = self.layer1(x) # 🌟 1/4 分辨率特征
        x = self.layer2(low_level)
        deep_feat = self.layer3(x) # 🌟 1/16 分辨率特征
        return low_level, deep_feat

## baseline
class PureResNet50_FCN(nn.Module):
    """
    纯粹的 ResNet-50 分割基线模型 (经典 FCN-32s 架构)。
    用于评估 DCNN 最原始的特征提取能力（性能下限），
    没有任何额外的多尺度融合(ASPP)或解码器(Decoder)。
    """
    def __init__(self, out_channels=1, spatial_dims=2):
        super().__init__()
        self.spatial_dims = spatial_dims
        weights = models.ResNet50_Weights.DEFAULT
        res = models.resnet50(weights=weights)
        
        # 2. 提取完整的基础特征提取层 (包含全部 4 个 Layer)
        # 图像经过 layer4 后，尺寸会缩小 32 倍 (例如 256 -> 8x8)，通道数变为 2048
        self.backbone = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool,
            res.layer1, # 1/4 分辨率
            res.layer2, # 1/8 分辨率
            res.layer3, # 1/16 分辨率
            res.layer4  # 1/32 分辨率 (这是原始 DCNN 最深度的特征)
        )
        
        # 3. 极简分割头 (Segmentation Head)
        # 仅仅使用一个 1x1 卷积，把 2048 个特征通道极速压缩成 1 个通道 (Mask)
        self.head = get_conv_layer(spatial_dims, 2048, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        input_shape = x.shape[2:] # 记录输入尺寸 (例如 256x256)
        
        # 步骤 1: 提取深层特征 (尺寸变为原图的 1/32，即 8x8)
        feat = self.backbone(x)
        
        # 步骤 2: 映射到类别预测 (尺寸还是 8x8, 但通道变为 out_channels)
        out = self.head(feat)
        
        # 步骤 3: 暴力上采样 (直接将 8x8 强行双线性插值拉伸 32 倍，恢复到 256x256)
        out = F.interpolate(out, size=input_shape, mode='bilinear' if self.spatial_dims==2 else 'trilinear', align_corners=False)
        
        return out

class AttentionGate(nn.Module):
    """工程级注意力门 (带奇数尺寸防护)"""
    def __init__(self, g_channels, s_channels, F_int):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, F_int, kernel_size=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, F_int, kernel_size=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        g1 = self.Wg(g)       
        s1 = self.Ws(s)       
        
        if g1.shape[2:] != s1.shape[2:]:
            g1 = F.interpolate(g1, size=s1.shape[2:], mode='bilinear', align_corners=False)
            
        out = F.relu(g1 + s1) 
        psi = self.psi(out)   
        return s * psi
    
class WindowAttention(nn.Module):
    """窗口多头自注意力机制 (包含相对位置编码)"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1) 
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        rel_pos_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous() 
        attn = attn + rel_pos_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x