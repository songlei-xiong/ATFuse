import math

import torch
from torch import einsum, nn
import numpy as np
from functools import partial
import torch.nn.functional as F
from torch.nn import Softmin


class Conv2d_BN(nn.Module):
    """Convolution with BN module."""

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity(
        )

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class DWConv2d_BN(nn.Module):
    """Depthwise Separable Convolution with BN module."""

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):
        """
        foward function
        """
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding."""

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
        )

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, embed_dim, num_path=4, isPool=False):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList([
            DWCPatchEmbed(
                in_chans=embed_dim,
                embed_dim=embed_dim,
                patch_size=3,
                stride=2 if isPool and idx == 0 else 1,
            ) for idx in range(num_path)
        ])

    def forward(self, inputs):
        """foward function"""
        att_inputs = []
        for x, pe in zip(inputs, self.patch_embeds):
            x = pe(x)
            att_inputs.append(x)

        return att_inputs


class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class."""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, q, k, v, minus=True):
        B, N, C = q.shape

        # Generate Q, K, V.
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Factorized attention.
        use_efficient = minus
        if use_efficient:

            k_softmax = k.softmax(dim=2)

            k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
            factor_att = einsum("b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v)
        else:
            # minus = Softmin(dim=2)
            # k_softmax = minus(k)
            # k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
            # factor_att = einsum("b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v)
            k_softmax = k.softmax(dim=2)

            k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
            factor_att = einsum("b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v)
        # else:
        #     q_dot_k = einsum("b h n k, b h n v -> b h k v", q, k)
        #     q_dot_k_softmax = q_dot_k.softmax(dim=2)
        #     factor_att = einsum("b h n v, b h n v -> b h n v", q_dot_k_softmax, v)

        # Convolutional relative position encoding.
        # if self.crpe:
        #     crpe = self.crpe(q, v, size=size)
        # else:
        #     crpe = 0

        # Merge and reshape.
        if use_efficient:
            x = factor_att  # + crpe    ViIr2用的0.5
        else:
            x = v - factor_att
        x = x.transpose(1, 2).reshape(B, N, C)

        # Output projection.

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    """Feed-forward network (FFN, a.k.a.

    MLP) class.
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """foward function"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MHCABlock(nn.Module):
    """Multi-Head Convolutional self-Attention block."""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=3,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            shared_cpe=None,
            shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.fuse = nn.Linear(dim * 2, dim)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)

        self.norm2 = norm_layer(dim)

    def forward(self, q, k, v, minus=True):
        """foward function"""
        """foward function"""
        b, c, h, w = q.size(0), q.size(1), q.size(2), q.size(3)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        x = q + self.factoratt_crpe(q, k, v, minus)
        cur = self.norm2(x)
        x = x + self.mlp(cur)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return x


class UpScale(nn.Module):
    def __init__(self, is_feature_sum, embed_dim, ):
        super(UpScale, self).__init__()
        self.is_feature_sum = is_feature_sum
        if is_feature_sum:
            self.conv11_headSum = nn.Conv2d(embed_dim, embed_dim, kernel_size=3,
                                            stride=1, padding=1, bias=True)
        else:
            self.conv11_head = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3,
                                         stride=1, padding=1, bias=True)
        self.conv12 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3,
                                stride=1, padding=1, bias=True)
        self.ps12 = nn.PixelShuffle(2)
        self.conv11_tail = nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3,
                                     stride=1, padding=1, bias=True)

    def forward(self, x, x_res):
        x11 = x  # B, C, H, W
        if self.is_feature_sum:
            x = x + x_res  # B, C, H, W
            x = self.conv11_headSum(x)  # B, C, H, W
        else:
            x = torch.cat([x, x_res], dim=1)  # B, 2*C, H, W
            x = self.conv11_head(x)  # B, C, H, W
        x = x + x11  # B, C, H, W
        x22 = self.conv12(x)  # B, 2*C, H, W
        x = F.relu(self.ps12(x22))  # B, C, 2*H, 2*W
        x = self.conv11_tail(x)  # 这里考虑一下是否使用relu函数  B, C, 2*H, 2*W
        return x


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + self.conv1(x1)
        return out


class ATF(nn.Module):
    def __init__(self, opt):
        super(ATF, self).__init__()
        self.num_stage = opt["networks"]["num_stage"]
        self.factor = opt["scale"]
        self.use_aggregate = opt["networks"]["use_aggregate"]
        in_chans = opt["networks"]["in_channels"]
        out_chans = opt["networks"]["out_channels"]
        embed_dims = opt["networks"]["embed_dims"]
        num_paths = opt["networks"]["num_paths"]
        num_heads = opt["networks"]["num_heads"]
        mlp_ratio = opt["networks"]["mlp_ratio"]
        self.num_paths = num_paths

        self.stem1 = nn.ModuleList([
            Conv2d_BN(
                in_chans,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ) for _ in range(num_paths[0])
        ])  # B,C,H/2,W/2, 对所有的通道进行处理

        self.patch_embed_stages1 = Patch_Embed_stage(
            embed_dims[0],
            num_path=num_paths[0],
            isPool=False
        )  # B,C,H/2,W/2, 对所有的通道进行处理

        self.mhca_stage = MHCABlock(
            embed_dims[0],
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_scale=None,
            shared_cpe=None,
            shared_crpe=None,
        )
        # B,C,H/2,W/2, 对所有的通道进行处理

        self.ir1_attn1 = MHCABlock(embed_dims[0],
                                   num_heads=num_heads,
                                   mlp_ratio=mlp_ratio,
                                   qk_scale=None,
                                   shared_cpe=None,
                                   shared_crpe=None, )

        self.ir2_attn1 = MHCABlock(embed_dims[0],
                                   num_heads=num_heads,
                                   mlp_ratio=mlp_ratio,
                                   qk_scale=None,
                                   shared_cpe=None,
                                   shared_crpe=None, )

        self.vi1_attn1 = MHCABlock(embed_dims[0],
                                   num_heads=num_heads,
                                   mlp_ratio=mlp_ratio,
                                   qk_scale=None,
                                   shared_cpe=None,
                                   shared_crpe=None, )

        self.vi2_attn1 = MHCABlock(embed_dims[0],
                                   num_heads=num_heads,
                                   mlp_ratio=mlp_ratio,
                                   qk_scale=None,
                                   shared_cpe=None,
                                   shared_crpe=None, )

        self.mhca_stage1_2 = MHCABlock(
            embed_dims[0],
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_scale=None,
            shared_cpe=None,
            shared_crpe=None,
        )  # B,C,H/2,W/2, 对所有的通道进行处理

        self.ir1_attn1_2 = MHCABlock(embed_dims[0],
                                     num_heads=num_heads,
                                     mlp_ratio=mlp_ratio,
                                     qk_scale=None,
                                     shared_cpe=None,
                                     shared_crpe=None, )

        self.ir2_attn1_2 = MHCABlock(embed_dims[0],
                                     num_heads=num_heads,
                                     mlp_ratio=mlp_ratio,
                                     qk_scale=None,
                                     shared_cpe=None,
                                     shared_crpe=None, )

        self.stem2 = nn.ModuleList([
            Conv2d_BN(
                embed_dims[0],
                embed_dims[1],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ) for _ in range(num_paths[0] + 1)
        ])  # B,C,H/4,W/4, 对所有的通道进行处理

        self.patch_embed_stages2 = Patch_Embed_stage(
            embed_dims[1],
            num_path=num_paths[0] + 1,
            isPool=False
        )  # B,C,H/4,W/4, 对所有的通道进行处理

        self.mhca_stage2 = MHCABlock(
            embed_dims[0],
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_scale=None,
            shared_cpe=None,
            shared_crpe=None,
        )  # B,C,H/4,W/4, 对所有的通道进行处理

        self.mhca_stage2_2 = MHCABlock(
            embed_dims[0],
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_scale=None,
            shared_cpe=None,
            shared_crpe=None,
        )  # B,C,H/4,W/4, 对所有的通道进行处理

        if self.use_aggregate:
            self.up_scale1_aggregate = UpScale(opt["networks"]["feature_sum"], embed_dim=embed_dims[1] * num_paths[1])
            self.resBlock = ResBlock(in_channels=embed_dims[0] * num_paths[1],
                                     out_channels=embed_dims[0] * num_paths[1])
            # self.resBlock2 = ResBlock(in_channels=embed_dims[0] * num_paths[1],
            #                           out_channels=embed_dims[0] * num_paths[1])  # add for ewc
            self.up_scale2_aggregate = UpScale(opt["networks"]["feature_sum"], embed_dim=embed_dims[0] * num_paths[1])
            # self.head = ResBlock(embed_dims[0] * 2, embed_dims[0])
            self.head = ResBlock(embed_dims[0] // 2, embed_dims[0] // 4)
            self.head_final = ResBlock(embed_dims[0] // 4, out_chans)
        else:
            # upscale 部分
            self.up_scale1 = nn.ModuleList([
                UpScale(opt["networks"]["feature_sum"], embed_dims[1]) for _ in range(num_paths[1])
            ])

            self.up_scale2 = nn.ModuleList([
                UpScale(opt["networks"]["feature_sum"], embed_dims[0]) for _ in range(num_paths[1])
            ])

            # head 部分
            self.head_aggregate = True
            if self.head_aggregate:
                self.head = ResBlock(embed_dims[0] * 2, embed_dims[0])
                self.head_final = ResBlock(embed_dims[0], 4)
            else:
                self.head = nn.ModuleList([
                    ResBlock(embed_dims[0] // 2, out_chans) for _ in range(num_paths[1])
                ])

    def inject_fusion(self, ms, pan):
        pass

    def forward(self, vi, ir):
        horizontal = 0
        perpendicular = 0
        outPre = vi
        with torch.no_grad():
            a = vi.shape[2] % 4
            b = vi.shape[3] % 4
            left = nn.ReplicationPad2d((1, 0, 0, 0))
            upper = nn.ReplicationPad2d((0, 0, 1, 0))

            while a % 4 != 0:
                vi = upper(vi)
                ir = upper(ir)
                horizontal += 1
                a += 1
            while b % 4 != 0:
                vi = left(vi)
                ir = left(ir)
                perpendicular += 1
                b += 1


            inputs = [ir,vi]  # pan(V), pan_up(K), ms(Q)

        att_outputs = []
        for x, model in zip(inputs, self.stem1):
            att_outputs.append(model(x))

        att_outputs = self.patch_embed_stages1(att_outputs)

        ir, vi = att_outputs[0], att_outputs[1]

        att_outputs2 = []  # 这里的结果后面需要使用

        att_outputs2.append(self.mhca_stage(att_outputs[1], ir, ir, minus=False))

        att_outputs2_1 = []  # 这里的结果后面需要使用

        att_outputs2_1.append(self.mhca_stage1_2(att_outputs2[0], vi, vi, minus=True))

        att_outputs3 = []  # 这里的结果后面需要使用

        att_outputs3.append(self.mhca_stage2(att_outputs2_1[0], ir, ir, minus=True))

        outv1 = ir
        outk1 = vi

        if self.use_aggregate:
            x11 = att_outputs3[0]  # (4,512,64,64)
            x11_skip = att_outputs2[0]  # (4,512,64,64)   (4,128,64,64)

            x22 = self.up_scale1_aggregate(x11, x11_skip)  # (4,256,128,128)    (4,64,128,128)

            x = self.head(x22)

            x = self.head_final(x)

            if horizontal != 0:
                x = x[:, :, horizontal:, :]
            if perpendicular != 0:
                x = x[:, :, :, perpendicular:]

            x = x.clamp(min=0, max=255)
            output = {"pred": x,
                      "k": outk1,
                      "v": outv1,
                      "outPre": outPre
                      }
        else:
            x = []

            att_outputs3 = []
            for i in range(self.num_paths[1]):
                att_outputs3.append(self.up_scale2[i](x[i], att_outputs2[i]))

            if self.head_aggregate:
                att_outputs3 = torch.cat(att_outputs3, dim=1)
                x = self.head(att_outputs3)
                x = self.head_final(x)
                output = {"pred": x, }
            else:
                x = []
                for i in range(self.num_paths[1]):
                    x.append(self.head[i](att_outputs3[i]))
                output = {"pred": torch.cat(x, dim=1)}

        return output
