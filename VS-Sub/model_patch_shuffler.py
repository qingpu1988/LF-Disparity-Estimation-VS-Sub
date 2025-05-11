import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import numpy as np
import math
import time
from einops import rearrange
from condconv import CondConv2D


class SubCostNet(nn.Module):
    def __init__(self, opt):
        super(SubCostNet, self).__init__()
        fn = opt.fn
        interval = opt.interval
        an = opt.angRes
        disp_max = opt.disp_max
        disp_min = opt.disp_min
        disp_num = int((disp_max - disp_min) / interval + 1)
        self.relu = nn.PReLU()
        self.bn = nn.BatchNorm2d(fn)
        self.an = an
        self.disp_num = disp_num
        self.disp_max = disp_max
        self.disp_min = disp_min
        self.fea_extraction = nn.Sequential(
            nn.Conv2d(1, fn, kernel_size=3, stride=1, dilation=an, padding=an, bias=False),
            nn.BatchNorm2d(fn),
            nn.PReLU(),
            nn.Conv2d(fn, fn, kernel_size=3, stride=1, dilation=an, padding=an, bias=False),
            nn.BatchNorm2d(fn),
            nn.PReLU(),
            Fea_extraction(an),
            Fea_extraction(an),
            Fea_extraction(an),
            Fea_extraction(an),
            Fea_extraction(an),
            Fea_extraction(an),
            Fea_extraction(an),
            Fea_extraction(an),
            nn.Conv2d(fn, fn, kernel_size=3, stride=1, dilation=an, padding=an, bias=False),
            nn.BatchNorm2d(fn),
            nn.PReLU(),
            nn.Conv2d(fn, fn, kernel_size=3, stride=1, dilation=an, padding=an, bias=False),
        )
        self.macpi_up = MacPI_Shuffler(an)
        cost_fn = 160
        self.costbuild = BuildCostVolume(fn, cost_fn, an)
        self.aggregation = nn.Sequential(
            nn.Conv3d(cost_fn, cost_fn, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False),
            nn.BatchNorm3d(cost_fn),
            nn.PReLU(),
            ResB3D(cost_fn),
            ResB3D(cost_fn),
            nn.Conv3d(cost_fn, cost_fn, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False),
            nn.BatchNorm3d(cost_fn),
            nn.PReLU(),
            nn.Conv3d(cost_fn, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False),
        )
        self.disp = faster_disparityregression()
        self.conf = Confidence_Estimation(disp_num)

    def forward(self, lf):
        lf = SAI2MacPI(lf, angRes=9)
        fea_lf = self.fea_extraction(lf)
        fea_lf_2 = self.macpi_up(fea_lf, 2)
        fea_lf_4 = self.macpi_up(fea_lf, 4)
        cost_full = self.costbuild(fea_lf, fea_lf_2, fea_lf_4, self.disp_min, self.disp_max, self.disp_num)
        cost = self.aggregation(cost_full)
        cost = cost.squeeze(1)
        disp_main = self.disp(cost, self.disp_max, self.disp_min, self.disp_num)
        conf = self.conf(cost)
        return disp_main, cost, conf


class Fea_extraction(nn.Module):
    def __init__(self, an):
        super(Fea_extraction, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, dilation=an, padding=an, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, dilation=an, padding=an, bias=False),
            nn.BatchNorm2d(16),
        )

    def forward(self, x):
        buffer = self.body(x)
        return buffer + x


class MacPI_Shuffler(nn.Module):
    def __init__(self, angRes):
        super(MacPI_Shuffler, self).__init__()
        self.unshuffler = nn.PixelUnshuffle(angRes)
        self.shuffler = nn.PixelShuffle(angRes)

    def forward(self, x, scale):
        fea_temp = self.unshuffler(x)
        fea_up = F.interpolate(fea_temp, scale_factor=scale, mode='bilinear', align_corners=True)
        fea_out = self.shuffler(fea_up)
        return fea_out


class BuildCostVolume(nn.Module):
    def __init__(self, channel_in, channel_out, angRes):
        super(BuildCostVolume, self).__init__()
        self.condconv = CondConv2D(channel_in, channel_out, angRes)
        self.angRes = angRes

    def forward(self, x_0, x_2, x_4, mindisp, maxdisp, dispnum):
        cost_list = []
        interval = (maxdisp - mindisp) / (dispnum - 1)
        if dispnum == 33:
            for i in range(dispnum):
                d = mindisp + i * interval
                if i % 4 == 0:
                    if d < 0:
                        dilat = int(abs(d) * self.angRes + 1)
                        pad = self.angRes // 2 * dilat - self.angRes // 2
                        stride = self.angRes
                    if d == 0:
                        dilat = 1
                        pad = self.angRes // 2 * dilat - self.angRes // 2
                        stride = self.angRes
                    if d > 0:
                        dilat = int(abs(d) * self.angRes - 1)
                        pad = self.angRes // 2 * dilat - self.angRes // 2
                        stride = self.angRes
                    # cost_lu = self.condconv(x_0, mask[0], stride, dilat, pad)
                    # cost_ru = self.condconv(x_0, mask[1], stride, dilat, pad)
                    # cost_lb = self.condconv(x_0, mask[2], stride, dilat, pad)
                    # cost_rb = self.condconv(x_0, mask[3], stride, dilat, pad)
                    cost = self.condconv(x_0, stride, dilat, pad)
                elif i % 4 == 2:
                    d = d * 2
                    if d < 0:
                        dilat = int(abs(d) * self.angRes + 1)
                        pad = self.angRes // 2 * dilat - self.angRes // 2
                        stride = self.angRes * 2
                    if d > 0:
                        dilat = int(abs(d) * self.angRes - 1)
                        pad = self.angRes // 2 * dilat - self.angRes // 2
                        stride = self.angRes * 2
                    # cost_lu = self.condconv(x_2, mask[0], stride, dilat, pad)
                    # cost_ru = self.condconv(x_2, mask[1], stride, dilat, pad)
                    # cost_lb = self.condconv(x_2, mask[2], stride, dilat, pad)
                    # cost_rb = self.condconv(x_2, mask[3], stride, dilat, pad)
                    cost = self.condconv(x_2, stride, dilat, pad)
                else:
                    d = d * 4
                    if d < 0:
                        dilat = int(abs(d) * self.angRes + 1)
                        pad = self.angRes // 2 * dilat - self.angRes // 2
                        stride = self.angRes * 4
                    if d > 0:
                        dilat = int(abs(d) * self.angRes - 1)
                        pad = self.angRes // 2 * dilat - self.angRes // 2
                        stride = self.angRes * 4
                    # cost_lu = self.condconv(x_4, mask[0], stride, dilat, pad)
                    # cost_ru = self.condconv(x_4, mask[1], stride, dilat, pad)
                    # cost_lb = self.condconv(x_4, mask[2], stride, dilat, pad)
                    # cost_rb = self.condconv(x_4, mask[3], stride, dilat, pad)
                    cost = self.condconv(x_4, stride, dilat, pad)
                # cost_concat = torch.cat([cost_lu, cost_ru, cost_lb, cost_rb], dim=1)
                # cost = self.qal(cost_concat)
                cost_list.append(cost)
        cost_volume = torch.stack(cost_list, dim=2)
        return cost_volume


class ResNet2d(nn.Module):
    def __init__(self, fn):
        super(ResNet2d, self).__init__()
        body = [nn.Conv2d(fn, fn, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False), nn.BatchNorm2d(fn),
                nn.PReLU()]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        fea = x
        fea = self.body(fea)
        fea = fea + x
        return fea


def make_layer(block, fn, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(fn))
    return nn.Sequential(*layers)


class faster_disparityregression(nn.Module):
    def __init__(self):
        super(faster_disparityregression, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, maxdisp, mindisp, disp_sample_number):
        interval = (maxdisp - mindisp) / (disp_sample_number - 1)
        score = self.softmax(x)
        d_values = torch.arange(disp_sample_number, device=score.device).float()
        disp_values = mindisp + interval * d_values
        temp = score * disp_values.unsqueeze(1).unsqueeze(1)

        temp1 = torch.zeros(score.shape).to(score.device)
        for d in range(disp_sample_number):
            temp1[:, d, :, :] = score[:, d, :, :] * (mindisp + interval * d)
        disp = torch.sum(temp, dim=1, keepdim=True)
        return disp


class ResB3D(nn.Module):
    def __init__(self, channels):
        super(ResB3D, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )

    def forward(self, x):
        buffer = self.body(x)
        return buffer + x


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


class Confidence_Estimation(nn.Module):
    def __init__(self, channel_in):
        super(Confidence_Estimation, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.theta = nn.Linear(channel_in, channel_in, bias=False)
        self.phi = nn.Linear(channel_in, channel_in, bias=False)
        self.psi = nn.Linear(channel_in, channel_in, bias=False)
        self.channel_fuse = nn.Conv2d(channel_in, 1, kernel_size=1, stride=1, padding=0, dilation=1)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(0.005 * torch.ones(1))
        nn.init.kaiming_normal_(self.theta.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.phi.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.psi.weight, mode='fan_out', nonlinearity='relu')
        self.layer_norm = nn.LayerNorm(channel_in)

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()

        # 将空间维度展平
        x_flat = x.view(batch_size, num_channels, -1).transpose(1, 2)  # (B, H*W, C)
        x_flat = self.layer_norm(x_flat)  # (B, H*W, C)
        # print("Input feature range: ", x_flat.min().item(), x_flat.max().item())
        # 应用全连接层
        theta = self.theta(x_flat)  # (B, H*W, C)
        phi = self.phi(x_flat)  # (B, H*W, C)
        psi = self.psi(x_flat)  # (B, H*W, C)

        # 计算注意力分数并应用softmax
        attn = torch.bmm(theta, phi.transpose(1, 2))  # (B, H*W, H*W)
        # attn_max = attn.max(dim=-1, keepdim=True)[0]
        # attn = attn - attn_max
        attn = F.softmax(attn / num_channels ** 0.5, dim=-1)  # (B, H*W, H*W)

        # 将注意力应用到psi上
        out = torch.bmm(attn, psi).transpose(1, 2).view(batch_size, num_channels, height, width)  # (B, C, H, W)

        # 与原始特征结合
        out = self.gamma * out + x

        # 生成置信度图
        confidence_map = self.channel_fuse(out)  # (B, 1, H, W)
        confidence_map = self.sigmoid(confidence_map)
        confidence_map = self.alpha * (1 - confidence_map) + self.beta


        return confidence_map
