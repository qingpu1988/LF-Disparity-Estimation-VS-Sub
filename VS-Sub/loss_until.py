from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

def laplace_disp2prob(gt, prob, mindisp=-4, numdisp=33, interval=0.25):
    n, _, h, w = gt.shape
    b = prob
    d_values = torch.arange(numdisp, device=gt.device).float()
    disp = mindisp + interval * d_values
    disp = disp.reshape(1, numdisp, 1, 1).repeat(n, 1, h, w)
    cost = -torch.abs(disp - gt) / b
    return F.softmax(cost, dim=1)

class loss_prob(nn.Module):
    def __init__(self):
        super(loss_prob, self).__init__()
        self.kl = nn.KLDivLoss(reduction="none")

    def forward(self, Cost, Disp, Prob, dis):
        eps = 1e-8
        N, C, H, W = Cost.shape
        gtDisp = Disp.clone()
        gtProb = laplace_disp2prob(gtDisp, Prob)
        gtProb = torch.clamp(gtProb, min=eps)

        # stereo focal loss
        estProb = F.log_softmax(Cost, dim=1)
        estSoft = F.softmax(Cost, dim=1)
        estSoft = torch.clamp(estSoft, min=eps)
        m = 0.5 * (gtProb + estSoft)
        kl_div1 = self.kl(estProb, m)
        kl_div2 = self.kl(gtProb.log(), m)
        js_div = 0.5 * (kl_div1 + kl_div2)
        js_div = torch.clamp(js_div, min=eps)
        t0 = js_div.max().item()
        t1 = js_div.min().item()
        # loss = -((gtProb * estProb) ).sum(dim=1, keepdim=True).mean()
        # kl_div = F.kl_div(estProb, gtProb, reduction='none'
        temp = js_div ** 0.5
        t2 = temp.max().item()

        loss = ((0.8 + temp) * torch.abs(Disp - dis)).mean()
        return loss


def loss_reg(Prob):
    loss = (-1.0 * F.logsigmoid(Prob)).mean()
    return loss

class gradient_loss_cal(torch.nn.Module):
    def __init__(self, angRes):
        super(gradient_loss_cal, self).__init__()
        self.angRes = angRes
        if angRes == 3:
            k = torch.Tensor([[.274, .452, .274]])
        if angRes == 5:
            k = torch.Tensor([[.05, .25, .4, .25, .05]])
        if angRes == 7:
            k = torch.Tensor([[.0044, .054, .242, .3991, .2420, .054, .0044]])
        if angRes == 9:
            k = torch.Tensor([[.0001, .0044, .054, .242, .3989, .2420, .054, .0044, .0001]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).unsqueeze(1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = nn.L1Loss()
        # self.tv_cal = TV_Cal()
        # self.ssim_cal = pytorch_ssim.SSIM(window_size=angRes)

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


def gradient_cal(img, angRes):
    if angRes == 3:
        k = torch.Tensor([[.274, .452, .274]])
    if angRes == 5:
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
    if angRes == 7:
        k = torch.Tensor([[.0044, .054, .242, .3991, .2420, .054, .0044]])
    if angRes == 9:
        k = torch.Tensor([[.0001, .0044, .054, .242, .3989, .2420, .054, .0044, .0001]])
    kernel = torch.matmul(k.t(), k).unsqueeze(0).unsqueeze(1)
    if torch.cuda.is_available():
        kernel = kernel.cuda()
    filtered = conv_gauss_1(img, kernel)
    down = filtered[:, :, ::2, ::2]
    new_filter = torch.zeros_like(filtered)
    new_filter[:, :, ::2, ::2] = down * 4
    filtered = conv_gauss_1(new_filter, kernel)
    diff = img - filtered
    return diff

