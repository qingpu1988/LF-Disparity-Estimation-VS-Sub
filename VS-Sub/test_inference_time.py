import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=9, help="angular resolution")
    parser.add_argument('--model_name', type=str, default='DistgDisp')
    parser.add_argument('--testset_dir', type=str, default='dataset/test/')
    parser.add_argument('--crop', type=bool, default=True)
    parser.add_argument('--patchsize', type=int, default=64)
    parser.add_argument('--minibatch_test', type=int, default=32)
    parser.add_argument('--model_path', type=str, default='./log/DistgDisp16200.pth.tar')
    parser.add_argument('--save_path', type=str, default='./Results/')
    return parser.parse_args()


class Corse_To_Fine_net(nn.Module):
    def __init__(self, opt):
        super(Corse_To_Fine_net, self).__init__()
        fn = 16
        an = 9
        self.an = an
        disp_num = 33
        disp_max = 4
        disp_min = -4
        self.relu = nn.PReLU()
        self.bn = nn.BatchNorm2d(fn)
        self.disp_num = disp_num
        self.disp_max = disp_max
        self.disp_min = disp_min
        # self.edge_first = nn.Conv2d(1, disp_num, kernel_size=3, stride=1, padding=1, bias=False)
        self.first = nn.Conv2d(1, fn, kernel_size=3, stride=1, padding=1, bias=False)
        # self.edge_fea = make_layer(ResNet2d, disp_num, edge_block)
        self.fea_extraction = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, dilation=an, padding=an, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, dilation=an, padding=an, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            Fea_extraction(an),
            Fea_extraction(an),
            Fea_extraction(an),
            Fea_extraction(an),
            Fea_extraction(an),
            Fea_extraction(an),
            Fea_extraction(an),
            Fea_extraction(an),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, dilation=an, padding=an, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, dilation=an, padding=an, bias=False),
        )
        channel_3D = 160
        channel_fc = 40
        self.build_cost = Sub_Cost_Volum(fn, channel_3D, an, channel_fc)
        self.aggregation = nn.Sequential(
            nn.Conv3d(channel_3D, channel_3D, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False),
            nn.BatchNorm3d(channel_3D),
            nn.PReLU(),
            ResB3D(channel_3D),
            ResB3D(channel_3D),
            nn.Conv3d(channel_3D, channel_3D, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False),
            nn.BatchNorm3d(channel_3D),
            nn.PReLU(),
            nn.Conv3d(channel_3D, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False),
        )
        self.cost = Feature_Fuse(channel_3D * disp_num, disp_num)

        # body = []
        # for i in range(6):
        #     body.append(feature_fusion())
        self.disp = faster_disparityregression()

    def forward(self, lf):
        lf = SAI2MacPI(lf, angRes=9)
        lf = MacPI2SAI(lf, angRes=9)
        #
        start = time.time()
        fea_lf = self.fea_extraction(lf)
        fea_lf_2 = F.interpolate(fea_lf, scale_factor=2, mode='bilinear', align_corners=True)
        fea_lf_4 = F.interpolate(fea_lf, scale_factor=2, mode='bilinear', align_corners=True)
        fea_lf = SAI2MacPI(fea_lf, angRes=9)
        fea_lf_2 = SAI2MacPI(fea_lf_2, angRes=9)
        fea_lf_4 = SAI2MacPI(fea_lf_4, angRes=9)
        time_cost = time.time() - start

        cost_casual, cost_full = self.build_cost(fea_lf, self.disp_min, self.disp_max, self.disp_num)

        cost = self.cost(cost_full, cost_casual)
        cost = self.aggregation(cost)
        cost = cost.squeeze(1)
        disp_main = self.disp(cost, self.disp_max, self.disp_min, self.disp_num)

        return disp_main, time_cost


class Sub_Cost_Volum(nn.Module):
    def __init__(self, channel_in, channel_out, angRes, channel_fc):
        super(Sub_Cost_Volum, self).__init__()
        self.cost_lu = Casual_Subcost_layer(channel_in, channel_fc, angRes)
        self.cost_ru = Casual_Subcost_layer(channel_in, channel_fc, angRes)
        self.cost_lb = Casual_Subcost_layer(channel_in, channel_fc, angRes)
        self.cost_rb = Casual_Subcost_layer(channel_in, channel_fc, angRes)
        self.build_full_cost = Full_Subcost_layer(channel_in, channel_out, angRes)
        self.casual_cost = Quardrant_Attention_Layer(channel_fc, groups=4)

        self.angRes = angRes
        self.casual_fuse = nn.Conv2d(channel_fc * 4, channel_out, kernel_size=1, padding=0, bias=False)

    def forward(self, x, dispmin, dispmax, dispnum):
        interval = (dispmax - dispmin) / (dispnum - 1)
        cost_list_casual = []
        cost_list_full = []
        x_crop = x[:, :, self.angRes // 2:, self.angRes // 2:]
        for i in range(dispnum):
            cost_lu = self.cost_lu(x_crop, interval, i, dispmin, case=1)
            cost_ru = self.cost_ru(x_crop, interval, i, dispmin, case=2)
            cost_lb = self.cost_lb(x_crop, interval, i, dispmin, case=3)
            cost_rb = self.cost_rb(x_crop, interval, i, dispmin, case=4)
            cost_casual = torch.cat([cost_lu, cost_ru, cost_lb, cost_rb], dim=1)
            cost_casual = self.casual_cost(cost_casual)
            cost_casual = self.casual_fuse(cost_casual)
            cost_full = self.build_full_cost(x, interval, i, dispmin)

            cost_list_casual.append(cost_casual)
            cost_list_full.append(cost_full)
        cost_volumn_casual = torch.stack(cost_list_casual, dim=2)
        cost_volumn_full = torch.stack(cost_list_full, dim=2)
        return cost_volumn_casual, cost_volumn_full


class Quardrant_Attention_Layer(nn.Module):
    def __init__(self, channel_in, groups):
        super(Quardrant_Attention_Layer, self).__init__()
        num_heads = 4
        self.quarters = groups
        self.channels = channel_in
        self.num_heads = num_heads
        self.gamma = nn.Parameter(torch.zeros(1))
        self.theta = nn.Linear(channel_in * groups, channel_in * groups * num_heads, bias=False)
        self.phi = nn.Linear(channel_in * groups, channel_in * groups * num_heads, bias=False)
        self.psi = nn.Linear(channel_in * groups, channel_in * groups, bias=False)

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        g = num_channels // self.quarters
        feas = x
        theta = self.theta(feas.view(-1, num_channels)).view(batch_size, height * width, num_channels, self.num_heads)
        phi = self.phi(feas.view(-1, num_channels)).view(batch_size, height * width, num_channels, self.num_heads)
        psi = self.psi(phi.transpose(2, 3)).view(batch_size, height * width, num_channels, self.num_heads)

        # Calculate attention scores and softmax along the groups dimension
        attn = theta * phi
        attn = attn / (num_channels // self.quarters)
        attn = attn.view(batch_size, height * width, -1, self.num_heads)
        attn = torch.softmax(attn, dim=2)
        out = (attn * psi).sum(dim=3).view(batch_size, num_channels, height, width)
        out = self.gamma * out + feas

        return out


class Feature_Fuse(nn.Module):
    def __init__(self, channel_in, depth):
        super(Feature_Fuse, self).__init__()
        fn = channel_in
        self.query_conv = nn.Conv2d(fn, fn, kernel_size=1)
        self.key_conv = nn.Conv2d(fn, fn, kernel_size=1)
        self.value_conv = nn.Conv2d(fn, fn, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable parameter to control attention blending
        self.num_depth = depth

    def forward(self, x1, x2):
        batch_size, channels, depth, height, width = x1.size()
        query = self.query_conv(x1.view(batch_size, -1, height,width))
        key = self.key_conv(x2.view(batch_size, -1, height,width))
        value = self.value_conv(x2.view(batch_size, -1, height,width))
        query = query.view(batch_size, -1, self.num_depth, query.size(-2), query.size(-1))
        key = key.view(batch_size, -1, self.num_depth,key.size(-2), key.size(-1))
        value = value.view(batch_size, -1, self.num_depth, value.size(-2), value.size(-1))
        attention_map = torch.matmul(query.view(-1, query.size(-2), query.size(-1)),
                                     key.view(-1, key.size(-2), key.size(-1)).permute(0, 2, 1))
        attention_map = torch.softmax(attention_map, dim=-1)

        # Apply attention to the value
        out = torch.matmul(attention_map, value.view(-1, value.size(-2), value.size(-1)))
        out = out.view(batch_size, value.size(1), self.num_depth, value.size(3), value.size(4))

        # Apply gamma to blend original and attended features
        out = self.gamma * out + x1
        return out


class Full_Subcost_layer(nn.Module):
    def __init__(self, channel_in, channel_out, angRes):
        super(Full_Subcost_layer, self).__init__()
        self.DSAFE_0 = nn.Conv2d(channel_in, channel_out, kernel_size=angRes, stride=angRes, padding=0, bias=False)
        self.DSAFE_1 = nn.Conv2d(channel_in, channel_out, kernel_size=angRes // 2 + 1, stride=angRes, padding=0,
                                 bias=False)
        self.DSAFE_2 = nn.Conv2d(channel_in, channel_out, kernel_size=angRes // 4 + 1, stride=angRes, padding=0,
                                 bias=False)
        self.angRes = angRes

    def forward(self, x, interval, index, mindisp):
        d = mindisp + index * interval

        if d == 0:
            dilat = 1
            pad = 0
            conv_weight = self.DSAFE_0.weight
        elif index % 4 == 0:
            dilat = int(sign(d) * (d * self.angRes - 1))
            pad = int(self.angRes // 2 * dilat - self.angRes // 2)
            conv_weight = self.DSAFE_0.weight
        elif index % 4 == 2:
            dilat = int(sign(d) * 2 * (d * self.angRes - 1))
            pad = int(self.angRes // 4 * dilat - self.angRes // 2)
            conv_weight = self.DSAFE_1.weight
        else:
            dilat = int(sign(d) * 4 * (d * self.angRes - 1))
            pad = int(self.angRes // 8 * dilat - self.angRes // 2)
            conv_weight = self.DSAFE_2.weight

        cost = F.conv2d(x, weight=conv_weight, stride=self.angRes, dilation=dilat, padding=pad)
        return cost


class Casual_Subcost_layer(nn.Module):
    def __init__(self, channel_in, channel_out, angRes):
        super(Casual_Subcost_layer, self).__init__()
        self.DSAFE_0 = nn.Conv2d(channel_in, channel_out, kernel_size=angRes // 2 + 1, stride=angRes, padding=0,
                                 bias=False)
        self.DSAFE_1 = nn.Conv2d(channel_in, channel_out, kernel_size=angRes // 4 + 1, stride=angRes, padding=0,
                                 bias=False)
        self.DSAFE_2 = nn.Conv2d(channel_in, channel_out, kernel_size=angRes // 8 + 1, stride=angRes, padding=0,
                                 bias=False)
        self.angRes = angRes

    def forward(self, x, interval, index, mindisp, case):
        d = mindisp + index * interval
        if d == 0:
            dilat = 1
            if case == 1:
                pad = (int(self.angRes // 2 * dilat), int(0),
                       int(self.angRes // 2 * dilat), int(0))
                x_pad = F.pad(x, pad, mode="reflect")
            elif case == 2:
                pad = (int(0), int(self.angRes // 2 * dilat),
                       int(self.angRes // 2 * dilat), int(0))
                x_pad = F.pad(x, pad, mode="reflect")
            elif case == 3:
                pad = (int(self.angRes // 2 * dilat), int(0),
                       int(0), int(self.angRes // 2 * dilat))
                x_pad = F.pad(x, pad, mode="reflect")
            else:
                pad = (int(0), int(self.angRes // 2 * dilat),
                       int(0), int(self.angRes // 2 * dilat))
                x_pad = F.pad(x, pad, mode="reflect")
            cost = F.conv2d(x_pad, weight=self.DSAFE_0.weight, stride=self.angRes, dilation=dilat,
                            padding=0)
        else:
            if index % 4 == 0:
                dilat = int(sign(d) * (d * self.angRes - 1))
                if case == 1:
                    pad = (int(self.angRes // 2 * dilat), int(0),
                           int(self.angRes // 2 * dilat), int(0))
                    x_pad = F.pad(x, pad, mode="reflect")
                elif case == 2:
                    pad = (int(0), int(self.angRes // 2 * dilat),
                           int(self.angRes // 2 * dilat), int(0))
                    x_pad = F.pad(x, pad, mode="reflect")
                elif case == 3:
                    pad = (int(self.angRes // 2 * dilat), int(0),
                           int(0), int(self.angRes // 2 * dilat))
                    x_pad = F.pad(x, pad, mode="reflect")
                else:
                    pad = (int(0), int(self.angRes // 2 * dilat),
                           int(0), int(self.angRes // 2 * dilat))
                    x_pad = F.pad(x, pad, mode="reflect")
                cost = F.conv2d(x_pad, weight=self.DSAFE_0.weight, stride=self.angRes, dilation=dilat,
                                padding=0)
            elif index % 4 == 2:
                dilat = int(sign(d) * 2 * (d * self.angRes - 1))
                if case == 1:
                    pad = (int(self.angRes // 4 * dilat), int(0),
                           int(self.angRes // 4 * dilat), int(0))
                    x_pad = F.pad(x, pad, mode="reflect")
                elif case == 2:
                    pad = (int(0), int(self.angRes // 4 * dilat),
                           int(self.angRes // 4 * dilat), int(0))
                    x_pad = F.pad(x, pad, mode="reflect")
                elif case == 3:
                    pad = (int(self.angRes // 4 * dilat), int(0),
                           int(0), int(self.angRes // 4 * dilat))
                    x_pad = F.pad(x, pad, mode="reflect")
                else:
                    pad = (int(0), int(self.angRes // 4 * dilat),
                           int(0), int(self.angRes // 4 * dilat))
                    x_pad = F.pad(x, pad, mode="reflect")
                cost = F.conv2d(x_pad, weight=self.DSAFE_1.weight, stride=self.angRes, dilation=dilat,
                                padding=0)
            else:
                dilat = int(sign(d) * 4 * (d * self.angRes - 1))
                if case == 1:
                    pad = (int(self.angRes // 8 * dilat), int(0),
                           int(self.angRes // 8 * dilat), int(0))
                    x_pad = F.pad(x, pad, mode="reflect")
                elif case == 2:
                    pad = (int(0), int(self.angRes // 8 * dilat),
                           int(self.angRes // 8 * dilat), int(0))
                    x_pad = F.pad(x, pad, mode="reflect")
                elif case == 3:
                    pad = (int(self.angRes // 8 * dilat), int(0),
                           int(0), int(self.angRes // 8 * dilat))
                    x_pad = F.pad(x, pad, mode="reflect")
                else:
                    pad = (int(0), int(self.angRes // 8 * dilat),
                           int(0), int(self.angRes // 8 * dilat))
                    x_pad = F.pad(x, pad, mode="reflect")
                cost = F.conv2d(x_pad, weight=self.DSAFE_2.weight, stride=self.angRes, dilation=dilat,
                                padding=0)
        return cost



def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


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


class faster_disparityregression(nn.Module):
    def __init__(self):
        super(faster_disparityregression, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, maxdisp, mindisp, disp_sample_number):
        interval = (maxdisp - mindisp) / (disp_sample_number - 1)

        score = self.softmax(x)

        a = time.time()
        d_values = torch.arange(disp_sample_number, device=score.device).float()
        disp_values = mindisp + interval * d_values
        temp = score * disp_values.unsqueeze(1).unsqueeze(1)
        # temp = torch.zeros(score.shape).to(score.device)
        # for d in range(disp_sample_number):
        #     temp[:, d, :, :] = score[:, d, :, :] * (mindisp + interval * d)
        b = time.time()
        c = b - a
        disp = torch.sum(temp, dim=1, keepdim=True)
        return disp


def _make_layer(block, inplanes, planes, blocks, stride, pad, dilation):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion), )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, pad, dilation))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, 1, None, pad, dilation))

    return nn.Sequential(*layers)


class Get_edge(nn.Module):
    def __init__(self):
        super(Get_edge, self).__init__()
        kernel_tensor_x = torch.tensor([[1.0, 0.0, -1.0],
                                        [2.0, 0.0, -2.0],
                                        [1.0, 0.0, -1.0]]).unsqueeze(0).unsqueeze(0)
        kernel_tensor_y = torch.tensor([[1.0, 2.0, 1.0],
                                        [0.0, 0.0, 0.0],
                                        [-1.0, -2.0, -1.0]]).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_tensor_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_tensor_y, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_x = F.conv2d(x_i.unsqueeze(1), self.weight_x, padding=1)
            x_i_y = F.conv2d(x_i.unsqueeze(1), self.weight_y, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_x, 2) + torch.pow(x_i_y, 2))
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        x = x / 4.
        return x


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


def make_layer(block, fn, angRes, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(fn, angRes))
    return nn.Sequential(*layers)


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


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out



if __name__ == "__main__":
    h, w = 32, 32
    net = Corse_To_Fine_net(h).cuda()
    input = torch.randn(32, 1, h * 9, w * 9).cuda()
    time_cost_list = []
    for i in range(50):
        with torch.no_grad():
            out, time_cost = net(input)
        print('time_cost = %f' % (time_cost))
        if i >= 10:
            time_cost_list.append(time_cost)
    time_cost_avg = np.mean(time_cost_list)
    print('average time cost = %f' % (time_cost_avg))
