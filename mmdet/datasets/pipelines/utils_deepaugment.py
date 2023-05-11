import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gelu
from torch.nn.functional import conv2d

import math

##############
### common ###
##############

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


############
### EDSR ###
############

class EDSR(nn.Module):
    def __init__(self,
                 n_resblocks=16, n_feats=64, scale=4, res_scale=1, rgb_range=255, n_colors=3,
                 conv=default_conv):
        super(EDSR, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, pre_distortions={None}, body_distortions={None}):
        # print("Using FF pre distortions = ", pre_distortions)
        # print("Using FF body distortions = ", body_distortions)
        x = self.sub_mean(x)
        x = self.head(x)

        ######################################################################
        # PRE - DISTORTIONS
        ######################################################################

        if 1 in pre_distortions:
            for _ in range(5):
                c1, c2 = random.randint(0, 63), random.randint(0, 63)
                x[:, c1], x[:, c2] = x[:, c2], x[:, c1]

        if 2 in pre_distortions:
            rand_filter_weight = torch.round(torch.rand_like(x) + 0.45) # Random matrix of 1s and 0s
            x = x * rand_filter_weight
            del rand_filter_weight

        if 3 in pre_distortions:
            rand_filter_weight = (torch.round(torch.rand_like(x) + 0.475) * 2) - 1 # Random matrix of 1s and -1s
            x = x * rand_filter_weight
            del rand_filter_weight

        ######################################################################
        # BODY - DISTORTIONS
        ######################################################################

        if 1 in body_distortions:
            res = self.body[:5](x)
            res = -res
            res = self.body[5:](res)
        elif 2 in body_distortions:
            if random.randint(0, 2) == 1:
                act = F.relu
            else:
                act = F.gelu
            res = self.body[:5](x)
            res = act(res)
            res = self.body[5:](res)
        elif 3 in body_distortions:
            if random.randint(0, 2) == 1:
                axes = [1, 2]
            else:
                axes = [1, 3]
            res = self.body[:5](x)
            res = torch.flip(res, axes)
            res = self.body[5:](res)
        elif 4 in body_distortions:
            to_skip = set([random.randint(2, 16) for _ in range(3)])
            for i in range(len(self.body)):
                if i not in to_skip:
                    res = self.body[i](x)
        else:
            res = self.body(x)

        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        # print(f"EDSR: predistortion={pre_distortions}, bodydistortion={body_distortions}")

        return x


###############
### WEIGHTS ###
###############

def _get_weights_EDSR():
    weights = torch.load('/ws/external/mmdet/datasets/pipelines/weights/edsr_baseline_x4.pt')

    random_sample_list = np.random.randint(0,17, size=3)
    for option in list(random_sample_list):
        if option == 0:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = torch.flip(weights['body.'+str(i)+'.body.0.weight'], (0,))
            weights['body.'+str(i)+'.body.0.bias'] = torch.flip(weights['body.'+str(i)+'.body.0.bias'], (0,))
            weights['body.'+str(i)+'.body.2.weight'] = torch.flip(weights['body.'+str(i)+'.body.2.weight'], (0,))
            weights['body.'+str(i)+'.body.2.bias'] = torch.flip(weights['body.'+str(i)+'.body.2.bias'], (0,))
        elif option == 1:
            i = np.random.choice(np.arange(1,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = -weights['body.'+str(i)+'.body.0.weight']
            weights['body.'+str(i)+'.body.0.bias'] = -weights['body.'+str(i)+'.body.0.bias']
        elif option == 2:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = 0 * weights['body.'+str(i)+'.body.0.weight']
            weights['body.'+str(i)+'.body.0.bias'] = 0*weights['body.'+str(i)+'.body.0.bias']
        elif option == 3:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = -gelu(weights['body.'+str(i)+'.body.0.weight'])
            weights['body.'+str(i)+'.body.2.weight'] = -gelu(weights['body.'+str(i)+'.body.2.weight'])
        elif option == 4:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = weights['body.'+str(i)+'.body.0.weight'] *\
            torch.Tensor([[0, 1, 0],[1, -4., 1], [0, 1, 0]]).view(1,1,3,3).cuda()
        elif option == 5:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = weights['body.'+str(i)+'.body.0.weight'] *\
            torch.Tensor([[-1, -1, -1],[-1, 8., -1], [-1, -1, -1]]).view(1,1,3,3).cuda()
        elif option == 6:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.2.weight'] = weights['body.'+str(i)+'.body.2.weight'] *\
            (1 + 2 * np.float32(np.random.uniform()) * (2*torch.rand_like(weights['body.'+str(i)+'.body.2.weight']-1)))
        elif option == 7:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = torch.flip(weights['body.'+str(i)+'.body.0.weight'], (-1,))
            weights['body.'+str(i)+'.body.2.weight'] = -1 * weights['body.'+str(i)+'.body.2.weight']
        elif option == 8:
            i = np.random.choice(np.arange(1,13,4))
            z = torch.zeros_like(weights['body.'+str(i)+'.body.0.weight'])
            for j in range(z.size(0)):
                shift_x, shift_y = np.random.randint(3, size=(2,))
                z[:,j,shift_x,shift_y] = np.random.choice([1.,-1.])
            weights['body.'+str(i)+'.body.0.weight'] = conv2d(weights['body.'+str(i)+'.body.0.weight'], z, padding=1)
        elif option == 9:
            i = np.random.choice(np.arange(0,10,3))
            z = (2*torch.rand_like(weights['body.'+str(i)+'.body.0.weight'])*np.float32(np.random.uniform()) - 1)/6.
            weights['body.'+str(i)+'.body.0.weight'] = conv2d(weights['body.'+str(i)+'.body.0.weight'], z, padding=1)
        elif option == 10:
            i = np.random.choice(np.arange(1,12,4))
            z = torch.FloatTensor(np.random.dirichlet([0.1] * 9, (64,64))).view(64,64,3,3).cuda() # 2.weight
            weights['body.'+str(i)+'.body.2.weight'] = conv2d(weights['body.'+str(i)+'.body.2.weight'], z, padding=1)
        elif option == 11: ############ Start Saurav's changes ############
            i = random.choice(list(range(15)))
            noise = (torch.rand_like(weights['body.'+str(i)+'.body.2.weight']) - 0.5) * 1.0
            weights['body.'+str(i)+'.body.2.weight'] += noise
        elif option == 12:
            _ij = [[random.choice(list(range(15))), random.choice([0, 2])] for _ in range(5)]
            for i, j in _ij:
                _k = random.randint(1, 3)
                if random.randint(0, 1) == 0:
                    _dims = (2,3)
                else:
                    _dims = (0,1)
                weights['body.'+str(i)+'.body.'+str(j)+'.weight'] = torch.rot90(weights['body.'+str(i)+'.body.'+str(j)+'.weight'], k=_k, dims=_dims)
        elif option == 13:
            _i = [random.choice(list(range(15))) for _ in range(5)]
            for i in _i:
                rand_filter_weight = torch.round(torch.rand_like(weights['body.'+str(i)+'.body.0.weight'])) * 2 - 1 # Random matrix of 1s and -1s
                weights['body.'+str(i)+'.body.0.weight'] = weights['body.'+str(i)+'.body.0.weight'] * rand_filter_weight
        elif option == 14:
            # Barely noticable difference here
            _i = [random.choice(list(range(15))) for _ in range(5)]
            for i in _i:
                rand_filter_weight = torch.round(torch.rand_like(weights['body.'+str(i)+'.body.0.weight'])) # Random matrix of 1s and 0s
                weights['body.'+str(i)+'.body.0.weight'] = weights['body.'+str(i)+'.body.0.weight'] * rand_filter_weight
        elif option == 15:
            # Negate some entire filters. Definitely a noticable difference
            _i = [random.choice(list(range(15))) for _ in range(5)]
            for i in _i:
                filters_to_be_zeroed = [random.choice(list(range(64))) for _ in range(32)]
                weights['body.'+str(i)+'.body.0.weight'][filters_to_be_zeroed] *= -1
        elif option == 16:
            # Only keep the max filter value in the conv
            _ij = [[random.choice(list(range(15))), random.choice([0, 2])] for _ in range(5)]
            for i, j in _ij:
                w = torch.reshape(weights['body.'+str(i)+'.body.'+str(j)+'.weight'], shape=(64, 64, 9))
                res = torch.topk(w, k=1)

                w_new = torch.zeros_like(w).scatter(2, res.indices, res.values)
                w_new = w_new.reshape(64, 64, 3, 3)
                weights['body.'+str(i)+'.body.'+str(j)+'.weight'] = w_new
        else:
            raise NotImplementedError()
    return weights


def _get_weights_CAE():
    weight_keys = ['e_conv_1.1.weight', 'e_conv_1.1.bias', 'e_conv_2.1.weight', 'e_conv_2.1.bias', 'e_block_1.1.weight',
                   'e_block_1.1.bias', 'e_block_1.4.weight', 'e_block_1.4.bias', 'e_block_2.1.weight',
                   'e_block_2.1.bias', 'e_block_2.4.weight', 'e_block_2.4.bias', 'e_block_3.1.weight',
                   'e_block_3.1.bias', 'e_block_3.4.weight', 'e_block_3.4.bias', 'e_conv_3.0.weight', 'e_conv_3.0.bias',
                   'd_up_conv_1.0.weight', 'd_up_conv_1.0.bias', 'd_up_conv_1.3.weight', 'd_up_conv_1.3.bias',
                   'd_block_1.1.weight', 'd_block_1.1.bias', 'd_block_1.4.weight', 'd_block_1.4.bias',
                   'd_block_2.1.weight', 'd_block_2.1.bias', 'd_block_2.4.weight', 'd_block_2.4.bias',
                   'd_block_3.1.weight', 'd_block_3.1.bias', 'd_block_3.4.weight', 'd_block_3.4.bias',
                   'd_up_conv_2.0.weight', 'd_up_conv_2.0.bias', 'd_up_conv_2.3.weight', 'd_up_conv_2.3.bias',
                   'd_up_conv_3.0.weight', 'd_up_conv_3.0.bias', 'd_up_conv_3.3.weight', 'd_up_conv_3.3.bias']
    key_mapping = dict(
        [(str(int(i / 2)) + ".weight", key) if i % 2 == 0 else (str(int(i / 2)) + ".bias", key) for i, key in
         enumerate(weight_keys)])
    NUM_LAYERS = int(len(key_mapping.values()) / 2)  # 21
    NUM_DISTORTIONS = 8
    MODEL_PATH = "/ws/external/mmdet/datasets/pipelines/weights/model_final.state"
    OPTION_LAYER_MAPPING = {0: range(11, NUM_LAYERS - 5), 1: range(8, NUM_LAYERS - 7), 2: range(8, NUM_LAYERS - 7),
                            3: range(10, NUM_LAYERS - 7), 4: range(8, NUM_LAYERS - 7), 5: range(8, NUM_LAYERS - 7),
                            6: range(8, NUM_LAYERS - 7), 7: range(8, NUM_LAYERS - 7), 8: range(8, NUM_LAYERS - 7)}

    def get_name(i, tpe):
        return key_mapping[str(i) + "." + tpe]

    weights = torch.load(MODEL_PATH)
    for option in random.sample(range(NUM_DISTORTIONS), 1):
        i = np.random.choice(OPTION_LAYER_MAPPING[option])
        j = np.random.choice(OPTION_LAYER_MAPPING[option])
        weight_i = get_name(i, "weight")
        bias_i = get_name(i, "bias")
        weight_j = get_name(j, "weight")
        bias_j = get_name(j, "weight")
        if option == 0:
            weights[weight_i] = torch.flip(weights[weight_i], (0,))
            weights[bias_i] = torch.flip(weights[bias_i], (0,))
            weights[weight_j] = torch.flip(weights[weight_j], (0,))
            weights[bias_j] = torch.flip(weights[bias_j], (0,))
        elif option == 1:
            for k in [np.random.choice(weights[weight_i].size()[0]) for _ in range(12)]:
                weights[weight_i][k] = -weights[weight_i][k]
                weights[bias_i][k] = -weights[bias_i][k]
        elif option == 2:
            for k in [np.random.choice(weights[weight_i].size()[0]) for _ in range(25)]:
                weights[weight_i][k] = 0 * weights[weight_i][k]
                weights[bias_i][k] = 0 * weights[bias_i][k]
        elif option == 3:
            for k in [np.random.choice(weights[weight_i].size()[0]) for _ in range(25)]:
                weights[weight_i][k] = -gelu(weights[weight_i][k])
                weights[bias_i][k] = -gelu(weights[bias_i][k])
        elif option == 4:
            weights[weight_i] = weights[weight_i] * \
                                (1 + 2 * np.float32(np.random.uniform()) * (4 * torch.rand_like(weights[weight_i] - 1)))
            weights[weight_j] = weights[weight_j] * \
                                (1 + 2 * np.float32(np.random.uniform()) * (4 * torch.rand_like(weights[weight_j] - 1)))
        elif option == 5:  ##### begin saurav #####
            if random.random() < 0.5:
                mask = torch.round(torch.rand_like(weights[weight_i]))
            else:
                mask = torch.round(torch.rand_like(weights[weight_i])) * 2 - 1
            weights[weight_i] *= mask
        elif option == 6:
            _k = random.randint(1, 3)
            weights[weight_i] = torch.rot90(weights[weight_i], k=_k, dims=(2, 3))
        elif option == 7:
            out_filters = weights[weight_i].shape[0]
            to_zero = list(set([random.choice(list(range(out_filters))) for _ in range(out_filters // 5)]))
            weights[weight_i][to_zero] = weights[weight_i][to_zero] * -1.0
        elif option == 8:
            # Only keep the max filter value in the conv
            c1, c2, width = weights[weight_i].shape[0], weights[weight_i].shape[1], weights[weight_i].shape[2]
            assert weights[weight_i].shape[2] == weights[weight_i].shape[3]

            w = torch.reshape(weights[weight_i], shape=(c1, c2, width ** 2))
            res = torch.topk(w, k=1)

            w_new = torch.zeros_like(w).scatter(2, res.indices, res.values)
            w_new = w_new.reshape(c1, c2, width, width)
            weights[weight_i] = w_new

    return weights


###########
### CAE ###
###########


class CAE(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)
    Latent representation: 32x32x32 bits per patch => 240KB per image (for 720p)
    """

    def __init__(self):
        super(CAE, self).__init__()

        self.encoded = None

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 32x32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Tanh()
        )

        # DECODER

        # a
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )

        # 128x64x64
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 256x128x128
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation
        if self.use_flip:
            option = np.random.choice(range(9))
        else:
            option_list = list(range(9))
            option_list.remove(4)
            option_list.remove(5)
            option = np.random.choice(option_list)
        if option == 1:
            # set some weights to zero
            H = ec3.size()[2]
            W = ec3.size()[3]
            mask = (torch.cuda.FloatTensor(H, W).uniform_() > 0.2).float().cuda()
            ec3 = ec3 * mask
            del mask
        elif option == 2:
            # negare some of the weights
            H = ec3.size()[2]
            W = ec3.size()[3]
            mask = (((torch.cuda.FloatTensor(H, W).uniform_() > 0.1).float() * 2) - 1).cuda()
            ec3 = ec3 * mask
            del mask
        elif option == 3:
            num_channels = 10
            perm = np.array(list(np.random.permutation(num_channels)) + list(range(num_channels, ec3.size()[1])))
            ec3 = ec3[:, perm, :, :]
        elif option == 4:
            num_channels = ec3.shape[1]
            num_channels_transform = 5

            _k = random.randint(1, 3)
            _dims = [0, 1, 2]
            random.shuffle(_dims)
            _dims = _dims[:2] # [0,1]: vertical_flip;  [0,2]: horizontal_flip; [1,2]: vertical & horizontal flip;

            for i in range(num_channels_transform):
                filter_select = random.choice(list(range(num_channels)))
                ec3[:, filter_select] = torch.flip(ec3[:, filter_select], dims=_dims)
        elif option == 5:
            num_channels = ec3.shape[1]
            num_channels_transform = num_channels

            _k = random.randint(1, 3)
            _dims = [0, 1, 2]
            random.shuffle(_dims)
            _dims = _dims[:2]

            for i in range(num_channels_transform):
                if i == num_channels_transform / 2:
                    _dims = [_dims[1], _dims[0]]
                ec3[:, i] = torch.flip(ec3[:, i], dims=_dims) # horizontal flip
        elif option == 6:
            with torch.no_grad():
                c, h, w = ec3.shape[1], ec3.shape[2], ec3.shape[3]
                z = torch.zeros(c, c, 3, 3).cuda()
                for j in range(z.size(0)):
                    shift_x, shift_y = 1, 1  # np.random.randint(3, size=(2,))
                    z[j, j, shift_x, shift_y] = 1  # np.random.choice([1.,-1.])

                # Without this line, z would be the identity convolution
                z = z + ((torch.rand_like(z) - 0.5) * 0.2)
                ec3 = F.conv2d(ec3, z, padding=1)
                del z
        elif option == 7:
            with torch.no_grad():
                c, h, w = ec3.shape[1], ec3.shape[2], ec3.shape[3]
                z = torch.zeros(c, c, 3, 3).cuda()
                for j in range(z.size(0)):
                    shift_x, shift_y = 1, 1  # np.random.randint(3, size=(2,))
                    z[j, j, shift_x, shift_y] = 1  # np.random.choice([1.,-1.])

                    if random.random() < 0.5:
                        rand_layer = random.randint(0, c - 1)
                        z[j, rand_layer, random.randint(-1, 1), random.randint(-1, 1)] = 1

                ec3 = F.conv2d(ec3, z, padding=1)
                del z
        elif option == 8:
            with torch.no_grad():
                c, h, w = ec3.shape[1], ec3.shape[2], ec3.shape[3]
                z = torch.zeros(c, c, 3, 3).cuda()
                shift_x, shift_y = np.random.randint(3, size=(2,))
                for j in range(z.size(0)):
                    if random.random() < 0.2:
                        shift_x, shift_y = np.random.randint(3, size=(2,))

                    z[j, j, shift_x, shift_y] = 1  # np.random.choice([1.,-1.])

                # Without this line, z would be the identity convolution
                # z = z + ((torch.rand_like(z) - 0.5) * 0.2)
                ec3 = F.conv2d(ec3, z, padding=1)
                del z

        # stochastic binarization
        with torch.no_grad():
            rand = torch.rand(ec3.shape).cuda()
            prob = (1 + ec3) / 2
            eps = torch.zeros(ec3.shape).cuda()
            eps[rand <= prob] = (1 - ec3)[rand <= prob]
            eps[rand > prob] = (-ec3 - 1)[rand > prob]

        # encoded tensor
        self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)
        if option == 0:
            self.encoded = self.encoded * \
                           (3 + 2 * np.float32(np.random.uniform()) * (2 * torch.rand_like(self.encoded - 1)))

        # print(f"CAE: option={option}")
        return self.decode(self.encoded)

    def decode(self, encoded):
        y = encoded * 2.0 - 1  # (0|1) -> (-1|1)

        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec
