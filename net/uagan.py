# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

from net.unet import UNetEncoder, UNetUpSamplingBlock, UNetConvBlock, ConvNormRelu


class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=4, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = [nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.001)]

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim *= 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, inputs):
        h = self.main(inputs)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                       nn.Sigmoid())

    def forward(self, inputs):
        return self.attention(inputs)


class Decoder(nn.Module):
    def __init__(self, out_channels=2, feature_maps=64, levels=4, norm_type='instance', bias=True):
        super(Decoder, self).__init__()

        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()

        for i in range(levels):
            att = AttentionBlock(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps)
            self.features.add_module('atten%d' % (i + 1), att)

            w = ConvNormRelu(2**(levels-i-1) * feature_maps, 2**(levels-i-1) * feature_maps,
                                 norm_type=norm_type, bias=bias)
            self.features.add_module('w%d' % (i + 1), w)

            upconv = UNetUpSamplingBlock(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                         deconv=True, bias=bias)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            conv_block = UNetConvBlock(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                       norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        self.score = nn.Conv2d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, self_encoder_outputs, aux_encoder_outputs):
        decoder_outputs = []
        self_encoder_outputs.reverse()
        aux_encoder_outputs.reverse()

        outputs = inputs
        for i in range(self.levels):
            w = getattr(self.features, 'w%d' % (i+1))(aux_encoder_outputs[i])
            a = getattr(self.features, 'atten%d' % (i+1))(aux_encoder_outputs[i])
            aux_encoder_output = w.mul(a)
            fuse_encoder_output = aux_encoder_output + self_encoder_outputs[i]
            outputs = getattr(self.features, 'upconv%d' % (i+1))(fuse_encoder_output, outputs)
            
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
            decoder_outputs.append(outputs)

        self_encoder_outputs.reverse()
        aux_encoder_outputs.reverse()
        return decoder_outputs, self.score(outputs)


class UAGAN(nn.Module):
    def __init__(self, seg_in, seg_out, syn_in, syn_out, feature_maps=64, levels=4, norm_type='instance',
                 bias=True, use_dropout=True):
        super(UAGAN, self).__init__()

        self.seg_encoder = UNetEncoder(seg_in, feature_maps, levels, norm_type, use_dropout, bias=bias,
                                       use_last_block=False)
        self.syn_encoder = UNetEncoder(syn_in, feature_maps, levels, norm_type, use_dropout, bias=bias,
                                       use_last_block=False)
        self.center_conv = UNetConvBlock(2**(levels-1) * feature_maps, 2**levels * feature_maps)
        self.seg_decoder = Decoder(seg_out, feature_maps, levels, norm_type, bias=bias)
        self.syn_decoder = Decoder(syn_out, feature_maps, levels, norm_type, bias=bias)

    def forward(self, seg_inputs, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, seg_inputs.size(2), seg_inputs.size(3))
        syn_inputs = torch.cat([seg_inputs, c], dim=1)

        seg_encoder_outputs, seg_output = self.seg_encoder(seg_inputs)
        syn_encoder_outputs, syn_output = self.syn_encoder(syn_inputs)

        seg_bottleneck = self.center_conv(seg_output)
        syn_bottleneck = self.center_conv(syn_output)

        _, seg_score = self.seg_decoder(seg_bottleneck, seg_encoder_outputs, syn_encoder_outputs)
        _, syn_score = self.syn_decoder(syn_bottleneck, syn_encoder_outputs, seg_encoder_outputs)
        return seg_score, syn_score


if __name__ == '__main__':
    n = UAGAN(1, 2, 4, 1)

    k = 0
    for p in list(n.parameters()):
        l = 1
        for j in p.size():
            l *= j
        k += l
    print(k / 1000000)
