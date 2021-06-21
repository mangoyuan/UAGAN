# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1, norm_type='instance'):

        super(ConvNormRelu, self).__init__()
        norm = nn.BatchNorm2d if norm_type == 'batch' else nn.InstanceNorm2d
        if padding == 'SAME':
            p = kernel_size // 2
        else:
            p = 0

        self.unit = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias, dilation=dilation),
                                  norm(out_channels),
                                  nn.LeakyReLU(0.01))

    def forward(self, inputs):
        return self.unit(inputs)


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', norm_type='instance', bias=True):
        super(UNetConvBlock, self).__init__()

        self.conv1 = ConvNormRelu(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                  norm_type=norm_type, bias=bias)
        self.conv2 = ConvNormRelu(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                  norm_type=norm_type, bias=bias)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UNetUpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bias=True):
        super(UNetUpSamplingBlock, self).__init__()
        self.deconv = deconv
        if self.deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=False)
            )
            
    def forward(self, *inputs):
        if len(inputs) == 2:
            return self.forward_concat(inputs[0], inputs[1])
        else:
            return self.forward_standard(inputs[0])

    def forward_concat(self, inputs1, inputs2):
        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):
        return self.up(inputs)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm_type='instance', use_dropout=True, bias=True,
                 use_last_block=True):
        super(UNetEncoder, self).__init__()

        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()
        self.use_dropout = use_dropout
        self.use_last_block = use_last_block
        if self.use_dropout:
            self.dropout = nn.Dropout()

        in_features = in_channels
        for i in range(levels):
            out_features = (2**i) * feature_maps

            conv_block = UNetConvBlock(in_features, out_features, norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i+1), pool)

            in_features = out_features
        if use_last_block:
            self.center_conv = UNetConvBlock(2**(levels-1) * feature_maps, 2**levels * feature_maps)

    def forward(self, inputs):
        encoder_outputs = []

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
            if i == self.levels - 1 and self.use_dropout:
                encoder_outputs.append(self.dropout(outputs))
            else:
                encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i+1))(outputs)
        if self.use_last_block:
            outputs = self.center_conv(outputs)
        return encoder_outputs, outputs


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, feature_maps=64, levels=4, norm_type='instance', bias=True):
        super(UNetDecoder, self).__init__()

        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()

        for i in range(levels):
            upconv = UNetUpSamplingBlock(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps, deconv=True,
                                         bias=bias)
            self.features.add_module('upconv%d' % (i+1), upconv)

            conv_block = UNetConvBlock(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps,
                                       norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

        self.score = nn.Conv2d(feature_maps, out_channels, kernel_size=1, bias=bias)

    def forward(self, inputs, encoder_outputs):
        decoder_outputs = []
        encoder_outputs.reverse()

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'upconv%d' % (i+1))(encoder_outputs[i], outputs)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
            decoder_outputs.append(outputs)
        encoder_outputs.reverse()
        return decoder_outputs, self.score(outputs)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, feature_map=64, levels=4, norm_type='instance',
                 use_dropout=True, bias=True):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, feature_map, levels, norm_type, use_dropout, bias=bias)
        self.decoder = UNetDecoder(out_channels, feature_map, levels, norm_type, bias=bias)

    def forward(self, inputs):
        encoder_outputs, final_output = self.encoder(inputs)
        decoder_outputs, outputs = self.decoder(final_output, encoder_outputs)
        return outputs


if __name__ == '__main__':
    u = UNet(1, 4)
    x = torch.Tensor(8, 1, 190, 190)
    y = u(x)
