#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: voxnet.py
Created: 2020-01-21 21:32:40
Author : Yangmaonan
Email : 59786677@qq.com
Description: VoxNet 网络结构
'''

import torch
import torch.nn as nn
from collections import OrderedDict


class VoxNet(nn.Module):
    def __init__(self, n_classes=10, input_shape=(32, 32, 32)):
        super(VoxNet, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.mlp = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(dim_feat, 128)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4)),
            ('fc2', torch.nn.Linear(128, self.n_classes))
        ]))

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


class VoxNetAutoEncoder(nn.Module):
    def __init__(self, input_shape=(32, 32, 32), z_dim=128):
        super().__init__()
        self.input_shape = input_shape

        shape_after_conv = [
            ((dim - 5 + 1) // 2 - 3 + 1) // 2
            for dim in input_shape
        ]
        shape_after_conv.insert(0, 32)

        shape_after_conv = tuple(shape_after_conv)
        dim_feat = 1 # out channel size
        for dim_size in shape_after_conv:
            dim_feat *= dim_size

        self.encoder = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3)),
            ('flatten', torch.nn.Flatten()),
            ('fc1', torch.nn.Linear(dim_feat, z_dim))
        ]))

        self.decoder = torch.nn.Sequential(OrderedDict([
            ('fc2', torch.nn.Linear(z_dim, dim_feat)),
            ('relu3', torch.nn.ReLU()),
            ('unflatten', torch.nn.Unflatten(-1, shape_after_conv)),
            ('convtrans3d_1', torch.nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=1, stride=2, output_padding=1)),
            ('relu4', torch.nn.ReLU()),
            ('convtrans3d_2', torch.nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu5', torch.nn.ReLU()),
            ('convtrans3d_3', torch.nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=5, stride=2, output_padding=1)),            
            
        ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class EnhancedVoxNetAutoEncoder(nn.Module):
    def __init__(self, input_shape=(32, 32, 32), z_dim=256):
        super().__init__()
        self.input_shape = input_shape

        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)),
            ('bn1', nn.BatchNorm3d(32)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)),
            ('bn2', nn.BatchNorm3d(64)),
            ('relu2', nn.ReLU()),
            ('drop1', nn.Dropout3d(0.3)),
            ('conv3', nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)),
            ('bn3', nn.BatchNorm3d(128)),
            ('relu3', nn.ReLU()),
            ('drop2', nn.Dropout3d(0.3)),
            ('flatten', nn.Flatten())
        ]))

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            feat_dim = self.encoder(dummy).shape[1]

        self.fc_enc = nn.Linear(feat_dim, z_dim)

        # Decoder
        self.fc_dec = nn.Linear(z_dim, feat_dim)

        self.decoder = nn.Sequential(OrderedDict([
            ('unflatten', nn.Unflatten(1, (128, input_shape[0] // 4, input_shape[1] // 4, input_shape[2] // 4))),
            ('deconv1', nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)),
            ('bn4', nn.BatchNorm3d(64)),
            ('relu4', nn.ReLU()),
            ('deconv2', nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)),
            ('bn5', nn.BatchNorm3d(32)),
            ('relu5', nn.ReLU()),
            ('deconv3', nn.ConvTranspose3d(32, 1, kernel_size=3, padding=1)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc_enc(x)
        x = self.fc_dec(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = EnhancedVoxNetAutoEncoder()
    dummy_input = torch.rand(4, 1, 32, 32, 32)
    output = model(dummy_input)
    print(output.shape)  # expected: (4, 1, 32, 32, 32)
