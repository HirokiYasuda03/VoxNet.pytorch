#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: train.py
Description: Training script for EnhancedVoxNetAutoEncoder on ModelNet10
'''

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from voxnet import EnhancedVoxNetAutoEncoder
from data.modelnet10 import ModelNet10
import numpy as np
import torch.nn.functional as F
import math

# Constants
CLASSES = {
    0: 'bathtub',
    1: 'chair',
    2: 'dresser',
    3: 'night_stand',
    4: 'sofa',
    5: 'toilet',
    6: 'bed',
    7: 'desk',
    8: 'monitor',
    9: 'table'
}
N_CLASSES = len(CLASSES)  

# Color print function
def blue(x): return '\033[94m' + x + '\033[0m'

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='data/ModelNet10', help="dataset path")
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--n-epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='denoise', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
opt = parser.parse_args()

# Create output directory
os.makedirs(opt.outf, exist_ok=True)

# Set random seed
opt.manualSeed = random.randint(1, 10000)
print("Random Seed:", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Load dataset
train_dataset = ModelNet10(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='train')
test_dataset = ModelNet10(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='test')
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)

# Model initialization
voxnet = EnhancedVoxNetAutoEncoder(z_dim=1024)
print(voxnet)

if opt.model:
    voxnet.load_state_dict(torch.load(opt.model))

voxnet.cuda()
optimizer = optim.Adam(voxnet.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()
num_batch = len(train_dataset) / opt.batchSize

# noise function
def add_salt_pepper_noise(voxel, noise_ratio = 0.03):
    mask = np.random.choice(2, size=voxel.size(), p=[1 - noise_ratio, noise_ratio])
    indices = np.where(mask)
    voxel[indices] = 1 - voxel[indices]
    return voxel

def add_occlusion_noise(voxel, block_size=8, n_blocks=1):
    for _ in range(n_blocks):
        x = np.random.randint(4, 28 - block_size)
        y = np.random.randint(4, 28 - block_size)
        z = np.random.randint(4, 28 - block_size)
        voxel[:, :, x:x+block_size, y:y+block_size, z:z+block_size] = 0
    return voxel

def rotate_voxels(voxels, angles_deg):
    """
    voxels: (B, 1, D, H, W) float tensor (e.g., on GPU)
    angles_deg: list or tensor of shape (B,) with rotation angles in degrees
    returns: rotated voxels of shape (B, 1, D, H, W)
    """
    B, C, X, Y, Z = voxels.shape
    device = voxels.device

    voxels = voxels.permute(0, 1, 4, 2, 3)
    
    # 正規化された座標に対する2D回転行列（Z軸回転）
    theta = []
    for angle in angles_deg:
        rad = math.radians(angle)
        cos = math.cos(rad)
        sin = math.sin(rad)

        # 3D affine行列（バッチごとに定義）
        # Z軸回転なので X-Y平面に回転行列を適用、Zはそのまま
        mat = torch.tensor([
            [cos, -sin, 0, 0],
            [sin,  cos, 0, 0],
            [0,    0,   1, 0]
        ], dtype=torch.float32, device=device)
        theta.append(mat)

    theta = torch.stack(theta)  # (B, 3, 4)

    # `affine_grid` expects normalized coordinates in [-1, 1]
    grid = F.affine_grid(theta, size=voxels.size(), align_corners=True)  # (B, D, H, W, 3)

    # grid_sample: 3Dボクセルに対する補間付きサンプリング
    rotated = F.grid_sample(voxels, grid, align_corners=True, mode='nearest', padding_mode='zeros')
    rotated = rotated.permute(0, 1, 3, 4, 2)
    return rotated


# Loss tracking
train_losses = []
test_losses = []
best_test_loss = float('inf')
no_improve_count = 0

# Training loop
for epoch in range(opt.n_epoch):
    epoch_train_loss = 0.0
    voxnet.train()
    for i, sample in enumerate(train_dataloader):
        voxel = sample['voxel'].float().cuda()
        voxel = rotate_voxels(voxel, torch.rand(voxel.shape[0]) * 360)
        
        choices = np.random.randint(0, 5, (voxel.shape[0], ))
        noised_voxel = voxel.detach().clone()
        noised_voxel[((choices == 1) | (choices == 3))] = add_salt_pepper_noise(noised_voxel[(choices==1) | (choices == 3)])
        noised_voxel[((choices == 2) | (choices == 3))] = add_occlusion_noise(noised_voxel[((choices == 2) | (choices == 3))])

        optimizer.zero_grad()
        pred = voxnet(noised_voxel)
        loss = loss_fn(pred, voxel)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        if i % 5 == 0:
            print(f'Epoch {epoch}, Batch {i}, Train Loss: {loss.item():.6f}')

    avg_train_loss = epoch_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # Evaluate on test set
    voxnet.eval()
    epoch_test_loss = 0.0
    with torch.no_grad():
        for sample in test_dataloader:
            voxel = sample['voxel'].float().cuda()
            noised_voxel = voxel.detach().clone()
            noised_voxel = add_salt_pepper_noise(noised_voxel)
            noised_voxel = add_occlusion_noise(noised_voxel)
            pred = voxnet(noised_voxel)
            loss = loss_fn(pred, voxel)
            epoch_test_loss += loss.item()

    avg_test_loss = epoch_test_loss / len(test_dataloader)
    test_losses.append(avg_test_loss)
    print(f'Epoch {epoch}, Test Loss: {avg_test_loss:.6f}')

    # Save model
    torch.save(voxnet.state_dict(), f'{opt.outf}/denoise_model_{epoch}.pth')

    # Early stopping
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= 3:
            print("Early stopping: test loss did not improve for 3 consecutive epochs.")
            break

# Final report
print("Training complete.")
print("Train Losses:", train_losses)
print("Test Losses:", test_losses)
