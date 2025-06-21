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
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--n-epoch', type=int, default=15, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='recon', help='output folder')
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
voxnet = EnhancedVoxNetAutoEncoder()
print(voxnet)

if opt.model:
    voxnet.load_state_dict(torch.load(opt.model))

voxnet.cuda()
optimizer = optim.Adam(voxnet.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()
num_batch = len(train_dataset) / opt.batchSize

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

        optimizer.zero_grad()
        pred = voxnet(voxel)
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
            pred = voxnet(voxel)
            loss = loss_fn(pred, voxel)
            epoch_test_loss += loss.item()

    avg_test_loss = epoch_test_loss / len(test_dataloader)
    test_losses.append(avg_test_loss)
    print(f'Epoch {epoch}, Test Loss: {avg_test_loss:.6f}')

    # Save model
    torch.save(voxnet.state_dict(), f'{opt.outf}/recon_model_{epoch}.pth')

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
