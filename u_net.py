import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
    
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
    
class VoxNetAE_UNet(nn.Module):
    def __init__(self, input_shape=(32, 32, 32), z_dim=1024):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(0.3)

        # flatten前のサイズを動的に計算
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            h1 = self.enc1(dummy)
            h2 = self.enc2(self.pool(h1))
            h3 = self.enc3(self.pool(h2))
            self.h1_shape = h1.shape[2:]
            self.h2_shape = h2.shape[2:]
            self.h3_shape = h3.shape[2:]
            feat_dim = h3.view(1, -1).shape[1]

        self.fc_enc = nn.Linear(feat_dim, z_dim)
        self.fc_dec = nn.Linear(z_dim, feat_dim)

        # Decoder
        self.up3 = UpConvBlock(128, 64)
        self.up2 = UpConvBlock(64 + 64, 32)
        self.up1 = nn.Sequential(
            nn.Conv3d(32 + 32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # --- Encoder ---
        h1 = self.enc1(x)                # [B, 32, 32, 32, 32]
        h2 = self.enc2(self.pool(h1))   # [B, 64, 16, 16, 16]
        h3 = self.enc3(self.pool(h2))   # [B, 128, 8, 8, 8]

        h3_flat = self.dropout(h3).flatten(1)

        # --- Bottleneck ---
        z = self.fc_enc(h3_flat)
        h3_rec_flat = self.fc_dec(z)
        h3_rec = h3_rec_flat.view(-1, 128, *self.h3_shape)

        # --- Decoder + skip connections ---
        d3 = self.up3(h3_rec)                # [B, 64, 16, 16, 16]
        d3_cat = torch.cat([d3, h2], dim=1)  # [B, 128, 16, 16, 16]

        d2 = self.up2(d3_cat)                # [B, 32, 32, 32, 32]
        d2_cat = torch.cat([d2, h1], dim=1)  # [B, 64, 32, 32, 32]

        out = self.up1(d2_cat)               # [B, 1, 32, 32, 32]

        return out