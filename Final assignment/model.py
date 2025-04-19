import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class SwinEncoder(nn.Module):
    def __init__(self):
        super(SwinEncoder, self).__init__()

        # Load a swin-t v2 model pre-trained on ImageNet
        self.model = models.swin_v2_t(weights='IMAGENET1K_V1')

        # Remove the classification head
        self.model.head = nn.Identity()

    def forward(self, x):
        
        # Create list to save feature maps, shape (B, C, H, W)
        featuremaps = []

        # Extract features before last pooling layer
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        featuremaps.append(x.permute(0, 3, 1, 2)) # Save feature map after first transformer stage: (B, 128, 128, 128)
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        featuremaps.append(x.permute(0, 3, 1, 2)) # Save feature map after second transformer stage: (B, 256, 256, 256)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        featuremaps.append(x.permute(0, 3, 1, 2)) # Save feature map after third transformer stage: (B, 512, 32, 32)
        x = self.model.features[6](x)
        x = self.model.features[7](x)
        featuremaps.append(x.permute(0, 3, 1, 2)) # Save feature map at transformer output: (B, 1024, 16, 16)

        return featuremaps

class FaPNDecoder(nn.Module):
    def __init__(self,in_channels=[768,384,192,96], out_channels=256, n_classes=19, reg_weight=0.01):
        super(FaPNDecoder, self).__init__()

        self.offset_scalers = nn.Parameter(torch.ones(len(in_channels)) * 3.0)
        self.reg_weight = reg_weight
        self.offset_reg_loss = 0.0
        
        # Offset generation layers
        self.gen_offset = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch//2, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch//2),
                nn.GELU(),
                nn.Conv2d(ch//2, 18, kernel_size=3, padding=1),
                # Optional to prevent exploding offsets
                nn.Tanh()
            ) for ch in in_channels
        ])

        # Deformable convolution layers for feature allignment
        self.deform_conv = nn.ModuleList([
            DeformConv2d(ch, out_channels, kernel_size=3, padding=1) for ch in in_channels
        ])

        # GroupNorm after deformable conv
        self.post_deform_norm = nn.ModuleList([
            nn.GroupNorm(num_groups=32, num_channels=out_channels)
            for _ in in_channels
        ])

        # Upsampling path
        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=out_channels),
                nn.ReLU()
            ) for _ in range(4)
        ])

        # Final upsample
        self.final_upsample = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=out_channels),
                nn.GELU()
            ) for _ in range(2)
        ])

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Conv2d(out_channels, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ReLU(),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        )

    def forward(self, features):
        maps = []

        # Reverse order of features from deepest to shallowest
        features = features[::-1]

        # Take deepest feature as starting point
        x = features[0]

        # L2 norm for regularization
        total_offset_reg = 0.0
    
        for i in range(len(features)):
            # Generate offsets and perform deformable convolution
            offset = self.gen_offset[i](features[i])*self.offset_scalers[i]
            deformed_feat = self.deform_conv[i](features[i], offset)

            # Compute L2 penalty
            total_offset_reg += offset.pow(2).mean()
        
            # Step 2: Upsample
            if i != 0:
                x = self.upsamples[i-1](x) + deformed_feat
                maps.append(x)
            else:
                x = deformed_feat  # Initialize with deepest deformed feature

        # Compute and save L2 penalty
        total_offset_reg = total_offset_reg * self.reg_weight
        self.offset_reg_loss = total_offset_reg
        
        # Last upsample to produce detailed feature map for masking
        x = self.final_upsample[0](x)
        x = self.final_upsample[1](x)

        # Classification head
        x = self.classifier(x)

        return x
    
class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super(Model, self).__init__()
        
        self.encoder = SwinEncoder()
        self.pixel_decoder = FaPNDecoder()
        
    def forward(self, x):

        featuremaps = self.encoder(x)
        out = self.pixel_decoder(featuremaps)

        return out