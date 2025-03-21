import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

# Second implementation of the model
# Swin-B v2 as encoder and FaPN as decoder

class SwinEncoder(nn.Module):
    def __init__(self):
        super(SwinEncoder, self).__init__()

        # Load a swin-b v2 model pre-trained on ImageNet
        self.model = models.swin_v2_b(weights='IMAGENET1K_V1')

        # Remove the classification head
        self.model.head = nn.Identity()

    def forward(self, x):
        
        # Create list to save feature maps
        featuremaps = []

        # Extract features before last pooling layer
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        featuremaps.append(x.permute(0, 3, 1, 2)) # Save feature map after first transformer stage
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        featuremaps.append(x.permute(0, 3, 1, 2)) # Save feature map after second transformer stage
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        featuremaps.append(x.permute(0, 3, 1, 2)) # Save feature map after third transformer stage
        x = self.model.features[6](x)
        x = self.model.features[7](x)
        featuremaps.append(x.permute(0, 3, 1, 2)) # Save feature map at transformer output

        # Permutate dimensions of feature maps to (N, C, H, W)
        #for i in len(featuremaps):
            #featuremaps[i] = featuremaps[i].permute(0, 3, 1, 2)

        return featuremaps
    
class FaPNDecoder(nn.Module):
    def __init__(self,in_channels=1024, out_channels=256, num_classes=19):
        super(FaPNDecoder, self).__init__()

        # Offset generation layers
        self.gen_offset1 = nn.Conv2d(1024, 18, kernel_size=3, padding=1)
        self.gen_offset2 = nn.Conv2d(512, 18, kernel_size=3, padding=1)
        self.gen_offset3 = nn.Conv2d(256, 18, kernel_size=3, padding=1)
        self.gen_offset4 = nn.Conv2d(128, 18, kernel_size=3, padding=1)

        # Deformable convolution layers for feature allignment
        self.deform_conv1 = DeformConv2d(1024, out_channels, kernel_size=3, padding=1)
        self.deform_conv2 = DeformConv2d(512, out_channels, kernel_size=3, padding=1)
        self.deform_conv3 = DeformConv2d(256, out_channels, kernel_size=3, padding=1)
        self.deform_conv4 = DeformConv2d(128, out_channels, kernel_size=3, padding=1)

        # Upsampling path (assuming 512x512 input --> 16x16x1024 after swin encoder)
        self.up1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

        # Additional deformable convolution for refinement + offset
        self.offset5 = nn.Conv2d(out_channels, 18, kernel_size=3, padding=1)
        self.deform_conv5 = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        

        # Final convolution layer
        self.conv = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, features):

        # Generate offsets for deformable convolution
        offset0 = self.gen_offset1(features[3]) # 16x16x18
        offset1 = self.gen_offset2(features[2]) # 32x32x18
        offset2 = self.gen_offset3(features[1]) # 64x64x18
        offset3 = self.gen_offset4(features[0]) # 128x128x18

        
        # Perform deformable convolution on intermediary feature maps from swin encoder
        f0 = self.deform_conv1(features[3], offset0) # 16x16x256
        f1 = self.deform_conv2(features[2], offset1) # 32x32x256
        f2 = self.deform_conv3(features[1], offset2) # 64x64x256
        f3 = self.deform_conv4(features[0], offset3) # 128x128x256

        # Upsample and align features
        f1 = self.up1(f0) + f1 # 32x32x256
        f2 = self.up2(f1) + f2 # 64x64x256
        f3 = self.up3(f2) + f3 # 128x128x256

        # Additional upsampling to match input size
        x = self.up4(f3) # 256x256x256
        x = self.up5(x) # 512x512x256

        # Refine feature maps
        offset5 = self.offset5(x)
        x = self.deform_conv5(x, offset5) # 512x512x256

        # Segmentation map output
        x = self.conv(x) # 512x512x19

        return x
    
class model(nn.Module):
    def __init__(self,in_channels=3, num_classes=19):
        super(model, self).__init__()
        
        self.encoder = SwinEncoder()
        self.decoder = FaPNDecoder(num_classes=num_classes)
        
    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x