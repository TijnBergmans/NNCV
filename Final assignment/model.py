import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

# Third implementation of the model
# Swin-T v2 as encoder and FaPN and Transformer decoder as decoder

class SwinEncoder(nn.Module):
    def __init__(self):
        super(SwinEncoder, self).__init__()

        # Load a swin-b v2 model pre-trained on ImageNet
        self.model = models.swin_v2_t(weights='IMAGENET1K_V1')

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

        return featuremaps
    
class PositionEmbedding(nn.Module):
    def __init__(self, embed_dim=256, max_size=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_size = max_size
        
        # Save the positional encoding
        self.register_buffer('pos_embed', self.create_pos_embed())
    
    def create_pos_embed(self):
        half_dim = self.embed_dim // 4
        
        grid_y = torch.arange(self.max_size, dtype=torch.float32)
        grid_x = torch.arange(self.max_size, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        
        freq_bands = (1.0 / (10000 ** (torch.arange(half_dim, dtype=torch.float32) / half_dim)))
        print("Frequency bands shape: ", freq_bands.shape)
        
        x_enc = grid_x.unsqueeze(-1) * freq_bands.view(1, 1, -1)
        print("X encoding shape: ", x_enc.shape)
        y_enc = grid_y.unsqueeze(-1) * freq_bands.view(1, 1, -1)
        print("Y encoding shape: ", y_enc.shape)
        
        pos_enc = torch.cat([
            torch.sin(x_enc), torch.cos(x_enc),
            torch.sin(y_enc), torch.cos(y_enc)
        ], dim=-1)
        print("Positional encoding shape: ", pos_enc.shape)
        
        return pos_enc.permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):

        # Get the positional encoding for the input shape
        _, _, H, W = x.shape
        return self.pos_embed[:, :, :H, :W]
    
class FaPNDecoder(nn.Module):
    def __init__(self,in_channels=[768,384,192,96], out_channels=256, n_classes=19):
        super(FaPNDecoder, self).__init__()

        # Offset generation layers
        self.gen_offset1 = nn.Conv2d(in_channels[0], 18, kernel_size=3, padding=1)
        self.gen_offset2 = nn.Conv2d(in_channels[1], 18, kernel_size=3, padding=1)
        self.gen_offset3 = nn.Conv2d(in_channels[2], 18, kernel_size=3, padding=1)
        self.gen_offset4 = nn.Conv2d(in_channels[3], 18, kernel_size=3, padding=1)

        # Deformable convolution layers for feature allignment
        self.deform_conv1 = DeformConv2d(in_channels[0], out_channels, kernel_size=3, padding=1)
        self.deform_conv2 = DeformConv2d(in_channels[1], out_channels, kernel_size=3, padding=1)
        self.deform_conv3 = DeformConv2d(in_channels[2], out_channels, kernel_size=3, padding=1)
        self.deform_conv4 = DeformConv2d(in_channels[3], out_channels, kernel_size=3, padding=1)

        # Upsampling path (assuming 512x512 input --> 16x16x768 after swin encoder)
        self.up1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

        # Additional deformable convolution for refinement + offset
        self.offset5 = nn.Conv2d(out_channels, 18, kernel_size=3, padding=1)
        self.deform_conv5 = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Final convolution layer
        self.conv = nn.Conv2d(out_channels, n_classes, kernel_size=1)

        # Positional embedding
        self.pos_embed = PositionEmbedding(embed_dim=out_channels, max_size=512)

    def forward(self, features):

        maps = []

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
        maps.append(f1)
        f2 = self.up2(f1) + f2 # 64x64x256
        maps.append(f2)
        f3 = self.up3(f2) + f3 # 128x128x256
        maps.append(f3)
        x = self.up4(f3) # 256x256x256

        # Add positional embedding
        x = x + self.pos_embed(x) # 256x256x256

        # Upsample to original image size
        x = self.up5(x) # 512x512x256

        # Refine feature maps
        offset5 = self.offset5(x)
        x = self.deform_conv5(x, offset5) # 512x512x256

        return x, maps

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, dropout=0.1, mlp_ratio=4, H=64, W=64):
        super(TransformerDecoderLayer, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        # Masked attention with mask guide
        self.mask_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Self attention
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio*embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio*embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        # Dropout layers
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

        # Soft masking layers
        self.temperature = nn.Parameter(torch.ones(1))
        self.mask_head = nn.Linear(embed_dim, 1)

    def forward(self, queries, features, prev_mask=None):
        N, B, _ = queries.shape 
        H = W = int(features.shape[0]**0.5)

        # Prepare mask
        attn_mask = None

        if prev_mask is not None:
            attn_mask = -self.temperature * (1 - prev_mask.sigmoid())
            attn_mask = attn_mask.flatten(2)
            attn_mask = attn_mask.repeat(self.n_heads, 1, 1)

        v = k = features
        q = queries

        # Masked attention
        queries2 = self.mask_attn(q, k, v, attn_mask=attn_mask)[0]
        
        # Add and norm
        queries = queries + self.drop1(queries2)
        queries = self.norm1(queries)
        
        # Self attention
        queries2 = self.self_attn(queries, features, features)[0]

        # Add and norm
        queries = queries + self.drop2(queries2)
        queries = self.norm2(queries)

        # FFN
        queries2 = self.mlp(queries)

        # Add and norm
        queries = queries + self.drop3(queries2)
        queries = self.norm3(queries)

        # Soft mask prediction
        mask_logits = self.mask_head(queries.transpose(0, 1))
        mask_pred = mask_logits.view(B, N, 1, 1).expand(-1, -1, H, W)

        return queries, mask_pred
    
class TransFormerDecoderBlock(nn.Module):
    def __init__(self, n_queries=100, embed_dim=256, n_classes=19):
        super(TransFormerDecoderBlock, self).__init__()

        self.n_queries = n_queries

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim=embed_dim) for _ in range(3)
        ])

        # Learnable query embeddings
        self.query = nn.Embedding(n_queries, embed_dim)
        self.query_pos = nn.Parameter(torch.zeros(n_queries, embed_dim))
        nn.init.normal_(self.query_pos, std=0.02)

        # Output projections
        self.class_pred = nn.Linear(embed_dim, n_classes)
        self.mask_pred = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, features):

        B = features[0].shape[0]
        # Prepare queries
        queries = self.query.weight + self.query_pos
        queries = queries.unsqueeze(1).repeat(1, B, 1)

        # Pass through transformer decoder layers
        mask_pred = []
        for layer_idx, layer in enumerate(self.layers):

            feat = features[layer_idx]
            H, W = feat.shape[-2:]

            memory = feat.flatten(2).permute(2,0,1)
            
            # Prepare mask
            prev_mask = mask_pred[-1] if mask_pred else None
            if prev_mask is not None:
                prev_mask = F.interpolate(
                    prev_mask,
                    size=(H, W),
                    mode='bilinear'
                ).squeeze(1)
            
            # Pass through decoder layer
            queries, mask_logits = layer(queries, memory, prev_mask=prev_mask)
            mask_pred.append(mask_logits)

        # Predictions
        class_logits = self.class_pred(queries.transpose(0,1))
        mask_embedding = self.mask_pred(queries.transpose(0,1))

        return class_logits, mask_embedding, mask_pred
    
class SemanticSegmentationHead(nn.Module):
    def __init__(self, in_channels=256, n_classes=19):
        super(SemanticSegmentationHead, self).__init__()

        self.mask_embed = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

        # Final classification layer
        self.class_embed = nn.Linear(in_channels, n_classes)

    def forward(self, pixel_decoder_features, mask_embeds, class_logits):

        mask_embeds = self.mask_embed(mask_embeds)

        # Compute mask predictions
        masks = torch.einsum('bnc,bchw->bnhw', mask_embeds, pixel_decoder_features)

        # Compute class predictions
        class_pred = F.softmax(class_logits, dim=-1)

        # Compute final segmentation map
        seg_map = torch.einsum('bnhw,bnc->bchw', masks, class_pred)

        return seg_map

class Model(nn.Module):
    def __init__(self,in_channels=3, n_classes=19):
        super(Model, self).__init__()
        
        self.encoder = SwinEncoder()
        self.pixel_decoder = FaPNDecoder(n_classes=n_classes)
        self.transformer_decoder = TransFormerDecoderBlock()
        self.segmentation_head = SemanticSegmentationHead(n_classes=n_classes)
        
    def forward(self, x):

        features = self.encoder(x)
        pixel_decoder_features, memory = self.pixel_decoder(features)

        class_logits, mask_pred, mask_logits = self.transformer_decoder(memory)

        seg_map = self.segmentation_head(pixel_decoder_features, mask_pred, class_logits)

        return seg_map