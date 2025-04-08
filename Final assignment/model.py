import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class SwinEncoder(nn.Module):
    def __init__(self):
        super(SwinEncoder, self).__init__()

        # Load a swin-b v2 model pre-trained on ImageNet
        self.model = models.swin_v2_b(weights='IMAGENET1K_V1')

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
    def __init__(self,in_channels=[1024,512,256,128], out_channels=256, n_classes=19):
        super(FaPNDecoder, self).__init__()

        # Offset generation layers
        self.gen_offset = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch//2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(ch//2, 18, kernel_size=3, padding=1),
                # Optional to prevent exploding offsets
                nn.Tanh()
            ) for ch in in_channels
        ])

        # Deformable convolution layers for feature allignment
        self.deform_conv = nn.ModuleList([
            DeformConv2d(ch, out_channels, kernel_size=3, padding=1) for ch in in_channels
        ])

        # Upsampling path
        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for _ in range(4)
        ])

        # Final upsample
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, features):
        maps = []

        # Reverse order of features from deepest to shallowest
        features = features[::-1]

        # Take deepest feature as starting point
        x = features[0]
    
        for i in range(len(features)):
            # Generate offsets and perform deformable convolution
            offset = self.gen_offset[i](features[i])
            deformed_feat = self.deform_conv[i](features[i], offset)
        
            # Step 2: Upsample
            if i != 0:
                x = self.upsamples[i-1](x) + deformed_feat
                maps.append(x)
            else:
                x = deformed_feat  # Initialize with deepest deformed feature

        # Last upsample to produce detailed feature map for masking
        x = self.final_upsample(x)
    
        return maps, x
    
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

    def forward(self, queries, features, attn_mask=None, pos=None, query_pos=None):

        v = features
        k = features + pos
        q = queries + query_pos

        # Masked attention
        queries2 = self.mask_attn(q, k, v, attn_mask=attn_mask)[0]
        
        # Add and norm
        queries = queries + self.drop1(queries2)
        queries = self.norm1(queries)
        
        # Self attention
        q = k = queries + query_pos
        v = queries
        queries2 = self.self_attn(q, k, v)[0]

        # Add and norm
        queries = queries + self.drop2(queries2)
        queries = self.norm2(queries)

        # FFN
        queries2 = self.mlp(queries)

        # Add and norm
        queries = queries + self.drop3(queries2)
        queries = self.norm3(queries)        

        return queries
    
class TransformerDecoder(nn.Module):
    def __init__(self, n_queries=100, n_layers=6, n_heads=8, n_maps=3, embed_dim=256, n_classes=19):
        super().__init__()
        self.n_queries = n_queries
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        # Query initialization
        self.query_feat = nn.Embedding(n_queries, embed_dim)
        self.query_pos = nn.Embedding(n_queries, embed_dim)

        # Level embedding
        self.level_embed = nn.Embedding(n_maps, embed_dim)

        # Position embedding
        self.pos_embed = PositionEmbedding(embed_dim=embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim) 
            for _ in range(n_layers)
        ])
        
        # Prediction heads
        self.class_embed = nn.Linear(embed_dim, n_classes)
        self.mask_embed = MLP(embed_dim, embed_dim, embed_dim, 3)
        
        # Layer norm
        self.decoder_norm = nn.LayerNorm(embed_dim)

    def forward_prediction_heads(self, output, mask_features, target_size):
        output = self.decoder_norm(output).transpose(0,1)  # [B, N, C]
        
        # Class predictions
        class_logits = self.class_embed(output)
        
        # Mask predictions
        mask_emb = self.mask_embed(output)  # [B, N, C]
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_emb, mask_features)
        
        # Attention mask for next layer
        attn_mask = F.interpolate(mask_pred, size=target_size, mode='bilinear')
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.n_heads, 1, 1).flatten(0,1) < 0.5).bool()        
        attn_mask = attn_mask.detach()
        
        return class_logits, mask_pred, attn_mask

    def forward(self, features, mask_features):
        B = features[0].shape[0]
        
        # Initialize queries and positional encodings
        queries = self.query_feat.weight.unsqueeze(1).repeat(1,B,1)  # [N,B,C]
        query_pos = self.query_pos.weight.unsqueeze(1).repeat(1,B,1) # [N,B,C]
        
        # Prepare feature maps
        memories = []
        positions = []
        for i, feat in enumerate(features):
            # Flatten and project featuremaps
            memories.append((feat.flatten(2) + self.level_embed.weight[i][None, :, None]).permute(2,0,1))  # [H*W, B, C]
            # Flatten and add positional embedding
            positions.append(self.pos_embed(feat).flatten(2).permute(2,0,1))  # [H*W, B, C]
        
        # Initial predictions
        predictions_class = []
        predictions_mask = []
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            queries, mask_features, target_size=features[0].shape[-2:]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        # Transformer layers
        for i in range(self.n_layers):
            level_idx = i % len(features)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # Decoder layer
            queries = self.layers[i](
                queries,
                features=memories[level_idx],
                attn_mask=attn_mask,
                pos=positions[level_idx],
                query_pos=query_pos
            )
            
            # Update predictions
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                queries, mask_features,
                target_size=features[(i+1)%len(features)].shape[-2:]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            aux_class = predictions_class[:-1]
            aux_mask = predictions_mask[:-1]
            pred_logits = predictions_class[-1]
            pred_masks = predictions_mask[-1]
            
        return pred_logits, pred_masks, aux_class, aux_mask

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SegmentationHead(nn.Module):
    def __init__(self, n_classes=19):
        super(SegmentationHead, self).__init__()

        # Refinement for small objects
        self.refine = nn.Sequential(
            nn.Conv2d(n_classes, 64, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, n_classes, kernel_size = 3, padding=1)
        )
    
    def forward(self, pred_logits, pred_masks, target_size):

        # Softmax for class logits
        pred_logits = F.softmax(pred_logits, dim=1)

        # Sigmoid for mask predictions
        pred_masks = pred_masks.sigmoid()

        # Aggregate predictions to find segmentation map
        seg_map = torch.einsum('bnc,bnhw->bchw', pred_logits, pred_masks) # [B, C, H, W]

        # Refine segmentation map
        seg_map = self.refine(seg_map)

        # Upscale to target size
        seg_map = F.interpolate(seg_map, size=target_size, mode='bilinear', align_corners=False)

        return seg_map

class PositionEmbedding(nn.Module):    
    def __init__(self, embed_dim=256, temperature=10000):
        super().__init__()
        assert embed_dim % 2 == 0, "Embed dim must be even"
        self.embed_dim = embed_dim // 2
        self.temperature = temperature
        
        # Create frequency bands
        dim_t = torch.arange(self.embed_dim, dtype=torch.float32)
        self.register_buffer('dim_t', temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.embed_dim))

    def forward(self, x):
        B, _, H, W = x.shape
        
        # Create normalized grid coordinates
        y_embed = torch.arange(H, device=x.device).float().unsqueeze(1).expand(H, W) / H
        x_embed = torch.arange(W, device=x.device).float().unsqueeze(0).expand(H, W) / W
        
        # Calculate positional encodings
        pos_x = x_embed.reshape(-1, 1) / self.dim_t  # [H*W, embed_dim]
        pos_y = y_embed.reshape(-1, 1) / self.dim_t
        
        # Alternate sin/cos pattern
        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1)
        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1)
        
        # Combine and reshape
        pos = torch.cat([pos_y, pos_x], dim=1).transpose(0,1).reshape(1, self.embed_dim*2, H, W)
        return pos.expand(B, -1, -1, -1)  # [B, C, H, W]

class FeatureFusion(nn.Module):
    def __init__(self, embed_dim=256, n_features=4):
        super(FeatureFusion, self).__init__()

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(embed_dim*n_features, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

    def forward(self, maps, mask_features):
        target_size = mask_features.shape[-2:]

        upsampled_maps = torch.cat([F.interpolate(m, size=target_size, mode='bilinear', align_corners=False) for m in maps], dim=1)
        
        fused_features = torch.cat((mask_features,upsampled_maps), dim=1)

        fused_features = self.fuse_conv(fused_features)

        return fused_features

class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super(Model, self).__init__()
        
        self.encoder = SwinEncoder()
        self.pixel_decoder = FaPNDecoder()
        self.fuse_feat = FeatureFusion()
        self.transformer_decoder = TransformerDecoder()
        self.seg_head = SegmentationHead(n_classes=n_classes)
        
    def forward(self, x):

        H, W = x.shape[-2:]

        featuremaps = self.encoder(x)
        maps, mask_features = self.pixel_decoder(featuremaps)
        fused_features = self.fuse_feat(maps, mask_features)
        pred_logits, pred_masks, aux_class, aux_mask = self.transformer_decoder(maps, fused_features)

        # Process auxiliary outputs
        aux_seg_map = []
        for logits, mask in zip(aux_class, aux_mask):
            seg_map = self.seg_head(logits, mask, target_size=(H,W))
            aux_seg_map.append(seg_map)

        # Generate segmentation map
        seg_map = self.seg_head(pred_logits, pred_masks, target_size=(H, W))

        out = {
            'segmentation': seg_map,
            'aux_segmentation': aux_seg_map
        }

        return out