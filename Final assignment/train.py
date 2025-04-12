import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ColorJitter,
    GaussianBlur,
    RandomApply
)
from torch.amp import GradScaler, autocast

from model import Model as model_module

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Config:
    # Pre-training
    COARSE_EPOCHS = 30
    COARSE_LR = 3e-5
    COARSE_WARMUP = 3

    # Training
    FINE_EPOCHS = 100
    FINE_LR = 6e-5
    SWIN_LR = 3e-5
    FINE_WARMUP = 10

    # Shared
    BATCH_SIZE = 16
    WEIGHT_DECAY = 0.01
    IMG_SIZE = 512

class SemanticSegmentationCriterion(nn.Module):
    def __init__(self, model, class_weights=None, ignore_index=255, dice_weight=0.3, ce_weight=0.7, aux_weight=0.4, reg_weight=0.01, reg_loss=0, smooth=1e-5):
        super().__init__()
        
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.weight = {'ce':ce_weight, 'dice':dice_weight, 'reg':reg_weight,'aux':aux_weight}
        self.reg_loss = reg_loss

    def forward(self, out, label):
        
        seg_map = out['segmentation']

        # Process auxiliary losses if necessary
        aux_loss = 0
        if 'aux_segmentation' in out:
            for aux in out['aux_segmentation']:
                aux_loss += self.ce_loss(aux, label) * self.weight['ce'] * self.weight['aux']
                aux_loss += self.dice_loss(aux, label) * self.weight['dice'] * self.weight['aux']
        
        ce_loss = self.ce_loss(seg_map, label) * self.weight['ce']
        dice_loss = self.dice_loss(seg_map, label) * self.weight['dice']
        reg_loss = self.reg_loss * self.weight['reg']
        aux_loss = aux_loss/len(out['aux_segmentation'])
        
        # Compute losses
        loss = {
            'ce_loss': ce_loss,
            'dice_loss': dice_loss,
            'reg_loss': reg_loss,
            'aux_loss': aux_loss
        }
        
        return sum(loss.values()), loss
    
class DiceLoss(nn.Module):
    def __init__(self, n_classes=19, ignore_index=255, smooth=1e-6):
        super(DiceLoss, self).__init__()

        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, preds, targets):
        # Apply softmax to get class probabilities for Dice loss
        preds_softmax = F.softmax(preds, dim=1)  # [B, C, H, W]

        # Create one-hot encoding of targets, ignoring ignored pixels
        targets_onehot = F.one_hot(targets.clamp(0, self.n_classes - 1), num_classes=self.n_classes)  # [B, H, W, C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Mask out ignored pixels
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()  # [B, 1, H, W]

        # Compute Dice loss
        intersection = (preds_softmax * targets_onehot * valid_mask).sum(dim=(2, 3))  # [B, C]
        union = (preds_softmax * valid_mask).sum(dim=(2, 3)) + (targets_onehot * valid_mask).sum(dim=(2, 3))  # [B, C]
        dice_loss = 1 - ((2. * intersection + self.smooth) / (union + self.smooth)).mean()  # Scalar

        return dice_loss

class DiceScore(nn.Module):
    def __init__(self, n_classes=19, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        
    def forward(self, preds, targets):
        # Apply softmax to get class probabilities for Dice loss
        preds_softmax = F.softmax(preds, dim=1)  # [B, C, H, W]

        # Create one-hot encoding of targets, ignoring ignored pixels
        targets_onehot = F.one_hot(targets.clamp(0, self.n_classes - 1), num_classes=self.n_classes)  # [B, H, W, C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Mask out ignored pixels
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()  # [B, 1, H, W]

        # Compute Dice loss
        intersection = (preds_softmax * targets_onehot * valid_mask).sum(dim=(2, 3))  # [B, C]
        union = (preds_softmax * valid_mask).sum(dim=(2, 3)) + (targets_onehot * valid_mask).sum(dim=(2, 3))  # [B, C]
        dice = ((2. * intersection + self.smooth) / (union + self.smooth)).mean()  # Scalar

        return dice

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        else:
            self.best_score = score
            self.counter = 0
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
    
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
        
# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image

def compute_class_weights(dataset, epsilon=1e-7):
    
    class_counts = torch.zeros(19)

    for _, label in dataset:
        # Convert labels to train ID's and count
        label = convert_to_train_id(label)
        counts = torch.bincount(label.flatten(), minlength=20)[:19]
        class_counts += counts

    # Add epsilon to avoid division by 0
    class_counts = class_counts + epsilon

    # Compute frequency
    frequency = class_counts / class_counts.sum()

    # Invert and root to compute class weight and normalize
    weights = 1/(frequency**0.5)
    weights = weights / weights.mean()

    return weights

def get_class_weights(train_dataset, weights_path='cityscapes_weights.pt', force_recompute=False):

    if not force_recompute and os.path.exists(weights_path):
        print(f"Loading precomputed class weights from {weights_path}")
        weights = torch.load(weights_path)
    else:
        print("Computing class weights from training set...")
        weights = compute_class_weights(train_dataset)
        torch.save(weights, weights_path)
        print(f"Saved class weights to {weights_path}")
    
    return weights

def get_args_parser():

    parser = ArgumentParser("Training script for a Mask2Former model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="mask2former training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--augmentation", type=str, default="standard", choices=["standard", "lsj"], help="Augmentation stragety")
    parser.add_argument("--pre-train", type=int, default=0, help="Pre-training")

    return parser

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        if "offset" in str(m):  # Special case for deformable conv offsets
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):  # Query embeddings
        nn.init.uniform_(m.weight, -0.08, 0.08)

def main(args):

    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Empty CUDA cache
    torch.cuda.empty_cache()

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data

    coarse_transform = Compose([
        ToImage(),
        Resize((512,512)),
        RandomHorizontalFlip(p=0.5),
        RandomResizedCrop(
            size=(512, 512),
            scale=(0.3, 1.0),
            ratio=(0.8, 1.25)),
        RandomApply([
            ColorJitter(
                brightness=0.3,
                contrast=0.3,
                hue=0.1)],
            p=0.5
        ),
        RandomApply([
            GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.0)
            )],
            p=0.1
        ),
        ToDtype(torch.float32, scale=True),
        Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
        )
    ])

    transform = Compose([
            ToImage(),
            Resize((512, 512)),
            RandomHorizontalFlip(p=0.5),
            RandomResizedCrop(
                size=(512, 512), 
                scale=(0.1, 2.0),
                ratio=(0.5, 2.0)),
            RandomApply([
                ColorJitter(
                    brightness=0.4, 
                    contrast=0.4, 
                    saturation=0.4, 
                    hue=0.15)],
                p=0.8
            ),
            RandomApply([
                GaussianBlur(
                    kernel_size=(5, 9),
                    sigma=(0.1, 2.0)
                )],
                p=0.3
            ),
            ToDtype(torch.float32, scale=True),
            Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
        ])

    val_transform = Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset and make a manual split for training and validation
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=val_transform
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    if args.pre_train == 1:
        coarse_dataset_full = Cityscapes(
            args.data_dir,
            split="train_extra",
            mode="coarse",
            target_type="semantic",
            transforms=coarse_transform
        )

        coarse_dataset_full = wrap_dataset_for_transforms_v2(coarse_dataset_full)

        dataset_size = len(coarse_dataset_full)
        train_size = int(dataset_size * 0.95)
        val_size = dataset_size - train_size

        coarse_train, coarse_val = random_split(
            coarse_dataset_full, 
            [train_size, val_size], 
            generator=torch.Generator().manual_seed(args.seed)
        )

        coarse_dataloader = DataLoader(
            coarse_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        coarse_val_dataloader = DataLoader(
            coarse_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    # Define the model
    model = model_module(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Make sure all tensors are pushed to the same device
    for name, param in model.named_parameters():
        param.data = param.data.to(device)
    for name, buffer in model.named_buffers():
        buffer.data = buffer.data.to(device)

    # Initialize weights
    model.apply(init_weights)

    # Carefully handle deformable Convs
    for m in model.modules():
        if isinstance(m, DeformConv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)

    # Make sure to initialize offset generation layers to 0
    for layer in model.pixel_decoder.gen_offset:
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # Load class weights
    class_weights = get_class_weights(
        train_dataset=train_dataset,
        weights_path=os.path.join(
                    output_dir, 
                    "cityscapes_class_weights.pth"
                ),
        force_recompute=False    
    ).to(device)

    # Define Dice metric
    dice_metric = DiceScore().to(device)

    # Initialize EMA
    ema = EMA(model, decay=0.999)

    # Initialize gradient scaler
    scaler = GradScaler('cuda')

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)

    # --- Pre Training ---

    if args.pre_train == 1:

        # Give less weight to DICE loss due to label noise
        ce_weight = 0.9
        dice_weight = 0.1
        aux_weight = 0.2
        
        # Define the loss function
        criterion = SemanticSegmentationCriterion(
            model=model,
            class_weights=class_weights, 
            ce_weight=ce_weight, 
            dice_weight=dice_weight,
            aux_weight=aux_weight,
            reg_loss=model.pixel_decoder.offset_reg_loss
        ).to(device)

        # Freeze Swin
        for param in model.encoder.parameters():
            param.requires_grad = False

        # Freeze deformable convolutions to regular convolutions
        for param in model.pixel_decoder.gen_offset.parameters():
            param.requires_grad = False

        # Set LR for other modules
        optimizer = AdamW([
            {'params': model.pixel_decoder.parameters(), 'lr': Config.COARSE_LR},
            {'params': model.transformer_decoder.parameters(), 'lr': Config.COARSE_LR},
            {'params': model.fuse_feat.parameters(), 'lr': Config.COARSE_LR},
            {'params': model.seg_head.parameters(), 'lr': Config.COARSE_LR}
            ],
            weight_decay=Config.WEIGHT_DECAY
        )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-6,
                    end_factor=1.0,
                    total_iters=Config.COARSE_WARMUP*len(coarse_dataloader)
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=(Config.COARSE_EPOCHS - Config.COARSE_WARMUP)*len(coarse_dataloader),
                    eta_min=1e-6
                )
            ],
            milestones=[Config.COARSE_WARMUP*len(coarse_dataloader)]
        )

        checkpoints = 0.0

        print("Starting pre-training...")

        for epoch in range(Config.COARSE_EPOCHS):
            print(f"Epoch {epoch+1:04}/{Config.COARSE_EPOCHS:04}")
            
            model.train()

            for i, (images, labels) in enumerate(coarse_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)
            
                optimizer.zero_grad()

                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss, _ = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    norm_type=2.0
                )
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                wandb.log({
                    "pre_train_loss": loss.item(),
                    "pre_train_learning_rate": optimizer.param_groups[0]['lr'],
                    "pre_train_epoch": epoch + 1
                    }, 
                    step = epoch * len(coarse_dataloader) + i
                )

            # Validation
            model.eval()
            with torch.no_grad():
                coarse_losses = []
                for i, (images, labels) in enumerate(coarse_val_dataloader):

                    labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                    images, labels = images.to(device), labels.to(device)

                    labels = labels.long().squeeze(1)  # Remove channel dimension

                    outputs = model(images)
                    loss, _ = criterion(outputs, labels)
                    dice = dice_metric(outputs['segmentation'], labels)
                    coarse_losses.append(loss.item())

                    outputs = outputs['segmentation']
                
                    if i == 0:
                        predictions = outputs.softmax(1).argmax(1)

                        predictions = predictions.unsqueeze(1)
                        labels = labels.unsqueeze(1)

                        predictions = convert_train_id_to_color(predictions)
                        labels = convert_train_id_to_color(labels)

                        predictions_img = make_grid(predictions.cpu(), nrow=8)
                        labels_img = make_grid(labels.cpu(), nrow=8)

                        predictions_img = predictions_img.permute(1, 2, 0).numpy()
                        labels_img = labels_img.permute(1, 2, 0).numpy()

                        wandb.log({
                            "pre_train_predictions": [wandb.Image(predictions_img)],
                            "pre_train_labels": [wandb.Image(labels_img)],
                        }, step=(epoch + 1) * len(coarse_dataloader) - 1)
                
                coarse_valid_loss = sum(coarse_losses) / len(coarse_losses)
                wandb.log({
                    "pre_train_valid_loss": coarse_valid_loss,
                    "DICE": dice
                }, step=(epoch + 1) * len(coarse_dataloader) - 1)

                if (epoch + 1) % 5 == 0:
                    checkpoint_path = os.path.join(output_dir, f"pretrained_checkpoint_epoch_{epoch + 1}.pth")
                    torch.save(model.state_dict(), checkpoint_path)
                    checkpoints = checkpoints + 1
                    wandb.log({
                        "checkpoints": checkpoints
                        }, 
                        step=(epoch + 1) * len(coarse_dataloader) - 1)


                # Early stopping check
                if early_stopping(coarse_valid_loss, model):
                    print("Early stopping triggered")
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            output_dir,
                            "pre_trained_model.pth"
                        )
                    )
                    break
        
        # Save final model
        torch.save(
            model.state_dict(),
            os.path.join(
                output_dir,
                "pre_trained_model.pth"
            )
        )
        
        print("Pre-training finished!")

    pre_train_path = os.path.join(output_dir,"pre_trained_model.pth")

    # If the model is not pre trained, see if pre-trained weights are available
    if args.pre_train == 0 and os.path.exists(pre_train_path):
        print(f"Loading precomputed class weights from {pre_train_path}")
        weights = torch.load(pre_train_path, map_location=device)
        model.load_state_dict(weights)
        print("Pre-trained weights loaded")

    # --- Training ---

    print("Starting training...")

    checkpoints = 0.0

    # Give more weight to DICE loss for fine-tuning
    ce_weight = 0.5
    dice_weight = 0.5
    aux_weight = 0.3
        
    # Define the loss function
    criterion = SemanticSegmentationCriterion(
        model=model,
        class_weights=class_weights, 
        ce_weight=ce_weight, 
        dice_weight=dice_weight,
        aux_weight=aux_weight,
        reg_loss=model.pixel_decoder.offset_reg_loss
    ).to(device)

    # Set new LR
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': Config.SWIN_LR},
        {'params': model.pixel_decoder.parameters(), 'lr': Config.FINE_LR},
        {'params': model.transformer_decoder.parameters(), 'lr': Config.FINE_LR},
        {'params': model.fuse_feat.parameters(), 'lr': Config.FINE_LR},
        {'params': model.seg_head.parameters(), 'lr': Config.FINE_LR},
        ],
        weight_decay=Config.WEIGHT_DECAY    
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-6,
                    end_factor=1.0,
                    total_iters=Config.FINE_WARMUP*len(train_dataloader)
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=(Config.FINE_EPOCHS - Config.FINE_WARMUP)*len(train_dataloader),
                    eta_min=1e-6
                )
            ],
            milestones=[Config.FINE_WARMUP*len(train_dataloader)]
        )
    
    # Unfreeze Swin encoder
    for param in model.encoder.parameters():
            param.requires_grad = True

    best_valid_loss = float('inf')
    best_valid_loss_ema = float('inf')
    current_best_model_path = None
    current_best_model_ema_path = None

    for epoch in range(args.epochs):
        
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Unfreeze offset generators after 5 epochs
        if epoch == 5:
            # Re-initialize offset generation layers
            with torch.no_grad():
                for layer in model.pixel_decoder.gen_offset:
                    for m in layer.modules():
                        if isinstance(m, nn.Conv2d):
                            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0.0)

            # Unfreeze deformable convolutions after 5 epochs
            for param in model.pixel_decoder.gen_offset.parameters():
                param.requires_grad = True
            
            optimizer.add_param_group({
                'params': [p for n,p in model.named_parameters() if 'gen_offset' in n],
                'lr': Config.FINE_LR * 0.1,
                'weight_decay': Config.WEIGHT_DECAY
            })

        # Training
        model.train()

        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)
        
            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, labels)
                dice = dice_metric(outputs['segmentation'], labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                norm_type=2.0
            )
            scaler.step(optimizer)
            scaler.update()

            ema.update()
            lr_scheduler.step()

            wandb.log({
                "train_loss": loss.item(),
                "losses": loss_dict,
                "DICE": dice,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1
                }, 
                step = epoch * len(train_dataloader) + i
            )
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses_ema = []
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                # Apply EMA
                ema.apply_shadow()
                outputs = model(images)
                loss, _ = criterion(outputs, labels)
                dice_ema = dice_metric(outputs['segmentation'], labels)
                losses_ema.append(loss.item())

                # Restore EMA
                ema.restore()
                outputs = model(images)
                loss, _ = criterion(outputs, labels)
                dice = dice_metric(outputs['segmentation'], labels)
                losses.append(loss.item())

                dice_val = {
                    "val_dice": dice,
                    "val_dice_ema": dice_ema
                }

                outputs = outputs['segmentation']
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss_ema = sum(losses_ema) / len(losses_ema)
            valid_loss = sum(losses) / len(losses)
            wandb.log({
                "valid_loss": valid_loss,
                "valid_loss_ema": valid_loss_ema,
                "DICE": dice_val
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            # Checkpointing

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)

            if valid_loss_ema < best_valid_loss_ema:
                best_valid_loss_ema = valid_loss_ema
                if current_best_model_ema_path:
                    os.remove(current_best_model_ema_path)
                current_best_model_ema_path = os.path.join(
                    output_dir, 
                    f"best_model_ema-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(ema.shadow, current_best_model_ema_path)
            
            # Save training state once every 5 epochs
            if (epoch + 1) % 5 == 0:
                    checkpoint_path = os.path.join(output_dir, f"train_checkpoint_epoch_{epoch + 1}.pth")
                    
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'checkpoints': checkpoints
                        }, 
                        checkpoint_path
                    )
                    
                    checkpoints = checkpoints + 1
                    
                    wandb.log({
                        "checkpoints": checkpoints
                        }, 
                        step=(epoch + 1) * len(train_dataloader) - 1)

            # Early stopping check
            if early_stopping(valid_loss, model):
                print("Early stopping triggered")
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        output_dir,
                        f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                    )
                )
                torch.save(
                    ema.shadow,
                    os.path.join(
                        output_dir,
                        f"final_model_ema-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                    )
                )
                break
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    torch.save(
        ema.shadow,
        os.path.join(
            output_dir,
            f"final_model_ema-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )

    wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)