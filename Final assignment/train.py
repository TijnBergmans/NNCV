import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from collections import defaultdict
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
    COARSE_EPOCHS = 8
    COARSE_LR = 3e-4
    COARSE_WARMUP = 500

    # Training
    FINE_EPOCHS = 100
    FINE_LR = 6e-5
    SWIN_LR = 3e-5
    FINE_WARMUP = 1000

    # Shared
    BATCH_SIZE = 16
    WEIGHT_DECAY = 0.05
    IMG_SIZE = 512

class SemanticSegmentationCriterion(nn.Module):
    def __init__(self, class_weights, ignore_index=255, dice_weight=0.3, ce_weight=0.7, aux_weight=0.4, smooth=1e-5):
        super().__init__()
        
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.weight = {'ce':ce_weight, 'dice':dice_weight, 'aux':aux_weight}

    def forward(self, out, label):
        
        seg_map = out['segmentation']

        # Process auxiliary losses if necessary
        aux_loss = 0
        if 'aux_segmentation' in out:
            for aux in out['aux_segmentation']:
                aux_loss += self.ce_loss(aux, label) * self.weight['ce'] * self.weight['aux']
                aux_loss += self.dice_loss(aux, label) * self.weight['dice'] * self.weight['aux']
            aux_loss = aux_loss/len(out['aux_segmentation'])
            
        ce_loss = self.ce_loss(seg_map, label) * self.weight['ce']
        dice_loss = self.dice_loss(seg_map, label) * self.weight['dice']
        
        # Compute losses
        loss = {
            'ce_loss': ce_loss,
            'dice_loss': dice_loss,
            'aux_loss': aux_loss
        }
        
        return sum(loss.values()), loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes=19, ignore_index=255, smooth=1e-5):
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
        valid_mask = (targets != self.ignore_index).float()  # [B, H, W]
        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]

        # Compute Dice loss
        intersection = (preds_softmax * targets_onehot * valid_mask).sum(dim=(2, 3))  # [B, C]
        union = (preds_softmax * valid_mask).sum(dim=(2, 3)) + (targets_onehot * valid_mask).sum(dim=(2, 3))  # [B, C]
        dice_loss = 1 - ((2. * intersection + self.smooth) / (union + self.smooth)).mean()  # Scalar

        return dice_loss

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class EMA:
    def __init__(self, model, decay=0.9995):
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

    # Invert to compute class weight and normalize
    weights = 1/frequency
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
    parser.add_argument("--pre-train", type=bool, default=False, help="Pre-training")

    return parser

def get_scheduler(optimizer, total_steps, warmup_steps):
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-6,
                    end_factor=1.0,
                    total_itters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    total_steps - warmup_steps
                )
        ],
        milestones=[warmup_steps]
    )

    return scheduler

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

    # Load the dataset and make a split for training and validation
    coarse_dataset = Cityscapes(
        args.data_dir,
        split="train_extra",
        mode="coarse",
        target_type="semantic",
        transforms=coarse_transform
    )

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

    coarse_dataset = wrap_dataset_for_transforms_v2(coarse_dataset)
    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    coarse_dataloader = DataLoader(
        coarse_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

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

    # Define the model
    model = model_module(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Load class weights
    class_weights = get_class_weights(
        train_dataset=train_dataset,
        weights_path='./weights/cityscapes_weights.pt',
        force_recompute=False    
    ).to(device)

    # Define the loss function
    criterion = SemanticSegmentationCriterion(class_weights=class_weights).to(device)

    # Define the optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.05
    )

    # Initialize EMA
    ema = EMA(model, decay=0.9995)

    # Initialize gradient scaler
    scaler = GradScaler('cuda')

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)

    # --- Pre Training ---
    if args.pre_train:

        # Freeze Swin
        for param in model.encoder.parameters():
            param.requires_grad = False

        # Freeze Positional embedding
        for param in model.transformer_decoder.pos_embed.parameters():
            param.requires_grad = False

        # Set higher LR for other modules
        optimizer = AdamW([
            {'params': model.pixel_decoder.parameters(), 'lr': Config.COARSE_LR},
            {'params': model.transformer_decoder.parameters(), 'lr': Config.COARSE_LR},
            {'params': model.seg_head.parameters(), 'lr': Config.COARSE_LR},

            {'params': model.transformer_decoder.query_feat.parameters(), 'lr': 1e-4},
            {'params': model.transformer_decoder.query_pos.parameters(), 'lr': 1e-4},
            {'params': model.transformer_decoder.level_embed.parameters(), 'lr': 1e-4},
            ],
            weight_decay=Config.WEIGHT_DECAY
        )

        lr_scheduler = get_scheduler(
            optimizer,
            total_steps=Config.COARSE_EPOCHS*len(coarse_dataloader),
            warmup_steps=Config.COARSE_WARMUP
        )

        print("Starting pre-training...")

        for epoch in range(Config.COARSE_EPOCHS):
            print(f"Epoch {epoch+1:04}/{Config.COARSE_EPOCHS:04}")
            
            model.train()

            for i, (images, labels) in enumerate(coarse_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)

                # Label smoothing
                epsilon = 0.1
                labels = (1-epsilon)*labels + epsilon/19
            
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
                    step = epoch * len(train_dataloader) + i
                )

        # Save pre-trained model

        # Save the model
        torch.save(
            model.state_dict(),
            os.path.join(
                output_dir,
                f"pre_trained_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
            )
    )

        print("Pre-training finished!")

    # --- Training ---

    print("Starting training...")

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    # Set new LR
    
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': Config.SWIN_LR},
        {'params': model.pixel_decoder.parameters(), 'lr': Config.FINE_LR},
        {'params': model.transformer_decoder.parameters(), 'lr': Config.FINE_LR},
        {'params': model.seg_head.parameters(), 'lr': Config.FINE_LR},
        ],
        weight_decay=Config.WEIGHT_DECAY    
    )

    lr_scheduler = get_scheduler(
        optimizer,
        total_steps=Config.FINE_EPOCHS*len(train_dataloader),
        warmup_steps=Config.FINE_WARMUP
    )

    best_valid_loss = float('inf')
    best_valid_loss_ema = float('inf')
    current_best_model_path = None
    current_best_model_ema_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

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
                "ce_loss": loss_dict['ce_loss'].item(),
                "dice_loss": loss_dict['dice_loss'].item(),
                "aux_loss": loss_dict['aux_loss'].item(),
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
                losses_ema.append(loss.item())

                # Restore EMA
                ema.restore()
                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
            
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
                "valid_loss_ema": valid_loss_ema
            }, step=(epoch + 1) * len(train_dataloader) - 1)

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