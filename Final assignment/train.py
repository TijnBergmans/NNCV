"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
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
)

from torch.amp import GradScaler, autocast

from model import Model as model_module

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

class SemanticSegmentationCriterion(nn.Module):
    def __init__(self, num_classes=19, ignore_index=255, dice_weight=0.5, ce_weight=0.5, smooth=1e-6):
        """
        Combined Dice + Cross-Entropy Loss for segmentation.
        
        :param num_classes: Number of classes
        :param ignore_index: Label index to ignore in the loss computation
        :param dice_weight: Weight for Dice loss component
        :param ce_weight: Weight for Cross-Entropy loss component
        :param smooth: Smoothing factor for Dice loss to prevent division by zero
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, targets):
        """
        preds: [B, num_classes, H, W] (Raw logits, softmax will be applied)
        targets: [B, H, W] (Class indices)
        """
        B, C, H, W = preds.shape

        # Cross-Entropy Loss
        ce_loss = self.ce_loss(preds, targets)

        # Apply softmax to get class probabilities for Dice loss
        preds_softmax = F.softmax(preds, dim=1)  # [B, C, H, W]

        # Create one-hot encoding of targets, ignoring ignored pixels
        targets_onehot = F.one_hot(targets.clamp(0, self.num_classes - 1), num_classes=self.num_classes)  # [B, H, W, C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Mask out ignored pixels
        valid_mask = (targets != self.ignore_index).float()  # [B, H, W]
        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]

        # Compute Dice loss
        intersection = (preds_softmax * targets_onehot * valid_mask).sum(dim=(2, 3))  # [B, C]
        union = (preds_softmax * valid_mask).sum(dim=(2, 3)) + (targets_onehot * valid_mask).sum(dim=(2, 3))  # [B, C]
        dice_loss = 1 - ((2. * intersection + self.smooth) / (union + self.smooth)).mean()  # Scalar

        # Weighted sum of losses
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return total_loss

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

    return parser

def get_augmentation(mode='standard'):
    if mode == 'standard':
        return Compose([
            ToImage(),
            Resize((256, 256)),
            RandomHorizontalFlip(p=0.5),
            RandomResizedCrop((256, 256), scale=(0.5, 2.0)),
            ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif mode == 'lsj':
        return Compose([
            ToImage(),
            Resize((256, 256)),
            RandomHorizontalFlip(p=0.5),
            RandomResizedCrop((256, 256), scale=(0.1, 2.0)),
            ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

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
    transform = get_augmentation(args.augmentation)
    val_transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset and make a split for training and validation
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

    # Define the model
    model = model_module(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Define the loss function
    criterion = SemanticSegmentationCriterion(num_classes=19).to(device)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    # Define the learning rate scheduler
    total_steps = args.epochs * len(train_dataloader)  # ~186 steps/epoch for Cityscapes
    warmup_steps = 500  # ~2-3 epochs

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            # Warmup phase
            torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=1e-6, 
                end_factor=1.0, 
                total_iters=warmup_steps
            ),
            # Cosine decay
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=args.lr/25  # 4% of initial LR
            )
        ],
        milestones=[warmup_steps]
    )

    # Mixed precision training
    scaler = GradScaler('cuda')  # Initialize the gradient scaler

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)
        
            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            train_loss += loss.item()

            if i % 50 == 0:
                print(f"Batch: {i} Loss: {loss.item():.4f}")

                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + (i/len(train_dataloader)),
                })
            
         # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

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
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({
                "valid_loss": valid_loss
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
        

            # Early stopping check
            #if early_stopping(avg_valid_loss, model):
            #    print("Early stopping triggered")
            #    break
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
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


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)