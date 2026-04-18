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

### Regular imports - Start - ###
import os
from argparse import ArgumentParser
### Regular imports - End - ###


### ML Specific imports - Start - ###
import wandb
import torch
import torch.nn as nn
from torch.optim import (AdamW,RMSprop)
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
    RandomHorizontalFlip
)

import segmentation_models_pytorch as smp
from torchvision.transforms.v2 import functional as F
import random
from codecarbon import EmissionsTracker #CodeCarbon Estimator
from thop import profile #FLOS estimator
### ML Specific imports - End - ###

### Model Import - Start - ###
from model import get_model
### Model Import - End - ###
#TODO Improve the looping mechanism once you have reach a super optimal zone;

# 1. Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes} # Creates a dictionary of mapped classes through dict-comprehension
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    """Maps raw Cityscapes class IDs to the 19 standard training IDs used for evaluation, setting ignored classes to 255."""
    return label_img.apply_(lambda x: id_to_trainid[x])

# 2. Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    """
    Builds a Paint-by-Number map. Acts as a sort of engine painter
    e.g.,
    The Logic: It looks at the official Cityscapes class list and says: "If the ID is 0 (Road), use the color Purple (128, 64, 128). If it's 13 (Car), use Blue (0, 0, 142)."
    The Catch: Since we mapped all the "junk" classes to 255 earlier, it explicitly tells the code: "Anything labeled 255 should be painted Black (0, 0, 0)."
    """

    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image

def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="unet-training")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--lrs", nargs=2, type=float, default=[1e-5, 3e-4], help='Differential LRs: first backbone, second for head')
    # Select criteria by name
    parser.add_argument("--criterion1", type=str, default="DiceLoss",choices=["CrossEntropyLoss", "DiceLoss", "FocalLoss"],help="Primary loss")
    parser.add_argument("--criterion2", type=str, default="FocalLoss",choices=["CrossEntropyLoss", "DiceLoss", "FocalLoss"],help="Secondary loss")
    # Criteria utilities
    parser.add_argument("--label-smoothing", type=float, default=0.0,help="Only used by CrossEntropyLoss")
    parser.add_argument("--focal-gamma", type=float, default=2.0,help="Only used by FocalLoss (SMP)")
    parser.add_argument("--focal-alpha", type=float, default=None,help="Only used by FocalLoss (SMP)")
    parser.add_argument("--dice-smooth", type=float, default=0.0, help="Only used by DiceLoss (SMP)")

    return parser

def make_criterion(name: str, args):

    name = name.strip()
    if name == "CrossEntropyLoss":
        # ignore_index masks out pixels equal to that value (no loss/grad contribution). [1](https://gist.github.com/ivechan/806faa4193c00ed41971c7f6878b4eca)[2](https://segmentation-models-pytorch.readthedocs.io/en/latest/_modules/segmentation_models_pytorch/losses/dice.html)
        return nn.CrossEntropyLoss(
            ignore_index=args.ignore_index,
            label_smoothing=args.label_smoothing,
        )

    elif name == "DiceLoss":
        # DiceLoss supports ignore_index and from_logits=True for raw logits. [3](https://stackoverflow.com/questions/73135768/how-to-use-ignore-index-in-torch-nn-crossentropyloss)[4](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
        return smp.losses.DiceLoss(
            mode="multiclass",
            ignore_index=args.ignore_index,
            from_logits=True,
            smooth=args.dice_smooth,
        )

    elif name == "FocalLoss":
        # FocalLoss supports ignore_index; implementation assumes logits by default. [5](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)[3](https://stackoverflow.com/questions/73135768/how-to-use-ignore-index-in-torch-nn-crossentropyloss)
        return smp.losses.FocalLoss(
            mode="multiclass",
            ignore_index=args.ignore_index,
            gamma=args.focal_gamma,
            alpha=args.focal_alpha)
    else:
        raise ValueError(f"Unknown criterion: {name}")

class CityscapesPipeline:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, image, target):
        # 1. Resize (Bilinear for image, Nearest for mask to avoid blending class IDs)
        image = F.resize(image, (299, 299), interpolation=InterpolationMode.BILINEAR)
        target = F.resize(target, (299, 299), interpolation=InterpolationMode.NEAREST)

        # 2. Synchronized Sample-Level Augmentation
        # Rolls the dice for each individual image, keeping image and mask perfectly aligned
        if self.is_train and random.random() > 0.5:
            image = F.horizontal_flip(image)
            target = F.horizontal_flip(target)

        # 3. Format and Normalize
        image = F.to_dtype(F.to_image(image), torch.float32, scale=True)
        image = F.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        target = F.to_dtype(F.to_image(target), torch.int64)

        return image, target

def main(args):

    # Initializinh external energy and performance related trackers
    # 1.1 CodeCarbon Initialization
    tracker = EmissionsTracker(
        project_name = "NNCV",
        measure_power_secs = 15, # How often it pings the API
        api_key="cpt_Yhs9XqbNLtS-J-L7Fig8fuaySZQXyEcbp7ipCnumPHg",
        save_to_api = True,
        experiment_id="a1ac9d91-e078-4104-a93e-df86c1b25b90"
    )
    tracker.start()
    
    # 1.2 W&B initialization
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

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

    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=CityscapesPipeline(is_train=True) # Uses the joint pipeline
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transforms=CityscapesPipeline(is_train=False) # Skips the flip for validation
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


    # 1. Initialize the model and activate CUDA
    Model = get_model()
    Model = Model.to(device)

    # 2. Group Parameters for Differential Learning Rates
    backbone_params = Model.encoder.parameters()
    head_params = list(Model.decoder.parameters()) + list(Model.segmentation_head.parameters())

    # Define the loss function
    criterion1 = make_criterion(args.criterion1, args).to(device)
    criterion2 = make_criterion(args.criterion2, args).to(device)
    print("Using criterion 1 {} and criterion 2 {}".format(criterion1, criterion2))

    # Define the optimizer
    optimizer = RMSprop([{'params':backbone_params,'lr':args.lrs[0]},{'params':head_params,'lr':args.lrs[1]}], alpha=0.9, momentum=0.9, weight_decay=1e-5, eps=0.001)
    #optimizer = AdamW([{'params':backbone_params,'lr':args.lrs[0]},{'params':head_params, 'lr':args.lrs[1]}],weight_decay=1e-5,eps=0.001)

    # Define the Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # ---------------------------------
    # Determining number of FLOPs by feeding the model with a dummy image
    print("Calculating Model FLOPs...")
    dummy_input = torch.randn(1, 3, 299, 299).to(device)
    macs, params = profile(Model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2  # MACs (Multiply-Accumulate) are roughly half a FLOP
    gflops = flops / 1e9 # Convert to GigaFLOPs for readability
    print(f"Model GFLOPs: {gflops:.3f}")
    # Define an independent Dice function just for observation
    observation_dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=255, from_logits=True)
    # Threshold
    BASELINE_DICE_SCORE:float = 0.65
    # ---------------------------------
    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        Model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            # 1. Map IDs first (Vectorized is best here)
            labels = convert_to_train_id(labels) 

            # 2. Move to device and format dimensions
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)  # Remove channel dimension

            # 3. Forward and Backward Pass
            optimizer.zero_grad()
            outputs = Model(images)
            loss = ((0.5*criterion2(outputs, labels)) + (0.5*criterion1(outputs, labels)))
            loss.backward()
            optimizer.step()

            # 4. Logging
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
      	# Validation
        Model.eval()
        with torch.no_grad():
            losses:list = []
            dice_scores:list = []

            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = Model(images)
                loss = ((0.5*criterion2(outputs, labels)) + (0.5*criterion1(outputs,labels)))
                losses.append(loss.item())

                # ---------- CALCULATING EMPIRICAL DICE OBSERVATION ------------ #
                batch_dice_score = 1.0 - observation_dice(outputs, labels).item()
                dice_scores.append(batch_dice_score)
                # ---------------------------------------------------------------#

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

            #  ------------ Logging Dice/Gflops metric ---------------- #
            # Calculating differential loss over v. set
            valid_loss = sum(losses) / len(losses)
            # Calculating empirical dice loss
            valid_dice = sum(dice_scores)/len(dice_scores)

            if valid_dice <(0.80*BASELINE_DICE_SCORE):
                efficiency_metric = 0
            else:
                efficiency_metric = valid_dice/ gflops
            # --------------------------------------------------------- #

            wandb.log({
                "valid_loss": valid_loss,
                "valid_dice_score":valid_dice, # 
                "efficiency_metric":efficiency_metric 
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                )
                torch.save(Model.state_dict(), current_best_model_path)
        scheduler.step()
        wandb.log({"learning_rate": optimizer.param_groups[0]['lr']}, step=(epoch + 1) * len(train_dataloader))
    print("Training complete!")

    # Save the model
    torch.save(
        Model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        )
    )

    tracker.stop()
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    # Make sure cache memory is free
    torch.cuda.empty_cache()
    main(args)
