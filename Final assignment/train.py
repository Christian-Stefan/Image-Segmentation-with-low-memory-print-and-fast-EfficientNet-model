"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` 
script or directly via the command line. It is entirely focused on finding the best design choices for RMSprop. 
This reduces the searching space in the attempt to improve one pivotal component.
2. **Data Augmentation Pipeline:** A mini pipeline wrapped in a standalone class (`CityscapesPipeline`) has been created 
in order to ease the implementation of new data augmentation techniques.
3. **List of weights:** To gain full control of what classes the model should stress most during the training phase, 
a list ('CITYSCAPES_CLASS_WEIGHTS') containing weighting float factors for all the 19 classes has been created that is converted 
into a 'weights_tensor,' which is being passed as a custom argument to the CrossEntropyLoss 'defined in 'make_criterion.'
4. **Craft criterion:** To impose experimental settings in a convenient and efficient fashion over the optimizer's parameters, 
a function 'make_criterion()' has been defined, which allows us to do this in one single go.
5. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
6. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.
"""

### \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Start Imports \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ ###
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
    RandomHorizontalFlip,
    RandomResizedCrop
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
### \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ End Imports \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ ###

### \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Start Setup Settings \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ ### 

# 1.
# Multipliers to heavily penalize missing small/rare objects (e.g., Pedestrians = 4x penalty)
# ... hard coded float values all together wrapped up in a list and \
# ... further passed to eturn nn.CrossEntropyLoss(weight = weight_tensr = CITYSCAPES_CLASS_WEIGHTS ...)
# ... to check the corresponding labels please have a closer look to  [1](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py), 62...99
CITYSCAPES_CLASS_WEIGHTS:list = [
    1.0,  # 0: Road
    2.0,  # 1: Sidewalk
    1.5,  # 2: Building
    3.0,  # 3: Wall
    3.0,  # 4: Fence
    3.0,  # 5: Pole
    3.0,  # 6: Traffic Light
    3.0,  # 7: Traffic Sign
    1.5,  # 8: Vegetation
    2.0,  # 9: Terrain
    1.5,  # 10: Sky
    4.0,  # 11: Person
    4.0,  # 12: Rider
    2.0,  # 13: Car
    4.0,  # 14: Truck
    4.0,  # 15: Bus
    4.0,  # 16: Train
    4.0,  # 17: Motorcycle
    4.0   # 18: Bicycle
]

# 2.
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 3. Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes} # Creates a dictionary of mapped classes through dict-comprehension
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    """Maps raw Cityscapes class IDs to the 19 standard training IDs used for evaluation, setting ignored classes to 255."""
    return label_img.apply_(lambda x: id_to_trainid[x])

# 4. Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

# 5. Building painted-by-number segmentation mask.
def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    """
    Def: Builds a Paint-by-Number map. Acts as a sort of engine painter

    e.g./personal note, 
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

# 6. Definition of all non-immutable setting-related arguments
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
    parser.add_argument("--criterion3", type=str, default="CrossEntropyLoss",choices=["CrossEntropyLoss", "DiceLoss","FocalLoss"],help="Third loss - optional")
    # Criteria utilities
    parser.add_argument("--label-smoothing", type=float, default=0.0,help="Only used by CrossEntropyLoss")
    parser.add_argument("--focal-gamma", type=float, default=4.0,help="Only used by FocalLoss (SMP)")
    parser.add_argument("--focal-alpha", type=float, default=None,help="Only used by FocalLoss (SMP)")
    parser.add_argument("--dice-smooth", type=float, default=0.0, help="Only used by DiceLoss (SMP)")

    return parser

# 7. Predefined loss criterion whose arguments can be modified in one go before execution starts
def make_criterion(name: str, args):
    """
    Def: Acts as a criterion constructor that permits the user to define and implement changes at a parameter level across three distinct loss functions in one single go. 
    All the desired changes will be applied globally—primarily within "main()`

    :param string name: the criteiron name (e.g., "--criterion1", type=str, default="DiceLoss",choices=["CrossEntropyLoss"'...) predefined as an .env var in parser container 
    :param args: other arguments predefined as .env vars in parser container (e.g. "--label-smoothing", type=float, default=0.0; --label-smoothing", type=float, default=0.0)

    return: object of the criteria with implicitly defined parameters
    """

    # 7.1 Remove extraneous characters that might undermine the natural flow of the process
    name = name.strip()
    # 7.2 Convert our list into a GPU tensor
    weights_tensor = torch.tensor(CITYSCAPES_CLASS_WEIGHTS, dtype=torch.float32).to(device)
    
    if name == "CrossEntropyLoss":
        # ignore_index masks out pixels equal to that value (no loss/grad contribution). [2](https://gist.github.com/ivechan/806faa4193c00ed41971c7f6878b4eca)[2](https://segmentation-models-pytorch.readthedocs.io/en/latest/_modules/segmentation_models_pytorch/losses/dice.html)
        return nn.CrossEntropyLoss(
            weight = weights_tensor,
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
        # FocalLoss supports ignore_index; implementation assumes logits by default. [4](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)[3](https://stackoverflow.com/questions/73135768/how-to-use-ignore-index-in-torch-nn-crossentropyloss)
        return smp.losses.FocalLoss(
            mode="multiclass",
            ignore_index=args.ignore_index,
            gamma=args.focal_gamma,
            alpha=args.focal_alpha)
    else:
        raise ValueError(f"Unknown criterion: {name}")

### \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ End Setup Settings \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ ### 

### \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Start Training & Testing \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ ### 
# 8. Building a mini pipeline;
class CityscapesPipeline:
    """
    Motivation: Chosing a "dunder" (e.g., __name_of_your_method__) it's a better practice as it converts our class into a memory-based method 
    which requires only one time instantiation and "rembers" the constructor's attributes;
    
    Def: Class used to augment data;

    --- Constructor ---
    :param tuple(int, int) size: Desired output dimensions for images and masks.
    :param bool is_train: Toggles between training augmentations and validation resizing.
    
    --- Class function __call__ ---
    :param PIL.Image/Tensor image: The input RGB image to be processed.
    :param PIL.Image/Tensor target: The ground truth segmentation mask.
    :return: Processed image (normalized float32) and target (int64).
    """
    def __init__(self,size=(512,512),is_train=True):
        self.is_train = is_train
        self.size = size
        
    def __call__(self, image, target):
        if self.is_train:
        	# Applicable to training;
            # 8.1. Resize (Bilinear for image, Nearest for mask to avoid blending class IDs)
            i, j, h, w = RandomResizedCrop.get_params(
		    image, scale=(0.3, 1.0), ratio=(1.0, 1.0))
            image = F.resized_crop(image, i, j, h, w, self.size, interpolation=InterpolationMode.BILINEAR)
            target = F.resized_crop(target, i, j, h, w, self.size, interpolation=InterpolationMode.NEAREST)

	        # 8.2. Synchronized Sample-Level Augmentation
	        # ...rolls the dice for each individual image, keeping image and mask perfectly aligned
            if self.is_train and random.random() > 0.5: 
                image = F.horizontal_flip(image)
                target = F.horizontal_flip(target)
        else:
        # Applicable to testing;
		# 8.3 For validation, we use a standard Resize to maintain consistency
            image = F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
            target = F.resize(target, self.size, interpolation=InterpolationMode.NEAREST)
        
	# 8.4. Format and Normalize
        image = F.to_dtype(F.to_image(image), torch.float32, scale=True)
        image = F.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # recommended distribution parameters to be used by [5](https://github.com/IliaZenkov/DCGAN-Rectangular-GANHacks2/blob/main/DCGAN_rectangular.ipynb)
        target = F.to_dtype(F.to_image(target), torch.int64)

        return image, target

# 9. Building the wrapper 
def main(args):

    # Initializinh external energy and performance related trackers
    # 9.1 CodeCarbon setting initialization and tracker start
    tracker = EmissionsTracker(
        project_name = "NNCV", # Project Name
        measure_power_secs = 15, # How often it pings the API
        api_key="None",
        save_to_api = True,
        experiment_id="a1ac9d91-e078-4104-a93e-df86c1b25b90"
    )
    tracker.start()
    
    # 9.2 W&B initialization
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

    # 9.3 Creating the data loaders based on the custom pipeline class
    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=CityscapesPipeline(is_train=True) # Uses custom pipeline class ('the joint pipeline') 
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


    # 9.4 Initialize the model and activate CUDA
    Model = get_model()
    Model = Model.to(device)

    # 9.5. Group Parameters for Differential Learning Rates
    backbone_params = Model.encoder.parameters() # Conservative learning rate (e.g., args.lrs[0]) must be applied to maintain the underlying luggage of knowledge intact 
    head_params = list(Model.decoder.parameters()) + list(Model.segmentation_head.parameters()) # Not conservative (e.g., args.lrs[1])

    # Define the loss function
    criterion1 = make_criterion(args.criterion1, args).to(device) # C.E.
    criterion2 = make_criterion(args.criterion2, args).to(device) # FocalLoss
    criterion3 = make_criterion(args.criterion3, args).to(device) # DiceLoss

    # Debug statement 1
    # print("Using criterion 1 {} and criterion 2 {}".format(criterion1, criterion2))

    # 9.6 Define the optimizer
    optimizer = RMSprop([{'params':backbone_params,'lr':args.lrs[0]},{'params':head_params,'lr':args.lrs[1]}], 
                        alpha=0.9, 
                        momentum=0.85, 
                        weight_decay=1e-5, 
                        eps=0.001)
    # optimizer = AdamW([{'params':backbone_params,'lr':args.lrs[0]},{'params':head_params, 'lr':args.lrs[1]}],
    #                   weight_decay=1e-4)

    # 9.7 Define the Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---------------FLOPs Calculation Starts ----------------
    # Determining number of FLOPs by feeding the model with a dummy image
    print("Calculating Model FLOPs...")
    dummy_input = torch.randn(1, 3, 299, 299).to(device)
    macs, params = profile(Model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2  # MACs (Multiply-Accumulate) are roughly half a FLOP
    gflops = flops / 1e9 # Convert to GigaFLOPs for readability
    print(f"Model GFLOPs: {gflops:.3f}")
    # Testing
    # ...define an independent Dice function just for observation
    observation_dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=255, from_logits=True)
    # Threshold
    BASELINE_DICE_SCORE:float = 0.65
   # ---------------FLOPs Calculation ends ----------------

    # 10. Training loop
    # 10.1 Defining the training-loop variables for recording performance across testing set;
    best_valid_loss:float = float('inf')
    current_best_model_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # 10.2. Initializing Training session
        Model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            # 10.3. Map IDs first (Vectorized is best here)
            labels = convert_to_train_id(labels) 
            # 10.4 Move to device and format dimensions
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)  # Remove channel dimension
            # 10.5 Forward and Backward Pass
            optimizer.zero_grad()
            outputs = Model(images)
            loss = ((0.4*criterion2(outputs, labels)) + (0.4*criterion1(outputs, labels))+(0.2*criterion3(outputs, labels)))
            loss.backward()
            optimizer.step()

            ### --- Start Logging training loss, learning rate evolution and epoch ---###
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            ### --- End Logging training loss, learning rate evolution and epoch ---###

      	# 11. Validation loop initialization
        Model.eval()
        with torch.no_grad():
            # 11.1 Defining containers wherein official and empirical loss (e.g., dice) will be separately stored
            losses:list = []
            dice_scores:list = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = Model(images) # Predict
                loss = ((0.4*criterion2(outputs, labels)) + (0.4*criterion1(outputs,labels)) + (0.2 * criterion3(outputs,labels))) # Quantify how much of a good prediction we got
                losses.append(loss.item()) # Keep the records of loss

                # ---------- Start calculating empirical dice observation ------------ #
                batch_dice_score = 1.0 - observation_dice(outputs, labels).item()     # Used just as an aditional observation mean to 
                dice_scores.append(batch_dice_score)                                 # help assess how close the performance of the model gets
                # ---------- End calculating empirical dice observation ------------ # from the relevant quantification metric (e.g., Dice/FLOP(s))
            
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

        ### --- Start Logging testing loss, dice score, and efficiency metric (Dice/FLOPs) --- ###
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
         ### --- End Logging testing loss, dice score, and efficiency metric (Dice/FLOPs) --- ###
    print("Training complete!")

    # 12. Save the model
    torch.save(
        Model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        )
    )

    tracker.stop()
    wandb.finish()
### \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ End Training & Testing \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ ### 


### \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Start Main Console \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ ### 
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    # Make sure cache memory is free
    torch.cuda.empty_cache()
    main(args)
## \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ End Main Console \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ ### 
