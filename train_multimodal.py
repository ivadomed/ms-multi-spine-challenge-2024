import os
import torch
import numpy as np
import argparse
from datetime import datetime
from loguru import logger
import yaml
import nibabel as nib
from datetime import datetime
import numpy as np
import wandb
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import torch.multiprocessing
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from monai.transforms import (
    AsDiscrete,
    ConcatItemsd,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    Spacingd,
    RandRotate90d,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    BatchInverseTransform,
    RandAdjustContrastd,
    AsDiscreted,
    RandHistogramShiftd,
    ToTensord,
    ResizeWithPadOrCropd,
    EnsureTyped,
    RandLambdad,
    CropForegroundd,
    RandGaussianNoised,
    LabelToContourd,
    Invertd,
    SaveImage,
    EnsureType,
    Rand3DElasticd,
    RandSimulateLowResolutiond,
    RandBiasFieldd,
    RandAffined, 
    RandRotated, 
    RandZoomd,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    ScaleIntensityd,
    EnsureChannelFirstd,
)
from monai.networks.nets import ViT,  UNet, BasicUNet, AttentionUnet, SwinUNETR, UNETR
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.optimizers import Novograd
from monai.inferers import sliding_window_inference
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt 
from augment import *


config = {
    "max_iteration": 5000,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "model": UNETR ,
    "weight_decay": 1e-5, 
}


def plot_slices(image, gt, pred, debug=False):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    # bring everything to numpy 
    ## added the .float() because of issue : TypeError: Got unsupported ScalarType BFloat16
    image = image.float().numpy()
    gt = gt.float().numpy()
    pred = pred.float().numpy()


    mid_sagittal = image.shape[0]//2
    # plot X slices before and after the mid-sagittal slice in a grid
    fig, axs = plt.subplots(3, 6, figsize=(10, 6))
    fig.suptitle('Original Image --> Ground Truth --> Prediction')
    for i in range(6):
        axs[0, i].imshow(image[mid_sagittal-3+i,:,:].T, cmap='gray'); axs[0, i].axis('off') 
        axs[1, i].imshow(gt[mid_sagittal-3+i,:,:].T); axs[1, i].axis('off')
        axs[2, i].imshow(pred[mid_sagittal-3+i,:,:].T); axs[2, i].axis('off')

    # fig, axs = plt.subplots(1, 3, figsize=(10, 8))
    # fig.suptitle('Original Image --> Ground Truth --> Prediction')
    # slice = image.shape[2]//2

    # axs[0].imshow(image[:, :, slice].T, cmap='gray'); axs[0].axis('off') 
    # axs[1].imshow(gt[:, :, slice].T); axs[1].axis('off')
    # axs[2].imshow(pred[:, :, slice].T); axs[2].axis('off')
    
    plt.tight_layout()
    fig.show()
    return fig



def plot_slices_combined(combined, gt, pred, debug=False):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    # bring everything to numpy 
    ## added the .float() because of issue : TypeError: Got unsupported ScalarType BFloat16
    combined = combined.float().numpy()
    gt = gt.float().numpy()
    pred = pred.float().numpy()

    
    mid_sagittal = combined.shape[1]//2
    
    # plot X slices before and after the mid-sagittal slice in a grid
    fig, axs = plt.subplots(4, 6, figsize=(10, 6))
    fig.suptitle('Original Image --> Ground Truth --> Prediction')
    if np.all(combined == 0):
        print("Array contains only zeros")
    for i in range(6):
        axs[0, i].imshow(combined[0,mid_sagittal-3+i,:,:].T, cmap='gray'); axs[0, i].axis('off') 
        axs[1, i].imshow(gt[mid_sagittal-3+i,:,:].T); axs[1, i].axis('off')
        axs[2, i].imshow(pred[mid_sagittal-3+i,:,:].T); axs[2, i].axis('off')
        axs[3, i].imshow(combined[1,mid_sagittal-3+i,:,:].T, cmap='gray'); axs[3, i].axis('off') 
    
    plt.tight_layout()
    fig.show()
    return fig


def validation(epoch_iterator_val):
    model.eval()
    counter = 0 
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["combined"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, target_specs["image"]["shape"], 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
            
            if counter%10 == 0 : 
                val_image= val_inputs[0].detach().cpu().squeeze()
                val_gt= val_labels[0].detach().cpu().squeeze()
                val_pred= val_outputs[0].detach().cpu().squeeze()

                fig = plot_slices_combined(combined=val_image,
                            gt=val_gt,
                            pred=val_pred,
                                    )

                wandb.log({"validation images": wandb.Image(fig)})
                plt.close(fig)
            
            counter+=1 
        
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["combined"].cuda(), batch["label"].cuda())

        logit_map = model(x)
        output = F.relu(logit_map) / F.relu(logit_map).max() if bool(F.relu(logit_map).max()) else F.relu(logit_map)
        
        loss = loss_function(output, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(  # noqa: B038
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        global_step += 1

        
        if global_step%10 == 0 : 
            """train_gt= y[0].detach().cpu().squeeze()
            train_pred= output[0].detach().cpu().squeeze()

            t2_image = batch["image1"][0].detach().cpu().squeeze()
            fig = plot_slices(image=t2_image,
                        gt=train_gt,
                        pred=train_pred,
                                )
            
            wandb.log({"t2 images": wandb.Image(fig)})
            plt.close(fig)

            image = batch["image2"][0].detach().cpu().squeeze()
            fig = plot_slices(image=image,
                        gt=train_gt,
                        pred=train_pred,
                                )
            
            wandb.log({"other images": wandb.Image(fig)})
            plt.close(fig)"""

            train_image= x[0].detach().cpu().squeeze()
            train_gt= y[0].detach().cpu().squeeze()
            train_pred= output[0].detach().cpu().squeeze()

            fig = plot_slices_combined(combined=train_image,
                        gt=train_gt,
                        pred=train_pred,
                                )

            wandb.log({"training images": wandb.Image(fig)})
            plt.close(fig)
            
            

    
    epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
    dice_val = validation(epoch_iterator_val)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    metric_values.append(dice_val)
    if dice_val > dice_val_best:
        dice_val_best = dice_val
        global_step_best = global_step
        torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
        print(
            "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
        )
    else:
        print(
            "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                dice_val_best, dice_val
            )
        )
       
    wandb_logs = {
                "train_loss": epoch_loss,
                "val_loss": dice_val,
            }
    
    wandb_logs.clear()
    
    

    return global_step, dice_val_best, global_step_best






     

output_path = os.path.join("output_path", str(datetime.now().date()) +"_" +str(datetime.now().time()))
os.makedirs(output_path, exist_ok=True)

wandb.init(project=f'monai-ms-lesion-seg-unetr', config=config, save_code=True, dir=output_path)


exp_logger = pl.loggers.WandbLogger(
                    name="test",
                    save_dir=output_path,
                    group="ms-seg-challenge",
                    log_model=True, # save best model using checkpoint callback
                    config=config)

# Saving training script to wandb
wandb.save(str(config))

# Paths
train_dir = "dataset_split/train"
val_dir = "dataset_split/val"
root_dir = "models"
os.makedirs(root_dir, exist_ok=True)

# Specifications
target_specs = {
    "image": {"resolution": (3.0, 0.7, 0.7), "shape": (16, 512, 528)},
    "seg": {"resolution": (3.0, 0.7, 0.7), "shape": (16, 512, 528)}  # Same as T2
}

train_transforms = Compose(
    [
        LoadImaged(keys=["image1", "image2", "label"]),
        EnsureChannelFirstd(keys=["image1",'image2', "label"]),
        Orientationd(keys=["image1","image2", "label"], axcodes="RPI"),
        Spacingd(
            keys=["image1",'image2', "label"],
            pixdim=target_specs["image"]["resolution"],
            mode=("bilinear", "bilinear", "nearest"),
        ),
        ScaleIntensityd(
            keys=["image1","image2"],
        ),
        ResizeWithPadOrCropd(keys=["image1","image2", "label"], spatial_size=target_specs["image"]["shape"]),
        RandLambdad(
            keys=["image1","image2"],
            func=aug_log,
            prob=0.1,
        ),

        RandLambdad(
            keys=["image1","image2"],
            func=aug_sqrt,
            prob=0.1,
        ),

        RandLambdad(
            keys=["image1","image2"],
            func=aug_sin,
            prob=0.1,
        ),

        RandLambdad(
            keys=["image1","image2"],
            func=aug_exp,
            prob=0.1,
        ),

        RandLambdad(
            keys=["image1","image2"],
            func=aug_sig,
            prob=0.1,
        ),

        RandLambdad(
            keys=["image1","image2"],
            func=aug_laplace,
            prob=0.1,
        ),

        RandLambdad(
            keys=["image1","image2"],
            func=aug_inverse,
            prob=0.1,
        ),

        ConcatItemsd(keys=["image1","image2"], name="combined"),
        ToTensord(keys=["combined"])
    ]
)


val_transforms = Compose(
    [
        LoadImaged(keys=["image1", "image2", "label"]),
        EnsureChannelFirstd(keys=["image1",'image2', "label"]),
        Orientationd(keys=["image1","image2", "label"], axcodes="RPI"),
        Spacingd(
            keys=["image1",'image2', "label"],
            pixdim=target_specs["image"]["resolution"],
            mode=("bilinear", "bilinear", "nearest"),
        ),
        ScaleIntensityd(
            keys=["image1","image2"],
        ),
        ResizeWithPadOrCropd(keys=["image1","image2", "label"], spatial_size=target_specs["image"]["shape"]),
        ConcatItemsd(keys=["image1","image2"], name="combined"),
        ToTensord(keys=["combined"])
    ]
)
data_dir = ""
split_json = "dataset_split_image2.json"

datasets = data_dir + split_json
datalist = load_decathlon_datalist(datasets, True, "training")


val_files = load_decathlon_datalist(datasets, True, "validation")

train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=8, pin_memory=True)

val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms, 
    cache_num=6, 
    cache_rate=1.0, 
    num_workers=4
)
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""model = ViT(
    in_channels=1,  # Number of input modalities
    img_size=target_specs["T2"]["shape"],  # Input image shape
    patch_size=(4, 16, 16),
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed_type="sincos",
    classification=False,
    num_classes=1,  # Output channel for segmentation
    dropout_rate=0.1
).to(device)"""

model = UNETR(
    in_channels=2,
    out_channels=1,
    img_size=target_specs["image"]["shape"],
    feature_size=4,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True, smooth_dr=1e-4)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

max_iterations = config["max_iteration"]
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

wandb.finish()  


