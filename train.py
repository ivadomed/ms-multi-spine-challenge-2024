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
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
)
from monai.transforms import (
    AsDiscrete,
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
    RandGaussianSharpend
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
import resource

def dice_score(prediction, groundtruth, smooth=1.):
    numer = (prediction * groundtruth).sum()

    denor = (prediction + groundtruth).sum()

    # loss = (2 * numer + self.smooth) / (denor + self.smooth)
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

config = {
    "max_iteration": 5000,
    "batch_size": 1,
    "learning_rate": 1e-4,  
    "model": UNETR ,
    "weight_decay": 1e-5, 
    "feature_size": 16,
    "data_augmentation": "no aug", #"aug_sqrt, aug_sin,aug_exp,aug_sig,aug_laplace,aug_inverse, RandGaussianNoised,RandGaussianSharpend,tio.RescaleIntensity," 
}


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    dice_epoch = 0 
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())

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

        
            
        dice_epoch += dice_score(prediction= output,  groundtruth = y).detach().cpu().item()
        
        if global_step%30 == 0 : 
            train_image= x[0].detach().cpu().squeeze()
            train_gt= y[0].detach().cpu().squeeze()
            train_pred= output[0].detach().cpu().squeeze()

            fig = plot_slices(image=train_image,
                        gt=train_gt,
                        pred=train_pred,
                                )

            wandb.log({"training images": wandb.Image(fig)})
            plt.close(fig)
            
    
    
    dice_epoch /= step
    wandb.log({"train_dice": dice_epoch, "epoch": global_step//80})
    print(dice_epoch)
    

    epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
    dice_val = validation(epoch_iterator_val)
    epoch_loss /= step
    wandb.log({"train_loss": epoch_loss, "epoch": global_step//80})  # Log training loss
    print(epoch_loss)
    
    wandb.log({"val_dice": dice_val, "epoch": global_step//80})  # Log training loss
    print(dice_val)
       
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
       
   
    
    
    

    return global_step, dice_val_best, global_step_best


def validation(epoch_iterator_val):
    model.eval()
    counter = 0 
    val_dice_epoch = 0 
    number = 0 
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            outputs = sliding_window_inference(val_inputs, target_specs["T2"]["shape"], mode="gaussian",
                                           sw_batch_size=4, predictor=model, overlap=0.5,) 
             # get probabilities from logits
            outputs = F.relu(outputs) / F.relu(outputs).max() if bool(F.relu(outputs).max()) else F.relu(outputs)
        
            loss = loss_function(outputs, val_labels)
      
            val_outputs = [post_pred(i) for i in decollate_batch(outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
           
            number += len(val_outputs)
            dice = dice_score(val_outputs[0], val_labels[0]).detach().cpu().sum().item()
            
            val_dice_epoch += dice

            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 20.0))  # noqa: B038

            if True or counter%10 == 0 : 
                val_image= val_inputs[0].detach().cpu().squeeze()
                val_gt= val_labels[0].detach().cpu().squeeze()
                val_pred= val_outputs[0].detach().cpu().squeeze()

                fig = plot_slices(image=val_image,
                            gt=val_gt,
                            pred=val_pred,
                                    )

                wandb.log({"validation images": wandb.Image(fig)})
                plt.close(fig)
            
            counter+=1 
        
        
        val_dice_epoch /= number
       
        
    return val_dice_epoch


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

    
    
    plt.tight_layout()
    fig.show()
    return fig
     
output_path = os.path.join("output_path", str(datetime.now().date()) +"_" +str(datetime.now().time()))
os.makedirs(output_path, exist_ok=True)

wandb.init(project=f'monai-ms-lesion-seg-transformer-approach', config=config, save_code=True, dir=output_path)



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
    "T2": {"resolution": (3.0, 0.7, 0.7), "shape": (16, 512, 528)},
    "seg": {"resolution": (3.0, 0.7, 0.7), "shape": (16, 512, 528)}  # Same as T2
}

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RPI"),
        Spacingd(
            keys=["image", "label"],
            pixdim=target_specs["T2"]["resolution"],
            mode=("bilinear", "nearest"),
        ),
        
        #RandLambdad(keys=["image"],func=aug_sqrt,prob=0.1,),
        #RandLambdad(keys=["image"],func=aug_sin,prob=0.1,),
        #RandLambdad(keys=["image"],func=aug_exp,prob=0.1,),
        #RandLambdad(keys=["image"],func=aug_sig,prob=0.1, ),
        #RandLambdad(keys=["image"],func=aug_laplace,prob=0.1,),
        #RandLambdad(keys=["image"],func=aug_inverse,prob=0.1, ),        
        #RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.1),
        #RandGaussianSharpend(keys=["image"], prob=0.1),   
        #tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), include=["image"]),

        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=target_specs["T2"]["shape"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RPI"),
        Spacingd(
            keys=["image", "label"],
            pixdim=target_specs["T2"]["resolution"],
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityd(
            keys=["image"],
        ),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=target_specs["T2"]["shape"]),
    ]
)

data_dir = ""
split_json = "dataset_split.json"

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

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = UNETR(
    in_channels=1,
    out_channels=1,
    img_size=(16, 512, 528),
    feature_size=config["feature_size"],
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)




loss_function = DiceCELoss(sigmoid = False, smooth_dr=1e-4) #added sigmoid=False discussion with PL
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

max_iterations = config["max_iteration"]
post_label = Compose([EnsureType()])
post_pred = Compose([EnsureType()])
#dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

wandb.finish()  