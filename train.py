import os
import torch
import numpy as np
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityd, ResizeWithPadOrCropd, EnsureTyped,
    ConcatItemsd, ToTensord
)
from monai.networks.nets import ViT
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.optimizers import Novograd
from monai.inferers import sliding_window_inference
import nibabel as nib



# Paths
train_dir = "dataset_split/train"
val_dir = "dataset_split/val"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Specifications
target_specs = {
    "T2": {"resolution": (3.0, 0.7, 0.7), "shape": (15, 512, 532)},
    "PSIR": {"resolution": (3.15, 0.34, 0.34), "shape": (15, 640, 640)},
    "STIR": {"resolution": (3.30, 0.57, 0.57), "shape": (15, 512, 544)},
    "MP2RAGE": {"resolution": (1.0, 0.94, 0.94), "shape": (176, 260, 180)},
    "seg": {"resolution": (3.0, 0.49, 0.49), "shape": (15, 512, 532)}  # Same as T2
}

# Data Preparation
def prepare_data(data_dir, transform):
    data = []
    labels = ["T2", "PSIR", "STIR", "MP2RAGE", "seg"]
    count = 0 
    for subject in os.listdir(data_dir):
        if "sub" in subject:
            subject_dir = os.path.join(data_dir, subject, "anat")
            if os.path.isdir(subject_dir):
                label_dict = {label: "default.nii.gz" for label in labels}
                for file in os.listdir(subject_dir):
                    if file.endswith(".nii.gz") and "preproc" not in file:
                        file_path = os.path.join(subject_dir, file)
                        for label in labels:
                            if label in file:
                                label_dict[label] = file_path
                count += 1
                data.append(label_dict)
    print(f'donnees chargees: {count}')
    return Dataset(data=data, transform=transform)

# Train Transforms
train_transforms = Compose([
    LoadImaged(keys=["T2", "PSIR", "STIR", "MP2RAGE", "seg"]),
    EnsureChannelFirstd(keys=["T2", "PSIR", "STIR", "MP2RAGE", "seg"]),
    Orientationd(keys=["T2", "PSIR", "STIR", "MP2RAGE", "seg"], axcodes="RAS"),
    Spacingd(keys=["T2", "seg"], pixdim=target_specs["T2"]["resolution"], mode=("bilinear", "nearest")),
    Spacingd(keys=["PSIR"], pixdim=target_specs["T2"]["resolution"], mode="bilinear"),
    Spacingd(keys=["STIR"], pixdim=target_specs["T2"]["resolution"], mode="bilinear"),
    Spacingd(keys=["MP2RAGE"], pixdim=target_specs["T2"]["resolution"], mode="bilinear"),
    
    ResizeWithPadOrCropd(keys=["T2", "seg"], spatial_size=target_specs["T2"]["shape"]),
    ResizeWithPadOrCropd(keys=["PSIR"], spatial_size=target_specs["T2"]["shape"]),
    ResizeWithPadOrCropd(keys=["STIR"], spatial_size=target_specs["T2"]["shape"]),
    ResizeWithPadOrCropd(keys=["MP2RAGE"], spatial_size=target_specs["T2"]["shape"]),
    ScaleIntensityd(keys=["T2", "PSIR", "STIR", "MP2RAGE"]),
    #EnsureTyped(keys=["T2", "PSIR", "STIR", "MP2RAGE", "seg"], nonzero=True, channel_wise=True),
    ConcatItemsd(keys=["T2", "PSIR", "STIR", "MP2RAGE"], name="comb"),
    ToTensord(keys=["comb"] )   
])

# Validation Transforms
val_transforms = train_transforms  # Use the same transforms as train for consistency

# Datasets and Loaders
train_ds = prepare_data(data_dir=train_dir, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

val_ds = prepare_data(data_dir=val_dir, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

# Define Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") 
model = ViT(
    in_channels=4,  # Number of input modalities
    img_size=target_specs["T2"]["shape"],  # Input image shape
    patch_size=(4, 16, 16),
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed_type="sincos",
    classification=False,
    num_classes=1,  # Output channel for segmentation
    dropout_rate=0.1
).to(device)


# Loss and Optimizer
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = Novograd(model.parameters(), lr=1e-4, weight_decay=1e-5)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

# Training Loop
max_epochs = 50
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []

for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    for batch_data in train_loader:
        inputs = batch_data["comb"].to(device)
        labels = batch_data["seg"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs[0]
        batch_size = labels.size(0)
        spatial_shape = target_specs["T2"]["shape"]  # (15, 512, 532)

        # Reshape outputs to match label dimensions
        outputs = outputs.view(batch_size, 1, *spatial_shape)

        segmentation_output = segmentation_output.view(batch_size, 1, *spatial_shape)

        
        loss = loss_function(labels, output)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    epoch_loss_values.append(epoch_loss)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    # Validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            dice_metric.reset()
            for val_data in val_loader:
                val_inputs = val_data["comb"].to(device)
                val_labels = val_data["seg"].to(device)
                val_outputs = sliding_window_inference(val_inputs, target_specs["T2"]["shape"], 4, model)
                dice_metric(y_pred=val_outputs, y=val_labels)
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            print(f"Validation Dice: {metric:.4f}")
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
                print("Saved new best model!")

print(f"Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")
