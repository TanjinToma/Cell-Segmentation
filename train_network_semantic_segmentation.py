#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:52:22 2021

@author: toma
"""
# import packages 
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset
from monai.config import print_config
import torch
import tempfile
import os
import sys
import math
import glob
import numpy as np
import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
torch.cuda.empty_cache()

import resource
rlimit=resource.getrlimit(resource.RLIMIT_NOFILE)
print(f'getrlimit before: {resource.getrlimit(resource.RLIMIT_NOFILE)}')
resource.setrlimit(resource.RLIMIT_NOFILE,(4096, rlimit[1]))
print(f'getrlimit after: {resource.getrlimit(resource.RLIMIT_NOFILE)}')


def main():
    # Set device for using CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    
    # set path for data and model

    data_dir = Path.cwd() / 'train_data' / 'semantic_seg_train_data'
     
    path_model = Path.cwd() / 'Trained_model_semantic_seg' # new trained model as output
    path_model.mkdir(exist_ok=True) 
    
    # Read train and validation dataset
    train_images = sorted(
        glob.glob(os.path.join(data_dir, "img", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "label", "*.nii.gz")))
           
            
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    
    train_size=math.floor(len(data_dicts)*0.85) # 85 percent train data
    val_size=len(data_dicts)-train_size
    
    train_files, val_files = data_dicts[:-val_size], data_dicts[-val_size:] # do 80-20 split 
    #print(train_files)
    #print(val_files)
    
    set_determinism(seed=0)
    
    # Setup transforms for training and validation
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
    
            ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=255.0,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 32),
                pos=1,
                neg=1,
                num_samples=4, #4 (3)
                image_key="image",
                image_threshold=0,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=255.0,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )
    # Define CacheDataset and DataLoader for training and validation
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4) 
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    
    # Create Model, Loss, Optimizer
    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=3, #2 for binary classification
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
       
    loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True) 
    
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    
    #Execute a typical PyTorch training process
    max_epochs = 100 
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=3)
    post_label = AsDiscrete(to_onehot=True, n_classes=3)
    
    
    all_metric_val = None
    
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        start = time.time()
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (96, 96, 32)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = post_pred(val_outputs)
                    val_labels = post_label(val_labels)
                    value = compute_meandice(
                        y_pred=val_outputs,
                        y=val_labels,
                        include_background=False,
                    )               
                    # to check dice of both classes
                    if all_metric_val is None:
                            all_metric_val = value.cpu().numpy()
                    else:
                            all_metric_val = np.append(all_metric_val, value.cpu().numpy(), axis=0)
                    # metric_count += len(value)
                    metric_count += value.shape[1]
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        path_model, "best_metric_model_dicefocal.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
        # Epoch training time
        print('Epoch training time: {:.1f}s'.format(time.time() - start))
    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")
    
    
if __name__ == "__main__":

    main()  
    
