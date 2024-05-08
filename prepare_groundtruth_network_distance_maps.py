#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:52:31 2024

@author: toma
"""




import argparse
import json
import math
import numpy as np
import random
import torch
import warnings
from PIL import Image,ImageSequence
import matplotlib.pyplot as plt
import skimage.io as io
import tifffile as tiff
from scipy.io import savemat
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.morphology import binary_closing
from segmentation.training.train_data_representations_3D import *
from segmentation.utils.utils import get_nucleus_ids


warnings.filterwarnings("ignore", category=UserWarning)

path_datasets = Path.cwd() / 'train_data'
dir_id='01' # rename if required
ctc_set_path = path_datasets / 'ctc_train_set'

#Create directories (just run once)
Path.mkdir(ctc_set_path, exist_ok=True)
Path.mkdir(ctc_set_path / 'train', exist_ok=True)
Path.mkdir(ctc_set_path / 'val', exist_ok=True)

# create train-val list 

cell_ids=[]
# read each image and generate transformed labels
radius=50
mode='ST'
label_ids = sorted((path_datasets / ( dir_id +'_' + mode) / 'SEG').glob('*.tif'))

for i, label_id in enumerate(label_ids): 
    #print(i)
    # Read uint16 label image and display MIP
    file_id = label_id.name.split('man_seg')[-1]
    file_name = path_datasets.stem + '_' + file_id.split('.tif')[0]
    cell_ids.append(file_name) 
    
train_size=math.floor(len(cell_ids)*0.80) # 80 percent train data
val_size=len(cell_ids)-train_size
train_ind=random.sample(range(0,len(cell_ids)), train_size)
train_ind.sort()

val_ind=set(list(range(0,len(cell_ids))))-set(train_ind)
val_ind=list(val_ind)
val_ind.sort()
    

train_ids=[cell_ids[x] for x in train_ind]
val_ids=[cell_ids[x] for x in val_ind]

train_val_ids = {"train" : train_ids, "val" : val_ids}

for i, label_id in enumerate(label_ids): 
    print(i)
    # Read uint16 label image and display MIP
    file_id = label_id.name.split('man_seg')[-1]
    label=io.imread(label_id)
    # fig = plt.figure()
    # plt.imshow(np.max(label,axis=0), cmap='gray')
    
    img = io.imread(str(label_id.parents[2] / (dir_id) / ('t' + file_id)))
    # fig = plt.figure()
    # plt.imshow(np.max(img,axis=0), cmap='gray')
    
    # crop to 64*128*128 patch volumes
    z_range=list(range(4,(label.shape[0]-64)))
    y_range=list(range(8,(label.shape[1]-128))) 
    x_range=list(range(8,(label.shape[2]-128)))
    
    for num_crops in range(10): # from one image create 10 randomly cropped patch
        z_start=random.choice(z_range)
        z_end=z_start+64
        y_start=random.choice(y_range)
        y_end=y_start+128
        x_start=random.choice(x_range)
        x_end=x_start+128
        
        img_crop=img[z_start:z_end,y_start:y_end,x_start:x_end]
        label_crop=label[z_start:z_end,y_start:y_end,x_start:x_end]
        if np.max(label_crop)==0:
            label_dist_crop=np.zeros_like(label_crop)
            label_dist_neighbor_crop=np.zeros_like(label_crop)
        else:
            label_dist_crop, label_dist_neighbor_crop = dist_label_3d_edge_scaled_boundary_map_max(label=label_crop, neighbor_radius=radius, num_pixels=1)
        # fig = plt.figure();plt.imshow(np.max(label_dist_crop,axis=0), cmap='gray'); 
        # fig = plt.figure(); plt.imshow(np.max(label_dist_neighbor_crop,axis=0), cmap='gray')
        
        # Add pseudo color channel and min-max normalize to 0 - 65535
        img_crop = np.expand_dims(img_crop, axis=-1)
        img_crop = 65535 * (img_crop.astype(np.float32) - img_crop.min()) / (img_crop.max() - img_crop.min())
        img_crop = np.clip(img_crop, 0, 65535).astype(np.uint16)
        label_crop = np.expand_dims(label_crop, axis=-1).astype(np.uint16)
        label_dist_crop = np.expand_dims(label_dist_crop, axis=-1).astype(np.float32)
        label_dist_neighbor_crop = np.expand_dims(label_dist_neighbor_crop, axis=-1).astype(np.float32)
               
        # Save
        file_name = path_datasets.stem + '_' + file_id
        crop_name = file_name.split('.tif')[0] + '_{:01d}.tif'.format(num_crops)
        for mode in ['train','val']:
            if file_name.split('.tif')[0] in train_val_ids[mode]: 
                if np.max(label_crop)==0:
                    num=np.random.randint(3, size=1); num=num.tolist()
                    if num[0] < 1:
                        tiff.imwrite(str(ctc_set_path / mode / ('img_' + crop_name)), img_crop)
                        tiff.imwrite(str(ctc_set_path / mode / ('mask_' + crop_name)), label_crop)               
                        tiff.imwrite(str(ctc_set_path / mode / ('dist_cell_' + crop_name)), label_dist_crop)
                        tiff.imwrite(str(ctc_set_path / mode / ('dist_neighbor_' + crop_name)), label_dist_neighbor_crop)
                else:
                    tiff.imwrite(str(ctc_set_path / mode / ('img_' + crop_name)), img_crop)
                    tiff.imwrite(str(ctc_set_path / mode / ('mask_' + crop_name)), label_crop)               
                    tiff.imwrite(str(ctc_set_path / mode / ('dist_cell_' + crop_name)), label_dist_crop)
                    tiff.imwrite(str(ctc_set_path / mode / ('dist_neighbor_' + crop_name)), label_dist_neighbor_crop)







