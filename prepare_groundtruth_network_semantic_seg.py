#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:14:24 2024

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
import skimage.io as io
import tifffile as tiff
from scipy.io import savemat
import nibabel as nib
import os 
import skimage.segmentation as seg

from pathlib import Path
from skimage.morphology import binary_closing
from segmentation.training.train_data_representations_3D import *
from segmentation.utils.utils import get_nucleus_ids


warnings.filterwarnings("ignore", category=UserWarning)

path_datasets = Path.cwd() / 'train_data'
dir_id='01' # rename if required
save_dir = path_datasets / 'semantic_seg_train_data'

#Create directories (just run once)
Path.mkdir(save_dir, exist_ok=True)
Path.mkdir(save_dir / 'img', exist_ok=True)
Path.mkdir(save_dir / 'label', exist_ok=True)

# read each image and generate input and labels
radius=50
mode='ST'
label_ids = sorted((path_datasets / ( dir_id +'_' + mode) / 'SEG').glob('*.tif'))

for id in range(len(label_ids)): 
    print(id)
    label_id=label_ids[id]
    label=io.imread(label_id)     
    
    file_id = label_id.name.split('man_seg')[-1]
    # create input 
    label_dist, label_dist_neighbor = dist_label_3d_edge_scaled_boundary_map_max(label=label, neighbor_radius=radius, num_pixels=1)
    
    diff_img=(label_dist - label_dist_neighbor)
    diff_img[diff_img>1]=1
    diff_img[diff_img<0]=0
    # fig = plt.figure();plt.imshow(np.max(diff_img,axis=0), cmap='gray'); 
    
    diff_img=diff_img*255
    diff_img=diff_img.astype(np.uint8)
   
    # save as .nii image 
    diff_img=np.transpose(diff_img,[1,2,0])
    diff_img = nib.Nifti1Image(diff_img,affine=None)
    example_filename = os.path.join(save_dir,'img', 'biofilm_'+str(id+1)+'.nii.gz')
    nib.save(diff_img, example_filename)
    
    # create corresponding three-class semantic mask
    semantic_mask=np.zeros(label.shape)

    edges = seg.find_boundaries(label, mode = 'thick')
    interior = 2*(label > 0)
    semantic_mask = edges + interior
    semantic_mask[semantic_mask == 3] = 1
    #   Swap category names - edges category 2, interior category 1, background category 0
    semantic_mask_temp = np.zeros(semantic_mask.shape, dtype = 'int')
    semantic_mask_temp[semantic_mask == 1] = 2
    semantic_mask_temp[semantic_mask == 2] = 1
    semantic_mask = semantic_mask_temp
    #print(semantic_mask.shape)
    
    semantic_mask=semantic_mask.astype(np.uint8)
   
    # save as .nii image 
    semantic_mask=np.transpose(semantic_mask,[1,2,0])
    semantic_mask = nib.Nifti1Image(semantic_mask,affine=None)
    example_filename = os.path.join(save_dir,'label', 'biofilm_'+str(id+1)+'.nii.gz')
    nib.save(semantic_mask, example_filename)
