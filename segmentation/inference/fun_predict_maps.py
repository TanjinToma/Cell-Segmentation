#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:10:58 2024

@author: toma
"""

import numpy as np
import torch
from segmentation.utils.utils_3D import min_max_normalization, zero_pad_model_input_bacteria, find_block_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_cell_and_border_maps(img,net):
    # Get position of the color channel
    if len(img.shape) == 3:  # add pseudo color channel
        img = np.expand_dims(img, axis=-1)
    

    # Min-max normalize the image to [0, 65535] (uint16 range)
    img = 65535 * (img.astype(np.float32) - img.min()) / (img.max() - img.min())
    img = np.clip(img, 0, 65535).astype(np.uint16)

    # Check if zero-padding is needed to apply the model           
    img, pads = zero_pad_model_input_bacteria(img=img)
    
    # find block size in all three dimesions
    block_sizeZ, block_sizeY, block_sizeX=find_block_size(img)

    # First convert image into the range [-1, 1] to get a zero-centered input for the network
    net_input = min_max_normalization(img=img, min_value=0, max_value=65535)

    # Bring input into the shape [batch, channel, height, width]
    net_input = np.transpose(np.expand_dims(net_input, axis=0), [0, 4, 1, 2, 3])

    # Prediction
    #print('         ... processing {0}{1} ...'.format(file.stem, file.suffix))
    net_input = torch.from_numpy(net_input).to(device)
    

    prediction_cell=np.zeros(net_input.shape,dtype=np.float32); prediction_cell = torch.from_numpy(prediction_cell).to(device)
    prediction_border=np.zeros(net_input.shape,dtype=np.float32); prediction_border = torch.from_numpy(prediction_border).to(device)

    step_sizeX=int(np.floor(block_sizeX/2))
    step_sizeY=int(np.floor(block_sizeY/2))
    step_sizeZ=int(np.floor(block_sizeZ/2))
    Z=list(range(0,net_input.shape[2],step_sizeZ))
    Y=list(range(0,net_input.shape[3],step_sizeY))
    X=list(range(0,net_input.shape[4],step_sizeX))
    for k in Z[:-1]:
        for j in Y[:-1]:
            for i in X[:-1]:
                net_input_block=net_input[:,:,k:k+block_sizeZ,j:j+block_sizeY,i:i+block_sizeX]
                prediction_border_block, prediction_cell_block = net(net_input_block)
                prediction_cell[:,:,k:k+block_sizeZ,j:j+block_sizeY,i:i+block_sizeX]=torch.max(prediction_cell_block, prediction_cell[:,:,k:k+block_sizeZ,j:j+block_sizeY,i:i+block_sizeX])
                prediction_border[:,:,k:k+block_sizeZ,j:j+block_sizeY,i:i+block_sizeX]=torch.max(prediction_border_block, prediction_border[:,:,k:k+block_sizeZ,j:j+block_sizeY,i:i+block_sizeX])
    
    prediction_cell = prediction_cell[0, 0, pads[0]:, pads[1]:, pads[2]:].cpu().numpy()
    prediction_border = prediction_border[0, 0, pads[0]:, pads[1]:, pads[2]:].cpu().numpy()
    
    return prediction_cell, prediction_border
