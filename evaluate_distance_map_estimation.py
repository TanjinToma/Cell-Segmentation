#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 12:56:33 2024

@author: toma
"""



import argparse
import json
import numpy as np
import random
import torch
import warnings
import time
import matplotlib.pyplot as plt
import tifffile as tiff
from pathlib import Path


from segmentation.utils import unets_3D
from segmentation.inference.fun_predict_maps import predict_cell_and_border_maps
from segmentation.inference.functions.fun_process_diff_map import process_difference_map
from skimage import morphology
from skimage.morphology import ball

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    # random.seed()
    # np.random.seed()
    
    # data directory
    path_dataset = Path.cwd() / 'test_data'

    # Make results directory
    path_results = Path.cwd() / 'network_distance_maps_outputs'
    path_results.mkdir(exist_ok=True)

    # trained model path
    #path_model = Path.cwd() / 'Trained_model_distance_maps' # newly trained model
    path_model = Path.cwd() / 'Pre-Trained_models' / 'Pre-Trained_model_distance_maps' 
 
    # Set device for using CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    num_gpus = torch.cuda.device_count()
    
    # Get paths of trained model
    label_type='distance'
    model = sorted(path_model.glob('{}_model*.pth'.format(label_type)))
    
    start = time.time()
    model = next(iter(model))
    with open(model.parent / (model.stem + '.json')) as f:
        model_settings = json.load(f)
    
    
    # Inference of the trained model
    num_gpus = 1
    net = unets_3D.build_unet(unet_type=model_settings['architecture'][0],
                      act_fun=model_settings['architecture'][2],
                      pool_method=model_settings['architecture'][1],
                      normalization=model_settings['architecture'][3],
                      device=device,
                      num_gpus=num_gpus,
                      ch_in=1,
                      ch_out=1,
                      filters=model_settings['architecture'][4],
                      print_path=None)

    if num_gpus > 1:
        net.module.load_state_dict(torch.load(str(model), map_location=device))
    else:
        net.load_state_dict(torch.load(str(model), map_location=device))
    net.eval()
    torch.set_grad_enabled(False)

    # test img filenames
    files = sorted((path_dataset).glob('*.tif'))  
  
    # Prediction process (iterate over images/files)
    for ind, file in enumerate(files):

        # Load image
        img = tiff.imread(str(file))
        #plt.figure();plt.imshow(np.max(img,axis=0),cmap='gray')
        # predict cell distance map and border neighbor distance map
        print('         ... processing {0}{1} ...'.format(file.stem, file.suffix))
        prediction_cell, prediction_border=predict_cell_and_border_maps(img,net)
        
        #plt.figure();plt.imshow(np.max(prediction_cell,axis=0),cmap='gray') # visualize maximum intensity projection
        #plt.figure();plt.imshow(prediction_border[60,:,:],cmap='gray') # visualize a slice
       
        # normalize maps between 0 and 1
        prediction_cell[prediction_cell>1.0]=1.0
        prediction_cell[prediction_cell<0.0]=0.0
        
        prediction_border[prediction_border>1.0]=1.0
        prediction_border[prediction_border<0.0]=0.0

        # differece map
        diff_map = prediction_cell - prediction_border
        diff_map[diff_map < 0] = 0
        
        diff_map_labeled=process_difference_map(diff_map)
        
        # save maps
        prediction_cell=np.expand_dims(prediction_cell,axis=0)
        prediction_border=np.expand_dims(prediction_border,axis=0)
        diff_map_labeled=np.expand_dims(diff_map_labeled,axis=0)
        prediction_maps=np.concatenate((prediction_cell, prediction_border,diff_map_labeled), axis=0)
        
        file_id = file.name.split('.tif')
        file_id = file_id[0].split('t')
        fname='map'+file_id[1]+'.tif'
        tiff.imwrite(str(path_results/fname), prediction_maps)
        
        
    # Clear memory
    del net

    # # Test time
    # print('Test time: {:.1f}s'.format(time.time() - start))
    
    

   
    
    
