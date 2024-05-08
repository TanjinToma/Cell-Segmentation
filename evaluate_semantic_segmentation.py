#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 18:58:29 2024

@author: toma
"""




from monai.networks.nets import UNet
from monai.networks.layers import Norm

from monai.inferers import sliding_window_inference

import torch

import numpy as np
import time
from pathlib import Path
import tifffile as tiff
from skimage.segmentation import watershed


if __name__ == "__main__":

    # data directory
    path_dataset = Path.cwd() / 'network_distance_maps_outputs'

    # Make results directory
    path_results = Path.cwd() / 'segmentation_results'
    path_results.mkdir(exist_ok=True)

    # trained model path
    #path_model = Path.cwd() / 'Trained_model_semantic_seg' / 'best_metric_model_dicefocal.pth' # newly trained model
    path_model = Path.cwd() / 'Pre-Trained_models'/ 'Pre-Trained_model_semantic_seg' / 'best_metric_model_dicefocal.pth'

    # Set device for using CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256,512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)


    # test img filenames
    files = sorted((path_dataset).glob('*.tif'))  

    # Evaluate the model
    model.load_state_dict(torch.load(path_model))
    model.eval()
    with torch.no_grad():
        for ind, file in enumerate(files):
            # Load maps
            maps = tiff.imread(str(file))
            cell_map=maps[0,:,:,:]
            cell_map=cell_map.astype(np.float32)
            border_map=maps[1,:,:,:]
            border_map=border_map.astype(np.float32)
            diff_map = cell_map - border_map
            diff_map[diff_map < 0] = 0
            diff_map=np.expand_dims(np.transpose(diff_map,(1,2,0)),axis=0)
            diff_map=torch.from_numpy(np.expand_dims(diff_map,axis=0))

            diff_map_labeled=maps[2,:,:,:]
            diff_map_labeled=diff_map_labeled.astype(np.int64)

            roi_size = (96, 96, 32)
            sw_batch_size = 1
            print('         ... processing {0}{1} ...'.format(file.stem, file.suffix))
            start = time.time()
            # semantic segmentation 
            test_outputs = sliding_window_inference(
                diff_map.to(device), roi_size, sw_batch_size, model)    
            image_output=torch.argmax(test_outputs, dim=1).detach().cpu()[0, :, :, :]
            image_output = np.transpose(image_output.numpy(),(2,0,1))               
            # # Marker-based watershed
            prediction_instance = watershed(image=-cell_map, markers=diff_map_labeled, mask=image_output.astype(bool), watershed_line=False)    
            prediction_instance=prediction_instance.astype(np.uint16)
             # Test time
            print('Test time: {:.1f}s'.format(time.time() - start))  
            # save result
            file_id = file.name.split('.tif')
            file_id = file_id[0].split('map')
            fname='mask'+file_id[1]+'.tif'

            tiff.imwrite(str(path_results/fname), prediction_instance)
