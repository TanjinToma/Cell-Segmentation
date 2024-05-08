#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 19:12:32 2022

@author: toma
"""



import json
import numpy as np
import random
import torch
import warnings

from pathlib import Path

from segmentation.training.cell_segmentation_dataset import CellSegDataset
from segmentation.training.mytransforms_3D import augmentors
from segmentation.training.training import train
from segmentation.utils import utils, unets_3D


warnings.filterwarnings("ignore", category=UserWarning)
torch.cuda.empty_cache()

def main():
    path_datasets = Path.cwd() / 'train_data'
 
    path_models = Path.cwd() / 'Trained_model_distance_maps'  # new trained model as output

    random.seed()
    np.random.seed()
        
    # load settings
    with open(Path.cwd() / 'settings_network_distance_maps.json') as f:
        settings = json.load(f)
    
    # Set device for using CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    num_gpus = torch.cuda.device_count()

    # Make directory for the trained model
    path_models.mkdir(exist_ok=True) 
    # set number of training epochs
    settings['max_epochs']=150 
       
    train_configs = {'architecture': settings['architecture'],
                      'batch_size': 4,
                      'break_condition': settings['break_condition'],
                      'label_type': 'distance',
                      'learning_rate': settings['learning_rate'],
                      'lr_patience': settings['lr_patience'],
                      'loss': settings['loss'],
                      'max_epochs': settings['max_epochs'],
                      'num_gpus': num_gpus,
                      'run_name': settings['run_name'] 
                      }

    train_configs['architecture'][4]=[32, 512]
    
    net = unets_3D.build_unet(unet_type=train_configs['architecture'][0],
                            act_fun=train_configs['architecture'][2],
                            pool_method=train_configs['architecture'][1],
                            normalization=train_configs['architecture'][3],
                            device=device,
                            num_gpus=num_gpus,
                            ch_in=1,
                            ch_out=1,
                            filters=train_configs['architecture'][4],
                            print_path=path_models)
      
    # # The training images are uint16 crops of a min-max normalized image
    data_transforms = augmentors(label_type=train_configs['label_type'], min_value=0, max_value=65535)
    train_configs['data_transforms'] = str(data_transforms)
    
    # # Load training and validation set
    datasets = {x: CellSegDataset(root_dir=path_datasets / 'ctc_train_set',
                                  label_type=train_configs['label_type'],
                                  mode=x,
                                  transform=data_transforms[x])
                for x in ['train', 'val']}
                
    # # Train model
    train(net=net, datasets=datasets, configs=train_configs, device=device, path_models=path_models)
    
    # # Write information to json-file
    utils.write_train_info(configs=train_configs, path=path_models)
    
if __name__ == "__main__":

    main()
