#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:19:55 2024

@author: toma
"""
import numpy as np
from skimage import filters, measure, morphology
from skimage.morphology import ball

from segmentation.inference.functions.fun_param_bounds import cell_param_limits
from segmentation.inference.functions.fun_multi_otsu import multi_otsu_process

def process_difference_map(Id):
    
    th = filters.threshold_otsu(Id)
    Id_log = Id > th
    #plt.figure(); plt.imshow(np.max(Id_log,axis=0),cmap='gray')
    Id_log = morphology.binary_erosion(Id_log, ball(1))
    Id_label = measure.label(Id_log.astype(int), background=0, return_num=False, connectivity=1)
    #plt.figure(); plt.imshow(np.max(Id_label,axis=0),cmap='gray')
    Id_label = morphology.dilation(Id_label, ball(1))
    size_th=10 # 
    Id_label = morphology.remove_small_objects(Id_label, size_th) #
    Id_log_img = Id_log * Id
    #plt.figure(); plt.imshow(np.max(Id_log_img,axis=0),cmap='gray')  
    vol_bounds, vc_ratio_upper_bound = cell_param_limits(Id_label)
    labels_final = multi_otsu_process(Id_label, Id_log_img, np.amax(Id_label), vol_bounds[0], vol_bounds[1], vc_ratio_upper_bound, min_seed=size_th) 
    return labels_final