#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:06:01 2024

@author: toma
"""
import numpy as np
from skimage import measure


def filter_instances(input_labels, input_map_img, max_vol, vc_ratio_upper_bound):
    process_further_binary = np.zeros_like(input_labels)
    props = measure.regionprops_table(input_labels, properties={'label','area','convex_area'})
    label = props['label']    
    vol = props['area']
    convex_hull_vol = props['convex_area']
    vc_ratio = convex_hull_vol / vol

    for i in range(len(label)):
        if vol[i] > max_vol or vc_ratio[i] > vc_ratio_upper_bound:
            temp_var = input_labels == label[i]
            input_labels[input_labels == label[i]] = 0 
            process_further_binary += temp_var.astype(int)           
    labels_processed = input_labels
    postprocess_map_img = input_map_img * process_further_binary
    return process_further_binary, postprocess_map_img, labels_processed