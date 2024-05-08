#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:44:45 2024

@author: toma
"""
import numpy as np
from skimage import measure,filters,morphology
from segmentation.inference.functions.fun_seeded_watershed import seeded_watershed
from segmentation.inference.functions.fun_aggregate import labels_aggregate
from segmentation.inference.functions.fun_find_irregular_instances import filter_instances


def multi_otsu_process(input_labels, input_map_img, max_label, max_vol, min_vol, vc_ratio_upper_bound, min_seed): 
    
    process_further_binary_init, postprocess_map_img_init, labels_processed_init = filter_instances(input_labels, input_map_img, max_vol, vc_ratio_upper_bound)
    if np.sum(process_further_binary_init) > 0:
        thresh_further = filters.threshold_multiotsu(postprocess_map_img_init, classes=5)
        # threshold interior
        postprocess_labels_init = seeded_watershed(process_further_binary_init, postprocess_map_img_init, 
                                            thresh_further[-2], min_seed)
    else:
        postprocess_labels_init = process_further_binary_init
        
    if np.sum(postprocess_labels_init) > 0:
        # threshold border
        process_further_binary_final, postprocess_map_img_final, labels_processed_final = filter_instances(postprocess_labels_init, 
                                                         postprocess_map_img_init, max_vol, vc_ratio_upper_bound)
        postprocess_labels_final = seeded_watershed(process_further_binary_final, postprocess_map_img_final, 
                                            thresh_further[-1], min_seed)
    else:
        postprocess_labels_final = postprocess_labels_init
        labels_processed_final = postprocess_labels_init
    # aggregate
    labels_all = labels_processed_final + postprocess_labels_final
    labels_all = measure.label(labels_all, background=0, return_num=False, connectivity=1)
    labels_final = labels_aggregate(labels_processed_init, labels_all, max_label, min_vol)
    
    return labels_final



    

