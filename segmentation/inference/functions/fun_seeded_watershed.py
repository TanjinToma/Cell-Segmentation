#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:50:04 2024

@author: toma
"""

import numpy as np
from skimage import measure,morphology
from skimage.segmentation import watershed

def seeded_watershed(input_logical, input_bin_rep, ts, min_seed):
    seed_binary = input_bin_rep > ts
    seed_label = measure.label(seed_binary.astype(int), background=0, return_num=False, connectivity=1)
    seed_label = morphology.remove_small_objects(seed_label, min_size=min_seed)  
    # watershed
    labels_temp = watershed(-input_bin_rep, seed_label, mask=input_logical)
    labels = labels_temp + input_logical
    labels = measure.label(labels, background=0, return_num=False, connectivity=1)
    labels = morphology.remove_small_objects(labels, min_size=min_seed)
    return labels