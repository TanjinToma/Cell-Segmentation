#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:54:11 2024

@author: toma
"""

import numpy as np
from skimage import measure,morphology


def labels_aggregate(labels_processed, labels_process_further, label_max,min_vol):
    labels_process_further[labels_process_further>0] += label_max
    # aggregate instances
    labels_all = labels_processed + labels_process_further
    labels_all = measure.label(labels_all, background=0, return_num=False)
    labels_all = morphology.remove_small_objects(labels_all, min_size=min_vol)
    return labels_all