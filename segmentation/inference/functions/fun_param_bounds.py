#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:59:59 2024

@author: toma
"""

import numpy as np
from skimage import measure

def find_upper_limit_percentile(data):
    quartile1 = np.percentile(data, 25)
    quartile3 = np.percentile(data, 75)
    inter_quartile_range = quartile3-quartile1
    res = quartile3 + 1.5 * inter_quartile_range
    return res

def cell_param_limits(label_img):
    props = measure.regionprops_table(label_img, properties={'area','convex_area'})
    vol = props['area']
    convex_hull_vol = props['convex_area']
    vc_ratio = convex_hull_vol / vol
    vc_ratio_upper_bound = find_upper_limit_percentile(vc_ratio)
    max_vol = find_upper_limit_percentile(vol)
    min_vol = max_vol / 10
    vol_bounds=[max_vol, min_vol]
    return vol_bounds, vc_ratio_upper_bound