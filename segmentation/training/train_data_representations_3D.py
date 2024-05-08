#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 17:09:25 2022

@author: toma
"""


import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, distance_transform_cdt, grey_closing
from skimage import measure
from segmentation.utils.utils import get_nucleus_ids
import skimage.segmentation as seg
from scipy.ndimage import gaussian_filter, binary_erosion

def generate_multipixel_boundaries(nucleus,num_pixels):
    edges_combined=np.zeros(nucleus.shape,dtype=bool)
    for i in range(num_pixels):
        edges = seg.find_boundaries(nucleus, mode = 'inner')
        edges_combined=np.logical_or(edges,edges_combined)
        edges_temp=np.array(edges,dtype=int)
        nucleus_temp=np.array(nucleus,dtype=int)
        nucleus=nucleus_temp-edges_temp
    return edges_combined

def dist_label_3d(label, neighbor_radius=None, apply_grayscale_closing=True):
    """ Cell and neigbhor distance label creation (Euclidean distance).

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param neighbor_radius: Defines the area to look for neighbors (smaller radius in px decreases the computation time)
        :type neighbor_radius: int
    :param apply_grayscale_closing: close gaps in between neighbor labels.
        :type apply_grayscale_closing: bool
    :return: Cell distance label image, neighbor distance label image.
    """
    # Relabel label to avoid some errors/bugs
    label_dist = np.zeros(shape=label.shape, dtype=np.float)
    label_dist_neighbor = np.zeros(shape=label.shape, dtype=np.float)

    props = measure.regionprops(label)
    if neighbor_radius is None:
        mean_diameter = []
        for i in range(len(props)):
            mean_diameter.append(props[i].equivalent_diameter)
        mean_diameter = np.mean(np.array(mean_diameter))
        neighbor_radius = 3 * mean_diameter

    # Find centroids, crop image, calculate distance transform
    for i in range(len(props)):

        # Get nucleus and Euclidean distance transform for each nucleus
        nucleus = (label == props[i].label)
        centroid, diameter = np.round(props[i].centroid), int(np.ceil(props[i].equivalent_diameter))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
                       int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
                       int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
                       ]
        nucleus_crop_dist = distance_transform_edt(nucleus_crop)
        if np.max(nucleus_crop_dist) > 0:
            nucleus_crop_dist = nucleus_crop_dist / np.max(nucleus_crop_dist)
        label_dist[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
        int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
        ] += nucleus_crop_dist

        # Get crop containing neighboring nuclei
        nucleus_neighbor_crop = np.copy(label[
                                int(max(centroid[0] - neighbor_radius, 0)):int(
                                    min(centroid[0] + neighbor_radius, label.shape[0])),
                                int(max(centroid[1] - neighbor_radius, 0)):int(
                                    min(centroid[1] + neighbor_radius, label.shape[1])),
                                int(max(centroid[2] - neighbor_radius, 0)):int(
                                    min(centroid[2] + neighbor_radius, label.shape[2]))
                                ])
        num_cells=np.unique(nucleus_neighbor_crop)
        if num_cells.shape[0]==2:
            nucleus_neighbor_crop_nucleus = nucleus_neighbor_crop == props[i].label
            nucleus_neighbor_crop_dist = 1
        else:            
            # Convert background to nucleus id
            nucleus_neighbor_crop_nucleus = nucleus_neighbor_crop == props[i].label
            nucleus_neighbor_crop[nucleus_neighbor_crop == 0] = props[i].label
            nucleus_neighbor_crop[nucleus_neighbor_crop != props[i].label] = 0
            nucleus_neighbor_crop = nucleus_neighbor_crop > 0
            nucleus_neighbor_crop_dist = distance_transform_edt(nucleus_neighbor_crop)
            nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist * nucleus_neighbor_crop_nucleus
            if np.max(nucleus_neighbor_crop_dist) > 0:
                nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist / np.max(nucleus_neighbor_crop_dist)
            else:
                nucleus_neighbor_crop_dist = 1
        nucleus_neighbor_crop_dist = (1 - nucleus_neighbor_crop_dist) * nucleus_neighbor_crop_nucleus
        label_dist_neighbor[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
        int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
        ] += nucleus_neighbor_crop_dist

    if apply_grayscale_closing:
        label_dist_neighbor = grey_closing(label_dist_neighbor, size=(2, 2, 2)) # changed for 3D
    label_dist_neighbor = label_dist_neighbor ** 3 # changed from 3 (org) to 4/5
    if (label_dist_neighbor.max() - label_dist_neighbor.min()) > 0.3: # changed from 0.5 to 0.3
        label_dist_neighbor = (label_dist_neighbor - label_dist_neighbor.min()) / (label_dist_neighbor.max() - label_dist_neighbor.min())
    else:
        label_dist_neighbor = np.zeros(shape=label.shape, dtype=np.float)
    #label_dist = label_dist ** 2  # added by Tanjin (2 initially by Tanjin)
    return label_dist, label_dist_neighbor


def dist_label_3d_edge_scaled_boundary_map_max(label, neighbor_radius=None, num_pixels=2, apply_grayscale_closing=True):  
    label_dist = np.zeros(shape=label.shape, dtype=float)
    label_dist_neighbor = np.zeros(shape=label.shape, dtype=float)

    props = measure.regionprops(label)

    if neighbor_radius is None:
        mean_diameter = []
        for i in range(len(props)):
            mean_diameter.append(props[i].equivalent_diameter)
        mean_diameter = np.mean(np.array(mean_diameter))
        neighbor_radius = 3 * mean_diameter
    for i in range(len(props)): 
        nucleus = (label == props[i].label)
        centroid, diameter = np.round(props[i].centroid), int(np.ceil(props[i].equivalent_diameter))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
                       int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
                       int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
                       ]
        nucleus_crop_dist = distance_transform_edt(nucleus_crop)

        if np.max(nucleus_crop_dist) > 0:
            nucleus_crop_dist = nucleus_crop_dist / np.max(nucleus_crop_dist)
        label_dist[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
        int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
        ] += nucleus_crop_dist
        
        #edges = seg.find_boundaries(nucleus_crop, mode = 'thick')
        edges = generate_multipixel_boundaries(nucleus_crop,num_pixels)
        edges=np.array(edges,dtype=float)
        # Get crop containing neighboring nuclei
        nucleus_neighbor_crop = np.copy(label[
                                int(max(centroid[0] - neighbor_radius, 0)):int(
                                    min(centroid[0] + neighbor_radius, label.shape[0])),
                                int(max(centroid[1] - neighbor_radius, 0)):int(
                                    min(centroid[1] + neighbor_radius, label.shape[1])),
                                int(max(centroid[2] - neighbor_radius, 0)):int(
                                    min(centroid[2] + neighbor_radius, label.shape[2]))
                                ])
        num_cells=np.unique(nucleus_neighbor_crop)
        if num_cells.shape[0]==2:
            nucleus_neighbor_crop_nucleus = nucleus_neighbor_crop == props[i].label
            nucleus_neighbor_crop_dist = 1
        else:            
            # Convert background to nucleus id
            nucleus_neighbor_crop_nucleus = nucleus_neighbor_crop == props[i].label
            nucleus_neighbor_crop[nucleus_neighbor_crop == 0] = props[i].label
            nucleus_neighbor_crop[nucleus_neighbor_crop != props[i].label] = 0
            nucleus_neighbor_crop = nucleus_neighbor_crop > 0
            nucleus_neighbor_crop_dist = distance_transform_edt(nucleus_neighbor_crop)
            nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist * nucleus_neighbor_crop_nucleus
            if np.max(nucleus_neighbor_crop_dist) > 0:
                nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist / np.max(nucleus_neighbor_crop_dist)
            else:
                nucleus_neighbor_crop_dist = 1
        nucleus_neighbor_crop_dist = (1 - nucleus_neighbor_crop_dist) * nucleus_neighbor_crop_nucleus
        
        nucleus_neighbor_crop_dist=np.multiply(nucleus_neighbor_crop_dist,edges) # element wise multiply with edge binary map
        
        label_dist_neighbor[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
        int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
        ] += nucleus_neighbor_crop_dist
        
   
    if (label_dist_neighbor.max() - label_dist_neighbor.min()) > 0.05: # changed from 0.5 to 0.3
        if apply_grayscale_closing:
            label_dist_neighbor = grey_closing(label_dist_neighbor, size=(2, 2, 2)) # (1,2,2) initially
        label_dist_neighbor = (label_dist_neighbor - label_dist_neighbor.min()) / (label_dist_neighbor.max() - label_dist_neighbor.min())
    else:
        label_dist_neighbor = np.zeros(shape=label.shape, dtype=float)
   
    return label_dist, label_dist_neighbor


def dist_label_3d_edge_scaled_boundary_map_sum(label, neighbor_radius=None, num_pixels=2, apply_grayscale_closing=True):  
    label_dist = np.zeros(shape=label.shape, dtype=float)
    label_dist_neighbor = np.zeros(shape=label.shape, dtype=float)

    props = measure.regionprops(label)

    if neighbor_radius is None:
        mean_diameter = []
        for i in range(len(props)):
            mean_diameter.append(props[i].equivalent_diameter)
        mean_diameter = np.mean(np.array(mean_diameter))
        neighbor_radius = 3 * mean_diameter
    for i in range(len(props)): 
        nucleus = (label == props[i].label)
        centroid, diameter = np.round(props[i].centroid), int(np.ceil(props[i].equivalent_diameter))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
                       int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
                       int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
                       ]
        nucleus_crop_dist = distance_transform_edt(nucleus_crop)

        if np.max(nucleus_crop_dist) > 0:
            nucleus_crop_dist = nucleus_crop_dist / np.max(nucleus_crop_dist)
        label_dist[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
        int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
        ] += nucleus_crop_dist
        
        #edges = seg.find_boundaries(nucleus_crop, mode = 'thick')
        edges = generate_multipixel_boundaries(nucleus_crop,num_pixels)
        edges=np.array(edges,dtype=float)
        # Get crop containing neighboring nuclei
        nucleus_neighbor_crop = np.copy(label[
                                int(max(centroid[0] - neighbor_radius, 0)):int(
                                    min(centroid[0] + neighbor_radius, label.shape[0])),
                                int(max(centroid[1] - neighbor_radius, 0)):int(
                                    min(centroid[1] + neighbor_radius, label.shape[1])),
                                int(max(centroid[2] - neighbor_radius, 0)):int(
                                    min(centroid[2] + neighbor_radius, label.shape[2]))
                                ])
        num_cells=np.unique(nucleus_neighbor_crop)
        if num_cells.shape[0]==2:
            nucleus_neighbor_crop_nucleus = nucleus_neighbor_crop == props[i].label
            nucleus_neighbor_crop_dist = 1
        else:            
            # Convert background to nucleus id
            nucleus_neighbor_crop_nucleus = nucleus_neighbor_crop == props[i].label
            nucleus_neighbor_crop[nucleus_neighbor_crop == 0] = props[i].label
            nucleus_neighbor_crop[nucleus_neighbor_crop != props[i].label] = 0
            nucleus_neighbor_crop = nucleus_neighbor_crop > 0
            nucleus_neighbor_crop_dist = distance_transform_edt(nucleus_neighbor_crop)
            nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist * nucleus_neighbor_crop_nucleus
            if np.max(nucleus_neighbor_crop_dist) > 0:
                nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist / np.sum(nucleus_neighbor_crop_dist)
            else:
                nucleus_neighbor_crop_dist = 1
        nucleus_neighbor_crop_dist = (1 - nucleus_neighbor_crop_dist) * nucleus_neighbor_crop_nucleus
        
        nucleus_neighbor_crop_dist=np.multiply(nucleus_neighbor_crop_dist,edges) # element wise multiply with edge binary map
        
        label_dist_neighbor[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
        int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
        ] += nucleus_neighbor_crop_dist
        
   
    if (label_dist_neighbor.max() - label_dist_neighbor.min()) > 0.05: # changed from 0.5 to 0.3
        if apply_grayscale_closing:
            label_dist_neighbor = grey_closing(label_dist_neighbor, size=(2, 3, 3)) # (1,2,2) initially
        label_dist_neighbor = (label_dist_neighbor - label_dist_neighbor.min()) / (label_dist_neighbor.max() - label_dist_neighbor.min())
    else:
        label_dist_neighbor = np.zeros(shape=label.shape, dtype=float)
   
    return label_dist, label_dist_neighbor

def dist_label_3d_edge_boundary_map(label, neighbor_radius=None, apply_grayscale_closing=False):  
  
    label_dist = np.zeros(shape=label.shape, dtype=float)
    label_dist_neighbor = np.zeros(shape=label.shape, dtype=float)
    
    props = measure.regionprops(label)
   
    if neighbor_radius is None:
        mean_diameter = []
        for i in range(len(props)):
            mean_diameter.append(props[i].equivalent_diameter)
        mean_diameter = np.mean(np.array(mean_diameter))
        neighbor_radius = 3 * mean_diameter
    for i in range(len(props)): 
        nucleus = (label == props[i].label)
        centroid, diameter = np.round(props[i].centroid), int(np.ceil(props[i].equivalent_diameter))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
                       int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
                       int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
                       ]
       
        nucleus_crop_dist = distance_transform_edt(nucleus_crop)
   
        if np.max(nucleus_crop_dist) > 0:
            nucleus_crop_dist = nucleus_crop_dist / np.max(nucleus_crop_dist)
        label_dist[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
        int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
        ] += nucleus_crop_dist
        
        edges = seg.find_boundaries(nucleus_crop, mode = 'thick')      
        edges_crop_dist = distance_transform_edt(edges)
                
        if np.max(edges_crop_dist) > 0:
            edges_crop_dist = edges_crop_dist / np.max(edges_crop_dist)
        
        label_dist_neighbor[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
        int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
        ] += edges_crop_dist
        
    
    # if apply_grayscale_closing:
    #     label_dist_neighbor = grey_closing(label_dist_neighbor, size=(1, 1, 1)) # changed for 3D
        
    sigma_border = (0.5, 1.0, 1.0)  
    label_dist_neighbor = gaussian_filter(label_dist_neighbor, sigma=sigma_border)
    label_dist_neighbor = (label_dist_neighbor - label_dist_neighbor.min()) / (label_dist_neighbor.max() - label_dist_neighbor.min())
  
    return label_dist, label_dist_neighbor


def dist_label_3d_BCM3D_2(label, neighbor_radius=None, apply_grayscale_closing=True):  
 
    label_dist = np.zeros(shape=label.shape, dtype=float)
    label_dist_neighbor = np.zeros(shape=label.shape, dtype=float)
    
    props = measure.regionprops(label)
    
    if neighbor_radius is None:
        mean_diameter = []
        for i in range(len(props)):
            mean_diameter.append(props[i].equivalent_diameter)
        mean_diameter = np.mean(np.array(mean_diameter))
        neighbor_radius = 3 * mean_diameter
    for i in range(len(props)): 
        nucleus = (label == props[i].label)
        centroid, diameter = np.round(props[i].centroid), int(np.ceil(props[i].equivalent_diameter))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
                       int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
                       int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
                       ]
        #plt.figure();plt.imshow(np.max(nucleus_crop,axis=0),cmap='gray')
        nucleus_crop_dist = distance_transform_edt(nucleus_crop)
        #plt.figure();plt.imshow(np.max(nucleus_crop_dist,axis=0),cmap='gray')
    
        if np.max(nucleus_crop_dist) > 0:
            nucleus_crop_dist = nucleus_crop_dist / np.max(nucleus_crop_dist)
        label_dist[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
        int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
        ] += nucleus_crop_dist
        
        # Get crop containing neighboring nuclei
        nucleus_neighbor_crop = np.copy(label[
                                int(max(centroid[0] - neighbor_radius, 0)):int(
                                    min(centroid[0] + neighbor_radius, label.shape[0])),
                                int(max(centroid[1] - neighbor_radius, 0)):int(
                                    min(centroid[1] + neighbor_radius, label.shape[1])),
                                int(max(centroid[2] - neighbor_radius, 0)):int(
                                    min(centroid[2] + neighbor_radius, label.shape[2]))
                                ])
        num_cells=np.unique(nucleus_neighbor_crop)
        if num_cells.shape[0]==2:
            nucleus_neighbor_crop_nucleus = nucleus_neighbor_crop == props[i].label
            nucleus_neighbor_crop_dist = 1
        else:            
            # Convert background to nucleus id
            nucleus_neighbor_crop_nucleus = nucleus_neighbor_crop == props[i].label
            nucleus_neighbor_crop[nucleus_neighbor_crop == 0] = props[i].label
            nucleus_neighbor_crop[nucleus_neighbor_crop != props[i].label] = 0
            nucleus_neighbor_crop = nucleus_neighbor_crop > 0
            nucleus_neighbor_crop_dist = distance_transform_edt(nucleus_neighbor_crop)
            nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist * nucleus_neighbor_crop_nucleus
            if np.max(nucleus_neighbor_crop_dist) > 0:
                nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist / np.max(nucleus_neighbor_crop_dist)
            else:
                nucleus_neighbor_crop_dist = 1
        nucleus_neighbor_crop_dist = (1 - nucleus_neighbor_crop_dist) * nucleus_neighbor_crop_nucleus
       
        label_dist_neighbor[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
        int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
        ] += nucleus_neighbor_crop_dist
        
    if (label_dist_neighbor.max() - label_dist_neighbor.min()) > 0.05: # changed from 0.5 to 0.3
        label_dist_neighbor = label_dist_neighbor ** 1.5
        label_dist_neighbor = (label_dist_neighbor - label_dist_neighbor.min()) / (label_dist_neighbor.max() - label_dist_neighbor.min())
    else:
        label_dist_neighbor = np.zeros(shape=label.shape, dtype=float)   
        
    label=np.array(label,dtype=float)
    label[label>0]=1
   
    diff_img=label-label_dist
    after_multiply=np.multiply(diff_img,label_dist_neighbor)
      
    after_multiply_closing = grey_closing(after_multiply, size=(1, 2, 2)) # changed for 3D    
    sigma_border = (0.5, 1.0, 1.0)   #(0.5, 1.5, 1.5)
    final_map_boundary = gaussian_filter(after_multiply_closing, sigma=sigma_border)
    final_map_boundary = (final_map_boundary - final_map_boundary.min()) / (final_map_boundary.max() - final_map_boundary.min())
    
    label_dist = label_dist ** 3.0
    sigma_cell = (0.5, 1.0, 1.0)   #(0.5, 1.5, 1.5)
    final_map_cell = gaussian_filter(label_dist, sigma=sigma_cell)
    
    return final_map_cell, final_map_boundary
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
