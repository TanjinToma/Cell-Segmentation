#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:07:37 2022

@author: toma
"""

import cv2
import numpy as np
import random
import scipy
import torch
#from imgaug import augmenters as iaa
from torchvision import transforms

from segmentation.utils.utils import min_max_normalization


def augmentors(label_type, min_value, max_value):
    """ Get augmentations for the training process.

    :param label_type: Type of the label images, e.g., 'boundary' or 'distance'.
        :type label_type: str
    :param min_value: Minimum value for the min-max normalization.
        :type min_value: float
    :param max_value: Minimum value for the min-max normalization.
        :type min_value: float
    :return: Dict of augmentations.
    """

    data_transforms = {'train': transforms.Compose([Contrast(p=0.0), 
                                                    Blur(p=0.0),
                                                    ToTensor(label_type=label_type,
                                                             min_value=min_value,
                                                             max_value=max_value)]),
                       'val': ToTensor(label_type=label_type, min_value=min_value, max_value=max_value)}

    return data_transforms
    # Contrast (p=0.3), Blur(p=0.3)

class Blur(object):
    """ Blur augmentation (label-preserving transformation) """

    def __init__(self, p=0.75): #p=1
        """

        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p
        
    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:

            sigma = 5 * random.random()
            sample['image'] = scipy.ndimage.gaussian_filter(sample['image'], sigma, order=0)
        
        return sample


class Contrast(object):
    """ Contrast augmentation (label-preserving transformation) """

    def __init__(self, p=1):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p
        
    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:

        # Contrast and gamma adjustment

            img = sample['image']
            dtype = img.dtype
            img = (img.astype(np.float32) - np.iinfo(dtype).min) / (np.iinfo(dtype).max - np.iinfo(dtype).min)
            contrast_range, gamma_range = (0.65, 1.35), (0.5, 1.5)

            # Contrast
            img_mean, img_min, img_max = img.mean(), img.min(), img.max()
            factor = np.random.uniform(contrast_range[0], contrast_range[1])
            img = (img - img_mean) * factor + img_mean

            # Gamma
            img_mean, img_std, img_min, img_max = img.mean(), img.std(), img.min(), img.max()
            gamma = np.random.uniform(gamma_range[0], gamma_range[1])
            rnge = img_max - img_min
            img = np.power(((img - img_min) / float(rnge + 1e-7)), gamma) * rnge + img_min
            if random.random() < 0.5:
                img = img - img.mean() + img_mean
                img = img / (img.std() + 1e-8) * img_std

            img = np.clip(img, 0, 1)
            img = img * (np.iinfo(dtype).max - np.iinfo(dtype).min) - np.iinfo(dtype).min
            img = img.astype(dtype)

            sample['image'] = img

        return sample


# class Flip(object):
#     """ Flip and rotation augmentation (label-preserving transformation) """

#     def __init__(self, p=0.5):
#         """

#         param p: Probability to apply augmentation to an image.
#             :type p: float
#         """
#         self.p = p

#     def __call__(self, sample):
#         """

#         :param sample: Dictionary containing image and label image (numpy arrays).
#             :type sample: dict
#         :return: Dictionary containing augmented image and label image (numpy arrays).
#         """
#         img = sample['image']

#         if random.random() < self.p:

#             # img.shape: (imgWidth, imgHeight, imgChannels)
#             if img.shape[0] == img.shape[1]:
#                 h = random.randint(0, 2)
#             else:
#                 h = random.randint(0, 1)

#             if h == 0:  # Flip left-right

#                 sample['image'] = np.flip(img, axis=1).copy()
#                 if len(sample) == 3:
#                     sample['label'] = np.flip(sample['label'], axis=1).copy()
#                 elif len(sample) == 4:
#                     sample['border_label'] = np.flip(sample['border_label'], axis=1).copy()
#                     sample['cell_label'] = np.flip(sample['cell_label'], axis=1).copy()
#                 elif len(sample) == 5:
#                     sample['border_label'] = np.flip(sample['border_label'], axis=1).copy()
#                     sample['cell_dist_label'] = np.flip(sample['cell_dist_label'], axis=1).copy()
#                     sample['cell_label'] = np.flip(sample['cell_label'], axis=1).copy()
#                 else:
#                     raise Exception('Unsupported sample format.')

#             elif h == 1:  # Flip up-down

#                 sample['image'] = np.flip(img, axis=0).copy()
#                 if len(sample) == 3:
#                     sample['label'] = np.flip(sample['label'], axis=0).copy()
#                 elif len(sample) == 4:
#                     sample['border_label'] = np.flip(sample['border_label'], axis=0).copy()
#                     sample['cell_label'] = np.flip(sample['cell_label'], axis=0).copy()
#                 elif len(sample) == 5:
#                     sample['border_label'] = np.flip(sample['border_label'], axis=0).copy()
#                     sample['cell_dist_label'] = np.flip(sample['cell_dist_label'], axis=0).copy()
#                     sample['cell_label'] = np.flip(sample['cell_label'], axis=0).copy()
#                 else:
#                     raise Exception('Unsupported sample format.')

#             elif h == 2:  # Rotate 90°

#                 sample['image'] = np.rot90(img, axes=(0, 1)).copy()
#                 if len(sample) == 3:
#                     sample['label'] = np.rot90(sample['label'], axes=(0, 1)).copy()
#                 elif len(sample) == 4:
#                     sample['border_label'] = np.rot90(sample['border_label'], axes=(0, 1)).copy()
#                     sample['cell_label'] = np.rot90(sample['cell_label'], axes=(0, 1)).copy()
#                 elif len(sample) == 5:
#                     sample['border_label'] = np.rot90(sample['border_label'], axes=(0, 1)).copy()
#                     sample['cell_dist_label'] = np.rot90(sample['cell_dist_label'], axes=(0, 1)).copy()
#                     sample['cell_label'] = np.rot90(sample['cell_label'], axes=(0, 1)).copy()
#                 else:
#                     raise Exception('Unsupported sample format.')

#         return sample


class Noise(object):
    """ Gaussian noise augmentation """

    def __init__(self, p=0.25):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:

            # Add noise with sigma 1-7% of image maximum
            sigma = random.randint(1, 7) / 100 * np.max(sample['image'])

            # Add noise to the stack
            img = sample['image']
            #img_org=img
            dtype = img.dtype
            mean=0.0
            depth,row,col,ch= img.shape
            gauss = np.random.normal(mean,sigma,(depth,row,col,ch))

            gauss=gauss.astype(np.float32)
            img=img.astype(np.float32)
            img=img+gauss
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = img * (np.iinfo(dtype).max - np.iinfo(dtype).min) - np.iinfo(dtype).min
            #img = img * (np.max(img_org) - np.min(img_org)) - np.min(img_org)
            img = img.astype(dtype)
            sample['image'] = img

        return sample

 
class Rotate(object):
    """ Rotation augmentation (label-changing augmentation) """

    def __init__(self, p=0.75): # p=1
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p
        
    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        angle = (-180, 180)
        
        if random.random() < self.p:

            angle = random.uniform(angle[0], angle[1])
                
            sample['image'] = scipy.ndimage.rotate(sample['image'], angle, axes=(1,2),reshape=False)
    
            border_label = scipy.ndimage.rotate(sample['border_label'], angle, axes=(1,2),reshape=False)
            border_label = (border_label - np.min(border_label)) / (np.max(border_label) - np.min(border_label))
            sample['border_label']=border_label
            
            cell_label = scipy.ndimage.rotate(sample['cell_label'], angle, axes=(1,2),reshape=False)
            cell_label = (cell_label - np.min(cell_label)) / (np.max(cell_label) - np.min(cell_label))
            sample['cell_label']=cell_label
            
        return sample


# class Scaling(object):
#     """ Scaling augmentation (label-changing transformation) """

#     def __init__(self, p=1):
#         """

#         param p: Probability to apply augmentation to an image.
#             :type p: float
#         """
#         self.p = p

#     def __call__(self, sample):
#         """

#         :param sample: Dictionary containing image and label image (numpy arrays).
#             :type sample: dict
#         :return: Dictionary containing augmented image and label image (numpy arrays).
#         """

#         scale = (0.85, 1.15)

#         scale1 = random.uniform(scale[0], scale[1])
#         scale2 = random.uniform(scale[0], scale[1])

#         if random.random() < self.p:

#             seq1 = iaa.Sequential([iaa.Affine(scale={"x": scale1, "y": scale2})])
#             seq2 = iaa.Sequential([iaa.Affine(scale={"x": scale1, "y": scale2}, order=0)])
#             sample['image'] = seq1.augment_image(sample['image'])

#             if len(sample) == 3:
#                 if sample['label'].dtype == np.uint8:
#                     sample['label'] = seq2.augment_image(sample['label'])
#                 else:
#                     sample['label'] = seq1.augment_image(sample['label']).copy()
#             elif len(sample) == 4:
#                 if sample['border_label'].dtype == np.uint8:
#                     sample['border_label'] = seq2.augment_image(sample['border_label'])
#                 else:
#                     sample['border_label'] = seq1.augment_image(sample['border_label'])

#                 if sample['cell_label'].dtype == np.uint8:
#                     sample['cell_label'] = seq2.augment_image(sample['cell_label'])
#                 else:
#                     sample['cell_label'] = seq1.augment_image(sample['cell_label'])

#             elif len(sample) == 5:  # Dual U-Net: boundary + cell discrete, cell_dist continuous
#                 sample['cell_label'] = seq2.augment_image(sample['cell_label'])
#                 sample['border_label'] = seq2.augment_image(sample['border_label'])
#                 sample['cell_dist_label'] = seq1.augment_image(sample['cell_dist_label'])

#             else:
#                 raise Exception('Unsupported sample format.')

#         return sample

  
class ToTensor(object):
    """ Convert image and label image to Torch tensors """
    
    def __init__(self, label_type, min_value, max_value):
        """

        :param min_value: Minimum value for the normalization. All values below this value are clipped
            :type min_value: int
        :param max_value: Maximum value for the normalization. All values above this value are clipped.
            :type max_value: int
        """
        self.min_value = min_value
        self.max_value = max_value
        self.label_type = label_type
        
    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        # Normalize image
        sample['image'] = min_max_normalization(sample['image'], min_value=self.min_value, max_value=self.max_value)

        # Swap axes from (D, H, W, Channels) to (Channels, D, H, W) (for 3D)
        for key in sample:
            if key != 'id':
                #sample[key] = np.transpose(sample[key], (2, 0, 1))
                sample[key] = np.transpose(sample[key], (3, 0, 1, 2))

        img = torch.from_numpy(sample['image']).to(torch.float)

        if self.label_type in ['boundary', 'border', 'pena']:

            # loss function (crossentropy) needs long tensor with shape [batch, height, width]
            label = torch.from_numpy(sample['label'])[0, :, :].to(torch.long)

            return img, label

        elif self.label_type == 'adapted_border':

            # loss function (crossentropy) needs long tensor with shape [batch, height, width]
            border_label = torch.from_numpy(sample['border_label'])[0, :, :].to(torch.long)

            # loss function (binary crossentropy) needs float tensor with shape [batch, channels, height, width]
            cell_label = torch.from_numpy(sample['cell_label']).to(torch.float)

            return img, border_label, cell_label

        elif self.label_type == 'dual_unet':

            border_label = torch.from_numpy(sample['border_label']).to(torch.float)
            cell_label = torch.from_numpy(sample['cell_label']).to(torch.float)
            cell_dist_label = torch.from_numpy(sample['cell_dist_label']).to(torch.float)

            return img, border_label, cell_dist_label, cell_label

        elif self.label_type == 'distance':

            # loss function (l1loss/l2loss) needs float tensor with shape [batch, channels, height, width]
            cell_label = torch.from_numpy(sample['cell_label']).to(torch.float)
            border_label = torch.from_numpy(sample['border_label']).to(torch.float)

            return img, border_label, cell_label
