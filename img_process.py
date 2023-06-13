# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:28:52 2023

@author: ankro
"""

import numpy as np
import cv2

def rmvbgr(image_rgb):
    
    image_rgb = cv2.imdecode(np.frombuffer(image_rgb, np.uint8), cv2.IMREAD_UNCHANGED)
    #image_rgb=cv2.imread(image_rgb)
    
    rectangle = (60, 110, 280, 260) 
    
    mask = np.zeros(image_rgb.shape[:2], np.uint8) 
    
    
    bgdModel = np.zeros((1, 65), np.float64) 
    fgdModel = np.zeros((1, 65), np.float64) 

    cv2.grabCut(image_rgb, mask, rectangle, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

    mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    image_rgd_nobg = image_rgb * mask_2[:, :, np.newaxis]
    
    img = cv2.resize(image_rgd_nobg,dsize=(150,150))
    img = img.reshape((1,) + img.shape)
    img = img.astype(np.float32) / 255.0
    
    return img