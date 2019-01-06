# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:02:48 2018

@author: MorganLab
"""

'''Usage
from utils_format_convert import convert_image_stack_to_h5,convert_raw_seg_stack_to_h5

path="C:\\Users\\MorganLab\\Documents\\LGNs1_P32_smallHighres\\"
stack_n=175
EM_name_pattern="tweakedImageVolume2_LRexport_s%03d.png"
raw_name_pattern="Segmentation1-LX_8-14.vsseg_LRexport_s%03d_1184x1072_16bpp.raw"
EM_stack = convert_image_stack_to_h5(path=path, pattern=,stack_n=stack_n,output="grayscale_maps_LR.h5") 
seg_stack = convert_raw_seg_stack_to_h5(path=path, raw_pattern=raw_name_pattern, 
    stack_n=stack_n, raw_shape=(1072,1184), img_shape=EM_stack.shape[1:], output="groundtruth_LR.h5")


You will find the h5 files in the `path` ! 
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import h5py
from os.path import join

def convert_image_stack_to_h5(path, pattern, stack_n, output = "grayscale_maps_zyx.h5"):
    ''' 
    
    E.g. 
    path = "C:\\Users\\MorganLab\\Documents\\LGNs1_P32_smallHighres\\"
    pattern = "tweakedImageVolume2_export_s%03d.png"
    stack_n = 175
    img_shape = (2116, 2360)
    
    convert_image_stack_to_h5(path=path, pattern="tweakedImageVolume2_LRexport_s%03d.png",stack_n=175,output="grayscale_maps_LR.h5") 
    
    '''
    tmp_img = plt.imread((join(path,pattern)) % 0)
    img_shape = tmp_img.shape[:2]
    # Read in image stacks (PNG gray scale)
    EM_image_stacks = np.zeros((stack_n,*img_shape,), dtype=np.uint8) 
    print("Image stack shape: ")
    print((stack_n,*img_shape,))
    for i in range(stack_n): 
        img = plt.imread((join(path,pattern)) % (i))
        img = (img[:,:,0]*255).astype(dtype=np.uint8)
        EM_image_stacks[i,:,:] = img
        plt.imshow(img, cmap="Greys") 
        plt.axis('off')
    f = h5py.File(join(path, output), "w")
    fstack=f.create_dataset("raw", (stack_n,*img_shape,), dtype='uint8')
    fstack[:] = EM_image_stacks
    f.close() 
    return EM_image_stacks

def convert_raw_seg_stack_to_h5(path, raw_pattern, stack_n, raw_shape, img_shape, output="groundtruth_zyx.h5", in_dtype=np.int16, out_dtype='int16'):
    '''
    raw_shape: shape marked in the name of file(inverse order)
    img_shape: shape of image stack, use to crop the seg_stack as there are 0 padding 
    out_dtype: 'int16', 'int64' either is ok
    E.g. 
    path = "C:\\Users\\MorganLab\\Documents\\LGNs1_P32_smallHighres\\"
    pattern = "Segmentation1-LX_8-14.vsseg_export_s%03d.png"
    raw_pattern = "Segmentation1-LX_8-14.vsseg_export_s%03d_2368x2128_16bpp.raw"
    raw_shape = (2128, 2368) 
    img_shape = (2116, 2360)
    stack_n = 175
    
    seg_stack2 = convert_raw_seg_stack_to_h5(path=path, raw_pattern="Segmentation1-LX_8-14.vsseg_LRexport_s%03d_1184x1072_16bpp.raw", 
    stack_n=175, raw_shape=(1072,1184), img_shape=(1058, 1180), output="groundtruth_LR.h5")
    '''
    print("Raw Data stack shape: ")
    print((stack_n,*raw_shape,))
    image_stacks = np.zeros((stack_n,*raw_shape), dtype=in_dtype)
    for i in range(stack_n):
        # img = plt.imread((join(path,raw_pattern)) % (i))
        data_from_raw = np.fromfile((join(path,raw_pattern)) % (i), dtype=in_dtype)
        data_from_raw.shape = raw_shape
        image_stacks[i,:,:] = data_from_raw
        plt.imshow(data_from_raw)
        plt.axis('off')
    ids = np.unique(image_stacks)
    print("Get %d unique ids"%len(ids))
    image_stacks = image_stacks[:,:img_shape[0],:img_shape[1]]  # crop to same size
    print("Image stack shape: ", image_stacks.shape)
    f = h5py.File(join(path, output), "w")
    fstack=f.create_dataset("stack", (stack_n,*img_shape,), dtype=out_dtype) # Note they only take int64 input
    fstack[:] = image_stacks
    f.close() 
    return image_stacks
    
