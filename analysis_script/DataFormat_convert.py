# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:45:29 2018

@author: Binxu
http://docs.h5py.org/en/latest/quick.html
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import h5py
from os.path import join
#%% Inspect into h5 files 

fmap = h5py.File(r"C:\Users\MorganLab\Documents\Binxu Notes\ffn-master\third_party\neuroproof_examples\validation_sample\grayscale_maps.h5", 'r')
graymap = fmap['raw']
graymap_array = graymap[:,:,:]
pxl_vals = np.unique(graymap_array) # range can be parsed out of this ndarray 
#%%
fseg = h5py.File(r"C:\Users\MorganLab\Documents\Binxu Notes\ffn-master\third_party\neuroproof_examples\validation_sample\groundtruth.h5", 'r')
segments = fseg['stack']
segment_array = segments[:,:,:]
seg_ids = np.unique(segment_array)
#%% Read in image stacks 
path = "C:\\Users\\MorganLab\\Documents\\LGNs1_P32_smallHighres\\"
pattern = "Segmentation1-LX_8-14.vsseg_export_s%03d.png"
raw_pattern = "Segmentation1-LX_8-14.vsseg_export_s%03d_2368x2128_16bpp.raw"
raw_shape = (2128,2368) 
stack_n = 175
image_stacks = np.zeros((stack_n,*raw_shape), dtype=np.int16)
for i in range(stack_n):
    # img = plt.imread((join(path,raw_pattern)) % (i))
    data_from_raw = np.fromfile((join(path,raw_pattern)) % (i), dtype=np.int16)
    data_from_raw.shape = raw_shape
    image_stacks[i,:,:] = data_from_raw
    plt.imshow(data_from_raw)
    plt.axis('off')
ids = np.unique(image_stacks)
image_stacks = image_stacks[:,:img_shape[0],:img_shape[1]]  # crop to same size
#%%
#%% Read in image stacks (PNG gray scale)
path = "C:\\Users\\MorganLab\\Documents\\LGNs1_P32_smallHighres\\"
pattern = "tweakedImageVolume2_export_s%03d.png"
stack_n = 175
img_shape = (2116, 2360)
EM_image_stacks = np.zeros((stack_n,*img_shape,), dtype=np.uint8) 
for i in range(stack_n): 
    img = plt.imread((join(path,pattern)) % (i))
    img = (img[:,:,0]*255).astype(dtype=np.uint8)
    EM_image_stacks[i,:,:] = img
    plt.imshow(data_from_raw, cmap="Greys") 
    plt.axis('off')
#ids = np.unique(image_stacks)
#%% Convert the RGB code back to int code! 
#%% Save into h5 files 
output = "groundtruth_zyx.h5"
f = h5py.File(join(path, output), "w")
fstack=f.create_dataset("stack", (stack_n,*img_shape,), dtype='int16') # Note they only take 1nt64 input
fstack[:] = image_stacks
f.close() 

#%%
output = "grayscale_maps_zyx.h5"
f = h5py.File(join(path, output), "w")
fstack=f.create_dataset("raw", (stack_n,*img_shape,), dtype='uint8')
fstack[:] = EM_image_stacks
f.close() 

#%% Export ndarry to stakced image files
for i in range(250):
    plt.imshow(seg[:,:,i])
    plt.imsave("img%03d.png"%i,seg[:,:,i],cmap='Greys')
    plt.axis('off')
    plt.colorbar()
    plt.show()
    
#%%
for i in range(10,12):
    plt.imshow(image_stacks[:,:,i])
    #plt.imsave("img%03d.png"%i,seg[:,:,i])
    plt.axis('off')
    plt.colorbar()
    plt.show()


#%%