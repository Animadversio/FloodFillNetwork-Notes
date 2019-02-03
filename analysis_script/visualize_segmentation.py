"""
Utilities to visualize masks from `canvas` object, segmentation, or
any volume numpy array
"""
import numpy as np
import matplotlib.pyplot as plt
import ffn.inference.storage as storage
from ffn.inference import inference
from os.path import isdir, join
import os
import logging
# logging.getLogger().setLevel(logging.INFO) # set the information level to show INFO logs

#%%
def visualize_seed(canvas, zid, threshold=0.6, baseline=0.1):
    plt.figure(figsize=[15,15])
    plt.imshow(canvas.image[zid,:,:]*((canvas.seed[zid,:,:]>threshold) + baseline), cmap='gray')
    plt.show()
    plt.close()

def visualize_mask(canvas, mask, zid, baseline=0.1):
    plt.figure(figsize=[15,15])
    plt.imshow(canvas.image[zid,:,:]*(mask[zid,:,:] + baseline), cmap='gray')
    plt.show()
    plt.close()

if __name__=="__main__":
    #%%
    visualize_seed(canvas, 10, baseline=0.1)
    #%%
    id = 2
    data = np.load("/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_point/seg%03d.npz" % id)
    mask = data['segmentation']
    #%%
    visualize_mask(canvas, canvas.segmentation, 350, baseline=1.2)#    canvas.segmentation, canvas.seed>0.6

