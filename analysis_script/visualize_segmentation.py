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
#%%
visualize_seed(canvas, 41, baseline=0.1)


#%%
def visualize_mask(canvas, mask, zid, baseline=0.1):
    plt.figure(figsize=[15,15])
    plt.imshow(canvas.image[zid,:,:]*(mask[zid,:,:] + baseline), cmap='gray')
    plt.show()
    plt.close()
#%%
id = 2
data = np.load("/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_point/seg%03d.npz"%id)
mask = data['segmentation']

#%%
visualize_mask(canvas, canvas.seed>0.6, 400, baseline=0.05)

