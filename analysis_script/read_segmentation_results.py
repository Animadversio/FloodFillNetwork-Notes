# -*- coding: utf-8 -*-
"""
Read and Export the Segmentation files to VAST
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import ffn.inference.storage as storage
from ffn.inference import inference
import logging
from os.path import isdir, join
import os
logging.getLogger().setLevel(logging.INFO) # set the information level to show INFO logs

#%% 
# testSegLoc = '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/seg-0_0_0.npz'
# testProbLoc = '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/seg-0_0_0.prob'

def load_segmentation_output(output_dir, corner):
    testSegLoc = storage.subvolume_path(output_dir, corner, 'npz')
    testProbLoc = storage.subvolume_path(output_dir, corner, 'prob')
    data = np.load(testSegLoc)
    segmentation = data['segmentation']
    data_prob = np.load(testProbLoc)
    qprob = data_prob['qprob']
    return segmentation, qprob

seg_dir = '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_Mov_full/'  # Longterm wide field, lowthreshold file
# '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_WF_cluster/'
corner = (0, 0, 0)
segmentation, qprob = load_segmentation_output(seg_dir, corner)
#%%
plt.figure()
plt.imshow(segmentation[80, :, :])
plt.show()
#%%
# idx, cnts=np.unique(segmentation[125,:,:],return_counts=True)
#%%
def visualize_supervoxel_size_dist():
    idx, cnts = np.unique(segmentation, return_counts=True)
    assert idx[0] == 0
    idx = idx[1:]
    cnts = cnts[1:]  # discard the background supervoxel!
    plt.hist(np.log10(cnts), bins=50)
    # plt.hist(cnts, bins=50, log=True)
    plt.title("Voxel count distribution for all labels\n Label count: %d, Mean voxel #: %d, median voxel #:%d"
              % (len(idx), cnts.mean(), np.median(cnts)))  # dis
    plt.xlabel("log10(voxel number)")
    plt.ylabel("supervoxel count")
    plt.show()
    if len(idx) > 2 ** 16:
        print("Too many labels, more than the import maximum of VAST %d " % 2 ** 16)
        logging.warning("Too many labels %d, more than the import maximum of VAST %d " % (len(idx), 2 ** 16))
    return idx, cnts
idx, cnts = visualize_supervoxel_size_dist()

#%% Export
#%%
def export_segmentation_to_VAST(export_dir, segmentation, show_fig=False, suffix='tif'):
    '''Turn a segmentation(numpy ndarray) into importable tif or png files

    Use the GB bytes to code integer label
    Example:
    exportLoc = '/home/morganlab/Documents/Autoseg_result/LGN_Autoseg_Mov_point'
    export_segmentation_to_VAST(exportLoc, canvas.segmentation)
    '''
    # if len(idx) > 2 ** 16: # if idx is not defined
    if segmentation.max() > 2 ** 16:
        print("Too many labels, more than the import maximum of VAST %d " % 2 ** 16)
        logging.warning("Too many labels %d, more than the import maximum of VAST %d " % (len(idx), 2 ** 16))
    os.makedirs(export_dir, exist_ok=True)
    for i in range(segmentation.shape[0]):
        # code the integer labels in G and B channel of the color image!
        export_img = np.zeros((segmentation.shape[1], segmentation.shape[2], 3), dtype=np.uint8)
        export_img[:, :, 1], export_img[:, :, 2] = np.divmod(segmentation[i, :, :], 256)
        # export_img[:, :, 0], export_img[:, :, 1] = np.divmod(export_img[:, :, 1], 256)
        plt.figure()
        plt.imsave(join(export_dir, "seg_%03d.%s" % (i, suffix)), export_img)
        if show_fig:
            plt.imshow(export_img)
            plt.show()
        plt.close()
#%%
exportLoc = "/home/morganlab/Documents/Sample1_branch109/Autoseg/Longtime_Mov_point"
    # '/home/morganlab/Documents/Autoseg_result/LGN_Autoseg_Mov_full'
export_segmentation_to_VAST(exportLoc, np.nan_to_num(canvas.seed>0.6)) # canvas.segmentation  segmentation
#%%
# tmp = plt.imread(exportLoc+"seg_%03d.tif"%10)
