# -*- coding: utf-8 -*-
"""
Read and Export the Segmentation files to VAST
and calculate simple statistics of segmentation
"""


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import ffn.inference.storage as storage
from ffn.inference import inference
import logging
from os.path import isdir, join
import os
from scipy import ndimage
from scipy.misc import imresize
from analysis_script.utils_format_convert import read_image_vol_from_h5
logging.getLogger().setLevel(logging.INFO)  # set the information level to show INFO logs

#%%
def load_segmentation_output(output_dir, corner):
    testSegLoc = storage.subvolume_path(output_dir, corner, 'npz')
    testProbLoc = storage.subvolume_path(output_dir, corner, 'prob')
    data = np.load(testSegLoc)
    segmentation = data['segmentation']
    data_prob = np.load(testProbLoc)
    qprob = data_prob['qprob']
    return segmentation, qprob
#%%
def visualize_supervoxel_size_dist(segmentation, show_fig=True, save_fig=True, save_dir=''):
    idx, cnts = np.unique(segmentation, return_counts=True)
    assert idx[0] == 0
    idx = idx[1:]
    cnts = cnts[1:]  # discard the background supervoxel!
    plt.figure()
    plt.hist(np.log10(cnts), bins=50)
    # plt.hist(cnts, bins=50, log=True)
    plt.title("Voxel count distribution for all labels\n Label count: %d, Mean voxel #: %d, median voxel #:%d"
              % (len(idx), cnts.mean(), np.median(cnts)))  # dis
    plt.xlabel("log10(voxel number)")
    plt.ylabel("supervoxel count")
    if save_fig:
        plt.savefig(join(save_dir, "segment_dist.png"))
    if show_fig:
        plt.show()
    plt.close()
    if len(idx) > 2 ** 16:
        print("Too many labels, more than the import maximum of VAST %d " % 2 ** 16)
        logging.warning("Too many labels %d, more than the import maximum of VAST %d " % (len(idx), 2 ** 16))
    return idx, cnts

#%% Export segmentation
def export_segmentation_to_VAST(export_dir, segmentation, show_fig=False, suffix='tif', resize=1):
    '''Turn a segmentation(numpy ndarray) into importable tif or png files

    resize: is used in case the segmentation is done on a **Higher resolution** than the VAST volume
        then the segmentation has to be resized (downsampled) to be imported into VAST
        if ffn is run on a lower resolution image then VAST volume, then VAST can import it.

    Use the G,B bytes in tif image to code integer label
    Example:
    exportLoc = '/home/morganlab/Documents/Autoseg_result/LGN_Autoseg_Mov_point'
    export_segmentation_to_VAST(exportLoc, canvas.segmentation)
    '''
    # if len(idx) > 2 ** 16: # if idx is not defined
    if segmentation.max() > 2 ** 16:
        print("Too many labels, more than the import maximum of VAST %d " % 2 ** 16)
        logging.warning("Too many labels %d, more than the import maximum of VAST %d " % (len(idx), 2 ** 16))
    os.makedirs(export_dir, exist_ok=True)
    out_img_size = (int(segmentation.shape[1]*resize), int(segmentation.shape[2]*resize), 3)
    for i in range(segmentation.shape[0]):
        # code the integer labels in G and B channel of the color image!
        export_img = np.zeros((segmentation.shape[1], segmentation.shape[2], 3), dtype=np.uint8)
        export_img[:, :, 1], export_img[:, :, 2] = np.divmod(segmentation[i, :, :], 256)
        # export_img[:, :, 0], export_img[:, :, 1] = np.divmod(export_img[:, :, 1], 256)
        if resize==1:
            out_img = export_img
        else:
            out_img = np.zeros(out_img_size, dtype=np.uint8)
            # out_img = ndimage.zoom(export_img, (resize, resize, 1)) # do not zoom on channel direction
            out_img = imresize(export_img, out_img_size, interp='nearest')  # much faster, or there will be label change
        plt.figure()
        plt.imsave(join(export_dir, "seg_%03d.%s" % (i, suffix)), out_img)
        if show_fig:
            plt.imshow(out_img)
            plt.show()
        plt.close()
#%%
def export_composite_image(segmentation, image_stack, export_dir, suffix="png",
                           alpha=0.2, resize=1, bbox=None, show_fig=False):
    """bbox like [(0,0,0),(175, 1058, 1180)] in zyx order"""
    if bbox == None:
        assert segmentation.shape == image_stack.shape
    else:
        image_stack = image_stack[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]]
        assert segmentation.shape == image_stack.shape
    stack_n = image_stack.shape[0]
    for z in range(stack_n):
        export_img = np.zeros((segmentation.shape[1], segmentation.shape[2], 3), dtype=np.uint8)
        out_img = np.zeros((segmentation.shape[1], segmentation.shape[2], 3), dtype=np.uint8)
        export_img[:, :, 1], export_img[:, :, 2] = np.divmod(segmentation[z, :, :], 256)
        background_mask = (segmentation[z, :, :] > 0.5) * alpha # alpha_mask
        out_img[:, :, 0] =  image_stack[z, :, :]
        out_img[:, :, 1] = (image_stack[z, :, :] * (1 - background_mask)) + background_mask * export_img[:, :, 1]
        out_img[:, :, 2] = (image_stack[z, :, :] * (1 - background_mask)) + background_mask * export_img[:, :, 2]

        plt.figure()
        plt.imsave(join(export_dir, "seg_render_%03d.%s" % (z, suffix)), out_img)
        if show_fig:
            plt.imshow(out_img)
            plt.show()
        plt.close()

#%%
if __name__=="__main__":
    #%%
    # testSegLoc = '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/seg-0_0_0.npz'
    # testProbLoc = '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/seg-0_0_0.prob'
    seg_dir = '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_success2/'  # Longterm wide field, lowthreshold file
    # '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_WF_cluster/'
    corner = (0, 0, 0)
    segmentation, qprob = load_segmentation_output(seg_dir, corner)
    #%%
    plt.figure()
    plt.imshow(segmentation[80, :, :])
    plt.show()
    #%%
    # idx, cnts=np.unique(segmentation[125,:,:],return_counts=True)
    idx, cnts = visualize_supervoxel_size_dist(segmentation)
    #%%
    exportLoc = '/home/morganlab/Documents/Autoseg_result/LGN_Autoseg_full2'
        # "/home/morganlab/Documents/Sample1_branch109/Autoseg/UpSp_Longtime_point"
        # '/home/morganlab/Documents/Autoseg_result/LGN_Autoseg_Mov_full'
    export_segmentation_to_VAST(exportLoc, segmentation, resize=1)  # canvas.segmentation  segmentation  np.nan_to_num(canvas.seed>0.6)
    #%% composite image for examine
    imgh5path = ""
    composite_export_dir = ""
    img_stack = read_image_vol_from_h5(imgh5path)
    export_composite_image(segmentation, img_stack, composite_export_dir, bbox=[(0,0,0), (175, 1058, 1180)])
