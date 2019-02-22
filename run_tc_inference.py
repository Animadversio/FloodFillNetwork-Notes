import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from tissue_classify.pixel_classifier2D import pixel_classifier_2d, inference_on_image
from scipy.misc import imresize
from PIL import Image
from glob import glob, iglob
from os.path import join, getctime
import os
#%%
img_dir = "/Users/binxu/Connectomics_Code/tissue_classifier/Train_Img/"
img = plt.imread(img_dir+"Soma_s092.png") # "Soma_s081_DS.png"
img = np.uint8(img[:, :, 0]*255)
#%%
pc2 = pixel_classifier_2d(img_rows=65, img_cols=65,)
inference_model = pc2.transfer_weight_to_inference("/Users/binxu/Connectomics_Code/tissue_classifier/Models/net_soma_ds-02-0.97.hdf5")
label_map = inference_on_image(img, inference_model)
plt.imshow(label_map)
plt.show()

plt.imshow(imresize(label_map, 2.0, interp='nearest'))
plt.show()
seg = imresize(label_map, 2.0, interp='nearest')
seg = Image.fromarray(seg).convert('L')
im = Image.open(img_dir + "Soma_s081.png").convert('L')
lut = [0]*256
lut[2] = 100
lut[3] = 50
out_img = Image.merge("HSV", (seg.point(lut=lut), seg.point(lut=lut), im))

#%%
pc2 = pixel_classifier_2d(img_rows=65, img_cols=65, proj_dir="/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/")
ckpt_path = max(iglob(join(pc2.model_dir, '*')), key=os.path.getctime)
inference_model = pc2.transfer_weight_to_inference(ckpt_path)
#%%


# "/Users/binxu/Connectomics_Code/tissue_classifier/Train_Img/Soma_s091.png"
downsample_factor = 2
img_pattern = "Soma_test_s*.png"
img_dir = "/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/Test_Img/" # "/scratch/binxu.wang/tissue_classifier/Train_Img/"
out_dir = "/home/morganlab/Documents/ixP11LGN/TissueClassifier_Soma/Test_Result/"# "/scratch/binxu.wang/tissue_classifier/Train_Result/"
def proc_img_dir(img_dir, img_pattern, out_dir, downsample_factor):
    lut = [0] * 256
    lut[2] = 100
    lut[3] = 50
    img_list = sorted(glob(img_dir+img_pattern))
    for img_name in img_list:
        print("Process ", img_name)
        im = Image.open(img_name).convert('L')
        if not downsample_factor == 1:
            ds_size = tuple(int(i // downsample_factor) for i in im.size)
            if any([not i%downsample_factor==0 for i in im.size]):
                print("Warning: Downsample has margin")
                new_size = tuple(int(i * downsample_factor) for i in ds_size)
                im = im.crop((0, 0, new_size[0], new_size[1]))
            im_ds = im.resize(ds_size)
            label_map = inference_on_image(np.array(im_ds), inference_model)
            seg = Image.fromarray(imresize(label_map, float(downsample_factor)))
        else:
            label_map = inference_on_image(np.array(im), inference_model)
            seg = Image.fromarray(label_map)
        print("Label finish ", img_name)
        out_img = Image.merge("HSV", (seg.point(lut=lut), seg.point(lut=lut), im))
        _, filename = os.path.split(img_name)
        out_img.convert("RGB").save(
            out_dir+filename[:filename.find(".")]+"_label.png")
        print("Merge finish ", img_name)
#%%
import h5py
vol_dir = "/home/morganlab/Documents/ixP11LGN/EM_data/grayscale_ixP11_5_align.h5"
f=h5py.File(vol_dir, 'r')
image_stack = f['raw']
image_stack = image_stack[:]
f.close()
#%%
def proc_vol(vol_dir, out_dir, downsample_factor=1, dataset='raw'):
    f = h5py.File(vol_dir, 'r')
    image_stack = f[dataset]
    image_stack = image_stack[:]
    f.close()
    _, filename = os.path.split(vol_dir)
    label_volume = np.zeros(image_stack.shape, dtype=np.uint8)
    for zid, img in enumerate(image_stack):
        print("Process %03d"%zid)
        label_map = inference_on_image(img, inference_model)

        if not downsample_factor == 1:
            ds_size = tuple(int(i // downsample_factor) for i in img.shape)
            if any([not i%downsample_factor==0 for i in img.shape]):
                print("Warning: Downsample has margin")
                new_size = tuple(int(i * downsample_factor) for i in ds_size)
                # im = im.crop((0, 0, new_size[0], new_size[1]))
            else:
                new_size = img.shape
            label_map = inference_on_image(imresize(img, float(1/downsample_factor)), inference_model)
            label_volume[zid, 0:new_size[0], 0:new_size[1]] = imresize(label_map, float(downsample_factor))
        else:
            label_map = inference_on_image(img, inference_model)
            label_volume[zid, :, :] = label_map
        print("Label finish %03d" % zid)
        im = Image.fromarray(img).convert('L')
        seg = Image.fromarray(label_volume[zid, :, :])
        out_img = Image.merge("HSV", (seg.point(lut=lut), seg.point(lut=lut), im))
        out_img.convert("RGB").save(
            join(out_dir, filename[:filename.find(".")]+"_label%03d.png"%zid))
        print("Merge finish %03d" % zid)
    # Post_processing

    f = h5py.File(join(out_dir, filename[:filename.find(".")]+"_label.h5"), "w")
    fstack = f.create_dataset("mask", (1,) + label_volume.shape, dtype='uint8')  # Note they only take int64 input
    fstack[0, :] = label_volume
    f.close()
    return label_volume
#%%
vol_dir = "/home/morganlab/Documents/ixP11LGN/EM_data/grayscale_ixP11_5_align.h5"
out_dir = "/home/morganlab/Documents/ixP11LGN/EM_data/mask/"
label_volume = proc_vol(vol_dir, out_dir, downsample_factor=2, dataset='raw')

#%%
import h5py
f = h5py.File("/home/morganlab/Documents/ixP11LGN/EM_data/mask/grayscale_ixP11_5_align_label1.h5", "w")
fstack = f.create_dataset("mask", (1,) + label_volume.shape, dtype='uint8')  # Note they only take int64 input
fstack[0, :] = label_volume
f.close()
#%%
import h5py
f = h5py.File("/home/morganlab/Documents/ixP11LGN/EM_data/mask/grayscale_ixP11_5_align_label.h5", "r")
# fstack = f.create_dataset("mask", (1,) + label_volume.shape, dtype='uint8')  # Note they only take int64 input
label_volume = f["mask"]
label_volume = label_volume[0, :]
f.close()

#%% median filter the z direction

import scipy.ndimage as ndimage
import skimage.measure
#%% clear all small islands
background_label = [0]
min_size = 1000000
def clear_small_island(label_volume, background_label=[0], min_size=1000000):
    labels = np.unique(label_volume)
    binary_mask = np.zeros(label_volume.shape, dtype=np.bool)

    # for label in labels:
    #     if label in background_label:
    #         continue
    #     binary_mask = np.logical_or(label_volume == label, binary_mask)
    valid_labels = np.setdiff1d(labels, background_label)
    binary_mask = np.in1d(label_volume, valid_labels).reshape(label_volume.shape)
    print("Get merge mask.")
    #%%

    isld_label_array, n_islands = ndimage.label(binary_mask) # skimage.measure.label(binary_mask, connectivity=1)
    print("%d islands found." % n_islands)
    #%%
    isld_labels, isld_sizes = np.unique(isld_label_array, return_counts=True)

    # large_isld_mask = np.zeros(label_volume.shape, dtype=np.bool)
    small = isld_labels[isld_sizes < min_size]
    print("Apply threshold %d, %d islands left, size %s." % (min_size, sum(isld_sizes > min_size), str(isld_sizes[isld_sizes > min_size])))
    small_mask = np.in1d(isld_label_array, small).reshape(isld_label_array.shape)
    large_mask_array = label_volume.copy()
    large_mask_array[small_mask] = 0
    return large_mask_array
#%%
plt.imshow(large_mask_array[40, :, :])
plt.show()
#%%

f = h5py.File("/home/morganlab/Documents/ixP11LGN/EM_data/mask/grayscale_ixP11_5_align_soma_mask.h5", "w")
fstack = f.create_dataset("mask", (1,) + label_volume.shape, dtype='uint8', compression="gzip")  # Note they only take int64 input
fstack[0, :] = large_mask_array
f.close()

#%%
# ndimage.median_filter
# skimage.measure

#%%

# plt.histogram(image_stack[40,:,:].flat)
# plt.show()


def normalize_img_stack_with_mask(path, output, EM_stack, upper=196, lower=80, up_outlier=245, low_outlier=30):
    low_p = []
    high_p = []
    for img in EM_stack:
        img1d = img.flatten()
        img1d = img1d[np.logical_and(img1d < up_outlier, img1d > low_outlier)]
        low_p.append(np.percentile(img1d, 5))
        high_p.append(np.percentile(img1d, 95))
    low_p = np.array(low_p)
    high_p = np.array(high_p)
    # low_p = np.percentile(EM_stack, 5, axis=[1,2])
    # high_p = np.percentile(EM_stack, 95, axis=[1,2])
    scaler = (upper-lower) / (high_p - low_p)
    shift = lower - (low_p*scaler)
    norm_img = scaler.reshape((-1,1,1)) * EM_stack + shift.reshape((-1,1,1))
    print("max: %.2f, min: %.2f after scaling"%(norm_img.max(), norm_img.min()))
    int_img = np.clip(norm_img, 0, 255, )
    int_img = int_img.astype('uint8')
    img_shape = EM_stack.shape
    f = h5py.File(join(path, output), "w")
    fstack=f.create_dataset("raw", img_shape, dtype='uint8', compression="gzip") # Note they only take int64 input
    fstack[:] = int_img
    f.close()
    return int_img

#%%
norm_img = normalize_img_stack_with_mask("/home/morganlab/Documents/ixP11LGN/EM_data", "grayscale_ixP11_5_align_norm_new.h5", image_stack)
