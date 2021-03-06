#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:52:18 2019

@author: morganlab

Search for connecting components in a object tensor
Using morgan's move and unique the pair strategy
(Parallel Computing Version)
"""
#%%
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp
import ctypes
import sys

#%%
# "/scratch/binxu.wang/ffn-Data/results/LGN/testing_LR/"
# "/Users/binxu/Connectomics_Code/results/LGN/testing_LR/0/0/"
ResultPath = "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/"
# if len(sys.argv)>1:
#     ResultPath = sys.argv[1]
output_path = "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/"# "/Users/binxu/Connectomics_Code/results/LGN/"
# if len(sys.argv)>2:
#     output_path = sys.argv[2]
# '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/'
testSegLoc = ResultPath + "seg-0_0_0.npz"
testProbLoc = ResultPath + 'seg-0_0_0.prob'

data = np.load(testSegLoc)
segmentation = data['segmentation']
#%%
move_vec = (10, 10, 3)
vx, vy, vz = move_vec
BASE = segmentation.max() + 1
def _sel(i):
    if i == 0:
        return slice(None)
    elif i > 0:
        return slice(i, None)
    else:
        return slice(None, i)
composite_map = segmentation[_sel(-vz), _sel(-vx), _sel(-vy)].astype(np.int) + BASE * segmentation[_sel(vz), _sel(vx), _sel(vy)].astype(np.int)

pair_idx, pair_cnts=np.unique(composite_map, return_counts=True)
idx2, idx1 = np.divmod(pair_idx, BASE)
pair_array = np.array([idx1, idx2]).T
def symmetrize_pair_array(pair_array, pair_cnts):
    pair_array_sym = np.sort(pair_array, axis=1)
    pair_array_sym = np.unique(pair_array_sym,axis=0)
    pair_idx_sym = pair_array_sym[:, 0] + pair_array_sym[:, 1]*BASE
    pair_cnts_sym = np.zeros(pair_array_sym.shape[0])
    for i in range(len(pair_cnts)):
        relid1 = np.where(pair_idx_sym==(idx1[i] +BASE*idx2[i]))[0]
        relid2 = np.where(pair_idx_sym==(idx2[i] +BASE*idx1[i]))[0]
        if len(relid1)==0:
            pair_cnts_sym[relid2] += pair_cnts[i]
        elif len(relid2)==0:
            pair_cnts_sym[relid1] += pair_cnts[i]
        else:  # same index idx1[i]==idx2[i]
            assert relid1==relid2
            pair_cnts_sym[relid2] += pair_cnts[i]
    return pair_array_sym, pair_cnts_sym
pair_array_sym, pair_cnts_sym = symmetrize_pair_array(pair_array, pair_cnts)
assert pair_cnts_sym.sum() == pair_cnts.sum()
# Threshold of overlap size can be added !
#%%
threshold = 50
valid_mask = (pair_array_sym[:,0]!=pair_array_sym[:,1]) * \
             (pair_array_sym[:,0]*pair_array_sym[:,1]!=0) * \
             (pair_cnts_sym > threshold)
pair_num = sum(valid_mask)
print("Pairs to process %d." % (pair_num)) # exclude background and same type
pair_array_sym = pair_array_sym[valid_mask,:]
pair_cnts_sym = pair_cnts_sym[valid_mask]
# Prepare shared array ()Note this part will induce error ! very slow and inefficient
# inz, iny, inx = composite_map.shape
# X = mp.RawArray(ctypes.c_int16, inz * iny * inx)  # Note the datatype esp. when wrapping
# # Wrap X as an inumpy array so we can easily manipulates its data.
# composite_map_sh = np.frombuffer(X, dtype=np.int16).reshape(composite_map.shape)
# # Copy data to our shared array.
# np.copyto(composite_map_sh, composite_map)  # seg_array is int16 array.

#%% generate integer seed around the center of mass
def seed_regularize(com_vec, shift_vec):
    com_vec_reg = [(int(vec[0] + vz/2), int(vec[1] + vx/2), int(vec[2] + vy/2)) for vec in com_vec]
    # assert seeds in bound
    return com_vec_reg
#%% for each pair associate the center for each intersecting island
# seed_dict = {} # manage the map from pair to seed
# for i in range(len(pair_cnts_sym)): # range(5000,5500):  #
def worker_func(id_pair):
    global composite_map_sh,BASE
    cur_idx1, cur_idx2 = id_pair[0], id_pair[1]
    if cur_idx1 == cur_idx2 or cur_idx1 * cur_idx2 == 0:
        return []  # ignore the overlap with background and samething overlap
    mask = (composite_map == cur_idx1 + cur_idx2 * BASE) | (composite_map == cur_idx2 + cur_idx1 * BASE)
    label_im, nb_labels = ndimage.label(mask, structure=ndimage.generate_binary_structure(3,3))
    # find sizes of
    _, sizes = np.unique(label_im, return_counts=True)
    # sizes = ndimage.sum(mask, label_im, range(1, nb_labels + 1))
    sizes = sizes[1:] # discard the 0
    assert len(sizes) == nb_labels
    significance_id = np.nonzero(sizes > threshold)[0] + 1
    # significance_id = sizes.argmax() + 1
    com_vec = ndimage.measurements.center_of_mass(mask, label_im, significance_id)
    # com_vec = ndimage.measurements.center_of_mass(mask, mask, 1)
    # ndimage.find_objects()
    if not len(com_vec) == 0:
        com_vec = seed_regularize(com_vec, move_vec)
        # seed_dict[(cur_idx1, cur_idx2)] = seed_dict.get((cur_idx1, cur_idx2), []).extend(com_vec)  # note extend here
        print("{id_a:%d id_b:%d point {%s} } size %d \n" % (cur_idx1, cur_idx2, str(com_vec), sizes.sum()))
        return com_vec
    else:
        return []

#%% parallelize the program
pair_list = list(pair_array_sym)
pool = mp.Pool(processes=mp.cpu_count())  # the code above does not work in Python 2.x but do in 3.6
result = pool.map(worker_func, pair_list)
# Save result to dict
seed_dict = {}
for result_vec, id_pair in zip(result, pair_list):
    cur_idx1, cur_idx2 = id_pair[0], id_pair[1]
    if len(result_vec)!=0:
        seed_dict[(cur_idx1, cur_idx2)] = seed_dict.get((cur_idx1, cur_idx2), []) + result_vec  # note extend here #seed_dict[(cur_idx1, cur_idx2)] =

pickle.dump(seed_dict, open(output_path+'seed_dict.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
# Write the pb file
file = open(output_path+"resegment_point_list.txt", "w")
for pair in seed_dict:
    for pnt in seed_dict[pair]:
        file.write("points {id_a:%d id_b:%d point {x: %d y: %d z: %d} } \n" % (pair[0], pair[1], pnt[1], pnt[2], pnt[0]))

file.close()
#%% merge the lists generated by different movement vectors

