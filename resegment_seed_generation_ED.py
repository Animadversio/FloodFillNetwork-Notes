#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:52:18 2019

@author: morganlab

Search for connecting components in a object tensor
Generate the decision points for the algorithm using the mid point of nearest point pair
Using Euclidean distance transform
(Parallel Computing Version)
"""
#%%
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp
from contextlib import closing
import ctypes
import os
from os.path import join
import argparse
from ffn.inference.storage import subvolume_path
import resource

#%% General Settings

metric = [8, 12, 30]  # voxel size in x,y,z order in nm
dist_threshold = 50  # maximum distance in nm to be considered valid pair
threshold = 50  # minimum overlap to be consider a pair
move_vec = (5, 5, 3)

seg_path = "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12/"#"/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/"
# "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12/" #
    # "/Users/binxu/Connectomics_Code/results/LGN/testing_LR/0/0/"
    # "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/"
# "/Users/binxu/Connectomics_Code/results/LGN/"
# '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/'
output_path = "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/"# '/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/' # "/home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12/"
# "/scratch/binxu.wang/ffn-Data/results/LGN/testing_LR/"
# "/Users/binxu/Connectomics_Code/results/LGN/testing_LR/"


ap = argparse.ArgumentParser()
ap.add_argument(
    '--seg_path',
    help='')
ap.add_argument(
    '--output_path', help='Obtain the Neuroglancer client code from the specified URL.')
args = ap.parse_args()
if args.seg_path:
    seg_path = args.seg_path
if args.output_path:
    output_path = args.output_path
elif args.seg_path:
    output_path = args.seg_path

metric = np.array(metric)
metric = metric.reshape((-1, 1))  # reshape to ensure the computation below
metric = metric[::-1]  # in z y x order


def find_projection_point(seg_a, seg_b, metric=[30, 12, 8]):
    ''' Find the coordinate of the nearest point in seg_a and seg_b boolean mask'''
    coord_a = np.array(seg_a.nonzero())
    coord_b = np.array(seg_b.nonzero())
    dist_mat = np.zeros((coord_a.shape[1], coord_b.shape[1]))
    metric = np.array(metric)
    metric = metric.reshape((-1, 1))
    for i in range(coord_a.shape[1]):
        dist_mat[i, :] = np.sqrt(np.sum((metric * (coord_b - coord_a[:, [i]])) ** 2, axis=0))
    (i, j) = np.unravel_index(dist_mat.argmin(), dist_mat.shape)
    near_coord_a = coord_a[:, i]
    near_coord_b = coord_b[:, j]
    return near_coord_a, near_coord_b


def worker_func(id_pair):
    global composite_map_sh, BASE, metric, segmentation
    cur_idx1, cur_idx2 = id_pair[0], id_pair[1]
    if cur_idx1 == cur_idx2 or cur_idx1 * cur_idx2 == 0:
        return []  # ignore the overlap with background and samething overlap
    seg_a = segmentation == cur_idx1
    seg_b = segmentation == cur_idx2
    print('[%d] After calculate segment Memory usage: %s (kb)' % (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    coord_a = np.array(seg_a.nonzero())
    coord_b = np.array(seg_b.nonzero())
    print('[%d] After fetching coordinates Memory usage: %s (kb)' % (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    dist_mat = np.zeros((coord_a.shape[1], coord_b.shape[1])) # this is very memory costing!!!!!!
    for i in range(coord_a.shape[1]):
        dist_mat[i, :] = np.sqrt(np.sum((metric * (coord_b - coord_a[:, [i]])) ** 2, axis=0))
    print('[%d] After dist_mat Memory usage: %s (kb)'% (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    (i, j) = np.unravel_index(dist_mat.argmin(), dist_mat.shape)
    # del seg_a, seg_b
    near_coord_a = coord_a[:, i]
    near_coord_b = coord_b[:, j]
    if dist_mat[i, j] < dist_threshold:
        com_vec = ((near_coord_b + near_coord_a) / 2).astype(int)
        if len(com_vec.shape) == 2:  # list of nearest points
            com_vec = list(com_vec)
        else:
            com_vec = [com_vec]
        print("{id_a:%d id_b:%d point {%s} } min dist %.1f \n" % (cur_idx1, cur_idx2, str(com_vec), dist_mat[i, j]))
        del seg_a, seg_b, coord_a, coord_b, dist_mat
        print('[%d] Before printing Memory usage: %s (kb)' %
              (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        return com_vec
    else:
        del seg_a, seg_b, coord_a, coord_b, dist_mat
        return []


#%%
if __name__=="__main__":
    print('[%d] At start Memory usage: %s (kb)' % (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    data = np.load(subvolume_path(seg_path, (0, 0, 0), 'npz'))
    segmentation = data['segmentation']
    segmentation = segmentation[0:300, 0:300, 0:300]
    data.close()
    segmentation = segmentation.astype(np.int)  # make sure full byte width, or BASE * will outflow
    print('[%d] After cast type Memory usage: %s (kb)' %
          (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    vx, vy, vz = move_vec
    BASE = segmentation.max() + 1
    def _sel(i):
        if i == 0:
            return slice(None)
        elif i > 0:
            return slice(i, None)
        else:
            return slice(None, i)
    composite_map = segmentation[_sel(-vz), _sel(-vx), _sel(-vy)] + BASE * segmentation[_sel(vz), _sel(vx), _sel(vy)]

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
    valid_mask = (pair_array_sym[:,0]!=pair_array_sym[:,1]) * \
                 (pair_array_sym[:,0]*pair_array_sym[:,1]!=0) * \
                 (pair_cnts_sym > threshold)
    pair_num = sum(valid_mask)
    print("Pairs to process %d." % (pair_num))  # exclude background and same type
    pair_array_sym = pair_array_sym[valid_mask, :]
    pair_cnts_sym = pair_cnts_sym[valid_mask]
    # Prepare shared array ()Note this part will induce error ! very slow and inefficient
    # inz, iny, inx = composite_map.shape
    # X = mp.RawArray(ctypes.c_int16, inz * iny * inx)  # Note the datatype esp. when wrapping
    # # Wrap X as an inumpy array so we can easily manipulates its data.
    # composite_map_sh = np.frombuffer(X, dtype=np.int16).reshape(composite_map.shape)
    # # Copy data to our shared array.
    # np.copyto(composite_map_sh, composite_map)  # seg_array is int16 array.

    #%% parallelize the program
    print('[%d] Before starting Pool Memory usage: %s (kb)' %
          (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    pair_list = list(pair_array_sym)
    # mp.cpu_count())  # the code above does not work in Python 2.x but do in 3.6
    with closing(mp.Pool(processes=4, maxtasksperchild=2)) as pool: #  mp.cpu_count()//2
        result = pool.imap(worker_func, pair_list, chunksize=100)  # returns a generator _unordered
    # result_list = list(result)
    # pickle.dump(result_list, open(join(output_path, 'seed_result.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

    #%%
    print('[%d] Before writing down Memory usage: %s (kb)' %
          (os.getpid(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    # Save result to dict
    seed_dict = {}
    for result_vec, id_pair in zip(result, pair_list):
        print(id_pair)
        cur_idx1, cur_idx2 = id_pair[0], id_pair[1]
        if len(result_vec)!=0:
            if type(result_vec) == list:
                seed_dict[(cur_idx1, cur_idx2)] = seed_dict.get((cur_idx1, cur_idx2), []) + result_vec  # note extend here #seed_dict[(cur_idx1, cur_idx2)] =
            else:
                seed_dict[(cur_idx1, cur_idx2)] = seed_dict.get((cur_idx1, cur_idx2), []) + [result_vec]
    pickle.dump(seed_dict, open(join(output_path, 'seed_dict.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    # Write the pb file
    file = open(join(output_path, "resegment_point_list.txt"), "w")
    for pair in seed_dict:
        for pnt in seed_dict[pair]:
            file.write("points {id_a:%d id_b:%d point {x: %d y: %d z: %d} } \n" % (pair[0], pair[1], pnt[2], pnt[1], pnt[0]))
    file.close()

    pool.close()
    print("closed pool")
    print("joining pool")
    pool.join()
    print("joined pool")
#%% merge the lists generated by different movement vectors

